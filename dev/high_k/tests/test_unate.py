# -*- coding: utf-8 -*-
"""Tests for high_k Espresso prime implicants (:mod:`high_k.unate`).

Syntactic unate checks live in :mod:`cana.function_properties.unate` and are
covered by ``tests/test_function_properties_unate.py``.
"""
from __future__ import annotations

import time

import pytest

from cana.function_properties.unate import check_syntactic_unate
from high_k.bbm_rules import as_bnet, unique_regulators
from high_k.timing import call_in_subprocess, import_detached_function
from high_k.unate import on_off_primes, primes


def _require_pyeda() -> None:
    pytest.importorskip("pyeda")


def _freeze(prime_list) -> frozenset:
    return frozenset(frozenset(p.items()) for p in prime_list)


def _qm_on_off_primes(expr: str):
    """ON/OFF primes as PrimeDicts via CANA Quine–McCluskey schemata."""
    names = unique_regulators(expr)
    fo = import_detached_function(as_bnet(expr), "Y")
    fo.build_bit_lut_from_positional_expression()
    schemata = fo.compute_one_symbol_schemata_from_bit_lut()

    def convert(sset):
        out = []
        for s in sset:
            d = {}
            for i, ch in enumerate(s):
                if ch == "1":
                    d[names[i]] = True
                elif ch == "0":
                    d[names[i]] = False
            out.append(d)
        return out

    return convert(schemata.get(1, ())), convert(schemata.get(0, ()))


@pytest.mark.parametrize(
    "expr",
    [
        "a0 & a1",
        "a0 | a1",
        "(a0 | a1) & (a2 | a3)",
        "(a0 & a1) | (a2 & a3)",
        "a and not b",
        "a & b | a & c",
        "!(a & b)",
    ],
)
def test_on_off_primes_match_qm(expr: str):
    _require_pyeda()
    assert check_syntactic_unate(expr)["is_unate"]
    on_p, off_p = on_off_primes(expr, timeout_s=0)
    qm_on, qm_off = _qm_on_off_primes(expr)
    assert _freeze(on_p) == _freeze(qm_on)
    assert _freeze(off_p) == _freeze(qm_off)
    assert _freeze(primes(expr, "on", timeout_s=0)) == _freeze(qm_on)
    assert _freeze(primes(expr, "off", timeout_s=0)) == _freeze(qm_off)


def test_on_off_primes_rejects_binate():
    _require_pyeda()
    with pytest.raises(ValueError, match="syntactically unate"):
        on_off_primes("a | !a", timeout_s=0)


def test_call_in_subprocess_timeout():
    def _slow(_conn):
        time.sleep(5.0)
        _conn.send(("ok", None))
        _conn.close()

    with pytest.raises(TimeoutError, match="timeout"):
        call_in_subprocess(_slow, (), timeout_s=0.2)


def test_on_off_primes_timeout(monkeypatch):
    _require_pyeda()

    def _slow(_expr):
        time.sleep(5.0)
        return [], []

    monkeypatch.setattr(
        "high_k.unate._espresso_on_off_primes_inprocess", _slow
    )
    with pytest.raises(TimeoutError, match="timeout"):
        on_off_primes("a & b", timeout_s=0.2)
