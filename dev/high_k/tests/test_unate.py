# -*- coding: utf-8 -*-
"""Tests for :mod:`high_k.unate`."""
from __future__ import annotations

import time

import pytest

from high_k.bbm_rules import as_bnet, unique_regulators
from high_k.rof import check_syntactic_rof
from high_k.timing import call_in_subprocess, import_detached_outputs
from high_k.unate import (
    check_syntactic_unate,
    on_off_primes,
    primes,
    require_syntactic_unate,
)


def test_all_positive_is_unate():
    info = check_syntactic_unate("a & b | c")
    assert info["is_unate"] is True
    assert info["positive"] == ["a", "b", "c"]
    assert info["negative"] == []
    assert info["binate"] == []
    assert info["n_unique_variables"] == 3


def test_all_negative_is_unate():
    info = check_syntactic_unate("!a & !b")
    assert info["is_unate"] is True
    assert info["positive"] == []
    assert info["negative"] == ["a", "b"]
    assert info["binate"] == []


def test_mixed_consistent_polarities_is_unate():
    info = check_syntactic_unate("a & !b")
    assert info["is_unate"] is True
    assert info["positive"] == ["a"]
    assert info["negative"] == ["b"]
    assert info["binate"] == []


def test_binate_variable_rejected():
    info = check_syntactic_unate("a | !a")
    assert info["is_unate"] is False
    assert info["binate"] == ["a"]
    assert info["positive"] == []
    assert info["negative"] == []


def test_de_morgan_creates_binate():
    info = check_syntactic_unate("!(a & b) | a")
    assert info["is_unate"] is False
    assert info["binate"] == ["a"]
    assert info["negative"] == ["b"]
    assert info["positive"] == []


def test_de_morgan_stays_unate():
    info = check_syntactic_unate("!(a & b)")
    assert info["is_unate"] is True
    assert info["positive"] == []
    assert info["negative"] == ["a", "b"]
    assert info["binate"] == []


def test_repeated_same_polarity_is_unate():
    info = check_syntactic_unate("a & a")
    assert info["is_unate"] is True
    assert info["positive"] == ["a"]
    assert info["n_unique_variables"] == 1


def test_vacuous_empty_and_constant():
    empty = check_syntactic_unate("")
    assert empty["is_unate"] is True
    assert empty["positive"] == []
    assert empty["negative"] == []
    assert empty["binate"] == []
    assert empty["n_unique_variables"] == 0
    assert check_syntactic_unate("1")["is_unate"] is True
    assert check_syntactic_unate("0")["is_unate"] is True


def test_require_syntactic_unate():
    require_syntactic_unate("a & !b")
    with pytest.raises(ValueError, match="syntactically unate"):
        require_syntactic_unate("a | !a")


def test_python_operators():
    info = check_syntactic_unate("a and not b")
    assert info["is_unate"] is True
    assert info["positive"] == ["a"]
    assert info["negative"] == ["b"]


def test_stacked_negation_nnf():
    assert check_syntactic_unate("!!a") == {
        "is_unate": True,
        "positive": ["a"],
        "negative": [],
        "binate": [],
        "n_unique_variables": 1,
    }
    # !!a | a ≡ a | a → still unate (same polarity).
    assert check_syntactic_unate("!!a | a")["is_unate"] is True
    assert check_syntactic_unate("!!!a")["negative"] == ["a"]
    assert check_syntactic_unate("!!!a")["positive"] == []
    # Odd stack + positive occurrence → binate (not a fake '! a' name).
    info = check_syntactic_unate("!!!a | a")
    assert info["is_unate"] is False
    assert info["binate"] == ["a"]


def test_stops_at_first_binate():
    # Binate first, then a long unate suffix that must not be visited
    # enough to populate positive with the suffix names — but
    # n_unique_variables must still count the full formula.
    suffix = " & ".join(f"x{i}" for i in range(20))
    expr = f"(z | !z) & {suffix}"
    info = check_syntactic_unate(expr)
    assert info["is_unate"] is False
    assert info["binate"] == ["z"]
    assert info["positive"] == []
    assert len(info["binate"]) == 1
    assert info["n_unique_variables"] == 21  # z + x0..x19


def test_n_unique_matches_rof_on_binate():
    expr = "(a | !a) & b & c & d & e"
    unate = check_syntactic_unate(expr)
    rof = check_syntactic_rof(expr)
    assert unate["is_unate"] is False
    assert unate["n_unique_variables"] == rof["n_unique_variables"] == 5


def test_deep_nesting_does_not_hit_recursion_limit():
    # BBM-style left-nested parens; recursive descent used to raise
    # RecursionError well below this depth.
    depth = 5_000
    expr = "(" * depth + "a & !b" + ")" * depth
    info = check_syntactic_unate(expr)
    assert info["is_unate"] is True
    assert info["positive"] == ["a"]
    assert info["negative"] == ["b"]
    # Polarity flip under a deep outer negation.
    info_neg = check_syntactic_unate("!" + expr)
    assert info_neg["is_unate"] is True
    assert info_neg["positive"] == ["b"]
    assert info_neg["negative"] == ["a"]


# ---------------------------------------------------------------------------
# Espresso prime implicants (requires pyeda)
# ---------------------------------------------------------------------------


def _require_pyeda() -> None:
    pytest.importorskip("pyeda")


def _freeze(prime_list) -> frozenset:
    return frozenset(frozenset(p.items()) for p in prime_list)


def _qm_on_off_primes(expr: str):
    """ON/OFF primes as PrimeDicts via CANA Quine–McCluskey schemata."""
    names = unique_regulators(expr)
    fo = import_detached_outputs(as_bnet(expr), "Y")
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
