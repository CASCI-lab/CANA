# -*- coding: utf-8 -*-
"""Tests for :mod:`high_k.general_primes` (exact prime fallbacks).

Espresso is intentionally not covered: PyEDA's Espresso API returns a
minimal SOP cover, not the full prime-implicant set for binate formulas.
"""
from __future__ import annotations

import time

import pytest

from high_k.general_primes import (
    DEFAULT_TIMEOUT_S,
    EXPLORATION_TIMEOUT_S,
    bdd_on_off_primes,
    complete_sum_on_off_primes,
    pyboolnet_on_off_primes,
    qm_on_off_primes,
)
from high_k.timing import call_in_subprocess


def _freeze(prime_list) -> frozenset:
    return frozenset(frozenset(p.items()) for p in prime_list)


def _match(a, b) -> bool:
    return _freeze(a[0]) == _freeze(b[0]) and _freeze(a[1]) == _freeze(b[1])


# Binate / general formulas where full primes include consensus terms.
_CASES = [
    "a & b | !a & c",
    "a & b | ~a & c",
    "a | !a",
    "a & b",
    "(a0 | a1) & (a2 | a3)",
    "a & b | a & c",
    "a0 & a1 & a2 | !a0 & !a1",
]


def test_timeout_constants():
    assert DEFAULT_TIMEOUT_S == 10.0
    assert EXPLORATION_TIMEOUT_S == 1.0


@pytest.mark.parametrize("expr", _CASES)
def test_qm_binate_includes_consensus_when_expected(expr: str):
    on_p, off_p = qm_on_off_primes(expr, timeout_s=0)
    assert isinstance(on_p, list) and isinstance(off_p, list)
    if expr in ("a & b | !a & c", "a & b | ~a & c"):
        # Full ON primes: a&b, ~a&c, and consensus b&c.
        assert _freeze(on_p) == _freeze(
            [
                {"a": True, "b": True},
                {"a": False, "c": True},
                {"b": True, "c": True},
            ]
        )


@pytest.mark.parametrize("expr", _CASES)
def test_complete_sum_matches_qm(expr: str):
    pytest.importorskip("pyeda")
    qm = qm_on_off_primes(expr, timeout_s=0)
    cs = complete_sum_on_off_primes(expr, timeout_s=0)
    assert _match(cs, qm)


@pytest.mark.parametrize("expr", _CASES)
def test_bdd_matches_qm(expr: str):
    pytest.importorskip("dd")
    qm = qm_on_off_primes(expr, timeout_s=0)
    bd = bdd_on_off_primes(expr, timeout_s=0)
    assert _match(bd, qm)


@pytest.mark.parametrize("expr", _CASES)
def test_pyboolnet_matches_qm(expr: str):
    pytest.importorskip("pyboolnet")
    qm = qm_on_off_primes(expr, timeout_s=0)
    pb = pyboolnet_on_off_primes(expr, timeout_s=0)
    assert _match(pb, qm)


def test_qm_subprocess_timeout(monkeypatch):
    def _slow(_expr):
        time.sleep(5.0)
        return [], []

    monkeypatch.setattr(
        "high_k.general_primes._qm_on_off_primes_inprocess", _slow
    )
    with pytest.raises(TimeoutError, match="timeout"):
        qm_on_off_primes("a & b", timeout_s=0.2)


def test_complete_sum_subprocess_timeout(monkeypatch):
    pytest.importorskip("pyeda")

    def _slow(_expr):
        time.sleep(5.0)
        return [], []

    monkeypatch.setattr(
        "high_k.general_primes._complete_sum_on_off_primes_inprocess",
        _slow,
    )
    with pytest.raises(TimeoutError, match="timeout"):
        complete_sum_on_off_primes("a & b", timeout_s=0.2)


def test_bdd_subprocess_timeout(monkeypatch):
    pytest.importorskip("dd")

    def _slow(_expr):
        time.sleep(5.0)
        return [], []

    monkeypatch.setattr(
        "high_k.general_primes._bdd_on_off_primes_inprocess", _slow
    )
    with pytest.raises(TimeoutError, match="timeout"):
        bdd_on_off_primes("a & b", timeout_s=0.2)


def test_call_in_subprocess_still_works():
    def _slow(_conn):
        time.sleep(5.0)
        _conn.send(("ok", None))
        _conn.close()

    with pytest.raises(TimeoutError, match="timeout"):
        call_in_subprocess(_slow, (), timeout_s=0.2)
