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
    MAX_DENSE_LUT_K,
    MAX_PRIMER_PAREN_DEPTH,
    MAX_PRIMER_PRIMES,
    MAX_PYEDA_PAREN_DEPTH,
    complete_sum_on_off_primes,
    primer_on_off_primes,
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
    assert MAX_DENSE_LUT_K == 20
    assert MAX_PRIMER_PRIMES == 1_000_000
    assert MAX_PYEDA_PAREN_DEPTH == 120
    assert MAX_PRIMER_PAREN_DEPTH == 220


def test_qm_refuses_huge_k():
    # k = 21 > MAX_DENSE_LUT_K; predictable Θ(2^k) RAM.
    names = [f"a{i}" for i in range(MAX_DENSE_LUT_K + 1)]
    expr = " & ".join(names)
    with pytest.raises(ValueError, match="MAX_DENSE_LUT_K"):
        qm_on_off_primes(expr, timeout_s=0)


def _right_nested(depth: int) -> str:
    expr = "a"
    for _ in range(depth):
        expr = f"({expr} & a)"
    return expr


def test_pyeda_refuses_extreme_paren_nesting_even_with_short_timeout():
    """Deep nesting must be ``error``, not a wall-clock ``timeout``."""
    # Just above MAX_PYEDA_PAREN_DEPTH (hard fail ≈ 140).
    expr = _right_nested(MAX_PYEDA_PAREN_DEPTH + 1)
    with pytest.raises(RecursionError, match="parenthesis depth"):
        complete_sum_on_off_primes(expr, timeout_s=0.05)


def test_primer_refuses_extreme_paren_nesting_even_with_short_timeout():
    """Primer nesting refuse must be ``error``, not a wall-clock ``timeout``."""
    pytest.importorskip("pysat")
    # Just above MAX_PRIMER_PAREN_DEPTH (hard fail ≈ 246).
    expr = _right_nested(MAX_PRIMER_PAREN_DEPTH + 1)
    with pytest.raises(RecursionError, match="parenthesis depth"):
        primer_on_off_primes(expr, timeout_s=0.05)


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
def test_primer_matches_qm(expr: str):
    pytest.importorskip("pysat")
    qm = qm_on_off_primes(expr, timeout_s=0)
    pr = primer_on_off_primes(expr, timeout_s=0)
    assert _match(pr, qm)


def test_primer_refuses_when_prime_count_exceeds_cap():
    """Partial enumeration must not be returned once the cap is hit."""
    pytest.importorskip("pysat")
    # Full ON primes for this expr: a&b, ~a&c, b&c (3 primes).
    expr = "a & b | !a & c"
    with pytest.raises(
        ValueError, match="prime count exceeded MAX_PRIMER_PRIMES=2"
    ):
        primer_on_off_primes(expr, timeout_s=0, max_primes=2)


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


def test_primer_subprocess_timeout(monkeypatch):
    pytest.importorskip("pysat")

    def _slow(_expr, max_primes=None):  # noqa: ARG001
        time.sleep(5.0)
        return [], []

    monkeypatch.setattr(
        "high_k.general_primes._primer_on_off_primes_inprocess", _slow
    )
    with pytest.raises(TimeoutError, match="timeout"):
        primer_on_off_primes("a & b", timeout_s=0.2)


def test_call_in_subprocess_still_works():
    def _slow(_conn):
        time.sleep(5.0)
        _conn.send(("ok", None))
        _conn.close()

    with pytest.raises(TimeoutError, match="timeout"):
        call_in_subprocess(_slow, (), timeout_s=0.2)


def test_call_in_subprocess_signal_death_is_timeout():
    """OOM/SIGKILL mid-run must count as timeout, not ``exited without result``."""
    import os
    import signal as signal_mod

    def _suicide(conn) -> None:
        # Die by signal without sending a result (simulates OOM killer).
        os.kill(os.getpid(), signal_mod.SIGKILL)
        conn.send(("ok", None))
        conn.close()

    with pytest.raises(TimeoutError, match="killed by signal"):
        call_in_subprocess(_suicide, (), timeout_s=2.0)


def test_call_in_subprocess_kills_grandchild(tmp_path):
    """Process-group kill must reap subprocess.Popen grandchildren."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    pidfile = tmp_path / "grandchild.pid"

    def _worker(path: str, conn) -> None:
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(120)"]
        )
        Path(path).write_text(str(child.pid), encoding="utf-8")
        time.sleep(120.0)
        conn.send(("ok", None))
        conn.close()

    with pytest.raises(TimeoutError, match="timeout"):
        call_in_subprocess(_worker, (str(pidfile),), timeout_s=0.3)

    # Allow the kernel a moment to reap the grandchild.
    deadline = time.perf_counter() + 2.0
    gp_pid = int(pidfile.read_text(encoding="utf-8"))
    while time.perf_counter() < deadline:
        try:
            os.kill(gp_pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    # Still alive → fail.
    try:
        os.kill(gp_pid, 0)
        alive = True
    except ProcessLookupError:
        alive = False
    assert not alive, f"grandchild pid={gp_pid} survived process-group kill"
