# -*- coding: utf-8 -*-
"""Tests for :mod:`cana.primes.primer` and :func:`call_in_subprocess`."""
from __future__ import annotations

import time

import pytest

from cana.function_outputs import FunctionOutputs
from cana.nodes_builder import named_expression_to_detached_rule
from cana.primes.primer import (
    DEFAULT_TIMEOUT_S,
    MAX_PRIMER_PAREN_DEPTH,
    MAX_PRIMER_PRIMES,
    primer_on_off_primes,
)
from cana.primes.timeout import call_in_subprocess
from cana.primes.util import MAX_DENSE_LUT_K, one_symbol_schemata_to_primes


def _freeze(prime_list) -> frozenset:
    return frozenset(frozenset(p.items()) for p in prime_list)


def _match(a, b) -> bool:
    return _freeze(a[0]) == _freeze(b[0]) and _freeze(a[1]) == _freeze(b[1])


def _qm_on_off(expr: str):
    rule = named_expression_to_detached_rule("Y", expr)
    fo = rule.outputs
    fo.build_bit_lut_from_positional_expression()
    schemata = fo.compute_one_symbol_schemata_from_bit_lut()
    return one_symbol_schemata_to_primes(schemata, rule.input_names)


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
    assert MAX_DENSE_LUT_K == 20
    assert MAX_PRIMER_PRIMES == 1_000_000
    assert MAX_PRIMER_PAREN_DEPTH == 220


def _right_nested(depth: int) -> str:
    expr = "a"
    for _ in range(depth):
        expr = f"({expr} & a)"
    return expr


def test_primer_refuses_extreme_paren_nesting_even_with_short_timeout():
    pytest.importorskip("pysat")
    expr = _right_nested(MAX_PRIMER_PAREN_DEPTH + 1)
    with pytest.raises(RecursionError, match="parenthesis depth"):
        primer_on_off_primes(expr, timeout_s=0.05)


@pytest.mark.parametrize("expr", _CASES)
def test_primer_matches_qm(expr: str):
    pytest.importorskip("pysat")
    qm = _qm_on_off(expr)
    pr = primer_on_off_primes(expr, timeout_s=0)
    assert _match(pr, qm)


def test_primer_refuses_when_prime_count_exceeds_cap():
    pytest.importorskip("pysat")
    expr = "a & b | !a & c"
    with pytest.raises(
        ValueError, match="prime count exceeded MAX_PRIMER_PRIMES=2"
    ):
        primer_on_off_primes(expr, timeout_s=0, max_primes=2)


def test_primer_subprocess_timeout(monkeypatch):
    pytest.importorskip("pysat")

    def _slow(_expr, max_primes=None):  # noqa: ARG001
        time.sleep(5.0)
        return [], []

    monkeypatch.setattr(
        "cana.primes.primer._primer_on_off_primes_inprocess", _slow
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
    import os
    import signal as signal_mod

    def _suicide(conn) -> None:
        os.kill(os.getpid(), signal_mod.SIGKILL)
        conn.send(("ok", None))
        conn.close()

    with pytest.raises(TimeoutError, match="killed by signal"):
        call_in_subprocess(_suicide, (), timeout_s=2.0)


def test_call_in_subprocess_kills_grandchild(tmp_path):
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

    deadline = time.perf_counter() + 2.0
    gp_pid = int(pidfile.read_text(encoding="utf-8"))
    while time.perf_counter() < deadline:
        try:
            os.kill(gp_pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    try:
        os.kill(gp_pid, 0)
        alive = True
    except ProcessLookupError:
        alive = False
    assert not alive, f"grandchild pid={gp_pid} survived process-group kill"
