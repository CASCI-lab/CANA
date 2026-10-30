# -*- coding: utf-8 -*-
"""Tests for :mod:`cana.primes.primer` (in-process; no wall-clock timeout)."""
from __future__ import annotations

import pytest

from cana.nodes_builder import named_expression_to_detached_rule
from cana.primes.primer import (
    MAX_PRIMER_PAREN_DEPTH,
    MAX_PRIMER_PRIMES,
    primer_on_off_primes,
)
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


def test_structural_guard_constants():
    assert MAX_DENSE_LUT_K == 20
    assert MAX_PRIMER_PRIMES == 1_000_000
    assert MAX_PRIMER_PAREN_DEPTH == 220


def _right_nested(depth: int) -> str:
    expr = "a"
    for _ in range(depth):
        expr = f"({expr} & a)"
    return expr


def test_primer_refuses_extreme_paren_nesting():
    pytest.importorskip("pysat")
    expr = _right_nested(MAX_PRIMER_PAREN_DEPTH + 1)
    with pytest.raises(RecursionError, match="parenthesis depth"):
        primer_on_off_primes(expr)


@pytest.mark.parametrize("expr", _CASES)
def test_primer_matches_qm(expr: str):
    pytest.importorskip("pysat")
    qm = _qm_on_off(expr)
    pr = primer_on_off_primes(expr)
    assert _match(pr, qm)


def test_primer_refuses_when_prime_count_exceeds_cap():
    pytest.importorskip("pysat")
    expr = "a & b | !a & c"
    with pytest.raises(
        ValueError, match="prime count exceeded MAX_PRIMER_PRIMES=2"
    ):
        primer_on_off_primes(expr, max_primes=2)
