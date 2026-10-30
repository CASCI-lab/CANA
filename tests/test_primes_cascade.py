# -*- coding: utf-8 -*-
"""Tests for expression cascade and FunctionOutputs prime priority."""
from __future__ import annotations

import pytest

from cana.bit_lut import BitLut
from cana.function_outputs import FunctionOutputs
from cana.primes.cascade import on_off_primes_from_expression
from cana.primes.util import MAX_DENSE_LUT_K, MAX_PRIMES


def test_cascade_rof_ok():
    result = on_off_primes_from_expression("a & !b")
    assert result.ok
    assert result.method == "rof"
    assert result.on_primes == [{"a": True, "b": False}]
    assert result.n_on == 1
    assert result.n_off == 2


def test_cascade_rof_too_many():
    # Product of 21 pairwise ORs → 2**21 ON primes.
    pairs = [f"(a{i} | b{i})" for i in range(21)]
    expr = " & ".join(pairs)
    result = on_off_primes_from_expression(expr, max_primes=1000)
    assert result.status == "too_many"
    assert result.method == "rof"
    assert result.n_on == 2**21
    assert result.on_primes is None


def test_cascade_non_rof_primer_or_unsuccessful():
    expr = "a & b | !a & c"
    result = on_off_primes_from_expression(expr, timeout_s=0)
    if result.ok:
        assert result.method == "primer"
        assert result.n_on == 3
    else:
        # Missing pysat / other soft failure — not a hard crash.
        assert result.status in (
            "unsuccessful",
            "timeout",
            "refused",
            "error",
        )


def test_fo_rof_stores_schemata_without_lut():
    fo = FunctionOutputs(positional_expression="x0 and not x1", k=2)
    assert fo.bit_lut is None
    schemata = fo.get_one_symbol_schemata()
    assert fo.bit_lut is None
    assert fo.one_symbol_is_prime_implicant is True
    # ON: x0 & !x1 → "10"; OFF: !x0 → "02", x1 → "21"
    assert schemata[1] == {"10"}
    assert schemata[0] == {"02", "21"}


def test_fo_cascade_miss_existing_bit_lut_uses_qm():
    # LUT-only FO skips the expression cascade and uses QM.
    src = FunctionOutputs(
        positional_expression="(x0 and x1) or ((not x0) and x2)", k=3
    )
    lut = BitLut.from_sequence(src.get_bit_lut())
    fo = FunctionOutputs(bit_lut=lut)
    assert fo.positional_expression is None
    schemata = fo.get_one_symbol_schemata()
    assert fo.one_symbol_is_prime_implicant is True
    assert len(schemata[1]) == 3


def test_fo_cascade_miss_builds_lut_when_k_small(monkeypatch):
    from cana.primes.cascade import PrimesResult

    def _fail(_expr, **_kwargs):
        return PrimesResult(status="unsuccessful", message="forced")

    monkeypatch.setattr(
        "cana.primes.cascade.on_off_primes_from_expression", _fail
    )

    fo = FunctionOutputs(
        positional_expression="(x0 and x1) or ((not x0) and x2)", k=3
    )
    assert fo.bit_lut is None
    schemata = fo.get_one_symbol_schemata()
    assert fo.bit_lut is not None
    assert fo.one_symbol_is_prime_implicant is True
    assert len(schemata[1]) == 3


def test_fo_refuses_huge_k_after_cascade_miss(monkeypatch):
    from cana.primes.cascade import PrimesResult

    def _fail(_expr, **_kwargs):
        return PrimesResult(status="unsuccessful", message="forced")

    monkeypatch.setattr(
        "cana.primes.cascade.on_off_primes_from_expression", _fail
    )

    k = MAX_DENSE_LUT_K + 1
    names = " and ".join(f"x{i}" for i in range(k))
    fo = FunctionOutputs(positional_expression=names, k=k)
    with pytest.raises(ValueError, match="MAX_DENSE_LUT_K"):
        fo.get_one_symbol_schemata()
    assert fo.bit_lut is None


def test_max_primes_constant():
    assert MAX_PRIMES == 1_000_000
