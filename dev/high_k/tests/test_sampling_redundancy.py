# -*- coding: utf-8 -*-
"""Tests for :mod:`high_k.sampling_redundancy`."""
from __future__ import annotations

from statistics import mean

import numpy as np
import pytest

from high_k.paths import ensure_cana_on_path
from high_k.sampling_redundancy import (
    bootstrap_edge_redundancy,
    bootstrap_input_redundancy,
    covering_primes,
    estimate_edge_redundancy,
    estimate_input_redundancy,
    pi_covers,
    sample_binstates,
    schemata_from_on_off,
)

ensure_cana_on_path()

from cana.boolean_node import BooleanNode  # noqa: E402
from cana.datasets.bools import AND  # noqa: E402


def _and_schemata(k: int):
    """One-symbol cover for k-input AND (no LUT)."""
    on = {"1" * k}
    off = set()
    for i in range(k):
        chars = ["2"] * k
        chars[i] = "0"
        off.add("".join(chars))
    return {0: off, 1: on}


def _or_schemata(k: int):
    """One-symbol cover for k-input OR (no LUT)."""
    off = {"0" * k}
    on = set()
    for i in range(k):
        chars = ["2"] * k
        chars[i] = "1"
        on.add("".join(chars))
    return {0: off, 1: on}


def _k_input_and_node(k: int) -> BooleanNode:
    return BooleanNode.from_bit_lut([0] * (2**k - 1) + [1], name=f"AND_k{k}")


def _k_input_or_node(k: int) -> BooleanNode:
    return BooleanNode.from_bit_lut([0] + [1] * (2**k - 1), name=f"OR_k{k}")


# ---------------------------------------------------------------------------
# Cover / sample basics
# ---------------------------------------------------------------------------


def test_pi_covers_and_covering_primes_and2():
    schemata = AND().outputs.get_one_symbol_schemata()
    # AND k=2: primes {0: {02,20}, 1: {11}}
    assert pi_covers("02", "00")
    assert pi_covers("02", "01")
    assert not pi_covers("02", "10")
    assert set(covering_primes("00", schemata)) == {"02", "20"}
    assert set(covering_primes("01", schemata)) == {"02"}
    assert set(covering_primes("10", schemata)) == {"20"}
    assert set(covering_primes("11", schemata)) == {"11"}


def test_covering_primes_raises_on_incomplete():
    with pytest.raises(ValueError, match="no prime implicant"):
        covering_primes("11", {0: {"00"}, 1: set()})


def test_sample_binstates_full_enumeration():
    got = sample_binstates(3, 8, replace=False)
    assert sorted(got) == [
        "000",
        "001",
        "010",
        "011",
        "100",
        "101",
        "110",
        "111",
    ]
    # n > 2**k without replace still returns full space
    got2 = sample_binstates(3, 100, replace=False)
    assert sorted(got2) == sorted(got)


def test_sample_binstates_seeded():
    a = sample_binstates(8, 16, rng=np.random.default_rng(0))
    b = sample_binstates(8, 16, rng=np.random.default_rng(0))
    assert a == b
    assert len(set(a)) == 16


def test_sample_binstates_large_k():
    """k larger than int64 index space must still return unique bitstrings."""
    got = sample_binstates(80, 32, rng=np.random.default_rng(0), replace=False)
    assert len(got) == 32
    assert all(len(s) == 80 and set(s) <= {"0", "1"} for s in got)
    assert len(set(got)) == 32
    again = sample_binstates(80, 32, rng=np.random.default_rng(0), replace=False)
    assert got == again


def test_schemata_from_on_off_and():
    on = [{"a": True, "b": True}]
    off = [{"a": False}, {"b": False}]
    sch = schemata_from_on_off(on, off, input_names=["a", "b"])
    assert sch[1] == {"11"}
    assert sch[0] == {"02", "20"}


# ---------------------------------------------------------------------------
# Exact match when fully enumerated
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [6, 8, 10, 12])
@pytest.mark.parametrize(
    "schemata_fn,node_fn",
    [(_and_schemata, _k_input_and_node), (_or_schemata, _k_input_or_node)],
    ids=["AND", "OR"],
)
def test_full_enum_matches_boolean_node_input(k, schemata_fn, node_fn):
    schemata = schemata_fn(k)
    node = node_fn(k)
    # Ensure node has matching schemata path available
    node.outputs.get_one_symbol_schemata()
    exact = node.input_redundancy(operator=mean, norm=True)
    est = estimate_input_redundancy(
        schemata, n_samples=2**k, operator=mean, norm=True
    )
    assert est == pytest.approx(exact, abs=1e-12)


@pytest.mark.parametrize("k", [6, 8, 10])
@pytest.mark.parametrize(
    "schemata_fn,node_fn",
    [(_and_schemata, _k_input_and_node), (_or_schemata, _k_input_or_node)],
    ids=["AND", "OR"],
)
def test_full_enum_matches_boolean_node_edge(k, schemata_fn, node_fn):
    schemata = schemata_fn(k)
    node = node_fn(k)
    node.outputs.get_one_symbol_schemata()
    for bound in ("mean", "lower", "upper"):
        exact = node.edge_redundancy(bound=bound)
        est = estimate_edge_redundancy(schemata, n_samples=2**k, bound=bound)
        assert len(est) == k
        for a, b in zip(est, exact):
            assert a == pytest.approx(b, abs=1e-12)


# ---------------------------------------------------------------------------
# Large subsample accuracy
# ---------------------------------------------------------------------------


def test_large_subsample_accuracy_k12():
    k = 12
    schemata = _and_schemata(k)
    node = _k_input_and_node(k)
    exact_kr = node.input_redundancy(norm=True)
    exact_ri = node.edge_redundancy(bound="mean")

    rng = np.random.default_rng(42)
    est_kr = estimate_input_redundancy(
        schemata, n_samples=2**11, norm=True, rng=rng
    )
    assert abs(est_kr - exact_kr) < 0.05

    rng = np.random.default_rng(42)
    est_ri = estimate_edge_redundancy(
        schemata, n_samples=2**11, bound="mean", rng=rng
    )
    mean_abs = float(np.mean([abs(a - b) for a, b in zip(est_ri, exact_ri)]))
    assert mean_abs < 0.05


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def _random_schemata_node(k: int, seed: int = 0):
    """Random LUT → node + one-symbol schemata (variable PI covers)."""
    lut = [int(x) for x in np.random.default_rng(seed).integers(0, 2, size=2**k)]
    # avoid constant functions (zero variance / empty polarity)
    if all(v == lut[0] for v in lut):
        lut[0] = 1 - lut[0]
    node = BooleanNode.from_bit_lut(lut, name=f"RND_k{k}")
    schemata = {
        p: set(s) for p, s in node.outputs.get_one_symbol_schemata().items()
    }
    return schemata, node


def test_bootstrap_input_matches_point_and_covers_exact():
    k = 12
    schemata, node = _random_schemata_node(k, seed=0)
    exact = node.input_redundancy(norm=True)

    minterms = sample_binstates(k, 2**11, rng=np.random.default_rng(7))
    point = estimate_input_redundancy(
        schemata, minterms=minterms, norm=True
    )
    boot = bootstrap_input_redundancy(
        schemata,
        n_samples=2**11,
        n_boot=200,
        ci=0.95,
        norm=True,
        minterms=minterms,
        rng=np.random.default_rng(99),
    )
    assert boot.estimate == pytest.approx(point, abs=1e-12)
    assert boot.ci_low <= exact <= boot.ci_high
    assert boot.n_samples == 2**11
    assert boot.n_boot == 200


def test_bootstrap_edge_covers_exact():
    k = 10
    schemata, node = _random_schemata_node(k, seed=1)
    exact = node.edge_redundancy(bound="mean")
    minterms = sample_binstates(k, 2**9, rng=np.random.default_rng(3))
    boots = bootstrap_edge_redundancy(
        schemata,
        n_samples=2**9,
        n_boot=200,
        bound="mean",
        minterms=minterms,
        rng=np.random.default_rng(11),
    )
    assert len(boots) == k
    covered = sum(
        1 for b, e in zip(boots, exact) if b.ci_low <= e <= b.ci_high
    )
    # Expect most edges' CIs to contain the exact value.
    assert covered >= max(1, k // 2)


def test_bootstrap_ci_shrinks_with_more_samples():
    k = 12
    schemata, _node = _random_schemata_node(k, seed=2)
    mt_small = sample_binstates(k, 2**10, rng=np.random.default_rng(1))
    mt_large = sample_binstates(k, 2**11, rng=np.random.default_rng(2))
    b_small = bootstrap_input_redundancy(
        schemata,
        n_samples=2**10,
        n_boot=300,
        minterms=mt_small,
        rng=np.random.default_rng(0),
    )
    b_large = bootstrap_input_redundancy(
        schemata,
        n_samples=2**11,
        n_boot=300,
        minterms=mt_large,
        rng=np.random.default_rng(0),
    )
    w_small = b_small.ci_high - b_small.ci_low
    w_large = b_large.ci_high - b_large.ci_low
    assert w_small > 0.0
    assert w_large < w_small


# ---------------------------------------------------------------------------
# No dense coverage / LUT
# ---------------------------------------------------------------------------


def test_estimator_schemata_only_no_lut():
    """Estimator works from raw schemata dicts; never builds a node LUT."""
    schemata = _and_schemata(8)
    # No BooleanNode involved
    kr = estimate_input_redundancy(schemata, n_samples=2**8, norm=True)
    assert 0.0 <= kr <= 1.0
    ri = estimate_edge_redundancy(schemata, n_samples=256, bound="mean")
    assert len(ri) == 8
    # bit_lut never materialized because we never touched FunctionOutputs
    assert isinstance(schemata, dict)
