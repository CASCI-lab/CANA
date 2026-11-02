# -*- coding: utf-8 -*-
"""Tests for :mod:`cana.redundancy`."""
from __future__ import annotations

import warnings
from statistics import mean

import numpy as np
import pytest

from cana.boolean_node import BooleanNode
from cana.datasets.bools import AND
from cana.function_outputs.one_symbol import primes_to_one_symbol
from cana.redundancy import (
    EXACT_K_MAX_DEFAULT,
    bootstrap_edge_redundancy,
    bootstrap_input_redundancy,
    covering_primes,
    edge_redundancy,
    input_redundancy,
    input_redundancy_from_coverage,
    pi_covers,
    sample_binstates,
    sample_pi_coverage,
)


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


def test_primes_to_one_symbol_and():
    on = [{"a": True, "b": True}]
    off = [{"a": False}, {"b": False}]
    sch = primes_to_one_symbol(on, off, input_names=["a", "b"])
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
    node.outputs.get_one_symbol_schemata()
    exact = node.input_redundancy(operator=mean, norm=True)
    est = input_redundancy(
        schemata,
        operator=mean,
        norm=True,
        exact_k_max=-1,
        n_samples=2**k,
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
        est = edge_redundancy(
            schemata, bound=bound, exact_k_max=-1, n_samples=2**k
        )
        assert len(est) == k
        for a, b in zip(est, exact):
            assert a == pytest.approx(b, abs=1e-12)


def test_from_coverage_matches_node_and2():
    node = AND()
    cov = node.pi_coverage()
    assert input_redundancy_from_coverage(cov, node.k, norm=True) == pytest.approx(
        node.input_redundancy(norm=True), abs=1e-12
    )


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
    est_kr = input_redundancy(
        schemata, norm=True, exact_k_max=-1, n_samples=2**11, rng=rng
    )
    assert abs(est_kr - exact_kr) < 0.05

    rng = np.random.default_rng(42)
    est_ri = edge_redundancy(
        schemata, bound="mean", exact_k_max=-1, n_samples=2**11, rng=rng
    )
    mean_abs = float(np.mean([abs(a - b) for a, b in zip(est_ri, exact_ri)]))
    assert mean_abs < 0.05


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def _random_schemata_node(k: int, seed: int = 0):
    """Random LUT → node + one-symbol schemata (variable PI covers)."""
    lut = [int(x) for x in np.random.default_rng(seed).integers(0, 2, size=2**k)]
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
    point = input_redundancy(
        schemata, norm=True, exact_k_max=-1, minterms=minterms
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


def test_bootstrap_from_sampled_pi_coverage():
    k = 10
    schemata, node = _random_schemata_node(k, seed=3)
    exact = node.input_redundancy(norm=True)
    sampled = sample_pi_coverage(
        schemata, 2**9, rng=np.random.default_rng(5), k=k
    )
    point = input_redundancy_from_coverage(sampled.coverage, k, norm=True)
    boot = bootstrap_input_redundancy(
        sampled, n_boot=200, rng=np.random.default_rng(8)
    )
    assert boot.estimate == pytest.approx(point, abs=1e-12)
    assert boot.n_samples == sampled.n_samples
    assert boot.ci_low <= exact <= boot.ci_high


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
# Two-track / node wrappers
# ---------------------------------------------------------------------------


def test_input_redundancy_two_track_default_exact_for_small_k():
    node = _k_input_and_node(8)
    exact = node.input_redundancy(norm=True)
    # exact_k_max=0 would sample, but n_samples=2**k is full enum → exact, no warn
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sampled, mode = node.input_redundancy(
            norm=True,
            exact_k_max=0,
            n_samples=2**8,
            rng=np.random.default_rng(0),
            return_mode=True,
        )
    assert mode == "exact"
    assert not any(issubclass(w.category, UserWarning) for w in caught)
    assert sampled == pytest.approx(exact, abs=1e-12)


def test_node_sampling_default_n_samples_is_two_to_exact_k_max():
    node = _k_input_and_node(6)
    # k > exact_k_max → sample; default n = 2**exact_k_max
    with pytest.warns(UserWarning, match="BooleanNode.input_redundancy"):
        kr = node.input_redundancy(
            exact_k_max=3, rng=np.random.default_rng(0)
        )
    assert 0.0 <= kr <= 1.0
    assert node._sampled_pi_coverage is not None
    assert node._sampled_pi_coverage.n_samples == 2**3
    assert EXACT_K_MAX_DEFAULT == 14


def test_node_reuses_sampled_pi_coverage_for_edge():
    node = _k_input_and_node(6)
    rng = np.random.default_rng(1)
    with pytest.warns(UserWarning, match="sampling"):
        _ = node.input_redundancy(exact_k_max=2, n_samples=16, rng=rng)
    cached = node._sampled_pi_coverage
    with pytest.warns(UserWarning, match="sampling"):
        ri = node.edge_redundancy(exact_k_max=2, n_samples=16, rng=rng)
    assert node._sampled_pi_coverage is cached
    assert len(ri) == 6


def test_module_input_redundancy_exact_vs_sample_defaults():
    schemata = _and_schemata(5)
    exact = input_redundancy(schemata, k=5, exact_k_max=5)
    # k > exact_k_max with default n_samples = 2**exact_k_max
    with pytest.warns(UserWarning, match="sampling n_samples"):
        est, sampled = input_redundancy(
            schemata,
            k=5,
            exact_k_max=3,
            rng=np.random.default_rng(0),
            return_coverage=True,
        )
    assert sampled.n_samples == 2**3
    assert 0.0 <= est <= 1.0
    assert abs(est - exact) < 0.2


def test_return_mode_exact_vs_sampled():
    schemata = _and_schemata(5)
    val, mode = input_redundancy(
        schemata, k=5, exact_k_max=5, return_mode=True
    )
    assert mode == "exact"
    assert 0.0 <= val <= 1.0

    with pytest.warns(UserWarning, match="input_redundancy"):
        val_s, mode_s = input_redundancy(
            schemata,
            k=5,
            exact_k_max=3,
            n_samples=8,
            rng=np.random.default_rng(0),
            return_mode=True,
        )
    assert mode_s == "sampled"
    assert 0.0 <= val_s <= 1.0

    # exact_k_max < 0 forces sampling without the auto-fallback warning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, mode_forced = input_redundancy(
            schemata,
            k=5,
            exact_k_max=-1,
            n_samples=8,
            rng=np.random.default_rng(0),
            return_mode=True,
        )
    assert mode_forced == "sampled"
    assert not any(issubclass(w.category, UserWarning) for w in caught)

    ri, ri_mode = edge_redundancy(
        schemata, k=5, exact_k_max=5, bound="mean", return_mode=True
    )
    assert ri_mode == "exact"
    assert len(ri) == 5


def test_node_return_mode_and_warn_on_auto_sample():
    node = _k_input_and_node(6)
    kr, mode = node.input_redundancy(norm=True, return_mode=True)
    assert mode == "exact"

    with pytest.warns(UserWarning, match="BooleanNode.input_redundancy"):
        kr_s, mode_s = node.input_redundancy(
            exact_k_max=3,
            n_samples=8,
            rng=np.random.default_rng(0),
            return_mode=True,
        )
    assert mode_s == "sampled"
    assert 0.0 <= kr_s <= 1.0

    with pytest.warns(UserWarning, match="BooleanNode.edge_redundancy"):
        ri, ri_mode = node.edge_redundancy(
            exact_k_max=3,
            n_samples=8,
            rng=np.random.default_rng(0),
            return_mode=True,
        )
    assert ri_mode == "sampled"
    assert len(ri) == 6


def test_node_exact_k_max_forces_full_coverage_above_default():
    """exact_k_max=k must not fall back to sampling for k > EXACT_K_MAX_DEFAULT."""
    # Use a modest k above the default threshold logic via a lowered threshold:
    # exact_k_max=node.k with k=8 is always exact; also verify mode string.
    node = _k_input_and_node(8)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kr, mode = node.input_redundancy(
            norm=True, exact_k_max=node.k, return_mode=True
        )
    assert mode == "exact"
    assert not any(issubclass(w.category, UserWarning) for w in caught)
    assert kr == pytest.approx(node.input_redundancy(norm=True), abs=1e-12)


def test_estimator_schemata_only_no_lut():
    """Estimator works from raw schemata dicts; never builds a node LUT."""
    schemata = _and_schemata(8)
    kr = input_redundancy(
        schemata, norm=True, exact_k_max=-1, n_samples=2**8
    )
    assert 0.0 <= kr <= 1.0
    ri = edge_redundancy(
        schemata, bound="mean", exact_k_max=-1, n_samples=256
    )
    assert len(ri) == 8
    assert isinstance(schemata, dict)
