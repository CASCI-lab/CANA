# -*- coding: utf-8 -*-
"""Tests for :mod:`cana.function_properties.sensitivity` bootstrap helpers."""
from __future__ import annotations

import numpy as np
import pytest

from cana.boolean_node import BooleanNode
from cana.function_properties.redundancy import sample_binstates, sample_pi_coverage
from cana.function_properties.sensitivity import (
    activities_from_schemata,
    bootstrap_activities,
    bootstrap_sensitivity,
    sensitivity_from_schemata,
)


def _and_schemata(k: int):
    """One-symbol cover for k-input AND (no LUT / QM)."""
    on = {"1" * k}
    off = set()
    for i in range(k):
        chars = ["2"] * k
        chars[i] = "0"
        off.add("".join(chars))
    return {0: off, 1: on}


def _and_node(k: int) -> BooleanNode:
    return BooleanNode.from_bit_lut([0] * (2**k - 1) + [1], name=f"AND_k{k}")


def test_bootstrap_activities_match_point_and_cover_exact():
    k = 10
    schemata = _and_schemata(k)
    exact = _and_node(k).activities_from_bit_lut()
    minterms = sample_binstates(k, 2**9, rng=np.random.default_rng(7))
    point = activities_from_schemata(
        schemata, exact_k_max=-1, minterms=minterms
    )
    boots = bootstrap_activities(
        schemata,
        n_samples=2**9,
        n_boot=200,
        ci=0.95,
        minterms=minterms,
        rng=np.random.default_rng(99),
    )
    assert len(boots) == k
    for b, p in zip(boots, point):
        assert b.estimate == pytest.approx(p, abs=1e-12)
    covered = sum(
        1 for b, e in zip(boots, exact) if b.ci_low <= e <= b.ci_high
    )
    assert covered >= max(1, k // 2)
    assert boots[0].n_samples == 2**9
    assert boots[0].n_boot == 200


def test_bootstrap_sensitivity_matches_point_and_covers_exact():
    k = 12
    schemata = _and_schemata(k)
    exact = _and_node(k).sensitivity_from_bit_lut(norm=False)
    minterms = sample_binstates(k, 2**11, rng=np.random.default_rng(3))
    point = sensitivity_from_schemata(
        schemata, exact_k_max=-1, minterms=minterms
    )
    boot = bootstrap_sensitivity(
        schemata,
        n_samples=2**11,
        n_boot=200,
        ci=0.95,
        norm=False,
        minterms=minterms,
        rng=np.random.default_rng(11),
    )
    assert boot.estimate == pytest.approx(point, abs=1e-12)
    assert boot.ci_low <= exact <= boot.ci_high
    assert boot.n_samples == 2**11
    assert boot.n_boot == 200


def test_bootstrap_sensitivity_from_sampled_pi_coverage():
    k = 10
    schemata = _and_schemata(k)
    exact = _and_node(k).sensitivity_from_bit_lut(norm=False)
    sampled = sample_pi_coverage(
        schemata, 2**9, rng=np.random.default_rng(5), k=k
    )
    point = sensitivity_from_schemata(
        schemata,
        exact_k_max=-1,
        minterms=sampled.minterms,
    )
    boot = bootstrap_sensitivity(
        sampled, n_boot=200, rng=np.random.default_rng(8)
    )
    assert boot.estimate == pytest.approx(point, abs=1e-12)
    assert boot.n_samples == sampled.n_samples
    assert boot.ci_low <= exact <= boot.ci_high


def test_bootstrap_sensitivity_ci_shrinks_with_more_samples():
    # Dense random LUT at modest k (schemata cheap; CI shrink is stable).
    k = 8
    lut = [int(x) for x in np.random.default_rng(2).integers(0, 2, size=2**k)]
    if all(v == lut[0] for v in lut):
        lut[0] = 1 - lut[0]
    node = BooleanNode.from_bit_lut(lut, name="RND_k8")
    schemata = {
        p: set(s) for p, s in node.outputs.get_one_symbol_schemata().items()
    }
    mt_large = sample_binstates(k, 2**7, rng=np.random.default_rng(2))
    mt_small = mt_large[: 2**6]
    b_small = bootstrap_sensitivity(
        schemata,
        n_samples=len(mt_small),
        n_boot=400,
        minterms=mt_small,
        rng=np.random.default_rng(0),
    )
    b_large = bootstrap_sensitivity(
        schemata,
        n_samples=len(mt_large),
        n_boot=400,
        minterms=mt_large,
        rng=np.random.default_rng(0),
    )
    w_small = b_small.ci_high - b_small.ci_low
    w_large = b_large.ci_high - b_large.ci_low
    assert w_small > 0.0
    assert w_large < w_small


def test_bootstrap_activities_from_sampled_pi_coverage():
    k = 8
    schemata = _and_schemata(k)
    sampled = sample_pi_coverage(
        schemata, 2**7, rng=np.random.default_rng(6), k=k
    )
    point = activities_from_schemata(
        schemata, exact_k_max=-1, minterms=sampled.minterms
    )
    boots = bootstrap_activities(
        sampled, n_boot=100, rng=np.random.default_rng(7)
    )
    assert len(boots) == k
    for b, p in zip(boots, point):
        assert b.estimate == pytest.approx(p, abs=1e-12)
