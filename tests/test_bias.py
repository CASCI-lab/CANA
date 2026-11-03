# -*- coding: utf-8 -*-
"""Tests for :mod:`cana.function_properties.bias`."""
from __future__ import annotations

import numpy as np
import pytest

from cana.boolean_node import BooleanNode
from cana.datasets.bools import AND, OR, XOR
from cana.function_outputs.bit_lut import MAX_DENSE_LUT_K
from cana.function_properties.bias import (
    bias_from_bit_lut,
    bias_from_schemata,
    bootstrap_bias,
)
from cana.function_properties.redundancy import sample_pi_coverage


def _and_schemata(k: int):
    on = {"1" * k}
    off = set()
    for i in range(k):
        chars = ["2"] * k
        chars[i] = "0"
        off.add("".join(chars))
    return {0: off, 1: on}


@pytest.mark.parametrize(
    "node_fn,expected",
    [
        (AND, 0.25),
        (OR, 0.75),
        (XOR, 0.5),
    ],
)
def test_bias_lut_and_schemata_agree(node_fn, expected):
    node = node_fn()
    lut_bias = bias_from_bit_lut(node.outputs.get_bit_lut())
    assert lut_bias == pytest.approx(expected, abs=1e-12)
    assert node.bias_from_bit_lut() == pytest.approx(expected, abs=1e-12)

    schemata = node.outputs.get_one_symbol_schemata()
    assert bias_from_schemata(schemata, k=node.k) == pytest.approx(
        expected, abs=1e-12
    )
    assert node.bias_from_schemata() == pytest.approx(expected, abs=1e-12)
    assert node.bias() == pytest.approx(expected, abs=1e-12)


def test_sampled_and_bias_near_exact():
    k = 10
    schemata = _and_schemata(k)
    sampled = sample_pi_coverage(
        schemata, 2**9, rng=np.random.default_rng(4), k=k
    )
    est = bias_from_schemata(
        schemata,
        k=k,
        exact_k_max=-1,
        minterms=sampled.minterms,
    )
    # With n=512 on a 1/1024 ON function, estimate is typically 0 or small.
    assert 0.0 <= est <= 0.05


def test_bootstrap_bias_from_sampled_pi_coverage():
    # Balanced LUT so a half-space sample still yields a CI that covers exact.
    k = 8
    lut = [int(x) for x in np.random.default_rng(2).integers(0, 2, size=2**k)]
    if all(v == lut[0] for v in lut):
        lut[0] = 1 - lut[0]
    node = BooleanNode.from_bit_lut(lut, name="RND_k8")
    exact = node.bias_from_bit_lut()
    schemata = {
        p: set(s) for p, s in node.outputs.get_one_symbol_schemata().items()
    }
    sampled = sample_pi_coverage(
        schemata, 2**7, rng=np.random.default_rng(6), k=k
    )
    point = bias_from_schemata(
        schemata, k=k, exact_k_max=-1, minterms=sampled.minterms
    )
    boot = bootstrap_bias(
        sampled, n_boot=200, rng=np.random.default_rng(8)
    )
    assert boot.estimate == pytest.approx(point, abs=1e-12)
    assert boot.n_samples == sampled.n_samples
    assert boot.ci_low <= exact <= boot.ci_high


def test_high_k_constant_one_bias_via_sampling():
    k = MAX_DENSE_LUT_K + 1
    n = BooleanNode.from_one_symbol_schemata({1: {"2" * k}}, k=k)
    # Known PI cover; verify_one_symbol_schemata would need a dense LUT.
    n.outputs.one_symbol_is_prime_implicant = True
    assert n.outputs.bit_lut is None
    with pytest.raises(ValueError, match="MAX_DENSE_LUT_K"):
        n.outputs.get_bit_lut()

    b = n.bias(exact_k_max=-1, n_samples=64, rng=np.random.default_rng(0))
    assert b == pytest.approx(1.0, abs=1e-12)
