# -*- coding: utf-8 -*-
#
# Tests for ``boolean_node.py``
# These tests were manually calculated by Luis M. Rocha and implemented by Rion B. Correia.
#
from math import isclose

from cana.datasets.bools import CONTRADICTION, AND, OR, XOR, COPYx1, RULE90, RULE110
from cana.boolean_node import BooleanNode
import cana.function_properties.sensitivity as sensitivity
import numpy as np
import pytest


#
# Test Input Redundancy
#

def test_input_redundancy_constant():
    """Test Input Redundancy - constant"""
    n = BooleanNode(k=1, outputs=list("00"))
    k_r, true_k_r = n.input_redundancy(norm=False), 1
    assert (k_r == true_k_r), ('Input Redundancy (mean) for CONSTANT node does not match. %s != %s' % (k_r, true_k_r))

    k_r, true_k_r = n.input_redundancy(norm=True), 1
    assert (k_r == true_k_r), ('Input Redundancy (mean, normed) for CONSTANT node does not match. %s != %s' % (k_r, true_k_r))

def test_input_redundancy_identity():
    """Test Input Redundancy - identity"""
    n = BooleanNode(k=1, outputs=list("01"))
    k_r, true_k_r = n.input_redundancy(norm=False), 0
    assert (k_r == true_k_r), ('Input Redundancy (mean) for identity node does not match. %s != %s' % (k_r, true_k_r))

    k_r, true_k_r = n.input_redundancy(norm=True), 0
    assert (k_r == true_k_r), ('Input Redundancy (mean, normed) for identity node does not match. %s != %s' % (k_r, true_k_r))

# AND
def test_input_redundancy_AND():
    """Test Input Redundancy - AND"""
    n = AND()
    k_r, true_k_r = n.input_redundancy(norm=False), (3 / 4)
    assert (k_r == true_k_r), ('Input Redundancy (mean) for AND node does not match. %s != %s' % (k_r, true_k_r))

    k_r, true_k_r = n.input_redundancy(norm=True), (3 / 4) / 2
    assert (k_r == true_k_r), ('Input Redundancy (mean, normed) for AND node does not match. %s != %s' % (k_r, true_k_r))


# OR
def test_input_redundancy_OR():
    """Test Input Redundancy - OR"""
    n = OR()
    k_r, true_k_r = n.input_redundancy(norm=False), 3 / 4
    assert (k_r == true_k_r), ('Input Redundancy (mean) for OR node does not match. %s != %s' % (k_r, true_k_r))

    k_r, true_k_r = n.input_redundancy(norm=True), (3 / 4) / 2
    assert (k_r == true_k_r), ('Input Redundancy (mean, normed) for OR node does not match. %s != %s' % (k_r, true_k_r))


# XOR
def test_input_redundancy_XOR():
    """Test Input Redundancy - XOR"""
    n = XOR()
    k_r, true_k_r = n.input_redundancy(norm=False), 0
    assert (k_r == true_k_r), ('Input Redundancy (mean) for XOR node does not match. %s != %s' % (k_r, true_k_r))

    k_r, true_k_r = n.input_redundancy(norm=True), 0
    assert (k_r == true_k_r), ('Input Redundancy (mean, normed) for XOR node does not match. %s != %s' % (k_r, true_k_r))


# CONTRADICTION
def test_input_redundancy_CONTRADICTION():
    """Test Input Redundancy - CONTRADICTION"""
    n = CONTRADICTION()
    k_r, true_k_r = n.input_redundancy(norm=False), 2.
    assert (k_r == true_k_r), ('Input Redundancy (mean) for CONTRADICTION node does not match. %s != %s' % (k_r, true_k_r))

    k_r, true_k_r = n.input_redundancy(norm=True), 1.
    assert (k_r == true_k_r), ('Input Redundancy (mean, normed) for CONTRADICTION node does not match. %s != %s' % (k_r, true_k_r))


# COPYx1
def test_input_redundancy_COPYx1():
    """Test Input Redundancy - COPYx1"""
    n = COPYx1()
    k_r, true_k_r = n.input_redundancy(norm=False), 1.
    assert (k_r == true_k_r), ('Input Redundancy (upper) for COPYx1 node does not match. %s != %s' % (k_r, true_k_r))

    k_r, true_k_r = n.input_redundancy(norm=True), 1 / 2
    assert (k_r == true_k_r), ('Input Redundancy (upper, normed) for COPYx1 node does not match. %s != %s' % (k_r, true_k_r))


# RULE 90
def test_input_redundancy_RULE90():
    """Test Input Redundancy - RULE90"""
    n = RULE90()
    k_r, true_k_r = n.input_redundancy(norm=False), 8 / 8
    assert (k_r == true_k_r), ('Input Redundancy (upper) for RULE90 node does not match. %s != %s' % (k_r, true_k_r))

    k_r, true_k_r = n.input_redundancy(norm=True), (8 / 8) / 3
    assert (k_r == true_k_r), ('Input Redundancy (upper, normed) for RULE90 node does not match. %s != %s' % (k_r, true_k_r))


# RULE 110
def test_input_redundancy_RULE110():
    """Test Input Redundancy - RULE110"""
    n = RULE110()
    k_r, true_k_r = n.input_redundancy(norm=False), 7 / 8
    assert (k_r == true_k_r), ('Input Redundancy (upper) for RULE110 node does not match. %s != %s' % (k_r, true_k_r))

    k_r, true_k_r = n.input_redundancy(norm=True), (7 / 8) / 3
    assert (k_r == true_k_r), ('Input Redundancy (upper, normed) for RULE110 node does not match. %s != %s' % (k_r, true_k_r))


def _k_input_or(k):
    """k-input OR: output 0 only on the all-zero input."""
    return BooleanNode.from_bit_lut([0] + [1] * (2**k - 1), name="OR_k%d" % k)


def _k_input_and(k):
    """k-input AND: output 1 only on the all-one input."""
    return BooleanNode.from_bit_lut([0] * (2**k - 1) + [1], name="AND_k%d" % k)


def _and_or_mean_input_redundancy(k):
    """Mean input redundancy for k-input AND or OR: (2^k-1)/2^k * (k-1)."""
    return ((2**k - 1) / (2**k)) * (k - 1)


def _and_or_mean_edge_redundancy(k):
    """Mean edge redundancy for each input of k-input AND or OR."""
    return ((2**k - 1) / (2**k)) * ((k - 1) / k)


@pytest.mark.parametrize("k", range(1, 7))
@pytest.mark.parametrize("factory", [_k_input_or, _k_input_and], ids=["OR", "AND"])
def test_input_redundancy_AND_OR_general_k(factory, k):
    """Mean input redundancy for k-input AND/OR is (2^k-1)/2^k * (k-1)."""
    n = factory(k)
    expected = _and_or_mean_input_redundancy(k)
    k_r = n.input_redundancy(norm=False)
    assert isclose(k_r, expected), (
        "Input Redundancy (mean) for %s (k=%d) does not match. %s != %s"
        % (n.name, k, k_r, expected)
    )
    k_r_norm = n.input_redundancy(norm=True)
    assert isclose(k_r_norm, expected / k), (
        "Input Redundancy (mean, normed) for %s (k=%d) does not match. %s != %s"
        % (n.name, k, k_r_norm, expected / k)
    )


#
# Test Edge Redundancy
#

# AND
def test_edge_redundancy_AND():
    """Test Edge Redundancy - AND"""
    n = AND()
    r_ji, true_r_ji = n.edge_redundancy(bound='upper'), [1 / 2., 1 / 2]
    assert (r_ji == true_r_ji), ('Edge Redundancy (upper) for AND node does not match. %s != %s' % (r_ji, true_r_ji))
    r_ji, true_r_ji = n.edge_redundancy(bound='mean'), [3 / 8., 3 / 8]
    assert (r_ji == true_r_ji), ('Edge Redundancy (mean) for AND node does not match. %s != %s' % (r_ji, true_r_ji))
    r_ji, true_r_ji = n.edge_redundancy(bound='lower'), [1 / 4., 1 / 4]
    assert (r_ji == true_r_ji), ('Edge Redundancy (lower) for AND node does not match. %s != %s' % (r_ji, true_r_ji))
    #
    r_ji, true_r_ji = n.edge_redundancy(bound='tuple'), [(0.25, 0.5), (0.25, 0.5)]
    assert (r_ji == true_r_ji), ('Edge Redundancy (tuples) for AND node does not match. %s != %s' % (r_ji, true_r_ji))


# OR
def test_edge_redundancy_OR():
    """Test Edge Redundancy - OR"""
    n = OR()
    r_ji, true_r_ji = n.edge_redundancy(bound='upper'), [1 / 2., 1 / 2]
    assert (r_ji == true_r_ji), ('Edge Redundancy (upper) for OR node does not match. %s != %s' % (r_ji, true_r_ji))
    r_ji, true_r_ji = n.edge_redundancy(bound='mean'), [3 / 8., 3 / 8]
    assert (r_ji == true_r_ji), ('Edge Redundancy (mean) for OR node does not match. %s != %s' % (r_ji, true_r_ji))
    r_ji, true_r_ji = n.edge_redundancy(bound='lower'), [1 / 4., 1 / 4]
    assert (r_ji == true_r_ji), ('Edge Redundancy (lower) for OR node does not match. %s != %s' % (r_ji, true_r_ji))

    r_ji, true_r_ji = n.edge_redundancy(bound='tuple'), [(0.25, 0.5), (0.25, 0.5)]
    assert (r_ji == true_r_ji), ('Edge Redundancy (tuples) for OR node does not match. %s != %s' % (r_ji, true_r_ji))


@pytest.mark.parametrize("k", range(1, 7))
@pytest.mark.parametrize("factory", [_k_input_or, _k_input_and], ids=["OR", "AND"])
def test_edge_redundancy_AND_OR_general_k(factory, k):
    """Mean edge redundancy for each input of k-input AND/OR is (2^k-1)/2^k * (k-1)/k."""
    n = factory(k)
    expected = _and_or_mean_edge_redundancy(k)
    r_ji = n.edge_redundancy(bound="mean")
    assert len(r_ji) == k
    for i, r in enumerate(r_ji):
        assert isclose(r, expected), (
            "Edge Redundancy (mean) for %s input %d (k=%d) does not match. %s != %s"
            % (n.name, i, k, r, expected)
        )


# XOR
def test_edge_redundancy_XOR():
    """Test Edge Redundancy - XOR"""
    n = XOR()
    r_ji, true_r_ji = n.edge_redundancy(bound='upper'), [0, 0]
    assert (r_ji == true_r_ji), ('Edge Redundancy (upper) for XOR node does not match. %s != %s' % (r_ji, true_r_ji))
    r_ji, true_r_ji = n.edge_redundancy(bound='mean'), [0, 0]
    assert (r_ji == true_r_ji), ('Edge Redundancy (mean) for XOR node does not match. %s != %s' % (r_ji, true_r_ji))
    r_ji, true_r_ji = n.edge_redundancy(bound='lower'), [0, 0]
    assert (r_ji == true_r_ji), ('Edge Redundancy (lower) for XOR node does not match. %s != %s' % (r_ji, true_r_ji))

    r_ji, true_r_ji = n.edge_redundancy(bound='tuple'), [(0.0, 0.0), (0.0, 0.0)]
    assert (r_ji == true_r_ji), ('Edge Redundancy (tuples) for XOR node does not match. %s != %s' % (r_ji, true_r_ji))


# CONTRADICTION
def test_edge_redundancy_CONTRADICTION():
    """Test Edge Redundancy - CONTRADICTION"""
    n = CONTRADICTION()
    r_ji, true_r_ji = n.edge_redundancy(bound='upper'), [1., 1.]
    assert (r_ji == true_r_ji), ('Edge Redundancy (upper) for CONTRADICTION node does not match. %s != %s' % (r_ji, true_r_ji))
    r_ji, true_r_ji = n.edge_redundancy(bound='mean'), [1., 1.]
    assert (r_ji == true_r_ji), ('Edge Redundancy (mean) for CONTRADICTION node does not match. %s != %s' % (r_ji, true_r_ji))
    r_ji, true_r_ji = n.edge_redundancy(bound='lower'), [1., 1.]
    assert (r_ji == true_r_ji), ('Edge Redundancy (lower) for CONTRADICTION node does not match. %s != %s' % (r_ji, true_r_ji))

    r_ji, true_r_ji = n.edge_redundancy(bound='tuple'), [(1.0, 1.0), (1.0, 1.0)]
    assert (r_ji == true_r_ji), ('Edge Redundancy (tuples) for CONTRADICTION node does not match. %s != %s' % (r_ji, true_r_ji))


# COPYx1
def test_edge_redundancy_COPYx1():
    """Test Edge Redundancy - COPYx1"""
    n = COPYx1()
    r_ji, true_r_ji = n.edge_redundancy(bound='upper'), [0., 1.]
    assert (r_ji == true_r_ji), ('Edge Redundancy (upper) for COPYx1 node does not match. %s != %s' % (r_ji, true_r_ji))
    r_ji, true_r_ji = n.edge_redundancy(bound='mean'), [0., 1.]
    assert (r_ji == true_r_ji), ('Edge Redundancy (mean) for COPYx1 node does not match. %s != %s' % (r_ji, true_r_ji))
    r_ji, true_r_ji = n.edge_redundancy(bound='lower'), [0., 1.]
    assert (r_ji == true_r_ji), ('Edge Redundancy (lower) for COPYx1 node does not match. %s != %s' % (r_ji, true_r_ji))

    r_ji, true_r_ji = n.edge_redundancy(bound='tuple'), [(0.0, 0.0), (1.0, 1.0)]
    assert (r_ji == true_r_ji), ('Edge Redundancy (tuples) for COPYx1 node does not match. %s != %s' % (r_ji, true_r_ji))


# RULE 90
def test_edge_redundancy_RULE90():
    """Test Edge Redundancy - RULE90"""
    n = RULE90()
    r_ji, true_r_ji = n.edge_redundancy(bound='upper'), [0., 1., 0.]
    assert (r_ji == true_r_ji), ('Edge Redundancy (upper bound) for RULE90 node does not match. %s != %s' % (r_ji, true_r_ji))
    r_ji, true_r_ji = n.edge_redundancy(bound='mean'), [0., 1., 0.]
    assert (r_ji == true_r_ji), ('Edge Redundancy (mean) for RULE90 node does not match. %s != %s' % (r_ji, true_r_ji))
    r_ji, true_r_ji = n.edge_redundancy(bound='lower'), [0., 1., 0.]
    assert (r_ji == true_r_ji), ('Edge Redundancy (lower bound) for RULE90 node does not match. %s != %s' % (r_ji, true_r_ji))

    r_ji, true_r_ji = n.edge_redundancy(bound='tuple'), [(0.0, 0.0), (1.0, 1.0), (0.0, 0.0)]
    assert (r_ji == true_r_ji), ('Edge Redundancy (tuples) for RULE90 node does not match. %s != %s' % (r_ji, true_r_ji))


# RULE 110
def test_edge_redundancy_RULE110():
    """Test Edge Redundancy - RULE110"""
    n = RULE110()
    r_ji, true_r_ji = n.edge_redundancy(bound='upper'), [6 / 8, 2 / 8, 2 / 8]
    assert (r_ji == true_r_ji), ('Edge Redundancy (upper) for RULE110 node does not match. %s != %s' % (r_ji, true_r_ji))
    r_ji, true_r_ji = n.edge_redundancy(bound='mean'), [5 / 8, 1 / 8, 1 / 8]
    assert (r_ji == true_r_ji), ('Edge Redundancy (mean) for RULE110 node does not match. %s != %s' % (r_ji, true_r_ji))
    r_ji, true_r_ji = n.edge_redundancy(bound='lower'), [4 / 8., 0 / 8, 0 / 8]
    assert (r_ji == true_r_ji), ('Edge Redundancy (lower) for RULE110 node does not match. %s != %s' % (r_ji, true_r_ji))

    r_ji, true_r_ji = n.edge_redundancy(bound='tuple'), [(0.5, 0.75), (0.0, 0.25), (0.0, 0.25)]
    assert (r_ji == true_r_ji), ('Edge Redundancy (tuples) for RULE110 node does not match. %s != %s' % (r_ji, true_r_ji))


#
# Test Effective Connectivity
#

# AND
def test_effective_connectivity_AND():
    """Test Effective Connectivity - AND"""
    n = AND()
    k_e, true_k_e = n.effective_connectivity(norm=False), 5 / 4
    assert (k_e == true_k_e), ('Effective Connectivity (node,upper bound) for AND node does not match. %s != %s' % (k_e, true_k_e))

    k_e, true_k_e = n.effective_connectivity(norm=True), (5 / 4) / 2
    assert (k_e == true_k_e), ('Effective Connectivity (node,upper bound,normed) for AND node does not match. %s != %s' % (k_e, true_k_e))


# XOR
def test_effective_connectivity_XOR():
    """Test Effective Connectivity - XOR"""
    n = XOR()
    k_e, true_k_e = n.effective_connectivity(norm=False), 2.
    assert (k_e == true_k_e), ('Effective Connectivity (node,upper bound) for XOR node does not match. %s != %s' % (k_e, true_k_e))

    k_e, true_k_e = n.effective_connectivity(norm=True), 2. / 2
    assert (k_e == true_k_e), ('Effective Connectivity (node,upper bound,normed) for XOR node does not match. %s != %s' % (k_e, true_k_e))


# CONTRADICTION
def test_effective_connectivity_CONTRADICTION():
    """Test Effective Connectivity - CONTRADICTION"""
    n = CONTRADICTION()
    k_e, true_k_e = n.effective_connectivity(norm=False), 0
    assert (k_e == true_k_e), ('Effective Connectivity (node,upper bound) for CONTRADICTION node does not match. %s != %s' % (k_e, true_k_e))

    k_e, true_k_e = n.effective_connectivity(norm=True), 0
    assert (k_e == true_k_e), ('Effective Connectivity (node,upper bound,normed) for CONTRADICTION node does not match. %s != %s' % (k_e, true_k_e))


# COPYx1
def test_effective_connectivity_COPYx1():
    """Test Effective Connectivity - COPYx1"""
    n = COPYx1()
    k_e, true_k_e = n.effective_connectivity(norm=False), 1
    assert (k_e == true_k_e), ('Effective Connectivity (node,upper bound) for COPYx1 node does not match. %s != %s' % (k_e, true_k_e))

    k_e, true_k_e = n.effective_connectivity(norm=True), 1 / 2
    assert (k_e == true_k_e), ('Effective Connectivity (node,upper bound,normed) for COPYx1 node does not match. %s != %s' % (k_e, true_k_e))


# RULE90
def test_effective_connectivity_RULE90():
    """Test Effective Connectivity - RULE90"""
    n = RULE90()
    k_e, true_k_e = n.effective_connectivity(norm=False), 3 - 1
    assert (k_e == true_k_e), ('Effective Connectivity (node,upper bound) for RULE90 node does not match. %s != %s' % (k_e, true_k_e))

    k_e, true_k_e = n.effective_connectivity(norm=True), (3 - 1) / 3
    assert (k_e == true_k_e), ('Effective Connectivity (node,upper bound,normed) for RULE90 node does not match. %s != %s' % (k_e, true_k_e))


# RULE110
def test_effective_connectivity_RULE110():
    """Test Effective Connectivity - RULE110"""
    n = RULE110()
    k_e, true_k_e = n.effective_connectivity(norm=False), 3 - (7 / 8)
    assert (k_e == true_k_e), ('Effective Connectivity (node,upper bound) for RULE110 node does not match. %s != %s' % (k_e, true_k_e))

    k_e, true_k_e = n.effective_connectivity(norm=True), (3 - (7 / 8)) / 3
    assert (k_e == true_k_e), ('Effective Connectivity (node,upper bound,normed) for RULE110 node does not match. %s != %s' % (k_e, true_k_e))


#
# Test Edge Effectiveness
#

# AND
def test_edge_effectiveness_AND():
    """Test Edge Effectiveness - AND"""
    n = AND()
    e_ji, true_e_ji = n.edge_effectiveness(bound='upper'), [1 - (2 / 4), 1 - (2 / 4)]
    assert (e_ji == true_e_ji), ('Input Redundancy (input,upper bound) for AND node does not match. %s != %s' % (e_ji, true_e_ji))
    e_ji, true_e_ji = n.edge_effectiveness(bound='mean'), [1 - (3 / 8), 1 - (3 / 8)]
    assert (e_ji == true_e_ji), ('Input Redundancy (input,mean) for AND node does not match. %s != %s' % (e_ji, true_e_ji))
    e_ji, true_e_ji = n.edge_effectiveness(bound='lower'), [1 - (1 / 4), 1 - (1 / 4)]
    assert (e_ji == true_e_ji), ('Input Redundancy (input,lower bound) for AND node does not match. %s != %s' % (e_ji, true_e_ji))


# XOR
def test_edge_effectiveness_XOR():
    """Test Edge Effectiveness - XOR"""
    n = XOR()
    e_ji, true_e_ji = n.edge_effectiveness(bound='upper'), [1 - (0), 1 - (0)]
    assert (e_ji == true_e_ji), ('Input Redundancy (input,upper bound) for XOR node does not match. %s != %s' % (e_ji, true_e_ji))
    e_ji, true_e_ji = n.edge_effectiveness(bound='mean'), [1 - (0), 1 - (0)]
    assert (e_ji == true_e_ji), ('Input Redundancy (input,mean) for XOR node does not match. %s != %s' % (e_ji, true_e_ji))
    e_ji, true_e_ji = n.edge_effectiveness(bound='lower'), [1 - (0), 1 - (0)]
    assert (e_ji == true_e_ji), ('Input Redundancy (input,lower bound) for XOR node does not match. %s != %s' % (e_ji, true_e_ji))


# CONTRADICTION
def test_edge_effectiveness_CONTRADICTION():
    """Test Edge Effectiveness - CONTRADICTION"""
    n = CONTRADICTION()
    e_ji, true_e_ji = n.edge_effectiveness(bound='upper'), [0, 0]
    assert (e_ji == true_e_ji), ('Input Redundancy (input,upper bound) for CONTRADICTION node does not match. %s != %s' % (e_ji, true_e_ji))
    e_ji, true_e_ji = n.edge_effectiveness(bound='mean'), [0, 0]
    assert (e_ji == true_e_ji), ('Input Redundancy (input,mean) for CONTRADICTION node does not match. %s != %s' % (e_ji, true_e_ji))
    e_ji, true_e_ji = n.edge_effectiveness(bound='lower'), [0, 0]
    assert (e_ji == true_e_ji), ('Input Redundancy (input,lower bound) for CONTRADICTION node does not match. %s != %s' % (e_ji, true_e_ji))


# COPYx1
def test_edge_effectiveness_COPYx1():
    """Test Edge Effectiveness - COPYx1"""
    n = COPYx1()
    e_ji, true_e_ji = n.edge_effectiveness(bound='upper'), [1, 0]
    assert (e_ji == true_e_ji), ('Input Redundancy (input,upper bound) for COPYx1 node does not match. %s != %s' % (e_ji, true_e_ji))
    e_ji, true_e_ji = n.edge_effectiveness(bound='mean'), [1, 0]
    assert (e_ji == true_e_ji), ('Input Redundancy (input,mean) for COPYx1 node does not match. %s != %s' % (e_ji, true_e_ji))
    e_ji, true_e_ji = n.edge_effectiveness(bound='lower'), [1, 0]
    assert (e_ji == true_e_ji), ('Input Redundancy (input,lower bound) for COPYx1 node does not match. %s != %s' % (e_ji, true_e_ji))


# RULE90
def test_edge_effectiveness_RULE90():
    """Test Edge Effectiveness - RULE90"""
    n = RULE90()
    e_ji, true_e_ji = n.edge_effectiveness(bound='upper'), [1, 0, 1]
    assert (e_ji == true_e_ji), ('Input Redundancy (input,upper bound) for RULE90 node does not match. %s != %s' % (e_ji, true_e_ji))
    e_ji, true_e_ji = n.edge_effectiveness(bound='mean'), [1, 0, 1]
    assert (e_ji == true_e_ji), ('Input Redundancy (input,mean) for RULE90 node does not match. %s != %s' % (e_ji, true_e_ji))
    e_ji, true_e_ji = n.edge_effectiveness(bound='lower'), [1, 0, 1]
    assert (e_ji == true_e_ji), ('Input Redundancy (input,lower bound) for RULE90 node does not match. %s != %s' % (e_ji, true_e_ji))


# RULE110
def test_edge_effectiveness_RULE110():
    """Test Edge Effectiveness - RULE110"""
    n = RULE110()
    e_ji, true_e_ji = n.edge_effectiveness(bound='upper'), [0.25, 0.75, 0.75]
    assert (e_ji == true_e_ji), ('Input Redundancy (input,upper bound) for RULE110 node does not match. %s != %s' % (e_ji, true_e_ji))
    e_ji, true_e_ji = n.edge_effectiveness(bound='mean'), [0.375, 0.875, 0.875]
    assert (e_ji == true_e_ji), ('Input Redundancy (input,mean) for RULE110 node does not match. %s != %s' % (e_ji, true_e_ji))
    e_ji, true_e_ji = n.edge_effectiveness(bound='lower'), [0.5, 1., 1.]
    assert (e_ji == true_e_ji), ('Input Redundancy (input,lower bound) for RULE110 node does not match. %s != %s' % (e_ji, true_e_ji))

#
# Test Sensitivity
#
def test_sensitivity_AND():
    """Test Sensitivity - AND"""
    n = AND()
    s, true_s = n.c_sensitivity(1), 1 / 2
    assert isclose(s, true_s), ('c-sensitivity(1) for AND does not match, %s != %s' % (s, true_s))
    s, true_s = n.c_sensitivity(2), 1 / 2
    assert isclose(s, true_s), ('c-sensitivity(2) for AND does not match, %s != %s' % (s, true_s))
    s, true_s = n.c_sensitivity(1, 'forceK', 3), 1 / 3
    assert isclose(s, true_s), ("c-sensitivity(1,'forceK',3) for AND does not match, %s != %s" % (s, true_s))
    s, true_s = n.c_sensitivity(2, 'forceK', 3), 1 / 2
    assert isclose(s, true_s), ("c-sensitivity(2,'forceK',3) for AND does not match, %s != %s" % (s, true_s))


def test_sensitivity_XOR():
    """Test Sensitivity - XOR"""
    n = XOR()
    s, true_s = n.c_sensitivity(1), 1.
    assert isclose(s, true_s), ('c-sensitivity(1) for XOR does not match, %s != %s' % (s, true_s))
    s, true_s = n.c_sensitivity(2), 0.
    assert isclose(s, true_s), ('c-sensitivity(2) for XOR does not match, %s != %s' % (s, true_s))
    s, true_s = n.c_sensitivity(1, 'forceK', 3), 2 / 3
    assert isclose(s, true_s), ("c-sensitivity(1,'forceK',3) for XOR does not match, %s != %s" % (s, true_s))
    s, true_s = n.c_sensitivity(2, 'forceK', 3), 2 / 3
    assert isclose(s, true_s), ("c-sensitivity(2,'forceK',3) for XOR does not match, %s != %s" % (s, true_s))

# input symmetry tests (new)
def test_input_symmetry_AND():
    n = AND()
    k_s, true_k_s = n.input_symmetry(aggOp="mean", kernel="numDots"), 3.0/2
    assert (k_s == true_k_s), f"Input symmetry: AND (mean): returned {k_s}, true value is {true_k_s}"
    k_s, true_k_s = n.input_symmetry(aggOp="max", kernel="numDots"), 3.0/2
    assert (k_s == true_k_s), f"Input symmetry: AND (max): returned {k_s}, true value is {true_k_s}"
    k_s, true_k_s = n.input_symmetry_mean(), 3.0/2
    assert (k_s == true_k_s), f"Input symmetry simp: AND (mean): returned {k_s}, true value is {true_k_s}"

    # k_s, true_k_s = n.input_symmetry(aggOp="mean", kernel="numDots", sameSymbol=True), 2.0
    # assert (k_s == true_k_s), f"Input symmetry: AND (mean, sameSymbol): returned {k_s}, true value is {true_k_s}"
    # k_s, true_k_s = n.input_symmetry(aggOp="max", kernel="numDots", sameSymbol=True), 2.0
    # assert (k_s == true_k_s), f"Input symmetry: AND (max, sameSymbol): returned {k_s}, true value is {true_k_s}"

def test_input_symmetry_XOR():
    n = XOR()
    k_s, true_k_s = n.input_symmetry(aggOp="mean", kernel="numDots"), 1.0
    assert (k_s == true_k_s), f"Input symmetry: XOR (mean): returned {k_s}, true value is {true_k_s}"
    k_s, true_k_s = n.input_symmetry_mean(), 1.0
    assert (k_s == true_k_s), f"Input symmetry simp: XOR (mean): returned {k_s}, true value is {true_k_s}"
    k_s, true_k_s = n.input_symmetry(aggOp="max", kernel="numDots"), 1.0
    assert (k_s == true_k_s), f"Input symmetry: XOR (max): returned {k_s}, true value is {true_k_s}"

    # k_s, true_k_s = n.input_symmetry(aggOp="mean", kernel="numDots", sameSymbol=True), 2.0
    # assert (k_s == true_k_s), f"Input symmetry: XOR (mean, sameSymbol): returned {k_s}, true value is {true_k_s}"
    # k_s, true_k_s = n.input_symmetry_mean(), 2.0
    # assert (k_s == true_k_s), f"Input symmetry simp: XOR (mean, sameSymbol): returned {k_s}, true value is {true_k_s}"
    # k_s, true_k_s = n.input_symmetry(aggOp="max", kernel="numDots", sameSymbol=True), 2.0
    # assert (k_s == true_k_s), f"Input symmetry: XOR (max, sameSymbol): returned {k_s}, true value is {true_k_s}"

def test_input_symmetry_COPYx1():
    n = COPYx1()
    k_s, true_k_s = n.input_symmetry(aggOp="mean", kernel="numDots"), 0
    assert (k_s == true_k_s), f"Input symmetry: COPYx1 (mean): returned {k_s}, true value is {true_k_s}"
    k_s, true_k_s = n.input_symmetry_mean(), 0
    assert (k_s == true_k_s), f"Input symmetry simp: COPYx1 (mean): returned {k_s}, true value is {true_k_s}"
    k_s, true_k_s = n.input_symmetry(aggOp="max", kernel="numDots"), 0
    assert (k_s == true_k_s), f"Input symmetry: COPYx1 (max): returned {k_s}, true value is {true_k_s}"

    # k_s, true_k_s = n.input_symmetry(aggOp="mean", kernel="numDots", sameSymbol=True), 0.0
    # assert (k_s == true_k_s), f"Input symmetry: COPYx1 (mean, sameSymbol): returned {k_s}, true value is {true_k_s}"
    # k_s, true_k_s = n.input_symmetry_mean(), 0.0
    # assert (k_s == true_k_s), f"Input symmetry simp: COPYx1 (mean, sameSymbol): returned {k_s}, true value is {true_k_s}"
    # k_s, true_k_s = n.input_symmetry(aggOp="max", kernel="numDots", sameSymbol=True), 0.0
    # assert (k_s == true_k_s), f"Input symmetry: COPYx1 (max, sameSymbol): returned {k_s}, true value is {true_k_s}"

def test_input_symmetry_RULE90():
    n = RULE90()
    k_s, true_k_s = n.input_symmetry(aggOp="mean", kernel="numDots"), 1.0
    assert (k_s == true_k_s), f"Input symmetry: RULE90 (mean): returned {k_s}, true value is {true_k_s}"
    k_s, true_k_s = n.input_symmetry_mean(), 1.0
    assert (k_s == true_k_s), f"Input symmetry simp: RULE90 (mean): returned {k_s}, true value is {true_k_s}"
    # k_s, true_k_s = n.input_symmetry(aggOp="max", kernel="numDots", sameSymbol=True), 1.0
    assert (k_s == true_k_s), f"Input symmetry: RULE90 (max): returned {k_s}, true value is {true_k_s}"

    # k_s, true_k_s = n.input_symmetry(aggOp="mean", kernel="numDots", sameSymbol=True), 2.0
    # assert (k_s == true_k_s), f"Input symmetry: RULE90 (mean, sameSymbol): returned {k_s}, true value is {true_k_s}"
    # k_s, true_k_s = n.input_symmetry_mean(), 2.0
    # assert (k_s == true_k_s), f"Input symmetry simp: RULE90 (mean, sameSymbol): returned {k_s}, true value is {true_k_s}"
    # k_s, true_k_s = n.input_symmetry(aggOp="max", kernel="numDots", sameSymbol=True), 2.0
    # assert (k_s == true_k_s), f"Input symmetry: RULE90 (max, sameSymbol): returned {k_s}, true value is {true_k_s}"

def test_input_symmetry_SBF():
    n = BooleanNode(outputs=list("0111" + "0"*12), k=4)
    k_s, true_k_s = n.input_symmetry(aggOp="mean", kernel="numDots"), 1.6875
    assert (k_s == true_k_s), f"Input symmetry: SBF (mean): returned {k_s}, true value is {true_k_s}"
    k_s, true_k_s = n.input_symmetry_mean(), 1.6875
    assert (k_s == true_k_s), f"Input symmetry simp: SBF (mean): returned {k_s}, true value is {true_k_s}"
    k_s, true_k_s = n.input_symmetry(aggOp="max", kernel="numDots"), 1.875
    assert (k_s == true_k_s), f"Input symmetry: SBF (max): returned {k_s}, true value is {true_k_s}"

    # k_s, true_k_s = n.input_symmetry(aggOp="mean", kernel="numDots", sameSymbol=True), 4.0
    # assert (k_s == true_k_s), f"Input symmetry: SBF (mean, sameSymbol): returned {k_s}, true value is {true_k_s}"
    # k_s, true_k_s = n.input_symmetry_mean(), 4.0
    # assert (k_s == true_k_s), f"Input symmetry simp: SBF (mean, sameSymbol): returned {k_s}, true value is {true_k_s}"
    # k_s, true_k_s = n.input_symmetry(aggOp="max", kernel="numDots", sameSymbol=True), 4.0
    # assert (k_s == true_k_s), f"Input symmetry: SBF (max, sameSymbol): returned {k_s}, true value is {true_k_s}"


#
# Test sensitivity / activities (LUT and schemata paths)
#

def test_sensitivity_hand_values():
    """sensitivity: mean number of single-input flips that change the output"""
    from cana.function_outputs.bit_lut import BitLut

    cases = [
        (list("00"), 0.0),          # constant
        (list("01"), 1.0),          # identity
        (list("0000"), 0.0),        # constant, k=2
        (list("0110"), 2.0),        # XOR: every flip changes the output
        (list("0001"), 1.0),        # AND: 00,01,10 have one sensitive input, 11 has two
        (list("00010111"), 1.5),    # majority k=3: 000 and 111 have none, the rest have two
    ]
    for outputs, true_s in cases:
        n = BooleanNode.from_bit_lut(outputs)
        s = n.sensitivity(norm=False)
        assert s == true_s, f"sensitivity for {''.join(outputs)}: returned {s}, true value is {true_s}"
        assert sensitivity.sensitivity_from_bit_lut(BitLut.from_sequence(outputs)) == true_s


def test_sensitivity_lut_matches_schemata():
    """bit-lut and schemata sensitivity paths must be bit-exactly equal on the full space"""
    import random
    nodes = [AND(), OR(), XOR(), COPYx1(), CONTRADICTION(), RULE90(), RULE110()]
    rng = random.Random(20260921)
    for k in range(1, 8):
        for _ in range(50 if k <= 5 else 15):
            nodes.append(BooleanNode.from_bit_lut([rng.randint(0, 1) for _ in range(2**k)]))
    for n in nodes:
        for norm in (False, True):
            s_lut = n.sensitivity_from_bit_lut(norm=norm)
            s_sch = n.sensitivity_from_schemata(norm=norm)
            assert s_lut == s_sch, (
                f"sensitivity(norm={norm}) for k={n.k} outputs="
                f"{''.join(map(str, n.outputs))}: lut {s_lut!r} != schemata {s_sch!r}"
            )
            assert n.sensitivity(norm=norm) == s_lut
        assert abs(n.sensitivity(norm=True) - n.c_sensitivity(1)) < 1e-12


def test_activities_hand_values():
    """activities: per input, the fraction of input states where flipping it changes the output"""
    from cana.function_outputs.bit_lut import BitLut

    cases = [
        (list("00"), [0.0]),                     # constant
        (list("01"), [1.0]),                     # identity
        (list("0011"), [1.0, 0.0]),              # copy of input 1; input 2 is fictitious
        (list("0110"), [1.0, 1.0]),              # XOR: every flip changes the output
        (list("0001"), [0.5, 0.5]),              # AND: input i matters only when the other is 1
        (list("00010111"), [0.5, 0.5, 0.5]),     # majority k=3: input i matters when the others disagree
        (list("00011111"), [0.75, 0.25, 0.25]),  # x1 OR (x2 AND x3)
    ]
    for outputs, true_a in cases:
        n = BooleanNode.from_bit_lut(outputs)
        a = n.activities()
        assert a == true_a, f"activities for {''.join(outputs)}: returned {a}, true value is {true_a}"
        assert sensitivity.activities_from_bit_lut(BitLut.from_sequence(outputs)) == true_a


def test_activities_lut_matches_schemata():
    """bit-lut and schemata activities paths must be bit-exactly equal on the full space"""
    import random
    nodes = [AND(), OR(), XOR(), COPYx1(), CONTRADICTION(), RULE90(), RULE110()]
    rng = random.Random(20260928)
    for k in range(1, 8):
        for _ in range(50 if k <= 5 else 15):
            nodes.append(BooleanNode.from_bit_lut([rng.randint(0, 1) for _ in range(2**k)]))
    for n in nodes:
        a_lut = n.activities_from_bit_lut()
        a_sch = n.activities_from_schemata()
        assert a_lut == a_sch, (
            f"activities for k={n.k} outputs={''.join(map(str, n.outputs))}: "
            f"lut {a_lut!r} != schemata {a_sch!r}"
        )
        assert n.activities() == a_lut


def test_activities_sensitivity_auto_fallback_paths():
    """default methods prefer cached coverage, else LUT when k <= exact_k_max"""
    lut_node = BooleanNode.from_bit_lut([0, 0, 0, 1], name="AND")
    assert lut_node.outputs.bit_lut is not None
    assert lut_node.activities() == lut_node.activities_from_bit_lut()
    assert lut_node.sensitivity() == lut_node.sensitivity_from_bit_lut()
    assert lut_node.bias() == lut_node.bias_from_bit_lut()

    # no coverage and k <= exact_k_max → bit_lut for both
    fresh = BooleanNode.from_one_symbol_schemata(
        {0: {"02", "20"}, 1: {"11"}}, k=2, name="AND"
    )
    assert fresh.outputs.bit_lut is None
    assert fresh._pi_coverage is None and fresh._sampled_pi_coverage is None
    assert fresh.activities() == fresh.activities_from_bit_lut()
    assert fresh.sensitivity() == fresh.sensitivity_from_bit_lut()
    assert fresh.bias() == fresh.bias_from_bit_lut()
    assert fresh.activities() == lut_node.activities()
    assert fresh.sensitivity() == lut_node.sensitivity()
    assert fresh.bias() == lut_node.bias()

    # once pi_coverage is cached, both prefer the schemata path
    covered = BooleanNode.from_one_symbol_schemata(
        {0: {"02", "20"}, 1: {"11"}}, k=2, name="AND"
    )
    assert covered.outputs.verify_one_symbol_schemata() is True
    _ = covered.pi_coverage()
    assert covered._pi_coverage is not None
    assert covered.activities() == covered.activities_from_schemata()
    assert covered.sensitivity() == covered.sensitivity_from_schemata()
    assert covered.bias() == covered.bias_from_schemata()
    assert covered.bias() == pytest.approx(0.25, abs=1e-12)


def test_activities_sensitivity_high_k_schemata_sampling():
    """schemata-only nodes above MAX_DENSE_LUT_K can still estimate via sampling"""
    import numpy as np
    from cana.function_outputs.bit_lut import MAX_DENSE_LUT_K

    k = MAX_DENSE_LUT_K + 1
    # constant-1 on all inputs: every input is fictitious → activity 0
    n = BooleanNode.from_one_symbol_schemata({1: {"2" * k}}, k=k)
    assert n.outputs.bit_lut is None
    with pytest.raises(ValueError, match="MAX_DENSE_LUT_K"):
        n.outputs.get_bit_lut()

    rng = np.random.default_rng(0)
    a = n.activities(exact_k_max=-1, n_samples=64, rng=rng)
    s = n.sensitivity(exact_k_max=-1, n_samples=64, rng=np.random.default_rng(0))
    assert a == [0.0] * k
    assert s == 0.0

#
# Test representation factories
#

def test_from_bit_lut_rejects_non_power_of_two():
    """from_bit_lut must not silently truncate k on a malformed list"""
    for bad in ([], [0, 1, 1], [0] * 6, [1] * 129):
        with pytest.raises(ValueError):
            BooleanNode.from_bit_lut(bad)

def test_from_bit_lut_returns_subclass():
    """from_bit_lut must instantiate the class it is called on"""
    class MyNode(BooleanNode):
        pass

    n = MyNode.from_bit_lut([0, 1, 1, 0], name="xor")
    assert type(n) is MyNode
    assert (n.k, n.name, n.outputs) == (2, "xor", list("0110"))
    assert n.input_redundancy(norm=False) == XOR().input_redundancy(norm=False)

def test_from_output_list_deprecated_alias():
    """from_output_list still works and warns toward from_bit_lut"""
    with pytest.warns(DeprecationWarning, match="from_bit_lut"):
        n = BooleanNode.from_output_list([0, 0, 0, 1], name="AND")
    assert n.k == 2
    assert list(n.outputs) == [0, 0, 0, 1]

def test_from_positional_expression_factory():
    n = BooleanNode.from_positional_expression("x0 and x1", k=2, name="AND", inputs=[0, 1])
    assert n.outputs.positional_expression == "x0 and x1"
    assert list(n.outputs.get_bit_lut()) == [0, 0, 0, 1]

def test_from_one_symbol_schemata_factory():
    # Complete AND cover (both polarities) so get_bit_lut can expand.
    schemata = {0: {"02", "20"}, 1: {"11"}}
    n = BooleanNode.from_one_symbol_schemata(
        schemata, k=2, name="AND", inputs=[0, 1]
    )
    assert n.outputs.one_symbol_schemata == schemata
    assert list(n.outputs.get_bit_lut()) == [0, 0, 0, 1]

def test_from_two_symbol_schemata_factory():
    # OR as two-symbol: one wild entry covering both singleton ON minterms.
    # Use a concrete two-symbol obtained from a known LUT via FunctionOutputs.
    from cana.function_outputs import FunctionOutputs

    fo = FunctionOutputs.from_sequence([0, 1, 1, 1])
    ts = fo.get_two_symbol_schemata()
    n = BooleanNode.from_two_symbol_schemata(ts, k=2, name="OR", inputs=[0, 1])
    assert n.outputs.two_symbol_schemata is not None
    assert list(n.outputs.get_bit_lut()) == [0, 1, 1, 1]

def test_coverage_refuses_huge_k():
    """pi_coverage / ts_coverage refuse Θ(2^k) expansion above MAX_DENSE_LUT_K."""
    from cana.function_outputs.bit_lut import MAX_DENSE_LUT_K

    k = MAX_DENSE_LUT_K + 1
    n_pi = BooleanNode.from_one_symbol_schemata({1: {"2" * k}}, k=k)
    with pytest.raises(ValueError, match="pi_coverage.*MAX_DENSE_LUT_K"):
        n_pi.pi_coverage()
    assert n_pi._pi_coverage is None

    ts = ([], [["2" * k, [], []]])
    n_ts = BooleanNode.from_two_symbol_schemata(ts, k=k)
    with pytest.raises(ValueError, match="ts_coverage.*MAX_DENSE_LUT_K"):
        n_ts.ts_coverage()
    assert n_ts._ts_coverage is None


def test_input_symmetry_mean_on_fresh_node():
    """input_symmetry_mean must compute its own coverage instead of requiring input_symmetry() to run first"""
    n = BooleanNode(outputs=list("0111" + "0"*12), k=4)
    assert n.input_symmetry_mean() == 1.6875
