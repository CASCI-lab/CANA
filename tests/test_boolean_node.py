# -*- coding: utf-8 -*-
#
# Tests for ``boolean_node.py``
# These tests were manually calculated by Luis M. Rocha and implemented by Rion B. Correia.
#
from cana.datasets.bools import CONTRADICTION, AND, OR, XOR, COPYx1, RULE90, RULE110
from cana.utils import *
from cana.boolean_node import BooleanNode
import numpy as np


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


# Three famous rules.
THREE_FAMOUS_RULES = {
    "GKL": (
        "0000000001011111000000000101111100000000010111110000000001011111"
        "0000000001011111111111110101111100000000010111111111111101011111"
    ), # G ́acs, P., Kurdyumov, G. L., and Levin, L. A. (1978). Onedimensional uniform arrays that wash out finite islands. Problemy Peredachi Informatsii, 14(3):92–96.
    "GP": (
        "0000010100000000010101010000010100000101000000000101010100000101"
        "0101010111111111010101011111111101010101111111110101010111111111"
    ), # Andre, D., Bennett III, F. H., and Koza, J. R. (1996). Discovery by genetic programming of a cellular automata rule that is better than any known rule for the majority classification problem. Genetic programming, 96:3–11.
    "COMP1": (
        "0000000000000001000100110000000100010011010111110001001101011111"
        "0001001100000001111111110101111100010011010111111111111101011111"
    ), # Kari, J., & Le Gloannec, B. (2012). Modified Traffic Cellular Automaton for the Density Classification Task. Fundamenta Informaticae, 116(1–4), 141–156. https://doi.org/10.3233/FI-2012-675


}

# Expected values computed from legacy implementations and verified manually.
EXPECTED_SYMMETRY_VALUES = {
    "GKL": {"raw": 0.6642857142857145, "distinct": 0.6371323529411765},
    "GP": {"raw": 0.6363095238095241, "distinct": 0.6079044117647053},
    "COMP1": {"raw": 0.7238095238095235, "distinct": 0.6994485294117648},
}


def test_new_symmetry_matches_expected_values():
    """Check exact expected outputs for the new symmetry implementations."""
    for rule_name in ["GKL", "GP", "COMP1"]:
        node = BooleanNode.from_output_list(THREE_FAMOUS_RULES[rule_name])
        raw_value = node.raw_symmetry()
        distinct_value = node.distinct_symmetry()

        assert isclose(raw_value, EXPECTED_SYMMETRY_VALUES[rule_name]["raw"]), (
            "Raw symmetry mismatch for %s: %s != %s"
            % (rule_name, raw_value, EXPECTED_SYMMETRY_VALUES[rule_name]["raw"])
        )
        assert isclose(
            distinct_value, EXPECTED_SYMMETRY_VALUES[rule_name]["distinct"]
        ), (
            "Distinct symmetry mismatch for %s: %s != %s"
            % (
                rule_name,
                distinct_value,
                EXPECTED_SYMMETRY_VALUES[rule_name]["distinct"],
            )
        )


def test_symmetry_values_are_bounded():
    """Symmetry metrics are ratios and must stay inside [0, 1]."""
    for rule_name in ["GKL", "GP", "COMP1"]:
        node = BooleanNode.from_output_list(THREE_FAMOUS_RULES[rule_name])
        raw_value = node.raw_symmetry()
        distinct_value = node.distinct_symmetry()

        assert 0.0 <= raw_value <= 1.0
        assert 0.0 <= distinct_value <= 1.0


def test_smallest_k_behavior():
    """Verify behavior for the smallest non-trivial LUT (k=1)."""
    node = BooleanNode.from_output_list("01")

    # For k=1, each Hamming-weight bucket has one row, so raw symmetry is 1.
    assert isclose(node.raw_symmetry(), 1.0)

    # Distinct metric excludes identity permutation; with only one bit there are
    # no non-identity permutations, so each row contributes 0.
    assert isclose(node.distinct_symmetry(), 0.0)


def test_distinct_symmetry_classic_gates():
    """Check distinct symmetry values for AND/OR/XOR/COPYx1."""
    gate_factories = {
        "AND": (AND, 0.5),
        "OR": (OR, 0.5),
        "XOR": (XOR, 0.5),
        "COPYx1": (COPYx1, 0.0),
    }

    for gate_name, (gate_factory, expected_value) in gate_factories.items():
        node = gate_factory()
        observed_value = node.distinct_symmetry()
        assert isclose(observed_value, expected_value), (
            "Distinct symmetry for %s does not match, %s != %s"
            % (gate_name, observed_value, expected_value)
        )


def test_raw_symmetry_classic_gates():
    """Check raw symmetry values for AND/OR/XOR/COPYx1."""
    gate_factories = {
        "AND": (AND, 1.0),
        "OR": (OR, 1.0),
        "XOR": (XOR, 1.0),
        "COPYx1": (COPYx1, 0.75),
    }

    for gate_name, (gate_factory, expected_value) in gate_factories.items():
        node = gate_factory()
        observed_value = node.raw_symmetry()
        assert isclose(observed_value, expected_value), (
            "Raw symmetry for %s does not match, %s != %s"
            % (gate_name, observed_value, expected_value)
        )


