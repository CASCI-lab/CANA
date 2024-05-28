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


# Tests for partially-specified functions
def test_partial_lut():
    partial_lut = [
        [('00','1'),('01','1')],
        [('0-','1'),('10','1')],
        [('001','1'),('01-','1'),('1-1','0')],
        [('00--', '0'), ('1--1','1'), ('11--','0')]
    ]
    expected_filled = [
        [('00','1'),('01','1'),('10','?'),('11','?')],
        [('00','1'),('01','1'),('10','1'),('11','?')],
        [('000','?'),('001','1'),('010','1'),('011','1'),('100','?'),('101','0'),('110','?'),('111','0')],
        [('0000', '0'), ('0001', '0'), ('0010', '0'), ('0011', '0'), ('0100', '?'), ('0101', '?'), ('0110', '?'), ('0111', '?'), ('1000', '?'), ('1001', '1'), ('1010', '?'), ('1011', '1'), ('1100', '0'), ('1101', '!'),('1110', '0'), ('1111', '!')]
    ]
    for i, partial in enumerate(partial_lut):
        filled = fill_out_lut(partial)
        expected = expected_filled[i]
        assert(filled==expected), f"Partial LUT filling failed: {filled} != {expected}"