# -*- coding: utf-8 -*-
#
# Tests for ``boolean_node.py``
# These tests were manually calculated by Luis M. Rocha and implemented by Rion B. Correia.
#
from cana.datasets.bools import CONTRADICTION, AND, OR, XOR, COPYx1, RULE90, RULE110
from cana.utils import *


#
# Test Input Redundancy
#

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
# Test Edge Symmetry
#

# AND
def test_edge_symmetry_AND():
    """Test Edge Symmetry - AND"""
    n = AND()
    k_s, true_k_s = n.edge_symmetry(bound='upper'), [2., 2.]
    assert (k_s == true_k_s), ('Input Symmetry (upper) for AND node does not match. %s != %s' % (k_s, true_k_s))
    k_s, true_k_s = n.edge_symmetry(bound='mean'), [2., 2.]
    assert (k_s == true_k_s), ('Input Symmetry (mean) for AND node does not match. %s != %s' % (k_s, true_k_s))
    k_s, true_k_s = n.edge_symmetry(bound='lower'), [2., 2.]
    assert (k_s == true_k_s), ('Input Symmetry (lower) for AND node does not match. %s != %s' % (k_s, true_k_s))

    k_s, true_k_s = n.edge_symmetry(bound='tuple'), [[(2, 2)] * 4, [(2, 2)] * 4]
    assert (k_s == true_k_s), ('Edge symmetry (tuples) for AND node does not match. %s != %s' % (k_s, true_k_s))


# OR
def test_edge_symmetry_OR():
    """Test Edge Symmetry - OR"""
    n = OR()
    k_s, true_k_s = n.edge_symmetry(bound='upper'), [2., 2.]
    assert (k_s == true_k_s), ('Input Symmetry (input,upper bound) for OR node does not match. %s != %s' % (k_s, true_k_s))
    k_s, true_k_s = n.edge_symmetry(bound='mean'), [2., 2.]
    assert (k_s == true_k_s), ('Input Symmetry (input,mean) for OR node does not match. %s != %s' % (k_s, true_k_s))
    k_s, true_k_s = n.edge_symmetry(bound='lower'), [2., 2.]
    assert (k_s == true_k_s), ('Input Symmetry (input,lower bound) for OR node does not match. %s != %s' % (k_s, true_k_s))

    k_s, true_k_s = n.edge_symmetry(bound='tuple'), [[(2, 2)] * 4, [(2, 2)] * 4]
    assert (k_s == true_k_s), ('Input Redundancy (input,tuples) for XOR node does not match. %s != %s' % (k_s, true_k_s))


# XOR
def test_edge_symmetry_XOR():
    """Test Edge Symmetry - XOR"""
    n = XOR()
    k_s, true_k_s = n.edge_symmetry(bound='upper'), [2., 2.]
    assert (k_s == true_k_s), ('Input Symmetry (input,upper bound) for XOR node does not match. %s != %s' % (k_s, true_k_s))
    k_s, true_k_s = n.edge_symmetry(bound='mean'), [2., 2.]
    assert (k_s == true_k_s), ('Input Symmetry (input,mean) for XOR node does not match. %s != %s' % (k_s, true_k_s))
    k_s, true_k_s = n.edge_symmetry(bound='lower'), [2., 2.]
    assert (k_s == true_k_s), ('Input Symmetry (input,lower bound) for XOR node does not match. %s != %s' % (k_s, true_k_s))

    k_s, true_k_s = n.edge_symmetry(bound='tuple'), [[(2, 2)] * 4, [(2, 2)] * 4]
    assert (k_s == true_k_s), ('Input Redundancy (input,tuples) for XOR node does not match. %s != %s' % (k_s, true_k_s))


# CONTRADICTION
def test_edge_symmetry_CONTRADICTION():
    """Test Edge Symmetry - CONTRADICTION"""
    n = CONTRADICTION()
    k_s, true_k_s = n.edge_symmetry(bound='upper'), [2., 2.]
    assert (k_s == true_k_s), ('Input Symmetry (input,upper bound) for CONTRADICTION node does not match. %s != %s' % (k_s, true_k_s))
    k_s, true_k_s = n.edge_symmetry(bound='mean'), [2., 2.]
    assert (k_s == true_k_s), ('Input Symmetry (input,mean) for CONTRADICTION node does not match. %s != %s' % (k_s, true_k_s))
    k_s, true_k_s = n.edge_symmetry(bound='lower'), [2., 2.]
    assert (k_s == true_k_s), ('Input Symmetry (input,lower bound) for CONTRADICTION node does not match. %s != %s' % (k_s, true_k_s))

    k_s, true_k_s = n.edge_symmetry(bound='tuple'), [[(2, 2)] * 4, [(2, 2)] * 4]
    assert (k_s == true_k_s), ('Input Redundancy (input,tuples) for CONTRADICTION node does not match. %s != %s' % (k_s, true_k_s))


# COPYx1
def test_edge_symmetry_COPYx1():
    """Test Edge Symmetry - COPYx1"""
    n = COPYx1()
    k_s, true_k_s = n.edge_symmetry(bound='upper'), [0., 0.]
    assert (k_s == true_k_s), ('Input Symmetry (input,upper bound) for COPYx1 node does not match. %s != %s' % (k_s, true_k_s))
    k_s, true_k_s = n.edge_symmetry(bound='mean'), [0., 0.]
    assert (k_s == true_k_s), ('Input Symmetry (input,mean) for COPYx1 node does not match. %s != %s' % (k_s, true_k_s))
    k_s, true_k_s = n.edge_symmetry(bound='lower'), [0., 0.]
    assert (k_s == true_k_s), ('Input Symmetry (input,lower bound) for COPYx1 node does not match. %s != %s' % (k_s, true_k_s))

    k_s, true_k_s = n.edge_symmetry(bound='tuple'), [[(0, 0)] * 4, [(0, 0)] * 4]
    assert (k_s == true_k_s), ('Input Redundancy (input,tuples) for COPYx1 node does not match. %s != %s' % (k_s, true_k_s))


# RULE90
def test_edge_symmetry_RULE90():
    """Test Edge Symmetry - RULE90"""
    n = RULE90()
    k_s, true_k_s = n.edge_symmetry(bound='upper'), [8 / 4, 0., 8 / 4]
    assert (k_s == true_k_s), ('Input Symmetry (input,upper bound) for RULE90 node does not match. %s != %s' % (k_s, true_k_s))
    k_s, true_k_s = n.edge_symmetry(bound='mean'), [8 / 4., 0., 8 / 4]
    assert (k_s == true_k_s), ('Input Symmetry (input,mean) for RULE90 node does not match. %s != %s' % (k_s, true_k_s))
    k_s, true_k_s = n.edge_symmetry(bound='lower'), [8 / 4., 0., 8 / 4]
    assert (k_s == true_k_s), ('Input Symmetry (input,lower bound) for RULE90 node does not match. %s != %s' % (k_s, true_k_s))

    k_s, true_k_s = n.edge_symmetry(bound='tuple'), [[(2, 2)] * 8, [(0, 0)] * 8, [(2, 2)] * 8]
    assert (k_s == true_k_s), ('Input Redundancy (input,tuples) for RULE90 node does not match. %s != %s' % (k_s, true_k_s))


# RULE110
def test_input_symmetry_RULE110():
    """Test Edge Symmetry - RULE110"""
    n = RULE110()
    k_s, true_k_s = n.edge_symmetry(bound='upper'), [13 / 8, 17 / 8, 17 / 8]
    assert (k_s == true_k_s), ('Input Symmetry (input,upper bound) for RULE110 node does not match. %s != %s' % (k_s, true_k_s))
    k_s, true_k_s = n.edge_symmetry(bound='mean'), [0.95833333333333326, 1.8333333333333333, 1.8333333333333333]  # NOT SURE ITS CORRECT!!!
    assert (k_s == true_k_s), ('Input Symmetry (input,mean) for RULE110 node does not match. %s != %s' % (k_s, true_k_s))
    k_s, true_k_s = n.edge_symmetry(bound='lower'), [0.375, 1.375, 1.375]  # NOT SURE ITS CORRECT!!!
    assert (k_s == true_k_s), ('Input Symmetry (input,lower bound) for RULE110 node does not match. %s != %s' % (k_s, true_k_s))

    k_s, true_k_s = n.edge_symmetry(bound='tuple'), [[(0, 0), (0, 2), (0, 2), (0, 2), (0, 0), (0, 2), (0, 2), (3, 3)], [(2, 2), (2, 2), (0, 2), (0, 2), (2, 2), (2, 2), (0, 2), (3, 3)], [(2, 2), (0, 2), (2, 2), (0, 2), (2, 2), (0, 2), (2, 2), (3, 3)]]
    assert (k_s == true_k_s), ('Input Redundancy (input,tuples) for RULE110 node does not match. %s != %s' % (k_s, true_k_s))


#
# Test Input Symmetry
#

# AND
def test_input_symmetry_AND():
    """Test Input Symmetry - AND"""
    n = AND()
    k_s, true_k_s = n.input_symmetry(bound='upper', norm=False), 8 / 4
    assert (k_s == true_k_s), ('Input Symmetry (node,upper bound) for AND node does not match. %s != %s' % (k_s, true_k_s))
    k_s, true_k_s = n.input_symmetry(bound='lower', norm=False), 8 / 4
    assert (k_s == true_k_s), ('Input Symmetry (node,lower bound) for AND node does not match. %s != %s' % (k_s, true_k_s))

    k_s, true_k_s = n.input_symmetry(bound='upper', norm=True), (8 / 4) / 2
    assert (k_s == true_k_s), ('Input Symmetry (node,upper bound,normed) for AND node does not match. %s != %s' % (k_s, true_k_s))
    k_s, true_k_s = n.input_symmetry(bound='lower', norm=True), (8 / 4) / 2
    assert (k_s == true_k_s), ('Input Symmetry (node,lower bound,normed) for AND node does not match. %s != %s' % (k_s, true_k_s))


# OR
def test_input_symmetry_OR():
    """Test Input Symmetry - OR"""
    n = OR()
    k_s, true_k_s = n.input_symmetry(bound='upper', norm=False), 8 / 4
    assert (k_s == true_k_s), ('Input Symmetry (node,upper bound) for OR node does not match. %s != %s' % (k_s, true_k_s))
    k_s, true_k_s = n.input_symmetry(bound='lower', norm=False), 8 / 4
    assert (k_s == true_k_s), ('Input Symmetry (node,lower bound) for OR node does not match. %s != %s' % (k_s, true_k_s))

    k_s, true_k_s = n.input_symmetry(bound='upper', norm=True), (8 / 4) / 2
    assert (k_s == true_k_s), ('Input Symmetry (node,upper bound,normed) for OR node does not match. %s != %s' % (k_s, true_k_s))
    k_s, true_k_s = n.input_symmetry(bound='lower', norm=True), (8 / 4) / 2
    assert (k_s == true_k_s), ('Input Symmetry (node,lower bound,normed) for OR node does not match. %s != %s' % (k_s, true_k_s))


# XOR
def test_input_symmetry_XOR():
    """Test Input Symmetry - XOR"""
    n = XOR()
    k_s, true_k_s = n.input_symmetry(bound='upper', norm=False), 8 / 4
    assert (k_s == true_k_s), ('Input Symmetry (node,upper bound) for XOR node does not match. %s != %s' % (k_s, true_k_s))
    k_s, true_k_s = n.input_symmetry(bound='lower', norm=False), 8 / 4
    assert (k_s == true_k_s), ('Input Symmetry (node,lower bound) for XOR node does not match. %s != %s' % (k_s, true_k_s))

    k_s, true_k_s = n.input_symmetry(bound='upper', norm=True), (8 / 4) / 2
    assert (k_s == true_k_s), ('Input Symmetry (node,upper bound,normed) for XOR node does not match. %s != %s' % (k_s, true_k_s))
    k_s, true_k_s = n.input_symmetry(bound='lower', norm=True), (8 / 4) / 2
    assert (k_s == true_k_s), ('Input Symmetry (node,lower bound,normed) for XOR node does not match. %s != %s' % (k_s, true_k_s))


# CONTRADICTION
def test_input_symmetry_CONTRADICTION():
    """Test Input Symmetry - CONTRADICTION"""
    n = CONTRADICTION()
    k_s, true_k_s = n.input_symmetry(bound='upper', norm=False), 8 / 4
    assert (k_s == true_k_s), ('Input Symmetry (node,upper bound) for CONTRADICTION node does not match. %s != %s' % (k_s, true_k_s))
    k_s, true_k_s = n.input_symmetry(bound='lower', norm=False), 8 / 4
    assert (k_s == true_k_s), ('Input Symmetry (node,lower bound) for CONTRADICTION node does not match. %s != %s' % (k_s, true_k_s))

    k_s, true_k_s = n.input_symmetry(bound='upper', norm=True), (8 / 4) / 2
    assert (k_s == true_k_s), ('Input Symmetry (node,upper bound,normed) for CONTRADICTION node does not match. %s != %s' % (k_s, true_k_s))
    k_s, true_k_s = n.input_symmetry(bound='lower', norm=True), (8 / 4) / 2
    assert (k_s == true_k_s), ('Input Symmetry (node,lower bound,normed) for CONTRADICTION node does not match. %s != %s' % (k_s, true_k_s))


# COPYx1
def test_input_symmetry_COPYx1():
    """Test Input Symmetry - COPYx1"""
    n = COPYx1()
    k_s, true_k_s = n.input_symmetry(bound='upper', norm=False), 0 / 4
    assert (k_s == true_k_s), ('Input Symmetry (node,upper bound) for COPYx1 node does not match. %s != %s' % (k_s, true_k_s))
    k_s, true_k_s = n.input_symmetry(bound='lower', norm=False), 0 / 4
    assert (k_s == true_k_s), ('Input Symmetry (node,lower bound) for COPYx1 node does not match. %s != %s' % (k_s, true_k_s))

    k_s, true_k_s = n.input_symmetry(bound='upper', norm=True), (0 / 4) / 2
    assert (k_s == true_k_s), ('Input Symmetry (node,upper bound,normed) for COPYx1 node does not match. %s != %s' % (k_s, true_k_s))
    k_s, true_k_s = n.input_symmetry(bound='lower', norm=True), (0 / 4) / 2
    assert (k_s == true_k_s), ('Input Symmetry (node,lower bound,normed) for COPYx1 node does not match. %s != %s' % (k_s, true_k_s))


# RULE90
def test_input_symmetry_RULE90():
    """Test Input Symmetry - RULE90"""
    n = RULE90()
    k_s, true_k_s = n.input_symmetry(bound='upper', norm=False), 4 / 3
    assert (k_s == true_k_s), ('Input Symmetry (node,upper bound) for RULE90 node does not match. %s != %s' % (k_s, true_k_s))
    k_s, true_k_s = n.input_symmetry(bound='lower', norm=False), 4 / 3
    assert (k_s == true_k_s), ('Input Symmetry (node,lower bound) for RULE90 node does not match. %s != %s' % (k_s, true_k_s))

    k_s, true_k_s = n.input_symmetry(bound='upper', norm=True), (4 / 3) / 3
    assert (k_s == true_k_s), ('Input Symmetry (node,upper bound,normed) for RULE90 node does not match. %s != %s' % (k_s, true_k_s))
    k_s, true_k_s = n.input_symmetry(bound='lower', norm=True), (4 / 3) / 3
    assert (k_s == true_k_s), ('Input Symmetry (node,lower bound,normed) for RULE90 node does not match. %s != %s' % (k_s, true_k_s))


# RULE110
def test_input_symmetry_RULE110():
    """Test Input Symmetry - RULE110"""
    n = RULE110()
    k_s, true_k_s = n.input_symmetry(bound='upper', norm=False), (13 / 8 + 17 / 8 + 17 / 8) / 3
    assert (k_s == true_k_s), ('Input Symmetry (node,upper bound) for RULE110 node does not match. %s != %s' % (k_s, true_k_s))
    k_s, true_k_s = n.input_symmetry(bound='lower', norm=False), (3 / 8 + 11 / 8 + 11 / 8) / 3
    assert (k_s == true_k_s), ('Input Symmetry (node,lower bound) for RULE110 node does not match. %s != %s' % (k_s, true_k_s))

    k_s, true_k_s = n.input_symmetry(bound='upper', norm=True), ((13 / 8 + 17 / 8 + 17 / 8) / 3) / 3
    assert (k_s == true_k_s), ('Input Symmetry (node,upper bound,normed) for RULE110 node does not match. %s != %s' % (k_s, true_k_s))
    k_s, true_k_s = n.input_symmetry(bound='lower', norm=True), ((3 / 8 + 11 / 8 + 11 / 8) / 3) / 3
    assert (k_s == true_k_s), ('Input Symmetry (node,lower bound,normed) for RULE110 node does not match. %s != %s' % (k_s, true_k_s))

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
