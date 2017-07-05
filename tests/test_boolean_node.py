# -*- coding: utf-8 -*-
#
# Tests for ``boolean_node.py``
# These tests were hand calculated by Luis M. Rocha and implemented by Rion B. Correia.
#
from __future__ import division
from cana.boolean_node import BooleanNode
from cana.datasets.bools import CONTRADICTION,AND,OR,XOR,COPYx1,RULE90,RULE110

#
# Test Input Redundancy
#
# AND
def test_input_redundancy_AND():
	"""Test Input Redundancy - AND"""
	#from cana.networks.bools import AND
	n = AND()
	k_r, true_k_r = n.input_redundancy(mode='node',bound='upper',norm=False) , 3/4
	assert (k_r ==  true_k_r) , ('Input Redundancy (node,lower bound) for AND node does not match. %s != %s' % (k_r,true_k_r))
	k_r, true_k_r = n.input_redundancy(mode='node',bound='lower',norm=False) , 3/4
	assert (k_r ==  true_k_r) , ('Input Redundancy (node,lower bound) for AND node does not match. %s != %s' % (k_r,true_k_r))
	
	k_r, true_k_r = n.input_redundancy(mode='node',bound='upper',norm=True) , (3/4)/2
	assert (k_r ==  true_k_r) , ('Input Redundancy (node,lower bound) for AND node does not match. %s != %s' % (k_r,true_k_r))
	k_r, true_k_r = n.input_redundancy(mode='node',bound='lower',norm=True) , (3/4)/2
	assert (k_r ==  true_k_r) , ('Input Redundancy (node,lower bound) for AND node does not match. %s != %s' % (k_r,true_k_r))

	k_r, true_k_r = n.input_redundancy(mode='input',bound='upper',norm=False) , [1/2.,1/2]
	assert (k_r ==  true_k_r) , ('Input Redundancy (input,upper bound) for AND node does not match. %s != %s' % (k_r,true_k_r))
	k_r, true_k_r = n.input_redundancy(mode='input',bound='mean',norm=False) , [3/8.,3/8]
	assert (k_r ==  true_k_r) , ('Input Redundancy (input,mean) for AND node does not match. %s != %s' % (k_r,true_k_r))
	k_r, true_k_r = n.input_redundancy(mode='input',bound='lower',norm=False) , [1/4.,1/4]
	assert (k_r ==  true_k_r) , ('Input Redundancy (input,lower bound) for AND node does not match. %s != %s' % (k_r,true_k_r))

	k_r, true_k_r = n.input_redundancy(mode='input',bound='tuple',norm=False) , [(0.25, 0.5), (0.25, 0.5)]
	assert (k_r ==  true_k_r) , ('Input Redundancy (input,tuples) for AND node does not match. %s != %s' % (k_r,true_k_r))

# OR
def test_input_redundancy_OR():
	"""Test Input Redundancy - OR"""
	n = OR()
	k_r, true_k_r = n.input_redundancy(mode='node',bound='upper',norm=False) , 3/4
	assert (k_r ==  true_k_r) , ('Input Redundancy (node,lower bound) for OR node does not match. %s != %s' % (k_r,true_k_r))
	k_r, true_k_r = n.input_redundancy(mode='node',bound='lower',norm=False) , 3/4
	assert (k_r ==  true_k_r) , ('Input Redundancy (node,lower bound) for OR node does not match. %s != %s' % (k_r,true_k_r))
	
	k_r, true_k_r = n.input_redundancy(mode='node',bound='upper',norm=True) , (3/4)/2
	assert (k_r ==  true_k_r) , ('Input Redundancy (node,lower bound) for OR node does not match. %s != %s' % (k_r,true_k_r))
	k_r, true_k_r = n.input_redundancy(mode='node',bound='lower',norm=True) , (3/4)/2
	assert (k_r ==  true_k_r) , ('Input Redundancy (node,lower bound) for OR node does not match. %s != %s' % (k_r,true_k_r))

	k_r, true_k_r = n.input_redundancy(mode='input',bound='upper',norm=False) , [1/2.,1/2]
	assert (k_r ==  true_k_r) , ('Input Redundancy (input,upper bound) for OR node does not match. %s != %s' % (k_r,true_k_r))
	k_r, true_k_r = n.input_redundancy(mode='input',bound='mean',norm=False) , [3/8.,3/8]
	assert (k_r ==  true_k_r) , ('Input Redundancy (input,mean) for OR node does not match. %s != %s' % (k_r,true_k_r))
	k_r, true_k_r = n.input_redundancy(mode='input',bound='lower',norm=False) , [1/4.,1/4]
	assert (k_r ==  true_k_r) , ('Input Redundancy (input,lower bound) for OR node does not match. %s != %s' % (k_r,true_k_r))

	k_r, true_k_r = n.input_redundancy(mode='input',bound='tuple',norm=False) , [(0.25, 0.5), (0.25, 0.5)]
	assert (k_r ==  true_k_r) , ('Input Redundancy (input,tuples) for OR node does not match. %s != %s' % (k_r,true_k_r))

# XOR
def test_input_redundancy_XOR():
	"""Test Input Redundancy - XOR"""
	n = XOR()
	k_r, true_k_r = n.input_redundancy(mode='node',bound='upper',norm=False) , 0
	assert (k_r ==  true_k_r) , ('Input Redundancy (node,lower bound) for XOR node does not match. %s != %s' % (k_r,true_k_r))
	k_r, true_k_r = n.input_redundancy(mode='node',bound='lower',norm=False) , 0
	assert (k_r ==  true_k_r) , ('Input Redundancy (node,lower bound) for XOR node does not match. %s != %s' % (k_r,true_k_r))
	
	k_r, true_k_r = n.input_redundancy(mode='node',bound='upper',norm=True) , 0
	assert (k_r ==  true_k_r) , ('Input Redundancy (node,lower bound) for XOR node does not match. %s != %s' % (k_r,true_k_r))
	k_r, true_k_r = n.input_redundancy(mode='node',bound='lower',norm=True) , 0
	assert (k_r ==  true_k_r) , ('Input Redundancy (node,lower bound) for XOR node does not match. %s != %s' % (k_r,true_k_r))

	k_r, true_k_r = n.input_redundancy(mode='input',bound='upper',norm=False) , [0,0]
	assert (k_r ==  true_k_r) , ('Input Redundancy (input,upper bound) for XOR node does not match. %s != %s' % (k_r,true_k_r))
	k_r, true_k_r = n.input_redundancy(mode='input',bound='mean',norm=False) , [0,0]
	assert (k_r ==  true_k_r) , ('Input Redundancy (input,mean) for XOR node does not match. %s != %s' % (k_r,true_k_r))
	k_r, true_k_r = n.input_redundancy(mode='input',bound='lower',norm=False) , [0,0]
	assert (k_r ==  true_k_r) , ('Input Redundancy (input,lower bound) for XOR node does not match. %s != %s' % (k_r,true_k_r))

	k_r, true_k_r = n.input_redundancy(mode='input',bound='tuple',norm=False) , [(0.0, 0.0), (0.0, 0.0)]
	assert (k_r ==  true_k_r) , ('Input Redundancy (input,tuples) for XOR node does not match. %s != %s' % (k_r,true_k_r))

# CONTRADICTION
def test_input_redundancy_CONTRADICTION():
	"""Test Input Redundancy - CONTRADICTION"""
	n = CONTRADICTION()
	k_r, true_k_r = n.input_redundancy(mode='node',bound='upper',norm=False) , 2.
	assert (k_r ==  true_k_r) , ('Input Redundancy (node,upper bound) for CONTRADICTION node does not match. %s != %s' % (k_r,true_k_r))
	k_r, true_k_r = n.input_redundancy(mode='node',bound='lower',norm=False) , 2.
	assert (k_r ==  true_k_r) , ('Input Redundancy (node,lower bound) for CONTRADICTION node does not match. %s != %s' % (k_r,true_k_r))

	k_r, true_k_r = n.input_redundancy(mode='node',bound='upper',norm=True) , 1.
	assert (k_r ==  true_k_r) , ('Input Redundancy (node,upper bound,normed) for CONTRADICTION node does not match. %s != %s' % (k_r,true_k_r))
	k_r, true_k_r = n.input_redundancy(mode='node',bound='lower',norm=True) , 1.
	assert (k_r ==  true_k_r) , ('Input Redundancy (node,lower bound,normed) for CONTRADICTION node does not match. %s != %s' % (k_r,true_k_r))

	k_r, true_k_r = n.input_redundancy(mode='input',bound='upper',norm=False) , [1.,1.]
	assert (k_r ==  true_k_r) , ('Input Redundancy (input,upper bound) for CONTRADICTION node does not match. %s != %s' % (k_r,true_k_r))
	k_r, true_k_r = n.input_redundancy(mode='input',bound='mean',norm=False) , [1.,1.]
	assert (k_r ==  true_k_r) , ('Input Redundancy (input,mean) for CONTRADICTION node does not match. %s != %s' % (k_r,true_k_r))
	k_r, true_k_r = n.input_redundancy(mode='input',bound='lower',norm=False) , [1.,1.]
	assert (k_r ==  true_k_r) , ('Input Redundancy (input,lower bound) for CONTRADICTION node does not match. %s != %s' % (k_r,true_k_r))
	
	k_r, true_k_r = n.input_redundancy(mode='input',bound='tuple',norm=False) , [(1.0, 1.0), (1.0, 1.0)]
	assert (k_r ==  true_k_r) , ('Input Redundancy (input,tuples) for CONTRADICTION node does not match. %s != %s' % (k_r,true_k_r))

# COPYx1
def test_input_redundancy_COPYx1():
	"""Test Input Redundancy - COPYx1"""
	n = COPYx1()
	k_r, true_k_r = n.input_redundancy(mode='node',bound='upper',norm=False) , 1.
	assert (k_r ==  true_k_r) , ('Input Redundancy (node,upper bound) for COPYx1 node does not match. %s != %s' % (k_r,true_k_r))
	k_r, true_k_r = n.input_redundancy(mode='node',bound='lower',norm=False) , 1.
	assert (k_r ==  true_k_r) , ('Input Redundancy (node,lower bound) for COPYx1 node does not match. %s != %s' % (k_r,true_k_r))

	k_r, true_k_r = n.input_redundancy(mode='node',bound='upper',norm=True) , 1/2
	assert (k_r ==  true_k_r) , ('Input Redundancy (node,upper bound,normed) for COPYx1 node does not match. %s != %s' % (k_r,true_k_r))
	k_r, true_k_r = n.input_redundancy(mode='node',bound='lower',norm=True) , 1/2
	assert (k_r ==  true_k_r) , ('Input Redundancy (node,lower bound,normed) for COPYx1 node does not match. %s != %s' % (k_r,true_k_r))

	k_r, true_k_r = n.input_redundancy(mode='input',bound='upper',norm=False) , [0.,1.]
	assert (k_r ==  true_k_r) , ('Input Redundancy (input,upper bound) for COPYx1 node does not match. %s != %s' % (k_r,true_k_r))
	k_r, true_k_r = n.input_redundancy(mode='input',bound='mean',norm=False) , [0.,1.]
	assert (k_r ==  true_k_r) , ('Input Redundancy (input,mean) for COPYx1 node does not match. %s != %s' % (k_r,true_k_r))
	k_r, true_k_r = n.input_redundancy(mode='input',bound='lower',norm=False) , [0.,1.]
	assert (k_r ==  true_k_r) , ('Input Redundancy (input,lower bound) for COPYx1 node does not match. %s != %s' % (k_r,true_k_r))
	
	k_r, true_k_r = n.input_redundancy(mode='input',bound='tuple',norm=False) , [(0.0, 0.0), (1.0, 1.0)]
	assert (k_r ==  true_k_r) , ('Input Redundancy (input,tuples) for COPYx1 node does not match. %s != %s' % (k_r,true_k_r))

# RULE 90
def test_input_redundancy_RULE90():
	"""Test Input Redundancy - RULE90"""
	n = RULE90()
	k_r, true_k_r = n.input_redundancy(mode='node',bound='upper',norm=False) , 8/8
	assert (k_r ==  true_k_r) , ('Input Redundancy (node,upper bound) for RULE90 node does not match. %s != %s' % (k_r,true_k_r))
	k_r, true_k_r = n.input_redundancy(mode='node',bound='lower',norm=False) , 8/8
	assert (k_r ==  true_k_r) , ('Input Redundancy (node,lower bound) for RULE90 node does not match. %s != %s' % (k_r,true_k_r))

	k_r, true_k_r = n.input_redundancy(mode='node',bound='upper',norm=True) , (8/8)/3
	assert (k_r ==  true_k_r) , ('Input Redundancy (node,upper bound,normed) for RULE90 node does not match. %s != %s' % (k_r,true_k_r))
	k_r, true_k_r = n.input_redundancy(mode='node',bound='lower',norm=True) , (8/8)/3
	assert (k_r ==  true_k_r) , ('Input Redundancy (node,lower bound,normed) for RULE90 node does not match. %s != %s' % (k_r,true_k_r))

	k_r, true_k_r = n.input_redundancy(mode='input',bound='upper',norm=False) , [0.,1.,0.]
	assert (k_r ==  true_k_r) , ('Input Redundancy (input,upper bound) for RULE90 node does not match. %s != %s' % (k_r,true_k_r))
	k_r, true_k_r = n.input_redundancy(mode='input',bound='mean',norm=False) , [0.,1.,0.]
	assert (k_r ==  true_k_r) , ('Input Redundancy (input,mean) for RULE90 node does not match. %s != %s' % (k_r,true_k_r))
	k_r, true_k_r = n.input_redundancy(mode='input',bound='lower',norm=False) , [0.,1.,0.]
	assert (k_r ==  true_k_r) , ('Input Redundancy (input,lower bound) for RULE90 node does not match. %s != %s' % (k_r,true_k_r))
	
	k_r, true_k_r = n.input_redundancy(mode='input',bound='tuple',norm=False) , [(0.0, 0.0), (1.0, 1.0), (0.0, 0.0)]
	assert (k_r ==  true_k_r) , ('Input Redundancy (input,tuples) for RULE90 node does not match. %s != %s' % (k_r,true_k_r))

# RULE 110
def test_input_redundancy_RULE110():
	"""Test Input Redundancy - RULE110"""
	n = RULE110()
	k_r, true_k_r = n.input_redundancy(mode='node',bound='upper',norm=False) , 7/8
	assert (k_r ==  true_k_r) , ('Input Redundancy (node,upper bound) for RULE110 node does not match. %s != %s' % (k_r,true_k_r))
	k_r, true_k_r = n.input_redundancy(mode='node',bound='lower',norm=False) , 7/8
	assert (k_r ==  true_k_r) , ('Input Redundancy (node,lower bound) for RULE110 node does not match. %s != %s' % (k_r,true_k_r))

	k_r, true_k_r = n.input_redundancy(mode='node',bound='upper',norm=True) , (7/8)/3
	assert (k_r ==  true_k_r) , ('Input Redundancy (node,upper bound,normed) for RULE110 node does not match. %s != %s' % (k_r,true_k_r))
	k_r, true_k_r = n.input_redundancy(mode='node',bound='lower',norm=True) , (7/8)/3
	assert (k_r ==  true_k_r) , ('Input Redundancy (node,lower bound,normed) for RULE110 node does not match. %s != %s' % (k_r,true_k_r))

	k_r, true_k_r = n.input_redundancy(mode='input',bound='upper',norm=False) , [6/8,2/8,2/8]
	assert (k_r ==  true_k_r) , ('Input Redundancy (input,upper bound) for RULE110 node does not match. %s != %s' % (k_r,true_k_r))
	k_r, true_k_r = n.input_redundancy(mode='input',bound='mean',norm=False) , [5/8,1/8,1/8]
	assert (k_r ==  true_k_r) , ('Input Redundancy (input,mean) for RULE110 node does not match. %s != %s' % (k_r,true_k_r))
	k_r, true_k_r = n.input_redundancy(mode='input',bound='lower',norm=False) , [4/8.,0/8,0/8]
	assert (k_r ==  true_k_r) , ('Input Redundancy (input,lower bound) for RULE110 node does not match. %s != %s' % (k_r,true_k_r))
	
	k_r, true_k_r = n.input_redundancy(mode='input',bound='tuple',norm=False) , [(0.5, 0.75), (0.0, 0.25), (0.0, 0.25)]
	assert (k_r ==  true_k_r) , ('Input Redundancy (input,tuples) for RULE110 node does not match. %s != %s' % (k_r,true_k_r))


#
# Test Effective Connectivity
#

# AND
def test_effective_connectivity_AND():
	"""Test Effective Connectivity - AND"""
	n = AND()
	k_e, true_k_e = n.effective_connectivity(mode='node',bound='upper',norm=False) , 5/4
	assert (k_e ==  true_k_e) , ('Effective Connectivity (node,upper bound) for AND node does not match. %s != %s' % (k_e,true_k_e))
	k_e, true_k_e = n.effective_connectivity(mode='node',bound='lower',norm=False) , 5/4
	assert (k_e ==  true_k_e) , ('Effective Connectivity (node,lower bound) for AND node does not match. %s != %s' % (k_e,true_k_e))

	k_e, true_k_e = n.effective_connectivity(mode='node',bound='upper',norm=True) , (5/4)/2
	assert (k_e ==  true_k_e) , ('Effective Connectivity (node,upper bound,normed) for AND node does not match. %s != %s' % (k_e,true_k_e))
	k_e, true_k_e = n.effective_connectivity(mode='node',bound='lower',norm=True) , (5/4)/2
	assert (k_e ==  true_k_e) , ('Effective Connectivity (node,lower bound,normed) for AND node does not match. %s != %s' % (k_e,true_k_e))

	k_e, true_k_e = n.effective_connectivity(mode='input',bound='upper',norm=False) , [1-(2/4),1-(2/4)]
	assert (k_e ==  true_k_e) , ('Input Redundancy (input,upper bound) for AND node does not match. %s != %s' % (k_e,true_k_e))
	k_e, true_k_e = n.effective_connectivity(mode='input',bound='mean',norm=False) , [1-(3/8),1-(3/8)]
	assert (k_e ==  true_k_e) , ('Input Redundancy (input,mean) for AND node does not match. %s != %s' % (k_e,true_k_e))
	k_e, true_k_e = n.effective_connectivity(mode='input',bound='lower',norm=False) , [1-(1/4),1-(1/4)]
	assert (k_e ==  true_k_e) , ('Input Redundancy (input,lower bound) for AND node does not match. %s != %s' % (k_e,true_k_e))

# XOR
def test_effective_connectivity_XOR():
	"""Test Effective Connectivity - XOR"""
	n = XOR()
	k_e, true_k_e = n.effective_connectivity(mode='node',bound='upper',norm=False) , 2.
	assert (k_e ==  true_k_e) , ('Effective Connectivity (node,upper bound) for XOR node does not match. %s != %s' % (k_e,true_k_e))
	k_e, true_k_e = n.effective_connectivity(mode='node',bound='lower',norm=False) , 2.
	assert (k_e ==  true_k_e) , ('Effective Connectivity (node,lower bound) for XOR node does not match. %s != %s' % (k_e,true_k_e))

	k_e, true_k_e = n.effective_connectivity(mode='node',bound='upper',norm=True) , 2./2
	assert (k_e ==  true_k_e) , ('Effective Connectivity (node,upper bound,normed) for XOR node does not match. %s != %s' % (k_e,true_k_e))
	k_e, true_k_e = n.effective_connectivity(mode='node',bound='lower',norm=True) , 2./2
	assert (k_e ==  true_k_e) , ('Effective Connectivity (node,lower bound,normed) for XOR node does not match. %s != %s' % (k_e,true_k_e))

	k_e, true_k_e = n.effective_connectivity(mode='input',bound='upper',norm=False) , [1-(0),1-(0)]
	assert (k_e ==  true_k_e) , ('Input Redundancy (input,upper bound) for XOR node does not match. %s != %s' % (k_e,true_k_e))
	k_e, true_k_e = n.effective_connectivity(mode='input',bound='mean',norm=False) , [1-(0),1-(0)]
	assert (k_e ==  true_k_e) , ('Input Redundancy (input,mean) for XOR node does not match. %s != %s' % (k_e,true_k_e))
	k_e, true_k_e = n.effective_connectivity(mode='input',bound='lower',norm=False) , [1-(0),1-(0)]
	assert (k_e ==  true_k_e) , ('Input Redundancy (input,lower bound) for XOR node does not match. %s != %s' % (k_e,true_k_e))

# CONTRADICTION
def test_effective_connectivity_CONTRADICTION():
	"""Test Effective Connectivity - CONTRADICTION"""
	n = CONTRADICTION()
	k_e, true_k_e = n.effective_connectivity(mode='node',bound='upper',norm=False) , 0
	assert (k_e ==  true_k_e) , ('Effective Connectivity (node,upper bound) for CONTRADICTION node does not match. %s != %s' % (k_e,true_k_e))
	k_e, true_k_e = n.effective_connectivity(mode='node',bound='lower',norm=False) , 0
	assert (k_e ==  true_k_e) , ('Effective Connectivity (node,lower bound) for CONTRADICTION node does not match. %s != %s' % (k_e,true_k_e))

	k_e, true_k_e = n.effective_connectivity(mode='node',bound='upper',norm=True) , 0
	assert (k_e ==  true_k_e) , ('Effective Connectivity (node,upper bound,normed) for CONTRADICTION node does not match. %s != %s' % (k_e,true_k_e))
	k_e, true_k_e = n.effective_connectivity(mode='node',bound='lower',norm=True) , 0
	assert (k_e ==  true_k_e) , ('Effective Connectivity (node,lower bound,normed) for CONTRADICTION node does not match. %s != %s' % (k_e,true_k_e))

	k_e, true_k_e = n.effective_connectivity(mode='input',bound='upper',norm=False) , [0,0]
	assert (k_e ==  true_k_e) , ('Input Redundancy (input,upper bound) for CONTRADICTION node does not match. %s != %s' % (k_e,true_k_e))
	k_e, true_k_e = n.effective_connectivity(mode='input',bound='mean',norm=False) , [0,0]
	assert (k_e ==  true_k_e) , ('Input Redundancy (input,mean) for CONTRADICTION node does not match. %s != %s' % (k_e,true_k_e))
	k_e, true_k_e = n.effective_connectivity(mode='input',bound='lower',norm=False) , [0,0]
	assert (k_e ==  true_k_e) , ('Input Redundancy (input,lower bound) for CONTRADICTION node does not match. %s != %s' % (k_e,true_k_e))

# COPYx1
def test_effective_connectivity_COPYx1():
	"""Test Effective Connectivity - COPYx1"""
	n = COPYx1()
	k_e, true_k_e = n.effective_connectivity(mode='node',bound='upper',norm=False) , 1
	assert (k_e ==  true_k_e) , ('Effective Connectivity (node,upper bound) for COPYx1 node does not match. %s != %s' % (k_e,true_k_e))
	k_e, true_k_e = n.effective_connectivity(mode='node',bound='lower',norm=False) , 1
	assert (k_e ==  true_k_e) , ('Effective Connectivity (node,lower bound) for COPYx1 node does not match. %s != %s' % (k_e,true_k_e))

	k_e, true_k_e = n.effective_connectivity(mode='node',bound='upper',norm=True) , 1/2
	assert (k_e ==  true_k_e) , ('Effective Connectivity (node,upper bound,normed) for COPYx1 node does not match. %s != %s' % (k_e,true_k_e))
	k_e, true_k_e = n.effective_connectivity(mode='node',bound='lower',norm=True) , 1/2
	assert (k_e ==  true_k_e) , ('Effective Connectivity (node,lower bound,normed) for COPYx1 node does not match. %s != %s' % (k_e,true_k_e))

	k_e, true_k_e = n.effective_connectivity(mode='input',bound='upper',norm=False) , [1,0]
	assert (k_e ==  true_k_e) , ('Input Redundancy (input,upper bound) for COPYx1 node does not match. %s != %s' % (k_e,true_k_e))
	k_e, true_k_e = n.effective_connectivity(mode='input',bound='mean',norm=False) , [1,0]
	assert (k_e ==  true_k_e) , ('Input Redundancy (input,mean) for COPYx1 node does not match. %s != %s' % (k_e,true_k_e))
	k_e, true_k_e = n.effective_connectivity(mode='input',bound='lower',norm=False) , [1,0]
	assert (k_e ==  true_k_e) , ('Input Redundancy (input,lower bound) for COPYx1 node does not match. %s != %s' % (k_e,true_k_e))

# RULE90
def test_effective_connectivity_RULE90():
	"""Test Effective Connectivity - RULE90"""
	n = RULE90()
	k_e, true_k_e = n.effective_connectivity(mode='node',bound='upper',norm=False) , 3-1
	assert (k_e ==  true_k_e) , ('Effective Connectivity (node,upper bound) for RULE90 node does not match. %s != %s' % (k_e,true_k_e))
	k_e, true_k_e = n.effective_connectivity(mode='node',bound='lower',norm=False) , 3-1
	assert (k_e ==  true_k_e) , ('Effective Connectivity (node,lower bound) for RULE90 node does not match. %s != %s' % (k_e,true_k_e))

	k_e, true_k_e = n.effective_connectivity(mode='node',bound='upper',norm=True) , (3-1)/3
	assert (k_e ==  true_k_e) , ('Effective Connectivity (node,upper bound,normed) for RULE90 node does not match. %s != %s' % (k_e,true_k_e))
	k_e, true_k_e = n.effective_connectivity(mode='node',bound='lower',norm=True) , (3-1)/3
	assert (k_e ==  true_k_e) , ('Effective Connectivity (node,lower bound,normed) for RULE90 node does not match. %s != %s' % (k_e,true_k_e))

	k_e, true_k_e = n.effective_connectivity(mode='input',bound='upper',norm=False) , [1,0,1]
	assert (k_e ==  true_k_e) , ('Input Redundancy (input,upper bound) for RULE90 node does not match. %s != %s' % (k_e,true_k_e))
	k_e, true_k_e = n.effective_connectivity(mode='input',bound='mean',norm=False) , [1,0,1]
	assert (k_e ==  true_k_e) , ('Input Redundancy (input,mean) for RULE90 node does not match. %s != %s' % (k_e,true_k_e))
	k_e, true_k_e = n.effective_connectivity(mode='input',bound='lower',norm=False) , [1,0,1]
	assert (k_e ==  true_k_e) , ('Input Redundancy (input,lower bound) for RULE90 node does not match. %s != %s' % (k_e,true_k_e))

# RULE110
def test_effective_connectivity_RULE110():
	"""Test Effective Connectivity - RULE110"""
	n = RULE110()
	k_e, true_k_e = n.effective_connectivity(mode='node',bound='upper',norm=False) , 3 - (7/8)
	assert (k_e ==  true_k_e) , ('Effective Connectivity (node,upper bound) for RULE110 node does not match. %s != %s' % (k_e,true_k_e))
	k_e, true_k_e = n.effective_connectivity(mode='node',bound='lower',norm=False) , 3 - (7/8)
	assert (k_e ==  true_k_e) , ('Effective Connectivity (node,lower bound) for RULE110 node does not match. %s != %s' % (k_e,true_k_e))

	k_e, true_k_e = n.effective_connectivity(mode='node',bound='upper',norm=True) , (3 - (7/8))/3
	assert (k_e ==  true_k_e) , ('Effective Connectivity (node,upper bound,normed) for RULE110 node does not match. %s != %s' % (k_e,true_k_e))
	k_e, true_k_e = n.effective_connectivity(mode='node',bound='lower',norm=True) , (3 - (7/8))/3
	assert (k_e ==  true_k_e) , ('Effective Connectivity (node,lower bound,normed) for RULE110 node does not match. %s != %s' % (k_e,true_k_e))

	k_e, true_k_e = n.effective_connectivity(mode='input',bound='upper',norm=False) , [0.25,0.75,0.75]
	assert (k_e ==  true_k_e) , ('Input Redundancy (input,upper bound) for RULE110 node does not match. %s != %s' % (k_e,true_k_e))
	k_e, true_k_e = n.effective_connectivity(mode='input',bound='mean',norm=False) , [0.375,0.875,0.875]
	assert (k_e ==  true_k_e) , ('Input Redundancy (input,mean) for RULE110 node does not match. %s != %s' % (k_e,true_k_e))
	k_e, true_k_e = n.effective_connectivity(mode='input',bound='lower',norm=False) , [0.5,1.,1.]
	assert (k_e ==  true_k_e) , ('Input Redundancy (input,lower bound) for RULE110 node does not match. %s != %s' % (k_e,true_k_e))


#
# Test Input Redundancy
#

# AND
def test_input_symmetry_AND():
	"""Test Input Symmetry - AND"""
	n = AND()
	k_s, true_k_s = n.input_symmetry(mode='node',bound='upper',norm=False) , 8/4
	assert (k_s ==  true_k_s) , ('Input Symmetry (node,upper bound) for AND node does not match. %s != %s' % (k_s,true_k_s))
	k_s, true_k_s = n.input_symmetry(mode='node',bound='lower',norm=False) , 8/4
	assert (k_s ==  true_k_s) , ('Input Symmetry (node,lower bound) for AND node does not match. %s != %s' % (k_s,true_k_s))

	k_s, true_k_s = n.input_symmetry(mode='node',bound='upper',norm=True) , (8/4)/2
	assert (k_s ==  true_k_s) , ('Input Symmetry (node,upper bound,normed) for AND node does not match. %s != %s' % (k_s,true_k_s))
	k_s, true_k_s = n.input_symmetry(mode='node',bound='lower',norm=True) , (8/4)/2
	assert (k_s ==  true_k_s) , ('Input Symmetry (node,lower bound,normed) for AND node does not match. %s != %s' % (k_s,true_k_s))

	k_s, true_k_s = n.input_symmetry(mode='input',bound='upper',norm=False) , [2.,2.]
	assert (k_s ==  true_k_s) , ('Input Symmetry (input,upper bound) for AND node does not match. %s != %s' % (k_s,true_k_s))
	k_s, true_k_s = n.input_symmetry(mode='input',bound='mean',norm=False) , [2.,2.]
	assert (k_s ==  true_k_s) , ('Input Symmetry (input,mean) for AND node does not match. %s != %s' % (k_s,true_k_s))
	k_s, true_k_s = n.input_symmetry(mode='input',bound='lower',norm=False) , [2.,2.]
	assert (k_s ==  true_k_s) , ('Input Symmetry (input,lower bound) for AND node does not match. %s != %s' % (k_s,true_k_s))
	
	k_s, true_k_s = n.input_symmetry(mode='input',bound='tuple',norm=False) , [ [(2,2)]*4 , [(2,2)]*4 ]
	assert (k_s ==  true_k_s) , ('Input Redundancy (input,tuples) for AND node does not match. %s != %s' % (k_s,true_k_s))

# OR
def test_input_symmetry_OR():
	"""Test Input Symmetry - OR"""
	n = OR()
	k_s, true_k_s = n.input_symmetry(mode='node',bound='upper',norm=False) , 8/4
	assert (k_s ==  true_k_s) , ('Input Symmetry (node,upper bound) for OR node does not match. %s != %s' % (k_s,true_k_s))
	k_s, true_k_s = n.input_symmetry(mode='node',bound='lower',norm=False) , 8/4
	assert (k_s ==  true_k_s) , ('Input Symmetry (node,lower bound) for OR node does not match. %s != %s' % (k_s,true_k_s))

	k_s, true_k_s = n.input_symmetry(mode='node',bound='upper',norm=True) , (8/4)/2
	assert (k_s ==  true_k_s) , ('Input Symmetry (node,upper bound,normed) for OR node does not match. %s != %s' % (k_s,true_k_s))
	k_s, true_k_s = n.input_symmetry(mode='node',bound='lower',norm=True) , (8/4)/2
	assert (k_s ==  true_k_s) , ('Input Symmetry (node,lower bound,normed) for OR node does not match. %s != %s' % (k_s,true_k_s))

	k_s, true_k_s = n.input_symmetry(mode='input',bound='upper',norm=False) , [2.,2.]
	assert (k_s ==  true_k_s) , ('Input Symmetry (input,upper bound) for OR node does not match. %s != %s' % (k_s,true_k_s))
	k_s, true_k_s = n.input_symmetry(mode='input',bound='mean',norm=False) , [2.,2.]
	assert (k_s ==  true_k_s) , ('Input Symmetry (input,mean) for OR node does not match. %s != %s' % (k_s,true_k_s))
	k_s, true_k_s = n.input_symmetry(mode='input',bound='lower',norm=False) , [2.,2.]
	assert (k_s ==  true_k_s) , ('Input Symmetry (input,lower bound) for OR node does not match. %s != %s' % (k_s,true_k_s))
	
	k_s, true_k_s = n.input_symmetry(mode='input',bound='tuple',norm=False) , [ [(2,2)]*4 , [(2,2)]*4 ]
	assert (k_s ==  true_k_s) , ('Input Redundancy (input,tuples) for XOR node does not match. %s != %s' % (k_s,true_k_s))

# XOR
def test_input_symmetry_XOR():
	"""Test Input Symmetry - XOR"""
	n = XOR()
	k_s, true_k_s = n.input_symmetry(mode='node',bound='upper',norm=False) , 8/4
	assert (k_s ==  true_k_s) , ('Input Symmetry (node,upper bound) for XOR node does not match. %s != %s' % (k_s,true_k_s))
	k_s, true_k_s = n.input_symmetry(mode='node',bound='lower',norm=False) , 8/4
	assert (k_s ==  true_k_s) , ('Input Symmetry (node,lower bound) for XOR node does not match. %s != %s' % (k_s,true_k_s))

	k_s, true_k_s = n.input_symmetry(mode='node',bound='upper',norm=True) , (8/4)/2
	assert (k_s ==  true_k_s) , ('Input Symmetry (node,upper bound,normed) for XOR node does not match. %s != %s' % (k_s,true_k_s))
	k_s, true_k_s = n.input_symmetry(mode='node',bound='lower',norm=True) , (8/4)/2
	assert (k_s ==  true_k_s) , ('Input Symmetry (node,lower bound,normed) for XOR node does not match. %s != %s' % (k_s,true_k_s))

	k_s, true_k_s = n.input_symmetry(mode='input',bound='upper',norm=False) , [2.,2.]
	assert (k_s ==  true_k_s) , ('Input Symmetry (input,upper bound) for XOR node does not match. %s != %s' % (k_s,true_k_s))
	k_s, true_k_s = n.input_symmetry(mode='input',bound='mean',norm=False) , [2.,2.]
	assert (k_s ==  true_k_s) , ('Input Symmetry (input,mean) for XOR node does not match. %s != %s' % (k_s,true_k_s))
	k_s, true_k_s = n.input_symmetry(mode='input',bound='lower',norm=False) , [2.,2.]
	assert (k_s ==  true_k_s) , ('Input Symmetry (input,lower bound) for XOR node does not match. %s != %s' % (k_s,true_k_s))
	
	k_s, true_k_s = n.input_symmetry(mode='input',bound='tuple',norm=False) , [ [(2,2)]*4 , [(2,2)]*4 ]
	assert (k_s ==  true_k_s) , ('Input Redundancy (input,tuples) for XOR node does not match. %s != %s' % (k_s,true_k_s))


# CONTRADICTION
def test_input_symmetry_CONTRADICTION():
	"""Test Input Symmetry - CONTRADICTION"""
	n = CONTRADICTION()
	k_s, true_k_s = n.input_symmetry(mode='node',bound='upper',norm=False) , 8/4
	assert (k_s ==  true_k_s) , ('Input Symmetry (node,upper bound) for CONTRADICTION node does not match. %s != %s' % (k_s,true_k_s))
	k_s, true_k_s = n.input_symmetry(mode='node',bound='lower',norm=False) , 8/4
	assert (k_s ==  true_k_s) , ('Input Symmetry (node,lower bound) for CONTRADICTION node does not match. %s != %s' % (k_s,true_k_s))

	k_s, true_k_s = n.input_symmetry(mode='node',bound='upper',norm=True) , (8/4)/2
	assert (k_s ==  true_k_s) , ('Input Symmetry (node,upper bound,normed) for CONTRADICTION node does not match. %s != %s' % (k_s,true_k_s))
	k_s, true_k_s = n.input_symmetry(mode='node',bound='lower',norm=True) , (8/4)/2
	assert (k_s ==  true_k_s) , ('Input Symmetry (node,lower bound,normed) for CONTRADICTION node does not match. %s != %s' % (k_s,true_k_s))

	k_s, true_k_s = n.input_symmetry(mode='input',bound='upper',norm=False) , [2.,2.]
	assert (k_s ==  true_k_s) , ('Input Symmetry (input,upper bound) for CONTRADICTION node does not match. %s != %s' % (k_s,true_k_s))
	k_s, true_k_s = n.input_symmetry(mode='input',bound='mean',norm=False) , [2.,2.]
	assert (k_s ==  true_k_s) , ('Input Symmetry (input,mean) for CONTRADICTION node does not match. %s != %s' % (k_s,true_k_s))
	k_s, true_k_s = n.input_symmetry(mode='input',bound='lower',norm=False) , [2.,2.]
	assert (k_s ==  true_k_s) , ('Input Symmetry (input,lower bound) for CONTRADICTION node does not match. %s != %s' % (k_s,true_k_s))
	
	k_s, true_k_s = n.input_symmetry(mode='input',bound='tuple',norm=False) , [ [(2,2)]*4 , [(2,2)]*4 ]
	assert (k_s ==  true_k_s) , ('Input Redundancy (input,tuples) for CONTRADICTION node does not match. %s != %s' % (k_s,true_k_s))

# COPYx1
def test_input_symmetry_COPYx1():
	"""Test Input Symmetry - COPYx1"""
	n = COPYx1()
	k_s, true_k_s = n.input_symmetry(mode='node',bound='upper',norm=False) , 0/4
	assert (k_s ==  true_k_s) , ('Input Symmetry (node,upper bound) for COPYx1 node does not match. %s != %s' % (k_s,true_k_s))
	k_s, true_k_s = n.input_symmetry(mode='node',bound='lower',norm=False) , 0/4
	assert (k_s ==  true_k_s) , ('Input Symmetry (node,lower bound) for COPYx1 node does not match. %s != %s' % (k_s,true_k_s))

	k_s, true_k_s = n.input_symmetry(mode='node',bound='upper',norm=True) , (0/4)/2
	assert (k_s ==  true_k_s) , ('Input Symmetry (node,upper bound,normed) for COPYx1 node does not match. %s != %s' % (k_s,true_k_s))
	k_s, true_k_s = n.input_symmetry(mode='node',bound='lower',norm=True) , (0/4)/2
	assert (k_s ==  true_k_s) , ('Input Symmetry (node,lower bound,normed) for COPYx1 node does not match. %s != %s' % (k_s,true_k_s))

	k_s, true_k_s = n.input_symmetry(mode='input',bound='upper',norm=False) , [0.,0.]
	assert (k_s ==  true_k_s) , ('Input Symmetry (input,upper bound) for COPYx1 node does not match. %s != %s' % (k_s,true_k_s))
	k_s, true_k_s = n.input_symmetry(mode='input',bound='mean',norm=False) , [0.,0.]
	assert (k_s ==  true_k_s) , ('Input Symmetry (input,mean) for COPYx1 node does not match. %s != %s' % (k_s,true_k_s))
	k_s, true_k_s = n.input_symmetry(mode='input',bound='lower',norm=False) , [0.,0.]
	assert (k_s ==  true_k_s) , ('Input Symmetry (input,lower bound) for COPYx1 node does not match. %s != %s' % (k_s,true_k_s))
	
	k_s, true_k_s = n.input_symmetry(mode='input',bound='tuple',norm=False) , [ [(0,0)]*4 , [(0,0)]*4 ]
	assert (k_s ==  true_k_s) , ('Input Redundancy (input,tuples) for COPYx1 node does not match. %s != %s' % (k_s,true_k_s))

# RULE90
def test_input_symmetry_RULE90():
	"""Test Input Symmetry - RULE90"""
	n = RULE90()
	k_s, true_k_s = n.input_symmetry(mode='node',bound='upper',norm=False) , 4/3
	assert (k_s ==  true_k_s) , ('Input Symmetry (node,upper bound) for RULE90 node does not match. %s != %s' % (k_s,true_k_s))
	k_s, true_k_s = n.input_symmetry(mode='node',bound='lower',norm=False) , 4/3
	assert (k_s ==  true_k_s) , ('Input Symmetry (node,lower bound) for RULE90 node does not match. %s != %s' % (k_s,true_k_s))

	k_s, true_k_s = n.input_symmetry(mode='node',bound='upper',norm=True) , (4/3)/3
	assert (k_s ==  true_k_s) , ('Input Symmetry (node,upper bound,normed) for RULE90 node does not match. %s != %s' % (k_s,true_k_s))
	k_s, true_k_s = n.input_symmetry(mode='node',bound='lower',norm=True) , (4/3)/3
	assert (k_s ==  true_k_s) , ('Input Symmetry (node,lower bound,normed) for RULE90 node does not match. %s != %s' % (k_s,true_k_s))

	k_s, true_k_s = n.input_symmetry(mode='input',bound='upper',norm=False) , [8/4,0.,8/4]
	assert (k_s ==  true_k_s) , ('Input Symmetry (input,upper bound) for RULE90 node does not match. %s != %s' % (k_s,true_k_s))
	k_s, true_k_s = n.input_symmetry(mode='input',bound='mean',norm=False) , [8/4.,0.,8/4]
	assert (k_s ==  true_k_s) , ('Input Symmetry (input,mean) for RULE90 node does not match. %s != %s' % (k_s,true_k_s))
	k_s, true_k_s = n.input_symmetry(mode='input',bound='lower',norm=False) , [8/4.,0.,8/4]
	assert (k_s ==  true_k_s) , ('Input Symmetry (input,lower bound) for RULE90 node does not match. %s != %s' % (k_s,true_k_s))
	
	k_s, true_k_s = n.input_symmetry(mode='input',bound='tuple',norm=False) , [ [(2,2)]*8 , [(0,0)]*8 , [(2,2)]*8 ]
	assert (k_s ==  true_k_s) , ('Input Redundancy (input,tuples) for RULE90 node does not match. %s != %s' % (k_s,true_k_s))

# RULE110
def test_input_symmetry_RULE110():
	"""Test Input Symmetry - RULE110"""
	n = RULE110()
	k_s, true_k_s = n.input_symmetry(mode='node',bound='upper',norm=False) , (13/8 + 17/8 + 17/8)/3
	assert (k_s ==  true_k_s) , ('Input Symmetry (node,upper bound) for RULE110 node does not match. %s != %s' % (k_s,true_k_s))
	k_s, true_k_s = n.input_symmetry(mode='node',bound='lower',norm=False) , (3/8 + 11/8 + 11/8)/3
	assert (k_s ==  true_k_s) , ('Input Symmetry (node,lower bound) for RULE110 node does not match. %s != %s' % (k_s,true_k_s))

	k_s, true_k_s = n.input_symmetry(mode='node',bound='upper',norm=True) , ((13/8 + 17/8 + 17/8)/3)/3
	assert (k_s ==  true_k_s) , ('Input Symmetry (node,upper bound,normed) for RULE110 node does not match. %s != %s' % (k_s,true_k_s))
	k_s, true_k_s = n.input_symmetry(mode='node',bound='lower',norm=True) , ((3/8 + 11/8 + 11/8)/3)/3
	assert (k_s ==  true_k_s) , ('Input Symmetry (node,lower bound,normed) for RULE110 node does not match. %s != %s' % (k_s,true_k_s))

	k_s, true_k_s = n.input_symmetry(mode='input',bound='upper',norm=False) , [13/8 , 17/8 , 17/8]
	assert (k_s ==  true_k_s) , ('Input Symmetry (input,upper bound) for RULE110 node does not match. %s != %s' % (k_s,true_k_s))
	k_s, true_k_s = n.input_symmetry(mode='input',bound='mean',norm=False) , [0.95833333333333326, 1.8333333333333333, 1.8333333333333333] # NOT SURE ITS CORRECT!!!
	assert (k_s ==  true_k_s) , ('Input Symmetry (input,mean) for RULE110 node does not match. %s != %s' % (k_s,true_k_s))
	k_s, true_k_s = n.input_symmetry(mode='input',bound='lower',norm=False) , [0.375, 1.375, 1.375] # NOT SURE ITS CORRECT!!!
	assert (k_s ==  true_k_s) , ('Input Symmetry (input,lower bound) for RULE110 node does not match. %s != %s' % (k_s,true_k_s))
	
	k_s, true_k_s = n.input_symmetry(mode='input',bound='tuple',norm=False) , [[(0, 0), (0, 2), (0, 2), (0, 2), (0, 0), (0, 2), (0, 2), (3, 3)], [(2, 2), (2, 2), (0, 2), (0, 2), (2, 2), (2, 2), (0, 2), (3, 3)], [(2, 2), (0, 2), (2, 2), (0, 2), (2, 2), (0, 2), (2, 2), (3, 3)]]
	assert (k_s ==  true_k_s) , ('Input Redundancy (input,tuples) for RULE110 node does not match. %s != %s' % (k_s,true_k_s))












