# -*- coding: utf-8 -*-
#
# Tests for ``boolean_canalization.py``
# These tests were hand calculated by Luis M. Rocha and implemented by Rion B. Correia.
# Checks were made with the online tool: http://www.mathematik.uni-marburg.de/~thormae/lectures/ti1/code/qmc/
#
from cana.canalization.boolean_canalization import *

#
# 
#
def test_AND():
	"""Test Canalization - AND (k=2, outputs=[0,0,0,1])"""
	k, outputs = 2, [0,0,0,1]
	# Prime Implicants
	true_pi0s = set(['02','20'])
	true_pi1s = set(['11'])

	tdt0, tdt1 = make_transition_density_tables(k=k, outputs=outputs)
	pi0s, pi1s = find_implicants_qm(tdt0) , find_implicants_qm(tdt1)

	assert (pi0s == true_pi0s) , ('Prime Implicants for 0 does not match. %s != %s' % (pi0s,true_pi0s))
	assert (pi1s == true_pi1s) , ('Prime Implicants for 1 does not match. %s != %s' % (pi1s,true_pi1s))
	# Two Symbols
	true_ts0s = [('02',[[0,1]],[])]
	true_ts1s = [('11',[],[[0,1]])]

	ts0s,ts1s = find_two_symbols_v2(k=k, prime_implicants=pi0s) , find_two_symbols_v2(k=k, prime_implicants=pi1s)

	assert (ts0s == true_ts0s) , ('Two Symbol for 0 does not match. %s != %s' % (ts0s,true_ts0s))
	assert (ts1s == true_ts1s) , ('Two Symbol for 1 does not match. %s != %s' % (ts1s,true_ts1s))

def test_OR():
	"""Test Canalization - OR (k=2, outputs=[0,1,1,1])"""
	k, outputs = 2, [0,1,1,1]
	# Prime Implicants
	true_pi0s = set(['00'])
	true_pi1s = set(['12','21'])

	tdt0, tdt1 = make_transition_density_tables(k=k, outputs=outputs)
	pi0s, pi1s = find_implicants_qm(tdt0) , find_implicants_qm(tdt1)

	assert (pi0s == true_pi0s) , ('Prime Implicants for 0 does not match. %s != %s' % (pi0s,true_pi0s))
	assert (pi1s == true_pi1s) , ('Prime Implicants for 1 does not match. %s != %s' % (pi1s,true_pi1s))
	# Two Symbols
	true_ts0s = [('00',[],[[0,1]])]
	true_ts1s = [('12',[[0,1]],[])]

	ts0s,ts1s = find_two_symbols_v2(k=k, prime_implicants=pi0s) , find_two_symbols_v2(k=k, prime_implicants=pi1s)

	assert (ts0s == true_ts0s) , ('Two Symbol for 0 does not match. %s != %s' % (ts0s,true_ts0s))
	assert (ts1s == true_ts1s) , ('Two Symbol for 1 does not match. %s != %s' % (ts1s,true_ts1s))

def test_XOR():
	"""Test Canalization - XOR (k=2, outputs=[0,1,1,0])"""
	k, outputs = 2, [0,1,1,0]

	true_pi0s = set(['00','11'])
	true_pi1s = set(['01','10'])

	tdt0, tdt1 = make_transition_density_tables(k=k, outputs=outputs)
	pi0s, pi1s = find_implicants_qm(tdt0) , find_implicants_qm(tdt1)

	assert (pi0s == true_pi0s) , ('Prime Implicants for 0 does not match. %s != %s' % (pi0s,true_pi0s))
	assert (pi1s == true_pi1s) , ('Prime Implicants for 1 does not match. %s != %s' % (pi1s,true_pi1s))
	# Two Symbols
	true_ts0s = [('11',[],[[0,1]]),('00',[],[[0,1]])]
	true_ts1s = [('10',[[0,1]],[])]

	ts0s,ts1s = find_two_symbols_v2(k=k, prime_implicants=pi0s) , find_two_symbols_v2(k=k, prime_implicants=pi1s)

	assert (ts0s == true_ts0s) , ('Two Symbol for 0 does not match. %s != %s' % (ts0s,true_ts0s))
	assert (ts1s == true_ts1s) , ('Two Symbol for 1 does not match. %s != %s' % (ts1s,true_ts1s))


def test_CONTRADICTION():
	"""Test Canalization - CONTRADICTION (k=2, outputs=[0,0,0,0])"""
	k, outputs = 2, [0,0,0,0]
	# Prime Implicants
	true_pi0s = set(['22'])
	true_pi1s = set([])

	tdt0, tdt1 = make_transition_density_tables(k=k, outputs=outputs)
	pi0s, pi1s = find_implicants_qm(tdt0) , find_implicants_qm(tdt1)

	assert (pi0s == true_pi0s) , ('Prime Implicants for 0 does not match. %s != %s' % (pi0s,true_pi0s))
	assert (pi1s == true_pi1s) , ('Prime Implicants for 1 does not match. %s != %s' % (pi1s,true_pi1s))
	# Two Symbols
	true_ts0s = [('22',[],[[0,1]])]
	true_ts1s = []

	ts0s,ts1s = find_two_symbols_v2(k=k, prime_implicants=pi0s) , find_two_symbols_v2(k=k, prime_implicants=pi1s)

	assert (ts0s == true_ts0s) , ('Two Symbol for 0 does not match. %s != %s' % (ts0s,true_ts0s))
	assert (ts1s == true_ts1s) , ('Two Symbol for 1 does not match. %s != %s' % (ts1s,true_ts1s))

def test_COPYxi():
	"""Test Canalization - COPYxI_1 (k=2, outputs=[0,0,1,1])"""
	k, outputs = 2, [0,0,1,1]
	# Prime Implicants
	true_pi0s = set(['02'])
	true_pi1s = set(['12'])

	tdt0, tdt1 = make_transition_density_tables(k=k, outputs=outputs)
	pi0s, pi1s = find_implicants_qm(tdt0) , find_implicants_qm(tdt1)

	assert (pi0s == true_pi0s) , ('Prime Implicants for 0 does not match. %s != %s' % (pi0s,true_pi0s))
	assert (pi1s == true_pi1s) , ('Prime Implicants for 1 does not match. %s != %s' % (pi1s,true_pi1s))
	# Two Symbols
	true_ts0s = [('02',[],[])]
	true_ts1s = [('12',[],[])]

	ts0s,ts1s = find_two_symbols_v2(k=k, prime_implicants=pi0s) , find_two_symbols_v2(k=k, prime_implicants=pi1s)

	assert (ts0s == true_ts0s) , ('Two Symbol for 0 does not match. %s != %s' % (ts0s,true_ts0s))
	assert (ts1s == true_ts1s) , ('Two Symbol for 1 does not match. %s != %s' % (ts1s,true_ts1s))

def test_RULE_90():
	"""Test Canalization - RULE 90 (k=3, outputs=[0,1,0,1,1,0,1,0])"""
	k, outputs = 3, [0,1,0,1,1,0,1,0]
	# Prime Implicants
	true_pi0s = set(['020','121'])
	true_pi1s = set(['021','120'])

	tdt0, tdt1 = make_transition_density_tables(k=k, outputs=outputs)
	pi0s, pi1s = find_implicants_qm(tdt0) , find_implicants_qm(tdt1)

	assert (pi0s == true_pi0s) , ('Prime Implicants for 0 does not match. %s != %s' % (pi0s,true_pi0s))
	assert (pi1s == true_pi1s) , ('Prime Implicants for 1 does not match. %s != %s' % (pi1s,true_pi1s))
	# Two Symbols
	true_ts0s = [('121',[],[[0,2]]),('020',[],[[0,2]])]
	true_ts1s = [('120',[[0,2]],[])]

	ts0s,ts1s = find_two_symbols_v2(k=k, prime_implicants=pi0s) , find_two_symbols_v2(k=k, prime_implicants=pi1s)

	assert (ts0s == true_ts0s) , ('Two Symbol for 0 does not match. %s != %s' % (ts0s,true_ts0s))
	assert (ts1s == true_ts1s) , ('Two Symbol for 1 does not match. %s != %s' % (ts1s,true_ts1s))

def test_RULE_110():
	"""Test Canalization - RULE 110 (k=3, outputs=[0,1,1,1,0,1,1,0])"""
	k, outputs = 3, [0,1,1,1,0,1,1,0]

	true_pi0s = set(['200','111'])
	true_pi1s = set(['021','201','012','210'])

	tdt0, tdt1 = make_transition_density_tables(k=k, outputs=outputs)
	pi0s, pi1s = find_implicants_qm(tdt0) , find_implicants_qm(tdt1)

	assert (pi0s == true_pi0s) , ('Prime Implicants for 0 does not match. %s != %s' % (pi0s,true_pi0s))
	assert (pi1s == true_pi1s) , ('Prime Implicants for 1 does not match. %s != %s' % (pi1s,true_pi1s))
	# Two Symbols
	true_ts0s = [('200',[],[[1,2]]),('111',[],[[0,1,2]])]
	true_ts1s = [('201',[[0,1]],[]),('012',[[1,2]],[]),('201',[[1,2]],[]),('012',[[0,2]],[])]

	ts0s,ts1s = find_two_symbols_v2(k=k, prime_implicants=pi0s) , find_two_symbols_v2(k=k, prime_implicants=pi1s)

	assert (ts0s == true_ts0s) , ('Two Symbol for 0 does not match. %s != %s' % (ts0s,true_ts0s))
	assert (ts1s == true_ts1s) , ('Two Symbol for 1 does not match. %s != %s' % (ts1s,true_ts1s))



