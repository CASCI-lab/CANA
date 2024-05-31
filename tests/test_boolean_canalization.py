# -*- coding: utf-8 -*-
#
# Tests for ``boolean_canalization.py``
# These tests were hand calculated by Luis M. Rocha and implemented by Rion B. Correia.
# Checks were made with the online tool: http://www.mathematik.uni-marburg.de/~thormae/lectures/ti1/code/qmc/
#
from cana.canalization.cboolean_canalization import find_implicants_qm
from cana.cutils import outputs_to_binstates_of_given_type
from helpers.helper import reorderTwoSymbolOutput, randNode, enumerateImplicants, expandPi 
from cana.canalization.boolean_canalization import find_two_symbols_v2


def test_AND():
    """Test Canalization - AND (k=2, outputs=[0,0,0,1])"""
    k, outputs = 2, '0001'

    # Prime Implicants
    true_pi0 = set(['0#', '#0'])
    true_pi1 = set(['11'])

    bs0 = outputs_to_binstates_of_given_type(outputs=outputs, output='0', k=k)
    bs1 = outputs_to_binstates_of_given_type(outputs=outputs, output='1', k=k)

    pi0, pi1 = find_implicants_qm(bs0), find_implicants_qm(bs1)

    assert (pi0 == true_pi0), ('Prime Implicants for 0 does not match. %s != %s' % (pi0, true_pi0))
    assert (pi1 == true_pi1), ('Prime Implicants for 1 does not match. %s != %s' % (pi1, true_pi1))

    # Two Symbols
    true_ts0 = [('20', [[0, 1]], [])]
    true_ts1 = [('11', [], [[0, 1]])]

    # Temporary fix to make find_two_symbols_v2 work until we update it to work more output types.
    pi0 = set([pi.replace('#', '2') for pi in pi0])
    pi1 = set([pi.replace('#', '2') for pi in pi1])

    ts0, ts1 = find_two_symbols_v2(k=k, prime_implicants=pi0), find_two_symbols_v2(k=k, prime_implicants=pi1)
    
    # convert to unique representation so that equality test works
    true_ts0 = reorderTwoSymbolOutput(true_ts0)
    true_ts1 = reorderTwoSymbolOutput(true_ts1)
    ts0 = reorderTwoSymbolOutput(ts0)
    ts1 = reorderTwoSymbolOutput(ts1)

    assert (ts0 == true_ts0), ('Two Symbol for 0 does not match. %s != %s' % (ts0, true_ts0))
    assert (ts1 == true_ts1), ('Two Symbol for 1 does not match. %s != %s' % (ts1, true_ts1))

def test_K1_constant():
    k, outputs = 1, '00'

    # Prime Implicants
    true_pi0 = set(['#'])
    true_pi1 = set([])

    bs0 = outputs_to_binstates_of_given_type(outputs=outputs, output='0', k=k)
    bs1 = outputs_to_binstates_of_given_type(outputs=outputs, output='1', k=k)

    pi0, pi1 = find_implicants_qm(bs0), find_implicants_qm(bs1)
    print(pi0, pi1)

    assert (pi0 == true_pi0), ('Prime Implicants for 0 does not match. %s != %s' % (pi0, true_pi0))
    assert (pi1 == true_pi1), ('Prime Implicants for 1 does not match. %s != %s' % (pi1, true_pi1))

    # # Two Symbols
    true_ts0 = [('2', [], [])]
    true_ts1 = []

    # Temporary fix to make find_two_symbols_v2 work until we update it to work more output types.
    pi0 = set([pi.replace('#', '2') for pi in pi0])
    pi1 = set([pi.replace('#', '2') for pi in pi1])

    ts0, ts1 = find_two_symbols_v2(k=k, prime_implicants=pi0), find_two_symbols_v2(k=k, prime_implicants=pi1)
    
    # convert to unique representation so that equality test works
    true_ts0 = reorderTwoSymbolOutput(true_ts0)
    true_ts1 = reorderTwoSymbolOutput(true_ts1)
    ts0 = reorderTwoSymbolOutput(ts0)
    ts1 = reorderTwoSymbolOutput(ts1)

    assert (ts0 == true_ts0), ('Two Symbol for 0 does not match. %s != %s' % (ts0, true_ts0))
    assert (ts1 == true_ts1), ('Two Symbol for 1 does not match. %s != %s' % (ts1, true_ts1))

def test_K1_identity():
    k, outputs = 1, '01'

    # Prime Implicants
    true_pi0 = set(['0'])
    true_pi1 = set(["1"])

    bs0 = outputs_to_binstates_of_given_type(outputs=outputs, output='0', k=k)
    bs1 = outputs_to_binstates_of_given_type(outputs=outputs, output='1', k=k)

    pi0, pi1 = find_implicants_qm(bs0), find_implicants_qm(bs1)
    print(pi0, pi1)

    assert (pi0 == true_pi0), ('Prime Implicants for 0 does not match. %s != %s' % (pi0, true_pi0))
    assert (pi1 == true_pi1), ('Prime Implicants for 1 does not match. %s != %s' % (pi1, true_pi1))

    # # Two Symbols
    true_ts0 = [('0', [], [])]
    true_ts1 = [("1", [], [])]

    # Temporary fix to make find_two_symbols_v2 work until we update it to work more output types.
    pi0 = set([pi.replace('#', '2') for pi in pi0])
    pi1 = set([pi.replace('#', '2') for pi in pi1])

    ts0, ts1 = find_two_symbols_v2(k=k, prime_implicants=pi0), find_two_symbols_v2(k=k, prime_implicants=pi1)
    
    # convert to unique representation so that equality test works
    true_ts0 = reorderTwoSymbolOutput(true_ts0)
    true_ts1 = reorderTwoSymbolOutput(true_ts1)
    ts0 = reorderTwoSymbolOutput(ts0)
    ts1 = reorderTwoSymbolOutput(ts1)

    assert (ts0 == true_ts0), ('Two Symbol for 0 does not match. %s != %s' % (ts0, true_ts0))
    assert (ts1 == true_ts1), ('Two Symbol for 1 does not match. %s != %s' % (ts1, true_ts1))

def test_OR():
    """Test Canalization - OR (k=2, outputs=[0,1,1,1])"""
    k, outputs = 2, '0111'
    # Prime Implicants
    true_pi0 = set(['00'])
    true_pi1 = set(['1#', '#1'])

    bs0 = outputs_to_binstates_of_given_type(outputs=outputs, output='0', k=k)
    bs1 = outputs_to_binstates_of_given_type(outputs=outputs, output='1', k=k)

    pi0, pi1 = find_implicants_qm(bs0), find_implicants_qm(bs1)

    assert (pi0 == true_pi0), ('Prime Implicants for 0 does not match. %s != %s' % (pi0, true_pi0))
    assert (pi1 == true_pi1), ('Prime Implicants for 1 does not match. %s != %s' % (pi1, true_pi1))

    # Two Symbols
    true_ts0 = [('00', [], [[0, 1]])]
    true_ts1 = [('12', [[0, 1]], [])]

    # Temporary fix to make find_two_symbols_v2 work until we update it to work more output types.
    pi0 = set([pi.replace('#', '2') for pi in pi0])
    pi1 = set([pi.replace('#', '2') for pi in pi1])

    ts0, ts1 = find_two_symbols_v2(k=k, prime_implicants=pi0), find_two_symbols_v2(k=k, prime_implicants=pi1)
    # convert to unique representation so that equality test works
    true_ts0 = reorderTwoSymbolOutput(true_ts0)
    true_ts1 = reorderTwoSymbolOutput(true_ts1)
    ts0 = reorderTwoSymbolOutput(ts0)
    ts1 = reorderTwoSymbolOutput(ts1)

    assert (ts0 == true_ts0), ('Two Symbol for 0 does not match. %s != %s' % (ts0, true_ts0))
    assert (ts1 == true_ts1), ('Two Symbol for 1 does not match. %s != %s' % (ts1, true_ts1))


def test_XOR():
    """Test Canalization - XOR (k=2, outputs=[0,1,1,0])"""
    k, outputs = 2, '0110'

    true_pi0 = set(['00', '11'])
    true_pi1 = set(['01', '10'])

    bs0 = outputs_to_binstates_of_given_type(outputs=outputs, output='0', k=k)
    bs1 = outputs_to_binstates_of_given_type(outputs=outputs, output='1', k=k)
    pi0, pi1 = find_implicants_qm(bs0), find_implicants_qm(bs1)

    assert (pi0 == true_pi0), ('Prime Implicants for 0 does not match. %s != %s' % (pi0, true_pi0))
    assert (pi1 == true_pi1), ('Prime Implicants for 1 does not match. %s != %s' % (pi1, true_pi1))
    # Two Symbols
    true_ts0 = [('00', [], [[0, 1]]), ('11', [], [[0, 1]])]
    true_ts1 = [('01', [[0, 1]], [])]

    # Temporary fix to make find_two_symbols_v2 work until we update it to work more output types.
    pi0 = set([pi.replace('#', '2') for pi in pi0])
    pi1 = set([pi.replace('#', '2') for pi in pi1])

    ts0, ts1 = find_two_symbols_v2(k=k, prime_implicants=pi0), find_two_symbols_v2(k=k, prime_implicants=pi1)
    # convert to unique representation so that equality test works
    true_ts0 = reorderTwoSymbolOutput(true_ts0)
    true_ts1 = reorderTwoSymbolOutput(true_ts1)
    ts0 = reorderTwoSymbolOutput(ts0)
    ts1 = reorderTwoSymbolOutput(ts1)

    assert (ts0 == true_ts0), ('Two Symbol for 0 does not match. %s != %s' % (ts0, true_ts0))
    assert (ts1 == true_ts1), ('Two Symbol for 1 does not match. %s != %s' % (ts1, true_ts1))


def test_CONTRADICTION():
    """Test Canalization - CONTRADICTION (k=2, outputs=[0,0,0,0])"""
    k, outputs = 2, '0000'
    # Prime Implicants
    true_pi0 = set(['##'])
    true_pi1 = set([])

    bs0 = outputs_to_binstates_of_given_type(outputs=outputs, output='0', k=k)
    bs1 = outputs_to_binstates_of_given_type(outputs=outputs, output='1', k=k)
    pi0, pi1 = find_implicants_qm(bs0), find_implicants_qm(bs1)

    assert (pi0 == true_pi0), ('Prime Implicants for 0 does not match. %s != %s' % (pi0, true_pi0))
    assert (pi1 == true_pi1), ('Prime Implicants for 1 does not match. %s != %s' % (pi1, true_pi1))
    # Two Symbols
    true_ts0 = [('22', [], [[0, 1]])]
    true_ts1 = []

    # Temporary fix to make find_two_symbols_v2 work until we update it to work more output types.
    pi0 = set([pi.replace('#', '2') for pi in pi0])
    pi1 = set([pi.replace('#', '2') for pi in pi1])

    ts0, ts1 = find_two_symbols_v2(k=k, prime_implicants=pi0), find_two_symbols_v2(k=k, prime_implicants=pi1)
    # convert to unique representation so that equality test works
    true_ts0 = reorderTwoSymbolOutput(true_ts0)
    true_ts1 = reorderTwoSymbolOutput(true_ts1)
    ts0 = reorderTwoSymbolOutput(ts0)
    ts1 = reorderTwoSymbolOutput(ts1)

    assert (ts0 == true_ts0), ('Two Symbol for 0 does not match. %s != %s' % (ts0, true_ts0))
    assert (ts1 == true_ts1), ('Two Symbol for 1 does not match. %s != %s' % (ts1, true_ts1))


def test_COPYxi():
    """Test Canalization - COPYxI_1 (k=2, outputs=[0,0,1,1])"""
    k, outputs = 2, '0011'
    # Prime Implicants
    true_pi0 = set(['0#'])
    true_pi1 = set(['1#'])

    bs0 = outputs_to_binstates_of_given_type(outputs=outputs, output='0', k=k)
    bs1 = outputs_to_binstates_of_given_type(outputs=outputs, output='1', k=k)
    pi0, pi1 = find_implicants_qm(bs0), find_implicants_qm(bs1)

    assert (pi0 == true_pi0), ('Prime Implicants for 0 does not match. %s != %s' % (pi0, true_pi0))
    assert (pi1 == true_pi1), ('Prime Implicants for 1 does not match. %s != %s' % (pi1, true_pi1))
    # Two Symbols
    true_ts0 = [('02', [], [])]
    true_ts1 = [('12', [], [])]

    # Temporary fix to make find_two_symbols_v2 work until we update it to work more output types.
    pi0 = set([pi.replace('#', '2') for pi in pi0])
    pi1 = set([pi.replace('#', '2') for pi in pi1])

    ts0, ts1 = find_two_symbols_v2(k=k, prime_implicants=pi0), find_two_symbols_v2(k=k, prime_implicants=pi1)
    # convert to unique representation so that equality test works
    true_ts0 = reorderTwoSymbolOutput(true_ts0)
    true_ts1 = reorderTwoSymbolOutput(true_ts1)
    ts0 = reorderTwoSymbolOutput(ts0)
    ts1 = reorderTwoSymbolOutput(ts1)

    assert (ts0 == true_ts0), ('Two Symbol for 0 does not match. %s != %s' % (ts0, true_ts0))
    assert (ts1 == true_ts1), ('Two Symbol for 1 does not match. %s != %s' % (ts1, true_ts1))


def test_RULE_90():
    """Test Canalization - RULE 90 (k=3, outputs=[0,1,0,1,1,0,1,0])"""
    k, outputs = 3, '01011010'
    # Prime Implicants
    true_pi0 = set(['0#0', '1#1'])
    true_pi1 = set(['0#1', '1#0'])

    bs0 = outputs_to_binstates_of_given_type(outputs=outputs, output='0', k=k)
    bs1 = outputs_to_binstates_of_given_type(outputs=outputs, output='1', k=k)
    pi0, pi1 = find_implicants_qm(bs0), find_implicants_qm(bs1)

    assert (pi0 == true_pi0), ('Prime Implicants for 0 does not match. %s != %s' % (pi0, true_pi0))
    assert (pi1 == true_pi1), ('Prime Implicants for 1 does not match. %s != %s' % (pi1, true_pi1))
    # Two Symbols
    true_ts0 = [('020', [], [[0, 2]]), ('121', [], [[0, 2]])]
    true_ts1 = [('120', [[0, 2]], [])]

    # Temporary fix to make find_two_symbols_v2 work until we update it to work more output types.
    pi0 = set([pi.replace('#', '2') for pi in pi0])
    pi1 = set([pi.replace('#', '2') for pi in pi1])

    ts0, ts1 = find_two_symbols_v2(k=k, prime_implicants=pi0), find_two_symbols_v2(k=k, prime_implicants=pi1)
    # convert to unique representation so that equality test works
    true_ts0 = reorderTwoSymbolOutput(true_ts0)
    true_ts1 = reorderTwoSymbolOutput(true_ts1)
    ts0 = reorderTwoSymbolOutput(ts0)
    ts1 = reorderTwoSymbolOutput(ts1)

    assert (ts0 == true_ts0), ('Two Symbol for 0 does not match. %s != %s' % (ts0, true_ts0))
    assert (ts1 == true_ts1), ('Two Symbol for 1 does not match. %s != %s' % (ts1, true_ts1))


def test_RULE_110():
    """Test Canalization - RULE 110 (k=3, outputs=[0,1,1,1,0,1,1,0])"""
    k, outputs = 3, '01110110'

    true_pi0 = set(['#00', '111'])
    true_pi1 = set(['0#1', '#01', '01#', '#10'])

    bs0 = outputs_to_binstates_of_given_type(outputs=outputs, output='0', k=k)
    bs1 = outputs_to_binstates_of_given_type(outputs=outputs, output='1', k=k)
    pi0, pi1 = find_implicants_qm(bs0), find_implicants_qm(bs1)

    assert (pi0 == true_pi0), ('Prime Implicants for 0 does not match. %s != %s' % (pi0, true_pi0))
    assert (pi1 == true_pi1), ('Prime Implicants for 1 does not match. %s != %s' % (pi1, true_pi1))
    # Two Symbols
    true_ts0 = [('111', [], [[0, 1, 2]]), ('200', [], [[1, 2]])]
    true_ts1 = [('012', [[0, 2]], []), ('012', [[1, 2]], []), ('201', [[0, 1]], []), ('201', [[1, 2]], [])]

    # Temporary fix to make find_two_symbols_v2 work until we update it to work more output types.
    pi0 = set([pi.replace('#', '2') for pi in pi0])
    pi1 = set([pi.replace('#', '2') for pi in pi1])

    ts0, ts1 = find_two_symbols_v2(k=k, prime_implicants=pi0), find_two_symbols_v2(k=k, prime_implicants=pi1)
    # convert to unique representation so that equality test works
    true_ts0 = reorderTwoSymbolOutput(true_ts0)
    true_ts1 = reorderTwoSymbolOutput(true_ts1)
    ts0 = reorderTwoSymbolOutput(ts0)
    ts1 = reorderTwoSymbolOutput(ts1)

    assert (ts0 == true_ts0), ('Two Symbol for 0 does not match. %s != %s' % (ts0, true_ts0))
    assert (ts1 == true_ts1), ('Two Symbol for 1 does not match. %s != %s' % (ts1, true_ts1))

def test_prime_implicants_random():
    """Test if prime implicants are computed correctly on random functions."""

    # generate list of random functions
    nodes_tmp = [randNode(k) for k in range(1, 10) for i in range(500)]
    nodes = {"".join(map(str, n.outputs)): n for n in nodes_tmp}
    # compute prime implicants
    pis = dict()
    for f, n in nodes.items():
        n._check_compute_canalization_variables(prime_implicants=True)
        if "0" in n._prime_implicants:
            zeros = n._prime_implicants["0"]
        else:
            zeros = set()
        if "1" in n._prime_implicants:
            ones = n._prime_implicants["1"]
        else:
            ones = set()
        pis[f] = {"0": set(i.replace("#", "2") for i in zeros),
                  "1": set(i.replace("#", "2") for i in ones)}

    # expand and compare
    for f in nodes:
        truth = enumerateImplicants(f)
        for out in ["0", "1"]:
            expanded = set()
            # for pi in pis[f][out]:
            #     print(pi)
            #     expanded = expanded.union(expandPi(pi))
            expanded = expandPi(pis[f][out])
            assert expanded == truth[out], expanded

if __name__ == "__main__":
    test_prime_implicants_random()