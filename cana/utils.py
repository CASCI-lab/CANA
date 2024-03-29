import copy
import math
import operator as op
from functools import reduce
from itertools import takewhile

import networkx as nx
import numpy as np

from cana.cutils import binstate_to_statenum, flip_binstate_bit, statenum_to_binstate


def flip_bitset_in_strstates(strstates, idxs):
    """Flips the binary value for a set of bits in a binary state.

    Args:
        binstate (string) : The binary state to flip.
        idxs (int) : The indexes of the bits to flip.

    Returns:
        (list) : The flipped states

    Example:
        >>> flip_bit_in_strstates('000',[0, 2])
        ['100','001']
    """
    return [flip_binstate_bit(strstates, idx) for idx in idxs]


def flip_binstate_bit_set(binstate, idxs):
    """Flips the binary value for a set of bits in a binary state.

    Args:
        binstate (string) : The binary state to flip.
        idxs (int) : The indexes of the bits to flip.

    Returns:
        (list) : The flipped states
    """
    flipset = []
    if len(idxs) != 0:
        fb = idxs.pop()
        flipset.extend(flip_binstate_bit_set(binstate, copy.copy(idxs)))
        flipset.extend(
            flip_binstate_bit_set(flip_binstate_bit(binstate, fb), copy.copy(idxs))
        )
    else:
        flipset.append(binstate)
    return flipset


def negate_LUT_input(outputs, idx):
    """For a LUT defined by the output list, it negates the input.

    Args:
        outputs (list) : The output list defining the LUT.
        idxs (int) : The indexes of the input to negate.

    Returns:
        (list) : The new output with input idx negated
    """
    k = int(np.log2(len(outputs)))
    return [
        outputs[
            binstate_to_statenum(
                flip_binstate_bit(statenum_to_binstate(istate, k), idx)
            )
        ]
        for istate in range(len(outputs))
    ]


def print_logic_table(outputs):
    """Print Logic Table

    Args:
        outputs (list) : The transition outputs of the function.

    Returns:
        print : a print-out of the logic table.

    Example:
        >>> print_logic_table([0,0,1,1])
        00 : 0
        01 : 0
        10 : 1
        11 : 1

    """
    k = int(math.log(len(outputs)) / math.log(2))
    for statenum in range(2**k):
        print(statenum_to_binstate(statenum, base=k) + " : " + str(outputs[statenum]))


def entropy(prob_vector, logbase=2.0):
    """Calculates the entropy given a probability vector

    Todo:
        This should be calculated using ``scipy.entropy``
    """
    prob_vector = np.array(prob_vector)
    pos_prob_vector = prob_vector[prob_vector > 0]
    return -np.sum(pos_prob_vector * np.log(pos_prob_vector) / np.log(logbase))


def ncr(n, r):
    """Return the combination number.
    The combination of selecting `r` items from `n` iterms, order doesn't matter.

    Args:
        n (int): number of elements in collection
        r (int): length of combination

    Returns:
        int
    """
    r = min(r, n - r)
    if r == 0:
        return 1
    numer = reduce(op.mul, range(n, n - r, -1))
    denom = reduce(op.mul, range(1, r + 1))
    return numer // denom


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    """Python 2 doesn't have math.isclose()
    Here is an equivalent function
    Use this to tell whether two float numbers are close enough
    considering using == to compare floats is dangerous!
    2.0*3.3 != 3.0*2.2 in python!

    Args:
        a (float) : the first float number
        b (float) : the second float number
        rel_tol (float) : the relative difference threshold between a and b
        abs_tol (float) : absolute difference threshold. not recommended for float

    Returns:
        bool
    """
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def output_transitions(eval_line, input_list):
    """Returns an output list from combinatorically trying all input values

    Args:
        eval_line (string) : logic or arithmetic line to evaluate
        input_list (list) : list of input variables

    Returns:
        list of all possible output transitions (list)

    Example:
        RAS*=(GRB2 or PLCG1) and not GAP

        .. code-block:: python

            >>> eval_line = "(GRB2 or PLCG1) and not GAP"
            >>> input_list = ['GRB2', 'PLCG1', 'GAP']
            >>> output_transitions(eval_line, input_list)
            000
            001
            010
            011
            100
            101
            110
            111

        A variable is dynamically created for each member of the input list
        and assigned the corresponding value from each trail string.
        The original eval_line is then evaluated with each assignment
        which results in the output list [0, 0, 1, 0, 1, 0, 1, 0]
    """
    total = 2 ** len(input_list)  # Total combinations to try
    output_list = []
    for i in range(total):
        trial_string = statenum_to_binstate(i, len(input_list))
        # Evaluate trial_string by assigning value to each input variable
        for j, input in enumerate(input_list):
            exec(input + "=" + trial_string[j])
        output_list.append(int(eval(eval_line)))

    return output_list


def seq_upto(seq, obj):
    """
    TODO: description
    """
    return takewhile(lambda el: el != obj, iter(seq))


def mindist_from_source(G, source):
    """
    TODO: description
    """
    dag = nx.bfs_tree(G, source)
    dist = {}  # stores [node, distance] pair
    for node in nx.topological_sort(dag):
        # pairs of dist,node for all incoming edges
        pairs = [(dist[v][0] + 1, v) for v in dag.pred[node]]
        if pairs:
            dist[node] = min(pairs)
        else:
            dist[node] = (0, node)

    return dist


def pathlength(p, weights, rule="sum"):
    """Calculate the length of path p, with weighted edges, given the length rule of:

    Ars:
        weights:

        rule (str):
            'sum' - sum of edge weights along path
            'prod' - product of edge weights along path
            'min' - minimum of edge weights along path (weakest-link)
            'max' - maximum of edge weights along path

    TODO: update description
    """
    if rule == "sum":
        return sum(weights[(p[ie], p[ie + 1])] for ie in range(len(p) - 1))
    elif rule == "prod":
        return np.prod([weights[(p[ie], p[ie + 1])] for ie in range(len(p) - 1)])
    elif rule == "min":
        return min(weights[(p[ie], p[ie + 1])] for ie in range(len(p) - 1))
    elif rule == "max":
        return max(weights[(p[ie], p[ie + 1])] for ie in range(len(p) - 1))


def function_monotone(outputs, method="exact", nsamples=100, random_seed=None):
    """
    Determine if a given LUT is monotone.

    Here we test every pair of inputs that are Hamming distance 1. (see Goldreich et al 2000)

    Args:
        outputs (list) : The transition outputs of the function.

        method (str) :
            'exact' - test all pairs of inputs
            TODO: 'random' - sample pairs of inputs

        nsamples (int) : when method=='random', specifies the number of samples.

    Returns:
        (Bool) : True if monotone.

    Example:
        >>> is_monotone(outputs=[0,0,0,1])
    """
    # np.random.seed(random_seed)

    k = int(np.log2(len(outputs)))

    # for all input configurations
    for input_confignum in range(2**k):
        input_configbin = statenum_to_binstate(input_confignum, k)

        # for all input states that are 0
        for idx, state in enumerate(input_configbin):
            if state == "0":
                # we flip the 0 and check for monotone along the edge
                next_confignum = binstate_to_statenum(
                    flip_binstate_bit(input_configbin, idx)
                )

                # if the monotone property fails
                if outputs[input_confignum] > outputs[next_confignum]:
                    return False
    return True


def input_monotone(outputs, input_idx, activation=1):
    """
    Determine if a given input is activating or inhibiting in a given function.

    Args:
        outputs (list) : The transition outputs of the function.

        input_idx (int) : The input to test.

        activation (1 or -1) : Whether to test for activation or inhibition.

    Returns:
        (Bool) : True if monotone with respect to activation or inhibition.

    Example:
        >>> input_monotone([0,1,0,0], 0, activation=1) == False
        >>> input_monotone([0,1,0,0], 0, activation=-1) == True
    """

    k = int(np.log2(len(outputs)))

    if k == 1:
        return True
    else:
        monotone_configs = []
        # for all input configurations
        for input_confignum in range(2 ** (k - 1)):
            other_input_configbin = statenum_to_binstate(input_confignum, k - 1)

            input_confignum_0 = binstate_to_statenum(
                other_input_configbin[:input_idx]
                + "0"
                + other_input_configbin[input_idx:]
            )

            input_confignum_1 = binstate_to_statenum(
                other_input_configbin[:input_idx]
                + "1"
                + other_input_configbin[input_idx:]
            )

            if activation == 1:
                monotone_configs.append(
                    outputs[input_confignum_0] <= outputs[input_confignum_1]
                )
            elif activation == -1:
                monotone_configs.append(
                    outputs[input_confignum_0] >= outputs[input_confignum_1]
                )

        return all(monotone_configs)
