import networkx as nx
import numpy as np
from itertools import takewhile
import copy
import math
import operator as op
from functools import reduce
from cana.cutils import *


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
    return [flip_bit_in_strstates(strstates, idx) for idx in idxs]


def flip_binstate_bit_set(binstate, idxs):
    """Flips the binary value for a set of bits in a binary state.

    Args:
        binstate (string) : The binary state to flip.
        idxs (int) : The indexes of the bits to flip.

    Returns:
        (list) : The flipped states
    """
    flipset = []
    if (len(idxs) != 0):
        fb = idxs.pop()
        flipset.extend(flip_binstate_bit_set(binstate, copy.copy(idxs)))
        flipset.extend(flip_binstate_bit_set(flip_binstate_bit(binstate, fb), copy.copy(idxs)))
    else:
        flipset.append(binstate)
    return flipset


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


def entropy(prob_vector, logbase=2.):
    """Calculates the entropy given a probability vector

    Todo:
        This should be calculated using ``scipy.entropy``
    """
    prob_vector = np.array(prob_vector)
    pos_prob_vector = prob_vector[prob_vector > 0]
    return - np.sum(pos_prob_vector * np.log(pos_prob_vector) / np.log(logbase))


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
    total = 2**len(input_list)  # Total combinations to try
    output_list = []
    for i in range(total):
        trial_string = statenum_to_binstate(i, len(input_list))
        # Evaluate trial_string by assigning value to each input variable
        for j, input in enumerate(input_list):
            exec(input + '=' + trial_string[j])
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


def pathlength(p, weights, rule='sum'):
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
    if rule == 'sum':
        return sum(weights[(p[ie], p[ie + 1])] for ie in range(len(p) - 1))
    elif rule == 'prod':
        return np.prod([weights[(p[ie], p[ie + 1])] for ie in range(len(p) - 1)])
    elif rule == 'min':
        return min(weights[(p[ie], p[ie + 1])] for ie in range(len(p) - 1))
    elif rule == 'max':
        return max(weights[(p[ie], p[ie + 1])] for ie in range(len(p) - 1))
