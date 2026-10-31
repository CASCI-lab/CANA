import copy
import operator as op
import warnings
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
    """Negate one LUT input; deprecated alias of :meth:`BitLut.negate_input`.

    .. deprecated::
        Use :meth:`cana.function_outputs.bit_lut.BitLut.negate_input` instead.

    Args:
        outputs (list) : The output list defining the LUT.
        idx (int) : The index of the input to negate.

    Returns:
        (list) : The new output with input idx negated
    """
    warnings.warn(
        "cana.utils.negate_LUT_input is deprecated; use BitLut.negate_input instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from cana.function_outputs.bit_lut import BitLut

    if isinstance(outputs, BitLut):
        return outputs.negate_input(idx).to_list()
    return BitLut.from_sequence(outputs).negate_input(idx).to_list()


def print_logic_table(outputs):
    """Print a LUT; deprecated alias of :meth:`BitLut.lut_as_str`.

    .. deprecated::
        Use :meth:`cana.function_outputs.bit_lut.BitLut.lut_as_str` instead.

    Args:
        outputs (list) : The transition outputs of the function.

    Example:
        >>> print_logic_table([0,0,1,1])
        00 : 0
        01 : 0
        10 : 1
        11 : 1
    """
    warnings.warn(
        "cana.utils.print_logic_table is deprecated; use BitLut.lut_as_str instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from cana.function_outputs.bit_lut import BitLut

    if isinstance(outputs, BitLut):
        lut = outputs
    else:
        lut = BitLut.from_sequence(outputs)
    print(lut.lut_as_str())


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
    """Returns an output list from combinatorically trying all input values.

    Each input variable is assigned every possible binary combination (0/1)
    via a namespace dict, and the boolean expression is evaluated with
    ``eval()`` (builtins disabled, but this is not a security boundary).
    Expressions must be trusted boolean rules using ``and``, ``or``,
    ``not``, and parentheses, as produced by CANA model files.

    Refuses when ``len(input_list)`` exceeds
    :data:`~cana.function_outputs.bit_lut.MAX_DENSE_LUT_K`.

    Args:
        eval_line (string) : boolean expression to evaluate (e.g. "A and not B")
        input_list (list) : list of input variable names

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

        Each input variable is assigned the corresponding value from each
        trial string via a namespace dict, and the expression is evaluated
        which results in the output list [0, 0, 1, 0, 1, 0, 1, 0]
    """
    from cana.function_outputs.bit_lut import refuse_dense_lut_k

    refuse_dense_lut_k(len(input_list), via="output_transitions")
    total = 2 ** len(input_list)  # Total combinations to try
    output_list = []
    # Use an explicit namespace dict for exec/eval so that dynamically
    # created variables are visible across calls.  In Python 3.13+
    # (PEP 667) bare exec() inside a function writes to a snapshot of
    # locals that eval() cannot see.
    ns = {}
    safe_globals = {"__builtins__": {}}
    code = compile(eval_line.strip(), "<string>", "eval")
    for i in range(total):
        trial_string = statenum_to_binstate(i, len(input_list))
        # Evaluate trial_string by assigning value to each input variable
        for j, input_name in enumerate(input_list):
            ns[input_name] = int(trial_string[j])
        output_list.append(int(eval(code, safe_globals, ns)))

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
    """Deprecated alias of :func:`~cana.function_properties.unate.is_function_monotone_increasing`.

    .. deprecated::
        Use :func:`cana.function_properties.is_function_monotone_increasing` instead.

    ``method``, ``nsamples``, and ``random_seed`` are ignored (legacy signature).
    """
    warnings.warn(
        "cana.utils.function_monotone is deprecated; "
        "use cana.function_properties.is_function_monotone_increasing instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from cana.function_outputs.bit_lut import BitLut
    from cana.function_properties import is_function_monotone_increasing

    lut = outputs if isinstance(outputs, BitLut) else BitLut.from_sequence(outputs)
    return is_function_monotone_increasing(lut)


def input_monotone(outputs, input_idx, activation=1):
    """Deprecated alias of per-input positive/negative unate checks.

    .. deprecated::
        Use :func:`cana.function_properties.is_input_positive_unate` or
        :func:`cana.function_properties.is_input_negative_unate` instead.

    Example:
        >>> input_monotone([0,1,0,0], 0, activation=1) == False
        >>> input_monotone([0,1,0,0], 0, activation=-1) == True
    """
    warnings.warn(
        "cana.utils.input_monotone is deprecated; "
        "use cana.function_properties.is_input_positive_unate / "
        "is_input_negative_unate instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from cana.function_outputs.bit_lut import BitLut
    from cana.function_properties import (
        is_input_negative_unate,
        is_input_positive_unate,
    )

    lut = outputs if isinstance(outputs, BitLut) else BitLut.from_sequence(outputs)
    if activation == 1:
        return is_input_positive_unate(lut, input_idx)
    if activation == -1:
        return is_input_negative_unate(lut, input_idx)
    raise ValueError("activation must be 1 or -1; got {0!r}".format(activation))
