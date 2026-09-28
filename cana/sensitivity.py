# -*- coding: utf-8 -*-
"""
Sensitivity
===========

Average sensitivity and edge activities of a Boolean node, computed directly on its
Look Up Table (LUT). :func:`sensitivity_old` and :func:`activities_old` keep the previous
formulations for reference and testing.

"""


def sensitivity(outputs, k, norm=False):
    """Average sensitivity of a Boolean function: the mean, over all input states,
    of the number of single-input flips that change the output.

    Up to CANA 1.0.2 :func:`cana.boolean_node.BooleanNode.sensitivity` computed this as
    ``sum(node.activities())``, which goes through the prime-implicant coverage; that
    formulation is kept as :func:`sensitivity_old`. This function computes it directly
    from the LUT with :math:`k 2^k` lookups and needs no canalization variables. The two
    are bit-exactly equal: the count is divided by a power of two, so the result is an
    exactly representable float either way. This is asserted in
    ``tests/test_boolean_node.py`` (``test_sensitivity_matches_original_implementation``).

    Args:
        outputs (list) : The LUT outputs, one per input state, indexed by the integer
            value of the state with input 1 as the most significant bit (CANA's default).
        k (int) : The number of inputs to the node.
        norm (bool) : Normalize by the number of inputs ``k``.

    Returns:
        (float)

    See also:
        :func:`sensitivity_old`,
        :func:`cana.boolean_node.BooleanNode.sensitivity`,
        :func:`cana.boolean_node.BooleanNode.activities`,
        :func:`cana.boolean_node.BooleanNode.c_sensitivity`.
    """
    changes = 0
    for state in range(2**k):
        out = outputs[state]
        for bit in range(k):
            if outputs[state ^ (1 << bit)] != out:
                changes += 1
    x = changes / 2**k
    if norm:
        return x / k
    return x


def activities(outputs, k):
    """Activity of each input: the fraction of input states in which flipping it
    changes the output. Bit-exactly equal to :func:`activities_old`.

    Args:
        outputs (list) : The LUT outputs, indexed with input 1 as the most significant bit.
        k (int) : The number of inputs to the node.

    Returns:
        (list) : The activity of each input.
    """
    changes = [0] * k
    for state in range(2**k):
        out = outputs[state]
        for i in range(k):
            if outputs[state ^ (1 << (k - 1 - i))] != out:
                changes[i] += 1
    return [c / 2**k for c in changes]


def sensitivity_old(node, norm=False):
    """Average sensitivity as computed by ``BooleanNode.sensitivity`` up to CANA 1.0.2.

    Sums the edge activities, ``sum(activities_old(node))``. Each activity is the upper
    bound of the edge effectiveness, :math:`1 - r_i` with :math:`r_i` the edge
    redundancy read from the prime-implicant coverage of the LUT. That makes this
    formulation depend on the Quine-McCluskey prime implicants and the coverage map
    being computed first, which :func:`sensitivity` avoids.

    Kept, unchanged, as the reference the direct computation is checked against:
    ``tests/test_boolean_node.py`` asserts that :func:`sensitivity` and this function
    return bit-exactly equal values. Not used by the library itself.

    Args:
        node (BooleanNode) : the node; only :func:`activities_old` and ``node.k`` are used.
        norm (bool) : Normalize by the number of inputs ``k``.

    Returns:
        (float)

    See also:
        :func:`sensitivity`, :func:`activities_old`.
    """
    x = sum(activities_old(node))
    if norm:
        return x / node.k
    return x


def activities_old(node):
    """Edge activities as computed by ``BooleanNode.activities`` up to CANA 1.0.2,
    ``node.edge_effectiveness(bound="upper")``. Kept as the reference for :func:`activities`.

    Args:
        node (BooleanNode) : the node.

    Returns:
        (list) : The activity of each input.
    """
    return node.edge_effectiveness(bound="upper")
