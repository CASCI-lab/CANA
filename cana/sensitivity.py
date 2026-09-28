# -*- coding: utf-8 -*-
"""
Sensitivity
===========

Average sensitivity and edge activities of a Boolean node, computed directly on its
Look Up Table (LUT). :func:`sensitivity_old` and :func:`activities_old` keep the previous
formulations for reference and testing.

"""


def sensitivity(outputs, k, norm=False):
    """Average sensitivity: the mean, over all input states, of the number of
    single-input flips that change the output, i.e. the sum of the :func:`activities`.
    Bit-exactly equal to :func:`sensitivity_old`.

    Args:
        outputs (list) : The LUT outputs, indexed with input 1 as the most significant bit.
        k (int) : The number of inputs to the node.
        norm (bool) : Normalize by the number of inputs ``k``.

    Returns:
        (float)
    """
    x = sum(activities(outputs, k))
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
