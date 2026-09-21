# -*- coding: utf-8 -*-
"""
Sensitivity
===========

Average sensitivity of a Boolean node, computed directly on its Look Up Table (LUT).
:func:`cana.boolean_node.BooleanNode.sensitivity` delegates to :func:`sensitivity`;
:func:`sensitivity_old` keeps the previous formulation for reference and testing.

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


def sensitivity_old(node, norm=False):
    """Average sensitivity as computed by ``BooleanNode.sensitivity`` up to CANA 1.0.2.

    Sums the edge activities, ``sum(node.activities())``. Each activity is the upper
    bound of the edge effectiveness, :math:`1 - r_i` with :math:`r_i` the edge
    redundancy read from the prime-implicant coverage of the LUT. That makes this
    formulation depend on the Quine-McCluskey prime implicants and the coverage map
    being computed first, which :func:`sensitivity` avoids.

    Kept, unchanged, as the reference the direct computation is checked against:
    ``tests/test_boolean_node.py`` asserts that :func:`sensitivity` and this function
    return bit-exactly equal values. Not used by the library itself.

    Args:
        node (BooleanNode) : the node; only ``node.activities()`` and ``node.k`` are used.
        norm (bool) : Normalize by the number of inputs ``k``.

    Returns:
        (float)

    See also:
        :func:`sensitivity`, :func:`cana.boolean_node.BooleanNode.activities`.
    """
    x = sum(node.activities())
    if norm:
        return x / node.k
    return x
