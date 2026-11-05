# -*- coding: utf-8 -*-
"""
Symmetry
=========

Permutation symmetry measures computed on the Look Up Table (LUT) of a node.

"""
from itertools import permutations

from cana.cutils import statenum_to_binstate


def distinct_symmetry(outputs, k):
    """Compute the distinct permutation symmetry of a node LUT.

    For each LUT entry, this computes the fraction of distinct input
    permutations that preserve the same output, excluding the identity
    permutation from both numerator and denominator.

    Args:
        outputs (list) : The LUT outputs, one per input state.
        k (int) : The number of inputs to the node.

    Returns:
        (float)

    See also:
        :func:`raw_symmetry`
    """
    if not outputs:
        return 0.0

    lut = list(map(str, outputs))
    total_ratio = 0.0
    row_count = len(lut)

    # Rows with the same number of 1s share the same distinct permutations.
    perm_cache = {}

    for index, output_symbol in enumerate(lut):
        input_bits = statenum_to_binstate(index, base=k)
        cache_key = (len(input_bits), input_bits.count("1"))

        if cache_key not in perm_cache:
            perm_cache[cache_key] = tuple(
                sorted({"".join(perm) for perm in permutations(input_bits)})
            )

        distinct_perms = perm_cache[cache_key]
        total_perms = len(distinct_perms)

        if total_perms <= 1:
            continue

        matches = 0
        for perm_bits in distinct_perms:
            perm_index = int(perm_bits, 2)
            if lut[perm_index] == output_symbol:
                matches += 1

        total_ratio += (matches - 1) / (total_perms - 1)

    return total_ratio / row_count if row_count else 0.0


def raw_symmetry(outputs, k):
    """Compute the raw symmetry of a node LUT.

    LUT rows are grouped by input Hamming weight. For each row, this computes
    the fraction of rows in the same weight group that have the same output,
    then averages across all LUT rows.

    Args:
        outputs (list) : The LUT outputs, one per input state.
        k (int) : The number of inputs to the node.

    Returns:
        (float)

    See also:
        :func:`distinct_symmetry`
    """
    if not outputs:
        return 0.0

    lut = list(map(str, outputs))
    rows_by_weight = {}

    for index, output_symbol in enumerate(lut):
        input_bits = statenum_to_binstate(index, base=k)
        weight = input_bits.count("1")
        rows_by_weight.setdefault(weight, []).append(output_symbol)

    total_ratio = 0.0
    total_rows = 0

    for symbols in rows_by_weight.values():
        group_size = len(symbols)
        if group_size == 0:
            continue

        counts = {}
        for symbol in symbols:
            counts[symbol] = counts.get(symbol, 0) + 1

        for symbol in symbols:
            total_ratio += counts[symbol] / group_size
            total_rows += 1

    return total_ratio / total_rows if total_rows else 0.0
