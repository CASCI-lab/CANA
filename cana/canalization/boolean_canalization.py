# -*- coding: utf-8 -*-
"""
Boolean Canalization
=====================

Functions to compute the Quine-McCluskey algorithm.

"""
#   Copyright (C) 2021 by
#   Rion Brattig Correia <rionbr@gmail.com>
#   Alex Gates <ajgates@gmail.com>
#   Etienne Nzabarushimana <enzabaru@indiana.edu>
#   All rights reserved.
#   MIT license.

import itertools
from collections import deque

import numpy as np
import schematodes as sc

from ..cutils import binstate_to_density, statenum_to_binstate

__author__ = """\n""".join(
    [
        "Alex Gates <ajgates@umail.iu.edu>",
        "Etienne Nzabarushimana <enzabaru@indiana.edu>",
        "Rion Brattig Correia <rionbr@gmail.com>",
    ]
)


# Quine-McCluskey Functions
def make_transition_density_tables(k=1, outputs=[0, 1]):
    """This method creates a tuple-of-lists that is used to calculate Prime Implicants in the first step of the Quine-McCluskey algorithm :cite:`Quine:1955`.
    In practice it separates the positive and negative transitions (tuple), then further separates it by counting the number of 1's in each (lists).

    Args:
        k (int) : the ``k`` number of inputs
        outputs (list) : a list of ``[0,1]`` output for each state number.

    Returns:
        tables (tuple) : a tuple where [0] is the negative table and [1] is the positive table.
    """
    # make sure outputs are integers
    outputs = list(map(int, outputs))

    # we need to split up the LUT based on the transition (to either 0 or 1) and the density of 1s in the binstate
    transition_density_tuple = [
        [[] for density in range(k + 1)] for transition in [0, 1]
    ]
    for statenum in range(2**k):
        binstate = statenum_to_binstate(statenum, base=k)
        density = binstate_to_density(binstate)
        transition = outputs[statenum]
        # Account for Dont-Care (2) transition states
        if transition == 2:
            transition_density_tuple[0][density].append(binstate)
            transition_density_tuple[1][density].append(binstate)
        else:
            transition_density_tuple[transition][density].append(binstate)
    #
    return transition_density_tuple


def find_implicants_qmOLD(column, verbose=False):
    """Finds the prime implicants (PI) using the Quine-McCluskey algorithm :cite:`Quine:1955`.

    Args:
        column (list) : A list-of-lists containing the counts of ``1`` for each input.
            This is given by `make_transition_density_tables`.

    Returns:
        PI (set): a set of prime implicants.

    # Authors: Alex Gates and Etienne Nzabarushimana
    """

    N = len(column) - 1

    # we start with an empty set of implicants
    prime_implicants = set()
    done = False

    # repeat the following until no matches are found
    while not done:
        done = True

        # default everything to empty with no matches
        next_column = [set() for _ in range(N + 1)]
        matches = [
            [False for _ in range(len(column[density]))] for density in range(N + 1)
        ]

        # loop through the possible densities
        for density in range(N):
            # compare the implicants from successive densities
            for i, implicant in enumerate(column[density]):
                for j, candidate in enumerate(column[density + 1]):
                    # check if the implicants differ on only one variable
                    match = _adjacent(implicant, candidate)
                    if match:
                        matches[density][i] = matches[density + 1][j] = True
                        matches_density = sum([var != "0" for var in match])
                        next_column[matches_density].add(match)
                        done = False

        # now add back the implicants that were not matched
        for i in range(N + 1):
            for j in range(len(matches[i])):
                if not matches[i][j]:
                    prime_implicants.add(column[i][j])

        # use the simplified table as the starting point of the next pass
        column = [list(g) for g in next_column]

    return prime_implicants


def _adjacent(imp1, imp2):
    """Determine if two implicants are adjacent: ie differ on only one variable.

    Args:
        imp1 (string): implicant 1
        imp1 (string): implicant 2

    Returns:
        (bool)
    """
    differences = 0
    match = []
    for m1, m2 in zip(imp1, imp2):
        if m1 == m2:
            match.append(m1)
        elif differences:
            return False
        else:
            differences += 1
            match.append("2")
    return "".join(match)


def __pi_covers(implicant, input, symbol=["2", "#", 2]):
    """Determines if a minterm is covered by a specific implicant.

    Args:
        implicant (string): the implicant.
        minterm (string): the minterm.

    Returns:
        x (bool): True if covered else False.

    """
    for i, m in zip(implicant, input):
        if i in symbol:
            continue
        if int(i) != int(m):
            return False
    return True


def computes_pi_coverage(k, outputs, prime_implicants):
    """Computes the input coverage by Prime Implicant schematas.

    Args:
        k (int): the number of inputs.
        outpus (list): the list of transition outputs.
        prime_implicants (tuple): a tuple containing a list negative and positive prime implicants. This is returned by `find_implicants_qm`.

    Returns:
        pi_coverage (dict) : a dictionary of coverage where keys are input states and values are lists of the Prime Implicants covering that input.

    Note: based on code from Alex Gates and Etienne Nzabarushimana.
    """
    # make sure outputs are integers
    outputs = list(map(int, outputs))

    pi_coverage = {}
    for statenum in range(2**k):
        binstate = statenum_to_binstate(statenum, base=k)
        pi_coverage[binstate] = covering_implicants = []
        transition = outputs[statenum]
        # Add support for DontCare (2) transition
        if transition == 2:
            transition = [0, 1]
        else:
            transition = [outputs[statenum]]
        for t in transition:
            for prime_implicant in prime_implicants[t]:
                if __pi_covers(prime_implicant, binstate):
                    covering_implicants.append(prime_implicant)
    #
    return pi_coverage


# Two Symbols Functions
def find_two_symbols_v2(k=1, prime_implicants=None, verbose=False, verbose_level=0):
    """This function calculates the permutation, two-symbol (TS), list of schematas.
    This implementation considers '11' and '00' as a possible input permutation.

    Finds totally symmetric groups of input variables on the maximal subsets of the one-symbol schemata.

    Args:
        k (int): The number of inputs.
        prime_implicants (list): The prime implicants computed.

    Returns:
        final_list (list) : The list of two-symbol schematas.

    Note: This is a modification of the original algorithm that can be found in Marques-Pita & Rocha [2013].
    """
    if verbose or verbose_level > 0:
        print("Finding TS schematas...")

    if not prime_implicants:
        return []

    # If this node has no input, yet it affects other nodes (fixed variable)
    if k == 0:
        TSf = []
        for pi in prime_implicants:
            TSf.append((pi, [], []))
        return TSf

    prime_implicants = [[int(c) for c in pi] for pi in prime_implicants]
    tss = sc.schemer(prime_implicants, max_symbol=2)
    TSf = []  # collect the two-symbol schemata and calculate the one-symbol symmetries
    for c in tss:
        representative = c.redescribed_schemata[0]
        bubble_indices = c.bubble_indices
        same_symbols_all = [
            [i for i, _ in enumerate(representative) if representative[i] == x]
            for x in [0, 1, 2]
        ]
        same_symbols = [x for x in same_symbols_all if len(x) > 1]
        representative_str = "".join(map(str, representative))
        TSf.append([representative_str, bubble_indices, same_symbols])
    return TSf


def _calc_ts_complexity(tss, pers):
    """Calculates the complexity of a TS schema
    Complexity = (Number of Schemas + Number of Permutable Symbols + Lenght of each Permutable Symbol)
    """
    return len(tss) + sum([len(per) for ts, per in zip(tss, pers)])


def _check_schema_within_schema(la, lb, dir=None, verbose=False):
    """Check is a Two-Symbol schemata is covered by another.
    This is used to simplify the number of TS schematas returned.
    The arguments for this function are generated by `_expand_ts_logic`.

    Args:
        tsa (list) : A list of :math:`F'` schematas that a Two-Symbol :math:`F''` schemata can cover.
        tsb (list) : A list of :math:`F'` schematas that a Two-Symbol :math:`F''` schemata can cover.
        dir (string) : The direction to check, either ``a`` or ``b`` is in the other.
            Defaults to both directions.
    """
    a_in_b, b_in_a = None, None
    #
    if dir != "b":
        a_in_b = all([(xa in lb) for xa in la])
        if verbose:
            print("%s in %s : %s" % (la, lb, a_in_b))
    if dir != "a":
        b_in_a = all([(xb in la) for xb in lb])
        if verbose:
            print("%s in %s : %s" % (lb, la, b_in_a))
    #
    return a_in_b, b_in_a


def _expand_ts_logic(two_symbols, permut_indexes):
    """Expands the Two-Symbol logic to all possible prime-implicants variations being covered.

    Args:
        two_symbols (list) : Two-Symbol schematas list-of-lists.

    Returns:
        (list) : a list of :math:`F'` covered by this Two-Symbol.
    """
    # If receiving a binary string, convert to list of lists
    if isinstance(two_symbols, str):
        two_symbols = [list(two_symbols)]
    # Queue
    Q = deque()
    Q.extend(two_symbols)
    logics = []
    #
    while Q:
        implicant = np.array(Q.pop())
        for idxs in permut_indexes:
            # Permutation of all possible combinations of the values that are permutable.
            for vals in itertools.permutations(implicant[idxs], len(idxs)):
                # Generate a new schema
                _implicant = np.copy(implicant)
                _implicant[idxs] = vals
                # Insert to list of logics if not already there
                if _implicant.tolist() not in logics:
                    logics.append(_implicant.tolist())
                    Q.append(_implicant.tolist())
    return logics


def _check_schemata_permutations(schema, perm_groups, verbose=None, verbose_level=None):
    """
    schematas = matrix
    perm_groups = lists
    """
    if verbose or verbose_level > 0:
        print("Verifying schemata permutations...")
    # check that all pairs specified in perm_group are still valid
    allowed_perm_groups = []
    for perm_group in perm_groups:
        for i, x in enumerate(perm_group):
            for j in range(i, len(perm_group)):
                y = perm_group[j]
                if not _can_swap_v3(schema, x, y):
                    return None
        allowed_perm_groups.append(perm_group)
    return allowed_perm_groups


def _can_swap_v3(schema, i, j):
    swapped = schema.copy()
    swapped[:, [i, j]] = schema[:, [j, i]]
    for row in swapped:
        if not np.any(np.all(schema == row, axis=1)):
            return False
    return True


def _check_col_counts(counts_matrix, verbose=False, verbose_level=0):
    """This function is used to find permutable symbols.

    Args:
        counts_matrix (numpy.ndarray) : a matrix where rows are inputs and columns are possible input types (0,1 or #)

    Returns:
        perm_groups (list) : a list of the indexes that can be permuted.
    """
    if verbose and verbose_level > 30:
        print("-- Check Col Counts (v3) --")

    counts = {}  # Multi Counts
    perm_groups = []  # A list of groups of Permutable Indexes

    for i, row in enumerate(counts_matrix, start=0):
        # a tuple (hashable) version of the row counts
        row_tuple = tuple(row)

        if row_tuple in counts:
            # we have seen this one before, so add it to the permutation group
            counts[row_tuple].append(i)
        elif np.count_nonzero(row) >= 2:
            # we have not seen this count before, it is not a fixed variable, so create a new entry for it
            counts[row_tuple] = [i]
        else:
            # we will skip fixed variables
            pass

    # Append non-constants that have permutable positions
    for col, idxs in counts.items():
        if verbose and verbose_level > 40:
            print(col, ":", idxs)

        if len(idxs) == 1:
            return -1
        elif len(idxs) >= 1:
            perm_groups.append(idxs)

    if verbose and verbose_level > 40:
        print("counts:", counts)
        print("perm_groups:", perm_groups)

    if len(perm_groups):
        return perm_groups
    else:
        return -1


def _check_identical_cols_count_symbols_v2(
    counts_matrix, verbose=False, verbose_level=0
):
    """This function is used to find same symbol permutables. In practice it is a variance of `_check_cols_symbols_vX`

    Args:
        counts_matrix (numpy.ndarray) : a matrix where rows are inputs and columns are possible input types (0,1 or #)

    Returns:
        perm_groups (list) : a list of the indexes that can be permuted
    """
    if verbose and verbose_level > 20:
        print("-- Check Identical Col Counts (v2) --")

    counts = {}  # Multi Counts
    perm_groups = []  # A list of groups of Permutable Indexes

    for i, row in enumerate(counts_matrix, start=0):
        # a tuple (hashable) version of the row counts
        row = row.tolist()
        row_tuple = tuple(row)

        if verbose and verbose_level > 30:
            print("RC: %s : %s" % (i, row_tuple))

        if row_tuple in counts:
            # we have seen this one before, so add it to the permutation group
            counts[row_tuple].append(i)
        else:
            # we have not seen this count before, so create a new entry for it
            counts[row_tuple] = [i]

    # Append non-constants that have permutable positions
    for col, idxs in counts.items():
        if verbose and verbose_level > 30:
            print(col, ":", idxs)

        if len(idxs) >= 2:
            perm_groups.append(idxs)

    if verbose and verbose_level > 30:
        print("counts:", counts)
        print("sames_groups:", perm_groups)

    if len(perm_groups):
        return perm_groups
    else:
        return []


def _count_cols_symbols(pi_matrix=None, verbose=False, verbose_level=0):
    """Given a matrix, where each row is a prime implicant, counts how many 0's, 1's and 2's are found in each column.

    Args:
        pi_matrix (numpy.ndarray) : a matrix ``n \times k`` of ``n`` prime implicants.

    Returns:
        counts (numpy.ndarray) : a matrix ``n \times 3`` where the entries are counts.
    """
    if verbose and verbose_level > 20:
        print(" -- Count Cols (v2) --")
    # How many PI?
    n = pi_matrix.shape[1]
    # Instanciate count matrix
    counts = np.zeros((n, 3), dtype=int)
    for i, col in enumerate(pi_matrix.T):
        # Count how many values are found and update the matrix of counts
        val, cnt = np.unique(col, return_counts=True)
        # print(val, cnt)
        counts[i, val] = cnt

    return counts


def _ts_covers(two_symbol, permut_indexes, input, verbose=False):
    """Helper method to test if an input is being covered by a two symbol permuted implicant

    Args:
        two_symbol (string): the two_symbol implicant.
        permut_indexes (list): a list-of-lists of the implicant indexes that are permutables.
        input (string): the input string to be checked.

    Returns:
        x (bool): True if covered else False.
    """
    if verbose:
        print("Evaluating permutation coverage")
    # No permutation, just plain implicant coverage?
    if not len(permut_indexes):
        if __pi_covers(two_symbol, input):
            return True
    # There are permutations to generate and check
    else:
        # NEW METHOD: Generates the expanded logic of the Two-Symbol Schema
        for gen_implicant in _expand_ts_logic(two_symbol, permut_indexes):
            if __pi_covers(gen_implicant, input):
                return True
        """
        # OLD METHOD
        for idxs in permut_indexes:
            # Extract the charactes that can be permuted
            chars = [implicant[idx] for idx in idxs]
            # Generate all possible permutations of these symbols
            permut_chars = itertools.permutations(chars, len(idxs))
            for permut_chars in permut_chars:
                # Generate a new implicant and substitute the charactes with the permuted ones
                tmp = list(implicant)
                for idx,char in zip(idxs,permut_chars):
                    tmp[idx] = char
                # The new permuted implicate is covered?
                if __pi_covers(tmp, input):
                    return True
        """
    return False


def computes_ts_coverage(k, outputs, two_symbols):
    """Computes the input coverage by Two Symbol schematas.

    Args:
        k (int): the number of inputs.
        outpus (list): the list of transition outputs.
        two_symbols (list): The final list of Two Symbol permutable schematas. This is returned by `find_two_symbols`.

    Returns:
        coverage (dict): a dictionary of coverage where keys are inputs states and values are lists of the Two Symbols covering that input.
    """
    ts_coverage = {}
    for statenum in range(2**k):
        binstate = statenum_to_binstate(statenum, base=k)
        ts_coverage[binstate] = covering_twosymbols = []
        output = int(outputs[statenum])
        if output == 2:
            output = [0, 1]
        else:
            output = [int(outputs[statenum])]
        for t in output:
            for implicant, permut_indxs, same_symbols_indxs in two_symbols[t]:
                if _ts_covers(implicant, permut_indxs, binstate):
                    covering_twosymbols.append(
                        (implicant, permut_indxs, same_symbols_indxs)
                    )
    #
    return ts_coverage
