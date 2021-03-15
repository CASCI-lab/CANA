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

import numpy as np

import itertools
from .. utils import *

from collections import deque

__author__ = """\n""".join([
    'Alex Gates <ajgates@umail.iu.edu>',
    'Etienne Nzabarushimana <enzabaru@indiana.edu>',
    'Rion Brattig Correia <rionbr@gmail.com>'
])


# Quine-McCluskey Functions
def make_transition_density_tables(k=1, outputs=[0, 1]):
    """ This method creates a tuple-of-lists that is used to calculate Prime Implicants in the first step of the Quine-McCluskey algorithm :cite:`Quine:1955`.
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
    transition_density_tuple = [[[] for density in range(k + 1)] for transition in [0, 1]]
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
    """ Finds the prime implicants (PI) using the Quine-McCluskey algorithm :cite:`Quine:1955`.

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
        matches = [[False for _ in range(len(column[density]))] for density in range(N + 1)]

        # loop through the possible densities
        for density in range(N):

            # compare the implicants from successive densities
            for i, implicant in enumerate(column[density]):
                for j, candidate in enumerate(column[density + 1]):

                    # check if the implicants differ on only one variable
                    match = _adjacent(implicant, candidate)
                    if match:
                        matches[density][i] = matches[density + 1][j] = True
                        matches_density = sum([var != '0' for var in match])
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
            match.append('2')
    return "".join(match)


def __pi_covers(implicant, input, symbol=['2', '#', 2]):
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

    Args:
        k (int): The number of inputs.
        prime_implicants (list): The prime implicants computed.

    Returns:
        final_list (list) : The list of two-symbol schematas.

    Note: This is a modification of the original algorithm that can be found in Marques-Pita & Rocha [2013].
    """
    if not len(prime_implicants):
        return []

    # If this node has no input, yet it affects other nodes (fixed variable)
    if k == 0:
        TSf = []
        for pi in prime_implicants:
            TSf.append((pi, [], []))
        return TSf

    # Init
    # n_pi = len(prime_implicants) # never used
    pi_matrix = np.array(tuple(map(tuple, prime_implicants)), dtype=int)

    # List of the Two-Symbol Schematas
    TS = []

    # Init Queue
    Q = deque()
    Q_history = set()
    Q.append(pi_matrix)
    i = 0

    while len(Q):

        schematas = Q.popleft()
        n_schematas = schematas.shape[0]
        i += 1

        if verbose:
            if verbose_level == 1:
                if (i % 500 == 0):
                    print('>>> QUEUE: pop | A (m=%d) | Queue size: %d' % (n_schematas, len(Q)))
            elif verbose_level > 5:
                print('>>> QUEUE: pop | A (m=%d) | Queue size: %d' % (n_schematas, len(Q)))
            if verbose_level > 10:
                print(schematas)

        # count the number of [0's, 1's, 2's] in each column
        column_counts = _count_cols_symbols_v2(pi_matrix=schematas, verbose=verbose, verbose_level=verbose_level)
        if verbose and verbose_level > 10:
            print('>>> COLUMN Schema Counts:')
        # find the permutation groups based on column counts
        perm_groups = _check_col_counts_v3(counts_matrix=column_counts, verbose=verbose, verbose_level=verbose_level)

        if (perm_groups != -1):
            if verbose and verbose_level > 10:
                print('>>> There are permutable groups! Lets loop them')
            for x_group in perm_groups:
                if verbose and verbose_level > 20:
                    print('>>> x_group:', x_group)
                    print('>>> Truncated schemata matrix:')
                    print(schematas[:, x_group].T)
                # find the row counts by taking the transpose of the truncated schemata list
                row_counts = _count_cols_symbols_v2(pi_matrix=schematas[:, x_group].T, verbose=verbose, verbose_level=verbose_level)
                if verbose and verbose_level > 20:
                    print('>>> ROW Schema Counts:')
                    print(row_counts)
                # make sure all row counts are the same
                if not (row_counts == row_counts[0]).all():
                    if verbose and verbose_level > 20:
                        print('>>> row_counts are NOT the same (-1)')
                    perm_groups = -1

        if verbose and verbose_level > 10:
            print(">>> Exists permutation groups?:", (perm_groups != -1))
            print(">>> Are groups already in F''?:", ((schematas.tolist(), perm_groups) in TS))
        if (perm_groups != -1) and not ((schematas.tolist(), perm_groups) in TS):
            # do some weird permutation group testing
            allowed_perm_groups = _check_schemata_permutations_v2(schematas, perm_groups, verbose=verbose, verbose_level=verbose_level)
            if verbose and verbose_level > 15:
                print('>>> Permutation testing result:', allowed_perm_groups)
            if allowed_perm_groups is not None:
                if verbose and verbose_level > 15:
                    print(">>> RESULTS: adding F'': %s , Idxs: %s" % (schematas.tolist(), allowed_perm_groups))
                TS.append((schematas.tolist(), allowed_perm_groups))

        else:
            if verbose and verbose_level > 10:
                print('>>> Generate combinations of schematas (m-1) and add to Queue')
            if schematas.shape[0] > 2:
                for idxs_subset in itertools.combinations(np.arange(0, n_schematas), (n_schematas - 1)):
                    idxs_subset = list(idxs_subset)
                    schemata_subset = schematas[idxs_subset, :]
                    # This schemata has already been inserted onto the Queue before?
                    if schemata_subset.tostring() not in Q_history:
                        if verbose and verbose_level > 25:
                            print('>>> QUEUE: appending (idxs: %s)' % (idxs_subset))
                            print(schemata_subset)
                        Q.append(schemata_subset)
                        Q_history.add(schemata_subset.tostring())
                    else:
                        if verbose and verbose_level > 25:
                            print('>>> QUEUE: duplicate, skip (idxs: %s)' % (idxs_subset))

    if verbose:
        print('>>> TWO-SYMBOLS:')
        for i, (tss, perms) in enumerate(TS):
            print("F''-%d: %s | Perms: %s" % (i, tss, perms))

    # Simplification. Check if there are TSs that are completely contained within others.
    # 'ts' = Two-Symbol
    # 'cx' = Complexity
    # 'xs' = Expanded Logic
    TSs = {
        i: {
            'tss': tss,
            'perms': perms,
            'cx': _calc_ts_complexity(tss, perms),
            'xl': _expand_ts_logic(tss, perms)
        } for i, (tss, perms) in enumerate(TS)
    }

    # Loops all combinations (2) of TS
    for (i, j) in itertools.combinations(TSs.keys(), 2):
        try:
            a_in_b, b_in_a = _check_schema_within_schema(TSs[i]['xl'], TSs[j]['xl'])
        except e:
            continue
        else:
            cx_a = TSs[i]['cx']
            cx_b = TSs[j]['cx']
            # A or B contained in the other, keep only contained.
            if a_in_b and not b_in_a:
                del TSs[i]
            elif b_in_a and not a_in_b:
                del TSs[j]
            elif a_in_b and b_in_a:
                # Keep most complex
                if cx_a < cx_b:
                    del TSs[i]
                elif cx_b < cx_a:
                    del TSs[j]
                else:
                    # they are equal, delete either one
                    del TSs[i]

    if verbose:
        print('>>> TWO-SYMBOLS (simplified):')
        for i, (tss) in TSs.items():
            print("F''-%d: %s | Perms: %s | CX: %s" % (i, tss['tss'], tss['perms'], tss['cx']))

    # Final List (from simplified)
    TSf = [(tss['tss'][0], tss['perms'], []) for tss in TSs.values()]

    # Check if all PI are being covered. If not, include the PI on the TS list
    if verbose:
        print('>>> Check all PI are accounted for in the TS')
    for i, pi in enumerate(pi_matrix, start=0):
        if not any([_check_schema_within_schema([pi.tolist()], tss['xl'], dir='a', verbose=verbose)[0] for tss in TSs.values()]):
            if verbose:
                print("PI-%d '%s' Not in list, ADDING." % (i, pi.tolist()))
            TSf.append((pi.tolist(), [], []))
        else:
            if verbose:
                print("PI-%d '%s' OK." % (i, pi.tolist()))

    if verbose:
        print('>>> Check for Same-Symbol permutables')

    # NEW: Step to include same-symbol permutables
    for ts, perms, sames in TSf:
        # Indices of permutables inputs
        idxs = list(set([idx for idxs in perms for idx in idxs]))

        # Makes the F'' into a Collum Array so it can be used by '_count_cols_symbols_vX'
        ts_matrix = np.array([ts]).T

        # Remove Inputs (columns) that already have permutable symbols. Only if there are permutables
        if len(idxs):
            rmask = np.array(idxs)
            ts_matrix_left = ts_matrix[~rmask, :]
        else:
            ts_matrix_left = ts_matrix

        if verbose and verbose_level > 10:
            print("> F'' Original:")
            print(ts_matrix)
            print("> Permutables: %s" % (perms))
            print("> F'' without permutables:")
            print(ts_matrix_left)

        counts_matrix = _count_cols_symbols_v2(pi_matrix=ts_matrix_left.T, verbose=False, verbose_level=verbose_level)
        perm_groups = _check_identical_cols_count_symbols_v2(counts_matrix=counts_matrix, verbose=verbose, verbose_level=verbose_level)
        sames.extend(perm_groups)

    # Step to convert the pi list to string
    for i, (ts, perms, sames) in enumerate(TSf, start=0):
        ts = ''.join(map(str, ts))
        TSf[i] = (ts, perms, sames)

    # Final list after all PI were accounted for
    if verbose:
        print('>>> TS (final list):')
        for i, tss, sms in TSf:
            print("TS: '%s' | Perm Idx: %s | Sms Idx: %s" % (i, tss, sms))

    return TSf


def _calc_ts_complexity(tss, pers):
    """ Calculates the complexity of a TS schema
    Complexity = (Number of Schemas + Number of Permutable Symbols + Lenght of each Permutable Symbol)
    """
    return len(tss) + sum([len(per) for ts, per in zip(tss, pers)])


def _check_schema_within_schema(la, lb, dir=None, verbose=False):
    """ Check is a Two-Symbol schemata is covered by another.
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
    if dir != 'b':
        a_in_b = all([(xa in lb) for xa in la])
        if verbose:
            print('%s in %s : %s' % (la, lb, a_in_b))
    if dir != 'a':
        b_in_a = all([(xb in la) for xb in lb])
        if verbose:
            print('%s in %s : %s' % (lb, la, b_in_a))
    #
    return a_in_b, b_in_a


def _expand_ts_logic(two_symbols, permut_indexes):
    """ Expands the Two-Symbol logic to all possible prime-implicants variations being covered.

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
                _implicant = copy.copy(implicant)
                _implicant[idxs] = vals
                # Insert to list of logics if not already there
                if not(_implicant.tolist() in logics):
                    logics.append(_implicant.tolist())
                    Q.append(_implicant.tolist())
    return logics


def _check_schemata_permutations_v2(schematas, perm_groups, verbose=False, verbose_level=0):
    """ Checks if the permutations are possible

    Note:
        Not sure if this is really needed.
    """
    if verbose and verbose_level > 20:
        print("-- Check Schemata Permutations (v2) : g(H',L) --")
    allowed_perm_groups = []
    all_indices = set([i_index for x_group in perm_groups for i_index in x_group])
    for x_group in perm_groups:
        sofar = []
        for i_index in range(len(x_group) - 1):
            x_index = x_group[i_index]
            small_group = [x_index]
            if not (x_index in sofar):
                sofar.append(x_index)

                for y_index in x_group[(i_index + 1)::]:

                    if (not(y_index in sofar)) and _can_swap_v2(schematas[:, [x_index, y_index]], verbose=verbose, verbose_level=verbose_level):
                        small_group.append(y_index)
                        sofar.append(y_index)
                if len(small_group) > 1:
                    allowed_perm_groups.append(small_group)

        if verbose and verbose_level > 30:
            print('> allowed_perm_groups', allowed_perm_groups)
        if set([i_index for x_group in allowed_perm_groups for i_index in x_group]) == all_indices:
            return allowed_perm_groups
    return None


def _can_swap_v2(schemata_subset, verbose=False, verbose_level=0):
    """Determines if two schemata subsets can be swapped"""
    if verbose and verbose_level > 40:
        print('> Can Swap?:',)
    can_switch = 1
    for row in schemata_subset[:, [1, 0]]:
        can_switch *= np.any(np.all(schemata_subset == row, axis=1))
    if verbose and verbose_level > 40:
        print(can_switch)
    return can_switch


def _check_col_counts_v3(counts_matrix, verbose=False, verbose_level=0):
    """ This function is used to find permutable symbols.

    Args:
        counts_matrix (numpy.ndarray) : a matrix where rows are inputs and columns are possible input types (0,1 or #)

    Returns:
        perm_groups (list) : a list of the indexes that can be permuted.
    """
    if verbose and verbose_level > 30:
        print('-- Check Col Counts (v3) --')

    counts = {}         # Multi Counts
    perm_groups = []    # A list of groups of Permutable Indexes

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
            print(col, ':', idxs)

        if len(idxs) == 1:
            return -1
        elif len(idxs) >= 1:
            perm_groups.append(idxs)

    if verbose and verbose_level > 40:
        print('counts:', counts)
        print('perm_groups:', perm_groups)

    if len(perm_groups):
        return perm_groups
    else:
        return -1


def _check_identical_cols_count_symbols_v2(counts_matrix, verbose=False, verbose_level=0):
    """ This function is used to find same symbol permutables. In practice it is a variance of `_check_cols_symbols_vX`

    Args:
        counts_matrix (numpy.ndarray) : a matrix where rows are inputs and columns are possible input types (0,1 or #)

    Returns:
        perm_groups (list) : a list of the indexes that can be permuted
    """
    if verbose and verbose_level > 20:
        print('-- Check Identical Col Counts (v2) --')

    counts = {}         # Multi Counts
    perm_groups = []    # A list of groups of Permutable Indexes

    for i, row in enumerate(counts_matrix, start=0):
        # a tuple (hashable) version of the row counts
        row = row.tolist()
        row_tuple = tuple(row)

        if verbose and verbose_level > 30:
            print('RC: %s : %s' % (i, row_tuple))

        if row_tuple in counts:
            # we have seen this one before, so add it to the permutation group
            counts[row_tuple].append(i)
        else:
            # we have not seen this count before, so create a new entry for it
            counts[row_tuple] = [i]

    # Append non-constants that have permutable positions
    for col, idxs in counts.items():
        if verbose and verbose_level > 30:
            print(col, ':', idxs)

        if len(idxs) >= 2:
            perm_groups.append(idxs)

    if verbose and verbose_level > 30:
        print('counts:', counts)
        print('sames_groups:', perm_groups)

    if len(perm_groups):
        return perm_groups
    else:
        return []


def _count_cols_symbols_v2(pi_matrix=None, verbose=False, verbose_level=0):
    """ Given a matrix, where each row is a prime implicant, counts how many 0's, 1's and 2's are found in each column.

    Args:
        pi_matrix (numpy.ndarray) : a matrix ``n \times k`` of ``n`` prime implicants.

    Returns:
        counts (numpy.ndarray) : a matrix ``n \times 3`` where the entries are counts.
    """
    if verbose and verbose_level > 20:
        print(' -- Count Cols (v2) --')
    # How many PI?
    n = pi_matrix.shape[1]
    # Instanciate count matrix
    counts = np.zeros((n, 3), dtype=int)
    for i, col in enumerate(pi_matrix.T):
        # Count how many values are found and update the matrix of counts
        val, cnt = np.unique(col, return_counts=True)
        counts[i, val] = cnt

    return counts


############ START OF TWO SYMBOL v.1 ############


# This version does not conside '11' and '00' as permutable symbols and had other bugs solved by v2
def find_two_symbols_v1(k=1, prime_implicants=None, verbose=False):
    two_symbol_schemata_list = []

    Partition_Options = [prime_implicants]
    while len(Partition_Options) > 0:
        # take first partition out of the set
        schemata_list = np.array(map(list, Partition_Options.pop()))
        if verbose:
            print('== Partitions (v1) ==')
            print('>>> A (m=%d)' % schemata_list.shape[0])
            print(schemata_list)

        # count the number of [0's, 1's, 2's] in each column
        column_counts = _three_symbol_count_cols_symbols_v1(k=k, transition_list=schemata_list)

        # find the permutation groups based on column counts
        permutation_groups = _check_counts_v1(column_counts)

        if (permutation_groups != -1):
            if verbose:
                print('>>> There are permutable groups! Lets loop them')
            for x_group in permutation_groups:

                # find the row counts by taking the transpose of the truncated schemata list
                row_counts = _three_symbol_count_cols_symbols_v1(k=schemata_list.shape[0], transition_list=schemata_list[:, x_group].T)
                if verbose:
                    print('>>> ROW Schema Counts:')
                    print(row_counts)
                # make sure all row counts are the same
                if len(row_counts) != row_counts.count(row_counts[0]):
                    permutation_groups = -1
        if verbose:
            print(">>> Permutation groups:", (permutation_groups != -1), permutation_groups)
            print(">>> Permuted groups already in F'':", ((schemata_list.tolist(), permutation_groups) in two_symbol_schemata_list))
        if (permutation_groups != -1) and not ((schemata_list.tolist(), permutation_groups) in two_symbol_schemata_list):
            # do some weird permutation group testing
            allowed_permutation_groups = _check_schemata_permutations_v1(schemata_list, permutation_groups)
            if allowed_permutation_groups != []:
                if verbose:
                    print("ADDING to F'':", schemata_subset)
                two_symbol_schemata_list.append((schemata_list.tolist(), allowed_permutation_groups))

        else:
            if schemata_list.shape[0] > 2:
                for schemata_subset in itertools.combinations(schemata_list, (schemata_list.shape[0] - 1)):
                    if verbose:
                        print('ADDING to Queue:', schemata_subset)
                    Partition_Options.append(np.array(schemata_subset))

    if verbose:
        print('Partition_Options:', Partition_Options)
    final_list = []
    prime_accounted = []

    for p_implicant in prime_implicants:
        p_implicant = list(p_implicant)
        if not (p_implicant in prime_accounted):
            for f_double_prime, r_perm in two_symbol_schemata_list:
                for f_prime in f_double_prime:
                    if np.all(f_prime == p_implicant):
                        final_list.append((p_implicant, r_perm))
                        for account_prime in f_double_prime:
                            prime_accounted.append(list(account_prime))
        if not (p_implicant in prime_accounted):
            final_list.append((p_implicant, []))

    return final_list

def _check_schemata_permutations_v1(schemata_list, permutation_groups):

    allowed_permutation_groups = []
    all_indices = set([i_index for x_group in permutation_groups for i_index in x_group])

    for x_group in permutation_groups:
        sofar = []

        for i_index in range(len(x_group) - 1):
            x_index = x_group[i_index]
            small_group = [x_index]

            if not (x_index in sofar):
                sofar.append(x_index)

                for y_index in x_group[(i_index + 1)::]:
                    if (not(y_index in sofar)) and _can_swap_v1(schemata_list[:, [x_index, y_index]]):
                        small_group.append(y_index)
                        sofar.append(y_index)
                if len(small_group) > 1:
                    allowed_permutation_groups.append(small_group)

        if set([i_index for x_group in allowed_permutation_groups for i_index in x_group]) == all_indices:
            return allowed_permutation_groups
    return []


def _can_swap_v1(schemata_subset):
    can_switch = 1
    for row in schemata_subset[:, [1, 0]]:
        can_switch *= np.any(np.all(schemata_subset == row, axis=1))
    return can_switch


def _check_counts_v1(column_counts=[]):
    print('-- Column Counts (v1) --')
    print(column_counts)
    unique_col_counts = []
    permutation_groups = []

    for i_col_count, x_col_count in enumerate(column_counts):
        print('RC: %d : %s' % (i_col_count, x_col_count))
        if x_col_count.count(0) >= 2:
            # this is a constant column so skip it
            pass

        elif x_col_count in unique_col_counts:
            # we have seen this one before, so add it to the permutation group
            permutation_groups[unique_col_counts.index(x_col_count)].append(i_col_count)

        else:
            # we have not seen this count before, so create a new entry for it
            unique_col_counts.append(x_col_count)
            permutation_groups.append([i_col_count])

    # check if a singleton permutation group exists
    if [len(x_group) for x_group in permutation_groups].count(1) > 0:
        print('counts:', permutation_groups)
        return -1
    else:
        print('counts:', permutation_groups)
        return permutation_groups


def _three_symbol_count_cols_symbols_v1(k=1, transition_list=None):
    column_counts = [[0, 0, 0] for i_col in range(k)]
    for x_col in transition_list:
        for i_entry, x_entry in enumerate(x_col):
            if x_entry == '0':
                column_counts[i_entry][0] += 1
            elif x_entry == '1':
                column_counts[i_entry][1] += 1
            elif x_entry == '2':
                column_counts[i_entry][2] += 1

    return column_counts


############ END OF TWO SYMBOL v.1 ############


def __ts_covers(two_symbol, permut_indexes, input, verbose=False):
    """Helper method to test if an input is being covered by a two symbol permuted implicant

    Args:
        two_symbol (string): the two_symbol implicant.
        permut_indexes (list): a list-of-lists of the implicant indexes that are permutables.
        input (string): the input string to be checked.

    Returns:
        x (bool): True if covered else False.
    """
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
    """ Computes the input coverage by Two Symbol schematas.

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
                if __ts_covers(implicant, permut_indxs, binstate):
                    covering_twosymbols.append((implicant, permut_indxs, same_symbols_indxs))
    #
    return ts_coverage
