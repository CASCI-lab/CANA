# -*- coding: utf-8 -*-
"""
(Cythonized) Boolean Canalization
=====================

Functions to compute the Quine-McCluskey algorithm in cython for increaed computation speed.

"""
#   Copyright (C) 2021 by
#   Rion Brattig Correia <rionbr@gmail.com>
#   Alex Gates <ajgates42@gmail.com>
#
#   All rights reserved.
#   MIT license.
from cana.cutils import *

WILDCARD_SYMBOL = '#'
SYMMETRIC_WILDCARD_SYMBOL = '*'


#
# Quine-McCluskey Functions
#
def make_density_groups(input_binstates):
    """
    """

    density_groups = dict()
    for binstate in input_binstates:
        density = binary_density(binstate)
        if density not in density_groups:
            density_groups[density] = set()
        density_groups[density].add(binstate)

    return density_groups


def find_wildcards(binstate1, binstate2):
    """
    Compare two binary states and replace any differing bits by a wildcard.
    Args:
        binstate1, binstate2 : the two binary states to be compared

    Return:
        c (list, bool) : a list of comparisons

    """
    # assert len(s1) == len(s2) , "The two binstates must have the same length"
    return "".join([b0 if (b0 == b1) else WILDCARD_SYMBOL for b0, b1 in zip(binstate1, binstate2)])


def binary_density(binstate):
    """
    Find the density (number of 1s) for a term with possible wildcards.
    """
    return binstate.count('1')


def replace_wildcard(binstate, idx):
    """
    Return the binary state with a wildcard at the idx position.
    """
    return binstate[:idx] + WILDCARD_SYMBOL + binstate[idx + 1:]


def find_implicants_qm(input_binstates, verbose=False):
    """ Finds the prime implicants (PI) using the Quine-McCluskey algorithm :cite:`Quine:1955`.

    Args:
        input_binstates (list / set) : A the binstates to condense.

    Returns:
        PI (set): a set of prime implicants.

    # Authors: Alex Gates
    """

    # we start with an empty set of implicants
    matched_implicants = set()
    done = False

    # repeat the following until no matches are found
    while not done:

        # split up the input_binstates into groups based on the number of 1s (density)
        density_groups = make_density_groups(input_binstates)

        # now clear everything for the new pass
        input_binstates = set()
        used = set()

        # Find the prime implicants

        # for each possible density
        for density in density_groups:
            # first make sure there are other binstates with the next density
            if density + 1 in density_groups:

                # then we pass through the binstates
                for binstate0 in density_groups[density]:

                    # An optimization due to Thomas Pircher, https://github.com/tpircher/quine-mccluskey/blob/master/quine_mccluskey/qm.py
                    # The Quine-McCluskey algorithm compares t1 with
                    # each element of the next group. (Normal approach)
                    # But in reality it is faster to construct all
                    # possible permutations of t1 by adding a '1' in
                    # opportune positions and check if this new term is
                    # contained in the set groups[key_next].

                    for idx, b0 in enumerate(binstate0):
                        if b0 == '0':
                            binstate1 = flip_binstate_bit(binstate0, idx)
                            if binstate1 in density_groups[density + 1]:
                                # keep track of the covered binary states
                                used.add(binstate0)
                                used.add(binstate1)
                                # keep the new wildcard binstate for the next round
                                input_binstates.add(replace_wildcard(binstate0, idx))

        # now add back the implicants that were not matched
        for groups in list(density_groups.values()):
            matched_implicants |= groups - used

        # finally, check if this pass actually compressed any terms
        # we finish when we cant make any further compressions
        if len(used) == 0:
            done = True

    # finish up by adding back all of the uncovered binary states
    prime_implicants = matched_implicants
    for groups in list(density_groups.values()):
        prime_implicants |= groups

    return prime_implicants


def __pi_covers(implicant, binstate):
    """Determines if a binarystate is covered by a specific implicant.
    Args:
        implicant (string): the implicant.
        minterm (string): the minterm.
    Returns:
        x (bool): True if covered else False.

    """
    return all(i == WILDCARD_SYMBOL or m == i for i, m in zip(implicant, input))


def expand_wildcard_schemata(schemata):
    """
    Expand a wildcard schemata to list all binary states it covers.

    Args:
        schemata (string): the wildcard shemata
    Returns:
        binary_states (list): list of all binary states covered by the schemata
    """

    # count the number of wildcard symbols
    nwildcards = schemata.count(WILDCARD_SYMBOL)

    # if there arent any symbols, return the original schemata
    if nwildcards == 0:
        return [schemata]
    else:
        binary_states = []
        for wildstatenum in range(2**nwildcards):
            wildstates = statenum_to_binstate(wildstatenum, nwildcards)
            wnum = 0
            newstate = ''
            for b in schemata:
                if b == WILDCARD_SYMBOL:
                    newstate += wildstates[wnum]
                    wnum += 1
                else:
                    newstate += b
            binary_states.append(newstate)
        return binary_states


def return_pi_coverage(prime_implicants):
    """Computes the binary states coverage by Prime Implicant schematas.

    Args:
        prime_implicants (set): a set of prime implicants.
            This is returned by `find_implicants_qm`.
    Returns:
        pi_coverage (dict) : a dictionary of coverage where keys are input states and values are lists of the Prime Implicants covering that input.
    """

    pi_coverage = dict()
    for pi in prime_implicants:
        for binstate in expand_wildcard_schemata(pi):
            if binstate not in pi_coverage:
                pi_coverage[binstate] = set()
            pi_coverage[binstate].add(pi)

    return pi_coverage


def input_wildcard_coverage(pi_coverage):
    """Computes the binary states coverage by Prime Implicant schematas.

    Args:
        pi_coverage (dict): a dict mapping binary states to their prime implicants.
            This is returned by `return_pi_coverage`.
    Returns:
        input_wildcard_coverage (dict) : a dictionary of coverage where keys are inputs and values are lists of wither a WildCard covers that input.

    """
    # number of inputs
    k = len(next(iter(pi_coverage)))

    input_to_wildcards = {i: dict() for i in range(k)}
    for binstate, piset in pi_coverage.items():
        for i in range(k):
            input_to_wildcards[i][binstate] = tuple(pi[i] == WILDCARD_SYMBOL for pi in piset)

    return input_to_wildcards
