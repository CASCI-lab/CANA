# -*- coding: utf-8 -*-
"""
Minimum Dominating Set
=======================


"""
#   Copyright (C) 2021 by
#   Alex Gates <ajgates42@gmail.com>
#   Rion Brattig Correia <rionbr@gmail.com>
#   All rights reserved.
#   MIT license.
import itertools


#
# Minimum Dominating Set
#
def mds(directed_graph, max_search=5, keep_self_loops=True):
    """The minimum dominating set method.

    Args:
        directed_graph (networkx.DiGraph) : The structural graph.
        max_search (int) : Maximum search of additional variables. Defaults to 5.
        keep_self_loops (bool) : If self-loops are used in the computation.
    Returns:
        (list) : A list of sets with the driver nodes.
    """
    N = len(directed_graph)
    root_var = _root_variables(directed_graph, keep_self_loops=keep_self_loops)

    if len(_get_dominated_set(directed_graph, root_var)) == N:
        return [root_var]
    else:
        MDS_sets = []
        nonroot_variables = set(directed_graph.nodes()) - set(root_var)
        for num_additional_var in range(1, max_search):
            for an_combo in itertools.combinations(nonroot_variables, num_additional_var):
                possible_dvs = root_var.union(an_combo)
                if len(_get_dominated_set(directed_graph, possible_dvs)) == N:
                    MDS_sets.append(possible_dvs)
            if len(MDS_sets) > 0:
                break
        return MDS_sets


def _get_dominated_set(directed_graph, dominatingset):
    """
    TODO
    """
    dominatedset = set(dominatingset)
    for dn in dominatingset:
        dominatedset.update(directed_graph.neighbors(dn))
    return dominatedset


def _root_variables(directed_graph, keep_self_loops=True):
    """
    """
    return set([n for n in directed_graph.nodes()
                if (directed_graph.in_degree(n) == 0) or ((not keep_self_loops) and (directed_graph.neighbors(n) == [n]))])
