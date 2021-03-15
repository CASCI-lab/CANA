# -*- coding: utf-8 -*-
"""
Feedback Vertex Set
====================

The NP-complete Feedback Vertex Set (FVS) problem is defined as the minimum number of vertices
that need to be removed from a directed graph so that the resulting graph has no direct cycle. :cite:`Pardalos:1998` :cite:`Mochizuki:2013`.

Two methods are implemented here. A bruteforce and the Greedy Randomized Adaptive Search Procedure (GRASP) method by :cite:`Pardalos:1998`.

"""
#   Copyright (C) 2021 by
#   Alex Gates <ajgates42@gmail.com>
#   Rion Brattig Correia <rionbr@gmail.com>
#   All rights reserved.
#   MIT license.
import networkx as nx
import numpy as np
import itertools
import copy


#
# GRASP method
#
def fvs_grasp(directed_graph, max_iter=100, keep_self_loops=True):
    """The Feedback Vertex Set GRASP implementation.
    This implementation is not exact but it is recommended for very large graphs.

    Args:
        directed_graph (networkx.DiGraph) : The structure graph.
        max_iter (int) : Maximum number of iterations for the search.
        keep_self_loops (bool) : If self-loops are used in the computation. By FVS theory, all self-loop nodes are needed for control.
    Returns:
        (list) : A list of sets with the driver nodes.
    See also:
        :func:`fvs_bruteforce`.
    """
    if keep_self_loops:
        S = set(nx.nodes_with_selfloops(directed_graph))

    else:
        directed_graph = copy.deepcopy(directed_graph)
        directed_graph.remove_edges_from(directed_graph.selfloop_edges())
        S = set([])

    root_var = _root_variables(directed_graph, keep_self_loops=keep_self_loops)
    S = S.union(root_var)

    minfvc = set([frozenset(directed_graph.nodes())])

    reduced_graph = directed_graph.copy()

    for i_iter in range(max_iter):
        alpha = np.random.random()
        S = _construct_greedy_randomized_solution(reduced_graph, alpha, S)
        S = _local_search(directed_graph.copy(), S)
        compare_set = next(iter(minfvc))
        if len(S) == len(compare_set):
            minfvc.add(frozenset(S))
        elif len(S) < len(compare_set):
            minfvc = set([frozenset(S)])

    return list(minfvc)


#
# Bruteforce method
#
def fvs_bruteforce(directed_graph, max_search=5, keep_self_loops=True):
    """The Feedback Vertex Set bruteforce implementation.

    Args:
        directed_graph (networkx.DiGraph) : The structure graph.
        max_search (int) : Maximum number of additional variables to include in the search.
        keep_self_loops (bool) : If self-loops are used in the computation. By FVS theory, all self-loop nodes are needed for control.
    Returns:
        (list) : A list of sets with with the driver nodes.
    Warning:
        Use the GRASP method for large graphs.
    See also:
        :func:`fvs_grasp`.

    """
    if keep_self_loops:
        minfvc = set(nx.nodes_with_selfloops(directed_graph))

    else:
        directed_graph = copy.deepcopy(directed_graph)
        directed_graph.remove_edges_from(directed_graph.selfloop_edges())
        minfvc = set([])

    root_var = _root_variables(directed_graph, keep_self_loops=keep_self_loops)
    minfvc = minfvc.union(root_var)

    if _is_acyclic(directed_graph):
        return [minfvc]

    else:
        FVC_sets = []
        nonfvc_variables = set(directed_graph.nodes()) - minfvc

        num_additional_var = 0
        while num_additional_var <= max_search and len(FVC_sets) == 0:
            for an_combo in itertools.combinations(nonfvc_variables, num_additional_var):
                possible_fvs = minfvc.union(an_combo)

                if _is_acyclic(_graph_minus(directed_graph, possible_fvs)):
                    FVC_sets.append(possible_fvs)

            num_additional_var += 1

        return FVC_sets


def _graph_minus(graph, nodeset):
    """
    """
    newgraph = nx.DiGraph()
    for (n1, n2) in graph.edges():
        if n1 not in nodeset and n2 not in nodeset:
            newgraph.add_edge(n1, n2)

    for n in graph.nodes():
        if n not in nodeset:
            newgraph.add_node(n)

    return newgraph


def _is_acyclic(graph):
    """
    """
    return nx.is_directed_acyclic_graph(graph)


def _in0out0(directed_graph, S):
    """
    """
    remove_set = set([n for n in directed_graph.nodes()
                      if (directed_graph.in_degree(n) == 0 or directed_graph.out_degree(n) == 0)])
    directed_graph.remove_nodes_from(remove_set)
    S = S.union(remove_set)
    return directed_graph, S, len(remove_set) == 0


def _in1(directed_graph):
    """
    """
    remove_set = set([n for n in directed_graph.nodes() if (directed_graph.in_degree(n) == 1)])

    for u in remove_set:
        v = list(directed_graph.predecessors(u))[0]
        directed_graph.add_edges_from([(v, w) for w in directed_graph.successors(u)])
    directed_graph.remove_nodes_from(remove_set)
    return directed_graph, len(remove_set) == 0


def _out1(directed_graph):
    """
    """
    remove_set = set([n for n in directed_graph.nodes() if (directed_graph.out_degree(n) == 1)])

    for u in remove_set:
        v = list(directed_graph.successors(u))[0]
        directed_graph.add_edges_from([(w, v) for w in directed_graph.predecessors(u)])
    directed_graph.remove_nodes_from(remove_set)

    return directed_graph, len(remove_set) == 0


def _selfloop(directed_graph, S):
    """
    """
    remove_set = list(nx.nodes_with_selfloops(directed_graph))

    S = S.union(set(remove_set))

    directed_graph.remove_nodes_from(remove_set)
    return directed_graph, S, len(remove_set) == 0


def _reduce_instance_size(directed_graph, S):
    """
    """
    all_done = False

    while not all_done:
        all_done = True

        reduced_graph, S, no_change = _selfloop(directed_graph, S)
        all_done *= no_change

        reduced_graph, S, no_change = _in0out0(reduced_graph, S)
        all_done *= no_change

        reduced_graph, no_change = _in1(reduced_graph)
        all_done *= no_change

        reduced_graph, no_change = _out1(reduced_graph)
        all_done *= no_change

    return reduced_graph, S


def _restricted_candidate_list(directed_graph, alpha):
    """
    """
    nodelist = directed_graph.nodes()

    G = np.multiply(directed_graph.in_degree(nodelist), directed_graph.out_degree(nodelist))
    Gmin = min(G)
    Gmax = max(G)

    return np.array(nodelist)[G >= (Gmin + alpha * (Gmax - Gmin))]


def _construct_greedy_randomized_solution(directed_graph, alpha, S):
    """"
    """
    reduced_graph, S = _reduce_instance_size(directed_graph, S)

    while len(reduced_graph) > 0:
        RCL = set(_restricted_candidate_list(reduced_graph, alpha))
        s = np.random.choice(RCL)
        S.add(s)
        reduced_graph.remove_node(s)
        reduced_graph, S = _reduce_instance_size(reduced_graph, S)

    return S


def _local_search(directed_graph, S):
    """
    """
    keep_going = True

    while keep_going:
        keep_going = False
        for localnode in S:
            remove_set = S.copy()
            remove_set.discard(localnode)
            reduced_graph = directed_graph.copy()
            reduced_graph.remove_nodes_from(remove_set)
            if nx.is_directed_acyclic_graph(reduced_graph):
                S = remove_set
                keep_going = True
                break

    return S


def _root_variables(directed_graph, keep_self_loops=True):
    """
    """
    return set([n for n in directed_graph.nodes()
                if (directed_graph.in_degree(n) == 0) or ((not keep_self_loops) and (directed_graph.neighbors(n) == [n]))])
