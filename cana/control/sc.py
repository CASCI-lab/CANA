# -*- coding: utf-8 -*-
"""
Structural Controllability
===========================


"""
#   Copyright (C) 2021 by
#   Alex Gates <ajgates42@gmail.com>
#   Rion Brattig Correia <rionbr@gmail.com>
#   All rights reserved.
#   MIT license.
import networkx as nx


def sc(directed_graph, keep_self_loops=True):
    """The Structural Contolability driver sets.

    Parameters:
        directed_graph (networkx.DiGraph) : The directed graph capturing the network structure.
        keep_self_loops (bool) : Determines if the self loops are kept in the determination of structural control.

    Returns:
        (list) : A list of sets with the driver nodes.

    See also:
        If you only need the size of the set, see :func:`sc_min_size`.
    """
    # get the bipartite represenation of the directed graph
    bipartite_graph, vertex_out, vertex_in = _directed_to_bipartite(directed_graph, keep_self_loops=keep_self_loops)

    # do the maximum matching on the bipartite representation
    max_matching = nx.bipartite.hopcroft_karp_matching(bipartite_graph, top_nodes=vertex_out)

    all_max_matchings = _enumerate_maximum_matchings(bipartite_graph, vertex_out, vertex_in, max_matching)

    sc_sets = []
    for M in all_max_matchings:
        # find the vertex set whose in-coming vertices were matched
        matched_incoming = set(vertex_in) & set(M.keys())
        matched_original_vertices = set([sink[0] for sink in matched_incoming])
        new_dv_set = sorted(set(directed_graph.nodes()) - matched_original_vertices)
        if new_dv_set not in sc_sets:
            sc_sets.append(sorted(new_dv_set))

    return sc_sets


def sc_min_size(directed_graph, keep_self_loops=True):
    """The minimum number of driver variables required by structural controllability.

    Parameters:
        directed_graph (networkx.DiGraph) : The directed graph capturing the network structure.
        keep_self_loops (bool) : Determines if the self loops are kept in the determination of structural control.

    TODO:
        Implement the removal of self loops

    Returns:
        (int) : The number of driver variables necessary to render the graph structurally controlled.

    See also:
        :func:`sc`
    """
    # get the bipartite represenation of the directed graph
    bipartite_graph, vertex_out, vertex_in = _directed_to_bipartite(directed_graph, keep_self_loops=keep_self_loops)

    # do the maximum matching on the bipartite representation
    max_matching = nx.bipartite.hopcroft_karp_matching(bipartite_graph)

    # find the vertex set whose in-coming vertices were matched
    matched_incoming = set(vertex_in) & set(max_matching.keys())
    matched_original_vertices = set([sink[0] for sink in matched_incoming])

    # driver variables are those whose in-coming edges were not matched
    # although this set is not unique, only the size of the set is gaurenteed to be
    # a minimum
    sc_min_size = len(set(directed_graph.nodes()) - matched_original_vertices)

    return sc_min_size


def _directed_to_bipartite(directed_graph, keep_self_loops=True):
    """Create the undirected Bipartite representation of a directed graph.

    In this represtation, each vertex is a tuple (n, direction)
    where  one vertex set (labeled with "+") denotes the out-going
    directed edges while the other vertex set (labeled with "-") denotes the
    in-coming directed edges.  Two vertices are connected with an undirected edge
    if a directed edge exsisted in the original graph.

    Args:
        directed_graph (networkx.DiGraph) : The directed graph to be transformed.

    Returns:
        bipartite_graph (networkx.Graph) : The bipartite representation of the directed graph.
        vertex_out (list) : The list of out-going vertices.
        vertex_in (list) : The list of in-coming vertices.
    """
    # get the two vertex sets
    vertex_out = [(n, True) for n in directed_graph.nodes()]
    vertex_in = [(n, False) for n in directed_graph.nodes()]

    # make the bipartite graph
    bipartite_graph = nx.Graph()
    bipartite_graph.add_nodes_from(vertex_out, bipartite=0)
    bipartite_graph.add_nodes_from(vertex_in, bipartite=1)

    # add the edges
    for n in directed_graph.nodes():
        for v in directed_graph.successors(n):
            if keep_self_loops:
                bipartite_graph.add_edge((n, True), (v, False))
            elif (n != v):
                bipartite_graph.add_edge((n, True), (v, False))

    return bipartite_graph, vertex_out, vertex_in


def _bipartite_to_directed(bipartite_graph):
    """Recover the directed graph from its bipartite representation.

    Args:
        bipartite_graph (networkx.Graph) : The bipartite represtation of a directed graph consisting of two vertex sets:
            vertex_out and vertex_in, to be transformed back to a directed graph.

    Returns:
        directed_graph (networkx.DiGraph) : The directed graph.

    See also:
        :attr:`_directed_to_bipartite`.
    """
    directed_graph = nx.DiGraph()
    for edge, d in bipartite_graph.edges():
        for node in edge:
            if node[1] is True:
                source = node[0]
            elif node[1] is False:
                sink = node[0]
        directed_graph.add_edge(source, sink)
    return directed_graph


def _enumerate_maximum_matchings(bipartite_graph, vertex_out, vertex_in, max_matching):
    """
    @@@ Uno (1999) ???

    TODO:
        Could not find the original paper to cite here.
    """
    # this is the first matching in our list
    matchings_list = [max_matching]

    D = _make_matching_digraph(bipartite_graph, vertex_out, vertex_in, max_matching)
    D = _trim_unnecessary_edges(D)
    matchings_list = _enumerate_maximum_matchings_iter(bipartite_graph, vertex_out, vertex_in, max_matching, D, matchings_list)

    return matchings_list


def _make_matching_digraph(bipartite_graph, U, V, M):
    """
    @@@ Uno (1999) ???

    TODO:
        Could not find the original paper to cite here.

    """
    matching_digraph = nx.DiGraph()
    matching_digraph.add_nodes_from(bipartite_graph.nodes())

    matched_vertices = M.keys()

    # loop over the first vertex set
    for v in V:
        # and loop over all neighbors of v
        for u in bipartite_graph.neighbors(v):
            # check if the edge is in the maximum matching
            if v in matched_vertices and M[v] == u:
                # if the edge is matched, place a directed edge from U to V
                matching_digraph.add_edge(u, v)
            else:
                # otherwise, place a directed edge from V to U
                matching_digraph.add_edge(v, u)

    return matching_digraph


def _trim_unnecessary_edges(matching_digraph):
    """
    """
    scc_subgraphs = [matching_digraph.subgraph(scc) for scc in nx.strongly_connected_components(matching_digraph)]
    trimmed_graph = scc_subgraphs[0]
    for next_subgraph in scc_subgraphs[1:]:
        trimmed_graph = nx.union(trimmed_graph, next_subgraph)
    return trimmed_graph


def _enumerate_maximum_matchings_iter(G, U, V, M, D, matchings_list):
    """

    """
    if len(G) > 0 and not (D is None):

        # find the cycles in the matching digraph
        cycles = [c for c in nx.simple_cycles(D)]

        if len(cycles) > 0:
            # swap the edges in the cycle
            e, Mprime = _swap_edges_in_cycle(cycles[0], M)
            # this creates a new maximum matching, see if we already have it or add it
            if Mprime not in matchings_list:
                matchings_list.append(Mprime)

            # now we have two subpromblems which result in new iterations
            # Problem plus:
            Gplus, Uplus, Vplus, Mplus = _make_graph_plus(G, U, V, M, e)
            Dplus = _make_matching_digraph(Gplus, Uplus, Vplus, Mplus)
            Dplus = _trim_unnecessary_edges(Dplus)
            matchings_list = _enumerate_maximum_matchings_iter(Gplus, Uplus, Vplus, Mplus, Dplus, matchings_list)

            # and Problem minus:
            Gminus = _make_graph_minus(G, e)
            Dminus = _make_matching_digraph(Gminus, U, V, Mprime)
            Dminus = _trim_unnecessary_edges(Dminus)
            matchings_list = _enumerate_maximum_matchings_iter(Gminus, U, V, Mprime, Dminus, matchings_list)

        else:
            # if there are no cycles call the iter again without the matching digraph
            matchings_list = _enumerate_maximum_matchings_iter(G, U, V, M, None, matchings_list)

    elif len(G) > 0:
        path = _find_path_length_two(G, V, M)
        if not (path is None):
            # swap edges in the path
            e, Mprime = _swap_edges_in_path(path, M)
            # this creates a new maximum matching, see if we already have it or add it
            if Mprime not in matchings_list:
                matchings_list.append(Mprime)

            # now we have two subpromblems which result in new iterations
            # Problem plus:
            Gplus, Uplus, Vplus, Mplus = _make_graph_plus(G, U, V, M, e)
            matchings_list = _enumerate_maximum_matchings_iter(Gplus, Uplus, Vplus, Mprime, None, matchings_list)

            # Problem minus with M:
            Gminus = _make_graph_minus(G, e)
            matchings_list = _enumerate_maximum_matchings_iter(Gminus, U, V, M, None, matchings_list)

    return matchings_list


def _make_graph_plus(G, U, V, M, e):
    """
    """
    Gplus = G.copy()
    # remove the two endpoints and all adjacent edges
    Gplus.remove_node(e[0])
    Gplus.remove_node(e[1])

    Uplus = set(U)
    Vplus = set(V)
    if e[0] in U:
        Uplus.remove(e[0])
        Vplus.remove(e[1])
    else:
        Uplus.remove(e[1])
        Vplus.remove(e[0])

    Mplus = dict(M)

    return Gplus, Uplus, Vplus, Mplus


def _make_graph_minus(G, e):
    """
    """
    Gminus = G.copy()
    # remove the edge
    Gminus.remove_edge(e[0], e[1])

    return Gminus


def _find_path_length_two(G, V, M):
    """
    """
    path = None
    for v in _listminus(V, M.keys()):
        for u in G.neighbors(v):
            for w in G.neighbors(u):
                if w in M and M[w] == u:
                    path = [v, u, w]
                    break
    return path


def _listminus(list1, list2):
    """
    """
    return [a for a in list1 if a not in list2]


def _swap_edges_in_cycle(cycle, M):
    """
    """
    Mprime = dict(M)
    e = None
    cycle = cycle + [cycle[0]]
    for i in range(len(cycle) - 1):
        if cycle[i] in Mprime:
            if M[cycle[i]] == cycle[i + 1]:
                e = (cycle[i], cycle[i + 1])
                Mprime[cycle[i]] = cycle[i - 1]
                Mprime[cycle[i - 1]] = Mprime[cycle[i]]
            else:
                e = (cycle[i], cycle[i - 1])
                Mprime[cycle[i]] = cycle[i + 1]
                Mprime[cycle[i + 1]] = Mprime[cycle[i]]
    return e, Mprime


def _swap_edges_in_path(path, M):
    """
    """
    Mprime = dict(M)
    e = (path[2], path[1])
    m = Mprime[path[2]]
    del Mprime[path[2]]
    del Mprime[m]
    Mprime[path[0]] = path[1]
    Mprime[path[1]] = path[0]
    return e, Mprime
