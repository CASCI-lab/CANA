# -*- coding: utf-8 -*-
"""
Random Boolean Network
======================

Methods to generate random ensembles of Boolean networks.

"""
import random

#   Copyright (C) 2021 by
#   Alex Gates <ajgates@indiana.edu>
#   Rion Brattig Correia <rionbr@gmail.com>
#   Thomas Parmer <tjparmer@indiana.edu>
#   All rights reserved.
#   MIT license.
from collections import Counter

import networkx as nx

from cana.boolean_network import BooleanNetwork
from cana.boolean_node import BooleanNode
from cana.function_generation import random_automata_table


def regular_boolean_network(
    N=10,
    K=2,
    bias=0.5,
    bias_constraint="soft",
    keep_constants=True,
    remove_multiedges=True,
    niter_remove=1000,
):
    """
    TODO: description
    """
    din = [K] * N  # in-degree distrubtion
    dout = [K] * N  # out-degree distrubtion

    regular_graph = nx.directed_configuration_model(din, dout)

    # the configuration graph creates a multigraph with self loops
    # the self loops are OK, but we should only have one copy of each edge
    if remove_multiedges:
        regular_graph = _remove_duplicate_edges(
            graph=regular_graph, niter_remove=niter_remove
        )

    nodes = []
    for node in range(N):
        inputs = sorted(n for n in regular_graph.predecessors(node))
        nodes.append(
            BooleanNode.from_bit_lut(
                random_automata_table(
                    regular_graph.in_degree(node), bias, bias_constraint
                ),
                id=node,
                name=str(node),
                inputs=inputs,
            )
        )

    return BooleanNetwork.from_nodes(nodes, keep_constants=keep_constants)


def er_boolean_network(
    N=10,
    p=0.2,
    bias=0.5,
    bias_constraint="soft",
    remove_multiedges=True,
    niter_remove=1000,
):
    """
    TODO: description
    """
    er_graph = nx.erdos_renyi_graph(N, p, directed=True)

    # the configuration graph creates a multigraph with self loops
    # the self loops are OK, but we should only have one copy of each edge
    if remove_multiedges:
        er_graph = _remove_duplicate_edges(graph=er_graph, niter_remove=niter_remove)

    nodes = []
    for node in range(N):
        inputs = sorted(n for n in er_graph.predecessors(node))
        nodes.append(
            BooleanNode.from_bit_lut(
                random_automata_table(
                    er_graph.in_degree(node), bias, bias_constraint
                ),
                id=node,
                name=str(node),
                inputs=inputs,
            )
        )

    return BooleanNetwork.from_nodes(nodes)


def _remove_duplicate_edges(graph, niter_remove=100):
    """
    TODO: description
    """
    edge_list = list(graph.edges())
    edge_frequency = Counter(edge_list)

    duplicate_edges = [
        edge for edge, num_edge in edge_frequency.items() if num_edge > 1
    ]

    iremove_iter = 0
    while len(duplicate_edges) > 0 and iremove_iter < niter_remove:
        for dedge in duplicate_edges:
            if edge_frequency[dedge] > 1:
                switch_edge = random.choice(edge_list)

                # exchange edges
                graph.remove_edge(dedge[0], dedge[1])
                graph.remove_edge(switch_edge[0], switch_edge[1])
                graph.add_edge(dedge[0], switch_edge[1])
                graph.add_edge(switch_edge[0], dedge[1])

                edge_frequency[dedge] -= 1
                edge_frequency[switch_edge] -= 1

                edge_frequency[(dedge[0], switch_edge[1])] += 1
                edge_frequency[(switch_edge[0], dedge[1])] += 1

                edge_list = list(graph.edges())

        duplicate_edges = [
            edge for edge, num_edge in edge_frequency.items() if num_edge > 1
        ]
        iremove_iter += 1

    if iremove_iter >= niter_remove:
        print(
            "Warning: multi-edges were not successfully removed after %s iterations!!"
            % str(iremove_iter)
        )

    return graph
