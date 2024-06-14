# -*- coding: utf-8 -*-
"""
Random Boolean Network
======================

Methods to generate random ensembles of Boolean networks.

"""
import random
import re

#   Copyright (C) 2021 by
#   Alex Gates <ajgates@indiana.edu>
#   Rion Brattig Correia <rionbr@gmail.com>
#   Thomas Parmer <tjparmer@indiana.edu>
#   All rights reserved.
#   MIT license.
from collections import Counter, defaultdict
from io import StringIO

import networkx as nx

from cana.boolean_network import BooleanNetwork
from cana.utils import output_transitions


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
    Generate a random regular boolean network.
    
    Args:
        N (int) : Number of nodes in the network.
        K (int) : Degree of each node in the network.
        bias (float) : Bias for the output transitions.
        bias_constraint (str) : Constraint for the bias. Options are 'soft', 'hard', 'soft_no_constant'.
        keep_constants (bool) : Keep constant nodes.
        remove_multiedges (bool) : Remove multi-edges.
        niter_remove (int) : Number of iterations to try to remove duplicate edges.

    Returns:
        (BooleanNetwork) : The boolean network object.

    Examples:
        A regular boolean network with 10 nodes, each with 2 inputs and a bias of 0.5.

        >>> bn = regular_boolean_network(N=10, K=2, bias=0.5)

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

    # A dict that contains the network logic {<id>:{'name':<string>,'in':<list-input-node-id>,'out':<list-output-transitions>},..}
    bn_dict = {
        node: {
            "name": str(node),
            "in": sorted([n for n in regular_graph.predecessors(node)]),
            "out": random_automata_table(
                regular_graph.in_degree(node), bias, bias_constraint
            ),
        }
        for node in range(N)
    }

    return BooleanNetwork.from_dict(bn_dict, keep_constants=keep_constants)


def er_boolean_network(
    N=10,
    p=0.2,
    bias=0.5,
    bias_constraint="soft",
    remove_multiedges=True,
    niter_remove=1000,
):
    """
    Generate a random Erdos-Renyi boolean network.

    Args:
        N (int) : Number of nodes in the network.
        p (float) : Probability for edge creation.
        bias (float) : Bias for the output transitions.
        bias_constraint (str) : Constraint for the bias. Options are 'soft', 'hard', 'soft_no_constant'.
        remove_multiedges (bool) : Remove multi-edges.
        niter_remove (int) : Number of iterations to try to remove duplicate edges.

    Returns:
        (BooleanNetwork) : The boolean network object.

    Examples:
        A random Erdos-Renyi boolean network with 10 nodes and a probability of 0.2.

        >>> bn = er_boolean_network(N=10, p=0.2)
        
    """
    er_graph = nx.erdos_renyi_graph(N, p, directed=True)

    # the configuration graph creates a multigraph with self loops
    # the self loops are OK, but we should only have one copy of each edge
    if remove_multiedges:
        er_graph = _remove_duplicate_edges(graph=er_graph, niter_remove=niter_remove)

    # A dict that contains the network logic {<id>:{'name':<string>,'in':<list-input-node-id>,'out':<list-output-transitions>},..}
    bn_dict = {
        node: {
            "name": str(node),
            "in": sorted([n for n in er_graph.predecessors(node)]),
            "out": random_automata_table(
                er_graph.in_degree(node), bias, bias_constraint
            ),
        }
        for node in range(N)
    }

    return BooleanNetwork.from_dict(bn_dict)


def random_automata_table(indegree, bias, bias_constraint="soft"):
    """
    Generate a random automata table.

    Args:
        indegree (int) : Number of inputs.
        bias (float) : Bias for the output transitions.
        bias_constraint (str) : Constraint for the bias. Options are 'soft', 'hard', 'soft_no_constant'.

    Returns:
        (list) : A list of output transitions.

    Examples:
        A random automata table with 2 inputs and a bias of 0.5.

        >>> random_automata_table(indegree=2, bias=0.5)

    """
    if bias_constraint == "soft":
        return [int(random.random() < bias) for b in range(2**indegree)]

    elif bias_constraint == "hard":
        n_ones = int(bias * (2**indegree))
        output = [0] * (2**indegree - n_ones) + [1] * n_ones
        random.shuffle(output)
        return output

    elif bias_constraint == "soft_no_constant":
        output = [int(random.random() < bias) for b in range(2**indegree)]
        if sum(output) == 0:
            output[0] = 1
            random.shuffle(output)
        return output


def _remove_duplicate_edges(graph, niter_remove=100):
    """
    Remove duplicate edges from a graph.

    Args:
        graph (nx.DiGraph) : A directed graph.
        niter_remove (int) : Number of iterations to try to remove duplicate edges.

    Returns:
        (nx.DiGraph) : A directed graph without duplicate edges.

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


def from_string_boolean(self, string, keep_constants=True, **kwargs):
    """
    Instanciates a Boolean Network from a Boolean update rules format.

    Args:
        string (string) : A boolean update rules format representation of a Boolean Network.

    Returns:
        (BooleanNetwork) : The boolean network object.

    Examples:
        String should be structured as follow:

        .. code-block:: text

            # BOOLEAN RULES (this is a comment)
            # node_name*=node_input_1 [logic operator] node_input_2 ...
            NODE3*=NODE1 AND NODE2 ...

    See also:
        :func:`from_string` :func:`from_dict`
    """

    logic = defaultdict(dict)

    # parse lines to receive node names
    network_file = StringIO(string)
    line = network_file.readline()
    i = 0
    while line != "":
        if line[0] == "#":
            line = network_file.readline()
            continue
        logic[i] = {"name": line.split("*")[0].strip(), "in": [], "out": []}
        line = network_file.readline()
        i += 1

    # Parse lines again to determine inputs and output sequence
    network_file = StringIO(string)
    line = network_file.readline()
    i = 0
    while line != "":
        if line[0] == "#":
            line = network_file.readline()
            continue
        eval_line = line.split("=")[1]  # logical condition to evaluate
        # RE checks for non-alphanumeric character before/after node name (node names are included in other node names)
        # Additional characters added to eval_line to avoid start/end of string complications
        input_names = [
            logic[node]["name"]
            for node in logic
            if re.compile(r"\W" + logic[node]["name"] + r"\W").search(
                "*" + eval_line + "*"
            )
        ]
        input_nums = [
            node
            for input in input_names
            for node in logic
            if input == logic[node]["name"]
        ]
        logic[i]["in"] = input_nums
        # Determine output transitions
        logic[i]["out"] = output_transitions(eval_line, input_names)
        line = network_file.readline()
        i += 1

    return self.from_dict(logic, keep_constants=keep_constants, **kwargs)
