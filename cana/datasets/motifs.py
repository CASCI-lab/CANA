# -*- coding: utf-8 -*-
"""
Network Motifs
===============

Simple network motifs in Networkx.DiGraph format that can be directly loaded.

"""
#   Copyright (C) 2021 by
#   Alex Gates <ajgates@gmail.com>
#   Rion Brattig Correia <rionbr@gmail.com>
#   All rights reserved.
#   MIT license.
import networkx as nx


def network_motif(name=None):
    """Graph motifs from :cite:`Milo:2012`.

    Args:
        name (string): The name of the motif.
            Possible values are : ``FeedForward``, ``Fan``, ``FeedForwardSelf1``,
                ``FeedForwardSelf2``, ``FeedForwardSelf3``, ``FeedForwardSelf123``,
                ``BiFan``, ``CoRegulated``, ``CoRegulating``, ``BiParallel``,
                ``TriParallel``, ``Dominating4``, ``Dominating4Undir``, ``3Loop``,
                ``4Loop``, ``3LoopSelf123``, ``FourLoop``, ``FourCoLoop``,
                ``DirectedTwoLoop``, ``BiParallelLoop``, ``5Chain``, ``3Chain``,
                ``KeffStudy3``, ``KeffStudy4``, ``CoRegulatedSelf``, ``KeffLine4``,
                ``KeffLineLoop4``, ``3Full``, ``6Pyramid``, ``4Split``, ``5BiParallel``,
                ``6BiParallelDilation``, ``6BiParallelDilationLoop``, ``5combine``, ``4tree``.

    Returns:
        (networkx.DiGraph) : The directed graph motif.
    """

    graph = nx.DiGraph()

    if name == "FeedForward":
        graph.add_edge(0, 1)
        graph.add_edge(0, 2)
        graph.add_edge(1, 2)

    elif name == "Fan":
        graph.add_edge(0, 1)
        graph.add_edge(0, 2)

    elif name == "FeedForwardSelf1":
        graph.add_edge(0, 1)
        graph.add_edge(0, 2)
        graph.add_edge(1, 2)
        graph.add_edge(0, 0)

    elif name == "FeedForwardSelf2":
        graph.add_edge(0, 1)
        graph.add_edge(0, 2)
        graph.add_edge(1, 2)
        graph.add_edge(1, 1)

    elif name == "FeedForwardSelf3":
        graph.add_edge(0, 1)
        graph.add_edge(0, 2)
        graph.add_edge(1, 2)
        graph.add_edge(2, 2)

    elif name == "FeedForwardSelf123":
        graph.add_edge(0, 1)
        graph.add_edge(0, 2)
        graph.add_edge(1, 2)
        graph.add_edge(0, 0)
        graph.add_edge(1, 1)
        graph.add_edge(2, 2)

    elif name == "BiFan":
        graph.add_edge(0, 2)
        graph.add_edge(0, 3)
        graph.add_edge(1, 2)
        graph.add_edge(1, 3)

    elif name == "CoRegulated":
        graph.add_edge(0, 1)
        graph.add_edge(0, 2)
        graph.add_edge(1, 2)
        graph.add_edge(2, 1)

    elif name == "CoRegulating":
        graph.add_edge(0, 1)
        graph.add_edge(0, 2)
        graph.add_edge(1, 0)
        graph.add_edge(1, 2)

    elif name == "BiParallel":
        graph.add_edge(0, 1)
        graph.add_edge(0, 2)
        graph.add_edge(1, 3)
        graph.add_edge(2, 3)

    elif name == "TriParallel":
        graph.add_edge(0, 1)
        graph.add_edge(0, 2)
        graph.add_edge(1, 3)
        graph.add_edge(2, 3)
        graph.add_edge(0, 3)

    elif name == "Dominating4":
        graph.add_edge(0, 1)
        graph.add_edge(0, 2)
        graph.add_edge(0, 3)
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)
        graph.add_edge(3, 1)

    elif name == "Dominating4Undir":
        graph.add_edge(0, 1)
        graph.add_edge(0, 2)
        graph.add_edge(0, 3)
        graph.add_edge(1, 0)
        graph.add_edge(2, 0)
        graph.add_edge(3, 0)
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)
        graph.add_edge(3, 1)

    elif name == "3Loop":
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)
        graph.add_edge(2, 0)

    elif name == "4Loop":
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)
        graph.add_edge(3, 0)

    elif name == "3LoopSelf123":
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)
        graph.add_edge(2, 0)
        graph.add_edge(0, 0)
        graph.add_edge(1, 1)
        graph.add_edge(2, 2)

    elif name == "FourLoop":
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)
        graph.add_edge(3, 1)

    elif name == "FourCoLoop":
        graph.add_edge(0, 1)
        graph.add_edge(0, 2)
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)
        graph.add_edge(3, 1)

    elif name == "DirectedTwoLoop":
        graph.add_edge(0, 1)
        graph.add_edge(1, 0)
        graph.add_edge(2, 3)
        graph.add_edge(3, 2)
        graph.add_edge(0, 2)
        graph.add_edge(1, 3)

    elif name == "BiParallelLoop":
        graph.add_edge(0, 1)
        graph.add_edge(0, 2)
        graph.add_edge(1, 3)
        graph.add_edge(2, 3)
        graph.add_edge(3, 0)

    elif name == "5Chain":
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)
        graph.add_edge(2, 3)
        graph.add_edge(3, 4)

    elif name == "3Chain":
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)

    elif name == "KeffStudy3":
        graph.add_edge(0, 1)
        graph.add_edge(1, 0)
        graph.add_edge(0, 2)
        graph.add_edge(2, 0)

    elif name == "KeffStudy4":
        graph.add_edge(0, 1)
        graph.add_edge(1, 0)
        graph.add_edge(0, 2)
        graph.add_edge(2, 0)
        graph.add_edge(0, 3)
        graph.add_edge(3, 0)

    elif name == "CoRegulatedSelf":
        graph.add_edge(0, 1)
        graph.add_edge(0, 2)
        graph.add_edge(1, 2)
        graph.add_edge(2, 1)
        graph.add_edge(0, 0)

    elif name == "KeffLine4":
        graph.add_edge(0, 1)
        graph.add_edge(0, 3)
        graph.add_edge(1, 2)
        graph.add_edge(1, 3)
        graph.add_edge(2, 3)

    elif name == "KeffLineLoop4":
        graph.add_edge(0, 1)
        graph.add_edge(0, 3)
        graph.add_edge(1, 2)
        graph.add_edge(1, 3)
        graph.add_edge(2, 3)
        graph.add_edge(3, 0)

    elif name == "3Full":
        graph.add_edge(0, 0)
        graph.add_edge(0, 1)
        graph.add_edge(0, 2)
        graph.add_edge(1, 0)
        graph.add_edge(1, 1)
        graph.add_edge(1, 2)
        graph.add_edge(2, 0)
        graph.add_edge(2, 1)
        graph.add_edge(2, 2)

    elif name == "6Pyramid":
        graph.add_edge(0, 1)
        graph.add_edge(0, 2)
        graph.add_edge(1, 3)
        graph.add_edge(1, 4)
        graph.add_edge(1, 5)
        graph.add_edge(2, 3)
        graph.add_edge(2, 4)
        graph.add_edge(2, 5)

    elif name == "4Split":
        graph.add_edge(0, 1)
        graph.add_edge(0, 2)
        graph.add_edge(1, 3)
        graph.add_edge(2, 3)
        graph.add_edge(0, 3)

    elif name == "5BiParallel":
        graph.add_edge(0, 2)
        graph.add_edge(0, 3)
        graph.add_edge(0, 4)
        graph.add_edge(1, 2)
        graph.add_edge(1, 3)
        graph.add_edge(1, 4)

    elif name == "6BiParallelDilation":
        graph.add_edge(0, 2)
        graph.add_edge(0, 3)
        graph.add_edge(0, 4)
        graph.add_edge(1, 2)
        graph.add_edge(1, 3)
        graph.add_edge(1, 4)
        graph.add_edge(2, 5)
        graph.add_edge(3, 5)
        graph.add_edge(4, 5)

    elif name == "6BiParallelDilationLoop":
        graph.add_edge(0, 2)
        graph.add_edge(0, 3)
        graph.add_edge(0, 4)
        graph.add_edge(1, 2)
        graph.add_edge(1, 3)
        graph.add_edge(1, 4)
        graph.add_edge(2, 5)
        graph.add_edge(3, 5)
        graph.add_edge(4, 5)
        graph.add_edge(5, 1)

    elif name == "5combine":
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)
        graph.add_edge(0, 2)
        graph.add_edge(1, 3)
        graph.add_edge(3, 0)
        graph.add_edge(3, 4)
        graph.add_edge(4, 2)

    elif name == "4tree":
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)
        graph.add_edge(1, 3)

    else:
        raise TypeError('The motif name could not be found.')

    return graph
