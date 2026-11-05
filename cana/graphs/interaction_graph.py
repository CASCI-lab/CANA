# -*- coding: utf-8 -*-
"""Interaction (wiring) graphs of Boolean networks.

Nodes are variables; an edge ``source -> target`` exists when ``source`` is
an input of ``target``.  Edge ``weight`` is ``1.0`` (topology only).  Edge
``sign`` is an ``Optional[frozenset[int]]``:

- ``None`` when ``signed=False`` (not computed)
- ``frozenset({1})`` / ``frozenset({-1})`` — pure activation / inhibition
- ``frozenset({1, -1})`` — binate / dual-sign
- ``frozenset()`` — irrelevant (no influence)
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import networkx as nx


def fill_wiring_signs(G, nodes: Sequence) -> nx.DiGraph:
    """Set edge ``sign`` frozensets from each target node's ``input_signs``.

    Args:
        G (networkx.DiGraph) : Digraph whose nodes are integer node ids.
        nodes (sequence) : Nodes with ``.inputs`` and ``.input_signs()``,
            indexed by id (typically ``BooleanNetwork.nodes``).

    Returns:
        G (networkx.DiGraph) : The same graph, mutated in place.
    """
    for target in range(len(nodes)):
        if target not in G:
            continue
        signs = nodes[target].input_signs()
        for idx, source in enumerate(nodes[target].inputs):
            if G.has_edge(source, target):
                G[source][target]["sign"] = signs[idx]
    G.graph["signs_computed"] = True
    return G


def build_interaction_graph(
    nodes: Sequence,
    name: str = "",
    remove_constants: bool = False,
    signed: bool = True,
    constant_ids: Optional[Iterable] = None,
) -> nx.DiGraph:
    """Build the interaction graph from a sequence of Boolean nodes.

    Args:
        nodes (sequence) : Nodes with ``.name``, ``.inputs``, and
            ``.input_signs()`` (when ``signed=True``).
        name (str) : Network name used in the digraph ``name`` attribute.
        remove_constants (bool) : Drop constant nodes. Defaults to ``False``.
        signed (bool) : Compute and attach interaction signs. Defaults to
            ``True``.
        constant_ids (iterable, optional) : Node ids removed when
            ``remove_constants`` is True.  Ignored otherwise.

    Returns:
        G (networkx.DiGraph) : The interaction graph.
    """
    G = nx.DiGraph(name="Interaction Graph: " + name)
    G.graph["signs_computed"] = bool(signed)

    G.add_nodes_from((i, {"label": n.name}) for i, n in enumerate(nodes))
    for target in range(len(nodes)):
        if signed:
            signs = nodes[target].input_signs()
        else:
            signs = [None] * len(nodes[target].inputs)
        for idx, source in enumerate(nodes[target].inputs):
            G.add_edge(source, target, **{"weight": 1.0, "sign": signs[idx]})

    if remove_constants and constant_ids is not None:
        G.remove_nodes_from(constant_ids)
    return G
