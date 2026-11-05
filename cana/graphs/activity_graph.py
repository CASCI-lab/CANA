# -*- coding: utf-8 -*-
"""Activity graphs of Boolean networks.

As proposed in Ghanbarnejad & Klemm (2012) EPL, 99.  Edge weights are
per-input activities (:meth:`~cana.boolean_node.BooleanNode.activities`).
Optional interaction ``sign`` frozensets use the same conventions as
:mod:`~cana.graphs.interaction_graph`.

The unthresholded graph includes every wiring edge (activity may be ``0``)
and records ``G.graph["threshold"] is None``.  Pass that graph as ``base_ag``
to apply a threshold without recomputing activities.  Applying a lower
threshold (or ``None``) than the base is an error — filtered edges cannot be
recovered.
"""
from __future__ import annotations

import copy
from typing import Optional, Sequence

import networkx as nx

from cana.graphs.interaction_graph import fill_wiring_signs


def _require_threshold_compatible(base, threshold):
    base_thr = base.graph.get("threshold", None)
    if base_thr is None:
        return
    if threshold is None or threshold < base_thr:
        raise ValueError(
            "cannot apply threshold={0!r} to an activity graph whose "
            "threshold is {1!r}; pass an unthresholded base_ag "
            "(graph['threshold'] is None)".format(threshold, base_thr)
        )


def build_activity_graph(
    nodes: Sequence,
    name: str = "",
    threshold: Optional[float] = None,
    signed: bool = False,
    base_ag: Optional[nx.DiGraph] = None,
) -> nx.DiGraph:
    """Build the activity graph from a sequence of Boolean nodes.

    Pass ``base_ag`` (typically an unsigned, unthresholded AG) to reuse a
    cached graph without recomputing activities; ``base_ag`` is never mutated.

    Args:
        nodes (sequence) : Nodes with ``.name``, ``.inputs``,
            ``.activities()``, and ``.input_signs()`` (when ``signed=True``).
            Ignored for edge weights when ``base_ag`` is provided.
        name (str) : Network name used in the digraph ``name`` attribute.
        threshold (float, optional) : Keep edges with activity ``> threshold``.
            When ``None``, keep all wiring edges (including weight ``0``).
        signed (bool) : Attach interaction ``sign`` frozensets.  When True,
            works on a deep copy of the unsigned graph. Defaults to ``False``.
        base_ag (networkx.DiGraph, optional) : Cached activity graph.  Must
            satisfy ``base_ag.graph["threshold"] is None`` or
            ``threshold >= base_ag.graph["threshold"]``.  Built from ``nodes``
            when omitted.

    Returns:
        G (networkx.DiGraph) : Activity graph with ``weight`` and ``sign``.
            ``G.graph["threshold"]`` records the applied threshold.
    """
    if base_ag is None:
        base = nx.DiGraph(name="Activity Graph: " + name + "(Threshold: None)")
        base.graph["signs_computed"] = False
        base.graph["threshold"] = None
        for i, node in enumerate(nodes):
            base.add_node(i, **{"label": node.name})
        for i, node in enumerate(nodes):
            a_is = node.activities()
            for source, a_i in zip(node.inputs, a_is):
                base.add_edge(source, i, **{"weight": a_i, "sign": None})
    else:
        _require_threshold_compatible(base_ag, threshold)
        base = base_ag

    if signed:
        base = copy.deepcopy(base)
        fill_wiring_signs(base, nodes)

    if threshold is not None:
        thresholded = nx.DiGraph(
            name="Activity Graph: " + name + "(Threshold: %.2f)" % threshold
        )
        thresholded.graph["signs_computed"] = base.graph.get(
            "signs_computed", False
        )
        thresholded.graph["threshold"] = threshold
        thresholded.add_nodes_from(base.nodes(data=True))
        for i, j, d in base.edges(data=True):
            if d["weight"] > threshold:
                thresholded.add_edge(
                    i,
                    j,
                    **{"weight": d["weight"], "sign": d.get("sign")},
                )
        return thresholded
    return base
