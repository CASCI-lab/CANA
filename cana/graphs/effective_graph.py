# -*- coding: utf-8 -*-
"""Effective graphs of Boolean networks.

Edge weights are per-input edge effectiveness
(:meth:`~cana.boolean_node.BooleanNode.edge_effectiveness`).  Optional
interaction ``sign`` frozensets use the same conventions as
:mod:`~cana.graphs.interaction_graph`.
"""
from __future__ import annotations

import copy
from typing import Dict, Mapping, Optional, Sequence

import networkx as nx

from cana.boolean_node import BooleanNode
from cana.cutils import binstate_to_statenum, statenum_to_binstate
from cana.function_outputs import BooleanFunction
from cana.function_outputs.bit_lut import BitLut
from cana.graphs.interaction_graph import fill_wiring_signs


def _require_threshold_compatible(base, threshold):
    base_thr = base.graph.get("threshold", None)
    if base_thr is None:
        return
    if threshold is None or threshold < base_thr:
        raise ValueError(
            "cannot apply threshold={0!r} to an effective graph whose "
            "threshold is {1!r}; pass an unthresholded base_eg "
            "(graph['threshold'] is None)".format(threshold, base_thr)
        )


def build_effective_graph(
    nodes: Sequence,
    name: str = "",
    bound: str = "mean",
    threshold: Optional[float] = None,
    signed: bool = False,
    base_eg: Optional[nx.DiGraph] = None,
) -> nx.DiGraph:
    """Build the effective graph of a Boolean network.

    Edge weights are edge effectiveness.  Pass ``base_eg`` (an unsigned,
    unthresholded EG) to reuse a cached graph without recomputing
    effectiveness; ``base_eg`` is never mutated.

    Args:
        nodes (sequence) : Nodes with ``.name``, ``.inputs``, and
            ``.edge_effectiveness`` (and ``.input_signs`` when signed).
        name (str) : Network name used in the digraph ``name`` attribute.
        bound (str) : Bound forwarded to ``edge_effectiveness``.
            One of ``"lower"``, ``"mean"``, ``"upper"``, ``"tuple"``.
            Ignored when ``base_eg`` is provided.
        threshold (float, optional) : Keep edges with ``weight > threshold``.
        signed (bool) : Attach interaction ``sign`` frozensets.  When True,
            works on a deep copy of the unsigned graph.
        base_eg (networkx.DiGraph, optional) : Cached effective graph.  Must
            satisfy ``base_eg.graph["threshold"] is None`` or
            ``threshold >= base_eg.graph["threshold"]``.  Built from ``nodes``
            when omitted.

    Returns:
        G (networkx.DiGraph) : Possibly signed and/or thresholded EG.
            ``G.graph["threshold"]`` records the applied threshold.
    """
    if base_eg is None:
        base = nx.DiGraph(name="Effective Graph: " + name + "(Threshold: None)")
        base.graph["signs_computed"] = False
        base.graph["threshold"] = None
        for i, node in enumerate(nodes):
            base.add_node(i, **{"label": node.name})
        for i, node in enumerate(nodes):
            e_is = node.edge_effectiveness(bound=bound)
            for source, e_i in zip(node.inputs, e_is):
                base.add_edge(source, i, **{"weight": e_i, "sign": None})
    else:
        _require_threshold_compatible(base_eg, threshold)
        base = base_eg

    if signed:
        base = copy.deepcopy(base)
        fill_wiring_signs(base, nodes)

    if threshold is not None:
        thresholded_eg = nx.DiGraph(
            name="Effective Graph: "
            + name
            + "(Threshold: {threshold:.2f})".format(threshold=threshold)
        )
        thresholded_eg.graph["signs_computed"] = base.graph.get(
            "signs_computed", False
        )
        thresholded_eg.graph["threshold"] = threshold
        thresholded_eg.add_nodes_from(base.nodes(data=True))
        for i, j, d in base.edges(data=True):
            if d["weight"] > threshold:
                thresholded_eg.add_edge(
                    i,
                    j,
                    **{"weight": d["weight"], "sign": d.get("sign")},
                )
        return thresholded_eg
    return base


def build_conditional_effective_graph(
    nodes: Sequence,
    conditioned_nodes: Optional[Mapping] = None,
    bound: str = "mean",
    threshold: Optional[float] = None,
    signed: bool = False,
    name: str = "",
    base_eg: Optional[nx.DiGraph] = None,
) -> nx.DiGraph:
    """Build an effective graph conditioned on fixed node states.

    Propagates constants induced by conditioning, recomputing edge
    effectiveness on the remaining inputs of affected successors.

    Args:
        nodes (sequence) : Original network nodes (not mutated).
        conditioned_nodes (mapping) : ``{node_id: state}`` fixed values.
        bound (str) : Bound for edge effectiveness.
        threshold (float, optional) : Drop edges with ``weight <= threshold``.
        signed (bool) : Attach signs from conditioned LUTs.
        name (str) : Network name for digraph naming.
        base_eg (networkx.DiGraph, optional) : Unsigned unthresholded EG to
            start from.  Built via :func:`build_effective_graph` if omitted.

    Returns:
        (networkx.DiGraph) : Conditioned effective graph.
    """
    if conditioned_nodes is None:
        conditioned_nodes = {}

    if base_eg is None:
        base_eg = build_effective_graph(nodes, name=name, bound=bound)

    conditional_eg = copy.deepcopy(base_eg)
    conditioned_subgraph = set([])

    conditioned_logic = [
        {
            "inputs": list(node.inputs),
            "function": copy.deepcopy(node.function),
        }
        for node in nodes
    ]

    all_conditioned_nodes = dict(conditioned_nodes)
    nodes2condition = list(conditioned_nodes.keys())

    while len(nodes2condition) > 0:
        conditioned_node = nodes2condition.pop(0)
        conditioned_value = str(all_conditioned_nodes[conditioned_node])
        conditioned_subgraph.add(conditioned_node)

        successors = [
            n
            for n in list(conditional_eg.neighbors(conditioned_node))
            if n != conditioned_node
        ]
        conditioned_subgraph.update(successors)

        for n in successors:
            conditioned_node_idx = conditioned_logic[n]["inputs"].index(
                conditioned_node
            )

            conditioned_subgraph.update(conditioned_logic[n]["inputs"])

            new_successor_inputs = (
                conditioned_logic[n]["inputs"][:conditioned_node_idx]
                + conditioned_logic[n]["inputs"][(conditioned_node_idx + 1) :]
            )
            newk = len(new_successor_inputs)

            n_lut = conditioned_logic[n]["function"].get_bit_lut()
            if newk == 0:
                new_successor_outputs = [
                    n_lut[binstate_to_statenum(conditioned_value)]
                ] * 2
            else:
                new_successor_outputs = []
                for sn in range(2**newk):
                    binstate = statenum_to_binstate(sn, newk)
                    binstate = (
                        binstate[:conditioned_node_idx]
                        + conditioned_value
                        + binstate[conditioned_node_idx:]
                    )
                    new_successor_outputs.append(
                        n_lut[binstate_to_statenum(binstate)]
                    )

            new_edge_effectiveness = (
                BooleanNode()
                .from_bit_lut(new_successor_outputs)
                .edge_effectiveness(bound=bound)
            )

            for i in range(newk):
                conditional_eg[new_successor_inputs[i]][n][
                    "weight"
                ] = new_edge_effectiveness[i]

            conditioned_logic[n]["inputs"] = new_successor_inputs
            conditioned_logic[n]["function"] = BooleanFunction(
                bit_lut=BitLut.from_sequence(new_successor_outputs)
            )

            if n not in all_conditioned_nodes and len(set(new_successor_outputs)) == 1:
                nodes2condition.append(n)
                all_conditioned_nodes[n] = new_successor_outputs[0]

    conditional_eg.name = (
        "Conditioned Effective Graph: {name:s} conditioned on {nodes:s}".format(
            name=name, nodes=str(dict(conditioned_nodes))
        )
    )
    if threshold is None:
        conditional_eg.name = conditional_eg.name + " (Threshold: None)"
    else:
        conditional_eg.name = (
            conditional_eg.name
            + " (Threshold: {threshold:.2f})".format(threshold=threshold)
        )
        remove_edges = [
            (i, j)
            for i, j, d in conditional_eg.edges(data=True)
            if d["weight"] <= threshold
        ]
        conditional_eg.remove_edges_from(remove_edges)

    dict_conditioned_subgraph = {
        n: (n in conditioned_subgraph) for n in conditional_eg.nodes()
    }
    nx.set_node_attributes(
        conditional_eg,
        values=dict_conditioned_subgraph,
        name="conditioned_subgraph",
    )

    dict_all_conditioned_nodes: Dict = {
        n: all_conditioned_nodes.get(n, None) for n in conditional_eg.nodes()
    }
    nx.set_node_attributes(
        conditional_eg, values=dict_all_conditioned_nodes, name="conditioned_state"
    )

    if signed:
        from cana.function_properties.unate import input_signs as lut_input_signs

        for target, logic in enumerate(conditioned_logic):
            if target not in conditional_eg:
                continue
            inputs = logic["inputs"]
            if not inputs:
                continue
            signs = lut_input_signs(logic["function"].get_bit_lut())
            for idx, source in enumerate(inputs):
                if conditional_eg.has_edge(source, target):
                    conditional_eg[source][target]["sign"] = signs[idx]
        conditional_eg.graph["signs_computed"] = True
    else:
        for _u, _v, data in conditional_eg.edges(data=True):
            data["sign"] = None
        conditional_eg.graph["signs_computed"] = False

    return conditional_eg
