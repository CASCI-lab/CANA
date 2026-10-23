# -*- coding: utf-8 -*-
"""
Simplify Canalizing Map (CM) / Dynamics Canalization Map (DCM) graphs.

Provides threshold-unit classification, compatibility-gated reduction, and
:func:`simplify_canalizing_map` — a deterministic fixed-point rewrite that
reduces uppermost (farthest-from-out) units first.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx

# ---------------------------------------------------------------------------
# Unit classes and compatibility
# ---------------------------------------------------------------------------

CLASS_LINEAR = "LINEAR"
CLASS_AND = "AND"
CLASS_OR = "OR"
CLASS_OTHER = "OTHER"
REDUCIBLE_CLASSES = frozenset(
    {CLASS_LINEAR, CLASS_AND, CLASS_OR, CLASS_OTHER}
)

_COMPAT: Dict[str, frozenset] = {
    CLASS_LINEAR: frozenset(
        {CLASS_AND, CLASS_OR, CLASS_LINEAR, CLASS_OTHER}
    ),
    CLASS_AND: frozenset({CLASS_AND, CLASS_LINEAR}),
    CLASS_OR: frozenset({CLASS_OR, CLASS_LINEAR}),
    # OTHER (1 < τ < n) only into LINEAR buffers (τ=1, n=1).
    CLASS_OTHER: frozenset({CLASS_LINEAR}),
}

_THRESHOLD_LIKE = frozenset({"threshold", "fusion"})


# ---------------------------------------------------------------------------
# Classification / compatibility
# ---------------------------------------------------------------------------


def is_threshold_like(graph: nx.DiGraph, node_id) -> bool:
    """True for ``threshold`` nodes and legacy ``fusion`` (τ≡1) units."""
    return graph.nodes[node_id].get("type") in _THRESHOLD_LIKE


def unit_tau(graph: nx.DiGraph, node_id) -> Optional[int]:
    """Effective τ: fusion ≡ 1; threshold uses ``tau``; else None."""
    data = graph.nodes[node_id]
    ntype = data.get("type")
    if ntype == "fusion":
        return 1
    if ntype == "threshold":
        tau = data.get("tau")
        return None if tau is None else int(tau)
    return None


def classify_unit(graph: nx.DiGraph, node_id) -> Optional[str]:
    """Return LINEAR / AND / OR / OTHER, or None if not classifiable as a unit.

    Variables are always OR (fixed τ; never bumped).
    Fusion nodes are threshold-like with ``tau=1`` (LINEAR if ``n==1``, else OR).
    """
    if node_id not in graph:
        return None
    data = graph.nodes[node_id]
    ntype = data.get("type")
    if ntype == "variable":
        return CLASS_OR
    if ntype not in _THRESHOLD_LIKE:
        return None
    tau = unit_tau(graph, node_id)
    if tau is None:
        return None
    n = graph.in_degree(node_id)
    if tau == 1 and n == 1:
        return CLASS_LINEAR
    if tau == n:
        return CLASS_AND
    if tau == 1:
        return CLASS_OR
    return CLASS_OTHER


def reduction_skip_reason(graph: nx.DiGraph, node_id) -> Optional[str]:
    """Return a skip reason, or None if reduction may be attempted."""
    if node_id not in graph:
        return "missing"
    if not is_threshold_like(graph, node_id):
        return "not_threshold"
    tau = unit_tau(graph, node_id)
    if tau is None:
        return "tau_none"
    if tau == 0:
        return "tau0"
    n = int(graph.in_degree(node_id))
    if n == 0:
        return "n0"
    succs = list(graph.successors(node_id))
    if not succs:
        return "no_successors"
    cls = classify_unit(graph, node_id)
    if cls is None:
        return "unclassified"
    if cls not in REDUCIBLE_CLASSES:
        return "not_reducible"
    for s in succs:
        scls = classify_unit(graph, s)
        if scls is None:
            return "succ_unclassified"
        allowed = _COMPAT.get(cls, frozenset())
        if scls not in allowed:
            return "incompatible"
    return None


def reduction_compatible(graph: nx.DiGraph, node_id) -> bool:
    """True if :func:`reduce_threshold_unit` may be applied to *node_id*."""
    return reduction_skip_reason(graph, node_id) is None


# ---------------------------------------------------------------------------
# Reduction rewrite
# ---------------------------------------------------------------------------


def _bump_successor_tau(graph: nx.DiGraph, succ_id, bump: int) -> None:
    """Add *bump* to a threshold/fusion successor's τ (never variables).

    Fusion with τ'≠1 is promoted to ``type=threshold`` so evaluation respects
    the new threshold.
    """
    if bump == 0:
        return
    sdata = graph.nodes[succ_id]
    ntype = sdata.get("type")
    if ntype == "threshold":
        tau = sdata.get("tau")
        if tau is None:
            return
        new_tau = int(tau) + bump
        graph.nodes[succ_id]["tau"] = new_tau
        graph.nodes[succ_id]["label"] = "{:d}".format(new_tau)
    elif ntype == "fusion":
        new_tau = 1 + bump
        graph.nodes[succ_id]["type"] = "threshold"
        graph.nodes[succ_id]["tau"] = new_tau
        graph.nodes[succ_id]["label"] = "{:d}".format(new_tau)


def reduce_threshold_unit(
    graph: nx.DiGraph, node_id, *, copy: bool = True
) -> nx.DiGraph:
    """Apply the uniform threshold-unit reduction to *node_id*.

    Wire all predecessors to all successors, bump each threshold/fusion
    successor by ``tau(T)-1``, delete ``T``.  Assumes
    :func:`reduction_compatible` already passed.
    """
    G = graph.copy() if copy else graph
    preds = list(G.predecessors(node_id))
    succs = list(G.successors(node_id))
    tau_t = unit_tau(G, node_id)
    if tau_t is None:
        raise ValueError("node {!r} has no effective tau".format(node_id))
    bump = int(tau_t) - 1

    for pred in preds:
        for succ in succs:
            if G.has_edge(pred, succ):
                continue
            if G.has_edge(node_id, succ):
                edge_data = dict(G.edges[node_id, succ])
            elif G.has_edge(pred, node_id):
                edge_data = dict(G.edges[pred, node_id])
            else:
                edge_data = {}
            edge_data["mode"] = "threshold_reduce"
            G.add_edge(pred, succ, **edge_data)

    for succ in succs:
        _bump_successor_tau(G, succ, bump)

    G.remove_node(node_id)
    return G


# ---------------------------------------------------------------------------
# Distance to out / candidate ordering
# ---------------------------------------------------------------------------


def _distance_to_out(graph: nx.DiGraph, node_id) -> int:
    """Directed shortest-path length from *node_id* to any variable node.

    Returns ``-1`` if no variable is reachable (sorts after reachable units
    when maximizing distance).
    """
    if node_id not in graph:
        return -1
    if graph.nodes[node_id].get("type") == "variable":
        return 0

    # BFS forward toward out sinks.
    visited = {node_id}
    queue: List[Tuple[object, int]] = [(node_id, 0)]
    while queue:
        cur, dist = queue.pop(0)
        for succ in graph.successors(cur):
            if succ in visited:
                continue
            visited.add(succ)
            if graph.nodes[succ].get("type") == "variable":
                return dist + 1
            queue.append((succ, dist + 1))
    return -1


def _candidate_sort_key(distance: int, node_id) -> Tuple:
    """Farthest-from-out first, then lexicographic node id."""
    # Distance -1 (unreachable) sorts after all non-negative distances when
    # we use (-distance, ...) for reachable nodes; map -1 → a large positive
    # so (-large) is smallest and loses the max-distance contest.
    dist_key = distance if distance >= 0 else -10**9
    return (-dist_key, str(node_id))


def _edge_delta_after_reduce(graph: nx.DiGraph, node_id) -> int:
    """Net edge-count change if *node_id* were reduced (preds→succs wiring).

    Removing ``T`` drops ``|P| + |S|`` incident edges and adds one edge per
    ``(pred, succ)`` pair that does not already exist. Positive means the
    edge count would increase.
    """
    preds = list(graph.predecessors(node_id))
    succs = list(graph.successors(node_id))
    removed = len(preds) + len(succs)
    added = sum(
        1
        for pred in preds
        for succ in succs
        if not graph.has_edge(pred, succ)
    )
    return added - removed


def _collect_candidates(
    graph: nx.DiGraph, *, skip_if_edges_increase: bool = False
) -> List[Tuple[object, int]]:
    """Return ``(node_id, distance)`` for reduction-compatible units."""
    candidates: List[Tuple[object, int]] = []
    for n, data in graph.nodes(data=True):
        if data.get("type") not in _THRESHOLD_LIKE:
            continue
        if not reduction_compatible(graph, n):
            continue
        if skip_if_edges_increase and _edge_delta_after_reduce(graph, n) > 0:
            continue
        candidates.append((n, _distance_to_out(graph, n)))
    return candidates


def _pick_next_candidate(
    graph: nx.DiGraph, *, skip_if_edges_increase: bool = False
):
    """Select the next unit to reduce: farthest from out, then min node id."""
    candidates = _collect_candidates(
        graph, skip_if_edges_increase=skip_if_edges_increase
    )
    if not candidates:
        return None
    node_id, _dist = min(
        candidates, key=lambda c: _candidate_sort_key(c[1], c[0])
    )
    return node_id


def _simplify_one_step(
    graph: nx.DiGraph, *, skip_if_edges_increase: bool = False
):
    """Apply one reduction in place. Return the removed node id, or None."""
    node_id = _pick_next_candidate(
        graph, skip_if_edges_increase=skip_if_edges_increase
    )
    if node_id is None:
        return None
    reduce_threshold_unit(graph, node_id, copy=False)
    return node_id


# ---------------------------------------------------------------------------
# Public simplify entry point
# ---------------------------------------------------------------------------


def simplify_canalizing_map(
    cm: nx.DiGraph,
    *,
    copy: bool = True,
    remove_isolates: bool = True,
    skip_if_edges_increase: bool = False,
) -> nx.DiGraph:
    """Simplify a CM / DCM by threshold-unit reduction (fixed point).

    Safe rewrite rules applied until no candidate remains:

    1. **Never** remove or rewrite ``tau=0`` thresholds (constant schemata).
    2. **Threshold-unit reduction:** if a threshold/fusion unit is
       LINEAR / AND / OR / OTHER and every successor class is compatible,
       wire all predecessors to all successors, bump each threshold/fusion
       successor by ``tau(T)-1``, and delete ``T``. Variables are never
       bumped. Compatibility:

       - LINEAR → AND, OR, LINEAR, OTHER
       - AND → AND, LINEAR
       - OR → OR, LINEAR
       - OTHER → LINEAR

       Non-schema OR/LINEAR units that claim variables are eligible.

    Candidates are ordered **farthest-from-out first** (largest directed
    distance to a ``variable`` node), then by ``str(node_id)``. One rewrite
    is applied per iteration.

    Parameters
    ----------
    cm : networkx.DiGraph
        Canalizing map or dynamics canalization map.
    copy : bool, optional
        If ``True`` (default), operate on a copy and leave *cm* unchanged.
    remove_isolates : bool, optional
        If ``True`` (default), drop isolated nodes after rewriting.
    skip_if_edges_increase : bool, optional
        If ``True``, skip a reduction when wiring preds→succs would add more
        edges than removing the unit deletes. Reductions that leave the edge
        count unchanged are still applied. Default ``False``.

    Returns
    -------
    networkx.DiGraph
        Simplified graph (a copy unless ``copy=False``).
    """
    G = cm.copy() if copy else cm

    while (
        _simplify_one_step(G, skip_if_edges_increase=skip_if_edges_increase)
        is not None
    ):
        pass

    if remove_isolates:
        G.remove_nodes_from(list(nx.isolates(G)))

    return G
