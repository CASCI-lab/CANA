# -*- coding: utf-8 -*-
"""
Reconstruct a look-up table (LUT) from a Canalizing Map (CM) graph.

This module interprets a CM strictly as a ``networkx.DiGraph`` with the
node/edge types produced by ``BooleanNode.canalizing_map``.  It does not
call canalization / two-symbol code.

Notes
-----
A CM encodes each schema as a McCulloch–Pitts-style *t-unit*.

**Node kinds**

- ``variable`` + ``mode=input`` : an input at a fixed polarity (``value`` 0/1).
- ``variable`` + ``mode=output`` : the claimed output polarity.
- ``threshold`` : the t-unit itself; attribute ``tau`` is the firing threshold.
- ``fusion`` : active iff **any** of its fusing parents is active (OR).

**Edge kinds**

- ``literal`` : input-var → t-unit; contributes 1 when that input equals its
  ``value``.
- ``fusing`` : input-var → fusion.
- ``fused`` : fusion → t-unit; contributes 1 when the fusion is active.
- ``out`` : t-unit → output-var; if the t-unit fires, claim that output.

**Firing rule**

.. code-block:: text

    score = (# active literal parents) + (# active fusion parents)
    fire  <=>  score >= tau

**Per input assignment**, collect claimed outputs from all firing t-units:

- ``{0}`` or ``{1}`` → that LUT entry.
- ``{0, 1}`` → **conflict** at that row.
- ``{}`` → **uncovered** (incomplete).

Inputs that never appear in the CM are free variables over the full ``2^k``
table (they do not affect any t-unit).  Input order is taken from
``input_order`` if given, otherwise from sorted input ids discovered in the
graph (preferring numeric ids parsed from node names ``var-{id}-out-{val}``,
falling back to labels).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import networkx as nx

# Node name pattern used by BooleanNode.canalizing_map
_VAR_NODE_RE = re.compile(r"^var-(\d+)-out-([01])$")


@dataclass
class CMLutResult:
    """Result of reconstructing a LUT from a canalizing map.

    Attributes
    ----------
    outputs : list of {0, 1, None}
        LUT entries indexed by the integer whose binary representation is the
        input state (MSB = first input).  ``None`` means uncovered or
        conflicted.
    input_names : list of int or str
        Ordered input identifiers used for the LUT (MSB first).
    has_conflict : bool
        ``True`` if at least one input state is claimed by both output
        polarities.
    is_incomplete : bool
        ``True`` if at least one input state is claimed by no t-unit.
    conflict_states : list of str
        Binary state strings (MSB first) that are conflicted.
    uncovered_states : list of str
        Binary state strings (MSB first) that are uncovered.
    """

    outputs: List[Optional[int]]
    input_names: List[Union[int, str]]
    has_conflict: bool
    is_incomplete: bool
    conflict_states: List[str] = field(default_factory=list)
    uncovered_states: List[str] = field(default_factory=list)

    def lut_string(self, uncovered: str = "?", conflict: str = "X") -> str:
        """Return a compact string view of the LUT.

        Parameters
        ----------
        uncovered : str, optional
            Character used for uncovered rows.  Default is ``'?'``.
        conflict : str, optional
            Character used for conflicted rows.  Default is ``'X'``.

        Returns
        -------
        str
            One character per LUT row, length ``2**k``.
        """
        chars = []
        conflict_set = set(self.conflict_states)
        uncovered_set = set(self.uncovered_states)
        for i, out in enumerate(self.outputs):
            state = _index_to_binstate(i, len(self.input_names))
            if state in conflict_set:
                chars.append(conflict)
            elif state in uncovered_set:
                chars.append(uncovered)
            else:
                chars.append(str(out))
        return "".join(chars)


def _index_to_binstate(index: int, k: int) -> str:
    """Convert a LUT row index to a binary state string.

    Parameters
    ----------
    index : int
        Row index in ``[0, 2**k)``.
    k : int
        Number of inputs.

    Returns
    -------
    str
        Binary string of length ``k`` (empty if ``k == 0``).
    """
    if k == 0:
        return ""
    return format(index, "0{}b".format(k))


def _binstate_to_index(binstate: str) -> int:
    """Convert a binary state string to a LUT row index.

    Parameters
    ----------
    binstate : str
        Binary string (MSB first).  Empty string maps to ``0``.

    Returns
    -------
    int
        Integer value of ``binstate``.
    """
    if not binstate:
        return 0
    return int(binstate, 2)


def _parse_var_node(node_id: str) -> Optional[Tuple[int, int]]:
    """Parse a CM variable node name ``var-{id}-out-{val}``.

    Parameters
    ----------
    node_id : str
        Graph node identifier.

    Returns
    -------
    (int, int) or None
        ``(variable_id, value)`` if the name matches the pattern, else
        ``None``.
    """
    m = _VAR_NODE_RE.match(str(node_id))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def iter_nodes_of_type(cm: nx.DiGraph, node_type: str) -> Iterable[Tuple[Any, dict]]:
    """Yield CM nodes whose ``type`` attribute equals ``node_type``.

    Parameters
    ----------
    cm : networkx.DiGraph
        Canalizing map graph.
    node_type : str
        Node type to select (``'variable'``, ``'threshold'``, or
        ``'fusion'``).

    Yields
    ------
    node_id : hashable
        Graph node identifier.
    data : dict
        Node attribute dictionary.
    """
    for n, d in cm.nodes(data=True):
        if d.get("type") == node_type:
            yield n, d


def discover_input_keys(cm: nx.DiGraph) -> List[Union[int, str]]:
    """Discover input identifiers present in the CM.

    Prefers numeric ids parsed from node names ``var-{id}-out-{val}``,
    otherwise uses the node ``label`` (or the node id).  Returned sorted for
    a stable LUT order.

    Parameters
    ----------
    cm : networkx.DiGraph
        Canalizing map graph.

    Returns
    -------
    list of int or str
        Sorted input keys (integers first, then string labels).
    """
    keys: Set[Union[int, str]] = set()
    for n, d in cm.nodes(data=True):
        if d.get("type") != "variable" or d.get("mode") != "input":
            continue
        parsed = _parse_var_node(n)
        if parsed is not None:
            keys.add(parsed[0])
        elif "label" in d:
            keys.add(d["label"])
        else:
            keys.add(n)

    def _sort_key(x):
        if isinstance(x, int):
            return (0, x, "")
        return (1, 0, str(x))

    return sorted(keys, key=_sort_key)


def _assignment_activates_var(
    cm: nx.DiGraph, var_node: Any, assignment: Dict[Union[int, str], int]
) -> bool:
    """Return whether an input variable node is active under an assignment.

    Parameters
    ----------
    cm : networkx.DiGraph
        Canalizing map graph.
    var_node : hashable
        Input variable node id.
    assignment : dict
        Mapping from input key to ``0`` or ``1``.

    Returns
    -------
    bool
        ``True`` if ``assignment[key]`` equals the node's ``value``.
    """
    d = cm.nodes[var_node]
    value = int(d["value"])
    parsed = _parse_var_node(var_node)
    if parsed is not None:
        key: Union[int, str] = parsed[0]
    else:
        key = d.get("label", var_node)
    if key not in assignment:
        return False
    return int(assignment[key]) == value


def fusion_active(
    cm: nx.DiGraph, fusion_id: Any, assignment: Dict[Union[int, str], int]
) -> bool:
    """Return whether a fusion node is active under an assignment.

    A fusion is active iff **any** of its fusing parent variables is active
    (logical OR).

    Parameters
    ----------
    cm : networkx.DiGraph
        Canalizing map graph.
    fusion_id : hashable
        Fusion node id.
    assignment : dict
        Mapping from input key to ``0`` or ``1``.

    Returns
    -------
    bool
        ``True`` if at least one fusing parent is active.
    """
    for pred in cm.predecessors(fusion_id):
        edge = cm.edges[pred, fusion_id]
        if edge.get("type") != "fusing":
            continue
        if _assignment_activates_var(cm, pred, assignment):
            return True
    return False


def tunit_score(
    cm: nx.DiGraph, thr_id: Any, assignment: Dict[Union[int, str], int]
) -> int:
    """Compute the score of a t-unit under an assignment.

    The score is the number of active literal parents plus the number of
    active fusion parents.

    Parameters
    ----------
    cm : networkx.DiGraph
        Canalizing map graph.
    thr_id : hashable
        Threshold (t-unit) node id.
    assignment : dict
        Mapping from input key to ``0`` or ``1``.

    Returns
    -------
    int
        Non-negative contribution count.
    """
    score = 0
    for pred in cm.predecessors(thr_id):
        edge = cm.edges[pred, thr_id]
        etype = edge.get("type")
        if etype == "literal":
            if _assignment_activates_var(cm, pred, assignment):
                score += 1
        elif etype == "fused":
            if fusion_active(cm, pred, assignment):
                score += 1
    return score


def tunit_fires(
    cm: nx.DiGraph, thr_id: Any, assignment: Dict[Union[int, str], int]
) -> bool:
    """Return whether a t-unit fires under an assignment.

    Fires when ``tunit_score(cm, thr_id, assignment) >= tau``.

    Parameters
    ----------
    cm : networkx.DiGraph
        Canalizing map graph.
    thr_id : hashable
        Threshold (t-unit) node id.
    assignment : dict
        Mapping from input key to ``0`` or ``1``.

    Returns
    -------
    bool
        ``True`` if the t-unit fires.

    Raises
    ------
    ValueError
        If ``thr_id`` is not a threshold node.
    """
    data = cm.nodes[thr_id]
    if data.get("type") != "threshold":
        raise ValueError("Node {!r} is not a threshold t-unit".format(thr_id))
    tau = int(data["tau"])
    return tunit_score(cm, thr_id, assignment) >= tau


def claimed_outputs(
    cm: nx.DiGraph, assignment: Dict[Union[int, str], int]
) -> Set[int]:
    """Return output polarities claimed by firing t-units.

    Parameters
    ----------
    cm : networkx.DiGraph
        Canalizing map graph.
    assignment : dict
        Mapping from input key to ``0`` or ``1``.

    Returns
    -------
    set of int
        Subset of ``{0, 1}`` claimed via ``out`` edges from firing t-units.
        Empty if nothing fires; both values if there is a conflict.
    """
    claimed: Set[int] = set()
    for thr_id, d in iter_nodes_of_type(cm, "threshold"):
        if not tunit_fires(cm, thr_id, assignment):
            continue
        for succ in cm.successors(thr_id):
            edge = cm.edges[thr_id, succ]
            if edge.get("type") != "out":
                continue
            out_data = cm.nodes[succ]
            if out_data.get("type") != "variable":
                continue
            claimed.add(int(out_data["value"]))
    return claimed


def lut_from_canalizing_map(
    cm: nx.DiGraph,
    input_order: Optional[Sequence[Union[int, str]]] = None,
) -> CMLutResult:
    """Build a (possibly partial or conflicting) LUT from a canalizing map.

    Parameters
    ----------
    cm : networkx.DiGraph
        Canalizing map as a directed graph with the standard CM node/edge
        types.
    input_order : sequence of int or str, optional
        Explicit input order (MSB first).  If omitted, inputs are discovered
        from the graph and sorted via :func:`discover_input_keys`.

    Returns
    -------
    CMLutResult
        Reconstructed LUT together with conflict / incompleteness flags and
        the lists of affected binary states.

    Notes
    -----
    This function is purely graph-based.  It does not call
    ``BooleanNode.canalizing_map`` or two-symbol canalization code.
    """
    if input_order is None:
        input_names = discover_input_keys(cm)
    else:
        input_names = list(input_order)

    k = len(input_names)
    n = 1 << k if k > 0 else 1
    outputs: List[Optional[int]] = [None] * n
    conflict_states: List[str] = []
    uncovered_states: List[str] = []

    # Enumerate all 2^k assignments in lexicographic / integer order
    if k == 0:
        assignments = [()]
    else:
        assignments = list(product([0, 1], repeat=k))

    for bits in assignments:
        if k == 0:
            assignment: Dict[Union[int, str], int] = {}
            binstate = ""
            index = 0
        else:
            assignment = {input_names[i]: int(bits[i]) for i in range(k)}
            binstate = "".join(str(b) for b in bits)
            index = _binstate_to_index(binstate)

        claimed = claimed_outputs(cm, assignment)
        if len(claimed) == 0:
            outputs[index] = None
            uncovered_states.append(binstate)
        elif len(claimed) == 1:
            outputs[index] = next(iter(claimed))
        else:
            outputs[index] = None
            conflict_states.append(binstate)

    return CMLutResult(
        outputs=outputs,
        input_names=input_names,
        has_conflict=bool(conflict_states),
        is_incomplete=bool(uncovered_states),
        conflict_states=conflict_states,
        uncovered_states=uncovered_states,
    )


# ---------------------------------------------------------------------------
# Hand-built CM helpers (useful for tests and notebooks)
# ---------------------------------------------------------------------------


def make_input_node(
    G: nx.DiGraph, var_id: int, value: int, label: Optional[str] = None
) -> str:
    """Add (or reuse) an input variable node on a CM graph.

    Parameters
    ----------
    G : networkx.DiGraph
        Graph to mutate.
    var_id : int
        Numeric input identifier used in the node name.
    value : int
        Polarity of this variable node (``0`` or ``1``).
    label : str, optional
        Display label.  Defaults to ``str(var_id)``.

    Returns
    -------
    str
        Node name ``var-{var_id}-out-{value}``.
    """
    name = "var-{nid:d}-out-{out:d}".format(nid=var_id, out=value)
    if name not in G:
        G.add_node(
            name,
            label=label if label is not None else str(var_id),
            type="variable",
            mode="input",
            value=value,
            group=0,
        )
    return name


def make_output_node(
    G: nx.DiGraph, node_id: int, value: int, label: str = "x"
) -> str:
    """Add (or reuse) an output variable node on a CM graph.

    Parameters
    ----------
    G : networkx.DiGraph
        Graph to mutate.
    node_id : int
        Numeric id of the function/node whose output this is.
    value : int
        Output polarity (``0`` or ``1``).
    label : str, optional
        Display label.  Default is ``'x'``.

    Returns
    -------
    str
        Node name ``var-{node_id}-out-{value}``.
    """
    name = "var-{nid:d}-out-{out:d}".format(nid=node_id, out=value)
    if name not in G:
        G.add_node(
            name,
            label=label,
            type="variable",
            mode="output",
            value=value,
            constant=False,
            group=node_id,
        )
    return name


def make_threshold(
    G: nx.DiGraph, tid: int, node_id: int, out: int, tau: int
) -> str:
    """Add a threshold (t-unit) node on a CM graph.

    Parameters
    ----------
    G : networkx.DiGraph
        Graph to mutate.
    tid : int
        Threshold index (unique per CM among t-units).
    node_id : int
        Owning function/node id (used in naming / ``group``).
    out : int
        Output polarity this t-unit claims (``0`` or ``1``).
    tau : int
        Firing threshold.

    Returns
    -------
    str
        Node name ``thr-{tid}-var-{node_id}-out-{out}``.
    """
    name = "thr-{tid:d}-var-{nid:d}-out-{out:d}".format(
        tid=tid, nid=node_id, out=out
    )
    G.add_node(name, label=str(tau), type="threshold", tau=tau, group=node_id)
    return name


def make_fusion(
    G: nx.DiGraph, fus: int, tid: int, node_id: int, out: int
) -> str:
    """Add a fusion junction node on a CM graph.

    Parameters
    ----------
    G : networkx.DiGraph
        Graph to mutate.
    fus : int
        Fusion index within the t-unit.
    tid : int
        Parent threshold index.
    node_id : int
        Owning function/node id.
    out : int
        Naming polarity tag (matches CANA CM naming).

    Returns
    -------
    str
        Node name ``fus-{fus}-thr-{tid}-var-{node_id}-out-{out}``.
    """
    name = "fus-{fus:d}-thr-{thr:d}-var-{nid:d}-out-{out:d}".format(
        fus=fus, thr=tid, nid=node_id, out=out
    )
    G.add_node(name, type="fusion", group=node_id)
    return name
