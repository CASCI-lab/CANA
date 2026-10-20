# -*- coding: utf-8 -*-
"""
Reconstruct a look-up table (LUT) from a Canalizing Map (CM) graph.

This module interprets a CM as a ``networkx.DiGraph`` of typed nodes.
Evaluation is **modular and topology-based**: each node activates from its
predecessors according to its node kind.  Edge ``type`` attributes (if
present) are ignored for scoring — they may still be used for drawing or
legacy CANA compatibility.

Notes
-----
A CM encodes schemata as McCulloch–Pitts-style *threshold* units, optionally
nested to express two-symbol / permutation structure.

**Node kinds**

- ``variable`` + ``mode=input`` : active iff the assignment matches ``value``.
- ``variable`` + ``mode=output`` : claims its polarity if any predecessor is
  active.
- ``threshold`` : active iff ``(# active predecessors) >= tau``.
  Optional annotation ``role`` (see :data:`THRESHOLD_ROLES`) describes the
  canalization layer; it does not change evaluation.
- ``fusion`` : active iff **any** predecessor is active (logical OR).
  Equivalent to a threshold with ``tau=1``; kept for legacy / drawing.

**Firing / claim rule**

.. code-block:: text

    score(thr) = # active predecessors
    fire       <=>  score >= tau
    claim(out) <=>  any predecessor of the output variable is active

**Per input assignment**, collect claimed output polarities:

- ``{0}`` or ``{1}`` → that LUT entry.
- ``{0, 1}`` → **conflict** at that row.
- ``{}`` → **uncovered** (incomplete).

Inputs that never appear in the CM are free variables over the full ``2^k``
table.  Input order is taken from ``input_order`` if given, otherwise from
sorted input ids discovered in the graph.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import networkx as nx

# Node name pattern used by BooleanNode.canalizing_map
_VAR_NODE_RE = re.compile(r"^var-(\d+)-out-([01])$")

# ---------------------------------------------------------------------------
# Threshold role annotations (documentation / drawing only)
# ---------------------------------------------------------------------------

ROLE_SCHEMA = "schema"
"""Two-symbol schema unit (typically a root that feeds an output)."""

ROLE_SYMMETRY = "symmetry"
"""Permutation-symmetry aggregate over one or more polarity groups."""

ROLE_POLARITY0 = "polarity0"
"""0-polarity fibre within a permutation symmetry."""

ROLE_POLARITY1 = "polarity1"
"""1-polarity fibre within a permutation symmetry."""

THRESHOLD_ROLES = frozenset(
    {ROLE_SCHEMA, ROLE_SYMMETRY, ROLE_POLARITY0, ROLE_POLARITY1}
)
"""Allowed values for :func:`make_threshold`'s ``role`` argument."""


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


def node_active(
    cm: nx.DiGraph,
    node_id: Any,
    assignment: Dict[Union[int, str], int],
    memo: Optional[Dict[Any, Optional[bool]]] = None,
) -> bool:
    """Return whether a CM node is active under an assignment.

    Activation is defined by node kind and predecessors only (edge types are
    ignored):

    - input variable: assignment matches ``value``
    - output variable: any predecessor is active
    - fusion: any predecessor is active
    - threshold: number of active predecessors ``>= tau``

    Parameters
    ----------
    cm : networkx.DiGraph
        Canalizing map graph.
    node_id : hashable
        Node to evaluate.
    assignment : dict
        Mapping from input key to ``0`` or ``1``.
    memo : dict, optional
        Per-assignment cache.  Values are ``True``/``False`` when resolved,
        or ``None`` while a node is on the recursion stack (cycle guard).

    Returns
    -------
    bool
        ``True`` if the node is active.

    Raises
    ------
    ValueError
        If evaluating ``node_id`` encounters a directed cycle, or if
        ``node_id`` has an unknown ``type``.
    """
    if memo is None:
        memo = {}
    if node_id in memo:
        cached = memo[node_id]
        if cached is None:
            raise ValueError(
                "Cycle detected while evaluating CM node {!r}".format(node_id)
            )
        return cached

    memo[node_id] = None  # in-progress marker
    data = cm.nodes[node_id]
    ntype = data.get("type")

    if ntype == "variable":
        mode = data.get("mode")
        if mode == "input":
            result = _assignment_activates_var(cm, node_id, assignment)
        elif mode == "output":
            result = any(
                node_active(cm, pred, assignment, memo)
                for pred in cm.predecessors(node_id)
            )
        else:
            raise ValueError(
                "Variable node {!r} has unknown mode {!r}".format(node_id, mode)
            )
    elif ntype == "fusion":
        result = any(
            node_active(cm, pred, assignment, memo)
            for pred in cm.predecessors(node_id)
        )
    elif ntype == "threshold":
        score = sum(
            1
            for pred in cm.predecessors(node_id)
            if node_active(cm, pred, assignment, memo)
        )
        result = score >= int(data["tau"])
    else:
        raise ValueError(
            "Unknown CM node type {!r} on node {!r}".format(ntype, node_id)
        )

    memo[node_id] = result
    return result


def fusion_active(
    cm: nx.DiGraph, fusion_id: Any, assignment: Dict[Union[int, str], int]
) -> bool:
    """Return whether a fusion node is active under an assignment.

    A fusion is active iff **any** of its predecessors is active (logical OR).

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
        ``True`` if at least one predecessor is active.

    Raises
    ------
    ValueError
        If ``fusion_id`` is not a fusion node.
    """
    data = cm.nodes[fusion_id]
    if data.get("type") != "fusion":
        raise ValueError("Node {!r} is not a fusion".format(fusion_id))
    return node_active(cm, fusion_id, assignment)


def tunit_score(
    cm: nx.DiGraph, thr_id: Any, assignment: Dict[Union[int, str], int]
) -> int:
    """Compute the score of a threshold unit under an assignment.

    The score is the number of active predecessors (inputs, fusions, or nested
    thresholds).

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

    Raises
    ------
    ValueError
        If ``thr_id`` is not a threshold node.
    """
    data = cm.nodes[thr_id]
    if data.get("type") != "threshold":
        raise ValueError("Node {!r} is not a threshold t-unit".format(thr_id))
    memo: Dict[Any, Optional[bool]] = {}
    return sum(
        1
        for pred in cm.predecessors(thr_id)
        if node_active(cm, pred, assignment, memo)
    )


def tunit_fires(
    cm: nx.DiGraph, thr_id: Any, assignment: Dict[Union[int, str], int]
) -> bool:
    """Return whether a threshold unit fires under an assignment.

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
    return node_active(cm, thr_id, assignment)


def claimed_outputs(
    cm: nx.DiGraph, assignment: Dict[Union[int, str], int]
) -> Set[int]:
    """Return output polarities claimed under an assignment.

    An output variable claims its polarity when any of its predecessors is
    active (typically a firing schema-level threshold).

    Parameters
    ----------
    cm : networkx.DiGraph
        Canalizing map graph.
    assignment : dict
        Mapping from input key to ``0`` or ``1``.

    Returns
    -------
    set of int
        Subset of ``{0, 1}``.  Empty if nothing claims; both values if there
        is a conflict.
    """
    claimed: Set[int] = set()
    memo: Dict[Any, Optional[bool]] = {}
    for node_id, data in iter_nodes_of_type(cm, "variable"):
        if data.get("mode") != "output":
            continue
        for pred in cm.predecessors(node_id):
            if node_active(cm, pred, assignment, memo):
                claimed.add(int(data["value"]))
                break
    return claimed


def lut_from_canalizing_map(
    cm: nx.DiGraph,
    input_order: Optional[Sequence[Union[int, str]]] = None,
) -> CMLutResult:
    """Build a (possibly partial or conflicting) LUT from a canalizing map.

    Parameters
    ----------
    cm : networkx.DiGraph
        Canalizing map as a directed graph of typed nodes.  Evaluation uses
        topology and node kinds only (see module docstring).
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
#
# Construction is label-driven on a plain ``nx.DiGraph``.  Bookkeeping lives
# in ``G.graph``; callers never assign ``var_id``, ``tid``, or edge types.


def _ensure_registry(G: nx.DiGraph) -> None:
    """Initialize CM construction bookkeeping on ``G.graph`` if needed."""
    g = G.graph
    g.setdefault("input_labels", [])
    g.setdefault("input_ids", {})
    g.setdefault("output_label", None)
    g.setdefault("output_id", None)
    g.setdefault("next_var_id", 0)
    g.setdefault("next_tid", 0)
    g.setdefault("next_fus", 0)


def _alloc_var_id(G: nx.DiGraph) -> int:
    """Allocate the next shared variable id (inputs and output)."""
    _ensure_registry(G)
    vid = int(G.graph["next_var_id"])
    G.graph["next_var_id"] = vid + 1
    return vid


def _input_id_for(G: nx.DiGraph, label: str) -> int:
    """Return the numeric id for an input label, allocating if new."""
    _ensure_registry(G)
    ids = G.graph["input_ids"]
    if label not in ids:
        vid = _alloc_var_id(G)
        ids[label] = vid
        G.graph["input_labels"].append(label)
    return int(ids[label])


def input_ids_in_order(G: nx.DiGraph) -> List[int]:
    """Return input variable ids in first-seen label order.

    Suitable as ``input_order`` for :func:`lut_from_canalizing_map`.
    """
    _ensure_registry(G)
    ids = G.graph["input_ids"]
    return [int(ids[lab]) for lab in G.graph["input_labels"]]


def _validate_bit(value: int, name: str = "value") -> int:
    """Coerce and validate a 0/1 polarity."""
    v = int(value)
    if v not in (0, 1):
        raise ValueError("{} must be 0 or 1, got {!r}".format(name, value))
    return v


def _node_kind(G: nx.DiGraph, node_id: Any) -> str:
    """Return a construction kind: ``input``, ``output``, ``threshold``, ``fusion``."""
    if node_id not in G:
        raise ValueError("Unknown node {!r}".format(node_id))
    data = G.nodes[node_id]
    ntype = data.get("type")
    if ntype == "variable":
        mode = data.get("mode")
        if mode == "input":
            return "input"
        if mode == "output":
            return "output"
        raise ValueError(
            "Variable node {!r} has unknown mode {!r}".format(node_id, mode)
        )
    if ntype in ("threshold", "fusion"):
        return ntype
    raise ValueError(
        "Unsupported node type {!r} on {!r}".format(ntype, node_id)
    )


_CONNECT_EDGE_TYPES = {
    ("input", "threshold"): "literal",
    ("input", "fusion"): "fusing",
    ("fusion", "threshold"): "fused",
    ("threshold", "output"): "out",
    ("threshold", "threshold"): "feed",
}


def connect(G: nx.DiGraph, src: Any, dst: Any) -> None:
    """Add an edge, inferring ``type`` from endpoint node kinds.

    Does not mutate threshold attributes (no ``out`` stamping).

    Parameters
    ----------
    G : networkx.DiGraph
        CM graph.
    src, dst : hashable
        Existing node ids.

    Raises
    ------
    ValueError
        If the ``(src_kind, dst_kind)`` pair is not allowed.
    """
    sk = _node_kind(G, src)
    dk = _node_kind(G, dst)
    etype = _CONNECT_EDGE_TYPES.get((sk, dk))
    if etype is None:
        raise ValueError(
            "Cannot connect {} ({}) → {} ({})".format(src, sk, dst, dk)
        )
    G.add_edge(src, dst, type=etype)


def make_input_node(G: nx.DiGraph, label: str, value: int) -> str:
    """Add an input variable node; ``label`` determines the shared ``var_id``.

    Parameters
    ----------
    G : networkx.DiGraph
        Graph to mutate.
    label : str
        Input name.  New labels get a fresh id; existing labels reuse theirs.
    value : int
        Polarity (``0`` or ``1``).

    Returns
    -------
    str
        Node name ``var-{id}-out-{value}``.

    Raises
    ------
    ValueError
        If this ``(label, value)`` pair already exists, or ``value`` is not 0/1.
    """
    value = _validate_bit(value)
    var_id = _input_id_for(G, label)
    name = "var-{nid:d}-out-{out:d}".format(nid=var_id, out=value)
    if name in G:
        raise ValueError(
            "Input {!r} with value {} already exists as {!r}".format(
                label, value, name
            )
        )
    G.add_node(
        name,
        label=label,
        type="variable",
        mode="input",
        value=value,
        group=0,
    )
    return name


def make_output_node(G: nx.DiGraph, label: str, value: int) -> str:
    """Add an output variable node for the CM's single function label.

    Parameters
    ----------
    G : networkx.DiGraph
        Graph to mutate.
    label : str
        Function / output name.  Both polarities share one allocated id.
    value : int
        Output polarity (``0`` or ``1``).

    Returns
    -------
    str
        Node name ``var-{output_id}-out-{value}``.

    Raises
    ------
    ValueError
        If a different output label is already registered, if this polarity
        already exists, or if ``value`` is not 0/1.
    """
    value = _validate_bit(value)
    _ensure_registry(G)
    existing = G.graph["output_label"]
    if existing is None:
        G.graph["output_label"] = label
        G.graph["output_id"] = _alloc_var_id(G)
    elif existing != label:
        raise ValueError(
            "Graph already has output label {!r}; cannot add {!r}".format(
                existing, label
            )
        )
    node_id = int(G.graph["output_id"])
    name = "var-{nid:d}-out-{out:d}".format(nid=node_id, out=value)
    if name in G:
        raise ValueError(
            "Output {!r} with value {} already exists as {!r}".format(
                label, value, name
            )
        )
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
    G: nx.DiGraph,
    tau: int,
    role: str = ROLE_SCHEMA,
) -> str:
    """Add a threshold (t-unit) node; ``tid`` is assigned automatically.

    Thresholds do **not** store an ``out`` polarity — claims come only from
    edges into output variables, so build order does not matter.

    Parameters
    ----------
    G : networkx.DiGraph
        Graph to mutate.
    tau : int
        Firing threshold.
    role : str, optional
        Canalization-layer annotation (does not affect evaluation).  One of
        :data:`THRESHOLD_ROLES`.  Default is ``'schema'``.

    Returns
    -------
    str
        Node name ``thr-{tid}``.

    Raises
    ------
    ValueError
        If ``role`` is not in :data:`THRESHOLD_ROLES`.
    """
    if role not in THRESHOLD_ROLES:
        raise ValueError(
            "Unknown threshold role {!r}; expected one of {}".format(
                role, sorted(THRESHOLD_ROLES)
            )
        )
    _ensure_registry(G)
    tid = int(G.graph["next_tid"])
    G.graph["next_tid"] = tid + 1
    out_id = G.graph["output_id"]
    group = int(out_id) if out_id is not None else 0
    name = "thr-{tid:d}".format(tid=tid)
    G.add_node(
        name,
        label=str(int(tau)),
        type="threshold",
        tau=int(tau),
        role=role,
        group=group,
    )
    return name


def make_fusion(G: nx.DiGraph) -> str:
    """Add a fusion junction node; index is assigned automatically.

    Fusion is legacy OR sugar (equivalent to a ``tau=1`` threshold over
    inputs).  Prefer nested thresholds for new maps.

    Parameters
    ----------
    G : networkx.DiGraph
        Graph to mutate.

    Returns
    -------
    str
        Node name ``fus-{index}``.
    """
    _ensure_registry(G)
    fus = int(G.graph["next_fus"])
    G.graph["next_fus"] = fus + 1
    out_id = G.graph["output_id"]
    group = int(out_id) if out_id is not None else 0
    name = "fus-{fus:d}".format(fus=fus)
    G.add_node(name, type="fusion", group=group)
    return name
