# -*- coding: utf-8 -*-
"""
Build Canalizing Map (CM) graphs from two-symbol schemata.

A canalizing map is a directed ``networkx.DiGraph`` that encodes how input
literals (and permutation-symmetric groups of them) force a Boolean node's
output.  Schemata come from two-symbol canalization; each schema becomes a
threshold unit that claims one output polarity when enough of its inputs fire.

Graph vocabulary (legacy CANA naming, kept for drawing / LUT round-trips)
-------------------------------------------------------------------------
- ``variable`` nodes ``var-{id}-out-{0|1}`` — an input or output at a fixed
  polarity.
- ``threshold`` nodes — fire when at least ``tau`` predecessors are active.
- ``fusion`` nodes — OR of predecessors (used when a symmetry fibre needs
  only one matching literal).

Callers may select a builder by name via :data:`CM_BUILDERS` or pass a
callable with the same signature as :func:`build_canalizing_map_legacy`.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import networkx as nx

# One two-symbol schema: (schema_string, permutation_groups, same_symbol_groups).
# ``same_symbol_groups`` is part of the canalization payload but unused by the
# legacy CM wiring below.
SchemaEntry = Tuple[str, Sequence[Sequence[int]], Sequence]
# ``(schemata_for_output_0, schemata_for_output_1)``
TwoSymbols = Tuple[Sequence[SchemaEntry], Sequence[SchemaEntry]]

BuilderFn = Callable[..., nx.DiGraph]

# One polarity fibre inside a permutation group:
# (group_index, member_positions, required_polarity, count_needed)
PolarityFibre = Tuple[int, Sequence[int], int, int]


# ---------------------------------------------------------------------------
# Stable node-id helpers (string formats are part of the public CM contract)
# ---------------------------------------------------------------------------

def _variable_node_id(var_id: int, polarity: int) -> str:
    return "var-{nid:d}-out-{out:d}".format(nid=var_id, out=polarity)


def _schema_threshold_id(threshold_index: int, node_id: int, output: int) -> str:
    return "thr-{tid:d}-var-{nid:d}-out-{out:d}".format(
        tid=threshold_index, nid=node_id, out=output
    )


def _fusion_node_id(
    group_index: int, threshold_index: int, node_id: int, polarity: int
) -> str:
    # Legacy quirk: the trailing ``out-`` field stores the fibre polarity,
    # not the schema output polarity.
    return (
        "fus-0-grp-{g:d}-thr-{thr:d}-var-{nid:d}-out-{out:d}"
    ).format(g=group_index, thr=threshold_index, nid=node_id, out=polarity)


def _fibre_threshold_id(
    threshold_index: int,
    group_index: int,
    polarity: int,
    node_id: int,
    output: int,
) -> str:
    return (
        "thr-{tid:d}-grp-{g:d}-pol-{p:d}-var-{nid:d}-out-{out:d}"
    ).format(
        tid=threshold_index,
        g=group_index,
        p=polarity,
        nid=node_id,
        out=output,
    )


# ---------------------------------------------------------------------------
# Graph mutators
# ---------------------------------------------------------------------------

def _add_output_variable(
    graph: nx.DiGraph,
    *,
    node_id: int,
    name: str,
    polarity: int,
    constant: bool,
) -> None:
    graph.add_node(
        _variable_node_id(node_id, polarity),
        label=name,
        type="variable",
        mode="output",
        value=polarity,
        constant=constant,
        group=node_id,
    )


def _ensure_input_literal(
    graph: nx.DiGraph,
    *,
    input_id: int,
    polarity: int,
    label: str,
    owner_node_id: int,
) -> str:
    """Return the variable-node id for an input literal, creating it if needed."""
    node_name = _variable_node_id(input_id, polarity)
    if node_name not in graph:
        graph.add_node(
            node_name,
            label=label,
            type="variable",
            mode="input",
            value=polarity,
            group=owner_node_id,
        )
    return node_name


def _positions_outside_permutations(
    schema: str,
    permutation_groups: Sequence[Sequence[int]],
    k: int,
) -> List[int]:
    """Input indices fixed to 0/1 and not covered by any permutation group."""
    in_a_group = {j for group in permutation_groups for j in group}
    return [
        j
        for j in range(k)
        if j not in in_a_group and schema[j] in ("0", "1")
    ]


def _polarity_fibres(
    schema: str,
    permutation_groups: Sequence[Sequence[int]],
) -> List[PolarityFibre]:
    """Non-empty (group, polarity) fibres implied by a schema string."""
    fibres: List[PolarityFibre] = []
    for group_index, group in enumerate(permutation_groups):
        for polarity in (0, 1):
            count = sum(1 for j in group if schema[j] == str(polarity))
            if count > 0:
                fibres.append((group_index, group, polarity, count))
    return fibres


def _wire_schema_unit(
    graph: nx.DiGraph,
    *,
    schema: str,
    permutation_groups: Sequence[Sequence[int]],
    output: int,
    threshold_index: int,
    node_id: int,
    inputs: Sequence[int],
    input_labels: Mapping[int, str],
    k: int,
) -> None:
    """Add one schema's threshold unit and its literal / symmetry inputs."""
    fixed_positions = _positions_outside_permutations(
        schema, permutation_groups, k
    )
    fibres = _polarity_fibres(schema, permutation_groups)
    tau = len(fixed_positions) + len(fibres)

    threshold_name = _schema_threshold_id(threshold_index, node_id, output)
    graph.add_node(
        threshold_name,
        label="{:d}".format(tau),
        type="threshold",
        tau=tau,
        group=node_id,
    )

    output_name = _variable_node_id(node_id, output)
    if not (graph.has_node(threshold_name) and graph.has_node(output_name)):
        raise RuntimeError(
            "Cannot wire schema threshold to missing output variable node."
        )
    graph.add_edge(threshold_name, output_name, type="out")

    # Fixed (non-permutable) literals feed the schema threshold directly.
    for position in fixed_positions:
        input_id = inputs[position]
        polarity = int(schema[position])
        literal_name = _ensure_input_literal(
            graph,
            input_id=input_id,
            polarity=polarity,
            label=input_labels.get(input_id, str(input_id)),
            owner_node_id=node_id,
        )
        graph.add_edge(literal_name, threshold_name, type="literal")

    # Each polarity fibre is either a fusion (need 1) or a nested threshold.
    for group_index, group, polarity, count in fibres:
        if count == 1:
            fibre_name = _fusion_node_id(
                group_index, threshold_index, node_id, polarity
            )
            graph.add_node(fibre_name, type="fusion", group=node_id)
            incoming_edge_type = "fusing"
        else:
            fibre_name = _fibre_threshold_id(
                threshold_index, group_index, polarity, node_id, output
            )
            graph.add_node(
                fibre_name,
                label="{:d}".format(count),
                type="threshold",
                tau=count,
                group=node_id,
            )
            incoming_edge_type = "literal"

        for position in group:
            input_id = inputs[position]
            literal_name = _ensure_input_literal(
                graph,
                input_id=input_id,
                polarity=polarity,
                label=input_labels.get(input_id, str(input_id)),
                owner_node_id=node_id,
            )
            graph.add_edge(literal_name, fibre_name, type=incoming_edge_type)

        graph.add_edge(fibre_name, threshold_name, type="fused")


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

def build_canalizing_map_legacy(
    two_symbols: TwoSymbols,
    *,
    node_id: int,
    inputs: Sequence[int],
    name: str,
    constant: bool = False,
    output: Optional[int] = None,
    input_labels: Optional[Mapping[int, str]] = None,
    k: Optional[int] = None,
) -> nx.DiGraph:
    """Build a CM graph from two-symbol schemata (legacy CANA wiring).

    Parameters
    ----------
    two_symbols : tuple
        ``(schemata_for_0, schemata_for_1)`` as produced by two-symbol
        canalization.  Each schema entry is
        ``(schema_str, permutation_groups, same_symbol_groups)``.
    node_id : int
        Variable id for output nodes ``var-{node_id}-out-{0|1}``.
    inputs : sequence of int
        Input variable ids in LUT order (length ``k``).
    name : str
        Output variable label.
    constant : bool, optional
        Stored on output variable nodes.
    output : int or None, optional
        If ``0`` or ``1``, only that polarity is built; ``None`` builds both.
    input_labels : mapping, optional
        Map from input variable id to display label.  Defaults to ``str(id)``.
    k : int, optional
        Number of inputs.  Defaults to ``len(inputs)``.

    Returns
    -------
    networkx.DiGraph
        Canalizing map graph.
    """
    if k is None:
        k = len(inputs)
    labels = input_labels if input_labels is not None else {}

    graph = nx.DiGraph(name="CM: %s" % name)

    if output is None or output == 0:
        _add_output_variable(
            graph, node_id=node_id, name=name, polarity=0, constant=constant
        )
    if output is None or output == 1:
        _add_output_variable(
            graph, node_id=node_id, name=name, polarity=1, constant=constant
        )

    threshold_index = 0
    for output_polarity, schemata in zip((0, 1), two_symbols):
        if not schemata:
            continue
        if output is not None and output != output_polarity:
            continue

        for schema, permutation_groups, _same_symbol_groups in schemata:
            _wire_schema_unit(
                graph,
                schema=schema,
                permutation_groups=permutation_groups,
                output=output_polarity,
                threshold_index=threshold_index,
                node_id=node_id,
                inputs=inputs,
                input_labels=labels,
                k=k,
            )
            threshold_index += 1

    return graph


CM_BUILDERS: Dict[str, BuilderFn] = {
    "legacy": build_canalizing_map_legacy,
}
"""Named CM builders.  Register future methods here (e.g. ``canonical``)."""


def build_canalizing_map(
    two_symbols: TwoSymbols,
    *,
    node_id: int,
    inputs: Sequence[int],
    name: str,
    constant: bool = False,
    output: Optional[int] = None,
    input_labels: Optional[Mapping[int, str]] = None,
    k: Optional[int] = None,
    builder: Union[str, BuilderFn] = "legacy",
) -> nx.DiGraph:
    """Build a canalizing map using a named or callable builder.

    Parameters
    ----------
    builder : str or callable, optional
        Registry key in :data:`CM_BUILDERS` (default ``'legacy'``) or a
        callable with the same signature as :func:`build_canalizing_map_legacy`.

    Other parameters
    ----------------
    Same as :func:`build_canalizing_map_legacy`.

    Returns
    -------
    networkx.DiGraph
        Canalizing map graph.

    Raises
    ------
    KeyError
        If ``builder`` is a string not present in :data:`CM_BUILDERS`.
    TypeError
        If ``builder`` is neither a string nor a callable.
    """
    if isinstance(builder, str):
        try:
            build_fn = CM_BUILDERS[builder]
        except KeyError:
            raise KeyError(
                "Unknown CM builder {!r}; known: {}".format(
                    builder, sorted(CM_BUILDERS)
                )
            )
    elif callable(builder):
        build_fn = builder
    else:
        raise TypeError(
            "builder must be a str or callable, got {!r}".format(type(builder))
        )

    return build_fn(
        two_symbols,
        node_id=node_id,
        inputs=inputs,
        name=name,
        constant=constant,
        output=output,
        input_labels=input_labels,
        k=k,
    )
