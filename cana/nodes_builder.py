# -*- coding: utf-8 -*-
"""Build and compare CANA networks as ``list[BooleanNode]`` hubs.

Invariant: for all ``i``, ``nodes[i].id == i``, and every ``inputs`` entry
is an index into that list.  Node sets are dense ``0 .. N-1``; CANA does
not delete BN nodes after construction.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from cana.bit_lut import BitLut
from cana.boolean_node import BooleanNode
from cana.function_outputs import FunctionOutputs
from cana.function_outputs.expression import (
    expression_identifiers,
    normalize_operators,
    regulator_names_in_expression,
    to_positional,
)


def expressions_to_nodes(
    expressions: Mapping[str, str],
    extra_names: Optional[Iterable[str]] = None,
) -> List[BooleanNode]:
    """Build a dense ``list[BooleanNode]`` from ``{target: boolean_expression}``.

    Names that appear only as inputs (free inputs) receive the identity
    update ``f(x) = x``, matching BBM's booleannet representation of inputs.
    Literal constants use empty ``inputs`` and positional ``True``/``False``.

    Each node's ``outputs`` stores a **positional** python expression
    (``x0``…``x{k-1}``) without building a dense LUT at import time.
    """
    # Preserve insertion order of targets; append free inputs alphabetically.
    ordered: List[str] = list(expressions.keys())
    seen = set(ordered)
    mentioned: set = set()
    for expr in expressions.values():
        mentioned.update(
            expression_identifiers(normalize_operators(expr, "python"))
        )
    if extra_names:
        mentioned.update(extra_names)
    for name in sorted(mentioned):
        if name not in seen:
            ordered.append(name)
            seen.add(name)

    name_to_id = {name: i for i, name in enumerate(ordered)}
    completed: Dict[str, str] = dict(expressions)
    for name in ordered:
        if name not in completed:
            completed[name] = name  # identity

    nodes: List[BooleanNode] = []
    for i, name in enumerate(ordered):
        expr = normalize_operators(completed[name], "python")
        if expr in ("True", "true", "1"):
            nodes.append(
                BooleanNode(
                    id=i,
                    name=name,
                    k=0,
                    inputs=[],
                    outputs=FunctionOutputs(expression="True", k=0),
                )
            )
            continue
        if expr in ("False", "false", "0"):
            nodes.append(
                BooleanNode(
                    id=i,
                    name=name,
                    k=0,
                    inputs=[],
                    outputs=FunctionOutputs(expression="False", k=0),
                )
            )
            continue
        input_names = regulator_names_in_expression(expr, ordered)
        if expr.strip() == name and name not in input_names:
            input_names = [name]
        input_ids = [name_to_id[n] for n in input_names]
        positional = to_positional(expr, input_names)
        nodes.append(
            BooleanNode(
                id=i,
                name=name,
                k=len(input_names),
                inputs=input_ids,
                outputs=FunctionOutputs(
                    expression=positional, k=len(input_names)
                ),
            )
        )
    return nodes


def input_names(nodes: Sequence[BooleanNode], node_id: int) -> List[str]:
    """Return input names for ``node_id`` in LUT bit order."""
    return [nodes[j].name for j in nodes[node_id].inputs]


def _as_nodes(network_or_nodes: Any) -> Sequence[BooleanNode]:
    """Accept a :class:`~cana.boolean_network.BooleanNetwork` or node list."""
    if hasattr(network_or_nodes, "nodes") and not isinstance(
        network_or_nodes, (list, tuple)
    ):
        return network_or_nodes.nodes
    return network_or_nodes


def name_keyed_luts(
    network_or_nodes: Any,
) -> Dict[str, Tuple[Tuple[str, ...], Tuple[int, ...]]]:
    """Return ``{name: (sorted_input_names, lut)}`` for equivalence checks.

    Accepts a :class:`~cana.boolean_network.BooleanNetwork` or a
    ``Sequence[BooleanNode]``.  Input order is canonicalized by sorting
    names; the LUT is reordered to match.
    """
    nodes = _as_nodes(network_or_nodes)
    result: Dict[str, Tuple[Tuple[str, ...], Tuple[int, ...]]] = {}
    for node in nodes:
        input_name_list = [nodes[j].name for j in node.inputs]
        sorted_names = tuple(sorted(input_name_list))
        outputs = node.outputs
        if isinstance(outputs, FunctionOutputs):
            lut = outputs.get_bit_lut()
        elif isinstance(outputs, BitLut):
            lut = outputs
        else:
            lut = BitLut.from_sequence(outputs)
        lut = lut.reorder(input_name_list, list(sorted_names))
        result[node.name] = (sorted_names, tuple(int(v) for v in lut))
    return result


def equivalent_networks(a: Any, b: Any) -> bool:
    """True if two networks / node lists have the same name-keyed LUTs."""
    return name_keyed_luts(a) == name_keyed_luts(b)
