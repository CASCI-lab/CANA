# -*- coding: utf-8 -*-
"""BoolNet / PyBoolNet ``*.bnet`` (``targets,factors``) import and export."""
from __future__ import annotations

from typing import Dict, List, Sequence

from cana.boolean_node import BooleanNode
from cana.function_outputs.expression import (
    expression_identifiers,
    function_outputs_to_expression,
    normalize_operators,
)
from cana.nodes_builder import expressions_to_nodes, input_names


def parse_bnet(text: str) -> List[BooleanNode]:
    """Parse BoolNet / PyBoolNet ``*.bnet`` into a node list."""
    expressions: Dict[str, str] = {}
    extras: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("targets") and "," in line:
            continue
        if "," not in line:
            continue
        name, expr = line.split(",", 1)
        name = name.strip()
        expr = expr.strip()
        if not name:
            continue
        expressions[name] = expr
        extras.extend(
            expression_identifiers(normalize_operators(expr, "python"))
        )
    if not expressions:
        raise ValueError("No bnet rules found")
    return expressions_to_nodes(expressions, extras)


def nodes_to_bnet(
    nodes: Sequence[BooleanNode],
    *,
    warn: bool = True,
) -> str:
    """Export a node list to BoolNet ``targets,factors`` text.

    Prefers stored expression, then one-symbol schemata, then LUT DNF.
    Soft-warns when verification flags are not ``True`` (never blocks).
    """
    lines = ["targets,factors"]
    for i, node in enumerate(nodes):
        names = input_names(nodes, i)
        expr = function_outputs_to_expression(
            node.outputs,
            names,
            style="symbolic",
            node_label=node.name,
            warn=warn,
        )
        lines.append("{0}, {1}".format(node.name, expr))
    return "\n".join(lines) + "\n"
