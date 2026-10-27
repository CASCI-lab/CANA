# -*- coding: utf-8 -*-
"""BooleanNet / ``*.booleannet.txt`` (``name* = expr``) import and export."""
from __future__ import annotations

from typing import Dict, List, Sequence

from cana.boolean_node import BooleanNode
from cana.function_outputs.expression import (
    expression_identifiers,
    function_outputs_to_expression,
    normalize_operators,
)
from cana.nodes_builder import expressions_to_nodes, input_names


def parse_booleannet(text: str) -> List[BooleanNode]:
    """Parse BooleanNet / ``*.booleannet.txt`` rules into a node list."""
    expressions: Dict[str, str] = {}
    extras: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "*" not in line or "=" not in line:
            continue
        name = line.split("*", 1)[0].strip()
        expr = line.split("=", 1)[1].strip()
        if not name:
            continue
        expressions[name] = expr
        extras.extend(
            expression_identifiers(normalize_operators(expr, "python"))
        )
    if not expressions:
        raise ValueError("No booleannet rules found")
    return expressions_to_nodes(expressions, extras)


def nodes_to_booleannet(
    nodes: Sequence[BooleanNode],
    *,
    warn: bool = True,
) -> str:
    """Export a node list to BooleanNet text (``name* = expr``).

    Prefers stored expression, then one-symbol schemata, then LUT DNF.
    Soft-warns when verification flags are not ``True`` (never blocks).
    """
    lines: List[str] = []
    for i, node in enumerate(nodes):
        names = input_names(nodes, i)
        expr = function_outputs_to_expression(
            node.outputs,
            names,
            style="python",
            node_label=node.name,
            warn=warn,
        )
        lines.append("{0}* = {1}".format(node.name, expr))
    return "\n".join(lines) + "\n"
