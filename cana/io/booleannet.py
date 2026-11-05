# -*- coding: utf-8 -*-
"""BooleanNet / ``*.booleannet.txt`` (``name* = expr``) import and export."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from cana.boolean_node import BooleanNode
from cana.function_outputs.expression import (
    expression_identifiers,
    function_outputs_to_named_expression,
    normalize_operators,
)
from cana.nodes_builder import input_names, update_rules_to_nodes


def parse_booleannet_update_rule(line: str) -> Optional[Tuple[str, str]]:
    """Parse one ``name* = named_expression`` line, or ``None`` if not a rule."""
    raw = line.strip()
    if not raw or raw.startswith("#"):
        return None
    if "*" not in raw or "=" not in raw:
        return None
    name = raw.split("*", 1)[0].strip()
    expr = raw.split("=", 1)[1].strip()
    if not name:
        return None
    return name, expr


def parse_booleannet(text: str) -> List[BooleanNode]:
    """Parse BooleanNet / ``*.booleannet.txt`` rules into a node list."""
    update_rules: Dict[str, str] = {}
    extras: List[str] = []
    for raw in text.splitlines():
        rule = parse_booleannet_update_rule(raw)
        if rule is None:
            continue
        name, expr = rule
        update_rules[name] = expr
        extras.extend(
            expression_identifiers(normalize_operators(expr, "python"))
        )
    if not update_rules:
        raise ValueError("No booleannet rules found")
    return update_rules_to_nodes(update_rules, extras)


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
        expr = function_outputs_to_named_expression(
            node.outputs,
            names,
            style="python",
            node_label=node.name,
            warn=warn,
        )
        lines.append("{0}* = {1}".format(node.name, expr))
    return "\n".join(lines) + "\n"
