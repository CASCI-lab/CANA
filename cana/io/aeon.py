# -*- coding: utf-8 -*-
"""AEON ``*.aeon`` (regulations + ``$var: expr``) import and export."""
from __future__ import annotations

import re
from typing import Dict, List, Sequence

from cana.boolean_node import BooleanNode
from cana.function_outputs.expression import (
    expression_identifiers,
    function_outputs_to_expression,
    normalize_operators,
)
from cana.nodes_builder import expressions_to_nodes, input_names

AEON_EDGE_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(-[>|?]?|\|)\s*([A-Za-z_][A-Za-z0-9_]*)\s*$"
)
AEON_UPDATE_RE = re.compile(
    r"^\s*\$([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+?)\s*$"
)


def parse_aeon(text: str) -> List[BooleanNode]:
    """Parse AEON ``*.aeon`` into a node list."""
    expressions: Dict[str, str] = {}
    names: set = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m_upd = AEON_UPDATE_RE.match(line)
        if m_upd:
            name, expr = m_upd.group(1), m_upd.group(2)
            expressions[name] = expr
            names.add(name)
            names.update(
                expression_identifiers(normalize_operators(expr, "python"))
            )
            continue
        m_edge = AEON_EDGE_RE.match(line)
        if m_edge:
            names.add(m_edge.group(1))
            names.add(m_edge.group(3))
            continue
    if not expressions and not names:
        raise ValueError("No aeon rules or regulations found")
    return expressions_to_nodes(expressions, names)


def nodes_to_aeon(
    nodes: Sequence[BooleanNode],
    *,
    warn: bool = True,
) -> str:
    """Export a node list to AEON text (regulations + update functions).

    Prefers stored expression, then one-symbol schemata, then LUT DNF.
    Soft-warns when verification flags are not ``True`` (never blocks).
    """
    lines: List[str] = []
    for node in nodes:
        for src_id in node.inputs:
            src = nodes[src_id].name
            lines.append("{0} -? {1}".format(src, node.name))
    for i, node in enumerate(nodes):
        names = input_names(nodes, i)
        expr = function_outputs_to_expression(
            node.outputs,
            names,
            style="symbolic",
            node_label=node.name,
            warn=warn,
        )
        lines.append("${0}: {1}".format(node.name, expr))
    return "\n".join(lines) + "\n"
