# -*- coding: utf-8 -*-
"""AEON ``*.aeon`` (regulations + ``$var: expr``) import and export."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping

from cana.function_outputs.expression import (
    expression_identifiers,
    function_outputs_to_expression,
    normalize_operators,
)
from cana.network_dict import (
    NetworkDict,
    expressions_to_network_dict,
    input_names_for_node,
    sorted_node_ids,
)

AEON_EDGE_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(-[>|?]?|\|)\s*([A-Za-z_][A-Za-z0-9_]*)\s*$"
)
AEON_UPDATE_RE = re.compile(
    r"^\s*\$([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+?)\s*$"
)


def parse_aeon(text: str) -> NetworkDict:
    """Parse AEON ``*.aeon`` into a ``NetworkDict``."""
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
    return expressions_to_network_dict(expressions, names)


def network_dict_to_aeon(
    network_dict: Mapping[int, Mapping[str, Any]],
    *,
    warn: bool = True,
) -> str:
    """Export ``NetworkDict`` to AEON text (regulations + update functions).

    Prefers stored expression, then one-symbol schemata, then LUT DNF.
    Soft-warns when verification flags are not ``True`` (never blocks).
    """
    lines: List[str] = []
    for i in sorted_node_ids(network_dict):
        target = network_dict[i]["name"]
        for src_id in network_dict[i]["inputs"]:
            src = network_dict[src_id]["name"]
            lines.append("{0} -? {1}".format(src, target))
    for i in sorted_node_ids(network_dict):
        name = network_dict[i]["name"]
        inputs = input_names_for_node(network_dict, i)
        expr = function_outputs_to_expression(
            network_dict[i]["outputs"],
            inputs,
            style="symbolic",
            node_label=name,
            warn=warn,
        )
        lines.append("${0}: {1}".format(name, expr))
    return "\n".join(lines) + "\n"
