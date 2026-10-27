# -*- coding: utf-8 -*-
"""AEON ``*.aeon`` (regulations + ``$var: expr``) import and export."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping

from cana.io.expressions import (
    expression_identifiers,
    lut_to_expression,
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


def network_dict_to_aeon(network_dict: Mapping[int, Mapping[str, Any]]) -> str:
    """Export ``NetworkDict`` to AEON text (regulations + update functions)."""
    lines: List[str] = []
    for i in sorted_node_ids(network_dict):
        target = network_dict[i]["name"]
        for src_id in network_dict[i]["inputs"]:
            src = network_dict[src_id]["name"]
            lines.append("{0} -? {1}".format(src, target))
    for i in sorted_node_ids(network_dict):
        name = network_dict[i]["name"]
        inputs = input_names_for_node(network_dict, i)
        lut = network_dict[i]["outputs"].get_bit_lut()
        expr = lut_to_expression(inputs, lut, style="symbolic")
        lines.append("${0}: {1}".format(name, expr))
    return "\n".join(lines) + "\n"
