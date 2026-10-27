# -*- coding: utf-8 -*-
"""BoolNet / PyBoolNet ``*.bnet`` (``targets,factors``) import and export."""
from __future__ import annotations

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


def parse_bnet(text: str) -> NetworkDict:
    """Parse BoolNet / PyBoolNet ``*.bnet`` into a ``NetworkDict``."""
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
    return expressions_to_network_dict(expressions, extras)


def network_dict_to_bnet(network_dict: Mapping[int, Mapping[str, Any]]) -> str:
    """Export ``NetworkDict`` to BoolNet ``targets,factors`` text."""
    lines = ["targets,factors"]
    for i in sorted_node_ids(network_dict):
        name = network_dict[i]["name"]
        inputs = input_names_for_node(network_dict, i)
        lut = network_dict[i]["outputs"].get_bit_lut()
        expr = lut_to_expression(inputs, lut, style="symbolic")
        lines.append("{0}, {1}".format(name, expr))
    return "\n".join(lines) + "\n"
