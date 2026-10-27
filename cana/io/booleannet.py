# -*- coding: utf-8 -*-
"""BooleanNet / ``*.booleannet.txt`` (``name* = expr``) import and export."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping

from cana.function_outputs.expression import (
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


def parse_booleannet(text: str) -> NetworkDict:
    """Parse BooleanNet / ``*.booleannet.txt`` rules into a ``NetworkDict``."""
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
    return expressions_to_network_dict(expressions, extras)


def network_dict_to_booleannet(
    network_dict: Mapping[int, Mapping[str, Any]]
) -> str:
    """Export ``NetworkDict`` to BooleanNet text (``name* = expr``)."""
    lines: List[str] = []
    for i in sorted_node_ids(network_dict):
        name = network_dict[i]["name"]
        inputs = input_names_for_node(network_dict, i)
        lut = network_dict[i]["outputs"].get_bit_lut()
        expr = lut_to_expression(inputs, lut, style="python")
        lines.append("{0}* = {1}".format(name, expr))
    return "\n".join(lines) + "\n"
