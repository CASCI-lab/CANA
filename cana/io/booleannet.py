# -*- coding: utf-8 -*-
"""BooleanNet / ``*.booleannet.txt`` (``name* = expr``) import and export."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping

from cana.io.expressions import (
    expression_identifiers,
    lut_to_expression,
    normalize_operators,
)
from cana.lut_dict import (
    LutDict,
    expressions_to_lut_dict,
    regulator_names_for_node,
    sorted_node_ids,
)


def parse_booleannet(text: str) -> LutDict:
    """Parse BooleanNet / ``*.booleannet.txt`` rules into a ``LutDict``."""
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
    return expressions_to_lut_dict(expressions, extras)


def lut_dict_to_booleannet(lut_dict: Mapping[int, Mapping[str, Any]]) -> str:
    """Export ``LutDict`` to BooleanNet text (``name* = expr``)."""
    lines: List[str] = []
    for i in sorted_node_ids(lut_dict):
        name = lut_dict[i]["name"]
        inputs = regulator_names_for_node(lut_dict, i)
        expr = lut_to_expression(inputs, lut_dict[i]["lut"], style="python")
        lines.append("{0}* = {1}".format(name, expr))
    return "\n".join(lines) + "\n"
