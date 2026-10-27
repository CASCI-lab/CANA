# -*- coding: utf-8 -*-
"""Boolean expression string helpers (re-exported from function_outputs).

Implementation lives in :mod:`cana.function_outputs.expression`.  This module
keeps the historical ``cana.io.expressions`` import path used by format
parsers and :mod:`cana.network_dict`.
"""
from __future__ import annotations

from cana.function_outputs.expression import (  # noqa: F401
    BOOL_KEYWORDS,
    IDENTIFIER_RE,
    bit_lut_from_expression,
    expression_identifiers,
    from_positional,
    lut_to_expression,
    normalize_operators,
    positional_names,
    regulator_names_in_expression,
    reorder_positional,
    to_positional,
)

__all__ = [
    "BOOL_KEYWORDS",
    "IDENTIFIER_RE",
    "normalize_operators",
    "expression_identifiers",
    "regulator_names_in_expression",
    "positional_names",
    "to_positional",
    "from_positional",
    "reorder_positional",
    "bit_lut_from_expression",
    "lut_to_expression",
]
