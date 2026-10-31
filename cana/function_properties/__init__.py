# -*- coding: utf-8 -*-
"""Predicates on Boolean update functions (semantic and syntactic)."""
from cana.function_properties.rof import (
    check_syntactic_rof,
    check_syntactic_rof_ast,
    require_syntactic_rof_ast,
)
from cana.function_properties.unate import (
    check_syntactic_unate,
    check_syntactic_unate_ast,
    input_sign,
    input_signs,
    is_function_monotone_increasing,
    is_function_unate,
    is_input_negative_unate,
    is_input_positive_unate,
    require_syntactic_unate,
    require_syntactic_unate_ast,
    syntactic_input_signs,
)

__all__ = [
    "check_syntactic_rof",
    "check_syntactic_rof_ast",
    "require_syntactic_rof_ast",
    "check_syntactic_unate",
    "check_syntactic_unate_ast",
    "require_syntactic_unate",
    "require_syntactic_unate_ast",
    "syntactic_input_signs",
    "input_sign",
    "input_signs",
    "is_function_monotone_increasing",
    "is_function_unate",
    "is_input_negative_unate",
    "is_input_positive_unate",
]
