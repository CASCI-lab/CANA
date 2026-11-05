# -*- coding: utf-8 -*-
"""Predicates and quantitative measures on Boolean update functions.

Submodules:

- :mod:`~cana.function_properties.unate` / :mod:`~cana.function_properties.rof`
  — semantic and syntactic predicates
- :mod:`~cana.function_properties.sampling` — shared exact/sampled PI coverage
  and bootstrap helpers
- :mod:`~cana.function_properties.redundancy` — input / edge redundancy
- :mod:`~cana.function_properties.sensitivity` — activities and average sensitivity
- :mod:`~cana.function_properties.bias` — node bias (ON-fraction)
- :mod:`~cana.function_properties.symmetry` — LUT permutation symmetry

The package root re-exports the unate / ROF predicate API for convenience.
Measure APIs are imported from their submodules.
"""
from cana.function_properties.rof import (
    check_syntactic_rof,
    check_syntactic_rof_ast,
    require_syntactic_rof_ast,
)
from cana.function_properties.unate import (
    SIGN_BINATE,
    SIGN_IRRELEVANT,
    SIGN_NEGATIVE,
    SIGN_POSITIVE,
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
    "SIGN_POSITIVE",
    "SIGN_NEGATIVE",
    "SIGN_BINATE",
    "SIGN_IRRELEVANT",
    "input_sign",
    "input_signs",
    "is_function_monotone_increasing",
    "is_function_unate",
    "is_input_negative_unate",
    "is_input_positive_unate",
]
