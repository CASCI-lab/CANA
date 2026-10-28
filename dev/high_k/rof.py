# -*- coding: utf-8 -*-
"""Syntactic read-once (ROF) checks for Boolean expressions.

A formula is treated as an *obvious* / syntactic ROF when every identifier
occurs exactly once (boolean keywords stripped).  This matches the deprecated
``ROF.ipynb`` / ``BBM_ROF.ipynb`` definition; it does not prove semantic
read-once-ness after algebraic rewriting.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Union

from high_k.paths import ensure_cana_on_path

ensure_cana_on_path()

from cana.function_outputs.expression import (
    expression_identifiers,
    normalize_operators,
)

__all__ = ["check_syntactic_rof"]


def check_syntactic_rof(expr: str) -> Dict[str, Union[bool, int]]:
    """Return ``is_rof``, occurrence count, and unique-variable count.

    Constants (no identifiers) are vacuously ROFs.
    """
    toks = expression_identifiers(normalize_operators(expr, "python"))
    counts = Counter(toks)
    is_rof = all(c == 1 for c in counts.values())
    return {
        "is_rof": is_rof,
        "n_variables": len(toks),
        "n_unique_variables": len(counts),
    }
