# -*- coding: utf-8 -*-
"""Syntactic read-once (ROF) checks on expressions and ASTs.

**Syntactic ROF** means every identifier occurs exactly once in the written
expression / AST (``n_variables == n_unique_variables``; also called trivial
/ single-occurrence).  That is distinct from **semantic** read-once-ness: a
formula may become read-once only after algebraic rewriting (e.g.
``a & a`` ≡ ``a``), which this module does not attempt.

``is_rof`` in return dicts means *syntactic* ROF.

Prime count / generation for syntactic ROFs lives in
:mod:`cana.primes.rof_primes`.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Union

from cana.function_outputs.ast import LogicNode, ast_variable_names
from cana.function_outputs.expression import (
    expression_identifiers,
    normalize_operators,
)

__all__ = [
    "check_syntactic_rof",
    "check_syntactic_rof_ast",
    "require_syntactic_rof_ast",
]


def _occurrence_stats(
    names: List[str],
) -> Dict[str, Union[bool, int]]:
    """Return ``is_rof``, ``n_variables``, ``n_unique_variables`` for *names*."""
    counts = Counter(names)
    return {
        "is_rof": all(c == 1 for c in counts.values()),
        "n_variables": len(names),
        "n_unique_variables": len(counts),
    }


def check_syntactic_rof(expr: str) -> Dict[str, Union[bool, int]]:
    """Return syntactic-ROF stats for expression *expr*.

    ``is_rof`` means *syntactic* (single occurrence), not semantic
    read-once after simplification.  Constants (no identifiers) are
    vacuously ROFs.
    """
    toks = expression_identifiers(normalize_operators(expr, "python"))
    return _occurrence_stats(list(toks))


def check_syntactic_rof_ast(
    node: Optional[LogicNode],
) -> Dict[str, Union[bool, int]]:
    """Return syntactic-ROF stats for an AST *node* (leaf occurrence walk)."""
    return _occurrence_stats(ast_variable_names(node))


def require_syntactic_rof_ast(node: Optional[LogicNode]) -> None:
    """Raise if *node* is not a syntactic ROF (single occurrence).

    Rejects formulas that might only be read-once after simplification;
    this engine never rewrites.
    """
    info = check_syntactic_rof_ast(node)
    if info["is_rof"]:
        return
    names = ast_variable_names(node)
    counts = Counter(names)
    repeated = sorted(v for v, c in counts.items() if c > 1)
    raise ValueError(
        "AST is not a syntactic ROF (single occurrence); "
        f"repeated variables={repeated} "
        f"(n_occurrences={info['n_variables']}, "
        f"n_unique={info['n_unique_variables']}). "
        "This method does not simplify; a formula that is read-once "
        "only after rewriting (semantic ROF) is not accepted."
    )
