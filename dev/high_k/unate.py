# -*- coding: utf-8 -*-
"""Syntactic unate checks on boolean expressions / ASTs.

**Syntactic unate** means every identifier appears only positively or only
negatively after negation is pushed to the leaves (NNF).  A variable that
occurs both ways is *binate*.  Same-polarity repeats (e.g. ``a & a``) are
still unate; that is distinct from syntactic ROF (single occurrence).

This is distinct from **semantic** unateness (cofactor / truth-table
monotonicity), which this module does not attempt.

The string path is a **single left-to-right scan** with an explicit
polarity stack (iterative; deep nesting cannot blow the call stack).
It **always stops at the first binate** variable — it never enumerates all
binates.  ``binate`` in the return dict is empty or a one-element list;
``positive`` / ``negative`` reflect only what was seen before stopping when
the formula is not unate.  ``n_unique_variables`` is always a **full**
unique-identifier count (via :func:`~cana.function_outputs.expression.expression_identifiers`
or :func:`~high_k.ast.ast_variable_names`), independent of early stop.

``is_unate`` in return dicts means *syntactic* unate.

Reuses:
    - :mod:`high_k.ast` for symbolic normalize / AST walks
    - CANA :func:`~cana.function_outputs.expression.expression_identifiers`
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Union

from high_k.ast import LogicNode, ast_variable_names, to_symbolic_expr
from high_k.paths import ensure_cana_on_path

ensure_cana_on_path()

from cana.function_outputs.expression import (
    expression_identifiers,
    normalize_operators,
)

__all__ = [
    "check_syntactic_unate",
    "check_syntactic_unate_ast",
    "require_syntactic_unate_ast",
]

UnateInfo = Dict[str, Union[bool, int, List[str]]]


def _n_unique_from_expr(expr: str) -> int:
    """Full unique-identifier count (same tokenizer as syntactic ROF)."""
    toks = expression_identifiers(normalize_operators(expr, "python"))
    return len(set(toks))


def _n_unique_from_ast(node: Optional[LogicNode]) -> int:
    """Full unique-identifier count from an AST leaf walk."""
    return len(set(ast_variable_names(node)))


def _stats(
    positive: Set[str],
    negative: Set[str],
    first_binate: Optional[str],
    *,
    n_unique_variables: int,
) -> UnateInfo:
    if first_binate is not None:
        # Conflict name sits in the set for its first polarity only.
        # positive/negative are partial (scan stopped); n_unique is full.
        only_pos = sorted(positive - {first_binate})
        only_neg = sorted(negative - {first_binate})
        return {
            "is_unate": False,
            "positive": only_pos,
            "negative": only_neg,
            "binate": [first_binate],
            "n_unique_variables": n_unique_variables,
        }
    return {
        "is_unate": True,
        "positive": sorted(positive),
        "negative": sorted(negative),
        "binate": [],
        "n_unique_variables": n_unique_variables,
    }


def _nnf_scan(expr: str) -> UnateInfo:
    """Linear NNF polarity scan; stop at the first binate.

    Iterative: ``&`` / ``|`` are polarity-neutral; only ``!`` and
    parentheses update an explicit stack.  Nesting depth is unbounded
    by the CPython recursion limit.
    """
    s = to_symbolic_expr(expr)
    n = len(s)
    positive: Set[str] = set()
    negative: Set[str] = set()
    first_binate: Optional[str] = None

    def record(name: str, pol: int) -> bool:
        nonlocal first_binate
        if pol > 0:
            if name in negative:
                first_binate = name
                return True
            positive.add(name)
        else:
            if name in positive:
                first_binate = name
                return True
            negative.add(name)
        return False

    n_unique = _n_unique_from_expr(expr)

    if not s.strip():
        return _stats(positive, negative, first_binate, n_unique_variables=n_unique)

    # Top of stack = polarity inside the current paren group.
    pol_stack: List[int] = [+1]
    i = 0
    while i < n:
        while i < n and s[i].isspace():
            i += 1
        if i >= n:
            break
        c = s[i]
        if c in "&|":
            i += 1
            continue
        if c == ")":
            if len(pol_stack) > 1:
                pol_stack.pop()
            i += 1
            continue
        # Factor: optional bangs, then '(' or identifier.
        sign = 1
        while i < n and s[i] == "!":
            sign = -sign
            i += 1
            while i < n and s[i].isspace():
                i += 1
        while i < n and s[i].isspace():
            i += 1
        if i >= n:
            break
        if s[i] == "(":
            pol_stack.append(pol_stack[-1] * sign)
            i += 1
            continue
        if s[i].isalpha() or s[i] == "_":
            j = i + 1
            while j < n and (s[j].isalnum() or s[j] == "_"):
                j += 1
            if record(s[i:j], pol_stack[-1] * sign):
                break
            i = j
            continue
        # Skip unknown char to avoid hanging on malformed input.
        i += 1

    return _stats(positive, negative, first_binate, n_unique_variables=n_unique)


def _ast_scan(node: Optional[LogicNode]) -> UnateInfo:
    """AST leaf walk; stop at the first binate."""
    positive: Set[str] = set()
    negative: Set[str] = set()
    first_binate: Optional[str] = None

    def record(name: str, pol: int) -> bool:
        nonlocal first_binate
        if pol > 0:
            if name in negative:
                first_binate = name
                return True
            positive.add(name)
        else:
            if name in positive:
                first_binate = name
                return True
            negative.add(name)
        return False

    def walk(n: Optional[LogicNode], pol: int) -> bool:
        if n is None:
            return False
        if n.op == "LITERAL":
            assert isinstance(n.children, str)
            return record(n.children, pol)
        if n.op == "NEGATED_LITERAL":
            assert isinstance(n.children, str)
            return record(n.children, -pol)
        assert isinstance(n.children, list)
        for child in n.children:
            if walk(child, pol):
                return True
        return False

    walk(node, +1)
    return _stats(
        positive,
        negative,
        first_binate,
        n_unique_variables=_n_unique_from_ast(node),
    )


def check_syntactic_unate(expr: str) -> UnateInfo:
    """Return syntactic-unate stats for expression *expr*.

    Single-pass NNF polarity scan.  Stops at the first binate variable
    (``binate`` has at most one name).  ``n_unique_variables`` is always a
    full count (not truncated by early stop).  Constants / empty
    expressions are vacuously unate.
    """
    return _nnf_scan(expr)


def check_syntactic_unate_ast(node: Optional[LogicNode]) -> UnateInfo:
    """Return syntactic-unate stats for an AST *node*.

    Stops at the first binate variable (``binate`` has at most one name).
    ``n_unique_variables`` is always a full AST leaf count.
    """
    return _ast_scan(node)


def require_syntactic_unate_ast(node: Optional[LogicNode]) -> None:
    """Raise if *node* is not syntactically unate.

    Rejects formulas that might only be unate after simplification; this
    check never rewrites.  Stops at the first binate.
    """
    info = check_syntactic_unate_ast(node)
    if info["is_unate"]:
        return
    raise ValueError(
        "AST is not syntactically unate; "
        f"binate variables={info['binate']} "
        f"(n_unique={info['n_unique_variables']}). "
        "This method does not simplify; a formula that is unate "
        "only after rewriting (semantic unate) is not accepted."
    )
