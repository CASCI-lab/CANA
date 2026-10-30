# -*- coding: utf-8 -*-
"""Shared helpers for expression→prime conversion (nesting, pysat).

Named-cube ↔ one-symbol adapters live on
:mod:`cana.function_outputs.one_symbol` and are re-exported here for
backward compatibility.
"""
from __future__ import annotations

import re
import sys
from typing import Dict, List, Optional, Tuple

from cana.function_outputs.expression import normalize_operators
from cana.function_outputs.one_symbol import (
    one_symbol_to_primes,
    primes_to_one_symbol,
)

__all__ = [
    "PrimeDict",
    "OnOffPrimes",
    "MAX_SAFE_PAREN_DEPTH",
    "MAX_PRIMER_PAREN_DEPTH",
    "MAX_PRIMES",
    "MAX_DENSE_LUT_K",
    "paren_nesting_depth",
    "refuse_extreme_nesting",
    "to_symbolic_expr",
    "expr_to_pysat_formula",
    "primes_to_one_symbol",
    "one_symbol_to_primes",
]

PrimeDict = Dict[str, bool]
OnOffPrimes = Tuple[List[PrimeDict], List[PrimeDict]]

# Backend-specific parenthesis-depth refuses (measured under default
# ``sys.getrecursionlimit() == 1000``).  Refuse early so parsers fail
# predictably instead of overflowing the call stack mid-run.
#
# - Primer ``expr_to_pysat_formula``: ~4 stack frames/level; hard fail ≈ 246
# - generic / QM: conservative shared default
MAX_PRIMER_PAREN_DEPTH = 220
MAX_SAFE_PAREN_DEPTH = 200

# Shared PI enumeration / ROF-generate cap used by cascade and Primer-B.
MAX_PRIMES = 1_000_000
# Dense LUT / onset→QM paths allocate Θ(2^k) memory.
MAX_DENSE_LUT_K = 20


def paren_nesting_depth(expr: str) -> int:
    """Return the maximum ``()`` nesting depth in *expr*."""
    depth = max_depth = 0
    for ch in expr:
        if ch == "(":
            depth += 1
            if depth > max_depth:
                max_depth = depth
        elif ch == ")":
            depth -= 1
    return max_depth


def refuse_extreme_nesting(
    expr: str, *, limit: Optional[int] = None
) -> None:
    """Raise ``RecursionError`` if parenthesis nesting exceeds *limit*.

    Default *limit* is ``min(MAX_SAFE_PAREN_DEPTH, getrecursionlimit() - 50)``.
    Pass a backend-specific cap (e.g. :data:`MAX_PRIMER_PAREN_DEPTH`) when
    the parser's stack cost is known.
    """
    if limit is None:
        limit = min(MAX_SAFE_PAREN_DEPTH, sys.getrecursionlimit() - 50)
    depth = paren_nesting_depth(expr)
    if depth > limit:
        raise RecursionError(
            f"expression parenthesis depth {depth} > {limit} "
            f"(refusing parse; would hit RecursionError/parser limits)"
        )


def to_symbolic_expr(expr: str) -> str:
    """Normalize *expr* to symbolic ``&`` / ``|`` / ``!`` form."""
    return normalize_operators(
        normalize_operators(expr, "python"), "symbolic"
    )


_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def expr_to_pysat_formula(expr: str):
    """Parse a symbolic ``&`` / ``|`` / ``!`` expression into a pysat Formula.

    Precedence: ``!`` > ``&`` > ``|``; parentheses override.  Identifiers are
    ``[A-Za-z_][A-Za-z0-9_]*``.  Constants ``0`` / ``1`` / ``True`` / ``False``
    are accepted.  Raises ``ImportError`` if pysat is missing, ``ValueError``
    on syntax errors.
    """
    from pysat.formula import Atom, And, Or, Neg

    text = to_symbolic_expr(expr).strip()
    if not text:
        raise ValueError("empty expression")

    n = len(text)
    i = 0

    def _skip_ws() -> None:
        nonlocal i
        while i < n and text[i].isspace():
            i += 1

    def _parse_or():
        nonlocal i
        left = _parse_and()
        terms = [left]
        while True:
            _skip_ws()
            if i < n and text[i] == "|":
                i += 1
                terms.append(_parse_and())
            else:
                break
        if len(terms) == 1:
            return terms[0]
        return Or(*terms)

    def _parse_and():
        nonlocal i
        left = _parse_unary()
        terms = [left]
        while True:
            _skip_ws()
            if i < n and text[i] == "&":
                i += 1
                terms.append(_parse_unary())
            else:
                break
        if len(terms) == 1:
            return terms[0]
        return And(*terms)

    def _parse_unary():
        nonlocal i
        _skip_ws()
        if i < n and text[i] in ("!", "~"):
            i += 1
            return Neg(_parse_unary())
        return _parse_atom()

    def _parse_atom():
        nonlocal i
        _skip_ws()
        if i >= n:
            raise ValueError(f"unexpected end of expression near {text!r}")
        if text[i] == "(":
            i += 1
            inner = _parse_or()
            _skip_ws()
            if i >= n or text[i] != ")":
                raise ValueError(f"missing ')' in {text!r}")
            i += 1
            return inner
        # digit constants
        if text[i] in ("0", "1") and (
            i + 1 >= n or not (text[i + 1].isalnum() or text[i + 1] == "_")
        ):
            bit = text[i]
            i += 1
            return Atom(True) if bit == "1" else Atom(False)
        m = _IDENT_RE.match(text, i)
        if not m:
            raise ValueError(
                f"unexpected token at {text[i:]!r} in {text!r}"
            )
        name = m.group(0)
        i = m.end()
        low = name.lower()
        if low == "true":
            return Atom(True)
        if low == "false":
            return Atom(False)
        return Atom(name)

    formula = _parse_or()
    _skip_ws()
    if i != n:
        raise ValueError(f"trailing junk {text[i:]!r} in {text!r}")
    return formula
