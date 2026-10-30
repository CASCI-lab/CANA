# -*- coding: utf-8 -*-
"""Shared helpers for expression→prime conversion (nesting, pysat, schemata)."""
from __future__ import annotations

import re
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple

from cana.function_outputs.expression import normalize_operators, positional_names
from cana.function_outputs.one_symbol import OneSymbolSchemata

__all__ = [
    "PrimeDict",
    "OnOffPrimes",
    "MAX_SAFE_PAREN_DEPTH",
    "MAX_PRIMER_PAREN_DEPTH",
    "MAX_PRIMES",
    "DEFAULT_TIMEOUT_S",
    "MAX_DENSE_LUT_K",
    "paren_nesting_depth",
    "refuse_extreme_nesting",
    "to_symbolic_expr",
    "expr_to_pysat_formula",
    "primes_to_one_symbol_schemata",
    "one_symbol_schemata_to_primes",
]

PrimeDict = Dict[str, bool]
OnOffPrimes = Tuple[List[PrimeDict], List[PrimeDict]]

# Backend-specific parenthesis-depth refuses (measured under default
# ``sys.getrecursionlimit() == 1000``).  Refuse early so a short wall-clock
# kill cannot mis-label parse ``RecursionError`` as ``timeout``.
#
# - Primer ``expr_to_pysat_formula``: ~4 stack frames/level; hard fail ≈ 246
# - generic / QM: conservative shared default
MAX_PRIMER_PAREN_DEPTH = 220
MAX_SAFE_PAREN_DEPTH = 200

# Shared PI enumeration / ROF-generate cap used by cascade and Primer-B.
MAX_PRIMES = 1_000_000
DEFAULT_TIMEOUT_S = 10.0
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


def _prime_to_schema(prime: PrimeDict, names: Sequence[str]) -> str:
    chars: List[str] = []
    for name in names:
        if name not in prime:
            chars.append("2")
        elif prime[name]:
            chars.append("1")
        else:
            chars.append("0")
    return "".join(chars)


def primes_to_one_symbol_schemata(
    on_primes: List[PrimeDict],
    off_primes: List[PrimeDict],
    *,
    input_names: Optional[Sequence[str]] = None,
    k: Optional[int] = None,
) -> OneSymbolSchemata:
    """Convert named ON/OFF prime cubes to positional one-symbol schemata.

    *input_names* defaults to ``positional_names(k)``.  When *k* is omitted it
    is inferred from *input_names*, else from the union of prime keys (sorted).
    """
    if input_names is None:
        if k is None:
            keys: Set[str] = set()
            for p in on_primes:
                keys.update(p)
            for p in off_primes:
                keys.update(p)
            # Prefer positional x0.. order when present; else sorted names.
            if keys and all(re.fullmatch(r"x\d+", n) for n in keys):
                idxs = sorted(int(n[1:]) for n in keys)
                k = (max(idxs) + 1) if idxs else 0
                input_names = positional_names(k)
            else:
                input_names = sorted(keys)
                k = len(input_names)
        else:
            input_names = positional_names(int(k))
    names = list(input_names)
    return {
        1: {_prime_to_schema(p, names) for p in on_primes},
        0: {_prime_to_schema(p, names) for p in off_primes},
    }


def one_symbol_schemata_to_primes(
    schemata: OneSymbolSchemata,
    input_names: Sequence[str],
) -> OnOffPrimes:
    """Convert one-symbol schemata to named ON/OFF ``PrimeDict`` lists."""
    names = list(input_names)

    def _from_set(sset) -> List[PrimeDict]:
        out: List[PrimeDict] = []
        for s in sset:
            d: PrimeDict = {}
            for i, ch in enumerate(s):
                if ch == "1":
                    d[names[i]] = True
                elif ch == "0":
                    d[names[i]] = False
            out.append(d)
        return out

    return (
        _from_set(schemata.get(1, ())),
        _from_set(schemata.get(0, ())),
    )
