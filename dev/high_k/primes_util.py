# -*- coding: utf-8 -*-
"""Shared prime helpers: core in :mod:`cana.primes.util`, pyeda extras here."""
from __future__ import annotations

from typing import List

from high_k.paths import ensure_cana_on_path

ensure_cana_on_path()

from cana.primes.util import (  # noqa: E402
    MAX_PRIMER_PAREN_DEPTH,
    MAX_SAFE_PAREN_DEPTH,
    PrimeDict,
    expr_to_pysat_formula,
    paren_nesting_depth,
    refuse_extreme_nesting,
    to_symbolic_expr,
)

__all__ = [
    "PrimeDict",
    "MAX_SAFE_PAREN_DEPTH",
    "MAX_PYEDA_PAREN_DEPTH",
    "MAX_PRIMER_PAREN_DEPTH",
    "paren_nesting_depth",
    "refuse_extreme_nesting",
    "to_symbolic_expr",
    "to_pyeda_expr_str",
    "dnf_terms_to_primes",
    "expr_to_pysat_formula",
]

# pyeda ``expr`` / Espresso: hard fail ≈ depth 140 (kept in high_k only).
MAX_PYEDA_PAREN_DEPTH = 120


def to_pyeda_expr_str(expr: str) -> str:
    """Normalize *expr* to a form ``pyeda.inter.expr`` accepts (``~`` not ``!``)."""
    return to_symbolic_expr(expr).replace("!", "~")


def dnf_terms_to_primes(dnf) -> List[PrimeDict]:
    """Convert a pyeda DNF expression cover to ``[{var: True/False}, ...]``."""
    from pyeda.boolalg.expr import Complement, Variable

    if dnf is True or (hasattr(dnf, "is_one") and dnf.is_one()):
        return [{}]
    if dnf is False or (hasattr(dnf, "is_zero") and dnf.is_zero()):
        return []

    cover = getattr(dnf, "cover", None)
    if cover is None:
        if isinstance(dnf, Variable):
            return [{str(dnf.name): True}]
        if isinstance(dnf, Complement):
            return [{str(dnf.inputs[0].name): False}]
        return [{}] if bool(dnf) else []

    primes: List[PrimeDict] = []
    for cube in cover:
        prime: PrimeDict = {}
        for lit in cube:
            if isinstance(lit, Variable):
                prime[str(lit.name)] = True
            elif isinstance(lit, Complement):
                prime[str(lit.inputs[0].name)] = False
            else:
                s = str(lit)
                if s.startswith("~"):
                    prime[s[1:]] = False
                else:
                    prime[s] = True
        primes.append(prime)
    return primes
