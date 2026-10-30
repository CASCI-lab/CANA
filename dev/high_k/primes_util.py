# -*- coding: utf-8 -*-
"""Shared helpers for prime-implicant conversion (pyeda DNF covers)."""
from __future__ import annotations

from typing import Dict, List

from high_k.paths import ensure_cana_on_path

ensure_cana_on_path()

from cana.function_outputs.expression import normalize_operators

__all__ = [
    "PrimeDict",
    "to_symbolic_expr",
    "to_pyeda_expr_str",
    "dnf_terms_to_primes",
]

PrimeDict = Dict[str, bool]


def to_symbolic_expr(expr: str) -> str:
    """Normalize *expr* to symbolic ``&`` / ``|`` / ``!`` form."""
    return normalize_operators(
        normalize_operators(expr, "python"), "symbolic"
    )


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
