# -*- coding: utf-8 -*-
"""Syntactic read-once (ROF) expression builders.

Builders produce variables ``a0…a{k-1}`` with symbolic ``&`` / ``|``.
Each variable appears once.
"""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional

__all__ = [
    "and_chain",
    "or_chain",
    "half_or_half_and",
    "pairwise_and_or",
    "pairwise_or_and",
    "makino_ibaraki",
    "random_rof",
    "ROF_BUILDERS",
    "EVEN_K_BUILDERS",
    "MOD3_K_BUILDERS",
]


def and_chain(k: int) -> str:
    return " & ".join(f"a{i}" for i in range(k))


def or_chain(k: int) -> str:
    return " | ".join(f"a{i}" for i in range(k))


def half_or_half_and(k: int) -> str:
    """``(a0 | … | a{k/2-1}) & a{k/2} & … & a{k-1}`` (even *k*)."""
    assert k % 2 == 0
    half = k // 2
    left = " | ".join(f"a{i}" for i in range(half))
    right = " & ".join(f"a{i}" for i in range(half, k))
    return f"({left}) & {right}"


def pairwise_and_or(k: int) -> str:
    """``(a0 & a1) | (a2 & a3) | …`` (even *k*)."""
    assert k % 2 == 0
    return " | ".join(f"(a{i} & a{i+1})" for i in range(0, k, 2))


def pairwise_or_and(k: int) -> str:
    """``(a0 | a1) & (a2 | a3) & …`` (even *k*)."""
    assert k % 2 == 0
    return " & ".join(f"(a{i} | a{i+1})" for i in range(0, k, 2))


def makino_ibaraki(k: int) -> str:
    """Makino–Ibaraki-style product-of-sums: ``(a0|a1|a2) & (a3|a4|a5) & …``.

    Width-3 rows give the classic ``O(3^{k/3})`` prime-implicant growth for
    this unate / read-once family.  Requires *k* divisible by 3.
    """
    assert k % 3 == 0
    return " & ".join(
        f"(a{i} | a{i+1} | a{i+2})" for i in range(0, k, 3)
    )


def random_rof(k: int, seed: Optional[int] = None) -> str:
    """Random read-once formula over ``a0…a{k-1}`` (each var once)."""
    rng = random.Random(seed)
    variables = [f"a{i}" for i in range(k)]

    def build(vars_: List[str]) -> str:
        if len(vars_) == 1:
            return vars_[0]
        split = rng.randint(1, len(vars_) - 1)
        left = build(vars_[:split])
        right = build(vars_[split:])
        op = rng.choice(["&", "|"])
        return f"({left} {op} {right})"

    return build(variables)


ROF_BUILDERS: Dict[str, Callable[..., str]] = {
    "and": and_chain,
    "or": or_chain,
    "half_or_half_and": half_or_half_and,
    "pairwise_and_or": pairwise_and_or,
    "pairwise_or_and": pairwise_or_and,
    "makino_ibaraki": makino_ibaraki,
    "random": random_rof,
}

EVEN_K_BUILDERS = frozenset(
    {"half_or_half_and", "pairwise_and_or", "pairwise_or_and"}
)

MOD3_K_BUILDERS = frozenset({"makino_ibaraki"})
