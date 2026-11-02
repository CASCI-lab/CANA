# -*- coding: utf-8 -*-
"""Unate (but not syntactic ROF) expression builders.

Builders produce variables ``a0…a{k-1}`` with symbolic ``&`` / ``|``.
Variables may repeat across terms (multi-occurrence).
"""
from __future__ import annotations

import itertools
from typing import Callable, Dict, List

__all__ = [
    "majority",
    "UNATE_BUILDERS",
    "MULTI_OCCURRENCE_BUILDERS",
]


def majority(k: int) -> str:
    """Majority-of-*k*: OR of every AND of size ``⌊k/2⌋+1`` over ``a0…a{k-1}``.

    Output is 1 iff strictly more than half of the inputs are 1.  Variables
    repeat across terms (not a syntactic ROF).  Expression size is
    ``C(k, ⌊k/2⌋+1)`` clauses — keep *k* modest.
    """
    assert k >= 1
    threshold = (k // 2) + 1
    terms: List[str] = []
    for combo in itertools.combinations(range(k), threshold):
        term = " & ".join(f"a{i}" for i in combo)
        terms.append(f"({term})" if threshold > 1 else term)
    return " | ".join(terms)


UNATE_BUILDERS: Dict[str, Callable[..., str]] = {
    "majority": majority,
}

# Builders whose expressions reuse variables (not syntactic ROF).
MULTI_OCCURRENCE_BUILDERS = frozenset({"majority"})
