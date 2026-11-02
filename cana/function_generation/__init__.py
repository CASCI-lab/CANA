# -*- coding: utf-8 -*-
"""Builders that generate Boolean update functions.

Cohorts mirror high-*k* taxonomy:

- :mod:`~cana.function_generation.rof` — syntactic read-once expressions
- :mod:`~cana.function_generation.unate` — unate but not ROF expressions
- :mod:`~cana.function_generation.other` — general / biased LUT generators

Expression builders produce variables ``a0…a{k-1}`` with symbolic
``&`` / ``|``.  Wrap with high_k ``as_bnet`` when a one-rule ``.bnet``
string is needed.
"""
from __future__ import annotations

from typing import Callable, Dict

from cana.function_generation.other import random_automata_table
from cana.function_generation.rof import (
    EVEN_K_BUILDERS,
    MOD3_K_BUILDERS,
    ROF_BUILDERS,
    and_chain,
    half_or_half_and,
    makino_ibaraki,
    or_chain,
    pairwise_and_or,
    pairwise_or_and,
    random_rof,
)
from cana.function_generation.unate import (
    MULTI_OCCURRENCE_BUILDERS,
    UNATE_BUILDERS,
    majority,
)

__all__ = [
    "and_chain",
    "or_chain",
    "half_or_half_and",
    "pairwise_and_or",
    "pairwise_or_and",
    "majority",
    "makino_ibaraki",
    "random_rof",
    "random_automata_table",
    "builder_accepts_k",
    "ROF_BUILDERS",
    "UNATE_BUILDERS",
    "SYNTHETIC_BUILDERS",
    "EVEN_K_BUILDERS",
    "MOD3_K_BUILDERS",
    "MULTI_OCCURRENCE_BUILDERS",
]

SYNTHETIC_BUILDERS: Dict[str, Callable[..., str]] = {
    **ROF_BUILDERS,
    **UNATE_BUILDERS,
}


def builder_accepts_k(name: str, k: int) -> bool:
    """Whether builder *name* is defined for input size *k*."""
    if name in EVEN_K_BUILDERS and k % 2 != 0:
        return False
    if name in MOD3_K_BUILDERS and k % 3 != 0:
        return False
    if name == "majority" and k < 1:
        return False
    return True
