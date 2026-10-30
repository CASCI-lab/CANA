# -*- coding: utf-8 -*-
"""Shim: syntactic ROF / primes live in :mod:`cana.primes.rof`."""
from __future__ import annotations

from high_k.paths import ensure_cana_on_path

ensure_cana_on_path()

from cana.primes.rof import (  # noqa: E402
    SafeNetworkPrimes,
    check_syntactic_rof,
    check_syntactic_rof_ast,
    count_on_off_primes,
    count_rof_primes,
    generate_rof_primes,
    iter_on_off_primes,
    require_syntactic_rof_ast,
)

__all__ = [
    "check_syntactic_rof",
    "check_syntactic_rof_ast",
    "require_syntactic_rof_ast",
    "count_rof_primes",
    "generate_rof_primes",
    "count_on_off_primes",
    "iter_on_off_primes",
    "SafeNetworkPrimes",
]
