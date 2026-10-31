# -*- coding: utf-8 -*-
"""Shim: syntactic ROF checks + primes from CANA."""
from __future__ import annotations

from high_k.paths import ensure_cana_on_path

ensure_cana_on_path()

from cana.function_properties.rof import (  # noqa: E402
    check_syntactic_rof,
    check_syntactic_rof_ast,
    require_syntactic_rof_ast,
)
from cana.primes.rof_primes import (  # noqa: E402
    SafeNetworkPrimes,
    count_on_off_primes,
    count_rof_primes,
    generate_rof_primes,
    iter_on_off_primes,
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
