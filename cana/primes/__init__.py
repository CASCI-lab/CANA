# -*- coding: utf-8 -*-
"""Expression → prime implicants: syntactic ROF, Primer-B, and cascade.

LUT / Quine–McCluskey remain on
:class:`~cana.function_outputs.FunctionOutputs` and are tried *after* the
expression cascade when deriving one-symbol schemata.

Wall-clock timeouts are not part of this package; see ``dev/high_k.timing``.
"""
from cana.primes.ast import (
    LogicNode,
    ast_leaf_polarities,
    ast_variable_names,
    invert_ast,
    parse_to_ast,
    to_symbolic_expr,
)
from cana.primes.cascade import (
    CascadeMethod,
    CascadeStatus,
    PrimesResult,
    on_off_primes_from_expression,
)
from cana.primes.primer import primer_on_off_primes
from cana.primes.rof import (
    SafeNetworkPrimes,
    check_syntactic_rof,
    check_syntactic_rof_ast,
    count_on_off_primes,
    count_rof_primes,
    generate_rof_primes,
    iter_on_off_primes,
    require_syntactic_rof_ast,
)
from cana.primes.util import (
    MAX_PRIMES,
    MAX_PRIMER_PAREN_DEPTH,
    MAX_SAFE_PAREN_DEPTH,
    OnOffPrimes,
    PrimeDict,
    expr_to_pysat_formula,
    one_symbol_to_primes,
    paren_nesting_depth,
    primes_to_one_symbol,
    refuse_extreme_nesting,
)

# Backward-compatible alias used by high_k / tests.
MAX_PRIMER_PRIMES = MAX_PRIMES

__all__ = [
    "LogicNode",
    "ast_leaf_polarities",
    "ast_variable_names",
    "invert_ast",
    "parse_to_ast",
    "to_symbolic_expr",
    "CascadeMethod",
    "CascadeStatus",
    "PrimesResult",
    "on_off_primes_from_expression",
    "primer_on_off_primes",
    "SafeNetworkPrimes",
    "check_syntactic_rof",
    "check_syntactic_rof_ast",
    "count_on_off_primes",
    "count_rof_primes",
    "generate_rof_primes",
    "iter_on_off_primes",
    "require_syntactic_rof_ast",
    "MAX_PRIMES",
    "MAX_PRIMER_PRIMES",
    "MAX_PRIMER_PAREN_DEPTH",
    "MAX_SAFE_PAREN_DEPTH",
    "OnOffPrimes",
    "PrimeDict",
    "expr_to_pysat_formula",
    "one_symbol_to_primes",
    "paren_nesting_depth",
    "primes_to_one_symbol",
    "refuse_extreme_nesting",
]
