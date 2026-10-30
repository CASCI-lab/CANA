# -*- coding: utf-8 -*-
"""Syntactic unate checks and Espresso prime-implicant generation.

**Syntactic unate** means every identifier appears only positively or only
negatively after negation is pushed to the leaves (NNF).  A variable that
occurs both ways is *binate*.  Same-polarity repeats (e.g. ``a & a``) are
still unate; that is distinct from syntactic ROF (single occurrence).

This is distinct from **semantic** unateness (cofactor / truth-table
monotonicity), which this module does not attempt.

The string path is a **single left-to-right scan** with an explicit
polarity stack (iterative; deep nesting cannot blow the call stack).
It **always stops at the first binate** variable — it never enumerates all
binates.  ``binate`` in the return dict is empty or a one-element list;
``positive`` / ``negative`` reflect only what was seen before stopping when
the formula is not unate.  ``n_unique_variables`` is always a **full**
unique-identifier count (via
:func:`~cana.function_outputs.expression.expression_identifiers`),
independent of early stop.

``is_unate`` in return dicts means *syntactic* unate.

For syntactically unate functions, Espresso's minimal SOP equals the full
prime-implicant set.  :func:`on_off_primes` runs Espresso on ``f`` and
``~f`` (via pyeda) inside a killable subprocess so wall-clock timeouts
work even though Espresso is a C extension.

Reuses:
    - CANA :func:`~cana.function_outputs.expression.expression_identifiers`
    - CANA :func:`~cana.function_outputs.expression.normalize_operators`
    - pyeda ``espresso_exprs`` (optional; imported in the prime path only)
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional, Set, Tuple, Union

from high_k.paths import ensure_cana_on_path

ensure_cana_on_path()

from cana.function_outputs.expression import (
    expression_identifiers,
    normalize_operators,
)
from high_k.primes_util import (
    PrimeDict,
    dnf_terms_to_primes,
    refuse_extreme_nesting,
    to_pyeda_expr_str,
    to_symbolic_expr,
)

__all__ = [
    "check_syntactic_unate",
    "require_syntactic_unate",
    "on_off_primes",
    "primes",
]

UnateInfo = Dict[str, Union[bool, int, List[str]]]
Polarity = Literal["on", "off"]

DEFAULT_PRIMES_TIMEOUT_S = 10.0


def _to_symbolic_expr(expr: str) -> str:
    """Normalize *expr* to symbolic ``&`` / ``|`` / ``!`` form."""
    return to_symbolic_expr(expr)


def _n_unique_from_expr(expr: str) -> int:
    """Full unique-identifier count (same tokenizer as syntactic ROF)."""
    toks = expression_identifiers(normalize_operators(expr, "python"))
    return len(set(toks))


def _stats(
    positive: Set[str],
    negative: Set[str],
    first_binate: Optional[str],
    *,
    n_unique_variables: int,
) -> UnateInfo:
    if first_binate is not None:
        # Conflict name sits in the set for its first polarity only.
        # positive/negative are partial (scan stopped); n_unique is full.
        only_pos = sorted(positive - {first_binate})
        only_neg = sorted(negative - {first_binate})
        return {
            "is_unate": False,
            "positive": only_pos,
            "negative": only_neg,
            "binate": [first_binate],
            "n_unique_variables": n_unique_variables,
        }
    return {
        "is_unate": True,
        "positive": sorted(positive),
        "negative": sorted(negative),
        "binate": [],
        "n_unique_variables": n_unique_variables,
    }


def _nnf_scan(expr: str) -> UnateInfo:
    """Linear NNF polarity scan; stop at the first binate.

    Iterative: ``&`` / ``|`` are polarity-neutral; only ``!`` and
    parentheses update an explicit stack.  Nesting depth is unbounded
    by the CPython recursion limit.
    """
    s = _to_symbolic_expr(expr)
    n = len(s)
    positive: Set[str] = set()
    negative: Set[str] = set()
    first_binate: Optional[str] = None

    def record(name: str, pol: int) -> bool:
        nonlocal first_binate
        if pol > 0:
            if name in negative:
                first_binate = name
                return True
            positive.add(name)
        else:
            if name in positive:
                first_binate = name
                return True
            negative.add(name)
        return False

    n_unique = _n_unique_from_expr(expr)

    if not s.strip():
        return _stats(positive, negative, first_binate, n_unique_variables=n_unique)

    # Top of stack = polarity inside the current paren group.
    pol_stack: List[int] = [+1]
    i = 0
    while i < n:
        while i < n and s[i].isspace():
            i += 1
        if i >= n:
            break
        c = s[i]
        if c in "&|":
            i += 1
            continue
        if c == ")":
            if len(pol_stack) > 1:
                pol_stack.pop()
            i += 1
            continue
        # Factor: optional bangs, then '(' or identifier.
        sign = 1
        while i < n and s[i] == "!":
            sign = -sign
            i += 1
            while i < n and s[i].isspace():
                i += 1
        while i < n and s[i].isspace():
            i += 1
        if i >= n:
            break
        if s[i] == "(":
            pol_stack.append(pol_stack[-1] * sign)
            i += 1
            continue
        if s[i].isalpha() or s[i] == "_":
            j = i + 1
            while j < n and (s[j].isalnum() or s[j] == "_"):
                j += 1
            if record(s[i:j], pol_stack[-1] * sign):
                break
            i = j
            continue
        # Skip unknown char to avoid hanging on malformed input.
        i += 1

    return _stats(positive, negative, first_binate, n_unique_variables=n_unique)


def check_syntactic_unate(expr: str) -> UnateInfo:
    """Return syntactic-unate stats for expression *expr*.

    Single-pass NNF polarity scan.  Stops at the first binate variable
    (``binate`` has at most one name).  ``n_unique_variables`` is always a
    full count (not truncated by early stop).  Constants / empty
    expressions are vacuously unate.
    """
    return _nnf_scan(expr)


def require_syntactic_unate(expr: str) -> None:
    """Raise if *expr* is not syntactically unate.

    Rejects formulas that might only be unate after simplification; this
    check never rewrites.  Stops at the first binate.
    """
    info = check_syntactic_unate(expr)
    if info["is_unate"]:
        return
    raise ValueError(
        "expression is not syntactically unate; "
        f"binate variables={info['binate']} "
        f"(n_unique={info['n_unique_variables']}). "
        "This method does not simplify; a formula that is unate "
        "only after rewriting (semantic unate) is not accepted."
    )


def _to_pyeda_expr_str(expr: str) -> str:
    """Normalize *expr* to a form ``pyeda.inter.expr`` accepts (``~`` not ``!``)."""
    return to_pyeda_expr_str(expr)


def _dnf_terms_to_primes(dnf) -> List[PrimeDict]:
    """Convert a pyeda DNF expression cover to ``[{var: True/False}, ...]``."""
    return dnf_terms_to_primes(dnf)


def _espresso_on_off_primes_inprocess(expr: str) -> Tuple[List[PrimeDict], List[PrimeDict]]:
    """Run Espresso on *expr* and ``~expr``; return ``(on_primes, off_primes)``.

    Assumes *expr* is already known to be syntactically unate (caller checks).
    """
    from pyeda.inter import espresso_exprs, expr as pyeda_expr

    raw = expr.strip()
    if raw in ("",):
        # Vacuous: no variables; treat as constant False-ish empty cover.
        return [], []
    if raw in ("1", "True", "true"):
        return [{}], []
    if raw in ("0", "False", "false"):
        return [], [{}]

    refuse_extreme_nesting(raw)
    f = pyeda_expr(to_pyeda_expr_str(raw))
    f_dnf = f.to_dnf()
    # Constant folding after to_dnf.
    if f_dnf is True or (hasattr(f_dnf, "is_one") and f_dnf.is_one()):
        return [{}], []
    if f_dnf is False or (hasattr(f_dnf, "is_zero") and f_dnf.is_zero()):
        return [], [{}]

    on_min, = espresso_exprs(f_dnf)
    off_dnf = (~f).to_dnf()
    if off_dnf is True or (hasattr(off_dnf, "is_one") and off_dnf.is_one()):
        off_primes: List[PrimeDict] = [{}]
    elif off_dnf is False or (hasattr(off_dnf, "is_zero") and off_dnf.is_zero()):
        off_primes = []
    else:
        off_min, = espresso_exprs(off_dnf)
        off_primes = dnf_terms_to_primes(off_min)

    return dnf_terms_to_primes(on_min), off_primes


def _espresso_worker(expr: str, conn) -> None:
    """Subprocess entry: compute ON/OFF primes; send ``('ok', result)`` or err."""
    try:
        result = _espresso_on_off_primes_inprocess(expr)
        conn.send(("ok", result))
    except Exception as exc:  # noqa: BLE001
        conn.send(("err", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def on_off_primes(
    expr: str,
    *,
    timeout_s: float = DEFAULT_PRIMES_TIMEOUT_S,
) -> Tuple[List[PrimeDict], List[PrimeDict]]:
    """Return ``(on_primes, off_primes)`` for syntactically unate *expr*.

    For unate Boolean functions, Espresso's minimal SOP is the complete
    prime-implicant set.  ON primes come from minimizing ``f``; OFF primes
    from minimizing ``~f`` (pyeda ``espresso_exprs``).

    Raises:
        ValueError: if *expr* is not syntactically unate.
        TimeoutError: if wall-clock time exceeds *timeout_s* (subprocess kill).
        ImportError: if pyeda is not installed.

    *timeout_s* ``<= 0`` runs in-process with no wall-clock limit (debug).
    """
    require_syntactic_unate(expr)
    # Import early so missing pyeda fails in the parent, not a dead worker.
    import pyeda  # noqa: F401

    raw = expr.strip()
    # Parent-side nesting guard (see ``refuse_extreme_nesting``).
    refuse_extreme_nesting(raw)

    if timeout_s <= 0:
        return _espresso_on_off_primes_inprocess(raw)

    from high_k.timing import call_in_subprocess

    return call_in_subprocess(
        _espresso_worker,
        (raw,),
        timeout_s=timeout_s,
    )


def primes(
    expr: str,
    polarity: Polarity = "on",
    *,
    timeout_s: float = DEFAULT_PRIMES_TIMEOUT_S,
) -> List[PrimeDict]:
    """Return ON or OFF primes for syntactically unate *expr*.

    Computes both polarities in one timed Espresso job (see
    :func:`on_off_primes`), then returns the requested side.
    """
    on_p, off_p = on_off_primes(expr, timeout_s=timeout_s)
    return on_p if polarity == "on" else off_p
