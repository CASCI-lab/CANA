# -*- coding: utf-8 -*-
"""Syntactic unate checks (CANA) and Espresso prime-implicant generation.

Syntactic unate checks live in :mod:`cana.function_properties.unate`.
This module re-exports them and keeps Espresso ON/OFF prime generation
(pyeda) for high_k experiments.
"""
from __future__ import annotations

from typing import List, Literal, Tuple

from high_k.paths import ensure_cana_on_path

ensure_cana_on_path()

from cana.function_properties.unate import (  # noqa: E402
    check_syntactic_unate,
    check_syntactic_unate_ast,
    require_syntactic_unate,
    require_syntactic_unate_ast,
)
from high_k.primes_util import (  # noqa: E402
    PrimeDict,
    dnf_terms_to_primes,
    MAX_PYEDA_PAREN_DEPTH,
    refuse_extreme_nesting,
    to_pyeda_expr_str,
)

__all__ = [
    "check_syntactic_unate",
    "check_syntactic_unate_ast",
    "require_syntactic_unate",
    "require_syntactic_unate_ast",
    "on_off_primes",
    "primes",
]

Polarity = Literal["on", "off"]

DEFAULT_PRIMES_TIMEOUT_S = 10.0


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

    refuse_extreme_nesting(raw, limit=MAX_PYEDA_PAREN_DEPTH)
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
    # Parent-side nesting guard (pyeda stack; see ``MAX_PYEDA_PAREN_DEPTH``).
    refuse_extreme_nesting(raw, limit=MAX_PYEDA_PAREN_DEPTH)

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
