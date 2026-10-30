# -*- coding: utf-8 -*-
"""Exact ON/OFF prime implicants via PySAT Primer-B."""
from __future__ import annotations

from typing import List, Optional

from cana.primes.timeout import call_in_subprocess
from cana.primes.util import (
    DEFAULT_TIMEOUT_S,
    MAX_PRIMES,
    MAX_PRIMER_PAREN_DEPTH,
    OnOffPrimes,
    PrimeDict,
    expr_to_pysat_formula,
    refuse_extreme_nesting,
)

__all__ = [
    "DEFAULT_TIMEOUT_S",
    "MAX_PRIMES",
    "MAX_PRIMER_PRIMES",
    "MAX_PRIMER_PAREN_DEPTH",
    "primer_on_off_primes",
]

# Alias kept for callers / tests that used the high_k name.
MAX_PRIMER_PRIMES = MAX_PRIMES


def _constant_on_off(raw: str) -> Optional[OnOffPrimes]:
    """Handle empty / constant expressions; ``None`` if not a constant."""
    if raw in ("",):
        return [], []
    if raw in ("1", "True", "true"):
        return [{}], []
    if raw in ("0", "False", "false"):
        return [], [{}]
    return None


def _pysat_forms_to_prime(forms) -> PrimeDict:
    """Map ``Formula.formulas(..., atoms_only=True)`` to a ``PrimeDict``."""
    from pysat.formula import Atom, Neg

    prime: PrimeDict = {}
    for form in forms:
        if isinstance(form, Atom):
            obj = form.object
            if isinstance(obj, bool):
                continue
            prime[str(obj)] = True
        elif isinstance(form, Neg):
            sub = form.subformula
            if isinstance(sub, Atom):
                obj = sub.object
                if isinstance(obj, bool):
                    continue
                prime[str(obj)] = False
            else:
                raise ValueError(
                    f"unexpected Neg subformula in prime: {form!r}"
                )
        else:
            raise ValueError(f"unexpected formula in prime: {form!r}")
    return prime


def _primer_polarity_primes(
    formula, *, max_primes: int, polarity: str
) -> List[PrimeDict]:
    """Enumerate all prime implicants of *formula* via Primer-B.

    Raises:
        ValueError: if more than *max_primes* implicants are found.  Enumeration
        stops immediately; no partial list is returned.
    """
    from pysat.examples.primer import Primer
    from pysat.formula import Formula

    out: List[PrimeDict] = []
    with Primer(formula, implicates=False) as primer:
        for lits in primer.enumerate():
            if len(out) >= max_primes:
                raise ValueError(
                    f"refusing Primer-B: {polarity} prime count exceeded "
                    f"MAX_PRIMER_PRIMES={max_primes} "
                    f"(stopped enumeration; partial result discarded)"
                )
            if not lits:
                out.append({})
                continue
            forms = Formula.formulas(lits, atoms_only=True)
            out.append(_pysat_forms_to_prime(forms))
    return out


def _primer_on_off_primes_inprocess(
    expr: str, max_primes: int = MAX_PRIMER_PRIMES
) -> OnOffPrimes:
    from pysat.formula import Formula, Neg

    const = _constant_on_off(expr)
    if const is not None:
        return const

    refuse_extreme_nesting(expr, limit=MAX_PRIMER_PAREN_DEPTH)
    try:
        f = expr_to_pysat_formula(expr)
        on_p = _primer_polarity_primes(
            f, max_primes=max_primes, polarity="ON"
        )
        off_p = _primer_polarity_primes(
            Neg(f), max_primes=max_primes, polarity="OFF"
        )
        return on_p, off_p
    finally:
        Formula.cleanup()


def _primer_worker(expr: str, max_primes: int, conn) -> None:
    try:
        conn.send(
            ("ok", _primer_on_off_primes_inprocess(expr, max_primes=max_primes))
        )
    except Exception as exc:  # noqa: BLE001
        conn.send(("err", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def primer_on_off_primes(
    expr: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_primes: int = MAX_PRIMER_PRIMES,
) -> OnOffPrimes:
    """Exact ON/OFF primes via PySAT Primer-B on ``f`` and ``~f``.

    Builds a non-clausal ``pysat.formula`` tree, enumerates all prime
    implicants with :class:`~pysat.examples.primer.Primer`
    (``implicates=False``), and maps clausification IDs back to variable
    names.  If either polarity would exceed *max_primes*, enumeration
    stops and :class:`ValueError` is raised (no partial result).

    Raises:
        ImportError: if pysat (``python-sat``) is not installed.
        ValueError: if ON or OFF prime count exceeds *max_primes*.
        RecursionError: if parenthesis nesting exceeds
            :data:`~cana.primes.util.MAX_PRIMER_PAREN_DEPTH`.
        TimeoutError: if wall-clock time exceeds *timeout_s*.
    """
    import pysat  # noqa: F401

    raw = expr.strip()
    refuse_extreme_nesting(raw, limit=MAX_PRIMER_PAREN_DEPTH)
    if timeout_s <= 0:
        return _primer_on_off_primes_inprocess(raw, max_primes=max_primes)
    return call_in_subprocess(
        _primer_worker, (raw, max_primes), timeout_s=timeout_s
    )
