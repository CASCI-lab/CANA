# -*- coding: utf-8 -*-
"""Exact ON/OFF prime implicants for general (possibly binate) expressions.

Fallbacks when syntactic ROF / unate shortcuts do not apply.  Each backend
returns the **full** prime-implicant set for ``f`` and ``~f`` (given enough
time and memory).  Espresso is intentionally omitted: PyEDA's Espresso
binding only yields a minimal SOP cover, not all primes.

Backends (separate APIs; no cascade):

1. CANA Quine–McCluskey over a dense LUT
2. PyEDA :meth:`~pyeda.boolalg.expr.Expression.complete_sum`
3. PySAT Primer-B (core in :mod:`cana.primes.primer`; timed wrapper here)

Wall-clock limits use :func:`~high_k.timing.call_in_subprocess` (dev only).
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from high_k.bbm_rules import as_bnet, unique_regulators
from high_k.paths import ensure_cana_on_path
from high_k.primes_util import (
    MAX_PRIMER_PAREN_DEPTH,
    MAX_PYEDA_PAREN_DEPTH,
    PrimeDict,
    dnf_terms_to_primes,
    refuse_extreme_nesting,
    to_pyeda_expr_str,
    to_symbolic_expr,
)

ensure_cana_on_path()

from cana.primes.primer import primer_on_off_primes as _primer_core  # noqa: E402
from cana.primes.util import (  # noqa: E402
    MAX_DENSE_LUT_K,
    MAX_PRIMES as MAX_PRIMER_PRIMES,
)

__all__ = [
    "DEFAULT_TIMEOUT_S",
    "EXPLORATION_TIMEOUT_S",
    "MAX_DENSE_LUT_K",
    "MAX_PRIMER_PRIMES",
    "MAX_PRIMER_PAREN_DEPTH",
    "MAX_PYEDA_PAREN_DEPTH",
    "qm_on_off_primes",
    "complete_sum_on_off_primes",
    "primer_on_off_primes",
]

DEFAULT_TIMEOUT_S = 10.0
EXPLORATION_TIMEOUT_S = 1.0

OnOffPrimes = Tuple[List[PrimeDict], List[PrimeDict]]


def _run_timed(
    inprocess: Callable[[str], OnOffPrimes],
    worker: Callable[..., None],
    expr: str,
    *,
    timeout_s: float,
    nesting_limit: Optional[int] = None,
) -> OnOffPrimes:
    """Run *inprocess* directly or *worker* under a wall-clock kill."""
    raw = expr.strip()
    refuse_extreme_nesting(raw, limit=nesting_limit)
    if timeout_s <= 0:
        return inprocess(raw)
    from high_k.timing import call_in_subprocess

    return call_in_subprocess(worker, (raw,), timeout_s=timeout_s)


def _constant_on_off(raw: str) -> Optional[OnOffPrimes]:
    """Handle empty / constant expressions; ``None`` if not a constant."""
    if raw in ("",):
        return [], []
    if raw in ("1", "True", "true"):
        return [{}], []
    if raw in ("0", "False", "false"):
        return [], [{}]
    return None


def _for_cana_eval(expr: str) -> str:
    """Rewrite ``~`` to ``!`` so CANA LUT eval uses logical (not bitwise) NOT."""
    return to_symbolic_expr(expr).replace("~", "!")


def _refuse_huge_dense_lut(expr: str) -> None:
    """Raise ``ValueError`` when dense ``2**k`` materialization is unsafe."""
    names = unique_regulators(_for_cana_eval(expr))
    k = len(names)
    if k > MAX_DENSE_LUT_K:
        raise ValueError(
            f"refusing dense LUT/QM for k={k} > MAX_DENSE_LUT_K={MAX_DENSE_LUT_K}"
        )


def _schemata_to_primes(
    sset, names: List[str]
) -> List[PrimeDict]:
    out: List[PrimeDict] = []
    for s in sset:
        d: PrimeDict = {}
        for i, ch in enumerate(s):
            if ch == "1":
                d[names[i]] = True
            elif ch == "0":
                d[names[i]] = False
        out.append(d)
    return out


def _qm_on_off_primes_inprocess(expr: str) -> OnOffPrimes:
    """Build dense LUT and run Cython QM → ``(on_primes, off_primes)``."""
    from high_k.timing import import_detached_outputs

    const = _constant_on_off(expr)
    if const is not None:
        return const

    refuse_extreme_nesting(expr)
    _refuse_huge_dense_lut(expr)
    cana_expr = _for_cana_eval(expr)
    names = unique_regulators(cana_expr)
    fo = import_detached_outputs(as_bnet(cana_expr), "Y")
    fo.build_bit_lut_from_positional_expression()
    schemata = fo.compute_one_symbol_schemata_from_bit_lut()
    return (
        _schemata_to_primes(schemata.get(1, ()), names),
        _schemata_to_primes(schemata.get(0, ()), names),
    )


def _qm_worker(expr: str, conn) -> None:
    try:
        conn.send(("ok", _qm_on_off_primes_inprocess(expr)))
    except Exception as exc:  # noqa: BLE001
        conn.send(("err", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def qm_on_off_primes(
    expr: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> OnOffPrimes:
    """Exact ON/OFF primes via dense LUT + CANA Quine–McCluskey.

    Refuses expressions with more than :data:`MAX_DENSE_LUT_K` inputs
    (predictable ``2**k`` RAM).  Otherwise *timeout_s* and subprocess kill
    bound wall-clock time.  *timeout_s* ``<= 0`` runs in-process
    (tests / debug).
    """
    return _run_timed(
        _qm_on_off_primes_inprocess,
        _qm_worker,
        expr,
        timeout_s=timeout_s,
    )


def _complete_sum_on_off_primes_inprocess(expr: str) -> OnOffPrimes:
    from pyeda.inter import expr as pyeda_expr

    const = _constant_on_off(expr)
    if const is not None:
        return const

    refuse_extreme_nesting(expr, limit=MAX_PYEDA_PAREN_DEPTH)
    f = pyeda_expr(to_pyeda_expr_str(expr))
    on_cs = f.complete_sum()
    off_cs = (~f).complete_sum()
    return dnf_terms_to_primes(on_cs), dnf_terms_to_primes(off_cs)


def _complete_sum_worker(expr: str, conn) -> None:
    try:
        conn.send(("ok", _complete_sum_on_off_primes_inprocess(expr)))
    except Exception as exc:  # noqa: BLE001
        conn.send(("err", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def complete_sum_on_off_primes(
    expr: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> OnOffPrimes:
    """Exact ON/OFF primes via pyeda ``complete_sum`` on ``f`` and ``~f``.

    Raises:
        ImportError: if pyeda is not installed.
        RecursionError: if parenthesis nesting exceeds
            :data:`~high_k.primes_util.MAX_PYEDA_PAREN_DEPTH`.
        TimeoutError: if wall-clock time exceeds *timeout_s*.
    """
    import pyeda  # noqa: F401

    return _run_timed(
        _complete_sum_on_off_primes_inprocess,
        _complete_sum_worker,
        expr,
        timeout_s=timeout_s,
        nesting_limit=MAX_PYEDA_PAREN_DEPTH,
    )


def _primer_on_off_primes_inprocess(
    expr: str, max_primes: int = MAX_PRIMER_PRIMES
) -> OnOffPrimes:
    """In-process Primer-B via :func:`cana.primes.primer.primer_on_off_primes`."""
    return _primer_core(expr, max_primes=max_primes)


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
    """Exact ON/OFF primes via PySAT Primer-B, with optional wall-clock kill.

    Core algorithm is :func:`cana.primes.primer.primer_on_off_primes`.
    *timeout_s* ``<= 0`` runs in-process; otherwise uses
    :func:`~high_k.timing.call_in_subprocess`.
    """
    import pysat  # noqa: F401

    raw = expr.strip()
    refuse_extreme_nesting(raw, limit=MAX_PRIMER_PAREN_DEPTH)
    if timeout_s <= 0:
        return _primer_on_off_primes_inprocess(raw, max_primes=max_primes)
    from high_k.timing import call_in_subprocess

    return call_in_subprocess(
        _primer_worker, (raw, max_primes), timeout_s=timeout_s
    )
