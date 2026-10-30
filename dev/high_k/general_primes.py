# -*- coding: utf-8 -*-
"""Exact ON/OFF prime implicants for general (possibly binate) expressions.

Fallbacks when syntactic ROF / unate shortcuts do not apply.  Each backend
returns the **full** prime-implicant set for ``f`` and ``~f`` (given enough
time and memory).  Espresso is intentionally omitted: PyEDA's Espresso
binding only yields a minimal SOP cover, not all primes.

Backends (separate APIs; no cascade):

1. CANA Quine–McCluskey over a dense LUT
2. PyEDA :meth:`~pyeda.boolalg.expr.Expression.complete_sum`
3. PySAT Primer-B (non-clausal Formula → all implicants)

All wall-clock limits use :func:`~high_k.timing.call_in_subprocess` so
Cython / C / external tools (and their grandchildren) can be killed via
process-group signals.  The dense LUT / QM path also refuses
``k > MAX_DENSE_LUT_K`` up front.  Primer-B refuses when either polarity
exceeds :data:`MAX_PRIMER_PRIMES` (enumeration stops; no partial return).
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
    expr_to_pysat_formula,
    refuse_extreme_nesting,
    to_pyeda_expr_str,
    to_symbolic_expr,
)

ensure_cana_on_path()

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
# Dense LUT / onset→QM paths allocate Θ(2^k) memory.  Refuse above this *k*
# rather than relying on a wall-clock kill after the allocation attempt.
MAX_DENSE_LUT_K = 20
# Primer-B may emit exponentially many primes.  Stop and refuse (no partial
# result) once either polarity exceeds this count.
MAX_PRIMER_PRIMES = 1_000_000

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
    # Parent-side: fail before spawn so a short TIMEOUT_S cannot mis-label
    # extreme nesting as ``timeout`` (kill during parse).
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


# ---------------------------------------------------------------------------
# 1. CANA Quine–McCluskey
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 2. PyEDA complete_sum (all primes)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 3. PySAT Primer-B (all implicants)
# ---------------------------------------------------------------------------


def _pysat_forms_to_prime(forms) -> PrimeDict:
    """Map ``Formula.formulas(..., atoms_only=True)`` to a ``PrimeDict``."""
    from pysat.formula import Atom, Neg

    prime: PrimeDict = {}
    for form in forms:
        if isinstance(form, Atom):
            obj = form.object
            if isinstance(obj, bool):
                # Constant atoms should not appear in a proper prime cube.
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
            :data:`~high_k.primes_util.MAX_PRIMER_PAREN_DEPTH`.
        TimeoutError: if wall-clock time exceeds *timeout_s*.
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
