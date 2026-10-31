# -*- coding: utf-8 -*-
"""Expression-only prime cascade: syntactic ROF, then PySAT Primer-B.

Does **not** build a dense LUT or run Quine–McCluskey; callers that need a
LUT/QM fallback (e.g. :meth:`~cana.function_outputs.FunctionOutputs.get_one_symbol_schemata`)
handle that separately after an unsuccessful cascade.

When the Primer-B path is reached but ``pysat`` (``python-sat``) is missing,
a :class:`UserWarning` is emitted before returning ``status="unsuccessful"``
(so LUT/QM fallback is not silent about the skipped Primer attempt).

No wall-clock timeout — structural caps only (``max_primes``, nesting).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Literal, Optional

from cana.primes.rof import (
    check_syntactic_rof,
    count_on_off_primes,
    iter_on_off_primes,
)
from cana.primes.util import (
    MAX_PRIMES,
    OnOffPrimes,
    PrimeDict,
)

__all__ = [
    "CascadeStatus",
    "CascadeMethod",
    "PrimesResult",
    "on_off_primes_from_expression",
]

CascadeStatus = Literal[
    "ok",
    "too_many",
    "refused",
    "error",
    "unsuccessful",
]
CascadeMethod = Literal["rof", "primer"]

_PYSAT_INSTALL_HINT = "pip install python-sat"


def _warn_primer_unavailable(exc: BaseException, *, stacklevel: int) -> None:
    """Warn that Primer-B was attempted but could not run (usually missing pysat)."""
    warnings.warn(
        "Primer-B path skipped: pysat (python-sat) is not available "
        f"({type(exc).__name__}: {exc}). Install with: {_PYSAT_INSTALL_HINT}. "
        "Expression primes were not computed; callers may fall through to "
        "LUT/Quine–McCluskey.",
        UserWarning,
        stacklevel=stacklevel,
    )


@dataclass
class PrimesResult:
    """Outcome of :func:`on_off_primes_from_expression`."""

    status: CascadeStatus
    method: Optional[CascadeMethod] = None
    on_primes: Optional[List[PrimeDict]] = None
    off_primes: Optional[List[PrimeDict]] = None
    n_on: Optional[int] = None
    n_off: Optional[int] = None
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def as_on_off(self) -> OnOffPrimes:
        if not self.ok or self.on_primes is None or self.off_primes is None:
            raise ValueError(
                f"no primes available (status={self.status!r}: {self.message})"
            )
        return self.on_primes, self.off_primes


def on_off_primes_from_expression(
    expr: str,
    *,
    max_primes: int = MAX_PRIMES,
) -> PrimesResult:
    """Try syntactic ROF, then Primer-B; never builds a LUT or runs QM.

    Parameters
    ----------
    expr :
        Boolean expression (python or symbolic operators).
    max_primes :
        Cap on ON+OFF (ROF) or per-polarity (Primer) prime counts.

    Returns
    -------
    PrimesResult
        Soft guardrail hits use ``status`` rather than raising, so callers can
        fall through to a LUT/QM path.
    """
    raw = expr.strip()

    # --- syntactic ROF path ---
    info = check_syntactic_rof(raw)
    if info["is_rof"]:
        try:
            n_on, n_off = count_on_off_primes(raw)
        except ValueError as exc:
            return PrimesResult(
                status="error",
                method="rof",
                message=str(exc),
            )
        total = n_on + n_off
        if total > max_primes:
            return PrimesResult(
                status="too_many",
                method="rof",
                n_on=n_on,
                n_off=n_off,
                message=(
                    f"too many ROF primes: n_on={n_on}, n_off={n_off} "
                    f"(total={total} > max_primes={max_primes}); "
                    "not enumerating"
                ),
            )
        try:
            on_p = list(iter_on_off_primes(raw, "on"))
            off_p = list(iter_on_off_primes(raw, "off"))
        except ValueError as exc:
            return PrimesResult(
                status="error",
                method="rof",
                n_on=n_on,
                n_off=n_off,
                message=str(exc),
            )
        return PrimesResult(
            status="ok",
            method="rof",
            on_primes=on_p,
            off_primes=off_p,
            n_on=len(on_p),
            n_off=len(off_p),
        )

    # --- Primer-B path (non-ROF) ---
    try:
        from cana.primes.primer import primer_on_off_primes
    except Exception as exc:  # noqa: BLE001
        _warn_primer_unavailable(exc, stacklevel=3)
        return PrimesResult(
            status="unsuccessful",
            message=f"primer unavailable: {exc}",
        )

    try:
        on_p, off_p = primer_on_off_primes(raw, max_primes=max_primes)
    except ImportError as exc:
        _warn_primer_unavailable(exc, stacklevel=3)
        return PrimesResult(
            status="unsuccessful",
            method="primer",
            message=f"pysat not installed: {exc}",
        )
    except RecursionError as exc:
        return PrimesResult(
            status="refused",
            method="primer",
            message=str(exc),
        )
    except ValueError as exc:
        msg = str(exc)
        if "prime count exceeded" in msg or "MAX_PRIMER_PRIMES" in msg:
            return PrimesResult(
                status="too_many",
                method="primer",
                message=msg,
            )
        return PrimesResult(
            status="unsuccessful",
            method="primer",
            message=msg,
        )
    except Exception as exc:  # noqa: BLE001
        return PrimesResult(
            status="unsuccessful",
            method="primer",
            message=f"{type(exc).__name__}: {exc}",
        )

    return PrimesResult(
        status="ok",
        method="primer",
        on_primes=on_p,
        off_primes=off_p,
        n_on=len(on_p),
        n_off=len(off_p),
    )
