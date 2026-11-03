# -*- coding: utf-8 -*-
"""
Redundancy
==========

Input and edge redundancy of a Boolean node from one-symbol prime implicants.

For ``k <= exact_k_max`` (default 14) values are exact averages over the full
prime-implicant coverage. For larger ``k``, minterms are sampled (default
``n_samples = 2**exact_k_max``) and the same per-minterm contributions are
averaged — a :class:`UserWarning` is emitted on that automatic fallback
(suppress by passing ``exact_k_max=-1``, or force exact with
``exact_k_max >= k``). Pass ``return_mode=True`` to get
``\"exact\"`` / ``\"sampled\"`` alongside the value. Bootstrap confidence
intervals resample contributions from a preserved
:class:`~cana.function_properties.sampling.SampledPiCoverage` without
re-matching primes.

Shared sampling / bootstrap plumbing lives in
:mod:`~cana.function_properties.sampling` and is re-exported here for
backwards compatibility.
"""
from __future__ import annotations

import math
from statistics import mean
from typing import (
    Callable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from cana.function_properties.sampling import (
    EXACT_K_MAX_DEFAULT,
    WILDCARD,
    BootstrapResult,
    OneSymbolSchemata,
    PiCoverage,
    SampledPiCoverage,
    _as_sampled_or_schemata,
    _bootstrap_means_1d,
    _infer_k_from_coverage,
    _infer_k_from_schemata,
    _looks_like_pi_coverage,
    _pack_coverage_result,
    _percentile_ci,
    _redundancy_mode,  # noqa: F401  — BooleanNode / back-compat
    _resolve_sampled,
    _warn_if_sampled,  # noqa: F401  — BooleanNode / back-compat
    covering_primes,
    covering_primes_with_polarity,
    pi_covers,
    resolve_exact_or_sampled,
    sample_binstates,
    sample_pi_coverage,
)

__all__ = [
    "EXACT_K_MAX_DEFAULT",
    "WILDCARD",
    "BootstrapResult",
    "SampledPiCoverage",
    "OneSymbolSchemata",
    "pi_covers",
    "covering_primes",
    "covering_primes_with_polarity",
    "sample_binstates",
    "sample_pi_coverage",
    "input_redundancy_from_coverage",
    "edge_redundancy_from_coverage",
    "input_redundancy_contributions",
    "edge_redundancy_contributions",
    "input_redundancy",
    "edge_redundancy",
    "bootstrap_input_redundancy",
    "bootstrap_edge_redundancy",
]

Operator = Callable[[Sequence[float]], float]

# Re-export private helpers still used by BooleanNode / sibling measures.
_pack_redundancy_result = _pack_coverage_result


def input_redundancy_from_coverage(
    pi_coverage: PiCoverage,
    k: Optional[int] = None,
    *,
    operator: Operator = mean,
    norm: bool = True,
) -> float:
    """Average input redundancy over entries of a (full or sampled) coverage."""
    if not hasattr(operator, "__call__"):
        raise TypeError(
            'operator must be callable (e.g. statistics.mean, min, max)'
        )
    if not pi_coverage:
        raise ValueError("need at least one coverage entry")
    k_inf = int(k) if k is not None else _infer_k_from_coverage(pi_coverage)
    contrib = [
        float(operator([pi.count(WILDCARD) for pi in pis]))
        for pis in pi_coverage.values()
    ]
    # Prefer math.fsum over np.mean for correctly rounded, order-independent means.
    k_r = math.fsum(contrib) / len(contrib)
    if norm:
        if k_inf == 0:
            raise ValueError("cannot normalize input redundancy for k=0")
        k_r /= k_inf
    return k_r


def edge_redundancy_from_coverage(
    pi_coverage: PiCoverage,
    k: Optional[int] = None,
    *,
    bound: str = "mean",
) -> Union[List[float], List[Tuple[float, float]]]:
    """Per-input edge redundancy over a (full or sampled) coverage."""
    if not pi_coverage:
        raise ValueError("need at least one coverage entry")
    k_inf = int(k) if k is not None else _infer_k_from_coverage(pi_coverage)
    n = len(pi_coverage)

    if bound in ("lower", "mean", "avg", "upper"):
        totals = [0.0] * k_inf
        for pis in pi_coverage.values():
            for j in range(k_inf):
                flags = [pi[j] == WILDCARD for pi in pis]
                if bound == "lower":
                    totals[j] += float(all(flags))
                elif bound in ("mean", "avg"):
                    totals[j] += float(sum(flags) / len(flags))
                else:
                    totals[j] += float(any(flags))
        return [t / n for t in totals]

    if bound == "tuple":
        lo = [0.0] * k_inf
        hi = [0.0] * k_inf
        for pis in pi_coverage.values():
            for j in range(k_inf):
                flags = [pi[j] == WILDCARD for pi in pis]
                lo[j] += float(all(flags))
                hi[j] += float(any(flags))
        return [(a / n, b / n) for a, b in zip(lo, hi)]

    raise ValueError(
        'bound must be "lower", "mean", "avg", "upper", or "tuple"; '
        "got {0!r}".format(bound)
    )


def input_redundancy_contributions(
    coverage_or_schemata: Union[PiCoverage, OneSymbolSchemata],
    minterms: Sequence[str],
    operator: Operator = mean,
) -> np.ndarray:
    """Per-minterm Φ(#wildcards among covering PIs); shape ``(n,)``.

    *coverage_or_schemata* may be a pi-coverage dict (binstate → PIs) or
    one-symbol ON/OFF schemata (polarity → PIs). Coverage is detected by
    whether keys look like binstates.
    """
    if not hasattr(operator, "__call__"):
        raise TypeError(
            'operator must be callable (e.g. statistics.mean, min, max)'
        )
    out = np.empty(len(minterms), dtype=np.float64)
    is_coverage = _looks_like_pi_coverage(coverage_or_schemata)
    for i, binstate in enumerate(minterms):
        if is_coverage:
            pis = list(coverage_or_schemata[binstate])  # type: ignore[index]
        else:
            pis = covering_primes(
                binstate, coverage_or_schemata  # type: ignore[arg-type]
            )
        out[i] = float(operator([pi.count(WILDCARD) for pi in pis]))
    return out


def edge_redundancy_contributions(
    coverage_or_schemata: Union[PiCoverage, OneSymbolSchemata],
    minterms: Sequence[str],
    bound: str = "mean",
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Per-minterm edge contributions.

    For ``bound`` in ``lower`` / ``mean`` / ``upper``: shape ``(n, k)``.
    For ``bound == "tuple"``: ``(lower, upper)`` each shape ``(n, k)``.
    """
    if not minterms:
        if _looks_like_pi_coverage(coverage_or_schemata):
            k = _infer_k_from_coverage(coverage_or_schemata)  # type: ignore[arg-type]
        else:
            k = _infer_k_from_schemata(coverage_or_schemata)  # type: ignore[arg-type]
        empty = np.zeros((0, k), dtype=np.float64)
        if bound == "tuple":
            return empty, empty.copy()
        return empty

    k = len(minterms[0])
    n = len(minterms)
    is_coverage = _looks_like_pi_coverage(coverage_or_schemata)

    def _pis(binstate: str) -> Sequence[str]:
        if is_coverage:
            return coverage_or_schemata[binstate]  # type: ignore[index]
        return covering_primes(
            binstate, coverage_or_schemata  # type: ignore[arg-type]
        )

    if bound in ("lower", "mean", "avg", "upper"):
        out = np.empty((n, k), dtype=np.float64)
        for row, binstate in enumerate(minterms):
            pis = _pis(binstate)
            for j in range(k):
                flags = [pi[j] == WILDCARD for pi in pis]
                if bound == "lower":
                    out[row, j] = float(all(flags))
                elif bound in ("mean", "avg"):
                    out[row, j] = float(sum(flags) / len(flags))
                else:
                    out[row, j] = float(any(flags))
        return out

    if bound == "tuple":
        lower = np.empty((n, k), dtype=np.float64)
        upper = np.empty((n, k), dtype=np.float64)
        for row, binstate in enumerate(minterms):
            pis = _pis(binstate)
            for j in range(k):
                flags = [pi[j] == WILDCARD for pi in pis]
                lower[row, j] = float(all(flags))
                upper[row, j] = float(any(flags))
        return lower, upper

    raise ValueError(
        'bound must be "lower", "mean", "avg", "upper", or "tuple"; '
        "got {0!r}".format(bound)
    )


def input_redundancy(
    schemata: OneSymbolSchemata,
    k: Optional[int] = None,
    *,
    operator: Operator = mean,
    norm: bool = True,
    exact_k_max: int = EXACT_K_MAX_DEFAULT,
    n_samples: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    minterms: Optional[Sequence[str]] = None,
    return_coverage: bool = False,
    return_mode: bool = False,
) -> Union[
    float,
    Tuple[float, SampledPiCoverage],
    Tuple[float, str],
    Tuple[float, SampledPiCoverage, str],
]:
    """Input redundancy :math:`k_r` with exact / sampled two-track.

    If ``k <= exact_k_max`` and *minterms* is not provided, enumerates the full
    ``2**k`` space. Otherwise samples ``n_samples`` (default ``2**exact_k_max``)
    minterms.

    When sampling is used because ``k > exact_k_max`` (and ``exact_k_max >= 0``),
    a :class:`UserWarning` is emitted. Pass ``exact_k_max=-1`` to force sampling
    without that warning, or ``exact_k_max >= k`` for the exact value.
    With ``return_mode=True``, the second (or third) return value is
    ``\"exact\"`` or ``\"sampled\"``.
    """
    sampled, mode, k_inf = resolve_exact_or_sampled(
        schemata,
        what="input_redundancy",
        k=k,
        exact_k_max=exact_k_max,
        n_samples=n_samples,
        rng=rng,
        minterms=minterms,
    )
    estimate = input_redundancy_from_coverage(
        sampled.coverage, k_inf, operator=operator, norm=norm
    )
    return _pack_coverage_result(
        estimate,
        sampled,
        mode,
        return_coverage=return_coverage,
        return_mode=return_mode,
    )


def edge_redundancy(
    schemata: OneSymbolSchemata,
    k: Optional[int] = None,
    *,
    bound: str = "mean",
    exact_k_max: int = EXACT_K_MAX_DEFAULT,
    n_samples: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    minterms: Optional[Sequence[str]] = None,
    return_coverage: bool = False,
    return_mode: bool = False,
) -> Union[
    List[float],
    List[Tuple[float, float]],
    Tuple[Union[List[float], List[Tuple[float, float]]], SampledPiCoverage],
    Tuple[Union[List[float], List[Tuple[float, float]]], str],
    Tuple[
        Union[List[float], List[Tuple[float, float]]],
        SampledPiCoverage,
        str,
    ],
]:
    """Per-input edge redundancy :math:`r_i` with exact / sampled two-track.

    Same exact/sampled two-track and warning / ``return_mode`` behaviour as
    :func:`input_redundancy`.
    """
    sampled, mode, k_inf = resolve_exact_or_sampled(
        schemata,
        what="edge_redundancy",
        k=k,
        exact_k_max=exact_k_max,
        n_samples=n_samples,
        rng=rng,
        minterms=minterms,
    )
    estimate = edge_redundancy_from_coverage(
        sampled.coverage, k_inf, bound=bound
    )
    return _pack_coverage_result(
        estimate,
        sampled,
        mode,
        return_coverage=return_coverage,
        return_mode=return_mode,
    )


def bootstrap_input_redundancy(
    sampled_or_schemata: Optional[
        Union[SampledPiCoverage, OneSymbolSchemata]
    ] = None,
    *,
    schemata: Optional[OneSymbolSchemata] = None,
    n_samples: Optional[int] = None,
    n_boot: int = 1000,
    ci: float = 0.95,
    operator: Operator = mean,
    norm: bool = True,
    rng: Optional[np.random.Generator] = None,
    minterms: Optional[Sequence[str]] = None,
) -> BootstrapResult:
    """Estimate :math:`k_r` with a percentile bootstrap CI on the sample mean.

    Prefer passing a preserved :class:`SampledPiCoverage` so primes are not
    re-matched. Schemata + ``n_samples`` / ``minterms`` still work
    (including the legacy positional ``schemata`` call).
    """
    gen = rng if rng is not None else np.random.default_rng()
    sampled, schemata = _as_sampled_or_schemata(
        sampled_or_schemata, schemata
    )
    cov = _resolve_sampled(
        sampled,
        schemata,
        n_samples=n_samples,
        rng=gen,
        minterms=minterms,
    )
    if cov.n_samples < 1:
        raise ValueError("need at least one minterm to bootstrap")
    n_boot = int(n_boot)
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1, got {0}".format(n_boot))
    if not (0.0 < ci < 1.0):
        raise ValueError("ci must be in (0, 1), got {0}".format(ci))

    contrib = cov.input_contributions(operator=operator)
    if norm:
        if cov.k == 0:
            raise ValueError("cannot normalize input redundancy for k=0")
        contrib = contrib / cov.k

    estimate = float(np.mean(contrib))
    boot = _bootstrap_means_1d(contrib, n_boot=n_boot, rng=gen)
    lo, hi = _percentile_ci(boot, ci)
    return BootstrapResult(
        estimate=estimate,
        ci_low=lo,
        ci_high=hi,
        boot_mean=float(np.mean(boot)),
        boot_std=float(np.std(boot, ddof=1)) if n_boot > 1 else 0.0,
        n_samples=cov.n_samples,
        n_boot=n_boot,
        ci=float(ci),
    )


def bootstrap_edge_redundancy(
    sampled_or_schemata: Optional[
        Union[SampledPiCoverage, OneSymbolSchemata]
    ] = None,
    *,
    schemata: Optional[OneSymbolSchemata] = None,
    n_samples: Optional[int] = None,
    n_boot: int = 1000,
    ci: float = 0.95,
    bound: str = "mean",
    rng: Optional[np.random.Generator] = None,
    minterms: Optional[Sequence[str]] = None,
) -> Union[List[BootstrapResult], List[Tuple[BootstrapResult, BootstrapResult]]]:
    """Estimate per-input :math:`r_i` with percentile bootstrap CIs.

    For ``bound == "tuple"``, returns ``[(lower_result, upper_result), ...]``.
    Prefer a preserved :class:`SampledPiCoverage`; schemata + ``n_samples``
    still work (legacy positional ``schemata`` included).
    """
    gen = rng if rng is not None else np.random.default_rng()
    sampled, schemata = _as_sampled_or_schemata(
        sampled_or_schemata, schemata
    )
    cov = _resolve_sampled(
        sampled,
        schemata,
        n_samples=n_samples,
        rng=gen,
        minterms=minterms,
    )
    if cov.n_samples < 1:
        raise ValueError("need at least one minterm to bootstrap")
    n_boot = int(n_boot)
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1, got {0}".format(n_boot))
    if not (0.0 < ci < 1.0):
        raise ValueError("ci must be in (0, 1), got {0}".format(ci))

    def _pack(
        col_contrib: np.ndarray, boot_cols: np.ndarray
    ) -> List[BootstrapResult]:
        results: List[BootstrapResult] = []
        k_dim = boot_cols.shape[1]
        for j in range(k_dim):
            est = float(np.mean(col_contrib[:, j]))
            b = boot_cols[:, j]
            lo, hi = _percentile_ci(b, ci)
            results.append(
                BootstrapResult(
                    estimate=est,
                    ci_low=lo,
                    ci_high=hi,
                    boot_mean=float(np.mean(b)),
                    boot_std=float(np.std(b, ddof=1)) if n_boot > 1 else 0.0,
                    n_samples=cov.n_samples,
                    n_boot=n_boot,
                    ci=float(ci),
                )
            )
        return results

    contrib = cov.edge_contributions(bound=bound)
    n = cov.n_samples
    idx = gen.integers(0, n, size=(n_boot, n), dtype=np.int64)
    if bound == "tuple":
        lower, upper = contrib  # type: ignore[misc]
        boot_lo = lower[idx].mean(axis=1)
        boot_hi = upper[idx].mean(axis=1)
        lo_res = _pack(lower, boot_lo)
        hi_res = _pack(upper, boot_hi)
        return list(zip(lo_res, hi_res))

    arr = contrib  # type: ignore[assignment]
    boot = arr[idx].mean(axis=1)  # type: ignore[index]
    return _pack(arr, boot)  # type: ignore[arg-type]
