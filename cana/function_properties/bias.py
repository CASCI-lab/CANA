# -*- coding: utf-8 -*-
"""
Bias
====

Node bias (fraction of minterms with output 1) of a Boolean function.

Two equivalent routes:

* :func:`bias_from_bit_lut` — dense LUT popcount / ``2**k``.
* :func:`bias_from_schemata` — ON-fraction over (full or sampled)
  prime-implicant coverage. Prefer this when PIs are already available, and
  when ``k`` is too large to materialize a dense LUT
  (:data:`~cana.function_outputs.bit_lut.MAX_DENSE_LUT_K`).

High-``k`` sampling reuses the shared exact/sampled two-track
(``exact_k_max`` / ``n_samples``) from
:mod:`~cana.function_properties.sampling`. Percentile bootstrap CIs are
available via :func:`bootstrap_bias`.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from cana.function_outputs.bit_lut import BitLut
from cana.function_properties.sampling import (
    EXACT_K_MAX_DEFAULT,
    BootstrapResult,
    OneSymbolSchemata,
    SampledPiCoverage,
    _as_sampled_or_schemata,
    _bootstrap_means_1d,
    _pack_coverage_result,
    _percentile_ci,
    _resolve_sampled,
    resolve_exact_or_sampled,
)


def bias_from_bit_lut(bit_lut: BitLut) -> float:
    """Bias from a dense LUT: popcount / ``2**k``."""
    k = bit_lut.k
    return bit_lut.popcount() / float(1 << k)


def bias_from_schemata(
    schemata: OneSymbolSchemata,
    k: Optional[int] = None,
    *,
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
    """Bias with exact / sampled two-track over prime-implicant coverage.

    Same exact/sampled two-track and warning / ``return_mode`` behaviour as
    :func:`~cana.function_properties.redundancy.input_redundancy`.
    """
    sampled, mode, _k = resolve_exact_or_sampled(
        schemata,
        what="bias_from_schemata",
        k=k,
        exact_k_max=exact_k_max,
        n_samples=n_samples,
        rng=rng,
        minterms=minterms,
    )
    if sampled.n_samples < 1:
        raise ValueError("need at least one coverage entry")
    contrib = sampled.bias_contributions()
    estimate = float(math.fsum(contrib) / len(contrib))
    return _pack_coverage_result(
        estimate,
        sampled,
        mode,
        return_coverage=return_coverage,
        return_mode=return_mode,
    )


def bootstrap_bias(
    sampled_or_schemata: Optional[
        Union[SampledPiCoverage, OneSymbolSchemata]
    ] = None,
    *,
    schemata: Optional[OneSymbolSchemata] = None,
    n_samples: Optional[int] = None,
    n_boot: int = 1000,
    ci: float = 0.95,
    rng: Optional[np.random.Generator] = None,
    minterms: Optional[Sequence[str]] = None,
) -> BootstrapResult:
    """Estimate bias with a percentile bootstrap CI on the sample mean.

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

    contrib = cov.bias_contributions()
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
