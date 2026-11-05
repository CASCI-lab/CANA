# -*- coding: utf-8 -*-
"""
Sensitivity
===========

Average sensitivity and edge activities of a Boolean node.

Two equivalent routes:

* :func:`activities_from_bit_lut` / :func:`sensitivity_from_bit_lut` — direct
  single-input flip counts on a dense :class:`~cana.function_outputs.bit_lut.BitLut`.
* :func:`activities_from_schemata` / :func:`sensitivity_from_schemata` — upper
  edge effectiveness (:math:`1 - r_i` with ``bound="upper"``) from one-symbol
  (prime-implicant) schemata. Prefer this when PIs are already available, and
  when ``k`` is too large to materialize a dense LUT
  (:data:`~cana.function_outputs.bit_lut.MAX_DENSE_LUT_K`).

The two formulations are bit-exactly equal on the full state space.

High-``k`` sampling reuses the shared exact/sampled two-track
(``exact_k_max`` / ``n_samples``) from
:mod:`~cana.function_properties.sampling`. Percentile bootstrap CIs are
available via :func:`bootstrap_activities` and :func:`bootstrap_sensitivity`.
"""

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Sequence, Union

import numpy as np

from cana.function_outputs.bit_lut import BitLut
from cana.function_properties.redundancy import (
    bootstrap_edge_redundancy,
    edge_redundancy,
)
from cana.function_properties.sampling import (
    EXACT_K_MAX_DEFAULT,
    BootstrapResult,
    OneSymbolSchemata,
    SampledPiCoverage,
    _as_sampled_or_schemata,
    _bootstrap_means_1d,
    _percentile_ci,
    _resolve_sampled,
)


def activities_from_bit_lut(bit_lut: BitLut) -> List[float]:
    """Activity of each input from a dense LUT: fraction of states where flipping
    it changes the output.

    Args:
        bit_lut : Dense truth table; ``k`` is taken from ``bit_lut.k``.

    Returns:
        (list) : The activity of each input.
    """
    k = bit_lut.k
    n_states = len(bit_lut)
    flip_masks = [1 << (k - 1 - i) for i in range(k)]
    changes = [0] * k
    for state in range(n_states):
        out = bit_lut[state]
        for i, mask in enumerate(flip_masks):
            if bit_lut[state ^ mask] != out:
                changes[i] += 1
    return [c / n_states for c in changes]


def sensitivity_from_bit_lut(bit_lut: BitLut, norm: bool = False) -> float:
    """Average sensitivity from a dense LUT: sum of :func:`activities_from_bit_lut`.

    Args:
        bit_lut : Dense truth table; ``k`` is taken from ``bit_lut.k``.
        norm (bool) : Normalize by the number of inputs ``k``.

    Returns:
        (float)
    """
    k = bit_lut.k
    x = math.fsum(activities_from_bit_lut(bit_lut))
    if norm:
        return x / k
    return x


def activities_from_schemata(
    schemata: OneSymbolSchemata,
    k: Optional[int] = None,
    *,
    exact_k_max: int = EXACT_K_MAX_DEFAULT,
    n_samples: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    minterms: Optional[Sequence[str]] = None,
) -> List[float]:
    """Edge activities from one-symbol schemata: upper edge effectiveness.

    Equivalent to ``1 - r_i`` with ``bound="upper"`` (see
    :func:`cana.function_properties.redundancy.edge_redundancy`). Use when prime implicants are
    available, or when ``k`` exceeds the dense-LUT limit.

    Args:
        schemata : Polarity → set of length-``k`` schemas over ``{0,1,2}``.
        k (int, optional) : Inferred from schema length when omitted.
        exact_k_max, n_samples, rng, minterms : Forwarded to
            :func:`cana.function_properties.redundancy.edge_redundancy` for the exact/sampled track.

    Returns:
        (list) : The activity of each input.
    """
    r_i = edge_redundancy(
        schemata,
        k=k,
        bound="upper",
        exact_k_max=exact_k_max,
        n_samples=n_samples,
        rng=rng,
        minterms=minterms,
    )
    return [1.0 - x for x in r_i]


def sensitivity_from_schemata(
    schemata: OneSymbolSchemata,
    k: Optional[int] = None,
    *,
    norm: bool = False,
    exact_k_max: int = EXACT_K_MAX_DEFAULT,
    n_samples: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    minterms: Optional[Sequence[str]] = None,
) -> float:
    """Average sensitivity from one-symbol schemata: sum of
    :func:`activities_from_schemata`.

    Args:
        schemata : Polarity → set of length-``k`` schemas over ``{0,1,2}``.
        k (int, optional) : Inferred from schema length when omitted.
        norm (bool) : Normalize by the number of inputs ``k``.
        exact_k_max, n_samples, rng, minterms : Forwarded to
            :func:`activities_from_schemata`.

    Returns:
        (float)
    """
    acts = activities_from_schemata(
        schemata,
        k=k,
        exact_k_max=exact_k_max,
        n_samples=n_samples,
        rng=rng,
        minterms=minterms,
    )
    x = math.fsum(acts)
    if norm:
        k_inf = int(k) if k is not None else len(acts)
        return x / k_inf
    return x


def _activity_from_redundancy_boot(b: BootstrapResult) -> BootstrapResult:
    """Map an upper-edge redundancy bootstrap result to activity ``1 - r``."""
    return BootstrapResult(
        estimate=1.0 - b.estimate,
        ci_low=1.0 - b.ci_high,
        ci_high=1.0 - b.ci_low,
        boot_mean=1.0 - b.boot_mean,
        boot_std=b.boot_std,
        n_samples=b.n_samples,
        n_boot=b.n_boot,
        ci=b.ci,
    )


def bootstrap_activities(
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
) -> List[BootstrapResult]:
    """Estimate per-input activities with percentile bootstrap CIs.

    Activities are upper edge effectiveness (:math:`a_i = 1 - r_i` with
    ``bound="upper"``). Prefer a preserved :class:`SampledPiCoverage` so primes
    are not re-matched.
    """
    r_boots = bootstrap_edge_redundancy(
        sampled_or_schemata,
        schemata=schemata,
        n_samples=n_samples,
        n_boot=n_boot,
        ci=ci,
        bound="upper",
        rng=rng,
        minterms=minterms,
    )
    return [_activity_from_redundancy_boot(b) for b in r_boots]  # type: ignore[misc]


def bootstrap_sensitivity(
    sampled_or_schemata: Optional[
        Union[SampledPiCoverage, OneSymbolSchemata]
    ] = None,
    *,
    schemata: Optional[OneSymbolSchemata] = None,
    n_samples: Optional[int] = None,
    n_boot: int = 1000,
    ci: float = 0.95,
    norm: bool = False,
    rng: Optional[np.random.Generator] = None,
    minterms: Optional[Sequence[str]] = None,
) -> BootstrapResult:
    """Estimate average sensitivity with a percentile bootstrap CI.

    Bootstraps the **sum** of per-minterm activities
    (:math:`s_t = \\sum_i (1 - r_{t,i})` with upper-bound edge contributions),
    not the sum of per-input CIs. Prefer a preserved
    :class:`SampledPiCoverage`.
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

    edge = cov.edge_contributions(bound="upper")
    contrib = (1.0 - np.asarray(edge, dtype=float)).sum(axis=1)
    if norm:
        if cov.k == 0:
            raise ValueError("cannot normalize sensitivity for k=0")
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


def _coerce_bit_lut(
    outputs: Union[BitLut, Sequence[object]], k: Optional[int] = None
) -> BitLut:
    if isinstance(outputs, BitLut):
        if k is not None and int(k) != outputs.k:
            raise ValueError(
                "k={0} does not match bit_lut.k={1}".format(k, outputs.k)
            )
        return outputs
    lut = BitLut.from_sequence(outputs)
    if k is not None and int(k) != lut.k:
        raise ValueError(
            "k={0} does not match sequence length 2**{1}".format(k, lut.k)
        )
    return lut


def activities(outputs, k=None):
    """Deprecated alias of :func:`activities_from_bit_lut`.

    ``outputs`` may be a :class:`~cana.function_outputs.bit_lut.BitLut` or a
    length-``2**k`` 0/1 sequence.
    """
    warnings.warn(
        "cana.function_properties.sensitivity.activities is deprecated; use activities_from_bit_lut",
        DeprecationWarning,
        stacklevel=2,
    )
    return activities_from_bit_lut(_coerce_bit_lut(outputs, k))


def sensitivity(outputs, k=None, norm=False):
    """Deprecated alias of :func:`sensitivity_from_bit_lut`.

    ``outputs`` may be a :class:`~cana.function_outputs.bit_lut.BitLut` or a
    length-``2**k`` 0/1 sequence.
    """
    warnings.warn(
        "cana.function_properties.sensitivity.sensitivity is deprecated; use sensitivity_from_bit_lut",
        DeprecationWarning,
        stacklevel=2,
    )
    return sensitivity_from_bit_lut(_coerce_bit_lut(outputs, k), norm=norm)


def activities_old(node):
    """Deprecated alias of node :meth:`~cana.boolean_node.BooleanNode.activities_from_schemata`."""
    warnings.warn(
        "cana.function_properties.sensitivity.activities_old is deprecated; use activities_from_schemata",
        DeprecationWarning,
        stacklevel=2,
    )
    return node.activities_from_schemata()


def sensitivity_old(node, norm=False):
    """Deprecated alias of node :meth:`~cana.boolean_node.BooleanNode.sensitivity_from_schemata`."""
    warnings.warn(
        "cana.function_properties.sensitivity.sensitivity_old is deprecated; use sensitivity_from_schemata",
        DeprecationWarning,
        stacklevel=2,
    )
    return node.sensitivity_from_schemata(norm=norm)
