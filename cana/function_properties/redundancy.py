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
:class:`SampledPiCoverage` without re-matching primes.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from statistics import mean
from typing import (
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np

from cana.cutils import statenum_to_binstate

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

EXACT_K_MAX_DEFAULT = 14
WILDCARD = "2"

OneSymbolSchemata = Mapping[int, Set[str]]
Operator = Callable[[Sequence[float]], float]
PiCoverage = Mapping[str, Sequence[str]]


@dataclass(frozen=True)
class BootstrapResult:
    """Point estimate with a percentile bootstrap confidence interval."""

    estimate: float
    ci_low: float
    ci_high: float
    boot_mean: float
    boot_std: float
    n_samples: int
    n_boot: int
    ci: float


@dataclass(frozen=True)
class SampledPiCoverage:
    """Partial ``pi_coverage`` from sampled minterms.

    ``coverage`` maps each sampled binary state to the one-symbol primes that
    cover it (same shape as a full :meth:`~cana.boolean_node.BooleanNode.pi_coverage`
    dict, but only for the sample). ``outputs`` stores the polarity (0/1) of
    each sampled minterm so bias can be estimated without re-matching primes.
    Bootstrap helpers derive per-minterm contributions from this coverage
    without re-matching primes.
    """

    coverage: Dict[str, Tuple[str, ...]]
    k: int
    n_samples: int
    outputs: Dict[str, int]

    @property
    def minterms(self) -> List[str]:
        return list(self.coverage.keys())

    def input_contributions(self, operator: Operator = mean) -> np.ndarray:
        return input_redundancy_contributions(
            self.coverage, self.minterms, operator=operator
        )

    def edge_contributions(
        self, bound: str = "mean"
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        return edge_redundancy_contributions(
            self.coverage, self.minterms, bound=bound
        )

    def bias_contributions(self) -> np.ndarray:
        """Per-minterm output bits as ``float64``, shape ``(n_samples,)``."""
        return np.asarray(
            [self.outputs[m] for m in self.minterms], dtype=np.float64
        )


def pi_covers(schema: str, binstate: str) -> bool:
    """Return True if one-symbol *schema* covers binary *binstate*."""
    if len(schema) != len(binstate):
        raise ValueError(
            "schema length {0} != binstate length {1}".format(
                len(schema), len(binstate)
            )
        )
    return all(s == WILDCARD or s == b for s, b in zip(schema, binstate))


def covering_primes_with_polarity(
    binstate: str, schemata: OneSymbolSchemata
) -> Tuple[List[str], int]:
    """Return covering one-symbol PIs and the output polarity of *binstate*.

    Raises:
        ValueError: if no prime covers *binstate*, or ON and OFF primes both
            cover it (polarity conflict).
    """
    covered: List[str] = []
    found_polarity: Optional[int] = None
    for polarity in (0, 1):
        for pi in schemata.get(polarity, ()) or ():
            if pi_covers(pi, binstate):
                if found_polarity is not None and found_polarity != polarity:
                    raise ValueError(
                        "polarity conflict at binstate {0!r}: "
                        "covered by both ON and OFF primes".format(binstate)
                    )
                found_polarity = polarity
                covered.append(pi)
    if not covered or found_polarity is None:
        raise ValueError(
            "no prime implicant covers binstate {0!r} "
            "(incomplete ON/OFF schemata?)".format(binstate)
        )
    return covered, found_polarity


def covering_primes(binstate: str, schemata: OneSymbolSchemata) -> List[str]:
    """Return one-symbol PIs covering *binstate* (ON and OFF scanned).

    Raises:
        ValueError: if no prime covers *binstate* (incomplete cover), or ON
            and OFF primes both cover it (polarity conflict).
    """
    covered, _polarity = covering_primes_with_polarity(binstate, schemata)
    return covered


def sample_binstates(
    k: int,
    n: int,
    *,
    rng: Optional[np.random.Generator] = None,
    replace: bool = False,
) -> List[str]:
    """Uniform sample of *n* binary strings of length *k*.

    If ``not replace`` and ``n >= 2**k``, returns the full space
    ``0..2**k-1`` (exact enumeration path).

    For large *k* (where ``2**k`` does not fit in ``int64``), samples
    bitstrings directly instead of indexing into ``range(2**k)``.
    """
    k = int(k)
    n = int(n)
    if k < 0:
        raise ValueError("k must be >= 0, got {0}".format(k))
    if n < 0:
        raise ValueError("n must be >= 0, got {0}".format(n))
    if n == 0:
        return []

    # Full enumeration only when the space is materializable.
    if not replace and k <= 62:
        space = 1 << k
        if n >= space:
            return [statenum_to_binstate(i, base=k) for i in range(space)]

    gen = rng if rng is not None else np.random.default_rng()

    # Small/medium k: sample integer indices in ``[0, 2**k)``.
    if k <= 62:
        space = 1 << k
        if replace:
            idxs = gen.integers(0, space, size=n, dtype=np.int64)
        else:
            if n > space:
                raise ValueError(
                    "cannot sample n={0} without replacement from 2**k={1}".format(
                        n, space
                    )
                )
            idxs = gen.choice(space, size=n, replace=False)
        return [statenum_to_binstate(int(i), base=k) for i in idxs]

    # Large k: draw bit rows. Without replacement uses a set (n << 2**k).
    def _rows_to_strings(rows: np.ndarray) -> List[str]:
        return ["".join("1" if b else "0" for b in row) for row in rows]

    if replace:
        bits = gen.integers(0, 2, size=(n, k), dtype=np.uint8)
        return _rows_to_strings(bits)

    out: List[str] = []
    seen: set = set()
    while len(out) < n:
        need = n - len(out)
        batch = gen.integers(
            0, 2, size=(max(need * 2, 64), k), dtype=np.uint8
        )
        for s in _rows_to_strings(batch):
            if s in seen:
                continue
            seen.add(s)
            out.append(s)
            if len(out) >= n:
                break
    return out


def _infer_k_from_schemata(schemata: OneSymbolSchemata) -> int:
    for polarity in (0, 1):
        for pi in schemata.get(polarity, ()) or ():
            return len(pi)
    raise ValueError("schemata contain no prime implicants")


def _infer_k_from_coverage(pi_coverage: PiCoverage) -> int:
    if not pi_coverage:
        raise ValueError("pi_coverage is empty")
    return len(next(iter(pi_coverage)))


def _default_n_samples(exact_k_max: int) -> int:
    exact_k_max = int(exact_k_max)
    if exact_k_max < 0:
        raise ValueError(
            "exact_k_max must be >= 0, got {0}".format(exact_k_max)
        )
    return 1 << exact_k_max


def _redundancy_mode(k: int, n: int, *, use_exact: bool) -> str:
    """Return ``\"exact\"`` if the average is over the full ``2**k`` space."""
    if use_exact:
        return "exact"
    # Full-space request via n_samples / minterms is mathematically exact.
    if k < 63 and n >= (1 << k):
        return "exact"
    return "sampled"


def _warn_if_sampled(
    what: str,
    *,
    mode: str,
    k: int,
    exact_k_max: int,
    n: int,
) -> None:
    """Warn on automatic sampling fallback (``exact_k_max >= 0``).

    Callers that force the sampling path with ``exact_k_max < 0`` are silent.
    """
    if mode != "sampled" or exact_k_max < 0:
        return
    warnings.warn(
        "{0}: k={1} > exact_k_max={2}; sampling n_samples={3} minterms "
        "(pass exact_k_max>=k for the exact full-coverage value)".format(
            what, k, exact_k_max, n
        ),
        UserWarning,
        stacklevel=3,
    )


def _pack_redundancy_result(
    estimate,
    sampled: SampledPiCoverage,
    mode: str,
    *,
    return_coverage: bool,
    return_mode: bool,
):
    if return_coverage and return_mode:
        return estimate, sampled, mode
    if return_coverage:
        return estimate, sampled
    if return_mode:
        return estimate, mode
    return estimate


def sample_pi_coverage(
    schemata: OneSymbolSchemata,
    n_samples: int,
    *,
    rng: Optional[np.random.Generator] = None,
    minterms: Optional[Sequence[str]] = None,
    k: Optional[int] = None,
) -> SampledPiCoverage:
    """Sample minterms and build a partial pi-coverage dict."""
    k_inf = int(k) if k is not None else _infer_k_from_schemata(schemata)
    if minterms is not None:
        mt = list(minterms)
        if mt and any(len(m) != k_inf for m in mt):
            raise ValueError(
                "all minterms must have length k={0}".format(k_inf)
            )
    else:
        mt = sample_binstates(k_inf, int(n_samples), rng=rng, replace=False)

    coverage: Dict[str, Tuple[str, ...]] = {}
    outputs: Dict[str, int] = {}
    for binstate in mt:
        pis, polarity = covering_primes_with_polarity(binstate, schemata)
        coverage[binstate] = tuple(pis)
        outputs[binstate] = polarity
    return SampledPiCoverage(
        coverage=coverage,
        k=k_inf,
        n_samples=len(coverage),
        outputs=outputs,
    )


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


def _looks_like_pi_coverage(obj: Mapping) -> bool:
    """Heuristic: pi_coverage keys are binstates; schemata keys are 0/1."""
    if not obj:
        return False
    key = next(iter(obj))
    return isinstance(key, str)


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
    k_inf = int(k) if k is not None else _infer_k_from_schemata(schemata)
    exact_k_max = int(exact_k_max)
    use_exact = minterms is None and k_inf <= exact_k_max
    if use_exact:
        n = 1 << k_inf
    elif n_samples is None and minterms is None:
        n = _default_n_samples(exact_k_max)
    elif minterms is not None:
        n = len(minterms)
    else:
        n = int(n_samples)  # type: ignore[arg-type]

    mode = _redundancy_mode(k_inf, n, use_exact=use_exact)
    _warn_if_sampled(
        "input_redundancy",
        mode=mode,
        k=k_inf,
        exact_k_max=exact_k_max,
        n=n,
    )

    sampled = sample_pi_coverage(
        schemata, n, rng=rng, minterms=minterms, k=k_inf
    )
    estimate = input_redundancy_from_coverage(
        sampled.coverage, k_inf, operator=operator, norm=norm
    )
    return _pack_redundancy_result(
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
    k_inf = int(k) if k is not None else _infer_k_from_schemata(schemata)
    exact_k_max = int(exact_k_max)
    use_exact = minterms is None and k_inf <= exact_k_max
    if use_exact:
        n = 1 << k_inf
    elif n_samples is None and minterms is None:
        n = _default_n_samples(exact_k_max)
    elif minterms is not None:
        n = len(minterms)
    else:
        n = int(n_samples)  # type: ignore[arg-type]

    mode = _redundancy_mode(k_inf, n, use_exact=use_exact)
    _warn_if_sampled(
        "edge_redundancy",
        mode=mode,
        k=k_inf,
        exact_k_max=exact_k_max,
        n=n,
    )

    sampled = sample_pi_coverage(
        schemata, n, rng=rng, minterms=minterms, k=k_inf
    )
    estimate = edge_redundancy_from_coverage(
        sampled.coverage, k_inf, bound=bound
    )
    return _pack_redundancy_result(
        estimate,
        sampled,
        mode,
        return_coverage=return_coverage,
        return_mode=return_mode,
    )


def _percentile_ci(boot_stats: np.ndarray, ci: float) -> Tuple[float, float]:
    alpha = (1.0 - float(ci)) / 2.0
    lo = float(np.quantile(boot_stats, alpha))
    hi = float(np.quantile(boot_stats, 1.0 - alpha))
    return lo, hi


def _bootstrap_means_1d(
    contrib: np.ndarray,
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = contrib.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n), dtype=np.int64)
    return contrib[idx].mean(axis=1)


def _resolve_sampled(
    sampled: Optional[SampledPiCoverage],
    schemata: Optional[OneSymbolSchemata],
    *,
    n_samples: Optional[int],
    rng: Optional[np.random.Generator],
    minterms: Optional[Sequence[str]],
) -> SampledPiCoverage:
    if sampled is not None:
        return sampled
    if schemata is None:
        raise ValueError("provide sampled coverage or schemata")
    if n_samples is None and minterms is None:
        raise ValueError("provide n_samples, minterms, or sampled coverage")
    n = int(n_samples) if n_samples is not None else len(minterms)  # type: ignore[arg-type]
    return sample_pi_coverage(
        schemata, n, rng=rng, minterms=minterms
    )


def _as_sampled_or_schemata(
    sampled_or_schemata: Optional[Union[SampledPiCoverage, OneSymbolSchemata]],
    schemata: Optional[OneSymbolSchemata],
) -> Tuple[Optional[SampledPiCoverage], Optional[OneSymbolSchemata]]:
    """Accept SampledPiCoverage or schemata as the first positional arg."""
    if sampled_or_schemata is None:
        return None, schemata
    if isinstance(sampled_or_schemata, SampledPiCoverage):
        return sampled_or_schemata, schemata
    if schemata is not None:
        raise TypeError(
            "pass either a SampledPiCoverage or schemata, not both"
        )
    return None, sampled_or_schemata  # type: ignore[return-value]


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
