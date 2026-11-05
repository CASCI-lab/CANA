# -*- coding: utf-8 -*-
"""
Sampling
========

Shared exact / sampled prime-implicant coverage helpers for function measures
(redundancy, bias, sensitivity).

For ``k <= exact_k_max`` (default 14) averages are over the full
``2**k`` space. For larger ``k``, minterms are sampled (default
``n_samples = 2**exact_k_max``). Bootstrap helpers resample contributions from
a preserved :class:`SampledPiCoverage` without re-matching primes.
"""
from __future__ import annotations

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
    "PiCoverage",
    "pi_covers",
    "covering_primes",
    "covering_primes_with_polarity",
    "sample_binstates",
    "sample_pi_coverage",
    "resolve_exact_or_sampled",
]

EXACT_K_MAX_DEFAULT = 14
WILDCARD = "2"

OneSymbolSchemata = Mapping[int, Set[str]]
PiCoverage = Mapping[str, Sequence[str]]
Operator = Callable[[Sequence[float]], float]


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
        # Lazy import: redundancy measure APIs depend on this module.
        from cana.function_properties.redundancy import (
            input_redundancy_contributions,
        )

        return input_redundancy_contributions(
            self.coverage, self.minterms, operator=operator
        )

    def edge_contributions(
        self, bound: str = "mean"
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        from cana.function_properties.redundancy import (
            edge_redundancy_contributions,
        )

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


def _coverage_mode(k: int, n: int, *, use_exact: bool) -> str:
    """Return ``\"exact\"`` if the average is over the full ``2**k`` space."""
    if use_exact:
        return "exact"
    # Full-space request via n_samples / minterms is mathematically exact.
    if k < 63 and n >= (1 << k):
        return "exact"
    return "sampled"


# Back-compat alias used by BooleanNode / older call sites.
_redundancy_mode = _coverage_mode


def _warn_if_sampled(
    what: str,
    *,
    mode: str,
    k: int,
    exact_k_max: int,
    n: int,
    stacklevel: int = 3,
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
        stacklevel=stacklevel,
    )


def _pack_coverage_result(
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


# Back-compat alias.
_pack_redundancy_result = _pack_coverage_result


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


def resolve_exact_or_sampled(
    schemata: OneSymbolSchemata,
    *,
    what: str,
    k: Optional[int] = None,
    exact_k_max: int = EXACT_K_MAX_DEFAULT,
    n_samples: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    minterms: Optional[Sequence[str]] = None,
) -> Tuple[SampledPiCoverage, str, int]:
    """Run the exact / sampled two-track; return ``(coverage, mode, k)``.

    Same policy as the measure front-ends: exact when
    ``k <= exact_k_max`` and *minterms* is omitted; otherwise sample
    ``n_samples`` (default ``2**exact_k_max``) and optionally warn.
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

    mode = _coverage_mode(k_inf, n, use_exact=use_exact)
    # stacklevel=4: warn → here → resolve caller (measure) → external caller
    _warn_if_sampled(
        what,
        mode=mode,
        k=k_inf,
        exact_k_max=exact_k_max,
        n=n,
        stacklevel=4,
    )
    sampled = sample_pi_coverage(
        schemata, n, rng=rng, minterms=minterms, k=k_inf
    )
    return sampled, mode, k_inf


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


def _looks_like_pi_coverage(obj: Mapping) -> bool:
    """Heuristic: pi_coverage keys are binstates; schemata keys are 0/1."""
    if not obj:
        return False
    key = next(iter(obj))
    return isinstance(key, str)
