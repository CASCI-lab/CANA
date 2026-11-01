# -*- coding: utf-8 -*-
"""Monte Carlo estimates of input / edge redundancy from one-symbol primes.

Bypasses the Θ(2^k) ``pi_coverage`` dict used by
:meth:`cana.boolean_node.BooleanNode.input_redundancy` by sampling minterms,
matching covering prime implicants per sample, and averaging the same
per-minterm Φ contributions.  Bootstrap CIs resample those contributions
(without re-matching primes).
"""
from __future__ import annotations

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

from high_k.paths import ensure_cana_on_path

ensure_cana_on_path()

from cana.cutils import statenum_to_binstate  # noqa: E402
from cana.function_outputs.one_symbol import primes_to_one_symbol  # noqa: E402
from cana.primes.util import PrimeDict  # noqa: E402

__all__ = [
    "BootstrapResult",
    "OneSymbolSchemata",
    "pi_covers",
    "covering_primes",
    "sample_binstates",
    "schemata_from_on_off",
    "input_redundancy_contributions",
    "edge_redundancy_contributions",
    "estimate_input_redundancy",
    "estimate_edge_redundancy",
    "bootstrap_input_redundancy",
    "bootstrap_edge_redundancy",
]

WILDCARD = "2"
OneSymbolSchemata = Mapping[int, Set[str]]
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


def pi_covers(schema: str, binstate: str) -> bool:
    """Return True if one-symbol *schema* covers binary *binstate*."""
    if len(schema) != len(binstate):
        raise ValueError(
            f"schema length {len(schema)} != binstate length {len(binstate)}"
        )
    return all(s == WILDCARD or s == b for s, b in zip(schema, binstate))


def covering_primes(binstate: str, schemata: OneSymbolSchemata) -> List[str]:
    """Return one-symbol PIs covering *binstate* (ON and OFF scanned).

    Raises:
        ValueError: if no prime covers *binstate* (incomplete cover).
    """
    covered: List[str] = []
    for polarity in (0, 1):
        for pi in schemata.get(polarity, ()) or ():
            if pi_covers(pi, binstate):
                covered.append(pi)
    if not covered:
        raise ValueError(
            f"no prime implicant covers binstate {binstate!r} "
            "(incomplete ON/OFF schemata?)"
        )
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
        raise ValueError(f"k must be >= 0, got {k}")
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
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
                    f"cannot sample n={n} without replacement from 2**k={space}"
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


def schemata_from_on_off(
    on_primes: Sequence[PrimeDict],
    off_primes: Sequence[PrimeDict],
    *,
    input_names: Optional[Sequence[str]] = None,
    k: Optional[int] = None,
) -> Dict[int, Set[str]]:
    """Convert named ON/OFF prime cubes to positional one-symbol schemata."""
    return primes_to_one_symbol(
        list(on_primes),
        list(off_primes),
        input_names=input_names,
        k=k,
    )


def _infer_k(schemata: OneSymbolSchemata) -> int:
    for polarity in (0, 1):
        for pi in schemata.get(polarity, ()) or ():
            return len(pi)
    raise ValueError("schemata contain no prime implicants")


def _resolve_minterms(
    schemata: OneSymbolSchemata,
    *,
    n_samples: Optional[int],
    rng: Optional[np.random.Generator],
    minterms: Optional[Sequence[str]],
) -> Tuple[List[str], int]:
    k = _infer_k(schemata)
    if minterms is not None:
        out = list(minterms)
        if out and any(len(m) != k for m in out):
            raise ValueError(f"all minterms must have length k={k}")
        return out, k
    if n_samples is None:
        raise ValueError("provide n_samples or minterms")
    return sample_binstates(k, int(n_samples), rng=rng, replace=False), k


def input_redundancy_contributions(
    schemata: OneSymbolSchemata,
    minterms: Sequence[str],
    operator: Operator = mean,
) -> np.ndarray:
    """Per-minterm Φ(#wildcards among covering PIs); shape ``(n,)``."""
    if not hasattr(operator, "__call__"):
        raise TypeError(
            'operator must be callable (e.g. statistics.mean, min, max)'
        )
    out = np.empty(len(minterms), dtype=np.float64)
    for i, binstate in enumerate(minterms):
        pis = covering_primes(binstate, schemata)
        out[i] = float(operator([pi.count(WILDCARD) for pi in pis]))
    return out


def edge_redundancy_contributions(
    schemata: OneSymbolSchemata,
    minterms: Sequence[str],
    bound: str = "mean",
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Per-minterm edge contributions.

    For ``bound`` in ``lower`` / ``mean`` / ``upper``: shape ``(n, k)``.
    For ``bound == "tuple"``: ``(lower, upper)`` each shape ``(n, k)``.
    """
    if not minterms:
        k = _infer_k(schemata)
        empty = np.zeros((0, k), dtype=np.float64)
        if bound == "tuple":
            return empty, empty.copy()
        return empty

    k = len(minterms[0])
    n = len(minterms)
    if bound in ("lower", "mean", "avg", "upper"):
        out = np.empty((n, k), dtype=np.float64)
        for row, binstate in enumerate(minterms):
            pis = covering_primes(binstate, schemata)
            for j in range(k):
                flags = [pi[j] == WILDCARD for pi in pis]
                if bound == "lower":
                    out[row, j] = float(all(flags))
                elif bound in ("mean", "avg"):
                    out[row, j] = float(sum(flags) / len(flags))
                else:  # upper
                    out[row, j] = float(any(flags))
        return out

    if bound == "tuple":
        lower = np.empty((n, k), dtype=np.float64)
        upper = np.empty((n, k), dtype=np.float64)
        for row, binstate in enumerate(minterms):
            pis = covering_primes(binstate, schemata)
            for j in range(k):
                flags = [pi[j] == WILDCARD for pi in pis]
                lower[row, j] = float(all(flags))
                upper[row, j] = float(any(flags))
        return lower, upper

    raise ValueError(
        'bound must be "lower", "mean", "avg", "upper", or "tuple"; '
        f"got {bound!r}"
    )


def estimate_input_redundancy(
    schemata: OneSymbolSchemata,
    *,
    n_samples: Optional[int] = None,
    operator: Operator = mean,
    norm: bool = True,
    rng: Optional[np.random.Generator] = None,
    minterms: Optional[Sequence[str]] = None,
) -> float:
    """Monte Carlo estimate of :math:`k_r` (optionally normalized by *k*)."""
    mt, k = _resolve_minterms(
        schemata, n_samples=n_samples, rng=rng, minterms=minterms
    )
    if not mt:
        raise ValueError("need at least one minterm to estimate")
    contrib = input_redundancy_contributions(schemata, mt, operator=operator)
    k_r = float(np.mean(contrib))
    if norm:
        if k == 0:
            raise ValueError("cannot normalize input redundancy for k=0")
        k_r /= k
    return k_r


def estimate_edge_redundancy(
    schemata: OneSymbolSchemata,
    *,
    n_samples: Optional[int] = None,
    bound: str = "mean",
    rng: Optional[np.random.Generator] = None,
    minterms: Optional[Sequence[str]] = None,
) -> Union[List[float], List[Tuple[float, float]]]:
    """Monte Carlo estimate of per-input :math:`r_i` (same bounds as BooleanNode)."""
    mt, _k = _resolve_minterms(
        schemata, n_samples=n_samples, rng=rng, minterms=minterms
    )
    if not mt:
        raise ValueError("need at least one minterm to estimate")
    contrib = edge_redundancy_contributions(schemata, mt, bound=bound)
    if bound == "tuple":
        lower, upper = contrib  # type: ignore[misc]
        lo = np.mean(lower, axis=0)
        hi = np.mean(upper, axis=0)
        return [(float(a), float(b)) for a, b in zip(lo, hi)]
    means = np.mean(contrib, axis=0)  # type: ignore[arg-type]
    return [float(x) for x in means]


def _percentile_ci(
    boot_stats: np.ndarray, ci: float
) -> Tuple[float, float]:
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
    # mean over axis 1 for each bootstrap replicate
    return contrib[idx].mean(axis=1)


def bootstrap_input_redundancy(
    schemata: OneSymbolSchemata,
    *,
    n_samples: int,
    n_boot: int = 1000,
    ci: float = 0.95,
    operator: Operator = mean,
    norm: bool = True,
    rng: Optional[np.random.Generator] = None,
    minterms: Optional[Sequence[str]] = None,
) -> BootstrapResult:
    """Estimate :math:`k_r` with a percentile bootstrap CI on the sample mean."""
    gen = rng if rng is not None else np.random.default_rng()
    mt, k = _resolve_minterms(
        schemata, n_samples=n_samples, rng=gen, minterms=minterms
    )
    if not mt:
        raise ValueError("need at least one minterm to bootstrap")
    n_boot = int(n_boot)
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    if not (0.0 < ci < 1.0):
        raise ValueError(f"ci must be in (0, 1), got {ci}")

    contrib = input_redundancy_contributions(schemata, mt, operator=operator)
    if norm:
        if k == 0:
            raise ValueError("cannot normalize input redundancy for k=0")
        contrib = contrib / k

    estimate = float(np.mean(contrib))
    boot = _bootstrap_means_1d(contrib, n_boot=n_boot, rng=gen)
    lo, hi = _percentile_ci(boot, ci)
    return BootstrapResult(
        estimate=estimate,
        ci_low=lo,
        ci_high=hi,
        boot_mean=float(np.mean(boot)),
        boot_std=float(np.std(boot, ddof=1)) if n_boot > 1 else 0.0,
        n_samples=len(mt),
        n_boot=n_boot,
        ci=float(ci),
    )


def bootstrap_edge_redundancy(
    schemata: OneSymbolSchemata,
    *,
    n_samples: int,
    n_boot: int = 1000,
    ci: float = 0.95,
    bound: str = "mean",
    rng: Optional[np.random.Generator] = None,
    minterms: Optional[Sequence[str]] = None,
) -> Union[List[BootstrapResult], List[Tuple[BootstrapResult, BootstrapResult]]]:
    """Estimate per-input :math:`r_i` with percentile bootstrap CIs.

    For ``bound == "tuple"``, returns ``[(lower_result, upper_result), ...]``.
    """
    gen = rng if rng is not None else np.random.default_rng()
    mt, _k = _resolve_minterms(
        schemata, n_samples=n_samples, rng=gen, minterms=minterms
    )
    if not mt:
        raise ValueError("need at least one minterm to bootstrap")
    n_boot = int(n_boot)
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    if not (0.0 < ci < 1.0):
        raise ValueError(f"ci must be in (0, 1), got {ci}")

    def _pack(col_contrib: np.ndarray, boot_cols: np.ndarray) -> List[BootstrapResult]:
        # col_contrib: (n,), boot_cols: (n_boot, k) — pack one column at a time
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
                    n_samples=len(mt),
                    n_boot=n_boot,
                    ci=float(ci),
                )
            )
        return results

    contrib = edge_redundancy_contributions(schemata, mt, bound=bound)
    n = len(mt)
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
