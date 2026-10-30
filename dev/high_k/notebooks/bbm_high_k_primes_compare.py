# %% [markdown]
# BBM high-*k* multi-method primes compare
#
# For every BBM update rule with ``k > 12``, time six ON/OFF prime-implicant
# backends (``TIMEOUT_S`` wall-clock kill each, process-group) and check that
# successful results match exactly.
#
# Methods:
#
# 1. **ROF** — syntactic ROF check + ``iter_on_off_primes`` (ON and OFF)
# 2. **Unate Espresso** — syntactic unate check + ``on_off_primes``
# 3. **CANA LUT + QM** — ``qm_on_off_primes`` (skipped when ``k > MAX_DENSE_LUT_K``)
# 4. **pyeda** — ``complete_sum_on_off_primes``
# 5. **pyboolnet** — ``pyboolnet_on_off_primes``
# 6. **BDD** — ``bdd_on_off_primes`` (skipped when ``k > MAX_DENSE_LUT_K``)
#
# Cohorts (disjoint) and plot series:
#
# - **rof** — syntactic ROF → methods 1–6
# - **unate** — syntactic unate but not ROF → methods 2–6
# - **rest** — not unate → methods 3–6
#
# Timing plots / cohort averages treat ``timeout`` as ``TIMEOUT_S`` (lower
# bound on the true mean).  Each cohort figure stacks mean time (top) and
# success rate (bottom) on a shared *k* axis; method color/marker are fixed.
# Per-row ``ERROR`` lines and an end-of-run error-message tally print the
# underlying exception text (previously only the word ``error``).
#
# Output is teed to ``high_k/logs/bbm_high_k_primes_compare/``.

# %%
from __future__ import annotations

import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# notebooks/ → high_k/ → dev/
_DEV_DIR = Path(__file__).resolve().parents[2]
if str(_DEV_DIR) not in sys.path:
    sys.path.insert(0, str(_DEV_DIR))

from high_k.bbm_rules import BBM_MODELS, iter_bnet_rules, unique_regulators
from high_k.general_primes import (
    MAX_DENSE_LUT_K,
    bdd_on_off_primes,
    complete_sum_on_off_primes,
    pyboolnet_on_off_primes,
    qm_on_off_primes,
)
from high_k.rof import check_syntactic_rof, iter_on_off_primes
from high_k.run_log import run_log_dir, start_run_log
from high_k.timing import call_in_subprocess, fmt_s
from high_k.unate import check_syntactic_unate, on_off_primes

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MIN_K = 13  # keep rules with k > 12
TIMEOUT_S = 4.0  # 1.0
REPEATS = 1  # single wall-clock run (timeout kill)

NOTEBOOK_NAME = "bbm_high_k_primes_compare"
FIG_DIR = run_log_dir(NOTEBOOK_NAME)

log, LOG_PATH = start_run_log(NOTEBOOK_NAME)
log.info(
    f"MIN_K={MIN_K}  TIMEOUT_S={TIMEOUT_S}  REPEATS={REPEATS}  "
    f"MAX_DENSE_LUT_K={MAX_DENSE_LUT_K}"
)
log.info(f"BBM_MODELS={BBM_MODELS}  FIG_DIR={FIG_DIR}")

for _mod in ("pyeda", "pyboolnet", "dd"):
    try:
        __import__(_mod)
        log.info(f"optional dep {_mod}: available")
    except ImportError:
        log.info(f"optional dep {_mod}: MISSING (method will record error)")

# Method id → (time key, status key, plot label, marker, color).
# Color/marker are fixed so every cohort figure uses the same style per method.
METHOD_META: Tuple[Tuple[str, str, str, str, str, str], ...] = (
    ("rof", "t_rof", "st_rof", "ROF generate", "o", "#1f77b4"),
    ("unate", "t_unate", "st_unate", "Unate Espresso", "s", "#ff7f0e"),
    ("qm", "t_qm", "st_qm", "CANA LUT+QM", "^", "#2ca02c"),
    ("pyeda", "t_pyeda", "st_pyeda", "pyeda complete_sum", "D", "#d62728"),
    ("pyboolnet", "t_pyboolnet", "st_pyboolnet", "pyboolnet", "v", "#9467bd"),
    ("bdd", "t_bdd", "st_bdd", "BDD+QM", "P", "#8c564b"),
)

# %% [markdown]
# ## Helpers: freeze, timed runners, match


# %%
PrimeDict = Dict[str, bool]
OnOffPrimes = Tuple[List[PrimeDict], List[PrimeDict]]


def _freeze(prime_list: List[PrimeDict]) -> frozenset:
    return frozenset(frozenset(p.items()) for p in prime_list)


def _freeze_on_off(on_off: OnOffPrimes) -> Tuple[frozenset, frozenset]:
    return _freeze(on_off[0]), _freeze(on_off[1])


def _rof_path_inprocess(expr: str) -> OnOffPrimes:
    """Syntactic ROF check + enumerate ON and OFF primes."""
    info = check_syntactic_rof(expr)
    if not info["is_rof"]:
        raise ValueError("expression is not syntactically ROF")
    on_p = list(iter_on_off_primes(expr, "on"))
    off_p = list(iter_on_off_primes(expr, "off"))
    return on_p, off_p


def _rof_worker(expr: str, conn) -> None:
    try:
        conn.send(("ok", _rof_path_inprocess(expr)))
    except Exception as exc:  # noqa: BLE001
        conn.send(("err", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def run_rof_path(expr: str, *, timeout_s: float = TIMEOUT_S) -> OnOffPrimes:
    """Timed ROF check + generate under a killable subprocess."""
    if timeout_s <= 0:
        return _rof_path_inprocess(expr)
    return call_in_subprocess(
        _rof_worker, (expr.strip(),), timeout_s=timeout_s
    )


def _timed_call(
    fn: Callable[[], OnOffPrimes],
) -> Tuple[Optional[float], Optional[OnOffPrimes], str, Optional[str]]:
    """Return ``(seconds, result, status, err_msg)``.

    *status* is ``ok`` / ``timeout`` / ``error``.
    """
    t0 = time.perf_counter()
    try:
        result = fn()
        return time.perf_counter() - t0, result, "ok", None
    except TimeoutError as exc:
        return None, None, "timeout", str(exc)
    except Exception as exc:  # noqa: BLE001
        return None, None, "error", f"{type(exc).__name__}: {exc}"


def cohort_of(*, is_rof: bool, is_unate: bool) -> str:
    if is_rof:
        return "rof"
    if is_unate:
        return "unate"
    return "rest"


def time_all_methods(
    expr: str,
    *,
    k: int,
    is_rof: bool,
    is_unate: bool,
) -> Dict[str, Any]:
    """Run applicable methods once each; return timing + equality fields."""
    results: Dict[str, Optional[OnOffPrimes]] = {}
    times: Dict[str, Optional[float]] = {}
    statuses: Dict[str, str] = {}
    errors: Dict[str, Optional[str]] = {}

    # Predictable Θ(2^k) RAM: skip dense LUT QM and onset→QM BDD entirely.
    dense_ok = k <= MAX_DENSE_LUT_K

    runners: Dict[str, Optional[Callable[[], OnOffPrimes]]] = {
        "rof": (
            (lambda: run_rof_path(expr, timeout_s=TIMEOUT_S))
            if is_rof
            else None
        ),
        "unate": (
            (lambda: on_off_primes(expr, timeout_s=TIMEOUT_S))
            if is_unate
            else None
        ),
        "qm": (
            (lambda: qm_on_off_primes(expr, timeout_s=TIMEOUT_S))
            if dense_ok
            else None
        ),
        "pyeda": lambda: complete_sum_on_off_primes(
            expr, timeout_s=TIMEOUT_S
        ),
        "pyboolnet": lambda: pyboolnet_on_off_primes(
            expr, timeout_s=TIMEOUT_S
        ),
        "bdd": (
            (lambda: bdd_on_off_primes(expr, timeout_s=TIMEOUT_S))
            if dense_ok
            else None
        ),
    }

    for mid, runner in runners.items():
        if runner is None:
            times[mid] = None
            results[mid] = None
            statuses[mid] = "skip"
            errors[mid] = None
            continue
        t, res, st, err = _timed_call(runner)
        times[mid] = t
        results[mid] = res
        statuses[mid] = st
        errors[mid] = err

    ok_ids = [mid for mid, st in statuses.items() if st == "ok"]
    match: Optional[bool] = None
    mismatch_pairs: List[Tuple[str, str]] = []
    if len(ok_ids) >= 2:
        ref = _freeze_on_off(results[ok_ids[0]])  # type: ignore[arg-type]
        match = True
        for mid in ok_ids[1:]:
            other = _freeze_on_off(results[mid])  # type: ignore[arg-type]
            if other != ref:
                match = False
                mismatch_pairs.append((ok_ids[0], mid))
    elif len(ok_ids) == 1:
        match = True  # nothing to disagree with

    n_on = n_off = 0
    if ok_ids:
        on_p, off_p = results[ok_ids[0]]  # type: ignore[misc]
        n_on = len(on_p)
        n_off = len(off_p)

    out: Dict[str, Any] = {
        "n_pi_on": n_on,
        "n_pi_off": n_off,
        "match": match,
        "mismatch_pairs": mismatch_pairs,
        "n_ok": len(ok_ids),
    }
    for mid, t_key, st_key, _label, _marker, _color in METHOD_META:
        out[t_key] = times[mid]
        out[st_key] = statuses[mid]
        out[f"err_{mid}"] = errors[mid]
    return out


# %% [markdown]
# ## Collect: BBM rules with *k* > 12


# %%
rows: List[Dict[str, Any]] = []

bbm_seen = 0
bbm_kept = 0
bbm_failed = 0
cohort_counts = {"rof": 0, "unate": 0, "rest": 0}

log.info("")
log.info(f"=== BBM (all rules with k > {MIN_K - 1}) ===")
if not BBM_MODELS.is_dir():
    log.info(f"BBM models not found at {BBM_MODELS}")
else:
    header = (
        f"{'cohort':7s} {'k':>3s}  "
        f"{'rof':>10s} {'unate':>10s} {'qm':>10s} "
        f"{'pyeda':>10s} {'pbn':>10s} {'bdd':>10s}  "
        f"{'#on':>5s} {'#off':>5s}  match  target / model"
    )
    log.info(header)
    for model_name, target, expr in iter_bnet_rules(BBM_MODELS):
        bbm_seen += 1
        k = len(unique_regulators(expr))
        if k < MIN_K:
            continue
        bbm_kept += 1

        is_rof = bool(check_syntactic_rof(expr)["is_rof"])
        is_unate = bool(check_syntactic_unate(expr)["is_unate"])
        cohort = cohort_of(is_rof=is_rof, is_unate=is_unate)
        cohort_counts[cohort] += 1

        short_model = model_name
        if len(short_model) > 40:
            short_model = short_model[:37] + "..."

        try:
            r = time_all_methods(
                expr, k=k, is_rof=is_rof, is_unate=is_unate
            )
        except Exception as exc:  # noqa: BLE001
            bbm_failed += 1
            log.info(
                f"{cohort:7s} {k:3d}  FAILED: {exc}  "
                f"{target} / {short_model}"
            )
            continue

        r["source"] = "bbm"
        r["k"] = k
        r["cohort"] = cohort
        r["is_rof"] = is_rof
        r["is_unate"] = is_unate
        r["model"] = model_name
        r["target"] = target
        r["label"] = f"{model_name}::{target}"
        r["expr"] = expr
        rows.append(r)

        def _cell(mid: str) -> str:
            st = r[f"st_{mid}"]
            if st == "skip":
                return f"{'skip':>10s}"
            if st == "timeout":
                return f"{'timeout':>10s}"
            if st == "error":
                return f"{'error':>10s}"
            return f"{fmt_s(r[f't_{mid}']):>10s}"

        log.info(
            f"{cohort:7s} {k:3d}  "
            f"{_cell('rof')} {_cell('unate')} {_cell('qm')} "
            f"{_cell('pyeda')} {_cell('pyboolnet')} {_cell('bdd')}  "
            f"{r['n_pi_on']:5d} {r['n_pi_off']:5d}  "
            f"{r['match']!s:5s}  {target} / {short_model}"
        )
        # Surface exception text (truncated) — previously only the word "error".
        err_parts: List[str] = []
        for mid, _t, _st, label, _m, _c in METHOD_META:
            msg = r.get(f"err_{mid}")
            if r.get(f"st_{mid}") != "error" or not msg:
                continue
            short = msg if len(msg) <= 160 else msg[:157] + "..."
            err_parts.append(f"{label}: {short}")
        for part in err_parts:
            log.info(f"         ERROR  {part}")

log.info("")
log.info(
    f"BBM: seen={bbm_seen}  kept_k_gt_12={bbm_kept}  "
    f"failed={bbm_failed}  "
    f"cohorts rof={cohort_counts['rof']}  "
    f"unate={cohort_counts['unate']}  rest={cohort_counts['rest']}"
)

# %% [markdown]
# ## Match summary + status tallies


# %%
mismatches = [r for r in rows if r.get("match") is False]
log.info("")
log.info("=== exact ON/OFF prime match (among successful methods) ===")
if mismatches:
    log.info(f"{len(mismatches)} mismatches:")
    for r in mismatches[:40]:
        pairs = r.get("mismatch_pairs") or []
        log.info(
            f"  {r['cohort']} k={r['k']}  pairs={pairs}  "
            f"{r['target']} / {r['model']}"
        )
    if len(mismatches) > 40:
        log.info(f"  ... and {len(mismatches) - 40} more")
else:
    log.info("all compared successful method sets match")

log.info("")
log.info("=== status tallies by method ===")
for mid, _t, st_key, label, _m, _c in METHOD_META:
    counts: Dict[str, int] = defaultdict(int)
    for r in rows:
        counts[str(r[st_key])] += 1
    parts = "  ".join(f"{st}={n}" for st, n in sorted(counts.items()))
    log.info(f"  {label:22s}  {parts}")

log.info("")
log.info("=== error message tallies by method ===")
for mid, _t, st_key, label, _m, _c in METHOD_META:
    kind_counts: Dict[str, int] = defaultdict(int)
    for r in rows:
        if str(r[st_key]) != "error":
            continue
        msg = str(r.get(f"err_{mid}") or "")
        # Bucket by leading ``Type:`` / first clause so nesting RecursionErrors
        # and signal-kills collapse cleanly.
        if "RecursionError" in msg:
            kind = "RecursionError (nesting/parse)"
        elif "too many nested parentheses" in msg:
            kind = "SyntaxError: too many nested parentheses"
        elif "Parser stack overflowed" in msg:
            kind = "MemoryError: parser stack overflow"
        elif "refusing dense LUT" in msg:
            kind = "ValueError: MAX_DENSE_LUT_K"
        elif "killed by signal" in msg:
            kind = "TimeoutError-like: killed by signal"
        elif "exited without result" in msg:
            kind = "RuntimeError: exited without result"
        elif ":" in msg:
            kind = msg.split(":", 1)[0].strip()
            rest = msg.split(":", 1)[1].strip()
            if rest:
                # Keep a short discriminator (e.g. KeyError vs TypeError body).
                head = rest.split(":", 1)[0].strip()
                if head and head != rest and len(head) < 60:
                    kind = f"{kind}: {head}"
        else:
            kind = msg[:80] if msg else "(empty)"
        kind_counts[kind] += 1
    if not kind_counts:
        continue
    log.info(f"  {label}:")
    for kind, n in sorted(kind_counts.items(), key=lambda x: (-x[1], x[0])):
        log.info(f"    {n:4d}  {kind}")

# %% [markdown]
# ## Averages by cohort and *k*


# %%
def _censored_time(row: Dict[str, Any], mid: str) -> Optional[float]:
    """Wall time for means/plots: ``ok`` → measured; ``timeout`` → TIMEOUT_S.

    ``timeout`` is treated as a lower bound (right-censored at the kill
    limit).  ``skip`` / ``error`` are excluded.
    """
    st = str(row[f"st_{mid}"])
    if st == "ok":
        t = row[f"t_{mid}"]
        return float(t) if isinstance(t, (int, float)) else None
    if st == "timeout":
        return float(TIMEOUT_S)
    return None


def _mean_or_none(vals: List[object]) -> Optional[float]:
    nums = [float(v) for v in vals if isinstance(v, (int, float))]
    if not nums:
        return None
    return float(statistics.mean(nums))


log.info("")
log.info(
    "=== averages by (cohort, k)  "
    f"(timeout counted as {TIMEOUT_S:g}s lower bound) ==="
)
log.info(
    f"{'cohort':7s} {'k':>3s} {'n':>4s}  "
    f"{'rof':>10s} {'unate':>10s} {'qm':>10s} "
    f"{'pyeda':>10s} {'pbn':>10s} {'bdd':>10s}"
)

grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
for r in rows:
    grouped[(str(r["cohort"]), int(r["k"]))].append(r)

for key in sorted(grouped.keys(), key=lambda x: (x[0], x[1])):
    cohort, k = key
    group = grouped[key]
    log.info(
        f"{cohort:7s} {k:3d} {len(group):4d}  "
        f"{fmt_s(_mean_or_none([_censored_time(g, 'rof') for g in group])):>10s} "
        f"{fmt_s(_mean_or_none([_censored_time(g, 'unate') for g in group])):>10s} "
        f"{fmt_s(_mean_or_none([_censored_time(g, 'qm') for g in group])):>10s} "
        f"{fmt_s(_mean_or_none([_censored_time(g, 'pyeda') for g in group])):>10s} "
        f"{fmt_s(_mean_or_none([_censored_time(g, 'pyboolnet') for g in group])):>10s} "
        f"{fmt_s(_mean_or_none([_censored_time(g, 'bdd') for g in group])):>10s}"
    )

log.info(f"\n{len(rows)} timing rows  |  wrote {LOG_PATH}")

# %% [markdown]
# ## Plots: time + success vs *k* by cohort (shared style)


# %%
COHORT_METHODS = {
    "rof": ("rof", "unate", "qm", "pyeda", "pyboolnet", "bdd"),
    "unate": ("unate", "qm", "pyeda", "pyboolnet", "bdd"),
    "rest": ("qm", "pyeda", "pyboolnet", "bdd"),
}

META_BY_ID = {m[0]: m for m in METHOD_META}


def _time_series_by_k(
    subset: List[Dict[str, Any]], mid: str
) -> Tuple[List[int], List[float]]:
    """Mean censored time per *k* (timeout → TIMEOUT_S lower bound)."""
    by_k: Dict[int, List[float]] = defaultdict(list)
    for r in subset:
        t = _censored_time(r, mid)
        if t is not None:
            by_k[int(r["k"])].append(t)
    ks = sorted(by_k)
    return ks, [statistics.mean(by_k[k]) for k in ks]


def _success_series_by_k(
    subset: List[Dict[str, Any]], mid: str
) -> Tuple[List[int], List[float]]:
    """Fraction ``ok`` among attempted runs (excludes ``skip``) per *k*."""
    by_k: Dict[int, List[bool]] = defaultdict(list)
    st_key = META_BY_ID[mid][2]
    for r in subset:
        st = str(r[st_key])
        if st == "skip":
            continue
        by_k[int(r["k"])].append(st == "ok")
    ks = sorted(by_k)
    return ks, [sum(by_k[k]) / len(by_k[k]) for k in ks]


def plot_time_and_success_vs_k(
    subset: List[Dict[str, Any]],
    method_ids: Tuple[str, ...],
    title: str,
    outfile: Path,
) -> None:
    """Stacked panels: mean time (top) and success rate (bottom), shared *k*."""
    fig, (ax_t, ax_s) = plt.subplots(
        2,
        1,
        figsize=(7.5, 7.2),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.15, 1.0]},
    )
    any_series = False
    for mid in method_ids:
        _mid, _t_key, _st, label, marker, color = META_BY_ID[mid]
        style = dict(marker=marker, linestyle="-", color=color)
        ks_t, ts = _time_series_by_k(subset, mid)
        if ks_t:
            any_series = True
            ax_t.plot(ks_t, ts, label=label, **style)
        ks_s, rates = _success_series_by_k(subset, mid)
        if ks_s:
            any_series = True
            # Legend lives on the time panel only.
            ax_s.plot(ks_s, rates, **style)

    ax_t.axhline(
        TIMEOUT_S,
        color="0.5",
        linestyle="--",
        linewidth=1.0,
        label=f"timeout={TIMEOUT_S:g}s",
    )
    ax_t.set_yscale("log")
    ax_t.set_ylabel("time (s)  [timeout → lower bound]")
    ax_t.set_title(title)
    if any_series:
        ax_t.legend(fontsize=8)
    ax_t.grid(True, which="both", alpha=0.3)

    ax_s.set_xscale("log", base=2)
    ax_s.set_ylim(-0.05, 1.05)
    ax_s.set_xlabel("k (inputs)")
    ax_s.set_ylabel("success rate (ok / attempted)")
    ax_s.grid(True, which="both", alpha=0.3)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    log.info(f"wrote {outfile}")


for cohort, method_ids in COHORT_METHODS.items():
    subset = [r for r in rows if r["cohort"] == cohort]
    if not subset:
        log.info(f"no rows for cohort={cohort}; skipping plot")
        continue
    plot_time_and_success_vs_k(
        subset,
        method_ids,
        title=(
            f"Prime implicants — BBM k>{MIN_K - 1} "
            f"/ {cohort} (n={len(subset)})"
        ),
        outfile=FIG_DIR / f"time_success_vs_k__{cohort}.png",
    )

log.info(
    f"done  |  {len(rows)} rows  |  mismatches={len(mismatches)}"
)
log.close()
