# %% [markdown]
# BBM high-*k* multi-method primes compare
#
# For every BBM update rule with ``k > 12``, time six ON/OFF prime-implicant
# backends (``TIMEOUT_S`` wall-clock kill each, process-group) and check that
# successful results match exactly.
#
# Methods (timed; all still logged):
#
# 1. **ROF** — syntactic ROF check + ``iter_on_off_primes`` (ON and OFF)
# 2. **Unate Espresso** — syntactic unate check + ``on_off_primes``
# 3. **CANA LUT + QM** — ``qm_on_off_primes`` (skipped when ``k > MAX_DENSE_LUT_K``)
# 4. **pyeda** — ``complete_sum_on_off_primes``
# 5. **pyboolnet** — ``pyboolnet_on_off_primes``
# 6. **BDD** — ``bdd_on_off_primes`` (skipped when ``k > MAX_DENSE_LUT_K``)
#
# Cohorts (disjoint); **plots** only show ROF / Unate Espresso / pyeda:
#
# - **rof** — syntactic ROF → plot ROF, Unate, pyeda
# - **unate** — syntactic unate but not ROF → plot Unate, pyeda
# - **rest** — not unate → plot pyeda
#
# Timing plots / cohort summaries treat ``timeout`` as ``TIMEOUT_S``.  Each
# cohort figure stacks **median** time with Q1–Q3 shading (top) and success
# rate (bottom) on a shared *k* axis; method color/marker are fixed.
#
# Set ``REGENERATE_FROM_LOG_ONLY = True`` to rebuild figures from the existing
# timing log without re-running the BBM sweep (does not overwrite that log).
#
# Output is teed to ``high_k/logs/bbm_high_k_primes_compare/``.

# %%
from __future__ import annotations

import re
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
from high_k.run_log import TeeLogger, run_log_dir, start_run_log
from high_k.timing import call_in_subprocess, fmt_s
from high_k.unate import check_syntactic_unate, on_off_primes

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MIN_K = 13  # keep rules with k > 12
TIMEOUT_S = 4.0  # 1.0
REPEATS = 1  # single wall-clock run (timeout kill)

# True → rebuild figures from the existing timing log (no re-timing; keeps log).
# False → full BBM sweep (overwrites the timing log).
REGENERATE_FROM_LOG_ONLY = True

NOTEBOOK_NAME = "bbm_high_k_primes_compare"
FIG_DIR = run_log_dir(NOTEBOOK_NAME)
TIMING_LOG_PATH = FIG_DIR / f"{NOTEBOOK_NAME}.log"

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

# Plots omit QM / pyboolnet / BDD (still timed and logged on a full run).
PLOT_METHOD_IDS = ("rof", "unate", "pyeda")

COHORT_METHODS = {
    "rof": ("rof", "unate", "pyeda"),
    "unate": ("unate", "pyeda"),
    "rest": ("pyeda",),
}

META_BY_ID = {m[0]: m for m in METHOD_META}

# %% [markdown]
# ## Helpers: log parse, freeze, timed runners, match


# %%
PrimeDict = Dict[str, bool]
OnOffPrimes = Tuple[List[PrimeDict], List[PrimeDict]]

_TIME_CELL = r"(?:skip|timeout|error|\d+(?:\.\d+)?\s*(?:µs|ms|s))"
_TIME_RE = re.compile(
    r"^(?:skip|timeout|error|(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>µs|ms|s))$"
)
_ROW_RE = re.compile(
    rf"^(?P<cohort>rof|unate|rest)\s+(?P<k>\d+)\s+"
    rf"(?P<rof>{_TIME_CELL})\s+(?P<unate>{_TIME_CELL})\s+"
    rf"(?P<qm>{_TIME_CELL})\s+(?P<pyeda>{_TIME_CELL})\s+"
    rf"(?P<pbn>{_TIME_CELL})\s+(?P<bdd>{_TIME_CELL})\s+"
    rf"(?P<n_on>\d+)\s+(?P<n_off>\d+)\s+(?P<match>\S+)\s+"
    rf"(?P<target>\S+)\s+/\s+(?P<model>.+?)\s*$"
)
_CFG_RE = re.compile(
    r"MIN_K=(?P<min_k>\d+)\s+TIMEOUT_S=(?P<timeout>\d+(?:\.\d+)?)"
)


def parse_time_cell(cell: str) -> Tuple[Optional[float], str]:
    """Parse a log timing cell → ``(seconds_or_None, status)``.

    Status is ``ok`` / ``timeout`` / ``error`` / ``skip``.
    """
    text = cell.strip()
    if text == "skip":
        return None, "skip"
    if text == "timeout":
        return None, "timeout"
    if text == "error":
        return None, "error"
    m = _TIME_RE.match(text)
    if m is None or m.group("num") is None:
        return None, "error"
    val = float(m.group("num"))
    unit = m.group("unit")
    if unit == "µs":
        return val * 1e-6, "ok"
    if unit == "ms":
        return val * 1e-3, "ok"
    return val, "ok"


def parse_timing_log(
    path: Path,
) -> Tuple[List[Dict[str, Any]], float, int]:
    """Rebuild timing rows from a prior ``bbm_high_k_primes_compare`` log.

    Returns ``(rows, timeout_s, min_k)``.
    """
    text = path.read_text(encoding="utf-8")
    timeout_s = float(TIMEOUT_S)
    min_k = int(MIN_K)
    for line in text.splitlines():
        m_cfg = _CFG_RE.search(line)
        if m_cfg:
            timeout_s = float(m_cfg.group("timeout"))
            min_k = int(m_cfg.group("min_k"))
            break

    rows: List[Dict[str, Any]] = []
    in_table = False
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if line.startswith("=== BBM"):
            in_table = True
            continue
        if not in_table:
            continue
        if line.startswith("==="):
            break
        if not line.strip() or line.lstrip().startswith("cohort"):
            continue
        if " FAILED:" in line or line.lstrip().startswith("ERROR"):
            continue
        m = _ROW_RE.match(line)
        if not m:
            continue
        cohort = m.group("cohort")
        k = int(m.group("k"))
        match_raw = m.group("match")
        if match_raw == "True":
            match: Optional[bool] = True
        elif match_raw == "False":
            match = False
        else:
            match = None
        row: Dict[str, Any] = {
            "source": "bbm",
            "k": k,
            "cohort": cohort,
            "is_rof": cohort == "rof",
            "is_unate": cohort in ("rof", "unate"),
            "model": m.group("model"),
            "target": m.group("target"),
            "label": f"{m.group('model')}::{m.group('target')}",
            "n_pi_on": int(m.group("n_on")),
            "n_pi_off": int(m.group("n_off")),
            "match": match,
            "mismatch_pairs": [],
            "n_ok": 0,
        }
        cell_by_mid = {
            "rof": m.group("rof"),
            "unate": m.group("unate"),
            "qm": m.group("qm"),
            "pyeda": m.group("pyeda"),
            "pyboolnet": m.group("pbn"),
            "bdd": m.group("bdd"),
        }
        n_ok = 0
        for mid, _t_key, _st_key, _lab, _mk, _col in METHOD_META:
            t, st = parse_time_cell(cell_by_mid[mid])
            row[f"t_{mid}"] = t
            row[f"st_{mid}"] = st
            row[f"err_{mid}"] = None
            if st == "ok":
                n_ok += 1
        row["n_ok"] = n_ok
        rows.append(row)
    return rows, timeout_s, min_k


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
# ## Collect (or reload from log)


# %%
if REGENERATE_FROM_LOG_ONLY:
    if not TIMING_LOG_PATH.is_file():
        raise FileNotFoundError(
            f"REGENERATE_FROM_LOG_ONLY set but missing {TIMING_LOG_PATH}"
        )
    rows, TIMEOUT_S, MIN_K = parse_timing_log(TIMING_LOG_PATH)
    # Do not truncate the timing log; tee replot notes alongside it.
    LOG_PATH = FIG_DIR / f"{NOTEBOOK_NAME}.replot.log"
    log = TeeLogger(LOG_PATH)
    log._install_warning_tee()
    log.info(f"log_path={LOG_PATH}")
    log.info(
        f"REGENERATE_FROM_LOG_ONLY=True  source={TIMING_LOG_PATH}  "
        f"rows={len(rows)}  MIN_K={MIN_K}  TIMEOUT_S={TIMEOUT_S}"
    )
    log.info(f"FIG_DIR={FIG_DIR}  plot_methods={PLOT_METHOD_IDS}")
    cohort_counts = {"rof": 0, "unate": 0, "rest": 0}
    for r in rows:
        cohort_counts[str(r["cohort"])] += 1
else:
    log, LOG_PATH = start_run_log(NOTEBOOK_NAME)
    log.info(
        f"MIN_K={MIN_K}  TIMEOUT_S={TIMEOUT_S}  REPEATS={REPEATS}  "
        f"MAX_DENSE_LUT_K={MAX_DENSE_LUT_K}"
    )
    log.info(f"BBM_MODELS={BBM_MODELS}  FIG_DIR={FIG_DIR}")
    log.info(f"plot_methods={PLOT_METHOD_IDS}")

    for _mod in ("pyeda", "pyboolnet", "dd"):
        try:
            __import__(_mod)
            log.info(f"optional dep {_mod}: available")
        except ImportError:
            log.info(f"optional dep {_mod}: MISSING (method will record error)")

    rows = []
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

mismatches = [r for r in rows if r.get("match") is False]

# %% [markdown]
# ## Match summary + status tallies (full runs only)


# %%
if not REGENERATE_FROM_LOG_ONLY:
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
# ## Medians by cohort and *k*


# %%
def _censored_time(row: Dict[str, Any], mid: str) -> Optional[float]:
    """Wall time for plots: ``ok`` → measured; ``timeout`` → TIMEOUT_S.

    ``timeout`` is treated as the kill limit.  ``skip`` / ``error`` excluded.
    """
    st = str(row[f"st_{mid}"])
    if st == "ok":
        t = row[f"t_{mid}"]
        return float(t) if isinstance(t, (int, float)) else None
    if st == "timeout":
        return float(TIMEOUT_S)
    return None


def _median_or_none(vals: List[object]) -> Optional[float]:
    nums = [float(v) for v in vals if isinstance(v, (int, float))]
    if not nums:
        return None
    return float(statistics.median(nums))


def _quartiles(vals: List[float]) -> Tuple[float, float, float]:
    """Return ``(median, q1, q3)``; single-sample → all equal that value."""
    if len(vals) == 1:
        v = float(vals[0])
        return v, v, v
    if len(vals) == 2:
        lo, hi = sorted(float(v) for v in vals)
        med = float(statistics.median(vals))
        return med, lo, hi
    q1, q2, q3 = statistics.quantiles(vals, n=4, method="inclusive")
    return float(q2), float(q1), float(q3)


log.info("")
log.info(
    "=== medians by (cohort, k)  "
    f"(timeout counted as {TIMEOUT_S:g}s) ==="
)
log.info(
    f"{'cohort':7s} {'k':>3s} {'n':>4s}  "
    f"{'rof':>10s} {'unate':>10s} {'pyeda':>10s}"
)

grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
for r in rows:
    grouped[(str(r["cohort"]), int(r["k"]))].append(r)

for key in sorted(grouped.keys(), key=lambda x: (x[0], x[1])):
    cohort, k = key
    group = grouped[key]
    log.info(
        f"{cohort:7s} {k:3d} {len(group):4d}  "
        f"{fmt_s(_median_or_none([_censored_time(g, 'rof') for g in group])):>10s} "
        f"{fmt_s(_median_or_none([_censored_time(g, 'unate') for g in group])):>10s} "
        f"{fmt_s(_median_or_none([_censored_time(g, 'pyeda') for g in group])):>10s}"
    )

log.info(f"\n{len(rows)} timing rows  |  wrote {LOG_PATH}")

# %% [markdown]
# ## Plots: median time (Q1–Q3) + success vs *k* by cohort


# %%
def _time_series_by_k(
    subset: List[Dict[str, Any]], mid: str
) -> Tuple[List[int], List[float], List[float], List[float]]:
    """Per-*k* median / Q1 / Q3 of censored times (timeout → TIMEOUT_S)."""
    by_k: Dict[int, List[float]] = defaultdict(list)
    for r in subset:
        t = _censored_time(r, mid)
        if t is not None:
            by_k[int(r["k"])].append(t)
    ks = sorted(by_k)
    meds: List[float] = []
    q1s: List[float] = []
    q3s: List[float] = []
    for k in ks:
        med, q1, q3 = _quartiles(by_k[k])
        meds.append(med)
        q1s.append(q1)
        q3s.append(q3)
    return ks, meds, q1s, q3s


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
    """Stacked panels: median±IQR time (top) and success rate (bottom)."""
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
        ks_t, meds, q1s, q3s = _time_series_by_k(subset, mid)
        if ks_t:
            any_series = True
            ax_t.fill_between(
                ks_t, q1s, q3s, color=color, alpha=0.22, linewidth=0
            )
            ax_t.plot(ks_t, meds, label=label, **style)
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
    ax_t.set_ylabel("median time (s)  [timeout → limit; band = Q1–Q3]")
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
    f"done  |  {len(rows)} rows  |  mismatches={len(mismatches)}  "
    f"REGENERATE_FROM_LOG_ONLY={REGENERATE_FROM_LOG_ONLY}"
)
log.close()
