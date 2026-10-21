# %% [markdown]
# All-LUT Canalizing Map ↔ LUT roundtrip
#
# Enumerate Boolean look-up tables for every `k <= MAX_K` (default 5), build a
# canalizing map via `BooleanNode.canalizing_map`, reconstruct the LUT with
# `cm_lut.lut_from_canalizing_map`, and report fail fractions by `k`.
#
# * **Exhaustive** when `2^(2^k) <= SAMPLE_N` (default 2048 → full enum for
#   `k <= 3`).
# * **Sample** exactly `SAMPLE_N` uniform random LUTs whenever the full space
#   is larger (e.g. `k = 4` → 65536, `k = 5` → `2^32`).
#
# Set `SMOKE = True` to cap at `SMOKE_MAX_K` (default).  Logs go under
# `dev/logs/`; later cells parse the log, print stats, and write figures.

# %%
from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_DEV_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _DEV_DIR.parent  # .../CANA/CANA (contains the cana package)
for _p in (_DEV_DIR, _PKG_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cana.boolean_node import BooleanNode

from bbm_cm_roundtrip import (  # noqa: E402
    TeeLogger,
    _fmt_secs,
    _timing_suffix,
    roundtrip_node,
)
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SMOKE = False
SMOKE_MAX_K = 2
# Enumerate all LUTs for k = 1 .. MAX_K (smoke overrides to SMOKE_MAX_K).
MAX_K = 5
# Exhaustive if 2^(2^k) <= SAMPLE_N; otherwise always sample SAMPLE_N LUTs.
SAMPLE_N = 2048
SAMPLE_SEED = 0
# Wall-clock limits per function step; skip on expiry. None / <=0 disables.
# Default off: subprocess timeouts dominate cost when enumerating many LUTs.
TIMEOUT_CM_S: Optional[float] = None
TIMEOUT_LUT_S: Optional[float] = None
# Progress line every N evaluated/skipped functions within a k-bucket.
PROGRESS_EVERY = 256

LOG_DIR = _DEV_DIR / "logs"

_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
_LOG_NAME = (
    f"cm_all_lut_roundtrip_smoke_{_TIMESTAMP}.log"
    if SMOKE
    else f"cm_all_lut_roundtrip_{_TIMESTAMP}.log"
)
LOG_PATH = LOG_DIR / _LOG_NAME
FIG_DIR = LOG_DIR / f"{Path(_LOG_NAME).stem}_figures"

# %% [markdown]
# ## Helpers — LUT enumeration + per-k aggregation


# %%
def n_boolean_functions(k: int) -> int:
    """Number of Boolean functions of *k* inputs: ``2**(2**k)``."""
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    return 1 << (1 << k)


def outputs_from_func_id(k: int, func_id: int) -> List[int]:
    """Decode *func_id* as a length-``2**k`` output list (LSB = state 0)."""
    n_rows = 1 << k
    return [(func_id >> i) & 1 for i in range(n_rows)]


def lut_name(k: int, func_id: int) -> str:
    n_rows = 1 << k
    bits = "".join(str((func_id >> i) & 1) for i in range(n_rows))
    return f"k{k}_id{func_id}_{bits}"


def _sample_ids(total: int, n: int, rng: np.random.Generator) -> List[int]:
    """Uniform sample of *n* distinct ids in ``[0, total)`` without building the range."""
    n = min(n, total)
    # np.choice(arange(total)) is fine for modest spaces; for 2^32 use rejection.
    if total <= 10_000_000:
        return sorted(int(x) for x in rng.choice(total, size=n, replace=False))
    ids: set[int] = set()
    while len(ids) < n:
        ids.add(int(rng.integers(0, total)))
    return sorted(ids)


def iter_func_ids(k: int, sample_n: int, seed: int) -> Tuple[str, List[int]]:
    """Return ``(mode, func_ids)`` for one *k*.

    mode is ``exhaustive`` when the full space fits in *sample_n*, else ``sample``.
    """
    total = n_boolean_functions(k)
    if total <= sample_n:
        return "exhaustive", list(range(total))
    rng = np.random.default_rng(seed + k)
    return "sample", _sample_ids(total, sample_n, rng)


def make_node(k: int, func_id: int) -> BooleanNode:
    outputs = outputs_from_func_id(k, func_id)
    inputs = list(range(k))
    return BooleanNode.from_output_list(
        outputs, name=lut_name(k, func_id), inputs=inputs, id=func_id
    )


@dataclass
class KBucketResult:
    k: int
    mode: str  # exhaustive | sample
    n_total_space: int
    n_planned: int
    n_eval: int = 0
    n_fail: int = 0
    n_skip: int = 0
    n_error: int = 0
    cm_total_s: float = 0.0
    lut_total_s: float = 0.0


def effective_max_k(smoke: bool, max_k: int, smoke_max_k: int) -> int:
    return min(max_k, smoke_max_k) if smoke else max_k


# %% [markdown]
# ## Run enumeration (smoke or full)
#
# Call `main()` from a notebook cell, or run the file as a script.


# %%
def main() -> Path:
    log = TeeLogger(LOG_PATH)

    mode = "smoke" if SMOKE else "full"
    k_hi = effective_max_k(SMOKE, MAX_K, SMOKE_MAX_K)
    log.info(f"mode={mode}")
    if SMOKE:
        log.info(f"smoke_max_k={SMOKE_MAX_K}  (effective MAX_K override)")
    log.info(f"MAX_K={MAX_K}")
    log.info(f"effective_max_k={k_hi}")
    log.info(f"SAMPLE_N={SAMPLE_N}")
    log.info(f"SAMPLE_SEED={SAMPLE_SEED}")
    log.info(f"TIMEOUT_CM_S={TIMEOUT_CM_S}")
    log.info(f"TIMEOUT_LUT_S={TIMEOUT_LUT_S}")
    log.info(f"PROGRESS_EVERY={PROGRESS_EVERY}")
    log.info(f"log_path={LOG_PATH}")
    log.info(f"timestamp={_TIMESTAMP}")
    log.info("")

    buckets: List[KBucketResult] = []
    n_fn_eval = 0
    n_fn_fail = 0
    total_cm_s = 0.0
    total_lut_s = 0.0
    failing_ks: List[str] = []

    for k in range(1, k_hi + 1):
        space = n_boolean_functions(k)
        enum_mode, func_ids = iter_func_ids(k, sample_n=SAMPLE_N, seed=SAMPLE_SEED)
        br = KBucketResult(
            k=k,
            mode=enum_mode,
            n_total_space=space,
            n_planned=len(func_ids),
        )
        log.info(
            f"=== k={k}  mode={enum_mode}  space={space}  "
            f"planned={br.n_planned} ==="
        )

        t_k0 = time.perf_counter()
        for i, func_id in enumerate(func_ids):
            node = make_node(k, func_id)
            nr = roundtrip_node(
                node,
                max_k=k_hi,
                timeout_cm_s=TIMEOUT_CM_S,
                timeout_lut_s=TIMEOUT_LUT_S,
            )

            if nr.skipped:
                br.n_skip += 1
                log.info(
                    f"  SKIP  {nr.name}  k={nr.k}  ({nr.skip_reason})"
                    f"{_timing_suffix(nr)}"
                )
            elif nr.error is not None:
                br.n_error += 1
                log.info(
                    f"  ERROR {nr.name}  k={nr.k}  {nr.error}"
                    f"{_timing_suffix(nr)}"
                )
            else:
                br.n_eval += 1
                n_fn_eval += 1
                if nr.cm_s is not None:
                    br.cm_total_s += nr.cm_s
                    total_cm_s += nr.cm_s
                if nr.lut_s is not None:
                    br.lut_total_s += nr.lut_s
                    total_lut_s += nr.lut_s
                if not nr.exact:
                    br.n_fail += 1
                    n_fn_fail += 1
                    log.info(
                        f"  FAIL  {nr.name}  k={nr.k}  reason={nr.fail_reason}"
                        f"{_timing_suffix(nr)}"
                    )
                else:
                    # Keep logs readable: OK lines only every PROGRESS_EVERY,
                    # plus the first few in each bucket.
                    if i < 3 or (i + 1) % PROGRESS_EVERY == 0 or i + 1 == len(func_ids):
                        log.info(
                            f"  OK    {nr.name}  k={nr.k}{_timing_suffix(nr)}"
                        )

            if PROGRESS_EVERY and (i + 1) % PROGRESS_EVERY == 0:
                elapsed = time.perf_counter() - t_k0
                log.info(
                    f"  progress  {i + 1}/{br.n_planned}  "
                    f"eval={br.n_eval}  fail={br.n_fail}  "
                    f"skip={br.n_skip}  error={br.n_error}  "
                    f"elapsed={_fmt_secs(elapsed)}"
                )

        k_fail = br.n_fail > 0
        if k_fail:
            failing_ks.append(f"k={k} fail={br.n_fail}/{br.n_eval}")
        log.info(
            f"  k_summary  eval={br.n_eval}  fail={br.n_fail}  "
            f"skip={br.n_skip}  error={br.n_error}  k_fail={k_fail}  "
            f"cm_total={_fmt_secs(br.cm_total_s)}  "
            f"lut_total={_fmt_secs(br.lut_total_s)}"
        )
        log.info("")
        buckets.append(br)

    fn_frac = (n_fn_fail / n_fn_eval) if n_fn_eval else float("nan")
    n_k_eval = sum(1 for b in buckets if b.n_eval + b.n_error + b.n_skip > 0)
    n_k_fail = sum(1 for b in buckets if b.n_fail > 0)
    k_frac = (n_k_fail / n_k_eval) if n_k_eval else float("nan")

    log.info("=" * 60)
    log.info("SUMMARY")
    log.info(f"mode={mode}")
    log.info(f"function_fail_fraction = {n_fn_fail}/{n_fn_eval} = {fn_frac:.6f}")
    log.info(f"k_fail_fraction        = {n_k_fail}/{n_k_eval} = {k_frac:.6f}")
    log.info(f"cm_time_total          = {_fmt_secs(total_cm_s)}")
    log.info(f"lut_time_total         = {_fmt_secs(total_lut_s)}")
    if failing_ks:
        log.info("failing_k:")
        for item in failing_ks:
            log.info(f"  {item}")
    else:
        log.info("failing_k: (none)")
    log.info(f"log written to {LOG_PATH}")
    log.info("=" * 60)

    log.close()
    return LOG_PATH


if __name__ == "__main__":
    main()

# %% [markdown]
# ## Run
#
# Uncomment / execute to generate a fresh log (also happens under
# `__main__`).  Summary cells below point at `LOG_PATH`.

# %%
# LOG_PATH = main()
# FIG_DIR = LOG_PATH.parent / f"{LOG_PATH.stem}_figures"

# %% [markdown]
# ## Parse log + headline stats


# %%
_RE_HEADER = re.compile(r"^([A-Za-z0-9_]+)=(.*)$")
_RE_K = re.compile(
    r"^=== k=(?P<k>\d+)  mode=(?P<mode>\S+)  space=(?P<space>\d+)  "
    r"planned=(?P<planned>\d+) ===$"
)
_RE_NODE_HEAD = re.compile(
    r"^  (?P<status>OK|FAIL|SKIP|ERROR)\s+(?P<name>\S+)\s+k=(?P<k>\d+)(?P<rest>.*)$"
)
_RE_TIMING_TAIL = re.compile(r"\s+cm=(?P<cm>\S+)\s+lut=(?P<lut>\S+)\s*$")
_RE_K_SUMMARY = re.compile(
    r"^  k_summary  eval=(?P<eval>\d+)  fail=(?P<fail>\d+)  "
    r"skip=(?P<skip>\d+)  error=(?P<error>\d+)  k_fail=(?P<k_fail>\S+)  "
    r"cm_total=(?P<cm_total>\S+)  lut_total=(?P<lut_total>\S+)$"
)
_RE_FINAL = re.compile(
    r"^(function_fail_fraction|k_fail_fraction|cm_time_total|lut_time_total)"
    r"\s*=\s*(.+)$"
)


def parse_duration(token: Optional[str]) -> Optional[float]:
    if token is None or token in ("", "n/a"):
        return None
    token = token.strip()
    if token.endswith("us"):
        return float(token[:-2]) * 1e-6
    if token.endswith("ms"):
        return float(token[:-2]) * 1e-3
    if token.endswith("s"):
        return float(token[:-1])
    return float(token)


def classify_skip(reason: Optional[str]) -> str:
    if not reason:
        return "other"
    if "TIMEOUT_CM" in reason:
        return "TIMEOUT_CM"
    if "TIMEOUT_LUT" in reason:
        return "TIMEOUT_LUT"
    if "MAX_K" in reason:
        return "MAX_K"
    return "other"


def parse_node_line(line: str) -> Optional[Dict[str, Any]]:
    m = _RE_NODE_HEAD.match(line)
    if not m:
        return None
    status = m.group("status")
    rest = m.group("rest") or ""
    cm_s = lut_s = None
    tm = _RE_TIMING_TAIL.search(rest)
    if tm:
        cm_s = parse_duration(tm.group("cm"))
        lut_s = parse_duration(tm.group("lut"))
        rest = rest[: tm.start()]

    fail_reason = skip_reason = error_msg = None
    rest = rest.strip()
    if status == "FAIL":
        rm = re.match(r"reason=(\S+)", rest)
        fail_reason = rm.group(1) if rm else (rest or None)
    elif status == "SKIP":
        if rest.startswith("(") and rest.endswith(")"):
            skip_reason = rest[1:-1]
        else:
            skip_reason = rest or None
    elif status == "ERROR":
        error_msg = rest or None

    return {
        "node": m.group("name"),
        "k": int(m.group("k")),
        "status": status,
        "fail_reason": fail_reason,
        "skip_reason": skip_reason,
        "skip_class": classify_skip(skip_reason) if status == "SKIP" else None,
        "error": error_msg,
        "cm_s": cm_s,
        "lut_s": lut_s,
    }


def parse_log(
    path: Path,
) -> Tuple[Dict[str, str], pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    header: Dict[str, str] = {}
    nodes: List[Dict[str, Any]] = []
    buckets: List[Dict[str, Any]] = []
    final: Dict[str, str] = {}

    cur: Optional[Dict[str, Any]] = None
    in_header = True

    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")

            if line == "SUMMARY" or (line.startswith("=" * 10) and "SUMMARY" not in line):
                in_header = False
                continue

            m_k = _RE_K.match(line)
            if in_header and m_k is None:
                hm = _RE_HEADER.match(line)
                if hm:
                    header[hm.group(1)] = hm.group(2)
                continue

            if m_k:
                in_header = False
                if cur is not None and "k_fail" not in cur:
                    buckets.append(cur)
                cur = {
                    "k": int(m_k.group("k")),
                    "mode": m_k.group("mode"),
                    "space": int(m_k.group("space")),
                    "planned": int(m_k.group("planned")),
                }
                continue

            m = _RE_K_SUMMARY.match(line)
            if m and cur is not None:
                cur.update(
                    {
                        "eval": int(m.group("eval")),
                        "fail": int(m.group("fail")),
                        "skip": int(m.group("skip")),
                        "error": int(m.group("error")),
                        "k_fail": m.group("k_fail") == "True",
                        "cm_total_s": parse_duration(m.group("cm_total")),
                        "lut_total_s": parse_duration(m.group("lut_total")),
                    }
                )
                buckets.append(cur)
                cur = None
                continue

            if line.startswith("  progress"):
                continue

            node = parse_node_line(line)
            if node is not None and cur is not None:
                nodes.append(node)
                continue

            m = _RE_FINAL.match(line)
            if m:
                final[m.group(1)] = m.group(2).strip()

    if cur is not None:
        buckets.append(cur)

    return header, pd.DataFrame(nodes), pd.DataFrame(buckets), final


def summarize_log(path: Path) -> Tuple[Dict[str, str], pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    header, df_nodes, df_k, final = parse_log(path)
    print(f"log: {path.name}")
    print(f"header keys: {sorted(header.keys())}")
    print(f"k-buckets: {len(df_k)}  node lines: {len(df_nodes)}")
    if final:
        print("final SUMMARY present:", final)
    else:
        print("final SUMMARY: (absent — run incomplete)")

    if df_nodes.empty:
        print("(no node outcome lines parsed)")
        return header, df_nodes, df_k, final

    n_ok = int((df_nodes["status"] == "OK").sum())
    n_fail = int((df_nodes["status"] == "FAIL").sum())
    n_skip = int((df_nodes["status"] == "SKIP").sum())
    n_error = int((df_nodes["status"] == "ERROR").sum())
    n_eval = n_ok + n_fail

    print()
    print("=== Function outcomes (logged lines; OK may be subsampled) ===")
    print(f"  OK={n_ok}  FAIL={n_fail}  SKIP={n_skip}  ERROR={n_error}")
    print(
        f"  logged_eval≈{n_eval}  "
        f"fail_among_logged={n_fail}/{n_eval} = "
        f"{n_fail / n_eval if n_eval else float('nan'):.6f}"
    )
    print("  note: OK lines are thinned by PROGRESS_EVERY; use k_summary for rates.")

    if not df_k.empty and "eval" in df_k.columns:
        print()
        print("=== Per-k summaries ===")
        cols = [
            c
            for c in [
                "k",
                "mode",
                "space",
                "planned",
                "eval",
                "fail",
                "skip",
                "error",
                "k_fail",
                "cm_total_s",
                "lut_total_s",
            ]
            if c in df_k.columns
        ]
        print(df_k[cols].to_string(index=False))
        ev = int(df_k["eval"].sum())
        fa = int(df_k["fail"].sum())
        print()
        print(
            f"=== Aggregate from k_summary ===\n"
            f"  eval={ev}  fail={fa}  fail_fraction={fa}/{ev} = "
            f"{fa / ev if ev else float('nan'):.6f}"
        )

    print()
    print("=== FAIL reasons (logged) ===")
    print(
        df_nodes.loc[df_nodes["status"] == "FAIL", "fail_reason"]
        .value_counts(dropna=False)
        .to_string()
    )
    return header, df_nodes, df_k, final


# Prefer the log from this session; fall back to newest matching log if missing.
if not LOG_PATH.is_file():
    _cands = sorted(LOG_DIR.glob("cm_all_lut_roundtrip*.log"), key=lambda p: p.stat().st_mtime)
    if _cands:
        LOG_PATH = _cands[-1]
        FIG_DIR = LOG_PATH.parent / f"{LOG_PATH.stem}_figures"
        print(f"(no current-session log yet; using newest: {LOG_PATH.name})")

if LOG_PATH.is_file():
    header, df_nodes, df_k, final = summarize_log(LOG_PATH)
else:
    header, df_nodes, df_k, final = {}, pd.DataFrame(), pd.DataFrame(), {}
    print(f"No log at {LOG_PATH} — run main() first.")

# %% [markdown]
# ## Figures
#
# PNGs land under `FIG_DIR` (next to the log).


# %%
def save_fig(name: str) -> Path:
    path = FIG_DIR / f"{name}.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved {path.name}")
    plt.show()
    return path


if not df_k.empty and "eval" in df_k.columns:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"figures → {FIG_DIR}")

    plt.rcParams.update(
        {
            "figure.figsize": (7, 4),
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    # 1) Per-k fail rate (from k_summary — authoritative)
    fig, ax = plt.subplots()
    sub = df_k[df_k["eval"] > 0].copy()
    if len(sub):
        rate = sub["fail"] / sub["eval"]
        ax.bar(sub["k"], rate, color="#e76f51", alpha=0.85)
        for _, row in sub.iterrows():
            ax.text(
                row["k"],
                row["fail"] / row["eval"],
                f"{int(row['fail'])}/{int(row['eval'])}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_ylim(0, max(0.05, float(rate.max()) * 1.25))
    ax.set_xlabel("k (inputs)")
    ax.set_ylabel("fail rate")
    ax.set_title("FAIL rate vs k (from k_summary)")
    save_fig("01_fail_rate_vs_k")

    # 2) OK / fail / skip / error stacked counts (eval = OK + FAIL)
    fig, ax = plt.subplots()
    ks = df_k["k"].astype(int)
    bottoms = np.zeros(len(df_k))
    ok_counts = (df_k["eval"] - df_k["fail"]).clip(lower=0)
    series = [
        ("OK", ok_counts, "#2a9d8f"),
        ("FAIL", df_k["fail"], "#e76f51"),
        ("SKIP", df_k["skip"], "#e9c46a"),
        ("ERROR", df_k["error"], "#264653"),
    ]
    for label, vals, color in series:
        ax.bar(ks, vals, bottom=bottoms, label=label, color=color)
        bottoms = bottoms + vals.to_numpy(dtype=float)
    ax.set_xlabel("k")
    ax.set_ylabel("functions")
    ax.set_title("Outcomes per k (k_summary)")
    ax.legend(frameon=False)
    save_fig("02_outcomes_per_k")

    # 3) Timing totals vs k
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, title in (
        (axes[0], "cm_total_s", "CM time total"),
        (axes[1], "lut_total_s", "LUT time total"),
    ):
        if col not in df_k.columns or df_k[col].isna().all():
            ax.set_title(f"{title} (no data)")
            continue
        ax.bar(df_k["k"], df_k[col].fillna(0.0), color="#264653", alpha=0.85)
        ax.set_xlabel("k")
        ax.set_ylabel("seconds")
        ax.set_title(f"{title} vs k")
    save_fig("03_timing_totals_vs_k")

    # 4) Mode / coverage: planned vs space
    fig, ax = plt.subplots()
    ax.bar(df_k["k"] - 0.15, df_k["planned"], width=0.3, label="planned", color="#2a9d8f")
    ax.bar(df_k["k"] + 0.15, df_k["space"], width=0.3, label="space 2^(2^k)", color="#e9c46a")
    ax.set_yscale("log")
    ax.set_xlabel("k")
    ax.set_ylabel("count (log)")
    ax.set_title("Coverage: planned vs full Boolean space")
    ax.legend(frameon=False)
    save_fig("04_coverage")

    # 5) Logged FAIL reasons (if any FAIL lines)
    if not df_nodes.empty and (df_nodes["status"] == "FAIL").any():
        fig, ax = plt.subplots()
        vc = df_nodes.loc[df_nodes["status"] == "FAIL", "fail_reason"].value_counts()
        ax.bar(vc.index.astype(str), vc.values, color="#e76f51")
        ax.set_ylabel("logged FAIL lines")
        ax.set_title("FAIL reasons")
        ax.tick_params(axis="x", rotation=20)
        save_fig("05_fail_reasons")
else:
    print("Skip figures — no k_summary data (run main() first).")

# %% [markdown]
# ## Tables — failing LUTs (logged FAIL lines)


# %%
if not df_nodes.empty:
    fails = df_nodes[df_nodes["status"] == "FAIL"][
        [c for c in ["k", "node", "fail_reason", "cm_s", "lut_s"] if c in df_nodes.columns]
    ]
    print(f"FAIL rows ({len(fails)}; showing up to 40):")
    print(fails.head(40).to_string(index=False))
else:
    print("No node rows to tabulate.")
