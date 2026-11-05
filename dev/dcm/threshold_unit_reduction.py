# %% [markdown]
# Threshold-unit reduction exploration
#
# Enumerate Boolean LUTs (same scheme as `all_lut_dcm_simplify_roundtrip.py`),
# build a **raw** canalizing map (no stock simplify), and for **each threshold
# (or legacy fusion) unit independently** try a single reduction:
#
# * wire all predecessors of `T` to all successors of `T`
# * bump each threshold/fusion successor's `tau` by `tau(T)-1`
# * never bump variable nodes (fixed OR)
# * delete `T`
#
# Legacy ``fusion`` nodes are treated as threshold units with ``tau=1``.
# Classification: LINEAR (`tau==1==n`), AND (`tau==n`), OR (`tau==1`), OTHER.
# OTHER targets are never reduced. Compatibility (target vs every successor):
#
# * LINEAR → AND, OR, LINEAR, OTHER
# * AND → AND, LINEAR
# * OR → OR, LINEAR
#
# Each trial starts from a fresh copy of the raw CM. After a successful
# rewrite, reconstruct the LUT and check exact roundtrip.
#
# Set `SMOKE = True` to cap at `SMOKE_MAX_K`. Each run writes under `dev/logs/`.

# %%
from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

_DEV_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _DEV_DIR.parent  # .../CANA/CANA (contains the cana package)
for _p in (_DEV_DIR, _PKG_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cana.boolean_node import BooleanNode
from cana.canalization.cm_lut import lut_from_canalizing_map
from cana.canalization.cm_simplify import (
    classify_unit,
    is_threshold_like,
    reduce_threshold_unit,
    reduction_skip_reason,
)

from bbm_cm_roundtrip import (  # noqa: E402
    TeeLogger,
    _fmt_secs,
    is_exact_roundtrip,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SMOKE = False
SMOKE_MAX_K = 2
MAX_K = 5
SAMPLE_N = 2048
SAMPLE_SEED = 0
BUILDERS = ("legacy", "canonical")
PROGRESS_EVERY = 256

LOG_DIR = _DEV_DIR / "logs"

_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
_RUN_STEM = (
    f"threshold_unit_reduction_smoke_{_TIMESTAMP}"
    if SMOKE
    else f"threshold_unit_reduction_{_TIMESTAMP}"
)
RUN_DIR = LOG_DIR / _RUN_STEM
LOG_PATH = RUN_DIR / f"{_RUN_STEM}.log"
FIG_DIR = RUN_DIR

# %% [markdown]
# ## Helpers — LUT enumeration


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
    n = min(n, total)
    if total <= 10_000_000:
        return sorted(int(x) for x in rng.choice(total, size=n, replace=False))
    ids: set[int] = set()
    while len(ids) < n:
        ids.add(int(rng.integers(0, total)))
    return sorted(ids)


def iter_func_ids(k: int, sample_n: int, seed: int) -> Tuple[str, List[int]]:
    total = n_boolean_functions(k)
    if total <= sample_n:
        return "exhaustive", list(range(total))
    rng = np.random.default_rng(seed + k)
    return "sample", _sample_ids(total, sample_n, rng)


def make_node(k: int, func_id: int) -> BooleanNode:
    outputs = outputs_from_func_id(k, func_id)
    inputs = list(range(k))
    return BooleanNode.from_bit_lut(
        outputs, name=lut_name(k, func_id), inputs=inputs, id=func_id
    )


def effective_max_k(smoke: bool, max_k: int, smoke_max_k: int) -> int:
    return min(max_k, smoke_max_k) if smoke else max_k


# %% [markdown]
# ## Classification + reduction (from ``cana.canalization.cm_simplify``)


# %%
def successor_has_threshold(graph: nx.DiGraph, node_id) -> bool:
    """True if any successor is threshold-like (threshold or fusion)."""
    return any(is_threshold_like(graph, s) for s in graph.successors(node_id))


# %% [markdown]
# ## Per-unit trial + sweep


# %%
@dataclass
class UnitTrial:
    builder: str
    k: int
    func_id: int
    node_name: str
    unit_id: str
    unit_class: Optional[str]
    status: str  # skip | ok | fail | error
    skip_reason: Optional[str] = None
    fail_reason: Optional[str] = None
    error: Optional[str] = None
    succ_has_threshold: bool = False
    reduce_s: Optional[float] = None
    lut_s: Optional[float] = None


@dataclass
class KBucketResult:
    builder: str
    k: int
    mode: str
    n_total_space: int
    n_planned: int
    n_fn: int = 0
    n_units: int = 0
    n_attempt: int = 0
    n_ok: int = 0
    n_fail: int = 0
    n_skip: int = 0
    n_error: int = 0
    skip_by_reason: Dict[str, int] = field(default_factory=dict)
    by_class: Dict[str, Dict[str, int]] = field(default_factory=dict)


def _bump_class_stat(
    bucket: KBucketResult, cls: Optional[str], key: str
) -> None:
    c = cls or "UNKNOWN"
    if c not in bucket.by_class:
        bucket.by_class[c] = {
            "seen": 0,
            "attempt": 0,
            "ok": 0,
            "fail": 0,
            "skip": 0,
            "error": 0,
        }
    bucket.by_class[c][key] = bucket.by_class[c].get(key, 0) + 1


def trial_one_unit(
    raw: nx.DiGraph,
    unit_id: str,
    *,
    builder: str,
    k: int,
    func_id: int,
    node_name: str,
    outputs: Sequence[int],
    inputs: Sequence[int],
) -> UnitTrial:
    cls = classify_unit(raw, unit_id)
    succ_thr = (
        successor_has_threshold(raw, unit_id) if unit_id in raw else False
    )
    skip = reduction_skip_reason(raw, unit_id)
    if skip is not None:
        return UnitTrial(
            builder=builder,
            k=k,
            func_id=func_id,
            node_name=node_name,
            unit_id=str(unit_id),
            unit_class=cls,
            status="skip",
            skip_reason=skip,
            succ_has_threshold=succ_thr,
        )

    t0 = time.perf_counter()
    try:
        G = reduce_threshold_unit(raw, unit_id, copy=True)
    except Exception as exc:  # noqa: BLE001
        return UnitTrial(
            builder=builder,
            k=k,
            func_id=func_id,
            node_name=node_name,
            unit_id=str(unit_id),
            unit_class=cls,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            succ_has_threshold=succ_thr,
            reduce_s=time.perf_counter() - t0,
        )
    reduce_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    try:
        result = lut_from_canalizing_map(G, input_order=list(inputs))
    except Exception as exc:  # noqa: BLE001
        return UnitTrial(
            builder=builder,
            k=k,
            func_id=func_id,
            node_name=node_name,
            unit_id=str(unit_id),
            unit_class=cls,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            succ_has_threshold=succ_thr,
            reduce_s=reduce_s,
            lut_s=time.perf_counter() - t1,
        )
    lut_s = time.perf_counter() - t1

    exact = is_exact_roundtrip(outputs, result)
    if exact:
        return UnitTrial(
            builder=builder,
            k=k,
            func_id=func_id,
            node_name=node_name,
            unit_id=str(unit_id),
            unit_class=cls,
            status="ok",
            succ_has_threshold=succ_thr,
            reduce_s=reduce_s,
            lut_s=lut_s,
        )
    if result.has_conflict:
        fail_reason = "conflict"
    elif result.is_incomplete:
        fail_reason = "incomplete"
    else:
        fail_reason = "mismatch"
    return UnitTrial(
        builder=builder,
        k=k,
        func_id=func_id,
        node_name=node_name,
        unit_id=str(unit_id),
        unit_class=cls,
        status="fail",
        fail_reason=fail_reason,
        succ_has_threshold=succ_thr,
        reduce_s=reduce_s,
        lut_s=lut_s,
    )


def _format_trial_line(tr: UnitTrial) -> str:
    base = (
        f"  {tr.status.upper():5s}  {tr.node_name}  unit={tr.unit_id}  "
        f"class={tr.unit_class or '?'}  succ_thr={int(tr.succ_has_threshold)}"
    )
    if tr.status == "skip":
        return f"{base}  reason={tr.skip_reason}"
    if tr.status == "fail":
        return (
            f"{base}  reason={tr.fail_reason}  "
            f"reduce={_fmt_secs(tr.reduce_s)}  lut={_fmt_secs(tr.lut_s)}"
        )
    if tr.status == "error":
        return f"{base}  {tr.error}"
    return (
        f"{base}  reduce={_fmt_secs(tr.reduce_s)}  lut={_fmt_secs(tr.lut_s)}"
    )


def main() -> Path:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
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
    log.info(f"BUILDERS={','.join(BUILDERS)}")
    log.info(f"PROGRESS_EVERY={PROGRESS_EVERY}")
    log.info(f"run_dir={RUN_DIR}")
    log.info(f"log_path={LOG_PATH}")
    log.info(f"timestamp={_TIMESTAMP}")
    log.info("")

    buckets: List[KBucketResult] = []
    func_ids_by_k: Dict[int, Tuple[str, List[int]]] = {}
    n_attempt_total = 0
    n_ok_total = 0
    n_fail_total = 0
    n_skip_total = 0
    n_error_total = 0

    for builder in BUILDERS:
        log.info(f"######## builder={builder} ########")
        log.info("")
        for k in range(1, k_hi + 1):
            if k not in func_ids_by_k:
                func_ids_by_k[k] = iter_func_ids(
                    k, sample_n=SAMPLE_N, seed=SAMPLE_SEED
                )
            enum_mode, func_ids = func_ids_by_k[k]
            space = n_boolean_functions(k)
            br = KBucketResult(
                builder=builder,
                k=k,
                mode=enum_mode,
                n_total_space=space,
                n_planned=len(func_ids),
            )
            log.info(
                f"=== builder={builder}  k={k}  mode={enum_mode}  "
                f"space={space}  planned={br.n_planned} ==="
            )

            t_k0 = time.perf_counter()
            for i, func_id in enumerate(func_ids):
                node = make_node(k, func_id)
                try:
                    raw = node.canalizing_map(builder=builder)
                except Exception as exc:  # noqa: BLE001
                    log.info(
                        f"  ERROR {node.name}  CM build  "
                        f"{type(exc).__name__}: {exc}"
                    )
                    br.n_error += 1
                    n_error_total += 1
                    continue

                br.n_fn += 1
                thr_ids = [
                    nid
                    for nid in raw.nodes
                    if is_threshold_like(raw, nid)
                ]
                for unit_id in thr_ids:
                    br.n_units += 1
                    cls = classify_unit(raw, unit_id)
                    _bump_class_stat(br, cls, "seen")

                    tr = trial_one_unit(
                        raw,
                        unit_id,
                        builder=builder,
                        k=k,
                        func_id=func_id,
                        node_name=node.name,
                        outputs=list(node.outputs),
                        inputs=list(node.inputs),
                    )

                    if tr.status == "skip":
                        br.n_skip += 1
                        n_skip_total += 1
                        _bump_class_stat(br, tr.unit_class, "skip")
                        reason = tr.skip_reason or "unknown"
                        br.skip_by_reason[reason] = (
                            br.skip_by_reason.get(reason, 0) + 1
                        )
                    elif tr.status == "ok":
                        br.n_attempt += 1
                        br.n_ok += 1
                        n_attempt_total += 1
                        n_ok_total += 1
                        _bump_class_stat(br, tr.unit_class, "attempt")
                        _bump_class_stat(br, tr.unit_class, "ok")
                    elif tr.status == "fail":
                        br.n_attempt += 1
                        br.n_fail += 1
                        n_attempt_total += 1
                        n_fail_total += 1
                        _bump_class_stat(br, tr.unit_class, "attempt")
                        _bump_class_stat(br, tr.unit_class, "fail")
                        log.info(_format_trial_line(tr))
                    else:
                        br.n_error += 1
                        n_error_total += 1
                        _bump_class_stat(br, tr.unit_class, "error")
                        log.info(_format_trial_line(tr))

                    # Thin OK / skip logging
                    if tr.status in ("ok", "skip") and (
                        i < 1
                        or (i + 1) % PROGRESS_EVERY == 0
                        or i + 1 == len(func_ids)
                    ):
                        # Only log a sample of non-fail lines at progress points
                        pass

                if PROGRESS_EVERY and (i + 1) % PROGRESS_EVERY == 0:
                    elapsed = time.perf_counter() - t_k0
                    log.info(
                        f"  progress  {i + 1}/{br.n_planned}  "
                        f"fn={br.n_fn}  units={br.n_units}  "
                        f"attempt={br.n_attempt}  ok={br.n_ok}  "
                        f"fail={br.n_fail}  skip={br.n_skip}  "
                        f"error={br.n_error}  elapsed={_fmt_secs(elapsed)}"
                    )

            log.info(
                f"  k_summary  fn={br.n_fn}  units={br.n_units}  "
                f"attempt={br.n_attempt}  ok={br.n_ok}  fail={br.n_fail}  "
                f"skip={br.n_skip}  error={br.n_error}"
            )
            if br.skip_by_reason:
                parts = [
                    f"{r}={c}"
                    for r, c in sorted(br.skip_by_reason.items())
                ]
                log.info(f"  skip_reasons  {', '.join(parts)}")
            for cls_name, stats in sorted(br.by_class.items()):
                log.info(
                    f"  class_summary  class={cls_name}  "
                    f"seen={stats.get('seen', 0)}  "
                    f"attempt={stats.get('attempt', 0)}  "
                    f"ok={stats.get('ok', 0)}  "
                    f"fail={stats.get('fail', 0)}  "
                    f"skip={stats.get('skip', 0)}  "
                    f"error={stats.get('error', 0)}"
                )
            log.info("")
            buckets.append(br)

    fail_frac = (
        (n_fail_total / n_attempt_total) if n_attempt_total else float("nan")
    )
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info(f"mode={mode}")
    log.info(f"BUILDERS={','.join(BUILDERS)}")
    log.info(
        f"unit_attempt_fail_fraction = "
        f"{n_fail_total}/{n_attempt_total} = {fail_frac:.6f}"
    )
    log.info(f"unit_ok={n_ok_total}")
    log.info(f"unit_fail={n_fail_total}")
    log.info(f"unit_skip={n_skip_total}")
    log.info(f"unit_error={n_error_total}")
    for builder in BUILDERS:
        b_buckets = [b for b in buckets if b.builder == builder]
        att = sum(b.n_attempt for b in b_buckets)
        fa = sum(b.n_fail for b in b_buckets)
        ok = sum(b.n_ok for b in b_buckets)
        sk = sum(b.n_skip for b in b_buckets)
        frac = (fa / att) if att else float("nan")
        log.info(
            f"builder={builder}  attempt_fail_fraction = "
            f"{fa}/{att} = {frac:.6f}  ok={ok}  skip={sk}"
        )
    log.info(f"run_dir={RUN_DIR}")
    log.info(f"log written to {LOG_PATH}")
    log.info("=" * 60)

    log.close()
    return RUN_DIR


if __name__ == "__main__":
    RUN_DIR = main()
    LOG_PATH = RUN_DIR / f"{RUN_DIR.name}.log"
    FIG_DIR = RUN_DIR

# %% [markdown]
# ## Run
#
# Uncomment / execute to generate a fresh run directory (also happens under
# `__main__`).

# %%
# RUN_DIR = main()
# LOG_PATH = RUN_DIR / f"{RUN_DIR.name}.log"
# FIG_DIR = RUN_DIR

# %% [markdown]
# ## Parse log + headline stats


# %%
_RE_HEADER = re.compile(r"^([A-Za-z0-9_]+)=(.*)$")
_RE_K = re.compile(
    r"^=== builder=(?P<builder>\S+)  k=(?P<k>\d+)  mode=(?P<mode>\S+)  "
    r"space=(?P<space>\d+)  planned=(?P<planned>\d+) ===$"
)
_RE_K_SUMMARY = re.compile(
    r"^  k_summary  fn=(?P<fn>\d+)  units=(?P<units>\d+)  "
    r"attempt=(?P<attempt>\d+)  ok=(?P<ok>\d+)  fail=(?P<fail>\d+)  "
    r"skip=(?P<skip>\d+)  error=(?P<error>\d+)$"
)
_RE_CLASS = re.compile(
    r"^  class_summary  class=(?P<cls>\S+)  seen=(?P<seen>\d+)  "
    r"attempt=(?P<attempt>\d+)  ok=(?P<ok>\d+)  fail=(?P<fail>\d+)  "
    r"skip=(?P<skip>\d+)  error=(?P<error>\d+)$"
)
_RE_SKIP_REASONS = re.compile(r"^  skip_reasons  (?P<body>.*)$")
_RE_TRIAL = re.compile(
    r"^  (?P<status>OK|FAIL|SKIP|ERROR)\s+(?P<name>\S+)\s+"
    r"unit=(?P<unit>\S+)\s+class=(?P<cls>\S+)\s+"
    r"succ_thr=(?P<succ_thr>\d+)(?P<rest>.*)$"
)
_RE_SECTION = re.compile(r"^######## builder=(?P<builder>\S+) ########$")
_RE_FINAL = re.compile(
    r"^(unit_attempt_fail_fraction|unit_ok|unit_fail|unit_skip|unit_error)"
    r"\s*=\s*(.+)$"
)
_RE_BUILDER_FRAC = re.compile(
    r"^builder=(?P<builder>\S+)\s+attempt_fail_fraction\s*=\s*(.+)$"
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


def parse_log(
    path: Path,
) -> Tuple[Dict[str, str], pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    header: Dict[str, str] = {}
    trials: List[Dict[str, Any]] = []
    buckets: List[Dict[str, Any]] = []
    class_rows: List[Dict[str, Any]] = []
    final: Dict[str, str] = {}

    cur: Optional[Dict[str, Any]] = None
    section_builder: Optional[str] = None
    cur_k: Optional[int] = None
    in_header = True

    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")

            m_sec = _RE_SECTION.match(line)
            if m_sec:
                in_header = False
                section_builder = m_sec.group("builder")
                continue

            if line == "SUMMARY" or (
                line.startswith("=" * 10) and "SUMMARY" not in line
            ):
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
                if cur is not None and "attempt" not in cur:
                    buckets.append(cur)
                section_builder = m_k.group("builder")
                cur_k = int(m_k.group("k"))
                cur = {
                    "builder": m_k.group("builder"),
                    "k": cur_k,
                    "mode": m_k.group("mode"),
                    "space": int(m_k.group("space")),
                    "planned": int(m_k.group("planned")),
                }
                continue

            m = _RE_K_SUMMARY.match(line)
            if m and cur is not None:
                cur.update(
                    {
                        "fn": int(m.group("fn")),
                        "units": int(m.group("units")),
                        "attempt": int(m.group("attempt")),
                        "ok": int(m.group("ok")),
                        "fail": int(m.group("fail")),
                        "skip": int(m.group("skip")),
                        "error": int(m.group("error")),
                    }
                )
                buckets.append(cur)
                cur = None
                continue

            m = _RE_CLASS.match(line)
            if m and section_builder is not None and cur_k is not None:
                class_rows.append(
                    {
                        "builder": section_builder,
                        "k": cur_k,
                        "class": m.group("cls"),
                        "seen": int(m.group("seen")),
                        "attempt": int(m.group("attempt")),
                        "ok": int(m.group("ok")),
                        "fail": int(m.group("fail")),
                        "skip": int(m.group("skip")),
                        "error": int(m.group("error")),
                    }
                )
                continue

            m = _RE_SKIP_REASONS.match(line)
            if m and cur is not None:
                body = m.group("body")
                for part in body.split(", "):
                    if "=" in part:
                        r, c = part.split("=", 1)
                        cur[f"skip_{r}"] = int(c)
                continue

            if line.startswith("  progress"):
                continue

            m = _RE_TRIAL.match(line)
            if m:
                rest = (m.group("rest") or "").strip()
                fail_reason = skip_reason = error_msg = None
                if m.group("status") == "FAIL":
                    rm = re.search(r"reason=(\S+)", rest)
                    fail_reason = rm.group(1) if rm else rest or None
                elif m.group("status") == "SKIP":
                    rm = re.search(r"reason=(\S+)", rest)
                    skip_reason = rm.group(1) if rm else rest or None
                elif m.group("status") == "ERROR":
                    error_msg = rest or None
                trials.append(
                    {
                        "builder": section_builder or "legacy",
                        "k": cur_k,
                        "node": m.group("name"),
                        "unit": m.group("unit"),
                        "class": m.group("cls"),
                        "succ_thr": int(m.group("succ_thr")),
                        "status": m.group("status"),
                        "fail_reason": fail_reason,
                        "skip_reason": skip_reason,
                        "error": error_msg,
                    }
                )
                continue

            m_b = _RE_BUILDER_FRAC.match(line)
            if m_b:
                final[f"builder_{m_b.group('builder')}_fail_fraction"] = (
                    m_b.group(2).strip()
                )
                continue

            m = _RE_FINAL.match(line)
            if m:
                final[m.group(1)] = m.group(2).strip()

    if cur is not None:
        buckets.append(cur)

    return (
        header,
        pd.DataFrame(trials),
        pd.DataFrame(buckets),
        pd.DataFrame(class_rows),
        final,
    )


def summarize_log(
    path: Path,
) -> Tuple[Dict[str, str], pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    header, df_trials, df_k, df_class, final = parse_log(path)
    print(f"log: {path.name}")
    print(f"header keys: {sorted(header.keys())}")
    print(
        f"k-buckets: {len(df_k)}  logged trials: {len(df_trials)}  "
        f"class rows: {len(df_class)}"
    )
    if final:
        print("final SUMMARY:", final)
    else:
        print("final SUMMARY: (absent — run incomplete)")

    if not df_k.empty and "attempt" in df_k.columns:
        print()
        print("=== Per-(builder, k) summaries ===")
        cols = [
            c
            for c in [
                "builder",
                "k",
                "mode",
                "planned",
                "fn",
                "units",
                "attempt",
                "ok",
                "fail",
                "skip",
                "error",
            ]
            if c in df_k.columns
        ]
        print(df_k[cols].to_string(index=False))
        att = int(df_k["attempt"].sum())
        fa = int(df_k["fail"].sum())
        ok = int(df_k["ok"].sum())
        print()
        print(
            f"=== Aggregate ===\n"
            f"  attempt={att}  ok={ok}  fail={fa}  "
            f"fail_fraction={fa}/{att} = "
            f"{fa / att if att else float('nan'):.6f}"
        )
        if "builder" in df_k.columns:
            print()
            print("=== Per-builder ===")
            for builder, sub in df_k.groupby("builder", sort=True):
                batt = int(sub["attempt"].sum())
                bfa = int(sub["fail"].sum())
                print(
                    f"  {builder}: attempt={batt}  fail={bfa}  "
                    f"fail_fraction={bfa}/{batt} = "
                    f"{bfa / batt if batt else float('nan'):.6f}"
                )

    if not df_class.empty:
        print()
        print("=== Per-class aggregate ===")
        agg = (
            df_class.groupby("class", sort=True)[
                ["seen", "attempt", "ok", "fail", "skip", "error"]
            ]
            .sum()
            .astype(int)
        )
        print(agg.to_string())

    if not df_trials.empty:
        print()
        print("=== Logged FAIL trials (by class × succ_thr) ===")
        fails = df_trials[df_trials["status"] == "FAIL"]
        if fails.empty:
            print("(none logged — fails are always logged)")
        else:
            print(
                fails.groupby(
                    ["builder", "class", "succ_thr", "fail_reason"],
                    dropna=False,
                )
                .size()
                .to_string()
            )

    return header, df_trials, df_k, df_class, final


def _newest_run_dir() -> Optional[Path]:
    cands = sorted(
        (
            p
            for p in LOG_DIR.glob("threshold_unit_reduction*")
            if p.is_dir()
        ),
        key=lambda p: p.stat().st_mtime,
    )
    return cands[-1] if cands else None


def resolve_run_paths() -> Tuple[Path, Path, Path]:
    run_dir = RUN_DIR
    log_path = LOG_PATH
    if not log_path.is_file():
        newest = _newest_run_dir()
        if newest is not None:
            run_dir = newest
            log_path = run_dir / f"{run_dir.name}.log"
            print(f"(no current-session log yet; using newest: {run_dir.name})")
    return run_dir, log_path, run_dir


RUN_DIR, LOG_PATH, FIG_DIR = resolve_run_paths()

if LOG_PATH.is_file():
    header, df_trials, df_k, df_class, final = summarize_log(LOG_PATH)
else:
    header, df_trials, df_k, df_class, final = (
        {},
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        {},
    )
    print(f"No log at {LOG_PATH} — run main() first.")

# %% [markdown]
# ## Figures + README


# %%
def save_fig(name: str) -> Path:
    path = FIG_DIR / f"{name}.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved {path.name}")
    plt.show()
    return path


_BUILDER_COLORS = {
    "legacy": "#264653",
    "canonical": "#2a9d8f",
}

_CLASS_COLORS = {
    "LINEAR": "#2a9d8f",
    "AND": "#e76f51",
    "OR": "#e9c46a",
    "OTHER": "#264653",
}


def write_readme(
    run_dir: Path,
    header: Dict[str, str],
    df_k: pd.DataFrame,
    df_class: pd.DataFrame,
    final: Dict[str, str],
    fig_names: List[str],
) -> Path:
    stem = run_dir.name
    lines: List[str] = [f"# {stem}", ""]
    lines.append(
        "Raw CM → per-threshold-unit reduction (AND/OR/LINEAR) → LUT roundtrip."
    )
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append("| Setting | Value |")
    lines.append("|---------|-------|")
    ts = header.get("timestamp", "")
    lines.append(f"| timestamp | `{ts}` |")
    lines.append(f"| mode | `{header.get('mode', '')}` |")
    lines.append(f"| BUILDERS | `{header.get('BUILDERS', '')}` |")
    lines.append(f"| MAX_K | `{header.get('MAX_K', '')}` |")
    lines.append(
        f"| SAMPLE_N / SAMPLE_SEED | "
        f"`{header.get('SAMPLE_N', '')}` / `{header.get('SAMPLE_SEED', '')}` |"
    )
    if "effective_max_k" in header:
        lines.append(f"| effective_max_k | `{header['effective_max_k']}` |")
    lines.append("")

    lines.append("## Results")
    lines.append("")
    if not df_k.empty and "attempt" in df_k.columns:
        builders = sorted(df_k["builder"].dropna().unique().tolist())
        ks = sorted(df_k["k"].astype(int).unique().tolist())
        header_cells = ["k", "mode"] + [
            f"{b} fail/attempt" for b in builders
        ]
        lines.append("| " + " | ".join(header_cells) + " |")
        lines.append("|" + "|".join(["------"] * len(header_cells)) + "|")
        for k in ks:
            krows = df_k[df_k["k"] == k]
            mode = str(krows["mode"].iloc[0]) if len(krows) else ""
            cells = [str(k), mode]
            for b in builders:
                sub = krows[krows["builder"] == b]
                if sub.empty:
                    cells.append("—")
                else:
                    fa = int(sub["fail"].iloc[0])
                    att = int(sub["attempt"].iloc[0])
                    cells.append(f"{fa} / {att}")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        if final.get("unit_attempt_fail_fraction"):
            lines.append(
                f"- **overall attempt fail rate:** "
                f"`{final['unit_attempt_fail_fraction']}`"
            )
        lines.append("")
    else:
        lines.append("(no k_summary data)")
        lines.append("")

    if not df_class.empty and "builder" in df_class.columns:
        lines.append("## Per-class totals")
        lines.append("")
        class_order = ["LINEAR", "AND", "OR", "OTHER"]
        builders_c = sorted(df_class["builder"].dropna().unique().tolist())
        for builder in builders_c:
            lines.append(f"### {builder}")
            lines.append("")
            sub = df_class[df_class["builder"] == builder]
            agg = (
                sub.groupby("class")[
                    ["seen", "attempt", "ok", "fail", "skip"]
                ]
                .sum()
                .astype(int)
            )
            # Stable class order; append any unexpected labels at the end.
            ordered = [c for c in class_order if c in agg.index] + [
                c for c in agg.index if c not in class_order
            ]
            lines.append("| class | seen | attempt | ok | fail | skip |")
            lines.append("|-------|------|---------|----|------|------|")
            for cls in ordered:
                row = agg.loc[cls]
                lines.append(
                    f"| {cls} | {int(row['seen'])} | {int(row['attempt'])} | "
                    f"{int(row['ok'])} | {int(row['fail'])} | "
                    f"{int(row['skip'])} |"
                )
            lines.append("")
    elif not df_class.empty:
        lines.append("## Per-class totals")
        lines.append("")
        agg = (
            df_class.groupby("class")[
                ["seen", "attempt", "ok", "fail", "skip"]
            ]
            .sum()
            .astype(int)
        )
        lines.append("| class | seen | attempt | ok | fail | skip |")
        lines.append("|-------|------|---------|----|------|------|")
        for cls, row in agg.iterrows():
            lines.append(
                f"| {cls} | {row['seen']} | {row['attempt']} | "
                f"{row['ok']} | {row['fail']} | {row['skip']} |"
            )
        lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `{stem}.log`")
    for name in fig_names:
        lines.append(f"- `{name}.png`")
    lines.append("- `README.md`")
    lines.append("")

    path = run_dir / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"README → {path}")
    return path


fig_names_written: List[str] = []

if not df_k.empty and "attempt" in df_k.columns:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"figures → {FIG_DIR}")

    plt.rcParams.update(
        {
            "figure.figsize": (8, 4.5),
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    builders = sorted(df_k["builder"].dropna().unique().tolist())
    ks_all = sorted(df_k["k"].astype(int).unique().tolist())
    n_b = max(len(builders), 1)
    width = 0.8 / n_b

    # 1) Fail rate vs k by builder
    fig, ax = plt.subplots()
    sub = df_k[df_k["attempt"] > 0].copy()
    max_rate = 0.05
    for bi, builder in enumerate(builders):
        csub = sub[sub["builder"] == builder]
        if csub.empty:
            continue
        rate = csub["fail"] / csub["attempt"]
        if len(rate):
            max_rate = max(max_rate, float(rate.max()))
        offsets = csub["k"].astype(float) + (bi - (n_b - 1) / 2) * width
        ax.bar(
            offsets,
            rate,
            width=width * 0.9,
            label=builder,
            color=_BUILDER_COLORS.get(builder, f"C{bi}"),
            alpha=0.85,
        )
    ax.set_xticks(ks_all)
    ax.set_ylim(0, max_rate * 1.25 if max_rate > 0 else 0.05)
    ax.set_xlabel("k (inputs)")
    ax.set_ylabel("fail / attempt")
    ax.set_title("Reduction attempt FAIL rate vs k")
    ax.legend(frameon=False, fontsize=8)
    save_fig("01_fail_rate_vs_k")
    fig_names_written.append("01_fail_rate_vs_k")

    # 2) Outcomes stacked per k
    fig, axes = plt.subplots(
        1, n_b, figsize=(4.2 * n_b, 4), squeeze=False
    )
    for ax, builder in zip(axes[0], builders):
        bdf = df_k[df_k["builder"] == builder]
        if bdf.empty:
            ax.set_title(f"{builder} (no data)")
            continue
        ks = bdf["k"].astype(int)
        bottoms = np.zeros(len(bdf))
        series = [
            ("ok", bdf["ok"], "#2a9d8f"),
            ("fail", bdf["fail"], "#e76f51"),
            ("skip", bdf["skip"], "#e9c46a"),
            ("error", bdf["error"], "#264653"),
        ]
        for label, vals, color in series:
            ax.bar(ks, vals, bottom=bottoms, label=label, color=color)
            bottoms = bottoms + vals.to_numpy(dtype=float)
        ax.set_xlabel("k")
        ax.set_ylabel("threshold units")
        ax.set_title(builder, fontsize=9)
        ax.legend(frameon=False, fontsize=7)
    save_fig("02_outcomes_per_k")
    fig_names_written.append("02_outcomes_per_k")

    # 3) Per-class attempt / fail
    if not df_class.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        agg = (
            df_class.groupby("class")[["attempt", "ok", "fail", "skip"]]
            .sum()
            .reindex(["LINEAR", "AND", "OR", "OTHER"])
            .fillna(0)
        )
        classes = [c for c in agg.index if agg.loc[c].sum() > 0]
        colors = [_CLASS_COLORS.get(c, "#888") for c in classes]
        axes[0].bar(
            classes,
            [agg.loc[c, "attempt"] for c in classes],
            color=colors,
            alpha=0.85,
        )
        axes[0].set_title("Attempts by class")
        axes[0].set_ylabel("units")
        axes[1].bar(
            classes,
            [
                (
                    agg.loc[c, "fail"] / agg.loc[c, "attempt"]
                    if agg.loc[c, "attempt"] > 0
                    else 0.0
                )
                for c in classes
            ],
            color=colors,
            alpha=0.85,
        )
        axes[1].set_title("Fail rate by class (among attempts)")
        axes[1].set_ylabel("fail / attempt")
        save_fig("03_by_class")
        fig_names_written.append("03_by_class")

    # 4) Logged FAIL: OR × succ_thr if any
    if not df_trials.empty and (df_trials["status"] == "FAIL").any():
        fails = df_trials[df_trials["status"] == "FAIL"]
        fig, ax = plt.subplots()
        vc = (
            fails.groupby(["class", "succ_thr"])
            .size()
            .reset_index(name="n")
        )
        labels = [
            f"{r['class']}/thr={r['succ_thr']}" for _, r in vc.iterrows()
        ]
        ax.bar(labels, vc["n"], color="#e76f51")
        ax.set_ylabel("logged FAIL lines")
        ax.set_title("FAIL trials by class × succ_has_threshold")
        ax.tick_params(axis="x", rotation=25)
        save_fig("04_fail_by_class_succ")
        fig_names_written.append("04_fail_by_class_succ")

    write_readme(
        FIG_DIR, header, df_k, df_class, final, fig_names_written
    )
else:
    print("Skip figures / README — no k_summary data (run main() first).")

# %% [markdown]
# ## Tables — failing unit trials (logged FAIL lines)


# %%
if not df_trials.empty:
    fails = df_trials[df_trials["status"] == "FAIL"]
    print(f"FAIL rows ({len(fails)}; showing up to 40):")
    print(fails.head(40).to_string(index=False))
else:
    print("No trial rows to tabulate.")
