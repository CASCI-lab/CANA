# %% [markdown]
# All-LUT CM full-simplify minimality
#
# Sibling of `all_lut_cm_simplify_size.py` / `threshold_unit_reduction.py`.
# Enumerate Boolean LUTs for every `k <= MAX_K`, build a canalizing map, apply
# **full** simplify (`skip_if_edges_increase=False`), then check empirical
# minimality:
#
# 1. Simplified CM LUT-roundtrips exactly.
# 2. Structural fixed point: no `reduction_compatible` units remain.
# 3. For **every** remaining threshold/fusion unit, **force**
#    `reduce_threshold_unit` even when `reduction_skip_reason` is set; the LUT
#    roundtrip must **fail** (anomaly if it still succeeds).
#
# * **Exhaustive** when `2^(2^k) <= SAMPLE_N` (default 2048 → full enum for
#   `k <= 3`).
# * **Sample** exactly `SAMPLE_N` uniform random LUTs when the space is larger.
#
# Set `SMOKE = True` to cap at `SMOKE_MAX_K`.  Each run writes under `dev/logs/`.

# %%
from __future__ import annotations

import sys
import time
from collections import Counter
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
from cana.canalization.canalizing_map import simplify_canalizing_map
from cana.canalization.cm_lut import lut_from_canalizing_map
from cana.canalization.cm_simplify import (
    classify_unit,
    is_threshold_like,
    reduce_threshold_unit,
    reduction_compatible,
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
    f"cm_simplify_minimality_smoke_{_TIMESTAMP}"
    if SMOKE
    else f"cm_simplify_minimality_{_TIMESTAMP}"
)
RUN_DIR = LOG_DIR / _RUN_STEM
LOG_PATH = RUN_DIR / f"{_RUN_STEM}.log"
FN_CSV_PATH = RUN_DIR / "functions.csv"
UNIT_CSV_PATH = RUN_DIR / "unit_trials.csv"
FIG_DIR = RUN_DIR

# %% [markdown]
# ## Helpers — LUT enumeration + make_node


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


def apply_full_simplify(cm: nx.DiGraph) -> nx.DiGraph:
    """Unrestricted fixed-point simplify (copy + drop isolates)."""
    return simplify_canalizing_map(
        cm,
        copy=True,
        remove_isolates=True,
        skip_if_edges_increase=False,
    )


def roundtrip_fail_reason(outputs: Sequence[int], result) -> Optional[str]:
    """Return None if exact, else conflict / incomplete / mismatch."""
    if is_exact_roundtrip(outputs, result):
        return None
    if result.has_conflict:
        return "conflict"
    if result.is_incomplete:
        return "incomplete"
    return "mismatch"


# %% [markdown]
# ## Full simplify + force-reduce trial types


# %%
@dataclass
class UnitTrial:
    builder: str
    k: int
    func_id: int
    node_name: str
    unit_id: str
    unit_class: Optional[str]
    skip_reason: Optional[str]
    status: str  # fail | ok | error  (fail = expected for minimality)
    fail_reason: Optional[str] = None
    error: Optional[str] = None
    reduce_s: Optional[float] = None
    lut_s: Optional[float] = None


@dataclass
class FnResult:
    builder: str
    k: int
    func_id: int
    node_name: str
    enum_mode: str
    status: str  # ok | simplify_fail | still_compatible | force_ok | force_error | error
    simplify_exact: bool = False
    simplify_fail_reason: Optional[str] = None
    n_units: int = 0
    n_compatible: int = 0
    n_force_fail: int = 0
    n_force_ok: int = 0
    n_force_error: int = 0
    error: Optional[str] = None
    simplify_s: Optional[float] = None
    compatible_ids: List[str] = field(default_factory=list)


@dataclass
class KBucketResult:
    builder: str
    k: int
    mode: str
    n_total_space: int
    n_planned: int
    n_fn: int = 0
    n_simplify_exact: int = 0
    n_simplify_fail: int = 0
    n_still_compatible: int = 0
    n_fn_error: int = 0
    n_units: int = 0
    n_force_fail: int = 0
    n_force_ok: int = 0
    n_force_error: int = 0
    simplify_fail_by_reason: Dict[str, int] = field(default_factory=dict)
    skip_by_reason: Dict[str, int] = field(default_factory=dict)
    by_class: Dict[str, Dict[str, int]] = field(default_factory=dict)


def _bump_class_stat(
    bucket: KBucketResult, cls: Optional[str], key: str
) -> None:
    c = cls or "UNKNOWN"
    if c not in bucket.by_class:
        bucket.by_class[c] = {
            "seen": 0,
            "force_fail": 0,
            "force_ok": 0,
            "force_error": 0,
        }
    bucket.by_class[c][key] = bucket.by_class[c].get(key, 0) + 1


def trial_force_reduce(
    full: nx.DiGraph,
    unit_id: str,
    *,
    builder: str,
    k: int,
    func_id: int,
    node_name: str,
    outputs: Sequence[int],
    inputs: Sequence[int],
) -> UnitTrial:
    """Force-reduce *unit_id* even if reduction_skip_reason is set."""
    cls = classify_unit(full, unit_id)
    skip = reduction_skip_reason(full, unit_id)

    t0 = time.perf_counter()
    try:
        G = reduce_threshold_unit(full, unit_id, copy=True)
    except Exception as exc:  # noqa: BLE001
        return UnitTrial(
            builder=builder,
            k=k,
            func_id=func_id,
            node_name=node_name,
            unit_id=str(unit_id),
            unit_class=cls,
            skip_reason=skip,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
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
            skip_reason=skip,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            reduce_s=reduce_s,
            lut_s=time.perf_counter() - t1,
        )
    lut_s = time.perf_counter() - t1

    fail_reason = roundtrip_fail_reason(outputs, result)
    if fail_reason is None:
        return UnitTrial(
            builder=builder,
            k=k,
            func_id=func_id,
            node_name=node_name,
            unit_id=str(unit_id),
            unit_class=cls,
            skip_reason=skip,
            status="ok",  # anomaly: still exact after forced reduce
            reduce_s=reduce_s,
            lut_s=lut_s,
        )
    return UnitTrial(
        builder=builder,
        k=k,
        func_id=func_id,
        node_name=node_name,
        unit_id=str(unit_id),
        unit_class=cls,
        skip_reason=skip,
        status="fail",  # expected
        fail_reason=fail_reason,
        reduce_s=reduce_s,
        lut_s=lut_s,
    )


def evaluate_function(
    node: BooleanNode,
    *,
    builder: str,
    k: int,
    func_id: int,
    enum_mode: str,
) -> Tuple[FnResult, List[UnitTrial]]:
    """Full-simplify one LUT and force-reduce every remaining unit."""
    name = node.name
    trials: List[UnitTrial] = []

    t0 = time.perf_counter()
    try:
        raw = node.canalizing_map(builder=builder)
        full = apply_full_simplify(raw)
    except Exception as exc:  # noqa: BLE001
        return (
            FnResult(
                builder=builder,
                k=k,
                func_id=func_id,
                node_name=name,
                enum_mode=enum_mode,
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                simplify_s=time.perf_counter() - t0,
            ),
            trials,
        )
    simplify_s = time.perf_counter() - t0

    try:
        rt = lut_from_canalizing_map(full, input_order=list(node.inputs))
    except Exception as exc:  # noqa: BLE001
        return (
            FnResult(
                builder=builder,
                k=k,
                func_id=func_id,
                node_name=name,
                enum_mode=enum_mode,
                status="error",
                error=f"lut_from_canalizing_map: {type(exc).__name__}: {exc}",
                simplify_s=simplify_s,
            ),
            trials,
        )

    simp_fail = roundtrip_fail_reason(node.outputs, rt)
    simplify_exact = simp_fail is None

    thr_ids = [nid for nid in full.nodes if is_threshold_like(full, nid)]
    compatible_ids = [
        str(nid) for nid in thr_ids if reduction_compatible(full, nid)
    ]

    for unit_id in thr_ids:
        trials.append(
            trial_force_reduce(
                full,
                unit_id,
                builder=builder,
                k=k,
                func_id=func_id,
                node_name=name,
                outputs=list(node.outputs),
                inputs=list(node.inputs),
            )
        )

    n_force_fail = sum(1 for t in trials if t.status == "fail")
    n_force_ok = sum(1 for t in trials if t.status == "ok")
    n_force_error = sum(1 for t in trials if t.status == "error")

    if not simplify_exact:
        status = "simplify_fail"
    elif compatible_ids:
        status = "still_compatible"
    elif n_force_ok > 0:
        status = "force_ok"
    elif n_force_error > 0:
        status = "force_error"
    else:
        status = "ok"

    return (
        FnResult(
            builder=builder,
            k=k,
            func_id=func_id,
            node_name=name,
            enum_mode=enum_mode,
            status=status,
            simplify_exact=simplify_exact,
            simplify_fail_reason=simp_fail,
            n_units=len(thr_ids),
            n_compatible=len(compatible_ids),
            n_force_fail=n_force_fail,
            n_force_ok=n_force_ok,
            n_force_error=n_force_error,
            simplify_s=simplify_s,
            compatible_ids=compatible_ids,
        ),
        trials,
    )


def _format_unit_line(tr: UnitTrial) -> str:
    base = (
        f"  UNIT_{tr.status.upper():5s}  {tr.node_name}  unit={tr.unit_id}  "
        f"class={tr.unit_class or '?'}  skip={tr.skip_reason or '-'}"
    )
    if tr.status == "fail":
        return (
            f"{base}  reason={tr.fail_reason}  "
            f"reduce={_fmt_secs(tr.reduce_s)}  lut={_fmt_secs(tr.lut_s)}"
        )
    if tr.status == "ok":
        return (
            f"{base}  ANOMALY_exact  "
            f"reduce={_fmt_secs(tr.reduce_s)}  lut={_fmt_secs(tr.lut_s)}"
        )
    return f"{base}  {tr.error}"


def _format_fn_anomaly(fn: FnResult) -> str:
    if fn.status == "simplify_fail":
        return (
            f"  FN_ANOMALY  {fn.node_name}  simplify_fail  "
            f"reason={fn.simplify_fail_reason}"
        )
    if fn.status == "still_compatible":
        ids = ",".join(fn.compatible_ids[:8])
        more = "..." if len(fn.compatible_ids) > 8 else ""
        return (
            f"  FN_ANOMALY  {fn.node_name}  still_compatible  "
            f"n={fn.n_compatible}  ids={ids}{more}"
        )
    if fn.status == "force_ok":
        return (
            f"  FN_ANOMALY  {fn.node_name}  force_ok  "
            f"n_force_ok={fn.n_force_ok}/{fn.n_units}"
        )
    if fn.status == "force_error":
        return (
            f"  FN_ANOMALY  {fn.node_name}  force_error  "
            f"n_force_error={fn.n_force_error}/{fn.n_units}"
        )
    if fn.status == "error":
        return f"  FN_ANOMALY  {fn.node_name}  error  {fn.error}"
    return f"  FN_ANOMALY  {fn.node_name}  {fn.status}"


# %% [markdown]
# ## Run enumeration (smoke or full)
#
# Call `main()` from a notebook cell, or run the file as a script.
# Returns the run directory.  Artifacts: log, `functions.csv`, `unit_trials.csv`.


# %%
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
    log.info(f"fn_csv_path={FN_CSV_PATH}")
    log.info(f"unit_csv_path={UNIT_CSV_PATH}")
    log.info(f"timestamp={_TIMESTAMP}")
    log.info("")

    buckets: List[KBucketResult] = []
    fn_rows: List[Dict[str, Any]] = []
    unit_rows: List[Dict[str, Any]] = []
    func_ids_by_k: Dict[int, Tuple[str, List[int]]] = {}

    n_fn_total = 0
    n_simplify_exact_total = 0
    n_simplify_fail_total = 0
    n_still_compatible_total = 0
    n_fn_error_total = 0
    n_units_total = 0
    n_force_fail_total = 0
    n_force_ok_total = 0
    n_force_error_total = 0

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
                fn, trials = evaluate_function(
                    node,
                    builder=builder,
                    k=k,
                    func_id=func_id,
                    enum_mode=enum_mode,
                )

                br.n_fn += 1
                n_fn_total += 1
                br.n_units += fn.n_units
                n_units_total += fn.n_units
                br.n_force_fail += fn.n_force_fail
                br.n_force_ok += fn.n_force_ok
                br.n_force_error += fn.n_force_error
                n_force_fail_total += fn.n_force_fail
                n_force_ok_total += fn.n_force_ok
                n_force_error_total += fn.n_force_error

                if fn.simplify_exact:
                    br.n_simplify_exact += 1
                    n_simplify_exact_total += 1
                elif fn.simplify_fail_reason is not None:
                    br.n_simplify_fail += 1
                    n_simplify_fail_total += 1
                    reason = fn.simplify_fail_reason
                    br.simplify_fail_by_reason[reason] = (
                        br.simplify_fail_by_reason.get(reason, 0) + 1
                    )

                if fn.n_compatible > 0:
                    br.n_still_compatible += 1
                    n_still_compatible_total += 1

                if fn.status == "error":
                    br.n_fn_error += 1
                    n_fn_error_total += 1

                if fn.status != "ok":
                    log.info(_format_fn_anomaly(fn))

                for tr in trials:
                    _bump_class_stat(br, tr.unit_class, "seen")
                    if tr.status == "fail":
                        _bump_class_stat(br, tr.unit_class, "force_fail")
                    elif tr.status == "ok":
                        _bump_class_stat(br, tr.unit_class, "force_ok")
                        log.info(_format_unit_line(tr))
                    else:
                        _bump_class_stat(br, tr.unit_class, "force_error")
                        log.info(_format_unit_line(tr))

                    if tr.skip_reason:
                        br.skip_by_reason[tr.skip_reason] = (
                            br.skip_by_reason.get(tr.skip_reason, 0) + 1
                        )

                    unit_rows.append(
                        {
                            "builder": tr.builder,
                            "k": tr.k,
                            "func_id": tr.func_id,
                            "name": tr.node_name,
                            "unit_id": tr.unit_id,
                            "unit_class": tr.unit_class,
                            "skip_reason": tr.skip_reason,
                            "status": tr.status,
                            "fail_reason": tr.fail_reason,
                            "error": tr.error,
                            "reduce_s": tr.reduce_s,
                            "lut_s": tr.lut_s,
                        }
                    )

                fn_rows.append(
                    {
                        "builder": fn.builder,
                        "k": fn.k,
                        "func_id": fn.func_id,
                        "name": fn.node_name,
                        "enum_mode": fn.enum_mode,
                        "status": fn.status,
                        "simplify_exact": int(fn.simplify_exact),
                        "simplify_fail_reason": fn.simplify_fail_reason,
                        "n_units": fn.n_units,
                        "n_compatible": fn.n_compatible,
                        "n_force_fail": fn.n_force_fail,
                        "n_force_ok": fn.n_force_ok,
                        "n_force_error": fn.n_force_error,
                        "compatible_ids": ";".join(fn.compatible_ids),
                        "error": fn.error,
                        "simplify_s": fn.simplify_s,
                    }
                )

                if PROGRESS_EVERY and (i + 1) % PROGRESS_EVERY == 0:
                    elapsed = time.perf_counter() - t_k0
                    log.info(
                        f"  progress  {i + 1}/{br.n_planned}  "
                        f"fn={br.n_fn}  simplify_exact={br.n_simplify_exact}  "
                        f"simp_fail={br.n_simplify_fail}  "
                        f"still_compat={br.n_still_compatible}  "
                        f"units={br.n_units}  force_fail={br.n_force_fail}  "
                        f"force_ok={br.n_force_ok}  force_error={br.n_force_error}  "
                        f"elapsed={_fmt_secs(elapsed)}"
                    )

            log.info(
                f"  k_summary  fn={br.n_fn}  "
                f"simplify_exact={br.n_simplify_exact}  "
                f"simp_fail={br.n_simplify_fail}  "
                f"still_compat={br.n_still_compatible}  "
                f"fn_error={br.n_fn_error}  units={br.n_units}  "
                f"force_fail={br.n_force_fail}  force_ok={br.n_force_ok}  "
                f"force_error={br.n_force_error}"
            )
            if br.simplify_fail_by_reason:
                parts = [
                    f"{r}={c}"
                    for r, c in sorted(br.simplify_fail_by_reason.items())
                ]
                log.info(f"  simplify_fail_reasons  {', '.join(parts)}")
            if br.skip_by_reason:
                parts = [
                    f"{r}={c}" for r, c in sorted(br.skip_by_reason.items())
                ]
                log.info(f"  force_skip_reasons  {', '.join(parts)}")
            for cls_name, stats in sorted(br.by_class.items()):
                log.info(
                    f"  class_summary  class={cls_name}  "
                    f"seen={stats.get('seen', 0)}  "
                    f"force_fail={stats.get('force_fail', 0)}  "
                    f"force_ok={stats.get('force_ok', 0)}  "
                    f"force_error={stats.get('force_error', 0)}"
                )
            log.info("")
            buckets.append(br)

    df_fn = pd.DataFrame(fn_rows)
    df_unit = pd.DataFrame(unit_rows)
    df_fn.to_csv(FN_CSV_PATH, index=False)
    df_unit.to_csv(UNIT_CSV_PATH, index=False)

    force_attempt = n_force_fail_total + n_force_ok_total + n_force_error_total
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info(f"mode={mode}")
    log.info(f"BUILDERS={','.join(BUILDERS)}")
    log.info(f"n_fn={n_fn_total}")
    log.info(f"n_simplify_exact={n_simplify_exact_total}")
    log.info(f"n_simplify_fail={n_simplify_fail_total}")
    log.info(f"n_still_compatible={n_still_compatible_total}")
    log.info(f"n_fn_error={n_fn_error_total}")
    log.info(f"n_units={n_units_total}")
    log.info(f"n_force_fail={n_force_fail_total}")
    log.info(f"n_force_ok={n_force_ok_total}")
    log.info(f"n_force_error={n_force_error_total}")
    anomaly_ok_frac = (
        (n_force_ok_total / force_attempt) if force_attempt else float("nan")
    )
    log.info(
        f"force_ok_fraction = {n_force_ok_total}/{force_attempt} = "
        f"{anomaly_ok_frac:.6f}"
    )
    for builder in BUILDERS:
        b_buckets = [b for b in buckets if b.builder == builder]
        fn = sum(b.n_fn for b in b_buckets)
        se = sum(b.n_simplify_exact for b in b_buckets)
        sf = sum(b.n_simplify_fail for b in b_buckets)
        sc = sum(b.n_still_compatible for b in b_buckets)
        ff = sum(b.n_force_fail for b in b_buckets)
        fo = sum(b.n_force_ok for b in b_buckets)
        fe = sum(b.n_force_error for b in b_buckets)
        log.info(
            f"builder={builder}  fn={fn}  simplify_exact={se}  "
            f"simp_fail={sf}  still_compat={sc}  "
            f"force_fail={ff}  force_ok={fo}  force_error={fe}"
        )
    log.info(f"fn_csv_path={FN_CSV_PATH}")
    log.info(f"unit_csv_path={UNIT_CSV_PATH}")
    log.info(f"run_dir={RUN_DIR}")
    log.info(f"log written to {LOG_PATH}")
    log.info("=" * 60)

    # First-class findings
    log.info("")
    log.info("FINDINGS")
    log.info(
        f"  simplify roundtrip fails : {n_simplify_fail_total} "
        f"(expect 0)"
    )
    log.info(
        f"  still reduction_compatible after full simplify : "
        f"{n_still_compatible_total} functions (expect 0)"
    )
    log.info(
        f"  force-reduce still exact (ANOMALY) : {n_force_ok_total} units "
        f"(expect 0)"
    )
    log.info(
        f"  force-reduce fails (expected) : {n_force_fail_total} units"
    )

    log.close()
    return RUN_DIR


if __name__ == "__main__":
    RUN_DIR = main()
    LOG_PATH = RUN_DIR / f"{RUN_DIR.name}.log"
    FN_CSV_PATH = RUN_DIR / "functions.csv"
    UNIT_CSV_PATH = RUN_DIR / "unit_trials.csv"
    FIG_DIR = RUN_DIR

# %% [markdown]
# ## Run
#
# Uncomment / execute to generate a fresh run directory (also happens under
# `__main__`).


# %%
# RUN_DIR = main()
# LOG_PATH = RUN_DIR / f"{RUN_DIR.name}.log"
# FN_CSV_PATH = RUN_DIR / "functions.csv"
# UNIT_CSV_PATH = RUN_DIR / "unit_trials.csv"
# FIG_DIR = RUN_DIR

# %% [markdown]
# ## Load CSVs + headline stats


# %%
def _newest_run_dir() -> Optional[Path]:
    if not LOG_DIR.exists():
        return None
    cands = sorted(
        (
            p
            for p in LOG_DIR.glob("cm_simplify_minimality*")
            if p.is_dir() and (p / "functions.csv").exists()
        ),
        key=lambda p: p.stat().st_mtime,
    )
    return cands[-1] if cands else None


def resolve_run_paths() -> Tuple[Path, Path, Path, Path]:
    run_dir = RUN_DIR
    fn_csv = FN_CSV_PATH
    unit_csv = UNIT_CSV_PATH
    if not fn_csv.is_file():
        newest = _newest_run_dir()
        if newest is not None:
            run_dir = newest
            fn_csv = run_dir / "functions.csv"
            unit_csv = run_dir / "unit_trials.csv"
            print(f"(no current-session CSV yet; using newest: {run_dir.name})")
    return run_dir, fn_csv, unit_csv, run_dir


RUN_DIR, FN_CSV_PATH, UNIT_CSV_PATH, FIG_DIR = resolve_run_paths()
LOG_PATH = RUN_DIR / f"{RUN_DIR.name}.log"


def summarize_csvs(
    fn_csv: Path, unit_csv: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_fn = pd.read_csv(fn_csv) if fn_csv.is_file() else pd.DataFrame()
    df_unit = pd.read_csv(unit_csv) if unit_csv.is_file() else pd.DataFrame()

    print(f"functions : {fn_csv.name}  rows={len(df_fn)}")
    print(f"units     : {unit_csv.name}  rows={len(df_unit)}")

    if not df_fn.empty:
        print()
        print("=== Function status counts ===")
        print(df_fn.groupby(["builder", "status"]).size().unstack(fill_value=0))
        print()
        print("=== Per-(builder, k) ===")
        agg = (
            df_fn.groupby(["builder", "k"], sort=True)
            .agg(
                n_fn=("name", "count"),
                simplify_exact=("simplify_exact", "sum"),
                n_units=("n_units", "sum"),
                n_compatible=("n_compatible", "sum"),
                n_force_fail=("n_force_fail", "sum"),
                n_force_ok=("n_force_ok", "sum"),
                n_force_error=("n_force_error", "sum"),
            )
            .astype(int)
        )
        print(agg.to_string())

        n_simp_fail = int((df_fn["status"] == "simplify_fail").sum())
        n_still = int((df_fn["status"] == "still_compatible").sum())
        n_force_ok_fn = int((df_fn["status"] == "force_ok").sum())
        print()
        print("=== FINDINGS (expect zeros for anomalies) ===")
        print(f"  simplify_fail functions     : {n_simp_fail}")
        print(f"  still_compatible functions  : {n_still}")
        print(f"  force_ok functions          : {n_force_ok_fn}")

    if not df_unit.empty:
        print()
        print("=== Unit trial status ===")
        print(df_unit.groupby(["builder", "status"]).size().unstack(fill_value=0))
        if "skip_reason" in df_unit.columns:
            print()
            print("=== Force trials by skip_reason × status ===")
            print(
                df_unit.groupby(["skip_reason", "status"], dropna=False)
                .size()
                .unstack(fill_value=0)
                .to_string()
            )
        ok_units = df_unit[df_unit["status"] == "ok"]
        if not ok_units.empty:
            print()
            print("=== ANOMALY: force-reduce still exact ===")
            cols = [
                c
                for c in [
                    "builder",
                    "k",
                    "name",
                    "unit_id",
                    "unit_class",
                    "skip_reason",
                ]
                if c in ok_units.columns
            ]
            print(ok_units[cols].head(40).to_string(index=False))
            if len(ok_units) > 40:
                print(f"  ... ({len(ok_units)} total)")

    return df_fn, df_unit


if FN_CSV_PATH.is_file():
    df_fn, df_unit = summarize_csvs(FN_CSV_PATH, UNIT_CSV_PATH)
else:
    df_fn, df_unit = pd.DataFrame(), pd.DataFrame()
    print(f"No CSV at {FN_CSV_PATH} — run main() first.")

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


def write_readme(
    run_dir: Path,
    df_fn: pd.DataFrame,
    df_unit: pd.DataFrame,
    fig_names: List[str],
) -> Path:
    stem = run_dir.name
    lines: List[str] = [f"# {stem}", ""]
    lines.append(
        "Raw CM → full simplify (`skip_if_edges_increase=False`) → "
        "force-reduce each remaining threshold unit → LUT roundtrip."
    )
    lines.append("")
    lines.append(
        "Minimality: after full simplify, every forced unit reduction must "
        "break LUT roundtrip; no `reduction_compatible` units should remain."
    )
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append("| Setting | Value |")
    lines.append("|---------|-------|")
    lines.append(f"| SMOKE | `{SMOKE}` |")
    lines.append(f"| MAX_K | `{MAX_K}` |")
    lines.append(f"| SAMPLE_N | `{SAMPLE_N}` |")
    lines.append(f"| BUILDERS | `{', '.join(BUILDERS)}` |")
    lines.append("")

    if not df_fn.empty:
        n_fn = len(df_fn)
        n_simp_fail = int((df_fn["status"] == "simplify_fail").sum())
        n_still = int((df_fn["status"] == "still_compatible").sum())
        n_force_ok_fn = int((df_fn["status"] == "force_ok").sum())
        n_ok = int((df_fn["status"] == "ok").sum())
        lines.append("## Findings")
        lines.append("")
        lines.append(f"- Functions evaluated: **{n_fn}**")
        lines.append(f"- Status `ok` (minimal): **{n_ok}**")
        lines.append(
            f"- Anomaly `simplify_fail`: **{n_simp_fail}** (expect 0)"
        )
        lines.append(
            f"- Anomaly `still_compatible`: **{n_still}** (expect 0)"
        )
        lines.append(
            f"- Anomaly `force_ok`: **{n_force_ok_fn}** functions (expect 0)"
        )
        lines.append("")

    if not df_unit.empty:
        counts = Counter(df_unit["status"].tolist())
        lines.append("## Unit force-reduce trials")
        lines.append("")
        lines.append(f"- fail (expected): **{counts.get('fail', 0)}**")
        lines.append(f"- ok (anomaly): **{counts.get('ok', 0)}**")
        lines.append(f"- error: **{counts.get('error', 0)}**")
        lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `{stem}.log`")
    lines.append("- `functions.csv`")
    lines.append("- `unit_trials.csv`")
    for name in fig_names:
        lines.append(f"- `{name}.png`")
    lines.append("")

    path = run_dir / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {path.name}")
    return path


fig_names: List[str] = []

if not df_fn.empty and "k" in df_fn.columns:
    # 1) Force-ok (anomaly) rate vs k by builder
    fig, ax = plt.subplots(figsize=(7, 4))
    for builder in sorted(df_fn["builder"].unique()):
        sub = df_fn[df_fn["builder"] == builder]
        by_k = sub.groupby("k")
        ks = sorted(by_k.groups.keys())
        rates = [
            float((by_k.get_group(k)["n_force_ok"] > 0).mean()) for k in ks
        ]
        ax.plot(
            ks,
            rates,
            marker="o",
            label=str(builder),
            color=_BUILDER_COLORS.get(str(builder), None),
        )
    ax.set_xlabel("k")
    ax.set_ylabel("fraction of functions with ≥1 force_ok unit")
    ax.set_title("Anomaly: force-reduce still exact (vs k)")
    ax.legend()
    ax.set_ylim(bottom=0)
    fig_names.append("01_force_ok_frac_vs_k")
    save_fig("01_force_ok_frac_vs_k")

    # 2) Units remaining after full simplify vs k
    fig, ax = plt.subplots(figsize=(7, 4))
    for builder in sorted(df_fn["builder"].unique()):
        sub = df_fn[df_fn["builder"] == builder]
        by_k = sub.groupby("k")["n_units"].mean()
        ax.plot(
            by_k.index.tolist(),
            by_k.values.tolist(),
            marker="o",
            label=str(builder),
            color=_BUILDER_COLORS.get(str(builder), None),
        )
    ax.set_xlabel("k")
    ax.set_ylabel("mean remaining threshold units")
    ax.set_title("Units left after full simplify (vs k)")
    ax.legend()
    ax.set_ylim(bottom=0)
    fig_names.append("02_mean_units_vs_k")
    save_fig("02_mean_units_vs_k")

if not df_unit.empty and "status" in df_unit.columns:
    # 3) Force-trial outcome stacked by skip_reason
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pivot = (
        df_unit.fillna({"skip_reason": "(none)"})
        .groupby(["skip_reason", "status"])
        .size()
        .unstack(fill_value=0)
    )
    pivot.plot(kind="bar", stacked=True, ax=ax, width=0.8)
    ax.set_xlabel("skip_reason (reported; reduction still forced)")
    ax.set_ylabel("unit trials")
    ax.set_title("Force-reduce outcomes by skip_reason")
    ax.legend(title="status")
    plt.xticks(rotation=30, ha="right")
    fig_names.append("03_force_by_skip_reason")
    save_fig("03_force_by_skip_reason")

write_readme(RUN_DIR, df_fn, df_unit, fig_names)

# %% [markdown]
# Done.  Re-run `main()` for a fresh sweep; summary cells fall back to the
# newest `cm_simplify_minimality*` run directory when the current-session CSV
# is missing.
