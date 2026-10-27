# %% [markdown]
# All-LUT DCM simplify ↔ LUT roundtrip
#
# Sibling of `all_lut_cm_roundtrip.py`.  Enumerate Boolean LUTs for every
# `k <= MAX_K`, build a canalizing map for each builder in `BUILDERS`, optionally
# apply DCM `simplify` (`simplify_canalizing_map` / 
# `BooleanNetwork.dynamics_canalization_map`), reconstruct the LUT, and report
# fail fractions by `(builder, simplify, k)`.
#
# Four combinations: legacy/canonical × simplify False/True.
#
# * **Exhaustive** when `2^(2^k) <= SAMPLE_N` (default 2048 → full enum for
#   `k <= 3`).
# * **Sample** exactly `SAMPLE_N` uniform random LUTs when the space is larger.
#   The same sampled ids are reused for every `(builder, simplify)` mode.
#
# Set `SMOKE = True` to cap at `SMOKE_MAX_K`.  Each run writes a directory under
# `dev/logs/` containing the log, numbered figures, and a summary `README.md`.

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

from bbm_cm_roundtrip import (  # noqa: E402
    NodeRoundtrip,
    StepTimeout,
    TeeLogger,
    _fmt_secs,
    _run_in_process,
    _timing_suffix,
    is_exact_roundtrip,
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
# CM builders to roundtrip (same sampled LUT ids for each).
BUILDERS = ("legacy", "canonical")
# DCM stock-simplify modes (False = full CM; True = apply stock collapse).
SIMPLIFY_MODES = (False, True)
# Wall-clock limits per function step; skip on expiry. None / <=0 disables.
TIMEOUT_CM_S: Optional[float] = None
TIMEOUT_LUT_S: Optional[float] = None
# Progress line every N evaluated/skipped functions within a k-bucket.
PROGRESS_EVERY = 256

LOG_DIR = _DEV_DIR / "logs"

_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
_RUN_STEM = (
    f"dcm_simplify_all_lut_roundtrip_smoke_{_TIMESTAMP}"
    if SMOKE
    else f"dcm_simplify_all_lut_roundtrip_{_TIMESTAMP}"
)
RUN_DIR = LOG_DIR / _RUN_STEM
LOG_PATH = RUN_DIR / f"{_RUN_STEM}.log"
# Figures write directly into RUN_DIR (no nested _figures/).
FIG_DIR = RUN_DIR

# %% [markdown]
# ## Helpers — LUT enumeration + stock DCM simplify


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
    return BooleanNode.from_bit_lut(
        outputs, name=lut_name(k, func_id), inputs=inputs, id=func_id
    )


def apply_stock_dcm_simplify(cm: nx.DiGraph) -> nx.DiGraph:
    """Apply DCM simplify to a canalizing-map graph (copy).

    Delegates to :func:`~cana.canalization.canalizing_map.simplify_canalizing_map`
    — the same helper used by ``BooleanNetwork.dynamics_canalization_map``.
    """
    return simplify_canalizing_map(cm, copy=True, remove_isolates=True)


def _cm_worker_payload(
    payload: Tuple[Any, ...], conn: Any
) -> None:
    """Child: build CM, optionally stock-simplify, return graph."""
    try:
        name, nid, inputs, outputs, builder, simplify = payload
        node = BooleanNode(
            id=nid,
            name=name,
            k=len(inputs),
            inputs=list(inputs),
            outputs=list(outputs),
            network=None,
        )
        cm = node.canalizing_map(builder=builder)
        if simplify:
            cm = apply_stock_dcm_simplify(cm)
        conn.send(("ok", cm))
    except Exception as exc:  # noqa: BLE001
        conn.send(("err", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def _lut_worker(payload: Tuple[Any, ...], conn: Any) -> None:
    try:
        cm, input_order = payload
        result = lut_from_canalizing_map(cm, input_order=input_order)
        conn.send(("ok", result))
    except Exception as exc:  # noqa: BLE001
        conn.send(("err", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def roundtrip_node_with_simplify(
    node: BooleanNode,
    max_k: int,
    *,
    builder: str = "legacy",
    simplify: bool = False,
    timeout_cm_s: Optional[float] = None,
    timeout_lut_s: Optional[float] = None,
) -> NodeRoundtrip:
    """Build CM → optional stock DCM simplify → LUT reconstruct."""
    out = NodeRoundtrip(name=node.name, k=int(node.k))
    if out.k > max_k:
        out.skipped = True
        out.skip_reason = f"k={out.k} > MAX_K={max_k}"
        return out
    try:
        t0 = time.perf_counter()
        try:
            if timeout_cm_s is not None and timeout_cm_s > 0:
                cm = _run_in_process(
                    _cm_worker_payload,
                    (
                        node.name,
                        node.id,
                        list(node.inputs),
                        list(node.outputs),
                        builder,
                        simplify,
                    ),
                    timeout_cm_s,
                )
            else:
                cm = node.canalizing_map(builder=builder)
                if simplify:
                    cm = apply_stock_dcm_simplify(cm)
        except StepTimeout:
            out.cm_s = time.perf_counter() - t0
            out.skipped = True
            out.skip_reason = (
                f"TIMEOUT_CM after {_fmt_secs(out.cm_s)} "
                f"(limit={_fmt_secs(timeout_cm_s)})"
            )
            return out
        out.cm_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        try:
            if timeout_lut_s is not None and timeout_lut_s > 0:
                result = _run_in_process(
                    _lut_worker,
                    (cm, list(node.inputs)),
                    timeout_lut_s,
                )
            else:
                result = lut_from_canalizing_map(cm, input_order=node.inputs)
        except StepTimeout:
            out.lut_s = time.perf_counter() - t1
            out.skipped = True
            out.skip_reason = (
                f"TIMEOUT_LUT after {_fmt_secs(out.lut_s)} "
                f"(limit={_fmt_secs(timeout_lut_s)})"
            )
            return out
        out.lut_s = time.perf_counter() - t1

        out.conflict = bool(result.has_conflict)
        out.incomplete = bool(result.is_incomplete)
        out.exact = is_exact_roundtrip(node.outputs, result)
        if not out.exact:
            if out.conflict:
                out.fail_reason = "conflict"
            elif out.incomplete:
                out.fail_reason = "incomplete"
            else:
                out.fail_reason = "mismatch"
    except Exception as exc:  # noqa: BLE001
        out.error = f"{type(exc).__name__}: {exc}"
        out.fail_reason = f"exception: {out.error}"
    return out


@dataclass
class KBucketResult:
    builder: str
    simplify: bool
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


def _simplify_label(simplify: bool) -> str:
    return "True" if simplify else "False"


# %% [markdown]
# ## Run enumeration (smoke or full)
#
# Call `main()` from a notebook cell, or run the file as a script.
# Returns the run directory (`RUN_DIR`).


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
    log.info(
        f"SIMPLIFY_MODES={','.join(_simplify_label(s) for s in SIMPLIFY_MODES)}"
    )
    log.info(f"TIMEOUT_CM_S={TIMEOUT_CM_S}")
    log.info(f"TIMEOUT_LUT_S={TIMEOUT_LUT_S}")
    log.info(f"PROGRESS_EVERY={PROGRESS_EVERY}")
    log.info(f"run_dir={RUN_DIR}")
    log.info(f"log_path={LOG_PATH}")
    log.info(f"timestamp={_TIMESTAMP}")
    log.info("")

    buckets: List[KBucketResult] = []
    n_fn_eval = 0
    n_fn_fail = 0
    total_cm_s = 0.0
    total_lut_s = 0.0
    failing_ks: List[str] = []
    # Cache sampled ids per k so every (builder, simplify) sees the same LUTs.
    func_ids_by_k: Dict[int, Tuple[str, List[int]]] = {}

    for builder in BUILDERS:
        for simplify in SIMPLIFY_MODES:
            slog = _simplify_label(simplify)
            log.info(f"######## builder={builder} simplify={slog} ########")
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
                    simplify=simplify,
                    k=k,
                    mode=enum_mode,
                    n_total_space=space,
                    n_planned=len(func_ids),
                )
                log.info(
                    f"=== builder={builder}  simplify={slog}  k={k}  "
                    f"mode={enum_mode}  space={space}  "
                    f"planned={br.n_planned} ==="
                )

                t_k0 = time.perf_counter()
                for i, func_id in enumerate(func_ids):
                    node = make_node(k, func_id)
                    nr = roundtrip_node_with_simplify(
                        node,
                        max_k=k_hi,
                        builder=builder,
                        simplify=simplify,
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
                                f"  FAIL  {nr.name}  k={nr.k}  "
                                f"reason={nr.fail_reason}"
                                f"{_timing_suffix(nr)}"
                            )
                        else:
                            if (
                                i < 3
                                or (i + 1) % PROGRESS_EVERY == 0
                                or i + 1 == len(func_ids)
                            ):
                                log.info(
                                    f"  OK    {nr.name}  k={nr.k}"
                                    f"{_timing_suffix(nr)}"
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
                    failing_ks.append(
                        f"builder={builder} simplify={slog} k={k} "
                        f"fail={br.n_fail}/{br.n_eval}"
                    )
                log.info(
                    f"  k_summary  eval={br.n_eval}  fail={br.n_fail}  "
                    f"skip={br.n_skip}  error={br.n_error}  "
                    f"k_fail={k_fail}  "
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
    log.info(f"BUILDERS={','.join(BUILDERS)}")
    log.info(
        f"SIMPLIFY_MODES={','.join(_simplify_label(s) for s in SIMPLIFY_MODES)}"
    )
    log.info(f"function_fail_fraction = {n_fn_fail}/{n_fn_eval} = {fn_frac:.6f}")
    log.info(f"k_fail_fraction        = {n_k_fail}/{n_k_eval} = {k_frac:.6f}")
    log.info(f"cm_time_total          = {_fmt_secs(total_cm_s)}")
    log.info(f"lut_time_total         = {_fmt_secs(total_lut_s)}")
    for builder in BUILDERS:
        for simplify in SIMPLIFY_MODES:
            slog = _simplify_label(simplify)
            b_buckets = [
                b
                for b in buckets
                if b.builder == builder and b.simplify == simplify
            ]
            b_eval = sum(b.n_eval for b in b_buckets)
            b_fail = sum(b.n_fail for b in b_buckets)
            b_frac = (b_fail / b_eval) if b_eval else float("nan")
            log.info(
                f"builder={builder} simplify={slog}  "
                f"function_fail_fraction = "
                f"{b_fail}/{b_eval} = {b_frac:.6f}"
            )
    if failing_ks:
        log.info("failing_k:")
        for item in failing_ks:
            log.info(f"  {item}")
    else:
        log.info("failing_k: (none)")
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
# `__main__`).  Summary cells below point at `RUN_DIR` / `LOG_PATH`.

# %%
# RUN_DIR = main()
# LOG_PATH = RUN_DIR / f"{RUN_DIR.name}.log"
# FIG_DIR = RUN_DIR

# %% [markdown]
# ## Parse log + headline stats


# %%
_RE_HEADER = re.compile(r"^([A-Za-z0-9_]+)=(.*)$")
_RE_K = re.compile(
    r"^=== builder=(?P<builder>\S+)  simplify=(?P<simplify>\S+)  "
    r"k=(?P<k>\d+)  mode=(?P<mode>\S+)  space=(?P<space>\d+)  "
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
_RE_COMBO_FRAC = re.compile(
    r"^builder=(?P<builder>\S+)\s+simplify=(?P<simplify>\S+)\s+"
    r"function_fail_fraction\s*=\s*(.+)$"
)
_RE_SECTION = re.compile(
    r"^######## builder=(?P<builder>\S+) simplify=(?P<simplify>\S+) ########$"
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


def _parse_simplify(token: str) -> bool:
    return token.strip().lower() in ("true", "1", "yes")


def parse_log(
    path: Path,
) -> Tuple[Dict[str, str], pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    header: Dict[str, str] = {}
    nodes: List[Dict[str, Any]] = []
    buckets: List[Dict[str, Any]] = []
    final: Dict[str, str] = {}

    cur: Optional[Dict[str, Any]] = None
    section_builder: Optional[str] = None
    section_simplify: Optional[bool] = None
    in_header = True

    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")

            m_sec = _RE_SECTION.match(line)
            if m_sec:
                in_header = False
                section_builder = m_sec.group("builder")
                section_simplify = _parse_simplify(m_sec.group("simplify"))
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
                if cur is not None and "k_fail" not in cur:
                    buckets.append(cur)
                cur = {
                    "builder": m_k.group("builder"),
                    "simplify": _parse_simplify(m_k.group("simplify")),
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
                node["builder"] = cur.get("builder", section_builder or "legacy")
                node["simplify"] = cur.get(
                    "simplify",
                    section_simplify if section_simplify is not None else False,
                )
                nodes.append(node)
                continue

            m_c = _RE_COMBO_FRAC.match(line)
            if m_c:
                key = (
                    f"builder_{m_c.group('builder')}_simplify_"
                    f"{m_c.group('simplify')}_fail_fraction"
                )
                final[key] = m_c.group(3).strip()
                continue

            m = _RE_FINAL.match(line)
            if m:
                final[m.group(1)] = m.group(2).strip()

    if cur is not None:
        buckets.append(cur)

    return header, pd.DataFrame(nodes), pd.DataFrame(buckets), final


def summarize_log(
    path: Path,
) -> Tuple[Dict[str, str], pd.DataFrame, pd.DataFrame, Dict[str, str]]:
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
        print("=== Per-(builder, simplify, k) summaries ===")
        cols = [
            c
            for c in [
                "builder",
                "simplify",
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
        if "builder" in df_k.columns and "simplify" in df_k.columns:
            print()
            print("=== Per-(builder, simplify) aggregate ===")
            for (builder, simplify), sub in df_k.groupby(
                ["builder", "simplify"], sort=True
            ):
                bev = int(sub["eval"].sum())
                bfa = int(sub["fail"].sum())
                print(
                    f"  {builder} simplify={_simplify_label(bool(simplify))}: "
                    f"eval={bev}  fail={bfa}  "
                    f"fail_fraction={bfa}/{bev} = "
                    f"{bfa / bev if bev else float('nan'):.6f}"
                )

    print()
    print("=== FAIL reasons (logged) ===")
    fail_rows = df_nodes.loc[df_nodes["status"] == "FAIL"]
    if fail_rows.empty:
        print("(none)")
    else:
        group_cols = [
            c for c in ["builder", "simplify", "fail_reason"] if c in fail_rows.columns
        ]
        print(fail_rows.groupby(group_cols, dropna=False).size().to_string())
    return header, df_nodes, df_k, final


def _newest_run_dir() -> Optional[Path]:
    cands = sorted(
        (
            p
            for p in LOG_DIR.glob("dcm_simplify_all_lut_roundtrip*")
            if p.is_dir()
        ),
        key=lambda p: p.stat().st_mtime,
    )
    return cands[-1] if cands else None


def resolve_run_paths() -> Tuple[Path, Path, Path]:
    """Return ``(run_dir, log_path, fig_dir)`` for this session or newest run."""
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
    header, df_nodes, df_k, final = summarize_log(LOG_PATH)
else:
    header, df_nodes, df_k, final = {}, pd.DataFrame(), pd.DataFrame(), {}
    print(f"No log at {LOG_PATH} — run main() first.")

# %% [markdown]
# ## Figures + README
#
# PNGs and `README.md` land inside `RUN_DIR`.


# %%
def save_fig(name: str) -> Path:
    path = FIG_DIR / f"{name}.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved {path.name}")
    plt.show()
    return path


_COMBO_COLORS = {
    ("legacy", False): "#264653",
    ("legacy", True): "#e76f51",
    ("canonical", False): "#2a9d8f",
    ("canonical", True): "#e9c46a",
}


def _combo_label(builder: str, simplify: bool) -> str:
    return f"{builder} / simplify={_simplify_label(simplify)}"


def _combos_in(df: pd.DataFrame) -> List[Tuple[str, bool]]:
    if df.empty or "builder" not in df.columns:
        return [("legacy", False)]
    if "simplify" not in df.columns:
        return [(b, False) for b in sorted(df["builder"].dropna().unique())]
    out: List[Tuple[str, bool]] = []
    for builder, simplify in (
        df[["builder", "simplify"]].drop_duplicates().itertuples(index=False)
    ):
        out.append((str(builder), bool(simplify)))
    return sorted(out, key=lambda x: (x[0], x[1]))


def write_readme(
    run_dir: Path,
    header: Dict[str, str],
    df_k: pd.DataFrame,
    final: Dict[str, str],
    fig_names: List[str],
) -> Path:
    """Write summary README.md into the run directory."""
    stem = run_dir.name
    lines: List[str] = [f"# {stem}", ""]
    lines.append(
        "All-LUT CM → stock DCM simplify (True/False) → LUT roundtrip for "
        "`legacy` and `canonical` builders (4 combinations)."
    )
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append("| Setting | Value |")
    lines.append("|---------|-------|")
    ts = header.get("timestamp", stem.rsplit("_", 2)[-2] + "_" + stem.rsplit("_", 1)[-1])
    lines.append(f"| timestamp | `{ts}` |")
    lines.append(f"| mode | `{header.get('mode', '')}` |")
    lines.append(f"| BUILDERS | `{header.get('BUILDERS', '')}` |")
    lines.append(f"| SIMPLIFY_MODES | `{header.get('SIMPLIFY_MODES', '')}` |")
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
    if not df_k.empty and "eval" in df_k.columns:
        combos = _combos_in(df_k)
        # Pivot fail counts by k for a compact table.
        ks = sorted(df_k["k"].astype(int).unique().tolist())
        header_cells = ["k", "mode"]
        for builder, simplify in combos:
            header_cells.append(_combo_label(builder, simplify) + " fail")
        lines.append("| " + " | ".join(header_cells) + " |")
        lines.append("|" + "|".join(["------"] * len(header_cells)) + "|")
        for k in ks:
            krows = df_k[df_k["k"] == k]
            mode = str(krows["mode"].iloc[0]) if len(krows) else ""
            cells = [str(k), mode]
            for builder, simplify in combos:
                sub = krows[
                    (krows["builder"] == builder) & (krows["simplify"] == simplify)
                ]
                if sub.empty:
                    cells.append("—")
                else:
                    fa = int(sub["fail"].iloc[0])
                    ev = int(sub["eval"].iloc[0])
                    cells.append(f"{fa} / {ev}")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

        lines.append("Per-(builder, simplify) totals:")
        lines.append("")
        for builder, simplify in combos:
            sub = df_k[
                (df_k["builder"] == builder) & (df_k["simplify"] == simplify)
            ]
            bev = int(sub["eval"].sum())
            bfa = int(sub["fail"].sum())
            frac = (bfa / bev) if bev else float("nan")
            lines.append(
                f"- **{_combo_label(builder, simplify)}:** "
                f"{bfa} / {bev} = **{frac:.2%}**"
            )
        lines.append("")
        if final.get("function_fail_fraction"):
            lines.append(
                f"- **overall function fail rate:** "
                f"`{final['function_fail_fraction']}`"
            )
        if final.get("cm_time_total") or final.get("lut_time_total"):
            lines.append(
                f"- **timing:** cm ≈ {final.get('cm_time_total', 'n/a')}, "
                f"lut ≈ {final.get('lut_time_total', 'n/a')}"
            )
        lines.append("")
    else:
        lines.append("(no k_summary data)")
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

if not df_k.empty and "eval" in df_k.columns:
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

    combos = _combos_in(df_k)
    ks_all = sorted(df_k["k"].astype(int).unique().tolist())
    n_c = max(len(combos), 1)
    width = 0.8 / n_c

    # 1) Fail rate vs k by (builder, simplify)
    fig, ax = plt.subplots()
    sub = df_k[df_k["eval"] > 0].copy()
    max_rate = 0.05
    for ci, (builder, simplify) in enumerate(combos):
        csub = sub[
            (sub["builder"] == builder) & (sub["simplify"] == simplify)
        ]
        if csub.empty:
            continue
        rate = csub["fail"] / csub["eval"]
        if len(rate):
            max_rate = max(max_rate, float(rate.max()))
        offsets = csub["k"].astype(float) + (ci - (n_c - 1) / 2) * width
        color = _COMBO_COLORS.get((builder, simplify), f"C{ci}")
        ax.bar(
            offsets,
            rate,
            width=width * 0.9,
            label=_combo_label(builder, simplify),
            color=color,
            alpha=0.85,
        )
        for x, row in zip(offsets, csub.itertuples()):
            r = row.fail / row.eval
            ax.text(
                x,
                r,
                f"{int(row.fail)}/{int(row.eval)}",
                ha="center",
                va="bottom",
                fontsize=6,
            )
    ax.set_xticks(ks_all)
    ax.set_ylim(0, max_rate * 1.25 if max_rate > 0 else 0.05)
    ax.set_xlabel("k (inputs)")
    ax.set_ylabel("fail rate")
    ax.set_title("FAIL rate vs k by builder × simplify")
    ax.legend(frameon=False, fontsize=8)
    save_fig("01_fail_rate_vs_k")
    fig_names_written.append("01_fail_rate_vs_k")

    # 2) Outcomes per k — one panel per combo
    n_panels = n_c
    fig, axes = plt.subplots(
        1, n_panels, figsize=(4.2 * n_panels, 4), squeeze=False
    )
    for ax, (builder, simplify) in zip(axes[0], combos):
        bdf = df_k[
            (df_k["builder"] == builder) & (df_k["simplify"] == simplify)
        ]
        if bdf.empty:
            ax.set_title(f"{_combo_label(builder, simplify)} (no data)")
            continue
        ks = bdf["k"].astype(int)
        bottoms = np.zeros(len(bdf))
        ok_counts = (bdf["eval"] - bdf["fail"]).clip(lower=0)
        series = [
            ("OK", ok_counts, "#2a9d8f"),
            ("FAIL", bdf["fail"], "#e76f51"),
            ("SKIP", bdf["skip"], "#e9c46a"),
            ("ERROR", bdf["error"], "#264653"),
        ]
        for label, vals, color in series:
            ax.bar(ks, vals, bottom=bottoms, label=label, color=color)
            bottoms = bottoms + vals.to_numpy(dtype=float)
        ax.set_xlabel("k")
        ax.set_ylabel("functions")
        ax.set_title(_combo_label(builder, simplify), fontsize=9)
        ax.legend(frameon=False, fontsize=7)
    save_fig("02_outcomes_per_k")
    fig_names_written.append("02_outcomes_per_k")

    # 3) Timing totals vs k
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, title in (
        (axes[0], "cm_total_s", "CM time total"),
        (axes[1], "lut_total_s", "LUT time total"),
    ):
        if col not in df_k.columns or df_k[col].isna().all():
            ax.set_title(f"{title} (no data)")
            continue
        for ci, (builder, simplify) in enumerate(combos):
            bdf = df_k[
                (df_k["builder"] == builder) & (df_k["simplify"] == simplify)
            ]
            if bdf.empty:
                continue
            offsets = bdf["k"].astype(float) + (ci - (n_c - 1) / 2) * width
            color = _COMBO_COLORS.get((builder, simplify), f"C{ci}")
            ax.bar(
                offsets,
                bdf[col].fillna(0.0),
                width=width * 0.9,
                label=_combo_label(builder, simplify),
                color=color,
                alpha=0.85,
            )
        ax.set_xticks(ks_all)
        ax.set_xlabel("k")
        ax.set_ylabel("seconds")
        ax.set_title(f"{title} vs k")
        ax.legend(frameon=False, fontsize=7)
    save_fig("03_timing_totals_vs_k")
    fig_names_written.append("03_timing_totals_vs_k")

    # 4) Coverage (independent of combo; use first)
    fig, ax = plt.subplots()
    builder0, simplify0 = combos[0]
    cov = df_k[
        (df_k["builder"] == builder0) & (df_k["simplify"] == simplify0)
    ]
    ax.bar(
        cov["k"] - 0.15, cov["planned"], width=0.3, label="planned", color="#2a9d8f"
    )
    ax.bar(
        cov["k"] + 0.15,
        cov["space"],
        width=0.3,
        label="space 2^(2^k)",
        color="#e9c46a",
    )
    ax.set_yscale("log")
    ax.set_xlabel("k")
    ax.set_ylabel("count (log)")
    ax.set_title("Coverage: planned vs full Boolean space")
    ax.legend(frameon=False)
    save_fig("04_coverage")
    fig_names_written.append("04_coverage")

    # 5) FAIL reasons if any
    if not df_nodes.empty and (df_nodes["status"] == "FAIL").any():
        fails = df_nodes[df_nodes["status"] == "FAIL"]
        fail_combos = _combos_in(fails)
        fig, axes = plt.subplots(
            1,
            len(fail_combos),
            figsize=(4.5 * len(fail_combos), 4),
            squeeze=False,
        )
        for ax, (builder, simplify) in zip(axes[0], fail_combos):
            mask = (fails["builder"] == builder) & (fails["simplify"] == simplify)
            vc = fails.loc[mask, "fail_reason"].value_counts()
            if vc.empty:
                ax.set_title(f"{_combo_label(builder, simplify)} (none)")
                continue
            ax.bar(vc.index.astype(str), vc.values, color="#e76f51")
            ax.set_ylabel("logged FAIL lines")
            ax.set_title(_combo_label(builder, simplify), fontsize=9)
            ax.tick_params(axis="x", rotation=20)
        save_fig("05_fail_reasons")
        fig_names_written.append("05_fail_reasons")

    write_readme(FIG_DIR, header, df_k, final, fig_names_written)
else:
    print("Skip figures / README — no k_summary data (run main() first).")

# %% [markdown]
# ## Tables — failing LUTs (logged FAIL lines)


# %%
if not df_nodes.empty:
    fails = df_nodes[df_nodes["status"] == "FAIL"][
        [
            c
            for c in [
                "builder",
                "simplify",
                "k",
                "node",
                "fail_reason",
                "cm_s",
                "lut_s",
            ]
            if c in df_nodes.columns
        ]
    ]
    print(f"FAIL rows ({len(fails)}; showing up to 40):")
    print(fails.head(40).to_string(index=False))
else:
    print("No node rows to tabulate.")
