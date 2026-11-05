# %% [markdown]
# BBM high-*k* legacy vs PI measures
#
# For every BBM update rule with ``k > 12``, time **bias**, **activities**, and
# **edge-redundancy** under two pipelines, with each subprocess stage hard-killed
# after ``TIMEOUT_S``.
#
# **Legacy (exact)**
#
# - import → dense LUT → ``bias_from_bit_lut`` / ``activities_from_bit_lut``
# - import → dense LUT → QM → ``edge_redundancy(exact_k_max=k)``
#
# **New (API defaults)**
#
# - import → expression-cascade PIs → ``bias_from_schemata`` /
#   ``activities_from_schemata`` / ``edge_redundancy`` (exact for ``k ≤ 14``,
#   sample for ``k > 14``)
#
# Overlapping stages are timed once and **summed** into measure totals so shared
# work is not double-counted.  Figures:
#
# 1. functions / successes vs *k* (1×3 overlapping bars: all | new ok | legacy ok)
# 2. cumulated completion ECDFs (1×3, legacy vs new overlay)
#
# Set ``REGENERATE_FROM_LOG_ONLY = True`` to rebuild figures from the existing
# run log without re-running the BBM sweep (does not overwrite that log).
#
# Also writes ``README.md`` (PI success fractions + fully-processable model
# counts under a ``k < MIN_K`` always-ok assumption), including on replot.
#
# Output is teed to ``high_k/logs/bbm_high_k_legacy_vs_pi_measures/``.

# %%
from __future__ import annotations

import re
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

# Expected for API-default schemata measures when k > exact_k_max (14).
warnings.filterwarnings(
    "ignore",
    message=r".*> exact_k_max=.*; sampling n_samples=.*",
    category=UserWarning,
)

# notebooks/ → high_k/ → dev/
_DEV_DIR = Path(__file__).resolve().parents[2]
if str(_DEV_DIR) not in sys.path:
    sys.path.insert(0, str(_DEV_DIR))

from high_k.bbm_rules import (  # noqa: E402
    BBM_MODELS,
    as_bnet,
    iter_bnet_rules,
    model_keywords,
    unique_regulators,
)
from high_k.paths import ensure_cana_on_path  # noqa: E402
from high_k.run_log import TeeLogger, run_log_dir, start_run_log  # noqa: E402
from high_k.timing import call_in_subprocess, fmt_s  # noqa: E402

ensure_cana_on_path()

from cana.io import import_detached_rule  # noqa: E402
from cana.nodes_builder import detached_rule_to_node  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MIN_K = 13  # keep rules with k > 12
TIMEOUT_S = 10.0
PROGRESS_EVERY = 25

# True → rebuild figures from the existing run log (no re-timing; keeps log).
# False → full BBM sweep (overwrites the run log).
REGENERATE_FROM_LOG_ONLY = True

NOTEBOOK_NAME = "bbm_high_k_legacy_vs_pi_measures"
FIG_DIR = run_log_dir(NOTEBOOK_NAME)
RUN_LOG_PATH = FIG_DIR / f"{NOTEBOOK_NAME}.log"

# Stage order = run order.
STAGE_IDS: Tuple[str, ...] = (
    "import",
    "lut",
    "qm",
    "legacy_bias",
    "legacy_activities",
    "legacy_edge_redundancy",
    "pi",
    "new_bias",
    "new_activities",
    "new_edge_redundancy",
)

# Prerequisite stage that must be ok before running this stage (None = always).
STAGE_PREREQ: Dict[str, Optional[str]] = {
    "import": None,
    "lut": "import",
    "qm": "lut",
    "legacy_bias": "lut",
    "legacy_activities": "lut",
    "legacy_edge_redundancy": "qm",
    "pi": "import",
    "new_bias": "pi",
    "new_activities": "pi",
    "new_edge_redundancy": "pi",
}

# measure → method → ordered stages whose times sum to the measure total.
MEASURE_CHAINS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "bias": {
        "legacy": ("import", "lut", "legacy_bias"),
        "new": ("import", "pi", "new_bias"),
    },
    "activities": {
        "legacy": ("import", "lut", "legacy_activities"),
        "new": ("import", "pi", "new_activities"),
    },
    "edge_redundancy": {
        "legacy": ("import", "lut", "qm", "legacy_edge_redundancy"),
        "new": ("import", "pi", "new_edge_redundancy"),
    },
}

MEASURES: Tuple[str, ...] = ("bias", "activities", "edge_redundancy")
METHODS: Tuple[str, ...] = ("legacy", "new")

METHOD_STYLE = {
    "legacy": {"label": "legacy (exact)", "color": "#d62728", "marker": "o"},
    "new": {"label": "new (API default)", "color": "#1f77b4", "marker": "s"},
}

STAGE_METHODS: Dict[str, str] = {
    "import": "detached positional import → BooleanNode",
    "lut": "build dense BitLut from positional expression",
    "qm": "Quine–McCluskey from bit_lut → one-symbol schemata",
    "legacy_bias": "bias_from_bit_lut (exact)",
    "legacy_activities": "activities_from_bit_lut (exact)",
    "legacy_edge_redundancy": "edge_redundancy(exact_k_max=k)",
    "pi": "get_one_symbol_schemata (expression cascade)",
    "new_bias": "bias_from_schemata (API default exact_k_max)",
    "new_activities": "activities_from_schemata (API default)",
    "new_edge_redundancy": "edge_redundancy (API default)",
}

_TIME_CELL = r"(?:skip|timeout|error|\d+(?:\.\d+)?\s*(?:µs|ms|s))"
_TIME_RE = re.compile(
    r"^(?:skip|timeout|error|(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>µs|ms|s))$"
)
_CFG_RE = re.compile(
    r"MIN_K=(?P<min_k>\d+)\s+TIMEOUT_S=(?P<timeout>\d+(?:\.\d+)?)"
)
_ROW_RE = re.compile(
    r"^#\s*(?P<idx>\d+)\s+k=\s*(?P<k>\d+)\s+"
    r"(?P<target>\S+)\s+/\s+(?P<model>.+?)\s*$"
)
_TOTAL_RE = re.compile(
    r"^\s*TOTAL\s+(?P<method>legacy|new)\s+"
    r"(?P<measure>bias|activities|edge_redundancy)\s+"
    rf"(?P<cell>{_TIME_CELL})\s*$"
)
_STAGE_ALT = "|".join(sorted(STAGE_IDS, key=len, reverse=True))
_STAGE_RE = re.compile(
    rf"^\s+(?P<stage>{_STAGE_ALT})\s+"
    rf"(?P<cell>{_TIME_CELL})\s+"
    r"(?P<status>ok|timeout|skip|error)\b"
)


def parse_time_cell(cell: str) -> Tuple[Optional[float], str]:
    """Parse a log timing cell → ``(seconds_or_None, status)``."""
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


def _short_model(name: str) -> str:
    """Match the truncated model label used in run-log row headers."""
    if len(name) <= 40:
        return name
    return name[:37] + "..."


def parse_run_log(
    path: Path,
) -> Tuple[List[Dict[str, Any]], float, int]:
    """Rebuild rows from a prior run log.

    Returns ``(rows, timeout_s, min_k)``.  Restores ``k`` / label fields,
    per-stage ``st_*`` / ``t_*``, and measure-total ``st_*`` / ``t_*``
    (enough to replot and write the summary README).
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
    current: Optional[Dict[str, Any]] = None
    in_table = False
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if line.startswith("=== BBM"):
            in_table = True
            continue
        if not in_table:
            continue
        if line.startswith("===") or line.startswith("done sweep:"):
            break
        m_row = _ROW_RE.match(line)
        if m_row:
            if current is not None:
                rows.append(current)
            model = m_row.group("model")
            target = m_row.group("target")
            current = {
                "k": int(m_row.group("k")),
                "model": model,
                "target": target,
                "label": f"{model}::{target}",
            }
            continue
        if current is None:
            continue
        m_stage = _STAGE_RE.match(line)
        if m_stage:
            stage = m_stage.group("stage")
            status = m_stage.group("status")
            t, _parsed_status = parse_time_cell(m_stage.group("cell"))
            if status != "ok":
                t = None
            current[f"st_{stage}"] = status
            current[f"t_{stage}"] = t
            continue
        m_tot = _TOTAL_RE.match(line)
        if not m_tot:
            continue
        method = m_tot.group("method")
        measure = m_tot.group("measure")
        t, status = parse_time_cell(m_tot.group("cell"))
        key = f"{method}_{measure}"
        current[f"st_{key}"] = status
        current[f"t_{key}"] = t
        current[f"err_{key}"] = None if status == "ok" else status
    if current is not None:
        rows.append(current)
    return rows, timeout_s, min_k


# %% [markdown]
# ## Helpers: node, stages, totals


# %%
def make_node(bnet_text: str, target: str):
    """Detached positional import → hub ``BooleanNode`` (no dense LUT)."""
    rule = import_detached_rule(bnet_text, fmt="bnet", target=target)
    return detached_rule_to_node(rule, id=0, inputs=list(range(rule.k)))


def _summarize(value: Any) -> Any:
    """Make stage payloads log-friendly."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _summarize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) <= 6 and all(
            isinstance(x, (bool, int, float)) for x in value
        ):
            return list(value)
        if value and all(isinstance(x, (int, float)) for x in value):
            return {
                "len": len(value),
                "min": float(min(value)),
                "max": float(max(value)),
                "mean": float(sum(value) / len(value)),
            }
        return f"len={len(value)}"
    shape = getattr(value, "shape", None)
    if shape is not None:
        return f"shape={tuple(shape)}"
    return repr(value)[:120]


def _prepare_node(bnet_text: str, target: str, stage: str):
    """Build a fresh node and any untimed prerequisites for *stage*."""
    node = make_node(bnet_text, target)
    if stage == "import":
        return node
    if stage == "lut":
        return node
    if stage == "qm":
        node.function.build_bit_lut_from_positional_expression()
        return node
    if stage in ("legacy_bias", "legacy_activities"):
        node.function.build_bit_lut_from_positional_expression()
        return node
    if stage == "legacy_edge_redundancy":
        node.function.build_bit_lut_from_positional_expression()
        node.function.compute_one_symbol_schemata_from_bit_lut()
        return node
    if stage == "pi":
        return node
    if stage in ("new_bias", "new_activities", "new_edge_redundancy"):
        node.function.get_one_symbol_schemata()
        return node
    raise ValueError(f"unknown stage {stage!r}")


def _run_stage_body(bnet_text: str, target: str, stage: str) -> Dict[str, Any]:
    """Execute one stage in-process; return timing + summary value."""
    if stage == "import":
        t0 = time.perf_counter()
        node = make_node(bnet_text, target)
        dt = time.perf_counter() - t0
        return {
            "t": dt,
            "k": int(node.k),
            "value": {"k": int(node.k), "name": node.name},
        }

    node = _prepare_node(bnet_text, target, stage)
    t0 = time.perf_counter()
    if stage == "lut":
        lut = node.function.build_bit_lut_from_positional_expression()
        value: Any = {"k": int(lut.k), "n": len(lut)}
    elif stage == "qm":
        sch = node.function.compute_one_symbol_schemata_from_bit_lut()
        value = {
            0: len(sch.get(0, ()) or ()),
            1: len(sch.get(1, ()) or ()),
        }
    elif stage == "legacy_bias":
        value = float(node.bias_from_bit_lut())
    elif stage == "legacy_activities":
        value = _summarize(node.activities_from_bit_lut())
    elif stage == "legacy_edge_redundancy":
        value = _summarize(node.edge_redundancy(exact_k_max=node.k))
    elif stage == "pi":
        sch = node.function.get_one_symbol_schemata()
        value = {
            0: len(sch.get(0, ()) or ()),
            1: len(sch.get(1, ()) or ()),
        }
    elif stage == "new_bias":
        value = float(node.bias_from_schemata())
    elif stage == "new_activities":
        value = _summarize(node.activities_from_schemata())
    elif stage == "new_edge_redundancy":
        value = _summarize(node.edge_redundancy())
    else:
        raise ValueError(f"unknown stage {stage!r}")
    dt = time.perf_counter() - t0
    return {"t": dt, "k": int(node.k), "value": value}


def _stage_worker(bnet_text: str, target: str, stage: str, conn) -> None:
    # Re-apply in the child (fork inherits filters; keep explicit for clarity).
    warnings.filterwarnings(
        "ignore",
        message=r".*> exact_k_max=.*; sampling n_samples=.*",
        category=UserWarning,
    )
    try:
        conn.send(("ok", _run_stage_body(bnet_text, target, stage)))
    except Exception as exc:  # noqa: BLE001 — report to parent
        conn.send(("err", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def run_stage(
    bnet_text: str, target: str, stage: str, *, timeout_s: float
) -> Dict[str, Any]:
    """Hard-kill *stage* after *timeout_s*; return status row fields."""
    try:
        payload = call_in_subprocess(
            _stage_worker,
            (bnet_text, target, stage),
            timeout_s=timeout_s,
        )
        return {
            "status": "ok",
            "t": float(payload["t"]),
            "k": int(payload["k"]),
            "value": payload.get("value"),
            "error": None,
        }
    except TimeoutError as exc:
        return {
            "status": "timeout",
            "t": float(timeout_s),
            "k": None,
            "value": None,
            "error": str(exc),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "t": None,
            "k": None,
            "value": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def skipped_stage(prereq: str) -> Dict[str, Any]:
    """Status payload when a prerequisite stage failed."""
    return {
        "status": "skip",
        "t": None,
        "k": None,
        "value": None,
        "error": f"skipped: prerequisite {prereq!r} not ok",
    }


def chain_result(
    stage_results: Dict[str, Dict[str, Any]], chain: Tuple[str, ...]
) -> Dict[str, Any]:
    """Sum stage times when every stage in *chain* is ``ok``."""
    total = 0.0
    for stage in chain:
        r = stage_results[stage]
        if r["status"] != "ok":
            return {
                "status": str(r["status"]),
                "t": None,
                "error": r.get("error"),
            }
        total += float(r["t"])
    return {"status": "ok", "t": total, "error": None}


# %% [markdown]
# ## Collect (or reload from log)


# %%
if REGENERATE_FROM_LOG_ONLY:
    if not RUN_LOG_PATH.is_file():
        raise FileNotFoundError(
            f"REGENERATE_FROM_LOG_ONLY set but missing {RUN_LOG_PATH}"
        )
    rows, TIMEOUT_S, MIN_K = parse_run_log(RUN_LOG_PATH)
    # Do not truncate the run log; tee replot notes alongside it.
    LOG_PATH = FIG_DIR / f"{NOTEBOOK_NAME}.replot.log"
    log = TeeLogger(LOG_PATH)
    log._install_warning_tee()
    log.info(f"log_path={LOG_PATH}")
    log.info(
        f"REGENERATE_FROM_LOG_ONLY=True  source={RUN_LOG_PATH}  "
        f"rows={len(rows)}  MIN_K={MIN_K}  TIMEOUT_S={TIMEOUT_S}"
    )
    log.info(f"FIG_DIR={FIG_DIR}")
else:
    log, LOG_PATH = start_run_log(NOTEBOOK_NAME)
    log.info(f"MIN_K={MIN_K}  TIMEOUT_S={TIMEOUT_S}  BBM_MODELS={BBM_MODELS}")
    log.info(f"FIG_DIR={FIG_DIR}")
    log.info(f"STAGES={STAGE_IDS}")
    log.info("")
    log.info("=== stage methods ===")
    for stage in STAGE_IDS:
        log.info(f"  {stage:28s}  {STAGE_METHODS[stage]}")
    log.info("")
    log.info("=== measure chains (summed) ===")
    for measure in MEASURES:
        for method in METHODS:
            chain = MEASURE_CHAINS[measure][method]
            log.info(f"  {measure:16s} {method:7s}  {' + '.join(chain)}")

    rows = []
    bbm_seen = 0
    bbm_kept = 0

    log.info("")
    log.info(f"=== BBM (all rules with k > {MIN_K - 1}) ===")
    if not BBM_MODELS.is_dir():
        log.info(f"BBM models not found at {BBM_MODELS}")
    else:
        for model_name, target, expr in iter_bnet_rules(BBM_MODELS):
            bbm_seen += 1
            k = len(unique_regulators(expr))
            if k < MIN_K:
                continue
            bbm_kept += 1

            short_model = model_name
            if len(short_model) > 40:
                short_model = short_model[:37] + "..."
            label = f"{model_name}::{target}"
            bnet = as_bnet(expr, target=target)

            if bbm_kept == 1 or bbm_kept % PROGRESS_EVERY == 0:
                log.info(
                    f"--- progress kept={bbm_kept} seen={bbm_seen} "
                    f"current k={k} {target} / {short_model} ---"
                )

            log.info(
                f"#{bbm_kept:4d} k={k:3d}  {target} / {short_model}"
            )

            stage_results: Dict[str, Dict[str, Any]] = {}
            for stage in STAGE_IDS:
                prereq = STAGE_PREREQ[stage]
                if (
                    prereq is not None
                    and stage_results[prereq]["status"] != "ok"
                ):
                    r = skipped_stage(prereq)
                else:
                    r = run_stage(bnet, target, stage, timeout_s=TIMEOUT_S)
                stage_results[stage] = r

                if r["status"] == "ok":
                    log.info(
                        f"       {stage:28s}  {fmt_s(r['t']):>10s}  "
                        f"ok  value={_summarize(r['value'])!r}"
                    )
                elif r["status"] == "timeout":
                    log.info(
                        f"       {stage:28s}  {fmt_s(TIMEOUT_S):>10s}  "
                        f"timeout"
                    )
                elif r["status"] == "skip":
                    log.info(
                        f"       {stage:28s}  {'skip':>10s}  {r['error']}"
                    )
                else:
                    err = r["error"] or ""
                    short = err if len(err) <= 100 else err[:97] + "..."
                    log.info(
                        f"       {stage:28s}  {'error':>10s}  {short}"
                    )

            row: Dict[str, Any] = {
                "k": k,
                "model": model_name,
                "target": target,
                "label": label,
            }
            for stage in STAGE_IDS:
                r = stage_results[stage]
                row[f"st_{stage}"] = r["status"]
                row[f"t_{stage}"] = r["t"]
                row[f"err_{stage}"] = r["error"]

            for measure in MEASURES:
                for method in METHODS:
                    cr = chain_result(
                        stage_results, MEASURE_CHAINS[measure][method]
                    )
                    key = f"{method}_{measure}"
                    row[f"st_{key}"] = cr["status"]
                    row[f"t_{key}"] = cr["t"]
                    row[f"err_{key}"] = cr["error"]
                    status_cell = (
                        fmt_s(cr["t"]).strip()
                        if cr["status"] == "ok"
                        else cr["status"]
                    )
                    log.info(
                        f"       TOTAL {method:7s} {measure:16s}  "
                        f"{status_cell}"
                    )

            rows.append(row)

    log.info("")
    log.info(
        f"done sweep: kept={bbm_kept}  seen={bbm_seen}  rows={len(rows)}"
    )


# %% [markdown]
# ## Summary counts


# %%
if not REGENERATE_FROM_LOG_ONLY:
    log.info("")
    log.info("=== per-stage status counts ===")
    for stage in STAGE_IDS:
        n_ok = n_to = n_err = n_skip = 0
        for row in rows:
            st = row[f"st_{stage}"]
            if st == "ok":
                n_ok += 1
            elif st == "timeout":
                n_to += 1
            elif st == "skip":
                n_skip += 1
            else:
                n_err += 1
        log.info(
            f"{stage:28s}  ok={n_ok:4d}  timeout={n_to:4d}  "
            f"error={n_err:4d}  skip={n_skip:4d}"
        )

log.info("")
log.info("=== per-measure method success ===")
for measure in MEASURES:
    for method in METHODS:
        key = f"{method}_{measure}"
        n_ok = sum(1 for r in rows if r[f"st_{key}"] == "ok")
        n_att = len(rows)
        pct = 100.0 * n_ok / n_att if n_att else 0.0
        log.info(
            f"{measure:16s} {method:7s}  ok={n_ok:4d}/{n_att:4d}  ({pct:.1f}%)"
        )


# %% [markdown]
# ## Plots: success ratio by *k* and completion curves


# %%
def _counts_by_k(data: List[Dict[str, Any]]) -> Dict[int, int]:
    """Total number of timed rules at each *k*."""
    by_k: Dict[int, int] = defaultdict(int)
    for r in data:
        by_k[int(r["k"])] += 1
    return dict(by_k)


def _ok_counts_by_k(
    data: List[Dict[str, Any]], status_key: str
) -> Dict[int, int]:
    """Number of ``ok`` outcomes at each *k* for *status_key*."""
    by_k: Dict[int, int] = defaultdict(int)
    for r in data:
        if str(r.get(status_key)) == "ok":
            by_k[int(r["k"])] += 1
    return dict(by_k)


def _overall_success(
    data: List[Dict[str, Any]], status_key: str
) -> Tuple[int, int, Optional[float]]:
    n_ok = 0
    n_att = 0
    for r in data:
        st = str(r[status_key])
        if st == "skip":
            continue
        n_att += 1
        if st == "ok":
            n_ok += 1
    if n_att == 0:
        return 0, 0, None
    return n_ok, n_att, n_ok / n_att


def plot_success_ratio_by_k(
    data: List[Dict[str, Any]], outfile: Path
) -> None:
    """1×3 overlapping bars: total / legacy-ok / new-ok counts vs *k*."""
    fig, axes = plt.subplots(
        1, 3, figsize=(13.5, 4.2), sharey=True, constrained_layout=True
    )
    total_by_k = _counts_by_k(data)
    all_ks = sorted(total_by_k)
    n_total = len(data)
    for ax, measure in zip(axes, MEASURES):
        if not all_ks:
            ax.set_title(measure.replace("_", " "))
            continue
        x = np.arange(len(all_ks))
        width = 0.82
        totals = [total_by_k[k] for k in all_ks]
        ax.bar(
            x,
            totals,
            width=width,
            color="0.75",
            edgecolor="white",
            linewidth=0.4,
            label=f"all functions ({n_total})",
            zorder=1,
        )
        # Draw new then legacy so both show through as overlapping histograms.
        for method, alpha, zorder in (
            ("new", 0.55, 2),
            ("legacy", 0.55, 3),
        ):
            status_key = f"st_{method}_{measure}"
            ok_by_k = _ok_counts_by_k(data, status_key)
            y = [ok_by_k.get(k, 0) for k in all_ks]
            style = METHOD_STYLE[method]
            n_ok, n_att, rate = _overall_success(data, status_key)
            if rate is None:
                label = f"{style['label']} ok (n/a)"
            else:
                label = (
                    f"{style['label']} ok "
                    f"({n_ok}/{n_att} = {100.0 * rate:.0f}%)"
                )
            ax.bar(
                x,
                y,
                width=width,
                color=style["color"],
                alpha=alpha,
                edgecolor="white",
                linewidth=0.4,
                label=label,
                zorder=zorder,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in all_ks], rotation=90, fontsize=7)
        y_max = max(totals) if totals else 1
        ax.set_ylim(0.0, y_max * 1.08)
        ax.set_xlabel("k (inputs)")
        ax.set_title(measure.replace("_", " "))
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
    axes[0].set_ylabel("number of functions")
    fig.suptitle(
        f"BBM k>{MIN_K - 1}: functions per k and successes "
        f"(legacy vs new)",
        fontsize=11,
    )
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    log.info(f"wrote {outfile}")


def _completion_curve(
    times: List[float], n_attempted: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Step ECDF of completion: fraction done by time *t* (failures never finish)."""
    if n_attempted <= 0:
        return np.array([0.0]), np.array([0.0])
    if not times:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])
    ts = np.sort(np.asarray(times, dtype=float))
    ys = np.arange(1, len(ts) + 1, dtype=float) / float(n_attempted)
    # Include a left origin so the step starts at 0.
    x = np.concatenate([[0.0], ts, [ts[-1]]])
    y = np.concatenate([[0.0], ys, [ys[-1]]])
    return x, y


def plot_completion_curves(
    data: List[Dict[str, Any]], outfile: Path
) -> None:
    """1×3 ECDFs: cumulated completion vs wall time for legacy vs new."""
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.5, 4.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    all_ok_times: List[float] = []
    for ax, measure in zip(axes, MEASURES):
        for method in METHODS:
            status_key = f"st_{method}_{measure}"
            time_key = f"t_{method}_{measure}"
            style = METHOD_STYLE[method]
            ok_times = [
                float(r[time_key])
                for r in data
                if r[status_key] == "ok" and r[time_key] is not None
            ]
            all_ok_times.extend(ok_times)
            n_att = sum(1 for r in data if r[status_key] != "skip")
            n_ok, _n, rate = _overall_success(data, status_key)
            x, y = _completion_curve(ok_times, n_att)
            if rate is None:
                label = f"{style['label']} (n/a)"
            else:
                label = (
                    f"{style['label']} "
                    f"({n_ok}/{n_att} = {100.0 * rate:.0f}%)"
                )
            ax.step(
                x,
                y,
                where="post",
                color=style["color"],
                label=label,
                linewidth=1.8,
            )
        # Guide: one stage timeout (lower bound on a failed chain is not drawn;
        # this marks the per-stage budget for context).
        ax.axvline(
            TIMEOUT_S,
            color="0.55",
            linestyle="--",
            linewidth=1.0,
            label=f"stage timeout={TIMEOUT_S:g}s",
        )
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("wall time (s)  [summed stages]")
        ax.set_title(measure.replace("_", " "))
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    # Shared log-x range across panels (include stage-timeout guide).
    x_hi = max([TIMEOUT_S] + all_ok_times) if all_ok_times else TIMEOUT_S
    x_lo = min(all_ok_times) if all_ok_times else TIMEOUT_S / 100.0
    if x_lo <= 0:
        positives = [t for t in all_ok_times if t > 0]
        x_lo = min(positives) if positives else TIMEOUT_S / 100.0
    pad = 1.15
    axes[0].set_xlim(x_lo / pad, x_hi * pad)
    axes[0].set_ylabel("fraction completed")
    fig.suptitle(
        f"BBM k>{MIN_K - 1}: cumulated completion (legacy vs new)",
        fontsize=11,
    )
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    log.info(f"wrote {outfile}")


CELL_COLLECTIVE_TAG = "cell-collective"

_PI_SPECS: Tuple[Tuple[str, str, str], ...] = (
    ("legacy", "qm", "LUT → Quine–McCluskey one-symbol schemata"),
    ("new", "pi", "expression-cascade one-symbol schemata"),
)


def _pi_table_lines(data: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, int], int]:
    """Return ``(markdown rows, ok_by_pipeline, n_either)`` for PI stages."""
    n_rules = len(data)
    lines: List[str] = []
    pi_ok: Dict[str, int] = {}
    for pipeline, stage, desc in _PI_SPECS:
        n_ok = sum(1 for r in data if r.get(f"st_{stage}") == "ok")
        pi_ok[pipeline] = n_ok
        pct = 100.0 * n_ok / n_rules if n_rules else 0.0
        lines.append(
            f"| {pipeline} | `{stage}` | {desc} | {n_ok} | {n_rules} | "
            f"{pct:.1f}% |"
        )
    n_either = sum(
        1
        for r in data
        if r.get("st_qm") == "ok" or r.get("st_pi") == "ok"
    )
    either_pct = 100.0 * n_either / n_rules if n_rules else 0.0
    lines.append(
        f"| either | `qm` ∨ `pi` | at least one pipeline | "
        f"{n_either} | {n_rules} | {either_pct:.1f}% |"
    )
    return lines, pi_ok, n_either


def _measure_table_lines(data: List[Dict[str, Any]]) -> List[str]:
    n_rules = len(data)
    lines: List[str] = []
    for measure in MEASURES:
        for method in METHODS:
            key = f"{method}_{measure}"
            n_ok = sum(1 for r in data if r.get(f"st_{key}") == "ok")
            pct = 100.0 * n_ok / n_rules if n_rules else 0.0
            lines.append(
                f"| {measure.replace('_', ' ')} | {method} | "
                f"{n_ok}/{n_rules} | {pct:.1f}% |"
            )
    return lines


def _model_coverage_lines(
    models: List[str],
    high_by_model: Dict[str, List[str]],
    row_by_key: Dict[Tuple[str, str], Dict[str, Any]],
) -> Tuple[List[str], Dict[str, int]]:
    """Markdown block + counts for fully-processable models in *models*."""

    def _model_pi_ok(model: str, stage: str) -> bool:
        targets = high_by_model.get(model, [])
        if not targets:
            return True
        for target in targets:
            r = row_by_key.get((_short_model(model), target))
            if r is None or r.get(f"st_{stage}") != "ok":
                return False
        return True

    def _model_either_ok(model: str) -> bool:
        targets = high_by_model.get(model, [])
        if not targets:
            return True
        for target in targets:
            r = row_by_key.get((_short_model(model), target))
            if r is None:
                return False
            if r.get("st_qm") != "ok" and r.get("st_pi") != "ok":
                return False
        return True

    n_models = len(models)
    n_no_high = sum(1 for m in models if m not in high_by_model)
    n_with_high = n_models - n_no_high
    n_full_legacy = sum(1 for m in models if _model_pi_ok(m, "qm"))
    n_full_new = sum(1 for m in models if _model_pi_ok(m, "pi"))
    n_full_either = sum(1 for m in models if _model_either_ok(m))
    counts = {
        "n_models": n_models,
        "n_no_high": n_no_high,
        "n_with_high": n_with_high,
        "n_full_legacy": n_full_legacy,
        "n_full_new": n_full_new,
        "n_full_either": n_full_either,
    }
    if n_models == 0:
        return ["_No models in this cohort._", ""], counts

    lines = [
        f"| | count | / {n_models} models |",
        "|---|---:|---:|",
        f"| models in cohort | {n_models} | 100% |",
        f"| models with no `k >= {MIN_K}` rule (auto-ok) | "
        f"{n_no_high} | {100.0 * n_no_high / n_models:.1f}% |",
        f"| models with ≥1 high-*k* rule | {n_with_high} | "
        f"{100.0 * n_with_high / n_models:.1f}% |",
        f"| fully processable — **legacy** (`qm`) | "
        f"{n_full_legacy} | "
        f"{100.0 * n_full_legacy / n_models:.1f}% |",
        f"| fully processable — **new** (`pi`) | "
        f"{n_full_new} | "
        f"{100.0 * n_full_new / n_models:.1f}% |",
        f"| fully processable — **either** | "
        f"{n_full_either} | "
        f"{100.0 * n_full_either / n_models:.1f}% |",
        "",
        "Of the high-*k*-bearing models only:",
        "",
        f"- legacy PI for all high-*k* rules: "
        f"**{n_full_legacy - n_no_high}** / {n_with_high}",
        f"- new PI for all high-*k* rules: "
        f"**{n_full_new - n_no_high}** / {n_with_high}",
        f"- either pipeline for all high-*k* rules: "
        f"**{n_full_either - n_no_high}** / {n_with_high}",
        "",
    ]
    return lines, counts


def write_summary_readme(
    data: List[Dict[str, Any]], outfile: Path
) -> None:
    """Write PI / model-coverage summary (works for live rows and log reload)."""
    n_rules = len(data)
    pi_lines, pi_ok, n_either = _pi_table_lines(data)
    measure_lines = _measure_table_lines(data)

    row_by_key = {
        (_short_model(str(r["model"])), str(r["target"])): r for r in data
    }
    all_models: List[str] = []
    cc_models: List[str] = []
    high_by_model: Dict[str, List[str]] = defaultdict(list)
    if BBM_MODELS.is_dir():
        for p in sorted(BBM_MODELS.iterdir()):
            if not p.is_dir():
                continue
            all_models.append(p.name)
            kws = {k.lower() for k in model_keywords(p)}
            if CELL_COLLECTIVE_TAG in kws:
                cc_models.append(p.name)
        for model_name, target, expr in iter_bnet_rules(BBM_MODELS):
            if len(unique_regulators(expr)) >= MIN_K:
                high_by_model[model_name].append(target)

    cc_short = {_short_model(m) for m in cc_models}
    cc_rows = [
        r for r in data if _short_model(str(r["model"])) in cc_short
    ]
    cc_pi_lines, cc_pi_ok, cc_n_either = _pi_table_lines(cc_rows)
    cc_measure_lines = _measure_table_lines(cc_rows)

    cover_all, counts_all = _model_coverage_lines(
        all_models, high_by_model, row_by_key
    )
    cover_cc, counts_cc = _model_coverage_lines(
        cc_models, high_by_model, row_by_key
    )

    assumption = (
        f"Assumption: every update rule with `k < {MIN_K}` can be processed. "
        "A model is then fully processable iff **every** rule with "
        f"`k >= {MIN_K}` obtained prime implicants under that pipeline."
    )

    lines = [
        f"# {NOTEBOOK_NAME} — summary",
        "",
        f"- `MIN_K={MIN_K}` (timed rules with `k >= {MIN_K}`; "
        f"assume all `k < {MIN_K}` succeed)",
        f"- `TIMEOUT_S={TIMEOUT_S:g}`",
        f"- high-*k* rules timed: **{n_rules}**",
        f"- Cell Collective tag in metadata: `{CELL_COLLECTIVE_TAG}`",
        f"- figures: `success_ratio_by_k.png`, `completion_curves.png`",
        "",
        "## All BBM models",
        "",
        "### Prime implicants on high-*k* rules",
        "",
        "Fraction of timed rules (`k >= MIN_K`) where the PI / one-symbol "
        "schemata stage finished `ok` within the per-stage timeout.",
        "",
        "| pipeline | stage | method | ok | n | fraction |",
        "|---|---|---|---:|---:|---:|",
        *pi_lines,
        "",
        "### Measure success on high-*k* rules",
        "",
        "| measure | pipeline | ok/n | fraction |",
        "|---|---|---:|---:|",
        *measure_lines,
        "",
        "### Fully processable BBM models",
        "",
        assumption,
        "",
    ]
    if not all_models:
        lines.extend(
            [
                f"_BBM models not found at `{BBM_MODELS}`; "
                "skipped model inventory._",
                "",
            ]
        )
    else:
        lines.extend(cover_all)

    lines.extend(
        [
            f"## Cell Collective models (`{CELL_COLLECTIVE_TAG}`)",
            "",
            f"Subset of BBM models whose `metadata.json` keywords include "
            f"`{CELL_COLLECTIVE_TAG}` "
            f"({len(cc_models)} models; {len(cc_rows)} high-*k* rules timed).",
            "",
            "### Prime implicants on high-*k* rules",
            "",
            "| pipeline | stage | method | ok | n | fraction |",
            "|---|---|---|---:|---:|---:|",
            *cc_pi_lines,
            "",
            "### Measure success on high-*k* rules",
            "",
            "| measure | pipeline | ok/n | fraction |",
            "|---|---|---:|---:|",
            *cc_measure_lines,
            "",
            "### Fully processable Cell Collective models",
            "",
            assumption,
            "",
            *cover_cc,
        ]
    )

    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"wrote {outfile}")
    log.info(
        f"PI ok (all): legacy={pi_ok.get('legacy', 0)}/{n_rules}  "
        f"new={pi_ok.get('new', 0)}/{n_rules}  either={n_either}/{n_rules}"
    )
    log.info(
        f"PI ok (cell-collective): "
        f"legacy={cc_pi_ok.get('legacy', 0)}/{len(cc_rows)}  "
        f"new={cc_pi_ok.get('new', 0)}/{len(cc_rows)}  "
        f"either={cc_n_either}/{len(cc_rows)}"
    )
    if counts_all["n_models"]:
        log.info(
            f"fully processable (all, assume k<{MIN_K} ok): "
            f"legacy={counts_all['n_full_legacy']}/{counts_all['n_models']}  "
            f"new={counts_all['n_full_new']}/{counts_all['n_models']}  "
            f"either={counts_all['n_full_either']}/{counts_all['n_models']}"
        )
    if counts_cc["n_models"]:
        log.info(
            f"fully processable (cell-collective, assume k<{MIN_K} ok): "
            f"legacy={counts_cc['n_full_legacy']}/{counts_cc['n_models']}  "
            f"new={counts_cc['n_full_new']}/{counts_cc['n_models']}  "
            f"either={counts_cc['n_full_either']}/{counts_cc['n_models']}"
        )


if rows:
    plot_success_ratio_by_k(rows, FIG_DIR / "success_ratio_by_k.png")
    plot_completion_curves(rows, FIG_DIR / "completion_curves.png")
    write_summary_readme(rows, FIG_DIR / "README.md")
else:
    log.info("no rows; skipping plots and README")

log.info("")
log.info(f"log_path={LOG_PATH}")
log.close()
