# %% [markdown]
# BBM high-*k* measures probe (LUT / full-cover / sample-cover)
#
# Pick BBM update rules that already yielded **expression-based** ON/OFF prime
# implicants in ``bbm_high_k_primes_compare`` (``match=True``, ``k > 12``),
# then try tutorial-style CANA measures under a hard subprocess kill.
#
# Goal: see which APIs succeed from positional expression / primes **without**
# materializing a dense LUT or a Θ(``2**k``) coverage dict, and which hit
# ``pi_coverage`` / ``ts_coverage`` expansion or ``get_bit_lut`` (refuses above
# ``MAX_DENSE_LUT_K``).  Sampling probes estimate ``input_redundancy`` /
# ``edge_redundancy`` by matching covering PIs on a fixed minterm sample.
# **No** CANA measure code is patched — timeouts and errors are the result.
#
# Modes:
#
# - ``MODE = "starter"`` — same BBM rules as
#   ``bbm_high_k_redundancy_sampling``: enumerable-*k* exact cohort
#   (``11 ≤ k ≤ 18``) plus high-*k* rules (``k > 20``), ROF / unate / rest
# - ``MODE = "all_from_log"`` — every ``match=True`` row with ``k > 12`` from
#   that log (same 10s cutoff per probe)
#
# Output is teed to ``high_k/logs/bbm_high_k_measures_probe/``.

# %%
from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# notebooks/ → high_k/ → dev/
_DEV_DIR = Path(__file__).resolve().parents[2]
if str(_DEV_DIR) not in sys.path:
    sys.path.insert(0, str(_DEV_DIR))

from high_k.bbm_rules import (  # noqa: E402
    BBM_MODELS,
    as_bnet,
    resolve_model_dir,
)
from high_k.paths import ensure_cana_on_path  # noqa: E402
from high_k.run_log import run_log_dir, start_run_log  # noqa: E402
from high_k.timing import call_in_subprocess, fmt_s  # noqa: E402

ensure_cana_on_path()

from cana.io import import_detached_rule  # noqa: E402
from cana.nodes_builder import detached_rule_to_node  # noqa: E402
from cana.redundancy import (  # noqa: E402
    edge_redundancy,
    input_redundancy,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# "starter" | "all_from_log"
MODE = "starter"

TIMEOUT_S = 10.0
MIN_K = 13  # used when MODE == "all_from_log" (k > 12)
N_SAMPLES = 2**13  # minterm sample size for *_sample probes
SEED = 0

NOTEBOOK_NAME = "bbm_high_k_measures_probe"
COMPARE_LOG = (
    run_log_dir("bbm_high_k_primes_compare") / "bbm_high_k_primes_compare.log"
)

# Shared with bbm_high_k_redundancy_sampling (STARTER_EXACT + STARTER_HIGH).
# (model_id, target, expected_k, cohort)
STARTER: Tuple[Tuple[str, str, int, str], ...] = (
    # enumerable-k exact cohort (11 ≤ k ≤ 18)
    ("002", "v_MKK3", 11, "rest"),
    ("002", "v_Src", 11, "rest"),
    ("009", "v_ROS_CYT", 12, "rest"),
    ("018", "v_erbb13", 12, "unate"),
    ("004", "v_Nck", 14, "rof"),
    ("004", "v_Spry2", 14, "unate"),
    ("004", "v_p120_catenin", 16, "rof"),
    ("004", "v_B_catenin", 16, "rest"),
    ("004", "v_DGK", 18, "rof"),
    ("004", "v_PI3K", 18, "unate"),
    # high-k cohort (k > 20)
    ("263", "v_v33_acetyl_CoA_b2", 21, "rest"),
    ("004", "v_Shc", 27, "unate"),
    ("207", "v_TP53", 31, "rof"),
    ("251", "v_v52_Cdk1_cycB_Pool_b10", 36, "unate"),
    ("256", "v_v211_Oligomycin_b1", 80, "rof"),
)

# Probe ids (order = run / summary column order).
# effective_connectivity omitted: k_e = k - k_r (same cost as input_redundancy).
PROBE_IDS: Tuple[str, ...] = (
    "import",
    "one_symbol",
    "schemata_pi",
    "input_redundancy",
    "input_redundancy_sample",
    "edge_redundancy",
    "edge_redundancy_sample",
    "bias",
    "sensitivity",
    "input_symmetry",
)

# What each probe does (logged once at startup).
PROBE_METHODS: Dict[str, str] = {
    "import": (
        "detached positional import → BooleanNode "
        "(no LUT / no covers)"
    ),
    "one_symbol": (
        "expression cascade → one-symbol primes/schemata "
        "(no LUT / no covers)"
    ),
    "schemata_pi": (
        "PI schemata DataFrame from derived primes "
        "(no full covers)"
    ),
    "input_redundancy": (
        "exact k_r: force full pi_coverage (exact_k_max=k); "
        "refuses / times out when Θ(2**k) cover is too large"
    ),
    "input_redundancy_sample": (
        f"sample covers: estimate k_r from n={N_SAMPLES} minterms "
        "(match covering PIs; no pi_coverage dict)"
    ),
    "edge_redundancy": (
        "exact r_i: force full pi_coverage (exact_k_max=k); "
        "refuses / times out when Θ(2**k) cover is too large"
    ),
    "edge_redundancy_sample": (
        f"sample covers: estimate r_i from n={N_SAMPLES} minterms "
        "(match covering PIs; no pi_coverage dict)"
    ),
    "bias": (
        "build dense LUT (get_bit_lut), then popcount / 2**k"
    ),
    "sensitivity": (
        "build dense LUT, then single-flip average sensitivity"
    ),
    "input_symmetry": (
        "exact k_s: build full ts_coverage over all 2**k minterms"
    ),
}

log, LOG_PATH = start_run_log(NOTEBOOK_NAME)
log.info(
    f"MODE={MODE}  TIMEOUT_S={TIMEOUT_S}  MIN_K={MIN_K}  "
    f"N_SAMPLES={N_SAMPLES}  SEED={SEED}"
)
log.info(f"BBM_MODELS={BBM_MODELS}")
log.info(f"COMPARE_LOG={COMPARE_LOG}")
log.info(f"PROBES={PROBE_IDS}")
log.info("")
log.info("=== probe methods ===")
for probe in PROBE_IDS:
    log.info(f"  {probe:24s}  {PROBE_METHODS[probe]}")


# %% [markdown]
# ## Helpers: load rule, build node, timed probes


# %%
_TIME_CELL = r"(?:skip|timeout|error|\d+(?:\.\d+)?\s*(?:µs|ms|s))"
_ROW_RE = re.compile(
    rf"^(?:#\s*(?P<idx>\d+)\s+)?"
    rf"(?P<cohort>rof|unate|rest)\s+(?P<k>\d+)\s+"
    rf"(?P<rof>{_TIME_CELL})\s+(?P<pyeda>{_TIME_CELL})\s+"
    rf"(?P<primer>{_TIME_CELL})\s+"
    rf"(?P<n_on>\d+)\s+(?P<n_off>\d+)\s+(?P<match>\S+)\s+"
    rf"(?P<target>\S+)\s+/\s+(?P<model>.+?)\s*$"
)
_MODEL_ID_RE = re.compile(r"^\[id-(\d+)\]")


def load_bbm_rule(model_id: str, target: str) -> Tuple[str, str, str, str]:
    """Return ``(model_dir_name, target, expr, bnet_text)`` for one BBM rule."""
    model_dir = resolve_model_dir(BBM_MODELS, model_id)
    path = model_dir / "model.bnet"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("targets") and "," in line:
            continue
        if "," not in line:
            continue
        name, expr = line.split(",", 1)
        name = name.strip()
        expr = expr.strip()
        if name == target:
            return model_dir.name, target, expr, as_bnet(expr, target=target)
    raise KeyError(f"target {target!r} not in {path}")


def make_node(bnet_text: str, target: str):
    """Detached positional import → hub ``BooleanNode`` (no dense LUT)."""
    rule = import_detached_rule(bnet_text, fmt="bnet", target=target)
    return detached_rule_to_node(rule, id=0, inputs=list(range(rule.k)))


def _summarize(value: Any) -> Any:
    """Make probe payloads pipe-friendly / log-friendly."""
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


def _run_probe_body(bnet_text: str, target: str, probe: str) -> Dict[str, Any]:
    """Execute one probe in-process; return timing + summary value."""
    if probe == "import":
        t0 = time.perf_counter()
        node = make_node(bnet_text, target)
        dt = time.perf_counter() - t0
        return {
            "t": dt,
            "k": node.k,
            "value": {"k": node.k, "name": node.name},
        }

    node = make_node(bnet_text, target)
    t0 = time.perf_counter()
    if probe == "one_symbol":
        sch = node.outputs.get_one_symbol_schemata()
        value: Any = {
            0: len(sch.get(0, ()) or ()),
            1: len(sch.get(1, ()) or ()),
        }
    elif probe == "schemata_pi":
        df = node.schemata_look_up_table(type="pi")
        value = _summarize(df)
    elif probe == "input_redundancy":
        # Force the exact Θ(2**k) pi_coverage path (node default exact_k_max=14
        # would otherwise sample for k > 14).
        value = float(node.input_redundancy(exact_k_max=node.k))
    elif probe == "input_redundancy_sample":
        schemata = node.outputs.get_one_symbol_schemata()
        rng = np.random.default_rng(SEED)
        value = float(
            input_redundancy(
                schemata,
                norm=True,
                exact_k_max=-1,
                n_samples=N_SAMPLES,
                rng=rng,
            )
        )
    elif probe == "edge_redundancy":
        value = _summarize(node.edge_redundancy(exact_k_max=node.k))
    elif probe == "edge_redundancy_sample":
        schemata = node.outputs.get_one_symbol_schemata()
        rng = np.random.default_rng(SEED)
        value = _summarize(
            edge_redundancy(
                schemata,
                bound="mean",
                exact_k_max=-1,
                n_samples=N_SAMPLES,
                rng=rng,
            )
        )
    elif probe == "bias":
        value = float(node.bias())
    elif probe == "sensitivity":
        value = float(node.sensitivity())
    elif probe == "input_symmetry":
        value = float(node.input_symmetry())
    else:
        raise ValueError(f"unknown probe {probe!r}")
    dt = time.perf_counter() - t0
    return {
        "t": dt,
        "k": node.k,
        "value": value,
    }


def _probe_worker(bnet_text: str, target: str, probe: str, conn) -> None:
    try:
        conn.send(("ok", _run_probe_body(bnet_text, target, probe)))
    except Exception as exc:  # noqa: BLE001 — report to parent
        conn.send(("err", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def run_probe(
    bnet_text: str, target: str, probe: str, *, timeout_s: float
) -> Dict[str, Any]:
    """Hard-kill *probe* after *timeout_s*; return status row fields."""
    try:
        payload = call_in_subprocess(
            _probe_worker,
            (bnet_text, target, probe),
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


def parse_success_rows_from_compare_log(
    path: Path, *, min_k: int
) -> List[Dict[str, Any]]:
    """Rows with ``match=True`` and ``k >= min_k`` from the primes-compare log."""
    if not path.is_file():
        raise FileNotFoundError(path)
    out: List[Dict[str, Any]] = []
    seen: set = set()
    in_table = False
    for raw in path.read_text(encoding="utf-8").splitlines():
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
        if not m or m.group("match") != "True":
            continue
        k = int(m.group("k"))
        if k < min_k:
            continue
        model_raw = m.group("model")
        mid_m = _MODEL_ID_RE.match(model_raw)
        if not mid_m:
            continue
        model_id = mid_m.group(1)
        target = m.group("target")
        key = (model_id, target)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "model_id": model_id,
                "target": target,
                "k": k,
                "cohort": m.group("cohort"),
                "model": model_raw,
            }
        )
    return out


def cohort_funcs() -> List[Dict[str, Any]]:
    """Resolve which functions to probe under current ``MODE``."""
    if MODE == "starter":
        return [
            {
                "model_id": mid,
                "target": tgt,
                "k": k,
                "cohort": cohort,
            }
            for mid, tgt, k, cohort in STARTER
        ]
    if MODE == "all_from_log":
        return parse_success_rows_from_compare_log(COMPARE_LOG, min_k=MIN_K)
    raise ValueError(f"unknown MODE={MODE!r}")


# %% [markdown]
# ## Run probe matrix


# %%
funcs = cohort_funcs()
log.info("")
log.info(
    f"=== probe matrix  n_funcs={len(funcs)}  "
    f"n_probes={len(PROBE_IDS)}  timeout={TIMEOUT_S:g}s ==="
)

rows: List[Dict[str, Any]] = []
load_failed = 0

for i, fn in enumerate(funcs, start=1):
    model_id = str(fn["model_id"])
    target = str(fn["target"])
    cohort = str(fn["cohort"])
    expect_k = int(fn["k"])
    label = f"[id-{int(model_id):03d}]::{target}"

    try:
        model_name, _t, expr, bnet = load_bbm_rule(model_id, target)
    except Exception as exc:  # noqa: BLE001
        load_failed += 1
        log.info(f"#{i:4d} LOAD_FAIL {label}: {exc}")
        continue

    log.info(
        f"#{i:4d} {cohort:7s} expect_k={expect_k:3d}  {label}  "
        f"model={model_name[:48]}"
    )

    for probe in PROBE_IDS:
        r = run_probe(bnet, target, probe, timeout_s=TIMEOUT_S)
        row = {
            "model_id": model_id,
            "target": target,
            "cohort": cohort,
            "expect_k": expect_k,
            "probe": probe,
            "status": r["status"],
            "t": r["t"],
            "k": r["k"],
            "value": r["value"],
            "error": r["error"],
            "label": label,
        }
        rows.append(row)

        if r["status"] == "ok":
            log.info(
                f"       {probe:24s}  {fmt_s(r['t']):>10s}  "
                f"ok  value={_summarize(r['value'])!r}"
            )
        elif r["status"] == "timeout":
            log.info(
                f"       {probe:24s}  {fmt_s(TIMEOUT_S):>10s}  timeout"
            )
        else:
            err = r["error"] or ""
            short = err if len(err) <= 100 else err[:97] + "..."
            log.info(
                f"       {probe:24s}  {'error':>10s}  {short}"
            )

log.info("")
log.info(
    f"done: rows={len(rows)}  funcs_attempted={len(funcs)}  "
    f"load_failed={load_failed}"
)


# %% [markdown]
# ## Summary table (status / time per probe)


# %%
def _cell(row: Optional[Dict[str, Any]]) -> str:
    if row is None:
        return "—"
    st = row["status"]
    if st == "ok":
        return fmt_s(row["t"]).strip()
    if st == "timeout":
        return f"timeout({TIMEOUT_S:g}s)"
    return "error"


# Compact pivot: one line per function.
by_func: Dict[str, Dict[str, Dict[str, Any]]] = {}
func_meta: Dict[str, Dict[str, Any]] = {}
for row in rows:
    key = row["label"]
    by_func.setdefault(key, {})[row["probe"]] = row
    func_meta[key] = {
        "cohort": row["cohort"],
        "expect_k": row["expect_k"],
    }

header = (
    f"{'cohort':7s} {'k':>3s}  {'function':42s}  "
    + "  ".join(f"{p:>22s}" for p in PROBE_IDS)
)
log.info("")
log.info("=== summary (time | timeout | error) ===")
log.info(header)
for key in sorted(
    by_func.keys(),
    key=lambda k: (-int(func_meta[k]["expect_k"]), k),
):
    meta = func_meta[key]
    cells = [_cell(by_func[key].get(p)) for p in PROBE_IDS]
    short = key if len(key) <= 42 else key[:39] + "..."
    log.info(
        f"{meta['cohort']:7s} {meta['expect_k']:3d}  {short:42s}  "
        + "  ".join(f"{c:>22s}" for c in cells)
    )

# Counts per probe.
log.info("")
log.info("=== per-probe status counts ===")
for probe in PROBE_IDS:
    n_ok = n_to = n_err = 0
    for row in rows:
        if row["probe"] != probe:
            continue
        if row["status"] == "ok":
            n_ok += 1
        elif row["status"] == "timeout":
            n_to += 1
        else:
            n_err += 1
    log.info(
        f"{probe:24s}  ok={n_ok:4d}  timeout={n_to:4d}  error={n_err:4d}"
    )

log.info("")
log.info(f"log_path={LOG_PATH}")
log.close()
