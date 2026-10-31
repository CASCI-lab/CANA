# %% [markdown]
# BBM high-*k* measures probe (LUT dependence)
#
# Pick BBM update rules that already yielded **expression-based** ON/OFF prime
# implicants in ``bbm_high_k_primes_compare`` (``match=True``, ``k > 12``),
# then try tutorial-style CANA measures under a hard subprocess kill.
#
# Goal: see which APIs succeed from positional expression / primes **without**
# materializing a dense LUT, and which hit ``pi_coverage`` Θ(``2**k``) expansion
# or ``get_bit_lut`` (refuses above ``MAX_DENSE_LUT_K``).  **No** CANA measure
# code is patched to avoid LUT builds — timeouts and errors are the result.
#
# Modes:
#
# - ``MODE = "starter"`` — five hand-picked rules with ``k > 20`` (ROF + unate
#   non-ROF + rest), from the compare log
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# "starter" | "all_from_log"
MODE = "starter"

TIMEOUT_S = 10.0
MIN_K = 13  # used when MODE == "all_from_log" (k > 12)

NOTEBOOK_NAME = "bbm_high_k_measures_probe"
COMPARE_LOG = (
    run_log_dir("bbm_high_k_primes_compare") / "bbm_high_k_primes_compare.log"
)

# Hand-picked from bbm_high_k_primes_compare.log (match=True, k > 20).
# (model_id, target, expected_k, cohort)
STARTER: Tuple[Tuple[str, str, int, str], ...] = (
    ("256", "v_v211_Oligomycin_b1", 80, "rof"),
    ("207", "v_TP53", 31, "rof"),
    ("251", "v_v52_Cdk1_cycB_Pool_b10", 36, "unate"),
    ("004", "v_Shc", 27, "unate"),
    ("263", "v_v33_acetyl_CoA_b2", 21, "rest"),
)

# Probe ids (order = run / summary column order).
PROBE_IDS: Tuple[str, ...] = (
    "import",
    "one_symbol",
    "schemata_pi",
    "input_redundancy",
    "effective_connectivity",
    "edge_effectiveness",
    "bias",
    "sensitivity",
    "input_symmetry",
)

log, LOG_PATH = start_run_log(NOTEBOOK_NAME)
log.info(f"MODE={MODE}  TIMEOUT_S={TIMEOUT_S}  MIN_K={MIN_K}")
log.info(f"BBM_MODELS={BBM_MODELS}")
log.info(f"COMPARE_LOG={COMPARE_LOG}")
log.info(f"PROBES={PROBE_IDS}")


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
    """Execute one probe in-process; return timing + lut + summary value."""
    if probe == "import":
        t0 = time.perf_counter()
        node = make_node(bnet_text, target)
        dt = time.perf_counter() - t0
        return {
            "t": dt,
            "lut": node.outputs.bit_lut is not None,
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
        value = float(node.input_redundancy())
    elif probe == "effective_connectivity":
        value = float(node.effective_connectivity())
    elif probe == "edge_effectiveness":
        value = _summarize(node.edge_effectiveness())
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
        "lut": node.outputs.bit_lut is not None,
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
            "lut": bool(payload["lut"]),
            "k": int(payload["k"]),
            "value": payload.get("value"),
            "error": None,
        }
    except TimeoutError as exc:
        return {
            "status": "timeout",
            "t": float(timeout_s),
            "lut": None,
            "k": None,
            "value": None,
            "error": str(exc),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "t": None,
            "lut": None,
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
            "lut": r["lut"],
            "k": r["k"],
            "value": r["value"],
            "error": r["error"],
            "label": label,
        }
        rows.append(row)

        if r["status"] == "ok":
            lut_s = "Y" if r["lut"] else "N"
            log.info(
                f"       {probe:24s}  {fmt_s(r['t']):>10s}  "
                f"ok  lut={lut_s}  value={_summarize(r['value'])!r}"
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
# ## Summary table (status / time / lut per probe)


# %%
def _cell(row: Optional[Dict[str, Any]]) -> str:
    if row is None:
        return "—"
    st = row["status"]
    if st == "ok":
        lut = "Y" if row["lut"] else "N"
        return f"{fmt_s(row['t']).strip()}/lut={lut}"
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
    + "  ".join(f"{p:>18s}" for p in PROBE_IDS)
)
log.info("")
log.info("=== summary (time/lut=Y|N | timeout | error) ===")
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
        + "  ".join(f"{c:>18s}" for c in cells)
    )

# Counts per probe.
log.info("")
log.info("=== per-probe status counts ===")
for probe in PROBE_IDS:
    n_ok = n_to = n_err = 0
    n_lut = 0
    for row in rows:
        if row["probe"] != probe:
            continue
        if row["status"] == "ok":
            n_ok += 1
            if row["lut"]:
                n_lut += 1
        elif row["status"] == "timeout":
            n_to += 1
        else:
            n_err += 1
    log.info(
        f"{probe:24s}  ok={n_ok:4d} (lut_Y={n_lut})  "
        f"timeout={n_to:4d}  error={n_err:4d}"
    )

log.info("")
log.info(f"log_path={LOG_PATH}")
log.close()
