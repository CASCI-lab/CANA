# %% [markdown]
# BBM high-*k* redundancy sampling
#
# Given ON/OFF prime implicants (expression cascade: ROF → Primer-B), estimate
# ``input_redundancy`` / ``edge_redundancy`` by sampling minterms and matching
# covering PIs — **without** building the Θ(``2**k``) ``pi_coverage`` dict.
#
# - **Enumerable *k*** (``MIN_K_EXACT ≤ k ≤ MAX_K_EXACT``): compare sample
#   estimates (``n ≥ 2**10``) to exact ``BooleanNode`` measures. Exact vs
#   sampling only makes sense for ``k > 10`` so a ``2**10`` sample is a proper
#   subsample of ``2**k``.
# - **High *k*** (``k > MAX_K_EXACT``): exact coverage is impractical; report
#   bootstrap CIs.
#
# Output is teed to ``high_k/logs/bbm_high_k_redundancy_sampling/``.

# %%
from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# notebooks/ → high_k/ → dev/
_DEV_DIR = Path(__file__).resolve().parents[2]
if str(_DEV_DIR) not in sys.path:
    sys.path.insert(0, str(_DEV_DIR))

from high_k.bbm_rules import (  # noqa: E402
    BBM_MODELS,
    as_bnet,
    resolve_model_dir,
    unique_regulators,
)
from high_k.paths import ensure_cana_on_path  # noqa: E402
from high_k.run_log import run_log_dir, start_run_log  # noqa: E402
from high_k.timing import call_in_subprocess, fmt_s  # noqa: E402

ensure_cana_on_path()

from cana.boolean_node import BooleanNode  # noqa: E402
from cana.io import import_detached_rule  # noqa: E402
from cana.nodes_builder import detached_rule_to_node  # noqa: E402
from cana.function_outputs.one_symbol import primes_to_one_symbol  # noqa: E402
from cana.primes.cascade import on_off_primes_from_expression  # noqa: E402
from cana.function_properties.redundancy import (  # noqa: E402
    bootstrap_edge_redundancy,
    bootstrap_input_redundancy,
    edge_redundancy_from_coverage,
    input_redundancy_from_coverage,
    sample_pi_coverage,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# "starter" | "all_from_log"
MODE = "starter"

TIMEOUT_S = 10.0
MIN_K_EXACT = 11  # k > 10 so n=2**10 is a proper subsample
MAX_K_EXACT = 18  # try exact pi_coverage up to here (≤ MAX_DENSE_LUT_K)
MIN_N_SAMPLES = 2**10
N_SAMPLES_GRID: Tuple[int, ...] = (2**10, 2**11, 2**12, 2**13, 2**14)
# Single n for exact-vs-sample timing (most accurate grid size).
N_TIME_COMPARE = max(N_SAMPLES_GRID)
N_BOOT = 1000
CI = 0.95
SEED = 0

NOTEBOOK_NAME = "bbm_high_k_redundancy_sampling"
COMPARE_LOG = (
    run_log_dir("bbm_high_k_primes_compare") / "bbm_high_k_primes_compare.log"
)
FIG_DIR = run_log_dir(NOTEBOOK_NAME)

# Enumerable-k accuracy cohort (k > 10; BBM rules with known-good primes).
# (model_id, target, expected_k, cohort)
STARTER_EXACT: Tuple[Tuple[str, str, int, str], ...] = (
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
)

# High-k bootstrap cohort (from bbm_high_k_primes_compare / measures_probe).
STARTER_HIGH: Tuple[Tuple[str, str, int, str], ...] = (
    ("256", "v_v211_Oligomycin_b1", 80, "rof"),
    ("207", "v_TP53", 31, "rof"),
    ("251", "v_v52_Cdk1_cycB_Pool_b10", 36, "unate"),
    ("004", "v_Shc", 27, "unate"),
    ("263", "v_v33_acetyl_CoA_b2", 21, "rest"),
)

log, LOG_PATH = start_run_log(NOTEBOOK_NAME)
log.info(
    f"MODE={MODE}  TIMEOUT_S={TIMEOUT_S}  "
    f"MIN_K_EXACT={MIN_K_EXACT}  MAX_K_EXACT={MAX_K_EXACT}  "
    f"MIN_N_SAMPLES={MIN_N_SAMPLES}  N_SAMPLES_GRID={N_SAMPLES_GRID}  "
    f"N_TIME_COMPARE={N_TIME_COMPARE}  N_BOOT={N_BOOT}  CI={CI}  SEED={SEED}"
)
log.info(f"BBM_MODELS={BBM_MODELS}")
log.info(f"COMPARE_LOG={COMPARE_LOG}")
log.info(f"FIG_DIR={FIG_DIR}")


# %% [markdown]
# ## Helpers


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


def make_node(bnet_text: str, target: str) -> BooleanNode:
    rule = import_detached_rule(bnet_text, fmt="bnet", target=target)
    return detached_rule_to_node(rule, id=0, inputs=list(range(rule.k)))


def _primes_worker(expr: str, conn) -> None:
    try:
        res = on_off_primes_from_expression(expr)
        if res.status != "ok":
            conn.send(
                (
                    "err",
                    f"primes status={res.status!r} method={res.method!r}: "
                    f"{res.message}",
                )
            )
            return
        conn.send(
            (
                "ok",
                {
                    "on": res.on_primes,
                    "off": res.off_primes,
                    "method": res.method,
                    "n_on": res.n_on,
                    "n_off": res.n_off,
                },
            )
        )
    except Exception as exc:  # noqa: BLE001
        conn.send(("err", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def get_primes(
    expr: str, *, timeout_s: float
) -> Dict[str, Any]:
    """Hard-kill cascade primes; return status payload."""
    t0 = time.perf_counter()
    try:
        payload = call_in_subprocess(
            _primes_worker, (expr,), timeout_s=timeout_s
        )
        return {
            "status": "ok",
            "t": time.perf_counter() - t0,
            "on": payload["on"],
            "off": payload["off"],
            "method": payload["method"],
            "n_on": payload["n_on"],
            "n_off": payload["n_off"],
            "error": None,
        }
    except TimeoutError as exc:
        return {
            "status": "timeout",
            "t": float(timeout_s),
            "on": None,
            "off": None,
            "method": None,
            "n_on": None,
            "n_off": None,
            "error": str(exc),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "t": time.perf_counter() - t0,
            "on": None,
            "off": None,
            "method": None,
            "n_on": None,
            "n_off": None,
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


def cohort_exact() -> List[Dict[str, Any]]:
    """Enumerable-*k* accuracy set (compare log only has *k*>12)."""
    return [
        {
            "model_id": mid,
            "target": tgt,
            "k": k,
            "cohort": cohort,
        }
        for mid, tgt, k, cohort in STARTER_EXACT
    ]


def cohort_high() -> List[Dict[str, Any]]:
    if MODE == "starter":
        return [
            {
                "model_id": mid,
                "target": tgt,
                "k": k,
                "cohort": cohort,
            }
            for mid, tgt, k, cohort in STARTER_HIGH
        ]
    return parse_success_rows_from_compare_log(COMPARE_LOG, min_k=21)


# %% [markdown]
# ## Accuracy: exact vs sampled (enumerable *k*)


# %%
exact_funcs = cohort_exact()
accuracy_rows: List[Dict[str, Any]] = []

log.info("")
log.info(
    f"=== accuracy (exact vs sample)  n_funcs={len(exact_funcs)}  "
    f"k in [{MIN_K_EXACT},{MAX_K_EXACT}]  n >= {MIN_N_SAMPLES} ==="
)

for i, fn in enumerate(exact_funcs, start=1):
    model_id = str(fn["model_id"])
    target = str(fn["target"])
    cohort = str(fn["cohort"])
    expect_k = int(fn["k"])
    label = f"[id-{int(model_id):03d}]::{target}"

    try:
        model_name, _t, expr, bnet = load_bbm_rule(model_id, target)
    except Exception as exc:  # noqa: BLE001
        log.info(f"#{i:4d} LOAD_FAIL {label}: {exc}")
        continue

    names = unique_regulators(expr)
    k = len(names)
    if k != expect_k:
        log.info(
            f"#{i:4d} WARN k mismatch expect={expect_k} got={k}  {label}"
        )
    if k < MIN_K_EXACT:
        log.info(f"#{i:4d} skip (k={k} < MIN_K_EXACT={MIN_K_EXACT}) {label}")
        continue
    if k > MAX_K_EXACT:
        log.info(f"#{i:4d} skip (k={k} > MAX_K_EXACT) {label}")
        continue

    primes = get_primes(expr, timeout_s=TIMEOUT_S)
    if primes["status"] != "ok":
        log.info(
            f"#{i:4d} {cohort:7s} k={k:3d}  {label}  primes={primes['status']}  "
            f"{primes.get('error')}"
        )
        continue

    schemata = primes_to_one_symbol(
        primes["on"], primes["off"], input_names=names
    )
    node = make_node(bnet, target)
    # Attach expression-derived schemata so exact path uses same primes.
    node.outputs.one_symbol_schemata = schemata
    node.outputs.one_symbol_is_prime_implicant = True

    t0 = time.perf_counter()
    try:
        # Force full pi_coverage (node default exact_k_max=14 would sample for
        # k > 14 — the exact-vs-sample cohort goes up to MAX_K_EXACT).
        exact_kr, kr_mode = node.input_redundancy(
            norm=True, exact_k_max=k, return_mode=True
        )
        exact_ri, ri_mode = node.edge_redundancy(
            bound="mean", exact_k_max=k, return_mode=True
        )
        exact_kr = float(exact_kr)
        exact_ri = [float(x) for x in exact_ri]
    except Exception as exc:  # noqa: BLE001 — coverage may OOM/refuse
        log.info(
            f"#{i:4d} {cohort:7s} k={k:3d}  {label}  exact_FAIL  "
            f"{type(exc).__name__}: {exc}"
        )
        continue
    t_exact = time.perf_counter() - t0

    if kr_mode != "exact" or ri_mode != "exact":
        log.info(
            f"#{i:4d} {cohort:7s} k={k:3d}  {label}  exact_FAIL  "
            f"unexpected mode kr={kr_mode!r} ri={ri_mode!r} "
            f"(expected 'exact' with exact_k_max=k)"
        )
        continue

    log.info(
        f"#{i:4d} {cohort:7s} k={k:3d}  {label}  primes={primes['method']} "
        f"({primes['n_on']}+{primes['n_off']})  "
        f"exact_kr={exact_kr:.6f}  mode={kr_mode}  t_exact={fmt_s(t_exact)}"
    )

    space = 1 << k
    # Only n >= MIN_N_SAMPLES; include full space as the exact-sample check.
    n_list = sorted(
        {int(n) for n in N_SAMPLES_GRID if MIN_N_SAMPLES <= int(n) < space}
        | ({space} if space >= MIN_N_SAMPLES else set())
    )
    if not n_list:
        log.info(f"       no n_samples >= {MIN_N_SAMPLES} for k={k}; skip")
        continue

    for n_eff in n_list:
        # One shared SampledPiCoverage for point estimate + bootstrap CI.
        rng = np.random.default_rng(SEED + n_eff + k)
        t0 = time.perf_counter()
        sampled = sample_pi_coverage(schemata, n_eff, rng=rng, k=k)
        est_kr = float(
            input_redundancy_from_coverage(
                sampled.coverage, k, norm=True
            )
        )
        est_ri = [
            float(x)
            for x in edge_redundancy_from_coverage(
                sampled.coverage, k, bound="mean"
            )
        ]
        t_samp = time.perf_counter() - t0
        # Full-space n (== 2**k) is exact; otherwise sampled.
        samp_mode = "exact" if n_eff >= space else "sampled"

        rng_boot = np.random.default_rng(SEED + n_eff + k + 17)
        t0 = time.perf_counter()
        boot = bootstrap_input_redundancy(
            sampled,
            n_boot=min(N_BOOT, 400) if MODE == "starter" else N_BOOT,
            ci=CI,
            norm=True,
            rng=rng_boot,
        )
        t_boot = time.perf_counter() - t0

        if abs(boot.estimate - est_kr) > 1e-9:
            log.info(
                f"       WARN boot.estimate={boot.estimate:.6f} != "
                f"est_kr={est_kr:.6f}"
            )

        d_kr = abs(est_kr - exact_kr)
        d_ri = float(np.mean([abs(a - b) for a, b in zip(est_ri, exact_ri)]))
        covers = boot.ci_low <= exact_kr <= boot.ci_high
        row = {
            "label": label,
            "cohort": cohort,
            "k": k,
            "n_samples": n_eff,
            "exact_kr": exact_kr,
            "est_kr": est_kr,
            "d_kr": d_kr,
            "d_ri": d_ri,
            "t_exact": t_exact,
            "t_sample": t_samp,
            "t_boot": t_boot,
            "boot_lo": boot.ci_low,
            "boot_hi": boot.ci_high,
            "covers": covers,
            "primes_t": primes["t"],
            "method": primes["method"],
        }
        accuracy_rows.append(row)
        log.info(
            f"       n={n_eff:5d}  mode={samp_mode}  "
            f"est_kr={est_kr:.6f}  |Δkr|={d_kr:.4g}  "
            f"|Δri|_mean={d_ri:.4g}  "
            f"CI=[{boot.ci_low:.6f},{boot.ci_high:.6f}] cover={covers}  "
            f"t_samp={fmt_s(t_samp)}  t_boot={fmt_s(t_boot)}"
        )


# %% [markdown]
# ## High-*k* bootstrap (exact N/A)


# %%
high_funcs = cohort_high()
high_rows: List[Dict[str, Any]] = []

log.info("")
log.info(
    f"=== high-k bootstrap  n_funcs={len(high_funcs)}  "
    f"N_SAMPLES_GRID={N_SAMPLES_GRID}  N_BOOT={N_BOOT} ==="
)

for i, fn in enumerate(high_funcs, start=1):
    model_id = str(fn["model_id"])
    target = str(fn["target"])
    cohort = str(fn["cohort"])
    expect_k = int(fn["k"])
    label = f"[id-{int(model_id):03d}]::{target}"

    try:
        model_name, _t, expr, bnet = load_bbm_rule(model_id, target)
    except Exception as exc:  # noqa: BLE001
        log.info(f"#{i:4d} LOAD_FAIL {label}: {exc}")
        continue

    names = unique_regulators(expr)
    k = len(names)
    primes = get_primes(expr, timeout_s=TIMEOUT_S)
    if primes["status"] != "ok":
        log.info(
            f"#{i:4d} {cohort:7s} k={k:3d}  {label}  primes={primes['status']}  "
            f"{primes.get('error')}"
        )
        continue

    schemata = primes_to_one_symbol(
        primes["on"], primes["off"], input_names=names
    )
    log.info(
        f"#{i:4d} {cohort:7s} k={k:3d}  {label}  primes={primes['method']} "
        f"({primes['n_on']}+{primes['n_off']})  t_primes={fmt_s(primes['t'])}"
    )

    for n_samp in N_SAMPLES_GRID:
        n_eff = int(n_samp)
        if n_eff >= (1 << k):
            # Would be full enum; skip for high-k narrative
            continue
        rng = np.random.default_rng(SEED + n_eff + k)
        sampled = sample_pi_coverage(schemata, n_eff, rng=rng, k=k)
        n_boot = N_BOOT if MODE != "starter" else min(N_BOOT, 400)

        t0 = time.perf_counter()
        boot = bootstrap_input_redundancy(
            sampled,
            n_boot=n_boot,
            ci=CI,
            norm=True,
            rng=rng,
        )
        t_boot = time.perf_counter() - t0

        rng_edge = np.random.default_rng(SEED + n_eff + k + 1)
        t0 = time.perf_counter()
        edge_boots = bootstrap_edge_redundancy(
            sampled,
            n_boot=n_boot,
            ci=CI,
            bound="mean",
            rng=rng_edge,
        )
        t_edge = time.perf_counter() - t0

        widths = [b.ci_high - b.ci_low for b in edge_boots]
        ri_ests = [b.estimate for b in edge_boots]
        row = {
            "label": label,
            "cohort": cohort,
            "k": k,
            "n_samples": n_eff,
            "est_kr": boot.estimate,
            "boot_lo": boot.ci_low,
            "boot_hi": boot.ci_high,
            "ci_width": boot.ci_high - boot.ci_low,
            "t_boot": t_boot,
            "t_edge_boot": t_edge,
            "edge_width_mean": float(np.mean(widths)),
            "ri_min": float(min(ri_ests)),
            "ri_max": float(max(ri_ests)),
            "samples_per_s": n_eff / t_boot if t_boot > 0 else float("inf"),
            "primes_t": primes["t"],
            "method": primes["method"],
            "n_on": primes["n_on"],
            "n_off": primes["n_off"],
        }
        high_rows.append(row)
        log.info(
            f"       n={n_eff:5d}  kr={boot.estimate:.6f}  "
            f"CI=[{boot.ci_low:.6f},{boot.ci_high:.6f}]  "
            f"width={boot.ci_high - boot.ci_low:.4g}  "
            f"edge_w_mean={float(np.mean(widths)):.4g}  "
            f"ri=[{min(ri_ests):.3f},{max(ri_ests):.3f}]  "
            f"t={fmt_s(t_boot)}  samp/s={n_eff / t_boot:,.0f}"
        )


# %% [markdown]
# ## Plots


# %%
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- error vs n_samples (enumerable k) ---
if accuracy_rows:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for r in accuracy_rows:
        by_label.setdefault(r["label"], []).append(r)
    for label, rows in by_label.items():
        rows = sorted(rows, key=lambda r: r["n_samples"])
        short = label if len(label) <= 28 else label[:25] + "..."
        ax.plot(
            [r["n_samples"] for r in rows],
            [r["d_kr"] for r in rows],
            marker="o",
            label=f"{short} (k={rows[0]['k']})",
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("n_samples")
    ax.set_ylabel("|Δ input_redundancy| (norm)")
    ax.set_title("Sampling error vs n (enumerable k)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path = FIG_DIR / "error_vs_n_samples.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    log.info(f"wrote {path}")

    # time: exact vs sample at one fixed n (most accurate grid size)
    time_rows = [
        r for r in accuracy_rows if r["n_samples"] == N_TIME_COMPARE
    ]
    if time_rows:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        # one point per function at N_TIME_COMPARE
        for r in sorted(time_rows, key=lambda x: (x["k"], x["label"])):
            ax.scatter(
                [r["k"]], [r["t_exact"]], marker="s", color="C0", s=60, zorder=3
            )
            ax.scatter(
                [r["k"]], [r["t_sample"]], marker="o", color="C1", s=60, zorder=3
            )
        ax.scatter([], [], marker="s", color="C0", label="exact (pi_coverage)")
        ax.scatter(
            [],
            [],
            marker="o",
            color="C1",
            label=f"sample (n={N_TIME_COMPARE})",
        )
        ax.set_xlabel("k")
        ax.set_ylabel("time (s)")
        ax.set_yscale("log")
        ax.set_title(
            f"Exact coverage vs sampling time (n={N_TIME_COMPARE})"
        )
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        path = FIG_DIR / "time_exact_vs_sample.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        log.info(f"wrote {path}")
    else:
        log.info(
            f"skip time_exact_vs_sample.png: no rows with "
            f"n_samples==N_TIME_COMPARE={N_TIME_COMPARE} "
            f"(need k > {int(np.log2(N_TIME_COMPARE))})"
        )

# --- high-k CI width vs n ---
if high_rows:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    by_label = {}
    for r in high_rows:
        by_label.setdefault(r["label"], []).append(r)
    for label, rows in by_label.items():
        rows = sorted(rows, key=lambda r: r["n_samples"])
        short = label if len(label) <= 28 else label[:25] + "..."
        ax.plot(
            [r["n_samples"] for r in rows],
            [r["ci_width"] for r in rows],
            marker="o",
            label=f"{short} (k={rows[0]['k']})",
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("n_samples")
    ax.set_ylabel("95% CI width (norm k_r)")
    ax.set_title("Bootstrap CI width vs n (high k)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path = FIG_DIR / "bootstrap_ci_width_vs_n.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    log.info(f"wrote {path}")

    # forest plot at largest shared n
    n_plot = max(r["n_samples"] for r in high_rows)
    forest = [r for r in high_rows if r["n_samples"] == n_plot]
    forest = sorted(forest, key=lambda r: r["k"])
    if forest:
        fig, ax = plt.subplots(figsize=(8.0, max(3.0, 0.45 * len(forest) + 1)))
        y = np.arange(len(forest))
        est = [r["est_kr"] for r in forest]
        lo = [r["boot_lo"] for r in forest]
        hi = [r["boot_hi"] for r in forest]
        ax.errorbar(
            est,
            y,
            xerr=[
                [e - l for e, l in zip(est, lo)],
                [h - e for e, h in zip(est, hi)],
            ],
            fmt="o",
            capsize=3,
            color="C0",
        )
        labels = []
        for r in forest:
            short = r["label"] if len(r["label"]) <= 36 else r["label"][:33] + "..."
            labels.append(f"{short} (k={r['k']})")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(f"input_redundancy (norm) ± {CI:.0%} CI")
        ax.set_title(f"High-k sampled k_r (n={n_plot})")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        path = FIG_DIR / "high_k_kr_forest.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        log.info(f"wrote {path}")


# %% [markdown]
# ## Done


# %%
log.info("")
log.info(
    f"done: accuracy_rows={len(accuracy_rows)}  high_rows={len(high_rows)}"
)
log.info(f"log_path={LOG_PATH}")
log.info(f"fig_dir={FIG_DIR}")
log.close()
