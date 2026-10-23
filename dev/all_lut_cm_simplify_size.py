# %% [markdown]
# All-LUT CM simplify size comparison
#
# Sibling of `all_lut_dcm_simplify_roundtrip.py`.  Enumerate Boolean LUTs for
# every `k <= MAX_K`, build canalizing maps for each builder in `BUILDERS`, and
# measure node/edge counts for three modes:
#
# * **raw** — unsimplified CM
# * **nei** — `simplify_canalizing_map(..., skip_if_edges_increase=True)`
# * **full** — unrestricted fixed-point simplify (`skip_if_edges_increase=False`)
#
# Six combinations: legacy/canonical × raw/nei/full.  Same sampled LUT ids are
# reused for every combo.
#
# * **Exhaustive** when `2^(2^k) <= SAMPLE_N` (default 2048 → full enum for
#   `k <= 3`).
# * **Sample** exactly `SAMPLE_N` uniform random LUTs when the space is larger.
#
# Set `SMOKE = True` to cap at `SMOKE_MAX_K`.  Each run writes a directory under
# `dev/logs/` containing the log, `sizes.csv`, numbered figures, and a summary
# `README.md`.

# %%
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

_DEV_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _DEV_DIR.parent  # .../CANA/CANA (contains the cana package)
for _p in (_DEV_DIR, _PKG_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cana.boolean_node import BooleanNode
from cana.canalization.canalizing_map import simplify_canalizing_map

from bbm_cm_roundtrip import TeeLogger, _fmt_secs  # noqa: E402

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
# CM builders to size-compare (same sampled LUT ids for each).
BUILDERS = ("legacy", "canonical")
# Size modes: raw / edge-capped simplify / full simplify.
MODES = ("raw", "nei", "full")
# How many functions to show in the same-function bar sample plot.
SAMPLE_FUNCS_N = 8
SAMPLE_FUNCS_SEED = 1
# Progress line every N evaluated/errored functions within a k-bucket.
PROGRESS_EVERY = 256

LOG_DIR = _DEV_DIR / "logs"

_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
_RUN_STEM = (
    f"cm_simplify_size_smoke_{_TIMESTAMP}"
    if SMOKE
    else f"cm_simplify_size_{_TIMESTAMP}"
)
RUN_DIR = LOG_DIR / _RUN_STEM
LOG_PATH = RUN_DIR / f"{_RUN_STEM}.log"
CSV_PATH = RUN_DIR / "sizes.csv"
# Figures write directly into RUN_DIR (no nested _figures/).
FIG_DIR = RUN_DIR

# %% [markdown]
# ## Helpers — LUT enumeration + CM size


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
    return BooleanNode.from_output_list(
        outputs, name=lut_name(k, func_id), inputs=inputs, id=func_id
    )


def cm_size(g: nx.DiGraph) -> Tuple[int, int]:
    """Return ``(n_nodes, n_edges)`` for a canalizing-map graph."""
    return int(g.number_of_nodes()), int(g.number_of_edges())


def apply_simplify(
    cm: nx.DiGraph, *, skip_if_edges_increase: bool
) -> nx.DiGraph:
    """Stock fixed-point simplify (copy + drop isolates)."""
    return simplify_canalizing_map(
        cm,
        copy=True,
        remove_isolates=True,
        skip_if_edges_increase=skip_if_edges_increase,
    )


def sizes_for_builder(node: BooleanNode, builder: str) -> Dict[str, Tuple[int, int]]:
    """Build raw CM and both simplify modes; return mode → (nodes, edges)."""
    raw = node.canalizing_map(builder=builder)
    nei = apply_simplify(raw, skip_if_edges_increase=True)
    full = apply_simplify(raw, skip_if_edges_increase=False)
    return {
        "raw": cm_size(raw),
        "nei": cm_size(nei),
        "full": cm_size(full),
    }


def effective_max_k(smoke: bool, max_k: int, smoke_max_k: int) -> int:
    return min(max_k, smoke_max_k) if smoke else max_k


def _combo_label(builder: str, mode: str) -> str:
    return f"{builder}/{mode}"


# %% [markdown]
# ## Run enumeration (smoke or full)
#
# Call `main()` from a notebook cell, or run the file as a script.
# Returns the run directory (`RUN_DIR`).  Primary artifact is `sizes.csv`.


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
    log.info(f"MODES={','.join(MODES)}")
    log.info(f"PROGRESS_EVERY={PROGRESS_EVERY}")
    log.info(f"run_dir={RUN_DIR}")
    log.info(f"log_path={LOG_PATH}")
    log.info(f"csv_path={CSV_PATH}")
    log.info(f"timestamp={_TIMESTAMP}")
    log.info("")

    rows: List[Dict[str, object]] = []
    n_eval = 0
    n_error = 0
    # Cache sampled ids per k so every builder sees the same LUTs.
    func_ids_by_k: Dict[int, Tuple[str, List[int]]] = {}

    for k in range(1, k_hi + 1):
        if k not in func_ids_by_k:
            func_ids_by_k[k] = iter_func_ids(
                k, sample_n=SAMPLE_N, seed=SAMPLE_SEED
            )
        enum_mode, func_ids = func_ids_by_k[k]
        space = n_boolean_functions(k)
        log.info(
            f"=== k={k}  mode={enum_mode}  space={space}  "
            f"planned={len(func_ids)} ==="
        )

        t_k0 = time.perf_counter()
        k_eval = 0
        k_error = 0
        for i, func_id in enumerate(func_ids):
            name = lut_name(k, func_id)
            try:
                node = make_node(k, func_id)
                t0 = time.perf_counter()
                for builder in BUILDERS:
                    sizes = sizes_for_builder(node, builder)
                    for mode_key in MODES:
                        n_nodes, n_edges = sizes[mode_key]
                        rows.append(
                            {
                                "k": k,
                                "func_id": func_id,
                                "name": name,
                                "enum_mode": enum_mode,
                                "builder": builder,
                                "mode": mode_key,
                                "nodes": n_nodes,
                                "edges": n_edges,
                            }
                        )
                elapsed_fn = time.perf_counter() - t0
                k_eval += 1
                n_eval += 1
                if (
                    i < 3
                    or (i + 1) % PROGRESS_EVERY == 0
                    or i + 1 == len(func_ids)
                ):
                    log.info(
                        f"  OK    {name}  k={k}  "
                        f"fn={_fmt_secs(elapsed_fn)}"
                    )
            except Exception as exc:  # noqa: BLE001
                k_error += 1
                n_error += 1
                log.info(
                    f"  ERROR {name}  k={k}  "
                    f"{type(exc).__name__}: {exc}"
                )

            if PROGRESS_EVERY and (i + 1) % PROGRESS_EVERY == 0:
                elapsed = time.perf_counter() - t_k0
                log.info(
                    f"  progress  {i + 1}/{len(func_ids)}  "
                    f"eval={k_eval}  error={k_error}  "
                    f"elapsed={_fmt_secs(elapsed)}"
                )

        log.info(
            f"  k_summary  eval={k_eval}  error={k_error}  "
            f"elapsed={_fmt_secs(time.perf_counter() - t_k0)}"
        )
        log.info("")

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)

    log.info("=" * 60)
    log.info("SUMMARY")
    log.info(f"mode={mode}")
    log.info(f"BUILDERS={','.join(BUILDERS)}")
    log.info(f"MODES={','.join(MODES)}")
    log.info(f"n_eval={n_eval}")
    log.info(f"n_error={n_error}")
    log.info(f"n_rows={len(df)}")
    log.info(f"csv_path={CSV_PATH}")
    if not df.empty:
        agg = (
            df.groupby(["builder", "mode"], sort=True)[["nodes", "edges"]]
            .agg(["mean", "median"])
            .round(2)
        )
        log.info("mean/median nodes,edges by (builder, mode):")
        for (builder, mode_key), _ in agg.iterrows():
            sub = df[(df["builder"] == builder) & (df["mode"] == mode_key)]
            log.info(
                f"  {_combo_label(builder, mode_key)}  "
                f"nodes mean={sub['nodes'].mean():.2f} "
                f"median={sub['nodes'].median():.1f}  "
                f"edges mean={sub['edges'].mean():.2f} "
                f"median={sub['edges'].median():.1f}"
            )
    log.info(f"run_dir={RUN_DIR}")
    log.info(f"log written to {LOG_PATH}")
    log.info("=" * 60)

    log.close()
    return RUN_DIR


if __name__ == "__main__":
    RUN_DIR = main()
    LOG_PATH = RUN_DIR / f"{RUN_DIR.name}.log"
    CSV_PATH = RUN_DIR / "sizes.csv"
    FIG_DIR = RUN_DIR

# %% [markdown]
# ## Run
#
# Uncomment / execute to generate a fresh run directory (also happens under
# `__main__`).  Summary cells below point at `RUN_DIR` / `CSV_PATH`.

# %%
# RUN_DIR = main()
# LOG_PATH = RUN_DIR / f"{RUN_DIR.name}.log"
# CSV_PATH = RUN_DIR / "sizes.csv"
# FIG_DIR = RUN_DIR

# %% [markdown]
# ## Load sizes.csv + headline stats


# %%
def _latest_run_dir() -> Optional[Path]:
    if not LOG_DIR.exists():
        return None
    cands = sorted(
        (
            p
            for p in LOG_DIR.glob("cm_simplify_size*")
            if p.is_dir() and (p / "sizes.csv").exists()
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return cands[0] if cands else None


def load_sizes(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ("k", "func_id", "nodes", "edges"):
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df


if not CSV_PATH.exists():
    latest = _latest_run_dir()
    if latest is not None:
        RUN_DIR = latest
        LOG_PATH = RUN_DIR / f"{RUN_DIR.name}.log"
        CSV_PATH = RUN_DIR / "sizes.csv"
        FIG_DIR = RUN_DIR
        print(f"using latest run: {RUN_DIR}")

if CSV_PATH.exists():
    df_sizes = load_sizes(CSV_PATH)
    print(f"loaded {len(df_sizes)} rows from {CSV_PATH}")
    print()
    print("=== Per-(builder, mode) mean/median ===")
    print(
        df_sizes.groupby(["builder", "mode"], sort=True)[["nodes", "edges"]]
        .agg(["count", "mean", "median", "min", "max"])
        .round(2)
        .to_string()
    )
    print()
    print("=== Per-(k, builder, mode) mean nodes/edges ===")
    print(
        df_sizes.groupby(["k", "builder", "mode"], sort=True)[["nodes", "edges"]]
        .mean()
        .round(2)
        .to_string()
    )
else:
    df_sizes = pd.DataFrame()
    print(f"no sizes.csv at {CSV_PATH}; run main() first")

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


# Stable colors for the six (builder, mode) combos.
_COMBO_COLORS = {
    ("legacy", "raw"): "#264653",
    ("legacy", "nei"): "#2a9d8f",
    ("legacy", "full"): "#e76f51",
    ("canonical", "raw"): "#457b9d",
    ("canonical", "nei"): "#a8dadc",
    ("canonical", "full"): "#e9c46a",
}

_MODE_ORDER = ["raw", "nei", "full"]
_BUILDER_ORDER = ["legacy", "canonical"]


def _combos() -> List[Tuple[str, str]]:
    return [(b, m) for b in _BUILDER_ORDER for m in _MODE_ORDER]


def write_readme(
    run_dir: Path,
    df: pd.DataFrame,
    fig_names: List[str],
) -> Path:
    """Write summary README.md into the run directory."""
    stem = run_dir.name
    lines: List[str] = [f"# {stem}", ""]
    lines.append(
        "All-LUT CM size comparison: `legacy` / `canonical` × "
        "`raw` / `nei` (no edge increase) / `full` simplify."
    )
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append("| Setting | Value |")
    lines.append("|---------|-------|")
    lines.append(f"| timestamp | `{_TIMESTAMP}` |")
    lines.append(f"| mode | `{'smoke' if SMOKE else 'full'}` |")
    lines.append(f"| BUILDERS | `{','.join(BUILDERS)}` |")
    lines.append(f"| MODES | `{','.join(MODES)}` |")
    lines.append(f"| MAX_K | `{MAX_K}` |")
    lines.append(f"| SAMPLE_N / SAMPLE_SEED | `{SAMPLE_N}` / `{SAMPLE_SEED}` |")
    lines.append(
        f"| effective_max_k | "
        f"`{effective_max_k(SMOKE, MAX_K, SMOKE_MAX_K)}` |"
    )
    lines.append("")

    lines.append("## Results")
    lines.append("")
    if df.empty:
        lines.append("(no size rows)")
        lines.append("")
    else:
        lines.append(
            f"- **rows:** {len(df)} "
            f"({df['func_id'].nunique()} functions × "
            f"{df['builder'].nunique()} builders × "
            f"{df['mode'].nunique()} modes, aggregated across k)"
        )
        lines.append("")
        lines.append("Mean nodes / edges by (builder, mode):")
        lines.append("")
        lines.append("| builder | mode | mean nodes | mean edges | n |")
        lines.append("|---------|------|------------|------------|---|")
        for builder, mode_key in _combos():
            sub = df[(df["builder"] == builder) & (df["mode"] == mode_key)]
            if sub.empty:
                continue
            lines.append(
                f"| {builder} | {mode_key} | "
                f"{sub['nodes'].mean():.2f} | "
                f"{sub['edges'].mean():.2f} | "
                f"{len(sub)} |"
            )
        lines.append("")

        lines.append("Mean nodes by (k, builder, mode):")
        lines.append("")
        ks = sorted(df["k"].unique().tolist())
        header = ["k"] + [_combo_label(b, m) for b, m in _combos()]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["------"] * len(header)) + "|")
        for k in ks:
            cells = [str(k)]
            for builder, mode_key in _combos():
                sub = df[
                    (df["k"] == k)
                    & (df["builder"] == builder)
                    & (df["mode"] == mode_key)
                ]
                if sub.empty:
                    cells.append("—")
                else:
                    cells.append(f"{sub['nodes'].mean():.1f}")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `{stem}.log`")
    lines.append("- `sizes.csv`")
    for name in fig_names:
        lines.append(f"- `{name}.png`")
    lines.append("- `README.md`")
    lines.append("")

    path = run_dir / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"README → {path}")
    return path


def _pivot_wide(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pivot long sizes to one row per (k, func_id) with builder_mode columns."""
    wide = df.pivot_table(
        index=["k", "func_id"],
        columns=["builder", "mode"],
        values=metric,
        aggfunc="first",
    )
    wide.columns = [f"{b}_{m}" for b, m in wide.columns]
    return wide.reset_index()


fig_names_written: List[str] = []

if not df_sizes.empty:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"figures → {FIG_DIR}")

    plt.rcParams.update(
        {
            "figure.figsize": (10, 5),
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 10,
        }
    )

    combos = _combos()
    ks = sorted(df_sizes["k"].unique().tolist())

    # ------------------------------------------------------------------
    # 1) Per-k box distributions of nodes and edges
    # ------------------------------------------------------------------
    for metric in ("nodes", "edges"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
        for ax, builder in zip(axes, _BUILDER_ORDER):
            data = []
            positions = []
            colors = []
            labels_pos = []
            labels_lab = []
            width = 0.22
            for mi, mode_key in enumerate(_MODE_ORDER):
                for ki, k in enumerate(ks):
                    sub = df_sizes[
                        (df_sizes["k"] == k)
                        & (df_sizes["builder"] == builder)
                        & (df_sizes["mode"] == mode_key)
                    ][metric]
                    pos = ki + (mi - 1) * width
                    data.append(sub.values)
                    positions.append(pos)
                    colors.append(_COMBO_COLORS[(builder, mode_key)])
                    if mi == 1:
                        labels_pos.append(ki)
                        labels_lab.append(str(k))
            bp = ax.boxplot(
                data,
                positions=positions,
                widths=width * 0.9,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
            )
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
            ax.set_xticks(labels_pos)
            ax.set_xticklabels(labels_lab)
            ax.set_xlabel("k")
            ax.set_ylabel(metric)
            ax.set_title(f"{builder}: {metric} by mode")
            # Legend proxies
            handles = [
                Patch(
                    facecolor=_COMBO_COLORS[(builder, m)],
                    alpha=0.75,
                    label=m,
                )
                for m in _MODE_ORDER
            ]
            ax.legend(handles=handles, title="mode", fontsize=8)
        fig.suptitle(f"Per-k distribution of CM {metric} (box, no outliers)")
        fig_names_written.append(f"01_{metric}_boxplot_by_k")
        save_fig(fig_names_written[-1])

    # ------------------------------------------------------------------
    # 2) Mean nodes / edges vs k
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric in zip(axes, ("nodes", "edges")):
        for builder, mode_key in combos:
            sub = (
                df_sizes[
                    (df_sizes["builder"] == builder)
                    & (df_sizes["mode"] == mode_key)
                ]
                .groupby("k")[metric]
                .mean()
                .reindex(ks)
            )
            ax.plot(
                ks,
                sub.values,
                marker="o",
                color=_COMBO_COLORS[(builder, mode_key)],
                label=_combo_label(builder, mode_key),
            )
        ax.set_xlabel("k")
        ax.set_ylabel(f"mean {metric}")
        ax.set_title(f"Mean CM {metric} vs k")
        ax.legend(fontsize=7, ncol=2)
    fig_names_written.append("02_mean_nodes_edges_vs_k")
    save_fig(fig_names_written[-1])

    # ------------------------------------------------------------------
    # 3) Same-function scatter: legacy vs canonical (nodes + edges)
    # ------------------------------------------------------------------
    for metric in ("nodes", "edges"):
        wide = _pivot_wide(df_sizes, metric)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=False, sharey=False)
        for ax, mode_key in zip(axes, _MODE_ORDER):
            leg_col = f"legacy_{mode_key}"
            can_col = f"canonical_{mode_key}"
            if leg_col not in wide.columns or can_col not in wide.columns:
                ax.set_title(f"{mode_key} (missing)")
                continue
            for ki, k in enumerate(ks):
                sub = wide[wide["k"] == k]
                ax.scatter(
                    sub[leg_col],
                    sub[can_col],
                    s=12,
                    alpha=0.45,
                    label=f"k={k}",
                    color=plt.cm.viridis(ki / max(len(ks) - 1, 1)),
                )
            lo = float(
                min(wide[leg_col].min(), wide[can_col].min())
            )
            hi = float(
                max(wide[leg_col].max(), wide[can_col].max())
            )
            pad = max((hi - lo) * 0.05, 0.5)
            ax.plot(
                [lo - pad, hi + pad],
                [lo - pad, hi + pad],
                "k--",
                lw=0.8,
                alpha=0.5,
            )
            ax.set_xlabel(f"legacy {metric}")
            ax.set_ylabel(f"canonical {metric}")
            ax.set_title(f"mode={mode_key}")
            ax.legend(fontsize=7)
        fig.suptitle(
            f"Same function: legacy vs canonical {metric} "
            "(dashed = equal size)"
        )
        fig_names_written.append(f"03_scatter_legacy_vs_canonical_{metric}")
        save_fig(fig_names_written[-1])

    # ------------------------------------------------------------------
    # 4) Same-function deltas: full − nei (nodes + edges)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for bi, builder in enumerate(_BUILDER_ORDER):
        for mi, metric in enumerate(("nodes", "edges")):
            ax = axes[mi, bi]
            sub_b = df_sizes[df_sizes["builder"] == builder]
            piv = sub_b.pivot_table(
                index=["k", "func_id"],
                columns="mode",
                values=metric,
                aggfunc="first",
            )
            if "full" not in piv.columns or "nei" not in piv.columns:
                ax.set_title(f"{builder} {metric} (missing)")
                continue
            delta = (piv["full"] - piv["nei"]).dropna()
            # Attach k for coloring
            delta_df = delta.reset_index(name="delta")
            for ki, k in enumerate(ks):
                vals = delta_df.loc[delta_df["k"] == k, "delta"]
                if vals.empty:
                    continue
                ax.hist(
                    vals,
                    bins=20,
                    alpha=0.55,
                    label=f"k={k}",
                    color=plt.cm.plasma(ki / max(len(ks) - 1, 1)),
                )
            ax.axvline(0, color="k", ls="--", lw=0.8)
            ax.set_xlabel(f"{metric}_full − {metric}_nei")
            ax.set_ylabel("count")
            ax.set_title(f"{builder}: Δ{metric} (full − nei)")
            ax.legend(fontsize=7)
    fig.suptitle(
        "How much unrestricted simplify shrinks beyond edge-capped (nei)"
    )
    fig_names_written.append("04_delta_full_minus_nei")
    save_fig(fig_names_written[-1])

    # ------------------------------------------------------------------
    # 5) Small-multiples sample: same function across 6 combos
    # ------------------------------------------------------------------
    # Prefer extremes by legacy raw edge count, plus a seeded random fill.
    raw_leg = df_sizes[
        (df_sizes["builder"] == "legacy") & (df_sizes["mode"] == "raw")
    ][["k", "func_id", "edges", "name"]].copy()
    sample_keys: List[Tuple[int, int]] = []
    if not raw_leg.empty:
        # One extreme min/max per k when available, then fill randomly.
        for k in ks:
            sub = raw_leg[raw_leg["k"] == k]
            if sub.empty:
                continue
            imin = int(sub["edges"].idxmin())
            imax = int(sub["edges"].idxmax())
            sample_keys.append(
                (int(sub.loc[imin, "k"]), int(sub.loc[imin, "func_id"]))
            )
            if imax != imin:
                sample_keys.append(
                    (int(sub.loc[imax, "k"]), int(sub.loc[imax, "func_id"]))
                )
        # Dedup while preserving order
        seen = set()
        uniq: List[Tuple[int, int]] = []
        for key in sample_keys:
            if key not in seen:
                seen.add(key)
                uniq.append(key)
        sample_keys = uniq
        if len(sample_keys) < SAMPLE_FUNCS_N:
            rng = np.random.default_rng(SAMPLE_FUNCS_SEED)
            pool = list(
                zip(raw_leg["k"].astype(int), raw_leg["func_id"].astype(int))
            )
            pool = [p for p in pool if p not in seen]
            n_extra = min(SAMPLE_FUNCS_N - len(sample_keys), len(pool))
            if n_extra > 0:
                picks = rng.choice(len(pool), size=n_extra, replace=False)
                for idx in np.atleast_1d(picks):
                    sample_keys.append(pool[int(idx)])
        sample_keys = sample_keys[:SAMPLE_FUNCS_N]

    n_samp = max(len(sample_keys), 1)
    fig, axes = plt.subplots(
        n_samp,
        2,
        figsize=(11, max(2.2 * n_samp, 4)),
        squeeze=False,
    )
    x = np.arange(len(combos))
    combo_labs = [_combo_label(b, m) for b, m in combos]
    colors = [_COMBO_COLORS[c] for c in combos]
    for ri, (k, fid) in enumerate(sample_keys):
        sub = df_sizes[(df_sizes["k"] == k) & (df_sizes["func_id"] == fid)]
        name = (
            str(sub["name"].iloc[0])
            if not sub.empty and "name" in sub.columns
            else f"k{k}_id{fid}"
        )
        # Shorten name for title
        short = name if len(name) <= 42 else name[:39] + "..."
        node_vals = []
        edge_vals = []
        for builder, mode_key in combos:
            row = sub[
                (sub["builder"] == builder) & (sub["mode"] == mode_key)
            ]
            node_vals.append(
                int(row["nodes"].iloc[0]) if not row.empty else 0
            )
            edge_vals.append(
                int(row["edges"].iloc[0]) if not row.empty else 0
            )
        axes[ri, 0].bar(x, node_vals, color=colors)
        axes[ri, 0].set_ylabel("nodes")
        axes[ri, 0].set_title(short, fontsize=8, loc="left")
        axes[ri, 0].set_xticks(x)
        axes[ri, 0].set_xticklabels(
            combo_labs if ri == n_samp - 1 else [],
            rotation=35,
            ha="right",
            fontsize=7,
        )
        axes[ri, 1].bar(x, edge_vals, color=colors)
        axes[ri, 1].set_ylabel("edges")
        axes[ri, 1].set_title(f"k={k} id={fid}", fontsize=8, loc="left")
        axes[ri, 1].set_xticks(x)
        axes[ri, 1].set_xticklabels(
            combo_labs if ri == n_samp - 1 else [],
            rotation=35,
            ha="right",
            fontsize=7,
        )
    if not sample_keys:
        axes[0, 0].set_title("(no sample functions)")
    fig.suptitle(
        "Same function across builders × modes "
        "(extremes by legacy-raw edges + random fill)"
    )
    fig_names_written.append("05_sample_functions_six_combos")
    save_fig(fig_names_written[-1])

    write_readme(RUN_DIR, df_sizes, fig_names_written)
    print(f"done — {len(fig_names_written)} figures in {FIG_DIR}")
else:
    print("skip figures (empty df_sizes)")
