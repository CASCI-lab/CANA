# %% [markdown]
# CM ↔ LUT roundtrip — log summary
#
# Parse a `bbm_cm_roundtrip.py` log and print simple stats + figures.
# Default log: `logs/cm_bbm_roundtrip_20261020_235736.log` (partial run OK).

# %%
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

_DEV_DIR = Path(__file__).resolve().parent
LOG_PATH = _DEV_DIR / "logs" / "cm_bbm_roundtrip_20261020_235736.log"
# Figures land next to the log: logs/<log_stem>_figures/
FIG_DIR = LOG_PATH.parent / f"{LOG_PATH.stem}_figures"

# %% [markdown]
# ## Parse log


# %%
_RE_HEADER = re.compile(r"^([A-Za-z0-9_]+)=(.*)$")
_RE_MODEL = re.compile(
    r"^=== model id=(?P<id>\S+) name=(?P<name>\S+)"
    r"(?:  nodes=(?P<nodes>\d+)  lut_built=(?P<lut_built>\d+)  "
    r"load_skip=(?P<load_skip>\d+)"
    r"| LOAD FAILED: (?P<load_error>.+)) ===$"
)
_RE_NODE_HEAD = re.compile(
    r"^  (?P<status>OK|FAIL|SKIP|ERROR)\s+(?P<name>\S+)\s+k=(?P<k>\d+)(?P<rest>.*)$"
)
_RE_TIMING_TAIL = re.compile(r"\s+cm=(?P<cm>\S+)\s+lut=(?P<lut>\S+)\s*$")
_RE_MODEL_SUMMARY = re.compile(
    r"^  model_summary  eval=(?P<eval>\d+)  fail=(?P<fail>\d+)  "
    r"skip=(?P<skip>\d+)  error=(?P<error>\d+)  model_fail=(?P<model_fail>\S+)  "
    r"cm_total=(?P<cm_total>\S+)  lut_total=(?P<lut_total>\S+)$"
)
_RE_FINAL = re.compile(
    r"^(function_fail_fraction|model_fail_fraction|cm_time_total|lut_time_total)"
    r"\s*=\s*(.+)$"
)


def parse_duration(token: Optional[str]) -> Optional[float]:
    """Parse ``12.3ms`` / ``1.5s`` / ``n/a`` → seconds."""
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
    if "MAX_K" in reason or "LUT not built" in reason:
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


def parse_log(path: Path) -> Tuple[Dict[str, str], pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    header: Dict[str, str] = {}
    nodes: List[Dict[str, Any]] = []
    models: List[Dict[str, Any]] = []
    final: Dict[str, str] = {}

    cur: Optional[Dict[str, Any]] = None
    in_header = True

    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")

            if line == "SUMMARY" or (line.startswith("=" * 10) and "SUMMARY" not in line):
                in_header = False
                continue

            m_model = _RE_MODEL.match(line)
            if in_header and m_model is None:
                hm = _RE_HEADER.match(line)
                if hm:
                    header[hm.group(1)] = hm.group(2)
                if line.startswith("selected_models="):
                    header["selected_models_line"] = line
                continue

            if m_model:
                in_header = False
                if cur is not None and "model_fail" not in cur:
                    # incomplete model block (run cut off mid-model)
                    models.append(cur)
                cur = {
                    "model_id": m_model.group("id"),
                    "name": m_model.group("name"),
                    "n_nodes": int(m_model.group("nodes")) if m_model.group("nodes") else 0,
                    "lut_built": int(m_model.group("lut_built")) if m_model.group("lut_built") else 0,
                    "load_skip": int(m_model.group("load_skip")) if m_model.group("load_skip") else 0,
                    "load_error": m_model.group("load_error"),
                }
                continue

            m = _RE_MODEL_SUMMARY.match(line)
            if m and cur is not None:
                cur.update(
                    {
                        "eval": int(m.group("eval")),
                        "fail": int(m.group("fail")),
                        "skip": int(m.group("skip")),
                        "error": int(m.group("error")),
                        "model_fail": m.group("model_fail") == "True",
                        "cm_total_s": parse_duration(m.group("cm_total")),
                        "lut_total_s": parse_duration(m.group("lut_total")),
                    }
                )
                models.append(cur)
                cur = None
                continue

            node = parse_node_line(line)
            if node is not None and cur is not None:
                nodes.append(
                    {
                        "model_id": cur["model_id"],
                        "model_name": cur["name"],
                        **node,
                    }
                )
                continue

            m = _RE_FINAL.match(line)
            if m:
                final[m.group(1)] = m.group(2).strip()

    if cur is not None:
        models.append(cur)

    df_nodes = pd.DataFrame(nodes)
    df_models = pd.DataFrame(models)
    return header, df_nodes, df_models, final


header, df_nodes, df_models, final = parse_log(LOG_PATH)
print(f"log: {LOG_PATH.name}")
print(f"header keys: {sorted(header.keys())}")
print(f"models parsed: {len(df_models)}  nodes: {len(df_nodes)}")
if final:
    print("final SUMMARY present:", final)
else:
    print("final SUMMARY: (absent — run incomplete or still going)")

# %% [markdown]
# ## Headline stats


# %%
n_ok = int((df_nodes["status"] == "OK").sum())
n_fail = int((df_nodes["status"] == "FAIL").sum())
n_skip = int((df_nodes["status"] == "SKIP").sum())
n_error = int((df_nodes["status"] == "ERROR").sum())
n_eval = n_ok + n_fail  # evaluated = not skip/error

models_done = df_models["model_fail"].notna() if "model_fail" in df_models else pd.Series(dtype=bool)
df_done = df_models.loc[models_done].copy() if len(df_models) else df_models
n_model = len(df_done)
n_model_fail = int(df_done["model_fail"].sum()) if n_model and "model_fail" in df_done else 0

print("=== Function outcomes ===")
print(f"  OK={n_ok}  FAIL={n_fail}  SKIP={n_skip}  ERROR={n_error}")
print(f"  evaluated={n_eval}  fail_fraction={n_fail}/{n_eval} = {n_fail / n_eval if n_eval else float('nan'):.6f}")
print()
print("=== Model outcomes (completed blocks only) ===")
print(f"  models_with_summary={n_model}  model_fail={n_model_fail}")
print(
    f"  model_fail_fraction={n_model_fail}/{n_model} = "
    f"{n_model_fail / n_model if n_model else float('nan'):.6f}"
)
print()

if "cm_s" in df_nodes:
    timed = df_nodes.dropna(subset=["cm_s"])
    print("=== Timing (per function, where recorded) ===")
    print(f"  cm  sum={timed['cm_s'].sum():.3f}s  median={timed['cm_s'].median()*1e3:.2f}ms  "
          f"p95={timed['cm_s'].quantile(0.95)*1e3:.2f}ms  max={timed['cm_s'].max():.3f}s")
    timed_lut = df_nodes.dropna(subset=["lut_s"])
    print(f"  lut sum={timed_lut['lut_s'].sum():.3f}s  median={timed_lut['lut_s'].median()*1e3:.2f}ms  "
          f"p95={timed_lut['lut_s'].quantile(0.95)*1e3:.2f}ms  max={timed_lut['lut_s'].max():.3f}s")

print()
print("=== FAIL reasons ===")
print(df_nodes.loc[df_nodes["status"] == "FAIL", "fail_reason"].value_counts(dropna=False).to_string())
print()
print("=== SKIP classes ===")
print(df_nodes.loc[df_nodes["status"] == "SKIP", "skip_class"].value_counts(dropna=False).to_string())

# %% [markdown]
# ## Figures
#
# PNGs are written under `FIG_DIR` (next to the log).


# %%
FIG_DIR.mkdir(parents=True, exist_ok=True)
print(f"figures → {FIG_DIR}")


def save_fig(name: str) -> Path:
    """Save current figure as PNG and still show it in the notebook."""
    path = FIG_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved {path.name}")
    plt.show()
    return path


plt.rcParams.update(
    {
        "figure.figsize": (7, 4),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# 1) Outcome counts
fig, ax = plt.subplots()
counts = pd.Series({"OK": n_ok, "FAIL": n_fail, "SKIP": n_skip, "ERROR": n_error})
colors = {"OK": "#2a9d8f", "FAIL": "#e76f51", "SKIP": "#e9c46a", "ERROR": "#264653"}
bars = ax.bar(counts.index, counts.values, color=[colors[k] for k in counts.index])
ax.set_ylabel("functions")
ax.set_title("Roundtrip outcomes (all parsed functions)")
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=9)
save_fig("01_outcomes")

# %%
# 2) Fail rate vs k (evaluated only)
eval_df = df_nodes[df_nodes["status"].isin(["OK", "FAIL"])].copy()
by_k = (
    eval_df.groupby("k")
    .agg(n=("status", "size"), n_fail=("status", lambda s: (s == "FAIL").sum()))
    .assign(fail_rate=lambda d: d["n_fail"] / d["n"])
    .reset_index()
)

fig, ax = plt.subplots()
ax.bar(by_k["k"], by_k["fail_rate"], color="#e76f51", alpha=0.85)
ax.set_xlabel("k (inputs)")
ax.set_ylabel("fail rate")
ax.set_title("FAIL rate among evaluated functions vs k")
ax.set_ylim(0, max(0.05, float(by_k["fail_rate"].max()) * 1.15) if len(by_k) else 1)
# annotate counts
for _, row in by_k.iterrows():
    if row["n_fail"] > 0:
        ax.text(row["k"], row["fail_rate"], f"{int(row['n_fail'])}/{int(row['n'])}",
                ha="center", va="bottom", fontsize=7, rotation=90)
save_fig("02_fail_rate_vs_k")

print(by_k.to_string(index=False))

# %%
# 3) Timing vs k (CM and LUT)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
for ax, col, title in (
    (axes[0], "cm_s", "CM build time"),
    (axes[1], "lut_s", "LUT reconstruct time"),
):
    sub = df_nodes.dropna(subset=[col])
    if sub.empty:
        ax.set_title(f"{title} (no data)")
        continue
    # boxplot per k
    ks = sorted(sub["k"].unique())
    data = [sub.loc[sub["k"] == k, col].values * 1e3 for k in ks]  # ms
    ax.boxplot(data, positions=ks, widths=0.6, showfliers=False)
    ax.set_xlabel("k")
    ax.set_ylabel("ms")
    ax.set_title(f"{title} vs k (no outliers)")
save_fig("03_timing_vs_k")

# %%
# 4) Skip classes + failing models
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

skip_counts = df_nodes.loc[df_nodes["status"] == "SKIP", "skip_class"].value_counts()
if len(skip_counts):
    axes[0].bar(skip_counts.index.astype(str), skip_counts.values, color="#e9c46a")
    axes[0].set_ylabel("functions")
    axes[0].set_title("SKIP by class")
    axes[0].tick_params(axis="x", rotation=20)
else:
    axes[0].set_title("SKIP by class (none)")

if n_model:
    fail_models = df_done[df_done["model_fail"]].sort_values("fail", ascending=False)
    top = fail_models.head(15)
    if len(top):
        labels = [f"{r.model_id} {r.name[:28]}" for r in top.itertuples()]
        axes[1].barh(range(len(top)), top["fail"].values, color="#e76f51")
        axes[1].set_yticks(range(len(top)))
        axes[1].set_yticklabels(labels, fontsize=8)
        axes[1].invert_yaxis()
        axes[1].set_xlabel("# failing functions")
        axes[1].set_title(f"Top failing models ({n_model_fail}/{n_model})")
    else:
        axes[1].set_title("No failing models")
else:
    axes[1].set_title("No completed models")

save_fig("04_skips_and_failing_models")

# %%
# 5) Per-model fail fraction scatter (size ~ eval count)
if n_model and "eval" in df_done.columns:
    fig, ax = plt.subplots()
    x = df_done["eval"].astype(float)
    y = (df_done["fail"] / df_done["eval"].clip(lower=1)).astype(float)
    sizes = 20 + 2 * df_done["eval"].astype(float)
    colors_pts = ["#e76f51" if f else "#2a9d8f" for f in df_done["model_fail"]]
    ax.scatter(x, y, s=sizes, c=colors_pts, alpha=0.7, edgecolors="none")
    ax.set_xlabel("# evaluated functions")
    ax.set_ylabel("function fail fraction")
    ax.set_title("Per-model fail fraction (red = model_fail)")
    save_fig("05_model_fail_fraction")

# %% [markdown]
# ## Tables — failing models & sample FAIL rows


# %%
if n_model:
    cols = [c for c in ["model_id", "name", "eval", "fail", "skip", "error", "model_fail",
                        "cm_total_s", "lut_total_s"] if c in df_done.columns]
    display_df = df_done.loc[df_done["model_fail"], cols].sort_values("fail", ascending=False)
    print(f"Failing models ({len(display_df)}):")
    print(display_df.to_string(index=False))
else:
    print("No completed model summaries.")

print()
fails = df_nodes[df_nodes["status"] == "FAIL"][
    ["model_id", "model_name", "node", "k", "fail_reason", "cm_s", "lut_s"]
]
print(f"FAIL rows ({len(fails)}; showing up to 40):")
print(fails.head(40).to_string(index=False))
