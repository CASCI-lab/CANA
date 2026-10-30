# %% [markdown]
# ROF vs Unate primes timing (from logs)
#
# Read the already-run timing logs and compare **ROF generate** vs
# **Unate Espresso** on the functions that appear in the ROF log
# (syntactic ROF ⇒ also in the unate run when *k* overlaps).
#
# Inputs:
#
# - ``qm_vs_rof_primes_timing/qm_vs_rof_primes_timing.log``
# - ``qm_vs_unate_primes_timing/qm_vs_unate_primes_timing.log``
#
# Figures land in ``logs/rof_vs_unate_primes_compare/``.

# %%
from __future__ import annotations

import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOGS_DIR = Path(__file__).resolve().parent
ROF_LOG = LOGS_DIR / "qm_vs_rof_primes_timing" / "qm_vs_rof_primes_timing.log"
UNATE_LOG = (
    LOGS_DIR / "qm_vs_unate_primes_timing" / "qm_vs_unate_primes_timing.log"
)
FIG_DIR = LOGS_DIR / "rof_vs_unate_primes_compare"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# fmt_s cells are either "skip" or "<num> <unit>" (space before unit).
_TIME_CELL = r"(?:skip|\d+(?:\.\d+)?\s*(?:µs|ms|s))"
_TIME_RE = re.compile(
    rf"^(?:skip|(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>µs|ms|s))$"
)
_SYN_ROF_RE = re.compile(
    rf"^(?P<label>\S+)\s+(?P<k>\d+)\s+"
    rf"(?P<count>{_TIME_CELL})\s+(?P<gen>{_TIME_CELL})\s+(?P<qm>{_TIME_CELL})\s+"
    rf"(?P<n_on>\d+)\s+(?P<n_off>\d+)\s+(?P<match>\S+)\s*$"
)
_SYN_UNATE_RE = re.compile(
    rf"^(?P<label>\S+)\s+(?P<k>\d+)\s+"
    rf"(?P<unate>{_TIME_CELL})\s+(?P<qm>{_TIME_CELL})\s+"
    rf"(?P<n_on>\d+)\s+(?P<n_off>\d+)\s+(?P<match>\S+)\s*$"
)
# short_model may itself contain '[' / ']'; take through the last ']'.
_BBM_ROF_RE = re.compile(
    rf"^\s*(?P<k>\d+)\s+"
    rf"(?P<count>{_TIME_CELL})\s+(?P<gen>{_TIME_CELL})\s+(?P<qm>{_TIME_CELL})\s+"
    rf"(?P<n_on>\d+)\s+(?P<n_off>\d+)\s+(?P<match>\S+)\s+"
    rf"(?P<target>\S+)\s+\[(?P<model>.+)\]\s*$"
)
_BBM_UNATE_RE = re.compile(
    rf"^\s*(?P<k>\d+)\s+"
    rf"(?P<unate>{_TIME_CELL})\s+(?P<qm>{_TIME_CELL})\s+"
    rf"(?P<n_on>\d+)\s+(?P<n_off>\d+)\s+(?P<match>\S+)\s+"
    rf"(?P<target>\S+)\s+\[(?P<model>.+)\]\s*$"
)


def parse_time(cell: str) -> Optional[float]:
    """Parse a ``fmt_s`` cell (``skip`` / ``123.4 µs`` / ``1.23 ms`` / ``1.000 s``)."""
    m = _TIME_RE.match(cell.strip())
    if m is None or m.group("num") is None:
        return None
    val = float(m.group("num"))
    unit = m.group("unit")
    if unit == "µs":
        return val * 1e-6
    if unit == "ms":
        return val * 1e-3
    return val


def fmt_s(t: Optional[float]) -> str:
    if t is None:
        return "skip"
    if t < 1e-3:
        return f"{t * 1e6:.1f} µs"
    if t < 1.0:
        return f"{t * 1e3:.2f} ms"
    return f"{t:.3f} s"


def fmt_x(x: float) -> str:
    """Format a dimensionless speedup ratio."""
    if x >= 100:
        return f"{x:.1f}x"
    if x >= 10:
        return f"{x:.2f}x"
    return f"{x:.3f}x"


def _data_lines(path: Path) -> Iterable[Tuple[str, str]]:
    """Yield ``(section, line)`` for synthetic/BBM data rows only."""
    section = ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip("\n")
        if line.startswith("=== equality") or line.startswith("=== averages"):
            break
        if line.startswith("=== synthetic"):
            section = "synthetic"
            continue
        if line.startswith("=== BBM"):
            section = "bbm"
            continue
        if not section or not line.strip():
            continue
        if line.lstrip().startswith("k ") or line.lstrip().startswith("label"):
            continue
        if " FAILED " in line:
            continue
        yield section, line


def parse_rof_log(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for section, line in _data_lines(path):
        if section == "synthetic":
            m = _SYN_ROF_RE.match(line)
            if not m:
                continue
            rows.append(
                {
                    "source": "synthetic",
                    "label": m.group("label"),
                    "k": int(m.group("k")),
                    "t_rof_count": parse_time(m.group("count")),
                    "t_rof_gen": parse_time(m.group("gen")),
                    "t_qm": parse_time(m.group("qm")),
                    "n_on": int(m.group("n_on")),
                    "n_off": int(m.group("n_off")),
                    "key": ("synthetic", m.group("label"), int(m.group("k"))),
                }
            )
        else:
            m = _BBM_ROF_RE.match(line)
            if not m:
                continue
            target, model = m.group("target"), m.group("model")
            rows.append(
                {
                    "source": "bbm",
                    "label": f"{model}::{target}",
                    "target": target,
                    "model": model,
                    "k": int(m.group("k")),
                    "t_rof_count": parse_time(m.group("count")),
                    "t_rof_gen": parse_time(m.group("gen")),
                    "t_qm": parse_time(m.group("qm")),
                    "n_on": int(m.group("n_on")),
                    "n_off": int(m.group("n_off")),
                    "key": ("bbm", target, model),
                }
            )
    return rows


def parse_unate_log(path: Path) -> Dict[Tuple[object, ...], Dict[str, object]]:
    by_key: Dict[Tuple[object, ...], Dict[str, object]] = {}
    for section, line in _data_lines(path):
        if section == "synthetic":
            m = _SYN_UNATE_RE.match(line)
            if not m:
                continue
            key: Tuple[object, ...] = (
                "synthetic",
                m.group("label"),
                int(m.group("k")),
            )
            by_key[key] = {
                "t_unate": parse_time(m.group("unate")),
                "t_qm": parse_time(m.group("qm")),
                "n_on": int(m.group("n_on")),
                "n_off": int(m.group("n_off")),
            }
        else:
            m = _BBM_UNATE_RE.match(line)
            if not m:
                continue
            target, model = m.group("target"), m.group("model")
            key = ("bbm", target, model)
            by_key[key] = {
                "t_unate": parse_time(m.group("unate")),
                "t_qm": parse_time(m.group("qm")),
                "n_on": int(m.group("n_on")),
                "n_off": int(m.group("n_off")),
            }
    return by_key


# %%
rof_rows = parse_rof_log(ROF_LOG)
unate_by_key = parse_unate_log(UNATE_LOG)

joined: List[Dict[str, object]] = []
missing_unate = 0
for r in rof_rows:
    u = unate_by_key.get(r["key"])  # type: ignore[arg-type]
    if u is None:
        missing_unate += 1
        continue
    row = dict(r)
    row["t_unate"] = u["t_unate"]
    row["unate_n_on"] = u["n_on"]
    row["unate_n_off"] = u["n_off"]
    joined.append(row)

print(f"ROF log rows:              {len(rof_rows)}")
print(f"Unate log keys:            {len(unate_by_key)}")
print(f"Joined (ROF ∩ unate):      {len(joined)}")
print(f"ROF rows missing in unate: {missing_unate}")
print(f"FIG_DIR={FIG_DIR}")

# %% [markdown]
# ## Coverage + both-timed subset
#
# Fair head-to-head uses rows where **both** ``rof-gen`` and ``unate`` ran
# (neither skipped). ``rof-count`` is also summarized when present.


# %%
both = [
    r
    for r in joined
    if isinstance(r["t_rof_gen"], (int, float))
    and isinstance(r["t_unate"], (int, float))
]
rof_only_gen = [
    r
    for r in joined
    if isinstance(r["t_rof_gen"], (int, float)) and r["t_unate"] is None
]
unate_only = [
    r
    for r in joined
    if r["t_rof_gen"] is None and isinstance(r["t_unate"], (int, float))
]
neither = [
    r for r in joined if r["t_rof_gen"] is None and r["t_unate"] is None
]

print("=== coverage on joined ROF functions ===")
print(f"  both timed (gen & unate): {len(both)}")
print(f"  rof-gen only (unate skip): {len(rof_only_gen)}")
print(f"  unate only (rof-gen skip): {len(unate_only)}")
print(f"  neither timed:             {len(neither)}")

by_source = defaultdict(int)
for r in both:
    by_source[str(r["source"])] += 1
print("  both-timed by source:", dict(by_source))

# %%
def _stats(vals: List[float]) -> Dict[str, float]:
    vals = sorted(vals)
    return {
        "n": len(vals),
        "min": vals[0],
        "median": statistics.median(vals),
        "mean": statistics.mean(vals),
        "p90": vals[max(0, int(0.9 * (len(vals) - 1)))],
        "max": vals[-1],
    }


def _print_stats(
    name: str, vals: List[float], *, fmt=fmt_s
) -> None:
    if not vals:
        print(f"  {name}: (none)")
        return
    s = _stats(vals)
    print(
        f"  {name}: n={s['n']}  "
        f"min={fmt(s['min'])}  med={fmt(s['median'])}  "
        f"mean={fmt(s['mean'])}  p90={fmt(s['p90'])}  "
        f"max={fmt(s['max'])}"
    )


t_gen = [float(r["t_rof_gen"]) for r in both]  # type: ignore[arg-type]
t_unate = [float(r["t_unate"]) for r in both]  # type: ignore[arg-type]
t_count = [
    float(r["t_rof_count"])  # type: ignore[arg-type]
    for r in both
    if isinstance(r["t_rof_count"], (int, float))
]
# unate / rof-gen  (>1 ⇒ ROF gen faster)
speedup = [u / g for g, u in zip(t_gen, t_unate)]
rof_faster = sum(1 for s in speedup if s > 1.0)
unate_faster = sum(1 for s in speedup if s < 1.0)
ties = sum(1 for s in speedup if s == 1.0)

print("")
print("=== wall times on both-timed rows ===")
_print_stats("rof-gen ", t_gen)
_print_stats("unate   ", t_unate)
_print_stats("rof-count", t_count)

print("")
print("=== speedup = t_unate / t_rof_gen  (>1 ⇒ ROF gen faster) ===")
_print_stats("speedup ", speedup, fmt=fmt_x)
print(
    f"  wins: ROF gen faster={rof_faster}  "
    f"unate faster={unate_faster}  ties={ties}"
)
if t_gen and t_unate:
    print(
        f"  geometric-ish summary via medians: "
        f"med(unate)/med(rof-gen) = "
        f"{statistics.median(t_unate) / statistics.median(t_gen):.2f}x"
    )

# By source
print("")
print("=== by source (both-timed) ===")
for source in ("synthetic", "bbm"):
    sub = [r for r in both if r["source"] == source]
    if not sub:
        continue
    sg = [float(r["t_rof_gen"]) for r in sub]  # type: ignore[arg-type]
    su = [float(r["t_unate"]) for r in sub]  # type: ignore[arg-type]
    sp = [u / g for g, u in zip(sg, su)]
    print(
        f"  {source:10s}  n={len(sub):5d}  "
        f"med_gen={fmt_s(statistics.median(sg))}  "
        f"med_unate={fmt_s(statistics.median(su))}  "
        f"med_speedup={statistics.median(sp):.2f}x  "
        f"rof_wins={sum(1 for x in sp if x > 1)}"
    )

# By k (BBM + synthetic pooled)
print("")
print("=== by k (both-timed; mean over rules) ===")
print(
    f"{'k':>4s} {'n':>5s} {'med_gen':>10s} {'med_unate':>10s} "
    f"{'med_spd':>8s} {'rof_win%':>8s}"
)
by_k: Dict[int, List[Dict[str, object]]] = defaultdict(list)
for r in both:
    by_k[int(r["k"])].append(r)  # type: ignore[arg-type]
for k in sorted(by_k):
    sub = by_k[k]
    sg = [float(r["t_rof_gen"]) for r in sub]  # type: ignore[arg-type]
    su = [float(r["t_unate"]) for r in sub]  # type: ignore[arg-type]
    sp = [u / g for g, u in zip(sg, su)]
    print(
        f"{k:4d} {len(sub):5d} {fmt_s(statistics.median(sg)):>10s} "
        f"{fmt_s(statistics.median(su)):>10s} "
        f"{statistics.median(sp):8.2f}x "
        f"{100.0 * sum(1 for x in sp if x > 1) / len(sp):7.1f}%"
    )

# Synthetic labels that appear in both
print("")
print("=== synthetic labels (both-timed) ===")
syn_labels = sorted({str(r["label"]) for r in both if r["source"] == "synthetic"})
for label in syn_labels:
    sub = [r for r in both if r["source"] == "synthetic" and r["label"] == label]
    sg = [float(r["t_rof_gen"]) for r in sub]  # type: ignore[arg-type]
    su = [float(r["t_unate"]) for r in sub]  # type: ignore[arg-type]
    sp = [u / g for g, u in zip(sg, su)]
    print(
        f"  {label:22s}  n={len(sub):2d}  "
        f"med_gen={fmt_s(statistics.median(sg))}  "
        f"med_unate={fmt_s(statistics.median(su))}  "
        f"med_spd={statistics.median(sp):.2f}x"
    )

# Largest speedups / slowdowns
print("")
print("=== extreme speedups (unate / rof-gen) ===")
ranked = sorted(
    both,
    key=lambda r: float(r["t_unate"]) / float(r["t_rof_gen"]),  # type: ignore[arg-type]
)
print("  ROF gen most ahead (top 5):")
for r in ranked[-5:][::-1]:
    spd = float(r["t_unate"]) / float(r["t_rof_gen"])  # type: ignore[arg-type]
    print(
        f"    {spd:8.1f}x  k={r['k']:3d}  {r['source']:10s}  "
        f"{str(r['label'])[:50]}  "
        f"gen={fmt_s(float(r['t_rof_gen']))}  "  # type: ignore[arg-type]
        f"unate={fmt_s(float(r['t_unate']))}"  # type: ignore[arg-type]
    )
print("  Unate most ahead (top 5):")
for r in ranked[:5]:
    spd = float(r["t_unate"]) / float(r["t_rof_gen"])  # type: ignore[arg-type]
    print(
        f"    {spd:8.3f}x  k={r['k']:3d}  {r['source']:10s}  "
        f"{str(r['label'])[:50]}  "
        f"gen={fmt_s(float(r['t_rof_gen']))}  "  # type: ignore[arg-type]
        f"unate={fmt_s(float(r['t_unate']))}"  # type: ignore[arg-type]
    )

# %% [markdown]
# ## Figures


# %%
def _series_mean_by_k(
    rows: List[Dict[str, object]], key: str
) -> Tuple[List[int], List[float]]:
    buckets: Dict[int, List[float]] = defaultdict(list)
    for r in rows:
        t = r.get(key)
        if isinstance(t, (int, float)):
            buckets[int(r["k"])].append(float(t))  # type: ignore[arg-type]
    ks = sorted(buckets)
    return ks, [statistics.mean(buckets[k]) for k in ks]


# 1) Mean time vs k (joined both-timed)
fig, ax = plt.subplots(figsize=(7.0, 4.5))
for key, label, marker in (
    ("t_rof_gen", "ROF generate", "s"),
    ("t_unate", "Unate Espresso", "o"),
    ("t_rof_count", "ROF count", "^"),
):
    ks, ts = _series_mean_by_k(both, key)
    if ks:
        ax.plot(ks, ts, marker=marker, linestyle="-", label=label)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("k (inputs)")
ax.set_ylabel("mean time (s)")
ax.set_title("ROF vs Unate — mean time vs k (joined ROF functions)")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
out = FIG_DIR / "time_vs_k__joined.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"wrote {out}")

# Per-source overlays
for source, title_bit in (("bbm", "BBM"), ("synthetic", "synthetic")):
    sub = [r for r in both if r["source"] == source]
    if not sub:
        continue
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for key, label, marker in (
        ("t_rof_gen", "ROF generate", "s"),
        ("t_unate", "Unate Espresso", "o"),
        ("t_rof_count", "ROF count", "^"),
    ):
        ks, ts = _series_mean_by_k(sub, key)
        if ks:
            ax.plot(ks, ts, marker=marker, linestyle="-", label=label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("k (inputs)")
    ax.set_ylabel("mean time (s)")
    ax.set_title(f"ROF vs Unate — {title_bit}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / f"time_vs_k__{source}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")

# 2) Scatter t_rof_gen vs t_unate
fig, ax = plt.subplots(figsize=(6.0, 6.0))
for source, color in (("bbm", "#4C78A8"), ("synthetic", "#F58518")):
    sub = [r for r in both if r["source"] == source]
    if not sub:
        continue
    xs = [float(r["t_rof_gen"]) for r in sub]  # type: ignore[arg-type]
    ys = [float(r["t_unate"]) for r in sub]  # type: ignore[arg-type]
    ax.scatter(xs, ys, s=12, alpha=0.35, c=color, label=source, edgecolors="none")
lo = min(min(t_gen), min(t_unate))
hi = max(max(t_gen), max(t_unate))
ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("ROF generate (s)")
ax.set_ylabel("Unate Espresso (s)")
ax.set_title("Per-function: Unate vs ROF generate")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
out = FIG_DIR / "scatter_unate_vs_rof_gen.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"wrote {out}")

# 3) Speedup vs k
fig, ax = plt.subplots(figsize=(7.0, 4.5))
for source, color in (("bbm", "#4C78A8"), ("synthetic", "#F58518")):
    sub = [r for r in both if r["source"] == source]
    if not sub:
        continue
    xs = [int(r["k"]) for r in sub]  # type: ignore[arg-type]
    ys = [
        float(r["t_unate"]) / float(r["t_rof_gen"])  # type: ignore[arg-type]
        for r in sub
    ]
    ax.scatter(xs, ys, s=12, alpha=0.35, c=color, label=source, edgecolors="none")
ax.axhline(1.0, color="k", ls="--", lw=1)
ax.set_yscale("log")
ax.set_xlabel("k (inputs)")
ax.set_ylabel("t_unate / t_rof_gen")
ax.set_title("Speedup vs k  (>1 ⇒ ROF gen faster)")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
out = FIG_DIR / "speedup_vs_k.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"wrote {out}")

# 4) Speedup histogram
fig, ax = plt.subplots(figsize=(7.0, 4.0))
log_sp = [math.log10(s) for s in speedup if s > 0]
ax.hist(log_sp, bins=40, color="#4C78A8", edgecolor="white", linewidth=0.4)
ax.axvline(0.0, color="k", ls="--", lw=1)
ax.set_xlabel(r"log10(t_unate / t_rof_gen)")
ax.set_ylabel("count")
ax.set_title("Speedup distribution (joined, both-timed)")
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
out = FIG_DIR / "speedup_hist.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"wrote {out}")

print(f"\ndone  |  both-timed={len(both)}  |  figs in {FIG_DIR}")
