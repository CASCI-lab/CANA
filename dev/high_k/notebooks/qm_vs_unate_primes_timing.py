# %% [markdown]
# QM vs Unate primes timing
#
# Compare two ways to get prime implicants for Boolean expressions, and
# plot wall time vs *k* on log–log axes.
#
# 1. **Traditional** — import → dense LUT → Quine–McCluskey one-symbol
#    (``compute_one_symbol_schemata_from_bit_lut``). Skip when
#    ``2**k > MAX_QM_ROWS``.
# 2. **Unate Espresso** — ``on_off_primes`` (pyeda Espresso on ``f`` and
#    ``~f``). Requires syntactic unate; wall-clock gated by
#    ``UNATE_TIMEOUT_S`` (subprocess kill).
#
# When both run, their ON/OFF prime covers are compared as sets of
# one-symbol schemata (must match).
#
# Sources:
#
# - **synthetic** — every builder in ``SYNTHETIC_BUILDERS`` (includes
#   multi-occurrence unate builders such as ``majority``)
# - **bbm** — every syntactic-unate rule in ``BBM/models/*/model.bnet``
#
# Output is teed to ``high_k/logs/qm_vs_unate_primes_timing/``.

# %%
from __future__ import annotations

import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# notebooks/ → high_k/ → dev/
_DEV_DIR = Path(__file__).resolve().parents[2]
if str(_DEV_DIR) not in sys.path:
    sys.path.insert(0, str(_DEV_DIR))

from high_k.bbm_rules import (
    BBM_MODELS,
    as_bnet,
    iter_bnet_rules,
    unique_regulators,
)
from high_k.run_log import run_log_dir, start_run_log
from high_k.synthetic import SYNTHETIC_BUILDERS, builder_accepts_k
from high_k.timing import fmt_s, import_detached_outputs, median_time
from high_k.unate import check_syntactic_unate, on_off_primes

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Espresso / to_dnf does not scale like ROF count — keep *k* modest.
K_VALUES = [4, 6, 8, 10, 12, 16, 20, 24]

REPEATS = 3
RANDOM_SEED = 12345

# Skip QM when the dense LUT would exceed this many rows.
MAX_QM_ROWS = 2**12  # k <= 12

# Wall-clock limit for Espresso (subprocess kill inside on_off_primes).
UNATE_TIMEOUT_S = 10.0

# Majority DNF size is C(k, floor(k/2)+1); keep *k* small.
MAX_MAJORITY_K = 10

FIG_DIR = run_log_dir("qm_vs_unate_primes_timing")

log, LOG_PATH = start_run_log("qm_vs_unate_primes_timing")
log.info(
    f"K_VALUES={K_VALUES}  REPEATS={REPEATS}  "
    f"MAX_QM_ROWS={MAX_QM_ROWS}  UNATE_TIMEOUT_S={UNATE_TIMEOUT_S}  "
    f"MAX_MAJORITY_K={MAX_MAJORITY_K}"
)
log.info(f"BBM_MODELS={BBM_MODELS}  FIG_DIR={FIG_DIR}")

# %% [markdown]
# ## Helpers: runners + QM ↔ Unate equality


# %%
def prime_to_schema(
    prime: Dict[str, bool], names: Sequence[str]
) -> str:
    """Map a ``{var: bool}`` prime to a positional ``{0,1,2}`` schema."""
    return "".join(
        "2" if n not in prime else ("1" if prime[n] else "0") for n in names
    )


def primes_to_schemata(
    primes: Iterable[Dict[str, bool]], names: Sequence[str]
) -> Set[str]:
    return {prime_to_schema(p, names) for p in primes}


def run_qm(expr: str, target: str = "Y") -> Dict[int, Set[str]]:
    """Import → LUT → QM one-symbol; return ON/OFF schema sets."""
    fo = import_detached_outputs(as_bnet(expr, target=target), target)
    fo.build_bit_lut_from_positional_expression()
    schemata = fo.compute_one_symbol_schemata_from_bit_lut()
    return {
        1: set(schemata.get(1, ())),
        0: set(schemata.get(0, ())),
    }


def run_unate(
    expr: str,
) -> Tuple[List[Dict[str, bool]], List[Dict[str, bool]]]:
    """Espresso ON and OFF primes for a syntactically unate expression."""
    return on_off_primes(expr, timeout_s=UNATE_TIMEOUT_S)


def primes_match(
    qm: Dict[int, Set[str]],
    unate_on: Iterable[Dict[str, bool]],
    unate_off: Iterable[Dict[str, bool]],
    names: Sequence[str],
) -> bool:
    """True iff QM and Unate Espresso produce the same ON/OFF covers."""
    return (
        qm[1] == primes_to_schemata(unate_on, names)
        and qm[0] == primes_to_schemata(unate_off, names)
    )


def time_all_methods(
    expr: str,
    *,
    k: int,
    target: str = "Y",
    repeats: int = REPEATS,
) -> Dict[str, object]:
    """Run Unate Espresso then QM with gates; return timing + equality."""
    names = unique_regulators(expr)

    t_unate: Optional[float] = None
    unate_on: Optional[List[Dict[str, bool]]] = None
    unate_off: Optional[List[Dict[str, bool]]] = None
    n_on = 0
    n_off = 0
    do_unate = True
    try:
        t_unate, gen_result = median_time(
            lambda: run_unate(expr), repeats=repeats
        )
        unate_on, unate_off = gen_result  # type: ignore[misc]
        n_on = len(unate_on)
        n_off = len(unate_off)
    except TimeoutError:
        do_unate = False
        t_unate = None
        unate_on = None
        unate_off = None

    t_qm: Optional[float] = None
    qm_schemata: Optional[Dict[int, Set[str]]] = None
    do_qm = (2**k) <= MAX_QM_ROWS
    if do_qm:
        t_qm, qm_schemata = median_time(
            lambda: run_qm(expr, target=target), repeats=repeats
        )

    match: Optional[bool] = None
    if (
        qm_schemata is not None
        and unate_on is not None
        and unate_off is not None
    ):
        match = primes_match(qm_schemata, unate_on, unate_off, names)

    return {
        "k": k,
        "n_pi_on": n_on,
        "n_pi_off": n_off,
        "t_unate": t_unate,
        "t_qm": t_qm,
        "match": match,
        "do_unate": do_unate,
        "do_qm": do_qm,
    }


# %% [markdown]
# ## Collect: synthetic builders


# %%
rows: List[Dict[str, object]] = []

log.info("=== synthetic ===")
log.info(
    f"{'label':18s} {'k':>3s}  {'unate':>10s}  {'qm':>10s}  "
    f"{'#on':>6s}  {'#off':>6s}  match"
)

for fname, builder in SYNTHETIC_BUILDERS.items():
    for k in K_VALUES:
        if not builder_accepts_k(fname, k):
            continue
        if fname == "majority" and k > MAX_MAJORITY_K:
            continue
        if fname == "random":
            expr = builder(k, seed=RANDOM_SEED + k)
        else:
            expr = builder(k)
        try:
            r = time_all_methods(expr, k=k, target="Y", repeats=REPEATS)
        except Exception as exc:  # noqa: BLE001 — report and continue
            log.info(f"{fname:18s} {k:3d}  FAILED: {exc}")
            continue
        r["source"] = "synthetic"
        r["label"] = fname
        r["expr"] = expr
        rows.append(r)
        log.info(
            f"{fname:18s} {k:3d}  {fmt_s(r['t_unate']):>10s}  "
            f"{fmt_s(r['t_qm']):>10s}  "
            f"{r['n_pi_on']:6d}  {r['n_pi_off']:6d}  {r['match']!s}"
        )

# %% [markdown]
# ## Collect: full BBM corpus (syntactic unate only)


# %%
bbm_seen = 0
bbm_skipped_non_unate = 0
bbm_failed = 0

log.info("")
log.info("=== BBM (all syntactic-unate rules) ===")
if not BBM_MODELS.is_dir():
    log.info(f"BBM models not found at {BBM_MODELS}")
else:
    log.info(
        f"{'k':>3s}  {'unate':>10s}  {'qm':>10s}  "
        f"{'#on':>6s}  {'#off':>6s}  match  target / model"
    )
    for model_name, target, expr in iter_bnet_rules(BBM_MODELS):
        bbm_seen += 1
        if not check_syntactic_unate(expr)["is_unate"]:
            bbm_skipped_non_unate += 1
            continue
        names = unique_regulators(expr)
        k = len(names)
        if k == 0:
            continue
        try:
            # Diversity comes from distinct rules; time each once.
            r = time_all_methods(expr, k=k, target=target, repeats=1)
        except Exception as exc:  # noqa: BLE001
            bbm_failed += 1
            log.info(
                f"{k:3d}  FAILED on {target} [{model_name}]: {exc}"
            )
            continue
        r["source"] = "bbm"
        r["label"] = f"{model_name}::{target}"
        r["expr"] = expr
        rows.append(r)
        short_model = model_name.split("__")[-1][:28]
        log.info(
            f"{k:3d}  {fmt_s(r['t_unate']):>10s}  "
            f"{fmt_s(r['t_qm']):>10s}  "
            f"{r['n_pi_on']:6d}  {r['n_pi_off']:6d}  "
            f"{r['match']!s:5s}  {target[:40]}  [{short_model}]"
        )

log.info(
    f"BBM: seen={bbm_seen}  "
    f"kept_unate={bbm_seen - bbm_skipped_non_unate}  "
    f"skipped_non_unate={bbm_skipped_non_unate}  failed={bbm_failed}"
)

# %% [markdown]
# ## Equality summary


# %%
compared = [r for r in rows if r["match"] is not None]
mismatches = [r for r in compared if r["match"] is False]
log.info("")
log.info(
    f"=== equality (QM vs Unate Espresso): "
    f"{len(compared)} compared, {len(mismatches)} mismatches ==="
)
if mismatches:
    for r in mismatches[:20]:
        log.info(
            f"  MISMATCH  source={r['source']}  label={r['label']}  "
            f"k={r['k']}  on={r['n_pi_on']}  off={r['n_pi_off']}"
        )
    if len(mismatches) > 20:
        log.info(f"  ... and {len(mismatches) - 20} more")
else:
    log.info("all compared pairs match")

# %% [markdown]
# ## Summary averages


# %%
def _mean_or_none(vals: List[object]) -> Optional[float]:
    nums = [float(v) for v in vals if isinstance(v, (int, float))]
    if not nums:
        return None
    return float(statistics.mean(nums))


log.info("")
log.info("=== averages by (source, label, k) / (bbm, k) ===")
log.info(
    f"{'source':10s} {'label':40s} {'k':>3s} {'n':>3s}  "
    f"{'t_unate':>10s} {'t_qm':>10s}"
)

grouped: Dict[Tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)
for r in rows:
    if r["source"] == "bbm":
        key: Tuple[object, ...] = ("bbm", r["k"])
    else:
        key = (r["source"], r["label"], r["k"])
    grouped[key].append(r)

for key in sorted(
    grouped.keys(),
    key=lambda x: (
        str(x[0]),
        str(x[1]) if len(x) > 2 else "",
        int(x[-1]),
    ),  # type: ignore[arg-type]
):
    group = grouped[key]
    source = str(group[0]["source"])
    k = int(group[0]["k"])  # type: ignore[arg-type]
    if source == "bbm":
        label = f"avg of {len(group)} rules"
    else:
        label = str(group[0]["label"])
    log.info(
        f"{source:10s} {label[:40]:40s} {k:3d} {len(group):3d}  "
        f"{fmt_s(_mean_or_none([g['t_unate'] for g in group])):>10s} "
        f"{fmt_s(_mean_or_none([g['t_qm'] for g in group])):>10s}"
    )

log.info(f"\n{len(rows)} timing rows  |  wrote {LOG_PATH}")

# %% [markdown]
# ## Plots: time vs *k* (log–log)


# %%
METHOD_KEYS = (
    ("t_qm", "QM (LUT+one-symbol)", "o"),
    ("t_unate", "Unate Espresso", "s"),
)


def _series_by_k(
    subset: List[Dict[str, object]], time_key: str
) -> Tuple[List[int], List[float]]:
    """Mean time per *k* for rows that have *time_key* set."""
    by_k: Dict[int, List[float]] = defaultdict(list)
    for r in subset:
        t = r.get(time_key)
        if isinstance(t, (int, float)):
            by_k[int(r["k"])].append(float(t))  # type: ignore[arg-type]
    ks = sorted(by_k)
    return ks, [statistics.mean(by_k[k]) for k in ks]


def plot_time_vs_k(
    subset: List[Dict[str, object]],
    title: str,
    outfile: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    any_series = False
    for time_key, label, marker in METHOD_KEYS:
        ks, ts = _series_by_k(subset, time_key)
        if not ks:
            continue
        any_series = True
        ax.plot(ks, ts, marker=marker, linestyle="-", label=label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("k (inputs)")
    ax.set_ylabel("time (s)")
    ax.set_title(title)
    if any_series:
        ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    log.info(f"wrote {outfile}")


# Per synthetic builder
for fname in SYNTHETIC_BUILDERS:
    subset = [
        r for r in rows if r["source"] == "synthetic" and r["label"] == fname
    ]
    if not subset:
        continue
    plot_time_vs_k(
        subset,
        title=f"Prime implicants timing — synthetic / {fname}",
        outfile=FIG_DIR / f"time_vs_k__synthetic__{fname}.png",
    )

# BBM
bbm_rows = [r for r in rows if r["source"] == "bbm"]
if bbm_rows:
    plot_time_vs_k(
        bbm_rows,
        title="Prime implicants timing — BBM (mean over unate rules)",
        outfile=FIG_DIR / "time_vs_k__bbm.png",
    )

# Aggregated across all sources
if rows:
    plot_time_vs_k(
        rows,
        title="Prime implicants timing — aggregated (all sources)",
        outfile=FIG_DIR / "time_vs_k__aggregated.png",
    )

log.info(f"done  |  {len(rows)} rows  |  mismatches={len(mismatches)}")
log.close()
