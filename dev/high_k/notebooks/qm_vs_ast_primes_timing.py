# %% [markdown]
# QM vs AST primes timing
#
# Compare three ways to get (or count) prime implicants for Boolean
# expressions, and plot wall time vs *k* on log–log axes.
#
# 1. **Traditional** — import → dense LUT → Quine–McCluskey one-symbol
#    (``compute_one_symbol_schemata_from_bit_lut``). Lowest cutoff:
#    skip when ``2**k > MAX_QM_ROWS``.
# 2. **AST generate** — enumerate all ON+OFF primes via
#    ``iter_on_off_primes``. Only when method 3 reports a manageable
#    total (``n_on + n_off <= MAX_AST_GENERATE``).
# 3. **AST count** — ``count_on_off_primes`` (no enumeration). Extends
#    furthest in *k*.
#
# When both 1 and 2 run, their ON/OFF prime covers are compared as sets
# of one-symbol schemata (must match).
#
# Sources:
#
# - **synthetic** — every builder in ``SYNTHETIC_BUILDERS``
# - **bbm** — every syntactic-ROF rule in ``BBM/models/*/model.bnet``
#
# Output is teed to ``high_k/logs/qm_vs_ast_primes_timing/``.

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

from high_k.ast_primes import count_on_off_primes, iter_on_off_primes
from high_k.bbm_rules import (
    BBM_MODELS,
    as_bnet,
    iter_bnet_rules,
    unique_regulators,
)
from high_k.rof import check_syntactic_rof
from high_k.run_log import run_log_dir, start_run_log
from high_k.synthetic import EVEN_K_BUILDERS, SYNTHETIC_BUILDERS
from high_k.timing import fmt_s, import_detached_outputs, median_time

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

K_VALUES = [4, 6, 8, 10, 12, 16, 24, 32, 48, 64, 128, 256]

REPEATS = 3
RANDOM_SEED = 12345

# Skip QM when the dense LUT would exceed this many rows.
MAX_QM_ROWS = 2**12  # k <= 12

# Skip AST full enumeration when total ON+OFF primes exceed this.
MAX_AST_GENERATE = 100_000

FIG_DIR = run_log_dir("qm_vs_ast_primes_timing")

log, LOG_PATH = start_run_log("qm_vs_ast_primes_timing")
log.info(
    f"K_VALUES={K_VALUES}  REPEATS={REPEATS}  "
    f"MAX_QM_ROWS={MAX_QM_ROWS}  MAX_AST_GENERATE={MAX_AST_GENERATE}"
)
log.info(f"BBM_MODELS={BBM_MODELS}  FIG_DIR={FIG_DIR}")

# %% [markdown]
# ## Helpers: runners + QM ↔ AST equality


# %%
def ast_prime_to_schema(
    prime: Dict[str, bool], names: Sequence[str]
) -> str:
    """Map an AST ``{var: bool}`` prime to a positional ``{0,1,2}`` schema."""
    return "".join(
        "2" if n not in prime else ("1" if prime[n] else "0") for n in names
    )


def ast_primes_to_schemata(
    primes: Iterable[Dict[str, bool]], names: Sequence[str]
) -> Set[str]:
    return {ast_prime_to_schema(p, names) for p in primes}


def run_qm(expr: str, target: str = "Y") -> Dict[int, Set[str]]:
    """Import → LUT → QM one-symbol; return ON/OFF schema sets."""
    fo = import_detached_outputs(as_bnet(expr, target=target), target)
    fo.build_bit_lut_from_positional_expression()
    schemata = fo.compute_one_symbol_schemata_from_bit_lut()
    return {
        1: set(schemata.get(1, ())),
        0: set(schemata.get(0, ())),
    }


def run_ast_generate(
    expr: str,
) -> Tuple[List[Dict[str, bool]], List[Dict[str, bool]]]:
    """Enumerate all ON and OFF AST primes."""
    on = list(iter_on_off_primes(expr, "on"))
    off = list(iter_on_off_primes(expr, "off"))
    return on, off


def primes_match(
    qm: Dict[int, Set[str]],
    ast_on: Iterable[Dict[str, bool]],
    ast_off: Iterable[Dict[str, bool]],
    names: Sequence[str],
) -> bool:
    """True iff QM and AST produce the same ON and OFF schema covers."""
    return (
        qm[1] == ast_primes_to_schemata(ast_on, names)
        and qm[0] == ast_primes_to_schemata(ast_off, names)
    )


def time_all_methods(
    expr: str,
    *,
    k: int,
    target: str = "Y",
    repeats: int = REPEATS,
) -> Dict[str, object]:
    """Run methods 3 → 2 → 1 with gates; return timing + equality fields."""
    names = unique_regulators(expr)

    t_count, counts = median_time(
        lambda: count_on_off_primes(expr), repeats=repeats
    )
    n_on, n_off = counts  # type: ignore[misc]
    n_on = int(n_on)
    n_off = int(n_off)
    n_total = n_on + n_off

    t_gen: Optional[float] = None
    ast_on: Optional[List[Dict[str, bool]]] = None
    ast_off: Optional[List[Dict[str, bool]]] = None
    do_gen = n_total <= MAX_AST_GENERATE
    if do_gen:
        t_gen, gen_result = median_time(
            lambda: run_ast_generate(expr), repeats=repeats
        )
        ast_on, ast_off = gen_result  # type: ignore[misc]

    t_qm: Optional[float] = None
    qm_schemata: Optional[Dict[int, Set[str]]] = None
    do_qm = (2**k) <= MAX_QM_ROWS
    if do_qm:
        t_qm, qm_schemata = median_time(
            lambda: run_qm(expr, target=target), repeats=repeats
        )

    match: Optional[bool] = None
    if qm_schemata is not None and ast_on is not None and ast_off is not None:
        match = primes_match(qm_schemata, ast_on, ast_off, names)

    return {
        "k": k,
        "n_pi_on": n_on,
        "n_pi_off": n_off,
        "t_ast_count": t_count,
        "t_ast_gen": t_gen,
        "t_qm": t_qm,
        "match": match,
        "do_gen": do_gen,
        "do_qm": do_qm,
    }


# %% [markdown]
# ## Collect: synthetic builders


# %%
rows: List[Dict[str, object]] = []

log.info("=== synthetic ===")
log.info(
    f"{'label':18s} {'k':>3s}  {'count':>10s}  {'ast-gen':>10s}  "
    f"{'qm':>10s}  {'#on':>6s}  {'#off':>6s}  match"
)

for fname, builder in SYNTHETIC_BUILDERS.items():
    for k in K_VALUES:
        if fname in EVEN_K_BUILDERS and k % 2 != 0:
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
            f"{fname:18s} {k:3d}  {fmt_s(r['t_ast_count']):>10s}  "
            f"{fmt_s(r['t_ast_gen']):>10s}  {fmt_s(r['t_qm']):>10s}  "
            f"{r['n_pi_on']:6d}  {r['n_pi_off']:6d}  {r['match']!s}"
        )

# %% [markdown]
# ## Collect: full BBM corpus (syntactic ROF only)


# %%
bbm_seen = 0
bbm_skipped_non_rof = 0
bbm_failed = 0

log.info("")
log.info("=== BBM (all syntactic-ROF rules) ===")
if not BBM_MODELS.is_dir():
    log.info(f"BBM models not found at {BBM_MODELS}")
else:
    log.info(
        f"{'k':>3s}  {'count':>10s}  {'ast-gen':>10s}  {'qm':>10s}  "
        f"{'#on':>6s}  {'#off':>6s}  match  target / model"
    )
    for model_name, target, expr in iter_bnet_rules(BBM_MODELS):
        bbm_seen += 1
        if not check_syntactic_rof(expr)["is_rof"]:
            bbm_skipped_non_rof += 1
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
            f"{k:3d}  {fmt_s(r['t_ast_count']):>10s}  "
            f"{fmt_s(r['t_ast_gen']):>10s}  {fmt_s(r['t_qm']):>10s}  "
            f"{r['n_pi_on']:6d}  {r['n_pi_off']:6d}  "
            f"{r['match']!s:5s}  {target[:40]}  [{short_model}]"
        )

log.info(
    f"BBM: seen={bbm_seen}  kept_rof={bbm_seen - bbm_skipped_non_rof}  "
    f"skipped_non_rof={bbm_skipped_non_rof}  failed={bbm_failed}"
)

# %% [markdown]
# ## Equality summary


# %%
compared = [r for r in rows if r["match"] is not None]
mismatches = [r for r in compared if r["match"] is False]
log.info("")
log.info(
    f"=== equality (QM vs AST gen): "
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
    f"{'t_count':>10s} {'t_gen':>10s} {'t_qm':>10s}"
)

grouped: Dict[Tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)
for r in rows:
    if r["source"] == "bbm":
        key: Tuple[object, ...] = ("bbm", r["k"])
    else:
        key = (r["source"], r["label"], r["k"])
    grouped[key].append(r)

for key in sorted(grouped.keys(), key=lambda x: (str(x[0]), str(x[1]) if len(x) > 2 else "", int(x[-1]))):  # type: ignore[arg-type]
    group = grouped[key]
    source = str(group[0]["source"])
    k = int(group[0]["k"])  # type: ignore[arg-type]
    if source == "bbm":
        label = f"avg of {len(group)} rules"
    else:
        label = str(group[0]["label"])
    log.info(
        f"{source:10s} {label[:40]:40s} {k:3d} {len(group):3d}  "
        f"{fmt_s(_mean_or_none([g['t_ast_count'] for g in group])):>10s} "
        f"{fmt_s(_mean_or_none([g['t_ast_gen'] for g in group])):>10s} "
        f"{fmt_s(_mean_or_none([g['t_qm'] for g in group])):>10s}"
    )

log.info(f"\n{len(rows)} timing rows  |  wrote {LOG_PATH}")

# %% [markdown]
# ## Plots: time vs *k* (log–log)


# %%
METHOD_KEYS = (
    ("t_qm", "QM (LUT+one-symbol)", "o"),
    ("t_ast_gen", "AST generate", "s"),
    ("t_ast_count", "AST count", "^"),
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
        title="Prime implicants timing — BBM (mean over ROF rules)",
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
