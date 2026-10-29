# %% [markdown]
# Import + **syntactic** ROF check
#
# This notebook checks **syntactic** read-once formulas only: every
# identifier occurs exactly once in the written expression
# (``n_variables == n_unique_variables``).  It does **not** test semantic
# read-once-ness (algebraic rewriting that might make a formula read-once).
# ``is_rof`` in results / logs means syntactic ROF.
#
# For each Boolean rule (as a one-rule ``.bnet`` string):
#
# 1. **import** — ``import_detached_rule`` only (expression → positional
#    ``FunctionOutputs``; no LUT / one-symbol; timed)
# 2. **syntactic ROF check** — single-occurrence test above
#    (timed; result stored as ``is_rof``)
#
# Sources (toggle below):
#
# - **synthetic** — ``high_k.synthetic`` syntactic ROFs (``and`` / ``or`` /
#   ``pairwise_*`` / ``random``) at each ``K_VALUES`` size
# - **bbm** — every rule in ``BBM/models/*/model.bnet``, then corpus stats
#
# Shared helpers: ``high_k.rof`` (``check_syntactic_rof``),
# ``high_k.timing.time_import_and_rof``, ``high_k.synthetic``,
# ``high_k.bbm_rules``.
#
# Output is teed to ``high_k/logs/rof_check/``.

# %%
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# notebooks/ → high_k/ → dev/
_DEV_DIR = Path(__file__).resolve().parents[2]
if str(_DEV_DIR) not in sys.path:
    sys.path.insert(0, str(_DEV_DIR))

from high_k.bbm_rules import (
    BBM_MODELS,
    as_bnet,
    iter_bnet_rules,
    model_id_from_dirname,
)
from high_k.run_log import start_run_log
from high_k.synthetic import SYNTHETIC_BUILDERS, builder_accepts_k
from high_k.timing import fmt_s, time_import_and_rof

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# "synthetic" | "bbm" | "both"
SOURCE = "both"

# Synthetic input sizes (k). Each size gets expression-import + syntactic-ROF timing.
K_VALUES = [8, 16, 32, 64, 128, 256, 512, 1024]
# Median over this many runs for import and for the syntactic ROF check.
REPEATS = 3
RANDOM_SEED = 12345

# Which synthetic builders to run (subset of SYNTHETIC_BUILDERS keys).
SYNTHETIC_FNS = ["and", "or", "half_or_half_and", "pairwise_and_or", "pairwise_or_and", "random"]

# BBM keyword filters (see ``high_k.bbm_rules.iter_bnet_rules``)
INCLUDE_KEYWORDS: List[str] = []  # e.g. ["curated"]
EXCLUDE_KEYWORDS: List[str] = []  # e.g. ["multi-valued"]

# Stats threshold for "large" functions (unique variables)
LARGE_K = 12

log, LOG_PATH = start_run_log("rof_check")
log.info(f"SOURCE={SOURCE}")
log.info(f"K_VALUES={K_VALUES}  REPEATS={REPEATS}")
log.info(f"SYNTHETIC_FNS={SYNTHETIC_FNS}  BBM_MODELS={BBM_MODELS}")
log.info(
    f"INCLUDE_KEYWORDS={INCLUDE_KEYWORDS}  EXCLUDE_KEYWORDS={EXCLUDE_KEYWORDS}"
)


# %% [markdown]
# ## Synthetic: import + syntactic ROF vs *k*


# %%
synth_rows: List[Dict[str, object]] = []

if SOURCE in ("synthetic", "both"):
    log.info("=== synthetic (syntactic ROF check) ===")
    log.info(
        f"{'fn':10s} {'k':>9s}  {'import':>10s}  {'syn_rof':>10s}  "
        f"{'is_rof':>6s}  {'n_occ':>7s}  {'n_uniq':>7s}"
    )
    for fname in SYNTHETIC_FNS:
        builder = SYNTHETIC_BUILDERS[fname]
        for k in K_VALUES:
            if not builder_accepts_k(fname, k):
                continue
            if fname == "random":
                expr = builder(k, seed=RANDOM_SEED + k)
            else:
                expr = builder(k)
            bnet = as_bnet(expr, target="Y")
            try:
                r = time_import_and_rof(bnet, "Y", expr, repeats=REPEATS)
            except Exception as exc:  # noqa: BLE001
                log.info(f"{fname:10s} {k:9,d}  FAILED: {exc}")
                continue

            row = {
                "source": "synthetic",
                "label": fname,
                "k": int(r["k"]),  # type: ignore[arg-type]
                "t_import": float(r["t_import"]),  # type: ignore[arg-type]
                "t_rof": float(r["t_rof"]),  # type: ignore[arg-type]
                "is_rof": bool(r["is_rof"]),
                "n_variables": int(r["n_variables"]),  # type: ignore[arg-type]
                "n_unique_variables": int(r["n_unique_variables"]),  # type: ignore[arg-type]
            }
            synth_rows.append(row)
            log.info(
                f"{fname:10s} {k:9,d}  {fmt_s(row['t_import']):>10s}  "
                f"{fmt_s(row['t_rof']):>10s}  {str(row['is_rof']):>6s}  "
                f"{row['n_variables']:7,d}  {row['n_unique_variables']:7,d}"
            )

# %% [markdown]
# ## BBM: import + syntactic ROF for every rule


# %%
bbm_rows: List[Dict[str, object]] = []

if SOURCE in ("bbm", "both"):
    log.info("")
    log.info("=== BBM (all rules; syntactic ROF check) ===")
    if not BBM_MODELS.is_dir():
        log.info(f"BBM models not found at {BBM_MODELS}")
    else:
        n_done = 0
        for model_name, target, expr in iter_bnet_rules(
            BBM_MODELS,
            include_keywords=INCLUDE_KEYWORDS,
            exclude_keywords=EXCLUDE_KEYWORDS,
        ):
            model_id = model_id_from_dirname(model_name) or model_name
            bnet = as_bnet(expr, target=target)
            try:
                r = time_import_and_rof(bnet, target, expr, repeats=1)
            except Exception as exc:  # noqa: BLE001
                log.info(f"FAILED {model_id} {target}: {exc}")
                continue
            bbm_rows.append(
                {
                    "source": "bbm",
                    "model_id": model_id,
                    "model_directory": model_name,
                    "target": target,
                    "function": expr,
                    "k": r["k"],
                    "t_import": r["t_import"],
                    "t_rof": r["t_rof"],
                    "is_rof": r["is_rof"],
                    "n_variables": r["n_variables"],
                    "n_unique_variables": r["n_unique_variables"],
                }
            )
            n_done += 1
            if n_done % 2000 == 0:
                log.info(f"  … {n_done:,} rules")

        log.info(f"timed {len(bbm_rows):,} BBM rules")

# %% [markdown]
# ## BBM corpus stats


# %%
def _mean(vals: List[float]) -> Optional[float]:
    return sum(vals) / len(vals) if vals else None


if bbm_rows:
    n_models = len({r["model_id"] for r in bbm_rows})
    n_fn = len(bbm_rows)
    n_rof = sum(1 for r in bbm_rows if r["is_rof"])
    rof_rate = n_rof / n_fn

    by_model: Dict[object, List[bool]] = defaultdict(list)
    for r in bbm_rows:
        by_model[r["model_id"]].append(bool(r["is_rof"]))
    n_all_rof_models = sum(1 for flags in by_model.values() if all(flags))
    model_rof_rate = n_all_rof_models / len(by_model)

    large = [r for r in bbm_rows if int(r["n_unique_variables"]) > LARGE_K]  # type: ignore[arg-type]
    large_frac = len(large) / n_fn
    large_rof = sum(1 for r in large if r["is_rof"])
    large_rof_rate = (large_rof / len(large)) if large else None

    models_with_large = {
        r["model_id"]
        for r in bbm_rows
        if int(r["n_unique_variables"]) > LARGE_K  # type: ignore[arg-type]
    }
    large_model_frac = len(models_with_large) / n_models

    large_by_model: Dict[object, List[bool]] = defaultdict(list)
    for r in large:
        large_by_model[r["model_id"]].append(bool(r["is_rof"]))
    n_large_all_rof = sum(1 for flags in large_by_model.values() if all(flags))
    large_model_all_rof_frac = (
        n_large_all_rof / len(large_by_model) if large_by_model else None
    )

    t_imports = [float(r["t_import"]) for r in bbm_rows]  # type: ignore[arg-type]
    t_rofs = [float(r["t_rof"]) for r in bbm_rows]  # type: ignore[arg-type]

    log.info("")
    log.info("=== BBM stats (syntactic ROF) ===")
    log.info(f"Models:    {n_models:,}")
    log.info(f"Functions: {n_fn:,}")
    log.info(f"Syntactic ROFs:      {n_rof:,}")
    log.info(f"Syntactic ROF rate:  {rof_rate:.2%}")
    log.info(
        f"Fraction of models with only syntactic ROFs: {model_rof_rate:.2%}"
    )
    log.info(
        f"Fraction of functions with >{LARGE_K} unique variables: "
        f"{large_frac:.2%}"
    )
    if large_rof_rate is not None:
        log.info(
            f"Syntactic ROF rate for functions with >{LARGE_K} unique "
            f"variables: {large_rof_rate:.2%}"
        )
    log.info(
        f"Fraction of models with ≥1 function with >{LARGE_K} variables: "
        f"{large_model_frac:.2%}"
    )
    if large_model_all_rof_frac is not None:
        log.info(
            f"Among those, fraction where all >{LARGE_K}-var functions "
            f"are syntactic ROFs: {large_model_all_rof_frac:.2%}"
        )
    log.info("")
    log.info(
        f"Models with only syntactic ROFs: "
        f"{n_all_rof_models:,} / {n_models:,}"
    )
    log.info(
        f"Functions with >{LARGE_K} variables: "
        f"{len(large):,} / {n_fn:,}"
    )
    log.info(
        f"Syntactic ROFs with >{LARGE_K} variables: "
        f"{large_rof:,} / {len(large):,}"
    )
    log.info(
        f"Models with >{LARGE_K}-variable function: "
        f"{len(models_with_large):,} / {n_models:,}"
    )
    log.info(
        f"Models where all >{LARGE_K}-variable functions are syntactic "
        f"ROFs: {n_large_all_rof:,} / {len(large_by_model):,}"
    )

    # Extra-variable histogram for large functions (n_occ - n_unique)
    if large:
        log.info("")
        log.info(
            f"Extra variable occurrences among >{LARGE_K}-unique functions:"
        )
        extras = [
            int(r["n_variables"]) - int(r["n_unique_variables"])  # type: ignore[arg-type]
            for r in large
        ]
        total = len(extras)
        for n in range(20):
            count = sum(1 for e in extras if e == n)
            label = "synROF" if n == 0 else f"+{n}"
            log.info(
                f"  {label:>6}: {count / total:.2%} ({count:,} / {total:,})"
            )
        rest = sum(1 for e in extras if e >= 20)
        log.info(f"  {'rest':>6}: {rest / total:.2%} ({rest:,} / {total:,})")

    log.info("")
    log.info(
        f"import time: mean={fmt_s(_mean(t_imports))}  "
        f"min={fmt_s(min(t_imports))}  max={fmt_s(max(t_imports))}"
    )
    log.info(
        f"syntactic ROF check time: mean={fmt_s(_mean(t_rofs))}  "
        f"min={fmt_s(min(t_rofs))}  max={fmt_s(max(t_rofs))}"
    )

# %% [markdown]
# ## Summary


# %%
log.info("")
log.info(
    f"synthetic rows={len(synth_rows)}  bbm rows={len(bbm_rows)}  "
    f"|  SOURCE={SOURCE}"
)
log.info(f"wrote {LOG_PATH}")
log.close()
