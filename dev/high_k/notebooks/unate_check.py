# %% [markdown]
# Import + **syntactic** unate check
#
# This notebook checks **syntactic** unateness only: after negation is
# pushed to the leaves (NNF), every identifier appears only positively or
# only negatively.  The checker stops at the **first** binate variable
# (never enumerates all).  Same-polarity repeats (e.g. ``a & a``) are
# still unate; that is distinct from syntactic ROF (single occurrence).
# It does **not** test semantic unateness (cofactor / truth-table
# monotonicity).  ``is_unate`` in results / logs means syntactic unate.
#
# For each Boolean rule (as a one-rule ``.bnet`` string):
#
# 1. **import** — ``import_detached_rule`` only (expression → positional
#    ``FunctionOutputs``; no LUT / one-symbol; timed)
# 2. **syntactic unate check** — leaf-polarity test above
#    (timed; result stored as ``is_unate``)
# 3. **syntactic ROF** — single-occurrence flag (untimed; for
#    not-ROF-but-unate stats)
#
# Sources (toggle below):
#
# - **synthetic** — ``high_k.synthetic`` builders (``and`` / ``or`` /
#   ``pairwise_*`` / ``random``) at each ``K_VALUES`` size
# - **bbm** — every rule in ``BBM/models/*/model.bnet``, then corpus stats.
#   Each BBM function is bounded by ``BBM_TIMEOUT_S`` (import + unate);
#   oversized ASTs that exceed the limit are skipped.
#
# Shared helpers: ``high_k.unate`` (``check_syntactic_unate``),
# ``high_k.timing.time_import_and_unate``, ``high_k.synthetic``,
# ``high_k.bbm_rules``.
#
# Output is teed to ``high_k/logs/unate_check/``.

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
from high_k.synthetic import EVEN_K_BUILDERS, SYNTHETIC_BUILDERS
from high_k.timing import call_with_timeout, fmt_s, time_import_and_unate

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# "synthetic" | "bbm" | "both"
SOURCE = "both"

# Synthetic input sizes (k). Each size gets expression-import + syntactic-unate timing.
K_VALUES = [8, 16, 32, 64, 128, 256, 512, 1024]
# Median over this many runs for import and for the syntactic unate check.
REPEATS = 3
RANDOM_SEED = 12345

# Which synthetic builders to run (subset of SYNTHETIC_BUILDERS keys).
SYNTHETIC_FNS = ["and", "or", "half_or_half_and", "pairwise_and_or", "pairwise_or_and", "random"]

# BBM keyword filters (see ``high_k.bbm_rules.iter_bnet_rules``)
INCLUDE_KEYWORDS: List[str] = []  # e.g. ["curated"]
EXCLUDE_KEYWORDS: List[str] = []  # e.g. ["multi-valued"]

# Stats threshold for "large" functions (unique variables)
LARGE_K = 12

# Per-function wall-clock limit for BBM import + syntactic unate (seconds).
BBM_TIMEOUT_S = 10.0

log, LOG_PATH = start_run_log("unate_check")
log.info(f"SOURCE={SOURCE}")
log.info(f"K_VALUES={K_VALUES}  REPEATS={REPEATS}")
log.info(f"SYNTHETIC_FNS={SYNTHETIC_FNS}  BBM_MODELS={BBM_MODELS}")
log.info(
    f"INCLUDE_KEYWORDS={INCLUDE_KEYWORDS}  EXCLUDE_KEYWORDS={EXCLUDE_KEYWORDS}"
)
log.info(f"BBM_TIMEOUT_S={BBM_TIMEOUT_S}")


# %% [markdown]
# ## Synthetic: import + syntactic unate vs *k*


# %%
synth_rows: List[Dict[str, object]] = []

if SOURCE in ("synthetic", "both"):
    log.info("=== synthetic (syntactic unate check) ===")
    log.info(
        f"{'fn':10s} {'k':>9s}  {'import':>10s}  {'syn_unate':>10s}  "
        f"{'is_unate':>8s}  {'is_rof':>6s}  {'n_uniq':>7s}"
    )
    for fname in SYNTHETIC_FNS:
        builder = SYNTHETIC_BUILDERS[fname]
        for k in K_VALUES:
            if fname in EVEN_K_BUILDERS and k % 2 != 0:
                continue
            if fname == "random":
                expr = builder(k, seed=RANDOM_SEED + k)
            else:
                expr = builder(k)
            bnet = as_bnet(expr, target="Y")
            try:
                r = time_import_and_unate(bnet, "Y", expr, repeats=REPEATS)
            except Exception as exc:  # noqa: BLE001
                log.info(f"{fname:10s} {k:9,d}  FAILED: {exc}")
                continue

            row = {
                "source": "synthetic",
                "label": fname,
                "k": int(r["k"]),  # type: ignore[arg-type]
                "t_import": float(r["t_import"]),  # type: ignore[arg-type]
                "t_unate": float(r["t_unate"]),  # type: ignore[arg-type]
                "is_unate": bool(r["is_unate"]),
                "is_rof": bool(r["is_rof"]),
                "n_variables": int(r["n_variables"]),  # type: ignore[arg-type]
                "n_unique_variables": int(r["n_unique_variables"]),  # type: ignore[arg-type]
            }
            synth_rows.append(row)
            log.info(
                f"{fname:10s} {k:9,d}  {fmt_s(row['t_import']):>10s}  "
                f"{fmt_s(row['t_unate']):>10s}  {str(row['is_unate']):>8s}  "
                f"{str(row['is_rof']):>6s}  {row['n_unique_variables']:7,d}"
            )

# %% [markdown]
# ## BBM: import + syntactic unate for every rule


# %%
bbm_rows: List[Dict[str, object]] = []
bbm_timed_out = 0
bbm_failed = 0

if SOURCE in ("bbm", "both"):
    log.info("")
    log.info(
        f"=== BBM (all rules; syntactic unate check; "
        f"timeout={BBM_TIMEOUT_S:g}s) ==="
    )
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
                r = call_with_timeout(
                    BBM_TIMEOUT_S,
                    lambda b=bnet, t=target, e=expr: time_import_and_unate(
                        b, t, e, repeats=1
                    ),
                )
            except TimeoutError as exc:
                bbm_timed_out += 1
                log.info(f"TIMEOUT {model_id} {target}: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001
                bbm_failed += 1
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
                    "t_unate": r["t_unate"],
                    "is_unate": r["is_unate"],
                    "is_rof": r["is_rof"],
                    "n_variables": r["n_variables"],
                    "n_unique_variables": r["n_unique_variables"],
                }
            )
            n_done += 1
            if n_done % 2000 == 0:
                log.info(
                    f"  … {n_done:,} rules  "
                    f"(timeout={bbm_timed_out:,}  failed={bbm_failed:,})"
                )

        log.info(
            f"timed {len(bbm_rows):,} BBM rules  "
            f"timeout={bbm_timed_out:,}  failed={bbm_failed:,}"
        )

# %% [markdown]
# ## BBM corpus stats (syntactic unate)


# %%
def _mean(vals: List[float]) -> Optional[float]:
    return sum(vals) / len(vals) if vals else None


if bbm_rows:
    n_models = len({r["model_id"] for r in bbm_rows})
    n_fn = len(bbm_rows)
    n_unate = sum(1 for r in bbm_rows if r["is_unate"])
    n_rof = sum(1 for r in bbm_rows if r["is_rof"])
    n_binate = n_fn - n_unate
    unate_rate = n_unate / n_fn
    rof_rate = n_rof / n_fn
    binate_rate = n_binate / n_fn

    by_model: Dict[object, List[bool]] = defaultdict(list)
    for r in bbm_rows:
        by_model[r["model_id"]].append(bool(r["is_unate"]))
    n_all_unate_models = sum(1 for flags in by_model.values() if all(flags))
    model_unate_rate = n_all_unate_models / len(by_model)

    large = [r for r in bbm_rows if int(r["n_unique_variables"]) > LARGE_K]  # type: ignore[arg-type]
    large_frac = len(large) / n_fn
    large_unate = sum(1 for r in large if r["is_unate"])
    large_unate_rate = (large_unate / len(large)) if large else None

    models_with_large = {
        r["model_id"]
        for r in bbm_rows
        if int(r["n_unique_variables"]) > LARGE_K  # type: ignore[arg-type]
    }
    large_model_frac = len(models_with_large) / n_models

    large_by_model: Dict[object, List[bool]] = defaultdict(list)
    for r in large:
        large_by_model[r["model_id"]].append(bool(r["is_unate"]))
    n_large_all_unate = sum(1 for flags in large_by_model.values() if all(flags))
    large_model_all_unate_frac = (
        n_large_all_unate / len(large_by_model) if large_by_model else None
    )

    t_imports = [float(r["t_import"]) for r in bbm_rows]  # type: ignore[arg-type]
    t_unates = [float(r["t_unate"]) for r in bbm_rows]  # type: ignore[arg-type]

    log.info("")
    log.info("=== BBM stats (syntactic unate) ===")
    log.info(f"Models:    {n_models:,}")
    log.info(f"Functions: {n_fn:,}")
    log.info(f"Syntactic unate:     {n_unate:,}")
    log.info(f"Syntactic unate rate: {unate_rate:.2%}")
    log.info(f"Syntactic ROF rate:   {rof_rate:.2%}")
    log.info(f"Binate (not unate) rate: {binate_rate:.2%}")
    log.info(
        f"Fraction of models with only syntactic unates: {model_unate_rate:.2%}"
    )
    log.info(
        f"Fraction of functions with >{LARGE_K} unique variables: "
        f"{large_frac:.2%}"
    )
    if large_unate_rate is not None:
        log.info(
            f"Syntactic unate rate for functions with >{LARGE_K} unique "
            f"variables: {large_unate_rate:.2%}"
        )
    log.info(
        f"Fraction of models with ≥1 function with >{LARGE_K} variables: "
        f"{large_model_frac:.2%}"
    )
    if large_model_all_unate_frac is not None:
        log.info(
            f"Among those, fraction where all >{LARGE_K}-var functions "
            f"are syntactic unates: {large_model_all_unate_frac:.2%}"
        )
    log.info("")
    log.info(
        f"Models with only syntactic unates: "
        f"{n_all_unate_models:,} / {n_models:,}"
    )
    log.info(
        f"Functions with >{LARGE_K} variables: "
        f"{len(large):,} / {n_fn:,}"
    )
    log.info(
        f"Syntactic unates with >{LARGE_K} variables: "
        f"{large_unate:,} / {len(large):,}"
    )
    log.info(
        f"Models with >{LARGE_K}-variable function: "
        f"{len(models_with_large):,} / {n_models:,}"
    )
    log.info(
        f"Models where all >{LARGE_K}-variable functions are syntactic "
        f"unates: {n_large_all_unate:,} / {len(large_by_model):,}"
    )

    log.info("")
    log.info(
        f"import time: mean={fmt_s(_mean(t_imports))}  "
        f"min={fmt_s(min(t_imports))}  max={fmt_s(max(t_imports))}"
    )
    log.info(
        f"syntactic unate check time: mean={fmt_s(_mean(t_unates))}  "
        f"min={fmt_s(min(t_unates))}  max={fmt_s(max(t_unates))}"
    )

# %% [markdown]
# ## Not syntactic ROF but syntactic unate
#
# ``ROF ⊆ unate ⊆ all``; the interesting slice is unate with repeated
# same-polarity identifiers (``is_unate and not is_rof``).


# %%
if bbm_rows:
    n_fn = len(bbm_rows)
    n_models = len({r["model_id"] for r in bbm_rows})

    unate_not_rof = [
        r for r in bbm_rows if r["is_unate"] and not r["is_rof"]
    ]
    n_unr = len(unate_not_rof)
    unr_rate = n_unr / n_fn

    large = [r for r in bbm_rows if int(r["n_unique_variables"]) > LARGE_K]  # type: ignore[arg-type]
    large_unr = [
        r for r in large if r["is_unate"] and not r["is_rof"]
    ]
    large_unr_rate = (len(large_unr) / len(large)) if large else None

    models_with_unr = {r["model_id"] for r in unate_not_rof}
    models_with_unr_frac = len(models_with_unr) / n_models

    # Models whose non-ROF functions are all still unate
    # (i.e. every non-ROF is unate-not-rof; no binate among non-ROFs)
    non_rof_by_model: Dict[object, List[bool]] = defaultdict(list)
    for r in bbm_rows:
        if not r["is_rof"]:
            non_rof_by_model[r["model_id"]].append(bool(r["is_unate"]))
    models_with_non_rof = len(non_rof_by_model)
    n_non_rof_all_unate = sum(
        1 for flags in non_rof_by_model.values() if all(flags)
    )
    non_rof_all_unate_frac = (
        n_non_rof_all_unate / models_with_non_rof
        if models_with_non_rof
        else None
    )

    log.info("")
    log.info("=== not syntactic ROF but syntactic unate ===")
    log.info(
        f"Functions that are syntactic unate but not ROF: "
        f"{n_unr:,} / {n_fn:,} ({unr_rate:.2%})"
    )
    if large_unr_rate is not None:
        log.info(
            f"Among >{LARGE_K}-unique functions: "
            f"{len(large_unr):,} / {len(large):,} ({large_unr_rate:.2%})"
        )
    log.info(
        f"Models with ≥1 unate-not-ROF function: "
        f"{len(models_with_unr):,} / {n_models:,} ({models_with_unr_frac:.2%})"
    )
    if non_rof_all_unate_frac is not None:
        log.info(
            f"Among models with ≥1 non-ROF, fraction where all non-ROFs "
            f"are still unate: {n_non_rof_all_unate:,} / "
            f"{models_with_non_rof:,} ({non_rof_all_unate_frac:.2%})"
        )

    if unate_not_rof:
        log.info("")
        log.info(
            "Extra variable occurrences among unate-not-ROF functions "
            "(n_occ - n_unique):"
        )
        extras = [
            int(r["n_variables"]) - int(r["n_unique_variables"])  # type: ignore[arg-type]
            for r in unate_not_rof
        ]
        total = len(extras)
        for n in range(1, 21):
            count = sum(1 for e in extras if e == n)
            log.info(
                f"  {'+' + str(n):>6}: {count / total:.2%} "
                f"({count:,} / {total:,})"
            )
        rest = sum(1 for e in extras if e >= 21)
        log.info(f"  {'rest':>6}: {rest / total:.2%} ({rest:,} / {total:,})")

    # Three-way split reminder
    n_rof = sum(1 for r in bbm_rows if r["is_rof"])
    n_unate = sum(1 for r in bbm_rows if r["is_unate"])
    n_binate = n_fn - n_unate
    log.info("")
    log.info("Three-way split (ROF ⊆ unate ⊆ all):")
    log.info(f"  syntactic ROF:           {n_rof:,} / {n_fn:,} ({n_rof / n_fn:.2%})")
    log.info(
        f"  unate \\ ROF:              {n_unr:,} / {n_fn:,} ({unr_rate:.2%})"
    )
    log.info(
        f"  binate (not unate):       {n_binate:,} / {n_fn:,} "
        f"({n_binate / n_fn:.2%})"
    )

# %% [markdown]
# ## Summary


# %%
log.info("")
log.info(
    f"synthetic rows={len(synth_rows)}  bbm rows={len(bbm_rows)}  "
    f"bbm_timeout={bbm_timed_out}  bbm_failed={bbm_failed}  "
    f"|  SOURCE={SOURCE}"
)
log.info(f"wrote {LOG_PATH}")
log.close()
