# %% [markdown]
# Expression → LUT → one-symbol timing vs *k*
#
# Benchmark the three CANA stages for Boolean functions given in **bnet**
# form (``target, expr`` with ``&`` / ``|`` / ``!``):
#
# 1. **import** — parse into positional ``FunctionOutputs.expression``
#    (no dense LUT yet; same path as ``parse_bnet`` / ``expressions_to_nodes``)
# 2. **build LUT** — ``build_bit_lut_from_expression()`` (``2**k`` evals)
# 3. **one-symbol** — Quine–McCluskey prime implicants via
#    ``compute_one_symbol_schemata_from_bit_lut()``
#
# Sources (toggle below):
#
# - **synthetic** — same ROFs as the deprecated ``AST.ipynb`` / ``AST_2.ipynb``
#   generators (``and``, ``or``, ``pairwise_*``, ``random``), wrapped as
#   single-rule ``.bnet`` strings
# - **bbm** — ``REPEATS`` real rules per *k* drawn from
#   ``BBM/models/*/model.bnet`` (reported separately, then averaged)
#
# Shared helpers live in ``high_k.synthetic``, ``high_k.bbm_rules``,
# ``high_k.timing``.  Keep *k* modest: dense LUT + QM grow as ``2**k``.
# Default ``K_VALUES = [4, 6, 8, 10, 12]``.
#
# Output is teed to ``high_k/logs/expr_lut_one_symbol_timing/``.

# %%
from __future__ import annotations

import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ``dev/`` on path so ``import high_k…`` works when run as a script / # %% notebook
# notebooks/ → high_k/ → dev/
_DEV_DIR = Path(__file__).resolve().parents[2]
if str(_DEV_DIR) not in sys.path:
    sys.path.insert(0, str(_DEV_DIR))

from high_k.bbm_rules import BBM_MODELS, as_bnet, bbm_rules_for_k
from high_k.run_log import start_run_log
from high_k.synthetic import EVEN_K_BUILDERS, SYNTHETIC_BUILDERS
from high_k.timing import fmt_s, time_expr_lut_one_symbol

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# "synthetic" | "bbm" | "both"
SOURCE = "both"

# Dense LUT + QM: stay well below the AST notebooks' k=64+ regime.
K_VALUES = [4, 6, 8, 10, 12]

REPEATS = 3
RANDOM_SEED = 12345

# Skip one-symbol (QM) when 2**k exceeds this (LUT still timed).
MAX_QM_ROWS = 2**12  # k <= 12

log, LOG_PATH = start_run_log("expr_lut_one_symbol_timing")
log.info(f"SOURCE={SOURCE}  K_VALUES={K_VALUES}  REPEATS={REPEATS}")
log.info(f"MAX_QM_ROWS={MAX_QM_ROWS}  BBM_MODELS={BBM_MODELS}")

# %% [markdown]
# ## Run: synthetic functions


# %%
rows: List[Dict[str, object]] = []

if SOURCE in ("synthetic", "both"):
    log.info("=== synthetic (AST-style ROFs as one-rule bnet) ===")
    log.info(
        f"{'fn':18s} {'k':>3s}  {'import':>10s}  {'lut':>10s}  "
        f"{'one-sym':>10s}  {'#PI on':>6s}  {'#PI off':>7s}"
    )
    for fname, builder in SYNTHETIC_BUILDERS.items():
        for k in K_VALUES:
            if fname in EVEN_K_BUILDERS and k % 2 != 0:
                continue
            if fname == "random":
                expr = builder(k, seed=RANDOM_SEED + k)
            else:
                expr = builder(k)
            bnet = as_bnet(expr, target="Y")
            do_qm = (2**k) <= MAX_QM_ROWS
            try:
                r = time_expr_lut_one_symbol(
                    bnet, "Y", repeats=REPEATS, do_one_symbol=do_qm
                )
            except Exception as exc:  # noqa: BLE001 — report and continue
                log.info(f"{fname:18s} {k:3d}  FAILED: {exc}")
                continue
            r["source"] = "synthetic"
            r["label"] = fname
            rows.append(r)
            log.info(
                f"{fname:18s} {k:3d}  {fmt_s(r['t_import']):>10s}  "
                f"{fmt_s(r['t_lut']):>10s}  {fmt_s(r['t_one_symbol']):>10s}  "
                f"{str(r['n_pi_on']):>6s}  {str(r['n_pi_off']):>7s}"
            )

# %% [markdown]
# ## Run: BBM rules (``REPEATS`` per *k*)


# %%
if SOURCE in ("bbm", "both"):
    log.info("")
    log.info(f"=== BBM ({REPEATS} matching rules per k; each reported) ===")
    if not BBM_MODELS.is_dir():
        log.info(f"BBM models not found at {BBM_MODELS}")
    else:
        log.info(
            f"{'k':>3s} {'rep':>3s}  {'import':>10s}  {'lut':>10s}  "
            f"{'one-sym':>10s}  {'#PI on':>6s}  {'#PI off':>7s}  "
            f"target / model"
        )
        for k in K_VALUES:
            hits = bbm_rules_for_k(BBM_MODELS, k, n=REPEATS)
            if not hits:
                log.info(f"{k:3d}  (no rule with k={k})")
                continue
            if len(hits) < REPEATS:
                log.info(
                    f"{k:3d}  (only {len(hits)}/{REPEATS} rules with k={k})"
                )
            do_qm = (2**k) <= MAX_QM_ROWS
            for rep, (model_name, target, _expr, bnet) in enumerate(hits, start=1):
                try:
                    # Diversity comes from distinct rules; time each once.
                    r = time_expr_lut_one_symbol(
                        bnet, target, repeats=1, do_one_symbol=do_qm
                    )
                except Exception as exc:  # noqa: BLE001
                    log.info(f"{k:3d} {rep:3d}  FAILED on {target}: {exc}")
                    continue
                r["source"] = "bbm"
                r["label"] = f"{model_name}::{target}"
                r["rep"] = rep
                rows.append(r)
                short_model = model_name.split("__")[-1][:28]
                log.info(
                    f"{k:3d} {rep:3d}  {fmt_s(r['t_import']):>10s}  "
                    f"{fmt_s(r['t_lut']):>10s}  "
                    f"{fmt_s(r['t_one_symbol']):>10s}  "
                    f"{str(r['n_pi_on']):>6s}  {str(r['n_pi_off']):>7s}  "
                    f"{target[:40]}  [{short_model}]"
                )

# %% [markdown]
# ## Summary table (averages over repeats)


# %%
def _mean_or_none(vals: List[object]) -> Optional[float]:
    nums = [float(v) for v in vals if isinstance(v, (int, float))]
    if not nums:
        return None
    return float(statistics.mean(nums))


log.info("")
log.info("=== collected rows (averages) ===")
log.info(
    f"{'source':10s} {'label':40s} {'k':>3s} {'n':>3s}  "
    f"{'t_import':>10s} {'t_lut':>10s} {'t_one':>10s}"
)

# Synthetic: one row per (label, k) already (internal median).
# BBM: collapse the REPEATS distinct rules per k into a mean.
grouped: Dict[Tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)
for r in rows:
    if r["source"] == "bbm":
        key: Tuple[object, ...] = ("bbm", r["k"])
    else:
        key = (r["source"], r["label"], r["k"])
    grouped[key].append(r)

n_summary = 0
for group in grouped.values():
    source = str(group[0]["source"])
    k = int(group[0]["k"])  # type: ignore[arg-type]
    if source == "bbm":
        label = f"avg of {len(group)} rules"
    else:
        label = str(group[0]["label"])
    t_import = _mean_or_none([g["t_import"] for g in group])
    t_lut = _mean_or_none([g["t_lut"] for g in group])
    t_one = _mean_or_none([g["t_one_symbol"] for g in group])
    log.info(
        f"{source:10s} {label[:40]:40s} {k:3d} {len(group):3d}  "
        f"{fmt_s(t_import):>10s} {fmt_s(t_lut):>10s} "
        f"{fmt_s(t_one):>10s}"
    )
    n_summary += 1

log.info(
    f"\n{n_summary} averaged rows from {len(rows)} timings  |  "
    f"K_VALUES={K_VALUES}  |  MAX_QM_ROWS={MAX_QM_ROWS}"
)
log.info(f"wrote {LOG_PATH}")
log.close()
