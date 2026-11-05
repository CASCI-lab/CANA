# %% [markdown]
# BBM multi-format import / conversion
#
# Load one Boolean model from the BBM corpus in all four expression formats
# (``.booleannet.txt``, ``.bnet``, ``.aeon``, ``.sbml``), check that they share
# the same name-keyed LUTs (CANA ``list[BooleanNode]`` hub), then exercise any↔any
# conversion and full round-trips.
#
# Also export to ``.cnet`` (BNS / BLIF-like) in both representations:
#
# - **dense LUT** — ``2**k`` rows per node (default ``to_cnet()`` / ``to_string``)
# - **one-symbol schemata** — wildcard ``2`` covers after
#   ``fill_function_outputs("one_symbol_schemata")``
#
# and round-trip each mode back through the expression formats.
#
# Free inputs (e.g. ``v_CycD`` in model 023) appear without an update rule in
# ``.bnet`` / ``.aeon``; CANA assigns the **identity** ``f(x)=x`` so they match
# BBM's ``model.booleannet.txt`` representation.
#
# Edit ``MODEL_ID`` below to switch models.
#
# Output is teed to ``high_k/logs/bbm_format_import/``.

# %%
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

# notebooks/ → high_k/ → dev/
_DEV_DIR = Path(__file__).resolve().parents[2]
if str(_DEV_DIR) not in sys.path:
    sys.path.insert(0, str(_DEV_DIR))

from high_k.bbm_rules import BBM_MODELS, resolve_model_dir
from high_k.paths import ensure_cana_on_path
from high_k.run_log import start_run_log

ensure_cana_on_path()

from cana.boolean_network import BooleanNetwork
from cana.nodes_builder import equivalent_networks, name_keyed_luts

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_ID = "023"  # any BBM summary ID, e.g. "001", "023", "100"

FORMATS: Tuple[str, ...] = ("booleannet", "bnet", "aeon", "sbml")
FILE_NAMES = {
    "booleannet": "model.booleannet.txt",
    "bnet": "model.bnet",
    "aeon": "model.aeon",
    "sbml": "model.sbml",
}

log, LOG_PATH = start_run_log("bbm_format_import")
log.info(f"MODEL_ID={MODEL_ID}")

# %% [markdown]
# ## Resolve model directory and load all formats


# %%
model_dir = resolve_model_dir(BBM_MODELS, MODEL_ID)
log.info(f"model_dir={model_dir}")

networks: Dict[str, BooleanNetwork] = {}
for fmt in FORMATS:
    path = model_dir / FILE_NAMES[fmt]
    N = BooleanNetwork.from_file(str(path))  # type inferred from suffix
    networks[fmt] = N
    log.info(
        f"  loaded {fmt:12s}  N={N.Nnodes}  names={sorted(n.name for n in N.nodes)}"
    )

# %% [markdown]
# ## What ``nodes`` looks like
#
# Each network stores a dense ``list[BooleanNode]`` with invariant
# ``nodes[i].id == i``.  Each node has ``name``, ``inputs`` (node ids), and
# ``outputs`` (:class:`FunctionOutputs`, usually with a dense ``bit_lut`` of
# length ``2**k``).  Formats may permute node ids; equivalence is checked by
# **node name**.


# %%
ref = networks["booleannet"]
log.info("Example node entry (first node of booleannet load):")
i0 = 0
node = ref.nodes[i0]
log.info(
    f"  nodes[{i0}] = BooleanNode("
    f"id={node.id}, name={node.name!r}, "
    f"inputs={node.inputs}, "
    f"outputs={node.outputs})"
)
log.info(
    "  (inputs are node ids; LUT row s uses inputs[0] as MSB — "
    "see BooleanNetwork docstring)"
)

# %% [markdown]
# ## Pairwise equivalence (name-keyed LUTs)


# %%
log.info("Pairwise equivalent_networks vs booleannet:")
all_ok = True
for fmt, N in networks.items():
    ok = equivalent_networks(ref, N)
    all_ok = all_ok and ok
    log.info(f"  {fmt:12s}  {ok}")
assert all_ok, "Formats disagree on name-keyed LUTs"

# Per-node LUT agreement table (canonical sorted inputs)
ref_luts = name_keyed_luts(ref)
log.info("")
log.info("Per-node LUT check (inputs sorted by name):")
log.info(f"{'node':12s}  {'k':>3s}  {'ok':>3s}  inputs")
for name in sorted(ref_luts):
    inputs, lut = ref_luts[name]
    oks = [
        name_keyed_luts(networks[fmt]).get(name) == ref_luts[name] for fmt in FORMATS
    ]
    log.info(
        f"{name:12s}  {len(inputs):3d}  {str(all(oks)):>3s}  {list(inputs)}"
    )

# %% [markdown]
# ## Any → any conversion (``src → string(dst) → network``)


# %%
log.info("Conversion matrix (row=src, col=dst); '.' = equivalent to ref:")
header = " ".join(f"{d:>12s}" for d in FORMATS)
log.info(f"{'':12s} {header}")
for src in FORMATS:
    cells: List[str] = []
    for dst in FORMATS:
        text = networks[src].to_string(type=dst)
        N2 = BooleanNetwork.from_string(text, type=dst)
        cells.append(f"{'.':>12s}" if ref.equivalent_to(N2) else f"{'FAIL':>12s}")
        assert ref.equivalent_to(N2), (src, dst)
    log.info(f"{src:12s} " + " ".join(cells))

# %% [markdown]
# ## Full round-trips ``src → dst → src``


# %%
log.info("Round-trips src → dst → src:")
for src in FORMATS:
    for dst in FORMATS:
        mid = networks[src].to_string(type=dst)
        back = BooleanNetwork.convert_string(mid, dst, src)
        N2 = BooleanNetwork.from_string(back, type=src)
        ok = ref.equivalent_to(N2)
        log.info(f"  {src:12s} → {dst:12s} → {src:12s}  {ok}")
        assert ok, (src, dst)

# %% [markdown]
# ## ``.cnet`` dense LUT export and round-trips
#
# BBM does not ship ``.cnet``; we synthesize dense (``bit_lut``) covers from
# each loaded expression format, re-import, and round-trip
# ``src → cnet → src``.


# %%
log.info("cnet dense LUT: src → cnet → network (~= ref):")
cnet_lut_texts: Dict[str, str] = {}
for src in FORMATS:
    cnet = networks[src].to_cnet()  # one_symbol_schemata=False (default)
    cnet_lut_texts[src] = cnet
    N2 = BooleanNetwork.from_string(cnet, type="cnet")
    ok = ref.equivalent_to(N2)
    log.info(f"  {src:12s} → cnet (lut)               {ok}")
    assert ok, ("cnet lut import", src)

log.info("")
log.info("cnet dense LUT round-trips src → cnet → src:")
for src in FORMATS:
    mid = cnet_lut_texts[src]
    back = BooleanNetwork.convert_string(mid, "cnet", src)
    N2 = BooleanNetwork.from_string(back, type=src)
    ok = ref.equivalent_to(N2)
    log.info(f"  {src:12s} → cnet (lut) → {src:12s}  {ok}")
    assert ok, ("cnet lut round-trip", src)

# %% [markdown]
# ## ``.cnet`` one-symbol-schemata export and round-trips
#
# Fill ``one_symbol_schemata`` on each loaded network, export with
# ``one_symbol_schemata=True`` (wildcard ``2`` covers), re-import, and
# round-trip ``src → cnet(schemata) → src``. Expression exporters resolve
# LUTs via ``get_bit_lut()``, so schemata-only re-imports still convert.


# %%
log.info("cnet one-symbol schemata: src → fill → cnet → network (~= ref):")
cnet_schema_texts: Dict[str, str] = {}
for src in FORMATS:
    networks[src].fill_function_outputs("one_symbol_schemata")
    cnet = networks[src].to_cnet(one_symbol_schemata=True)
    cnet_schema_texts[src] = cnet
    N2 = BooleanNetwork.from_string(cnet, type="cnet")
    ok = ref.equivalent_to(N2)
    has_wildcard = any(
        "2" in line.split()[0]
        for line in cnet.splitlines()
        if line and line[0] not in ".#" and " " in line
    )
    log.info(
        f"  {src:12s} → cnet (schemata)  wildcard={has_wildcard}  {ok}"
    )
    assert ok, ("cnet schemata import", src)

log.info("")
log.info("cnet one-symbol schemata round-trips src → cnet → src:")
for src in FORMATS:
    mid = cnet_schema_texts[src]
    # Schemata-only import has one_symbol_is_prime_implicant=None; soft-warn is expected.
    back = BooleanNetwork.convert_string(mid, "cnet", src, warn=False)
    N2 = BooleanNetwork.from_string(back, type=src)
    ok = ref.equivalent_to(N2)
    log.info(f"  {src:12s} → cnet (schemata) → {src:12s}  {ok}")
    assert ok, ("cnet schemata round-trip", src)

# Sample schemata cover for the first non-trivial node (k>=2)
log.info("")
log.info("Sample one-symbol schemata cnet lines (first k>=2 node):")
sample = cnet_schema_texts["booleannet"]
sample_lines = sample.splitlines()
printed = False
i = 0
while i < len(sample_lines):
    line = sample_lines[i]
    if line.startswith(".n "):
        parts = line.split()
        k = int(parts[2]) if len(parts) >= 3 else 0
        if k >= 2:
            log.info(f"  {line}")
            i += 1
            while i < len(sample_lines) and sample_lines[i] and not sample_lines[i].startswith("."):
                log.info(f"  {sample_lines[i]}")
                i += 1
            printed = True
            break
    i += 1
if not printed:
    log.info("  (no k>=2 node — model may be all identity/constants)")

log.info("")
log.info(
    "All format loads, conversions, and round-trips passed "
    "(incl. cnet lut + schemata)."
)
log.info(f"wrote {LOG_PATH}")
log.close()
