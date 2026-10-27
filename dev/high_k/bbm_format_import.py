# %% [markdown]
# BBM multi-format import / conversion
#
# Load one Boolean model from the BBM corpus in all four expression formats
# (``.booleannet.txt``, ``.bnet``, ``.aeon``, ``.sbml``), check that they share
# the same name-keyed LUTs (CANA ``NetworkDict`` hub), then exercise any↔any
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

# %%
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

_DEV_DIR = Path(__file__).resolve().parent.parent
_PKG_ROOT = _DEV_DIR.parent  # .../CANA/CANA (contains the cana package)
for _p in (_DEV_DIR, _PKG_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cana.boolean_network import BooleanNetwork
from cana.network_dict import equivalent_network_dicts, name_keyed_luts

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_ID = "023"  # any BBM summary ID, e.g. "001", "023", "100"

# .../CANA/CANA/dev → parents[1] = workspace root (.../CANA) containing BBM/
BBM_MODELS = _DEV_DIR.resolve().parents[1] / "BBM" / "models"

FORMATS: Tuple[str, ...] = ("booleannet", "bnet", "aeon", "sbml")
FILE_NAMES = {
    "booleannet": "model.booleannet.txt",
    "bnet": "model.bnet",
    "aeon": "model.aeon",
    "sbml": "model.sbml",
}

# %% [markdown]
# ## Resolve model directory and load all formats


# %%
def resolve_model_dir(models_root: Path, model_id: str) -> Path:
    """Match ``[id-{ID}]__*`` under *models_root* (ID zero-padded to 3).

    Uses a string prefix (not ``Path.glob``) because ``[`` / ``]`` are
    character-class metacharacters in glob patterns.
    """
    pad = f"{int(model_id):03d}"
    prefix = f"[id-{pad}]__"
    matches = sorted(
        p for p in models_root.iterdir() if p.is_dir() and p.name.startswith(prefix)
    )
    if not matches:
        raise FileNotFoundError(
            f"No BBM model for id={model_id!r} under {models_root}"
        )
    return matches[0]


model_dir = resolve_model_dir(BBM_MODELS, MODEL_ID)
print(f"MODEL_ID={MODEL_ID}")
print(f"model_dir={model_dir}")

networks: Dict[str, BooleanNetwork] = {}
for fmt in FORMATS:
    path = model_dir / FILE_NAMES[fmt]
    N = BooleanNetwork.from_file(str(path))  # type inferred from suffix
    networks[fmt] = N
    print(f"  loaded {fmt:12s}  N={N.Nnodes}  names={sorted(n.name for n in N.nodes)}")

# %% [markdown]
# ## What ``network_dict`` looks like
#
# Each network stores a dense dict
# ``network_dict[i] = {"name", "inputs", "outputs"}`` where ``outputs`` is a
# ``FunctionOutputs`` (usually with a dense ``bit_lut`` of length ``2**k``)
# for input node ids ``inputs``.
# Formats may permute node ids; equivalence is checked by **node name**.


# %%
ref = networks["booleannet"]
print("Example network_dict entry (first node of booleannet load):")
i0 = 0
entry = ref.network_dict[i0]
print(
    f"  network_dict[{i0}] = {{"
    f"'name': {entry['name']!r}, "
    f"'inputs': {entry['inputs']}, "
    f"'outputs': {entry['outputs']}"
    f"}}"
)
print(
    "  (inputs are node ids; LUT row s uses inputs[0] as MSB — "
    "see BooleanNetwork docstring)"
)

# %% [markdown]
# ## Pairwise equivalence (name-keyed LUTs)


# %%
print("Pairwise equivalent_network_dicts vs booleannet:")
all_ok = True
for fmt, N in networks.items():
    ok = equivalent_network_dicts(ref, N)
    all_ok = all_ok and ok
    print(f"  {fmt:12s}  {ok}")
assert all_ok, "Formats disagree on name-keyed LUTs"

# Per-node LUT agreement table (canonical sorted inputs)
ref_luts = name_keyed_luts(ref)
print("\nPer-node LUT check (inputs sorted by name):")
print(f"{'node':12s}  {'k':>3s}  {'ok':>3s}  inputs")
for name in sorted(ref_luts):
    inputs, lut = ref_luts[name]
    oks = [
        name_keyed_luts(networks[fmt]).get(name) == ref_luts[name] for fmt in FORMATS
    ]
    print(
        f"{name:12s}  {len(inputs):3d}  {str(all(oks)):>3s}  {list(inputs)}"
    )

# %% [markdown]
# ## Any → any conversion (``src → string(dst) → network``)


# %%
print("Conversion matrix (row=src, col=dst); '.' = equivalent to ref:")
header = " ".join(f"{d:>12s}" for d in FORMATS)
print(f"{'':12s} {header}")
for src in FORMATS:
    cells: List[str] = []
    for dst in FORMATS:
        text = networks[src].to_string(type=dst)
        N2 = BooleanNetwork.from_string(text, type=dst)
        cells.append(f"{'.':>12s}" if ref.equivalent_to(N2) else f"{'FAIL':>12s}")
        assert ref.equivalent_to(N2), (src, dst)
    print(f"{src:12s} " + " ".join(cells))

# %% [markdown]
# ## Full round-trips ``src → dst → src``


# %%
print("Round-trips src → dst → src:")
for src in FORMATS:
    for dst in FORMATS:
        mid = networks[src].to_string(type=dst)
        back = BooleanNetwork.convert_string(mid, dst, src)
        N2 = BooleanNetwork.from_string(back, type=src)
        ok = ref.equivalent_to(N2)
        print(f"  {src:12s} → {dst:12s} → {src:12s}  {ok}")
        assert ok, (src, dst)

# %% [markdown]
# ## ``.cnet`` dense LUT export and round-trips
#
# BBM does not ship ``.cnet``; we synthesize dense (``bit_lut``) covers from
# each loaded expression format, re-import, and round-trip
# ``src → cnet → src``.


# %%
print("cnet dense LUT: src → cnet → network (~= ref):")
cnet_lut_texts: Dict[str, str] = {}
for src in FORMATS:
    cnet = networks[src].to_cnet()  # one_symbol_schemata=False (default)
    cnet_lut_texts[src] = cnet
    N2 = BooleanNetwork.from_string(cnet, type="cnet")
    ok = ref.equivalent_to(N2)
    print(f"  {src:12s} → cnet (lut)               {ok}")
    assert ok, ("cnet lut import", src)

print("\ncnet dense LUT round-trips src → cnet → src:")
for src in FORMATS:
    mid = cnet_lut_texts[src]
    back = BooleanNetwork.convert_string(mid, "cnet", src)
    N2 = BooleanNetwork.from_string(back, type=src)
    ok = ref.equivalent_to(N2)
    print(f"  {src:12s} → cnet (lut) → {src:12s}  {ok}")
    assert ok, ("cnet lut round-trip", src)

# %% [markdown]
# ## ``.cnet`` one-symbol-schemata export and round-trips
#
# Fill ``one_symbol_schemata`` on each loaded network, export with
# ``one_symbol_schemata=True`` (wildcard ``2`` covers), re-import, and
# round-trip ``src → cnet(schemata) → src``. Expression exporters resolve
# LUTs via ``get_bit_lut()``, so schemata-only re-imports still convert.


# %%
print("cnet one-symbol schemata: src → fill → cnet → network (~= ref):")
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
    print(
        f"  {src:12s} → cnet (schemata)  wildcard={has_wildcard}  {ok}"
    )
    assert ok, ("cnet schemata import", src)

print("\ncnet one-symbol schemata round-trips src → cnet → src:")
for src in FORMATS:
    mid = cnet_schema_texts[src]
    back = BooleanNetwork.convert_string(mid, "cnet", src)
    N2 = BooleanNetwork.from_string(back, type=src)
    ok = ref.equivalent_to(N2)
    print(f"  {src:12s} → cnet (schemata) → {src:12s}  {ok}")
    assert ok, ("cnet schemata round-trip", src)

# Sample schemata cover for the first non-trivial node (k>=2)
print("\nSample one-symbol schemata cnet lines (first k>=2 node):")
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
            print(f"  {line}")
            i += 1
            while i < len(sample_lines) and sample_lines[i] and not sample_lines[i].startswith("."):
                print(f"  {sample_lines[i]}")
                i += 1
            printed = True
            break
    i += 1
if not printed:
    print("  (no k>=2 node — model may be all identity/constants)")

print("\nAll format loads, conversions, and round-trips passed (incl. cnet lut + schemata).")
