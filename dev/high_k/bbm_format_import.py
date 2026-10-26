# %% [markdown]
# BBM multi-format import / conversion
#
# Load one Boolean model from the BBM corpus in all four expression formats
# (``.booleannet.txt``, ``.bnet``, ``.aeon``, ``.sbml``), check that they share
# the same name-keyed LUTs (CANA ``logic`` hub), then exercise any↔any
# conversion and full round-trips.
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
from cana.io.formats import equivalent_logic, name_keyed_luts

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
# ## What ``logic`` looks like
#
# Each network stores a dense dict ``logic[i] = {"name", "in", "out"}`` where
# ``out`` is the Boolean LUT of length ``2**k`` for input node ids ``in``.
# Formats may permute node ids; equivalence is checked by **node name**.


# %%
ref = networks["booleannet"]
print("Example logic entry (first node of booleannet load):")
i0 = 0
entry = ref.logic[i0]
print(
    f"  logic[{i0}] = {{"
    f"'name': {entry['name']!r}, "
    f"'in': {entry['in']}, "
    f"'out': {entry['out']}"
    f"}}"
)
print(
    "  (inputs are node ids; LUT row s uses in[0] as MSB — "
    "see BooleanNetwork docstring)"
)

# %% [markdown]
# ## Pairwise equivalence (name-keyed LUTs)


# %%
print("Pairwise equivalent_logic vs booleannet:")
all_ok = True
for fmt, N in networks.items():
    ok = equivalent_logic(ref, N)
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

print("\nAll format loads, conversions, and round-trips passed.")
