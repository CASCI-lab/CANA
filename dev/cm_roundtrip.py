# %% [markdown]
# Canalizing Map ↔ LUT roundtrip
#
# Walkthrough:
# 1. Build a Boolean node from a LUT or boolean rules.
# 2. Convert to a Canalizing Map (CM) via the legacy builder
#    (`BooleanNode.canalizing_map` / `build_canalizing_map_legacy`).
# 3. Reconstruct a LUT from that CM via `cm_lut.lut_from_canalizing_map`.
# 4. Compare original vs reconstructed LUTs.
# 5. Also build hand / nested-threshold CMs and (attempt to) draw all three.
#
# The reverse step evaluates the CM modularly from node kinds + topology
# (see `cana.canalization.cm_lut` module docstring).

# %%
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import networkx as nx
from cana.boolean_network import BooleanNetwork
from cana.boolean_node import BooleanNode
from cana.canalization.canalizing_map import build_canalizing_map_legacy
from cana.canalization.cm_lut import (
    ROLE_POLARITY0,
    ROLE_POLARITY1,
    ROLE_SCHEMA,
    ROLE_SYMMETRY,
    connect,
    input_ids_in_order,
    lut_from_canalizing_map,
    make_fusion,
    make_input_node,
    make_output_node,
    make_threshold,
)
try:
    from cana.drawing.canalizing_map import draw_canalizing_map_graphviz
except Exception as _draw_import_err:  # missing graphviz, or drawer import bug
    draw_canalizing_map_graphviz = None
    _DRAW_IMPORT_ERROR = _draw_import_err
else:
    _DRAW_IMPORT_ERROR = None


def summarize_roundtrip(name, original_outputs, result):
    orig = "".join(str(x) for x in original_outputs)
    recon = result.lut_string()
    exact = (
        not result.has_conflict
        and not result.is_incomplete
        and [int(x) for x in original_outputs] == result.outputs
    )
    print(f"=== {name} ===")
    print(f"  inputs     : {result.input_names}")
    print(f"  original   : {orig}")
    print(f"  from CM    : {recon}")
    print(f"  conflict   : {result.has_conflict}  states={result.conflict_states}")
    print(f"  incomplete : {result.is_incomplete}  uncovered={result.uncovered_states[:8]}"
          f"{'...' if len(result.uncovered_states) > 8 else ''}")
    print(f"  exact match: {exact}")
    return exact


TRUE_CTNNB1 = "00001110111011100000000000000000"


# %% [markdown]
# ## Case 1 — AND / OR from LUT
#
# Simple canalizing functions.  The current forward CM builder usually gets
# these right, so the roundtrip can succeed.

# %%
and_node = BooleanNode.from_output_list(
    [0, 0, 0, 1], name="AND", inputs=[0, 1], id=2
)
and_cm = and_node.canalizing_map(builder="legacy")
and_result = lut_from_canalizing_map(and_cm, input_order=and_node.inputs)
summarize_roundtrip("AND", and_node.outputs, and_result)

or_node = BooleanNode.from_output_list(
    [0, 1, 1, 1], name="OR", inputs=[10, 11], id=5
)
or_cm = or_node.canalizing_map(builder="legacy")
or_result = lut_from_canalizing_map(or_cm, input_order=or_node.inputs)
summarize_roundtrip("OR", or_node.outputs, or_result)

# %% [markdown]
# ## Case 2 — CTNNB1 via legacy canalizing map builder
#
# ```
# CTNNB1 = (not SRY) and (WNT4 or RSPO1) and ((not AMH) or (not SOX9))
# ```
#
# Multi-group permutation schemata round-trip with the fibre-aware legacy
# ``build_canalizing_map_legacy`` / ``BooleanNode.canalizing_map(builder='legacy')``.

# %%
CTNNB1_BNET = (
    "CTNNB1*=(not SRY) and (WNT4 or RSPO1) and ((not AMH) or (not SOX9))\n"
    "SRY*=SRY\n"
    "WNT4*=WNT4\n"
    "RSPO1*=RSPO1\n"
    "AMH*=AMH\n"
    "SOX9*=SOX9\n"
)

N = BooleanNetwork.from_string_boolean(CTNNB1_BNET)
ct = next(n for n in N.nodes if n.name == "CTNNB1")
input_labels = [N.nodes[i].name for i in ct.inputs]
print("CTNNB1 inputs (LUT order):", list(zip(ct.inputs, input_labels)))
print("True LUT:", "".join(ct.outputs))

ct._check_compute_canalization_variables(two_symbols=True)
print("Two-symbol schemata (out=0):")
for row in ct._two_symbols[0]:
    print(" ", row)
print("Two-symbol schemata (out=1):")
for row in ct._two_symbols[1]:
    print(" ", row)

# Explicit legacy builder (same graph as ct.canalizing_map(builder="legacy")).
legacy_CTNNB1 = build_canalizing_map_legacy(
    ct._two_symbols,
    node_id=ct.id,
    inputs=ct.inputs,
    name=ct.name,
    constant=ct.constant,
    input_labels={i: N.nodes[i].name for i in ct.inputs},
    k=ct.k,
)
legacy_result = lut_from_canalizing_map(legacy_CTNNB1, input_order=ct.inputs)
exact = summarize_roundtrip("legacy_CTNNB1", ct.outputs, legacy_result)
assert exact, "Expected legacy CTNNB1 roundtrip to succeed"

# %% [markdown]
# ## Case 3 — Hand-correct CM (`hand_CTNNB1`)
#
# A correct CM for the full CTNNB1 function using fusion junctions for the
# OR-like fibres.  Reverse reconstruction recovers the LUT exactly — showing
# the interpreter itself is sound when the graph is right.

# %%
# Label-driven helpers: ids / tids / edge types are assigned automatically.

hand_CTNNB1 = nx.DiGraph(name="hand_CTNNB1")
out0 = make_output_node(hand_CTNNB1, "CTNNB1", 0)
out1 = make_output_node(hand_CTNNB1, "CTNNB1", 1)

thr0 = make_threshold(hand_CTNNB1, tau=1)
thr1 = make_threshold(hand_CTNNB1, tau=2)
thr2 = make_threshold(hand_CTNNB1, tau=2)
connect(hand_CTNNB1, thr0, out0)
connect(hand_CTNNB1, thr1, out0)
connect(hand_CTNNB1, thr2, out0)
connect(hand_CTNNB1, make_input_node(hand_CTNNB1, "SRY", 1), thr0)
connect(hand_CTNNB1, make_input_node(hand_CTNNB1, "WNT4", 0), thr1)
connect(hand_CTNNB1, make_input_node(hand_CTNNB1, "RSPO1", 0), thr1)
connect(hand_CTNNB1, make_input_node(hand_CTNNB1, "AMH", 1), thr2)
connect(hand_CTNNB1, make_input_node(hand_CTNNB1, "SOX9", 1), thr2)

thr3 = make_threshold(hand_CTNNB1, tau=3)
connect(hand_CTNNB1, thr3, out1)
fus0 = make_fusion(hand_CTNNB1)
fus1 = make_fusion(hand_CTNNB1)
connect(hand_CTNNB1, fus0, thr3)
connect(hand_CTNNB1, fus1, thr3)
connect(hand_CTNNB1, make_input_node(hand_CTNNB1, "SRY", 0), thr3)
connect(hand_CTNNB1, make_input_node(hand_CTNNB1, "WNT4", 1), fus0)
connect(hand_CTNNB1, make_input_node(hand_CTNNB1, "RSPO1", 1), fus0)
connect(hand_CTNNB1, make_input_node(hand_CTNNB1, "AMH", 0), fus1)
connect(hand_CTNNB1, make_input_node(hand_CTNNB1, "SOX9", 0), fus1)

hand_result = lut_from_canalizing_map(
    hand_CTNNB1, input_order=input_ids_in_order(hand_CTNNB1)
)
summarize_roundtrip("hand_CTNNB1", TRUE_CTNNB1, hand_result)
assert hand_result.outputs == [int(bit) for bit in TRUE_CTNNB1]
assert not hand_result.has_conflict and not hand_result.is_incomplete

# %% [markdown]
# ## Case 4 — Canonical nested-threshold CM (`canonical_CTNNB1`)
#
# Same CTNNB1 function with nested schema / symmetry / polarity thresholds.
# Build order is free; thresholds do not store an ``out`` attribute.

# %%
canonical_CTNNB1 = nx.DiGraph(name="canonical_CTNNB1")
out0 = make_output_node(canonical_CTNNB1, "CTNNB1", 0)
out1 = make_output_node(canonical_CTNNB1, "CTNNB1", 1)

# out=0: three independent two-symbol schemata
out0_two0 = make_threshold(canonical_CTNNB1, tau=1, role=ROLE_SCHEMA)
out0_two1 = make_threshold(canonical_CTNNB1, tau=2, role=ROLE_SCHEMA)
out0_two2 = make_threshold(canonical_CTNNB1, tau=2, role=ROLE_SCHEMA)
connect(canonical_CTNNB1, out0_two0, out0)
connect(canonical_CTNNB1, out0_two1, out0)
connect(canonical_CTNNB1, out0_two2, out0)
connect(canonical_CTNNB1, make_input_node(canonical_CTNNB1, "SRY", 1), out0_two0)
connect(canonical_CTNNB1, make_input_node(canonical_CTNNB1, "WNT4", 0), out0_two1)
connect(canonical_CTNNB1, make_input_node(canonical_CTNNB1, "RSPO1", 0), out0_two1)
connect(canonical_CTNNB1, make_input_node(canonical_CTNNB1, "AMH", 1), out0_two2)
connect(canonical_CTNNB1, make_input_node(canonical_CTNNB1, "SOX9", 1), out0_two2)

# out=1: schema over SRY literal + two symmetry fibres (leaf-first ok)
out1_two0 = make_threshold(canonical_CTNNB1, tau=3, role=ROLE_SCHEMA)
connect(canonical_CTNNB1, out1_two0, out1)
out1_two0_sym0 = make_threshold(canonical_CTNNB1, tau=1, role=ROLE_SYMMETRY)
out1_two0_sym1 = make_threshold(canonical_CTNNB1, tau=1, role=ROLE_SYMMETRY)
connect(canonical_CTNNB1, out1_two0_sym0, out1_two0)
connect(canonical_CTNNB1, out1_two0_sym1, out1_two0)
out1_two0_sym0_g1 = make_threshold(canonical_CTNNB1, tau=1, role=ROLE_POLARITY1)
out1_two0_sym1_g0 = make_threshold(canonical_CTNNB1, tau=1, role=ROLE_POLARITY0)
connect(canonical_CTNNB1, out1_two0_sym0_g1, out1_two0_sym0)
connect(canonical_CTNNB1, out1_two0_sym1_g0, out1_two0_sym1)
connect(canonical_CTNNB1, make_input_node(canonical_CTNNB1, "SRY", 0), out1_two0)
connect(canonical_CTNNB1, make_input_node(canonical_CTNNB1, "WNT4", 1), out1_two0_sym0_g1)
connect(canonical_CTNNB1, make_input_node(canonical_CTNNB1, "RSPO1", 1), out1_two0_sym0_g1)
connect(canonical_CTNNB1, make_input_node(canonical_CTNNB1, "AMH", 0), out1_two0_sym1_g0)
connect(canonical_CTNNB1, make_input_node(canonical_CTNNB1, "SOX9", 0), out1_two0_sym1_g0)

canon_result = lut_from_canalizing_map(
    canonical_CTNNB1, input_order=input_ids_in_order(canonical_CTNNB1)
)
summarize_roundtrip("canonical_CTNNB1", TRUE_CTNNB1, canon_result)
assert canon_result.outputs == [int(bit) for bit in TRUE_CTNNB1]
assert not canon_result.has_conflict and not canon_result.is_incomplete

# %% [markdown]
# ## Visualize the three CTNNB1 CMs
#
# Uses ``cana.drawing.canalizing_map.draw_canalizing_map_graphviz``.
# Writes ``.gv`` and ``.png`` under ``dev/logs/cm_roundtrip_figures/``.
# Nested ``feed`` edges are remapped for the legacy drawer.  If system
# Graphviz is missing, falls back to a portable ``dot_static`` + SVG→PNG.

# %%
_DEV_DIR = Path(__file__).resolve().parent
FIG_DIR = _DEV_DIR / "logs" / "cm_roundtrip_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
_TOOLS_DIR = _DEV_DIR / ".tools"
_DOT_STATIC = _TOOLS_DIR / "dot_static"
_DOT_STATIC_URL = (
    "https://raw.githubusercontent.com/restruct/dot-static/main/x64/dot_static"
)

CM_GRAPHS = {
    "legacy_CTNNB1": legacy_CTNNB1,
    "hand_CTNNB1": hand_CTNNB1,
    "canonical_CTNNB1": canonical_CTNNB1,
}


def _graph_for_legacy_drawer(graph: nx.DiGraph) -> nx.DiGraph:
    """Copy *graph*, remapping edge types the legacy drawer does not know.

    Nested-threshold CMs use ``feed`` (threshold→threshold); the GraphViz
    drawer only accepts out / literal / fusing / fused / simplified.  Treat
    ``feed`` as ``literal`` for display only.
    """
    H = graph.copy()
    for _s, _t, d in H.edges(data=True):
        if d.get("type") == "feed":
            d["type"] = "literal"
    return H


def _ensure_dot_executable() -> str | None:
    """Return a path to a usable ``dot`` binary, downloading a portable one if needed."""
    system_dot = shutil.which("dot")
    if system_dot:
        return system_dot
    if _DOT_STATIC.is_file() and os.access(_DOT_STATIC, os.X_OK):
        return str(_DOT_STATIC)
    try:
        import urllib.request

        _TOOLS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  downloading portable dot → {_DOT_STATIC}")
        urllib.request.urlretrieve(_DOT_STATIC_URL, _DOT_STATIC)
        _DOT_STATIC.chmod(0o755)
        return str(_DOT_STATIC)
    except Exception as exc:
        print(f"  could not obtain dot binary: {exc}")
        return None


def _dot_supports_png(dot_bin: str) -> bool:
    try:
        proc = subprocess.run(
            [dot_bin, "-Tpng"],
            input=b"digraph { a -> b }\n",
            capture_output=True,
            timeout=10,
        )
        return proc.returncode == 0 and proc.stdout[:8] == b"\x89PNG\r\n\x1a\n"
    except Exception:
        return False


def _render_png(dot_bin: str, gv_source: str, png_path: Path) -> None:
    """Render GraphViz source to PNG (direct, or SVG→PNG via cairosvg)."""
    if _dot_supports_png(dot_bin):
        proc = subprocess.run(
            [dot_bin, "-Tpng", "-o", str(png_path)],
            input=gv_source.encode("utf-8"),
            capture_output=True,
            timeout=60,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode("utf-8", errors="replace"))
        return

    svg_path = png_path.with_suffix(".svg")
    proc = subprocess.run(
        [dot_bin, "-Tsvg", "-o", str(svg_path)],
        input=gv_source.encode("utf-8"),
        capture_output=True,
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="replace"))
    try:
        from cairosvg import svg2png
    except ImportError as exc:
        raise RuntimeError(
            "dot has no PNG support; install cairosvg for SVG→PNG "
            "(pip install cairosvg)"
        ) from exc
    svg2png(url=str(svg_path), write_to=str(png_path))
    svg_path.unlink(missing_ok=True)


def try_draw_cm(name: str, graph: nx.DiGraph, out_dir: Path) -> None:
    print(f"--- draw {name} ({graph.number_of_nodes()} nodes, "
          f"{graph.number_of_edges()} edges) ---")
    if draw_canalizing_map_graphviz is None:
        print(f"  skipped (drawer unavailable: {_DRAW_IMPORT_ERROR})")
        return
    try:
        gv = draw_canalizing_map_graphviz(_graph_for_legacy_drawer(graph))
    except Exception as exc:
        print(f"  draw_canalizing_map_graphviz failed: {type(exc).__name__}: {exc}")
        return

    dot_path = out_dir / f"{name}.gv"
    png_path = out_dir / f"{name}.png"
    try:
        dot_path.write_text(gv.source)
        print(f"  wrote DOT → {dot_path}")
    except Exception as exc:
        print(f"  could not write DOT: {exc}")
        return

    dot_bin = _ensure_dot_executable()
    if dot_bin is None:
        print("  PNG skipped (no dot binary)")
        return
    try:
        _render_png(dot_bin, gv.source, png_path)
        print(f"  wrote PNG → {png_path}")
    except Exception as exc:
        print(f"  PNG failed: {type(exc).__name__}: {exc}")


for _name, _graph in CM_GRAPHS.items():
    try_draw_cm(_name, _graph, FIG_DIR)

print(f"figures dir → {FIG_DIR}")

# %% [markdown]
# ## Takeaway
#
# * Construction helpers are label-driven: ``connect`` infers edge types;
#   ``tid`` / input ids are automatic; thresholds do not store ``out``.
# * `lut_from_canalizing_map` evaluates nodes modularly from topology.
# * Threshold ``role`` annotates schema / symmetry / polarity0 / polarity1.
# * Forward legacy ``canalizing_map``, ``hand_CTNNB1``, and nested-threshold
#   ``canonical_CTNNB1`` all reconstruct CTNNB1.
# * Drawing writes ``.gv`` + ``.png`` (portable ``dot_static`` + SVG→PNG
#   when system Graphviz is unavailable).
