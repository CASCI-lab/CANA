# %% [markdown]
# Canalizing Map ↔ LUT roundtrip — ``X ∧ ¬Y ∧ (exactly two of A,B,C)``
#
# Walkthrough:
# 1. Build a Boolean node from the rule below.
# 2. Convert to a Canalizing Map (CM) via the **canonical** builder
#    (`BooleanNode.canalizing_map(builder="canonical")` /
#    `build_canalizing_map_canonical`), and also via the legacy builder.
# 3. Reconstruct a LUT from that CM via `cm_lut.lut_from_canalizing_map`.
# 4. Compare original vs reconstructed LUTs.
# 5. Also keep the hand-built nested-threshold CM and draw all of them.
#
# Rule
# ----
# ```
# F = X and not Y and (
#       (not A and B and C)
#    or (A and not B and C)
#    or (A and B and not C)
# )
# ```
# Equivalent to ``X ∧ ¬Y ∧ (A+B+C = 2)``.

# %%
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import networkx as nx
from cana.boolean_network import BooleanNetwork
from cana.boolean_node import BooleanNode
from cana.canalization.canalizing_map import (
    build_canalizing_map_canonical,
    build_canalizing_map_legacy,
)
from cana.canalization.cm_lut import (
    ROLE_POLARITY0,
    ROLE_POLARITY1,
    ROLE_SCHEMA,
    ROLE_SYMMETRY,
    connect,
    input_ids_in_order,
    lut_from_canalizing_map,
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


TRUE_F = "00000000000000000001011000000000"


# %% [markdown]
# ## Case 1 — AND / OR from LUT
#
# Simple canalizing functions.  Both forward builders round-trip these.

# %%
and_node = BooleanNode.from_output_list(
    [0, 0, 0, 1], name="AND", inputs=[0, 1], id=2
)
and_cm = and_node.canalizing_map(builder="canonical")
and_result = lut_from_canalizing_map(and_cm, input_order=and_node.inputs)
summarize_roundtrip("AND (canonical)", and_node.outputs, and_result)

or_node = BooleanNode.from_output_list(
    [0, 1, 1, 1], name="OR", inputs=[10, 11], id=5
)
or_cm = or_node.canalizing_map(builder="canonical")
or_result = lut_from_canalizing_map(or_cm, input_order=or_node.inputs)
summarize_roundtrip("OR (canonical)", or_node.outputs, or_result)

# %% [markdown]
# ## Case 2 — ``F`` via legacy canalizing map builder
#
# ```
# F = X and not Y and ((not A and B and C) or (A and not B and C)
#                                          or (A and B and not C))
# ```

# %%
F_BNET = (
    "F*= X and not Y and ((not A and B and C) or (A and not B and C) or (A and B and not C))\n"
    "X*=X\n"
    "Y*=Y\n"
    "A*=A\n"
    "B*=B\n"
    "C*=C\n"
)

N = BooleanNetwork.from_string_boolean(F_BNET)
fn = next(n for n in N.nodes if n.name == "F")
input_labels = [N.nodes[i].name for i in fn.inputs]
print("F inputs (LUT order):", list(zip(fn.inputs, input_labels)))
print("True LUT:", "".join(fn.outputs))

fn._check_compute_canalization_variables(two_symbols=True)
print("Two-symbol schemata (out=0):")
for row in fn._two_symbols[0]:
    print(" ", row)
print("Two-symbol schemata (out=1):")
for row in fn._two_symbols[1]:
    print(" ", row)

_input_labels = {i: N.nodes[i].name for i in fn.inputs}

legacy_F = build_canalizing_map_legacy(
    fn._two_symbols,
    node_id=fn.id,
    inputs=fn.inputs,
    name=fn.name,
    constant=fn.constant,
    input_labels=_input_labels,
    k=fn.k,
)
legacy_result = lut_from_canalizing_map(legacy_F, input_order=fn.inputs)
exact = summarize_roundtrip("legacy_F", fn.outputs, legacy_result)
assert exact, "Expected legacy F roundtrip to succeed"

# %% [markdown]
# ## Case 3 — ``F`` via canonical canalizing map builder
#
# Nested schema / symmetry / polarity thresholds; no fusion nodes.
# Same as ``fn.canalizing_map(builder="canonical")``.

# %%
canonical_F = build_canalizing_map_canonical(
    fn._two_symbols,
    node_id=fn.id,
    inputs=fn.inputs,
    name=fn.name,
    constant=fn.constant,
    input_labels=_input_labels,
    k=fn.k,
)
canonical_result = lut_from_canalizing_map(canonical_F, input_order=fn.inputs)
exact = summarize_roundtrip("canonical_F", fn.outputs, canonical_result)
assert exact, "Expected canonical F roundtrip to succeed"

# %% [markdown]
# ## Case 4 — Hand-built nested-threshold CM (`hand_canonical_F`)
#
# Same function with nested schema / symmetry / polarity thresholds on out=1
# (manual reference for the canonical builder).  Out=0 keeps flat schemata
# (including ``τ=2`` for ≥2 zeros among ABC).

# %%
hand_canonical_F = nx.DiGraph(name="hand_canonical_F")
out0 = make_output_node(hand_canonical_F, "F", 0)
out1 = make_output_node(hand_canonical_F, "F", 1)

x0 = make_input_node(hand_canonical_F, "X", 0)
x1 = make_input_node(hand_canonical_F, "X", 1)
y0 = make_input_node(hand_canonical_F, "Y", 0)
y1 = make_input_node(hand_canonical_F, "Y", 1)
a0 = make_input_node(hand_canonical_F, "A", 0)
a1 = make_input_node(hand_canonical_F, "A", 1)
b0 = make_input_node(hand_canonical_F, "B", 0)
b1 = make_input_node(hand_canonical_F, "B", 1)
c0 = make_input_node(hand_canonical_F, "C", 0)
c1 = make_input_node(hand_canonical_F, "C", 1)

# out=0: four schemata
out0_x = make_threshold(hand_canonical_F, tau=1, role=ROLE_SCHEMA)
out0_y = make_threshold(hand_canonical_F, tau=1, role=ROLE_SCHEMA)
out0_ge2zero = make_threshold(hand_canonical_F, tau=1, role=ROLE_SCHEMA)
out0_all1 = make_threshold(hand_canonical_F, tau=3, role=ROLE_SCHEMA)
connect(hand_canonical_F, out0_x, out0)
connect(hand_canonical_F, out0_y, out0)
connect(hand_canonical_F, out0_ge2zero, out0)
connect(hand_canonical_F, out0_all1, out0)
out0_ge2zero_sym = make_threshold(
    hand_canonical_F, tau=1, role=ROLE_SYMMETRY
)
connect(hand_canonical_F, out0_ge2zero_sym, out0_ge2zero)
out0_ge2zero_sym_pol0 = make_threshold(
    hand_canonical_F, tau=2, role=ROLE_POLARITY0
)
connect(hand_canonical_F, out0_ge2zero_sym_pol0, out0_ge2zero_sym)
connect(hand_canonical_F, x0, out0_x)
connect(hand_canonical_F, y1, out0_y)
connect(hand_canonical_F, a0, out0_ge2zero_sym_pol0)
connect(hand_canonical_F, b0, out0_ge2zero_sym_pol0)
connect(hand_canonical_F, c0, out0_ge2zero_sym_pol0)
connect(hand_canonical_F, a1, out0_all1)
connect(hand_canonical_F, b1, out0_all1)
connect(hand_canonical_F, c1, out0_all1)

# out=1: schema over X=1, Y=0 + polarity0 (τ=1) + polarity1 (τ=2) fibres
out1_schema = make_threshold(hand_canonical_F, tau=3, role=ROLE_SCHEMA)
connect(hand_canonical_F, out1_schema, out1)
out1_schema_sym = make_threshold(hand_canonical_F, tau=2, role=ROLE_SYMMETRY)
connect(hand_canonical_F, out1_schema_sym, out1_schema)
out1_schema_sym_pol0 = make_threshold(
    hand_canonical_F, tau=1, role=ROLE_POLARITY0
)
out1_schema_sym_pol1 = make_threshold(
    hand_canonical_F, tau=2, role=ROLE_POLARITY1
)
connect(hand_canonical_F, out1_schema_sym_pol0, out1_schema_sym)
connect(hand_canonical_F, out1_schema_sym_pol1, out1_schema_sym)
connect(hand_canonical_F, x1, out1_schema)
connect(hand_canonical_F, y0, out1_schema)
connect(hand_canonical_F, a0, out1_schema_sym_pol0)
connect(hand_canonical_F, b0, out1_schema_sym_pol0)
connect(hand_canonical_F, c0, out1_schema_sym_pol0)
connect(hand_canonical_F, a1, out1_schema_sym_pol1)
connect(hand_canonical_F, b1, out1_schema_sym_pol1)
connect(hand_canonical_F, c1, out1_schema_sym_pol1)

hand_canon_result = lut_from_canalizing_map(
    hand_canonical_F, input_order=input_ids_in_order(hand_canonical_F)
)
summarize_roundtrip("hand_canonical_F", TRUE_F, hand_canon_result)
assert hand_canon_result.outputs == [int(bit) for bit in TRUE_F]
assert not hand_canon_result.has_conflict and not hand_canon_result.is_incomplete

# %% [markdown]
# ## Visualize the ``F`` CMs
#
# Uses ``cana.drawing.canalizing_map.draw_canalizing_map_graphviz``.
# Writes ``.gv`` and ``.png`` under ``dev/logs/cm_roundtrip_xy_abc_figures/``.

# %%
_DEV_DIR = Path(__file__).resolve().parent
FIG_DIR = _DEV_DIR / "logs" / "cm_roundtrip_xy_abc_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
_TOOLS_DIR = _DEV_DIR / ".tools"
_DOT_STATIC = _TOOLS_DIR / "dot_static"
_DOT_STATIC_URL = (
    "https://raw.githubusercontent.com/restruct/dot-static/main/x64/dot_static"
)

CM_GRAPHS = {
    "legacy_F": legacy_F,
    "canonical_F": canonical_F,
    "hand_canonical_F": hand_canonical_F,
}


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
        gv = draw_canalizing_map_graphviz(graph)
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
# * ``F = X ∧ ¬Y ∧ (A+B+C = 2)`` round-trips via legacy and canonical
#   builders, and via hand nested-threshold ``hand_canonical_F``.
# * Out=0 schemata: ``X=0``, ``Y=1``, ``≥2`` zeros among ABC, ``A=B=C=1``.
# * Out=1: ``X=1``, ``Y=0``, plus ``≥1`` zero and ``≥2`` ones among ABC.
# * Drawing writes ``.gv`` + ``.png`` under ``cm_roundtrip_xy_abc_figures/``.
