# %% [markdown]
# Canalizing Map ↔ LUT roundtrip
#
# Walkthrough:
# 1. Pick a Boolean rule (swap the active ``BNET`` below).
# 2. Build a ``BooleanNode`` and read its LUT.
# 3. Build canalizing maps (legacy + canonical) and store them keyed by LUT.
# 4. Reconstruct the LUT from each CM and compare.
# 5. Draw the CMs under ``dev/logs/cm_roundtrip_figures/<lut>/``.

# %%
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import networkx as nx
from cana.boolean_network import BooleanNetwork
from cana.canalization.cm_lut import lut_from_canalizing_map

try:
    from cana.drawing.canalizing_map import draw_canalizing_map_graphviz
except Exception as _draw_import_err:  # missing graphviz, or drawer import bug
    draw_canalizing_map_graphviz = None
    _DRAW_IMPORT_ERROR = _draw_import_err
else:
    _DRAW_IMPORT_ERROR = None

_DEV_DIR = Path(__file__).resolve().parent
FIG_DIR = _DEV_DIR / "logs" / "cm_roundtrip_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
_TOOLS_DIR = _DEV_DIR / ".tools"
_DOT_STATIC = _TOOLS_DIR / "dot_static"
_DOT_STATIC_URL = (
    "https://raw.githubusercontent.com/restruct/dot-static/main/x64/dot_static"
)

BUILDERS = ("legacy", "canonical")

# ---------------------------------------------------------------------------
# Example functions — uncomment one ``BNET`` (and matching ``TARGET``) to swap.
# ---------------------------------------------------------------------------

# # F = X ∧ ¬Y ∧ (exactly two of A,B,C)
# BNET = (
#     "F*= X and not Y and ((not A and B and C) or (A and not B and C)"
#     " or (A and B and not C))\n"
#     "X*=X\n"
#     "Y*=Y\n"
#     "A*=A\n"
#     "B*=B\n"
#     "C*=C\n"
# )
# TARGET = "F"

# # CTNNB1 = (not SRY) and (WNT4 or RSPO1) and ((not AMH) or (not SOX9))
# BNET = (
#     "CTNNB1*=(not SRY) and (WNT4 or RSPO1) and ((not AMH) or (not SOX9))\n"
#     "SRY*=SRY\n"
#     "WNT4*=WNT4\n"
#     "RSPO1*=RSPO1\n"
#     "AMH*=AMH\n"
#     "SOX9*=SOX9\n"
# )
# TARGET = "CTNNB1"

# # F = (X XOR Y) ∧ (exactly two of A,B,C)
# BNET = (
#     "F*= ((X and not Y) or (not X and Y)) and ((not A and B and C) or (A and not B and C)"
#     " or (A and B and not C))\n"
#     "X*=X\n"
#     "Y*=Y\n"
#     "A*=A\n"
#     "B*=B\n"
#     "C*=C\n"
# )
# TARGET = "F"

# 0(01)(1#), 1(00#)#, 111##
BNET = (
    "F*= not A and ((not B and C) or (B and not C)) and (D or E)"
    " or A and ((not B and not C) or (not B and not D) or (not C and not D))"
    " or A and B and C\n"
    "A*=A\n"
    "B*=B\n"
    "C*=C\n"
    "D*=D\n"
    "E*=E\n"
)
TARGET = "F"

# %% [markdown]
# ## Helpers


# %%
def lut_key(outputs) -> str:
    """Encode a LUT as a bit-string for dict keys and figure paths."""
    return "".join(str(int(x)) for x in outputs)


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
    print(
        f"  incomplete : {result.is_incomplete}  "
        f"uncovered={result.uncovered_states[:8]}"
        f"{'...' if len(result.uncovered_states) > 8 else ''}"
    )
    print(f"  exact match: {exact}")
    return exact


def node_from_bnet(bnet: str, target: str):
    """Parse a booleannet string and return ``(network, target_node)``."""
    network = BooleanNetwork.from_string_booleannet(bnet)
    node = next(n for n in network.nodes if n.name == target)
    return network, node


def build_cms(node, builders=BUILDERS) -> dict[str, nx.DiGraph]:
    """Build canalizing maps for *node*; return ``{builder: DiGraph}``."""
    return {builder: node.canalizing_map(builder=builder) for builder in builders}


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
    print(
        f"--- draw {name} ({graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges) ---"
    )
    if draw_canalizing_map_graphviz is None:
        print(f"  skipped (drawer unavailable: {_DRAW_IMPORT_ERROR})")
        return
    try:
        gv = draw_canalizing_map_graphviz(graph)
    except Exception as exc:
        print(
            f"  draw_canalizing_map_graphviz failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return

    out_dir.mkdir(parents=True, exist_ok=True)
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


# %% [markdown]
# ## Function → LUT → canalizing maps → roundtrip


# %%
N, node = node_from_bnet(BNET, TARGET)
input_labels = [N.nodes[i].name for i in node.inputs]
lut = lut_key(node.outputs)

print(f"target     : {node.name}")
print(f"inputs     : {list(zip(node.inputs, input_labels))}")
print(f"k          : {node.k}")
print(f"LUT        : {lut}")

# Store CMs keyed by LUT encoding:  cms[lut][builder] → DiGraph
cms: dict[str, dict[str, nx.DiGraph]] = {}
cms[lut] = build_cms(node)

node._check_compute_canalization_variables(two_symbols=True)
print("Two-symbol schemata (out=0):")
for row in node._two_symbols[0]:
    print(" ", row)
print("Two-symbol schemata (out=1):")
for row in node._two_symbols[1]:
    print(" ", row)

for builder, cm in cms[lut].items():
    result = lut_from_canalizing_map(cm, input_order=node.inputs)
    exact = summarize_roundtrip(f"{node.name}/{builder}", node.outputs, result)
    assert exact, f"Expected {builder} roundtrip of {node.name} to succeed"

# Draw under figures/<lut>/; also record the source bnet.
lut_fig_dir = FIG_DIR / lut
lut_fig_dir.mkdir(parents=True, exist_ok=True)
bnet_path = lut_fig_dir / "source.bnet"
bnet_path.write_text(BNET if BNET.endswith("\n") else BNET + "\n")
print(f"  wrote bnet → {bnet_path}")

for builder, cm in cms[lut].items():
    try_draw_cm(builder, cm, lut_fig_dir)

print(f"stored CMs : cms[{lut!r}] → {list(cms[lut])}")
print(f"figures    → {lut_fig_dir}")
