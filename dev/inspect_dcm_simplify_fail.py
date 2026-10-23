# %% [markdown]
# Inspect CM before / after stock DCM simplify (failed roundtrip)
#
# Pick one LUT that failed the simplify=True roundtrip in
# `logs/dcm_simplify_all_lut_roundtrip_20261021_221427`, rebuild its canalizing
# map, apply the same stock simplify used in
# `all_lut_dcm_simplify_roundtrip.py`, then show / draw CM **before** and
# **after**.

# %%
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import networkx as nx

_DEV_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _DEV_DIR.parent
for _p in (_DEV_DIR, _PKG_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cana.boolean_node import BooleanNode
from cana.canalization.canalizing_map import simplify_canalizing_map
from cana.canalization.cm_lut import lut_from_canalizing_map
from bbm_cm_roundtrip import is_exact_roundtrip  # noqa: E402

try:
    from cana.drawing.canalizing_map import draw_canalizing_map_graphviz
except Exception as _draw_import_err:
    draw_canalizing_map_graphviz = None
    _DRAW_IMPORT_ERROR = _draw_import_err
else:
    _DRAW_IMPORT_ERROR = None

# ---------------------------------------------------------------------------
# Config — point at the failed run; swap FAIL_PICK to inspect another LUT.
# ---------------------------------------------------------------------------

RUN_DIR = (
    _DEV_DIR / "logs" / "dcm_simplify_all_lut_roundtrip_20261021_221427"
)
LOG_PATH = RUN_DIR / f"{RUN_DIR.name}.log"

# Which (builder, simplify) section of the log to harvest FAIL lines from.
# Fails only occur for simplify=True in this run.
BUILDER = "legacy"  # "legacy" | "canonical"
SIMPLIFY = True

# None → first FAIL in that section. Or a name like "k2_id6_0110".
FAIL_PICK: Optional[str] = "k1_id0_00"

FIG_DIR = RUN_DIR / "inspect_cm_before_after"
_TOOLS_DIR = _DEV_DIR / ".tools"
_DOT_STATIC = _TOOLS_DIR / "dot_static"
_DOT_STATIC_URL = (
    "https://raw.githubusercontent.com/restruct/dot-static/main/x64/dot_static"
)

# %% [markdown]
# ## Helpers — LUT rebuild + DCM simplify
#
# Uses :func:`cana.canalization.canalizing_map.simplify_canalizing_map`
# (same as ``BooleanNetwork.dynamics_canalization_map``).


# %%
def outputs_from_func_id(k: int, func_id: int) -> List[int]:
    n_rows = 1 << k
    return [(func_id >> i) & 1 for i in range(n_rows)]


def lut_name(k: int, func_id: int) -> str:
    n_rows = 1 << k
    bits = "".join(str((func_id >> i) & 1) for i in range(n_rows))
    return f"k{k}_id{func_id}_{bits}"


def make_node(k: int, func_id: int) -> BooleanNode:
    outputs = outputs_from_func_id(k, func_id)
    inputs = list(range(k))
    return BooleanNode.from_output_list(
        outputs, name=lut_name(k, func_id), inputs=inputs, id=func_id
    )


def apply_stock_dcm_simplify(cm: nx.DiGraph) -> nx.DiGraph:
    """Apply DCM simplify to a canalizing-map graph (copy)."""
    return simplify_canalizing_map(cm, copy=True, remove_isolates=True)


# %% [markdown]
# ## Parse FAIL lines from the run log


# %%
_FAIL_RE = re.compile(
    r"^\s*FAIL\s+(?P<name>\S+)\s+k=(?P<k>\d+)\s+reason=(?P<reason>\S+)"
)
_SECTION_RE = re.compile(
    r"^########\s+builder=(?P<builder>\S+)\s+simplify=(?P<simplify>\S+)\s+########"
)
_NAME_RE = re.compile(r"^k(?P<k>\d+)_id(?P<fid>\d+)_(?P<bits>[01]+)$")


@dataclass
class FailRecord:
    name: str
    k: int
    reason: str
    builder: str
    simplify: bool


def parse_fails(
    log_path: Path, *, builder: str, simplify: bool
) -> List[FailRecord]:
    want_simp = "True" if simplify else "False"
    cur_builder: Optional[str] = None
    cur_simplify: Optional[str] = None
    out: List[FailRecord] = []
    for line in log_path.read_text().splitlines():
        m_sec = _SECTION_RE.match(line)
        if m_sec:
            cur_builder = m_sec.group("builder")
            cur_simplify = m_sec.group("simplify")
            continue
        if cur_builder != builder or cur_simplify != want_simp:
            continue
        m = _FAIL_RE.match(line)
        if not m:
            continue
        out.append(
            FailRecord(
                name=m.group("name"),
                k=int(m.group("k")),
                reason=m.group("reason"),
                builder=builder,
                simplify=simplify,
            )
        )
    return out


def parse_lut_name(name: str) -> Tuple[int, int, str]:
    m = _NAME_RE.match(name)
    if not m:
        raise ValueError(f"unexpected LUT name: {name!r}")
    return int(m.group("k")), int(m.group("fid")), m.group("bits")


fails = parse_fails(LOG_PATH, builder=BUILDER, simplify=SIMPLIFY)
print(f"log        : {LOG_PATH}")
print(f"section    : builder={BUILDER}  simplify={SIMPLIFY}")
print(f"n_fails    : {len(fails)}")
if not fails:
    raise SystemExit("no FAIL lines in that section — nothing to inspect")

if FAIL_PICK is None:
    pick = fails[0]
else:
    matched = [f for f in fails if f.name == FAIL_PICK]
    if not matched:
        preview = ", ".join(f.name for f in fails[:12])
        raise SystemExit(
            f"{FAIL_PICK!r} not among fails for this section. "
            f"first few: {preview} ..."
        )
    pick = matched[0]

k, func_id, bits = parse_lut_name(pick.name)
print(f"picked     : {pick.name}  reason={pick.reason}")
print(f"k / id     : {k} / {func_id}")
print(f"LUT bits   : {bits}")

# %% [markdown]
# ## Build CM before / after stock simplify + roundtrip both


# %%
def summarize_cm(label: str, g: nx.DiGraph) -> None:
    by_type: dict[str, int] = {}
    for _, d in g.nodes(data=True):
        t = str(d.get("type", "?"))
        by_type[t] = by_type.get(t, 0) + 1
    print(f"=== CM {label} ===")
    print(f"  nodes={g.number_of_nodes()}  edges={g.number_of_edges()}")
    print(f"  node types: {dict(sorted(by_type.items()))}")
    for n, d in sorted(g.nodes(data=True), key=lambda x: str(x[0])):
        attrs = {k: v for k, v in d.items() if k != "label"}
        print(f"  N  {n!r}  {attrs}")
    for u, v, d in sorted(g.edges(data=True), key=lambda e: (str(e[0]), str(e[1]))):
        print(f"  E  {u!r} → {v!r}  {dict(d)}")


def summarize_roundtrip(label: str, outputs, result) -> bool:
    orig = "".join(str(x) for x in outputs)
    recon = result.lut_string()
    exact = is_exact_roundtrip(outputs, result)
    print(f"=== roundtrip {label} ===")
    print(f"  original   : {orig}")
    print(f"  from CM    : {recon}")
    print(f"  conflict   : {result.has_conflict}  states={result.conflict_states}")
    print(
        f"  incomplete : {result.is_incomplete}  "
        f"uncovered={result.uncovered_states[:8]}"
        f"{'...' if len(result.uncovered_states) > 8 else ''}"
    )
    print(f"  exact      : {exact}")
    return exact


node = make_node(k, func_id)
cm_before = node.canalizing_map(builder=BUILDER)
cm_after = apply_stock_dcm_simplify(cm_before)

summarize_cm("before simplify", cm_before)
print()
summarize_cm("after simplify", cm_after)
print()

rt_before = lut_from_canalizing_map(cm_before, input_order=node.inputs)
rt_after = lut_from_canalizing_map(cm_after, input_order=node.inputs)
ok_before = summarize_roundtrip(f"{BUILDER}/simplify=False", node.outputs, rt_before)
print()
ok_after = summarize_roundtrip(f"{BUILDER}/simplify=True", node.outputs, rt_after)
print()
print(f"before exact={ok_before}  after exact={ok_after}  (log reason={pick.reason})")

# %% [markdown]
# ## Draw before / after (GraphViz)


# %%
def _ensure_dot_executable() -> Optional[str]:
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
            "dot has no PNG support; install cairosvg for SVG→PNG"
        ) from exc
    svg2png(url=str(svg_path), write_to=str(png_path))
    svg_path.unlink(missing_ok=True)


def try_draw_cm(name: str, graph: nx.DiGraph, out_dir: Path) -> Optional[Path]:
    print(
        f"--- draw {name} ({graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges) ---"
    )
    if draw_canalizing_map_graphviz is None:
        print(f"  skipped (drawer unavailable: {_DRAW_IMPORT_ERROR})")
        return None
    try:
        gv = draw_canalizing_map_graphviz(graph)
    except Exception as exc:
        print(f"  draw failed: {type(exc).__name__}: {exc}")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    dot_path = out_dir / f"{name}.gv"
    png_path = out_dir / f"{name}.png"
    dot_path.write_text(gv.source)
    print(f"  wrote DOT → {dot_path}")

    dot_bin = _ensure_dot_executable()
    if dot_bin is None:
        print("  PNG skipped (no dot binary)")
        return None
    try:
        _render_png(dot_bin, gv.source, png_path)
        print(f"  wrote PNG → {png_path}")
        return png_path
    except Exception as exc:
        print(f"  PNG failed: {type(exc).__name__}: {exc}")
        return None


out_dir = FIG_DIR / f"{BUILDER}_simplify" / pick.name
png_before = try_draw_cm("cm_before", cm_before, out_dir)
png_after = try_draw_cm("cm_after", cm_after, out_dir)
print(f"figures → {out_dir}")

# Inline preview when running as a notebook / interactive session.
try:
    from IPython.display import Image, display
except ImportError:
    pass
else:
    for label, path in (("before", png_before), ("after", png_after)):
        if path is not None and path.is_file():
            print(f"preview {label}:")
            display(Image(filename=str(path)))
