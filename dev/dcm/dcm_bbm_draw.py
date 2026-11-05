# %% [markdown]
# DCM visualization (any BBM model)
#
# Load one Boolean model from the BBM corpus (default ``MODEL_ID = "023"``),
# build the Dynamics Canalization Map for all four combinations of
# ``legacy`` / ``canonical`` × ``simplify`` on/off, print compact graph stats,
# write ``.gv`` / ``.png`` under ``dev/logs/dcm_bbm_figures/<id>__<name>/``,
# and a ``README.md`` with per-variant node/edge counts and type/role breakdowns.
#
# Edit ``MODEL_ID`` at the top to switch models.  Builds DCMs via
# ``BooleanNetwork.dynamics_canalization_map(builder=..., simplify=...)``.

# %%
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx

_DEV_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _DEV_DIR.parent  # .../CANA/CANA (contains the cana package)
for _p in (_DEV_DIR, _PKG_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cana.boolean_network import BooleanNetwork

from bbm_cm_roundtrip import resolve_model_dir  # noqa: E402

try:
    from cana.drawing.canalizing_map import draw_canalizing_map_graphviz
except Exception as _draw_import_err:  # missing graphviz, or drawer import bug
    draw_canalizing_map_graphviz = None
    _DRAW_IMPORT_ERROR = _draw_import_err
else:
    _DRAW_IMPORT_ERROR = None

# ---------------------------------------------------------------------------
# Config — edit MODEL_ID to switch BBM models.
# ---------------------------------------------------------------------------

MODEL_ID = "023"  # any BBM summary ID, e.g. "001", "023", "100"
BUILDERS = ("legacy", "canonical")
SIMPLIFY_MODES = (False, True)

# .../CANA/CANA/dev → parents[1] = workspace root (.../CANA) containing BBM/
BBM_MODELS = _DEV_DIR.resolve().parents[1] / "BBM" / "models"

FIG_DIR = _DEV_DIR / "logs" / "dcm_bbm_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

_TOOLS_DIR = _DEV_DIR / ".tools"
_DOT_STATIC = _TOOLS_DIR / "dot_static"
_DOT_STATIC_URL = (
    "https://raw.githubusercontent.com/restruct/dot-static/main/x64/dot_static"
)

# %% [markdown]
# ## Helpers — load, draw, stats


# %%
def model_slug(model_dir: Path, model_id: str) -> str:
    """Build ``{pad}__{SHORT_NAME}`` from a BBM folder name."""
    pad = f"{int(model_id):03d}"
    prefix = f"[id-{pad}]__"
    name = model_dir.name
    if name.startswith(prefix):
        rest = name[len(prefix) :]
        # Drop ``[var-…]__[in-…]__`` segments if present; keep trailing name.
        parts = rest.split("__")
        short = parts[-1] if parts else rest
    else:
        short = name
    short = short.strip("[]") or "model"
    return f"{pad}__{short}"


def load_bbm_network(models_root: Path, model_id: str) -> Tuple[BooleanNetwork, Path]:
    """Resolve BBM ``model.booleannet.txt`` and load as ``BooleanNetwork``."""
    model_dir = resolve_model_dir(models_root, model_id)
    bnet_path = model_dir / "model.booleannet.txt"
    if not bnet_path.is_file():
        raise FileNotFoundError(f"Missing {bnet_path}")
    pad = f"{int(model_id):03d}"
    # Prefer the trailing human name from the folder slug.
    slug = model_slug(model_dir, model_id)
    net_name = slug.split("__", 1)[-1]
    network = BooleanNetwork.from_file(
        str(bnet_path),
        type="booleannet",
        name=net_name,
    )
    return network, model_dir


def variant_key(builder: str, simplify: bool) -> str:
    return f"{builder}_{'simplify' if simplify else 'full'}"


def dcm_stats(dcm: nx.DiGraph) -> Dict[str, object]:
    """Collect node / edge / type / tau / role counts for one DCM."""
    type_counts = Counter(d.get("type", "?") for _, d in dcm.nodes(data=True))
    edge_type_counts = Counter(
        d.get("type", "?") for _, _, d in dcm.edges(data=True)
    )
    tau_counts = Counter(
        d.get("tau")
        for _, d in dcm.nodes(data=True)
        if d.get("type") == "threshold"
    )
    role_counts = Counter(
        d.get("role", "(none)")
        for _, d in dcm.nodes(data=True)
        if d.get("type") == "threshold"
    )
    return {
        "nodes": dcm.number_of_nodes(),
        "edges": dcm.number_of_edges(),
        "node_types": dict(sorted(type_counts.items())),
        "edge_types": dict(sorted(edge_type_counts.items())),
        "thr_tau": dict(
            sorted(tau_counts.items(), key=lambda x: (x[0] is None, x[0]))
        ),
        "thr_roles": dict(sorted(role_counts.items())),
    }


def summarize_dcm(name: str, dcm: nx.DiGraph) -> Dict[str, object]:
    """Print compact stats for one DCM variant; return the same as ``dcm_stats``."""
    stats = dcm_stats(dcm)
    print(f"=== {name} ===")
    print(f"  nodes={stats['nodes']}  edges={stats['edges']}")
    print(f"  node types : {stats['node_types']}")
    print(f"  edge types : {stats['edge_types']}")
    print(f"  thr tau    : {stats['thr_tau']}")
    print(f"  thr roles  : {stats['thr_roles']}")
    return stats


def _count_get(d: object, key: object, default: int = 0) -> int:
    if isinstance(d, dict):
        return int(d.get(key, default))
    return default


def _union_keys(*dicts: object, sort_key=None) -> List[object]:
    keys: set = set()
    for d in dicts:
        if isinstance(d, dict):
            keys.update(d.keys())
    return sorted(keys, key=sort_key or (lambda x: (x is None, str(x))))


def _md_table(headers: List[str], rows: List[List[object]]) -> List[str]:
    """Build a GitHub-flavored markdown table."""
    lines = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return lines


def write_readme(
    out_dir: Path,
    *,
    model_id: str,
    network: BooleanNetwork,
    model_dir: Path,
    all_stats: Dict[str, Dict[str, object]],
    drawn: List[Tuple[str, Optional[Path]]],
) -> Path:
    """Write ``README.md`` next to the figures with per-variant graph stats."""
    n_const = sum(1 for n in network.nodes if n.constant)
    drawn_map = {key: png for key, png in drawn}
    variants = list(all_stats.keys())
    headers = ["", *[f"`{v}`" for v in variants]]

    size_rows: List[List[object]] = [
        [
            "figure",
            *[
                (
                    f"`{drawn_map[v].name}`"
                    if drawn_map.get(v) is not None
                    else "(missing)"
                )
                for v in variants
            ],
        ],
        ["nodes", *[all_stats[v]["nodes"] for v in variants]],
        ["edges", *[all_stats[v]["edges"] for v in variants]],
    ]

    node_type_keys = _union_keys(*(all_stats[v]["node_types"] for v in variants))
    node_type_rows = [
        [
            f"node type `{k}`",
            *[_count_get(all_stats[v]["node_types"], k) for v in variants],
        ]
        for k in node_type_keys
    ]

    edge_type_keys = _union_keys(*(all_stats[v]["edge_types"] for v in variants))
    edge_type_rows = [
        [
            f"edge type `{k}`",
            *[_count_get(all_stats[v]["edge_types"], k) for v in variants],
        ]
        for k in edge_type_keys
    ]

    tau_keys = _union_keys(
        *(all_stats[v]["thr_tau"] for v in variants),
        sort_key=lambda x: (x is None, x),
    )
    tau_rows = [
        [
            f"threshold τ={k}",
            *[_count_get(all_stats[v]["thr_tau"], k) for v in variants],
        ]
        for k in tau_keys
    ]

    role_keys = _union_keys(*(all_stats[v]["thr_roles"] for v in variants))
    role_rows = [
        [
            f"threshold role `{k}`",
            *[_count_get(all_stats[v]["thr_roles"], k) for v in variants],
        ]
        for k in role_keys
    ]

    lines: List[str] = [
        f"# DCM figures — {network.name}",
        "",
        f"- **MODEL_ID**: `{model_id}`",
        f"- **network nodes**: {len(network.nodes)} (constants={n_const})",
        f"- **model dir**: `{model_dir}`",
        "",
        "Four Dynamics Canalization Map variants: "
        "`legacy` / `canonical` × simplify on/off.",
        "",
        "Node types are `variable`, `threshold`, and (legacy only) `fusion`.",
        "Canonical threshold units carry **roles** "
        "(`schema`, `symmetry`, `polarity0`, `polarity1`); "
        "legacy thresholds typically have no role and may use fusion nodes "
        "instead of nested polarity/symmetry thresholds.",
        "",
        "## Comparison",
        "",
        *_md_table(
            headers,
            size_rows + node_type_rows + edge_type_rows + tau_rows + role_rows,
        ),
        "",
    ]
    readme_path = out_dir / "README.md"
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme_path


def _ensure_dot_executable() -> Optional[str]:
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
            timeout=120,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode("utf-8", errors="replace"))
        return

    svg_path = png_path.with_suffix(".svg")
    proc = subprocess.run(
        [dot_bin, "-Tsvg", "-o", str(svg_path)],
        input=gv_source.encode("utf-8"),
        capture_output=True,
        timeout=120,
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


def try_draw_cm(name: str, graph: nx.DiGraph, out_dir: Path) -> Optional[Path]:
    """Write ``{name}.gv`` and ``{name}.png`` under *out_dir*; return PNG path."""
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
        print(
            f"  draw_canalizing_map_graphviz failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    dot_path = out_dir / f"{name}.gv"
    png_path = out_dir / f"{name}.png"
    try:
        dot_path.write_text(gv.source)
        print(f"  wrote DOT → {dot_path}")
    except Exception as exc:
        print(f"  could not write DOT: {exc}")
        return None

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


# %% [markdown]
# ## Load model


# %%
N, model_dir = load_bbm_network(BBM_MODELS, MODEL_ID)
slug = model_slug(model_dir, MODEL_ID)
model_fig_dir = FIG_DIR / slug
model_fig_dir.mkdir(parents=True, exist_ok=True)

n_const = sum(1 for n in N.nodes if n.constant)
print(f"MODEL_ID   : {MODEL_ID}")
print(f"name       : {N.name}")
print(f"model_dir  : {model_dir}")
print(f"nodes      : {len(N.nodes)}  (constants={n_const})")
print(f"fig dir    : {model_fig_dir}")

# %% [markdown]
# ## Build four DCMs + summary counts


# %%
dcms: Dict[str, nx.DiGraph] = {}
for builder in BUILDERS:
    for simplify in SIMPLIFY_MODES:
        key = variant_key(builder, simplify)
        print(f"building {key} …")
        dcm = N.dynamics_canalization_map(builder=builder, simplify=simplify)
        dcm.name = f"DCM:{N.name}:{builder}:simplify={simplify}"
        dcms[key] = dcm

print()
all_stats: Dict[str, Dict[str, object]] = {}
for key, dcm in dcms.items():
    all_stats[key] = summarize_dcm(key, dcm)

# %% [markdown]
# ## Draw all four variants + README
#
# Writes ``.gv`` + ``.png`` and a ``README.md`` into ``model_fig_dir``.


# %%
drawn: List[Tuple[str, Optional[Path]]] = []
for key, dcm in dcms.items():
    png = try_draw_cm(key, dcm, model_fig_dir)
    drawn.append((key, png))

readme_path = write_readme(
    model_fig_dir,
    model_id=MODEL_ID,
    network=N,
    model_dir=model_dir,
    all_stats=all_stats,
    drawn=drawn,
)

print()
print(f"figures → {model_fig_dir}")
for key, png in drawn:
    status = png.name if png is not None else "(png missing)"
    print(f"  {key}: {status}")
print(f"readme  → {readme_path}")
