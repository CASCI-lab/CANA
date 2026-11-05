# -*- coding: utf-8 -*-
"""BBM corpus helpers: resolve models and pick rules by in-degree *k*."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator, List, Optional, Set, Tuple

from high_k.paths import BBM_MODELS, ensure_cana_on_path

ensure_cana_on_path()

from cana.function_outputs.expression import (
    expression_identifiers,
    normalize_operators,
)

__all__ = [
    "BBM_MODELS",
    "as_bnet",
    "resolve_model_dir",
    "model_id_from_dirname",
    "model_keywords",
    "unique_regulators",
    "iter_bnet_rules",
    "bbm_rule_for_k",
    "bbm_rules_for_k",
]

_MODEL_ID_RE = re.compile(r"^\[([^\]]+)\]__")


def model_id_from_dirname(dirname: str) -> Optional[str]:
    """Extract ``id-001``-style id from a ``[id-001]__…`` directory name."""
    m = _MODEL_ID_RE.match(dirname)
    return m.group(1) if m else None


def model_keywords(model_dir: Path) -> Set[str]:
    """Return ``keywords`` from ``metadata.json``, or empty if missing."""
    path = model_dir / "metadata.json"
    if not path.exists():
        return set()
    meta = json.loads(path.read_text())
    return set(meta.get("keywords", []))



def as_bnet(expr: str, target: str = "Y") -> str:
    """Wrap a symbolic expression as a minimal ``targets,factors`` bnet."""
    return f"targets,factors\n{target}, {expr}\n"


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


def unique_regulators(expr: str) -> List[str]:
    """Ordered unique identifiers in *expr* (boolean keywords stripped)."""
    toks = expression_identifiers(normalize_operators(expr, "python"))
    seen = set()
    out: List[str] = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def iter_bnet_rules(
    models_root: Path,
    *,
    include_keywords: Optional[List[str]] = None,
    exclude_keywords: Optional[List[str]] = None,
) -> Iterator[Tuple[str, str, str]]:
    """Yield ``(model_dir_name, target, expr)`` for every rule under *models_root*.

    Optional keyword filters match ``metadata.json`` ``keywords``:
    keep models that contain *all* ``include_keywords`` and *none* of
    ``exclude_keywords``.
    """
    include = list(include_keywords or ())
    exclude = list(exclude_keywords or ())
    if not models_root.is_dir():
        return
    for model_dir in sorted(models_root.iterdir()):
        if not model_dir.is_dir():
            continue
        if include or exclude:
            keywords = model_keywords(model_dir)
            if include and not all(k in keywords for k in include):
                continue
            if exclude and any(k in keywords for k in exclude):
                continue
        path = model_dir / "model.bnet"
        if not path.exists():
            continue
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("targets") and "," in line:
                continue
            if "," not in line:
                continue
            target, expr = line.split(",", 1)
            target = target.strip()
            expr = expr.strip()
            if not target:
                continue
            yield model_dir.name, target, expr


def bbm_rule_for_k(
    models_root: Path, k: int
) -> Optional[Tuple[str, str, str, str]]:
    """Return ``(model_dir_name, target, expr, bnet_text)`` for the first
    rule with exactly *k* unique regulators, or ``None``.
    """
    hits = bbm_rules_for_k(models_root, k, n=1)
    return hits[0] if hits else None


def bbm_rules_for_k(
    models_root: Path, k: int, n: int = 1
) -> List[Tuple[str, str, str, str]]:
    """Return up to *n* rules with exactly *k* unique regulators.

    Each hit is ``(model_dir_name, target, expr, bnet_text)``, in corpus
    iteration order (first *n* matches).
    """
    if n <= 0:
        return []
    out: List[Tuple[str, str, str, str]] = []
    for model_name, target, expr in iter_bnet_rules(models_root):
        if len(unique_regulators(expr)) != k:
            continue
        out.append((model_name, target, expr, as_bnet(expr, target=target)))
        if len(out) >= n:
            break
    return out
