# -*- coding: utf-8 -*-
"""Shared filesystem roots for ``dev/high_k`` notebooks."""
from __future__ import annotations

import sys
from pathlib import Path

HIGH_K_DIR = Path(__file__).resolve().parent
DEV_DIR = HIGH_K_DIR.parent
PKG_ROOT = DEV_DIR.parent  # .../CANA/CANA (contains the cana package)
# .../CANA/CANA/dev → parents[1] = workspace root (.../CANA) containing BBM/
WORKSPACE_ROOT = DEV_DIR.resolve().parents[1]
BBM_MODELS = WORKSPACE_ROOT / "BBM" / "models"
LOGS_DIR = HIGH_K_DIR / "logs"


def ensure_cana_on_path() -> None:
    """Insert ``dev/`` and the CANA package root on ``sys.path`` if needed."""
    for p in (DEV_DIR, PKG_ROOT):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
