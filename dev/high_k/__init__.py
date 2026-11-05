# -*- coding: utf-8 -*-
"""Shared helpers for ``dev/high_k`` notebooks (synthetic ROFs, BBM rules, timing)."""
from high_k.paths import (
    BBM_MODELS,
    DEV_DIR,
    LOGS_DIR,
    PKG_ROOT,
    ensure_cana_on_path,
)

ensure_cana_on_path()

__all__ = [
    "BBM_MODELS",
    "DEV_DIR",
    "LOGS_DIR",
    "PKG_ROOT",
    "ensure_cana_on_path",
]
