# -*- coding: utf-8 -*-
"""Put ``dev/`` on ``sys.path`` so ``import high_k…`` works under pytest."""
from __future__ import annotations

import sys
from pathlib import Path

_DEV_DIR = Path(__file__).resolve().parents[2]  # …/CANA/dev
if str(_DEV_DIR) not in sys.path:
    sys.path.insert(0, str(_DEV_DIR))
