# -*- coding: utf-8 -*-
"""Tee stdout lines (and Python warnings) into ``high_k/logs/<notebook>/``."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable, Optional, TextIO, Tuple

from high_k.paths import LOGS_DIR

__all__ = ["TeeLogger", "run_log_dir", "start_run_log"]


class TeeLogger:
    """Write ``info`` lines to stdout and a log file; mirror warnings into the file.

    ``warnings`` still go to stderr via the previous ``showwarning`` hook; this
    logger only *adds* them to the run log so a clean log is never a silent
    drop of stderr traffic.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fh: TextIO = path.open("w", encoding="utf-8")
        self._prev_showwarning: Optional[Callable[..., None]] = None

    def info(self, msg: str = "") -> None:
        print(msg)
        self._fh.write(msg + "\n")
        self._fh.flush()

    def _install_warning_tee(self) -> None:
        self._prev_showwarning = warnings.showwarning

        def _showwarning(message, category, filename, lineno, file=None, line=None):
            prev = self._prev_showwarning
            if prev is not None:
                prev(message, category, filename, lineno, file=file, line=line)
            text = warnings.formatwarning(
                message, category, filename, lineno, line
            )
            self._fh.write(text)
            if not text.endswith("\n"):
                self._fh.write("\n")
            self._fh.flush()

        warnings.showwarning = _showwarning  # type: ignore[assignment]

    def close(self) -> None:
        if self._prev_showwarning is not None:
            warnings.showwarning = self._prev_showwarning  # type: ignore[assignment]
            self._prev_showwarning = None
        self._fh.close()


def run_log_dir(notebook_name: str) -> Path:
    """Return ``high_k/logs/<notebook_name>/`` (created on demand by callers)."""
    return LOGS_DIR / notebook_name


def start_run_log(notebook_name: str) -> Tuple[TeeLogger, Path]:
    """Open a teeing logger at ``logs/<notebook_name>/<notebook_name>.log``.

    The file is truncated on each run so the latest output overwrites the
    previous log. Python ``warnings`` emitted while the logger is open are
    appended to the same file (stderr still receives them as usual).

    Returns:
        ``(logger, log_path)``.
    """
    log_path = run_log_dir(notebook_name) / f"{notebook_name}.log"
    log = TeeLogger(log_path)
    log._install_warning_tee()
    log.info(f"log_path={log_path}")
    return log, log_path
