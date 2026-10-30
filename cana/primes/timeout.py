# -*- coding: utf-8 -*-
"""Wall-clock kill via subprocess process-group signals."""
from __future__ import annotations

import multiprocessing as mp
import os
import signal
from typing import Any, Callable, Tuple

__all__ = ["call_in_subprocess"]


def _kill_process_group(pid: int, sig: int) -> None:
    """Send *sig* to the process group led by *pid*; ignore if already gone."""
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        return
    except PermissionError:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            return


def call_in_subprocess(
    worker: Callable[..., None],
    args: Tuple[Any, ...],
    *,
    timeout_s: float,
) -> Any:
    """Run ``worker(*args, conn)`` in a child process; kill on timeout.

    *worker* must ``conn.send(("ok", result))`` or
    ``conn.send(("err", message))`` then close *conn*.  Raises
    ``TimeoutError`` if the child exceeds *timeout_s* wall-clock seconds.
    Raises ``RuntimeError`` on worker error or a missing result.

    The child calls ``os.setsid()`` before running *worker*, so it becomes a
    new session / process-group leader.  On timeout the whole group is
    signaled (``SIGTERM`` then ``SIGKILL``), which also reaps grandchildren.
    """
    if timeout_s <= 0:
        raise ValueError("call_in_subprocess requires timeout_s > 0")

    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    ready = ctx.Event()

    def _entry() -> None:
        os.setsid()
        ready.set()
        try:
            worker(*args, child_conn)
        finally:
            try:
                child_conn.close()
            except Exception:  # noqa: BLE001
                pass

    proc = ctx.Process(target=_entry)
    proc.start()
    child_conn.close()
    if not ready.wait(timeout=5.0):
        if proc.pid is not None:
            _kill_process_group(proc.pid, signal.SIGKILL)
        proc.join(1.0)
        parent_conn.close()
        raise RuntimeError("worker failed to enter new session")

    proc.join(float(timeout_s))
    if proc.is_alive():
        assert proc.pid is not None
        _kill_process_group(proc.pid, signal.SIGTERM)
        proc.join(5.0)
        if proc.is_alive():
            _kill_process_group(proc.pid, signal.SIGKILL)
            proc.join(1.0)
        parent_conn.close()
        raise TimeoutError(
            f"exceeded {timeout_s:g}s timeout (subprocess killed)"
        )
    exitcode = proc.exitcode

    def _missing_result() -> None:
        if isinstance(exitcode, int) and exitcode < 0:
            sig = -exitcode
            raise TimeoutError(
                f"exceeded {timeout_s:g}s timeout "
                f"(worker killed by signal {sig}, exitcode={exitcode})"
            )
        raise RuntimeError(
            f"worker exited without result (exitcode={exitcode})"
        )

    if not parent_conn.poll():
        parent_conn.close()
        _missing_result()
    try:
        status, payload = parent_conn.recv()
    except EOFError:
        parent_conn.close()
        _missing_result()
    parent_conn.close()
    if status == "ok":
        return payload
    raise RuntimeError(str(payload))
