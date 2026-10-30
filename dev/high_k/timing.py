# -*- coding: utf-8 -*-
"""Timing helpers for high-*k* CANA benchmarks."""
from __future__ import annotations

import signal
import statistics
import time
from typing import Callable, Dict, List, Optional, Tuple, TypeVar

from high_k.paths import ensure_cana_on_path

ensure_cana_on_path()

from cana.function_outputs import FunctionOutputs
from cana.io import import_detached_rule
from cana.nodes_builder import DetachedRule
from cana.primes.timeout import call_in_subprocess

__all__ = [
    "median_time",
    "fmt_s",
    "call_with_timeout",
    "call_in_subprocess",
    "import_detached_outputs",
    "time_expr_lut_one_symbol",
    "time_import_and_rof",
    "time_import_and_unate",
]

_T = TypeVar("_T")


class _Timeout(Exception):
    """Raised internally by :func:`call_with_timeout` via ``SIGALRM``."""


def median_time(
    fn: Callable[[], object], repeats: int = 3
) -> Tuple[float, object]:
    """Run *fn* *repeats* times; return ``(median_seconds, last_result)``."""
    times: List[float] = []
    result: object = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), result


def fmt_s(t: Optional[float]) -> str:
    """Format a duration in seconds for compact table cells."""
    if t is None:
        return "  skip"
    if t < 1e-3:
        return f"{t * 1e6:7.1f} µs"
    if t < 1.0:
        return f"{t * 1e3:7.2f} ms"
    return f"{t:7.3f} s"


def call_with_timeout(
    seconds: float, fn: Callable[[], _T]
) -> _T:
    """Run *fn* on the main thread; raise ``TimeoutError`` after *seconds*.

    Uses ``signal.setitimer`` / ``SIGALRM`` (Unix, main thread only).  Pure
    Python work (e.g. AST parse) is interruptible; C extensions that block
    signals may overrun the limit until they return to Python.
    """
    if seconds <= 0:
        return fn()

    def _handler(signum, frame):  # noqa: ARG001
        raise _Timeout()

    prev = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn()
    except _Timeout as exc:
        raise TimeoutError(
            f"exceeded {seconds:g}s timeout"
        ) from exc
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, prev)


def import_detached_outputs(
    bnet_text: str, target: str
) -> FunctionOutputs:
    """Parse a one-rule bnet snippet → detached positional ``FunctionOutputs``.

    Uses :func:`~cana.io.import_detached_rule` (no free-input siblings).
    """
    rule = import_detached_rule(bnet_text, fmt="bnet", target=target)
    fo = rule.outputs
    if fo.positional_expression is None:
        raise RuntimeError("expected positional_expression on DetachedRule")
    return FunctionOutputs(
        positional_expression=fo.positional_expression, k=fo.k
    )


def time_expr_lut_one_symbol(
    bnet_text: str,
    target: str,
    *,
    repeats: int = 3,
    do_one_symbol: bool = True,
) -> Dict[str, object]:
    """Median times for import → LUT → one-symbol on *target*.

    Each stage starts from a fresh :func:`import_detached_outputs` so peer
    caches are not shared across stages.
    """

    def _import():
        return import_detached_outputs(bnet_text, target)

    t_import, fo0 = median_time(_import, repeats=repeats)
    assert isinstance(fo0, FunctionOutputs)
    k = fo0.k

    def _lut():
        fo = import_detached_outputs(bnet_text, target)
        return fo.build_bit_lut_from_positional_expression()

    t_lut, lut = median_time(_lut, repeats=repeats)

    t_one = None
    n_on = n_off = None
    if do_one_symbol:

        def _one():
            fo = import_detached_outputs(bnet_text, target)
            fo.build_bit_lut_from_positional_expression()
            return fo.compute_one_symbol_schemata_from_bit_lut()

        t_one, schemata = median_time(_one, repeats=repeats)
        n_on = len(schemata.get(1, ()))
        n_off = len(schemata.get(0, ()))

    return {
        "k": k,
        "t_import": t_import,
        "t_lut": t_lut,
        "t_one_symbol": t_one,
        "lut_len": len(lut) if lut is not None else None,
        "n_pi_on": n_on,
        "n_pi_off": n_off,
    }


def time_import_and_rof(
    bnet_text: str,
    target: str,
    expr: str,
    *,
    repeats: int = 3,
) -> Dict[str, object]:
    """Median times for expression import and a syntactic ROF check on *expr*.

    Import is :func:`import_detached_rule` only (no LUT / one-symbol).
    The ROF check runs on the original named *expr* string via
    :func:`~high_k.rof.check_syntactic_rof`.
    """
    from high_k.rof import check_syntactic_rof

    def _import():
        return import_detached_rule(bnet_text, fmt="bnet", target=target)

    t_import, rule = median_time(_import, repeats=repeats)
    assert isinstance(rule, DetachedRule)
    k = rule.k

    def _rof():
        return check_syntactic_rof(expr)

    t_rof, rof = median_time(_rof, repeats=repeats)
    assert isinstance(rof, dict)

    return {
        "k": k,
        "t_import": t_import,
        "t_rof": t_rof,
        "is_rof": bool(rof["is_rof"]),
        "n_variables": int(rof["n_variables"]),  # type: ignore[arg-type]
        "n_unique_variables": int(rof["n_unique_variables"]),  # type: ignore[arg-type]
    }


def time_import_and_unate(
    bnet_text: str,
    target: str,
    expr: str,
    *,
    repeats: int = 3,
) -> Dict[str, object]:
    """Median times for expression import and a syntactic unate check on *expr*.

    Import is :func:`import_detached_rule` only (no LUT / one-symbol).
    The unate check runs on the original named *expr* string via
    :func:`~high_k.unate.check_syntactic_unate` (stops at the first
    binate for polarity; ``n_unique_variables`` is still a full count).

    Also runs an untimed :func:`~high_k.rof.check_syntactic_rof` for
    cross-classification (``is_rof``, ``n_variables``). Occurrence
    counts come from the ROF pass so they match ``rof_check`` logs.
    """
    from high_k.rof import check_syntactic_rof
    from high_k.unate import check_syntactic_unate

    def _import():
        return import_detached_rule(bnet_text, fmt="bnet", target=target)

    t_import, rule = median_time(_import, repeats=repeats)
    assert isinstance(rule, DetachedRule)
    k = rule.k

    def _unate():
        return check_syntactic_unate(expr)

    t_unate, unate = median_time(_unate, repeats=repeats)
    assert isinstance(unate, dict)

    rof = check_syntactic_rof(expr)

    return {
        "k": k,
        "t_import": t_import,
        "t_unate": t_unate,
        "is_unate": bool(unate["is_unate"]),
        "is_rof": bool(rof["is_rof"]),
        "n_variables": int(rof["n_variables"]),  # type: ignore[arg-type]
        "n_unique_variables": int(rof["n_unique_variables"]),  # type: ignore[arg-type]
    }
