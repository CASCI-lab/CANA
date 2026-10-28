# -*- coding: utf-8 -*-
"""Timing helpers for high-*k* CANA benchmarks."""
from __future__ import annotations

import statistics
import time
from typing import Callable, Dict, List, Optional, Tuple

from high_k.paths import ensure_cana_on_path

ensure_cana_on_path()

from cana.function_outputs import FunctionOutputs
from cana.io.bnet import parse_bnet

__all__ = [
    "median_time",
    "fmt_s",
    "import_target_outputs",
    "time_expr_lut_one_symbol",
    "time_import_and_rof",
]


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


def import_target_outputs(bnet_text: str, target: str) -> FunctionOutputs:
    """Parse bnet → detached expression-only ``FunctionOutputs`` for *target*."""
    nodes = parse_bnet(bnet_text)
    by_name = {n.name: n for n in nodes}
    if target not in by_name:
        raise KeyError(f"target {target!r} not in parsed nodes")
    node = by_name[target]
    fo = node.outputs
    if not isinstance(fo, FunctionOutputs) or fo.expression is None:
        raise RuntimeError("expected expression-only import")
    return FunctionOutputs(expression=fo.expression, k=fo.k)


def time_expr_lut_one_symbol(
    bnet_text: str,
    target: str,
    *,
    repeats: int = 3,
    do_one_symbol: bool = True,
) -> Dict[str, object]:
    """Median times for import → LUT → one-symbol on *target*.

    Each stage starts from a fresh :func:`import_target_outputs` so peer
    caches are not shared across stages.
    """

    def _import():
        return import_target_outputs(bnet_text, target)

    t_import, fo0 = median_time(_import, repeats=repeats)
    assert isinstance(fo0, FunctionOutputs)
    k = fo0.k

    def _lut():
        fo = import_target_outputs(bnet_text, target)
        return fo.build_bit_lut_from_expression()

    t_lut, lut = median_time(_lut, repeats=repeats)

    t_one = None
    n_on = n_off = None
    if do_one_symbol:

        def _one():
            fo = import_target_outputs(bnet_text, target)
            fo.build_bit_lut_from_expression()
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
    """Median times for CANA import and a syntactic ROF check on *expr*.

    Import uses :func:`import_target_outputs` (same path as the LUT notebook).
    The ROF check runs on the original named *expr* string via
    :func:`~high_k.rof.check_syntactic_rof` (independent of the import result).
    """
    from high_k.rof import check_syntactic_rof

    def _import():
        return import_target_outputs(bnet_text, target)

    t_import, fo = median_time(_import, repeats=repeats)
    assert isinstance(fo, FunctionOutputs)
    k = fo.k

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
