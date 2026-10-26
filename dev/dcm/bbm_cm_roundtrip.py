# %% [markdown]
# BBM Canalizing Map ↔ LUT roundtrip
#
# Load Boolean models from the BBM corpus, build a canalizing map for every
# function via `BooleanNode.canalizing_map`, reconstruct the LUT with
# `cana.canalization.cm_lut.lut_from_canalizing_map`, and report:
#
# * fraction of **functions** that fail the roundtrip
# * fraction of **models** that fail (any evaluated function fails)
#
# Set `SMOKE = True` to run only the smallest models (default).  Logs are
# written under `dev/logs/`; smoke runs use a `_smoke_` filename and
# `mode=smoke` in the header.
#
# `MAX_K` caps both LUT expansion at load and CM roundtrip.  Nodes with more
# inputs are skipped without building a ``2^k`` table.
#
# `TIMEOUT_CM_S` / `TIMEOUT_LUT_S` skip a function if canalizing-map build or
# LUT reconstruction exceeds the limit (``None`` or ``<= 0`` disables).

# %%
from __future__ import annotations

import csv
import multiprocessing as mp
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO, Tuple

_DEV_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _DEV_DIR.parent  # .../CANA/CANA (contains the cana package)
for _p in (_DEV_DIR, _PKG_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cana.boolean_node import BooleanNode
from cana.canalization.cm_lut import lut_from_canalizing_map
from cana.utils import output_transitions

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SMOKE = False
SMOKE_N = 5
# Skip LUT build + CM for functions with more than this many inputs.
MAX_K = 12
# Wall-clock limits per function step; skip on expiry. None / <=0 disables.
TIMEOUT_CM_S: Optional[float] = 30.0
TIMEOUT_LUT_S: Optional[float] = 30.0

# .../CANA/CANA/dev → parents[1] = workspace root (.../CANA) containing BBM/
BBM_MODELS = _DEV_DIR.resolve().parents[1] / "BBM" / "models"
SUMMARY_CSV = BBM_MODELS / "summary.csv"
LOG_DIR = _DEV_DIR / "logs"

_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
_LOG_NAME = (
    f"cm_bbm_roundtrip_smoke_{_TIMESTAMP}.log"
    if SMOKE
    else f"cm_bbm_roundtrip_{_TIMESTAMP}.log"
)
LOG_PATH = LOG_DIR / _LOG_NAME

# %% [markdown]
# ## Helpers — logging, summary selection, roundtrip


# %%
class TeeLogger:
    """Write the same line to stdout and a log file."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: TextIO = path.open("w", encoding="utf-8")
        self.path = path

    def info(self, msg: str = "") -> None:
        print(msg)
        self._fh.write(msg + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def load_summary(csv_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, skipinitialspace=True)
        for raw in reader:
            row = {
                "ID": raw["ID"].strip(),
                "name": raw["name"].strip(),
                "variables": int(raw["variables"]),
                "inputs": int(raw["inputs"]),
                "regulations": int(raw["regulations"]),
            }
            row["total"] = row["variables"] + row["inputs"]
            rows.append(row)
    return rows


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
        raise FileNotFoundError(f"No model directory for id={pad} under {models_root}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple directories for id={pad}: {matches}")
    return matches[0]


def select_models(rows: List[Dict[str, Any]], smoke: bool, smoke_n: int) -> List[Dict[str, Any]]:
    if not smoke:
        return list(rows)
    ordered = sorted(
        rows,
        key=lambda r: (r["total"], r["variables"], r["regulations"], int(r["ID"])),
    )
    return ordered[:smoke_n]


@dataclass
class LoadedNode:
    """One booleannet rule, optionally with a built ``BooleanNode`` LUT."""

    name: str
    k: int
    node: Optional[BooleanNode] = None
    skip_reason: Optional[str] = None


def parse_booleannet_rules(text: str) -> List[Tuple[str, str]]:
    """Parse booleannet lines into ``(name, expression)`` in file order.

    Accepts both ``name*=expr`` and BBM's spaced form ``name* = expr``,
    matching CANA's ``split('*')`` / ``split('=')`` parsing.
    """
    rules: List[Tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "*" not in line or "=" not in line:
            continue
        name = line.split("*", 1)[0].strip()
        expr = line.split("=", 1)[1].strip()
        if not name:
            continue
        rules.append((name, expr))
    return rules


def load_model_nodes(path: Path, max_k: int) -> List[LoadedNode]:
    """Load a booleannet model, building LUTs only for nodes with ``k <= max_k``.

    Mirrors CANA's ``from_string_boolean`` input detection, but skips
    ``output_transitions`` (full ``2^k`` expansion) when ``k > max_k`` so
    high-indegree models (e.g. ERBB ``v_Shc`` ≈ 27) do not hang at load.
    """
    rules = parse_booleannet_rules(path.read_text(encoding="utf-8"))
    names = [name for name, _ in rules]
    name_to_id = {name: i for i, name in enumerate(names)}

    loaded: List[LoadedNode] = []
    for nid, (name, expr) in enumerate(rules):
        input_names = [
            other
            for other in names
            if re.compile(r"\W" + re.escape(other) + r"\W").search("*" + expr + "*")
        ]
        k = len(input_names)
        if k > max_k:
            loaded.append(
                LoadedNode(
                    name=name,
                    k=k,
                    skip_reason=f"k={k} > MAX_K={max_k} (LUT not built)",
                )
            )
            continue
        outputs = output_transitions(expr, input_names)
        input_ids = [name_to_id[inp] for inp in input_names]
        node = BooleanNode.from_output_list(
            outputs, name=name, inputs=input_ids, id=nid
        )
        loaded.append(LoadedNode(name=name, k=k, node=node))
    return loaded


def is_exact_roundtrip(original_outputs, result) -> bool:
    return (
        not result.has_conflict
        and not result.is_incomplete
        and [int(x) for x in original_outputs] == result.outputs
    )


@dataclass
class NodeRoundtrip:
    name: str
    k: int
    exact: bool = False
    conflict: bool = False
    incomplete: bool = False
    skipped: bool = False
    skip_reason: Optional[str] = None
    error: Optional[str] = None
    fail_reason: Optional[str] = None
    cm_s: Optional[float] = None
    lut_s: Optional[float] = None


def _fmt_secs(seconds: Optional[float]) -> str:
    if seconds is None:
        return "n/a"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f}us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f}ms"
    return f"{seconds:.3f}s"


class StepTimeout(Exception):
    """Raised when a timed step exceeds TIMEOUT_*_S."""


def _cm_worker(
    payload: Tuple[Any, ...], conn: Any
) -> None:
    """Child process: build CM (Cython QM must be killable via terminate)."""
    try:
        name, nid, inputs, outputs, builder = payload
        node = BooleanNode(
            id=nid,
            name=name,
            k=len(inputs),
            inputs=list(inputs),
            outputs=list(outputs),
            network=None,
        )
        cm = node.canalizing_map(builder=builder)
        conn.send(("ok", cm))
    except Exception as exc:  # noqa: BLE001
        conn.send(("err", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def _lut_worker(payload: Tuple[Any, ...], conn: Any) -> None:
    """Child process: reconstruct LUT from a pickled CM."""
    try:
        cm, input_order = payload
        result = lut_from_canalizing_map(cm, input_order=input_order)
        conn.send(("ok", result))
    except Exception as exc:  # noqa: BLE001
        conn.send(("err", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def _run_in_process(
    worker: Callable[..., None],
    payload: Tuple[Any, ...],
    timeout_s: Optional[float],
) -> Any:
    """Run *worker(payload, conn)* in a child; kill on timeout.

    ``SIGALRM`` cannot interrupt Cython Quine–McCluskey, so CM (and timed LUT)
    steps use a subprocess that can be ``terminate``d.
    """
    if timeout_s is None or timeout_s <= 0:
        # No timeout: call worker logic in-process via a local pipe simulation
        # is awkward; callers use the direct path instead.
        raise ValueError("_run_in_process requires a positive timeout")

    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=worker, args=(payload, child_conn))
    proc.start()
    child_conn.close()
    proc.join(float(timeout_s))
    if proc.is_alive():
        proc.terminate()
        proc.join(5.0)
        if proc.is_alive():
            proc.kill()
            proc.join(1.0)
        parent_conn.close()
        raise StepTimeout(
            f"step exceeded wall-clock limit ({timeout_s}s)"
        )
    if not parent_conn.poll():
        parent_conn.close()
        raise RuntimeError(
            f"worker exited without result (exitcode={proc.exitcode})"
        )
    status, payload_out = parent_conn.recv()
    parent_conn.close()
    if status == "ok":
        return payload_out
    raise RuntimeError(str(payload_out))


def roundtrip_node(
    node: BooleanNode,
    max_k: int,
    timeout_cm_s: Optional[float] = None,
    timeout_lut_s: Optional[float] = None,
    builder: str = "legacy",
) -> NodeRoundtrip:
    out = NodeRoundtrip(name=node.name, k=int(node.k))
    if out.k > max_k:
        out.skipped = True
        out.skip_reason = f"k={out.k} > MAX_K={max_k}"
        return out
    try:
        t0 = time.perf_counter()
        try:
            if timeout_cm_s is not None and timeout_cm_s > 0:
                cm = _run_in_process(
                    _cm_worker,
                    (
                        node.name,
                        node.id,
                        list(node.inputs),
                        list(node.outputs),
                        builder,
                    ),
                    timeout_cm_s,
                )
            else:
                cm = node.canalizing_map(builder=builder)
        except StepTimeout:
            out.cm_s = time.perf_counter() - t0
            out.skipped = True
            out.skip_reason = (
                f"TIMEOUT_CM after {_fmt_secs(out.cm_s)} "
                f"(limit={_fmt_secs(timeout_cm_s)})"
            )
            return out
        out.cm_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        try:
            if timeout_lut_s is not None and timeout_lut_s > 0:
                result = _run_in_process(
                    _lut_worker,
                    (cm, list(node.inputs)),
                    timeout_lut_s,
                )
            else:
                result = lut_from_canalizing_map(cm, input_order=node.inputs)
        except StepTimeout:
            out.lut_s = time.perf_counter() - t1
            out.skipped = True
            out.skip_reason = (
                f"TIMEOUT_LUT after {_fmt_secs(out.lut_s)} "
                f"(limit={_fmt_secs(timeout_lut_s)})"
            )
            return out
        out.lut_s = time.perf_counter() - t1

        out.conflict = bool(result.has_conflict)
        out.incomplete = bool(result.is_incomplete)
        out.exact = is_exact_roundtrip(node.outputs, result)
        if not out.exact:
            if out.conflict:
                out.fail_reason = "conflict"
            elif out.incomplete:
                out.fail_reason = "incomplete"
            else:
                out.fail_reason = "mismatch"
    except Exception as exc:  # noqa: BLE001 — log and continue corpus scan
        out.error = f"{type(exc).__name__}: {exc}"
        out.fail_reason = f"exception: {out.error}"
    return out


def _timing_suffix(nr: NodeRoundtrip) -> str:
    return f"  cm={_fmt_secs(nr.cm_s)}  lut={_fmt_secs(nr.lut_s)}"


@dataclass
class ModelResult:
    model_id: str
    name: str
    path: str
    n_nodes: int = 0
    n_eval: int = 0
    n_skip: int = 0
    n_fail: int = 0
    n_error: int = 0
    failed: bool = False
    load_error: Optional[str] = None
    node_results: List[NodeRoundtrip] = field(default_factory=list)

# %% [markdown]
# ## Run corpus (smoke or full)
#
# Call ``main()`` from a notebook cell, or run the file as a script.


# %%
def main() -> None:
    log = TeeLogger(LOG_PATH)

    mode = "smoke" if SMOKE else "full"
    log.info(f"mode={mode}")
    if SMOKE:
        log.info(
            f"smoke_n={SMOKE_N}  "
            "(smallest by total=variables+inputs, then variables, regulations)"
        )
    log.info(f"MAX_K={MAX_K}")
    log.info(f"TIMEOUT_CM_S={TIMEOUT_CM_S}")
    log.info(f"TIMEOUT_LUT_S={TIMEOUT_LUT_S}")
    log.info(f"BBM_MODELS={BBM_MODELS}")
    log.info(f"SUMMARY_CSV={SUMMARY_CSV}")
    log.info(f"log_path={LOG_PATH}")
    log.info(f"timestamp={_TIMESTAMP}")
    log.info("")

    rows = load_summary(SUMMARY_CSV)
    selected = select_models(rows, smoke=SMOKE, smoke_n=SMOKE_N)
    log.info(f"selected_models={len(selected)} / summary_total={len(rows)}")
    for r in selected:
        log.info(
            f"  id={r['ID']}  name={r['name']}  "
            f"total={r['total']}  var={r['variables']}  in={r['inputs']}  "
            f"reg={r['regulations']}"
        )
    log.info("")

    model_results: List[ModelResult] = []
    n_fn_eval = 0
    n_fn_fail = 0
    n_model_eval = 0
    n_model_fail = 0
    failing_models: List[str] = []
    total_cm_s = 0.0
    total_lut_s = 0.0

    for row in selected:
        model_id = row["ID"]
        name = row["name"]
        try:
            model_dir = resolve_model_dir(BBM_MODELS, model_id)
            bnet_path = model_dir / "model.booleannet.txt"
            if not bnet_path.is_file():
                raise FileNotFoundError(f"missing {bnet_path}")
            loaded_nodes = load_model_nodes(bnet_path, max_k=MAX_K)
        except Exception as exc:  # noqa: BLE001
            mr = ModelResult(
                model_id=model_id,
                name=name,
                path="",
                load_error=f"{type(exc).__name__}: {exc}",
                failed=True,
            )
            model_results.append(mr)
            n_model_eval += 1
            n_model_fail += 1
            failing_models.append(f"{model_id} {name} (load_error)")
            log.info(
                f"=== model id={model_id} name={name} "
                f"LOAD FAILED: {mr.load_error} ==="
            )
            log.info("")
            continue

        mr = ModelResult(
            model_id=model_id,
            name=name,
            path=str(bnet_path),
            n_nodes=len(loaded_nodes),
        )
        n_load_skip = sum(1 for ln in loaded_nodes if ln.node is None)
        log.info(
            f"=== model id={model_id} name={name}  nodes={mr.n_nodes}  "
            f"lut_built={mr.n_nodes - n_load_skip}  load_skip={n_load_skip} ==="
        )
        log.info(f"  path={bnet_path}")

        for ln in loaded_nodes:
            if ln.node is None:
                nr = NodeRoundtrip(
                    name=ln.name,
                    k=ln.k,
                    skipped=True,
                    skip_reason=ln.skip_reason,
                )
                mr.node_results.append(nr)
                mr.n_skip += 1
                log.info(f"  SKIP  {nr.name}  k={nr.k}  ({nr.skip_reason})")
                continue

            nr = roundtrip_node(
                ln.node,
                max_k=MAX_K,
                timeout_cm_s=TIMEOUT_CM_S,
                timeout_lut_s=TIMEOUT_LUT_S,
            )
            mr.node_results.append(nr)
            if nr.skipped:
                mr.n_skip += 1
                if nr.cm_s is not None or nr.lut_s is not None:
                    log.info(
                        f"  SKIP  {nr.name}  k={nr.k}  ({nr.skip_reason})"
                        f"{_timing_suffix(nr)}"
                    )
                else:
                    log.info(f"  SKIP  {nr.name}  k={nr.k}  ({nr.skip_reason})")
                continue
            if nr.error is not None:
                mr.n_error += 1
                log.info(
                    f"  ERROR {nr.name}  k={nr.k}  {nr.error}{_timing_suffix(nr)}"
                )
                continue

            mr.n_eval += 1
            n_fn_eval += 1
            if nr.cm_s is not None:
                total_cm_s += nr.cm_s
            if nr.lut_s is not None:
                total_lut_s += nr.lut_s
            if not nr.exact:
                mr.n_fail += 1
                n_fn_fail += 1
                log.info(
                    f"  FAIL  {nr.name}  k={nr.k}  reason={nr.fail_reason}"
                    f"{_timing_suffix(nr)}"
                )
            else:
                log.info(f"  OK    {nr.name}  k={nr.k}{_timing_suffix(nr)}")

        n_model_eval += 1
        mr.failed = mr.n_fail > 0
        if mr.failed:
            n_model_fail += 1
            failing_models.append(f"{model_id} {name}")

        timed = [
            nr
            for nr in mr.node_results
            if nr.cm_s is not None or nr.lut_s is not None
        ]
        sum_cm = sum(nr.cm_s or 0.0 for nr in timed)
        sum_lut = sum(nr.lut_s or 0.0 for nr in timed)
        log.info(
            f"  model_summary  eval={mr.n_eval}  fail={mr.n_fail}  "
            f"skip={mr.n_skip}  error={mr.n_error}  model_fail={mr.failed}  "
            f"cm_total={_fmt_secs(sum_cm)}  lut_total={_fmt_secs(sum_lut)}"
        )
        log.info("")
        model_results.append(mr)

    fn_frac = (n_fn_fail / n_fn_eval) if n_fn_eval else float("nan")
    model_frac = (n_model_fail / n_model_eval) if n_model_eval else float("nan")

    log.info("=" * 60)
    log.info("SUMMARY")
    log.info(f"mode={mode}")
    log.info(f"function_fail_fraction = {n_fn_fail}/{n_fn_eval} = {fn_frac:.6f}")
    log.info(f"model_fail_fraction    = {n_model_fail}/{n_model_eval} = {model_frac:.6f}")
    log.info(f"cm_time_total          = {_fmt_secs(total_cm_s)}")
    log.info(f"lut_time_total         = {_fmt_secs(total_lut_s)}")
    if failing_models:
        log.info("failing_models:")
        for item in failing_models:
            log.info(f"  {item}")
    else:
        log.info("failing_models: (none)")
    log.info(f"log written to {LOG_PATH}")
    log.info("=" * 60)

    log.close()


if __name__ == "__main__":
    main()
