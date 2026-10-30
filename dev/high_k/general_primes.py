# -*- coding: utf-8 -*-
"""Exact ON/OFF prime implicants for general (possibly binate) expressions.

Fallbacks when syntactic ROF / unate shortcuts do not apply.  Each backend
returns the **full** prime-implicant set for ``f`` and ``~f`` (given enough
time and memory).  Espresso is intentionally omitted: PyEDA's Espresso
binding only yields a minimal SOP cover, not all primes.

Backends (separate APIs; no cascade):

1. CANA Quine–McCluskey over a dense LUT
2. PyEDA :meth:`~pyeda.boolalg.expr.Expression.complete_sum`
3. PyBoolNet ``bnet2primes``
4. BDD onset enumeration (``dd``) + Quine–McCluskey

All wall-clock limits use :func:`~high_k.timing.call_in_subprocess` so
Cython / C / external tools (and their grandchildren) can be killed via
process-group signals.  Dense LUT / onset→QM paths also refuse
``k > MAX_DENSE_LUT_K`` up front.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from high_k.bbm_rules import as_bnet, unique_regulators
from high_k.paths import ensure_cana_on_path
from high_k.primes_util import (
    PrimeDict,
    dnf_terms_to_primes,
    refuse_extreme_nesting,
    to_pyeda_expr_str,
    to_symbolic_expr,
)

ensure_cana_on_path()

__all__ = [
    "DEFAULT_TIMEOUT_S",
    "EXPLORATION_TIMEOUT_S",
    "MAX_DENSE_LUT_K",
    "qm_on_off_primes",
    "complete_sum_on_off_primes",
    "pyboolnet_on_off_primes",
    "bdd_on_off_primes",
]

DEFAULT_TIMEOUT_S = 10.0
EXPLORATION_TIMEOUT_S = 1.0
# Dense LUT / onset→QM paths allocate Θ(2^k) memory.  Refuse above this *k*
# rather than relying on a wall-clock kill after the allocation attempt.
MAX_DENSE_LUT_K = 20

OnOffPrimes = Tuple[List[PrimeDict], List[PrimeDict]]


def _run_timed(
    inprocess: Callable[[str], OnOffPrimes],
    worker: Callable[..., None],
    expr: str,
    *,
    timeout_s: float,
) -> OnOffPrimes:
    """Run *inprocess* directly or *worker* under a wall-clock kill."""
    raw = expr.strip()
    # Parent-side: fail before spawn so a short TIMEOUT_S cannot mis-label
    # extreme nesting as ``timeout`` (kill during parse).
    refuse_extreme_nesting(raw)
    if timeout_s <= 0:
        return inprocess(raw)
    from high_k.timing import call_in_subprocess

    return call_in_subprocess(worker, (raw,), timeout_s=timeout_s)


def _constant_on_off(raw: str) -> Optional[OnOffPrimes]:
    """Handle empty / constant expressions; ``None`` if not a constant."""
    if raw in ("",):
        return [], []
    if raw in ("1", "True", "true"):
        return [{}], []
    if raw in ("0", "False", "false"):
        return [], [{}]
    return None


def _for_cana_eval(expr: str) -> str:
    """Rewrite ``~`` to ``!`` so CANA LUT eval uses logical (not bitwise) NOT."""
    return to_symbolic_expr(expr).replace("~", "!")


def _refuse_huge_dense_lut(expr: str) -> None:
    """Raise ``ValueError`` when dense ``2**k`` materialization is unsafe."""
    names = unique_regulators(_for_cana_eval(expr))
    k = len(names)
    if k > MAX_DENSE_LUT_K:
        raise ValueError(
            f"refusing dense LUT/QM for k={k} > MAX_DENSE_LUT_K={MAX_DENSE_LUT_K}"
        )


# ---------------------------------------------------------------------------
# 1. CANA Quine–McCluskey
# ---------------------------------------------------------------------------


def _schemata_to_primes(
    sset, names: List[str]
) -> List[PrimeDict]:
    out: List[PrimeDict] = []
    for s in sset:
        d: PrimeDict = {}
        for i, ch in enumerate(s):
            if ch == "1":
                d[names[i]] = True
            elif ch == "0":
                d[names[i]] = False
        out.append(d)
    return out


def _qm_on_off_primes_inprocess(expr: str) -> OnOffPrimes:
    """Build dense LUT and run Cython QM → ``(on_primes, off_primes)``."""
    from high_k.timing import import_detached_outputs

    const = _constant_on_off(expr)
    if const is not None:
        return const

    refuse_extreme_nesting(expr)
    _refuse_huge_dense_lut(expr)
    cana_expr = _for_cana_eval(expr)
    names = unique_regulators(cana_expr)
    fo = import_detached_outputs(as_bnet(cana_expr), "Y")
    fo.build_bit_lut_from_positional_expression()
    schemata = fo.compute_one_symbol_schemata_from_bit_lut()
    return (
        _schemata_to_primes(schemata.get(1, ()), names),
        _schemata_to_primes(schemata.get(0, ()), names),
    )


def _qm_worker(expr: str, conn) -> None:
    try:
        conn.send(("ok", _qm_on_off_primes_inprocess(expr)))
    except Exception as exc:  # noqa: BLE001
        conn.send(("err", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def qm_on_off_primes(
    expr: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> OnOffPrimes:
    """Exact ON/OFF primes via dense LUT + CANA Quine–McCluskey.

    Refuses expressions with more than :data:`MAX_DENSE_LUT_K` inputs
    (predictable ``2**k`` RAM).  Otherwise *timeout_s* and subprocess kill
    bound wall-clock time.  *timeout_s* ``<= 0`` runs in-process
    (tests / debug).
    """
    return _run_timed(
        _qm_on_off_primes_inprocess,
        _qm_worker,
        expr,
        timeout_s=timeout_s,
    )


# ---------------------------------------------------------------------------
# 2. PyEDA complete_sum (all primes)
# ---------------------------------------------------------------------------


def _complete_sum_on_off_primes_inprocess(expr: str) -> OnOffPrimes:
    from pyeda.inter import expr as pyeda_expr

    const = _constant_on_off(expr)
    if const is not None:
        return const

    refuse_extreme_nesting(expr)
    f = pyeda_expr(to_pyeda_expr_str(expr))
    on_cs = f.complete_sum()
    off_cs = (~f).complete_sum()
    return dnf_terms_to_primes(on_cs), dnf_terms_to_primes(off_cs)


def _complete_sum_worker(expr: str, conn) -> None:
    try:
        conn.send(("ok", _complete_sum_on_off_primes_inprocess(expr)))
    except Exception as exc:  # noqa: BLE001
        conn.send(("err", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def complete_sum_on_off_primes(
    expr: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> OnOffPrimes:
    """Exact ON/OFF primes via pyeda ``complete_sum`` on ``f`` and ``~f``.

    Raises:
        ImportError: if pyeda is not installed.
        TimeoutError: if wall-clock time exceeds *timeout_s*.
    """
    import pyeda  # noqa: F401

    return _run_timed(
        _complete_sum_on_off_primes_inprocess,
        _complete_sum_worker,
        expr,
        timeout_s=timeout_s,
    )


# ---------------------------------------------------------------------------
# 3. PyBoolNet
# ---------------------------------------------------------------------------


def _pyboolnet_cube_to_prime(cube: dict) -> PrimeDict:
    """Convert PyBoolNet ``{var: 0|1}`` cube to ``PrimeDict``."""
    out: PrimeDict = {}
    for var, bit in cube.items():
        if bit in (1, True, "1"):
            out[str(var)] = True
        elif bit in (0, False, "0"):
            out[str(var)] = False
        else:
            raise ValueError(
                f"unexpected PyBoolNet literal {var!r}={bit!r}"
            )
    return out


def _pyboolnet_on_off_primes_inprocess(expr: str) -> OnOffPrimes:
    from pyboolnet.file_exchange import bnet2primes

    const = _constant_on_off(expr)
    if const is not None:
        return const

    # PyBoolNet .bnet uses ``!`` for negation (not ``~``).
    primes = bnet2primes(as_bnet(_for_cana_eval(expr), target="Y"))
    if "Y" not in primes:
        raise RuntimeError(
            f"PyBoolNet primes missing target Y; keys={list(primes)}"
        )
    node = primes["Y"]
    # PyBoolNet 3.x: ``[off_cubes, on_cubes]``. Older docs used ``{0: …, 1: …}``.
    if isinstance(node, dict):
        on_raw = node.get(1, node.get("1", []))
        off_raw = node.get(0, node.get("0", []))
    elif isinstance(node, (list, tuple)) and len(node) >= 2:
        off_raw, on_raw = node[0], node[1]
    else:
        raise RuntimeError(
            f"unexpected PyBoolNet primes shape for Y: {type(node)!r}"
        )
    return (
        [_pyboolnet_cube_to_prime(c) for c in on_raw],
        [_pyboolnet_cube_to_prime(c) for c in off_raw],
    )


def _pyboolnet_worker(expr: str, conn) -> None:
    try:
        conn.send(("ok", _pyboolnet_on_off_primes_inprocess(expr)))
    except Exception as exc:  # noqa: BLE001
        conn.send(("err", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def pyboolnet_on_off_primes(
    expr: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> OnOffPrimes:
    """Exact ON/OFF primes via PyBoolNet ``bnet2primes``.

    PyBoolNet shells out to ``BNetToPrime``; the timed worker runs in a new
    process group so a timeout kills that binary too (not only the Python
    wrapper).

    Raises:
        ImportError: if pyboolnet is not installed.
        TimeoutError: if wall-clock time exceeds *timeout_s*.
    """
    import pyboolnet  # noqa: F401

    return _run_timed(
        _pyboolnet_on_off_primes_inprocess,
        _pyboolnet_worker,
        expr,
        timeout_s=timeout_s,
    )


# ---------------------------------------------------------------------------
# 4. BDD onset → Quine–McCluskey (dd)
# ---------------------------------------------------------------------------


def _bdd_to_dd_expr(expr: str) -> str:
    """Symbolic ``&`` / ``|`` / ``!`` → dd ``add_expr`` syntax (``~``)."""
    return to_symbolic_expr(expr).replace("!", "~")


def _assignment_to_bitstring(assignment: dict, names: List[str]) -> str:
    """Map a full ``{var: bool}`` assignment to a ``0``/``1`` string."""
    return "".join("1" if assignment[n] else "0" for n in names)


def _bitstring_primes_to_dicts(
    implicants: set, names: List[str]
) -> List[PrimeDict]:
    """Convert QM bitstrings (``0``/``1``/``2``) to ``PrimeDict`` list."""
    out: List[PrimeDict] = []
    for s in implicants:
        d: PrimeDict = {}
        for i, ch in enumerate(s):
            if ch == "1":
                d[names[i]] = True
            elif ch == "0":
                d[names[i]] = False
        out.append(d)
    return out


def _bdd_polarity_primes(bdd, node, names: List[str]) -> List[PrimeDict]:
    """Exact primes for one polarity: BDD onset minterms → Cython QM."""
    from cana.canalization.cboolean_canalization import find_implicants_qm

    if node == bdd.false:
        return []
    if node == bdd.true:
        return [{}]

    # ``pick_iter`` yields assignments over the node's *support* only.  Running
    # QM in that subspace avoids KeyError on don't-care inputs and avoids
    # exploding 2^k enumerations via ``care_vars=all``.  Primes only mention
    # support literals; absent inputs are implicit don't-cares.
    support = set(bdd.support(node))
    support_names = [n for n in names if n in support]
    if not support_names:
        return [{}]
    minterms = {
        _assignment_to_bitstring(a, support_names)
        for a in bdd.pick_iter(node)
    }
    if not minterms:
        return []
    if len(minterms) == (1 << len(support_names)):
        return [{}]
    return _bitstring_primes_to_dicts(
        find_implicants_qm(minterms), support_names
    )


def _bdd_on_off_primes_inprocess(expr: str) -> OnOffPrimes:
    from dd.autoref import BDD

    const = _constant_on_off(expr)
    if const is not None:
        return const

    refuse_extreme_nesting(expr)
    # Onset enumeration + QM is Θ(2^k) in the worst case; same guard as LUT QM.
    _refuse_huge_dense_lut(expr)
    names = unique_regulators(_for_cana_eval(expr))
    manager = BDD()
    for name in names:
        manager.declare(name)

    f = manager.add_expr(_bdd_to_dd_expr(expr))
    return (
        _bdd_polarity_primes(manager, f, names),
        _bdd_polarity_primes(manager, ~f, names),
    )


def _bdd_worker(expr: str, conn) -> None:
    try:
        conn.send(("ok", _bdd_on_off_primes_inprocess(expr)))
    except Exception as exc:  # noqa: BLE001
        conn.send(("err", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def bdd_on_off_primes(
    expr: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> OnOffPrimes:
    """Exact ON/OFF primes via BDD onset enumeration + Quine–McCluskey.

    Compiles *expr* with ``dd``, enumerates satisfying assignments, then
    runs CANA's Cython QM.  (Textbook Morreale recursion on BDDs does not
    produce the full prime set without extra machinery.)

    Refuses expressions with more than :data:`MAX_DENSE_LUT_K` inputs
    (worst-case onset size ``2**k``).

    Raises:
        ImportError: if dd is not installed.
        ValueError: if *k* exceeds :data:`MAX_DENSE_LUT_K`.
        TimeoutError: if wall-clock time exceeds *timeout_s*.
    """
    import dd  # noqa: F401

    return _run_timed(
        _bdd_on_off_primes_inprocess,
        _bdd_worker,
        expr,
        timeout_s=timeout_s,
    )
