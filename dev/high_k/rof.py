# -*- coding: utf-8 -*-
"""Syntactic read-once (ROF) checks and prime-implicant count / generation.

**Syntactic ROF** means every identifier occurs exactly once in the written
expression / AST (``n_variables == n_unique_variables``; also called trivial
/ single-occurrence).  That is distinct from **semantic** read-once-ness: a
formula may become read-once only after algebraic rewriting (e.g.
``a & a`` ≡ ``a``), which this module does not attempt.

``is_rof`` in return dicts means *syntactic* ROF.

Prime count / generate use the syntactic-ROF product-over-AND / sum-over-OR
rules (AST is only the evaluation vehicle).  :func:`count_rof_primes` and
:func:`generate_rof_primes` assert syntactic ROF on every call.

AND prime generation streams recursively (no intermediate product lists).

Reuses:
    - :mod:`high_k.ast` for parse / invert / leaf walks
    - CANA :func:`~cana.function_outputs.expression.expression_identifiers`
    - CANA :func:`~cana.io.bnet.parse_bnet_update_rule` for ``.bnet`` lines
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union

from high_k.ast import (
    LogicNode,
    ast_variable_names,
    invert_ast,
    parse_to_ast,
)
from high_k.paths import ensure_cana_on_path

ensure_cana_on_path()

from cana.function_outputs.expression import (
    expression_identifiers,
    normalize_operators,
)
from cana.io.bnet import parse_bnet_update_rule

__all__ = [
    "check_syntactic_rof",
    "check_syntactic_rof_ast",
    "require_syntactic_rof_ast",
    "count_rof_primes",
    "generate_rof_primes",
    "count_on_off_primes",
    "iter_on_off_primes",
    "SafeNetworkPrimes",
]

Polarity = Literal["on", "off"]
PrimeDict = Dict[str, bool]


def _occurrence_stats(
    names: List[str],
) -> Dict[str, Union[bool, int]]:
    """Return ``is_rof``, ``n_variables``, ``n_unique_variables`` for *names*."""
    counts = Counter(names)
    return {
        "is_rof": all(c == 1 for c in counts.values()),
        "n_variables": len(names),
        "n_unique_variables": len(counts),
    }


def check_syntactic_rof(expr: str) -> Dict[str, Union[bool, int]]:
    """Return syntactic-ROF stats for expression *expr*.

    ``is_rof`` means *syntactic* (single occurrence), not semantic
    read-once after simplification.  Constants (no identifiers) are
    vacuously ROFs.
    """
    toks = expression_identifiers(normalize_operators(expr, "python"))
    return _occurrence_stats(list(toks))


def check_syntactic_rof_ast(
    node: Optional[LogicNode],
) -> Dict[str, Union[bool, int]]:
    """Return syntactic-ROF stats for an AST *node* (leaf occurrence walk)."""
    return _occurrence_stats(ast_variable_names(node))


def require_syntactic_rof_ast(node: Optional[LogicNode]) -> None:
    """Raise if *node* is not a syntactic ROF (single occurrence).

    Rejects formulas that might only be read-once after simplification;
    this engine never rewrites.
    """
    info = check_syntactic_rof_ast(node)
    if info["is_rof"]:
        return
    names = ast_variable_names(node)
    counts = Counter(names)
    repeated = sorted(v for v, c in counts.items() if c > 1)
    raise ValueError(
        "AST is not a syntactic ROF (single occurrence); "
        f"repeated variables={repeated} "
        f"(n_occurrences={info['n_variables']}, "
        f"n_unique={info['n_unique_variables']}). "
        "This method does not simplify; a formula that is read-once "
        "only after rewriting (semantic ROF) is not accepted."
    )


def count_rof_primes(node: Optional[LogicNode]) -> int:
    """Count primes of a syntactic-ROF formula (no enumeration).

    Asserts syntactic ROF.  Formulas that are only semantic ROF after
    simplification are rejected.  The AST is only the evaluation vehicle.
    """
    require_syntactic_rof_ast(node)
    return _count_rof_primes(node)


def _count_rof_primes(node: Optional[LogicNode]) -> int:
    if node is None:
        return 0
    if node.op in ("LITERAL", "NEGATED_LITERAL"):
        return 1
    if node.op == "OR":
        assert isinstance(node.children, list)
        return sum(_count_rof_primes(child) for child in node.children)
    if node.op == "AND":
        assert isinstance(node.children, list)
        total = 1
        for child in node.children:
            total *= _count_rof_primes(child)
        return total
    return 0


def generate_rof_primes(
    node: Optional[LogicNode],
) -> Iterator[PrimeDict]:
    """Yield ON-set (or dualized OFF-set) primes as ``{var: True/False}``.

    Asserts syntactic ROF (AND merge via ``dict.update`` assumes disjoint
    variable sets).  Formulas that are only semantic ROF after simplification
    are rejected.  The AST is only the evaluation vehicle.

    AND children are combined recursively without materializing full child
    lists via ``itertools.product``.
    """
    require_syntactic_rof_ast(node)
    yield from _generate_rof_primes(node)


def _generate_rof_primes(
    node: Optional[LogicNode],
) -> Iterator[PrimeDict]:
    if node is None:
        return

    if node.op == "LITERAL":
        assert isinstance(node.children, str)
        yield {node.children: True}
        return

    if node.op == "NEGATED_LITERAL":
        assert isinstance(node.children, str)
        yield {node.children: False}
        return

    if node.op == "OR":
        assert isinstance(node.children, list)
        for child in node.children:
            yield from _generate_rof_primes(child)
        return

    if node.op == "AND":
        assert isinstance(node.children, list)

        def combine(
            children: List[LogicNode], prefix: Optional[PrimeDict] = None
        ) -> Iterator[PrimeDict]:
            if prefix is None:
                prefix = {}
            if not children:
                yield prefix
                return
            for implicant in _generate_rof_primes(children[0]):
                merged = prefix.copy()
                merged.update(implicant)
                yield from combine(children[1:], merged)

        yield from combine(node.children)


def _constant_ast(
    expr: str,
) -> Optional[Tuple[LogicNode, int, Optional[LogicNode], int]]:
    """Return ``(ast_on, on_count, ast_off, off_count)`` for 0/1 constants."""
    if expr in ("1", "True", "true"):
        return LogicNode("LITERAL", f"CONSTANT_{expr}"), 1, None, 0
    if expr in ("0", "False", "false"):
        # Treat as a single ON literal, no OFF dual.
        return LogicNode("LITERAL", f"CONSTANT_{expr}"), 1, None, 0
    return None


def count_on_off_primes(expr: str) -> Tuple[int, int]:
    """Return ``(n_on_primes, n_off_primes)`` for expression *expr*.

    *expr* must be a syntactic ROF; asserted via :func:`count_rof_primes`.
    """
    expr = expr.strip()
    const = _constant_ast(expr)
    if const is not None:
        return const[1], const[3]
    ast_on = parse_to_ast(expr)
    ast_off = invert_ast(ast_on) if ast_on is not None else None
    return count_rof_primes(ast_on), count_rof_primes(ast_off)


def iter_on_off_primes(
    expr: str,
    polarity: Polarity = "on",
) -> Iterator[PrimeDict]:
    """Yield prime implicants for the ON or OFF set of *expr*.

    *expr* must be a syntactic ROF; asserted via :func:`generate_rof_primes`.
    """
    expr = expr.strip()
    const = _constant_ast(expr)
    if const is not None:
        ast_on, _, ast_off, _ = const
        node = ast_on if polarity == "on" else ast_off
        yield from generate_rof_primes(node)
        return
    ast_on = parse_to_ast(expr)
    if polarity == "on":
        yield from generate_rof_primes(ast_on)
        return
    if ast_on is None:
        return
    yield from generate_rof_primes(invert_ast(ast_on))


class SafeNetworkPrimes:
    """Parse a ``.bnet`` string/path and expose per-node PI counts / generators.

    Count ON/OFF primes eagerly; expose lazy generators so callers can skip
    huge enumerations.

    Each update rule must be a syntactic ROF; enforced by
    :func:`count_rof_primes` / :func:`generate_rof_primes`.
    """

    def __init__(self, bnet_content_or_path: Union[str, Path]) -> None:
        self.nodes: Dict[str, dict] = {}
        raw_lines: List[str]
        if isinstance(bnet_content_or_path, Path):
            raw_lines = bnet_content_or_path.read_text().splitlines()
        else:
            text = str(bnet_content_or_path)
            # Avoid Path.is_file() on multi-line / huge strings (ENAMETOOLONG).
            looks_like_path = "\n" not in text and len(text) < 4096
            if looks_like_path and Path(text).is_file():
                raw_lines = Path(text).read_text().splitlines()
            else:
                raw_lines = text.strip().split("\n")
        self._parse_bnet(raw_lines)

    def _parse_bnet(self, lines: List[str]) -> None:
        for line in lines:
            rule = parse_bnet_update_rule(line)
            if rule is None:
                continue
            var_name, expr = rule
            const = _constant_ast(expr)
            if const is not None:
                ast_on, on_count, ast_off, off_count = const
            else:
                ast_on = parse_to_ast(expr)
                ast_off = invert_ast(ast_on) if ast_on is not None else None
                on_count = count_rof_primes(ast_on)
                off_count = count_rof_primes(ast_off)

            self.nodes[var_name] = {
                "expression": expr,
                "ast_on": ast_on,
                "ast_off": ast_off,
                "on_count": on_count,
                "off_count": off_count,
            }

    def get_on_generator(self, var_name: str) -> Optional[Iterator[PrimeDict]]:
        data = self.nodes.get(var_name)
        if data is None or data["ast_on"] is None:
            return None
        return generate_rof_primes(data["ast_on"])

    def get_off_generator(self, var_name: str) -> Optional[Iterator[PrimeDict]]:
        data = self.nodes.get(var_name)
        if data is None or data["ast_off"] is None:
            return None
        return generate_rof_primes(data["ast_off"])

    def summary(self) -> None:
        print(
            f"{'Node Variable':<15} | {'ON-Primes':<12} | "
            f"{'OFF-Primes':<12} | {'Equation Structure'}"
        )
        print("-" * 80)
        for var, data in self.nodes.items():
            on_c = data["on_count"]
            off_c = data["off_count"]
            on_str = f"{on_c:e}" if on_c > 10**6 else f"{on_c:,}"
            off_str = f"{off_c:e}" if off_c > 10**6 else f"{off_c:,}"
            print(
                f"{var:<15} | {on_str:<12} | {off_str:<12} | "
                f"{data['expression']}"
            )
