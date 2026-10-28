# -*- coding: utf-8 -*-
"""AST-based prime-implicant count / generation for read-once formulas.

Migrated from the deprecated ``AST.ipynb`` / ``AST_2.ipynb`` engines.  For
syntactic ROFs (and more generally unate formulas already in DNF/CNF-like
``&`` / ``|`` / ``!`` form), ON-set primes are the product over AND children
and the sum over OR children; OFF-set primes come from De Morgan dualization.

Reuses:
- CANA :func:`~cana.function_outputs.expression.normalize_operators` for
  python ↔ symbolic operator style
- CANA :func:`~cana.io.bnet.parse_bnet_update_rule` for ``.bnet`` lines
- :func:`~high_k.rof.check_syntactic_rof` when ``require_rof=True``

Generation follows the streaming AST_2 variant (no intermediate product lists).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union

from high_k.paths import ensure_cana_on_path

ensure_cana_on_path()

from cana.function_outputs.expression import normalize_operators
from cana.io.bnet import parse_bnet_update_rule

__all__ = [
    "LogicNode",
    "to_symbolic_expr",
    "parse_to_ast",
    "invert_ast",
    "count_prime_implicants",
    "generate_ast_primes",
    "count_on_off_primes",
    "iter_on_off_primes",
    "SafeNetworkPrimes",
]

Polarity = Literal["on", "off"]
PrimeDict = Dict[str, bool]


class LogicNode:
    """Node in a boolean expression AST.

    ``op`` is one of ``AND``, ``OR``, ``LITERAL``, ``NEGATED_LITERAL``.
    For leaves, ``children`` is the variable name (``str``); for gates it is
    a list of child :class:`LogicNode` instances.
    """

    __slots__ = ("op", "children")

    def __init__(
        self,
        op: str,
        children: Union[str, List["LogicNode"]],
    ) -> None:
        self.op = op
        self.children = children


def to_symbolic_expr(expr: str) -> str:
    """Normalize *expr* to symbolic ``&`` / ``|`` / ``!`` form."""
    return normalize_operators(
        normalize_operators(expr, "python"), "symbolic"
    )


def parse_to_ast(expr_str: str) -> Optional[LogicNode]:
    """Parse a boolean expression into a :class:`LogicNode` AST.

    Accepts python (``and``/``or``/``not``) or symbolic (``&``/``|``/``!``)
    operators.  Splits on top-level ``|`` then ``&``, unwraps outer
    parentheses, and treats ``!`` as a prefix (with De Morgan for
    ``!(…)`` groups).
    """
    expr_str = to_symbolic_expr(expr_str).strip()
    if not expr_str:
        return None

    paren_depth = 0
    or_indices: List[int] = []
    for idx, char in enumerate(expr_str):
        if char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth -= 1
        elif char == "|" and paren_depth == 0:
            or_indices.append(idx)

    if or_indices:
        parts: List[str] = []
        last_idx = 0
        for idx in or_indices:
            parts.append(expr_str[last_idx:idx].strip())
            last_idx = idx + 1
        parts.append(expr_str[last_idx:].strip())
        return LogicNode("OR", [parse_to_ast(p) for p in parts])  # type: ignore[misc]

    paren_depth = 0
    and_indices: List[int] = []
    for idx, char in enumerate(expr_str):
        if char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth -= 1
        elif char == "&" and paren_depth == 0:
            and_indices.append(idx)

    if and_indices:
        parts = []
        last_idx = 0
        for idx in and_indices:
            parts.append(expr_str[last_idx:idx].strip())
            last_idx = idx + 1
        parts.append(expr_str[last_idx:].strip())
        return LogicNode("AND", [parse_to_ast(p) for p in parts])  # type: ignore[misc]

    if expr_str.startswith("(") and expr_str.endswith(")"):
        return parse_to_ast(expr_str[1:-1])

    if expr_str.startswith("!"):
        inner_content = expr_str[1:].strip()
        if inner_content.startswith("(") and inner_content.endswith(")"):
            inner = parse_to_ast(inner_content)
            assert inner is not None
            return invert_ast(inner)
        return LogicNode("NEGATED_LITERAL", inner_content)

    return LogicNode("LITERAL", expr_str)


def invert_ast(node: LogicNode) -> LogicNode:
    """Apply De Morgan dualization (swap AND/OR, flip literals)."""
    if node.op == "LITERAL":
        return LogicNode("NEGATED_LITERAL", node.children)
    if node.op == "NEGATED_LITERAL":
        return LogicNode("LITERAL", node.children)
    if node.op == "OR":
        assert isinstance(node.children, list)
        return LogicNode("AND", [invert_ast(child) for child in node.children])
    if node.op == "AND":
        assert isinstance(node.children, list)
        return LogicNode("OR", [invert_ast(child) for child in node.children])
    return node


def count_prime_implicants(node: Optional[LogicNode]) -> int:
    """Count primes implied by a unate / syntactic-ROF AST (no enumeration)."""
    if node is None:
        return 0
    if node.op in ("LITERAL", "NEGATED_LITERAL"):
        return 1
    if node.op == "OR":
        assert isinstance(node.children, list)
        return sum(count_prime_implicants(child) for child in node.children)
    if node.op == "AND":
        assert isinstance(node.children, list)
        total = 1
        for child in node.children:
            total *= count_prime_implicants(child)
        return total
    return 0


def generate_ast_primes(
    node: Optional[LogicNode],
) -> Iterator[PrimeDict]:
    """Yield ON-set (or dualized OFF-set) primes as ``{var: True/False}``.

    Streaming form from ``AST_2.ipynb``: AND children are combined recursively
    without materializing full child lists via ``itertools.product``.
    """
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
            yield from generate_ast_primes(child)
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
            for implicant in generate_ast_primes(children[0]):
                merged = prefix.copy()
                merged.update(implicant)
                yield from combine(children[1:], merged)

        yield from combine(node.children)


def _maybe_require_rof(expr: str, require_rof: bool) -> None:
    if not require_rof:
        return
    from high_k.rof import check_syntactic_rof

    info = check_syntactic_rof(expr)
    if not info["is_rof"]:
        raise ValueError(
            "expression is not a syntactic ROF "
            f"(n_variables={info['n_variables']}, "
            f"n_unique_variables={info['n_unique_variables']})"
        )


def _constant_ast(
    expr: str,
) -> Optional[Tuple[LogicNode, int, Optional[LogicNode], int]]:
    """Return ``(ast_on, on_count, ast_off, off_count)`` for 0/1 constants."""
    if expr in ("1", "True", "true"):
        return LogicNode("LITERAL", f"CONSTANT_{expr}"), 1, None, 0
    if expr in ("0", "False", "false"):
        # Match notebook: treat as a single ON literal, no OFF dual.
        return LogicNode("LITERAL", f"CONSTANT_{expr}"), 1, None, 0
    return None


def count_on_off_primes(
    expr: str, *, require_rof: bool = False
) -> Tuple[int, int]:
    """Return ``(n_on_primes, n_off_primes)`` for expression *expr*."""
    _maybe_require_rof(expr, require_rof)
    expr = expr.strip()
    const = _constant_ast(expr)
    if const is not None:
        return const[1], const[3]
    ast_on = parse_to_ast(expr)
    ast_off = invert_ast(ast_on) if ast_on is not None else None
    return count_prime_implicants(ast_on), count_prime_implicants(ast_off)


def iter_on_off_primes(
    expr: str,
    polarity: Polarity = "on",
    *,
    require_rof: bool = False,
) -> Iterator[PrimeDict]:
    """Yield prime implicants for the ON or OFF set of *expr*."""
    _maybe_require_rof(expr, require_rof)
    expr = expr.strip()
    const = _constant_ast(expr)
    if const is not None:
        ast_on, _, ast_off, _ = const
        node = ast_on if polarity == "on" else ast_off
        yield from generate_ast_primes(node)
        return
    ast_on = parse_to_ast(expr)
    if polarity == "on":
        yield from generate_ast_primes(ast_on)
        return
    if ast_on is None:
        return
    yield from generate_ast_primes(invert_ast(ast_on))


class SafeNetworkPrimes:
    """Parse a ``.bnet`` string/path and expose per-node PI counts / generators.

    Same role as the notebook ``SafeNetworkPrimes`` helper: count ON/OFF primes
    eagerly; expose lazy generators so callers can skip huge enumerations.
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
                on_count = count_prime_implicants(ast_on)
                off_count = count_prime_implicants(ast_off)

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
        return generate_ast_primes(data["ast_on"])

    def get_off_generator(self, var_name: str) -> Optional[Iterator[PrimeDict]]:
        data = self.nodes.get(var_name)
        if data is None or data["ast_off"] is None:
            return None
        return generate_ast_primes(data["ast_off"])

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
