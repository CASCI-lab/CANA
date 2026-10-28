# -*- coding: utf-8 -*-
"""Boolean expression AST: parse, invert, and leaf walks.

Structural ``&`` / ``|`` / ``!`` trees only — no read-once assumption.
Callers that need syntactic ROF (single occurrence) should use
:mod:`high_k.rof`.
"""
from __future__ import annotations

from typing import List, Optional, Union

from high_k.paths import ensure_cana_on_path

ensure_cana_on_path()

from cana.function_outputs.expression import normalize_operators

__all__ = [
    "LogicNode",
    "to_symbolic_expr",
    "parse_to_ast",
    "invert_ast",
    "ast_variable_names",
]


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


def ast_variable_names(node: Optional[LogicNode]) -> List[str]:
    """Collect leaf variable names in *node* (preorder, with multiplicity)."""
    if node is None:
        return []
    if node.op in ("LITERAL", "NEGATED_LITERAL"):
        assert isinstance(node.children, str)
        return [node.children]
    assert isinstance(node.children, list)
    names: List[str] = []
    for child in node.children:
        names.extend(ast_variable_names(child))
    return names
