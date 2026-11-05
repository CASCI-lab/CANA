# -*- coding: utf-8 -*-
"""Tests for :mod:`cana.function_outputs.ast`."""
from __future__ import annotations

from cana.function_outputs.ast import (
    ast_leaf_polarities,
    ast_variable_names,
    invert_ast,
    expression_to_ast,
)
from cana.function_outputs.expression import normalize_operators


def _to_symbolic(expr: str) -> str:
    return normalize_operators(
        normalize_operators(expr, "python"), "symbolic"
    )


def test_normalize_operators_accepts_python_and_symbolic():
    assert _to_symbolic("a and not b") == _to_symbolic("a & !b")
    assert "&" in _to_symbolic("a and b")
    assert "!" in _to_symbolic("not a")


def test_parse_literal_and_negated():
    pos = expression_to_ast("GeneA")
    assert pos is not None and pos.op == "LITERAL" and pos.children == "GeneA"
    neg = expression_to_ast("!GeneA")
    assert neg is not None and neg.op == "NEGATED_LITERAL"
    assert neg.children == "GeneA"


def test_parse_negated_group_applies_de_morgan():
    node = expression_to_ast("!(A & B)")
    assert node is not None
    assert node.op == "OR"
    assert isinstance(node.children, list)
    assert {c.op for c in node.children} == {"NEGATED_LITERAL"}
    assert sorted(ast_variable_names(node)) == ["A", "B"]


def test_invert_is_involutive_on_structure():
    node = expression_to_ast("(a0 | a1) & !a2")
    assert node is not None
    twice = invert_ast(invert_ast(node))
    assert ast_variable_names(twice) == ast_variable_names(node)
    assert twice.op == node.op


def test_empty_expression():
    assert expression_to_ast("") is None
    assert expression_to_ast("   ") is None
    assert ast_variable_names(None) == []


def test_ast_variable_names_with_multiplicity():
    node = expression_to_ast("a & a")
    assert node is not None
    assert ast_variable_names(node) == ["a", "a"]


def test_ast_leaf_polarities():
    node = expression_to_ast("a & !b | a")
    assert node is not None
    assert ast_leaf_polarities(node) == [
        ("a", "pos"),
        ("b", "neg"),
        ("a", "pos"),
    ]
    assert ast_leaf_polarities(None) == []
    assert ast_leaf_polarities(expression_to_ast("!(A & B)")) == [
        ("A", "neg"),
        ("B", "neg"),
    ]


def test_parse_stacked_negation_cancels():
    pos = expression_to_ast("!!GeneA")
    assert pos is not None and pos.op == "LITERAL" and pos.children == "GeneA"
    neg = expression_to_ast("!!!GeneA")
    assert neg is not None and neg.op == "NEGATED_LITERAL"
    assert neg.children == "GeneA"
    group = expression_to_ast("!!(A & B)")
    assert group is not None and group.op == "AND"
    assert ast_variable_names(group) == ["A", "B"]
    assert ast_leaf_polarities(group) == [("A", "pos"), ("B", "pos")]
