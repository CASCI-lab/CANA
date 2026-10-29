# -*- coding: utf-8 -*-
"""Tests for :mod:`high_k.ast`."""
from __future__ import annotations

from high_k.ast import (
    ast_leaf_polarities,
    ast_variable_names,
    invert_ast,
    parse_to_ast,
    to_symbolic_expr,
)


def test_to_symbolic_accepts_python_and_symbolic():
    assert to_symbolic_expr("a and not b") == to_symbolic_expr("a & !b")
    assert "&" in to_symbolic_expr("a and b")
    assert "!" in to_symbolic_expr("not a")


def test_parse_literal_and_negated():
    pos = parse_to_ast("GeneA")
    assert pos is not None and pos.op == "LITERAL" and pos.children == "GeneA"
    neg = parse_to_ast("!GeneA")
    assert neg is not None and neg.op == "NEGATED_LITERAL"
    assert neg.children == "GeneA"


def test_parse_negated_group_applies_de_morgan():
    node = parse_to_ast("!(A & B)")
    assert node is not None
    assert node.op == "OR"
    assert isinstance(node.children, list)
    assert {c.op for c in node.children} == {"NEGATED_LITERAL"}
    assert sorted(ast_variable_names(node)) == ["A", "B"]


def test_invert_is_involutive_on_structure():
    node = parse_to_ast("(a0 | a1) & !a2")
    assert node is not None
    twice = invert_ast(invert_ast(node))
    assert ast_variable_names(twice) == ast_variable_names(node)
    assert twice.op == node.op


def test_empty_expression():
    assert parse_to_ast("") is None
    assert parse_to_ast("   ") is None
    assert ast_variable_names(None) == []


def test_ast_variable_names_with_multiplicity():
    node = parse_to_ast("a & a")
    assert node is not None
    assert ast_variable_names(node) == ["a", "a"]


def test_ast_leaf_polarities():
    node = parse_to_ast("a & !b | a")
    assert node is not None
    assert ast_leaf_polarities(node) == [
        ("a", "pos"),
        ("b", "neg"),
        ("a", "pos"),
    ]
    assert ast_leaf_polarities(None) == []
    assert ast_leaf_polarities(parse_to_ast("!(A & B)")) == [
        ("A", "neg"),
        ("B", "neg"),
    ]


def test_parse_stacked_negation_cancels():
    pos = parse_to_ast("!!GeneA")
    assert pos is not None and pos.op == "LITERAL" and pos.children == "GeneA"
    neg = parse_to_ast("!!!GeneA")
    assert neg is not None and neg.op == "NEGATED_LITERAL"
    assert neg.children == "GeneA"
    group = parse_to_ast("!!(A & B)")
    assert group is not None and group.op == "AND"
    assert ast_variable_names(group) == ["A", "B"]
    assert ast_leaf_polarities(group) == [("A", "pos"), ("B", "pos")]
