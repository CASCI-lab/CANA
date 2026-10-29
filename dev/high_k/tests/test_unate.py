# -*- coding: utf-8 -*-
"""Tests for :mod:`high_k.unate`."""
from __future__ import annotations

import pytest

from high_k.ast import parse_to_ast
from high_k.unate import (
    check_syntactic_unate,
    check_syntactic_unate_ast,
    require_syntactic_unate_ast,
)


def test_all_positive_is_unate():
    info = check_syntactic_unate("a & b | c")
    assert info["is_unate"] is True
    assert info["positive"] == ["a", "b", "c"]
    assert info["negative"] == []
    assert info["binate"] == []
    assert info["n_unique_variables"] == 3


def test_all_negative_is_unate():
    info = check_syntactic_unate("!a & !b")
    assert info["is_unate"] is True
    assert info["positive"] == []
    assert info["negative"] == ["a", "b"]
    assert info["binate"] == []


def test_mixed_consistent_polarities_is_unate():
    info = check_syntactic_unate("a & !b")
    assert info["is_unate"] is True
    assert info["positive"] == ["a"]
    assert info["negative"] == ["b"]
    assert info["binate"] == []


def test_binate_variable_rejected():
    info = check_syntactic_unate("a | !a")
    assert info["is_unate"] is False
    assert info["binate"] == ["a"]
    assert info["positive"] == []
    assert info["negative"] == []


def test_de_morgan_creates_binate():
    info = check_syntactic_unate("!(a & b) | a")
    assert info["is_unate"] is False
    assert info["binate"] == ["a"]
    assert info["negative"] == ["b"]
    assert info["positive"] == []


def test_de_morgan_stays_unate():
    info = check_syntactic_unate("!(a & b)")
    assert info["is_unate"] is True
    assert info["positive"] == []
    assert info["negative"] == ["a", "b"]
    assert info["binate"] == []


def test_repeated_same_polarity_is_unate():
    info = check_syntactic_unate("a & a")
    assert info["is_unate"] is True
    assert info["positive"] == ["a"]
    assert info["n_unique_variables"] == 1


def test_string_and_ast_parity():
    expr = "(a | !b) & !c"
    assert check_syntactic_unate(expr) == check_syntactic_unate_ast(
        parse_to_ast(expr)
    )


def test_vacuous_empty_and_constant():
    empty = check_syntactic_unate("")
    assert empty["is_unate"] is True
    assert empty["positive"] == []
    assert empty["negative"] == []
    assert empty["binate"] == []
    assert empty["n_unique_variables"] == 0
    assert check_syntactic_unate_ast(None) == empty


def test_require_syntactic_unate_ast():
    ok = parse_to_ast("a & !b")
    require_syntactic_unate_ast(ok)
    bad = parse_to_ast("a | !a")
    with pytest.raises(ValueError, match="syntactically unate"):
        require_syntactic_unate_ast(bad)


def test_python_operators():
    info = check_syntactic_unate("a and not b")
    assert info["is_unate"] is True
    assert info["positive"] == ["a"]
    assert info["negative"] == ["b"]


def test_stacked_negation_nnf():
    assert check_syntactic_unate("!!a") == {
        "is_unate": True,
        "positive": ["a"],
        "negative": [],
        "binate": [],
        "n_unique_variables": 1,
    }
    # !!a | a ≡ a | a → still unate (same polarity).
    assert check_syntactic_unate("!!a | a")["is_unate"] is True
    assert check_syntactic_unate("!!!a")["negative"] == ["a"]
    assert check_syntactic_unate("!!!a")["positive"] == []
    # Odd stack + positive occurrence → binate (not a fake '! a' name).
    info = check_syntactic_unate("!!!a | a")
    assert info["is_unate"] is False
    assert info["binate"] == ["a"]


def test_stacked_negation_string_ast_parity():
    for expr in ("!!a", "!!!a", "!!!a | a", "!!(a & b)", "!(!(a | !b))"):
        assert check_syntactic_unate(expr) == check_syntactic_unate_ast(
            parse_to_ast(expr)
        )


def test_stops_at_first_binate():
    # Binate first, then a long unate suffix that must not be visited
    # enough to populate positive with the suffix names — but
    # n_unique_variables must still count the full formula.
    suffix = " & ".join(f"x{i}" for i in range(20))
    expr = f"(z | !z) & {suffix}"
    info = check_syntactic_unate(expr)
    assert info["is_unate"] is False
    assert info["binate"] == ["z"]
    assert info["positive"] == []
    assert len(info["binate"]) == 1
    assert info["n_unique_variables"] == 21  # z + x0..x19
    ast_info = check_syntactic_unate_ast(parse_to_ast(expr))
    assert ast_info["is_unate"] is False
    assert ast_info["binate"] == ["z"]
    assert ast_info["positive"] == []
    assert ast_info["n_unique_variables"] == 21


def test_n_unique_matches_rof_on_binate():
    from high_k.rof import check_syntactic_rof

    expr = "(a | !a) & b & c & d & e"
    unate = check_syntactic_unate(expr)
    rof = check_syntactic_rof(expr)
    assert unate["is_unate"] is False
    assert unate["n_unique_variables"] == rof["n_unique_variables"] == 5
    ast_info = check_syntactic_unate_ast(parse_to_ast(expr))
    assert ast_info["n_unique_variables"] == 5


def test_deep_nesting_does_not_hit_recursion_limit():
    # BBM-style left-nested parens; recursive descent used to raise
    # RecursionError well below this depth.
    depth = 5_000
    expr = "(" * depth + "a & !b" + ")" * depth
    info = check_syntactic_unate(expr)
    assert info["is_unate"] is True
    assert info["positive"] == ["a"]
    assert info["negative"] == ["b"]
    # Polarity flip under a deep outer negation.
    info_neg = check_syntactic_unate("!" + expr)
    assert info_neg["is_unate"] is True
    assert info_neg["positive"] == ["b"]
    assert info_neg["negative"] == ["a"]
