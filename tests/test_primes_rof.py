# -*- coding: utf-8 -*-
"""Tests for :mod:`cana.primes.rof`."""
from __future__ import annotations

from pathlib import Path

import pytest

from cana.function_outputs import FunctionOutputs
from cana.nodes_builder import named_expression_to_detached_rule
from cana.primes.ast import invert_ast, parse_to_ast
from cana.primes.rof import (
    SafeNetworkPrimes,
    check_syntactic_rof,
    check_syntactic_rof_ast,
    count_on_off_primes,
    count_rof_primes,
    generate_rof_primes,
    iter_on_off_primes,
)


def _qm_pi_counts(expr: str) -> tuple[int, int]:
    rule = named_expression_to_detached_rule("Y", expr)
    fo = rule.outputs
    fo.build_bit_lut_from_positional_expression()
    schemata = fo.compute_one_symbol_schemata_from_bit_lut()
    return len(schemata.get(1, ())), len(schemata.get(0, ()))


def _freeze(primes) -> frozenset:
    return frozenset(frozenset(p.items()) for p in primes)


def test_check_syntactic_rof_accepts_single_occurrence():
    info = check_syntactic_rof("a & b")
    assert info["is_rof"] is True
    assert info["n_variables"] == 2
    assert info["n_unique_variables"] == 2


def test_check_syntactic_rof_rejects_repeated_variable():
    info = check_syntactic_rof("a & a")
    assert info["is_rof"] is False
    assert info["n_variables"] == 2
    assert info["n_unique_variables"] == 1


def test_check_syntactic_rof_ast_matches_string():
    expr = "(a0 | a1) & !a2"
    node = parse_to_ast(expr)
    assert check_syntactic_rof_ast(node) == check_syntactic_rof(expr)


@pytest.mark.parametrize(
    "expr",
    [
        "a0 & a1",
        "a0 | a1",
        "(a0 | a1) & (a2 | a3)",
        "(a0 & a1) | (a2 & a3)",
        "a0 and not a1",
    ],
)
def test_rof_counts_match_qm_one_symbol(expr: str):
    assert check_syntactic_rof(expr)["is_rof"]
    on_n, off_n = count_on_off_primes(expr)
    qm_on, qm_off = _qm_pi_counts(expr)
    assert (on_n, off_n) == (qm_on, qm_off)


def test_parse_negated_group_rof_count():
    node = parse_to_ast("!(A & B)")
    assert node is not None
    assert count_rof_primes(node) == 2


def test_invert_preserves_rof_primes():
    node = parse_to_ast("(a0 | a1) & !a2")
    assert node is not None
    twice = invert_ast(invert_ast(node))
    assert count_rof_primes(twice) == count_rof_primes(node)
    assert _freeze(generate_rof_primes(twice)) == _freeze(
        generate_rof_primes(node)
    )


def test_count_matches_generated_length():
    expr = "(a0 | a1) & (a2 | a3) & (a4 | a5)"
    on_n, off_n = count_on_off_primes(expr)
    assert on_n == sum(1 for _ in iter_on_off_primes(expr, "on"))
    assert off_n == sum(1 for _ in iter_on_off_primes(expr, "off"))


def test_empty_expression_primes():
    assert count_rof_primes(None) == 0
    assert list(generate_rof_primes(None)) == []


def test_python_style_count_and_iter():
    assert count_on_off_primes("a and not b") == (1, 2)
    assert list(iter_on_off_primes("a and not b", "on")) == [
        {"a": True, "b": False}
    ]


def test_rejects_repeated_variable_not_just_semantic_rof():
    with pytest.raises(ValueError, match="syntactic ROF"):
        count_on_off_primes("a & a")
    with pytest.raises(ValueError, match="syntactic ROF"):
        list(iter_on_off_primes("a | a", "on"))


@pytest.mark.parametrize("const", ["0", "1", "True", "False", "true", "false"])
def test_constant_expressions(const: str):
    on_n, off_n = count_on_off_primes(const)
    assert on_n == 1
    assert off_n == 0
    assert len(list(iter_on_off_primes(const, "on"))) == 1
    assert list(iter_on_off_primes(const, "off")) == []


def test_safe_network_from_path(tmp_path: Path):
    path = tmp_path / "toy.bnet"
    path.write_text("targets,factors\nY, a & !b\nZ, 1\n")
    net = SafeNetworkPrimes(path)
    assert set(net.nodes) == {"Y", "Z"}
    assert net.nodes["Y"]["on_count"] == 1
    assert net.nodes["Y"]["off_count"] == 2
    assert net.nodes["Z"]["on_count"] == 1
    assert net.nodes["Z"]["off_count"] == 0
    assert net.get_off_generator("Z") is None
    assert list(net.get_on_generator("Y")) == [{"a": True, "b": False}]


def test_safe_network_skips_comments_and_header():
    net = SafeNetworkPrimes(
        "# comment\ntargets,factors\nA, x | y\n\n# another\n"
    )
    assert list(net.nodes) == ["A"]
    assert net.nodes["A"]["on_count"] == 2
    assert net.nodes["A"]["off_count"] == 1
