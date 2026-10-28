# -*- coding: utf-8 -*-
"""Tests for :mod:`high_k.rof`."""
from __future__ import annotations

from pathlib import Path

import pytest

from high_k.ast import invert_ast, parse_to_ast
from high_k.bbm_rules import as_bnet
from high_k.rof import (
    SafeNetworkPrimes,
    check_syntactic_rof,
    check_syntactic_rof_ast,
    count_on_off_primes,
    count_rof_primes,
    generate_rof_primes,
    iter_on_off_primes,
)
from high_k.synthetic import (
    and_chain,
    or_chain,
    pairwise_and_or,
    pairwise_or_and,
    random_rof,
)
from high_k.timing import import_detached_outputs

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

MASSIVE_HUB_BNET = """
# Toy Biological Network Model
targets,factors
Protein_A,      Input_1 & Input_2
Protein_B,      (Input_3 | Input_4) & Protein_A
Massive_Hub,    (a1 | b1) & (a2 | b2) & (a3 | b3) & (a4 | b4) & (a5 | b5) & (a6 | b6) & (a7 | b7) & (a8 | b8) & (a9 | b9) & (a10 | b10)
Protein_C,      Massive_Hub | Protein_B
"""

NEGATION_BNET = """
# Regulatory Network with Inhibitors
Cell_Cycle_Start,   Growth_Factor & !DNA_Damage
S_Phase_Entry,      Cell_Cycle_Start & !Checkpoint_Arrest
Mitosis_Inhibitor,  (p53 | p21) & !Anaphase_Promoting_Complex
"""


def _qm_pi_counts(expr: str) -> tuple[int, int]:
    """ON/OFF prime counts via CANA Quine–McCluskey one-symbol schemata."""
    fo = import_detached_outputs(as_bnet(expr), "Y")
    fo.build_bit_lut_from_positional_expression()
    schemata = fo.compute_one_symbol_schemata_from_bit_lut()
    return len(schemata.get(1, ())), len(schemata.get(0, ()))


def _freeze(primes) -> frozenset:
    return frozenset(frozenset(p.items()) for p in primes)


# ---------------------------------------------------------------------------
# Syntactic ROF check
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Notebook regressions (AST.ipynb / AST_2.ipynb)
# ---------------------------------------------------------------------------


def test_massive_hub_network_counts():
    net = SafeNetworkPrimes(MASSIVE_HUB_BNET)
    assert net.nodes["Protein_A"]["on_count"] == 1
    assert net.nodes["Protein_A"]["off_count"] == 2
    assert net.nodes["Protein_B"]["on_count"] == 2
    assert net.nodes["Protein_B"]["off_count"] == 2
    assert net.nodes["Massive_Hub"]["on_count"] == 1024
    assert net.nodes["Massive_Hub"]["off_count"] == 10
    assert net.nodes["Protein_C"]["on_count"] == 2
    assert net.nodes["Protein_C"]["off_count"] == 1


def test_massive_hub_off_primes():
    net = SafeNetworkPrimes(MASSIVE_HUB_BNET)
    offs = list(net.get_off_generator("Massive_Hub"))
    assert len(offs) == 10
    assert offs[0] == {"a1": False, "b1": False}
    assert offs[-1] == {"a10": False, "b10": False}


def test_negation_network_cell_cycle_start():
    model = SafeNetworkPrimes(NEGATION_BNET)
    assert model.nodes["Cell_Cycle_Start"]["on_count"] == 1
    assert model.nodes["Cell_Cycle_Start"]["off_count"] == 2
    assert list(model.get_on_generator("Cell_Cycle_Start")) == [
        {"Growth_Factor": True, "DNA_Damage": False}
    ]
    offs = list(model.get_off_generator("Cell_Cycle_Start"))
    assert len(offs) == 2
    assert _freeze(offs) == _freeze(
        [
            {"Growth_Factor": False},
            {"DNA_Damage": True},
        ]
    )


# ---------------------------------------------------------------------------
# Cross-check vs CANA QM (small ROFs)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr",
    [
        "a0 & a1",
        "a0 | a1",
        "(a0 | a1) & (a2 | a3)",
        "(a0 & a1) | (a2 & a3)",
        "a0 and not a1",
        pairwise_or_and(6),
        pairwise_and_or(6),
        and_chain(4),
        or_chain(4),
        random_rof(6, seed=7),
    ],
)
def test_rof_counts_match_qm_one_symbol(expr: str):
    assert check_syntactic_rof(expr)["is_rof"]
    on_n, off_n = count_on_off_primes(expr)
    qm_on, qm_off = _qm_pi_counts(expr)
    assert (on_n, off_n) == (qm_on, qm_off)


@pytest.mark.parametrize(
    "expr,expected_on,expected_off",
    [
        (and_chain(5), 1, 5),
        (or_chain(5), 5, 1),
        (pairwise_or_and(8), 16, 4),
        (pairwise_and_or(8), 4, 16),
    ],
)
def test_synthetic_rof_closed_form_counts(expr, expected_on, expected_off):
    assert count_on_off_primes(expr) == (expected_on, expected_off)


# ---------------------------------------------------------------------------
# Count / generate basics
# ---------------------------------------------------------------------------


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
    expr = pairwise_or_and(6)
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


# ---------------------------------------------------------------------------
# Syntactic ROF assertion / constants / network I/O
# ---------------------------------------------------------------------------


def test_syntactic_rof_accepted():
    assert count_on_off_primes("a & b") == (1, 2)


def test_rejects_repeated_variable_not_just_semantic_rof():
    # ``a & a`` ≡ ``a`` (semantic ROF after simplification) but is not
    # syntactic / single-occurrence ROF; the engine must reject it.
    with pytest.raises(ValueError, match="syntactic ROF"):
        count_on_off_primes("a & a")
    with pytest.raises(ValueError, match="syntactic ROF"):
        list(iter_on_off_primes("a | a", "on"))
    node = parse_to_ast("a & a")
    with pytest.raises(ValueError, match="syntactic ROF"):
        count_rof_primes(node)
    with pytest.raises(ValueError, match="syntactic ROF"):
        list(generate_rof_primes(node))


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


def test_safe_network_unknown_node_generators():
    net = SafeNetworkPrimes("Y, a\n")
    assert net.get_on_generator("missing") is None
    assert net.get_off_generator("missing") is None


def test_summary_prints(capsys):
    net = SafeNetworkPrimes("Y, a & b\n")
    net.summary()
    out = capsys.readouterr().out
    assert "Y" in out
    assert "a & b" in out
