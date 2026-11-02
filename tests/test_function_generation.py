# -*- coding: utf-8 -*-
"""Tests for :mod:`cana.function_generation`."""
import pytest

from cana.function_generation import (
    EVEN_K_BUILDERS,
    MOD3_K_BUILDERS,
    MULTI_OCCURRENCE_BUILDERS,
    ROF_BUILDERS,
    SYNTHETIC_BUILDERS,
    UNATE_BUILDERS,
    builder_accepts_k,
    random_automata_table,
)
from cana.function_outputs.bit_lut import BitLut
from cana.function_properties import check_syntactic_rof, check_syntactic_unate


@pytest.mark.parametrize("name,builder", list(ROF_BUILDERS.items()))
def test_rof_builders_return_syntactic_rof(name, builder):
    k = 6
    if name in EVEN_K_BUILDERS:
        k = 6
    elif name in MOD3_K_BUILDERS:
        k = 6
    kwargs = {"seed": 0} if name == "random" else {}
    expr = builder(k, **kwargs) if kwargs else builder(k)
    assert isinstance(expr, str)
    assert check_syntactic_rof(expr)["is_rof"] is True
    assert check_syntactic_unate(expr)["is_unate"] is True


def test_majority_is_unate_not_rof():
    expr = UNATE_BUILDERS["majority"](3)
    assert isinstance(expr, str)
    assert check_syntactic_unate(expr)["is_unate"] is True
    assert check_syntactic_rof(expr)["is_rof"] is False
    assert "majority" in MULTI_OCCURRENCE_BUILDERS
    assert "majority" in UNATE_BUILDERS
    assert "majority" not in ROF_BUILDERS


def test_synthetic_builders_union():
    assert set(SYNTHETIC_BUILDERS) == set(ROF_BUILDERS) | set(UNATE_BUILDERS)
    assert "majority" in SYNTHETIC_BUILDERS
    assert "and" in SYNTHETIC_BUILDERS


def test_builder_accepts_k_constraints():
    assert builder_accepts_k("and", 5) is True
    assert builder_accepts_k("half_or_half_and", 4) is True
    assert builder_accepts_k("half_or_half_and", 5) is False
    assert builder_accepts_k("makino_ibaraki", 6) is True
    assert builder_accepts_k("makino_ibaraki", 5) is False
    assert builder_accepts_k("majority", 1) is True
    assert builder_accepts_k("majority", 0) is False


@pytest.mark.parametrize(
    "bias_constraint",
    ["soft", "hard", "soft_no_constant"],
)
def test_random_automata_table(bias_constraint):
    k = 3
    lut = random_automata_table(k, bias=0.5, bias_constraint=bias_constraint)
    assert isinstance(lut, BitLut)
    assert lut.k == k
    assert len(lut) == 2**k
    assert all(bit in (0, 1) for bit in lut)


def test_random_automata_table_unknown_constraint():
    with pytest.raises(ValueError, match="Unknown bias_constraint"):
        random_automata_table(2, 0.5, bias_constraint="nope")
