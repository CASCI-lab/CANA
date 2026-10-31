# -*- coding: utf-8 -*-
"""Tests for :mod:`cana.function_properties.unate`."""
import pytest

from cana.boolean_node import BooleanNode
from cana.datasets.bools import AND, OR, XOR
from cana.function_outputs.ast import expression_to_ast
from cana.function_outputs.bit_lut import BitLut
from cana.function_properties import (
    check_syntactic_rof,
    check_syntactic_unate,
    check_syntactic_unate_ast,
    input_sign,
    input_signs,
    is_function_monotone_increasing,
    is_function_unate,
    is_input_negative_unate,
    is_input_positive_unate,
    require_syntactic_unate,
    syntactic_input_signs,
)


def test_and_or_monotone_increasing_and_unate():
    and_lut = BitLut.from_sequence([0, 0, 0, 1])
    or_lut = BitLut.from_sequence([0, 1, 1, 1])
    assert is_function_monotone_increasing(and_lut)
    assert is_function_monotone_increasing(or_lut)
    assert is_function_unate(and_lut)
    assert is_function_unate(or_lut)
    assert input_signs(and_lut) == [1, 1]
    assert input_signs(or_lut) == [1, 1]


def test_xor_not_unate():
    xor = BitLut.from_sequence([0, 1, 1, 0])
    assert not is_function_monotone_increasing(xor)
    assert not is_function_unate(xor)
    assert input_signs(xor) == [0, 0]


def test_inhibiting_input():
    # f = (not A) and B  => outputs for AB: 00,01,10,11 = 0,1,0,0
    lut = BitLut.from_sequence([0, 1, 0, 0])
    assert not is_input_positive_unate(lut, 0)
    assert is_input_negative_unate(lut, 0)
    assert is_input_positive_unate(lut, 1)
    assert input_sign(lut, 0) == -1
    assert input_sign(lut, 1) == 1
    assert input_signs(lut) == [-1, 1]
    assert is_function_unate(lut)
    assert not is_function_monotone_increasing(lut)


def test_not_gate_negative_unate():
    not_lut = BitLut.from_sequence([1, 0])
    assert not is_input_positive_unate(not_lut, 0)
    assert is_input_negative_unate(not_lut, 0)
    assert input_signs(not_lut) == [-1]
    assert is_function_unate(not_lut)
    assert not is_function_monotone_increasing(not_lut)


def test_constant_prefers_positive_sign():
    lut = BitLut.from_sequence([0, 0, 0, 0])
    assert input_signs(lut) == [1, 1]
    assert is_function_monotone_increasing(lut)
    assert is_function_unate(lut)


def test_boolean_node_input_signs():
    assert AND().input_signs() == [1, 1]
    assert OR().input_signs() == [1, 1]
    assert XOR().input_signs() == [0, 0]


def test_deprecated_utils_aliases():
    with pytest.warns(DeprecationWarning):
        from cana.utils import function_monotone

        assert function_monotone([0, 0, 0, 1]) is True
    with pytest.warns(DeprecationWarning):
        from cana.utils import input_monotone

        assert input_monotone([0, 1, 0, 0], 0, activation=-1) is True
    with pytest.warns(DeprecationWarning):
        from cana.utils import negate_LUT_input

        assert negate_LUT_input([1, 1, 0, 1], 0) == list(
            BitLut.from_sequence([1, 1, 0, 1]).negate_input(0)
        )


# --- syntactic unate (NNF scan) ----------------------------------------------


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


def test_vacuous_empty_and_constant():
    empty = check_syntactic_unate("")
    assert empty["is_unate"] is True
    assert empty["positive"] == []
    assert empty["negative"] == []
    assert empty["binate"] == []
    assert empty["n_unique_variables"] == 0
    assert check_syntactic_unate("1")["is_unate"] is True
    assert check_syntactic_unate("0")["is_unate"] is True


def test_require_syntactic_unate():
    require_syntactic_unate("a & !b")
    with pytest.raises(ValueError, match="syntactically unate"):
        require_syntactic_unate("a | !a")


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
    assert check_syntactic_unate("!!a | a")["is_unate"] is True
    assert check_syntactic_unate("!!!a")["negative"] == ["a"]
    assert check_syntactic_unate("!!!a")["positive"] == []
    info = check_syntactic_unate("!!!a | a")
    assert info["is_unate"] is False
    assert info["binate"] == ["a"]


def test_stops_at_first_binate():
    suffix = " & ".join("x{0}".format(i) for i in range(20))
    expr = "(z | !z) & {0}".format(suffix)
    info = check_syntactic_unate(expr)
    assert info["is_unate"] is False
    assert info["binate"] == ["z"]
    assert info["positive"] == []
    assert len(info["binate"]) == 1
    assert info["n_unique_variables"] == 21


def test_n_unique_matches_rof_on_binate():
    expr = "(a | !a) & b & c & d & e"
    unate = check_syntactic_unate(expr)
    rof = check_syntactic_rof(expr)
    assert unate["is_unate"] is False
    assert unate["n_unique_variables"] == rof["n_unique_variables"] == 5


def test_check_syntactic_unate_ast_agrees_when_unate():
    expr = "a & !b | c"
    info = check_syntactic_unate(expr)
    info_ast = check_syntactic_unate_ast(expression_to_ast(expr))
    assert info["is_unate"] is True
    assert info_ast == info


def test_syntactic_input_signs():
    assert syntactic_input_signs("x0 and not x1", ["x0", "x1"]) == [1, -1]
    assert syntactic_input_signs("x0 | !x0", ["x0"]) is None
    # Absent input prefers +1.
    assert syntactic_input_signs("x0", ["x0", "x1"]) == [1, 1]


def test_boolean_node_input_signs_syntactic_fastpath():
    node = BooleanNode.from_positional_expression(
        "x0 and not x1", k=2, name="Y", inputs=[0, 1]
    )
    assert node.outputs.bit_lut is None
    assert node.input_signs() == [1, -1]
    assert node.outputs.bit_lut is None


def test_boolean_node_input_signs_falls_through_when_binate():
    node = BooleanNode.from_positional_expression(
        "x0 or not x0", k=1, name="Y", inputs=[0]
    )
    # Syntactic binate → LUT path; constant-True is vacuously positive unate.
    assert node.input_signs() == [1]
    assert node.outputs.bit_lut is not None
