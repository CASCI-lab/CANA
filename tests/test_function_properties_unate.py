# -*- coding: utf-8 -*-
"""Tests for :mod:`cana.function_properties.unate`."""
import pytest

from cana.datasets.bools import AND, OR, XOR
from cana.function_outputs.bit_lut import BitLut
from cana.function_properties import (
    input_sign,
    input_signs,
    is_function_monotone_increasing,
    is_function_unate,
    is_input_negative_unate,
    is_input_positive_unate,
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
