# -*- coding: utf-8 -*-
"""Tests for :class:`cana.function_outputs.FunctionOutputs`."""
import pytest

from cana.bit_lut import BitLut
from cana.boolean_network import BooleanNetwork
from cana.boolean_node import BooleanNode
from cana.function_outputs import BooleanEval, FunctionOutputs


def test_from_sequence_and_eval():
    fo = FunctionOutputs.from_sequence([0, 0, 0, 1])
    assert isinstance(fo.bit_lut, BitLut)
    assert fo.k == 2
    assert list(fo) == [0, 0, 0, 1]
    assert fo.eval(3) == 1
    assert fo[3] == 1
    assert isinstance(fo, BooleanEval)


def test_wraps_bit_lut_same_object():
    bit = BitLut.from_sequence([0, 1])
    fo = FunctionOutputs.from_sequence(bit)
    assert fo.bit_lut is bit


def test_no_dense_lut_errors():
    fo = FunctionOutputs(expression="A and B", k=2)
    assert fo.bit_lut is None
    assert fo.k == 2
    with pytest.raises(ValueError, match="bit_lut"):
        len(fo)
    with pytest.raises(ValueError, match="bit_lut"):
        fo[0]
    with pytest.raises(ValueError, match="bit_lut"):
        fo.eval(0)


def test_equality_with_list():
    fo = FunctionOutputs.from_sequence([0, 1, 1, 0])
    assert fo == [0, 1, 1, 0]
    assert fo == ["0", "1", "1", "0"]
    assert fo != [0, 0, 0, 0]


def test_boolean_node_uses_function_outputs():
    n = BooleanNode.from_output_list([0, 0, 0, 1], name="AND")
    assert isinstance(n.outputs, FunctionOutputs)
    assert isinstance(n.outputs.bit_lut, BitLut)
    assert list(n.outputs) == [0, 0, 0, 1]


def test_network_shares_bit_lut():
    rules = "\n".join(
        [
            "A *= B",
            "B *= A and B",
        ]
    )
    network = BooleanNetwork.from_string_booleannet(rules, keep_constants=True)
    for i, node in enumerate(network.nodes):
        assert isinstance(network.lut_dict[i]["lut"], BitLut)
        assert isinstance(node.outputs, FunctionOutputs)
        assert node.outputs.bit_lut is network.lut_dict[i]["lut"]
