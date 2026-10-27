# -*- coding: utf-8 -*-
"""Tests for :class:`cana.function_outputs.FunctionOutputs`."""
import pytest

from cana.bit_lut import BitLut
from cana.boolean_network import BooleanNetwork
from cana.boolean_node import BooleanNode
from cana.function_outputs import BooleanEval, FunctionOutputs
from cana.function_outputs import one_symbol as one_symbol_mod
from cana.function_outputs.one_symbol import bit_lut_from_schemata, reorder as reorder_one_symbol


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
    with pytest.raises(ValueError, match="neither bit_lut nor one_symbol_schemata"):
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


def test_network_shares_function_outputs():
    rules = "\n".join(
        [
            "A *= B",
            "B *= A and B",
        ]
    )
    network = BooleanNetwork.from_string_booleannet(rules, keep_constants=True)
    for i, node in enumerate(network.nodes):
        assert isinstance(network.network_dict[i]["outputs"], FunctionOutputs)
        assert isinstance(node.outputs, FunctionOutputs)
        assert node.outputs is network.network_dict[i]["outputs"]
        assert isinstance(node.outputs.bit_lut, BitLut)


def test_get_one_symbol_schemata_lazy():
    fo = FunctionOutputs.from_sequence([0, 0, 0, 1])  # AND
    assert fo.one_symbol_schemata is None
    first = fo.get_one_symbol_schemata()
    assert fo.one_symbol_schemata is first
    assert first[1] == {"11"}
    assert first[0] == {"02", "20"}
    second = fo.get_one_symbol_schemata()
    assert second is first


def test_build_bit_lut_from_one_symbol_schemata():
    original = [0, 0, 0, 1]
    fo = FunctionOutputs.from_sequence(original)
    schemata = fo.get_one_symbol_schemata()
    rebuilt = FunctionOutputs(one_symbol_schemata=schemata, k=2)
    assert rebuilt.bit_lut is None
    lut = rebuilt.build_bit_lut_from_one_symbol_schemata()
    assert list(lut) == original
    assert rebuilt.get_bit_lut() is lut


def test_eval_from_one_symbol_only():
    schemata = {0: {"02", "20"}, 1: {"11"}}  # AND cover with wildcards as '2'
    fo = FunctionOutputs(one_symbol_schemata=schemata, k=2)
    assert fo.bit_lut is None
    assert fo.eval(0) == 0
    assert fo.eval(3) == 1
    assert fo.bit_lut is not None
    assert list(fo.bit_lut) == [0, 0, 0, 1]


def test_compute_one_symbol_sets_prime_tag():
    fo = FunctionOutputs.from_sequence([0, 0, 0, 1])  # AND
    assert fo.one_symbol_is_prime_implicant is None
    fo.get_one_symbol_schemata()
    assert fo.one_symbol_is_prime_implicant is True


def test_verify_one_symbol_schemata_from_qm():
    fo = FunctionOutputs.from_sequence([0, 1, 1, 0])  # XOR
    schemata = fo.get_one_symbol_schemata()
    assert fo.one_symbol_is_prime_implicant is True
    assert fo.verify_one_symbol_schemata() is True
    assert fo.one_symbol_is_prime_implicant is True
    # Schemata-only copy of the same PI cover.
    only = FunctionOutputs(one_symbol_schemata=schemata, k=2)
    assert only.bit_lut is None
    assert only.verify_one_symbol_schemata() is True
    assert only.one_symbol_is_prime_implicant is True


def test_verify_one_symbol_incomplete_and_conflict():
    incomplete = FunctionOutputs(
        one_symbol_schemata={1: {"11"}}, k=2
    )
    assert incomplete.verify_one_symbol_schemata() is False
    assert incomplete.one_symbol_is_prime_implicant is False

    conflict = FunctionOutputs(
        one_symbol_schemata={0: {"11"}, 1: {"11"}}, k=2
    )
    assert conflict.verify_one_symbol_schemata() is False
    assert conflict.one_symbol_is_prime_implicant is False


def test_verify_one_symbol_non_prime_complete_cover():
    # Complete AND cover as minterms (not the QM prime-implicant set).
    schemata = {0: {"00", "01", "10"}, 1: {"11"}}
    fo = FunctionOutputs(one_symbol_schemata=schemata, k=2)
    assert fo.verify_one_symbol_schemata() is False
    assert fo.one_symbol_is_prime_implicant is False


def test_verify_representations_consistent():
    fo = FunctionOutputs.from_sequence([0, 0, 0, 1])
    assert fo.representations_consistent is None
    assert fo.verify_representations_consistent() is True  # vacuous: only bit_lut
    assert fo.representations_consistent is True

    fo.get_one_symbol_schemata()
    assert fo.verify_representations_consistent() is True
    assert fo.representations_consistent is True

    mismatched = FunctionOutputs.from_sequence([0, 0, 0, 1])
    mismatched.one_symbol_schemata = {0: {"00"}, 1: {"01", "10", "11"}}  # OR
    assert mismatched.verify_representations_consistent() is False
    assert mismatched.representations_consistent is False


def test_node_stores_one_symbol_on_outputs():
    n = BooleanNode.from_output_list([0, 0, 0, 1], name="AND")
    assert n.outputs.one_symbol_schemata is None
    n._check_compute_canalization_variables(prime_implicants=True)
    assert n.outputs.one_symbol_schemata is not None
    assert n._prime_implicants is n.outputs.one_symbol_schemata
    cached = n.outputs.one_symbol_schemata
    n._check_compute_canalization_variables(prime_implicants=True)
    assert n.outputs.one_symbol_schemata is cached
    n._check_compute_canalization_variables(two_symbols=True)
    assert n.outputs.two_symbol_schemata is not None
    assert n._two_symbols is n.outputs.two_symbol_schemata


def test_fill_function_outputs_node_and_network():
    schemata = {0: {"02", "20"}, 1: {"11"}}
    n = BooleanNode(
        id=0, name="AND", k=2, inputs=[0, 1], outputs=FunctionOutputs(
            one_symbol_schemata=schemata, k=2
        )
    )
    assert n.outputs.bit_lut is None
    n.fill_function_outputs("bit_lut")
    assert list(n.outputs.bit_lut) == [0, 0, 0, 1]

    n2 = BooleanNode.from_output_list([0, 1, 1, 0], name="XOR")
    assert n2.outputs.one_symbol_schemata is None
    n2.fill_function_outputs("one_symbol_schemata")
    assert n2.outputs.one_symbol_schemata is not None

    network = BooleanNetwork.from_dict(
        {
            0: {"name": "A", "inputs": [0], "outputs": [0, 1]},
            1: {
                "name": "B",
                "inputs": [0, 1],
                "outputs": FunctionOutputs(
                    one_symbol_schemata={0: {"02", "20"}, 1: {"11"}}, k=2
                ),
            },
        }
    )
    assert network.nodes[1].outputs.bit_lut is None
    network.fill_function_outputs("bit_lut")
    assert list(network.nodes[1].outputs.bit_lut) == [0, 0, 0, 1]
    assert network.network_dict[1]["outputs"].bit_lut is network.nodes[1].outputs.bit_lut
    network.fill_function_outputs("one_symbol_schemata")
    assert network.nodes[0].outputs.one_symbol_schemata is not None


def test_one_symbol_reorder_identity_returns_same():
    schemata = {0: {"02", "20"}, 1: {"11"}}
    assert reorder_one_symbol(schemata, ["A", "B"], ["A", "B"]) is schemata


def test_one_symbol_reorder_swaps_and_implies():
    # AND on (A, B): QM cover {0: 02,20; 1: 11}; swap keeps AND.
    and_schemata = {0: {"02", "20"}, 1: {"11"}}
    swapped_and = reorder_one_symbol(and_schemata, ["A", "B"], ["B", "A"])
    assert swapped_and[1] == {"11"}
    assert swapped_and[0] == {"02", "20"}
    assert list(bit_lut_from_schemata(swapped_and, 2)) == [0, 0, 0, 1]

    # Implies A->B on (A, B): [1,1,0,1]; after swap to (B, A): [1,0,1,1].
    implies = FunctionOutputs.from_sequence([1, 1, 0, 1])
    schemata = implies.get_one_symbol_schemata()
    swapped = reorder_one_symbol(schemata, ["A", "B"], ["B", "A"])
    assert list(bit_lut_from_schemata(swapped, 2)) == [1, 0, 1, 1]


def test_function_outputs_reorder_both_peers():
    fo = FunctionOutputs.from_sequence([1, 1, 0, 1])  # A->B
    fo.get_one_symbol_schemata()
    assert fo.one_symbol_is_prime_implicant is True
    fo.representations_consistent = True
    fo.two_symbol_schemata = ([], [])  # stub; should be cleared

    out = fo.reorder(["A", "B"], ["B", "A"])
    assert out is not fo
    assert list(out.bit_lut) == [1, 0, 1, 1]
    assert out.one_symbol_schemata is not None
    assert list(bit_lut_from_schemata(out.one_symbol_schemata, 2)) == [1, 0, 1, 1]
    assert out.one_symbol_is_prime_implicant is True
    assert out.representations_consistent is True
    assert out.two_symbol_schemata is None
    assert out.verify_representations_consistent() is True


def test_function_outputs_reorder_identity_returns_self():
    fo = FunctionOutputs.from_sequence([0, 0, 0, 1])
    assert fo.reorder(["A", "B"], ["A", "B"]) is fo


def test_one_symbol_module_public_api():
    assert "OneSymbolSchemata" in one_symbol_mod.__all__
    assert callable(one_symbol_mod.reorder)
    assert callable(one_symbol_mod.bit_lut_from_schemata)
    assert callable(one_symbol_mod.schemata_from_bit_lut)
