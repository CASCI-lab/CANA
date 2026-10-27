# -*- coding: utf-8 -*-
"""Tests for :class:`cana.function_outputs.FunctionOutputs`."""
from cana.bit_lut import BitLut
from cana.boolean_network import BooleanNetwork
from cana.boolean_node import BooleanNode
from cana.function_outputs import BooleanEval, FunctionOutputs
from cana.function_outputs import one_symbol as one_symbol_mod
from cana.function_outputs import two_symbol as two_symbol_mod
import warnings

import pytest

from cana.function_outputs.expression import (
    bit_lut_from_expression,
    from_positional,
    function_outputs_to_expression,
    one_symbol_to_expression,
    to_positional,
)
from cana.function_outputs.one_symbol import bit_lut_from_schemata, reorder as reorder_one_symbol
from cana.function_outputs.two_symbol import (
    one_symbol_from_two_symbol,
    reorder as reorder_two_symbol,
    two_symbol_to_expression,
)


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


def test_expression_only_sequence_materializes():
    fo = FunctionOutputs(expression="x0 and x1", k=2)
    assert fo.bit_lut is None
    assert fo.k == 2
    assert len(fo) == 4
    assert fo[3] == 1
    assert list(fo) == [0, 0, 0, 1]
    assert fo.bit_lut is not None
    assert fo.eval(3) == 1


def test_positional_to_from_roundtrip():
    named = "(A and not B) or C"
    names = ["A", "B", "C"]
    positional = to_positional(named, names)
    assert positional == "(x0 and not x1) or x2"
    assert from_positional(positional, names) == named


def test_one_symbol_to_expression_concise():
    # Cover {1: {"12"}} with inputs A,B → A (not two minterms).
    expr = one_symbol_to_expression(["A", "B"], {1: {"12"}}, style="python")
    assert expr == "A"
    # Sorted schemas: "01" then "10"
    assert one_symbol_to_expression(
        ["A", "B"], {1: {"01", "10"}}, style="python"
    ) == "(not A and B) or (A and not B)"
    assert one_symbol_to_expression(
        ["A", "B"], {1: {"01", "10"}}, style="symbolic"
    ) == "(!A & B) | (A & !B)"
    assert one_symbol_to_expression(["A", "B"], {1: set()}, style="python") == (
        "(A and not A)"
    )
    assert one_symbol_to_expression(["A", "B"], {1: {"22"}}, style="python") == (
        "True"
    )


def test_function_outputs_to_expression_ladder():
    names = ["A", "B"]
    # Expression wins over schemata / LUT.
    fo = FunctionOutputs(
        bit_lut=BitLut.from_sequence([0, 0, 0, 1]),
        one_symbol_schemata={1: {"11"}},
        expression="x0 or x1",
        k=2,
    )
    fo.representations_consistent = True
    fo.one_symbol_is_prime_implicant = True
    assert (
        function_outputs_to_expression(fo, names, warn=False) == "A or B"
    )

    # Schemata wins over LUT when no expression.
    fo2 = FunctionOutputs(
        bit_lut=BitLut.from_sequence([0, 1, 1, 1]),  # OR LUT
        one_symbol_schemata={1: {"12"}},  # would say A
        k=2,
    )
    fo2.one_symbol_is_prime_implicant = True
    fo2.representations_consistent = True
    assert function_outputs_to_expression(fo2, names, warn=False) == "A"

    # LUT fallback.
    fo3 = FunctionOutputs.from_sequence([0, 0, 0, 1])
    assert (
        function_outputs_to_expression(fo3, names, warn=False)
        == "(A and B)"
    )


def test_function_outputs_to_expression_warnings():
    names = ["A", "B"]
    schemata_only = FunctionOutputs(
        one_symbol_schemata={1: {"12"}}, k=2
    )
    assert schemata_only.one_symbol_is_prime_implicant is None
    with pytest.warns(UserWarning, match="one_symbol_schemata"):
        assert (
            function_outputs_to_expression(
                schemata_only, names, node_label="X"
            )
            == "A"
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        function_outputs_to_expression(schemata_only, names, warn=False)
        assert caught == []

    # Expression-only: no consistency warn (single slot).
    expr_only = FunctionOutputs(expression="x0 and x1", k=2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        function_outputs_to_expression(
            expr_only, names, node_label="B", warn=True
        )
        assert caught == []

    # Multi-slot expression with known mismatch → warn.
    multi = FunctionOutputs(
        bit_lut=BitLut.from_sequence([0, 0, 0, 1]),
        expression="x0 and x1",
        k=2,
    )
    multi.representations_consistent = False
    with pytest.warns(UserWarning, match="representations_consistent"):
        function_outputs_to_expression(multi, names, node_label="B")
    # Unchecked (None) multi-slot does not warn on expression path.
    multi.representations_consistent = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        function_outputs_to_expression(multi, names, node_label="B")
        assert caught == []


def test_boolean_node_expression_from_schemata():
    n = BooleanNode(
        id=0,
        name="AND",
        k=2,
        inputs=[1, 2],
        outputs=FunctionOutputs(one_symbol_schemata={1: {"11"}}, k=2),
    )
    # Attach a tiny network so input_names() works.
    class _Net(object):
        nodes = [
            n,
            BooleanNode(id=1, name="A", k=0, inputs=[], outputs=[0]),
            BooleanNode(id=2, name="B", k=0, inputs=[], outputs=[0]),
        ]

    n.network = _Net()
    with pytest.warns(UserWarning, match="one_symbol"):
        assert n.expression() == "(A and B)"
    assert n.expression(warn=False) == "(A and B)"
    assert n.expression(style="symbolic", warn=False) == "(A & B)"


def test_bit_lut_from_expression_and_get_bit_lut():
    lut = bit_lut_from_expression("x0 and x1", 2)
    assert list(lut) == [0, 0, 0, 1]
    fo = FunctionOutputs(expression="x0 or x1", k=2)
    assert fo.bit_lut is None
    built = fo.get_bit_lut()
    assert list(built) == [0, 1, 1, 1]
    assert fo.bit_lut is built


def test_function_outputs_reorder_expression():
    fo = FunctionOutputs(expression="x0 and not x1", k=2)
    fo.get_bit_lut()
    expected_lut = list(fo.bit_lut.reorder(["A", "B"], ["B", "A"]))
    out = fo.reorder(["A", "B"], ["B", "A"])
    assert out.expression == "x1 and not x0"
    assert list(out.bit_lut) == expected_lut
    assert out.verify_representations_consistent() is True


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
        # Expression formats store positional IR; LUT is lazy.
        assert node.outputs.expression is not None
        assert node.outputs.bit_lut is None
        assert isinstance(node.outputs.get_bit_lut(), BitLut)


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


def test_canalization_uses_stored_one_symbol_with_soft_warn():
    # Non-prime complete cover: previously would auto-QM and overwrite.
    schemata = {0: {"00", "01", "10"}, 1: {"11"}}
    n = BooleanNode(
        id=0,
        name="AND",
        k=2,
        inputs=[0, 1],
        outputs=FunctionOutputs(one_symbol_schemata=schemata, k=2),
    )
    assert n.outputs.one_symbol_is_prime_implicant is None
    assert n.outputs.bit_lut is None

    with pytest.warns(UserWarning, match="one_symbol_is_prime_implicant"):
        n._check_compute_canalization_variables(prime_implicants=True)

    # Stored cover kept; no LUT / QM forced.
    assert n.outputs.one_symbol_schemata is schemata
    assert n.outputs.bit_lut is None
    assert n.outputs.one_symbol_is_prime_implicant is None

    with pytest.warns(UserWarning, match="one_symbol_is_prime_implicant"):
        n._check_compute_canalization_variables(two_symbols=True)
    assert n.outputs.two_symbol_schemata is not None
    assert n.outputs.bit_lut is None
    assert n.outputs.one_symbol_schemata is schemata


def test_canalization_skips_warn_when_one_symbol_verified():
    schemata = {0: {"02", "20"}, 1: {"11"}}
    fo = FunctionOutputs(one_symbol_schemata=schemata, k=2)
    assert fo.verify_one_symbol_schemata() is True
    n = BooleanNode(
        id=0, name="AND", k=2, inputs=[0, 1], outputs=fo
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        n._check_compute_canalization_variables(prime_implicants=True)
        n._check_compute_canalization_variables(two_symbols=True)
    assert not [w for w in caught if issubclass(w.category, UserWarning)]
    assert n.outputs.one_symbol_schemata is schemata
    assert n.outputs.bit_lut is None


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
    fo.get_two_symbol_schemata()
    assert fo.one_symbol_is_prime_implicant is True
    fo.representations_consistent = True

    out = fo.reorder(["A", "B"], ["B", "A"])
    assert out is not fo
    assert list(out.bit_lut) == [1, 0, 1, 1]
    assert out.one_symbol_schemata is not None
    assert list(bit_lut_from_schemata(out.one_symbol_schemata, 2)) == [1, 0, 1, 1]
    assert out.one_symbol_is_prime_implicant is True
    assert out.representations_consistent is True
    assert out.two_symbol_schemata is not None
    expanded = one_symbol_from_two_symbol(out.two_symbol_schemata)
    assert list(bit_lut_from_schemata(expanded, 2)) == [1, 0, 1, 1]
    assert out.verify_representations_consistent() is True


def test_function_outputs_reorder_identity_returns_self():
    fo = FunctionOutputs.from_sequence([0, 0, 0, 1])
    assert fo.reorder(["A", "B"], ["A", "B"]) is fo


def test_one_symbol_module_public_api():
    assert "OneSymbolSchemata" in one_symbol_mod.__all__
    assert callable(one_symbol_mod.reorder)
    assert callable(one_symbol_mod.bit_lut_from_schemata)
    assert callable(one_symbol_mod.schemata_from_bit_lut)


def test_two_symbol_module_public_api():
    assert callable(two_symbol_mod.one_symbol_from_two_symbol)
    assert callable(two_symbol_mod.two_symbol_to_expression)
    assert callable(two_symbol_mod.reorder)


def test_two_symbol_reorder_identity_returns_same():
    ts = ([["00", [], [[0, 1]]]], [["12", [[0, 1]], []]])
    assert reorder_two_symbol(ts, ["A", "B"], ["A", "B"]) is ts


def test_two_symbol_reorder_swaps_or_and_same_groups():
    # OR ON entry: schema chars swap; groups stay ascending after remap.
    or_ts = ([["00", [], [[0, 1]]]], [["12", [[0, 1]], []]])
    swapped_or = reorder_two_symbol(or_ts, ["A", "B"], ["B", "A"])
    assert swapped_or == ([["00", [], [[0, 1]]]], [["21", [[0, 1]], []]])
    assert one_symbol_from_two_symbol(swapped_or)[1] == {"12", "21"}
    assert list(bit_lut_from_schemata(one_symbol_from_two_symbol(swapped_or), 2)) == [
        0,
        1,
        1,
        1,
    ]

    # AND: OFF has perm_groups; ON has same_groups — both remapped/sorted.
    and_ts = ([["02", [[0, 1]], []]], [["11", [], [[0, 1]]]])
    swapped_and = reorder_two_symbol(and_ts, ["A", "B"], ["B", "A"])
    assert swapped_and == ([["20", [[0, 1]], []]], [["11", [], [[0, 1]]]])
    assert list(bit_lut_from_schemata(one_symbol_from_two_symbol(swapped_and), 2)) == [
        0,
        0,
        0,
        1,
    ]


@pytest.mark.parametrize(
    "lut",
    [
        [0, 0, 0, 1],  # AND
        [0, 1, 1, 1],  # OR
        [0, 1, 1, 0],  # XOR
        [0, 0, 0, 0, 0, 0, 0, 1],  # ABC (k=3 AND)
    ],
)
def test_two_symbol_one_symbol_lut_roundtrip(lut):
    fo = FunctionOutputs.from_sequence(lut)
    original = list(fo.bit_lut)
    pre_one = {
        p: set(s) for p, s in fo.get_one_symbol_schemata().items()
    }
    ts = fo.get_two_symbol_schemata()

    # Drop dense / one-symbol; rebuild from two-symbol only.
    rebuilt = FunctionOutputs(two_symbol_schemata=ts, k=fo.k)
    assert rebuilt.bit_lut is None
    assert rebuilt.one_symbol_schemata is None

    expanded = rebuilt.get_one_symbol_schemata()
    assert rebuilt.one_symbol_schemata is expanded
    assert rebuilt.one_symbol_is_prime_implicant is None
    assert expanded[0] == pre_one[0]
    assert expanded[1] == pre_one[1]

    assert list(rebuilt.get_bit_lut()) == original
    assert rebuilt.bit_lut is not None


def test_get_bit_lut_from_two_symbol_is_two_step():
    # OR: polarity-1 schema "12" with perm group on both inputs.
    ts = (
        [["00", [], [[0, 1]]]],
        [["12", [[0, 1]], []]],
    )
    fo = FunctionOutputs(two_symbol_schemata=ts, k=2)
    assert fo.one_symbol_schemata is None
    assert fo.bit_lut is None
    lut = fo.get_bit_lut()
    assert fo.one_symbol_schemata is not None
    assert fo.one_symbol_schemata[1] == {"12", "21"}
    assert list(lut) == [0, 1, 1, 1]


def test_two_symbol_to_expression_unit_cases():
    names3 = ["x0", "x1", "x2"]
    # 022 + full perm → OR of negations (as ON cover for the unit test).
    ts_022 = ([], [["022", [[0, 1, 2]], []]])
    assert two_symbol_to_expression(names3, ts_022) == (
        "(not x0 or not x1 or not x2)"
    )
    # 112 + full perm → at-least-two ones.
    ts_112 = ([], [["112", [[0, 1, 2]], []]])
    assert two_symbol_to_expression(names3, ts_112) == (
        "((x0 and x1) or (x0 and x2) or (x1 and x2))"
    )
    # OR node polarity-1: "12" with perm on both inputs.
    names2 = ["x0", "x1"]
    ts_or = ([], [["12", [[0, 1]], []]])
    assert two_symbol_to_expression(names2, ts_or) == "(x0 or x1)"

    # Two perm groups ANDed: (x0|x1) and (x2|x3) from "1212".
    names4 = ["x0", "x1", "x2", "x3"]
    ts_multi = ([], [["1212", [[0, 1], [2, 3]], []]])
    assert two_symbol_to_expression(names4, ts_multi) == (
        "((x0 or x1) and (x2 or x3))"
    )

    assert two_symbol_to_expression(
        names2, ts_or, style="symbolic"
    ) == "(x0 | x1)"


def test_function_outputs_to_expression_from_two_symbol_only():
    ts = (
        [["00", [], []]],
        [["12", [[0, 1]], []]],
    )
    fo = FunctionOutputs(two_symbol_schemata=ts, k=2)
    assert (
        function_outputs_to_expression(fo, ["A", "B"], warn=False)
        == "(A or B)"
    )


def test_verify_representations_consistent_with_two_symbol():
    fo = FunctionOutputs.from_sequence([0, 1, 1, 1])
    fo.get_two_symbol_schemata()
    assert fo.verify_representations_consistent() is True

    # Agreeing LUT + two-symbol only (no stored one-symbol).
    ts = fo.two_symbol_schemata
    fo2 = FunctionOutputs(
        bit_lut=BitLut.from_sequence([0, 1, 1, 1]),
        two_symbol_schemata=ts,
        k=2,
    )
    assert fo2.verify_representations_consistent() is True

    # Mismatch.
    fo3 = FunctionOutputs(
        bit_lut=BitLut.from_sequence([0, 0, 0, 1]),
        two_symbol_schemata=ts,
        k=2,
    )
    assert fo3.verify_representations_consistent() is False


def test_one_symbol_from_two_symbol_no_perm_keeps_representative():
    ts = ([["11", [], [[0, 1]]]], [["00", [], []]])
    expanded = one_symbol_from_two_symbol(ts)
    assert expanded[0] == {"11"}
    assert expanded[1] == {"00"}
