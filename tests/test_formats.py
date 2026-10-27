"""Tests for multi-format Boolean network import / export / round-trips."""
from pathlib import Path

import pytest

from cana.boolean_network import BooleanNetwork
from cana.boolean_node import BooleanNode
from cana.function_outputs import FunctionOutputs
from cana.io import (
    convert,
    export_string,
    infer_format,
    nodes_to_cnet,
    parse_aeon,
    parse_bnet,
    parse_booleannet,
    parse_cnet,
    parse_string,
)
from cana.nodes_builder import (
    equivalent_networks,
    expressions_to_nodes,
    name_keyed_luts,
)

# Tiny network shared across formats:
#   A* = B
#   B* = A and C
#   C* = C          (free / identity input)
TINY_BNET = """targets,factors
A, B
B, (A & C)
"""

TINY_BOOLEANNET = """A* = B
B* = A and C
C* = C
"""

TINY_AEON = """B -? A
A -? B
C -? B
C -? C
$A: B
$B: A & C
"""


def _assert_tiny(nodes_or_net):
    luts = name_keyed_luts(nodes_or_net)
    assert set(luts) == {"A", "B", "C"}
    assert luts["A"] == (("B",), (0, 1))
    assert luts["C"] == (("C",), (0, 1))
    # B = A and C, inputs sorted (A, C): lut [0,0,0,1]
    assert luts["B"] == (("A", "C"), (0, 0, 0, 1))


def test_infer_format():
    assert infer_format("model.bnet") == "bnet"
    assert infer_format("model.aeon") == "aeon"
    assert infer_format("model.sbml") == "sbml"
    assert infer_format("model.booleannet.txt") == "booleannet"
    assert infer_format("model.cnet") == "cnet"
    assert infer_format("ambiguous.txt") is None


def test_parse_tiny_bnet_booleannet_aeon():
    _assert_tiny(parse_bnet(TINY_BNET))
    _assert_tiny(parse_booleannet(TINY_BOOLEANNET))
    _assert_tiny(parse_aeon(TINY_AEON))


def test_expressions_to_nodes_adds_free_input_identity():
    nodes = expressions_to_nodes({"A": "B", "B": "A and C"})
    _assert_tiny(nodes)


def test_import_stores_positional_expression_only():
    nodes = parse_booleannet(TINY_BOOLEANNET)
    # A* = B  -> inputs [B], expression x0
    # B* = A and C -> inputs [A, C] in discovery order, expression x0 and x1
    # C* = C -> inputs [C], expression x0
    by_name = {n.name: n for n in nodes}
    for node in by_name.values():
        assert node.outputs.bit_lut is None
        assert node.outputs.expression is not None
    assert by_name["A"].outputs.expression == "x0"
    assert by_name["B"].outputs.expression == "x0 and x1"
    assert by_name["C"].outputs.expression == "x0"
    # Symbolic operators normalize to python positional IR
    aeon = parse_aeon(TINY_AEON)
    assert {n.name: n.outputs.expression for n in aeon}["B"] == "x0 and x1"


def test_boolean_node_named_expression():
    network = BooleanNetwork.from_nodes(parse_booleannet(TINY_BOOLEANNET))
    by_name = {n.name: n for n in network.nodes}
    assert by_name["A"].expression() == "B"
    assert by_name["B"].expression() == "A and C"
    assert by_name["C"].expression() == "C"
    assert by_name["B"].input_names() == ["A", "C"]
    assert by_name["B"].expression(style="symbolic") == "A & C"


def test_export_preserves_imported_expression_strings():
    """Expression-first export keeps authored rules (not LUT minterm DNF)."""
    nodes = parse_booleannet(TINY_BOOLEANNET)
    text = export_string(nodes, "booleannet", warn=False)
    lines = {
        line.split("*", 1)[0].strip(): line.split("=", 1)[1].strip()
        for line in text.splitlines()
        if "*" in line and "=" in line
    }
    assert lines["A"] == "B"
    assert lines["B"] == "A and C"
    assert lines["C"] == "C"
    # Still LUT-equivalent after reimport.
    assert equivalent_networks(nodes, parse_booleannet(text))

    bnet_text = export_string(nodes, "bnet", warn=False)
    assert "B, A & C" in bnet_text or "B, A&C" in bnet_text.replace(" ", "")


def test_export_reimport_roundtrip_each_format():
    nodes = parse_booleannet(TINY_BOOLEANNET)
    for fmt in ("booleannet", "bnet", "aeon", "sbml", "cnet"):
        text = export_string(nodes, fmt, warn=False)
        again = parse_string(text, fmt)
        assert equivalent_networks(nodes, again), fmt


def test_any_to_any_convert():
    formats = ("booleannet", "bnet", "aeon", "sbml", "cnet")
    sources = {
        "booleannet": TINY_BOOLEANNET,
        "bnet": TINY_BNET,
        "aeon": TINY_AEON,
        "sbml": export_string(parse_booleannet(TINY_BOOLEANNET), "sbml"),
        "cnet": nodes_to_cnet(parse_booleannet(TINY_BOOLEANNET)),
    }
    ref = parse_booleannet(TINY_BOOLEANNET)
    for src in formats:
        for dst in formats:
            out = convert(sources[src], src, dst)
            assert equivalent_networks(ref, parse_string(out, dst)), (src, dst)


def test_parse_cnet_direct():
    nodes = parse_booleannet(TINY_BOOLEANNET)
    again = parse_cnet(nodes_to_cnet(nodes))
    assert equivalent_networks(nodes, again)


# Mixed: CycD dense identity; CycE wildcard one-symbol cover (mammalian-style).
MIXED_CNET = """\
.v 2

.l 1 CycD
.l 2 CycE

.n 1 1 1
0 0
1 1

.n 2 2 1 2
01 1
12 0
20 0
"""


def test_parse_cnet_wildcard_as_one_symbol_schemata():
    nd = parse_cnet(MIXED_CNET)
    cyc_e = nd[1].outputs
    assert cyc_e.bit_lut is None
    assert cyc_e.one_symbol_schemata is not None
    assert cyc_e.one_symbol_schemata[1] == {"01"}
    assert cyc_e.one_symbol_schemata[0] == {"12", "20"}
    # Dense sibling stays LUT-only (no auto-fill).
    cyc_d = nd[0].outputs
    assert cyc_d.bit_lut is not None
    assert cyc_d.one_symbol_schemata is None
    assert list(cyc_d.bit_lut) == [0, 1]


def test_parse_cnet_dense_minterms_not_schemata():
    # Fully specified 0/1 rows → dense LUT even if “PI-like”.
    dense = """\
.v 1
.l 1 AND
.n 1 2 1 1
00 0
01 0
10 0
11 1
"""
    nd = parse_cnet(dense)
    fo = nd[0].outputs
    assert fo.bit_lut is not None
    assert fo.one_symbol_schemata is None
    assert list(fo.bit_lut) == [0, 0, 0, 1]


def test_cnet_export_materializes_requested_representation():
    nd = parse_cnet(MIXED_CNET)
    # Dense export materializes bit_lut from schemata-only peers.
    dense_text = nodes_to_cnet(nd)
    assert "00 0" in dense_text or "0 0" in dense_text
    assert equivalent_networks(nd, parse_cnet(dense_text))

    # Schemata export materializes one-symbol from dense peers.
    schemata_text = nodes_to_cnet(nd, one_symbol_schemata=True)
    assert "2" in schemata_text
    assert equivalent_networks(nd, parse_cnet(schemata_text))

    # Truly empty FO still cannot export.
    empty = [
        BooleanNode(
            id=0, name="A", k=1, inputs=[0], outputs=FunctionOutputs(k=1)
        )
    ]
    with pytest.raises(ValueError, match=r"fill_function_outputs\('bit_lut'\)"):
        nodes_to_cnet(empty)


def test_boolean_network_cnet_schemata_and_fill():
    N = BooleanNetwork.from_string(MIXED_CNET, type="cnet")
    assert N.nodes[1].outputs.bit_lut is None
    assert N.nodes[1].outputs.one_symbol_schemata is not None
    N.fill_function_outputs("bit_lut")
    assert N.nodes[1].outputs.bit_lut is not None
    assert list(N.nodes[1].outputs.bit_lut) == [0, 1, 0, 0]
    N.fill_function_outputs("one_symbol_schemata")
    cnet = N.to_cnet(one_symbol_schemata=True)
    assert "2" in cnet
    N2 = BooleanNetwork.from_string(cnet, type="cnet")
    N2.fill_function_outputs("bit_lut")
    assert N.equivalent_to(N2)


def test_boolean_network_from_string_and_equivalent_to():
    N1 = BooleanNetwork.from_string(TINY_BOOLEANNET, type="booleannet")
    N2 = BooleanNetwork.from_string_bnet(TINY_BNET)
    N3 = BooleanNetwork.from_string_aeon(TINY_AEON)
    assert N1.equivalent_to(N2)
    assert N1.equivalent_to(N3)
    assert N1.Nnodes == 3


def test_to_string_roundtrip():
    N = BooleanNetwork.from_string(TINY_BOOLEANNET, type="booleannet")
    for fmt in ("booleannet", "bnet", "aeon", "sbml", "cnet"):
        text = N.to_string(type=fmt)
        N2 = BooleanNetwork.from_string(text, type=fmt)
        assert N.equivalent_to(N2), fmt


def test_convert_string_via_cnet():
    N = BooleanNetwork.from_string(TINY_BOOLEANNET, type="booleannet")
    cnet = BooleanNetwork.convert_string(
        TINY_BOOLEANNET, "booleannet", "cnet"
    )
    N2 = BooleanNetwork.from_string(cnet, type="cnet")
    assert N.equivalent_to(N2)
    back = BooleanNetwork.convert_string(cnet, "cnet", "bnet")
    N3 = BooleanNetwork.from_string(back, type="bnet")
    assert N.equivalent_to(N3)


def test_from_file_booleannet_explicit_type(tmp_path):
    path = tmp_path / "tiny.txt"
    path.write_text(TINY_BOOLEANNET)
    N = BooleanNetwork.from_file(str(path), type="booleannet")
    assert N.Nnodes == 3
    assert N.equivalent_to(BooleanNetwork.from_string(TINY_BNET, type="bnet"))


def test_ambiguous_format_aliases_rejected():
    with pytest.raises(ValueError, match="Unknown format"):
        parse_string(TINY_BOOLEANNET, "logical")
    with pytest.raises(ValueError, match="Unknown format"):
        parse_string(TINY_BOOLEANNET, "boolean")


def test_empty_inputs_implies_constant_not_converse():
    # Empty inputs => constant
    N = BooleanNetwork.from_nodes(
        [
            BooleanNode(id=0, name="Const1", k=0, inputs=[], outputs=[1]),
        ]
    )
    assert N.nodes[0].constant is True
    assert N.nodes[0].k == 0
    assert N.constants == {0: 1}
    assert N.Nconstants == 1
    assert list(N.get_constants()) == [0]

    # Flat LUT with inputs => constant, but inputs preserved
    N2 = BooleanNetwork.from_nodes(
        [
            BooleanNode(id=0, name="A", k=1, inputs=[0], outputs=[0, 1]),
            BooleanNode(id=1, name="Flat", k=1, inputs=[0], outputs=[1, 1]),
        ]
    )
    flat = N2.nodes[1]
    assert flat.constant is True
    assert flat.inputs == [0]
    assert N2.nodes[1].inputs == [0]
    assert N2.constants == {1: 1}
    assert N2.Nconstants == 1
    # Pinning shrinks free state space; constancy itself is independent.
    assert N2.keep_constants is True
    assert N2.Nstates == 2  # only node 0 is free

    N2_unpin = BooleanNetwork.from_nodes(
        [
            BooleanNode(id=0, name="A", k=1, inputs=[0], outputs=[0, 1]),
            BooleanNode(id=1, name="Flat", k=1, inputs=[0], outputs=[1, 1]),
        ],
        keep_constants=False,
    )
    assert N2_unpin.constants == {1: 1}
    assert N2_unpin.Nconstants == 1
    assert N2_unpin.Nstates == 4  # full 2**N when not pinning

    # Self-loop identity => not constant, in input_nodes
    N3 = BooleanNetwork.from_nodes(
        [
            BooleanNode(id=0, name="Src", k=1, inputs=[0], outputs=[0, 1]),
        ]
    )
    assert N3.nodes[0].constant is False
    assert 0 in N3.input_nodes
    assert N3.constants == {}
    assert N3.Nconstants == 0


BBM_023 = (
    Path(__file__).resolve().parents[2]
    / "BBM"
    / "models"
    / "[id-023]__[var-9]__[in-1]__[MAMMALIAN-CELL-CYCLE-2006]"
)


@pytest.mark.skipif(not BBM_023.is_dir(), reason="BBM corpus not checked out")
def test_bbm_023_all_formats_equivalent():
    files = {
        "booleannet": BBM_023 / "model.booleannet.txt",
        "bnet": BBM_023 / "model.bnet",
        "aeon": BBM_023 / "model.aeon",
        "sbml": BBM_023 / "model.sbml",
    }
    networks = {
        fmt: BooleanNetwork.from_file(str(path)) for fmt, path in files.items()
    }
    ref = networks["booleannet"]
    assert ref.Nnodes == 10
    for fmt, N in networks.items():
        assert ref.equivalent_to(N), fmt

    # Round-trip each through every other format
    for src_fmt, N in networks.items():
        for dst_fmt in files:
            text = N.to_string(type=dst_fmt)
            N2 = BooleanNetwork.from_string(text, type=dst_fmt)
            assert ref.equivalent_to(N2), (src_fmt, dst_fmt)
