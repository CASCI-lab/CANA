"""Tests for multi-format Boolean network import / export / round-trips."""
from pathlib import Path

import pytest

from cana.boolean_network import BooleanNetwork
from cana.io.formats import (
    convert,
    equivalent_logic,
    export_string,
    infer_format,
    logic_to_cnet,
    name_keyed_luts,
    parse_aeon,
    parse_bnet,
    parse_booleannet,
    parse_cnet,
    parse_sbml,
    parse_string,
    rules_to_logic,
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


def _assert_tiny(logic_or_net):
    luts = name_keyed_luts(logic_or_net)
    assert set(luts) == {"A", "B", "C"}
    assert luts["A"] == (("B",), (0, 1))
    assert luts["C"] == (("C",), (0, 1))
    # B = A and C, inputs sorted (A, C): out [0,0,0,1]
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


def test_rules_to_logic_adds_free_input_identity():
    logic = rules_to_logic({"A": "B", "B": "A and C"})
    _assert_tiny(logic)


def test_export_reimport_roundtrip_each_format():
    logic = parse_booleannet(TINY_BOOLEANNET)
    for fmt in ("booleannet", "bnet", "aeon", "sbml", "cnet"):
        text = export_string(logic, fmt)
        again = parse_string(text, fmt)
        assert equivalent_logic(logic, again), fmt


def test_any_to_any_convert():
    formats = ("booleannet", "bnet", "aeon", "sbml", "cnet")
    sources = {
        "booleannet": TINY_BOOLEANNET,
        "bnet": TINY_BNET,
        "aeon": TINY_AEON,
        "sbml": export_string(parse_booleannet(TINY_BOOLEANNET), "sbml"),
        "cnet": logic_to_cnet(parse_booleannet(TINY_BOOLEANNET)),
    }
    ref = parse_booleannet(TINY_BOOLEANNET)
    for src in formats:
        for dst in formats:
            out = convert(sources[src], src, dst)
            assert equivalent_logic(ref, parse_string(out, dst)), (src, dst)


def test_parse_cnet_direct():
    logic = parse_booleannet(TINY_BOOLEANNET)
    again = parse_cnet(logic_to_cnet(logic))
    assert equivalent_logic(logic, again)


def test_boolean_network_from_string_and_equivalent_to():
    N1 = BooleanNetwork.from_string(TINY_BOOLEANNET, type="booleannet")
    N2 = BooleanNetwork.from_string_bnet(TINY_BNET)
    N3 = BooleanNetwork.from_string_aeon(TINY_AEON)
    assert N1.equivalent_to(N2)
    assert N1.equivalent_to(N3)
    assert N1.Nnodes == 3


def test_to_string_roundtrip():
    N = BooleanNetwork.from_string(TINY_BOOLEANNET, type="logical")
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


def test_from_file_logical_alias_still_works(tmp_path):
    path = tmp_path / "tiny.txt"
    path.write_text(TINY_BOOLEANNET)
    N = BooleanNetwork.from_file(str(path), type="logical")
    assert N.Nnodes == 3
    assert N.equivalent_to(BooleanNetwork.from_string(TINY_BNET, type="bnet"))


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
