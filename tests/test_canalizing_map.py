# -*- coding: utf-8 -*-
"""
Tests for schemata → canalizing map construction and CM → LUT roundtrips.

Covers the legacy forward builder (``BooleanNode.canalizing_map`` /
``build_canalizing_map_legacy``) rather than hand-built CM graphs.
"""
from __future__ import annotations

import networkx as nx
import pytest

from cana.boolean_network import BooleanNetwork
from cana.boolean_node import BooleanNode
from cana.canalization.canalizing_map import (
    build_canalizing_map,
    build_canalizing_map_legacy,
)
from cana.canalization.cm_lut import lut_from_canalizing_map


def _graphs_equal(a: nx.DiGraph, b: nx.DiGraph) -> bool:
    """True if *a* and *b* have the same nodes, edges, and attributes."""
    if set(a.nodes) != set(b.nodes) or set(a.edges) != set(b.edges):
        return False
    if a.graph.get("name") != b.graph.get("name"):
        return False
    for n in a.nodes:
        if dict(a.nodes[n]) != dict(b.nodes[n]):
            return False
    for u, v in a.edges:
        if dict(a.edges[u, v]) != dict(b.edges[u, v]):
            return False
    return True


def _assert_exact_roundtrip(node: BooleanNode) -> None:
    cm = node.canalizing_map()
    result = lut_from_canalizing_map(cm, input_order=node.inputs)
    assert not result.has_conflict
    assert not result.is_incomplete
    assert result.outputs == [int(x) for x in node.outputs]


class TestParity:
    """BooleanNode wrapper matches build_canalizing_map_legacy."""

    @pytest.mark.parametrize(
        "outputs,name,inputs,nid",
        [
            ([0, 0, 0, 1], "AND", [0, 1], 2),
            ([0, 1, 1, 1], "OR", [10, 11], 5),
        ],
    )
    def test_node_matches_legacy_builder(self, outputs, name, inputs, nid):
        node = BooleanNode.from_output_list(
            outputs, name=name, inputs=inputs, id=nid
        )
        node._check_compute_canalization_variables(two_symbols=True)
        input_labels = {i: str(i) for i in node.inputs}

        via_node = node.canalizing_map()
        via_fn = build_canalizing_map_legacy(
            node._two_symbols,
            node_id=node.id,
            inputs=node.inputs,
            name=node.name,
            constant=node.constant,
            input_labels=input_labels,
            k=node.k,
        )
        via_dispatch = build_canalizing_map(
            node._two_symbols,
            node_id=node.id,
            inputs=node.inputs,
            name=node.name,
            constant=node.constant,
            input_labels=input_labels,
            k=node.k,
            builder="legacy",
        )
        assert _graphs_equal(via_node, via_fn)
        assert _graphs_equal(via_node, via_dispatch)


# Previously failing under old fibre wiring (all-LUT roundtrip logs).
K3_REGRESSION_LUTS = [
    "01101000",  # id 22
    "11101000",  # id 23
    "00010110",  # id 104
    "10010110",  # id 105
    "01101001",  # id 150
    "11101001",  # id 151
    "00010111",  # id 232
    "10010111",  # id 233
]


class TestK3RegressionRoundtrip:
    @pytest.mark.parametrize("lut", K3_REGRESSION_LUTS)
    def test_exact_roundtrip(self, lut):
        outputs = [int(b) for b in lut]
        node = BooleanNode.from_output_list(
            outputs, name="k3_{}".format(lut), inputs=[0, 1, 2], id=0
        )
        _assert_exact_roundtrip(node)


CTNNB1_BNET = (
    "CTNNB1*=(not SRY) and (WNT4 or RSPO1) and ((not AMH) or (not SOX9))\n"
    "SRY*=SRY\n"
    "WNT4*=WNT4\n"
    "RSPO1*=RSPO1\n"
    "AMH*=AMH\n"
    "SOX9*=SOX9\n"
)
CTNNB1_LUT = "00001110111011100000000000000000"


class TestCTNNB1Roundtrip:
    """k=5 CTNNB1 example from ``dev/cm_roundtrip.py``."""

    def test_exact_roundtrip_via_bnet(self):
        N = BooleanNetwork.from_string_boolean(CTNNB1_BNET)
        ct = next(n for n in N.nodes if n.name == "CTNNB1")
        assert "".join(str(x) for x in ct.outputs) == CTNNB1_LUT
        _assert_exact_roundtrip(ct)

    def test_exact_roundtrip_from_lut(self):
        outputs = [int(b) for b in CTNNB1_LUT]
        node = BooleanNode.from_output_list(
            outputs,
            name="CTNNB1",
            inputs=[0, 1, 2, 3, 4],
            id=5,
        )
        _assert_exact_roundtrip(node)
