# -*- coding: utf-8 -*-
"""
Firm tests for CM → LUT reconstruction (``cm_lut``).

All canalizing maps here are hand-built NetworkX graphs.  They do **not**
depend on ``BooleanNode.canalizing_map``, which is known to be incorrect for
non-trivial schemata.
"""
from __future__ import annotations

import os
import sys

import networkx as nx
import pytest

# Allow importing the sibling module when pytest is run from the repo root
_DEV_DIR = os.path.dirname(os.path.abspath(__file__))
if _DEV_DIR not in sys.path:
    sys.path.insert(0, _DEV_DIR)

from cm_lut import (  # noqa: E402
    ROLE_POLARITY0,
    ROLE_POLARITY1,
    ROLE_SCHEMA,
    ROLE_SYMMETRY,
    claimed_outputs,
    connect,
    fusion_active,
    input_ids_in_order,
    lut_from_canalizing_map,
    make_fusion,
    make_input_node,
    make_output_node,
    make_threshold,
    node_active,
    tunit_fires,
    tunit_score,
)


# ---------------------------------------------------------------------------
# Hand-built CM factories
# ---------------------------------------------------------------------------


def hand_and_out1_cm() -> nx.DiGraph:
    """AND output-1 half: literals A=1 and B=1 into tau=2 → claims out=1.

    Covers only input state ``11``.
    """
    G = nx.DiGraph(name="AND-out1")
    out1 = make_output_node(G, "AND", 1)
    thr = make_threshold(G, tau=2)
    connect(G, make_input_node(G, "A", 1), thr)
    connect(G, make_input_node(G, "B", 1), thr)
    connect(G, thr, out1)
    return G


def hand_and_out0_cm() -> nx.DiGraph:
    """AND output-0 half: fusion OR of A=0, B=0 into tau=1 → claims out=0.

    Covers ``00``, ``01``, ``10``.
    """
    G = nx.DiGraph(name="AND-out0")
    out0 = make_output_node(G, "AND", 0)
    thr = make_threshold(G, tau=1)
    fus = make_fusion(G)
    connect(G, make_input_node(G, "A", 0), fus)
    connect(G, make_input_node(G, "B", 0), fus)
    connect(G, fus, thr)
    connect(G, thr, out0)
    return G


def hand_and_cm() -> nx.DiGraph:
    """Full AND canalizing map (both output halves)."""
    G = nx.DiGraph(name="AND")
    out0 = make_output_node(G, "AND", 0)
    out1 = make_output_node(G, "AND", 1)
    thr0 = make_threshold(G, tau=1)
    fus0 = make_fusion(G)
    connect(G, make_input_node(G, "A", 0), fus0)
    connect(G, make_input_node(G, "B", 0), fus0)
    connect(G, fus0, thr0)
    connect(G, thr0, out0)
    thr1 = make_threshold(G, tau=2)
    connect(G, make_input_node(G, "A", 1), thr1)
    connect(G, make_input_node(G, "B", 1), thr1)
    connect(G, thr1, out1)
    return G


def hand_conflict_cm() -> nx.DiGraph:
    """Two literal t-units that both fire on ``11`` but claim opposite outputs."""
    G = nx.DiGraph(name="conflict")
    out0 = make_output_node(G, "X", 0)
    out1 = make_output_node(G, "X", 1)
    thr0 = make_threshold(G, tau=1)
    a1 = make_input_node(G, "A", 1)
    b1 = make_input_node(G, "B", 1)
    connect(G, a1, thr0)
    connect(G, thr0, out0)
    thr1 = make_threshold(G, tau=2)
    connect(G, a1, thr1)
    connect(G, b1, thr1)
    connect(G, thr1, out1)
    return G


def hand_tau0_cm() -> nx.DiGraph:
    """tau=0 t-unit with no parents: always fires, claims out=1 for all states."""
    G = nx.DiGraph(name="tau0")
    # Free variable so the LUT has dimension; never wired.
    make_input_node(G, "A", 0)
    make_input_node(G, "A", 1)
    out1 = make_output_node(G, "const1", 1)
    thr = make_threshold(G, tau=0)
    connect(G, thr, out1)
    return G


def hand_or_nested_cm() -> nx.DiGraph:
    """OR via nested thresholds (leaf-first, then root→output).

    out=0: both A=0 and B=0 into schema tau=2
    out=1: polarity1 {A=1, B=1} into symmetry tau=1 into schema tau=1
    """
    G = nx.DiGraph(name="OR-nested")
    out0 = make_output_node(G, "OR", 0)
    out1 = make_output_node(G, "OR", 1)

    thr0 = make_threshold(G, tau=2, role=ROLE_SCHEMA)
    connect(G, make_input_node(G, "A", 0), thr0)
    connect(G, make_input_node(G, "B", 0), thr0)
    connect(G, thr0, out0)

    # Leaf-first nesting, then connect schema to output last.
    grp = make_threshold(G, tau=1, role=ROLE_POLARITY1)
    sym = make_threshold(G, tau=1, role=ROLE_SYMMETRY)
    thr1 = make_threshold(G, tau=1, role=ROLE_SCHEMA)
    connect(G, make_input_node(G, "A", 1), grp)
    connect(G, make_input_node(G, "B", 1), grp)
    connect(G, grp, sym)
    connect(G, sym, thr1)
    connect(G, thr1, out1)
    return G


# True LUT for
#   CTNNB1 = (not SRY) and (WNT4 or RSPO1) and ((not AMH) or (not SOX9))
# with input order SRY, WNT4, RSPO1, AMH, SOX9.
CTNNB1_LUT = "00001110111011100000000000000000"


def hand_ctnnb1_cm() -> nx.DiGraph:
    """Hand-correct CTNNB1 CM using fusion fibres (from ``cm_roundtrip`` Case 3).

    out=0: three independent thresholds (SRY=1; WNT4=0 & RSPO1=0; AMH=1 & SOX9=1)
    out=1: tau=3 over SRY=0 plus two fusion ORs (WNT4|RSPO1, AMH|SOX9 negated)
    """
    G = nx.DiGraph(name="real_CTNNB1")
    out0 = make_output_node(G, "CTNNB1", 0)
    out1 = make_output_node(G, "CTNNB1", 1)

    thr0 = make_threshold(G, tau=1)
    thr1 = make_threshold(G, tau=2)
    thr2 = make_threshold(G, tau=2)
    connect(G, thr0, out0)
    connect(G, thr1, out0)
    connect(G, thr2, out0)
    connect(G, make_input_node(G, "SRY", 1), thr0)
    connect(G, make_input_node(G, "WNT4", 0), thr1)
    connect(G, make_input_node(G, "RSPO1", 0), thr1)
    connect(G, make_input_node(G, "AMH", 1), thr2)
    connect(G, make_input_node(G, "SOX9", 1), thr2)

    thr3 = make_threshold(G, tau=3)
    connect(G, thr3, out1)
    fus0 = make_fusion(G)
    fus1 = make_fusion(G)
    connect(G, fus0, thr3)
    connect(G, fus1, thr3)
    connect(G, make_input_node(G, "SRY", 0), thr3)
    connect(G, make_input_node(G, "WNT4", 1), fus0)
    connect(G, make_input_node(G, "RSPO1", 1), fus0)
    connect(G, make_input_node(G, "AMH", 0), fus1)
    connect(G, make_input_node(G, "SOX9", 0), fus1)
    return G


def hand_ctnnb1_canonical_cm() -> nx.DiGraph:
    """Canonical nested-threshold CTNNB1 CM (from ``cm_roundtrip`` Case 4).

    Same function as ``hand_ctnnb1_cm``, but out=1 uses nested
    schema / symmetry / polarity thresholds instead of fusion nodes.
    """
    G = nx.DiGraph(name="canonical_CTNNB1")
    out0 = make_output_node(G, "CTNNB1", 0)
    out1 = make_output_node(G, "CTNNB1", 1)

    # out=0: three independent two-symbol schemata
    out0_two0 = make_threshold(G, tau=1, role=ROLE_SCHEMA)
    out0_two1 = make_threshold(G, tau=2, role=ROLE_SCHEMA)
    out0_two2 = make_threshold(G, tau=2, role=ROLE_SCHEMA)
    connect(G, out0_two0, out0)
    connect(G, out0_two1, out0)
    connect(G, out0_two2, out0)
    connect(G, make_input_node(G, "SRY", 1), out0_two0)
    connect(G, make_input_node(G, "WNT4", 0), out0_two1)
    connect(G, make_input_node(G, "RSPO1", 0), out0_two1)
    connect(G, make_input_node(G, "AMH", 1), out0_two2)
    connect(G, make_input_node(G, "SOX9", 1), out0_two2)

    # out=1: schema over SRY literal + two symmetry fibres (leaf-first ok)
    out1_two0 = make_threshold(G, tau=3, role=ROLE_SCHEMA)
    connect(G, out1_two0, out1)
    out1_two0_sym0 = make_threshold(G, tau=1, role=ROLE_SYMMETRY)
    out1_two0_sym1 = make_threshold(G, tau=1, role=ROLE_SYMMETRY)
    connect(G, out1_two0_sym0, out1_two0)
    connect(G, out1_two0_sym1, out1_two0)
    out1_two0_sym0_g1 = make_threshold(G, tau=1, role=ROLE_POLARITY1)
    out1_two0_sym1_g0 = make_threshold(G, tau=1, role=ROLE_POLARITY0)
    connect(G, out1_two0_sym0_g1, out1_two0_sym0)
    connect(G, out1_two0_sym1_g0, out1_two0_sym1)
    connect(G, make_input_node(G, "SRY", 0), out1_two0)
    connect(G, make_input_node(G, "WNT4", 1), out1_two0_sym0_g1)
    connect(G, make_input_node(G, "RSPO1", 1), out1_two0_sym0_g1)
    connect(G, make_input_node(G, "AMH", 0), out1_two0_sym1_g0)
    connect(G, make_input_node(G, "SOX9", 0), out1_two0_sym1_g0)
    return G


def _thr(G: nx.DiGraph, tid: int) -> str:
    return "thr-{tid:d}".format(tid=tid)


# ---------------------------------------------------------------------------
# t-unit evaluation
# ---------------------------------------------------------------------------


class TestTunitFires:
    def test_literal_both_required(self):
        G = hand_and_out1_cm()
        thr = _thr(G, 0)
        order = input_ids_in_order(G)
        a, b = order[0], order[1]
        assert tunit_score(G, thr, {a: 1, b: 1}) == 2
        assert tunit_fires(G, thr, {a: 1, b: 1}) is True
        assert tunit_fires(G, thr, {a: 1, b: 0}) is False
        assert tunit_fires(G, thr, {a: 0, b: 1}) is False
        assert tunit_fires(G, thr, {a: 0, b: 0}) is False

    def test_fusion_or_tau1(self):
        G = hand_and_out0_cm()
        thr = _thr(G, 0)
        fus = "fus-0"
        order = input_ids_in_order(G)
        a, b = order[0], order[1]
        assert fusion_active(G, fus, {a: 0, b: 1}) is True
        assert fusion_active(G, fus, {a: 1, b: 0}) is True
        assert fusion_active(G, fus, {a: 0, b: 0}) is True
        assert fusion_active(G, fus, {a: 1, b: 1}) is False
        assert tunit_fires(G, thr, {a: 0, b: 1}) is True
        assert tunit_fires(G, thr, {a: 1, b: 1}) is False

    def test_tau_zero_always_fires(self):
        G = hand_tau0_cm()
        thr = _thr(G, 0)
        a = input_ids_in_order(G)[0]
        assert tunit_score(G, thr, {a: 0}) == 0
        assert tunit_fires(G, thr, {a: 0}) is True
        assert tunit_fires(G, thr, {a: 1}) is True

    def test_non_threshold_raises(self):
        G = hand_and_out1_cm()
        a1 = "var-{nid:d}-out-1".format(nid=input_ids_in_order(G)[0])
        with pytest.raises(ValueError):
            tunit_fires(G, a1, {input_ids_in_order(G)[0]: 1})


# ---------------------------------------------------------------------------
# LUT reconstruction
# ---------------------------------------------------------------------------


class TestLutFromCM:
    def test_literal_only_half_incomplete(self):
        G = hand_and_out1_cm()
        result = lut_from_canalizing_map(G, input_order=input_ids_in_order(G))
        assert result.outputs == [None, None, None, 1]
        assert result.is_incomplete is True
        assert result.has_conflict is False
        assert set(result.uncovered_states) == {"00", "01", "10"}
        assert result.conflict_states == []

    def test_fusion_or_half_incomplete(self):
        G = hand_and_out0_cm()
        result = lut_from_canalizing_map(G, input_order=input_ids_in_order(G))
        assert result.outputs == [0, 0, 0, None]
        assert result.is_incomplete is True
        assert result.has_conflict is False
        assert result.uncovered_states == ["11"]

    def test_full_and_complete(self):
        G = hand_and_cm()
        result = lut_from_canalizing_map(G, input_order=input_ids_in_order(G))
        assert result.outputs == [0, 0, 0, 1]
        assert result.is_incomplete is False
        assert result.has_conflict is False
        assert result.uncovered_states == []
        assert result.conflict_states == []

    def test_conflict_on_shared_state(self):
        G = hand_conflict_cm()
        order = input_ids_in_order(G)
        result = lut_from_canalizing_map(G, input_order=order)
        a, b = order[0], order[1]
        assert result.has_conflict is True
        assert "11" in result.conflict_states
        assert claimed_outputs(G, {a: 1, b: 1}) == {0, 1}
        assert result.outputs[3] is None  # index of 11
        assert result.outputs[2] == 0  # 10
        assert result.is_incomplete is True  # 00 and 01 uncovered

    def test_tau0_covers_all(self):
        G = hand_tau0_cm()
        result = lut_from_canalizing_map(G, input_order=input_ids_in_order(G))
        assert result.outputs == [1, 1]
        assert result.is_incomplete is False
        assert result.has_conflict is False

    def test_discover_input_order(self):
        G = hand_and_cm()
        result = lut_from_canalizing_map(G)  # no explicit order
        assert result.input_names == sorted(input_ids_in_order(G))
        assert result.outputs == [0, 0, 0, 1]

    def test_lut_string(self):
        G = hand_and_out1_cm()
        result = lut_from_canalizing_map(G, input_order=input_ids_in_order(G))
        assert result.lut_string() == "???1"

    def test_nested_or_complete(self):
        G = hand_or_nested_cm()
        result = lut_from_canalizing_map(G, input_order=input_ids_in_order(G))
        assert result.outputs == [0, 1, 1, 1]
        assert result.is_incomplete is False
        assert result.has_conflict is False

    def test_hand_ctnnb1_complete(self):
        G = hand_ctnnb1_cm()
        result = lut_from_canalizing_map(G, input_order=input_ids_in_order(G))
        expected = [int(bit) for bit in CTNNB1_LUT]
        assert result.outputs == expected
        assert result.is_incomplete is False
        assert result.has_conflict is False

    def test_canonical_ctnnb1_complete(self):
        G = hand_ctnnb1_canonical_cm()
        result = lut_from_canalizing_map(G, input_order=input_ids_in_order(G))
        expected = [int(bit) for bit in CTNNB1_LUT]
        assert result.outputs == expected
        assert result.is_incomplete is False
        assert result.has_conflict is False

    def test_ignores_edge_types(self):
        """Topology evaluation still works even if edge types are wrong."""
        G = nx.DiGraph(name="AND-no-edge-types")
        out1 = make_output_node(G, "AND", 1)
        thr = make_threshold(G, tau=2)
        a1 = make_input_node(G, "A", 1)
        b1 = make_input_node(G, "B", 1)
        G.add_edge(a1, thr, type="whatever")
        G.add_edge(b1, thr)  # no type
        G.add_edge(thr, out1, type="2sym")
        order = input_ids_in_order(G)
        a, b = order[0], order[1]
        result = lut_from_canalizing_map(G, input_order=order)
        assert result.outputs == [None, None, None, 1]
        assert node_active(G, thr, {a: 1, b: 1}) is True
        assert node_active(G, thr, {a: 1, b: 0}) is False


class TestThresholdRole:
    def test_default_role_is_schema(self):
        G = nx.DiGraph()
        thr = make_threshold(G, tau=1)
        assert G.nodes[thr]["role"] == ROLE_SCHEMA

    def test_roles_stored(self):
        G = nx.DiGraph()
        t0 = make_threshold(G, tau=1, role=ROLE_SCHEMA)
        t1 = make_threshold(G, tau=1, role=ROLE_SYMMETRY)
        t2 = make_threshold(G, tau=1, role=ROLE_POLARITY0)
        t3 = make_threshold(G, tau=1, role=ROLE_POLARITY1)
        assert G.nodes[t0]["role"] == "schema"
        assert G.nodes[t1]["role"] == "symmetry"
        assert G.nodes[t2]["role"] == "polarity0"
        assert G.nodes[t3]["role"] == "polarity1"
        assert t0 == "thr-0"
        assert t3 == "thr-3"
        assert "out" not in G.nodes[t0]

    def test_invalid_role_raises(self):
        G = nx.DiGraph()
        with pytest.raises(ValueError, match="Unknown threshold role"):
            make_threshold(G, tau=1, role="fibre")


class TestConstructionHelpers:
    def test_duplicate_input_polarity_raises(self):
        G = nx.DiGraph()
        make_input_node(G, "SRY", 1)
        with pytest.raises(ValueError, match="already exists"):
            make_input_node(G, "SRY", 1)

    def test_same_label_other_polarity_reuses_id(self):
        G = nx.DiGraph()
        s1 = make_input_node(G, "SRY", 1)
        s0 = make_input_node(G, "SRY", 0)
        assert G.nodes[s1]["label"] == "SRY"
        assert G.nodes[s0]["label"] == "SRY"
        assert input_ids_in_order(G) == [0]
        assert s1 == "var-0-out-1"
        assert s0 == "var-0-out-0"

    def test_auto_tid(self):
        G = nx.DiGraph()
        assert make_threshold(G, tau=1) == "thr-0"
        assert make_threshold(G, tau=2) == "thr-1"

    def test_connect_infers_edge_types(self):
        G = nx.DiGraph()
        out1 = make_output_node(G, "AND", 1)
        thr = make_threshold(G, tau=2)
        a1 = make_input_node(G, "A", 1)
        connect(G, a1, thr)
        connect(G, thr, out1)
        assert G.edges[a1, thr]["type"] == "literal"
        assert G.edges[thr, out1]["type"] == "out"

    def test_connect_threshold_to_threshold_is_feed(self):
        G = nx.DiGraph()
        t0 = make_threshold(G, tau=1)
        t1 = make_threshold(G, tau=1)
        connect(G, t0, t1)
        assert G.edges[t0, t1]["type"] == "feed"

    def test_connect_illegal_pair_raises(self):
        G = nx.DiGraph()
        out0 = make_output_node(G, "X", 0)
        a0 = make_input_node(G, "A", 0)
        with pytest.raises(ValueError, match="Cannot connect"):
            connect(G, a0, out0)

    def test_duplicate_output_polarity_raises(self):
        G = nx.DiGraph()
        make_output_node(G, "CTNNB1", 0)
        with pytest.raises(ValueError, match="already exists"):
            make_output_node(G, "CTNNB1", 0)

    def test_second_output_label_raises(self):
        G = nx.DiGraph()
        make_output_node(G, "CTNNB1", 0)
        with pytest.raises(ValueError, match="already has output label"):
            make_output_node(G, "OTHER", 1)

    def test_leaf_first_nested_still_evaluates(self):
        """Build polarity→symmetry→schema, connect to output last."""
        G = hand_or_nested_cm()
        # Nested thresholds must not rely on a stored ``out`` attribute.
        for n, d in G.nodes(data=True):
            if d.get("type") == "threshold":
                assert "out" not in d
        result = lut_from_canalizing_map(G, input_order=input_ids_in_order(G))
        assert result.outputs == [0, 1, 1, 1]
