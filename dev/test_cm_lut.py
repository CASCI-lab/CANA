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
    fusion_active,
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
    out1 = make_output_node(G, node_id=2, value=1, label="AND")
    thr = make_threshold(G, tid=0, node_id=2, out=1, tau=2)
    a1 = make_input_node(G, var_id=0, value=1, label="A")
    b1 = make_input_node(G, var_id=1, value=1, label="B")
    G.add_edge(a1, thr, type="literal")
    G.add_edge(b1, thr, type="literal")
    G.add_edge(thr, out1, type="out")
    return G


def hand_and_out0_cm() -> nx.DiGraph:
    """AND output-0 half: fusion OR of A=0, B=0 into tau=1 → claims out=0.

    Covers ``00``, ``01``, ``10``.
    """
    G = nx.DiGraph(name="AND-out0")
    out0 = make_output_node(G, node_id=2, value=0, label="AND")
    thr = make_threshold(G, tid=0, node_id=2, out=0, tau=1)
    fus = make_fusion(G, fus=0, tid=0, node_id=2, out=0)
    a0 = make_input_node(G, var_id=0, value=0, label="A")
    b0 = make_input_node(G, var_id=1, value=0, label="B")
    G.add_edge(a0, fus, type="fusing")
    G.add_edge(b0, fus, type="fusing")
    G.add_edge(fus, thr, type="fused")
    G.add_edge(thr, out0, type="out")
    return G


def hand_and_cm() -> nx.DiGraph:
    """Full AND canalizing map (both output halves)."""
    G = nx.DiGraph(name="AND")
    # out 0
    out0 = make_output_node(G, node_id=2, value=0, label="AND")
    thr0 = make_threshold(G, tid=0, node_id=2, out=0, tau=1)
    fus0 = make_fusion(G, fus=0, tid=0, node_id=2, out=0)
    a0 = make_input_node(G, var_id=0, value=0, label="A")
    b0 = make_input_node(G, var_id=1, value=0, label="B")
    G.add_edge(a0, fus0, type="fusing")
    G.add_edge(b0, fus0, type="fusing")
    G.add_edge(fus0, thr0, type="fused")
    G.add_edge(thr0, out0, type="out")
    # out 1
    out1 = make_output_node(G, node_id=2, value=1, label="AND")
    thr1 = make_threshold(G, tid=1, node_id=2, out=1, tau=2)
    a1 = make_input_node(G, var_id=0, value=1, label="A")
    b1 = make_input_node(G, var_id=1, value=1, label="B")
    G.add_edge(a1, thr1, type="literal")
    G.add_edge(b1, thr1, type="literal")
    G.add_edge(thr1, out1, type="out")
    return G


def hand_conflict_cm() -> nx.DiGraph:
    """Two literal t-units that both fire on ``11`` but claim opposite outputs."""
    G = nx.DiGraph(name="conflict")
    out0 = make_output_node(G, node_id=9, value=0, label="X")
    out1 = make_output_node(G, node_id=9, value=1, label="X")
    # thr claiming 0: A=1 alone, tau=1 → fires whenever A=1
    thr0 = make_threshold(G, tid=0, node_id=9, out=0, tau=1)
    a1 = make_input_node(G, var_id=0, value=1, label="A")
    b1 = make_input_node(G, var_id=1, value=1, label="B")
    G.add_edge(a1, thr0, type="literal")
    G.add_edge(thr0, out0, type="out")
    # thr claiming 1: A=1 and B=1, tau=2 → fires only on 11
    thr1 = make_threshold(G, tid=1, node_id=9, out=1, tau=2)
    G.add_edge(a1, thr1, type="literal")
    G.add_edge(b1, thr1, type="literal")
    G.add_edge(thr1, out1, type="out")
    return G


def hand_tau0_cm() -> nx.DiGraph:
    """tau=0 t-unit with no parents: always fires, claims out=1 for all states."""
    G = nx.DiGraph(name="tau0")
    # Still need at least one input so the LUT has dimension; add a dummy input
    # that is never wired — free variable.
    make_input_node(G, var_id=0, value=0, label="A")
    make_input_node(G, var_id=0, value=1, label="A")
    out1 = make_output_node(G, node_id=3, value=1, label="const1")
    thr = make_threshold(G, tid=0, node_id=3, out=1, tau=0)
    G.add_edge(thr, out1, type="out")
    return G


def hand_or_nested_cm() -> nx.DiGraph:
    """OR via nested thresholds (no fusion, no edge types).

    out=0: both A=0 and B=0 into schema tau=2
    out=1: polarity1 {A=1, B=1} into symmetry tau=1 into schema tau=1
    """
    G = nx.DiGraph(name="OR-nested")
    out0 = make_output_node(G, node_id=5, value=0, label="OR")
    out1 = make_output_node(G, node_id=5, value=1, label="OR")

    thr0 = make_threshold(
        G, tid=0, node_id=5, out=0, tau=2, role=ROLE_SCHEMA
    )
    a0 = make_input_node(G, var_id=0, value=0, label="A")
    b0 = make_input_node(G, var_id=1, value=0, label="B")
    G.add_edge(a0, thr0)
    G.add_edge(b0, thr0)
    G.add_edge(thr0, out0)

    thr1 = make_threshold(
        G, tid=1, node_id=5, out=1, tau=1, role=ROLE_SCHEMA
    )
    sym = make_threshold(
        G, tid=2, node_id=5, out=1, tau=1, role=ROLE_SYMMETRY
    )
    grp = make_threshold(
        G, tid=3, node_id=5, out=1, tau=1, role=ROLE_POLARITY1
    )
    a1 = make_input_node(G, var_id=0, value=1, label="A")
    b1 = make_input_node(G, var_id=1, value=1, label="B")
    G.add_edge(a1, grp)
    G.add_edge(b1, grp)
    G.add_edge(grp, sym)
    G.add_edge(sym, thr1)
    G.add_edge(thr1, out1)
    return G


# ---------------------------------------------------------------------------
# t-unit evaluation
# ---------------------------------------------------------------------------


class TestTunitFires:
    def test_literal_both_required(self):
        G = hand_and_out1_cm()
        thr = "thr-0-var-2-out-1"
        assert tunit_score(G, thr, {0: 1, 1: 1}) == 2
        assert tunit_fires(G, thr, {0: 1, 1: 1}) is True
        assert tunit_fires(G, thr, {0: 1, 1: 0}) is False
        assert tunit_fires(G, thr, {0: 0, 1: 1}) is False
        assert tunit_fires(G, thr, {0: 0, 1: 0}) is False

    def test_fusion_or_tau1(self):
        G = hand_and_out0_cm()
        thr = "thr-0-var-2-out-0"
        fus = "fus-0-thr-0-var-2-out-0"
        assert fusion_active(G, fus, {0: 0, 1: 1}) is True
        assert fusion_active(G, fus, {0: 1, 1: 0}) is True
        assert fusion_active(G, fus, {0: 0, 1: 0}) is True
        assert fusion_active(G, fus, {0: 1, 1: 1}) is False
        assert tunit_fires(G, thr, {0: 0, 1: 1}) is True
        assert tunit_fires(G, thr, {0: 1, 1: 1}) is False

    def test_tau_zero_always_fires(self):
        G = hand_tau0_cm()
        thr = "thr-0-var-3-out-1"
        assert tunit_score(G, thr, {0: 0}) == 0
        assert tunit_fires(G, thr, {0: 0}) is True
        assert tunit_fires(G, thr, {0: 1}) is True

    def test_non_threshold_raises(self):
        G = hand_and_out1_cm()
        with pytest.raises(ValueError):
            tunit_fires(G, "var-0-out-1", {0: 1, 1: 1})


# ---------------------------------------------------------------------------
# LUT reconstruction
# ---------------------------------------------------------------------------


class TestLutFromCM:
    def test_literal_only_half_incomplete(self):
        G = hand_and_out1_cm()
        result = lut_from_canalizing_map(G, input_order=[0, 1])
        assert result.outputs == [None, None, None, 1]
        assert result.is_incomplete is True
        assert result.has_conflict is False
        assert set(result.uncovered_states) == {"00", "01", "10"}
        assert result.conflict_states == []

    def test_fusion_or_half_incomplete(self):
        G = hand_and_out0_cm()
        result = lut_from_canalizing_map(G, input_order=[0, 1])
        assert result.outputs == [0, 0, 0, None]
        assert result.is_incomplete is True
        assert result.has_conflict is False
        assert result.uncovered_states == ["11"]

    def test_full_and_complete(self):
        G = hand_and_cm()
        result = lut_from_canalizing_map(G, input_order=[0, 1])
        assert result.outputs == [0, 0, 0, 1]
        assert result.is_incomplete is False
        assert result.has_conflict is False
        assert result.uncovered_states == []
        assert result.conflict_states == []

    def test_conflict_on_shared_state(self):
        G = hand_conflict_cm()
        result = lut_from_canalizing_map(G, input_order=[0, 1])
        # 01: A=0 B=1 → neither fires (thr0 needs A=1)
        # 10: A=1 B=0 → thr0 fires → 0
        # 11: both fire → conflict
        # 00: neither
        assert result.has_conflict is True
        assert "11" in result.conflict_states
        assert claimed_outputs(G, {0: 1, 1: 1}) == {0, 1}
        assert result.outputs[3] is None  # index of 11
        assert result.outputs[2] == 0  # 10
        assert result.is_incomplete is True  # 00 and 01 uncovered

    def test_tau0_covers_all(self):
        G = hand_tau0_cm()
        result = lut_from_canalizing_map(G, input_order=[0])
        assert result.outputs == [1, 1]
        assert result.is_incomplete is False
        assert result.has_conflict is False

    def test_discover_input_order(self):
        G = hand_and_cm()
        result = lut_from_canalizing_map(G)  # no explicit order
        assert result.input_names == [0, 1]
        assert result.outputs == [0, 0, 0, 1]

    def test_lut_string(self):
        G = hand_and_out1_cm()
        result = lut_from_canalizing_map(G, input_order=[0, 1])
        assert result.lut_string() == "???1"

    def test_nested_or_complete(self):
        G = hand_or_nested_cm()
        result = lut_from_canalizing_map(G, input_order=[0, 1])
        assert result.outputs == [0, 1, 1, 1]
        assert result.is_incomplete is False
        assert result.has_conflict is False

    def test_ignores_edge_types(self):
        """Same AND-out1 wiring but with nonsense / missing edge types."""
        G = nx.DiGraph(name="AND-no-edge-types")
        out1 = make_output_node(G, node_id=2, value=1, label="AND")
        thr = make_threshold(G, tid=0, node_id=2, out=1, tau=2)
        a1 = make_input_node(G, var_id=0, value=1, label="A")
        b1 = make_input_node(G, var_id=1, value=1, label="B")
        G.add_edge(a1, thr, type="whatever")
        G.add_edge(b1, thr)  # no type
        G.add_edge(thr, out1, type="2sym")
        result = lut_from_canalizing_map(G, input_order=[0, 1])
        assert result.outputs == [None, None, None, 1]
        assert node_active(G, thr, {0: 1, 1: 1}) is True
        assert node_active(G, thr, {0: 1, 1: 0}) is False


class TestThresholdRole:
    def test_default_role_is_schema(self):
        G = nx.DiGraph()
        thr = make_threshold(G, tid=0, node_id=1, out=1, tau=1)
        assert G.nodes[thr]["role"] == ROLE_SCHEMA

    def test_roles_stored(self):
        G = nx.DiGraph()
        t0 = make_threshold(G, tid=0, node_id=1, out=1, tau=1, role=ROLE_SCHEMA)
        t1 = make_threshold(
            G, tid=1, node_id=1, out=1, tau=1, role=ROLE_SYMMETRY
        )
        t2 = make_threshold(
            G, tid=2, node_id=1, out=1, tau=1, role=ROLE_POLARITY0
        )
        t3 = make_threshold(
            G, tid=3, node_id=1, out=1, tau=1, role=ROLE_POLARITY1
        )
        assert G.nodes[t0]["role"] == "schema"
        assert G.nodes[t1]["role"] == "symmetry"
        assert G.nodes[t2]["role"] == "polarity0"
        assert G.nodes[t3]["role"] == "polarity1"

    def test_invalid_role_raises(self):
        G = nx.DiGraph()
        with pytest.raises(ValueError, match="Unknown threshold role"):
            make_threshold(G, tid=0, node_id=1, out=1, tau=1, role="fibre")
