# -*- coding: utf-8 -*-
"""Tests for threshold-unit reduction and CM/DCM simplify."""
from __future__ import annotations

import networkx as nx
import pytest

from cana.boolean_node import BooleanNode
from cana.canalization.cm_lut import (
    ROLE_POLARITY0,
    ROLE_SCHEMA,
    ROLE_SYMMETRY,
    lut_from_canalizing_map,
)
from cana.canalization.cm_simplify import (
    CLASS_AND,
    CLASS_LINEAR,
    CLASS_OR,
    CLASS_OTHER,
    _distance_to_out,
    _edge_delta_after_reduce,
    _simplify_one_step,
    classify_unit,
    reduce_threshold_unit,
    reduction_compatible,
    reduction_skip_reason,
    simplify_canalizing_map,
    unit_tau,
)


def _assert_exact_roundtrip_simplified(
    node: BooleanNode, builder: str = "legacy"
) -> None:
    cm = simplify_canalizing_map(node.canalizing_map(builder=builder))
    result = lut_from_canalizing_map(cm, input_order=node.inputs)
    assert not result.has_conflict
    assert not result.is_incomplete
    expected = [int(x) for x in node.outputs]
    assert result.outputs == expected


# ---------------------------------------------------------------------------
# Classification / compatibility
# ---------------------------------------------------------------------------


class TestClassifyAndCompat:
    def test_classify_linear_and_or_other(self):
        G = nx.DiGraph()
        G.add_node("lin", type="threshold", tau=1)
        G.add_node("p0", type="variable", mode="input", value=0)
        G.add_edge("p0", "lin")
        assert classify_unit(G, "lin") == CLASS_LINEAR

        G.add_node("and", type="threshold", tau=2)
        G.add_node("p1", type="variable", mode="input", value=1)
        G.add_edge("p0", "and")
        G.add_edge("p1", "and")
        assert classify_unit(G, "and") == CLASS_AND

        G.add_node("orr", type="threshold", tau=1)
        G.add_edge("p0", "orr")
        G.add_edge("p1", "orr")
        assert classify_unit(G, "orr") == CLASS_OR

        G.add_node("oth", type="threshold", tau=2)
        G.add_node("p2", type="variable", mode="input", value=0)
        G.add_edge("p0", "oth")
        G.add_edge("p1", "oth")
        G.add_edge("p2", "oth")
        assert classify_unit(G, "oth") == CLASS_OTHER

        assert classify_unit(G, "p0") == CLASS_OR

    def test_fusion_tau_and_classify(self):
        G = nx.DiGraph()
        G.add_node("fus", type="fusion")
        G.add_node("p0", type="variable", mode="input", value=0)
        G.add_node("p1", type="variable", mode="input", value=1)
        G.add_edge("p0", "fus")
        G.add_edge("p1", "fus")
        assert unit_tau(G, "fus") == 1
        assert classify_unit(G, "fus") == CLASS_OR

    def test_and_incompatible_with_variable(self):
        G = nx.DiGraph()
        G.add_node("and", type="threshold", tau=2)
        G.add_node("p0", type="variable", mode="input", value=0)
        G.add_node("p1", type="variable", mode="input", value=1)
        G.add_node("out0", type="variable", mode="output", value=0)
        G.add_edge("p0", "and")
        G.add_edge("p1", "and")
        G.add_edge("and", "out0", type="out")
        assert reduction_skip_reason(G, "and") == "incompatible"
        assert not reduction_compatible(G, "and")

    def test_or_compatible_with_variable(self):
        G = nx.DiGraph()
        G.add_node("orr", type="threshold", tau=1)
        G.add_node("p0", type="variable", mode="input", value=0)
        G.add_node("p1", type="variable", mode="input", value=1)
        G.add_node("out0", type="variable", mode="output", value=0)
        G.add_edge("p0", "orr")
        G.add_edge("p1", "orr")
        G.add_edge("orr", "out0", type="out")
        assert reduction_compatible(G, "orr")

    def test_linear_compatible_with_other(self):
        G = nx.DiGraph()
        G.add_node("lin", type="threshold", tau=1)
        G.add_node("oth", type="threshold", tau=2)
        G.add_node("p0", type="variable", mode="input", value=0)
        G.add_node("p1", type="variable", mode="input", value=1)
        G.add_node("p2", type="variable", mode="input", value=0)
        G.add_edge("p0", "lin")
        G.add_edge("p1", "oth")
        G.add_edge("p2", "oth")
        G.add_edge("lin", "oth")
        # oth has 3 preds and tau=2 → OTHER; lin → OTHER is allowed.
        assert classify_unit(G, "oth") == CLASS_OTHER
        assert reduction_compatible(G, "lin")

    def test_tau0_skipped(self):
        G = nx.DiGraph()
        G.add_node("c", type="threshold", tau=0)
        G.add_node("out0", type="variable", mode="output", value=0)
        G.add_edge("c", "out0", type="out")
        assert reduction_skip_reason(G, "c") == "tau0"

    def test_other_into_linear_compatible(self):
        G = nx.DiGraph()
        G.add_node("oth", type="threshold", tau=2)
        G.add_node("lin", type="threshold", tau=1)
        G.add_node("p0", type="variable", mode="input", value=0)
        G.add_node("p1", type="variable", mode="input", value=1)
        G.add_node("p2", type="variable", mode="input", value=0)
        G.add_edge("p0", "oth")
        G.add_edge("p1", "oth")
        G.add_edge("p2", "oth")
        G.add_edge("oth", "lin")
        assert classify_unit(G, "oth") == CLASS_OTHER
        assert classify_unit(G, "lin") == CLASS_LINEAR
        assert reduction_compatible(G, "oth")

    def test_other_into_variable_incompatible(self):
        G = nx.DiGraph()
        G.add_node("oth", type="threshold", tau=2)
        G.add_node("p0", type="variable", mode="input", value=0)
        G.add_node("p1", type="variable", mode="input", value=1)
        G.add_node("p2", type="variable", mode="input", value=0)
        G.add_node("out0", type="variable", mode="output", value=0)
        G.add_edge("p0", "oth")
        G.add_edge("p1", "oth")
        G.add_edge("p2", "oth")
        G.add_edge("oth", "out0", type="out")
        assert reduction_skip_reason(G, "oth") == "incompatible"


# ---------------------------------------------------------------------------
# reduce_threshold_unit
# ---------------------------------------------------------------------------


class TestReduceThresholdUnit:
    def test_wires_and_bumps_and(self):
        G = nx.DiGraph()
        G.add_node("T", type="threshold", tau=2, label="2")
        G.add_node("B", type="threshold", tau=1, label="1")
        G.add_node("p0", type="variable", mode="input", value=0)
        G.add_node("p1", type="variable", mode="input", value=1)
        G.add_edge("p0", "T")
        G.add_edge("p1", "T")
        G.add_edge("T", "B")
        out = reduce_threshold_unit(G, "T", copy=True)
        assert "T" not in out
        assert set(out.predecessors("B")) == {"p0", "p1"}
        assert out.nodes["B"]["tau"] == 2  # 1 + (2-1)
        assert out.edges["p0", "B"].get("mode") == "threshold_reduce"

    def test_or_bump_zero_no_tau_change(self):
        G = nx.DiGraph()
        G.add_node("T", type="threshold", tau=1, label="1")
        G.add_node("B", type="threshold", tau=2, label="2")
        G.add_node("p0", type="variable", mode="input", value=0)
        G.add_node("p1", type="variable", mode="input", value=1)
        G.add_edge("p0", "T")
        G.add_edge("p1", "T")
        G.add_edge("T", "B")
        # OR → AND is incompatible; use LINEAR successor instead.
        G.remove_edge("p1", "T")
        # Now T is LINEAR (tau=1, n=1); bump=0.
        assert classify_unit(G, "T") == CLASS_LINEAR
        out = reduce_threshold_unit(G, "T", copy=True)
        assert out.nodes["B"]["tau"] == 2
        assert set(out.predecessors("B")) == {"p0"}

    def test_never_bumps_variable(self):
        G = nx.DiGraph()
        G.add_node("T", type="threshold", tau=1, label="1")
        G.add_node("p0", type="variable", mode="input", value=0)
        G.add_node("p1", type="variable", mode="input", value=1)
        G.add_node("out0", type="variable", mode="output", value=0)
        G.add_edge("p0", "T")
        G.add_edge("p1", "T")
        G.add_edge("T", "out0", type="out")
        out = reduce_threshold_unit(G, "T", copy=True)
        assert "tau" not in out.nodes["out0"]
        assert set(out.predecessors("out0")) == {"p0", "p1"}

    def test_fusion_promoted_on_bump(self):
        G = nx.DiGraph()
        G.add_node("T", type="threshold", tau=2, label="2")
        G.add_node("fus", type="fusion")
        G.add_node("p0", type="variable", mode="input", value=0)
        G.add_node("p1", type="variable", mode="input", value=1)
        G.add_edge("p0", "T")
        G.add_edge("p1", "T")
        G.add_edge("T", "fus")
        # fus has only T as pred → LINEAR; AND→LINEAR is OK.
        assert classify_unit(G, "fus") == CLASS_LINEAR
        assert reduction_compatible(G, "T")
        out = reduce_threshold_unit(G, "T", copy=True)
        assert out.nodes["fus"]["type"] == "threshold"
        assert out.nodes["fus"]["tau"] == 2  # 1 + 1

    def test_other_bumps_linear_successor(self):
        G = nx.DiGraph()
        G.add_node("T", type="threshold", tau=2, label="2")
        G.add_node("B", type="threshold", tau=1, label="1")
        G.add_node("p0", type="variable", mode="input", value=0)
        G.add_node("p1", type="variable", mode="input", value=1)
        G.add_node("p2", type="variable", mode="input", value=0)
        G.add_edge("p0", "T")
        G.add_edge("p1", "T")
        G.add_edge("p2", "T")
        G.add_edge("T", "B")
        assert classify_unit(G, "T") == CLASS_OTHER
        out = reduce_threshold_unit(G, "T", copy=True)
        assert "T" not in out
        assert out.nodes["B"]["tau"] == 2
        assert set(out.predecessors("B")) == {"p0", "p1", "p2"}


# ---------------------------------------------------------------------------
# Ordering / determinism
# ---------------------------------------------------------------------------


class TestSimplifyOrdering:
    def test_farthest_from_out_reduced_first(self):
        # far (OR) → mid (OR) → out0; mid has a second pred so both stay
        # reducible independently; far is farther from out.
        G = nx.DiGraph()
        G.add_node("far", type="threshold", tau=1, label="1")
        G.add_node("mid", type="threshold", tau=1, label="1")
        G.add_node("p0", type="variable", mode="input", value=0)
        G.add_node("p1", type="variable", mode="input", value=1)
        G.add_node("extra", type="variable", mode="input", value=0)
        G.add_node("out0", type="variable", mode="output", value=0)
        G.add_edge("p0", "far")
        G.add_edge("p1", "far")
        G.add_edge("far", "mid")
        G.add_edge("extra", "mid")
        G.add_edge("mid", "out0", type="out")

        assert _distance_to_out(G, "far") == 2
        assert _distance_to_out(G, "mid") == 1
        assert reduction_compatible(G, "far")
        assert reduction_compatible(G, "mid")

        removed = _simplify_one_step(G)
        assert removed == "far"
        assert "far" not in G
        assert G.has_node("mid")

    def test_deterministic_same_result(self):
        node = BooleanNode.from_output_list(
            [0, 1, 1, 0], name="xor", inputs=[0, 1], id=6
        )
        raw = node.canalizing_map(builder="canonical")
        a = simplify_canalizing_map(raw, copy=True)
        b = simplify_canalizing_map(raw, copy=True)
        assert set(a.nodes) == set(b.nodes)
        assert set(a.edges) == set(b.edges)
        for n in a.nodes:
            assert dict(a.nodes[n]) == dict(b.nodes[n])


# ---------------------------------------------------------------------------
# simplify_canalizing_map integration
# ---------------------------------------------------------------------------


class TestSimplifyCanalizingMap:
    def test_constant_tau0_preserved(self):
        node = BooleanNode.from_output_list(
            [0, 0, 0, 0], name="const0", inputs=[0, 1], id=0
        )
        for builder in ("legacy", "canonical"):
            before = node.canalizing_map(builder=builder)
            tau0 = [
                n
                for n, d in before.nodes(data=True)
                if d.get("type") == "threshold" and d.get("tau") == 0
            ]
            assert len(tau0) == 1
            after = simplify_canalizing_map(before)
            assert after.has_node(tau0[0])
            assert after.nodes[tau0[0]]["tau"] == 0
            _assert_exact_roundtrip_simplified(node, builder=builder)

    def test_reduces_other_into_linear_tau1(self):
        # OTHER (tau=2, n=3) → LINEAR schema: wire + bump τ(B) by τ(A)-1.
        G = nx.DiGraph()
        G.add_node("A", type="threshold", tau=2, label="2", role=ROLE_SYMMETRY)
        G.add_node("B", type="threshold", tau=1, label="1", role=ROLE_SCHEMA)
        G.add_node("p0", type="variable", mode="input", value=0)
        G.add_node("p1", type="variable", mode="input", value=1)
        G.add_node("p2", type="variable", mode="input", value=0)
        G.add_edge("p0", "A")
        G.add_edge("p1", "A")
        G.add_edge("p2", "A")
        G.add_edge("A", "B")
        assert classify_unit(G, "A") == CLASS_OTHER
        out = simplify_canalizing_map(G)
        assert "A" not in out
        assert out.has_node("B")
        assert out.nodes["B"]["tau"] == 2
        assert out.nodes["B"]["role"] == ROLE_SCHEMA
        assert set(out.predecessors("B")) == {"p0", "p1", "p2"}

    def test_reduces_and_into_linear_tau1_buffer(self):
        G = nx.DiGraph()
        G.add_node("A", type="threshold", tau=2, label="2", role=ROLE_SYMMETRY)
        G.add_node("B", type="threshold", tau=1, label="1", role=ROLE_SCHEMA)
        G.add_node("p0", type="variable", mode="input", value=0)
        G.add_node("p1", type="variable", mode="input", value=1)
        G.add_edge("p0", "A")
        G.add_edge("p1", "A")
        G.add_edge("A", "B")
        out = simplify_canalizing_map(G)
        assert "A" not in out
        assert out.has_node("B")
        assert out.nodes["B"]["tau"] == 2
        assert out.nodes["B"]["role"] == ROLE_SCHEMA
        assert set(out.predecessors("B")) == {"p0", "p1"}

    def test_does_not_collapse_multipred_tau1_into_wrong_class(self):
        # OR polarity into AND symmetry: incompatible.
        G = nx.DiGraph()
        G.add_node("pol", type="threshold", tau=1, label="1", role=ROLE_POLARITY0)
        G.add_node("sym", type="threshold", tau=2, label="2", role=ROLE_SYMMETRY)
        G.add_node("p0", type="variable", mode="input", value=0)
        G.add_node("p1", type="variable", mode="input", value=1)
        G.add_edge("p0", "pol")
        G.add_edge("p1", "pol")
        G.add_edge("pol", "sym")
        G.add_node("other", type="threshold", tau=1, label="1")
        G.add_edge("other", "sym")
        out = simplify_canalizing_map(G)
        assert out.has_node("pol")
        assert set(out.predecessors("pol")) == {"p0", "p1"}

    def test_bypasses_schema_and_nonschema_or(self):
        G = nx.DiGraph()
        G.add_node("pol", type="threshold", tau=1, label="1", role=ROLE_POLARITY0)
        G.add_node("p0", type="variable", mode="input", value=0)
        G.add_node("p1", type="variable", mode="input", value=1)
        G.add_node("out0", type="variable", mode="output", value=0)
        G.add_edge("p0", "pol")
        G.add_edge("p1", "pol")
        G.add_edge("pol", "out0", type="out")
        out = simplify_canalizing_map(G)
        assert "pol" not in out
        assert set(out.predecessors("out0")) == {"p0", "p1"}

    @pytest.mark.parametrize(
        "outputs",
        [
            [0, 0, 0, 1],  # AND
            [0, 1, 1, 1],  # OR
            [0, 1, 1, 0],  # XOR
            [0, 0, 0, 0],  # const 0
            [1, 1, 1, 1],  # const 1
        ],
    )
    @pytest.mark.parametrize("builder", ["legacy", "canonical"])
    def test_roundtrip_common_k2(self, outputs, builder):
        node = BooleanNode.from_output_list(
            outputs, name="f", inputs=[0, 1], id=1
        )
        _assert_exact_roundtrip_simplified(node, builder=builder)

    def test_xor_canonical_roundtrip(self):
        node = BooleanNode.from_output_list(
            [0, 1, 1, 0], name="xor", inputs=[0, 1], id=6
        )
        _assert_exact_roundtrip_simplified(node, builder="canonical")
        _assert_exact_roundtrip_simplified(node, builder="legacy")

    def test_reexport_from_canalizing_map(self):
        from cana.canalization.canalizing_map import (
            simplify_canalizing_map as reexported,
        )

        assert reexported is simplify_canalizing_map

    def test_skip_if_edges_increase_preserves_node(self):
        # |P|=3, |S|=2 → add 6, remove 5 → delta +1; skip when option on.
        G = nx.DiGraph()
        G.add_node("T", type="threshold", tau=1, label="1")
        G.add_node("B0", type="threshold", tau=1, label="1")
        G.add_node("B1", type="threshold", tau=1, label="1")
        for p in ("p0", "p1", "p2"):
            G.add_node(p, type="variable", mode="input", value=0)
            G.add_edge(p, "T")
        G.add_edge("T", "B0")
        G.add_edge("T", "B1")
        assert classify_unit(G, "T") == CLASS_OR
        assert reduction_compatible(G, "T")
        assert _edge_delta_after_reduce(G, "T") == 1

        kept = simplify_canalizing_map(G, skip_if_edges_increase=True)
        assert kept.has_node("T")
        assert set(kept.predecessors("T")) == {"p0", "p1", "p2"}

        reduced = simplify_canalizing_map(G, skip_if_edges_increase=False)
        assert "T" not in reduced
        assert set(reduced.predecessors("B0")) == {"p0", "p1", "p2"}
        assert set(reduced.predecessors("B1")) == {"p0", "p1", "p2"}

    def test_skip_if_edges_increase_allows_nonincreasing(self):
        # |P|=2, |S|=2 → add 4, remove 4 → delta 0; still reduce.
        G = nx.DiGraph()
        G.add_node("T", type="threshold", tau=1, label="1")
        G.add_node("B0", type="threshold", tau=1, label="1")
        G.add_node("B1", type="threshold", tau=1, label="1")
        G.add_node("p0", type="variable", mode="input", value=0)
        G.add_node("p1", type="variable", mode="input", value=1)
        G.add_edge("p0", "T")
        G.add_edge("p1", "T")
        G.add_edge("T", "B0")
        G.add_edge("T", "B1")
        assert _edge_delta_after_reduce(G, "T") == 0
        out = simplify_canalizing_map(G, skip_if_edges_increase=True)
        assert "T" not in out
        assert set(out.predecessors("B0")) == {"p0", "p1"}
        assert set(out.predecessors("B1")) == {"p0", "p1"}

    def test_edge_delta_accounts_for_existing_cross_edges(self):
        # |P|=2, |S|=2 with one pred→succ already present: add 3, remove 4.
        G = nx.DiGraph()
        G.add_node("T", type="threshold", tau=1, label="1")
        G.add_node("B0", type="threshold", tau=1, label="1")
        G.add_node("B1", type="threshold", tau=1, label="1")
        G.add_node("p0", type="variable", mode="input", value=0)
        G.add_node("p1", type="variable", mode="input", value=1)
        G.add_edge("p0", "T")
        G.add_edge("p1", "T")
        G.add_edge("T", "B0")
        G.add_edge("T", "B1")
        G.add_edge("p0", "B0")
        assert _edge_delta_after_reduce(G, "T") == -1
