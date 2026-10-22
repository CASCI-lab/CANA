# -*- coding: utf-8 -*-
"""
Tests for schemata → canalizing map construction and CM → LUT roundtrips.

Covers the legacy and canonical forward builders
(``BooleanNode.canalizing_map`` / ``build_canalizing_map_*``)
and ``BooleanNetwork.dynamics_canalization_map`` builder forwarding.
"""
from __future__ import annotations

import networkx as nx
import pytest

from cana.boolean_network import BooleanNetwork
from cana.boolean_node import BooleanNode
from cana.canalization.canalizing_map import (
    build_canalizing_map,
    build_canalizing_map_canonical,
    build_canalizing_map_legacy,
    simplify_canalizing_map,
)
from cana.canalization.cm_lut import (
    ROLE_POLARITY0,
    ROLE_POLARITY1,
    ROLE_SCHEMA,
    ROLE_SYMMETRY,
    lut_from_canalizing_map,
)


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


def _assert_exact_roundtrip(node: BooleanNode, builder: str = "legacy") -> None:
    cm = node.canalizing_map(builder=builder)
    result = lut_from_canalizing_map(cm, input_order=node.inputs)
    assert not result.has_conflict
    assert not result.is_incomplete
    assert result.outputs == [int(x) for x in node.outputs]


def _threshold_roles(cm: nx.DiGraph):
    return {
        d.get("role")
        for _, d in cm.nodes(data=True)
        if d.get("type") == "threshold"
    }


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


# ---------------------------------------------------------------------------
# Canonical builder
# ---------------------------------------------------------------------------

F_BNET = (
    "F*= X and not Y and ((not A and B and C) or (A and not B and C) "
    "or (A and B and not C))\n"
    "X*=X\n"
    "Y*=Y\n"
    "A*=A\n"
    "B*=B\n"
    "C*=C\n"
)
F_LUT = "00000000000000000001011000000000"


class TestCanonicalParity:
    """BooleanNode wrapper matches build_canalizing_map_canonical."""

    @pytest.mark.parametrize(
        "outputs,name,inputs,nid",
        [
            ([0, 0, 0, 1], "AND", [0, 1], 2),
            ([0, 1, 1, 1], "OR", [10, 11], 5),
        ],
    )
    def test_node_matches_canonical_builder(self, outputs, name, inputs, nid):
        node = BooleanNode.from_output_list(
            outputs, name=name, inputs=inputs, id=nid
        )
        node._check_compute_canalization_variables(two_symbols=True)
        input_labels = {i: str(i) for i in node.inputs}

        via_node = node.canalizing_map(builder="canonical")
        via_fn = build_canalizing_map_canonical(
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
            builder="canonical",
        )
        assert _graphs_equal(via_node, via_fn)
        assert _graphs_equal(via_node, via_dispatch)


class TestCanonicalRoundtrip:
    @pytest.mark.parametrize(
        "outputs,name,inputs,nid",
        [
            ([0, 0, 0, 1], "AND", [0, 1], 2),
            ([0, 1, 1, 1], "OR", [10, 11], 5),
        ],
    )
    def test_and_or(self, outputs, name, inputs, nid):
        node = BooleanNode.from_output_list(
            outputs, name=name, inputs=inputs, id=nid
        )
        _assert_exact_roundtrip(node, builder="canonical")

    @pytest.mark.parametrize("lut", K3_REGRESSION_LUTS)
    def test_k3_regression(self, lut):
        outputs = [int(b) for b in lut]
        node = BooleanNode.from_output_list(
            outputs, name="k3_{}".format(lut), inputs=[0, 1, 2], id=0
        )
        _assert_exact_roundtrip(node, builder="canonical")

    def test_ctnnb1(self):
        N = BooleanNetwork.from_string_boolean(CTNNB1_BNET)
        ct = next(n for n in N.nodes if n.name == "CTNNB1")
        _assert_exact_roundtrip(ct, builder="canonical")

    def test_f_xy_abc(self):
        N = BooleanNetwork.from_string_boolean(F_BNET)
        fn = next(n for n in N.nodes if n.name == "F")
        assert "".join(str(x) for x in fn.outputs) == F_LUT
        _assert_exact_roundtrip(fn, builder="canonical")


class TestCanonicalStructure:
    """No fusion; nested roles and schema tau match the hand examples."""

    def test_ctnnb1_out1_structure(self):
        N = BooleanNetwork.from_string_boolean(CTNNB1_BNET)
        ct = next(n for n in N.nodes if n.name == "CTNNB1")
        cm = ct.canalizing_map(builder="canonical")

        assert not any(d.get("type") == "fusion" for _, d in cm.nodes(data=True))
        roles = _threshold_roles(cm)
        assert ROLE_SCHEMA in roles
        assert ROLE_SYMMETRY in roles
        assert ROLE_POLARITY0 in roles
        assert ROLE_POLARITY1 in roles

        # out=1 schema: 1 fixed (SRY) + 2 permutation groups → tau=3
        out1 = "var-{nid:d}-out-1".format(nid=ct.id)
        schema_preds = [
            u
            for u, v, d in cm.in_edges(out1, data=True)
            if cm.nodes[u].get("role") == ROLE_SCHEMA
        ]
        assert len(schema_preds) == 1
        assert cm.nodes[schema_preds[0]]["tau"] == 3

        symmetries = [
            u
            for u, v in cm.in_edges(schema_preds[0])
            if cm.nodes[u].get("role") == ROLE_SYMMETRY
        ]
        assert len(symmetries) == 2
        for sym in symmetries:
            assert cm.nodes[sym]["tau"] == 1
            pols = list(cm.predecessors(sym))
            assert len(pols) == 1
            assert cm.nodes[pols[0]]["role"] in (ROLE_POLARITY0, ROLE_POLARITY1)
            assert cm.nodes[pols[0]]["tau"] == 1

    def test_f_out1_structure(self):
        N = BooleanNetwork.from_string_boolean(F_BNET)
        fn = next(n for n in N.nodes if n.name == "F")
        cm = fn.canalizing_map(builder="canonical")

        assert not any(d.get("type") == "fusion" for _, d in cm.nodes(data=True))
        roles = _threshold_roles(cm)
        assert ROLE_SCHEMA in roles
        assert ROLE_SYMMETRY in roles
        assert ROLE_POLARITY0 in roles
        assert ROLE_POLARITY1 in roles

        # out=1 schema: 2 fixed (X,Y) + 1 permutation group → tau=3
        out1 = "var-{nid:d}-out-1".format(nid=fn.id)
        schema_preds = [
            u
            for u, v in cm.in_edges(out1)
            if cm.nodes[u].get("role") == ROLE_SCHEMA
        ]
        assert len(schema_preds) == 1
        assert cm.nodes[schema_preds[0]]["tau"] == 3

        symmetries = [
            u
            for u in cm.predecessors(schema_preds[0])
            if cm.nodes[u].get("role") == ROLE_SYMMETRY
        ]
        assert len(symmetries) == 1
        assert cm.nodes[symmetries[0]]["tau"] == 2

        pol_by_role = {
            cm.nodes[u]["role"]: cm.nodes[u]["tau"]
            for u in cm.predecessors(symmetries[0])
            if cm.nodes[u].get("type") == "threshold"
        }
        assert pol_by_role[ROLE_POLARITY0] == 1
        assert pol_by_role[ROLE_POLARITY1] == 2


def _assert_exact_roundtrip_simplified(
    node: BooleanNode, builder: str = "legacy"
) -> None:
    cm = simplify_canalizing_map(node.canalizing_map(builder=builder))
    result = lut_from_canalizing_map(cm, input_order=node.inputs)
    assert not result.has_conflict
    assert not result.is_incomplete
    assert result.outputs == [int(x) for x in node.outputs]


def _graphs_equal_ignore_name(a: nx.DiGraph, b: nx.DiGraph) -> bool:
    """Like ``_graphs_equal`` but ignore DiGraph ``.name`` / graph attrs."""
    if set(a.nodes) != set(b.nodes) or set(a.edges) != set(b.edges):
        return False
    for n in a.nodes:
        if dict(a.nodes[n]) != dict(b.nodes[n]):
            return False
    for u, v in a.edges:
        if dict(a.edges[u, v]) != dict(b.edges[u, v]):
            return False
    return True


TINY_DCM_BNET = "\n".join(
    [
        "A *= B",
        "B *= A and C",
        "C *= A or B",
    ]
)


class TestDynamicsCanalizationMap:
    """Network DCM compose forwards builder; isolate cleanup is orthogonal."""

    @pytest.mark.parametrize("builder", ["legacy", "canonical"])
    def test_matches_manual_compose(self, builder):
        N = BooleanNetwork.from_string_boolean(TINY_DCM_BNET)
        via_net = N.dynamics_canalization_map(
            simplify=False, builder=builder, remove_isolates=True
        )
        cms = [
            n.canalizing_map(builder=builder)
            for n in N.nodes
            if N.keep_constants or not n.constant
        ]
        manual = nx.compose_all(cms)
        manual.remove_nodes_from(list(nx.isolates(manual)))
        assert _graphs_equal_ignore_name(via_net, manual)

    def test_legacy_default_smoke(self):
        N = BooleanNetwork.from_string_boolean(TINY_DCM_BNET)
        dcm = N.dynamics_canalization_map(simplify=True)
        assert dcm.number_of_nodes() > 0
        assert dcm.name.startswith("DCM:")

    def test_remove_isolates_false_matches_raw_compose(self):
        N = BooleanNetwork.from_string_boolean(TINY_DCM_BNET)
        via_net = N.dynamics_canalization_map(
            simplify=False, builder="canonical", remove_isolates=False
        )
        cms = [
            n.canalizing_map(builder="canonical")
            for n in N.nodes
            if N.keep_constants or not n.constant
        ]
        manual = nx.compose_all(cms)
        assert _graphs_equal_ignore_name(via_net, manual)

    def test_remove_isolates_false_keeps_injected_isolate(self):
        """Else-branch strip is skipped when ``remove_isolates=False``."""
        N = BooleanNetwork.from_string_boolean(TINY_DCM_BNET)
        orphan = "orphan_isolate_test"
        orig = N.nodes[0].canalizing_map

        def with_orphan(output=None, builder="legacy"):
            g = orig(output=output, builder=builder)
            g.add_node(orphan, type="variable", mode="input", value=0)
            return g

        N.nodes[0].canalizing_map = with_orphan
        kept = N.dynamics_canalization_map(
            simplify=False, builder="legacy", remove_isolates=False
        )
        stripped = N.dynamics_canalization_map(
            simplify=False, builder="legacy", remove_isolates=True
        )
        assert orphan in kept
        assert orphan not in stripped

    def test_remove_isolates_forwarded_to_simplify(self, monkeypatch):
        seen = {}

        def fake_simplify(cm, *, copy=True, remove_isolates=True):
            seen["remove_isolates"] = remove_isolates
            return cm

        monkeypatch.setattr(
            "cana.boolean_network.simplify_canalizing_map", fake_simplify
        )
        N = BooleanNetwork.from_string_boolean(TINY_DCM_BNET)
        N.dynamics_canalization_map(simplify=True, remove_isolates=False)
        assert seen["remove_isolates"] is False


class TestSimplifyCanalizingMap:
    """Safe DCM/CM simplify: keep tau=0; merge buffers; bypass schema tau=1 OR."""

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

    def test_xor_canonical_roundtrip(self):
        # Previously broken by stock simplify (multi-pred tau=1 polarity collapse).
        node = BooleanNode.from_output_list(
            [0, 1, 1, 0], name="xor", inputs=[0, 1], id=6
        )
        _assert_exact_roundtrip_simplified(node, builder="canonical")
        _assert_exact_roundtrip_simplified(node, builder="legacy")

    def test_reduces_sole_pred_tau1_buffer(self):
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
        # Surviving node keeps its own role (schema claimer stays schema).
        assert out.nodes["B"]["role"] == ROLE_SCHEMA
        assert set(out.predecessors("B")) == {"p0", "p1"}

    def test_does_not_collapse_multipred_tau1(self):
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

    def test_bypasses_variable_into_schema_tau1(self):
        G = nx.DiGraph()
        G.add_node("schema", type="threshold", tau=1, label="1", role=ROLE_SCHEMA)
        G.add_node("p0", type="variable", mode="input", value=1)
        G.add_node("out0", type="variable", mode="output", value=0)
        G.add_edge("p0", "schema")
        G.add_edge("schema", "out0", type="out")
        out = simplify_canalizing_map(G)
        assert "schema" not in out
        assert set(out.predecessors("out0")) == {"p0"}
        assert out.edges["p0", "out0"].get("type") == "out"

    def test_bypasses_schema_tau1_when_claim_mode_is_input(self):
        # DCM compose can leave shared var-* claim targets as mode=input.
        G = nx.DiGraph()
        G.add_node("schema", type="threshold", tau=1, label="1", role=ROLE_SCHEMA)
        G.add_node("p0", type="variable", mode="input", value=0)
        G.add_node("claim", type="variable", mode="input", value=0)
        G.add_edge("p0", "schema")
        G.add_edge("schema", "claim", type="out")
        out = simplify_canalizing_map(G)
        assert "schema" not in out
        assert set(out.predecessors("claim")) == {"p0"}
        assert out.edges["p0", "claim"].get("type") == "out"

    def test_bypasses_multipred_schema_tau1(self):
        G = nx.DiGraph()
        G.add_node("schema", type="threshold", tau=1, label="1", role=ROLE_SCHEMA)
        G.add_node("p0", type="variable", mode="input", value=0)
        G.add_node("p1", type="variable", mode="input", value=1)
        G.add_node("out1", type="variable", mode="output", value=1)
        G.add_edge("p0", "schema")
        G.add_edge("p1", "schema")
        G.add_edge("schema", "out1", type="out")
        out = simplify_canalizing_map(G)
        assert "schema" not in out
        assert set(out.predecessors("out1")) == {"p0", "p1"}

    def test_bypasses_fusion_into_schema_tau1(self):
        # Schema OR-bypass then fusion OR→variable; both are compatible.
        G = nx.DiGraph()
        G.add_node("schema", type="threshold", tau=1, label="1", role=ROLE_SCHEMA)
        G.add_node("fus", type="fusion")
        G.add_node("p0", type="variable", mode="input", value=0)
        G.add_node("p1", type="variable", mode="input", value=0)
        G.add_node("out0", type="variable", mode="output", value=0)
        G.add_edge("p0", "fus")
        G.add_edge("p1", "fus")
        G.add_edge("fus", "schema")
        G.add_edge("schema", "out0", type="out")
        out = simplify_canalizing_map(G)
        assert "schema" not in out
        assert "fus" not in out
        assert set(out.predecessors("out0")) == {"p0", "p1"}

    def test_does_not_bypass_schema_tau2(self):
        G = nx.DiGraph()
        G.add_node("schema", type="threshold", tau=2, label="2", role=ROLE_SCHEMA)
        G.add_node("p0", type="variable", mode="input", value=0)
        G.add_node("out0", type="variable", mode="output", value=0)
        G.add_edge("p0", "schema")
        G.add_edge("schema", "out0", type="out")
        out = simplify_canalizing_map(G)
        assert out.has_node("schema")
        assert set(out.predecessors("out0")) == {"schema"}

    def test_bypasses_nonschema_tau1_or_into_variable(self):
        # Compatible OR → variable reduction; schema role is not required.
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
