# -*- coding: utf-8 -*-
"""Tests for :mod:`cana.graphs.effective_graph`."""
import pytest

from cana.boolean_network import BooleanNetwork
from cana.function_properties import SIGN_NEGATIVE, SIGN_POSITIVE
from cana.graphs.effective_graph import (
    build_conditional_effective_graph,
    build_effective_graph,
)


AND_NET = "\n".join(
    [
        "A *= A",
        "B *= B",
        "C *= A and B",
    ]
)

SIGNED_ABC = "\n".join(
    [
        "A *= A",
        "B *= not A",
        "C *= (A and not B) or (not A and B)",
    ]
)

CASCADE = "\n".join(
    [
        "A *= A",
        "B *= A",
        "C *= B and D",
        "D *= D",
    ]
)

IRRELEVANT_NET = "\n".join(
    [
        "A *= A",
        "B *= A or (not A)",
        "C *= B",
    ]
)


def _and_net():
    return BooleanNetwork.from_string_booleannet(AND_NET, keep_constants=True)


def _cascade():
    return BooleanNetwork.from_string_booleannet(CASCADE, keep_constants=True)


def test_effective_graph_weights_match_edge_effectiveness():
    network = _and_net()
    for bound in ("mean", "upper", "lower"):
        G = build_effective_graph(network.nodes, bound=bound)
        assert G.graph["threshold"] is None
        assert G.graph["signs_computed"] is False
        assert G.name.startswith("Effective Graph:")
        for target, node in enumerate(network.nodes):
            e_is = node.edge_effectiveness(bound=bound)
            for source, e_i in zip(node.inputs, e_is):
                assert G[source][target]["weight"] == e_i
                assert G[source][target]["sign"] is None


def test_effective_graph_keeps_zero_weight_edges():
    network = BooleanNetwork.from_string_booleannet(
        IRRELEVANT_NET, keep_constants=True
    )
    G = build_effective_graph(network.nodes)
    assert G.has_edge(0, 1)
    assert G[0][1]["weight"] == 0.0

    Gt = build_effective_graph(network.nodes, threshold=0.0, base_eg=G)
    assert not Gt.has_edge(0, 1)
    assert Gt.graph["threshold"] == 0.0


def test_effective_graph_threshold_strict_gt():
    network = _and_net()
    base = build_effective_graph(network.nodes)
    # AND inputs to C have mean effectiveness 0.625
    Gt = build_effective_graph(network.nodes, threshold=0.625, base_eg=base)
    assert not Gt.has_edge(0, 2)
    assert not Gt.has_edge(1, 2)
    assert Gt.has_edge(0, 0)  # weight 1.0
    assert set(Gt.edges()).issubset(set(base.edges()))


def test_effective_graph_signed_does_not_mutate_base():
    network = BooleanNetwork.from_string_booleannet(
        SIGNED_ABC, keep_constants=True
    )
    base = build_effective_graph(network.nodes)
    signed = build_effective_graph(
        network.nodes, signed=True, base_eg=base
    )
    assert signed is not base
    assert signed.graph["signs_computed"] is True
    assert signed[0][1]["sign"] == SIGN_NEGATIVE
    assert base.graph["signs_computed"] is False
    assert all(d["sign"] is None for _, _, d in base.edges(data=True))


def test_effective_graph_base_eg_identity_when_unsigned_unthresholded():
    network = _and_net()
    base = build_effective_graph(network.nodes)
    reused = build_effective_graph(network.nodes, base_eg=base)
    assert reused is base


def test_effective_graph_threshold_compat_errors_and_higher_ok():
    network = _and_net()
    base = build_effective_graph(network.nodes)
    mid = build_effective_graph(network.nodes, threshold=0.1, base_eg=base)

    with pytest.raises(ValueError, match="unthresholded base_eg"):
        build_effective_graph(network.nodes, threshold=None, base_eg=mid)
    with pytest.raises(ValueError, match="unthresholded base_eg"):
        build_effective_graph(network.nodes, threshold=0.05, base_eg=mid)

    higher = build_effective_graph(network.nodes, threshold=0.5, base_eg=mid)
    assert higher.graph["threshold"] == 0.5
    assert set(higher.edges()).issubset(set(mid.edges()))


def test_conditional_effective_graph_empty_conditioning():
    network = _cascade()
    base = build_effective_graph(network.nodes)
    ceg = build_conditional_effective_graph(
        network.nodes, conditioned_nodes={}, base_eg=base
    )
    assert ceg is not base
    assert all(
        ceg.nodes[n]["conditioned_state"] is None for n in ceg.nodes
    )
    assert all(
        ceg.nodes[n]["conditioned_subgraph"] is False for n in ceg.nodes
    )
    assert "(Threshold: None)" in ceg.name


def test_conditional_effective_graph_cascade():
    network = _cascade()
    orig_inputs = [list(node.inputs) for node in network.nodes]
    orig_outputs = [list(node.outputs) for node in network.nodes]

    ceg = build_conditional_effective_graph(
        network.nodes, conditioned_nodes={0: 0}, name=network.name
    )

    # Original nodes unchanged.
    assert [list(node.inputs) for node in network.nodes] == orig_inputs
    assert [list(node.outputs) for node in network.nodes] == orig_outputs

    # A=0 forces B=0 then C=0; D remains free.
    assert ceg.nodes[0]["conditioned_state"] == 0
    assert ceg.nodes[1]["conditioned_state"] == 0
    assert ceg.nodes[2]["conditioned_state"] == 0
    assert ceg.nodes[3]["conditioned_state"] is None
    assert all(ceg.nodes[n]["conditioned_subgraph"] for n in ceg.nodes)

    # Remaining free input of C is D; constant LUT → effectiveness 0.
    assert ceg.has_edge(3, 2)
    assert ceg[3][2]["weight"] == 0.0
    assert "{0: 0}" in ceg.name


def test_conditional_effective_graph_threshold_and_signed():
    network = _cascade()
    ceg = build_conditional_effective_graph(
        network.nodes,
        conditioned_nodes={0: 0},
        threshold=0.0,
        signed=True,
    )
    assert "(Threshold: 0.00)" in ceg.name
    # Zero-weight D→C removed by <= threshold.
    assert not ceg.has_edge(3, 2)
    assert ceg.graph["signs_computed"] is True
    # Remaining self-loop on D is positive; irrelevant if any zero-LUT input remains signed path.
    assert ceg[3][3]["sign"] == SIGN_POSITIVE

    ceg_u = build_conditional_effective_graph(
        network.nodes, conditioned_nodes={0: 0}, signed=False
    )
    assert ceg_u.graph["signs_computed"] is False
    assert all(d["sign"] is None for _, _, d in ceg_u.edges(data=True))
