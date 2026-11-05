# -*- coding: utf-8 -*-
"""Tests for :mod:`cana.graphs.activity_graph`."""
import pytest

from cana.boolean_network import BooleanNetwork
from cana.function_properties import SIGN_NEGATIVE
from cana.graphs.activity_graph import build_activity_graph


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

IRRELEVANT_NET = "\n".join(
    [
        "A *= A",
        "B *= A or (not A)",
        "C *= B",
    ]
)


def _and_net():
    return BooleanNetwork.from_string_booleannet(AND_NET, keep_constants=True)


def test_activity_graph_weights_match_activities():
    network = _and_net()
    G = build_activity_graph(network.nodes, name=network.name)

    assert G.graph["threshold"] is None
    assert G.graph["signs_computed"] is False
    assert G.name.startswith("Activity Graph:")
    assert "(Threshold: None)" in G.name

    for target, node in enumerate(network.nodes):
        for source, a_i in zip(node.inputs, node.activities()):
            assert G[source][target]["weight"] == a_i
            assert G[source][target]["sign"] is None

    # AND activities are the classic [0.5, 0.5]
    assert G[0][2]["weight"] == 0.5
    assert G[1][2]["weight"] == 0.5


def test_activity_graph_keeps_zero_weight_edges():
    network = BooleanNetwork.from_string_booleannet(
        IRRELEVANT_NET, keep_constants=True
    )
    G = build_activity_graph(network.nodes)
    assert G.has_edge(0, 1)
    assert G[0][1]["weight"] == 0.0

    Gt = build_activity_graph(network.nodes, threshold=0.0, base_ag=G)
    assert not Gt.has_edge(0, 1)
    assert Gt.graph["threshold"] == 0.0
    assert "Threshold: 0.00" in Gt.name


def test_activity_graph_threshold_strict_gt():
    network = _and_net()
    base = build_activity_graph(network.nodes)
    Gt = build_activity_graph(network.nodes, threshold=0.5, base_ag=base)
    assert not Gt.has_edge(0, 2)
    assert not Gt.has_edge(1, 2)
    assert Gt.has_edge(0, 0)
    assert set(Gt.edges()).issubset(set(base.edges()))


def test_activity_graph_signed_does_not_mutate_base():
    network = BooleanNetwork.from_string_booleannet(
        SIGNED_ABC, keep_constants=True
    )
    base = build_activity_graph(network.nodes)
    signed = build_activity_graph(network.nodes, signed=True, base_ag=base)
    assert signed is not base
    assert signed.graph["signs_computed"] is True
    assert signed[0][1]["sign"] == SIGN_NEGATIVE
    assert base.graph["signs_computed"] is False
    assert all(d["sign"] is None for _, _, d in base.edges(data=True))


def test_activity_graph_base_ag_identity_when_unsigned_unthresholded():
    network = _and_net()
    base = build_activity_graph(network.nodes)
    reused = build_activity_graph(network.nodes, base_ag=base)
    assert reused is base


def test_activity_graph_threshold_compat_errors_and_higher_ok():
    network = _and_net()
    base = build_activity_graph(network.nodes)
    mid = build_activity_graph(network.nodes, threshold=0.1, base_ag=base)

    with pytest.raises(ValueError, match="unthresholded base_ag"):
        build_activity_graph(network.nodes, threshold=None, base_ag=mid)
    with pytest.raises(ValueError, match="unthresholded base_ag"):
        build_activity_graph(network.nodes, threshold=0.05, base_ag=mid)

    higher = build_activity_graph(network.nodes, threshold=0.5, base_ag=mid)
    assert higher.graph["threshold"] == 0.5
    assert set(higher.edges()).issubset(set(mid.edges()))
