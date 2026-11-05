# -*- coding: utf-8 -*-
"""Tests for :mod:`cana.graphs.interaction_graph`."""
import pytest

from cana.boolean_network import BooleanNetwork
from cana.function_properties import (
    SIGN_BINATE,
    SIGN_IRRELEVANT,
    SIGN_NEGATIVE,
    SIGN_POSITIVE,
)
from cana.graphs import (
    build_activity_graph,
    build_conditional_effective_graph,
    build_effective_graph,
    build_interaction_graph,
    fill_wiring_signs,
)
from cana.graphs.interaction_graph import (
    build_interaction_graph as build_ig_direct,
    fill_wiring_signs as fill_signs_direct,
)


SIGNED_ABC = "\n".join(
    [
        "A *= A",
        "B *= not A",
        "C *= (A and not B) or (not A and B)",
    ]
)

CONSTANTS_NET = "\n".join(
    [
        "A *= 1",
        "B *= A",
        "C *= B or A",
    ]
)

IRRELEVANT_NET = "\n".join(
    [
        "A *= A",
        "B *= A or (not A)",
        "C *= B",
    ]
)


def _signed_abc():
    return BooleanNetwork.from_string_booleannet(SIGNED_ABC, keep_constants=True)


def test_graphs_package_exports():
    assert build_interaction_graph is build_ig_direct
    assert fill_wiring_signs is fill_signs_direct
    assert callable(build_effective_graph)
    assert callable(build_conditional_effective_graph)
    assert callable(build_activity_graph)


def test_interaction_graph_topology_and_weights():
    network = _signed_abc()
    G = build_interaction_graph(network.nodes, name=network.name, signed=False)

    assert G.name.startswith("Interaction Graph:")
    assert set(G.nodes()) == {0, 1, 2}
    assert G.nodes[0]["label"] == "A"
    assert G.nodes[1]["label"] == "B"
    assert G.nodes[2]["label"] == "C"

    expected = set()
    for target, node in enumerate(network.nodes):
        for source in node.inputs:
            expected.add((source, target))
    assert set(G.edges()) == expected
    assert all(d["weight"] == 1.0 for _, _, d in G.edges(data=True))


def test_interaction_graph_signed_default():
    network = _signed_abc()
    G = build_interaction_graph(network.nodes, name="abc")

    assert G.graph["signs_computed"] is True
    assert G[0][0]["sign"] == SIGN_POSITIVE
    assert G[0][1]["sign"] == SIGN_NEGATIVE
    assert G[0][2]["sign"] == SIGN_BINATE
    assert G[1][2]["sign"] == SIGN_BINATE


def test_interaction_graph_unsigned():
    network = _signed_abc()
    G = build_interaction_graph(network.nodes, signed=False)

    assert G.graph["signs_computed"] is False
    assert all(d["sign"] is None for _, _, d in G.edges(data=True))


def test_interaction_graph_irrelevant_sign():
    network = BooleanNetwork.from_string_booleannet(
        IRRELEVANT_NET, keep_constants=True
    )
    G = build_interaction_graph(network.nodes)
    assert G[0][1]["sign"] == SIGN_IRRELEVANT


def test_fill_wiring_signs_mutates_in_place():
    network = _signed_abc()
    G = build_interaction_graph(network.nodes, signed=False)
    assert G.graph["signs_computed"] is False

    out = fill_wiring_signs(G, network.nodes)
    assert out is G
    assert G.graph["signs_computed"] is True
    assert G[0][0]["sign"] == SIGN_POSITIVE
    assert G[0][1]["sign"] == SIGN_NEGATIVE
    assert G[0][2]["sign"] == SIGN_BINATE


def test_fill_wiring_signs_skips_missing_targets():
    network = BooleanNetwork.from_string_booleannet(
        CONSTANTS_NET, keep_constants=True
    )
    G = build_interaction_graph(network.nodes, signed=False)
    G.remove_node(0)
    fill_wiring_signs(G, network.nodes)
    # Node 0 removed; remaining edges that still exist get signs from targets.
    assert 0 not in G
    assert G.graph["signs_computed"] is True
    if G.has_edge(1, 2):
        assert G[1][2]["sign"] == SIGN_POSITIVE


def test_interaction_graph_remove_constants():
    network = BooleanNetwork.from_string_booleannet(
        CONSTANTS_NET, keep_constants=True
    )
    constant_ids = list(network.get_constants().keys())
    assert constant_ids == [0]

    G = build_interaction_graph(
        network.nodes,
        remove_constants=True,
        signed=False,
        constant_ids=constant_ids,
    )
    assert 0 not in G
    assert set(G.nodes()) == {1, 2}
    assert not any(u == 0 or v == 0 for u, v in G.edges())
