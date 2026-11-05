from cana.boolean_network import BooleanNetwork
import networkx as nx
import pytest
from cana.datasets.bio import THALIANA, LEUKEMIA
from cana.function_properties import (
    SIGN_BINATE,
    SIGN_NEGATIVE,
    SIGN_POSITIVE,
)
from cana.graphs.activity_graph import build_activity_graph

# THALIANA loads schemata-only nodes from cnet; canalization soft-warns when
# one_symbol_is_prime_implicant / two_symbol_is_canonical are unset.
# Not under test here.
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*one_symbol_is_prime_implicant.*:UserWarning",
    "ignore:.*two_symbol_is_canonical.*:UserWarning",
)


def test_EG_weight_THALIANA():
    """Test that effective graph in-degree edge weights are computed correctly."""
    network = THALIANA()
    network.effective_graph()

    true = []
    for i, node in enumerate(network.nodes):
        # get sum from nx object
        edgews = {edge: network._eg.edges[edge]["weight"] for edge in network._eg.edges if edge[1]==i}
        true.append(sum(edgews.values()))
    assert network.effective_indegrees() == sorted(true, reverse=True)


def test_named_expression_to_bit_lut():
    """Test named_expression_to_bit_lut produces correct truth tables."""
    from cana.function_outputs.expression import named_expression_to_bit_lut

    assert named_expression_to_bit_lut('A', ['A']).to_list() == [0, 1]
    assert named_expression_to_bit_lut('A and B', ['A', 'B']).to_list() == [0, 0, 0, 1]
    assert named_expression_to_bit_lut('A or B', ['A', 'B']).to_list() == [0, 1, 1, 1]
    assert named_expression_to_bit_lut(
        '(A or B) and not C', ['A', 'B', 'C']
    ).to_list() == [0, 0, 1, 0, 1, 0, 1, 0]


def test_from_string_booleannet():
    """Test that BooleanNetwork.from_string_booleannet parses update rules."""
    rules = "\n".join([
        "A *= B",
        "B *= A and C",
        "C *= A or B",
    ])
    network = BooleanNetwork.from_string_booleannet(rules, keep_constants=True)
    assert network.Nnodes == 3
    # A = copy of B: output transitions [0, 1]
    assert list(network.nodes[0].outputs) == [0, 1]
    # B = A AND C: output transitions [0, 0, 0, 1]
    assert list(network.nodes[1].outputs) == [0, 0, 0, 1]
    # C = A OR B: output transitions [0, 1, 1, 1]
    assert list(network.nodes[2].outputs) == [0, 1, 1, 1]


def test_load_booleannet_LEUKEMIA():
    """Test loading a real model in booleannet update-rule format."""
    network = LEUKEMIA()
    assert network.Nnodes == 60
    # Spot-check first rule: CTLA4* = TCR (identity)
    assert network.nodes[0].name == 'CTLA4'
    assert list(network.nodes[0].outputs) == [0, 1]


def test_interaction_graph_signs():
    """Signed interaction graph uses frozenset signs; unsigned leaves sign=None."""
    rules = "\n".join(
        [
            "A *= A",
            "B *= not A",
            "C *= (A and not B) or (not A and B)",
        ]
    )
    network = BooleanNetwork.from_string_booleannet(rules, keep_constants=True)

    ig = network.interaction_graph(signed=True)
    assert ig.graph["signs_computed"] is True
    assert ig.name.startswith("Interaction Graph:")
    assert ig[0][0]["weight"] == 1.0
    assert ig[0][0]["sign"] == SIGN_POSITIVE
    assert ig[0][1]["sign"] == SIGN_NEGATIVE
    assert ig[0][2]["sign"] == SIGN_BINATE
    assert ig[1][2]["sign"] == SIGN_BINATE

    ig_u = network.interaction_graph(signed=False)
    assert ig_u.graph["signs_computed"] is False
    assert all(d["sign"] is None for _, _, d in ig_u.edges(data=True))

    eg = network.effective_graph(signed=False)
    assert eg.graph["signs_computed"] is False
    assert all(d["sign"] is None for _, _, d in eg.edges(data=True))

    eg_s = network.effective_graph(signed=True)
    assert eg_s.graph["signs_computed"] is True
    assert eg_s[0][1]["sign"] == SIGN_NEGATIVE
    # Cache stays unsigned.
    assert network._eg.graph["signs_computed"] is False

    ag = network.activity_graph(signed=False)
    assert ag.graph["signs_computed"] is False
    assert ag.graph["threshold"] is None
    assert all(d["sign"] is None for _, _, d in ag.edges(data=True))
    ag_s = network.activity_graph(signed=True)
    assert ag_s.graph["signs_computed"] is True
    assert ag_s[0][1]["sign"] == SIGN_NEGATIVE
    # Cache stays unsigned / unthresholded.
    assert network._ag.graph["signs_computed"] is False
    assert network._ag.graph["threshold"] is None

    ag_t = network.activity_graph(threshold=0.5)
    assert ag_t.graph["threshold"] == 0.5
    assert set(ag_t.edges()).issubset(set(network._ag.edges()))
    with pytest.raises(ValueError, match="unthresholded base_ag"):
        build_activity_graph(
            network.nodes,
            threshold=0.1,
            base_ag=ag_t,
        )


def test_structural_graph_deprecated_alias():
    rules = "A *= A\nB *= A"
    network = BooleanNetwork.from_string_booleannet(rules, keep_constants=True)
    with pytest.warns(DeprecationWarning, match="structural_graph"):
        sg = network.structural_graph()
    assert sg.graph["signs_computed"] is False
    with pytest.warns(DeprecationWarning, match="signed_interaction_graph"):
        sig = network.signed_interaction_graph()
    assert sig.graph["signs_computed"] is True
    assert sig[0][1]["sign"] == SIGN_POSITIVE
