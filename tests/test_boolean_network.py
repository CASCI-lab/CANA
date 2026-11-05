from cana.boolean_network import BooleanNetwork
import pytest
from cana.datasets.bio import THALIANA, LEUKEMIA
from cana.function_properties import SIGN_POSITIVE
from cana.graphs.effective_graph import build_effective_graph

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
    # A = copy of B: outputs [0, 1]
    assert list(network.nodes[0].function) == [0, 1]
    # B = A AND C: outputs [0, 0, 0, 1]
    assert list(network.nodes[1].function) == [0, 0, 0, 1]
    # C = A OR B: outputs [0, 1, 1, 1]
    assert list(network.nodes[2].function) == [0, 1, 1, 1]


def test_load_booleannet_LEUKEMIA():
    """Test loading a real model in booleannet update-rule format."""
    network = LEUKEMIA()
    assert network.Nnodes == 60
    # Spot-check first rule: CTLA4* = TCR (identity)
    assert network.nodes[0].name == 'CTLA4'
    assert list(network.nodes[0].function) == [0, 1]


def test_graph_wrappers_cache_unsigned():
    """BN graph methods cache unsigned unthresholded EG/AG; views do not mutate them."""
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

    eg = network.effective_graph(signed=False)
    assert eg.graph["signs_computed"] is False
    eg_s = network.effective_graph(signed=True)
    assert eg_s.graph["signs_computed"] is True
    assert network._eg.graph["signs_computed"] is False

    ag = network.activity_graph(signed=False)
    assert ag.graph["threshold"] is None
    ag_s = network.activity_graph(signed=True)
    assert ag_s.graph["signs_computed"] is True
    ag_t = network.activity_graph(threshold=0.5)
    assert ag_t.graph["threshold"] == 0.5
    assert network._ag.graph["signs_computed"] is False
    assert network._ag.graph["threshold"] is None


def test_effective_graph_bound_uses_first_cached_eg():
    """Later bound= changes are ignored once ``_eg`` is cached (current behavior)."""
    rules = "\n".join(
        [
            "A *= A",
            "B *= B",
            "C *= A and B",
        ]
    )
    network = BooleanNetwork.from_string_booleannet(rules, keep_constants=True)

    mean = network.effective_graph(bound="mean")
    # Same object path via cache: upper request still returns mean weights.
    again = network.effective_graph(bound="upper")
    assert again[0][2]["weight"] == mean[0][2]["weight"]
    assert again[0][2]["weight"] == build_effective_graph(
        network.nodes, bound="mean"
    )[0][2]["weight"]
    assert again[0][2]["weight"] != build_effective_graph(
        network.nodes, bound="upper"
    )[0][2]["weight"]


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
