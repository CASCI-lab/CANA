from cana.boolean_network import BooleanNetwork
import networkx as nx
from cana.datasets.bio import THALIANA, LEUKEMIA

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


def test_output_transitions():
    """Test output_transitions produces correct truth tables."""
    from cana.utils import output_transitions

    assert output_transitions('A', ['A']) == [0, 1]
    assert output_transitions('A and B', ['A', 'B']) == [0, 0, 0, 1]
    assert output_transitions('A or B', ['A', 'B']) == [0, 1, 1, 1]
    assert output_transitions('(A or B) and not C', ['A', 'B', 'C']) == [0, 0, 1, 0, 1, 0, 1, 0]


def test_from_string_boolean():
    """Test that BooleanNetwork.from_string_boolean parses logical rules."""
    rules = "\n".join([
        "A *= B",
        "B *= A and C",
        "C *= A or B",
    ])
    network = BooleanNetwork.from_string_boolean(rules, keep_constants=True)
    assert network.Nnodes == 3
    # A = copy of B: output transitions [0, 1]
    assert network.nodes[0].outputs == ['0', '1']
    # B = A AND C: output transitions [0, 0, 0, 1]
    assert network.nodes[1].outputs == ['0', '0', '0', '1']
    # C = A OR B: output transitions [0, 1, 1, 1]
    assert network.nodes[2].outputs == ['0', '1', '1', '1']


def test_load_logical_LEUKEMIA():
    """Test loading a real model in 'logical' boolean-rule format."""
    network = LEUKEMIA()
    assert network.Nnodes == 60
    # Spot-check first rule: CTLA4* = TCR (identity)
    assert network.nodes[0].name == 'CTLA4'
    assert network.nodes[0].outputs == ['0', '1']
