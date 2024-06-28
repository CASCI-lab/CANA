from cana.datasets.bio import THALIANA

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
