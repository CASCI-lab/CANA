from cana.datasets.bio import THALIANA, BUDDING_YEAST

N = THALIANA()
#N = BUDDING_YEAST()
print(N)

for n in N.nodes:
	print(n)

stg = N.state_transition_graph()
print('Number of nodes: {:d}'.format(stg.number_of_nodes()))
print('Number of edges: {:d}'.format(stg.number_of_edges()))
att = N.attractors(mode='stg')
print(att)
print('Number of attractors: {:d}'.format(len(att)))