from cana.datasets.NeuralNets import Neural
from cana.datasets.bio import THALIANA
from cana.boolean_network import BooleanNetwork
from cana.drawing.derrida_curve import derrida_curve
from cana.drawing.canalization_schema import plot_network_schema
from cana.drawing.redundancy_plots import NodeMeasures

version = "18_0.05-3"
# N = THALIANA()
N = Neural.getNeural(version,overlap=False)

# trajectory = BooleanNetwork.trajectory(N,length=2000)

#stg = N.state_transition_graph()
#print('Number of nodes: {:d}'.format(stg.number_of_nodes()))
#print('Number of edges: {:d}'.format(stg.number_of_edges()))
#att = N.attractors(mode='stg')
#print(att)
#print('Number of attractors: {:d}'.format(len(att)))

### Calculate Diedera Coefficients
# derridacurve = BooleanNetwork.derrida_curve(N,nsamples=100)
# derrida_curve(derridacurve,"outputs","%sDerridaCurve"%(N.name),OpenGraph=True)

### Plot network schema
# plot_network_schema(N)
### Redundancy Plots
NodeMeasures(N,None)
print("end")

##