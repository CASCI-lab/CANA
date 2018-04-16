from cana.datasets.bio import MARQUESPITA, THALIANA

#N = MARQUESPITA()
N = THALIANA()
print N
print 

FVS_g = N.feedback_vertex_set_driver_nodes(graph='structural', method='grasp', remove_constants=True)
print N.get_node_name(FVS_g) , '(grasp)'
FVS_bf = N.feedback_vertex_set_driver_nodes(graph='structural', method='bruteforce', remove_constants=True)
print N.get_node_name(FVS_bf) , '(bruteforce)'