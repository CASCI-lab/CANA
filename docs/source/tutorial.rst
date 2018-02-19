Tutorials
=========

Below are some examples of how you might use this package.

For more detailed examples see the ``tutorials/`` folder inside the package.
There you will find several ``.ipynb`` files.


Instanciante a Boolean Node
-------------------------------

To instanciante a node from scratch:

.. code-block:: python

	print BooleanNode.from_output_list(outputs=[0,0,0,1], name='AND', inputs=['in1','in2'])
	<BNode(name='AND', k=2, inputs=[in1,in2], state=0, outputs='[0,0,0,1]' constant=False)>

To load a predefined node

.. code-block:: python

	from cana.datasets.bools import *
	print AND()
	<BNode(name='AND', k=2, inputs=[i1,i2], state=0, outputs='[0,0,0,1]' constant=False)>
	
	print OR()
	<BNode(name='OR', k=2, inputs=[i1,i2], state=0, outputs='[0,1,1,1]' constant=False)>


Instanciante a Boolean Network
-------------------------------

To instanciante a network from scratch:

.. code-block:: python

	logic = {
		0:{'name':'in0', 'in':[0],'out':[0,1]}, 
		1:{'name':'in1', 'in':[0],'out':[0,1]}, 
		2:{'name':'out', 'in':[0,1], 'out':[0,0,0,1]}
	}
	print BooleanNetwork.from_dict(logic, name='AND Net')
	<BNetwork(Name='AND Net', N=3, Nodes=['in0', 'in1', 'out'])>

To load a predefined network.

.. code-block:: python

	from cana.datasets.bio import THALIANA
	t = THALIANA()
	print t
	<BNetwork(Name='Arabidopsis Thaliana', N=15, Nodes=['AP3', 'UFO', 'FUL', 'FT', 'AP1', 'EMF1', 'LFY', 'AP2', 'WUS', 'AG', 'LUG', 'CLF', 'TFL1', 'PI', 'SEP'])>

Check inside the ``datasets/`` folder for other networks in ``.cnet`` format.

State-Transition-Graph (STG) & Attractors
-------------------------------------------

To compute the State-Transition-Graph (STG) of a Boolean Network.

.. code-block:: python

	# this is a networkx.DiGraph() object
	STG = t.state_transition_graph()
	# a list of attractors
	attrs = t.attractors(mode='stg')
	
		
Control Driver Nodes
----------------------
	
Discover which nodes control the network based on different methods.
Note that for large networks, some of these methods do not scale.

.. code-block:: python

	# Find driver nodes (bruteforce, takes some time)
	A = t.attractor_driver_nodes(min_dvs=4, max_dvs=5, verbose=True)
	SC = t.structural_controllability_driver_nodes(keep_self_loops=False)
	MDS = t.minimum_dominating_set_driver_nodes(max_search=10)
	FVS = t.feedback_vertex_set_driver_nodes(method='bruteforce', remove_constants=True, keep_self_loops=True)
	#
	print t.get_node_name(SC)
	[['UFO', 'EMF1', 'LUG', 'CLF'], ['UFO', 'LUG', 'CLF', 'SEP']]

LUT, F' and F'' schematas
---------------------------

.. code-block:: python

	AND = BooleanNode.from_output_list([0,0,0,1])
	AND.look_up_table()
	  In:  Out:
	0  00     0
	1  01     0
	2  10     0
	3  11     1

.. code-block:: python

	AND.schemata_look_up_table(type='pi')
	  In:  Out:
	0  0#     0
	1  #0     0
	2  11     1

.. code-block:: python

	AND.schemata_look_up_table(type='ts')
	  In:  Out:
	0  0̊#̊     0
	1  11     1