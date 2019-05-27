# -*- coding: utf-8 -*-
"""
Random Boolean Network
================



"""
#	Copyright (C) 2017 by
#	Alex Gates <ajgates@indiana.edu>
#	Rion Brattig Correia <rionbr@gmail.com>
#	Thomas Parmer <tjparmer@indiana.edu>
#	All rights reserved.
#	MIT license.
from collections import defaultdict, Counter

import numpy as np
import networkx as nx
import random
import itertools

from cana.boolean_network import BooleanNetwork
from cana.boolean_node import BooleanNode

import warnings
import re
#
#

def regular_boolean_network(N=10, K=2, bias=0.5, keep_constants=True, remove_multiedges=True, niter_remove=1000):


	din = [K]*N   # in-degree distrubtion
	dout = [K]*N  # out-degree distrubtion

	regular_graph = nx.directed_configuration_model(din, dout)

	# the configuration graph creates a multigraph with self loops
	# the self loops are OK, but we should only have one copy of each edge
	if remove_multiedges:
		regular_graph = _remove_duplicate_edges(graph = regular_graph, niter_remove = niter_remove)


	# A dict that contains the network logic {<id>:{'name':<string>,'in':<list-input-node-id>,'out':<list-output-transitions>},..}
	bn_dict = {node:{'name':str(node), 'in':sorted([n for n in regular_graph.predecessors(node)]),
		'out':[int(random.random() < bias) for b in range(2**regular_graph.in_degree(node))]} for node in regular_graph.nodes()}

	return BooleanNetwork.from_dict(bn_dict)


def er_boolean_network(N=10, p=0.2, bias=0.5, keep_constants=True, remove_multiedges=True, niter_remove=1000):

	er_graph = nx.erdos_renyi_graph(N, p, directed=True)


	# the configuration graph creates a multigraph with self loops
	# the self loops are OK, but we should only have one copy of each edge
	if remove_multiedges:
		er_graph = _remove_duplicate_edges(graph = er_graph, niter_remove = niter_remove)


	# A dict that contains the network logic {<id>:{'name':<string>,'in':<list-input-node-id>,'out':<list-output-transitions>},..}
	bn_dict = {node:{'name':str(node), 'in':sorted([n for n in er_graph.predecessors(node)]),
		'out':[int(random.random() < bias) for b in range(2**er_graph.in_degree(node))]} for node in er_graph.nodes()}

	return BooleanNetwork.from_dict(bn_dict)


	

def _remove_duplicate_edges(graph, niter_remove = 100):
	edge_list = list(graph.edges())
	edge_frequency = Counter(edge_list)

	duplicate_edges = [edge for edge, num_edge in edge_frequency.items() if num_edge > 1]

	iremove_iter = 0
	while len(duplicate_edges) > 0 and iremove_iter < niter_remove:
		for dedge in duplicate_edges:
			if edge_frequency[dedge] > 1:
				switch_edge = random.choice(edge_list)

				# exchange edges
				graph.remove_edge(dedge[0], dedge[1])
				graph.remove_edge(switch_edge[0], switch_edge[1])
				graph.add_edge(dedge[0], switch_edge[1])
				graph.add_edge(switch_edge[0], dedge[1])

				edge_frequency[dedge] -= 1
				edge_frequency[switch_edge] -= 1

				edge_frequency[(dedge[0], switch_edge[1])] += 1
				edge_frequency[(switch_edge[0], dedge[1])] += 1

				edge_list = list(graph.edges())

		duplicate_edges = [edge for edge, num_edge in edge_frequency.items() if num_edge > 1]
		iremove_iter += 1

	if iremove_iter >= niter_remove:
		print("Warning: multi-edges were not successfully removed after %s iterations!!" % str(iremove_iter))

	return graph

def from_string_boolean(self, string, keep_constants=True, **kwargs):
	"""
	Instanciates a Boolean Network from a Boolean update rules format.

	Args:
		string (string) : A boolean update rules format representation of a Boolean Network.

	Returns:
		(BooleanNetwork) : The boolean network object.

	Examples:
		String should be structured as follow
		```
		#BOOLEAN RULES
		node_name*=node_input_1 [logic operator] node_input_2 ...
		```

	See also:
		:func:`from_string` :func:`from_dict`
	"""

	logic = defaultdict(dict)

	# parse lines to receive node names
	network_file = StringIO(string)
	line = network_file.readline()
	i = 0
	while line != "":
		if line[0] == '#':
			line = network_file.readline()
			continue
		logic[i] = {'name': line.split("*")[0].strip(), 'in':[], 'out':[]}
		line = network_file.readline()
		i += 1

	# Parse lines again to determine inputs and output sequence
	network_file = StringIO(string)
	line = network_file.readline()
	i = 0
	while line != "":
		if line[0] == '#':
			line = network_file.readline()
			continue
		eval_line = line.split("=")[1] #logical condition to evaluate
		# RE checks for non-alphanumeric character before/after node name (node names are included in other node names)
		# Additional characters added to eval_line to avoid start/end of string complications
		input_names = [logic[node]['name'] for node in logic if re.compile('\W'+logic[node]['name']+'\W').search('*'+eval_line+'*')]
		input_nums = [node for input in input_names for node in logic if input==logic[node]['name']]
		logic[i]['in'] = input_nums
		# Determine output transitions
		logic[i]['out'] = output_transitions(eval_line, input_names)
		line = network_file.readline()
		i += 1

	return self.from_dict(logic, keep_constants=keep_constants, **kwargs)


