# -*- coding: utf-8 -*-
"""
Boolean Network
================

description ... 

"""
#	Copyright (C) 2017 by
#	Rion Brattig Correia <rionbr@gmail.com>
#	Alex Gates <ajgates@indiana.edu>
#	All rights reserved.
#	MIT license.
from collections import defaultdict
import cStringIO
import numpy as np
import networkx as nx
import random
#import bnsAttractors.bns as bns @ NEEDS TO CONVERT THE FILE FIRST @
from boolean_node import BooleanNode
import bns
from base import deprecated
from utils import *
#
#
#
class BooleanNetwork:
	"""


	"""
	def __init__(self, name='', logic=None, structural_graph=None, Nnodes=0, *args, **kwargs):
		self.name = name 							# Name of the Network
		self.Nnodes = Nnodes 						# Number of Nodes
		self.logic = logic 							# A dict that contains the network logic
		self.stg = None 							# State-Transition-Graph (STG)
		self.attractors = None 						# ?
		self.attracting_states = None 				# ?
		self.structural_graph = structural_graph 	# A `networkx` Directed Graph
		self.constants = dict([]) 					# A dict that contains of constant variables in the network
		self.Nconstants = 0							# Number of constant variables
		self.keep_constants = True 					# ?
		self.Nstates = 2**Nnodes 					# Number of possible states in the network 2^N
		
		# Intanciate BooleanNodes
		self.nodes = []
		for i in xrange(Nnodes):
			name = logic[i]['name']
			k = len(logic[i]['in'])
			inputs = [logic[j]['name'] for j in logic[i]['in']]
			outputs = logic[i]['out']
			node = BooleanNode(name=name, k=k, inputs=inputs, outputs=outputs)
			self.nodes.append(node)

		# 
		self.func_bin2num = None 					# Helper function. Converts binstate to statenum. It gets updated by `_update_trans_func`
		self.func_num2bin = None 					# Helper function. Converts statenum to binstate. It gets updated by `_update_trans_func`
		#
		self._update_trans_func() 					# Updates helper functions and other variables

		# Canalization Variables
		self.effective_graph = None 				# A `networkx` Directed Graph resulting from the effective connectiviy computation per node input

		# Init Methods
		self.get_structural_graph()


	def __str__(self):
		node_names = [node.name for node in self.nodes]
		return "<BNetwork(Name='%s', N=%d, Nodes=%s)>" % (self.name, self.Nnodes, node_names)

	@classmethod
	def from_file(self, input_file, keep_constants=True):
		"""
		Load the Boolean Network from a file.

		Args:
			infile (string) : The name of a file containing the Boolean Network.

		Returns:
			BooleanNetwork (object) : The boolean network object.

		See also:
			:func:`from_string` :func:`from_dict`
		"""
		with open(input_file, 'r') as infile:
			return self.from_string(infile.read(), keep_constants=keep_constants)

	@classmethod
	def from_string(self, input_string, keep_constants=True):
		"""
		Load the Boolean Network from a string.

		Args:
			input_string (string): The representation of a Boolean Network.

		Returns:
			(BooleanNetwork)

		See also:
			:func:`from_file` :func:`from_dict`

		Note: see examples for more information.
		"""
		network_file = cStringIO.StringIO(input_string)
		logic = defaultdict(dict)
		
		line = network_file.readline()
		while line != "":
			if line[0] != '#' and line != '\n':
				# .v <#-nodes>
				if '.v' in line:
					self.Nnodes = int(line.split()[1])
					for inode in xrange(self.Nnodes):
						logic[inode] = {'name':'','in':[],'out':[]}
				# .l <node-id> <node-name>
				elif '.l' in line:
					logic[int(line.split()[1])-1]['name'] = line.split()[2]
				# .n <node-id> <#-inputs> <input-node-id>
				elif '.n' in line:
					inode = int(line.split()[1]) - 1
					indegree = int(line.split()[2])
					for jnode in xrange(indegree):
						logic[inode]['in'].append(int(line.split()[3 + jnode])-1)
					
					logic[inode]['out'] = [0 for i in xrange(2**indegree) if indegree > 0]

					logic_line = network_file.readline().strip()

					if indegree <= 0:
						if logic_line == '':
							logic[inode]['in'] = [inode]
							logic[inode]['out'] = [0,1]
						else:
							logic[inode]['out'] = [int(logic_line)]
					else:
						while logic_line != '\n' and logic_line != '' and len(logic_line)>1:
							for nlogicline in expand_logic_line(logic_line):
								logic[inode]['out'][binstate_to_statenum(nlogicline.split()[0])] = int(nlogicline.split()[1])
							logic_line = network_file.readline().strip()

				# .e = end of file
				elif '.e' in line:
					break
			line = network_file.readline()

		return self.from_dict(logic, keep_constants=keep_constants)

	@classmethod
	def from_dict(self, logic, keep_constants=True):
		"""Instanciaets a BoolleanNetwork from a logic dictionary.

		Args:
			logic (dict) : The logic dict.
			keep_constants (bool) : 

		Returns:
			(BooleanNetwork)

		See also:
			:func:`from_file` :func:`from_dict`
		"""
		Nnodes = len(logic)
		keep_constants = keep_constants
		constants = {}
		if keep_constants:
			for i, nodelogic in logic.iteritems():
				# No inputs? It's a constant!
				if len(nodelogic['in']) == 0:
					constants[i] = logic[i]['out'][0]

		return BooleanNetwork(logic=logic, Nnodes=Nnodes, constants=constants, keep_constants=keep_constants)

	def get_structural_graph(self, remove_constants=False):
		""" Calculates and returns the structural graph of the boolean network.

		Args:
			remove_constants (bool) : Remove constants from the graph. Defaults to ``False``.
		Returns:
			G (networkx.Digraph) : The boolean network structural graph.
		"""
		self.structural_graph = nx.DiGraph()
	
		# Add Nodes
		for i, node in enumerate(self.nodes, start=0):
			self.structural_graph.add_node(i, {'label':node.name})

		for target in xrange(self.Nnodes):
			for source in self.logic[target]['in']:
				self.structural_graph.add_edge(source, target, {'weight':1.})

		if remove_constants:
			self.structural_graph.remove_nodes_from(self.constants.keys())
		#
		return self.structural_graph

	def get_effective_graph(self, mode='input', bound='upper', threshold=None):
		"""Computes and returns the effective graph of the network.
		In practive it asks each :class:`~boolnets.boolean_node.BooleanNode` for their :func:`~boolnets.boolean_node.BooleanNode.effective_connectivity`.
	
		Args:
			mode (string) : Per "input" or per "node". Defaults to "node".
			bound (string) : The bound to which compute input redundancy.
			threshold (float) : Only return edges above a certain effective connectivity threshold.
				This is usefull when computing graph measures at diffent levels.

		Returns:
			(networkx.DiGraph) : directed graph
		
		See Also:
			:func:`~boolnets.boolean_node.BooleanNode.effective_connectivity`
		"""
		self.effective_graph = nx.DiGraph()

		# Add Nodes
		for i, node in enumerate(self.nodes, start=0):
			self.effective_graph.add_node(i, {'label':node.name})

		# Add Edges
		for i, node in enumerate(self.nodes, start=0):

			if mode == 'node':
				raise Exception('TODO')

			elif mode == 'input':
				e_is = node.effective_connectivity(mode=mode, bound=bound, norm=False)
				for inputs,e_i in zip(self.logic[i]['in'], e_is):
					# If there is a threshold, only return those number above the threshold. Else, return all edges.
					if (threshold is None) or ((threshold is not None) and (e_i > threshold)):
						self.effective_graph.add_edge(inputs,i,{'weight':e_i})						
			else:
				raise TypeError('The mode you selected does not exist. Try "node" or "input".')

		return self.effective_graph

	@deprecated
	def get_variable_names(self, variable_set=[]):
		"""
		# THIS IS USED TO GET THE NODE NAME. NOT NECESSARY ANyNORE
		"""
		return [self.logic[i]['name'] for i in variable_set]

	def number_interactions(self):
		return nx.number_of_edges(self.structural_graph)

	def structural_indegrees(self):
		"""The number of in-degrees in the structural graph.
		
		Returns:
			(int) : the number of in-degrees.
		See also:
			:func:`structural_outdegree`
		"""
		return sorted(self.structural_graph.in_degree().values(), reverse=True)

	def structural_outdegrees(self):
		"""The number of out-degrees in the structural graph.

		Returns:
			(int) : the number of out-degrees.

		See also:
			:func:`structural_indegree`
		"""
		return sorted(self.structural_graph.out_degree().values(), reverse=True)

	@deprecated
	def configuration(self, statenum):
		""" Returns
		Args:
			statenum (int) : The state number.
		Returns:
			binstate (string) : The binary state
		"""
		return self.func_num2bin(statenum)

	def get_state_transition_graph(self):
		"""Creates and returns the full State Transition Graph (STG) for the Boolean Network.

		Returns:
			(networkx.DiGraph) : The state transition graph for the Boolean Network.
				
		"""
		self.stg = nx.DiGraph(name=self.name + 'STG')
		self.stg.add_nodes_from( (i, {'label':self.func_num2bin(i)}) for i in xrange(self.Nstates) )
		for i in xrange(self.Nstates):
			b = self.func_num2bin(i)
			self.stg.add_edge(i, self.func_bin2num(self.step(b)))
		# 
		return self.stg

	def step(self, initial, n=1):
		""" Steps the boolean network 'n' step from the given initial input condition.
		Args:
			initial (string) : the initial state.
			n (int) : the number of steps.
		Returns:
			(string) : The stepped binary state.
		"""
		# for every node:
		#   node input = breaks down initial by node input
		#   asks node to step with the input
		#   append output to list
		# joins the results from each node output
		return ''.join( [ str(node.step( ''.join(initial[j] for j in self.logic[i]['in']) ) ) for i,node in enumerate(self.nodes, start=0) ] )

	def trajectory(self, initial, length=2):
		""" Computes the trajectory of ``length`` steps without the State Transition Graph (STG).

		@ TODO CONVERT STEP_ONE AND TEST THIS @
		"""
		trajectory = [initial]
		for istep in xrange(length):
			trajectory.append(step(trajectory[-1]))
		return trajectory

	def trajectory_to_attractor(self, initial):
		""" Computes the trajectory starting at ``initial`` until it reaches an attracor (this is garanteed)
		Args:
			initial (string): the initial state.
		Returns:
			trajectory (list): the state trajectory between initial and the final attractor state.
		"""
		# Must find the attactors first
		if self.attracting_states is None:
			self.get_attractors()

		trajectory = [initial]
		while (trajectory[-1] not in self.attracting_states):
			trajectory.append(step(trajectory[-1]))

		return trajectory

	def get_attractor(self, initial):
		""" Computes the trajectory starting at ``initial`` until it reaches an attracor (this is garanteed)

		Args:
			initial (string): the initial state.
		Returns:
			attractor (string): the atractor state.
		"""
		# Must find the attactors first
		if self.attracting_states is None:
			self.get_attractors()

		trajectory = trajectory_to_attractor(initial)
		for attractor in self.attractors:
			if trajectory[-1] in attractor:
				return attractor

	def get_attractors(self, mode='stg'):
		""" Find the attractors of the boolean network.

		Args:
			mode (string) : ``stg`` or ``sat``. Defaults to ``stg``.
					``stg``: Uses the full State Transition Graph (STG) and identifies the attractors as strongly connected components.
					``bns``: Uses the SAT-based `BNS<https://people.kth.se/~dubrova/bns.html>_` to find all attractors.

		Returns:
			attractors (list) : A list containing all attractors for the boolean network.
		"""
		if mode == 'stg':
			if self.stg is None:
				self.get_state_transition_graph()
			self.attractors = [list(a) for a in nx.attracting_components(self.stg)]
		
		elif mode == 'bns':
			self.attractors = bns.get_attractors(self.to_cnet(file=None, adjust_no_input=False))
		else:
			raise TypeError("Could not find the specified mode. Try 'stg' or 'bns'.")

		self.attractors.sort(key=len,reverse=True)
		self.attracting_states = [s for att in self.attractors for s in att]
		return self.attractors

	def network_bias(self):
		"""Network Bias. The sum of individual node biases divided by the number of nodes.
		Practically, it asks each node for their own bias.

		.. math:
			TODO

		See Also:
			:func:`~boolnets.boolean_node.bias`
		"""
		# Ask each node for their bias
		nodes_bias = [node.bias() for node in self.nodes]
		#
		return sum(nodes_bias) / self.Nnodes

	def get_stg_indegree(self):
		"""

		"""
		if self.stg is None:
			self.get_state_transition_graph()
		return sorted(self.stg.in_degree().values(), reverse=True)

	def get_basin_entropy(self, base = 2):
		"""

		"""
		if self.stg is None:
			self.get_state_transition_graph()
		prob_vec = np.array([len(wcc) for wcc in nx.weakly_connected_components(self.stg)])/2.0**self.Nnodes
		return entropy(prob_vec, base = base)

	def derrida_curve(self, nsamples=10, random_state=None):
		"""
		
		"""
		random.seed(random_state)

		dx = np.linspace(0,1,self.Nnodes)
		dy = np.zeros(self.Nnodes)

		# for each possible hamming distance between the starting states
		for hamm_dist in xrange(1, self.Nnodes + 1):
			
			# sample nsample times
			for isample in xrange(nsamples):
				rnd_config = [random.choice(['0', '1']) for b in xrange(self.Nnodes)]
				perturbed_var = random.sample(range(self.Nnodes), hamm_dist)
				perturbed_config = [_flip_bit(rnd_config[ivar]) if ivar in perturbed_var else rnd_config[ivar] for ivar in xrange(self.Nnodes)]
				dy[hamm_dist-1] += hamming_distance(self.step(rnd_config), self.step(perturbed_config))
		
		dy /= float(self.Nnodes * nsamples)

		return dx, dy
				
	def set_constant(self, node, value=None):
		""" Sets or unsets a node as a constant.

		Args:
			node (int) : The node ``id`` in the logic dict.
		Todo:
			This functions needs to better handle node_id and node_name
		"""
		if value is not None:
			self.nodes[node].constant = True
			self.nodes[node].constant_value = value
			self.Nconstants += 1
		else:
			self.nodes[node].constant = False
			self.nodes[node].constant_value = value
			self.Nconstants -= 1

		self._update_trans_func()

	def remove_all_constants(self):
		self.keep_constants = False
		for inode in self.constants:
			self.set_constant(inode, None)

	def _update_trans_func(self):
		"""

		"""
		if self.keep_constants:
			self.Nstates = 2**(self.Nnodes - self.Nconstants)
			constant_template = [None if not (ivar in self.constants.keys()) else self.constants[ivar] for ivar in xrange(self.Nnodes)]
			self.func_bin2num = lambda bs: constantbinstate_to_statenum(bs, constant_template)
			self.func_num2bin = lambda sn: binstate_to_constantbinstate(
				statenum_to_binstate(sn, base = self.Nnodes - self.Nconstants), constant_template)
		else:
			self.Nstates =  2**self.Nnodes
			self.func_bin2num = binstate_to_statenum
			self.func_num2bin = lambda sn: statenum_to_binstate(sn, base = self.Nnodes)


	def to_cnet(self, file=None, adjust_no_input=False):
		""" Outputs the network logic to ``.cnet`` format, which is similar to the Berkeley Logic Interchange Format (BLIF).
		This is the format used by BNS to compute attractors.

		Args:
			file (string,optional) : A string of the file to write the output to. If not supplied, a string will be returned.
			adjust_no_input (bool) : Adjust output string for nodes with no input.

		Returns:
			(string) : The ``.cnet`` format string.

		Note:
			See `BNS <https://people.kth.se/~dubrova/bns.html>`_ for more information.

		"""
		# Copy
		logic = self.logic.copy()
		#
		if adjust_no_input:
			for i, data in logic.iteritems():
				# updates in place
				if len(data['in']) == 0:
					data['in'] = [i + 1]
					data['out'] = [0,1]

		bns_string = '.v ' + str(self.Nnodes) + '\n' + '\n'
		for i in xrange(self.Nnodes):
			k = len(logic[i]['in'])
			bns_string += '.n ' + str(i + 1) + " " + str(k) + " " + " ".join([str(v + 1) for v in logic[i]['in']]) + "\n"
			for statenum in xrange(2**k):
				# If is a constant (TODO: This must come from the BooleanNode, not the logic)
				if len(logic[i]['out']) == 1:
					bns_string += str(logic[i]['out'][statenum]) + "\n"
				# Not a constant, print the state and output
				else:
					bns_string += statenum_to_binstate(statenum, base=k) + " " + str(logic[i]['out'][statenum]) + "\n"
			bns_string += "\n"

		if file is None:
			return bns_string
		else:
			if isinstance(file, string):
				with open(file, 'w') as iofile:
					iofile.write(bns_string)
					iofile.close()
			else:
				raise TypeError("File format not supported. Please specify a string.")

