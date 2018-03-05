# -*- coding: utf-8 -*-
"""
Drawing the Dynamics Canalization Map (DCM)
============================================

description ... 

"""
#	Copyright (C) 2017 by
#	Rion Brattig Correia <rionbr@gmail.com>
#	Alex Gates <ajgates@indiana.edu>
#	All rights reserved.
#	MIT license.try:
import warnings
try:
	import graphviz
except:
	warnings.warn("'Graphviz' could not be loaded, you won't be able to plot graphs. Try installing it first.")
from .. utils import *
import networkx as nx


def draw_dynamics_canalization_map_graphviz(DG, engine='neato', overlap='false',
		on_node_fillcolor='black', on_node_fontcolor='white',
		off_node_fillcolor='white', off_node_fontcolor='black',
		
		input_node_bordercolor='gray',
		output_node_bordercolor='red',
		controlled_node_bordercolor='green',
		constant_node_bordercolor='pink',
		
		literal_edge_color='#208120', output_edge_color='#812020',
		fusing_edge_color='#a5a5cc', fused_edge_color='#202081',
		simplified_edge_color='#cca37a',
		
		fusing_edge_arrowhead='none',
		fused_edge_arrowhead='dot',
		literal_edge_arrowhead='dot',
		out_edge_arrowhead='normal',
		simplified_edge_arrowhead='normal',
		*args, **kwargs):
	""" Draws the Dynamics Canalization Map (DCM) using the GraphViz plotting engine.
	
	Args:
		DG (networkx.DiGraph) : The node Canalizing Map (DCM).
		simpily (bool) : Attemps to simpify the DCM by removing thresholds nodes with :math:`\tao=1`
	Returns:
		(graphviz) : The network in graphviz dot format.
	"""
	G = graphviz.Digraph(engine=engine)
	G.graph_attr.update(overlap=overlap)
	G.node_attr.update(fontname='helvetica', shape='circle', fontcolor='black', fontsize='12', width='.4', fixedsize='true', style='filled', color='gray', penwidth='3')
	G.edge_attr.update(arrowhead='dot', color='gray', arrowsize='1', constraint='true')

	#cl = graphviz.Digraph('cluster_left')
	#cr = graphviz.Digraph('cluster_right')
	#cc = graphviz.Digraph('cluster_center')
	#cl.graph_attr.update(rankdir='LR', pos="0,0!", color='red')
	#cr.graph_attr.update(rankdir='LR', pos="100,0!", color='blue')
	
	#groupset = set( [d['group'] for n,d in DG.nodes(data=True) ] )
	#C = {name:graphviz.Digraph('cluster_%s' % name) for name in groupset}

	# Input Used
	input_used = set()

	# Nodes
	for n,d in DG.nodes(data=True):
		g = d['group']
		if 'type' not in d:
			raise AttributeError("Node type could not be found. Must be either 'variable', 'threshold' or 'fusion'.")

		# Variable Nodes
		if d['type'] == 'variable':

			if 'mode' in d:
				# Border Color. Dependents if 'input', 'output', 'controlled' or 'constant'
				if d['mode'] == 'input' or d['mode'] == 'output':
					mode_bordercolor = input_node_bordercolor
				elif d['mode'] == 'constant':
					mode_bordercolor = constant_node_bordercolor
				elif d['mode'] == 'controled':
					mode_bordercolor = controlled_node_bordercolor
			else:
				mode_bordercolor = output_node_bordercolor


			if d['value'] == 0:
				G.node(name=n, label=d['label'], fontcolor=off_node_fontcolor, fillcolor=off_node_fillcolor, color=mode_bordercolor)
			elif d['value'] == 1:
				G.node(name=n, label=d['label'], fontcolor=on_node_fontcolor, fillcolor=on_node_fillcolor, color=mode_bordercolor)

		# Threshold Nodes
		elif d['type'] == 'threshold':
			G.node(name=n, label=d['label'], shape='diamond', style='filled,solid', fillcolor='#dae8f4', fontcolor='black', color='#b5d1e9', width='.4', height='.4')

		elif d['type'] == 'fusion':
			G.node(name=n, label='', shape='none', width='0', height='0', margin='0')


	# Edges
	for s,t,d in DG.edges(data=True):
		color = literal_edge_color
		arrowhead = out_edge_arrowhead

		if 'type' in d:
			if d['type'] == 'out':
				arrowhead = out_edge_arrowhead
				color = output_edge_color

			elif d['type'] == 'literal':
				arrowhead = literal_edge_arrowhead
				color = literal_edge_color

			elif d['type'] == 'fusing':
				arrowhead = fusing_edge_arrowhead
				color = fusing_edge_color

			elif d['type'] == 'fused':
				arrowhead = fused_edge_arrowhead
				color = fused_edge_color
			
			elif d['type'] == 'simplified':
				if d['mode'] == 'selfloop':
					arrowhead = simplified_edge_arrowhead
					color = simplified_edge_color
				elif d['mode'] == 'direct':
					arrowhead = simplified_edge_arrowhead
					color = simplified_edge_color
			else:
				raise AttributeError("Node type could not be found. Must be either 'out', 'literal', 'fusing' or 'fused'.")
		G.edge(s, t, arrowhead=arrowhead, color=color)

	return G


