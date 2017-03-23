# -*- coding: utf-8 -*-
"""
Boolean Node
=============

Description...

"""
#   Copyright (C) 2017 by
#   Rion Brattig Correia <rionbr@gmail.com>
#   Alex Gates <ajgates@gmail.com>
#	Etienne Nzabarushimana <enzabaru@indiana.edu>
#   All rights reserved.
#   MIT license.
from __future__ import division
import numpy as np
import pandas as pd
import itertools
from collections import defaultdict
import boolean_canalization as BC
from base import deprecated
import warnings
try:
	import graphviz
except:
	warnings.warn("'Graphviz' could not be loaded, you won't be able to plot graphs. Try installing it first.")
from utils import *
#
#
#
class BooleanNode(object):
	"""
	
	
	"""
	def __init__(self, name='x', k=1, inputs=['i_0'], state=False, outputs=[0,1], constant=False, verbose=False, *args, **kwargs):
		self.name = name 				# the name of the node
		self.k = k 						# k is the number of inputs
		self.inputs = inputs 			# the name of the input variables
		self.state = state 				# the initial state of the node
		self.outputs = outputs 			# the list of transition outputs
		self.verbose = verbose 			# verbose mode

		# Consistency
		if (k != 0) and (k != int(np.log2(len(outputs)))):
			raise ValueError('Number of k (inputs) do not match the number of output transitions')

		# If all outputs are either positive or negative, this node can be treated as a constant.
		if (len(set(outputs))==1) or (constant):
			self.constant = True
		else:
			self.constant = False

		# Canalization Variables
		self.transition_density_tuple = None 	# A tuple of transition tables used in the first step of the QM algorithm.
		self.prime_implicants = None 			# A tuple of negative and positive prime implicants.
		self.two_symbols = None 				# The Two Symbol (TS) Schemata
		self.pi_coverage = None 				# The Coverage of inputs by Prime Implicants schemata
		self.ts_coverage = None 				# The Coverage of inputs by Two Symbol schemata

	def __str__(self):
		if len(self.outputs) > 10 :
			outputs = '[' + ','.join(map(str, self.outputs[:4])) + '...' + ','.join(map(str, self.outputs[-4:])) + ']'
		else:
			outputs = '[' + ','.join(map(str,self.outputs)) + ']'
		inputs = '[' + ','.join(self.inputs) + ']'
		return "<BNode(name='%s', k=%s, inputs=%s, state=%d, outputs='%s' constant=%s)>" % (self.name, self.k, inputs, self.state, outputs, self.constant)

	@classmethod
	def from_output_list(self, outputs=list(), *args, **kwargs):
		"""Instanciate a Boolean Node from a output transition list.

		Args:
			outputs (list) : The transition outputs of the node.
		Returns:
			(BooleanNode) : the instanciated object.
		Example:
			>>> node = BooleanNode.from_output_list(name="AND", outputs=[0,0,0,1])
		"""
		name = kwargs.pop('name') if 'name' in kwargs else 'x'
		k = int(np.log2(len(outputs)))
		inputs = kwargs.pop('inputs') if 'inputs' in kwargs else ['i%d' % (x+1) for x in xrange(k)]
		state = kwargs.pop('state') if 'state' in kwargs else False

		return BooleanNode(name=name, k=k, inputs=inputs, state=state, outputs=outputs, *args, **kwargs)

	def input_redundancy(self, mode='node', bound='upper', norm=True):
		r""" The Input Redundancy :math:`k_{r}` is the mean number of unnecessary inputs (or ``#``) in the Prime Implicants Look Up Table (LUT).
		Since there may be more than one redescription schema for each input entry, the input redundancy is bounded by an upper and lower limit.
		It can also be computed per input :math:`r_i`.
	

		.. math::

			k_{r}(x) = \frac{ \sum_{f_{\alpha} \in F} \Phi_{\theta:f_{\alpha} \in \Theta_{\theta}} (n^{\#}_{\theta} ) }{ |F| }

		.. math::

			r_i(x_i) = \frac{ \sum_{f_{\alpha} \in F} \Phi_{\theta:f_{\alpha} \in \Theta_{\theta}} (X^{\#}_{\theta i} ) }{ |F| }
	
		where :math:`\Phi` is a function (:math:`min` or :math:`max`) and :math:`F` is the node LUT.

		Args:
			mode (string) : Per "input" or per "node". Defaults to "node".
			bound (string) : The bound to which compute input redundancy.
				Mode "node" accepts: ["lower", "upper"].
				Mode "input" accepts: ["lower", "mean", "upper", "tuple"].
				Defaults to "upper".
			norm (bool) : Normalized between [0,1].
				Use this value when comparing nodes with different input sizes. (Defaults to "True".)

				:math:`k^{*}_r(x) = \frac{ k_r(x) }{ k(x) }`. 


		Returns:
			(float / list) : The :math:`k_r` value or a list of :math:`r_i`.
		
		Note:
			The complete mathematical description can be found in Marques-Pita & Rocha [2013].

		See also:
			:func:`effective_connectivity`, :func:`input_symmetry`.
		"""

		self._check_compute_cannalization_variables(pi_coverage=True)

		# Per Node
		if mode == 'node':

			if bound == 'upper':
				minmax = max
			elif bound == 'lower':
				minmax = min
			else:
				raise TypeError('The bound you selected does not exist. Try "upper", or "lower"')		

			redundancy = [minmax([pi.count('2') for pi in self.pi_coverage[binstate]]) for binstate in self.pi_coverage]
			
			k_r = sum(redundancy) / 2**self.k

			if (norm):
				# Normalizes
				k_r = k_r / self.k

			return k_r

		# Per Input
		elif mode == 'input':

			redundancies = []
			# Generate a per input coverage
			# ex: {0: {'11': [], '10': [], '00': [], '01': []}, 1: {'11': [], '10': [], '00': [], '01': []}}
			pi_input_coverage = { input : { binstate: [ pi[input] for pi in pis ] for binstate,pis in self.pi_coverage.items() } for input in xrange(self.k) }

			# Loop ever input node
			for input,binstates in pi_input_coverage.items():
				# {'numstate': [matches], '10': [True,False,True,...] ...}
				countslenghts = {binstate_to_statenum(binstate): ([pi=='2' for pi in pis]) for binstate,pis in binstates.items() }
				# A triplet of (min, mean, max) values
				if bound == 'lower':
					redundancy = sum( [all(pi) for pi in countslenghts.values()] ) / 2**self.k  # min(r_i)
				elif bound == 'mean':
					redundancy = sum( [sum(pi)/len(pi) for pi in countslenghts.values()] ) / 2**self.k  # <r_i>
				elif bound == 'upper':
					redundancy = sum( [any(pi) for pi in countslenghts.values()] ) / 2**self.k # max(r_i)
				elif bound == 'tuple':
					redundancy = ( sum([all(pi) for pi in countslenghts.values()]) / 2**self.k , sum([any(pi) for pi in countslenghts.values()]) / 2**self.k ) # (min,max)
				else:
					raise TypeError('The bound you selected does not exist. Try "upper", "mean", "lower" or "tuple".')
				
				redundancies.append(redundancy)

			return redundancies # r_i
		
		else:
			raise TypeError('The mode you selected does not exist. Try "node" or "input".')

	def effective_connectivity(self, mode='node', bound='upper', norm=True):
		r"""The Effective Connectiviy is the mean number of input nodes needed to determine the transition of the node.

		.. math::

			k_e(x) = k(x) - k_r(x)
		
		.. math::

			e_i(x_i) = k(x_i) - k_r(x_i)

		Args:
			mode (string) : Per "input" or per "node". Default is "node".
			bound (string) : The bound for the :math:`k_r` Input Redundancy
			norm (bool) : Normalized between [0,1].
				Use this value when comparing nodes with different input sizes. (Defaults to "True".)

				:math:`k^{*}_e(x) = \frac{ k_e(x) }{ k(x) }`. 

		
		Returns:
			(float/list) : The :math:`k_e` value or a list of :math:`e_r`.

		See Also:
			:func:`input_redundancy`, :func:`input_symmetry`, :func:`~boolnets.boolean_network.BooleanNetwork.effective_graph`.
		"""
		if mode == 'node':
			
			k_r = self.input_redundancy(mode=mode, bound=bound, norm=False)
			k_e = self.k - k_r
			if (norm):
				k_e = k_e / self.k
			return k_e

		elif mode == 'input':
			e_i = [1 - x_i for x_i in self.input_redundancy(mode=mode, bound=bound, norm=False)]
			return e_i
		else:
			raise TypeError('The mode you selected does not exist. Try "node" or "input".')

	def input_symmetry(self, mode='node', bound='upper', norm=True):
		r"""The Input Symmetry is a measure of permutation redundancy.
		Similar to the computation of Input Redundancy but using the Two-Symbol instead of the Prime Implicant schemata.
	
		.. math::

			k_s = \frac{ \sum_{f_{\alpha} \in F} \Phi_{\theta:f_{\alpha} \in \Theta_{\theta}} (n^{\circ}) }{ |F| }
		
		.. math::

			r_i = \frac{ \sum_{f_{\alpha} \in F} \Phi_{\theta:f_{\alpha} \in \Theta_{\theta}} (n^{\circ}_i)}{ |F| }

		where :math:`\Phi` is the function :math:`min` or :math:`max` and :math:`F` is the node LUT.

		Args:
			mode (string) : Per "input" or per "node". Default is "node".
			bound (string) : The bound to which compute input symmetry.
				Mode "node" accepts: ["lower", "upper"].
				Mode "input" accepts: ["lower", "mean", "upper", "tuple"].
				Defaults to "upper".
			norm (bool) : Normalized between [0,1].
				Use this value when comparing nodes with different input sizes. (Defaults to "True".)

				:math:`k^{*}_s(x) = \frac{ k_s(x) }{ k(x) }`. 

		Returns:
			(float/list) : The :math:`k_s` or a list of :math:`r_i`.
		
		See also:
			:func:`input_redundancy`, :func:`effective_connectivity`
		"""
		self._check_compute_cannalization_variables(ts_coverage=True)

		if mode == 'node':
			if bound == 'upper':
				minmax = max
			elif bound == 'lower':
				minmax = min

			symmetry = []
			for binstate in self.ts_coverage:
				# Every binary state can have multiple schemata covering it
				values = []
				# TwoSymbol , permutation_groups , same-symbols
				for schema,perms,sms in self.ts_coverage[binstate]:
					
					# NEW VERSION
					# For every input, sum the length of each permutable groups it belongs. Then divide by k
					values.append( 
						sum( [ sum([len(x) for x in perms+sms if i in x]) for i in xrange(self.k)] ) / self.k
					)
					
					# OLD VERSION
					"""
					# If there are permutation_groups, get their lenghts
					if len(perms) or len(sms):
						value += minmax([len(idx) for idx in perms+sms])
					"""
				symmetry.append(
					minmax(values)
					)
			k_s = sum(symmetry) / 2**self.k # k_s
			if (norm):
				k_s = k_s / self.k
			return k_s

		elif mode == 'input':
			symmetries = []
			# Generate a per input coverage
			# ex: {0: {'11': [], '10': [], '00': [], '01': []}, 1: {'11': [], '10': [], '00': [], '01': []}}
			ts_input_coverage = { input : { binstate: [ idxs.count(input) for schema,reps,sms in tss for idxs in reps+sms ] for binstate,tss in self.ts_coverage.items() } for input in xrange(self.k) }
			# Loop ever input node
			for input,binstates in ts_input_coverage.items():
				# {'numstate': [number-of-ts's for each match], '10': [0, 2] ...}
				numstates = {binstate_to_statenum(binstate): permuts for binstate,permuts in binstates.items() }
				# A triplet of (min, mean, max) values
				if bound in ['lower','mean','upper']:
					# Min, Mean or Max
					if bound == 'upper':
						minmax = max
					elif bound == 'mean':
						minmax = np.mean
					elif bound == 'lower':
						minmax = min

					symmetry = sum(minmax(permuts) if len(permuts) else 0 for permuts in numstates.values() ) / 2**self.k  # min(r_s)

				elif bound == 'tuple':
					# tuple (min,max) per input, per state
					symmetry = [ ( min(permuts) , max(permuts) ) if len(permuts) else (0,0) for permuts in numstates.values() ] # (min,max)
				else:
					raise TypeError('The bound you selected does not exist. Try "upper", "mean", "lower" or "tuple".')
				symmetries.append(symmetry)
			return symmetries # r_i

		else:
			raise TypeError('The mode you selected does not exist. Try "node" or "input".')

	def look_up_table(self):
		""" Returns the Look Up Table (LUT)
		
		Returns:
			df (pandas.DataFrame): the LUT

		Examples:
			>>> AND = BooleanNode.from_output_list([0,0,0,1])
			>>> AND.look_up_table()

		See also:
			:func:`schemata_look_up_table`

		"""
		d = []
		for statenum, output in zip( xrange(self.k**2), self.outputs):
			# Binary State, Transition
			d.append( (statenum_to_binstate(statenum, base=self.k), output) )
		df = pd.DataFrame(d, columns=['In:','Out:'])
		return df
	
	def schemata_look_up_table(self, type='pi', pi_symbol=u'#', ts_symbol_unicode=u"\u030A", ts_symbol_latex=u"\circ", format='html'):
		""" Returns the simplified schemata Look Up Table (LUT)
		
		Args:
			type (string) : The type of schemata to return, either Prime Implicants ``pi`` or Two-Symbol ``ts``. Defaults to 'pi'.
			pi_symbol (unicode) : The Prime Implicant don't care symbol. Default is ``#``.
			ts_symbols (list) : A list of Two Symbol permutable symbols. Default is ``[u"\u030A",u"\u032F"]``.
			format (string) : The text format inside the cells. Possible values are ``html`` (default) and ``latex``.
		
		Returns:
			df (pandas.DataFrame): the schemata LUT

		Examples:
			>>> AND = BooleanNode.from_output_list([0,0,0,1])
			>>> AND.schemata_look_up_table(type='pi')
	
		Note:
			See the full list of `combining characters <https://en.wikipedia.org/wiki/Combining_character>`_.

		See also:
			:func:`look_up_table`
		"""
		r = []
		# Prime Implicant LUT
		if type == 'pi':
			self._check_compute_cannalization_variables(prime_implicants=True)
			
			pi0s,pi1s = self.prime_implicants
			
			for output, pi in zip([0,1], [pi0s,pi1s]):
				for schemata in pi:
					r.append( (schemata, output) )
		
		# Two Symbol LUT
		elif type == 'ts':
			self._check_compute_cannalization_variables(two_symbols=True)

			ts0s, ts1s = self.two_symbols

			for output, ts in zip([0,1], [ts0s,ts1s]):
				for i,(schemata,permutables,samesymbols) in enumerate(ts):
					string = u''
					if len(permutables):
						# Permutable
						for j, permutable in enumerate(permutables):

							if format == 'latex':
								if j>0:
									string += u' \,  | \, '
								string += r' '.join([x if (k not in permutable) else '\overset{%s}{%s}' % (ts_symbol_latex,unicode(x)) for k,x in enumerate(schemata, start=0)])
							else:
								if j>0:
									string += u' | '
								string += u''.join([x if (k not in permutable) else u'%s%s' % (x,ts_symbol_unicode) for k,x in enumerate(schemata, start=0)])

					else:
						string += schemata
					"""
					if len(samesymbols):
						# Same Symbol
						for j,samesymbol in enumerate(samesymbols):
							if j>0:
								sstring+= ' | '
							sstring += ''.join([x if (k not in samesymbol) else unicode(x)+ts_symbols[-1] for k,x in enumerate(schemata, start=0)])
					"""
					r.append( (string, output) )
		else:
			raise TypeError('The schemata type could not be found. Try "PI" (Prime Implicants) or "TS" (Two-Symbol).')

		# Output Format (Latex Table or Pandas DataFrame)
		if format == 'latex':
			out = r"\begin{array}{ | c | r | l }" + "\n"
			out += r"\hline" + "\n"
			out += r" & In: & Out: \\" + "\n"
			out += r"\hline" + "\n"
			for i,(string,output) in enumerate(r):
				string = string.replace('2','\%s' % (pi_symbol))
				out += r"%d & %s & %s \\" % (i, string,output) + r"\hline" + "\n"
			out += r"\hline" + "\n"
			out += r"\end{array}" + "\n"
			return out
		
		elif format == 'pandas':
			
			r = [(schemata.replace('2',pi_symbol),output) for schemata,output in r]
			return pd.DataFrame(r, columns=['In:','Out:'])

		else:
			TypeError('The format type could not be found. Try "pandas" "latex".')


	def step(self, input):
		""" Returns the output of the node based on a specific input
		Args:
			input (list) : an input to the node.
		
		Returns:
			output (bool) : the output value.
		"""
		if self.constant:
			return self.outputs[0]
		else:
			if isinstance(input, str):
				input = ''.join(input)
			if len(input) != self.k:
				raise ValueError('Input length do not match number of k inputs')

			return self.outputs[binstate_to_statenum(input)]

	def draw_full_canalizing_map(self):
		""" Draws the node complete canalizing map.

		Returns:
			G (dot) : a graphviz representation of the node.
		"""
		self._check_compute_cannalization_variables(two_symbols=True)

		ts0s, ts1s = self.two_symbols
		
		nr_ts = len(ts0s) + len(ts1s)

		G = graphviz.Digraph(engine='dot')

		G.graph_attr.update(splines='curved',overlap='false')
		G.node_attr.update(fontname='helvetica', shape='circle', fontcolor='black', fontsize='12', width='.4', fixedsize='true', style='filled', color='gray', penwidth='3')
		G.edge_attr.update(arrowhead='dot', color='gray', arrowsize='1')

		# Input Used
		input_used = set()

		# Outputs
		G.node(name='x0', label=self.name, fontcolor='black', fillcolor='white')
		G.node(name='x1', label=self.name, fontcolor='white', fillcolor='black')

		# Thresholds
		for output, tspsss in zip( [0,1] , self.two_symbols ):

			if not len(tspsss):
				continue

			for t,(ts, ps, ss) in enumerate(tspsss, start=0):

				
				lits = []
				group0 = []
				group1 = []
				group2 = []
				nlit, ngrp0, ngrp1, ngrp2 = 0,0,0,0 # Tau is the threshold, counted as the sum of (0's and 1's literals; 0's in permutation group; 1's in permutation group)
				
				for i in xrange(self.k):			
					# Is this input in any permutation group?
					input = ts[i]
					if not any([i in group for group in ps]):
						if ts[i] in ['0','1']:
							nlit += 1
							source = i
							lits.append( source )
					else:
						if ts[i] == '0':
							ngrp0 += 1
							group0.append( i )
						elif ts[i] == '1':
							ngrp1 += 1
							group1.append( i )

				tau = nlit + ngrp0 + ngrp1

				# Threshold Node
				tname = 't%s o%s' % (t,output)
				label = "%d\l" % (tau)
				G.node(name=tname, label=label, shape='circle', fillcolor='white;.5:blue', width='.6')

				# Add Edges from Threshold node to output
				xname = 'x%s' % (output)
				G.edge(tname, xname, label='')

				# Literal Edges
				for lit in lits:
					lname = 'i%s o%s' % (lit,ts[lit])
					input_used.add(lname)
					G.edge(lname, tname, label='')

				# Group0
				for fusion in xrange(ngrp0):
					fname = 'f%s t%s' % (fusion, 0)
					G.node(name=fname, label='', shape='none',width='0',height='0',margin='0')
					for input in ps[0]:
						name = 'i%s o%s' % (input,0)
						input_used.add(name)
						G.edge(name,fname, arrowhead='none')
					G.edge(fname,tname, arrowhead='dot')
					
				# Group1
				for fusion in xrange(ngrp1):
					fname = 'f%s t%s' % (fusion, 1)
					G.node(name=fname, label='', shape='none',width='0',height='0',margin='0')
					for input in ps[0]:
						name = 'i%s o%s' % (input,1)
						input_used.add(name)
						G.edge(name,fname, arrowhead='none')
					G.edge(fname,tname)

				#break
		# Draw Input better
		for input in xrange(self.k):
			for transition, fillcolor, fontcolor in zip([0,1], ['white','black'],['black','white']):
				iname = 'i%s o%s' % (input,transition)
				if iname in input_used:
					style = 'filled,solid'
				else:
					style = 'filled,dashed'
				#label = "<i<SUB>%s</SUB>>" % (input+1)
				label = self.inputs[input]
				G.node(name=iname, label=label, shape='circle', fontcolor=fontcolor, style=style, fillcolor=fillcolor)
		#
		return G

	def get_pi_coverage(self):
		""" 
		#TODO DOCSTRING 
		"""
		self._check_compute_cannalization_variables(pi_coverage=True)
		#
		return self.pi_coverage

	def get_ts_coverage(self):
		"""
		#TODO DOCSTRING 
		"""
		self._check_compute_cannalization_variables(ts_coverage=True)
		#
		return self.ts_coverage

	def _check_compute_cannalization_variables(self, **kwargs):
		""" Recursevely check if the requested canalization variables are instantiated/computed, otherwise computes them in order.
		For example: to compute `two_symbols` we need `prime_implicants` first. Likewise, to compute `prime_implicants` we need `transition_density_tuple` first.
		"""
		if 'transition_density_tuple' in kwargs:
			if self.transition_density_tuple is None:
				if self.verbose: print "Computing: Transition Density Tuple Table"
				self.transition_density_tuple = BC.make_transition_density_tables(self.k, self.outputs)
		
		elif 'prime_implicants' in kwargs:
			self._check_compute_cannalization_variables(transition_density_tuple=True)
			if self.prime_implicants is None:
				if self.verbose: print "Computing: Prime Implicants"
				self.prime_implicants = \
					(
						BC.find_implicants_qm(column=self.transition_density_tuple[0]),
						BC.find_implicants_qm(column=self.transition_density_tuple[1])
					)

		elif 'pi_coverage' in kwargs:
			self._check_compute_cannalization_variables(prime_implicants=True)
			if self.pi_coverage is None:
				if self.verbose: print "Computing: Coverage of Prime Implicants"
				self.pi_coverage = BC.computes_pi_coverage(self.k, self.outputs, self.prime_implicants)

		elif 'two_symbols' in kwargs:
			self._check_compute_cannalization_variables(prime_implicants=True)
			if self.two_symbols is None:
				if self.verbose: print "Computing: Two Symbols"
				self.two_symbols = \
					(
						BC.find_two_symbols_v2(k=self.k, prime_implicants=self.prime_implicants[0]),
						BC.find_two_symbols_v2(k=self.k, prime_implicants=self.prime_implicants[1])
					)
		elif 'ts_coverage' in kwargs:
			self._check_compute_cannalization_variables(two_symbols=True)
			if self.ts_coverage is None:
				if self.verbose: print "Computing: Coverage of Two Symbols"
				self.ts_coverage = BC.computes_ts_coverage(self.k, self.outputs, self.two_symbols)

		else:
			pass
		return True

	def bias(self):
		r""" The node bias. The sum of the boolean output transitions divided by the number of entries (:math:`2^k`) in the LUT.

		.. math::

			bias(x) = \frac{ \sum_{f_{\alpha}\in F} s_{\alpha} }{ |F| }

		Returns:
			(float)

		See Also:
			:func:`~boolnets.boolean_network.network_bias`
		"""
		return sum(self.outputs) / 2**self.k




