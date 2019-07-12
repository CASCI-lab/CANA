import networkx as nx
import numpy as np
from itertools import product, zip_longest, count
import copy
import math
import random
import operator as op

from heapq import heappush, heappop

def recursive_map(f,d):
	"""Normal python map, but recursive

	Args:
		f (function) : a function to be applied to every item of the iterable
		d (iterable) : the iterable to which f will be applied itemwise.
	"""
	return [ not hasattr(x, "__iter__") and f(x) or recursive_map(f, x) for x in d ]

def binstate_to_statenum(binstate):
	"""Converts from binary state to state number.

	Args:
		binstate (string) : The binary state.

	Returns:
		int : The state number.

	Example:

		.. code-block:: python

			'000' -> 0
			'001' -> 1
			'010' -> 2 ...

	See also:
		:attr:`statenum_to_binstate`, :attr:`statenum_to_density`
	"""
	return int(binstate, 2)

def statenum_to_binstate(statenum, base):
	"""Converts an interger into the binary string.

	Args:
		statenum (int) : The state number.
		base (int) : The binary base

	Returns:
		string : The binary state.

	Example:

		.. code-block:: python

			0 -> '00' (base 2)
			1 -> '01' (base 2)
			2 -> '10' (base 2)
			...
			0 -> '000' (base 3)
			1 -> '001' (base 3)
			2 -> '010' (base 3)

	See also:
		:attr:`binstate_to_statenum`, :attr:`binstate_to_density`
	"""

	return bin(statenum)[2:].zfill(base)

def random_binstate(base, random_seed=None):
	"""
		create a random binary state
	"""
	random.seed(random_seed)

	return statenum_to_binstate(random.randint(0,2**base), base)

def binstate_pinned_to_binstate(binstate, pinned_binstate, pinned_var):
	"""Combines two binstates based on the locations of pinned variables.

	Args:
		binstate (str) : the binary state of non-pinned variables
		pinned_binstate (str) : the binary states of the pinned variables
		pinned_var (list of int) : the list of pinned variables

	Returns:
		string : The combined binary state.

	See also:
		:attr:'statenum_to_binstate'
	"""
	total_length = len(binstate) + len(pinned_binstate)
	new_binstate = list(statenum_to_binstate(0, base=total_length))
	ipin = 0
	ireg = 0
	for istate in range(total_length):
		if istate in pinned_var:
			new_binstate[pinned_var[ipin]] = pinned_binstate[ipin]
			ipin += 1
		else:
			new_binstate[istate] = binstate[ireg]
			ireg += 1
	return ''.join(new_binstate)


def statenum_to_output_list(statenum, base):
	"""Converts an interger into a list of 0 and 1, thus can feed to BooleanNode.from_output_list()

	Args:
		statenum (int) : the state number
		base (int) : the length of output list

	Returns:
		list : a list of length base, consisting of 0 and 1

	See also:
		:attr:'statenum_to_binstate'
	"""
	return [int(i) for i in statenum_to_binstate(statenum, base)]

def flip_bit(bit):
	"""Flips the binary value of a state.

	Args:
		bit (string/int/bool): The current bit position

	Returns:
		same as input: The flipped bit
	"""
	if isinstance(bit, str):
		return '0' if (bit=='1') else '1'
	elif isinstance(bit, int) or isintance(bit, bool):
		return 0 if (bit == 1) else 1
	else:
		raise TypeError("'bit' type format must be either 'string', 'int' or 'boolean'")

def flip_binstate_bit(binstate, idx):
	"""Flips the binary value of a bit in a binary state.
		Args:
			binstate (string) : A string of binary states.
			idx (int) : The index of the bit to flip.
		Returns:
			(string) : New binary state.
		
		Example:
			
			.. code-block:: python
				flip_bit_in_strstates('000',1) -> '010'
		"""
	if idx+1 > len(binstate):
		raise TypeError("Binary state '{}' length and index position '{}' mismatch.".format(binstate, idx))
	return binstate[:idx] + flip_bit(binstate[idx]) + binstate[idx+1:]

def flip_bitset_in_strstates(strstates, idxs):
	"""Flips the binary value for a set of bits in a binary state.
		Args:
			binstate (string) : The binary state to flip.
			idxs (int) : The indexes of the bits to flip.
		Returns:
			(list) : The flipped states
		Example:
			
			.. code-block:: python
				flip_bit_in_strstates('000',[0,2]) -> ['100','001']
		"""
	return [flip_bit_in_strstates(strstates,idx) for idx in idxs]

def flip_binstate_bit_old(binstate, idx):
	"""Flips the binary value of a bit in a binary state.

	Args:
		binstate (string) : The binary state.
		idx (int) : The index of the bit to flip.

	Returns:
		(string) : New binary state.

	"""
	if idx > len(binstate):
		raise TypeError('Binary state (%s) length and index position (%d) mismatch for.' % (binstate, idx))

	_binstate = list(binstate)
	_binstate[idx] = flip_bit(_binstate[idx])
	return ''.join(_binstate)

def flip_binstate_bit_set(binstate, idxs):
	"""Flips the binary value for a set of bits in a binary state.

	Args:
		binstate (string) : The binary state to flip.
		idxs (int) : The indexes of the bits to flip.

	Returns:
		(list) : The flipped states
	"""
	flipset = []
	if (len(idxs) != 0):
		fb = idxs.pop()
		flipset.extend(flip_binstate_bit_set(binstate, copy.copy(idxs) ) )
		flipset.extend(flip_binstate_bit_set(flip_binstate_bit(binstate, fb), copy.copy(idxs) ) )
	else:
		flipset.append(binstate)
	return flipset

def statenum_to_density(statenum):
	"""Converts from state number to density

	Args:
		statenum (int): The state number

	Returns:
		int: The density of ``1`` in that specific binary state number.

	Example:
		>>> statenum_to_binstate(14, base=2)
		>>> '1110'
		>>> statenum_to_density(14)
		>>> 3
	"""
	return sum(map(int, bin(statenum)[2::]))

def binstate_to_density(binstate):
	"""Converts from binary state to density

	Args:
		binstate (string) : The binary state

	Returns:
		int

	Example:
		>>> binstate_to_density('1110')
		>>> 3
	"""
	return sum(map(int, binstate))

def binstate_to_constantbinstate(binstate, constant_template):
	"""
	Todo:
		Documentation
	"""
	# This function is being used in the boolean_network._update_trans_func
	constantbinstate = ''
	iorig = 0
	for value in constant_template:
		if value is None:
			constantbinstate += binstate[iorig]
			iorig += 1
		else:
			constantbinstate += str(value)

	return constantbinstate

def constantbinstate_to_statenum(constantbinstate, constant_template):
	"""
	Todo:
		Documentation
	"""
	binstate = ''.join([constantbinstate[ivar] for ivar in range(len(constant_template)) if constant_template[ivar] is None])
	return binstate_to_statenum(binstate)

def random_binstate(N):
	"""
	generates a random binary state over N variables

	Args:
		N (int) : the length of the binary state

	Returns:
		binstate (str) : a random binary state
	"""

	return"".join([random.choice(['0', '1']) for bit in range(N)])

def expand_logic_line(line):
	"""This generator expands a logic line containing ``-`` (ie. ``00- 0`` or ``0-0 1``) to a series of logic lines containing only ``0`` and ``1``.

	Args:
		line (string) : The logic line. Format is <binary-state><space><output>.

	Returns:
		generator : a series of logic lines

	Example:
		>>> expand_logic_line('1-- 0')
		>>> 100 0
		>>> 101 0
		>>> 110 0
		>>> 111 0
	"""
	# helper function for expand_logic_line
	def _insert_char(la,lb):
		lc=[]
		for i in range(len(lb)):
			lc.append(la[i])
			lc.append(lb[i])
		lc.append(la[-1])
		return ''.join(lc)

	line1,line2=line.split()
	chunks=line1.split('-')
	if len(chunks)>1:
		for i in product(*[('0','1')]*(len(chunks)-1)):
			yield _insert_char(chunks,i)+' '+line2
	else:
		for i in [line]:
			yield i

def print_logic_table(outputs):
	"""Print Logic Table

	Args:
		outputs (list) : The transition outputs of the function.

	Returns:
		print : a print-out of the logic table.

	Example:
		>>> print_logic_table([0,0,1,1])
		>>> 00 : 0
		>>> 01 : 0
		>>> 10 : 1
		>>> 11 : 1

	"""
	k = int(math.log(len(outputs))/math.log(2))
	for statenum in range(2**k):
		print(statenum_to_binstate(statenum, base=k) + " : " + str(outputs[statenum]))

def entropy(prob_vector, logbase = 2.):
	"""Calculates the entropy given a probability vector

	Todo:
		This should be calculated using ``scipy.entropy``
	"""
	prob_vector = np.array(prob_vector)
	pos_prob_vector = prob_vector[prob_vector > 0]
	return - np.sum(pos_prob_vector * np.log(pos_prob_vector)/np.log(logbase))

def binstate_compare(binstate1, binstate2):
	"""
	Compare each element in two binary states 

	Args:
		binstate1, binstate2 : the two binary states to be compared

	Return:
		c (list, bool) : a list of comparisons
	"""
	return [ (b0==b1) for b0, b1 in zip_longest(binstate1, binstate2)]


def hamming_distance(s1, s2):
	"""Calculates the hamming distance between two configurations strings.

	Args:
		s1 (string): First string
		s2 (string): Second string

	Returns:
		float : The Hamming distance

	Example:
		>>> hamming_distance('001','101')
		>>> 1
	"""
	assert len(s1) == len(s2) , "The two strings must have the same length"
	return sum([ (b0!=b1) for b0, b1 in zip_longest(s1, s2)])


def ncr(n, r):
	"""Return the combination number.
	The combination of selecting `r` items from `n` iterms, order doesn't matter.

	Args:
		n (int): number of elements in collection
		r (int): length of combination

	Returns:
		int
	"""
	r = min(r, n - r)
	if r == 0: return 1
	numer = reduce(op.mul, range(n, n - r, -1))
	denom = reduce(op.mul, range(1, r + 1))
	return numer // denom

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
	"""Python 2 doesn't have math.isclose()
	Here is an equivalent function
	Use this to tell whether two float numbers are close enough
		considering using == to compare floats is dangerous!
		2.0*3.3 != 3.0*2.2 in python!
	Args:
		a (float) : the first float number
		b (float) : the second float number
		rel_tol (float) : the relative difference threshold between a and b
		abs_tol (float) : absolute difference threshold. not recommended for float

	Returns:
		bool
	"""
	return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def output_transitions(eval_line, input_list):
	"""Returns an output list from combinatorically trying all input values

	Args:
		eval_line (string) : logic or arithmetic line to evaluate
		input_list (list) : list of input variables

	Returns:
		list of all possible output transitions (list)

	Example:
		RAS*=(GRB2 or PLCG1) and not GAP
		eval_line = "(GRB2 or PLCG1) and not GAP"
		input_list = ['GRB2', 'PLCG1', 'GAP']

		This function generates the following trial strings:
			000
			001
			010
			011
			100
			101
			110
			111

			A variable is dynamically created for each member of the input list
			and assigned the corresponding value from each trail string.
			The original eval_line is then evaluated with each assignment
			which results in the output list [0, 0, 1, 0, 1, 0, 1, 0]
	"""
	total = 2**len(input_list) # Total combinations to try
	output_list = []
	for i in range(total):
		trial_string = statenum_to_binstate(i, len(input_list) )
		# Evaluate trial_string by assigning value to each input variable
		for j,input in enumerate(input_list):
			exec(input + '=' + trial_string[j])
		output_list.append(int(eval(eval_line)))

	return output_list

def mindist_from_source(G, source):
	dag = nx.bfs_tree(G, source)
	dist = {}  # stores [node, distance] pair
	for node in nx.topological_sort(dag):
		# pairs of dist,node for all incoming edges
		pairs = [(dist[v][0]+1, v) for v in dag.pred[node]]
		if pairs:
			dist[node] = min(pairs)
		else:
			dist[node] = (0, node)

	return dist


def probability_dijkstra_multisource(G, sources, weight, pred=None, paths=None,
						  target=None, min_prob=-10**10):
	"""Uses Dijkstra's algorithm to find shortest weighted paths when all values are negative
	(shortest path is closest to 0)
	modified from Network X

	"""
	G_succ = G._succ if G.is_directed() else G._adj

	push = heappush
	pop = heappop
	dist = {}  # dictionary of final distances
	seen = {}
	# fringe is heapq with 3-tuples (distance,c,node)
	# use the count c to avoid comparing nodes (may not be able to)
	c = count()
	fringe = []
	for source in sources:
		if source not in G:
			raise nx.NodeNotFound("Source {} not in G".format(source))
		seen[source] = 0
		push(fringe, (0, next(c), source))
	while fringe:
		(d, _, v) = pop(fringe)
		if v in dist:
			continue  # already searched this node.
		dist[v] = d
		if v == target:
			break
		for u, e in G_succ[v].items():
			cost = weight(v, u, e)
			if cost is None:
				continue
			vu_dist = dist[v] + cost
			if u not in seen or vu_dist > seen[u]:  # vu_dist < seen[u]:
				seen[u] = vu_dist
				push(fringe, (vu_dist, next(c), u))
				if paths is not None:
					paths[u] = paths[v] + [u]
				if pred is not None:
					pred[u] = [v]
			elif vu_dist == seen[u]:
				if pred is not None:
					pred[u].append(v)

	# The optional predecessor and path dictionaries can be accessed
	# by the caller via the pred and paths objects passed as arguments.
	return dist
