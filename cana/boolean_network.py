# -*- coding: utf-8 -*-
"""
Boolean Network
================

Main class for Boolean network objects.

"""
#   Copyright (C) 2021 by
#   Rion Brattig Correia <rionbr@gmail.com>
#   Alex Gates <ajgates@indiana.edu>
#   Thomas Parmer <tjparmer@indiana.edu>
#   All rights reserved.
#   MIT license.
from collections import defaultdict
try:
    import cStringIO.StringIO as StringIO
except ImportError:
    from io import StringIO
import numpy as np
import networkx as nx
import random
import itertools
from cana.boolean_node import BooleanNode
import cana.bns as bns
from cana.control import fvs, mds, sc
from cana.utils import *
import warnings
import re
import copy


class BooleanNetwork:
    """

    """

    def __init__(self, name='', Nnodes=0, logic=None, sg=None, stg=None, stg_r=None, _ef=None, attractors=None,
                 constants={}, Nconstants=0, keep_constants=False,
                 bin2num=binstate_to_statenum, num2bin=statenum_to_binstate,
                 verbose=False, *args, **kwargs):

        self.name = name                            # Name of the Network
        self.Nnodes = Nnodes                        # Number of Nodes
        self.logic = logic                          # A dict that contains the network logic {<id>:{'name':<string>,'in':<list-input-node-id>,'out':<list-output-transitions>},..}
        self._sg = sg                               # Structure-Graph (SG)
        self._stg = stg                             # State-Transition-Graph (STG)
        self._stg_r = stg_r                         # State-Transition-Graph Reachability dict (STG-R)
        self._eg = None                             # Effective Graph, computed from the effective connectivity
        self._attractors = attractors               # Network Attractors
        #
        self.keep_constants = keep_constants        # Keep/Include constants in some of the computations
        self.constants = constants                  # A dict that contains of constant variables in the network
        self.Nconstants = len(constants)            # Number of constant variables
        #
        self.Nstates = 2**Nnodes                    # Number of possible states in the network 2^N
        #
        self.verbose = verbose

        # Intanciate BooleanNodes
        self.name2int = {logic[i]['name']: i for i in range(Nnodes)}
        self.Nself_loops = sum([self.name2int[logic[i]['name']] in logic[i]['in'] for i in range(Nnodes)])

        self.nodes = list()
        for i in range(Nnodes):
            name = logic[i]['name']
            k = len(logic[i]['in'])
            inputs = [self.name2int[logic[j]['name']] for j in logic[i]['in']]
            outputs = logic[i]['out']
            node = BooleanNode(id=i, name=name, k=k, inputs=inputs, outputs=outputs, network=self)
            self.nodes.append(node)

        self.input_nodes = [i for i in range(Nnodes) if (self.nodes[i].constant or ((self.nodes[i].k == 1) and (i in self.nodes[i].inputs)))]
        #
        self.bin2num = bin2num                      # Helper function. Converts binstate to statenum. It gets updated by `_update_trans_func`
        self.num2bin = num2bin                      # Helper function. Converts statenum to binstate. It gets updated by `_update_trans_func`
        self._update_trans_func()                   # Updates helper functions and other variables

    def __str__(self):
        node_names = [node.name for node in self.nodes]
        return "<BNetwork(name='{name:s}', N={number_of_nodes:d}, Nodes={nodes:})>".format(name=self.name, number_of_nodes=self.Nnodes, nodes=node_names)

    #
    # I/O Methods
    #
    @classmethod
    def from_file(self, file, type='cnet', keep_constants=True, **kwargs):
        """
        Load the Boolean Network from a file.

        Args:
            file (string) : The name of a file containing the Boolean Network.
            type (string) : The type of file, either 'cnet' (default) or 'logical' for Boolean logical rules.

        Returns:
            BooleanNetwork (object) : The boolean network object.

        See also:
            :func:`from_string` :func:`from_dict`
        """
        with open(file, 'r') as infile:
            if type == 'cnet':
                return self.from_string_cnet(infile.read(), keep_constants=keep_constants, **kwargs)
            elif type == 'logical':
                return self.from_string_boolean(infile.read(), keep_constants=keep_constants, **kwargs)

    @classmethod
    def from_string_cnet(self, string, keep_constants=True, **kwargs):
        """
        Instanciates a Boolean Network from a string in cnet format.

        Args:
            string (string): A cnet format representation of a Boolean Network.

        Returns:
            (BooleanNetwork)

        Examples:
            String should be structured as follow:

            .. code-block:: text

                #.v = number of nodes
                .v 1
                #.l = node label
                .l 1 node-a
                .l 2 node-b
                #.n = (node number) (in-degree) (input node 1) … (input node k)
                .n 1 2 4 5
                01 1 # transition rule

        See also:
            :func:`from_file` :func:`from_dict`
        """
        network_file = StringIO(string)
        logic = defaultdict(dict)

        line = network_file.readline()
        while line != "":
            if line[0] != '#' and line != '\n':
                # .v <#-nodes>
                if '.v' in line:
                    Nnodes = int(line.split()[1])
                    for inode in range(Nnodes):
                        logic[inode] = {'name': '', 'in': [], 'out': []}
                # .l <node-id> <node-name>
                elif '.l' in line:
                    logic[int(line.split()[1]) - 1]['name'] = line.split()[2]
                # .n <node-id> <#-inputs> <input-node-id>
                elif '.n' in line:
                    inode = int(line.split()[1]) - 1
                    indegree = int(line.split()[2])
                    for jnode in range(indegree):
                        logic[inode]['in'].append(int(line.split()[3 + jnode]) - 1)

                    logic[inode]['out'] = [0 for i in range(2**indegree) if indegree > 0]

                    logic_line = network_file.readline().strip()

                    if indegree <= 0:
                        if logic_line == '':
                            logic[inode]['in'] = [inode]
                            logic[inode]['out'] = [0, 1]
                        else:
                            logic[inode]['out'] = [int(logic_line)]
                    else:
                        while logic_line != '\n' and logic_line != '' and len(logic_line) > 1:
                            for nlogicline in expand_logic_line(logic_line):
                                logic[inode]['out'][binstate_to_statenum(nlogicline.split()[0])] = int(nlogicline.split()[1])
                            logic_line = network_file.readline().strip()

                # .e = end of file
                elif '.e' in line:
                    break
            line = network_file.readline()

        return self.from_dict(logic, keep_constants=keep_constants, **kwargs)

    @classmethod
    def from_string_boolean(self, string, keep_constants=True, **kwargs):
        """
        Instanciates a Boolean Network from a Boolean update rules format.

        Args:
            string (string) : A boolean update rules format representation of a Boolean Network.

        Returns:
            (BooleanNetwork) : The boolean network object.

        Examples:
            String should be structured as follow

            .. code-block:: text

                # BOOLEAN RULES (this is a comment)
                # node_name*=node_input_1 [logic operator] node_input_2 ...
                NODE3*=NODE1 AND NODE2 ...

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
            logic[i] = {'name': line.split("*")[0].strip(), 'in': [], 'out': []}
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
            eval_line = line.split("=")[1]  # logical condition to evaluate
            # RE checks for non-alphanumeric character before/after node name (node names are included in other node names)
            # Additional characters added to eval_line to avoid start/end of string complications
            input_names = [logic[node]['name'] for node in logic if re.compile('\W' + logic[node]['name'] + '\W').search('*' + eval_line + '*')]
            input_nums = [node for input in input_names for node in logic if input == logic[node]['name']]
            logic[i]['in'] = input_nums
            # Determine output transitions
            logic[i]['out'] = output_transitions(eval_line, input_names)
            line = network_file.readline()
            i += 1

        return self.from_dict(logic, keep_constants=keep_constants, **kwargs)

    @classmethod
    def from_dict(self, logic, keep_constants=True, **kwargs):
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
        if 'name' in kwargs:
            name = kwargs['name']
        else:
            name = ''
        if keep_constants:
            for i, nodelogic in logic.items():
                # No inputs? It's a constant!
                if len(nodelogic['in']) == 0:
                    constants[i] = logic[i]['out'][0]

        return BooleanNetwork(name=name, logic=logic, Nnodes=Nnodes, constants=constants, keep_constants=keep_constants)

    def to_cnet(self, file=None, adjust_no_input=False):
        """Outputs the network logic to ``.cnet`` format, which is similar to the Berkeley Logic Interchange Format (BLIF).
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
            for i, data in logic.items():
                # updates in place
                if len(data['in']) == 0:
                    data['in'] = [i + 1]
                    data['out'] = [0, 1]

        bns_string = '.v ' + str(self.Nnodes) + '\n' + '\n'
        for i in range(self.Nnodes):
            k = len(logic[i]['in'])
            bns_string += '.n ' + str(i + 1) + " " + str(k) + " " + " ".join([str(v + 1) for v in logic[i]['in']]) + "\n"
            for statenum in range(2**k):
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
                raise AttributeError("File format not supported. Please specify a string.")

    #
    # Methods
    #
    def structural_graph(self, remove_constants=False):
        """Calculates and returns the structural graph of the boolean network.

        Args:
            remove_constants (bool) : Remove constants from the graph. Defaults to ``False``.

        Returns:
            G (networkx.Digraph) : The boolean network structural graph.
        """
        self._sg = nx.DiGraph(name="Structural Graph: " + self.name)

        # Add Nodes
        self._sg.add_nodes_from((i, {'label': n.name}) for i, n in enumerate(self.nodes))
        for target in range(self.Nnodes):
            for source in self.logic[target]['in']:
                self._sg.add_edge(source, target, **{'weight': 1.})

        if remove_constants:
            self._sg.remove_nodes_from(self.constants.keys())
        #
        return self._sg

    def number_interactions(self):
        """Returns the number of interactions in the Structural Graph (SG).
        Practically, it returns the number of edges of the SG.

        Returns:
            int
        """
        self._check_compute_variables(sg=True)
        return nx.number_of_edges(self._sg)

    def structural_indegrees(self):
        """Returns the in-degrees of the Structural Graph. Sorted.

        Returns:
            (int) : the number of in-degrees.

        See also:
            :func:`structural_outdegrees`, :func:`effective_indegrees`, :func:`effective_outdegrees`
        """
        self._check_compute_variables(sg=True)
        return sorted([d for n, d in self._sg.in_degree()], reverse=True)

    def structural_outdegrees(self):
        """Returns the out-degrees of the Structural Graph. Sorted.

        Returns:
            (list) : out-degrees.

        See also:
            :func:`structural_indegrees`, :func:`effective_indegrees`, :func:`effective_outdegrees`
        """
        self._check_compute_variables(sg=True)
        return sorted([d for n, d in self._sg.out_degree()], reverse=True)

    def effective_graph(self, bound='mean', threshold=None):
        """Computes and returns the effective graph of the network.
        In practive it asks each :class:`~cana.boolean_node.BooleanNode` for their :func:`~cana.boolean_node.BooleanNode.edge_effectiveness`.

        Args:
            bound (string) : The bound to which compute input redundancy.
                Can be one of : ["lower", "mean", "upper", "tuple"].
                Defaults to "mean".
            threshold (float) : Only return edges above a certain effective connectivity threshold.
                This is usefull when computing graph measures at diffent levels.

        Returns:
            (networkx.DiGraph) : directed graph

        See Also:
            :func:`~cana.boolean_node.BooleanNode.edge_effectiveness`
        """
        if threshold is not None:
            self._eg = nx.DiGraph(name="Effective Graph: " + self.name + "(Threshold: {threshold:.2f})".format(threshold=threshold))
        else:
            self._eg = nx.DiGraph(name="Effective Graph: " + self.name + "(Threshold: None)")

        # Add Nodes
        for i, node in enumerate(self.nodes, start=0):
            self._eg.add_node(i, **{'label': node.name})

        # Add Edges
        for i, node in enumerate(self.nodes, start=0):
            e_is = node.edge_effectiveness(bound=bound)
            for inputs, e_i in zip(self.logic[i]['in'], e_is):
                # If there is a threshold, only return those number above the threshold. Else, return all edges.
                if (threshold is None) or ((threshold is not None) and (e_i > threshold)):
                    self._eg.add_edge(inputs, i, **{'weight': e_i})

        return self._eg

    def conditional_effective_graph(self, conditioned_nodes={}, bound='mean', threshold=None):
        """Computes and returns the BN effective graph conditioned on some known states.

        Args:
            conditioned_nodes (dict) : a dictionary mapping node ids to their conditioned states.
                dict of form { nodeid : nodestate }
            bound (string) : The bound to which compute input redundancy.
                Can be one of : ["lower", "mean", "upper", "tuple"].
                Defaults to "mean".
            threshold (float) : Only return edges above a certain effective connectivity threshold.
                This is usefull when computing graph measures at diffent levels.

        Returns:
            (networkx.DiGraph) : directed graph

        See Also:
            :func:`~cana.boolean_network.BooleanNetwork.effective_graph`
        """

        conditional_eg = copy.deepcopy(self.effective_graph(bound=bound, threshold=None))
        conditioned_subgraph = set([])

        # make a copy of the logic dict so we can edit it
        conditioned_logic = copy.deepcopy(self.logic)

        # separate input conditioned nodes and nodes that get conditioned
        all_conditioned_nodes = dict(conditioned_nodes)
        # Queue of nodes to condition
        nodes2condition = list(conditioned_nodes.keys())

        while len(nodes2condition) > 0:

            conditioned_node = nodes2condition.pop(0)
            conditioned_value = str(all_conditioned_nodes[conditioned_node])
            conditioned_subgraph.add(conditioned_node)

            # take all successors of the conditioned node ignoring self-loops
            successors = [n for n in list(conditional_eg.neighbors(conditioned_node)) if n != conditioned_node]
            conditioned_subgraph.update(successors)

            # we have to loop through all of the successors of the conditioned node and change their logic
            for n in successors:

                # find the index of the conditioned node in the successor logic
                conditioned_node_idx = conditioned_logic[n]['in'].index(conditioned_node)

                conditioned_subgraph.update(conditioned_logic[n]['in'])

                # the new successor inputs without the conditioned node
                new_successor_inputs = conditioned_logic[n]['in'][:conditioned_node_idx] + conditioned_logic[n]['in'][(conditioned_node_idx + 1):]
                newk = len(new_successor_inputs)

                # now we create a conditioned LUT as the subset of the original for which the conditioned node is fixed to its value
                if newk == 0:
                    new_successor_outputs = [conditioned_logic[n]['out'][binstate_to_statenum(conditioned_value)]] * 2
                else:
                    new_successor_outputs = []
                    for sn in range(2**newk):
                        binstate = statenum_to_binstate(sn, newk)
                        binstate = binstate[:conditioned_node_idx] + conditioned_value + binstate[conditioned_node_idx:]
                        new_successor_outputs.append(conditioned_logic[n]['out'][binstate_to_statenum(binstate)])

                # use the new logic to calcuate a new edge effectiveness
                new_edge_effectiveness = BooleanNode().from_output_list(new_successor_outputs).edge_effectiveness(bound=bound)

                # and update the conditional effective graph with the new edge effectiveness values
                for i in range(newk):
                    conditional_eg[new_successor_inputs[i]][n]['weight'] = new_edge_effectiveness[i]

                # now update the conditioned_logic in case these nodes are further modified by additional conditioned variables
                conditioned_logic[n]['in'] = new_successor_inputs
                conditioned_logic[n]['out'] = new_successor_outputs

                # check if we just made a constant node
                if n not in all_conditioned_nodes and len(set(new_successor_outputs)) == 1:
                    # in which case, add it to the conditioned set and propagate the conditioned effect
                    nodes2condition.append(n)
                    all_conditioned_nodes[n] = new_successor_outputs[0]

        conditional_eg.name = "Conditioned Effective Graph: {name:s} conditioned on {nodes:s}".format(name=self.name, nodes=str(conditioned_nodes))
        if threshold is None:
            conditional_eg.name = conditional_eg.name + " (Threshold: None)"
        else:
            conditional_eg.name = conditional_eg.name + " (Threshold: {threshold:.2f})".format(threshold=threshold)
            remove_edges = [(i, j) for i, j, d in conditional_eg.edges(data=True) if d['weight'] <= threshold]
            conditional_eg.remove_edges_from(remove_edges)

        # add the conditional information into the effective graph object
        dict_conditioned_subgraph = {n: (n in conditioned_subgraph) for n in conditional_eg.nodes()}
        nx.set_node_attributes(conditional_eg, values=dict_conditioned_subgraph, name='conditioned_subgraph')

        dict_all_conditioned_nodes = {n: all_conditioned_nodes.get(n, None) for n in conditional_eg.nodes()}
        nx.set_node_attributes(conditional_eg, values=dict_all_conditioned_nodes, name='conditioned_state')

        return conditional_eg

    def effective_indegrees(self):
        """Returns the in-degrees of the Effective Graph. Sorted.

        Returns:
            (list)

        See also:
            :func:`effective_outdegrees`, :func:`structural_indegrees`, :func:`structural_outdegrees`
        """
        self._check_compute_variables(eg=True)
        return sorted([d for n, d in self._eg.in_degree()], reverse=True)

    def effective_outdegrees(self):
        """Returns the out-degrees of the Effective Graph. Sorted.

        Returns:
            (list)

        See also:
            :func:`effective_indegrees`, :func:`structural_indegrees`, :func:`structural_outdegrees`
        """
        self._check_compute_variables(eg=True)
        return sorted([d for n, d in self._eg.out_degree()], reverse=True)

    def activity_graph(self, threshold=None):
        """
        Returns the activity graph as proposed in
        Ghanbarnejad & Klemm (2012) EPL, 99

        Args:
            threshold (float) : Only return edges above a certain activity threshold.
                This is usefull when computing graph measures at diffent levels.

        Returns:
            (networkx.DiGraph) : directed graph

        """

        if threshold is not None:
            act_g = nx.DiGraph(name="Activity Graph: " + self.name + "(Threshold: %.2f)" % threshold)
        else:
            act_g = nx.DiGraph(name="Activity Graph: " + self.name + "(Threshold: None)")

        # Add Nodes
        for i, node in enumerate(self.nodes, start=0):
            act_g.add_node(i, **{'label': node.name})

        # Add Edges
        for i, node in enumerate(self.nodes, start=0):

            a_is = node.activities()
            for inputs, a_i in zip(self.logic[i]['in'], a_is):
                # If there is a threshold, only return those number above the threshold. Else, return all edges.
                if ((threshold is None) and (a_i > 0)) or ((threshold is not None) and (a_i > threshold)):
                    act_g.add_edge(inputs, i, **{'weight': a_i})

        return act_g

    def state_transition_graph(self):
        """Creates and returns the full State Transition Graph (STG) for the Boolean Network.

        Returns:
            (networkx.DiGraph) : The state transition graph for the Boolean Network.
        """
        self._stg = nx.DiGraph(name='STG: ' + self.name)
        self._stg.add_nodes_from((i, {'label': self.num2bin(i)}) for i in range(self.Nstates))
        for i in range(self.Nstates):
            b = self.num2bin(i)
            self._stg.add_edge(i, self.bin2num(self.step(b)))
        #
        return self._stg

    def stg_indegree(self):
        """Returns the In-degrees of the State-Transition-Graph (STG). Sorted.

        Returns:
            list
        """
        self._check_compute_variables(stg=True)
        return sorted(self._stg.in_degree().values(), reverse=True)

    def step(self, initial):
        """Steps the boolean network 'n' step from the given initial input condition.

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
        assert len(initial) == self.Nnodes
        return ''.join([node.step(node.input_mask(initial)) for node in self.nodes])

    def trajectory(self, initial, length=2):
        """Computes the trajectory of ``length`` steps without the State Transition Graph (STG)."""
        trajectory = [initial]
        for istep in range(length):
            trajectory.append(self.step(trajectory[-1]))
        return trajectory

    def trajectory_to_attractor(self, initial, precompute_attractors=True, return_attractor=False):
        """Computes the trajectory starting at `initial` until it reaches an attracor (this is garanteed).

        Args:
            initial (string) : the initial binstate.
            precompute_attractors (bool) : use precomputed attractors, default True.
            return_attractor (bool) : also return the attractor reached, default False.

        Returns:
            (list) : the state trajectory between initial and the final attractor state.
            if return_attractor: (list): the attractor
        """

        # if the attractors are already precomputed, then we can check when we reach a known state
        if precompute_attractors:
            self._check_compute_variables(attractors=True)
            attractor_states = [self.num2bin(s) for att in self._attractors for s in att]

            trajectory = [initial]
            while (trajectory[-1] not in attractor_states):
                trajectory.append(self.step(trajectory[-1]))

            if return_attractor:
                attractor = self.attractor(trajectory[-1])

        else:
            trajectory = [initial]
            while (trajectory[-1] not in trajectory[:-1]):
                trajectory.append(self.step(trajectory[-1]))

            # the attractor starts at the first occurence of the element
            idxatt = trajectory.index(trajectory[-1])
            if return_attractor:
                attractor = [self.bin2num(s) for s in trajectory[idxatt:-1]]
            trajectory = trajectory[:(idxatt + 1)]

        if return_attractor:
            return trajectory, attractor
        else:
            return trajectory

    def attractor(self, initial):
        """Computes the trajectory starting at ``initial`` until it reaches an attracor (this is garanteed)

        Args:
            initial (string): the initial state.
        Returns:
            attractor (string): the atractor state.
        """
        self._check_compute_variables(attractors=True)

        trajectory = self.trajectory_to_attractor(initial)
        for attractor in self._attractors:
            if self.bin2num(trajectory[-1]) in attractor:
                return attractor

    def attractors(self, mode='stg'):
        """Find the attractors of the boolean network.

        Args:
            mode (string) : ``stg`` or ``sat``. Defaults to ``stg``.
                ``stg``: Uses the full State Transition Graph (STG) and identifies the attractors as strongly connected components.
                ``bns``: Uses the SAT-based :mod:`cana.bns` to find all attractors.
        Returns:
            attractors (list) : A list containing all attractors for the boolean network.
        See also:
            :mod:`cana.bns`
        """
        self._check_compute_variables(stg=True)

        if mode == 'stg':
            self._attractors = [list(a) for a in nx.attracting_components(self._stg)]

        elif mode == 'bns':
            self._attractors = bns.attractors(self.to_cnet(file=None, adjust_no_input=False))
        else:
            raise AttributeError("Could not find the specified mode. Try 'stg' or 'bns'.")

        self._attractors.sort(key=len, reverse=True)
        return self._attractors

    def network_bias(self):
        """Network Bias. The sum of individual node biases divided by the number of nodes.
        Practically, it asks each node for their own bias.

        .. math:
            TODO

        See Also:
            :func:`~cana.boolean_node.BooleanNode.bias`
        """
        return sum([node.bias() for node in self.nodes]) / self.Nnodes

    def basin_entropy(self, base=2):
        """

        """
        self._check_compute_variables(stg=True)

        prob_vec = np.array([len(wcc) for wcc in nx.weakly_connected_components(self._stg)]) / 2.0**self.Nnodes
        return entropy(prob_vec, base=base)

    def set_constant(self, node, value=None):
        """Sets or unsets a node as a constant.

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
            constant_template = [None if not (ivar in self.constants.keys()) else self.constants[ivar] for ivar in range(self.Nnodes)]
            self.bin2num = lambda bs: constantbinstate_to_statenum(bs, constant_template)
            self.num2bin = lambda sn: binstate_to_constantbinstate(
                statenum_to_binstate(sn, base=self.Nnodes - self.Nconstants), constant_template)
        else:
            self.Nstates = 2**self.Nnodes
            self.bin2num = binstate_to_statenum
            self.num2bin = lambda sn: statenum_to_binstate(sn, base=self.Nnodes)

    #
    # Dynamical Control Methods
    #
    def state_transition_graph_reachability(self, filename=None):
        """Generates a State-Transition-Graph Reachability (STG-R) dictionary.
        This dict/file will be used by the State Transition Graph Control Analysis.

        Args:
            filename (string) : The location to a file where the STG-R will be stored.

        Returns:
            (dict) : The STG-R in dict format.
        """
        self._check_compute_variables(stg=True)

        self._stg_r = {}

        if (filename is None):
            for source in self._stg:
                self._stg_r[source] = len(self._dfs_reachable(self._stg, source)) - 1.0
        else:
            try:
                with open(filename, 'rb') as handle:
                    self._stg_r = pickle.load(handle)
            except IOError:
                print("Finding STG dict")
                for source in self._stg:
                    self._stg_r[source] = len(self._dfs_reachable(self._stg, source)) - 1.0
                with open(filename, 'wb') as handle:
                    pickle.dump(self._stg_r, handle)
        return self._stg_r

    def attractor_driver_nodes(self, min_dvs=1, max_dvs=4, verbose=False):
        """Get the minimum necessary driver nodes by iterating the combination of all possible driver nodes of length :math:`min <= x <= max`.

        Args:
            min_dvs (int) : Mininum number of driver nodes to search.
            max_dvs (int) : Maximum number of driver nodes to search.

        Returns:
            (list) : The list of driver nodes found in the search.

        Note:
            This is an inefficient bruit force search, maybe we can think of better ways to do this?

        TODO:
            Parallelize the search on each combination. Each CSTG is independent and can be searched in parallel.

        See also:
            :func:`controlled_state_transition_graph`, :func:`controlled_attractor_graph`.
        """
        nodeids = list(range(self.Nnodes))
        if self.keep_constants:
            for cv in self.constants.keys():
                nodeids.remove(cv)

        attractor_controllers_found = []
        nr_dvs = min_dvs
        while (len(attractor_controllers_found) == 0) and (nr_dvs <= max_dvs):
            if verbose:
                print("Trying with {:d} Driver Nodes".format(nr_dvs))
            for dvs in itertools.combinations(nodeids, nr_dvs):
                dvs = list(dvs)
                # cstg = self.controlled_state_transition_graph(dvs)
                cag = self.controlled_attractor_graph(dvs)
                att_reachable_from = self.mean_reachable_attractors(cag)

                if att_reachable_from == 1.0:
                    attractor_controllers_found.append(dvs)
            # Add another driver node
            nr_dvs += 1

        if len(attractor_controllers_found) == 0:
            warnings.warn("No attractor control driver variable sets found after exploring all subsets of size {:,d} to {:,d} nodes!!".format(min_dvs, max_dvs))

        return attractor_controllers_found

    def controlled_state_transition_graph(self, driver_nodes=[]):
        """Returns the Controlled State-Transition-Graph (CSTG).
        In practice, it copies the original STG, flips driver nodes (variables), and updates the CSTG.

        Args:
            driver_nodes (list) : The list of driver nodes.

        Returns:
            (networkx.DiGraph) : The Controlled State-Transition-Graph.

        See also:
            :func:`attractor_driver_nodes`, :func:`controlled_attractor_graph`.
        """
        self._check_compute_variables(attractors=True)

        if self.keep_constants:
            for dv in driver_nodes:
                if dv in self.constants:
                    warnings.warn("Cannot control a constant variable '%s'! Skipping" % self.nodes[dv].name )

        # attractor_states = [s for att in self._attractors for s in att]
        cstg = copy.deepcopy(self._stg)
        cstg.name = 'C-' + cstg.name + ' (' + ','.join(map(str, [self.nodes[dv].name for dv in driver_nodes])) + ')'

        # add the control pertubations applied to all other configurations
        for statenum in range(self.Nstates):
            binstate = self.num2bin(statenum)
            controlled_states = flip_binstate_bit_set(binstate, copy.copy(driver_nodes))
            controlled_states.remove(binstate)

            for constate in controlled_states:
                cstg.add_edge(statenum, self.bin2num(constate))

        return cstg

    def pinning_controlled_state_transition_graph(self, driver_nodes=[]):
        """Returns a dictionary of Controlled State-Transition-Graph (CSTG)
        under the assumptions of pinning controllability.

        In practice, it copies the original STG, flips driver nodes (variables), and updates the CSTG.

        Args:
            driver_nodes (list) : The list of driver nodes.

        Returns:
            (networkx.DiGraph) : The Pinning Controlled State-Transition-Graph.

        See also:
            :func:`controlled_state_transition_graph`, :func:`attractor_driver_nodes`, :func:`controlled_attractor_graph`.
        """
        self._check_compute_variables(attractors=True)

        if self.keep_constants:
            for dv in driver_nodes:
                if dv in self.constants:
                    warnings.warn("Cannot control a constant variable {dv:s}'! Skipping".format(dv=self.nodes[dv].name))

        uncontrolled_system_size = self.Nnodes - len(driver_nodes)

        pcstg_dict = {}
        for att in self._attractors:
            dn_attractor_transitions = [tuple(''.join([self.num2bin(s)[dn] for dn in driver_nodes]) for s in att_edge)
                                        for att_edge in self._stg.subgraph(att).edges()]

            pcstg_states = [self.bin2num(binstate_pinned_to_binstate(
                            statenum_to_binstate(statenum, base=uncontrolled_system_size), attsource, pinned_var=driver_nodes))
                            for statenum in range(2**uncontrolled_system_size) for attsource, attsink in dn_attractor_transitions]

            pcstg = nx.DiGraph(name='STG: ' + self.name)
            pcstg.name = 'PC-' + pcstg.name + ' (' + ','.join(map(str, [self.nodes[dv].name for dv in driver_nodes])) + ')'

            pcstg.add_nodes_from((ps, {'label': ps}) for ps in pcstg_states)

            for attsource, attsink in dn_attractor_transitions:
                for statenum in range(2**uncontrolled_system_size):
                    initial = binstate_pinned_to_binstate(statenum_to_binstate(statenum, base=uncontrolled_system_size), attsource, pinned_var=driver_nodes)
                    pcstg.add_edge(self.bin2num(initial), self.bin2num(self.pinned_step(initial, pinned_binstate=attsink, pinned_var=driver_nodes)))

            pcstg_dict[tuple(att)] = pcstg

        return pcstg_dict

    def pinned_step(self, initial, pinned_binstate, pinned_var):
        """Steps the boolean network 1 step from the given initial input condition when the driver variables are pinned
        to their controlled states.

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
        assert len(initial) == self.Nnodes
        return ''.join([str(node.step(''.join(initial[j] for j in self.logic[i]['in']))) if not (i in pinned_var) else initial[i] for i, node in enumerate(self.nodes, start=0)])

    def controlled_attractor_graph(self, driver_nodes=[]):
        """
        Args:
            cstg (networkx.DiGraph) : A Controlled State-Transition-Graph (CSTG)

        Returns:
            (networkx.DiGraph) : The Controlled Attractor Graph (CAG)

        See also:
            :func:`attractor_driver_nodes`, :func:`controlled_state_transition_graph`.
        """
        self._check_compute_variables(attractors=True)

        if self.keep_constants:
            for dv in driver_nodes:
                if dv in self.constants:
                    warnings.warn("Cannot control a constant variable '%s'! Skipping" % self.nodes[dv].name )

        attractor_states = [s for att in self._attractors for s in att]
        cstg = copy.deepcopy(self._stg)
        cstg.name = 'C-' + cstg.name + ' Att(' + ','.join(map(str, [self.nodes[dv].name for dv in driver_nodes])) + ')'

        # add the control pertubations applied to only attractor configurations
        for statenum in attractor_states:
            binstate = self.num2bin(statenum)
            controlled_states = flip_binstate_bit_set(binstate, copy.copy(driver_nodes))
            controlled_states.remove(binstate)

            for constate in controlled_states:
                cstg.add_edge(statenum, self.bin2num(constate))

        Nattract = len(self._attractors)

        cag = nx.DiGraph(name='CAG: ' + cstg.name)
        # Nodes
        for i, attr in enumerate(self._attractors):
            cag.add_node(i, **{'label': '|'.join([self.num2bin(a) for a in attr])})
        # Edges
        for i in range(Nattract):
            ireach = self._dfs_reachable(cstg, self._attractors[i][0])
            for j in range(i + 1, Nattract):
                if self._attractors[j][0] in ireach:
                    cag.add_edge(i, j)
                if self._attractors[i][0] in self._dfs_reachable(cstg, self._attractors[j][0]):
                    cag.add_edge(j, i)
        return cag

    def mean_reachable_configurations(self, cstg):
        """Returns the Mean Fraction of Reachable Configurations

        Args:
            cstg (networkx.DiGraph) : The Controlled State-Transition-Graph.
        Returns:
            (float) : Mean Fraction of Reachable Configurations
        """
        reachable_from = []

        for source in cstg:
            control_reach = len(self._dfs_reachable(cstg, source)) - 1.0
            reachable_from.append(control_reach)

        norm = (2.0**self.Nnodes - 1.0) * len(reachable_from)
        reachable_from = sum(reachable_from) / (norm)

        return reachable_from

    def mean_controlable_configurations(self, cstg):
        """The Mean Fraction of Controlable Configurations

        Args:
            cstg (networkx.DiGraph) : The Controlled State-Transition-Graph.
        Returns:
            (float) : Mean Fraction of Controlable Configurations.
        """
        self._check_compute_variables(stg_r=True)

        control_from, reachable_from = [], []

        for source in cstg:
            control_reach = len(self._dfs_reachable(cstg, source)) - 1.0
            control_from.append(control_reach - self._stg_r[source])
            reachable_from.append(control_reach)

        norm = (2.0**self.Nnodes - 1.0) * len(reachable_from)
        control_from = sum(control_from) / (norm)

        return control_from

    def mean_reachable_attractors(self, cag, norm=True):
        """The Mean Fraction of Reachable Attractors to a specific Controlled Attractor Graph (CAG).

        Args:
            cag (networkx.DiGraph) : A Controlled Attractor Graph (CAG).

        Returns:
            (float) Mean Fraction of Reachable Attractors
        """
        att_norm = (float(len(cag)) - 1.0) * len(cag)

        if att_norm == 0:
            # if there is only one attractor everything is reachable
            att_reachable_from = 1
        else:
            # otherwise find the reachable from each attractor
            att_reachable_from = [len(self._dfs_reachable(cag, idxatt)) - 1.0 for idxatt in cag]
            att_reachable_from = sum(att_reachable_from) / (att_norm)

        return att_reachable_from

    def fraction_pinned_attractors(self, pcstg_dict):
        """Returns the Number of Accessible Attractors

        Args:
            pcstg_dict (dict of networkx.DiGraph) : The dictionary of Pinned Controlled State-Transition-Graphs.

        Returns:
            (int) : Number of Accessible Attractors
        """
        reached_attractors = []
        for att, pcstg in pcstg_dict.items():
            pinned_att = list(nx.attracting_components(pcstg))
            print(set(att), pinned_att)
            reached_attractors.append(set(att) in pinned_att)
        return sum(reached_attractors) / float(len(pcstg_dict))

    def fraction_pinned_configurations(self, pcstg_dict):
        """Returns the Fraction of successfully Pinned Configurations

        Args:
            pcstg_dict (dict of networkx.DiGraph) : The dictionary of Pinned Controlled State-Transition-Graphs.

        Returns:
            (list) : the Fraction of successfully Pinned Configurations to each attractor
        """
        pinned_configurations = []
        for att, pcstg in pcstg_dict.items():
            att_reached = False
            for wcc in nx.weakly_connected_components(pcstg):
                if set(att) in list(nx.attracting_components(pcstg.subgraph(wcc))):
                    pinned_configurations.append(len(wcc) / len(pcstg))
                    att_reached = True
            if not att_reached:
                pinned_configurations.append(0)

        return pinned_configurations

    def mean_fraction_pinned_configurations(self, pcstg_dict):
        """Returns the mean Fraction of successfully Pinned Configurations

        Args:
            pcstg_dict (dict of networkx.DiGraph) : The dictionary of Pinned Controlled State-Transition-Graphs.

        Returns:
            (int) : the mean Fraction of successfully Pinned Configurations
        """
        return sum(self.fraction_pinned_configurations(pcstg_dict)) / len(pcstg_dict)

    def _dfs_reachable(self, G, source):
        """Produce nodes in a depth-first-search pre-ordering starting from source."""
        return [n for n in nx.dfs_preorder_nodes(G, source)]

    #
    # Feedback Vertex Set (FVS)
    #
    def feedback_vertex_set_driver_nodes(self, graph='structural', method='grasp', max_iter=1, max_search=11, keep_self_loops=True, *args, **kwargs):
        """The minimum set of necessary driver nodes to control the network based on Feedback Vertex Set (FVS) theory.

        Args:
            graph (string) : Which graph to perform computation
            method (string) : FVS method. ``bruteforce`` or ``grasp`` (default).
            max_iter (int) : The maximum number of iterations used by the grasp method.
            max_search (int) : The maximum number of searched used by the bruteforce method.
            keep_self_loops (bool) : Keep or remove self loop in the graph to be searched.

        Returns:
            (list) : A list-of-lists with FVS solution nodes.

        Note:
            When computing FVS on the structural graph, you might want to use ``remove_constants=True``
            to make sure the resulting set is minimal – since constants are not controlabled by definition.
            Also, when computing on the effective graph, you can define the desired ``threshold`` level.
        """
        self._check_compute_variables(sg=True)

        if graph == 'structural':
            dg = self.structural_graph(*args, **kwargs)
        elif graph == 'effective':
            dg = self.effective_graph(mode='input', bound='mean', threshold=None, *args, **kwargs)
        else:
            raise AttributeError("The graph type '%s' is not accepted. Try 'structural' or 'effective'." % graph)
        #
        if method == 'grasp':
            fvssets = fvs.fvs_grasp(dg, max_iter=max_iter, keep_self_loops=keep_self_loops)
        elif method == 'bruteforce':
            fvssets = fvs.fvs_bruteforce(dg, max_search=max_search, keep_self_loops=keep_self_loops)
        else:
            raise AttributeError("The FVS method '%s' does not exist. Try 'grasp' or 'bruteforce'." % method)

        fvssets = [fvc.union(set(self.input_nodes)) for fvc in fvssets]

        return fvssets  # [ [self.nodes[i].name for i in fvsset] for fvsset in fvssets]

    #
    # Minimum Dominating Set
    #
    def minimum_dominating_set_driver_nodes(self, graph='structural', max_search=5, keep_self_loops=True, *args, **kwargs):
        """The minimun set of necessary driver nodes to control the network based on Minimum Dominating Set (MDS) theory.

        Args:
            max_search (int) : Maximum search of additional variables. Defaults to 5.
            keep_self_loops (bool) : If self-loops are used in the computation.

        Returns:
            (list) : A list-of-lists with MDS solution nodes.
        """
        self._check_compute_variables(sg=True)
        #
        if graph == 'structural':
            dg = self.structural_graph(*args, **kwargs)
        elif graph == 'effective':
            dg = self.effective_graph(mode='input', bound='mean', threshold=None, *args, **kwargs)
        else:
            raise AttributeError("The graph type '%s' is not accepted. Try 'structural' or 'effective'." % graph)
        #

        mdssets = mds.mds(dg, max_search=max_search, keep_self_loops=keep_self_loops)
        return mdssets  # [ [self.nodes[i].name for i in mdsset] for mdsset in mdssets]

    # Structural Controllability
    #
    def structural_controllability_driver_nodes(self, graph='structural', keep_self_loops=True, *args, **kwargs):
        """The minimum set of necessary driver nodes to control the network based on Structural Controlability (SC) theory.

        Args:
            keep_self_loops (bool) : If self-loops are used in the computation.

        Returns:
            (list) : A list-of-lists with SC solution nodes.
        """
        self._check_compute_variables(sg=True)

        if graph == 'structural':
            dg = self.structural_graph(*args, **kwargs)
        elif graph == 'effective':
            dg = self.effective_graph(mode='input', bound='mean', threshold=None, *args, **kwargs)
        else:
            raise AttributeError("The graph type '%s' is not accepted. Try 'structural' or 'effective'." % graph)
        #
        scsets = [set(scset).union(set(self.input_nodes)) for scset in sc.sc(dg, keep_self_loops=keep_self_loops)]
        return scsets  # [ [self.nodes[i].name for i in scset] for scset in scsets]

    #
    # Dynamical Impact
    #
    def partial_derative_node(self, node, n_traj=10, t=1):
        """The partial derivative of node on all other nodes after t steps

        Args:
            node (int) : the node index for perturbations

            t (int) : the number of time steps the system is run before impact is calculated.

            n_traj (int) : the number of trajectories used to approximate the dynamical impact of a node.
                if 0 then the full STG is used to calculate the true value instead of the approximation method.

        Returns:
            (vector) : the partial derivatives
        """
        partial = np.zeros((t, self.Nnodes), dtype=float)
        if n_traj == 0:
            config_genderator = (self.num2bin(statenum) for statenum in range(self.Nstates))
            n_traj = self.Nstates
        else:
            # sample configurations
            config_genderator = (random_binstate(self.Nnodes) for itraj in range(n_traj))

        for config in config_genderator:
            perturbed_config = flip_binstate_bit(config, node)
            for n_step in range(t):
                config = self.step(config)
                perturbed_config = self.step(perturbed_config)
                partial[n_step] += np.logical_not(binstate_compare(config, perturbed_config))
        partial /= n_traj

        return partial

    def approx_dynamic_impact(self, source, n_steps=1, target_set=None, bound='mean', threshold=0.0):
        """Use the network structure to approximate the dynamical impact of a perturbation to node for each of n_steps
        for details see: Gates et al (2020).

        Args:
            source (int) : the source index for perturbations
            n_steps (int) : the number of time steps

        bound (str) : the bound for the effective graph
            'mean' - edge effectiveness
            'upper' - activity

        Returns:
            (matrix) : approximate dynamical impact for each node at each step (2 x n_steps x n_nodes)
        """

        if target_set is None:
            target_set = range(self.Nnodes)

        Gstr = self.structural_graph()

        Geff = self.effective_graph(bound=bound, threshold=threshold)

        # the maximum path with length given by product of weights is the same as minimal path of negative log weight
        def eff_weight_func(u, v, e):
            return -np.log(e['weight'])

        def inv_eff_weight_func(pathlength):
            return np.exp(-pathlength)


        impact_matrix = np.zeros((2, n_steps + 1, len(target_set)))
        impact_matrix[0, :, :] = self.Nnodes + 1  # if we can't reach the node, then the paths cant be longer than the number of nodes in the graph
        # note that by default: impact_matrix[1, :, :] = 0 the minimum path for nodes we cant reach in the effective graph

        # in the structural graph, calcluate the dijkstra shortest paths from the source to all targets that are shorter than the cufoff
        Gstr_shortest_dist, Gstr_shortest_paths = nx.single_source_dijkstra(Gstr, source, target=None, cutoff=n_steps)
        Gstr_shortest_dist = {n: int(l) for n, l in Gstr_shortest_dist.items()}

        # in the effective graph, calcluate the dijkstra shortest paths from the source to all targets that are shorter than the cufoff
        # where the edge weight is given by the effective weight function
        Geff_shortest_dist, Geff_shortest_paths = nx.single_source_dijkstra(Geff, source, target=None, cutoff=n_steps, weight=eff_weight_func)

        for itar, target in enumerate(target_set):

            # we dont need to worry about a path to iteself (source==target)
            # and if the target doesnt appear in the shortest path dict, then no path exists that is less than the cutoff
            if target != source and not Gstr_shortest_dist.get(target, None) is None:

                # the light cone is at least as big as the number of edges in the structural shorest path
                impact_matrix[0, list(range(Gstr_shortest_dist[target], n_steps + 1)), itar] = Gstr_shortest_dist[target]

                # if the path exists, then the number of edges (timesteps) is one less than the number of nodes
                if not Geff_shortest_paths.get(target, None) is None:
                    eff_path_steps = len(Geff_shortest_paths[target]) - 1
                else:
                    # or the path doesnt exist
                    eff_path_steps = n_steps + 100 # any number bigger than the longest path to represent we cannot reach the node


                # start by checking if the number of timesteps is less than the maximum allowable number of steps
                if eff_path_steps <= n_steps:

                    # now check if the most likely effective path is longer (in terms of # of timesteps) than the structural shortest path
                    if eff_path_steps > Gstr_shortest_dist[target]:

                        # if it is, then we need to find another effective path constrained by the light-cone
                        # for all time steps where the most likely effective path is longer (in terms of # of timesteps)
                        # than the structural shortest path
                        for istep in range(Gstr_shortest_dist[target], eff_path_steps):

                            # bc the effective graph has fully redundant edges, there may actually not be a path
                            try:
                                redo_dijkstra_dist, _ = nx.single_source_dijkstra(Geff,
                                    source=source,
                                    target=target,
                                    cutoff=istep,
                                    weight=eff_weight_func)
                                impact_matrix[1, istep, itar] = inv_eff_weight_func(redo_dijkstra_dist)
                            except nx.NetworkXNoPath:
                                pass

                    # once the lightcone includes the target node on the effective shortest path,
                    # then for all other steps the effective path is the best
                    impact_matrix[1, list(range(eff_path_steps, n_steps + 1)), itar] = inv_eff_weight_func(Geff_shortest_dist[target])

        return impact_matrix[:, 1:]


    def dist_from_attractor(self):
        """Find the distance from attractor for each configuration.

        Returns:
            distance (dict). Nodes are dictionary indexes and distances the values.
        """
        self._check_compute_variables(attractors=True)

        dist = {}  # stores [node, distance] pair
        for att in self._attractors:
            dist.update({a: (0, a) for a in att})
            dag = nx.bfs_tree(self._stg, att[0], reverse=True)
            attractor_states = set(att)
            for node in nx.topological_sort(dag):
                # pairs of dist,node for all incoming edges
                if node not in attractor_states:
                    pairs = [(dist[v][0] + 1, v) for v in dag.pred[node]]
                    if pairs:
                        dist[node] = min(pairs)
                    else:
                        dist[node] = (0, node)

        return dist

    def average_dist_from_attractor(self):
        dist = self.dist_from_attractor()
        return np.mean([d[0] for d in dist.values() if d[0] > 0])

    #
    # Dynamics Canalization Map (DCM)
    #
    def dynamics_canalization_map(self, output=None, simplify=True):
        """Computes the Dynamics Canalization Map (DCM).
        In practice, it asks each node to compute their Canalization Map and then puts them together, simplifying it if possible.

        Args:
            output (int) : The output DCM to return. Default is ``None``, retuning both [0,1].
            simplify (bool) : Attemps to simpify the DCM by removing thresholds nodes with :math:`\tao=1`.

        Returns:
            DCM (networkx.DiGraph) : a directed graph representation of the DCM.

        See Also:
            :func:`boolean_node.canalizing_map` for the CM and :func:`drawing.draw_dynamics_canalizing_map_graphviz` for plotting.
        """
        CMs = []
        for node in self.nodes:
            if self.keep_constants or not node.constant:
                CMs.append(node.canalizing_map(output))
        # https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.operators.all.compose_all.html
        DCM = nx.compose_all(CMs)
        DCM.name = 'DCM: %s' % (self.name)

        if simplify:
            # Loop all threshold nodes
            threshold_nodes = [(n, d) for n, d in DCM.nodes(data=True) if d['type'] == 'threshold']
            for n, d in threshold_nodes:

                # Constant, remove threshold node
                if d['tau'] == 0:
                    DCM.remove_node(n)

                # Tau == 1
                if d['tau'] == 1:
                    in_nei = list(DCM.in_edges(n))[0]
                    out_nei = list(DCM.out_edges(n))[0]

                    # neis = set(list(in_nei) + list(out_nei))

                    # Convert to self loop
                    if (in_nei == out_nei[::-1]):
                        DCM.remove_node(n)
                        DCM.add_edge(in_nei[0], out_nei[1], **{'type': 'simplified', 'mode': 'selfloop'})
                    # Link variables nodes directly
                    elif not any([DCM.nodes[tn]['type'] == 'fusion' for tn in in_nei]):
                        DCM.remove_node(n)
                        DCM.add_edge(in_nei[0], out_nei[1], **{'type': 'simplified', 'mode': 'direct'})
        # Remove Isolates
        isolates = list(nx.isolates(DCM))
        DCM.remove_nodes_from(isolates)

        return DCM

    def _check_compute_variables(self, **kwargs):
        """Recursevely check if the requested control variables are instantiated/computed.
        Otherwise computes them in order.
        """
        if 'sg' in kwargs:
            if self._sg is None:
                self._sg = self.structural_graph()

        elif 'eg' in kwargs:
            if self._eg is None:
                self._eg = self.effective_graph()

        elif 'stg' in kwargs:
            if self._stg is None:
                self._check_compute_variables(sg=True)
                self._stg = self.state_transition_graph()

        elif 'attractors' in kwargs:
            if self._attractors is None:
                self._check_compute_variables(stg=True)
                self._attractors = self.attractors()

        elif 'stg_r' in kwargs:
            if self._stg_r is None:
                self._check_compute_variables(stg=True)
                self._stg_r = self.state_transition_graph_reachability()
        else:
            raise Exception('Control variable name not found. %s' % kwargs)
        return True

    #
    # Get Node Names from Ids
    #
    def _get_node_name(self, id):
        """Return the name of the node based on its id.

        Args:
            id (int): id of the node.

        Returns:
            name (string): name of the node.
        """
        try:
            node = self.nodes[id]
        except error:
            raise AttributeError("Node with id {id:d} does not exist. {error::s}".format(id=id, error=error))
        else:
            return node.name

    def get_node_name(self, iterable=[]):
        """Return node names. Optionally, it returns only the names of the ids requested.

        Args:
            iterable (int,list, optional) : The id (or list of ids) of nodes to which return their names.

        Returns:
            names (list) : The name of the nodes.
        """
        # If only one id is passed, make it a list
        if not isinstance(iterable, list):
            iterable = [iterable]
        # No ids requested, return all the names
        if not len(iterable):
            return [n.name for n in self.nodes]
        # otherwise, use the recursive map to change ids to names
        else:
            return recursive_map(self._get_node_name, iterable)

    def average_trajectory_length(self, nsamples=10, random_seed=None, method='random'):
        """The average length of trajectories from a random initial configuration to its attractor.

        Args:
            nsamples (int) : The number of samples per hammimg distance to get.
            random_seed (int) : The random state seed.
            method (string) : specify the method you want. either 'random' or ....

        Returns:
            trajlen (float) : The average trajectory length to an attractor.
        """
        return sum(len(self.trajectory_to_attractor(random_binstate(self.Nnodes))) for isample in range(nsamples)) / nsamples

    def derrida_curve(self, nsamples=10, max_hamm=None, random_seed=None, method='random'):
        """The Derrida Curve (also reffered as criticality measure :math:`D_s`).
        When "mode" is set as "random" (default), it would use random sampling to estimate Derrida value
        If "mode" is set as "sensitivity", it would use c-sensitivity to calculate Derrida value (slower)
        You can refer to :cite:'kadelka2017influence' about why c-sensitivity can be used to caculate Derrida value

        Args:
            nsamples (int) : The number of samples per hammimg distance to get.
            max_hamm (int) : The maximum Hamming distance between starting states. default: self.Nnodes
            random_seed (int) : The random state seed.
            method (string) : specify the method you want. either 'random' or 'sensitivity'

        Returns:
            (dx,dy) (tuple) : The dx and dy of the curve.
        """
        random.seed(random_seed)

        if max_hamm is None or (max_hamm > self.Nnodes):
            max_hamm = self.Nnodes

        dx = np.linspace(0, 1, max_hamm, endpoint=True)
        dy = np.zeros(max_hamm + 1)

        if method == 'random':
            # for each possible hamming distance between the starting states
            for hamm_dist in range(1, max_hamm + 1):

                # sample nsample times
                for isample in range(nsamples):
                    rnd_config = random_binstate(self.Nnodes)
                    perturbed_var = random.sample(range(self.Nnodes), hamm_dist)
                    perturbed_config = [flip_bit(rnd_config[ivar]) if ivar in perturbed_var else rnd_config[ivar] for ivar in range(self.Nnodes)]
                    dy[hamm_dist] += hamming_distance(self.step(rnd_config), self.step(perturbed_config)) / self.Nnodes  # normalized Hamming Distance

            dy /= nsamples

        elif method == 'sensitivity':

            for hamm_dist in range(1, max_hamm + 1):
                dy[hamm_dist] = sum([node.c_sensitivity(hamm_dist, mode='forceK', max_k=self.Nnodes) for node in self.nodes]) / self.Nnodes

        return dx, dy

    def derrida_coefficient(self, nsamples=10, random_seed=None, method='random'):
        """The Derrida Coefficient.
        When "mode" is set as "random" (default), it would use random sampling to estimate Derrida value
        If "mode" is set as "sensitivity", it would use c-sensitivity to calculate Derrida value (slower)
        You can refer to :cite:'kadelka2017influence' about why c-sensitivity can be used to caculate Derrida value

        Args:
            nsamples (int) : The number of samples per hammimg distance to get.
            random_seed (int) : The random state seed.
            method (string) : specify the method you want. either 'random' or 'sensitivity'

        Returns:
            (dx,dy) (tuple) : The dx and dy of the curve.
        """
        random.seed(random_seed)
        hamm_dist = 1

        if method == 'random':
            # for each possible hamming distance between the starting states

            dy = 0
            # sample nsample times
            for isample in range(nsamples):
                rnd_config = random_binstate(self.Nnodes)
                perturbed_var = random.sample(range(self.Nnodes), hamm_dist)
                perturbed_config = [flip_bit(rnd_config[ivar]) if ivar in perturbed_var else rnd_config[ivar] for ivar in range(self.Nnodes)]
                dy += hamming_distance(self.step(rnd_config), self.step(perturbed_config))

            dy /= float(nsamples)

        elif method == 'sensitivity':
            # raise NotImplementedError
            dy = sum([node.c_sensitivity(hamm_dist, mode='forceK', max_k=self.Nnodes) for node in self.nodes])

        return dy / float(self.Nnodes)
