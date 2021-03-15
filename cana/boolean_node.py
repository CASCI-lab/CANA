# -*- coding: utf-8 -*-
"""
Boolean Node
=============

Main class for Boolean node objects.

"""
#   Copyright (C) 2021 by
#   Rion Brattig Correia <rionbr@gmail.com>
#   Alex Gates <ajgates@gmail.com>
#   Etienne Nzabarushimana <enzabaru@indiana.edu>
#   All rights reserved.
#   MIT license.
from __future__ import division
import numpy as np
import pandas as pd
from statistics import mean
from itertools import compress, combinations
from cana.canalization import boolean_canalization as BCanalization
from cana.canalization import cboolean_canalization as cBCanalization
from cana.utils import *
from cana.cutils import *


class BooleanNode(object):
    """
    """

    def __init__(self, id=0, name='x', k=1, inputs=[1], state=False, outputs=[0, 1], constant=False, network=None, verbose=False, *args, **kwargs):
        self.id = id                            # the id of the node
        self.name = name                        # the name of the node
        self.k = k                              # k is the number of inputs
        self.inputs = list(map(int, inputs))    # the ids of the input nodes
        self.state = state                      # the initial state of the node
        self.outputs = list(map(str, outputs))  # the list of transition outputs
        self.network = network                  # the BooleanNetwork object this nodes belongs to
        self.verbose = verbose                  # verbose mode

        # mask for inputs
        if len(self.inputs) > 0:
            self.mask = [(i in self.inputs) for i in range(max(self.inputs) + 1)]
        else:
            self.mask = []

        # Consistency
        if (k != 0) and (k != int(np.log2(len(outputs)))):
            raise ValueError('Number of k (inputs) do not match the number of output transitions')

        # If all outputs are either positive or negative, this node can be treated as a constant.
        if (len(set(outputs)) == 1) or (constant):
            self.constant = True
            self.step = self.constant_step
        else:
            self.constant = False
            self.step = self.dynamic_step

        # Canalization Variables
        self._prime_implicants = None           # A tuple of negative and positive prime implicants.
        self._two_symbols = None                # The Two Symbol (TS) Schemata
        self._pi_coverage = None                # The Coverage of inputs by Prime Implicants schemata
        self._ts_coverage = None                # The Coverage of inputs by Two Symbol schemata

    def __str__(self):
        if len(self.outputs) > 10:
            outputs = '[' + ','.join(map(str, self.outputs[:4])) + '...' + ','.join(map(str, self.outputs[-4:])) + ']'
        else:
            outputs = '[' + ','.join(map(str, self.outputs)) + ']'
        inputs = '[' + ','.join(map(str, self.inputs)) + ']'
        return "<BNode(id={id:d}, name='{name:s}', k={k:d}, inputs={inputs:s}, state={state:d}, outputs={outputs:s} constant={constant:b})>".format(
            id=self.id, name=self.name, k=self.k, inputs=inputs, state=self.state, outputs=outputs, constant=self.constant)

    @classmethod
    def from_output_list(self, outputs=list(), *args, **kwargs):
        """Instanciate a Boolean Node from a output transition list.

        Args:
            outputs (list) : The transition outputs of the node.

        Returns:
            (BooleanNode) : the instanciated object.

        Example:
            >>> BooleanNode.from_output_list(outputs=[0,0,0,1], name="AND")
        """
        id = kwargs.pop('id') if 'id' in kwargs else 0
        name = kwargs.pop('name') if 'name' in kwargs else 'x'
        k = int(np.log2(len(outputs)))
        inputs = kwargs.pop('inputs') if 'inputs' in kwargs else [(x + 1) for x in range(k)]
        state = kwargs.pop('state') if 'state' in kwargs else False

        return BooleanNode(id=id, name=name, k=k, inputs=inputs, state=state, outputs=outputs, *args, **kwargs)

    def input_redundancy(self, operator=mean, norm=True):
        r"""The Input Redundancy :math:`k_{r}` is the mean number of unnecessary inputs (or ``#``) in the Prime Implicants Look Up Table (LUT).
        Since there may be more than one redescription schema for each input entry, the input redundancy is bounded by an upper and lower limit.


        .. math::

            k_{r}(x) = \frac{ \sum_{f_{\alpha} \in F} \Phi_{\theta:f_{\alpha} \in \Theta_{\theta}} (n^{\#}_{\theta} ) }{ |F| }

        where :math:`\Phi` is a function (:math:`min` or :math:`max`) and :math:`F` is the node LUT.

        Args:
            operator (function) : The operator to use while computing input redundancy for the node.
                Defaults to `statistics.mean`, but `min` or `max` can be also considered.
            norm (bool) : Normalized between [0,1].
                Use this value when comparing nodes with different input sizes. (Defaults to "True".)

                :math:`k^{*}_r(x) = \frac{ k_r(x) }{ k(x) }`.


        Returns:
            (float) : The :math:`k_r` value.

        Note:
            The complete mathematical description can be found in :cite:`Marques-Pita:2013`.

        See also:
            :func:`effective_connectivity`, :func:`input_symmetry`, :func:`edge_redundancy`.
        """
        # Canalization can only occur when k>= 2
        if self.k < 2:
            return 0.0

        self._check_compute_canalization_variables(pi_coverage=True)

        if not hasattr(operator, '__call__'):
            raise AttributeError('The operator you selected must be a function. Try "min", "statitics.mean", or "max".')

        redundancy = [operator([pi.count('#') for pi in self._pi_coverage[binstate]]) for binstate in self._pi_coverage]

        k_r = sum(redundancy) / 2**self.k

        if (norm):
            # Normalizes
            k_r = k_r / self.k

        return k_r

    def edge_redundancy(self, bound='mean'):
        r""" The Edge Redundancy :math:`r_{i}` is the mean number of unnecessary inputs (or ``#``) in the Prime Implicants Look Up Table (LUT) for that input.
        Since there may be more than one redescription schema for each input entry, the input redundancy is bounded by an upper and lower limit.

        .. math::

            r_i(x_i) = \frac{ \sum_{f_{\alpha} \in F} \Phi_{\theta:f_{\alpha} \in \Theta_{\theta}} (X^{\#}_{\theta_i} ) }{ |F| }

        where :math:`\Phi` is a function (:math:`min` or :math:`max`) and :math:`F` is the node LUT.

        Args:
            bound (string) : The bound to which compute input redundancy.
                Mode "input" accepts: ["lower", "mean", "ave", upper", "tuple"].
                Defaults to "mean".

        Returns:
            (list) : The list of :math:`r_i` for inputs.

        Note:
            The complete mathematical description can be found in :cite:Gates:2020`.

        See also:
            :func:`effective_connectivity`, :func:`input_symmetry`.
        """

        self._check_compute_canalization_variables(pi_coverage=True)

        redundancies = []
        # Generate a per input coverage
        # ex: {0: {'11': [], '10': [], '00': [], '01': []}, 1: {'11': [], '10': [], '00': [], '01': []}}
        # pi_edge_coverage = { input : { binstate: [ pi[input] for pi in pis ] for binstate,pis in self._pi_coverage.items() } for input in range(self.k) }
        pi_edge_coverage = cBCanalization.input_wildcard_coverage(self._pi_coverage)
        # Loop ever input node
        for edge, binstates2wildcard in pi_edge_coverage.items():
            # {'numstate': [matches], '10': [True,False,True,...] ...}

            # countslenghts = {binstate_to_statenum(binstate): ([pi=='#' for pi in pis]) for binstate,pis in binstates.items() }
            # A triplet of (min, mean, max) values
            if bound == 'lower':
                redundancy = sum([all(pi) for pi in binstates2wildcard.values()]) / 2**self.k  # min(r_i)
            elif bound == 'mean' or bound == 'avg':
                redundancy = sum([sum(pi) / len(pi) for pi in binstates2wildcard.values()]) / 2**self.k  # <r_i>
            elif bound == 'upper':
                redundancy = sum([any(pi) for pi in binstates2wildcard.values()]) / 2**self.k  # max(r_i)
            elif bound == 'tuple':
                redundancy = (sum([all(pi) for pi in binstates2wildcard.values()]) / 2**self.k, sum([any(pi) for pi in binstates2wildcard.values()]) / 2**self.k)  # (min,max)
            else:
                raise AttributeError('The bound you selected does not exist. Try "upper", "mean", "lower" or "tuple".')

            redundancies.append(redundancy)

        return redundancies  # r_i

    def effective_connectivity(self, operator=mean, norm=True):
        r"""The Effective Connectiviy is the mean number of input nodes needed to determine the transition of the node.

        .. math::

            k_e(x) = k(x) - k_r(x)

        Args:
            operator (function) : The operator to use while computing input redundancy for the node.
                Defaults to `statistics.mean`, but `min` or `max` can be also considered.
            norm (bool) : Normalized between [0,1].
                Use this value when comparing nodes with different input sizes. (Defaults to "True".)

                :math:`k^{*}_e(x) = \frac{ k_e(x) }{ k(x) }`.

        Returns:
            (float) : The :math:`k_e` value.

        See Also:
            :func:`input_redundancy`, :func:`input_symmetry`, :func:`~cana.boolean_network.BooleanNetwork.effective_graph`.
        """
        #
        # Canalization can only occur when k>= 2
        if self.k < 2:
            return 0.0

        k_r = self.input_redundancy(operator=operator, norm=False)
        #
        k_e = self.k - k_r
        if (norm):
            k_e = k_e / self.k
        return k_e

    def edge_effectiveness(self, bound='mean'):
        r"""The Edge Effectiveness is the mean number of an input's states needed to determine the transition of the node.

        .. math::

            e_i(x_i) = 1 - r_i(x_i)

        Args:
            bound (string) : The bound for the :math:`k_r` Input Redundancy

        Returns:
            (list) : The list of :math:`e_r` values.

        See Also:
            :func:`input_redundancy`, :func:`input_symmetry`, :func:`~cana.boolean_network.BooleanNetwork.effective_graph`.
        """
        e_i = [1.0 - x_i for x_i in self.edge_redundancy(bound=bound)]
        return e_i

    def edge_symmetry(self, bound='upper'):
        r"""Edge Symmetry is a measure of permutation redundancy of a single input.
        Similar to the computation of Edge Effectiveness but using the Two-Symbol instead of the Prime Implicant schemata.

        .. math::

            s_i = \frac{ \sum_{f_{\alpha} \in F} \Phi_{\theta:f_{\alpha} \in \Theta_{\theta}} (n^{\circ}_i)}{ |F| }

        where :math:`\Phi` is the function :math:`min` or :math:`max` and :math:`F` is the node LUT.

        Args:
            bound (string) : The bound to which compute input symmetry.
                Mode "node" accepts: ["lower", "upper"].
                Mode "input" accepts: ["lower", "mean", "upper", "tuple"].
                Defaults to "upper".

        Returns:
            (float/list) : The :math:`k_s` or a list of :math:`r_i`.

        See also:
            :func:`input_redundancy`, :func:`effective_connectivity`
        """
        # Canalization can only occur when k>= 2
        if self.k < 2:
            return [0.0]

        self._check_compute_canalization_variables(ts_coverage=True)

        symmetries = []
        # Generate a per input coverage
        # ex: {0: {'11': [], '10': [], '00': [], '01': []}, 1: {'11': [], '10': [], '00': [], '01': []}}
        # ts_input_coverage = { input : { binstate: [ idxs.count(input) for schema,reps,sms in tss for idxs in reps+sms ] for binstate,tss in self._ts_coverage.items() } for input in range(self.k) }
        ts_input_coverage = {input: {binstate: [len(idxs) if input in idxs else 0 for schema, reps, sms in tss for idxs in reps + sms] for binstate, tss in self._ts_coverage.items()} for input in range(self.k)}

        # Loop ever input node
        for input, binstates in ts_input_coverage.items():
            # {'numstate': [number-of-ts's for each match], '10': [0, 2] ...}
            numstates = {binstate_to_statenum(binstate): permuts for binstate, permuts in binstates.items()}

            # A triplet of (min, mean, max) values
            if bound in ['lower', 'mean', 'upper']:
                # Min, Mean or Max
                if bound == 'upper':
                    minmax = max
                elif bound == 'mean':
                    minmax = np.mean
                elif bound == 'lower':
                    minmax = min

                s_i = sum(minmax(permuts) if len(permuts) else 0 for permuts in numstates.values()) / 2**self.k  # min(r_s)

            elif bound == 'tuple':
                # tuple (min,max) per input, per state
                s_i = [(min(permuts), max(permuts)) if len(permuts) else (0, 0) for permuts in numstates.values()]  # (min,max)
            else:
                raise AttributeError('The bound you selected does not exist. Try "upper", "mean", "lower" or "tuple".')
            symmetries.append(s_i)

        return symmetries  # s_i

    def input_symmetry(self, bound='upper', norm=True):
        r"""The Input Symmetry is a measure of permutation redundancy.
        Similar to the computation of Input Redundancy but using the Two-Symbol instead of the Prime Implicant schemata.

        .. math::

            k_s = \frac{ \sum_{f_{\alpha} \in F} \Phi_{\theta:f_{\alpha} \in \Theta_{\theta}} (n^{\circ}) }{ |F| }

        where :math:`\Phi` is the function :math:`min` or :math:`max` and :math:`F` is the node LUT.

        Args:
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
        # Canalization can only occur when k>= 2
        if self.k < 2:
            return 0.0

        self._check_compute_canalization_variables(ts_coverage=True)

        k_s = sum(self.edge_symmetry(bound=bound)) / self.k

        if (norm):
            k_s = k_s / self.k

        return k_s

    def look_up_table(self):
        """ Returns the Look Up Table (LUT)

        Returns:
            (pandas.DataFrame): the LUT

        Examples:
            >>> AND = BooleanNode.from_output_list([0,0,0,1])
            >>> AND.look_up_table()

        See also:
            :func:`schemata_look_up_table`

        """
        d = []
        if ((self.k < 2) and (self.constant is False)):
            k = 2
        else:
            k = self.k

        for statenum, output in zip(range(2**k), self.outputs):
            # Binary State, Transition
            d.append((statenum_to_binstate(statenum, base=self.k), output))

        df = pd.DataFrame(d, columns=['In:', 'Out:'])

        return df

    def schemata_look_up_table(self, type='pi', pi_symbol=u'#', ts_symbol_unicode=u"\u030A", ts_symbol_latex=u"\circ", format='pandas'):
        """ Returns the simplified schemata Look Up Table (LUT)

        Args:
            type (string) : The type of schemata to return, either Prime Implicants ``pi`` or Two-Symbol ``ts``. Defaults to 'pi'.
            pi_symbol (unicode) : The Prime Implicant don't care symbol. Default is ``#``.
            ts_symbol_unicode (unicode) : A unicode string for the Two Symbol permutable symbol. Default is ``u"\u030A"``.
            ts_symbol_latex (unicode) : The latex string for Two Symbol permutable symbol. Default is ``\circ``.
            format (string) : The format to return. Possible values are ``pandas`` (default) and ``latex``.

        Returns:
            (pandas.DataFrame or Latex): the schemata LUT

        Examples:
            >>> AND = BooleanNode.from_output_list([0,0,0,1])
            >>> AND.schemata_look_up_table(type='pi')

        Note:
            See the full list of `combining characters <https://en.wikipedia.org/wiki/Combining_character>`_ to use other symbols as the permutation symbol.

        See also:
            :func:`look_up_table`
        """
        r = []
        # Prime Implicant LUT
        if type == 'pi':
            self._check_compute_canalization_variables(prime_implicants=True)

            pi0s = self._prime_implicants.get('0', [])
            pi1s = self._prime_implicants.get('1', [])

            for output, pi in zip([0, 1], [pi0s, pi1s]):
                for schemata in pi:
                    r.append((schemata, output))

        # Two Symbol LUT
        elif type == 'ts':
            self._check_compute_canalization_variables(two_symbols=True)

            ts0s, ts1s = self._two_symbols

            for output, ts in zip([0, 1], [ts0s, ts1s]):
                for i, (schemata, permutables, samesymbols) in enumerate(ts):
                    string = u''
                    if len(permutables):
                        # Permutable
                        for j, permutable in enumerate(permutables):

                            if format == 'latex':
                                if j > 0:
                                    string += u' \,  | \, '
                                string += r' '.join([x if (k not in permutable) else '\overset{%s}{%s}' % (ts_symbol_latex, unicode(x)) for k,x in enumerate(schemata, start=0)])
                            else:
                                if j > 0:
                                    string += u' | '
                                string += u''.join([x if (k not in permutable) else u'%s%s' % (x, ts_symbol_unicode) for k, x in enumerate(schemata, start=0)])

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
                    r.append((string, output))
        else:
            raise AttributeError('The schemata type could not be found. Try "PI" (Prime Implicants) or "TS" (Two-Symbol).')

        # Output Format (Latex Table or Pandas DataFrame)
        if format == 'latex':
            out = r"\begin{array}{ | c | r | l }" + "\n"
            out += r"\hline" + "\n"
            out += r" & In: & Out: \\" + "\n"
            out += r"\hline" + "\n"
            for i, (string, output) in enumerate(r):
                string = string.replace('2','\%s' % (pi_symbol))
                out += r"%d & %s & %s \\" % (i, string, output) + r"\hline" + "\n"
            out += r"\hline" + "\n"
            out += r"\end{array}" + "\n"
            return out

        elif format == 'pandas':

            r = [(schemata.replace('2', pi_symbol), output) for schemata, output in r]
            return pd.DataFrame(r, columns=['In:', 'Out:'])

        else:
            AttributeError('The format type could not be found. Try "pandas" "latex".')

    def input_mask(self, binstate):
        """ Returns the mask applied to the binary state binstate

        Args:
            binstate (str) : the binary state

        Returns:
            output (str) : the masked state
        """
        return ''.join(compress(binstate, self.mask))

    def constant_step(self, input_state):
        """
            Treat the node as a constant
        """
        return self.outputs[0]

    def dynamic_step(self, input_state):
        """ Returns the output of the node based on a specific input

        Args:
            input (list) : an input to the node.

        Returns:
            output (bool) : the output value.
        """
        return self.outputs[binstate_to_statenum(input_state)]

    def activities(self):
        return self.effective_connectivity(mode='input', bound='upper')

    def canalizing_map(self, output=None):
        """ Computes the node Canalizing Map (CM).

        Args:
            output (int) : The output CM to return. Default is ``None``, retuning both [0,1].

        Returns:
            CM (networkx.DiGraph) : a directed graph representation of the CM.

        See Also:
            :func:`boolean_network.dynamics_canalization_map` for the DCM and :func:`drawing.draw_canalizing_map_graphviz` for plotting.
        """
        self._check_compute_canalization_variables(two_symbols=True)

        ts0s, ts1s = self._two_symbols

        G = nx.DiGraph(name='CM: %s' % self.name)

        # Outputs
        if output is None or output == 0:
            G.add_node('var-{nid:d}-out-0'.format(nid=self.id), **{'label': self.name, 'type': 'variable', 'mode': 'output', 'value': 0, 'constant': self.constant, 'group': self.id})

        if output is None or output == 1:
            G.add_node('var-{nid:d}-out-1'.format(nid=self.id), **{'label': self.name, 'type': 'variable', 'mode': 'output', 'value': 1, 'constant': self.constant, 'group': self.id})

        tid = 0
        for out, tspsss in zip([0, 1], self._two_symbols):

            # Only return the requested output
            if ((not len(tspsss)) or ((output is not None) and (output != out))):
                continue

            for ts, ps, ss in tspsss:

                lits = []
                group0 = []
                group1 = []
                # group2 = []
                # Tau is the threshold, counted as the sum of (0's and 1's literals; 0's in permutation group; 1's in permutation group)
                nlit, ngrp0, ngrp1 = 0, 0, 0

                for j in range(self.k):
                    # Is this input in any permutation group?
                    input = ts[j]
                    if not any([j in group for group in ps]):
                        if ts[j] in ['0', '1']:
                            nlit += 1
                            source = j
                            lits.append(source)
                    else:
                        if ts[j] == '0':
                            ngrp0 += 1
                            group0.append(j)
                        elif ts[j] == '1':
                            ngrp1 += 1
                            group1.append(j)

                tau = nlit + ngrp0 + ngrp1

                # Threshold Node
                tname = 'thr-{tid:d}-var-{nid:d}-out-{out:d}'.format(tid=tid, nid=self.id, out=out)
                label = "{:d}".format(tau)
                G.add_node(tname, **{'label': label, 'type': 'threshold', 'tau': tau, 'group': self.id})

                # Add Edges from Threshold node to output
                xname = 'var-{nid:d}-out-{out:d}'.format(nid=self.id, out=out)
                if G.has_node(tname) and G.has_node(xname):
                    G.add_edge(tname, xname, **{'type': 'out'})
                else:
                    raise BaseException("Adding edge to node that does not exist.")

                # Literal Edges
                for lit in lits:
                    iname = 'var-{nid:d}-out-{out:d}'.format(nid=self.inputs[lit], out=int(ts[lit]))
                    if iname not in G.nodes():
                        # Can we get the name of ther nodes? Are we attached to a network?
                        if self.network is not None:
                            ilabel = self.network.get_node_name(self.inputs[lit])[0]
                        else:
                            ilabel = str(self.inputs[lit])
                        iout = int(ts[lit])
                        G.add_node(iname, **{'label': ilabel, 'type': 'variable', 'mode': 'input', 'value': iout, 'group': self.id})
                    G.add_edge(iname, tname, **{'type': 'literal'})

                # Group0
                for fusion in range(ngrp0):
                    fname = 'fus-{fus:d}-thr-{thr:d}-var-{nid:d}-out-{out:d}'.format(fus=fusion, thr=tid, nid=self.id, out=0)
                    G.add_node(fname, **{'type': 'fusion', 'group': self.id})
                    for input in ps[0]:
                        iname = 'var-{nid:d}-out-{out:d}'.format(nid=self.inputs[input], out=0)
                        if iname not in G.nodes():
                            # Can we get the name of ther nodes? Are we attached to a network?
                            if self.network is not None:
                                ilabel = self.network.get_node_name(self.inputs[input])[0]
                            else:
                                ilabel = str(self.inputs[input])
                            G.add_node(iname, **{'label': ilabel, 'type': 'variable', 'mode': 'input', 'value': 0, 'group': self.id})
                        G.add_edge(iname, fname, **{'type': 'fusing'})
                    G.add_edge(fname, tname, **{'type': 'fused'})

                # Group1
                for fusion in range(ngrp1):
                    fname = 'fus-{fus:d}-thr-{thr:d}-var-{nid:d}-out-{out:d}'.format(fus=fusion, thr=tid, nid=self.id, out=1)
                    G.add_node(fname, **{'type': 'fusion', 'group': self.id})
                    for input in ps[0]:
                        iname = 'var-{var:d}-out-{out:d}'.format(var=self.inputs[input], out=1)
                        if iname not in G.nodes():
                            # Can we get the name of ther nodes? Are we attached to a network?
                            if self.network is not None:
                                ilabel = self.network.get_node_name(self.inputs[input])[0]
                            else:
                                ilabel = str(self.inputs[input])
                            iout = ts[input]
                            G.add_node(iname, **{'label': ilabel, 'type': 'variable', 'mode': 'input', 'value': 1, 'group': self.id})
                        G.add_edge(iname, fname, **{'type': 'fusing'})
                    G.add_edge(fname, tname, **{'type': 'fused'})

                tid += 1

        return G

    def pi_coverage(self):
        """ Returns the :math:`F'` (Prime Implicants) binary state coverage.

        Returns:
            (list)

        See also:
            :func:`ts_coverage`
        """
        self._check_compute_canalization_variables(pi_coverage=True)
        return self._pi_coverage

    def ts_coverage(self):
        """ Returns the :math:`F''` (Two-Symbol schematas) binary state coverage.

        Returns:
            (list)

        See also:
            :func:`pi_coverage`
        """
        self._check_compute_canalization_variables(ts_coverage=True)
        return self._ts_coverage

    def _check_compute_canalization_variables(self, **kwargs):
        """ Recursevely check if the requested canalization variables are instantiated/computed, otherwise computes them in order.
        For example: to compute `two_symbols` we need `prime_implicants` first.
        Likewise, to compute `prime_implicants` we need the `transition_density_table` first.
        """
        if 'prime_implicants' in kwargs:
            if self._prime_implicants is None:
                self._prime_implicants = dict()
                for output in set(self.outputs):
                    output_binstates = outputs_to_binstates_of_given_type(self.outputs, output=output, k=self.k)
                    self._prime_implicants[output] = cBCanalization.find_implicants_qm(input_binstates=output_binstates)

        elif 'pi_coverage' in kwargs:
            self._check_compute_canalization_variables(prime_implicants=True)
            if self._pi_coverage is None:
                self._pi_coverage = dict()
                for output, piset in self._prime_implicants.items():
                    self._pi_coverage.update(cBCanalization.return_pi_coverage(piset))

                # make sure every inputstate was covered by at least one prime implicant
                assert len(self._pi_coverage) == 2**self.k

        elif 'two_symbols' in kwargs:
            self._check_compute_canalization_variables(prime_implicants=True)
            if self._two_symbols is None:
                # this is a temporary fix until we update 'find_two_symbols' to cython.
                if '0' in self._prime_implicants:
                    pi0 = self._prime_implicants['0']
                    pi0 = set([pi.replace('#', '2') for pi in pi0])
                else:
                    pi0 = []
                if '1' in self._prime_implicants:
                    pi1 = self._prime_implicants['1']
                    pi1 = set([pi.replace('#', '2') for pi in pi1])
                else:
                    pi1 = []
                # /end fix
                self._two_symbols = \
                    (
                        BCanalization.find_two_symbols_v2(k=self.k, prime_implicants=pi0),
                        BCanalization.find_two_symbols_v2(k=self.k, prime_implicants=pi1)
                    )
        elif 'ts_coverage' in kwargs:
            self._check_compute_canalization_variables(two_symbols=True)
            if self._ts_coverage is None:
                self._ts_coverage = BCanalization.computes_ts_coverage(self.k, self.outputs, self._two_symbols)

        else:
            raise Exception('Canalization variable name not found. %s' % kwargs)
        return True

    def bias(self):
        r""" The node bias. The sum of the boolean output transitions divided by the number of entries (:math:`2^k`) in the LUT.

        .. math::

            bias(x) = \frac{ \sum_{f_{\alpha}\in F} s_{\alpha} }{ |F| }

        Returns:
            (float)

        See Also:
            :func:`~cana.boolean_network.BooleanNetwork.network_bias`
        """
        return sum(map(int, self.outputs)) / 2**self.k

    def c_sensitivity(self, c, mode="default", max_k=0):
        """ Node c-sensitivity.
        c-sensitivity is defined as: the mean probability that changing exactly
        ``c`` variables in input variables would change output value.
        There is another mode "forceK", which will be used to calculate Derrida value.
        In that mode, it would assume the number of input variables is specified as max_k
        this methods is equivalent to Derrida value in :cite:`Kadelka:2017`, only move a normalization
        coefficient from expression of Derrida value to c-sensitivity to simplify it

        Args:
            c (int) : the ``c`` in the definition of c-senstivity above
            mode (string) : either "default" or "forceK"
            max_k (int) : you must specify max_k when you set mode as 'forceK'

        Returns:
            (float)

        See Also:
            :func:`~cana.boolean_network.derrida_curve`
        """
        S_c_f = 0
        ic = min(c, self.k)

        if mode != 'forceK':
            for j in product('01', repeat=self.k):
                origin_config = list(j)
                for mut in combinations(range(self.k), ic):
                    mut_config = origin_config[:]
                    for i_mut in mut:
                        mut_config[i_mut] = flip_bit(mut_config[i_mut])
                    if self.step(''.join(origin_config)) != self.step(''.join(mut_config)):
                        S_c_f += 1
            if S_c_f == 0:
                return 0.
            return S_c_f / float(ncr(self.k, c)) / float(2 ** self.k)
        else:
            assert max_k >= self.k
            for ic in range(max(1, c + self.k - max_k), min(c, self.k) + 1):
                for j in product('01', repeat=self.k):
                    origin_config = list(j)
                    for mut in combinations(range(self.k), ic):
                        mut_config = origin_config[:]
                        for i_mut in mut:
                            mut_config[i_mut] = flip_bit(mut_config[i_mut])
                        if self.step(''.join(origin_config)) != self.step(
                                ''.join(mut_config)):
                            S_c_f += ncr(max_k - self.k, c - ic)
            return S_c_f / float(ncr(max_k, c)) / float(2 ** self.k)
