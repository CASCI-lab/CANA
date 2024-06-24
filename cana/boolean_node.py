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

from itertools import combinations, compress, islice, product, permutations
from statistics import mean

import networkx as nx
import numpy as np
import pandas as pd

import cana.canalization.boolean_canalization as BCanalization
import cana.canalization.cboolean_canalization as cBCanalization
from cana.cutils import (
    binstate_to_statenum,
    flip_bit,
    outputs_to_binstates_of_given_type,
    statenum_to_binstate,
)
from cana.utils import input_monotone, ncr, fill_out_lut
import random
import warnings
from math import comb


class BooleanNode(object):
    """ """

    def __init__(
        self,
        id=0,
        name="x",
        k=1,
        inputs=[1],
        state=False,
        outputs=[0, 1],
        constant=False,
        network=None,
        verbose=False,
        *args,
        **kwargs,
    ):
        self.id = id  # the id of the node
        self.name = name  # the name of the node
        self.k = k  # k is the number of inputs
        self.inputs = list(map(int, inputs))  # the ids of the input nodes
        self.state = str(
            int(state)
        )  # the initial state of the node. internally this is treated as a string.
        self.outputs = list(map(str, outputs))  # the list of transition outputs
        self.network = network  # the BooleanNetwork object this nodes belongs to
        self.verbose = verbose  # verbose mode

        # mask for inputs
        if len(self.inputs) > 0:
            self.mask = [(i in self.inputs) for i in range(max(self.inputs) + 1)]
        else:
            self.mask = []

        # Consistency
        if (k != 0) and (k != int(np.log2(len(outputs)))):
            raise ValueError(
                "Number of k (inputs) do not match the number of output transitions"
            )

        # If all outputs are either positive or negative, the node is treated as a constant.
        if (len(set(outputs)) == 1 and ("?" not in outputs)) or (constant):
            self.set_constant(constant=True, state=outputs[0])
        else:
            self.set_constant(constant=False)

        # Canalization Variables
        self._prime_implicants = (
            None  # A tuple of negative and positive prime implicants.
        )
        self._two_symbols = None  # The Two Symbol (TS) Schemata
        self._pi_coverage = None  # The Coverage of inputs by Prime Implicants schemata
        self._ts_coverage = None  # The Coverage of inputs by Two Symbol schemata

    def __str__(self):
        if len(self.outputs) > 10:
            outputs = (
                "["
                + ",".join(map(str, self.outputs[:4]))
                + "..."
                + ",".join(map(str, self.outputs[-4:]))
                + "]"
            )
        else:
            outputs = "[" + ",".join(map(str, self.outputs)) + "]"
        inputs = "[" + ",".join(map(str, self.inputs)) + "]"
        return "<BNode(id={id}, name='{name}', k={k}, inputs={inputs:s}, state='{state}', outputs={outputs} constant={constant})>".format(
            id=self.id,
            name=self.name,
            k=self.k,
            inputs=inputs,
            state=self.state,
            outputs=outputs,
            constant=self.constant,
        )

    @classmethod
    def from_output_list(self, outputs=list(), *args, **kwargs):
        """
        Instanciate a Boolean Node from a output transition list.

        Args:
            outputs (list) : The transition outputs of the node.

        Returns:
            (BooleanNode) : the instanciated object.

        Example:
            >>> BooleanNode.from_output_list(outputs=[0,0,0,1], name="EG")
        """
        id = kwargs.pop("id") if "id" in kwargs else 0
        name = kwargs.pop("name") if "name" in kwargs else "x"
        k = int(np.log2(len(outputs)))

        # checking if length of outputs is a power of 2, else raising an error.
        if 2**k != len(outputs):
            raise ValueError(
                "The number of outputs should be a power of 2. The length of the outputs list should be 2^k."
            )

        inputs = (
            kwargs.pop("inputs") if "inputs" in kwargs else [(x + 1) for x in range(k)]
        )
        state = kwargs.pop("state") if "state" in kwargs else False

        return BooleanNode(
            id=id,
            name=name,
            k=k,
            inputs=inputs,
            state=state,
            outputs=outputs,
            *args,
            **kwargs,
        )

    def set_constant(self, constant=True, state=None):
        """Sets whether the node is to be treated as a contant

        Args:
            constant (Boolean) : Whether to set or unset the node as a constant.
            state (str; optional) : The state value to which to set the node. Either '0' or '1'; default to current state value.
        """
        if constant:
            self.constant = True
            self.step = self.constant_step
        else:
            self.constant = False
            self.step = self.dynamic_step
        # Set an optional value to the node
        if state is not None:
            self.state = str(int(state))

    def constant_step(self, input_state):
        """Treat the node as a constant variable, always returning its state."""
        return self.state

    def dynamic_step(self, input_state):
        """Returns the output of the node based on a specific input.

        Args:
            input (str) : an input to the node (e.g.: '0111' -> 7).

        Returns:
            output (bool) : the output value.
        """
        self.state = self.outputs[binstate_to_statenum(input_state)]
        return self.state

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

        self._check_compute_canalization_variables(pi_coverage=True)

        if not hasattr(operator, "__call__"):
            raise AttributeError(
                'The operator you selected must be a function. Try "min", "statitics.mean", or "max".'
            )

        redundancy = [
            operator([pi.count("#") for pi in self._pi_coverage[binstate]])
            for binstate in self._pi_coverage
        ]

        k_r = sum(redundancy) / 2**self.k

        if norm:
            # Normalizes
            k_r = k_r / self.k

        return k_r

    def edge_redundancy(self, bound="mean"):
        r"""The Edge Redundancy :math:`r_{i}` is the mean number of unnecessary inputs (or ``#``) in the Prime Implicants Look Up Table (LUT) for that input.
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
            if bound == "lower":
                redundancy = (
                    sum([all(pi) for pi in binstates2wildcard.values()]) / 2**self.k
                )  # min(r_i)
            elif bound == "mean" or bound == "avg":
                redundancy = (
                    sum([sum(pi) / len(pi) for pi in binstates2wildcard.values()])
                    / 2**self.k
                )  # <r_i>
            elif bound == "upper":
                redundancy = (
                    sum([any(pi) for pi in binstates2wildcard.values()]) / 2**self.k
                )  # max(r_i)
            elif bound == "tuple":
                redundancy = (
                    sum([all(pi) for pi in binstates2wildcard.values()]) / 2**self.k,
                    sum([any(pi) for pi in binstates2wildcard.values()]) / 2**self.k,
                )  # (min,max)
            else:
                raise AttributeError(
                    'The bound you selected does not exist. Try "upper", "mean", "lower" or "tuple".'
                )

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
        k_r = self.input_redundancy(operator=operator, norm=False)
        #
        k_e = self.k - k_r
        if norm:
            k_e = k_e / self.k
        return k_e

    def edge_effectiveness(self, bound="mean"):
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

    def symKernel_numDots(self, ts, sameSymbol=False):
        """compute the number of variables involved in a symmetry group.

        Args:
            ts ([str, [[]], [[]]]) : two symbol schema
            sameSymbol (bool) : whether or not to consider same-symbol symmetry

        Returns:
            (int)
        """
        if not sameSymbol:
            groups = ts[1]
        else:
            groups = ts[1] + ts[2]
        return len(set(i for group in groups for i in group))

    def _input_symmetry(self, aggOp, kernel):
        self._check_compute_canalization_variables(ts_coverage=True)

        summand = []
        for fAlpha, fTheta in self._ts_coverage.items():
            summand.append(aggOp(list(map(kernel, fTheta))))
        return np.mean(summand)

    def input_symmetry(self, aggOp="mean", kernel="numDots", sameSymbol=False):
        """compute the input symmetry (k_s) of the boolean node, with variations via the specified functions.
        Convenience wrapper for lower-level function.

        Args:
            aggOp : the function aggregating over all two-symbol schemata that redescribe a LUT entry
            kernel : the function to compute on a given two-symbol schema

        Returns:
            (float) : the mean over all LUT entries of the aggOp applied to the kernel of all two-symbol schemata that redescribe the LUT entry
        """
        strToOp = {"mean": np.mean, "max": max, "min": min}
        strToKern = {"numDots": self.symKernel_numDots}

        def kernFunc(x):
            return strToKern[kernel](x, sameSymbol=sameSymbol)

        return self._input_symmetry(strToOp[aggOp], kernFunc)

    # refactor ks for speed, avg op only
    def input_symmetry_mean(self):
        """compute the input symmetry (k_s) of the boolean node.
           Specifically, computes it using the avg operator for the summand.
           Refactoring of input_symmetry for speed.

        Returns:
            (float)
        """
        summand = 0
        # fTheta = a list of TS
        for fTheta in self._ts_coverage.values():
            inner = 0
            for ts in fTheta:
                inner += sum(
                    len(i) for i in ts[1]
                )  # assumes that indicies will ever only be in at most 1 group
            summand += inner / len(fTheta)
        return summand / 2**self.k

    def look_up_table(self):
        """Returns the Look Up Table (LUT)

        Returns:
            (pandas.DataFrame): the LUT

        Examples:
            >>> AND = BooleanNode.from_output_list([0,0,0,1])
            >>> AND.look_up_table()

        See also:
            :func:`schemata_look_up_table`

        """
        d = []
        if (self.k < 2) and (self.constant is False):
            k = 2
        else:
            k = self.k

        for statenum, output in zip(range(2**k), self.outputs):
            # Binary State, Transition
            d.append((statenum_to_binstate(statenum, base=self.k), output))

        df = pd.DataFrame(d, columns=["In:", "Out:"])

        return df

    def schemata_look_up_table(
        self, type="pi", pi_symbol="#", ts_symbol_list=["\u030a", "\u032f"]
    ):
        """Returns the simplified schemata Look Up Table (LUT)

        Args:
            type (string) : The type of schemata to return, either Prime Implicants ``pi`` or Two-Symbol ``ts``. Defaults to 'pi'.
            pi_symbol (str) : The Prime Implicant don't care symbol. Default is ``#``.
            ts_symbol_list (list) : A list containing Two Symbol permutable symbols. Default is ``["\u030a", "\u032f"]``.

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
        # Check if the outputs contain '?' and generate an error message if it does.
        if "?" in self.outputs:
            raise ValueError(
                "The look-up table contains '?' values. The schemata look-up table cannot be generated."
            )

        r = []
        # Prime Implicant LUT
        if type == "pi":
            self._check_compute_canalization_variables(prime_implicants=True)

            pi0s = self._prime_implicants.get("0", [])
            pi1s = self._prime_implicants.get("1", [])

            for output, pi in zip([0, 1], [pi0s, pi1s]):
                for schemata in pi:
                    r.append((schemata, output))

        # Two Symbol LUT
        elif type == "ts":
            self._check_compute_canalization_variables(two_symbols=True)

            ts0s, ts1s = self._two_symbols

            for output, ts in zip([0, 1], [ts0s, ts1s]):
                for i, (schemata, permutables, samesymbols) in enumerate(ts):
                    string = ""
                    if len(permutables):
                        string += "("
                        # Permutable
                        for j, permutable in enumerate(permutables):
                            #
                            ts_symbol_unicode = ts_symbol_list[j]
                            if j > 0:
                                string += " | "
                            string += "".join(
                                [
                                    x
                                    if (k not in permutable)
                                    else "%s%s" % (x, ts_symbol_unicode)
                                    for k, x in enumerate(schemata, start=0)
                                ]
                            )
                        string += ")"

                    else:
                        string += "(" + schemata + ")"

                    if len(samesymbols):
                        string += " | ("
                        # Same Symbol
                        for j, samesymbol in enumerate(samesymbols):
                            ts_symbol_unicode = ts_symbol_list[j]
                            if j > 0:
                                string += " | "
                            string += "".join(
                                [
                                    x
                                    if (k not in samesymbol)
                                    else "%s%s" % (x, ts_symbol_unicode)
                                    for k, x in enumerate(schemata, start=0)
                                ]
                            )
                        string += ")"
                    r.append((string, output))
        else:
            raise AttributeError(
                'The schemata type could not be found. Try "PI" (Prime Implicants) or "TS" (Two-Symbol).'
            )

        r = [(schemata.replace("2", pi_symbol), output) for schemata, output in r]
        return pd.DataFrame(r, columns=["Input", "Output"])

    def input_mask(self, binstate):
        """Returns the mask applied to the binary state binstate

        Args:
            binstate (str) : the binary state

        Returns:
            output (str) : the masked state
        """
        return "".join(compress(binstate, self.mask))

    def activities(self):
        """compute the activities of each incoming edge of the node
        Returns:
            (list of floats)
        """
        return self.edge_effectiveness(bound="upper")

    def sensitivity(self, norm=False):
        """compute the sensitivity of the node. Does so by summing the activities of the edges
        Args:
            norm (bool) : whether or not to normalize by the number of inputs (k)
        Returns:
            (float)
        """
        x = sum(self.activities())
        if norm:
            return x / self.k
        else:
            return x

    def canalizing_map(self, output=None):
        """Computes the node Canalizing Map (CM).

        Args:
            output (int) : The output CM to return. Default is ``None``, retuning both [0,1].

        Returns:
            CM (networkx.DiGraph) : a directed graph representation of the CM.

        See Also:
            :func:`boolean_network.dynamics_canalization_map` for the DCM and :func:`drawing.draw_canalizing_map_graphviz` for plotting.
        """
        self._check_compute_canalization_variables(two_symbols=True)

        ts0s, ts1s = self._two_symbols

        G = nx.DiGraph(name="CM: %s" % self.name)

        # Outputs
        if output is None or output == 0:
            G.add_node(
                "var-{nid:d}-out-0".format(nid=self.id),
                **{
                    "label": self.name,
                    "type": "variable",
                    "mode": "output",
                    "value": 0,
                    "constant": self.constant,
                    "group": self.id,
                },
            )

        if output is None or output == 1:
            G.add_node(
                "var-{nid:d}-out-1".format(nid=self.id),
                **{
                    "label": self.name,
                    "type": "variable",
                    "mode": "output",
                    "value": 1,
                    "constant": self.constant,
                    "group": self.id,
                },
            )

        tid = 0
        for out, tspsss in zip([0, 1], self._two_symbols):
            # Only return the requested output
            if (not len(tspsss)) or ((output is not None) and (output != out)):
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
                        if ts[j] in ["0", "1"]:
                            nlit += 1
                            source = j
                            lits.append(source)
                    else:
                        if ts[j] == "0":
                            ngrp0 += 1
                            group0.append(j)
                        elif ts[j] == "1":
                            ngrp1 += 1
                            group1.append(j)

                tau = nlit + ngrp0 + ngrp1

                # Threshold Node
                tname = "thr-{tid:d}-var-{nid:d}-out-{out:d}".format(
                    tid=tid, nid=self.id, out=out
                )
                label = "{:d}".format(tau)
                G.add_node(
                    tname,
                    **{
                        "label": label,
                        "type": "threshold",
                        "tau": tau,
                        "group": self.id,
                    },
                )

                # Add Edges from Threshold node to output
                xname = "var-{nid:d}-out-{out:d}".format(nid=self.id, out=out)
                if G.has_node(tname) and G.has_node(xname):
                    G.add_edge(tname, xname, **{"type": "out"})
                else:
                    raise BaseException("Adding edge to node that does not exist.")

                # Literal Edges
                for lit in lits:
                    iname = "var-{nid:d}-out-{out:d}".format(
                        nid=self.inputs[lit], out=int(ts[lit])
                    )
                    if iname not in G.nodes():
                        # Can we get the name of ther nodes? Are we attached to a network?
                        if self.network is not None:
                            ilabel = self.network.get_node_name(self.inputs[lit])[0]
                        else:
                            ilabel = str(self.inputs[lit])
                        iout = int(ts[lit])
                        G.add_node(
                            iname,
                            **{
                                "label": ilabel,
                                "type": "variable",
                                "mode": "input",
                                "value": iout,
                                "group": self.id,
                            },
                        )
                    G.add_edge(iname, tname, **{"type": "literal"})

                # Group0
                for fusion in range(ngrp0):
                    fname = "fus-{fus:d}-thr-{thr:d}-var-{nid:d}-out-{out:d}".format(
                        fus=fusion, thr=tid, nid=self.id, out=0
                    )
                    G.add_node(fname, **{"type": "fusion", "group": self.id})
                    for input in ps[0]:
                        iname = "var-{nid:d}-out-{out:d}".format(
                            nid=self.inputs[input], out=0
                        )
                        if iname not in G.nodes():
                            # Can we get the name of ther nodes? Are we attached to a network?
                            if self.network is not None:
                                ilabel = self.network.get_node_name(self.inputs[input])[
                                    0
                                ]
                            else:
                                ilabel = str(self.inputs[input])
                            G.add_node(
                                iname,
                                **{
                                    "label": ilabel,
                                    "type": "variable",
                                    "mode": "input",
                                    "value": 0,
                                    "group": self.id,
                                },
                            )
                        G.add_edge(iname, fname, **{"type": "fusing"})
                    G.add_edge(fname, tname, **{"type": "fused"})

                # Group1
                for fusion in range(ngrp1):
                    fname = "fus-{fus:d}-thr-{thr:d}-var-{nid:d}-out-{out:d}".format(
                        fus=fusion, thr=tid, nid=self.id, out=1
                    )
                    G.add_node(fname, **{"type": "fusion", "group": self.id})
                    for input in ps[0]:
                        iname = "var-{var:d}-out-{out:d}".format(
                            var=self.inputs[input], out=1
                        )
                        if iname not in G.nodes():
                            # Can we get the name of ther nodes? Are we attached to a network?
                            if self.network is not None:
                                ilabel = self.network.get_node_name(self.inputs[input])[
                                    0
                                ]
                            else:
                                ilabel = str(self.inputs[input])
                            iout = ts[input]
                            G.add_node(
                                iname,
                                **{
                                    "label": ilabel,
                                    "type": "variable",
                                    "mode": "input",
                                    "value": 1,
                                    "group": self.id,
                                },
                            )
                        G.add_edge(iname, fname, **{"type": "fusing"})
                    G.add_edge(fname, tname, **{"type": "fused"})

                tid += 1

        return G

    def pi_coverage(self):
        """Returns the :math:`F'` (Prime Implicants) binary state coverage.

        Returns:
            (list)

        See also:
            :func:`ts_coverage`
        """
        self._check_compute_canalization_variables(pi_coverage=True)
        return self._pi_coverage

    def ts_coverage(self):
        """Returns the :math:`F''` (Two-Symbol schematas) binary state coverage.

        Returns:
            (list)

        See also:
            :func:`pi_coverage`
        """
        self._check_compute_canalization_variables(ts_coverage=True)
        return self._ts_coverage

    def _check_compute_canalization_variables(self, **kwargs):
        """Recursevely check if the requested canalization variables are instantiated/computed, otherwise computes them in order.
        For example: to compute `two_symbols` we need `prime_implicants` first.
        Likewise, to compute `prime_implicants` we need the `transition_density_table` first.
        """
        if "prime_implicants" in kwargs:
            if self._prime_implicants is None:
                self._prime_implicants = dict()
                for output in set(self.outputs):
                    output_binstates = outputs_to_binstates_of_given_type(
                        self.outputs, output=output, k=self.k
                    )
                    self._prime_implicants[output] = cBCanalization.find_implicants_qm(
                        input_binstates=output_binstates
                    )

        elif "pi_coverage" in kwargs:
            self._check_compute_canalization_variables(prime_implicants=True)
            if self._pi_coverage is None:
                self._pi_coverage = dict()
                for output, piset in self._prime_implicants.items():
                    self._pi_coverage.update(cBCanalization.return_pi_coverage(piset))

                # make sure every inputstate was covered by at least one prime implicant
                assert len(self._pi_coverage) == 2**self.k

        elif "two_symbols" in kwargs:
            self._check_compute_canalization_variables(prime_implicants=True)
            if self._two_symbols is None:
                # this is a temporary fix until we update 'find_two_symbols' to cython.
                if "0" in self._prime_implicants:
                    pi0 = self._prime_implicants["0"]
                    pi0 = set([pi.replace("#", "2") for pi in pi0])
                else:
                    pi0 = []
                if "1" in self._prime_implicants:
                    pi1 = self._prime_implicants["1"]
                    pi1 = set([pi.replace("#", "2") for pi in pi1])
                else:
                    pi1 = []
                # /end fix
                self._two_symbols = (
                    BCanalization.find_two_symbols_v2(k=self.k, prime_implicants=pi0),
                    BCanalization.find_two_symbols_v2(k=self.k, prime_implicants=pi1),
                )
        elif "ts_coverage" in kwargs:
            self._check_compute_canalization_variables(two_symbols=True)
            if self._ts_coverage is None:
                self._ts_coverage = BCanalization.computes_ts_coverage(
                    self.k, self.outputs, self._two_symbols
                )

        else:
            raise Exception("Canalization variable name not found. %s" % kwargs)
        return True

    def bias(self, verbose=True):
        r"""The node bias. The sum of the boolean output transitions divided by the number of entries (:math:`2^k`) in the LUT.

        .. math::

            bias(x) = \frac{ \sum_{f_{\alpha}\in F} s_{\alpha} }{ |F| }

        Returns:
            (float)

        See Also:
            :func:`~cana.boolean_network.BooleanNetwork.network_bias`
        """
        if verbose:
            if "?" in self.outputs:
                print(
                    "Warning: There is a '?' value in the output. It will be treated as zero for the bias calculation."
                )

        outputs = [
            0 if output == "?" else output for output in self.outputs
        ]  # added this condition so that bias function plays nice with '?' output values. It will treat missing outputs as 0.

        bias = sum(map(int, outputs)) / 2**self.k

        return bias

    def c_sensitivity(self, c, mode="default", max_k=0):
        """Node c-sensitivity.
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

        if mode != "forceK":
            for j in product("01", repeat=self.k):
                origin_config = list(j)
                for mut in combinations(range(self.k), ic):
                    mut_config = origin_config[:]
                    for i_mut in mut:
                        mut_config[i_mut] = flip_bit(mut_config[i_mut])
                    if self.step("".join(origin_config)) != self.step(
                        "".join(mut_config)
                    ):
                        S_c_f += 1
            if S_c_f == 0:
                return 0.0
            return S_c_f / float(ncr(self.k, c)) / float(2**self.k)
        else:
            assert max_k >= self.k
            for ic in range(max(1, c + self.k - max_k), min(c, self.k) + 1):
                for j in product("01", repeat=self.k):
                    origin_config = list(j)
                    for mut in combinations(range(self.k), ic):
                        mut_config = origin_config[:]
                        for i_mut in mut:
                            mut_config[i_mut] = flip_bit(mut_config[i_mut])
                        if self.step("".join(origin_config)) != self.step(
                            "".join(mut_config)
                        ):
                            S_c_f += ncr(max_k - self.k, c - ic)
            return S_c_f / float(ncr(max_k, c)) / float(2**self.k)

    def input_signs(self):
        """
        Determine if a each input can be considered activation (1), inhibition (-1), or neither (0).

        Here we test every pair of inputs that are Hamming distance 1. (see Goldreich et al 2000)

        Returns:
            (list) : The list of input signs.

        Example:
            >>> is_monotone(outputs=[0,0,0,1])
        """

        # first test if the inputs are activation inputs (more common in bio networks)
        input_sign_list = [
            int(input_monotone(self.outputs, idx, activation=1))
            for idx in range(self.k)
        ]

        # for those inputs that are not activation inputs (more common in bio networks),
        # then test for inhibition
        input_sign_list = [
            -1 * int(input_monotone(self.outputs, idx, activation=-1))
            if sign == 0
            else sign
            for idx, sign in enumerate(input_sign_list)
        ]

        return input_sign_list

    def from_partial_lut(
        partial_lut,
        fill_missing_output_randomly=False,
        required_node_bias=None,
        required_effective_connectivity=None,
        verbose=False,
        *args,
        **kwargs,
    ):
        """
        Instantiate a Boolean Node from a partial look-up table.

        Uses the fill_out_lut function to complete the look-up table. Extracts the output list from the completed look-up table. Then instantiates the Boolean Node from the output list using the from_output_list method.

        Args:
            partial_lut (list) : A partial look-up table of the node.
            fill_missing_output_randomly (bool) : If True, missing output values are filled with random 0 or 1. If False, missing output values are filled with '?'.
            verbose (bool) : If True, print additional information.

        Returns:
            (BooleanNode) : the instantiated object.

        Example:
            >>> BooleanNode.from_partial_lut(partial_lut=[('00', 0), ('01', 1), ('11', 1)], verbose=True, name="EG")
            >>> BooleanNode.from_partial_lut(partial_lut=[('00', 0), ('01', 1), ('11', 1)], fill_missing_output_randomly=True, verbose=False, name="EG")


        Note:
            The partial look-up table should be a list of tuples where each tuple contains a binary input state and the corresponding output value. For example, [('00', 0), ('01', 1), ('11', 1)].
            The fill_missing_output_randomly should be a boolean value.

        # TODO : [SRI] add tests for this
        """

        generated_lut = fill_out_lut(partial_lut, verbose=False)
        output_list = [x[1] for x in generated_lut]

        generated_node = BooleanNode.from_output_list(output_list, *args, **kwargs)

        # Fill missing output values with the specified bias or with specified effective connectivity or randomly

        if fill_missing_output_randomly: # TODO : [SRI] should this also return a list like the others? 
            # Replace '?' in generated_node.outputs with 0 or 1 randomly
            generated_node.outputs = [
                random.choice(["0", "1"]) if output == "?" else output
                for output in generated_node.outputs
            ]

        if verbose and "?" in generated_node.outputs:
            print(
                "The LUT is incomplete. Missing values are represented by '?'."
                if verbose
                else None
            )
        return generated_node

    def generate_with_required_bias(
        self,
        required_node_bias=None,
        limit=1000,
        verbose=False,
        *args,
        **kwargs,
    ):
        """
        Generate a node with the required bias.
        This node takes a boolean node with "?" output values and generates all possible nodes with the missing output values filled to achieve the required bias as closely as possible.

        Args:
            required_node_bias (float) : The required node bias to fill the missing output values with.
            verbose (bool) : If True, print additional information.

        Returns:
            List of BooleanNode objects with the required bias.

        Example:
            >>> BooleanNode.generate_with_required_bias(required_node_bias=0.5, verbose=True, name="EG")

        Note:
            The required node bias should be a float value between 0 and 1.

            # TODO : [SRI] check why when run in a notebook does the cell take so long to run. It says pending. and takes forever to initialize and the runtime when finished is close to zero.
        """
        generated_node = self
        bias = required_node_bias  # making a copy for print statement at the end of function
        # Checking if more than one out of required_effective_connectivity, requried_node_bias and fill_missing_output_randomly are True, then raise an error.
        if required_node_bias is None:
            raise ValueError(
                "Please specify the required node bias to generate the node with the required bias."
            )

        if (
            required_node_bias is not None
        ):  # If required node bias is specified, then fill missing output values with the specified bias.
            # Checking if required node bias is within the achievable bias range of the node.

            # Calculating max achievable bias
            max_achievable_output = [
                "1" if output == "?" else output for output in generated_node.outputs
            ]
            max_achievable_bias = (
                sum(map(int, max_achievable_output)) / 2**generated_node.k
            )

            # Calculating the number of '1' required to achieve the required bias.
            required_ones = int(required_node_bias * 2**generated_node.k)
            current_ones = generated_node.outputs.count("1")

            min_achievable_bias = current_ones / 2**generated_node.k
            min = False  # flag to check if the required bias is less than the minimum achievable bias.
            # Checking if the required bias is achievable.
            if required_node_bias > max_achievable_bias:
                if verbose:
                    warnings.warn(
                        f"Required Node Bias is greater than the maximum achievable bias ({max_achievable_bias}) of the node. Generating with the maximum achievable bias."
                    )
                required_node_bias = max_achievable_bias

            elif required_node_bias < min_achievable_bias:
                if verbose:
                    warnings.warn(
                        f"Required Node Bias is lower than the minimum achievable bias ({min_achievable_bias}) of the node. Generating with the minimum achievable bias."
                    )
                required_node_bias = min_achievable_bias
                min = True

            # Fill the missing output values to achieve the required bias as closely as possible.
            required_ones = int(
                required_node_bias * 2**generated_node.k
            )  # recalculating in case the required bias was adjusted in the above steps.
            ones_to_be_generated = required_ones - current_ones
            number_of_missing_values = generated_node.outputs.count("?")

            missing_output_values = (
                ["1"] * ones_to_be_generated
                + ["0"] * (number_of_missing_values - ones_to_be_generated)
            )  # creating a list of 1 and 0 to replace the '?' with the right ratio required to achieve the required bias.

            combinationsnumber = comb(number_of_missing_values, ones_to_be_generated)

            if combinationsnumber > limit:
                warnings.warn(
                    f"Total possible permutaions = {combinationsnumber}. Selecting {limit} permutations randomly."
                )
            # create a list of all possible unique arrangements of the missing output values
            # combinations = list(islice(set(permutations(missing_output_values)), limit))
            def unique_permutations_missing_values(elements, n):
                """
                Generate n unique permutations of elements.
                """
                seen = set()
                elements = list(elements)  # Ensure we can shuffle
                random.shuffle(elements)  # Shuffle to ensure randomness in subsets
                for perm in permutations(elements):
                    perm_as_str = str(perm)  # Convert to string for hashability
                    if perm_as_str not in seen:
                        seen.add(perm_as_str)
                        yield perm
                        if len(seen) == n:
                            return

            combinations = unique_permutations_missing_values(
                missing_output_values, combinationsnumber
            )
            generated_node_permutations = [None] * combinationsnumber

            def node_permutations(combinations, node_outputs, *args, **kwargs):
                for combination in combinations:
                    combination = list(combination)
                    generated_outputs = node_outputs.copy()
                    for i, output in enumerate(node_outputs):
                        if output == "?":
                            generated_outputs[i] = combination.pop()
                    yield BooleanNode.from_output_list(
                        generated_outputs, *args, **kwargs
                    )

            generated_node_permutations = node_permutations(
                combinations, generated_node.outputs, *args, **kwargs
            )

            # generated_node_permutations = [None] * combinationsnumber

            # for count, combination in enumerate(combinations):
            #     combination = list(combination)
            #     # random.shuffle(combination) # shuffling the combination creates duplicates and misses some combinations.
            #     generated_outputs = generated_node.outputs.copy()
            #     for i, output in enumerate(generated_node.outputs):
            #         if output == "?":
            #             generated_outputs[i] = combination.pop()
            #     generated_node_permutations[count] = BooleanNode.from_output_list(
            #         generated_outputs, *args, **kwargs
            #     )  # generating a list of nodes with all possible permutations of the missing output values that achieve the required bias.

            output_bias_for_print = (
                ones_to_be_generated + current_ones
            ) / 2**generated_node.k  # for the print message in the end
            if verbose:
                if min:
                    print(
                        f"Generated {combinationsnumber} node(s) with a bias of {output_bias_for_print}. This is the closest achievable bias to the required bias of {bias}."
                    )
                else:
                    print(
                        f"Generated {combinationsnumber} node(s) with a bias of {output_bias_for_print}. This is the closest bias less than or equal to the required bias of {bias}."
                    )
            return generated_node_permutations  # returning a list of BooleanNode objects with the required bias.

    def generate_with_required_effective_connectivity(
        self,
        required_effective_connectivity=None,
        # limit=50,
        verbose=False,
        *args,
        **kwargs,
    ):
        """
        Generate a node with the required effective connectivity.
        This node takes a boolean node with "?" output values and generates all possible nodes with the missing output values filled to achieve the required effective connectivity as closely as possible.

        Args:
            required_effective_connectivity (float) : The required effective connectivity to fill the missing output values with. It will generate a node with the closest possible effective connectivity to the required effective connectivity.
            verbose (bool) : If True, print additional information.

        Returns:
            (BooleanNode) : the instantiated object.

        Example:
            >>> BooleanNode.generate_with_required_effective_connectivity(required_effective_connectivity=0.5, verbose=True, name="EG")

        Note:
            The required effective connectivity should be a float value between 0 and 1.

        # TODO : [SRI] to  cover the entire space of permutations evenly, what if i fill each node randomly and calculate the effective connectivity . then add them to a list of all nodes with sufficiently close effective connectivity? This option will only be activated if the calculated permutation space goes beyond a predecided threshold.
        """

        generated_node = self
        if required_effective_connectivity is not None:
            generated_outputs = generated_node.outputs.copy()
            missing_output_indices = [
                i for i, x in enumerate(generated_outputs) if x == "?"
            ]
            # print(f"Missing output indices = {missing_output_indices}." if verbose else None)

            missing_output_count = generated_outputs.count("?")
            # print(f"No. of '?' in output = {missing_output_count}.")
            missing_permutations = list(product(*[("0", "1")] * (missing_output_count)))
            # print(permutations)
            generated_node_permutations = [None] * len(missing_permutations)

            for count, missing_permutation in enumerate(missing_permutations):
                for i, index in enumerate(missing_output_indices):
                    generated_outputs[index] = missing_permutation[i]
                generated_node_permutations[count] = BooleanNode.from_output_list(
                    generated_outputs, *args, **kwargs
                )  # generating a list of nodes with all possible permutations of the missing output values.

            # print(f"Total output permutations generated = {len(generated_node_permutations)}.")

            permutation_effective_connectivity = [
                x.effective_connectivity() for x in generated_node_permutations
            ]
            closest_value = min(
                permutation_effective_connectivity,
                key=lambda x: abs(x - required_effective_connectivity),
            )
            closest_index = permutation_effective_connectivity.index(closest_value)

            generated_node = generated_node_permutations[closest_index]
            print(
                f"Generated the node with the closest possible effective connectivity of {generated_node.effective_connectivity()}."
                if verbose
                else None
            )

        return generated_node
