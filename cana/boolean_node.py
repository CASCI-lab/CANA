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

import math
from itertools import combinations, compress, product
from statistics import mean

import numpy as np
import pandas as pd

import cana.canalization.boolean_canalization as BCanalization
import cana.canalization.cboolean_canalization as cBCanalization
from cana.canalization.canalizing_map import build_canalizing_map
import cana.symmetry as symmetry
import cana.sensitivity as sensitivity
from cana.cutils import (
    binstate_to_statenum,
    flip_bit,
    statenum_to_binstate,
)
from cana.function_outputs import FunctionOutputs
from cana.utils import input_monotone, ncr


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
        **kwargs
    ):
        self.id = id  # the id of the node
        self.name = name  # the name of the node
        self.k = k  # k is the number of inputs
        self.inputs = list(map(int, inputs))  # the ids of the input nodes
        self.state = str(
            int(state)
        )  # the initial state of the node. internally this is treated as a string.
        # Multi-representation outputs (dense BitLut; one-/two-symbol on demand).
        if isinstance(outputs, FunctionOutputs):
            self.outputs = outputs
        else:
            self.outputs = FunctionOutputs.from_sequence(outputs)
        self.network = network  # the BooleanNetwork object this nodes belongs to
        self.verbose = verbose  # verbose mode

        # When attached to a network, share the NetworkDict FunctionOutputs object.
        if (
            network is not None
            and getattr(network, "network_dict", None) is not None
            and id in network.network_dict
        ):
            shared = network.network_dict[id]["outputs"]
            if isinstance(shared, FunctionOutputs):
                self.outputs = shared
            else:
                fo = FunctionOutputs.from_sequence(shared)
                network.network_dict[id]["outputs"] = fo
                self.outputs = fo

        # mask for inputs
        if len(self.inputs) > 0:
            self.mask = [(i in self.inputs) for i in range(max(self.inputs) + 1)]
        else:
            self.mask = []

        # Consistency (use outputs.k so schemata-only nodes need no dense LUT)
        if (k != 0) and (k != self.outputs.k):
            raise ValueError(
                "Number of k (inputs) do not match the number of output transitions"
            )

        # If all outputs are either positive or negative, the node is treated as a constant.
        # Schemata-only nodes skip dense LUT constant detection (is_constant → None).
        const_val = self.outputs.is_constant()
        if (const_val is not None) or (constant):
            self.set_constant(
                constant=True,
                state=const_val if const_val is not None else self.outputs[0],
            )
        else:
            self.set_constant(constant=False)

        # Canalization Variables (PI / TS live on outputs.*; coverage stays here)
        self._pi_coverage = None  # The Coverage of inputs by Prime Implicants schemata
        self._ts_coverage = None  # The Coverage of inputs by Two Symbol schemata

    def fill_function_outputs(self, representation):
        """Fill a missing ``FunctionOutputs`` representation slot from another.

        Args:
            representation (str) : ``"bit_lut"`` or ``"one_symbol_schemata"``.

        Returns:
            (FunctionOutputs) : This node's ``outputs`` (mutated in place).
        """
        if representation == "bit_lut":
            self.outputs.get_bit_lut()
        elif representation == "one_symbol_schemata":
            self.outputs.get_one_symbol_schemata()
        else:
            raise ValueError(
                "Unknown representation {0!r}; expected 'bit_lut' or "
                "'one_symbol_schemata'".format(representation)
            )
        return self.outputs

    def input_names(self):
        """Return regulator names in LUT / expression bit order.

        Requires this node to be attached to a :class:`BooleanNetwork`.
        """
        if self.network is None:
            raise ValueError(
                "input_names requires a BooleanNode attached to a network"
            )
        return [self.network.nodes[j].name for j in self.inputs]

    def expression(self, style="python"):
        """Return a named boolean expression for this node's update rule.

        Rebinds positional FO IR (``x0``…) to :meth:`input_names`.  If FO has
        no expression but has a dense LUT, builds a DNF via
        :func:`~cana.function_outputs.expression.lut_to_expression`.

        Args:
            style (str) : ``"python"`` or ``"symbolic"``.
        """
        from cana.function_outputs.expression import (
            from_positional,
            lut_to_expression,
            normalize_operators,
        )

        names = self.input_names()
        if self.outputs.expression is not None:
            named = from_positional(self.outputs.expression, names)
            if style == "python":
                return named
            return normalize_operators(named, style)
        lut = self.outputs.bit_lut
        if lut is not None:
            return lut_to_expression(names, lut, style=style)
        raise ValueError(
            "Cannot build named expression: neither expression nor bit_lut "
            "is available on outputs"
        )

    def __str__(self):
        if self.outputs.bit_lut is not None:
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
        elif self.outputs.expression is not None:
            if self.network is not None:
                try:
                    outputs = "expression={0!r}".format(self.expression())
                except ValueError:
                    outputs = "expression={0!r}".format(self.outputs.expression)
            else:
                outputs = "expression={0!r}".format(self.outputs.expression)
        elif self.outputs.one_symbol_schemata is not None:
            outputs = "one_symbol_schemata"
        else:
            outputs = "k={0}".format(self.outputs.k)
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
    def from_output_list(cls, outputs=list(), *args, **kwargs):
        """Instanciate a Boolean Node from a output transition list.

        Args:
            outputs (list) : The transition outputs of the node. Its length must be a power of two, :math:`2^k`.

        Returns:
            (BooleanNode) : the instanciated object. Subclasses receive an instance of the subclass.

        Raises:
            ValueError : if the length of ``outputs`` is not a power of two.

        Example:
            >>> BooleanNode.from_output_list(outputs=[0,0,0,1], name="AND")
        """
        n = len(outputs)
        if n == 0 or n & (n - 1):
            raise ValueError(
                "The length of `outputs` must be a power of two (2**k); got {:d}.".format(n)
            )
        k = n.bit_length() - 1
        id = kwargs.pop("id") if "id" in kwargs else 0
        name = kwargs.pop("name") if "name" in kwargs else "x"
        inputs = (
            kwargs.pop("inputs") if "inputs" in kwargs else [(x + 1) for x in range(k)]
        )
        state = kwargs.pop("state") if "state" in kwargs else False

        return cls(
            id=id,
            name=name,
            k=k,
            inputs=inputs,
            state=state,
            outputs=outputs,
            *args,
            **kwargs
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
        self.state = str(self.outputs[binstate_to_statenum(input_state)])
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
            operator([pi.count("2") for pi in self._pi_coverage[binstate]])
            for binstate in self._pi_coverage
        ]

        k_r = math.fsum(redundancy) / 2**self.k

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
        return math.fsum(summand) / len(summand)

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
        self._check_compute_canalization_variables(ts_coverage=True)
        summand = []
        # fTheta = a list of TS
        for fTheta in self._ts_coverage.values():
            inner = 0
            for ts in fTheta:
                inner += sum(
                    len(i) for i in ts[1]
                )  # assumes that indicies will ever only be in at most 1 group
            summand.append(inner / len(fTheta))
        return math.fsum(summand) / 2**self.k

    def distinct_symmetry(self):
        """Compute the distinct permutation symmetry of the node LUT.

        For each LUT entry, this computes the fraction of distinct input
        permutations that preserve the same output, excluding the identity
        permutation from both numerator and denominator.

        Returns:
            (float)

        See also:
            :func:`cana.symmetry.distinct_symmetry`
        """
        return symmetry.distinct_symmetry(self.outputs, self.k)

    def raw_symmetry(self):
        """Compute the raw symmetry of the node LUT.

        LUT rows are grouped by input Hamming weight. For each row, this
        computes the fraction of rows in the same weight group that have the
        same output, then averages across all LUT rows.

        Returns:
            (float)

        See also:
            :func:`cana.symmetry.raw_symmetry`
        """
        return symmetry.raw_symmetry(self.outputs, self.k)

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
        self,
        type="pi",
        pi_symbol="#",
        ts_symbol_list=["\u030A", "\u032F", "\u0303", "\u0330", "\u0306", "\u032E"],
    ):
        """Returns the simplified schemata Look Up Table (LUT)

        Args:
            type (string) : The type of schemata to return, either Prime Implicants ``pi`` or Two-Symbol ``ts``. Defaults to 'pi'.
            pi_symbol (str) : The Prime Implicant don't care symbol. Default is ``#``.
            ts_symbol_list (list) : A list containing Two Symbol permutable symbols, one per permutation group of a schema (combining marks: ring above, inverted breve below, tilde, tilde below, breve, breve below). A ``ValueError`` is raised if a schema has more groups than symbols.

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
        if type == "pi":
            self._check_compute_canalization_variables(prime_implicants=True)

            pi0s = self._prime_implicants.get(0, [])
            pi1s = self._prime_implicants.get(1, [])

            for output, pi in zip([0, 1], [pi0s, pi1s]):
                # Sort on display form so row order matches the visible don't-care symbol.
                for schemata in sorted(s.replace("2", pi_symbol) for s in pi):
                    r.append((schemata, output))

        # Two Symbol LUT
        elif type == "ts":
            self._check_compute_canalization_variables(two_symbols=True)

            ts0s, ts1s = self._two_symbols

            for output, ts in zip([0, 1], [ts0s, ts1s]):
                for i, (schemata, permutables, samesymbols) in enumerate(ts):
                    if max(len(permutables), len(samesymbols)) > len(ts_symbol_list):
                        raise ValueError(
                            "Schema %s has %d permutation groups but ts_symbol_list has only %d symbols; pass a longer ts_symbol_list."
                            % (schemata, max(len(permutables), len(samesymbols)), len(ts_symbol_list))
                        )
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
        """compute the activities of each incoming edge of the node, directly from the LUT.
        See :func:`cana.sensitivity.activities`.

        Returns:
            (list of floats)
        """
        return sensitivity.activities(self.outputs, self.k)

    def sensitivity(self, norm=False):
        """compute the average sensitivity of the node: the mean, over all input states,
        of the number of single-input flips that change the output.

        Delegates to :func:`cana.sensitivity.sensitivity`, which computes it directly from
        the look-up table. Up to CANA 1.0.2 this was ``sum(self.activities())``, kept as
        :func:`cana.sensitivity.sensitivity_old`; the two are bit-exactly equal, as
        asserted in ``tests/test_boolean_node.py``
        (``test_sensitivity_matches_original_implementation``).

        Args:
            norm (bool) : whether or not to normalize by the number of inputs (k)
        Returns:
            (float)
        See Also:
            :func:`cana.sensitivity.sensitivity`, :func:`activities`, :func:`c_sensitivity`.
        """
        return sensitivity.sensitivity(self.outputs, self.k, norm=norm)

    def canalizing_map(self, output=None, builder="legacy"):
        """Computes the node Canalizing Map (CM).

        Args:
            output (int) : The output CM to return. Default is ``None``, retuning both [0,1].
            builder (str or callable) : CM construction method. Default is ``'legacy'``.
                See :func:`cana.canalization.canalizing_map.build_canalizing_map`.

        Returns:
            CM (networkx.DiGraph) : a directed graph representation of the CM.

        See Also:
            :func:`boolean_network.dynamics_canalization_map` for the DCM and :func:`drawing.draw_canalizing_map_graphviz` for plotting.
        """
        self._check_compute_canalization_variables(two_symbols=True)

        input_labels = {}
        for in_id in self.inputs:
            if self.network is not None:
                input_labels[in_id] = self.network.get_node_name(in_id)[0]
            else:
                input_labels[in_id] = str(in_id)

        return build_canalizing_map(
            self._two_symbols,
            node_id=self.id,
            inputs=self.inputs,
            name=self.name,
            constant=self.constant,
            output=output,
            input_labels=input_labels,
            k=self.k,
            builder=builder,
        )

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

    @property
    def _prime_implicants(self):
        """Proxy to :attr:`outputs.one_symbol_schemata` (one source of truth)."""
        return self.outputs.one_symbol_schemata

    @_prime_implicants.setter
    def _prime_implicants(self, value):
        self.outputs.one_symbol_schemata = value

    @property
    def _two_symbols(self):
        """Proxy to :attr:`outputs.two_symbol_schemata` (one source of truth)."""
        return self.outputs.two_symbol_schemata

    @_two_symbols.setter
    def _two_symbols(self, value):
        self.outputs.two_symbol_schemata = value

    def _ensure_prime_implicants(self):
        """Ensure ``outputs.one_symbol_schemata`` holds Quine–McCluskey PIs.

        Stored one-symbol covers are not treated as PIs unless
        :attr:`~cana.function_outputs.FunctionOutputs.one_symbol_is_prime_implicant`
        is ``True``.  Otherwise materialize a dense LUT (complete schemata or
        expression) and recompute.  Incomplete schemata cannot be materialized.
        """
        outs = self.outputs
        if outs.one_symbol_is_prime_implicant is True:
            outs.get_one_symbol_schemata()
            return
        outs.get_bit_lut()
        outs.compute_one_symbol_schemata_from_bit_lut()

    def _check_compute_canalization_variables(self, **kwargs):
        """Recursevely check if the requested canalization variables are instantiated/computed, otherwise computes them in order.
        For example: to compute `two_symbols` we need `prime_implicants` first.
        Likewise, to compute `prime_implicants` we need the dense LUT (or stored schemata) first.
        """
        if "prime_implicants" in kwargs:
            self._ensure_prime_implicants()

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
            self.outputs.get_two_symbol_schemata()
        elif "ts_coverage" in kwargs:
            self._check_compute_canalization_variables(two_symbols=True)
            if self._ts_coverage is None:
                self._ts_coverage = BCanalization.computes_ts_coverage(
                    self.k, self.outputs, self._two_symbols
                )

        else:
            raise Exception("Canalization variable name not found. %s" % kwargs)
        return True

    def bias(self):
        r"""The node bias. The sum of the boolean output transitions divided by the number of entries (:math:`2^k`) in the LUT.

        .. math::

            bias(x) = \frac{ \sum_{f_{\alpha}\in F} s_{\alpha} }{ |F| }

        Returns:
            (float)

        See Also:
            :func:`~cana.boolean_network.BooleanNetwork.network_bias`
        """
        if self.outputs.bit_lut is not None:
            return self.outputs.bit_lut.popcount() / 2**self.k
        return sum(self.outputs) / 2**self.k

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
