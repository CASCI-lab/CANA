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
import warnings
from itertools import combinations, compress, product
from statistics import mean

import numpy as np
import pandas as pd

import cana.canalization.boolean_canalization as BCanalization
import cana.canalization.cboolean_canalization as cBCanalization
from cana.canalization.canalizing_map import build_canalizing_map
import cana.function_properties.bias as bias
import cana.function_properties.redundancy as redundancy
import cana.function_properties.sampling as sampling
import cana.function_properties.symmetry as symmetry
import cana.function_properties.sensitivity as sensitivity
from cana.cutils import (
    binstate_to_statenum,
    flip_bit,
    statenum_to_binstate,
)
from cana.function_properties.sampling import EXACT_K_MAX_DEFAULT
from cana.function_outputs import FunctionOutputs
from cana.function_outputs.bit_lut import MAX_DENSE_LUT_K
from cana.function_outputs.expression import positional_names
from cana.function_properties import input_signs as lut_input_signs
from cana.function_properties import syntactic_input_signs


def _refuse_coverage_k(k, *, kind):
    """Raise ``ValueError`` when building a Θ(2^k) coverage dict would be too large."""
    if int(k) > MAX_DENSE_LUT_K:
        raise ValueError(
            "refusing to build {0}: k={1:d} > MAX_DENSE_LUT_K={2:d}".format(
                kind, int(k), MAX_DENSE_LUT_K
            )
        )


class BooleanNode(object):
    """ """

    # --- Construction ---

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

    # Coverage caches live on ``outputs`` and clear when one-/two-symbol
    # schemata are replaced with a different value.
    @property
    def _pi_coverage(self):
        return self.outputs._pi_coverage

    @_pi_coverage.setter
    def _pi_coverage(self, value):
        self.outputs._pi_coverage = value

    @property
    def _ts_coverage(self):
        return self.outputs._ts_coverage

    @_ts_coverage.setter
    def _ts_coverage(self, value):
        self.outputs._ts_coverage = value

    @property
    def _sampled_pi_coverage(self):
        return self.outputs._sampled_pi_coverage

    @_sampled_pi_coverage.setter
    def _sampled_pi_coverage(self, value):
        self.outputs._sampled_pi_coverage = value

    @staticmethod
    def _factory_common_kwargs(k, kwargs):
        """Pop shared factory kwargs; default ``inputs`` is ``1..k`` (legacy)."""
        node_id = kwargs.pop("id") if "id" in kwargs else 0
        name = kwargs.pop("name") if "name" in kwargs else "x"
        inputs = (
            kwargs.pop("inputs")
            if "inputs" in kwargs
            else [(x + 1) for x in range(k)]
        )
        state = kwargs.pop("state") if "state" in kwargs else False
        return node_id, name, inputs, state

    @classmethod
    def from_bit_lut(cls, lut=None, *args, **kwargs):
        """Instantiate a Boolean Node from a dense LUT (sequence or ``BitLut``).

        Args:
            lut : Transition outputs of length :math:`2^k`, or a
                :class:`~cana.function_outputs.bit_lut.BitLut`.  (Also accepted as the legacy
                keyword ``outputs=``.)

        Returns:
            (BooleanNode) : the instantiated object. Subclasses receive an
            instance of the subclass.

        Raises:
            ValueError : if a sequence length is not a power of two.

        Example:
            >>> BooleanNode.from_bit_lut([0, 0, 0, 1], name="AND")
        """
        from cana.function_outputs.bit_lut import BitLut

        if lut is None and "outputs" in kwargs:
            lut = kwargs.pop("outputs")
        if lut is None:
            lut = []

        if isinstance(lut, BitLut):
            k = lut.k
            outputs = FunctionOutputs(bit_lut=lut)
        elif isinstance(lut, FunctionOutputs):
            k = lut.k
            outputs = lut
        else:
            n = len(lut)
            if n == 0 or n & (n - 1):
                raise ValueError(
                    "The length of `lut` must be a power of two (2**k); got {:d}.".format(
                        n
                    )
                )
            k = n.bit_length() - 1
            outputs = FunctionOutputs.from_sequence(lut)

        node_id, name, inputs, state = cls._factory_common_kwargs(k, kwargs)
        return cls(
            id=node_id,
            name=name,
            k=k,
            inputs=inputs,
            state=state,
            outputs=outputs,
            *args,
            **kwargs
        )

    @classmethod
    def from_positional_expression(cls, expression, *args, **kwargs):
        """Instantiate from a positional python expression (``x0``…``x{k-1}``).

        Args:
            expression (str) : Positional IR stored on
                :class:`~cana.function_outputs.FunctionOutputs`.
            k (int) : Number of inputs (required keyword-only).

        Example:
            >>> BooleanNode.from_positional_expression(
            ...     "x0 and x1", k=2, name="AND"
            ... )
        """
        if "k" not in kwargs:
            raise TypeError(
                "from_positional_expression() missing required "
                "keyword-only argument: 'k'"
            )
        k = int(kwargs.pop("k"))
        node_id, name, inputs, state = cls._factory_common_kwargs(k, kwargs)
        return cls(
            id=node_id,
            name=name,
            k=k,
            inputs=inputs,
            state=state,
            outputs=FunctionOutputs(positional_expression=expression, k=k),
            *args,
            **kwargs
        )

    @classmethod
    def from_one_symbol_schemata(cls, schemata, *args, **kwargs):
        """Instantiate from a one-symbol (wildcard) cover.

        Args:
            schemata : Polarity → set of length-``k`` schemas over ``{0,1,2}``.
            k (int, optional) : Keyword-only; inferred from schema length when
                omitted.

        Example:
            >>> BooleanNode.from_one_symbol_schemata({1: {"11"}}, k=2, name="AND")
        """
        k = kwargs.pop("k") if "k" in kwargs else None
        fo = FunctionOutputs(one_symbol_schemata=schemata, k=k)
        k = fo.k
        node_id, name, inputs, state = cls._factory_common_kwargs(k, kwargs)
        return cls(
            id=node_id,
            name=name,
            k=k,
            inputs=inputs,
            state=state,
            outputs=fo,
            *args,
            **kwargs
        )

    @classmethod
    def from_two_symbol_schemata(cls, schemata, *args, **kwargs):
        """Instantiate from a two-symbol schemata pair ``(ts0, ts1)``.

        Args:
            schemata : ``(list, list)`` two-symbol structure.
            k (int, optional) : Keyword-only; inferred from schema length when
                omitted.
        """
        k = kwargs.pop("k") if "k" in kwargs else None
        fo = FunctionOutputs(two_symbol_schemata=schemata, k=k)
        k = fo.k
        node_id, name, inputs, state = cls._factory_common_kwargs(k, kwargs)
        return cls(
            id=node_id,
            name=name,
            k=k,
            inputs=inputs,
            state=state,
            outputs=fo,
            *args,
            **kwargs
        )

    # --- Representation ---

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
        elif self.outputs.positional_expression is not None:
            if self.network is not None:
                try:
                    outputs = "named_expression={0!r}".format(
                        self.named_expression()
                    )
                except ValueError:
                    outputs = "positional_expression={0!r}".format(
                        self.outputs.positional_expression
                    )
            else:
                outputs = "positional_expression={0!r}".format(
                    self.outputs.positional_expression
                )
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

    def input_names(self):
        """Return regulator names in LUT / expression bit order.

        Requires this node to be attached to a :class:`BooleanNetwork`.
        """
        if self.network is None:
            raise ValueError(
                "input_names requires a BooleanNode attached to a network"
            )
        return [self.network.nodes[j].name for j in self.inputs]

    def named_expression(self, style="python", warn=True):
        """Return a named boolean expression for this node's update rule.

        Uses :func:`~cana.function_outputs.expression.function_outputs_to_named_expression`:
        stored positional expression, else two-symbol, else one-symbol ON
        cover, else dense LUT DNF. Soft-warns when verification flags are not
        ``True`` for the chosen non-LUT source (never blocks).

        Args:
            style (str) : ``"python"`` or ``"symbolic"``.
            warn (bool) : If False, suppress verification soft-warnings.
        """
        from cana.function_outputs.expression import (
            function_outputs_to_named_expression,
        )

        return function_outputs_to_named_expression(
            self.outputs,
            self.input_names(),
            style=style,
            node_label=self.name,
            warn=warn,
        )

    def fill_function_outputs(self, representation):
        """Fill a missing ``FunctionOutputs`` representation slot from another.

        Args:
            representation (str) : ``"bit_lut"``, ``"one_symbol_schemata"``,
                ``"two_symbol_schemata"``, or ``"positional_expression"``.

        Returns:
            (FunctionOutputs) : This node's ``outputs`` (mutated in place).
        """
        outs = self.outputs
        if representation == "bit_lut":
            if outs.bit_lut is None:
                outs.get_bit_lut()
        elif representation == "one_symbol_schemata":
            if outs.one_symbol_schemata is None:
                outs.get_one_symbol_schemata()
        elif representation == "two_symbol_schemata":
            if outs.two_symbol_schemata is None:
                outs.get_two_symbol_schemata()
        elif representation == "positional_expression":
            if outs.positional_expression is None:
                outs.get_positional_expression()
        else:
            raise ValueError(
                "Unknown representation {0!r}; expected 'bit_lut', "
                "'one_symbol_schemata', 'two_symbol_schemata', or "
                "'positional_expression'".format(representation)
            )
        return self.outputs

    # --- Dynamics ---

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

    def input_mask(self, binstate):
        """Returns the mask applied to the binary state binstate

        Args:
            binstate (str) : the binary state

        Returns:
            output (str) : the masked state
        """
        return "".join(compress(binstate, self.mask))

    # --- Look-up tables ---

    def look_up_table(self):
        """Returns the Look Up Table (LUT)

        Returns:
            (pandas.DataFrame): the LUT

        Examples:
            >>> AND = BooleanNode.from_bit_lut([0,0,0,1])
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
            d.append((statenum_to_binstate(statenum, base=self.k), int(output)))

        # object dtype keeps Python ints through notebook concat/fillna (avoids 0.0).
        return pd.DataFrame(d, columns=["In:", "Out:"], dtype=object)

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
            >>> AND = BooleanNode.from_bit_lut([0,0,0,1])
            >>> AND.schemata_look_up_table(type='pi')

        Note:
            See the full list of `combining characters <https://en.wikipedia.org/wiki/Combining_character>`_ to use other symbols as the permutation symbol.

        See also:
            :func:`look_up_table`
        """
        r = []
        # Prime Implicant LUT
        if type == "pi":
            self.outputs.get_one_symbol_schemata()

            pi0s = self._prime_implicants.get(0, [])
            pi1s = self._prime_implicants.get(1, [])

            for output, pi in zip([0, 1], [pi0s, pi1s]):
                # Sort on display form so row order matches the visible don't-care symbol.
                for schemata in sorted(s.replace("2", pi_symbol) for s in pi):
                    r.append((schemata, output))

        # Two Symbol LUT
        elif type == "ts":
            self.outputs.get_two_symbol_schemata()

            ts0s, ts1s = self._two_symbols

            def _mark_schema(schemata, group, mark):
                return "".join(
                    x if (k not in group) else "%s%s" % (x, mark)
                    for k, x in enumerate(schemata)
                )

            for output, ts in zip([0, 1], [ts0s, ts1s]):
                rows = []
                for schemata, permutables, samesymbols in ts:
                    # Same-symbol groups are trivial (identical literals);
                    # only true permutation groups are shown.
                    if len(permutables) > len(ts_symbol_list):
                        raise ValueError(
                            "Schema %s has %d permutation groups but ts_symbol_list has only %d symbols; pass a longer ts_symbol_list."
                            % (schemata, len(permutables), len(ts_symbol_list))
                        )
                    if permutables:
                        parts = [
                            _mark_schema(schemata, group, ts_symbol_list[j])
                            for j, group in enumerate(permutables)
                        ]
                        string = (
                            "(%s)" % (" | ".join(parts))
                            if len(parts) > 1
                            else parts[0]
                        )
                    else:
                        string = schemata
                    rows.append((string, output))
                r.extend(rows)
        else:
            raise AttributeError(
                'The schemata type could not be found. Try "PI" (Prime Implicants) or "TS" (Two-Symbol).'
            )

        r = [
            (schemata.replace("2", pi_symbol), int(output)) for schemata, output in r
        ]
        if type == "ts":
            # Stable row order within each output polarity (sets → nondeterministic).
            r = sorted(r, key=lambda row: (row[1], row[0]))
        # object dtype keeps Python ints through notebook concat/fillna (avoids 0.0).
        return pd.DataFrame(r, columns=["In:", "Out:"], dtype=object)

    # --- Canalization (internals) ---

    @property
    def _prime_implicants(self):
        """Proxy to :attr:`outputs.one_symbol_schemata` (one source of truth).

        Soft-warns when the stored cover is not known to be prime implicants
        (same policy as :meth:`outputs.get_one_symbol_schemata`).
        """
        self.outputs._warn_if_one_symbol_unverified()
        return self.outputs.one_symbol_schemata

    @_prime_implicants.setter
    def _prime_implicants(self, value):
        self.outputs.one_symbol_schemata = value

    @property
    def _two_symbols(self):
        """Proxy to :attr:`outputs.two_symbol_schemata` (one source of truth).

        Soft-warns when the stored two-symbol is not known to be canonical
        (same policy as :meth:`outputs.get_two_symbol_schemata`).
        """
        self.outputs._warn_if_two_symbol_unverified()
        return self.outputs.two_symbol_schemata

    @_two_symbols.setter
    def _two_symbols(self, value):
        self.outputs.two_symbol_schemata = value

    def pi_coverage(self):
        """Return the :math:`F'` (Prime Implicants) binary state coverage.

        Lazily fills :attr:`_pi_coverage` (refuses oversized ``k``).

        See also:
            :meth:`sampled_pi_coverage`, :meth:`ts_coverage`
        """
        _refuse_coverage_k(self.k, kind="pi_coverage")
        self.outputs.get_one_symbol_schemata()
        if self._pi_coverage is None:
            self._pi_coverage = dict()
            for output, piset in self._prime_implicants.items():
                self._pi_coverage.update(cBCanalization.return_pi_coverage(piset))

            # make sure every inputstate was covered by at least one prime implicant
            assert len(self._pi_coverage) == 2**self.k
        return self._pi_coverage

    def sampled_pi_coverage(
        self,
        *,
        exact_k_max=EXACT_K_MAX_DEFAULT,
        n_samples=None,
        rng=None,
        reuse=True,
    ):
        """Sample a partial pi-coverage for high-``k`` measure estimates.

        Caches the result on ``_sampled_pi_coverage``. Pass ``reuse=False`` to
        force a fresh sample.

        See also:
            :meth:`pi_coverage`,
            :func:`cana.function_properties.sampling.sample_pi_coverage`.
        """
        n = (
            (1 << int(exact_k_max))
            if n_samples is None
            else int(n_samples)
        )
        if (
            reuse
            and self._sampled_pi_coverage is not None
            and self._sampled_pi_coverage.k == self.k
            and self._sampled_pi_coverage.n_samples == n
        ):
            return self._sampled_pi_coverage

        schemata = self.outputs.get_one_symbol_schemata()
        sampled = sampling.sample_pi_coverage(
            schemata, n, rng=rng, k=self.k
        )
        self._sampled_pi_coverage = sampled
        return sampled

    def ts_coverage(self):
        """Return the :math:`F''` (Two-Symbol schematas) binary state coverage.

        Lazily fills :attr:`_ts_coverage` (refuses oversized ``k``).

        See also:
            :meth:`pi_coverage`, :meth:`sampled_pi_coverage`
        """
        _refuse_coverage_k(self.k, kind="ts_coverage")
        self.outputs.get_two_symbol_schemata()
        if self._ts_coverage is None:
            self._ts_coverage = BCanalization.computes_ts_coverage(
                self.k, self.outputs, self._two_symbols
            )
        return self._ts_coverage

    def _check_compute_canalization_variables(self, **kwargs):
        """Deprecated kwargs dispatcher for canalization lazy-compute.

        .. deprecated::
            Use :meth:`outputs.get_one_symbol_schemata`,
            :meth:`outputs.get_two_symbol_schemata`, :meth:`pi_coverage`,
            or :meth:`ts_coverage` instead.
        """
        warnings.warn(
            "_check_compute_canalization_variables is deprecated; "
            "use outputs.get_one_symbol_schemata / "
            "outputs.get_two_symbol_schemata / "
            "pi_coverage / ts_coverage.",
            DeprecationWarning,
            stacklevel=2,
        )
        if "prime_implicants" in kwargs:
            self.outputs.get_one_symbol_schemata()
        elif "pi_coverage" in kwargs:
            self.pi_coverage()
        elif "two_symbols" in kwargs:
            self.outputs.get_two_symbol_schemata()
        elif "ts_coverage" in kwargs:
            self.ts_coverage()
        else:
            raise Exception("Canalization variable name not found. %s" % kwargs)
        return True

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
        self.outputs.get_two_symbol_schemata()

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

    # --- Canalization metrics ---

    def input_redundancy(
        self,
        operator=mean,
        norm=True,
        *,
        exact_k_max=EXACT_K_MAX_DEFAULT,
        n_samples=None,
        rng=None,
        return_mode=False,
    ):
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
            exact_k_max (int) : For ``k <= exact_k_max``, compute exactly from full
                ``pi_coverage``. Above that, sample minterms. Defaults to 14.
                Pass ``exact_k_max=k`` (or larger) to force the exact path for
                ``k > 14``. Pass ``exact_k_max=-1`` to force sampling without a
                warning.
            n_samples (int or None) : Sample size when ``k > exact_k_max``.
                Defaults to ``2**exact_k_max``.
            rng : Optional ``numpy.random.Generator`` for the sampling path.
            return_mode (bool) : If True, return ``(value, \"exact\"|\"sampled\")``.


        Returns:
            (float) : The :math:`k_r` value.
            or ``(float, str)`` when ``return_mode`` is True.

        Note:
            The complete mathematical description can be found in :cite:`Marques-Pita:2013`.
            For large ``k``, the last sampled coverage is cached on
            ``_sampled_pi_coverage`` for bootstrap CIs via
            :func:`cana.function_properties.redundancy.bootstrap_input_redundancy`.
            Automatic sampling (``k > exact_k_max`` with ``exact_k_max >= 0``)
            emits a :class:`UserWarning`.

        See also:
            :func:`effective_connectivity`, :func:`input_symmetry`, :func:`edge_redundancy`,
            :func:`cana.function_properties.redundancy.input_redundancy`.
        """
        if not hasattr(operator, "__call__"):
            raise AttributeError(
                'The operator you selected must be a function. Try "min", "statitics.mean", or "max".'
            )

        if self.k <= exact_k_max:
            value = redundancy.input_redundancy_from_coverage(
                self.pi_coverage(), self.k, operator=operator, norm=norm
            )
            mode = "exact"
        else:
            if n_samples is not None:
                n = int(n_samples)
            elif exact_k_max >= 0:
                n = 1 << int(exact_k_max)
            else:
                raise ValueError(
                    "n_samples is required when exact_k_max < 0 "
                    "(forced sampling path)"
                )
            mode = redundancy._redundancy_mode(
                self.k, n, use_exact=False
            )
            redundancy._warn_if_sampled(
                "BooleanNode.input_redundancy",
                mode=mode,
                k=self.k,
                exact_k_max=int(exact_k_max),
                n=n,
            )
            sampled = self.sampled_pi_coverage(
                exact_k_max=exact_k_max, n_samples=n_samples, rng=rng
            )
            value = redundancy.input_redundancy_from_coverage(
                sampled.coverage, self.k, operator=operator, norm=norm
            )

        if return_mode:
            return value, mode
        return value

    def edge_redundancy(
        self,
        bound="mean",
        *,
        exact_k_max=EXACT_K_MAX_DEFAULT,
        n_samples=None,
        rng=None,
        return_mode=False,
    ):
        r"""The Edge Redundancy :math:`r_{i}` is the mean number of unnecessary inputs (or ``#``) in the Prime Implicants Look Up Table (LUT) for that input.
        Since there may be more than one redescription schema for each input entry, the input redundancy is bounded by an upper and lower limit.

        .. math::

            r_i(x_i) = \frac{ \sum_{f_{\alpha} \in F} \Phi_{\theta:f_{\alpha} \in \Theta_{\theta}} (X^{\#}_{\theta_i} ) }{ |F| }

        where :math:`\Phi` is a function (:math:`min` or :math:`max`) and :math:`F` is the node LUT.

        Args:
            bound (string) : The bound to which compute input redundancy.
                Mode "input" accepts: ["lower", "mean", "ave", upper", "tuple"].
                Defaults to "mean".
            exact_k_max (int) : For ``k <= exact_k_max``, compute exactly from full
                ``pi_coverage``. Above that, sample minterms. Defaults to 14.
                Pass ``exact_k_max=k`` (or larger) to force the exact path for
                ``k > 14``. Pass ``exact_k_max=-1`` to force sampling without a
                warning.
            n_samples (int or None) : Sample size when ``k > exact_k_max``.
                Defaults to ``2**exact_k_max``.
            rng : Optional ``numpy.random.Generator`` for the sampling path.
            return_mode (bool) : If True, return ``(value, \"exact\"|\"sampled\")``.

        Returns:
            (list) : The list of :math:`r_i` for inputs.
            or ``(list, str)`` when ``return_mode`` is True.

        Note:
            The complete mathematical description can be found in :cite:Gates:2020`.
            For large ``k``, the last sampled coverage is cached on
            ``_sampled_pi_coverage`` for bootstrap CIs via
            :func:`cana.function_properties.redundancy.bootstrap_edge_redundancy`.
            Automatic sampling (``k > exact_k_max`` with ``exact_k_max >= 0``)
            emits a :class:`UserWarning`.

        See also:
            :func:`effective_connectivity`, :func:`input_symmetry`,
            :func:`cana.function_properties.redundancy.edge_redundancy`.
        """
        if bound not in ("lower", "mean", "avg", "upper", "tuple"):
            raise AttributeError(
                'The bound you selected does not exist. Try "upper", "mean", "lower" or "tuple".'
            )

        if self.k <= exact_k_max:
            value = redundancy.edge_redundancy_from_coverage(
                self.pi_coverage(), self.k, bound=bound
            )
            mode = "exact"
        else:
            if n_samples is not None:
                n = int(n_samples)
            elif exact_k_max >= 0:
                n = 1 << int(exact_k_max)
            else:
                raise ValueError(
                    "n_samples is required when exact_k_max < 0 "
                    "(forced sampling path)"
                )
            mode = redundancy._redundancy_mode(
                self.k, n, use_exact=False
            )
            redundancy._warn_if_sampled(
                "BooleanNode.edge_redundancy",
                mode=mode,
                k=self.k,
                exact_k_max=int(exact_k_max),
                n=n,
            )
            sampled = self.sampled_pi_coverage(
                exact_k_max=exact_k_max, n_samples=n_samples, rng=rng
            )
            value = redundancy.edge_redundancy_from_coverage(
                sampled.coverage, self.k, bound=bound
            )

        if return_mode:
            return value, mode
        return value

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

    def edge_effectiveness(
        self,
        bound="mean",
        *,
        exact_k_max=EXACT_K_MAX_DEFAULT,
        n_samples=None,
        rng=None,
        return_mode=False,
    ):
        r"""The Edge Effectiveness is the mean number of an input's states needed to determine the transition of the node.

        .. math::

            e_i(x_i) = 1 - r_i(x_i)

        Args:
            bound (string) : The bound for the :math:`k_r` Input Redundancy
            exact_k_max, n_samples, rng, return_mode : Forwarded to
                :func:`edge_redundancy`.

        Returns:
            (list) : The list of :math:`e_r` values.
            or ``(list, str)`` when ``return_mode`` is True.

        See Also:
            :func:`input_redundancy`, :func:`input_symmetry`, :func:`~cana.boolean_network.BooleanNetwork.effective_graph`.
        """
        r = self.edge_redundancy(
            bound=bound,
            exact_k_max=exact_k_max,
            n_samples=n_samples,
            rng=rng,
            return_mode=return_mode,
        )
        if return_mode:
            r_i, mode = r
            return [1.0 - x_i for x_i in r_i], mode
        return [1.0 - x_i for x_i in r]

    # --- Symmetry ---

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

    def input_symmetry(self, aggOp="mean", kernel="numDots", sameSymbol=False):
        """compute the input symmetry (k_s) of the boolean node, with variations via the specified functions.

        Args:
            aggOp : the function aggregating over all two-symbol schemata that redescribe a LUT entry
            kernel : the function to compute on a given two-symbol schema

        Returns:
            (float) : the mean over all LUT entries of the aggOp applied to the kernel of all two-symbol schemata that redescribe the LUT entry
        """
        strToOp = {"mean": np.mean, "max": max, "min": min}
        strToKern = {"numDots": self.symKernel_numDots}
        op = strToOp[aggOp]

        def kernFunc(x):
            return strToKern[kernel](x, sameSymbol=sameSymbol)

        ts_coverage = self.ts_coverage()

        summand = []
        for fAlpha, fTheta in ts_coverage.items():
            summand.append(op(list(map(kernFunc, fTheta))))
        return math.fsum(summand) / len(summand)

    def input_symmetry_mean(self):
        """compute the input symmetry (k_s) of the boolean node.
           Specifically, computes it using the avg operator for the summand.
           Refactoring of input_symmetry for speed.

        Returns:
            (float)
        """
        ts_coverage = self.ts_coverage()
        summand = []
        # fTheta = a list of TS
        for fTheta in ts_coverage.values():
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
            :func:`cana.function_properties.symmetry.distinct_symmetry`
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
            :func:`cana.function_properties.symmetry.raw_symmetry`
        """
        return symmetry.raw_symmetry(self.outputs, self.k)

    # --- Sensitivity ---

    def activities_from_bit_lut(self):
        """Activities from the dense LUT via :func:`cana.function_properties.sensitivity.activities_from_bit_lut`.

        Returns:
            (list of floats)
        """
        return sensitivity.activities_from_bit_lut(self.outputs.get_bit_lut())

    def sensitivity_from_bit_lut(self, norm=False):
        """Average sensitivity from the dense LUT via
        :func:`cana.function_properties.sensitivity.sensitivity_from_bit_lut`.

        Args:
            norm (bool) : whether or not to normalize by the number of inputs (k)

        Returns:
            (float)
        """
        return sensitivity.sensitivity_from_bit_lut(
            self.outputs.get_bit_lut(), norm=norm
        )

    def activities_from_schemata(
        self,
        *,
        exact_k_max=EXACT_K_MAX_DEFAULT,
        n_samples=None,
        rng=None,
    ):
        """Activities as upper edge effectiveness from one-symbol schemata.

        See :func:`cana.function_properties.sensitivity.activities_from_schemata`. Prefer this when
        prime implicants are available, or when ``k`` exceeds the dense-LUT limit.

        Args:
            exact_k_max, n_samples, rng : Forwarded to :func:`edge_effectiveness`.

        Returns:
            (list of floats)
        """
        return self.edge_effectiveness(
            bound="upper",
            exact_k_max=exact_k_max,
            n_samples=n_samples,
            rng=rng,
        )

    def sensitivity_from_schemata(
        self,
        norm=False,
        *,
        exact_k_max=EXACT_K_MAX_DEFAULT,
        n_samples=None,
        rng=None,
    ):
        """Average sensitivity from one-symbol schemata: sum of
        :meth:`activities_from_schemata`.

        Args:
            norm (bool) : whether or not to normalize by the number of inputs (k)
            exact_k_max, n_samples, rng : Forwarded to :meth:`activities_from_schemata`.

        Returns:
            (float)
        """
        x = math.fsum(
            self.activities_from_schemata(
                exact_k_max=exact_k_max, n_samples=n_samples, rng=rng
            )
        )
        if norm:
            return x / self.k
        return x

    def activities(
        self,
        *,
        exact_k_max=EXACT_K_MAX_DEFAULT,
        n_samples=None,
        rng=None,
    ):
        """Activities of each incoming edge.

        Prefers :meth:`activities_from_schemata` when full or sampled
        ``pi_coverage`` is already cached; otherwise uses
        :meth:`activities_from_bit_lut` when ``k <= exact_k_max``; otherwise
        :meth:`activities_from_schemata`. Sampling kwargs apply only on the
        schemata path.

        Returns:
            (list of floats)
        """
        if self._pi_coverage is not None or self._sampled_pi_coverage is not None:
            return self.activities_from_schemata(
                exact_k_max=exact_k_max, n_samples=n_samples, rng=rng
            )
        if self.k <= exact_k_max:
            return self.activities_from_bit_lut()
        return self.activities_from_schemata(
            exact_k_max=exact_k_max, n_samples=n_samples, rng=rng
        )

    def sensitivity(
        self,
        norm=False,
        *,
        exact_k_max=EXACT_K_MAX_DEFAULT,
        n_samples=None,
        rng=None,
    ):
        """Average sensitivity: mean number of single-input flips that change the output.

        Prefers :meth:`sensitivity_from_schemata` when full or sampled
        ``pi_coverage`` is already cached; otherwise uses
        :meth:`sensitivity_from_bit_lut` when ``k <= exact_k_max``; otherwise
        :meth:`sensitivity_from_schemata`. The two formulations are bit-exactly
        equal on the full state space. Sampling kwargs apply only on the
        schemata path.

        Args:
            norm (bool) : whether or not to normalize by the number of inputs (k)
            exact_k_max, n_samples, rng : Forwarded to the schemata path when used.

        Returns:
            (float)
        See Also:
            :meth:`activities`, :meth:`c_sensitivity`,
            :func:`cana.function_properties.sensitivity.sensitivity_from_bit_lut`,
            :func:`cana.function_properties.sensitivity.sensitivity_from_schemata`.
        """
        if self._pi_coverage is not None or self._sampled_pi_coverage is not None:
            return self.sensitivity_from_schemata(
                norm=norm,
                exact_k_max=exact_k_max,
                n_samples=n_samples,
                rng=rng,
            )
        if self.k <= exact_k_max:
            return self.sensitivity_from_bit_lut(norm=norm)
        return self.sensitivity_from_schemata(
            norm=norm,
            exact_k_max=exact_k_max,
            n_samples=n_samples,
            rng=rng,
        )

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
            return S_c_f / float(math.comb(self.k, c)) / float(2**self.k)
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
                            S_c_f += math.comb(max_k - self.k, c - ic)
            return S_c_f / float(math.comb(max_k, c)) / float(2**self.k)

    # --- Node properties ---

    def bias_from_bit_lut(self):
        """Bias from the dense LUT via :func:`cana.function_properties.bias.bias_from_bit_lut`.

        Returns:
            (float)
        """
        return bias.bias_from_bit_lut(self.outputs.get_bit_lut())

    def bias_from_schemata(
        self,
        *,
        exact_k_max=EXACT_K_MAX_DEFAULT,
        n_samples=None,
        rng=None,
        return_mode=False,
    ):
        """Bias as ON-fraction from one-symbol schemata.

        See :func:`cana.function_properties.bias.bias_from_schemata`. Prefer
        this when prime implicants are available, or when ``k`` exceeds the
        dense-LUT limit.

        Args:
            exact_k_max, n_samples, rng : Same two-track kwargs as
                :meth:`input_redundancy`.
            return_mode (bool) : If True, return ``(value, \"exact\"|\"sampled\")``.

        Returns:
            (float) or ``(float, str)`` when ``return_mode`` is True.
        """
        return bias.bias_from_schemata(
            self.outputs.get_one_symbol_schemata(),
            k=self.k,
            exact_k_max=exact_k_max,
            n_samples=n_samples,
            rng=rng,
            return_mode=return_mode,
        )

    def bias(
        self,
        *,
        exact_k_max=EXACT_K_MAX_DEFAULT,
        n_samples=None,
        rng=None,
        return_mode=False,
    ):
        r"""The node bias. The sum of the boolean output transitions divided by the number of entries (:math:`2^k`) in the LUT.

        .. math::

            bias(x) = \frac{ \sum_{f_{\alpha}\in F} s_{\alpha} }{ |F| }

        Prefers :meth:`bias_from_schemata` when full or sampled
        ``pi_coverage`` is already cached; otherwise uses
        :meth:`bias_from_bit_lut` when ``k <= exact_k_max``; otherwise
        :meth:`bias_from_schemata`. Sampling kwargs apply only on the
        schemata path.

        Args:
            exact_k_max, n_samples, rng : Forwarded to the schemata path when used.
            return_mode (bool) : If True, return ``(value, \"exact\"|\"sampled\")``.
                On the LUT path the mode is always ``\"exact\"``.

        Returns:
            (float) or ``(float, str)`` when ``return_mode`` is True.

        See Also:
            :func:`~cana.boolean_network.BooleanNetwork.network_bias`,
            :func:`cana.function_properties.bias.bias_from_bit_lut`,
            :func:`cana.function_properties.bias.bias_from_schemata`.
        """
        if self._pi_coverage is not None or self._sampled_pi_coverage is not None:
            return self.bias_from_schemata(
                exact_k_max=exact_k_max,
                n_samples=n_samples,
                rng=rng,
                return_mode=return_mode,
            )
        if self.k <= exact_k_max:
            value = self.bias_from_bit_lut()
            if return_mode:
                return value, "exact"
            return value
        return self.bias_from_schemata(
            exact_k_max=exact_k_max,
            n_samples=n_samples,
            rng=rng,
            return_mode=return_mode,
        )

    def input_signs(self):
        """
        Determine if each input can be considered activation (1), inhibition (-1), or neither (0).

        Prefers a fast syntactic NNF scan when a stored positional expression
        is available and syntactically unate (no LUT).  Otherwise uses
        semantic (truth-table) signs via
        :func:`~cana.function_properties.unate.input_signs`.

        Returns:
            (list) : The list of input signs.
        """
        expr = self.outputs.positional_expression
        if expr is not None:
            signs = syntactic_input_signs(expr, positional_names(self.k))
            if signs is not None:
                return signs
        return lut_input_signs(self.outputs.get_bit_lut())
