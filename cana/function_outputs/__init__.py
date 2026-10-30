# -*- coding: utf-8 -*-
"""Multi-representation Boolean function outputs for a node.

``FunctionOutputs`` is what :attr:`cana.boolean_node.BooleanNode.outputs`
stores.  Peer representation slots may hold a dense
:class:`~cana.bit_lut.BitLut`, one-symbol (prime-implicant) schemata,
two-symbol schemata, and/or a positional expression.  Multiple representations may
coexist; named converters document each derivation, and lazy getters
(:meth:`get_bit_lut`, :meth:`get_one_symbol_schemata`,
:meth:`get_two_symbol_schemata`, :meth:`get_positional_expression`) try every peer
slot, cache intermediates computed along the way, and raise
``ValueError`` when none can produce the requested representation.

Dense Sequence access (``len``, ``[i]``, iterate) and :meth:`eval` prefer
``bit_lut`` and materialize it from one-symbol schemata, two-symbol
schemata (via one-symbol), or a positional expression via
:meth:`get_bit_lut` when needed.
"""
from __future__ import annotations

import warnings
from typing import Any, Iterator, List, Optional, Sequence, Tuple, Union

from cana.bit_lut import BitLut
from cana.function_outputs.expression import (
    positional_expression_to_bit_lut,
    lut_to_expression,
    one_symbol_to_expression,
    positional_names,
    reorder_positional,
)
from cana.function_outputs.one_symbol import (
    OneSymbolSchemata,
    bit_lut_from_one_symbol_schemata,
    one_symbol_schemata_from_bit_lut,
    reorder as reorder_one_symbol,
)
from cana.function_outputs.two_symbol import (
    one_symbol_from_two_symbol,
    reorder as reorder_two_symbol,
    two_symbol_to_expression,
)

__all__ = [
    "FunctionOutputs",
    "OneSymbolSchemata",
    "TwoSymbolSchemata",
    "one_symbol_from_two_symbol",
    "two_symbol_to_expression",
]

# (ts0, ts1); each entry is [schema_str, perm_groups, same_groups].
TwoSymbolSchemata = Tuple[list, list]


class FunctionOutputs(object):
    """Container for one or more representations of a Boolean update rule.

    Attributes
    ----------
    bit_lut :
        Dense packed truth table of length ``2**k``, if known.
    one_symbol_schemata :
        Prime-implicant / one-symbol (wildcard) cover: ``dict`` keyed by
        polarity ``0``/``1`` with sets of length-``k`` strings over
        ``{0,1,2}`` (``2`` = don't-care), if known.
    two_symbol_schemata :
        Two-symbol schemata as ``(ts0, ts1)``, if known.  Each list holds
        ``[schema_str, perm_groups, same_groups]`` entries.
    positional_expression :
        Positional python IR (``and``/``or``/``not`` over ``x0``…``x{k-1}``),
        if known.  Network names are rebound at the
        :class:`~cana.boolean_node.BooleanNode` / I/O boundary.
    one_symbol_is_prime_implicant :
        ``None`` until known; ``True``/``False`` after QM compute or
        :meth:`verify_one_symbol_schemata`.
    representations_consistent :
        ``None`` until known; ``True``/``False`` after
        :meth:`verify_representations_consistent`.
    """

    __slots__ = (
        "bit_lut",
        "one_symbol_schemata",
        "two_symbol_schemata",
        "positional_expression",
        "one_symbol_is_prime_implicant",
        "representations_consistent",
        "_k",
    )

    def __init__(
        self,
        bit_lut: Optional[BitLut] = None,
        one_symbol_schemata: Optional[OneSymbolSchemata] = None,
        two_symbol_schemata: Optional[TwoSymbolSchemata] = None,
        positional_expression: Optional[Any] = None,
        k: Optional[int] = None,
    ) -> None:
        self.bit_lut = bit_lut
        self.one_symbol_schemata = one_symbol_schemata
        self.two_symbol_schemata = two_symbol_schemata
        self.positional_expression = positional_expression
        self.one_symbol_is_prime_implicant = None  # type: Optional[bool]
        self.representations_consistent = None  # type: Optional[bool]
        if bit_lut is not None:
            self._k = bit_lut.k
        elif k is not None:
            self._k = int(k)
        elif one_symbol_schemata:
            # Infer k from the first schema string when only schemata are given.
            inferred = None
            for piset in one_symbol_schemata.values():
                for schema in piset:
                    inferred = len(schema)
                    break
                if inferred is not None:
                    break
            self._k = inferred if inferred is not None else 0
        elif two_symbol_schemata:
            inferred = None
            for polarity_list in two_symbol_schemata:
                for entry in polarity_list or ():
                    schema = entry[0]
                    if isinstance(schema, str):
                        inferred = len(schema)
                    else:
                        inferred = len(schema)
                    break
                if inferred is not None:
                    break
            self._k = inferred if inferred is not None else 0
        else:
            self._k = 0

    @classmethod
    def from_sequence(cls, seq: Sequence[object]) -> "FunctionOutputs":
        """Build from a dense 0/1 sequence (packs into a :class:`BitLut`)."""
        if isinstance(seq, FunctionOutputs):
            return seq
        if isinstance(seq, BitLut):
            return cls(bit_lut=seq)
        return cls(bit_lut=BitLut.from_sequence(seq))

    @property
    def k(self) -> int:
        """Number of inputs inferred from the dense LUT or set explicitly."""
        if self.bit_lut is not None:
            return self.bit_lut.k
        return self._k

    def eval(self, statenum: int) -> int:
        """Evaluate the function at input state ``statenum``.

        Prefers ``bit_lut``.  If absent, materializes via :meth:`get_bit_lut`
        from one-symbol schemata, two-symbol schemata (via one-symbol), or a
        positional expression.
        """
        if self.bit_lut is not None:
            return self.bit_lut.eval(statenum)
        return self.get_bit_lut().eval(statenum)

    def to_list(self) -> List[int]:
        """Unpack the dense LUT to a ``list`` of ints."""
        return self.get_bit_lut().to_list()

    def is_constant(self) -> Optional[int]:
        """Return ``0``/``1`` if the function is constant, else ``None``.

        Uses ``bit_lut`` when present; otherwise recognizes positional
        ``True``/``False`` expressions with ``k=0``.
        """
        if self.bit_lut is not None:
            return self.bit_lut.is_constant()
        if self.positional_expression is not None and self._k == 0:
            if self.positional_expression in ("True", "true", "1"):
                return 1
            if self.positional_expression in ("False", "false", "0"):
                return 0
        return None

    # --- lazy getters (peer slots → requested representation) ---

    def get_bit_lut(self) -> BitLut:
        """Return the dense LUT, building from schemata or expression if needed.

        Tries stored ``bit_lut``, then one-symbol, then positional_expression, then
        two-symbol (expand to one-symbol, then build the LUT).  Intermediates
        are cached.  Raises ``ValueError`` if no peer can produce a LUT.
        """
        if self.bit_lut is not None:
            return self.bit_lut
        if self.one_symbol_schemata is not None:
            return self.build_bit_lut_from_one_symbol_schemata()
        if self.positional_expression is not None:
            return self.build_bit_lut_from_positional_expression()
        if self.two_symbol_schemata is not None:
            self.get_one_symbol_schemata()
            return self.build_bit_lut_from_one_symbol_schemata()
        raise ValueError(
            "Cannot get bit_lut: neither bit_lut, one_symbol_schemata, "
            "two_symbol_schemata, nor positional_expression is available"
        )

    def get_one_symbol_schemata(self) -> OneSymbolSchemata:
        """Return one-symbol schemata, deriving from any available peer slot.

        Preference order:

        1. stored ``one_symbol_schemata``
        2. expand ``two_symbol_schemata`` (does not set
           :attr:`one_symbol_is_prime_implicant`)
        3. expression cascade (syntactic ROF, then Primer-B) when
           ``positional_expression`` is set — stores schemata only, not a LUT
        4. Quine–McCluskey from stored ``bit_lut``, or build+store ``bit_lut``
           from ``positional_expression`` when ``k`` is small enough, then QM

        Intermediates are cached.  Raises ``ValueError`` if no peer can produce
        one-symbol schemata.
        """
        if self.one_symbol_schemata is not None:
            return self.one_symbol_schemata
        if self.two_symbol_schemata is not None:
            self.one_symbol_schemata = one_symbol_from_two_symbol(
                self.two_symbol_schemata
            )
            return self.one_symbol_schemata

        # Expression cascade before any LUT/QM (high-k friendly).
        if self.positional_expression is not None:
            from cana.primes.cascade import on_off_primes_from_expression
            from cana.function_outputs.one_symbol import (
                primes_to_one_symbol_schemata,
            )

            result = on_off_primes_from_expression(self.positional_expression)
            if (
                result.ok
                and result.on_primes is not None
                and result.off_primes is not None
            ):
                self.one_symbol_schemata = primes_to_one_symbol_schemata(
                    result.on_primes,
                    result.off_primes,
                    input_names=positional_names(self.k),
                )
                self.one_symbol_is_prime_implicant = True
                return self.one_symbol_schemata

        if self.bit_lut is not None:
            return self.compute_one_symbol_schemata_from_bit_lut()

        if self.positional_expression is not None:
            # Same QM fallback as one_symbol_schemata_from_positional_expression,
            # but cache the dense LUT on this container.
            from cana.primes.util import MAX_DENSE_LUT_K

            if self.k > MAX_DENSE_LUT_K:
                raise ValueError(
                    "Cannot get one_symbol_schemata from positional_expression: "
                    "expression cascade unsuccessful and k={0:d} > "
                    "MAX_DENSE_LUT_K={1:d} (refusing dense LUT/QM)".format(
                        self.k, MAX_DENSE_LUT_K
                    )
                )
            self.build_bit_lut_from_positional_expression()
            return self.compute_one_symbol_schemata_from_bit_lut()

        raise ValueError(
            "Cannot get one_symbol_schemata: neither one_symbol_schemata, "
            "two_symbol_schemata, bit_lut, nor positional_expression is available"
        )

    def get_two_symbol_schemata(self) -> TwoSymbolSchemata:
        """Return two-symbol schemata, deriving from any available peer slot.

        Prefer stored two-symbol; else derive from ``one_symbol_schemata``
        (soft-warns unless :attr:`one_symbol_is_prime_implicant` is ``True``);
        else obtain primes via :meth:`get_one_symbol_schemata` (expression
        cascade, then bit_lut/QM) and derive.  Intermediates are cached.  Does
        not re-run QM over a stored cover; call
        :meth:`verify_one_symbol_schemata` to check explicitly.  Raises
        ``ValueError`` if no peer can produce two-symbol schemata.
        """
        if self.two_symbol_schemata is not None:
            return self.two_symbol_schemata

        if self.one_symbol_schemata is not None:
            if self.one_symbol_is_prime_implicant is not True:
                warnings.warn(
                    "using one_symbol_schemata "
                    "(one_symbol_is_prime_implicant={0!r}); "
                    "two-symbol derivation assumes a prime-implicant cover. "
                    "Call verify_one_symbol_schemata() to check.".format(
                        self.one_symbol_is_prime_implicant
                    ),
                    UserWarning,
                    stacklevel=2,
                )
            return self._two_symbol_from_one_symbol(self.one_symbol_schemata)

        one_symbol = self.get_one_symbol_schemata()
        return self._two_symbol_from_one_symbol(one_symbol)

    def get_positional_expression(self) -> Any:
        """Return positional expression IR, deriving from any available peer.

        Prefer stored ``positional_expression``; else build from ``two_symbol_schemata``,
        then ``one_symbol_schemata``, then ``bit_lut`` (positional ``x0``…
        ``x{k-1}`` python IR).  Does not call :meth:`get_bit_lut` so an empty
        container fails here rather than via a side-effect materialization.
        Raises ``ValueError`` if no peer can produce an expression.

        Named / styled export stays in
        :func:`~cana.function_outputs.expression.function_outputs_to_named_expression`.
        """
        if self.positional_expression is not None:
            return self.positional_expression
        names = positional_names(self.k)
        if self.two_symbol_schemata is not None:
            self.positional_expression = two_symbol_to_expression(
                names, self.two_symbol_schemata
            )
            return self.positional_expression
        if self.one_symbol_schemata is not None:
            self.positional_expression = one_symbol_to_expression(
                names, self.one_symbol_schemata
            )
            return self.positional_expression
        if self.bit_lut is not None:
            self.positional_expression = lut_to_expression(names, self.bit_lut)
            return self.positional_expression
        raise ValueError(
            "Cannot get positional_expression: neither positional_expression, two_symbol_schemata, "
            "one_symbol_schemata, nor bit_lut is available"
        )

    # --- named converters (fill / derive a peer slot) ---

    def build_bit_lut_from_positional_expression(self) -> BitLut:
        """Build and store ``bit_lut`` from ``positional_expression``."""
        if self.positional_expression is None:
            raise ValueError(
                "build_bit_lut_from_positional_expression requires "
                "positional_expression"
            )
        self.bit_lut = positional_expression_to_bit_lut(self.positional_expression, self.k)
        self._k = self.bit_lut.k
        return self.bit_lut

    def build_bit_lut_from_one_symbol_schemata(self) -> BitLut:
        """Build and store ``bit_lut`` from ``one_symbol_schemata``."""
        if self.one_symbol_schemata is None:
            raise ValueError(
                "build_bit_lut_from_one_symbol_schemata requires "
                "one_symbol_schemata"
            )
        self.bit_lut = bit_lut_from_one_symbol_schemata(
            self.one_symbol_schemata, self.k
        )
        self._k = self.bit_lut.k
        return self.bit_lut

    # Module-level converter; kept as a static for callers that used FO.
    bit_lut_from_one_symbol_schemata = staticmethod(
        bit_lut_from_one_symbol_schemata
    )

    def compute_one_symbol_schemata_from_bit_lut(self) -> OneSymbolSchemata:
        """Compute and store one-symbol schemata from ``bit_lut`` via Quine-McCluskey."""
        if self.bit_lut is None:
            raise ValueError(
                "compute_one_symbol_schemata_from_bit_lut requires bit_lut"
            )
        schemata = one_symbol_schemata_from_bit_lut(self.bit_lut)
        self.one_symbol_schemata = schemata
        self.one_symbol_is_prime_implicant = True
        self._k = self.bit_lut.k
        return self.one_symbol_schemata

    @staticmethod
    def _one_symbol_schemata_from_bit_lut(bit_lut: BitLut) -> OneSymbolSchemata:
        """Run Quine-McCluskey on ``bit_lut`` without mutating a container."""
        return one_symbol_schemata_from_bit_lut(bit_lut)

    def _two_symbol_from_one_symbol(
        self, one_symbol: OneSymbolSchemata
    ) -> TwoSymbolSchemata:
        """Derive and store two-symbol schemata from a one-symbol cover."""
        # Local import avoids a circular import at module load time.
        import cana.canalization.boolean_canalization as BCanalization

        k = self.k
        pi0 = one_symbol.get(0, set())
        pi1 = one_symbol.get(1, set())
        self.two_symbol_schemata = (
            BCanalization.find_two_symbols_v2(k=k, prime_implicants=pi0),
            BCanalization.find_two_symbols_v2(k=k, prime_implicants=pi1),
        )
        return self.two_symbol_schemata

    # --- opt-in verification ---

    def verify_one_symbol_schemata(self) -> bool:
        """Opt-in: stored one-symbol schemata are complete, consistent, and PIs.

        Builds a LUT from the stored schemata (fails on conflict/incompleteness),
        re-runs Quine-McCluskey, and checks set equality with the stored cover.
        Sets :attr:`one_symbol_is_prime_implicant`.  Never called by normal
        canalization paths.
        """
        if self.one_symbol_schemata is None:
            raise ValueError(
                "verify_one_symbol_schemata requires one_symbol_schemata"
            )
        schemata = self.one_symbol_schemata
        try:
            lut = bit_lut_from_one_symbol_schemata(schemata, self.k)
        except ValueError:
            self.one_symbol_is_prime_implicant = False
            return False
        recomputed = one_symbol_schemata_from_bit_lut(lut)
        ok = (
            set(schemata.get(0, ())) == set(recomputed.get(0, ()))
            and set(schemata.get(1, ())) == set(recomputed.get(1, ()))
        )
        self.one_symbol_is_prime_implicant = ok
        return ok

    def verify_representations_consistent(self) -> bool:
        """Opt-in: all convertible representations describe the same function.

        Derives a LUT from each populated slot that can produce one (stored
        ``bit_lut``, ``one_symbol_schemata``, ``two_symbol_schemata``, and
        positional ``positional_expression``) and checks they agree.  Sets
        :attr:`representations_consistent`.  Never called by normal
        canalization paths.
        """
        candidates = []  # type: List[BitLut]
        if self.bit_lut is not None:
            candidates.append(self.bit_lut)
        if self.one_symbol_schemata is not None:
            try:
                candidates.append(
                    bit_lut_from_one_symbol_schemata(
                        self.one_symbol_schemata, self.k
                    )
                )
            except ValueError:
                self.representations_consistent = False
                return False
        if self.positional_expression is not None:
            try:
                candidates.append(
                    positional_expression_to_bit_lut(self.positional_expression, self.k)
                )
            except ValueError:
                self.representations_consistent = False
                return False
        if self.two_symbol_schemata is not None:
            try:
                expanded = one_symbol_from_two_symbol(self.two_symbol_schemata)
                candidates.append(
                    bit_lut_from_one_symbol_schemata(expanded, self.k)
                )
            except ValueError:
                self.representations_consistent = False
                return False
        if len(candidates) < 2:
            self.representations_consistent = True
            return True
        first = candidates[0]
        ok = all(c == first for c in candidates[1:])
        self.representations_consistent = ok
        return ok

    def reorder(
        self, old_labels: Sequence[object], new_labels: Sequence[object]
    ) -> "FunctionOutputs":
        """Return a copy with input order remapped across filled peer reps.

        Remaps ``bit_lut``, ``one_symbol_schemata``, ``two_symbol_schemata``,
        and positional ``positional_expression`` when present (no QM).  Preserves
        ``one_symbol_is_prime_implicant``.  If orders already match, returns
        ``self`` unchanged.
        """
        if list(old_labels) == list(new_labels):
            return self

        new_bit_lut = None
        if self.bit_lut is not None:
            new_bit_lut = self.bit_lut.reorder(old_labels, new_labels)

        new_one_symbol = None
        if self.one_symbol_schemata is not None:
            new_one_symbol = reorder_one_symbol(
                self.one_symbol_schemata, old_labels, new_labels
            )

        new_two_symbol = None
        if self.two_symbol_schemata is not None:
            new_two_symbol = reorder_two_symbol(
                self.two_symbol_schemata, old_labels, new_labels
            )

        new_expression = None
        if self.positional_expression is not None:
            try:
                new_expression = reorder_positional(
                    self.positional_expression, old_labels, new_labels
                )
            except (ValueError, KeyError):
                new_expression = None

        out = FunctionOutputs(
            bit_lut=new_bit_lut,
            one_symbol_schemata=new_one_symbol,
            two_symbol_schemata=new_two_symbol,
            positional_expression=new_expression,
            k=self.k,
        )
        out.one_symbol_is_prime_implicant = self.one_symbol_is_prime_implicant
        # Peers remapped with the same permutation stay consistent iff they were.
        out.representations_consistent = self.representations_consistent
        return out

    def __len__(self) -> int:
        return len(self.get_bit_lut())

    def __getitem__(self, index: Union[int, slice]) -> Union[int, List[int]]:
        return self.get_bit_lut()[index]

    def __setitem__(self, index: int, value: object) -> None:
        self.get_bit_lut()[index] = value

    def __iter__(self) -> Iterator[int]:
        return iter(self.get_bit_lut())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FunctionOutputs):
            if self.bit_lut is not None and other.bit_lut is not None:
                return self.bit_lut == other.bit_lut
            return (
                self.bit_lut == other.bit_lut
                and self.one_symbol_schemata == other.one_symbol_schemata
                and self.two_symbol_schemata == other.two_symbol_schemata
                and self.positional_expression == other.positional_expression
            )
        if self.bit_lut is not None and isinstance(other, Sequence) and not isinstance(
            other, (str, bytes, bytearray)
        ):
            return self.bit_lut == other
        return NotImplemented

    def __repr__(self) -> str:
        parts = []
        if self.bit_lut is not None:
            parts.append("bit_lut={0!r}".format(self.bit_lut))
        if self.one_symbol_schemata is not None:
            parts.append("one_symbol_schemata=...")
        if self.two_symbol_schemata is not None:
            parts.append("two_symbol_schemata=...")
        if self.positional_expression is not None:
            parts.append("positional_expression={0!r}".format(self.positional_expression))
        if self.one_symbol_is_prime_implicant is not None:
            parts.append(
                "one_symbol_is_prime_implicant={0!r}".format(
                    self.one_symbol_is_prime_implicant
                )
            )
        if self.representations_consistent is not None:
            parts.append(
                "representations_consistent={0!r}".format(
                    self.representations_consistent
                )
            )
        return "FunctionOutputs({0})".format(", ".join(parts) or "k={0}".format(self._k))
