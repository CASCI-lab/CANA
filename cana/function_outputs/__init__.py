# -*- coding: utf-8 -*-
"""Multi-representation Boolean function outputs for a node.

``FunctionOutputs`` is what :attr:`cana.boolean_node.BooleanNode.outputs`
stores.  Peer representation slots may hold a dense
:class:`~cana.bit_lut.BitLut`, one-symbol (prime-implicant) schemata,
two-symbol schemata, and/or an expression.  Multiple representations may
coexist; named converters document each derivation, and lazy getters fill
a missing slot from another available slot when possible.

Dense Sequence access (``len``, ``[i]``, iterate) and :meth:`eval` prefer
``bit_lut`` and materialize it from one-symbol schemata, two-symbol
schemata (via one-symbol), or a positional expression via
:meth:`get_bit_lut` when needed.
"""
from __future__ import annotations

import warnings
from typing import Any, Iterator, List, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

from cana.bit_lut import BitLut
from cana.function_outputs.expression import (
    bit_lut_from_expression,
    reorder_positional,
)
from cana.function_outputs.one_symbol import (
    OneSymbolSchemata,
    bit_lut_from_schemata,
    reorder as reorder_one_symbol,
    schemata_from_bit_lut,
)
from cana.function_outputs.two_symbol import (
    one_symbol_from_two_symbol,
    reorder as reorder_two_symbol,
    two_symbol_to_expression,
)

__all__ = [
    "BooleanEval",
    "FunctionOutputs",
    "OneSymbolSchemata",
    "TwoSymbolSchemata",
    "one_symbol_from_two_symbol",
    "two_symbol_to_expression",
]

# (ts0, ts1); each entry is [schema_str, perm_groups, same_groups].
TwoSymbolSchemata = Tuple[list, list]


@runtime_checkable
class BooleanEval(Protocol):
    """Minimal interface for evaluating a Boolean update function."""

    @property
    def k(self) -> int:
        """Number of input variables."""

    def eval(self, statenum: int) -> int:
        """Return the next value for input state number ``statenum``."""


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
    expression :
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
        "expression",
        "one_symbol_is_prime_implicant",
        "representations_consistent",
        "_k",
    )

    def __init__(
        self,
        bit_lut: Optional[BitLut] = None,
        one_symbol_schemata: Optional[OneSymbolSchemata] = None,
        two_symbol_schemata: Optional[TwoSymbolSchemata] = None,
        expression: Optional[Any] = None,
        k: Optional[int] = None,
    ) -> None:
        self.bit_lut = bit_lut
        self.one_symbol_schemata = one_symbol_schemata
        self.two_symbol_schemata = two_symbol_schemata
        self.expression = expression
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

    def get_one_symbol_schemata(self) -> OneSymbolSchemata:
        """Return one-symbol schemata, deriving from two-symbol or ``bit_lut``.

        Prefer stored one-symbol; else expand ``two_symbol_schemata`` (does not
        set :attr:`one_symbol_is_prime_implicant`); else Quine–McCluskey from
        ``bit_lut``.
        """
        if self.one_symbol_schemata is not None:
            return self.one_symbol_schemata
        if self.two_symbol_schemata is not None:
            self.one_symbol_schemata = one_symbol_from_two_symbol(
                self.two_symbol_schemata
            )
            return self.one_symbol_schemata
        self.get_bit_lut()
        return self.compute_one_symbol_schemata_from_bit_lut()

    def get_bit_lut(self) -> BitLut:
        """Return the dense LUT, building from schemata or expression if needed.

        Two-symbol materializes in two steps: expand to one-symbol, then build
        the LUT from that cover.
        """
        if self.bit_lut is not None:
            return self.bit_lut
        if self.one_symbol_schemata is not None:
            return self.build_bit_lut_from_one_symbol_schemata()
        if self.expression is not None:
            return self.build_bit_lut_from_expression()
        if self.two_symbol_schemata is not None:
            self.get_one_symbol_schemata()
            return self.build_bit_lut_from_one_symbol_schemata()
        raise ValueError(
            "Cannot get bit_lut: neither bit_lut, one_symbol_schemata, "
            "two_symbol_schemata, nor expression is available"
        )

    def build_bit_lut_from_expression(self) -> BitLut:
        """Build and store ``bit_lut`` from positional ``expression``."""
        if self.expression is None:
            raise ValueError("build_bit_lut_from_expression requires expression")
        self.bit_lut = bit_lut_from_expression(self.expression, self.k)
        self._k = self.bit_lut.k
        return self.bit_lut

    def get_two_symbol_schemata(self) -> TwoSymbolSchemata:
        """Return two-symbol schemata, deriving from one-symbol if needed.

        Uses stored ``one_symbol_schemata`` when present (soft-warns unless
        :attr:`one_symbol_is_prime_implicant` is ``True``).  Otherwise
        computes one-symbol via Quine–McCluskey from ``bit_lut``.  Does not
        re-run QM over a stored cover; call :meth:`verify_one_symbol_schemata`
        to check explicitly.
        """
        if self.two_symbol_schemata is not None:
            return self.two_symbol_schemata
        # Local import avoids a circular import at module load time.
        import cana.canalization.boolean_canalization as BCanalization

        if (
            self.one_symbol_schemata is not None
            and self.one_symbol_is_prime_implicant is not True
        ):
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

        one_symbol = self.get_one_symbol_schemata()
        k = self.k
        pi0 = one_symbol.get(0, set())
        pi1 = one_symbol.get(1, set())
        self.two_symbol_schemata = (
            BCanalization.find_two_symbols_v2(k=k, prime_implicants=pi0),
            BCanalization.find_two_symbols_v2(k=k, prime_implicants=pi1),
        )
        return self.two_symbol_schemata

    def compute_one_symbol_schemata_from_bit_lut(self) -> OneSymbolSchemata:
        """Compute and store one-symbol schemata from ``bit_lut`` via Quine-McCluskey."""
        if self.bit_lut is None:
            raise ValueError(
                "compute_one_symbol_schemata_from_bit_lut requires bit_lut"
            )
        schemata = schemata_from_bit_lut(self.bit_lut)
        self.one_symbol_schemata = schemata
        self.one_symbol_is_prime_implicant = True
        self._k = self.bit_lut.k
        return self.one_symbol_schemata

    @staticmethod
    def bit_lut_from_one_symbol_schemata(
        schemata: OneSymbolSchemata, k: int
    ) -> BitLut:
        """Build a :class:`BitLut` from one-symbol schemata without mutating self.

        Raises
        ------
        ValueError
            On polarity conflicts or incomplete coverage of the ``2**k`` rows.
        """
        return bit_lut_from_schemata(schemata, k)

    def build_bit_lut_from_one_symbol_schemata(self) -> BitLut:
        """Build and store ``bit_lut`` from ``one_symbol_schemata``."""
        if self.one_symbol_schemata is None:
            raise ValueError(
                "build_bit_lut_from_one_symbol_schemata requires "
                "one_symbol_schemata"
            )
        self.bit_lut = bit_lut_from_schemata(self.one_symbol_schemata, self.k)
        self._k = self.bit_lut.k
        return self.bit_lut

    @staticmethod
    def _one_symbol_schemata_from_bit_lut(bit_lut: BitLut) -> OneSymbolSchemata:
        """Run Quine-McCluskey on ``bit_lut`` without mutating a container."""
        return schemata_from_bit_lut(bit_lut)

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
            lut = bit_lut_from_schemata(schemata, self.k)
        except ValueError:
            self.one_symbol_is_prime_implicant = False
            return False
        recomputed = schemata_from_bit_lut(lut)
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
        positional ``expression``) and checks they agree.  Sets
        :attr:`representations_consistent`.  Never called by normal
        canalization paths.
        """
        candidates = []  # type: List[BitLut]
        if self.bit_lut is not None:
            candidates.append(self.bit_lut)
        if self.one_symbol_schemata is not None:
            try:
                candidates.append(
                    bit_lut_from_schemata(self.one_symbol_schemata, self.k)
                )
            except ValueError:
                self.representations_consistent = False
                return False
        if self.expression is not None:
            try:
                candidates.append(
                    bit_lut_from_expression(self.expression, self.k)
                )
            except ValueError:
                self.representations_consistent = False
                return False
        if self.two_symbol_schemata is not None:
            try:
                expanded = one_symbol_from_two_symbol(self.two_symbol_schemata)
                candidates.append(bit_lut_from_schemata(expanded, self.k))
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
        and positional ``expression`` when present (no QM).  Preserves
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
        if self.expression is not None:
            try:
                new_expression = reorder_positional(
                    self.expression, old_labels, new_labels
                )
            except (ValueError, KeyError):
                new_expression = None

        out = FunctionOutputs(
            bit_lut=new_bit_lut,
            one_symbol_schemata=new_one_symbol,
            two_symbol_schemata=new_two_symbol,
            expression=new_expression,
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
                and self.expression == other.expression
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
        if self.expression is not None:
            parts.append("expression={0!r}".format(self.expression))
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
        if self.expression is not None and self._k == 0:
            if self.expression in ("True", "true", "1"):
                return 1
            if self.expression in ("False", "false", "0"):
                return 0
        return None
