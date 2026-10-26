# -*- coding: utf-8 -*-
"""Multi-representation Boolean function outputs for a node.

``FunctionOutputs`` is what :attr:`cana.boolean_node.BooleanNode.outputs`
stores.  Today the dense slot is a :class:`~cana.bit_lut.BitLut`; later
slots may hold prime-implicant / one-symbol schemata and/or an expression
AST.  Multiple representations may coexist.

Dense Sequence access (``len``, ``[i]``, iterate) requires ``bit_lut``.
:meth:`eval` dispatches to whatever representation can answer.
"""
from __future__ import annotations

from typing import Any, Iterator, List, Optional, Protocol, Sequence, Union, runtime_checkable

from cana.bit_lut import BitLut

__all__ = ["BooleanEval", "FunctionOutputs"]


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
    schemata :
        Reserved for prime-implicant / one-symbol / wildcard covers
        (unused in this pass; canalization may migrate here later).
    expression :
        Reserved for a source expression string or AST (unused in this pass).
    """

    __slots__ = ("bit_lut", "schemata", "expression", "_k")

    def __init__(
        self,
        bit_lut: Optional[BitLut] = None,
        schemata: Optional[Any] = None,
        expression: Optional[Any] = None,
        k: Optional[int] = None,
    ) -> None:
        self.bit_lut = bit_lut
        self.schemata = schemata
        self.expression = expression
        if bit_lut is not None:
            self._k = bit_lut.k
        elif k is not None:
            self._k = int(k)
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

        Prefers ``bit_lut``.  Schemata / expression backends are not
        implemented yet; calling without a dense LUT raises ``ValueError``.
        """
        if self.bit_lut is not None:
            return self.bit_lut.eval(statenum)
        # Future: match schemata, then expression/AST.
        raise ValueError(
            "FunctionOutputs.eval requires a dense bit_lut "
            "(schemata/expression backends are not implemented yet)"
        )

    def _require_bit_lut(self) -> BitLut:
        if self.bit_lut is None:
            raise ValueError(
                "Dense LUT access requires bit_lut; this FunctionOutputs "
                "has no packed truth table (expression/schemata-only nodes "
                "cannot be indexed as a Sequence yet)"
            )
        return self.bit_lut

    def __len__(self) -> int:
        return len(self._require_bit_lut())

    def __getitem__(self, index: Union[int, slice]) -> Union[int, List[int]]:
        return self._require_bit_lut()[index]

    def __setitem__(self, index: int, value: object) -> None:
        self._require_bit_lut()[index] = value

    def __iter__(self) -> Iterator[int]:
        return iter(self._require_bit_lut())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FunctionOutputs):
            if self.bit_lut is not None and other.bit_lut is not None:
                return self.bit_lut == other.bit_lut
            return (
                self.bit_lut == other.bit_lut
                and self.schemata == other.schemata
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
        if self.schemata is not None:
            parts.append("schemata=...")
        if self.expression is not None:
            parts.append("expression={0!r}".format(self.expression))
        return "FunctionOutputs({0})".format(", ".join(parts) or "k={0}".format(self._k))

    def to_list(self) -> List[int]:
        """Unpack the dense LUT to a ``list`` of ints."""
        return self._require_bit_lut().to_list()

    def is_constant(self) -> Optional[int]:
        """Return ``0``/``1`` if the dense LUT is constant, else ``None``."""
        if self.bit_lut is None:
            return None
        return self.bit_lut.is_constant()
