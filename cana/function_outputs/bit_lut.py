# -*- coding: utf-8 -*-
"""Bit-packed dense Boolean look-up tables."""
from __future__ import annotations

from typing import Iterator, List, Optional, Sequence, Union

from cana.cutils import binstate_to_statenum, statenum_to_binstate

__all__ = ["BitLut", "MAX_DENSE_LUT_K", "refuse_dense_lut_k"]

# Building a dense BitLut from another representation allocates Θ(2^k).
MAX_DENSE_LUT_K = 20


def refuse_dense_lut_k(k: int, *, via: str) -> None:
    """Raise ``ValueError`` when materializing a dense BitLut would be too large."""
    if int(k) > MAX_DENSE_LUT_K:
        raise ValueError(
            "refusing to build dense BitLut via {0}: k={1:d} > MAX_DENSE_LUT_K={2:d}".format(
                via, int(k), MAX_DENSE_LUT_K
            )
        )


def _coerce_bit(value: object) -> int:
    bit = int(value)
    if bit not in (0, 1):
        raise ValueError("LUT entries must be 0 or 1; got {0!r}".format(value))
    return bit


def _k_from_length(n: int) -> int:
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(
            "LUT length must be a power of two (2**k); got {0:d}.".format(n)
        )
    return n.bit_length() - 1


class BitLut(object):
    """Dense Boolean truth table packed 8 entries per byte.

    Row ``i`` (input state number, matching
    :func:`~cana.cutils.statenum_to_binstate`) lives in bit ``(i & 7)`` of
    byte ``i >> 3`` (LSB-first within each byte).  Indexing and iteration
    yield ``int`` values in ``{0, 1}``.
    """

    __slots__ = ("_k", "_n", "_data")

    def __init__(self, k: int = 0, *, fill: int = 0) -> None:
        if k < 0:
            raise ValueError("k must be >= 0")
        self._k = int(k)
        self._n = 1 << self._k if self._k > 0 else 1
        nbytes = (self._n + 7) // 8
        bit = _coerce_bit(fill)
        self._data = bytearray([0xFF if bit else 0x00] * nbytes)
        if bit and (self._n & 7):
            # Clear unused high bits in the last byte.
            self._data[-1] &= (1 << (self._n & 7)) - 1

    @classmethod
    def from_sequence(cls, seq: Sequence[object]) -> "BitLut":
        """Build a ``BitLut`` from any 0/1 sequence of length ``2**k``."""
        n = len(seq)
        k = _k_from_length(n)
        lut = cls(k)
        data = lut._data
        for i, value in enumerate(seq):
            if _coerce_bit(value):
                data[i >> 3] |= 1 << (i & 7)
            # else already zero
        return lut

    @classmethod
    def from_bytes(cls, data: Union[bytes, bytearray], k: int) -> "BitLut":
        """Build from packed bytes with the same layout as :meth:`to_bytes`."""
        lut = cls(k)
        raw = bytes(data)
        if len(raw) != len(lut._data):
            raise ValueError(
                "expected {0:d} packed bytes for k={1:d}; got {2:d}".format(
                    len(lut._data), k, len(raw)
                )
            )
        lut._data[:] = raw
        if lut._n & 7:
            lut._data[-1] &= (1 << (lut._n & 7)) - 1
        return lut

    @property
    def k(self) -> int:
        """Number of inputs (``len(self) == 2**k``, or 1 when ``k == 0``)."""
        return self._k

    def eval(self, statenum: int) -> int:
        """Return the output for input state number ``statenum``."""
        return self[statenum]

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: Union[int, slice]) -> Union[int, List[int]]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self._n))]
        i = index if index >= 0 else index + self._n
        if i < 0 or i >= self._n:
            raise IndexError("BitLut index out of range")
        return (self._data[i >> 3] >> (i & 7)) & 1

    def __setitem__(self, index: int, value: object) -> None:
        i = index if index >= 0 else index + self._n
        if i < 0 or i >= self._n:
            raise IndexError("BitLut index out of range")
        bit = _coerce_bit(value)
        byte_i = i >> 3
        mask = 1 << (i & 7)
        if bit:
            self._data[byte_i] |= mask
        else:
            self._data[byte_i] &= ~mask & 0xFF

    def __iter__(self) -> Iterator[int]:
        data = self._data
        n = self._n
        for i in range(n):
            yield (data[i >> 3] >> (i & 7)) & 1

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BitLut):
            return self._n == other._n and self._data == other._data
        if isinstance(other, Sequence) and not isinstance(other, (str, bytes, bytearray)):
            if len(other) != self._n:
                return False
            return all(self[i] == _coerce_bit(other[i]) for i in range(self._n))
        return NotImplemented

    def __repr__(self) -> str:
        if self._n <= 16:
            body = ",".join(str(v) for v in self)
            return "BitLut.from_sequence([{0}])".format(body)
        return "BitLut(k={0})".format(self._k)

    def __copy__(self) -> "BitLut":
        return BitLut.from_bytes(self._data, self._k)

    def __deepcopy__(self, memo: dict) -> "BitLut":
        return self.__copy__()

    def to_list(self) -> List[int]:
        """Return an unpacked ``list`` of ``0``/``1`` ints."""
        return list(self)

    def to_bytes(self) -> bytes:
        """Return a copy of the packed byte buffer."""
        return bytes(self._data)

    def fill(self, value: object) -> None:
        """Set every row to ``0`` or ``1``."""
        bit = _coerce_bit(value)
        if bit:
            self._data[:] = b"\xff" * len(self._data)
            if self._n & 7:
                self._data[-1] &= (1 << (self._n & 7)) - 1
        else:
            self._data[:] = b"\x00" * len(self._data)

    def is_constant(self) -> Optional[int]:
        """Return ``0`` or ``1`` if every row equals that value, else ``None``."""
        if self._n == 0:
            return None
        first = self[0]
        data = self._data
        if first == 0:
            # All used bits must be clear.
            full, rem = divmod(self._n, 8)
            if any(data[i] for i in range(full)):
                return None
            if rem and (data[full] & ((1 << rem) - 1)):
                return None
            return 0
        # first == 1: all used bits set
        full, rem = divmod(self._n, 8)
        if any(data[i] != 0xFF for i in range(full)):
            return None
        if rem and (data[full] & ((1 << rem) - 1)) != ((1 << rem) - 1):
            return None
        return 1

    def popcount(self) -> int:
        """Number of rows equal to ``1``."""
        # int.bit_count is 3.10+
        return sum(byte.bit_count() for byte in self._data)

    def negate_input(self, idx: int) -> "BitLut":
        """Return a copy with input bit ``idx`` negated (MSB-first bit order).

        Row ``i`` of the result equals row ``i ^ (1 << (k - 1 - idx))`` of
        ``self``, matching the legacy ``negate_LUT_input`` remapping.
        """
        k = self._k
        if idx < 0 or idx >= k:
            raise IndexError(
                "input index {0:d} out of range for BitLut.k={1:d}".format(idx, k)
            )
        mask = 1 << (k - 1 - idx)
        out = BitLut(k)
        for i in range(self._n):
            out[i] = self[i ^ mask]
        return out

    def lut_as_str(self) -> str:
        """Return a multi-line string of ``binstate : output`` rows."""
        k = self._k
        lines = [
            "{0} : {1}".format(statenum_to_binstate(i, base=k), self[i])
            for i in range(self._n)
        ]
        return "\n".join(lines)

    def reorder(
        self, old_labels: Sequence[object], new_labels: Sequence[object]
    ) -> "BitLut":
        """Return a copy with input-bit order remapped from ``old_labels`` to ``new_labels``.

        Labels identify the ``k`` input variables in MSB-first bit order.
        ``new_labels`` must be a permutation of ``old_labels``.  If the orders
        already match, returns ``self`` unchanged.
        """
        if list(old_labels) == list(new_labels):
            return self
        k = len(old_labels)
        if k != self._k:
            raise ValueError(
                "old_labels length {0:d} does not match BitLut.k={1:d}".format(
                    k, self._k
                )
            )
        if k != len(new_labels) or set(old_labels) != set(new_labels):
            raise ValueError("old_labels and new_labels must be permutations")
        name_to_old = {n: i for i, n in enumerate(old_labels)}
        new_lut = BitLut(k)
        for new_state in range(self._n):
            new_bits = statenum_to_binstate(new_state, k)
            old_bits = ["0"] * k
            for new_i, name in enumerate(new_labels):
                old_bits[name_to_old[name]] = new_bits[new_i]
            old_state = binstate_to_statenum("".join(old_bits))
            new_lut[new_state] = self[old_state]
        return new_lut

    @classmethod
    def reorder_sequence(
        cls,
        lut: Sequence[object],
        old_labels: Sequence[object],
        new_labels: Sequence[object],
    ) -> "BitLut":
        """Like :meth:`reorder`, but accepts any 0/1 sequence or ``BitLut``."""
        if isinstance(lut, BitLut):
            source = lut
        else:
            source = cls.from_sequence(lut)
        return source.reorder(old_labels, new_labels)
