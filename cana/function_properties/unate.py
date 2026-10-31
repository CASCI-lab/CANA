# -*- coding: utf-8 -*-
"""Semantic (truth-table) unate and monotone-increasing checks.

These predicates inspect a dense :class:`~cana.function_outputs.bit_lut.BitLut`.
They are distinct from *syntactic* unate checks on Boolean expressions
(e.g. the helpers under ``dev/high_k/unate.py``), which inspect identifier
polarity in source text without expanding a truth table.

Naming:
- ``is_input_*`` / ``input_sign`` — one variable (input index) of the LUT.
- ``input_signs`` — one sign per input (not a single whole-function flag).
- ``is_function_*`` — a property of the whole LUT / Boolean function.

An input ``idx`` is **positive unate** when flipping it from ``0`` to ``1``
never decreases the output, and **negative unate** when that flip never
increases the output.  A function is **monotone increasing** when every
input is positive unate, and **unate** when every input has a nonzero sign.
"""
from __future__ import annotations

from typing import List

from cana.function_outputs.bit_lut import BitLut

__all__ = [
    "is_input_positive_unate",
    "is_input_negative_unate",
    "input_sign",
    "input_signs",
    "is_function_monotone_increasing",
    "is_function_unate",
]


def _mask(k: int, idx: int) -> int:
    if idx < 0 or idx >= k:
        raise IndexError(
            "input index {0:d} out of range for k={1:d}".format(idx, k)
        )
    return 1 << (k - 1 - idx)


def is_input_positive_unate(lut: BitLut, idx: int) -> bool:
    """Return whether input ``idx`` is positive unate (activating)."""
    k = lut.k
    if k == 0:
        return True
    mask = _mask(k, idx)
    # Only check edges where bit idx is currently 0.
    for state in range(len(lut)):
        if state & mask:
            continue
        if lut[state] > lut[state ^ mask]:
            return False
    return True


def is_input_negative_unate(lut: BitLut, idx: int) -> bool:
    """Return whether input ``idx`` is negative unate (inhibiting)."""
    k = lut.k
    if k == 0:
        return True
    mask = _mask(k, idx)
    for state in range(len(lut)):
        if state & mask:
            continue
        if lut[state] < lut[state ^ mask]:
            return False
    return True


def input_sign(lut: BitLut, idx: int) -> int:
    """Return ``+1`` (activating), ``-1`` (inhibiting), or ``0`` (neither).

    Prefers positive unateness when both hold (e.g. irrelevant inputs).
    """
    if is_input_positive_unate(lut, idx):
        return 1
    if is_input_negative_unate(lut, idx):
        return -1
    return 0


def input_signs(lut: BitLut) -> List[int]:
    """Return the sign of each input of ``lut`` (length ``lut.k``)."""
    return [input_sign(lut, idx) for idx in range(lut.k)]


def is_function_monotone_increasing(lut: BitLut) -> bool:
    """Return whether the whole function is monotone increasing.

    True when every input is positive unate (isotone / monotone up).
    """
    return all(is_input_positive_unate(lut, idx) for idx in range(lut.k))


def is_function_unate(lut: BitLut) -> bool:
    """Return whether the whole function is unate.

    True when every input has a nonzero sign (``+1`` or ``-1``).
    """
    return all(sign != 0 for sign in input_signs(lut))
