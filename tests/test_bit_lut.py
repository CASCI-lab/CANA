# -*- coding: utf-8 -*-
"""Tests for :class:`cana.function_outputs.bit_lut.BitLut`."""
from copy import copy, deepcopy

import pytest

from cana.function_outputs.bit_lut import BitLut


def test_from_sequence_roundtrip():
    seq = [0, 0, 0, 1]
    lut = BitLut.from_sequence(seq)
    assert lut.k == 2
    assert len(lut) == 4
    assert list(lut) == seq
    assert lut.to_list() == seq
    assert lut == seq
    assert lut == BitLut.from_sequence(seq)


def test_k0_constant():
    lut = BitLut.from_sequence([1])
    assert lut.k == 0
    assert len(lut) == 1
    assert lut[0] == 1
    assert lut.is_constant() == 1
    assert BitLut.from_sequence([0]).is_constant() == 0


def test_get_set_and_pack_layout():
    lut = BitLut(3)  # 8 zeros
    lut[0] = 1
    lut[7] = 1
    assert lut[0] == 1
    assert lut[1] == 0
    assert lut[7] == 1
    # LSB of first byte is state 0
    assert lut.to_bytes()[0] & 0x01
    assert lut.to_bytes()[0] & 0x80


def test_from_str_sequence():
    lut = BitLut.from_sequence(list("0110"))
    assert list(lut) == [0, 1, 1, 0]


def test_reject_bad_length():
    with pytest.raises(ValueError):
        BitLut.from_sequence([0, 1, 0])


def test_reject_bad_bit():
    with pytest.raises(ValueError):
        BitLut.from_sequence([0, 2, 0, 1])


def test_fill_popcount_constant():
    lut = BitLut(2)
    lut.fill(1)
    assert lut.popcount() == 4
    assert lut.is_constant() == 1
    lut.fill(0)
    assert lut.popcount() == 0
    assert lut.is_constant() == 0
    lut[1] = 1
    assert lut.is_constant() is None
    assert lut.popcount() == 1


def test_copy_deepcopy():
    lut = BitLut.from_sequence([0, 1, 1, 0])
    c1 = copy(lut)
    c2 = deepcopy(lut)
    assert c1 == lut and c2 == lut
    assert c1 is not lut
    c1[0] = 1
    assert lut[0] == 0


def test_eval_and_slice():
    lut = BitLut.from_sequence([0, 1, 1, 0])
    assert lut.eval(2) == 1
    assert lut[1:3] == [1, 1]


def test_from_bytes_roundtrip():
    lut = BitLut.from_sequence([1, 0, 1, 1, 0, 0, 0, 1])
    again = BitLut.from_bytes(lut.to_bytes(), k=3)
    assert again == lut


def test_reorder_identity_returns_self():
    lut = BitLut.from_sequence([0, 0, 0, 1])  # AND on (A, B)
    assert lut.reorder(["A", "B"], ["A", "B"]) is lut


def test_reorder_swaps_input_bits():
    # AND with order (A, B): only 11 -> 1  => [0,0,0,1]
    # After swap to (B, A): still AND => [0,0,0,1]
    and_lut = BitLut.from_sequence([0, 0, 0, 1])
    assert list(and_lut.reorder(["A", "B"], ["B", "A"])) == [0, 0, 0, 1]

    # Implies A->B with order (A, B): outputs for 00,01,10,11 = 1,1,0,1
    implies = BitLut.from_sequence([1, 1, 0, 1])
    # Swap to (B, A): new state bits are (B,A); old was (A,B)
    # new 00 -> old 00 -> 1
    # new 01 -> old 10 -> 0
    # new 10 -> old 01 -> 1
    # new 11 -> old 11 -> 1
    swapped = implies.reorder(["A", "B"], ["B", "A"])
    assert list(swapped) == [1, 0, 1, 1]


def test_reorder_sequence_from_list():
    lut = BitLut.reorder_sequence([1, 1, 0, 1], ["A", "B"], ["B", "A"])
    assert list(lut) == [1, 0, 1, 1]
