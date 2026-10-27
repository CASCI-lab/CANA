# -*- coding: utf-8 -*-
"""One-symbol (wildcard) schemata helpers.

Schemata are thin ``dict`` maps polarity ``0``/``1`` → sets of length-``k``
strings over ``{0,1,2}`` (``2`` = don't-care).  No wrapper class.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Set

from cana.bit_lut import BitLut
from cana.cutils import binstate_to_statenum, outputs_to_binstates_of_given_type

__all__ = [
    "OneSymbolSchemata",
    "reorder",
    "bit_lut_from_schemata",
    "schemata_from_bit_lut",
]

# Polarity -> set of length-k strings over {0,1,2} (2 = don't-care).
OneSymbolSchemata = Dict[int, Set[str]]


def _infer_k(schemata: OneSymbolSchemata) -> Optional[int]:
    for piset in schemata.values():
        for schema in piset:
            return len(schema)
    return None


def reorder(
    schemata: OneSymbolSchemata,
    old_labels: Sequence[object],
    new_labels: Sequence[object],
) -> OneSymbolSchemata:
    """Return a copy with schema character positions remapped to ``new_labels``.

    Labels identify the ``k`` input variables in MSB-first bit order (same
    convention as :meth:`~cana.bit_lut.BitLut.reorder`).  ``new_labels`` must
    be a permutation of ``old_labels``.  If the orders already match, returns
    ``schemata`` unchanged.
    """
    if list(old_labels) == list(new_labels):
        return schemata
    k = len(old_labels)
    if k != len(new_labels) or set(old_labels) != set(new_labels):
        raise ValueError("old_labels and new_labels must be permutations")
    inferred = _infer_k(schemata)
    if inferred is not None and inferred != k:
        raise ValueError(
            "old_labels length {0:d} does not match schema length {1:d}".format(
                k, inferred
            )
        )
    name_to_old = {n: i for i, n in enumerate(old_labels)}
    remapped = {}  # type: OneSymbolSchemata
    for polarity, piset in schemata.items():
        remapped[polarity] = {
            "".join(schema[name_to_old[name]] for name in new_labels)
            for schema in piset
        }
    return remapped


def bit_lut_from_schemata(schemata: OneSymbolSchemata, k: int) -> BitLut:
    """Build a :class:`~cana.bit_lut.BitLut` from one-symbol schemata.

    Raises
    ------
    ValueError
        On polarity conflicts or incomplete coverage of the ``2**k`` rows.
        Incomplete covers are not filled with a default output value.
    """
    import cana.canalization.cboolean_canalization as cBCanalization

    lut = BitLut(k)
    covered = [False] * len(lut)
    for output, piset in schemata.items():
        out_bit = int(output)
        for pi in piset:
            for binstate in cBCanalization.expand_wildcard_schemata(pi):
                idx = binstate_to_statenum(binstate)
                if covered[idx] and lut[idx] != out_bit:
                    raise ValueError(
                        "one-symbol schemata polarity conflict at "
                        "state {0!r}: {1:d} vs {2:d}".format(
                            binstate, lut[idx], out_bit
                        )
                    )
                lut[idx] = out_bit
                covered[idx] = True
    if not all(covered):
        missing = sum(1 for c in covered if not c)
        raise ValueError(
            "one-symbol schemata do not cover all 2**k states "
            "({0:d} uncovered for k={1:d})".format(missing, k)
        )
    return lut


def schemata_from_bit_lut(bit_lut: BitLut) -> OneSymbolSchemata:
    """Run Quine-McCluskey on ``bit_lut``; return one-symbol schemata."""
    import cana.canalization.cboolean_canalization as cBCanalization

    k = bit_lut.k
    schemata = {}  # type: OneSymbolSchemata
    for output in sorted(set(bit_lut)):
        output_binstates = outputs_to_binstates_of_given_type(
            bit_lut, output=output, k=k
        )
        schemata[output] = cBCanalization.find_implicants_qm(
            input_binstates=output_binstates
        )
    return schemata
