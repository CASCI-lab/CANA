# -*- coding: utf-8 -*-
"""One-symbol (wildcard) schemata helpers.

Schemata are thin ``dict`` maps polarity ``0``/``1`` → sets of length-``k``
strings over ``{0,1,2}`` (``2`` = don't-care).  No wrapper class.

This is the public representation of prime covers for
:class:`~cana.function_outputs.FunctionOutputs`.  Named ``PrimeDict`` cubes
remain algorithm-internal inside :mod:`cana.primes`; convert at the boundary
with :func:`primes_to_one_symbol_schemata` /
:func:`one_symbol_schemata_to_primes`.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

from cana.bit_lut import BitLut
from cana.cutils import binstate_to_statenum, outputs_to_binstates_of_given_type

__all__ = [
    "OneSymbolSchemata",
    "reorder",
    "bit_lut_from_one_symbol_schemata",
    "one_symbol_schemata_from_bit_lut",
    "primes_to_one_symbol_schemata",
    "one_symbol_schemata_to_primes",
    "one_symbol_schemata_from_positional_expression",
]

# Polarity -> set of length-k strings over {0,1,2} (2 = don't-care).
OneSymbolSchemata = Dict[int, Set[str]]

# Named cube encoding used by primes algorithms (algo → public boundary).
_PrimeCube = Dict[str, bool]
_OnOffCubes = Tuple[List[_PrimeCube], List[_PrimeCube]]


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


def bit_lut_from_one_symbol_schemata(
    schemata: OneSymbolSchemata, k: int
) -> BitLut:
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


def one_symbol_schemata_from_bit_lut(bit_lut: BitLut) -> OneSymbolSchemata:
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


def _prime_to_schema(prime: _PrimeCube, names: Sequence[str]) -> str:
    chars: List[str] = []
    for name in names:
        if name not in prime:
            chars.append("2")
        elif prime[name]:
            chars.append("1")
        else:
            chars.append("0")
    return "".join(chars)


def primes_to_one_symbol_schemata(
    on_primes: List[_PrimeCube],
    off_primes: List[_PrimeCube],
    *,
    input_names: Optional[Sequence[str]] = None,
    k: Optional[int] = None,
) -> OneSymbolSchemata:
    """Convert named ON/OFF prime cubes to positional one-symbol schemata.

    *input_names* defaults to ``positional_names(k)``.  When *k* is omitted it
    is inferred from *input_names*, else from the union of prime keys (sorted).
    """
    from cana.function_outputs.expression import positional_names

    if input_names is None:
        if k is None:
            keys: Set[str] = set()
            for p in on_primes:
                keys.update(p)
            for p in off_primes:
                keys.update(p)
            # Prefer positional x0.. order when present; else sorted names.
            if keys and all(re.fullmatch(r"x\d+", n) for n in keys):
                idxs = sorted(int(n[1:]) for n in keys)
                k = (max(idxs) + 1) if idxs else 0
                input_names = positional_names(k)
            else:
                input_names = sorted(keys)
                k = len(input_names)
        else:
            input_names = positional_names(int(k))
    names = list(input_names)
    return {
        1: {_prime_to_schema(p, names) for p in on_primes},
        0: {_prime_to_schema(p, names) for p in off_primes},
    }


def one_symbol_schemata_to_primes(
    schemata: OneSymbolSchemata,
    input_names: Sequence[str],
) -> _OnOffCubes:
    """Convert one-symbol schemata to named ON/OFF prime-cube lists."""
    names = list(input_names)

    def _from_set(sset) -> List[_PrimeCube]:
        out: List[_PrimeCube] = []
        for s in sset:
            d: _PrimeCube = {}
            for i, ch in enumerate(s):
                if ch == "1":
                    d[names[i]] = True
                elif ch == "0":
                    d[names[i]] = False
            out.append(d)
        return out

    return (
        _from_set(schemata.get(1, ())),
        _from_set(schemata.get(0, ())),
    )


def one_symbol_schemata_from_positional_expression(
    expr: str, k: int
) -> OneSymbolSchemata:
    """Derive one-symbol schemata from a positional boolean expression.

    Tries the expression cascade (syntactic ROF, then Primer-B) without
    building a dense LUT.  On cascade failure, falls back to LUT +
    Quine–McCluskey when ``k`` is at most ``MAX_DENSE_LUT_K``; otherwise
    raises ``ValueError``.
    """
    from cana.function_outputs.expression import (
        positional_expression_to_bit_lut,
        positional_names,
    )
    from cana.primes.cascade import on_off_primes_from_expression
    from cana.primes.util import MAX_DENSE_LUT_K

    result = on_off_primes_from_expression(expr)
    if result.ok and result.on_primes is not None and result.off_primes is not None:
        return primes_to_one_symbol_schemata(
            result.on_primes,
            result.off_primes,
            input_names=positional_names(k),
        )

    if k > MAX_DENSE_LUT_K:
        raise ValueError(
            "Cannot get one_symbol_schemata from positional_expression: "
            "expression cascade unsuccessful and k={0:d} > "
            "MAX_DENSE_LUT_K={1:d} (refusing dense LUT/QM)".format(
                k, MAX_DENSE_LUT_K
            )
        )
    return one_symbol_schemata_from_bit_lut(positional_expression_to_bit_lut(expr, k))
