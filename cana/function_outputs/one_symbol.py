# -*- coding: utf-8 -*-
"""One-symbol (wildcard) schemata helpers.

Schemata are thin ``dict`` maps polarity ``0``/``1`` → sets of length-``k``
strings over ``{0,1,2}`` (``2`` = don't-care).  No wrapper class.

This is the public representation of prime covers for
:class:`~cana.function_outputs.FunctionOutputs`.  Named ``PrimeDict`` cubes
remain algorithm-internal inside :mod:`cana.primes`; convert at the boundary
with :func:`primes_to_one_symbol` /
:func:`one_symbol_to_primes`.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

from cana.function_outputs.bit_lut import BitLut
from cana.cutils import binstate_to_statenum, outputs_to_binstates_of_given_type
from cana.function_outputs.expression import _contradiction_expr, _style_ops

__all__ = [
    "OneSymbolSchemata",
    "reorder",
    "one_symbol_to_bit_lut",
    "bit_lut_to_one_symbol",
    "primes_to_one_symbol",
    "one_symbol_to_primes",
    "positional_expression_to_one_symbol",
    "one_symbol_to_expression",
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
    convention as :meth:`~cana.function_outputs.bit_lut.BitLut.reorder`).  ``new_labels`` must
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


def one_symbol_to_bit_lut(
    schemata: OneSymbolSchemata, k: int
) -> BitLut:
    """Build a :class:`~cana.function_outputs.bit_lut.BitLut` from one-symbol schemata.

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


def bit_lut_to_one_symbol(bit_lut: BitLut) -> OneSymbolSchemata:
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


def primes_to_one_symbol(
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


def one_symbol_to_primes(
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


def positional_expression_to_one_symbol(
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
        return primes_to_one_symbol(
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
    return bit_lut_to_one_symbol(positional_expression_to_bit_lut(expr, k))


def one_symbol_to_expression(
    input_names: Sequence[str],
    schemata: OneSymbolSchemata,
    style: str = "python",
) -> str:
    """Build a DNF expression from a one-symbol ON cover (polarity ``1``).

    Schema characters: ``1`` → literal, ``0`` → negated literal, ``2`` →
    don't-care (omitted).  Incomplete covers are still exported as the
    positive DNF of whatever ON schemas are present (no LUT materialization).

    Positional vs named is chosen by ``input_names``: pass
    :func:`~cana.function_outputs.expression.positional_names` ``(k)`` for
    positional IR, or regulator names for a named expression.

    Args:
        input_names: Variable names in schema bit order (MSB first).
        schemata: One-symbol map polarity → set of length-``k`` schemas.
        style: ``"python"`` or ``"symbolic"``.
    """
    k = len(input_names)
    and_op, or_op, not_prefix = _style_ops(style)
    on_cover = schemata.get(1, set()) if schemata else set()

    if k == 0:
        return "True" if on_cover else "False"

    terms: List[str] = []
    for schema in sorted(on_cover):
        if len(schema) != k:
            raise ValueError(
                "schema length {0:d} does not match k={1:d}".format(
                    len(schema), k
                )
            )
        lits: List[str] = []
        for name, bit in zip(input_names, schema):
            if bit == "1":
                lits.append(name)
            elif bit == "0":
                lits.append(not_prefix + name)
            elif bit == "2":
                continue
            else:
                raise ValueError(
                    "invalid schema character {0!r} in {1!r}".format(
                        bit, schema
                    )
                )
        if not lits:
            # All don't-cares: covers the full cube → constant True.
            return "True"
        if len(lits) == 1:
            terms.append(lits[0])
        else:
            terms.append("(" + and_op.join(lits) + ")")

    if not terms:
        return _contradiction_expr(input_names, style)
    return or_op.join(terms)
