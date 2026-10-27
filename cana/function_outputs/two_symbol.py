# -*- coding: utf-8 -*-
"""Two-symbol schemata helpers for :class:`FunctionOutputs`.

Two-symbol entries are ``[schema_str, perm_groups, same_groups]``.  Expansion
to one-symbol and compact expression export use **perm_groups only**;
``same_groups`` are display metadata and are ignored.
"""
from __future__ import annotations

import itertools
from collections import deque
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from cana.function_outputs.expression import _contradiction_expr, _style_ops
from cana.function_outputs.one_symbol import OneSymbolSchemata

__all__ = [
    "one_symbol_from_two_symbol",
    "two_symbol_to_expression",
]

# (ts0, ts1); each entry is [schema_str, perm_groups, same_groups].
TwoSymbolSchemata = Tuple[list, list]


def _as_schema_str(schema) -> str:
    if isinstance(schema, str):
        return schema
    return "".join(map(str, schema))


def _expand_entry(schema_str: str, perm_groups: Optional[Iterable]) -> Set[str]:
    """Expand one two-symbol entry to one-symbol schema strings.

    Always includes the representative.  Empty ``perm_groups`` yields only
    the representative (unlike ``_expand_ts_logic``, which returns ``[]``).
    """
    base = list(schema_str)
    results = {"".join(base)}
    groups = list(perm_groups or ())
    if not groups:
        return results

    queue = deque([base])
    seen = {"".join(base)}
    while queue:
        implicant = queue.pop()
        for idxs in groups:
            vals = [implicant[i] for i in idxs]
            for perm in itertools.permutations(vals, len(idxs)):
                new_imp = list(implicant)
                for i, v in zip(idxs, perm):
                    new_imp[i] = v
                key = "".join(new_imp)
                if key not in seen:
                    seen.add(key)
                    results.add(key)
                    queue.append(new_imp)
    return results


def one_symbol_from_two_symbol(ts: TwoSymbolSchemata) -> OneSymbolSchemata:
    """Expand two-symbol schemata to one-symbol (F') covers.

    For each polarity and each ``[schema, perm_groups, _]``, permute values
    within each perm group and collect length-``k`` strings over ``{0,1,2}``.
    Does not re-run Quine–McCluskey; the result is the redescribed F' cover,
    not necessarily flagged as prime-implicant.
    """
    ts0, ts1 = ts
    out = {0: set(), 1: set()}  # type: OneSymbolSchemata
    for polarity, entries in ((0, ts0), (1, ts1)):
        for entry in entries or ():
            schema = _as_schema_str(entry[0])
            perm_groups = entry[1] if len(entry) > 1 else []
            out[polarity] |= _expand_entry(schema, perm_groups)
    return out


def _literal(name: str, bit, not_prefix: str) -> Optional[str]:
    b = str(bit)
    if b == "1":
        return name
    if b == "0":
        return not_prefix + name
    if b == "2":
        return None
    raise ValueError("invalid schema character {0!r}".format(bit))


def _join_and(lits: List[str], and_op: str) -> Optional[str]:
    if not lits:
        return None
    if len(lits) == 1:
        return lits[0]
    return "(" + and_op.join(lits) + ")"


def _join_or(parts: List[str], or_op: str) -> Optional[str]:
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return "(" + or_op.join(parts) + ")"


def _perm_group_expr(
    schema: str,
    indices: Sequence[int],
    input_names: Sequence[str],
    and_op: str,
    or_op: str,
    not_prefix: str,
) -> Optional[str]:
    """OR of distinct permutation clauses for one perm group."""
    vals = [schema[i] for i in indices]
    clauses: List[str] = []
    seen = set()
    for perm in itertools.permutations(vals, len(indices)):
        if perm in seen:
            continue
        seen.add(perm)
        lits: List[str] = []
        for idx, bit in zip(indices, perm):
            lit = _literal(input_names[idx], bit, not_prefix)
            if lit is not None:
                lits.append(lit)
        clause = _join_and(lits, and_op)
        if clause is None:
            # All don't-cares in this assignment → group is tautological.
            return None
        clauses.append(clause)
    return _join_or(clauses, or_op)


def _schema_to_term(
    schema: str,
    perm_groups: Sequence,
    input_names: Sequence[str],
    and_op: str,
    or_op: str,
    not_prefix: str,
) -> Optional[str]:
    """AND of fixed literals and perm-group subexpressions for one TS schema.

    Returns ``None`` for an all-don't-care / tautological schema term (caller
    treats that as constant True for the ON cover).
    """
    k = len(input_names)
    if len(schema) != k:
        raise ValueError(
            "schema length {0:d} does not match k={1:d}".format(len(schema), k)
        )

    in_group = set()
    for group in perm_groups or ():
        in_group.update(group)

    parts: List[str] = []

    for i, bit in enumerate(schema):
        if i in in_group:
            continue
        lit = _literal(input_names[i], bit, not_prefix)
        if lit is not None:
            parts.append(lit)

    for group in perm_groups or ():
        gexpr = _perm_group_expr(
            schema, group, input_names, and_op, or_op, not_prefix
        )
        if gexpr is None:
            # Tautological group (all 2s under permutation) — skip.
            continue
        parts.append(gexpr)

    if not parts:
        return None  # full cube
    return _join_and(parts, and_op)


def two_symbol_to_expression(
    input_names: Sequence[str],
    ts: TwoSymbolSchemata,
    style: str = "python",
) -> str:
    """Build a DNF-like expression from two-symbol ON schemata (polarity ``1``).

    Each schema becomes an AND of fixed literals and perm-group subexpressions
    (each group is an OR of the distinct permutation clauses).  Schemas are
    OR'd together.  Schema characters: ``1`` positive, ``0`` negated, ``2``
    don't-care.  ``same_groups`` are ignored.
    """
    k = len(input_names)
    and_op, or_op, not_prefix = _style_ops(style)
    _ts0, ts1 = ts
    on_entries = list(ts1 or ())

    if k == 0:
        return "True" if on_entries else "False"

    terms: List[str] = []
    for entry in on_entries:
        schema = _as_schema_str(entry[0])
        perm_groups = entry[1] if len(entry) > 1 else []
        term = _schema_to_term(
            schema, perm_groups, input_names, and_op, or_op, not_prefix
        )
        if term is None:
            return "True"
        terms.append(term)

    if not terms:
        return _contradiction_expr(input_names, style)
    if len(terms) == 1:
        return terms[0]
    return or_op.join(terms)
