# -*- coding: utf-8 -*-
"""Positional Boolean expression helpers for :class:`FunctionOutputs`.

In-memory peer IR uses python ``and``/``or``/``not`` over ``x0``…``x{k-1}``
(MSB = ``x0`` = ``inputs[0]``).  Named source strings are translated at the
I/O / :class:`~cana.boolean_node.BooleanNode` boundary.
"""
from __future__ import annotations

import re
from typing import Any, List, Sequence

from cana.bit_lut import BitLut
from cana.cutils import statenum_to_binstate
from cana.utils import output_transitions

__all__ = [
    "BOOL_KEYWORDS",
    "IDENTIFIER_RE",
    "POSITIONAL_RE",
    "normalize_operators",
    "expression_identifiers",
    "regulator_names_in_expression",
    "positional_names",
    "to_positional",
    "from_positional",
    "reorder_positional",
    "bit_lut_from_expression",
    "lut_to_expression",
]

BOOL_KEYWORDS = frozenset({"and", "or", "not", "true", "false", "xor"})
IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
POSITIONAL_RE = re.compile(r"\bx(\d+)\b")


def normalize_operators(expr: str, style: str = "python") -> str:
    """Normalize boolean operators to ``and``/``or``/``not`` or ``&``/``|``/``!``.

    Args:
        expr: Boolean expression string.
        style: ``"python"`` (for :func:`~cana.utils.output_transitions`) or
            ``"symbolic"``.
    """
    s = expr.strip()
    if style == "python":
        s = re.sub(r"\bAND\b", "and", s, flags=re.IGNORECASE)
        s = re.sub(r"\bOR\b", "or", s, flags=re.IGNORECASE)
        s = re.sub(r"\bNOT\b", "not", s, flags=re.IGNORECASE)
        s = s.replace("&&", " and ").replace("||", " or ")
        s = re.sub(r"(?<![A-Za-z0-9_])!(?!=)", " not ", s)
        s = s.replace("&", " and ").replace("|", " or ")
        s = re.sub(r"\s+", " ", s).strip()
        return s
    if style == "symbolic":
        s = re.sub(r"\band\b", "&", s, flags=re.IGNORECASE)
        s = re.sub(r"\bor\b", "|", s, flags=re.IGNORECASE)
        s = re.sub(r"\bnot\b", "!", s, flags=re.IGNORECASE)
        s = re.sub(r"\s+", " ", s).strip()
        return s
    raise ValueError("style must be 'python' or 'symbolic'")


def expression_identifiers(expr: str) -> List[str]:
    """Return identifier tokens in ``expr``, excluding boolean keywords."""
    return [
        tok
        for tok in IDENTIFIER_RE.findall(expr)
        if tok.lower() not in BOOL_KEYWORDS
    ]


def regulator_names_in_expression(
    expr: str, known_names: Sequence[str]
) -> List[str]:
    """Select known node names that appear as tokens in ``expr`` (stable order)."""
    padded = "*" + expr + "*"
    found: List[str] = []
    for name in known_names:
        if re.search(r"\W" + re.escape(name) + r"\W", padded):
            found.append(name)
    return found


def positional_names(k: int) -> List[str]:
    """Return ``['x0', 'x1', …, 'x{k-1}']``."""
    return ["x{0}".format(i) for i in range(int(k))]


def _replace_identifiers(expr: str, mapping: dict) -> str:
    """Replace identifier tokens via ``mapping``; longest keys first."""
    if not mapping:
        return expr

    def repl(match):
        tok = match.group(0)
        return mapping.get(tok, tok)

    # Longest-first alternation avoids Gene clobbering Gene1 when building
    # a single regex; mapping lookup still handles exact tokens.
    keys = sorted(mapping.keys(), key=len, reverse=True)
    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(k) for k in keys) + r")\b"
    )
    return pattern.sub(repl, expr)


def to_positional(expr: str, input_names: Sequence[str]) -> str:
    """Rewrite a named expression to positional ``x0``…``x{k-1}`` IR."""
    mapping = {name: "x{0}".format(i) for i, name in enumerate(input_names)}
    return _replace_identifiers(expr, mapping)


def from_positional(expr: str, input_names: Sequence[str]) -> str:
    """Rewrite positional ``x_i`` IR to ``input_names[i]``."""
    mapping = {"x{0}".format(i): name for i, name in enumerate(input_names)}
    return _replace_identifiers(expr, mapping)


def reorder_positional(
    expr: str, old_labels: Sequence[object], new_labels: Sequence[object]
) -> str:
    """Rewrite ``x_i`` indices when input order changes.

    ``old_labels`` / ``new_labels`` identify the same inputs in old vs new
    bit order (any hashable labels).  Token ``x{i}`` referred to
    ``old_labels[i]``; after reorder it must refer to the same label at its
    new index.
    """
    old_list = list(old_labels)
    new_list = list(new_labels)
    if len(old_list) != len(new_list):
        raise ValueError("old_labels and new_labels must have the same length")
    # old_index -> new_index for the same label
    new_index = {label: j for j, label in enumerate(new_list)}
    mapping = {}
    for i, label in enumerate(old_list):
        j = new_index[label]
        if i != j:
            mapping["x{0}".format(i)] = "x{0}".format(j)
    if not mapping:
        return expr
    # Two-phase replace via temps to avoid x0↔x1 clobber.
    old_tokens = list(mapping.keys())
    phase1 = {tok: "__T{0}__".format(i) for i, tok in enumerate(old_tokens)}
    phase2 = {
        "__T{0}__".format(i): mapping[tok] for i, tok in enumerate(old_tokens)
    }
    mid = _replace_identifiers(expr, phase1)
    return _replace_identifiers(mid, phase2)


def bit_lut_from_expression(expr: str, k: int) -> BitLut:
    """Evaluate positional (or already-bound) IR to a dense :class:`BitLut`."""
    k = int(k)
    if expr in ("True", "true", "1"):
        if k != 0:
            raise ValueError("constant True requires k=0")
        return BitLut.from_sequence([1])
    if expr in ("False", "false", "0"):
        if k != 0:
            raise ValueError("constant False requires k=0")
        return BitLut.from_sequence([0])
    names = positional_names(k)
    return BitLut.from_sequence(output_transitions(expr, names))


def lut_to_expression(
    input_names: Sequence[str], lut: Sequence[Any], style: str = "python"
) -> str:
    """Build a DNF expression for a LUT.

    Args:
        input_names: Input variable names in LUT bit order (MSB first).
            Pass ``positional_names(k)`` for positional IR.
        lut: Truth table of length ``2**k``.
        style: ``"python"`` (``and``/``or``/``not``) or ``"symbolic"`` (``&``/``|``/``!``).
    """
    k = len(input_names)
    if k == 0:
        val = int(lut[0]) if lut else 0
        return "True" if val else "False"

    and_op = " and " if style == "python" else " & "
    or_op = " or " if style == "python" else " | "
    not_prefix = "not " if style == "python" else "!"

    terms: List[str] = []
    for state in range(2**k):
        if int(lut[state]) != 1:
            continue
        bits = statenum_to_binstate(state, k)
        lits: List[str] = []
        for name, bit in zip(input_names, bits):
            if bit == "1":
                lits.append(name)
            else:
                lits.append(not_prefix + name)
        if len(lits) == 1:
            terms.append(lits[0])
        else:
            terms.append("(" + and_op.join(lits) + ")")

    if not terms:
        x = input_names[0]
        if style == "python":
            return "({x} and not {x})".format(x=x)
        return "({x} & !{x})".format(x=x)

    if len(lut) == 2**k and all(int(v) == 1 for v in lut):
        x = input_names[0]
        if style == "python":
            return "({x} or not {x})".format(x=x)
        return "({x} | !{x})".format(x=x)

    return or_op.join(terms)
