# -*- coding: utf-8 -*-
"""Boolean expression string helpers (operators, identifiers, LUT → DNF)."""
from __future__ import annotations

import re
from typing import Any, List, Sequence

from cana.cutils import statenum_to_binstate

BOOL_KEYWORDS = frozenset({"and", "or", "not", "true", "false", "xor"})
IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


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


def lut_to_expression(
    input_names: Sequence[str], lut: Sequence[Any], style: str = "python"
) -> str:
    """Build a DNF expression for a LUT.

    Args:
        input_names: Input variable names in LUT bit order (MSB first).
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
