# -*- coding: utf-8 -*-
"""Positional Boolean expression helpers for :class:`FunctionOutputs`.

In-memory peer IR uses python ``and``/``or``/``not`` over ``x0``…``x{k-1}``
(MSB = ``x0`` = ``inputs[0]``).  Named source strings are translated at the
I/O / :class:`~cana.boolean_node.BooleanNode` boundary.

Naming:
- ``*_positional*`` / ``named_to_positional`` / ``positional_to_named`` —
  fixed IR kind (positional ``x_i`` or named regulators).
- ``bit_lut_to_expression`` — emit whatever names are in ``input_names``
  (pass :func:`positional_names` for positional IR).
- :func:`normalize_operators` / :func:`expression_identifiers` — either kind.
"""
from __future__ import annotations

import re
import warnings
from typing import Any, List, Optional, Sequence

from cana.function_outputs.bit_lut import BitLut, refuse_dense_lut_k
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
    "named_to_positional",
    "positional_to_named",
    "reorder_positional",
    "positional_expression_to_bit_lut",
    "bit_lut_to_expression",
    "function_outputs_to_named_expression",
]

BOOL_KEYWORDS = frozenset({"and", "or", "not", "true", "false", "xor"})
IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
POSITIONAL_RE = re.compile(r"\bx(\d+)\b")


def normalize_operators(expr: str, style: str = "python") -> str:
    """Normalize boolean operators to ``and``/``or``/``not`` or ``&``/``|``/``!``.

    Works on positional or named expressions (operators only).

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
    """Return identifier tokens in ``expr``, excluding boolean keywords.

    Works on positional (``x0``…) or named expressions.
    """
    return [
        tok
        for tok in IDENTIFIER_RE.findall(expr)
        if tok.lower() not in BOOL_KEYWORDS
    ]


def regulator_names_in_expression(
    expr: str, known_names: Sequence[str]
) -> List[str]:
    """Select known node names that appear as tokens in a named expression.

    ``known_names`` order is preserved.  Intended for named (regulator)
    expressions, not positional ``x_i`` IR.
    """
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


def named_to_positional(expr: str, input_names: Sequence[str]) -> str:
    """Rewrite a named expression to positional ``x0``…``x{k-1}`` IR."""
    mapping = {name: "x{0}".format(i) for i, name in enumerate(input_names)}
    return _replace_identifiers(expr, mapping)


def positional_to_named(expr: str, input_names: Sequence[str]) -> str:
    """Rewrite positional ``x_i`` IR to a named expression over ``input_names``."""
    mapping = {"x{0}".format(i): name for i, name in enumerate(input_names)}
    return _replace_identifiers(expr, mapping)


def reorder_positional(
    expr: str, old_labels: Sequence[object], new_labels: Sequence[object]
) -> str:
    """Rewrite positional ``x_i`` indices when input order changes.

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


def positional_expression_to_bit_lut(expr: str, k: int) -> BitLut:
    """Evaluate positional ``x0``…``x{k-1}`` IR to a dense :class:`BitLut`.

    Refuses when ``k`` exceeds
    :data:`~cana.function_outputs.bit_lut.MAX_DENSE_LUT_K`.
    """
    k = int(k)
    if expr in ("True", "true", "1"):
        if k != 0:
            raise ValueError("constant True requires k=0")
        return BitLut.from_sequence([1])
    if expr in ("False", "false", "0"):
        if k != 0:
            raise ValueError("constant False requires k=0")
        return BitLut.from_sequence([0])
    refuse_dense_lut_k(k, via="positional_expression_to_bit_lut")
    names = positional_names(k)
    return BitLut.from_sequence(output_transitions(expr, names))


def _style_ops(style: str):
    """Return ``(and_op, or_op, not_prefix)`` for ``style``."""
    if style == "python":
        return " and ", " or ", "not "
    if style == "symbolic":
        return " & ", " | ", "!"
    raise ValueError("style must be 'python' or 'symbolic'")


def _contradiction_expr(input_names: Sequence[str], style: str) -> str:
    """Return a constant-False expression over ``input_names``."""
    x = input_names[0]
    if style == "python":
        return "({x} and not {x})".format(x=x)
    return "({x} & !{x})".format(x=x)


def _tautology_expr(input_names: Sequence[str], style: str) -> str:
    """Return a constant-True expression over ``input_names``."""
    x = input_names[0]
    if style == "python":
        return "({x} or not {x})".format(x=x)
    return "({x} | !{x})".format(x=x)


def bit_lut_to_expression(
    input_names: Sequence[str], lut: BitLut, style: str = "python"
) -> str:
    """Build a DNF expression for a :class:`BitLut` using ``input_names``.

    Positional vs named is chosen by ``input_names``: pass
    :func:`positional_names` ``(k)`` for positional IR, or regulator names for
    a named expression.  Callers with a plain 0/1 sequence should convert via
    :meth:`BitLut.from_sequence` first.

    Args:
        input_names: Input variable names in LUT bit order (MSB first).
        lut: Dense truth table (``lut.k`` must equal ``len(input_names)``).
        style: ``"python"`` (``and``/``or``/``not``) or ``"symbolic"`` (``&``/``|``/``!``).
    """
    k = len(input_names)
    if k != lut.k:
        raise ValueError(
            "len(input_names)={0:d} does not match BitLut.k={1:d}".format(
                k, lut.k
            )
        )
    if k == 0:
        return "True" if lut[0] else "False"

    and_op, or_op, not_prefix = _style_ops(style)

    const = lut.is_constant()
    if const == 0:
        return _contradiction_expr(input_names, style)
    if const == 1:
        return _tautology_expr(input_names, style)

    terms: List[str] = []
    for state in range(len(lut)):
        if lut[state] != 1:
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
        return _contradiction_expr(input_names, style)

    return or_op.join(terms)


def function_outputs_to_named_expression(
    outputs: Any,
    input_names: Sequence[str],
    style: str = "python",
    *,
    node_label: Optional[str] = None,
    warn: bool = True,
) -> str:
    """Build a named expression from ``FunctionOutputs`` peers.

    Priority: stored positional expression → two-symbol → one-symbol → LUT.

    Never runs ``verify_*``; only reads
    ``representations_consistent`` / ``one_symbol_is_prime_implicant``.
    Soft-warns (never raises) for the chosen non-LUT source:
    one-symbol when ``one_symbol_is_prime_implicant`` is not ``True``;
    two-symbol likewise when a one-symbol cover is also stored (two-symbol
    is often derived from it);
    consistency only when ``representations_consistent is False``
    (``None`` is silent — common after lazy LUT fill).

    Args:
        outputs: A :class:`~cana.function_outputs.FunctionOutputs` (duck-typed).
        input_names: Regulator names in LUT / expression bit order.
        style: ``"python"`` or ``"symbolic"``.
        node_label: Optional node name for warning text.
        warn: If False, suppress verification soft-warnings.
    """
    # Local imports avoid cycles with one_symbol / two_symbol importing
    # style helpers from this module.
    from cana.function_outputs.one_symbol import one_symbol_to_expression
    from cana.function_outputs.two_symbol import two_symbol_to_expression

    label = "Node {0!r}".format(node_label) if node_label is not None else "Node"

    def _multi_slot() -> bool:
        n = 0
        if getattr(outputs, "positional_expression", None) is not None:
            n += 1
        if getattr(outputs, "one_symbol_schemata", None) is not None:
            n += 1
        if getattr(outputs, "two_symbol_schemata", None) is not None:
            n += 1
        if getattr(outputs, "bit_lut", None) is not None:
            n += 1
        return n >= 2

    def _warn_consistency(source: str) -> None:
        # Only warn on known mismatch (False). ``None`` is normal when a LUT
        # was lazily filled from the same expression and never verified.
        if not warn or not _multi_slot():
            return
        flag = getattr(outputs, "representations_consistent", None)
        if flag is not False:
            return
        warnings.warn(
            "{0}: exporting from {1} "
            "(representations_consistent={2!r})".format(label, source, flag),
            UserWarning,
            stacklevel=3,
        )

    def _warn_prime(source: str) -> None:
        if not warn:
            return
        flag = getattr(outputs, "one_symbol_is_prime_implicant", None)
        if flag is True:
            return
        warnings.warn(
            "{0}: exporting from {1} "
            "(one_symbol_is_prime_implicant={2!r})".format(
                label, source, flag
            ),
            UserWarning,
            stacklevel=3,
        )

    expr = getattr(outputs, "positional_expression", None)
    if expr is not None:
        _warn_consistency("positional_expression")
        named = positional_to_named(expr, input_names)
        if style == "python":
            return named
        return normalize_operators(named, style)

    two_symbol = getattr(outputs, "two_symbol_schemata", None)
    one_symbol = getattr(outputs, "one_symbol_schemata", None)
    if two_symbol is not None:
        # Two-symbol is often built from one-symbol; warn if that cover
        # is present and not known to be prime.
        if one_symbol is not None:
            _warn_prime("two_symbol_schemata")
        _warn_consistency("two_symbol_schemata")
        return two_symbol_to_expression(
            input_names, two_symbol, style=style
        )

    if one_symbol is not None:
        _warn_prime("one_symbol_schemata")
        _warn_consistency("one_symbol_schemata")
        return one_symbol_to_expression(input_names, one_symbol, style=style)

    get_bit_lut = getattr(outputs, "get_bit_lut", None)
    if get_bit_lut is not None:
        lut = get_bit_lut()
    else:
        lut = getattr(outputs, "bit_lut", None)
        if lut is None:
            raise ValueError(
                "Cannot build named expression: neither "
                "positional_expression, one_symbol_schemata, "
                "two_symbol_schemata, nor bit_lut is available on outputs"
            )
    return bit_lut_to_expression(input_names, lut, style=style)
