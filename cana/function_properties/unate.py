# -*- coding: utf-8 -*-
"""Unate and monotone-increasing checks: semantic (LUT) and syntactic (NNF).

**Semantic** predicates inspect a dense
:class:`~cana.function_outputs.bit_lut.BitLut`.  An input ``idx`` is
**positive unate** when flipping it from ``0`` to ``1`` never decreases the
output, and **negative unate** when that flip never increases the output.
A function is **monotone increasing** when every input is positive unate,
and **unate** when every input has a nonzero sign.

**Syntactic** unate means every identifier appears only positively or only
negatively after negation is pushed to the leaves (NNF).  A variable that
occurs both ways is *binate*.  Same-polarity repeats (e.g. ``a & a``) are
still unate; that is distinct from syntactic ROF (single occurrence).
Named or positional expression IR both work (identifiers are tokens).

The string path is a **single left-to-right NNF scan** with an explicit
polarity stack (iterative; deep nesting cannot blow the call stack).  It
**always stops at the first binate** variable.  ``binate`` in the return
dict is empty or a one-element list; ``positive`` / ``negative`` reflect
only what was seen before stopping when the formula is not unate.
``n_unique_variables`` is always a **full** unique-identifier count.
``is_unate`` means *syntactic* unate.

An optional AST leaf-polarity walk
(:func:`check_syntactic_unate_ast`) is available when a tree already
exists; the string API does **not** parse to an AST.

Naming (semantic):
- ``is_input_*`` / ``input_sign`` — one variable (input index) of the LUT.
- ``input_signs`` — one sign per input (not a single whole-function flag).
- ``is_function_*`` — a property of the whole LUT / Boolean function.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Union

from cana.function_outputs.ast import LogicNode, ast_leaf_polarities
from cana.function_outputs.bit_lut import BitLut
from cana.function_outputs.expression import (
    expression_identifiers,
    normalize_operators,
)

__all__ = [
    "is_input_positive_unate",
    "is_input_negative_unate",
    "input_sign",
    "input_signs",
    "is_function_monotone_increasing",
    "is_function_unate",
    "check_syntactic_unate",
    "check_syntactic_unate_ast",
    "require_syntactic_unate",
    "require_syntactic_unate_ast",
    "syntactic_input_signs",
]

UnateInfo = Dict[str, Union[bool, int, List[str]]]


# --- semantic (BitLut) -------------------------------------------------------


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


# --- syntactic (NNF scan / optional AST) -------------------------------------


def _n_unique_from_expr(expr: str) -> int:
    """Full unique-identifier count (same tokenizer as syntactic ROF)."""
    toks = expression_identifiers(normalize_operators(expr, "python"))
    return len(set(toks))


def _stats(
    positive: Set[str],
    negative: Set[str],
    first_binate: Optional[str],
    *,
    n_unique_variables: int,
) -> UnateInfo:
    if first_binate is not None:
        # Conflict name sits in the set for its first polarity only.
        # positive/negative are partial (scan stopped); n_unique is full.
        only_pos = sorted(positive - {first_binate})
        only_neg = sorted(negative - {first_binate})
        return {
            "is_unate": False,
            "positive": only_pos,
            "negative": only_neg,
            "binate": [first_binate],
            "n_unique_variables": n_unique_variables,
        }
    return {
        "is_unate": True,
        "positive": sorted(positive),
        "negative": sorted(negative),
        "binate": [],
        "n_unique_variables": n_unique_variables,
    }


def _nnf_scan(expr: str) -> UnateInfo:
    """Linear NNF polarity scan; stop at the first binate.

    Iterative: ``&`` / ``|`` are polarity-neutral; only ``!`` and
    parentheses update an explicit stack.  Nesting depth is unbounded
    by the CPython recursion limit.
    """
    s = normalize_operators(
        normalize_operators(expr, "python"), "symbolic"
    )
    n = len(s)
    positive: Set[str] = set()
    negative: Set[str] = set()
    first_binate: Optional[str] = None

    def record(name: str, pol: int) -> bool:
        nonlocal first_binate
        if pol > 0:
            if name in negative:
                first_binate = name
                return True
            positive.add(name)
        else:
            if name in positive:
                first_binate = name
                return True
            negative.add(name)
        return False

    n_unique = _n_unique_from_expr(expr)

    if not s.strip():
        return _stats(positive, negative, first_binate, n_unique_variables=n_unique)

    # Top of stack = polarity inside the current paren group.
    pol_stack: List[int] = [+1]
    i = 0
    while i < n:
        while i < n and s[i].isspace():
            i += 1
        if i >= n:
            break
        c = s[i]
        if c in "&|":
            i += 1
            continue
        if c == ")":
            if len(pol_stack) > 1:
                pol_stack.pop()
            i += 1
            continue
        # Factor: optional bangs, then '(' or identifier.
        sign = 1
        while i < n and s[i] == "!":
            sign = -sign
            i += 1
            while i < n and s[i].isspace():
                i += 1
        while i < n and s[i].isspace():
            i += 1
        if i >= n:
            break
        if s[i] == "(":
            pol_stack.append(pol_stack[-1] * sign)
            i += 1
            continue
        if s[i].isalpha() or s[i] == "_":
            j = i + 1
            while j < n and (s[j].isalnum() or s[j] == "_"):
                j += 1
            if record(s[i:j], pol_stack[-1] * sign):
                break
            i = j
            continue
        # Skip unknown char to avoid hanging on malformed input.
        i += 1

    return _stats(positive, negative, first_binate, n_unique_variables=n_unique)


def check_syntactic_unate(expr: str) -> UnateInfo:
    """Return syntactic-unate stats for expression *expr*.

    Single-pass NNF polarity scan.  Stops at the first binate variable
    (``binate`` has at most one name).  ``n_unique_variables`` is always a
    full count (not truncated by early stop).  Constants / empty
    expressions are vacuously unate.  Does not parse to an AST.
    """
    return _nnf_scan(expr)


def check_syntactic_unate_ast(node: Optional[LogicNode]) -> UnateInfo:
    """Return syntactic-unate stats for an AST *node* (leaf polarity walk).

    Convenience when a tree already exists; enumerates all leaves (no
    early stop).  Prefer :func:`check_syntactic_unate` for expression
    strings.
    """
    leaves = ast_leaf_polarities(node)
    positive: Set[str] = set()
    negative: Set[str] = set()
    first_binate: Optional[str] = None
    for name, pol in leaves:
        if pol == "pos":
            if name in negative:
                first_binate = name
                break
            positive.add(name)
        else:
            if name in positive:
                first_binate = name
                break
            negative.add(name)
    names = [n for n, _ in leaves]
    return _stats(
        positive,
        negative,
        first_binate,
        n_unique_variables=len(set(names)),
    )


def require_syntactic_unate(expr: str) -> None:
    """Raise if *expr* is not syntactically unate.

    Rejects formulas that might only be unate after simplification; this
    check never rewrites.  Stops at the first binate.
    """
    info = check_syntactic_unate(expr)
    if info["is_unate"]:
        return
    raise ValueError(
        "expression is not syntactically unate; "
        f"binate variables={info['binate']} "
        f"(n_unique={info['n_unique_variables']}). "
        "This method does not simplify; a formula that is unate "
        "only after rewriting (semantic unate) is not accepted."
    )


def require_syntactic_unate_ast(node: Optional[LogicNode]) -> None:
    """Raise if *node* is not syntactically unate."""
    info = check_syntactic_unate_ast(node)
    if info["is_unate"]:
        return
    raise ValueError(
        "AST is not syntactically unate; "
        f"binate variables={info['binate']} "
        f"(n_unique={info['n_unique_variables']}). "
        "This method does not simplify; a formula that is unate "
        "only after rewriting (semantic unate) is not accepted."
    )


def syntactic_input_signs(
    expr: str, input_names: Sequence[str]
) -> Optional[List[int]]:
    """Return per-input signs if *expr* is syntactically unate; else ``None``.

    Uses the NNF string scan.  ``+1`` / ``-1`` come from the ``positive`` /
    ``negative`` sets keyed by *input_names*; inputs absent from the
    expression get ``+1`` (same preference as semantic irrelevant inputs).
    Returns ``None`` when not syntactically unate so callers can fall
    through to a LUT-based check (syntactic binate does not imply
    semantic binate).
    """
    info = check_syntactic_unate(expr)
    if not info["is_unate"]:
        return None
    pos = set(info["positive"])  # type: ignore[arg-type]
    neg = set(info["negative"])  # type: ignore[arg-type]
    signs: List[int] = []
    for name in input_names:
        if name in pos:
            signs.append(1)
        elif name in neg:
            signs.append(-1)
        else:
            signs.append(1)
    return signs
