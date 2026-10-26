# -*- coding: utf-8 -*-
"""Build and compare CANA ``LutDict`` hubs (per-node LUTs + regulators)."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TypedDict, Union
from pathlib import Path

from cana.cutils import binstate_to_statenum, statenum_to_binstate
from cana.io.expressions import (
    expression_identifiers,
    normalize_operators,
    regulator_names_in_expression,
)
from cana.utils import output_transitions

NodeLut = TypedDict(
    "NodeLut",
    {
        "name": str,
        "regulators": List[int],
        "lut": List[int],
    },
)
"""One node's entry in a CANA ``LutDict``.

Keys:
    name: Unique node name.
    regulators: Ordered list of input **node ids** (length ``k``).
    lut: Truth table of length ``2**k`` with entries in ``{0, 1}``.  Row ``s``
        is the next value when the input state (in the ``regulators`` order,
        MSB = ``regulators[0]``) equals ``s``, matching
        :func:`~cana.utils.output_transitions`.
"""

LutDict = Dict[int, NodeLut]
"""Network wiring + LUTs keyed by node id."""

PathLike = Union[str, Path]


def reorder_lut(
    lut: Sequence[Any], old_inputs: Sequence[str], new_inputs: Sequence[str]
) -> List[int]:
    """Reorder a LUT from ``old_inputs`` bit order to ``new_inputs`` bit order."""
    if list(old_inputs) == list(new_inputs):
        return [int(v) for v in lut]
    k = len(old_inputs)
    if k != len(new_inputs) or set(old_inputs) != set(new_inputs):
        raise ValueError("old_inputs and new_inputs must be permutations")
    name_to_old = {n: i for i, n in enumerate(old_inputs)}
    new_lut = [0] * (2**k)
    for new_state in range(2**k):
        new_bits = statenum_to_binstate(new_state, k)
        old_bits = ["0"] * k
        for new_i, name in enumerate(new_inputs):
            old_bits[name_to_old[name]] = new_bits[new_i]
        old_state = binstate_to_statenum("".join(old_bits))
        new_lut[new_state] = int(lut[old_state])
    return new_lut


def expressions_to_lut_dict(
    expressions: Mapping[str, str],
    extra_names: Optional[Iterable[str]] = None,
) -> LutDict:
    """Build a ``LutDict`` from ``{target: boolean_expression}``.

    Names that appear only as regulators (free inputs) receive the identity
    update ``f(x) = x``, matching BBM's booleannet representation of inputs.
    """
    # Preserve insertion order of targets; append free inputs alphabetically.
    ordered: List[str] = list(expressions.keys())
    seen = set(ordered)
    mentioned: set = set()
    for expr in expressions.values():
        mentioned.update(
            expression_identifiers(normalize_operators(expr, "python"))
        )
    if extra_names:
        mentioned.update(extra_names)
    for name in sorted(mentioned):
        if name not in seen:
            ordered.append(name)
            seen.add(name)

    name_to_id = {name: i for i, name in enumerate(ordered)}
    completed: Dict[str, str] = dict(expressions)
    for name in ordered:
        if name not in completed:
            completed[name] = name  # identity

    lut_dict: LutDict = {}
    for i, name in enumerate(ordered):
        expr = normalize_operators(completed[name], "python")
        if expr in ("True", "true", "1"):
            lut_dict[i] = {"name": name, "regulators": [], "lut": [1]}
            continue
        if expr in ("False", "false", "0"):
            lut_dict[i] = {"name": name, "regulators": [], "lut": [0]}
            continue
        input_names = regulator_names_in_expression(expr, ordered)
        if expr.strip() == name and name not in input_names:
            input_names = [name]
        regulator_ids = [name_to_id[n] for n in input_names]
        lut = output_transitions(expr, input_names)
        lut_dict[i] = {
            "name": name,
            "regulators": regulator_ids,
            "lut": [int(v) for v in lut],
        }
    return lut_dict


def sorted_node_ids(lut_dict: Mapping[int, Mapping[str, Any]]) -> List[int]:
    """Return node ids in ascending order (export order)."""
    return sorted(lut_dict.keys())


def regulator_names_for_node(
    lut_dict: Mapping[int, Mapping[str, Any]], node_id: int
) -> List[str]:
    """Return regulator names for ``node_id`` in LUT bit order."""
    return [lut_dict[j]["name"] for j in lut_dict[node_id]["regulators"]]


def name_keyed_luts(
    network_or_lut_dict: Any,
) -> Dict[str, Tuple[Tuple[str, ...], Tuple[int, ...]]]:
    """Return ``{name: (sorted_regulator_names, lut)}`` for equivalence checks.

    Accepts a :class:`~cana.boolean_network.BooleanNetwork` or a ``LutDict``.
    Regulator order is canonicalized by sorting names; the LUT is reordered
    to match.
    """
    if hasattr(network_or_lut_dict, "lut_dict") and hasattr(
        network_or_lut_dict, "nodes"
    ):
        lut_dict = network_or_lut_dict.lut_dict
        id_to_name = {
            i: network_or_lut_dict.nodes[i].name
            for i in range(network_or_lut_dict.Nnodes)
        }
    else:
        lut_dict = network_or_lut_dict
        id_to_name = {i: lut_dict[i]["name"] for i in lut_dict}

    result: Dict[str, Tuple[Tuple[str, ...], Tuple[int, ...]]] = {}
    for i, entry in lut_dict.items():
        name = entry["name"]
        input_names = [id_to_name[j] for j in entry["regulators"]]
        sorted_names = tuple(sorted(input_names))
        lut = reorder_lut(entry["lut"], input_names, list(sorted_names))
        result[name] = (sorted_names, tuple(int(v) for v in lut))
    return result


def equivalent_lut_dicts(a: Any, b: Any) -> bool:
    """True if two networks / LutDicts have the same name-keyed LUTs."""
    return name_keyed_luts(a) == name_keyed_luts(b)
