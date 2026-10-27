# -*- coding: utf-8 -*-
"""Build and compare CANA ``NetworkDict`` hubs (per-node wiring + outputs)."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, TypedDict, Union
from pathlib import Path

from cana.bit_lut import BitLut
from cana.function_outputs import FunctionOutputs
from cana.io.expressions import (
    expression_identifiers,
    normalize_operators,
    regulator_names_in_expression,
)
from cana.utils import output_transitions

NodeSpec = TypedDict(
    "NodeSpec",
    {
        "name": str,
        "inputs": List[int],
        "outputs": FunctionOutputs,
    },
)
"""One node's entry in a CANA ``NetworkDict``.

Keys:
    name: Unique node name.
    inputs: Ordered list of input **node ids** (length ``k``).  Empty means
        a structural constant (single-bit function); that implies a constant
        node, but a constant LUT can also exist with non-empty ``inputs``.
    outputs: :class:`~cana.function_outputs.FunctionOutputs` holding a dense
        :class:`~cana.bit_lut.BitLut` and/or other representations.  Row ``s``
        of the dense LUT is the next value when the input state (in the
        ``inputs`` order, MSB = ``inputs[0]``) equals ``s``, matching
        :func:`~cana.utils.output_transitions`.
"""

NetworkDict = Dict[int, NodeSpec]
"""Network wiring + per-node outputs keyed by node id."""

PathLike = Union[str, Path]


def expressions_to_network_dict(
    expressions: Mapping[str, str],
    extra_names: Optional[Iterable[str]] = None,
) -> NetworkDict:
    """Build a ``NetworkDict`` from ``{target: boolean_expression}``.

    Names that appear only as inputs (free inputs) receive the identity
    update ``f(x) = x``, matching BBM's booleannet representation of inputs.
    Literal constants use empty ``inputs`` and a length-1 LUT.
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

    network_dict: NetworkDict = {}
    for i, name in enumerate(ordered):
        expr = normalize_operators(completed[name], "python")
        if expr in ("True", "true", "1"):
            network_dict[i] = {
                "name": name,
                "inputs": [],
                "outputs": FunctionOutputs(
                    bit_lut=BitLut.from_sequence([1])
                ),
            }
            continue
        if expr in ("False", "false", "0"):
            network_dict[i] = {
                "name": name,
                "inputs": [],
                "outputs": FunctionOutputs(
                    bit_lut=BitLut.from_sequence([0])
                ),
            }
            continue
        input_names = regulator_names_in_expression(expr, ordered)
        if expr.strip() == name and name not in input_names:
            input_names = [name]
        input_ids = [name_to_id[n] for n in input_names]
        lut = BitLut.from_sequence(output_transitions(expr, input_names))
        network_dict[i] = {
            "name": name,
            "inputs": input_ids,
            "outputs": FunctionOutputs(bit_lut=lut),
        }
    return network_dict


def sorted_node_ids(network_dict: Mapping[int, Mapping[str, Any]]) -> List[int]:
    """Return node ids in ascending order (export order)."""
    return sorted(network_dict.keys())


def input_names_for_node(
    network_dict: Mapping[int, Mapping[str, Any]], node_id: int
) -> List[str]:
    """Return input names for ``node_id`` in LUT bit order."""
    return [network_dict[j]["name"] for j in network_dict[node_id]["inputs"]]


def name_keyed_luts(
    network_or_network_dict: Any,
) -> Dict[str, Tuple[Tuple[str, ...], Tuple[int, ...]]]:
    """Return ``{name: (sorted_input_names, lut)}`` for equivalence checks.

    Accepts a :class:`~cana.boolean_network.BooleanNetwork` or a ``NetworkDict``.
    Input order is canonicalized by sorting names; the LUT is reordered
    to match.
    """
    if hasattr(network_or_network_dict, "network_dict") and hasattr(
        network_or_network_dict, "nodes"
    ):
        network_dict = network_or_network_dict.network_dict
        id_to_name = {
            i: network_or_network_dict.nodes[i].name
            for i in range(network_or_network_dict.Nnodes)
        }
    else:
        network_dict = network_or_network_dict
        id_to_name = {i: network_dict[i]["name"] for i in network_dict}

    result: Dict[str, Tuple[Tuple[str, ...], Tuple[int, ...]]] = {}
    for i, entry in network_dict.items():
        name = entry["name"]
        input_names = [id_to_name[j] for j in entry["inputs"]]
        sorted_names = tuple(sorted(input_names))
        outputs = entry["outputs"]
        if isinstance(outputs, FunctionOutputs):
            lut = outputs.get_bit_lut()
        elif isinstance(outputs, BitLut):
            lut = outputs
        else:
            lut = BitLut.from_sequence(outputs)
        lut = lut.reorder(input_names, list(sorted_names))
        result[name] = (sorted_names, tuple(int(v) for v in lut))
    return result


def equivalent_network_dicts(a: Any, b: Any) -> bool:
    """True if two networks / NetworkDicts have the same name-keyed LUTs."""
    return name_keyed_luts(a) == name_keyed_luts(b)
