# -*- coding: utf-8 -*-
"""Build and compare CANA networks as ``list[BooleanNode]`` hubs.

Invariant: for all ``i``, ``nodes[i].id == i``, and every ``inputs`` entry
is an index into that list.  Node sets are dense ``0 .. N-1``; CANA does
not delete BN nodes after construction.

Glossary
--------
positional expression
    ``FunctionOutputs`` peer IR over ``x0``…``x{k-1}``.
named expression
    Gene-named formula string at the I/O boundary.
update rule
    ``(name, named_expression)`` from a format line (often a ``dict``).
detached rule
    :class:`DetachedRule` — name + ``input_names`` + positional FO, not
    hub-wired (no free-input siblings).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from cana.function_outputs.bit_lut import BitLut
from cana.boolean_node import BooleanNode
from cana.function_outputs import FunctionOutputs
from cana.function_outputs.expression import (
    expression_identifiers,
    positional_to_named,
    normalize_operators,
    regulator_names_in_expression,
    named_to_positional,
)

__all__ = [
    "DetachedRule",
    "named_expression_to_detached_rule",
    "detached_rule_to_node",
    "update_rules_to_nodes",
    "input_names",
    "name_keyed_luts",
    "equivalent_networks",
]


@dataclass(frozen=True)
class DetachedRule:
    """One update rule, detached from any network hub.

    ``outputs`` holds a positional expression peer.  ``input_names[i]`` is
    the regulator for positional ``xi`` (MSB = ``input_names[0]``).
    Network ``id`` / integer ``inputs`` are not set.
    """

    name: str
    input_names: Tuple[str, ...]
    outputs: FunctionOutputs

    @property
    def k(self) -> int:
        return int(self.outputs.k)

    def to_named_expression(self, style: str = "symbolic") -> str:
        """Reconstruct a named expression from positional IR + input names."""
        expr = self.outputs.positional_expression
        if expr is None:
            raise ValueError("DetachedRule.outputs has no positional_expression")
        named = positional_to_named(expr, self.input_names)
        if style == "python":
            return named
        return normalize_operators(named, style)

    def as_update_rule(self, style: str = "symbolic") -> Tuple[str, str]:
        """Return ``(name, named_expression)``."""
        return self.name, self.to_named_expression(style=style)


def named_expression_to_detached_rule(
    name: str,
    named_expression: str,
    *,
    input_names: Optional[Sequence[str]] = None,
) -> DetachedRule:
    """Bind a named expression to a :class:`DetachedRule`.

    If *input_names* is ``None``, regulators are unique identifiers in
    first-appearance order (single-rule / synthetic import).  Otherwise the
    caller supplies order (full-model link uses the global name list via
    :func:`regulator_names_in_expression`).
    """
    expr = normalize_operators(named_expression, "python")
    if expr in ("True", "true", "1"):
        return DetachedRule(
            name=name,
            input_names=(),
            outputs=FunctionOutputs(positional_expression="True", k=0),
        )
    if expr in ("False", "false", "0"):
        return DetachedRule(
            name=name,
            input_names=(),
            outputs=FunctionOutputs(positional_expression="False", k=0),
        )

    if input_names is None:
        toks = expression_identifiers(expr)
        seen = set()
        regs: List[str] = []
        for t in toks:
            if t not in seen:
                seen.add(t)
                regs.append(t)
        # Identity free-input style: bare name with no other tokens.
        if expr.strip() == name and name not in regs:
            regs = [name]
    else:
        regs = list(input_names)
        if expr.strip() == name and name not in regs:
            regs = [name]

    positional = named_to_positional(expr, regs)
    return DetachedRule(
        name=name,
        input_names=tuple(regs),
        outputs=FunctionOutputs(
            positional_expression=positional, k=len(regs)
        ),
    )


def detached_rule_to_node(
    rule: DetachedRule, *, id: int, inputs: Sequence[int]
) -> BooleanNode:
    """Wire a :class:`DetachedRule` into a hub :class:`BooleanNode`."""
    expr = rule.outputs.positional_expression
    if expr is None:
        raise ValueError("DetachedRule.outputs has no positional_expression")
    return BooleanNode.from_positional_expression(
        expr,
        k=rule.k,
        id=id,
        name=rule.name,
        inputs=list(inputs),
    )


def update_rules_to_nodes(
    update_rules: Mapping[str, str],
    extra_names: Optional[Iterable[str]] = None,
) -> List[BooleanNode]:
    """Build a dense ``list[BooleanNode]`` from ``{target: named_expression}``.

    Names that appear only as inputs (free inputs) receive the identity
    update ``f(x) = x``, matching BBM's booleannet representation of inputs.
    Literal constants use empty ``inputs`` and positional ``True``/``False``.

    Each node's ``outputs`` stores a **positional** expression without
    building a dense LUT at import time.
    """
    # Preserve insertion order of targets; append free inputs alphabetically.
    ordered: List[str] = list(update_rules.keys())
    seen = set(ordered)
    mentioned: set = set()
    for named_expression in update_rules.values():
        mentioned.update(
            expression_identifiers(
                normalize_operators(named_expression, "python")
            )
        )
    if extra_names:
        mentioned.update(extra_names)
    for name in sorted(mentioned):
        if name not in seen:
            ordered.append(name)
            seen.add(name)

    name_to_id = {name: i for i, name in enumerate(ordered)}
    completed: Dict[str, str] = dict(update_rules)
    for name in ordered:
        if name not in completed:
            completed[name] = name  # identity

    nodes: List[BooleanNode] = []
    for i, name in enumerate(ordered):
        named = completed[name]
        # Global regulator order (stable across the hub name list).
        norm = normalize_operators(named, "python")
        if norm in ("True", "true", "1", "False", "false", "0"):
            regs = []
        else:
            regs = regulator_names_in_expression(norm, ordered)
            if norm.strip() == name and name not in regs:
                regs = [name]
        rule = named_expression_to_detached_rule(
            name, named, input_names=regs
        )
        input_ids = [name_to_id[n] for n in rule.input_names]
        nodes.append(detached_rule_to_node(rule, id=i, inputs=input_ids))
    return nodes


def input_names(nodes: Sequence[BooleanNode], node_id: int) -> List[str]:
    """Return input names for ``node_id`` in LUT bit order."""
    return [nodes[j].name for j in nodes[node_id].inputs]


def _as_nodes(network_or_nodes: Any) -> Sequence[BooleanNode]:
    """Accept a :class:`~cana.boolean_network.BooleanNetwork` or node list."""
    if hasattr(network_or_nodes, "nodes") and not isinstance(
        network_or_nodes, (list, tuple)
    ):
        return network_or_nodes.nodes
    return network_or_nodes


def name_keyed_luts(
    network_or_nodes: Any,
) -> Dict[str, Tuple[Tuple[str, ...], Tuple[int, ...]]]:
    """Return ``{name: (sorted_input_names, lut)}`` for equivalence checks.

    Accepts a :class:`~cana.boolean_network.BooleanNetwork` or a
    ``Sequence[BooleanNode]``.  Input order is canonicalized by sorting
    names; the LUT is reordered to match.
    """
    nodes = _as_nodes(network_or_nodes)
    result: Dict[str, Tuple[Tuple[str, ...], Tuple[int, ...]]] = {}
    for node in nodes:
        input_name_list = [nodes[j].name for j in node.inputs]
        sorted_names = tuple(sorted(input_name_list))
        outputs = node.outputs
        if isinstance(outputs, FunctionOutputs):
            lut = outputs.get_bit_lut()
        elif isinstance(outputs, BitLut):
            lut = outputs
        else:
            lut = BitLut.from_sequence(outputs)
        lut = lut.reorder(input_name_list, list(sorted_names))
        result[node.name] = (sorted_names, tuple(int(v) for v in lut))
    return result


def equivalent_networks(a: Any, b: Any) -> bool:
    """True if two networks / node lists have the same name-keyed LUTs."""
    return name_keyed_luts(a) == name_keyed_luts(b)
