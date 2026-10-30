# -*- coding: utf-8 -*-
"""BNS / Cell Collective ``.cnet`` (BLIF-like) import and export."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from cana.function_outputs.bit_lut import BitLut
from cana.boolean_node import BooleanNode
from cana.cutils import binstate_to_statenum, expand_logic_line, statenum_to_binstate
from cana.function_outputs import FunctionOutputs, OneSymbolSchemata


def _logic_line_has_wildcard(bits: str) -> bool:
    """True if an input pattern uses don't-care ``2`` or ``-``."""
    return "2" in bits or "-" in bits


def _collect_logic_lines(
    lines: List[str], start: int
) -> Tuple[List[str], int]:
    """Collect consecutive logic lines starting at ``start``; return (lines, next_i)."""
    collected: List[str] = []
    i = start
    while i < len(lines):
        logic_line = lines[i].strip()
        if (
            logic_line == ""
            or logic_line == "\n"
            or len(logic_line) <= 1
            or logic_line[0] in ".#"
        ):
            break
        collected.append(logic_line)
        i += 1
    return collected, i


def _schemata_from_logic_lines(logic_lines: List[str]) -> OneSymbolSchemata:
    """Build polarity → schema-set from raw cnet logic lines (normalize ``-`` → ``2``)."""
    schemata: OneSymbolSchemata = {0: set(), 1: set()}
    for logic_line in logic_lines:
        parts = logic_line.split()
        bits, out_bit = parts[0], int(parts[1])
        bits = bits.replace("-", "2")
        schemata.setdefault(out_bit, set()).add(bits)
    # Drop empty polarity keys for a cleaner store.
    return {p: s for p, s in schemata.items() if s}


def _bit_lut_from_logic_lines(logic_lines: List[str], indegree: int) -> BitLut:
    """Expand logic lines (including wildcards) into a dense ``BitLut``."""
    lut = BitLut(indegree)
    for logic_line in logic_lines:
        for nlogicline in expand_logic_line(logic_line):
            bits, out_bit = nlogicline.split()[0], nlogicline.split()[1]
            lut[binstate_to_statenum(bits)] = int(out_bit)
    return lut


def parse_cnet(text: str) -> List[BooleanNode]:
    """Parse BNS / Cell Collective ``.cnet`` text into a node list.

    Format sketch::

        .v <N>
        .l <id> <name>
        .n <id> <k> <in1> ... <ink>
        <binstate> <out>   # may use ``2`` don't-care bits (``-`` also accepted)
        .e

    Per-node heuristic: if any logic line uses don't-care ``2``/``-``, store
    that node's cover as ``one_symbol_schemata`` only; otherwise import as a
    dense ``bit_lut`` (including ON-only minterm listings with implied 0s).

    Import is faithful: missing peer representations are not filled. Call
    ``BooleanNetwork.fill_function_outputs(...)`` (or the node helper) when
    you need the other slot.
    """
    # Scratch during parse; materialize via BooleanNode factories at the end.
    # Each entry: name, inputs, kind ("bit_lut"|"one_symbol"), payload, optional k.
    scratch: Dict[int, Dict[str, Any]] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if not line or line[0] == "#" or line.strip() == "":
            continue
        if ".v" in line:
            nnodes = int(line.split()[1])
            for inode in range(nnodes):
                scratch[inode] = {
                    "name": "",
                    "inputs": [],
                    "kind": "bit_lut",
                    "payload": BitLut(0),
                }
        elif ".l" in line:
            parts = line.split()
            # Tolerate missing / empty names (``.l <id>`` or ``.l <id> ``).
            if len(parts) >= 3:
                scratch[int(parts[1]) - 1]["name"] = parts[2]
        elif ".n" in line:
            parts = line.split()
            inode = int(parts[1]) - 1
            indegree = int(parts[2])
            for j in range(indegree):
                scratch[inode]["inputs"].append(int(parts[3 + j]) - 1)

            if indegree <= 0:
                # Constant / free-input: a single output token (len 1), not a
                # ``<bits> <out>`` pair — peek one line without the pair collector.
                if i >= len(lines):
                    logic_line = ""
                else:
                    logic_line = lines[i].strip()
                    i += 1
                if logic_line == "":
                    scratch[inode]["inputs"] = [inode]
                    scratch[inode]["kind"] = "bit_lut"
                    scratch[inode]["payload"] = BitLut.from_sequence([0, 1])
                else:
                    scratch[inode]["kind"] = "bit_lut"
                    scratch[inode]["payload"] = BitLut.from_sequence(
                        [int(logic_line)]
                    )
            else:
                logic_lines, i = _collect_logic_lines(lines, i)
                has_wildcard = any(
                    _logic_line_has_wildcard(ll.split()[0]) for ll in logic_lines
                )
                if has_wildcard:
                    scratch[inode]["kind"] = "one_symbol"
                    scratch[inode]["payload"] = _schemata_from_logic_lines(
                        logic_lines
                    )
                    scratch[inode]["k"] = indegree
                else:
                    scratch[inode]["kind"] = "bit_lut"
                    scratch[inode]["payload"] = _bit_lut_from_logic_lines(
                        logic_lines, indegree
                    )
        elif ".e" in line:
            break

    if not scratch:
        raise ValueError("No cnet nodes found")

    nodes: List[BooleanNode] = []
    for inode in range(len(scratch)):
        entry = scratch[inode]
        name = entry["name"]
        inputs = list(entry["inputs"])
        if entry["kind"] == "one_symbol":
            nodes.append(
                BooleanNode.from_one_symbol_schemata(
                    entry["payload"],
                    k=entry.get("k"),
                    id=inode,
                    name=name,
                    inputs=inputs,
                )
            )
        else:
            nodes.append(
                BooleanNode.from_bit_lut(
                    entry["payload"],
                    id=inode,
                    name=name,
                    inputs=inputs,
                )
            )
    return nodes


def _append_dense_lut_lines(
    parts: List[str], lut: BitLut, k: int
) -> None:
    """Append dense ``2**k`` (or constant) lut rows to ``parts``."""
    for statenum in range(2**k if k > 0 else 1):
        if k == 0:
            parts.append(str(lut[statenum]))
        else:
            parts.append(
                "{0} {1}".format(
                    statenum_to_binstate(statenum, base=k),
                    lut[statenum],
                )
            )


def _append_schemata_lines(
    parts: List[str], schemata: OneSymbolSchemata
) -> None:
    """Append one-symbol schema lines (``2`` don't-cares) to ``parts``."""
    for polarity in sorted(schemata):
        for schema in sorted(schemata[polarity]):
            parts.append("{0} {1}".format(schema, polarity))


def _node_label(i: int, name: str) -> str:
    return "{0} ({1!r})".format(i, name) if name else str(i)


def _require_representation(
    nodes: Sequence[BooleanNode],
    *,
    representation: str,
) -> None:
    """Raise if any node lacks the requested ``FunctionOutputs`` slot."""
    missing = []
    for i, node in enumerate(nodes):
        outputs = node.outputs
        if representation == "bit_lut":
            if outputs.bit_lut is None:
                try:
                    outputs.get_bit_lut()
                except ValueError:
                    missing.append(_node_label(i, node.name))
        elif representation == "one_symbol_schemata":
            if outputs.one_symbol_schemata is None:
                try:
                    outputs.get_one_symbol_schemata()
                except ValueError:
                    missing.append(_node_label(i, node.name))
        else:
            raise ValueError(
                "Unknown representation {0!r}".format(representation)
            )
    if missing:
        raise ValueError(
            "Cannot export cnet as {0}: missing on node(s) {1}. "
            "Run network.fill_function_outputs({0!r}) "
            "(or node.fill_function_outputs({0!r})) first.".format(
                representation, ", ".join(missing)
            )
        )


def nodes_to_cnet(
    nodes: Sequence[BooleanNode],
    *,
    adjust_no_input: bool = False,
    one_symbol_schemata: bool = False,
) -> str:
    """Export a node list to BNS / Cell Collective ``.cnet`` text.

    Args:
        nodes: Dense ``list[BooleanNode]`` hub (``nodes[i].id == i``).
        adjust_no_input: If True, rewrite empty-input nodes as self-loops
            with identity LUT ``[0, 1]`` (BNS compatibility).
        one_symbol_schemata: If True, export one-symbol (wildcard) covers;
            if False (default), export dense ``2**k`` LUT rows.

    Dense export materializes ``bit_lut`` via :meth:`FunctionOutputs.get_bit_lut`
    when only expression/schemata are present. Schemata export materializes
    ``one_symbol_schemata`` similarly. If a representation cannot be derived,
    raises ``ValueError``.
    """
    # Working copies so adjust_no_input does not mutate the caller's nodes.
    local_inputs: List[List[int]] = [list(n.inputs) for n in nodes]
    local_outputs: List[FunctionOutputs] = []
    for node in nodes:
        outputs = node.outputs
        if not isinstance(outputs, FunctionOutputs):
            if isinstance(outputs, BitLut):
                outputs = FunctionOutputs(bit_lut=outputs)
            else:
                outputs = FunctionOutputs.from_sequence(outputs)
        local_outputs.append(outputs)

    if adjust_no_input:
        for i in range(len(nodes)):
            if len(local_inputs[i]) == 0:
                local_inputs[i] = [i]
                local_outputs[i] = FunctionOutputs(
                    bit_lut=BitLut.from_sequence([0, 1])
                )

    # Temporary node views for the representation check / export loop.
    export_nodes = [
        BooleanNode(
            id=i,
            name=nodes[i].name,
            k=len(local_inputs[i]),
            inputs=local_inputs[i],
            outputs=local_outputs[i],
        )
        for i in range(len(nodes))
    ]

    if one_symbol_schemata:
        _require_representation(export_nodes, representation="one_symbol_schemata")
    else:
        _require_representation(export_nodes, representation="bit_lut")

    nnodes = len(export_nodes)
    parts: List[str] = [".v {0}".format(nnodes), ""]
    for i, node in enumerate(export_nodes):
        if node.name:
            parts.append(".l {0} {1}".format(i + 1, node.name))
    if any(n.name for n in export_nodes):
        parts.append("")
    for i, node in enumerate(export_nodes):
        k = len(node.inputs)
        parts.append(
            ".n {0} {1} {2}".format(
                i + 1,
                k,
                " ".join(str(v + 1) for v in node.inputs),
            ).rstrip()
        )
        outputs = node.outputs
        if one_symbol_schemata:
            if k == 0:
                # Constant cnet line is a single bit (not a schema pair).
                if outputs.bit_lut is not None:
                    parts.append(str(outputs.bit_lut[0]))
                else:
                    lut = FunctionOutputs.one_symbol_to_bit_lut(
                        outputs.one_symbol_schemata, 0
                    )
                    parts.append(str(lut[0]))
            else:
                _append_schemata_lines(parts, outputs.one_symbol_schemata)
        else:
            _append_dense_lut_lines(parts, outputs.bit_lut, k)
        parts.append("")
    return "\n".join(parts) + "\n"
