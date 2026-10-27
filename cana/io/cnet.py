# -*- coding: utf-8 -*-
"""BNS / Cell Collective ``.cnet`` (BLIF-like) import and export."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

from cana.bit_lut import BitLut
from cana.cutils import binstate_to_statenum, expand_logic_line, statenum_to_binstate
from cana.function_outputs import FunctionOutputs, OneSymbolSchemata
from cana.network_dict import NetworkDict


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


def parse_cnet(text: str) -> NetworkDict:
    """Parse BNS / Cell Collective ``.cnet`` text into a ``NetworkDict``.

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
    network_dict: Dict[int, Dict[str, Any]] = {}
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
                network_dict[inode] = {
                    "name": "",
                    "inputs": [],
                    "outputs": FunctionOutputs(bit_lut=BitLut(0)),
                }
        elif ".l" in line:
            parts = line.split()
            # Tolerate missing / empty names (``.l <id>`` or ``.l <id> ``).
            if len(parts) >= 3:
                network_dict[int(parts[1]) - 1]["name"] = parts[2]
        elif ".n" in line:
            parts = line.split()
            inode = int(parts[1]) - 1
            indegree = int(parts[2])
            for j in range(indegree):
                network_dict[inode]["inputs"].append(int(parts[3 + j]) - 1)

            if indegree <= 0:
                # Constant / free-input: a single output token (len 1), not a
                # ``<bits> <out>`` pair — peek one line without the pair collector.
                if i >= len(lines):
                    logic_line = ""
                else:
                    logic_line = lines[i].strip()
                    i += 1
                if logic_line == "":
                    network_dict[inode]["inputs"] = [inode]
                    network_dict[inode]["outputs"] = FunctionOutputs(
                        bit_lut=BitLut.from_sequence([0, 1])
                    )
                else:
                    network_dict[inode]["outputs"] = FunctionOutputs(
                        bit_lut=BitLut.from_sequence([int(logic_line)])
                    )
            else:
                logic_lines, i = _collect_logic_lines(lines, i)
                has_wildcard = any(
                    _logic_line_has_wildcard(ll.split()[0]) for ll in logic_lines
                )
                if has_wildcard:
                    schemata = _schemata_from_logic_lines(logic_lines)
                    network_dict[inode]["outputs"] = FunctionOutputs(
                        one_symbol_schemata=schemata, k=indegree
                    )
                else:
                    lut = _bit_lut_from_logic_lines(logic_lines, indegree)
                    network_dict[inode]["outputs"] = FunctionOutputs(bit_lut=lut)
        elif ".e" in line:
            break

    if not network_dict:
        raise ValueError("No cnet nodes found")
    return network_dict  # type: ignore[return-value]


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
    local: Mapping[int, Mapping[str, Any]],
    *,
    representation: str,
) -> None:
    """Raise if any node lacks the requested ``FunctionOutputs`` slot."""
    missing = []
    for i in sorted(local):
        outputs = local[i]["outputs"]
        if representation == "bit_lut":
            if outputs.bit_lut is None:
                try:
                    outputs.get_bit_lut()
                except ValueError:
                    missing.append(_node_label(i, local[i]["name"]))
        elif representation == "one_symbol_schemata":
            if outputs.one_symbol_schemata is None:
                try:
                    outputs.get_one_symbol_schemata()
                except ValueError:
                    missing.append(_node_label(i, local[i]["name"]))
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


def network_dict_to_cnet(
    network_dict: Mapping[int, Mapping[str, Any]],
    *,
    adjust_no_input: bool = False,
    one_symbol_schemata: bool = False,
) -> str:
    """Export ``NetworkDict`` to BNS / Cell Collective ``.cnet`` text.

    Args:
        network_dict: CANA ``NetworkDict``.
        adjust_no_input: If True, rewrite empty-input nodes as self-loops
            with identity LUT ``[0, 1]`` (BNS compatibility).
        one_symbol_schemata: If True, export one-symbol (wildcard) covers;
            if False (default), export dense ``2**k`` LUT rows.

    Dense export materializes ``bit_lut`` via :meth:`FunctionOutputs.get_bit_lut`
    when only expression/schemata are present. Schemata export materializes
    ``one_symbol_schemata`` similarly. If a representation cannot be derived,
    raises ``ValueError``.
    """
    local: Dict[int, Dict[str, Any]] = {}
    for i, data in network_dict.items():
        outputs = data["outputs"]
        if not isinstance(outputs, FunctionOutputs):
            if isinstance(outputs, BitLut):
                outputs = FunctionOutputs(bit_lut=outputs)
            else:
                outputs = FunctionOutputs.from_sequence(outputs)
        local[i] = {
            "name": data["name"],
            "inputs": list(data["inputs"]),
            "outputs": outputs,
        }

    if adjust_no_input:
        for i, data in local.items():
            if len(data["inputs"]) == 0:
                data["inputs"] = [i]
                data["outputs"] = FunctionOutputs(
                    bit_lut=BitLut.from_sequence([0, 1])
                )

    if one_symbol_schemata:
        _require_representation(local, representation="one_symbol_schemata")
    else:
        _require_representation(local, representation="bit_lut")

    nnodes = len(local)
    parts: List[str] = [".v {0}".format(nnodes), ""]
    for i in range(nnodes):
        name = local[i]["name"]
        if name:
            parts.append(".l {0} {1}".format(i + 1, name))
    if any(local[i]["name"] for i in range(nnodes)):
        parts.append("")
    for i in range(nnodes):
        k = len(local[i]["inputs"])
        parts.append(
            ".n {0} {1} {2}".format(
                i + 1,
                k,
                " ".join(str(v + 1) for v in local[i]["inputs"]),
            ).rstrip()
        )
        outputs = local[i]["outputs"]
        if one_symbol_schemata:
            if k == 0:
                # Constant cnet line is a single bit (not a schema pair).
                if outputs.bit_lut is not None:
                    parts.append(str(outputs.bit_lut[0]))
                else:
                    lut = FunctionOutputs.bit_lut_from_one_symbol_schemata(
                        outputs.one_symbol_schemata, 0
                    )
                    parts.append(str(lut[0]))
            else:
                _append_schemata_lines(parts, outputs.one_symbol_schemata)
        else:
            _append_dense_lut_lines(parts, outputs.bit_lut, k)
        parts.append("")
    return "\n".join(parts) + "\n"
