# -*- coding: utf-8 -*-
"""BNS / Cell Collective ``.cnet`` (BLIF-like) import and export."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping

from cana.bit_lut import BitLut
from cana.cutils import binstate_to_statenum, expand_logic_line, statenum_to_binstate
from cana.function_outputs import FunctionOutputs
from cana.network_dict import NetworkDict


def parse_cnet(text: str) -> NetworkDict:
    """Parse BNS / Cell Collective ``.cnet`` text into a ``NetworkDict``.

    Format sketch::

        .v <N>
        .l <id> <name>
        .n <id> <k> <in1> ... <ink>
        <binstate> <out>   # may use ``2`` don't-care bits (``-`` also accepted)
        .e
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
            network_dict[int(parts[1]) - 1]["name"] = parts[2]
        elif ".n" in line:
            parts = line.split()
            inode = int(parts[1]) - 1
            indegree = int(parts[2])
            for j in range(indegree):
                network_dict[inode]["inputs"].append(int(parts[3 + j]) - 1)

            if i >= len(lines):
                logic_line = ""
            else:
                logic_line = lines[i].strip()
                i += 1

            if indegree <= 0:
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
                lut = BitLut(indegree)
                while (
                    logic_line != ""
                    and logic_line != "\n"
                    and len(logic_line) > 1
                ):
                    for nlogicline in expand_logic_line(logic_line):
                        bits, out_bit = nlogicline.split()[0], nlogicline.split()[1]
                        lut[binstate_to_statenum(bits)] = int(out_bit)
                    if i >= len(lines):
                        break
                    logic_line = lines[i].strip()
                    i += 1
                network_dict[inode]["outputs"] = FunctionOutputs(bit_lut=lut)
        elif ".e" in line:
            break

    if not network_dict:
        raise ValueError("No cnet nodes found")
    return network_dict  # type: ignore[return-value]


def network_dict_to_cnet(
    network_dict: Mapping[int, Mapping[str, Any]],
    *,
    adjust_no_input: bool = False,
) -> str:
    """Export ``NetworkDict`` to BNS / Cell Collective ``.cnet`` text.

    Args:
        network_dict: CANA ``NetworkDict``.
        adjust_no_input: If True, rewrite empty-input nodes as self-loops
            with identity LUT ``[0, 1]`` (BNS compatibility).
    """
    local: Dict[int, Dict[str, Any]] = {}
    for i, data in network_dict.items():
        outputs = data["outputs"]
        if isinstance(outputs, FunctionOutputs):
            lut = outputs.get_bit_lut()
        elif isinstance(outputs, BitLut):
            lut = outputs
        else:
            lut = BitLut.from_sequence(outputs)
        local[i] = {
            "name": data["name"],
            "inputs": list(data["inputs"]),
            "lut": lut,
        }
    if adjust_no_input:
        for i, data in local.items():
            if len(data["inputs"]) == 0:
                data["inputs"] = [i]
                data["lut"] = BitLut.from_sequence([0, 1])

    nnodes = len(local)
    parts: List[str] = [".v {0}".format(nnodes), ""]
    for i in range(nnodes):
        parts.append(".l {0} {1}".format(i + 1, local[i]["name"]))
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
        lut = local[i]["lut"]
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
        parts.append("")
    return "\n".join(parts) + "\n"
