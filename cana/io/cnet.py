# -*- coding: utf-8 -*-
"""BNS / Cell Collective ``.cnet`` (BLIF-like) import and export."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping

from cana.cutils import binstate_to_statenum, expand_logic_line, statenum_to_binstate
from cana.lut_dict import LutDict


def parse_cnet(text: str) -> LutDict:
    """Parse BNS / Cell Collective ``.cnet`` text into a ``LutDict``.

    Format sketch::

        .v <N>
        .l <id> <name>
        .n <id> <k> <in1> ... <ink>
        <binstate> <out>   # may use ``-`` don't-care bits
        .e
    """
    lut_dict: Dict[int, Dict[str, Any]] = {}
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
                lut_dict[inode] = {"name": "", "regulators": [], "lut": []}
        elif ".l" in line:
            parts = line.split()
            lut_dict[int(parts[1]) - 1]["name"] = parts[2]
        elif ".n" in line:
            parts = line.split()
            inode = int(parts[1]) - 1
            indegree = int(parts[2])
            for j in range(indegree):
                lut_dict[inode]["regulators"].append(int(parts[3 + j]) - 1)

            lut_dict[inode]["lut"] = [
                0 for _ in range(2**indegree) if indegree > 0
            ]

            if i >= len(lines):
                logic_line = ""
            else:
                logic_line = lines[i].strip()
                i += 1

            if indegree <= 0:
                if logic_line == "":
                    lut_dict[inode]["regulators"] = [inode]
                    lut_dict[inode]["lut"] = [0, 1]
                else:
                    lut_dict[inode]["lut"] = [int(logic_line)]
            else:
                while (
                    logic_line != ""
                    and logic_line != "\n"
                    and len(logic_line) > 1
                ):
                    for nlogicline in expand_logic_line(logic_line):
                        bits, out_bit = nlogicline.split()[0], nlogicline.split()[1]
                        lut_dict[inode]["lut"][
                            binstate_to_statenum(bits)
                        ] = int(out_bit)
                    if i >= len(lines):
                        break
                    logic_line = lines[i].strip()
                    i += 1
        elif ".e" in line:
            break

    if not lut_dict:
        raise ValueError("No cnet nodes found")
    return lut_dict  # type: ignore[return-value]


def lut_dict_to_cnet(
    lut_dict: Mapping[int, Mapping[str, Any]],
    *,
    adjust_no_input: bool = False,
) -> str:
    """Export ``LutDict`` to BNS / Cell Collective ``.cnet`` text.

    Args:
        lut_dict: CANA ``LutDict``.
        adjust_no_input: If True, rewrite empty-regulator nodes as self-loops
            with identity LUT ``[0, 1]`` (BNS compatibility).
    """
    local: Dict[int, Dict[str, Any]] = {
        i: {
            "name": data["name"],
            "regulators": list(data["regulators"]),
            "lut": list(data["lut"]),
        }
        for i, data in lut_dict.items()
    }
    if adjust_no_input:
        for i, data in local.items():
            if len(data["regulators"]) == 0:
                data["regulators"] = [i]
                data["lut"] = [0, 1]

    nnodes = len(local)
    parts: List[str] = [".v {0}".format(nnodes), ""]
    for i in range(nnodes):
        parts.append(".l {0} {1}".format(i + 1, local[i]["name"]))
    parts.append("")
    for i in range(nnodes):
        k = len(local[i]["regulators"])
        parts.append(
            ".n {0} {1} {2}".format(
                i + 1,
                k,
                " ".join(str(v + 1) for v in local[i]["regulators"]),
            ).rstrip()
        )
        for statenum in range(2**k):
            if len(local[i]["lut"]) == 1:
                parts.append(str(local[i]["lut"][statenum]))
            else:
                parts.append(
                    "{0} {1}".format(
                        statenum_to_binstate(statenum, base=k),
                        local[i]["lut"][statenum],
                    )
                )
        parts.append("")
    return "\n".join(parts) + "\n"
