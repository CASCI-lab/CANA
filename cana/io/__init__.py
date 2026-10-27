# -*- coding: utf-8 -*-
"""
Import / export Boolean networks via a ``list[BooleanNode]`` hub.

File formats (booleannet, bnet, aeon, sbml, cnet) convert to/from a dense
list of :class:`~cana.boolean_node.BooleanNode` (invariant: ``nodes[i].id == i``).
Expression formats store positional IR on
:class:`~cana.function_outputs.FunctionOutputs` at import; exporters may
materialize dense LUTs. Name-keyed LUT equivalence is the round-trip
authority across formats.

Modules:
    booleannet, bnet, aeon, sbml, cnet: per-format parse / export

Builders / equivalence live in :mod:`cana.nodes_builder`.
Expression helpers live in :mod:`cana.function_outputs.expression`.

Format parsers/exporters are loaded lazily so importing a single helper
does not pull every format module.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Union

from cana.boolean_node import BooleanNode

PathLike = Union[str, Path]

FORMAT_ALIASES = {
    "booleannet": "booleannet",
    "bnet": "bnet",
    "aeon": "aeon",
    "sbml": "sbml",
    "cnet": "cnet",
}

# Name → (submodule, attribute) for lazy public re-exports.
_LAZY_EXPORTS = {
    "nodes_to_aeon": ("cana.io.aeon", "nodes_to_aeon"),
    "parse_aeon": ("cana.io.aeon", "parse_aeon"),
    "nodes_to_bnet": ("cana.io.bnet", "nodes_to_bnet"),
    "parse_bnet": ("cana.io.bnet", "parse_bnet"),
    "nodes_to_booleannet": ("cana.io.booleannet", "nodes_to_booleannet"),
    "parse_booleannet": ("cana.io.booleannet", "parse_booleannet"),
    "nodes_to_cnet": ("cana.io.cnet", "nodes_to_cnet"),
    "parse_cnet": ("cana.io.cnet", "parse_cnet"),
    "lut_to_expression": (
        "cana.function_outputs.expression",
        "lut_to_expression",
    ),
    "normalize_operators": (
        "cana.function_outputs.expression",
        "normalize_operators",
    ),
    "nodes_to_sbml": ("cana.io.sbml", "nodes_to_sbml"),
    "parse_sbml": ("cana.io.sbml", "parse_sbml"),
}


def __getattr__(name: str):
    """Lazy-load format / expression helpers without importing every submodule."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr = target
    import importlib

    value = getattr(importlib.import_module(mod_name), attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS) | set(__all__))


def normalize_format(fmt: Optional[str]) -> Optional[str]:
    """Map aliases to a canonical format name, or return ``None``."""
    if fmt is None:
        return None
    key = str(fmt).strip().lower()
    if key not in FORMAT_ALIASES:
        raise ValueError(
            "Unknown format {fmt!r}. Expected one of: {ok}".format(
                fmt=fmt, ok=", ".join(sorted(set(FORMAT_ALIASES.values())))
            )
        )
    return FORMAT_ALIASES[key]


def infer_format(path: PathLike) -> Optional[str]:
    """Infer format from a file path suffix, or ``None`` if ambiguous."""
    name = Path(path).name.lower()
    if name.endswith(".booleannet.txt"):
        return "booleannet"
    if name.endswith(".bnet"):
        return "bnet"
    if name.endswith(".aeon"):
        return "aeon"
    if name.endswith(".sbml") or name.endswith(".xml"):
        return "sbml"
    if name.endswith(".cnet"):
        return "cnet"
    return None


def parse_string(text: str, fmt: str):
    """Parse ``text`` of the given format into a ``list[BooleanNode]``."""
    fmt = normalize_format(fmt)
    if fmt == "booleannet":
        from cana.io.booleannet import parse_booleannet

        return parse_booleannet(text)
    if fmt == "bnet":
        from cana.io.bnet import parse_bnet

        return parse_bnet(text)
    if fmt == "aeon":
        from cana.io.aeon import parse_aeon

        return parse_aeon(text)
    if fmt == "sbml":
        from cana.io.sbml import parse_sbml

        return parse_sbml(text)
    if fmt == "cnet":
        from cana.io.cnet import parse_cnet

        return parse_cnet(text)
    raise ValueError("parse_string does not handle format {0!r}".format(fmt))


def export_string(
    nodes: Sequence[BooleanNode],
    fmt: str,
    **kwargs: Any,
) -> str:
    """Serialize a node list to the requested format string.

    Extra ``kwargs`` are forwarded to format-specific exporters (e.g.
    ``adjust_no_input`` for cnet, ``warn`` for expression formats).
    """
    fmt = normalize_format(fmt)
    if fmt == "booleannet":
        from cana.io.booleannet import nodes_to_booleannet

        return nodes_to_booleannet(nodes, **kwargs)
    if fmt == "bnet":
        from cana.io.bnet import nodes_to_bnet

        return nodes_to_bnet(nodes, **kwargs)
    if fmt == "aeon":
        from cana.io.aeon import nodes_to_aeon

        return nodes_to_aeon(nodes, **kwargs)
    if fmt == "sbml":
        from cana.io.sbml import nodes_to_sbml

        return nodes_to_sbml(nodes, **kwargs)
    if fmt == "cnet":
        from cana.io.cnet import nodes_to_cnet

        # ``warn`` applies to expression formats only.
        cnet_kwargs = {
            k: v for k, v in kwargs.items() if k != "warn"
        }
        return nodes_to_cnet(nodes, **cnet_kwargs)
    raise ValueError("export_string does not handle format {0!r}".format(fmt))


def convert(text: str, from_type: str, to_type: str, **kwargs: Any) -> str:
    """Convert a model string from ``from_type`` to ``to_type`` via a node list.

    Extra ``kwargs`` are forwarded to the destination exporter (e.g.
    ``adjust_no_input`` / ``one_symbol_schemata`` for cnet).
    """
    nodes = parse_string(text, from_type)
    return export_string(nodes, to_type, **kwargs)


__all__ = [
    "FORMAT_ALIASES",
    "convert",
    "export_string",
    "infer_format",
    "nodes_to_aeon",
    "nodes_to_bnet",
    "nodes_to_booleannet",
    "nodes_to_cnet",
    "nodes_to_sbml",
    "lut_to_expression",
    "normalize_format",
    "normalize_operators",
    "parse_aeon",
    "parse_bnet",
    "parse_booleannet",
    "parse_cnet",
    "parse_sbml",
    "parse_string",
]
