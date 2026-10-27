# -*- coding: utf-8 -*-
"""
Import / export Boolean networks via the CANA ``NetworkDict`` hub.

A ``NetworkDict`` maps node id → ``{name, inputs, outputs}``. File formats
(booleannet, bnet, aeon, sbml, cnet) convert to/from this structure. Expression
formats store positional IR on :class:`~cana.function_outputs.FunctionOutputs`
at import; exporters may materialize dense LUTs. Name-keyed LUT equivalence
is the round-trip authority across formats.

Modules:
    expressions: re-exports of :mod:`cana.function_outputs.expression`
    booleannet, bnet, aeon, sbml, cnet: per-format parse / export

The hub types and builders live in :mod:`cana.network_dict`.

Format parsers/exporters are loaded lazily so ``cana.io.expressions`` can be
imported without pulling every format module.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Union

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
    "network_dict_to_aeon": ("cana.io.aeon", "network_dict_to_aeon"),
    "parse_aeon": ("cana.io.aeon", "parse_aeon"),
    "network_dict_to_bnet": ("cana.io.bnet", "network_dict_to_bnet"),
    "parse_bnet": ("cana.io.bnet", "parse_bnet"),
    "network_dict_to_booleannet": (
        "cana.io.booleannet",
        "network_dict_to_booleannet",
    ),
    "parse_booleannet": ("cana.io.booleannet", "parse_booleannet"),
    "network_dict_to_cnet": ("cana.io.cnet", "network_dict_to_cnet"),
    "parse_cnet": ("cana.io.cnet", "parse_cnet"),
    "lut_to_expression": ("cana.io.expressions", "lut_to_expression"),
    "normalize_operators": ("cana.io.expressions", "normalize_operators"),
    "network_dict_to_sbml": ("cana.io.sbml", "network_dict_to_sbml"),
    "parse_sbml": ("cana.io.sbml", "parse_sbml"),
}


def __getattr__(name: str):
    """Lazy-load format helpers so ``import cana.io.expressions`` stays cycle-free."""
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
    """Parse ``text`` of the given format into a ``NetworkDict``."""
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
    network_dict: Mapping[int, Mapping[str, Any]],
    fmt: str,
    **kwargs: Any,
) -> str:
    """Serialize ``network_dict`` to the requested format string.

    Extra ``kwargs`` are forwarded to format-specific exporters (e.g.
    ``adjust_no_input`` for cnet).
    """
    fmt = normalize_format(fmt)
    if fmt == "booleannet":
        from cana.io.booleannet import network_dict_to_booleannet

        return network_dict_to_booleannet(network_dict)
    if fmt == "bnet":
        from cana.io.bnet import network_dict_to_bnet

        return network_dict_to_bnet(network_dict)
    if fmt == "aeon":
        from cana.io.aeon import network_dict_to_aeon

        return network_dict_to_aeon(network_dict)
    if fmt == "sbml":
        from cana.io.sbml import network_dict_to_sbml

        return network_dict_to_sbml(network_dict)
    if fmt == "cnet":
        from cana.io.cnet import network_dict_to_cnet

        return network_dict_to_cnet(network_dict, **kwargs)
    raise ValueError("export_string does not handle format {0!r}".format(fmt))


def convert(text: str, from_type: str, to_type: str, **kwargs: Any) -> str:
    """Convert a model string from ``from_type`` to ``to_type`` via ``NetworkDict``.

    Extra ``kwargs`` are forwarded to the destination exporter (e.g.
    ``adjust_no_input`` / ``one_symbol_schemata`` for cnet).
    """
    network_dict = parse_string(text, from_type)
    return export_string(network_dict, to_type, **kwargs)


__all__ = [
    "FORMAT_ALIASES",
    "convert",
    "export_string",
    "infer_format",
    "network_dict_to_aeon",
    "network_dict_to_bnet",
    "network_dict_to_booleannet",
    "network_dict_to_cnet",
    "network_dict_to_sbml",
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
