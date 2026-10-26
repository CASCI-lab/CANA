# -*- coding: utf-8 -*-
"""
Import / export Boolean networks via the CANA ``logic`` dict.

The ``logic`` dict is the network hub: a dense map from node id to a per-node
look-up table (LUT) plus wiring. Expression formats (booleannet, bnet, aeon,
sbml) are converted to/from this structure; textual expressions need not be
preserved across round-trips as long as name-keyed LUTs match.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TypedDict, Union

from cana.cutils import binstate_to_statenum, statenum_to_binstate
from cana.utils import output_transitions

# ---------------------------------------------------------------------------
# Logic schema
# ---------------------------------------------------------------------------


NodeLogic = TypedDict(
    "NodeLogic",
    {
        "name": str,
        # ``in`` is a reserved keyword, so the functional TypedDict form is required.
        "in": List[int],
        "out": List[int],
    },
)
"""One node's entry in a CANA ``logic`` dict.

Keys:
    name: Unique node name.
    in: Ordered list of input **node ids** (length ``k``).
    out: LUT of length ``2**k`` with entries in ``{0, 1}``.  Row ``s`` is
        the next value when the input state (in the ``in`` order, MSB =
        ``in[0]``) equals ``s``, matching :func:`output_transitions`.
"""


LogicDict = Dict[int, NodeLogic]

_BOOL_KEYWORDS = frozenset({"and", "or", "not", "true", "false", "xor"})
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_AEON_EDGE_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(-[>|?]?|\|)\s*([A-Za-z_][A-Za-z0-9_]*)\s*$"
)
_AEON_UPDATE_RE = re.compile(
    r"^\s*\$([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+?)\s*$"
)

FORMAT_ALIASES = {
    "logical": "booleannet",
    "boolean": "booleannet",
    "booleannet": "booleannet",
    "bnet": "bnet",
    "aeon": "aeon",
    "sbml": "sbml",
    "cnet": "cnet",
}

PathLike = Union[str, Path]


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


# ---------------------------------------------------------------------------
# Expression helpers
# ---------------------------------------------------------------------------


def normalize_operators(expr: str, style: str = "python") -> str:
    """Normalize boolean operators to ``and``/``or``/``not`` or ``&``/``|``/``!``.

    Args:
        expr: Boolean expression string.
        style: ``"python"`` (for :func:`output_transitions`) or ``"symbolic"``.
    """
    s = expr.strip()
    if style == "python":
        # Symbolic → python (word boundaries for & | ! handled carefully)
        s = re.sub(r"\bAND\b", "and", s, flags=re.IGNORECASE)
        s = re.sub(r"\bOR\b", "or", s, flags=re.IGNORECASE)
        s = re.sub(r"\bNOT\b", "not", s, flags=re.IGNORECASE)
        s = s.replace("&&", " and ").replace("||", " or ")
        s = re.sub(r"(?<![A-Za-z0-9_])!(?!=)", " not ", s)
        s = s.replace("&", " and ").replace("|", " or ")
        s = re.sub(r"\s+", " ", s).strip()
        return s
    if style == "symbolic":
        s = re.sub(r"\band\b", "&", s, flags=re.IGNORECASE)
        s = re.sub(r"\bor\b", "|", s, flags=re.IGNORECASE)
        s = re.sub(r"\bnot\b", "!", s, flags=re.IGNORECASE)
        s = re.sub(r"\s+", " ", s).strip()
        return s
    raise ValueError("style must be 'python' or 'symbolic'")


def _identifiers_in_expr(expr: str) -> List[str]:
    """Return identifier tokens in ``expr``, excluding boolean keywords."""
    return [
        tok
        for tok in _IDENT_RE.findall(expr)
        if tok.lower() not in _BOOL_KEYWORDS
    ]


def _inputs_for_expr(expr: str, known_names: Sequence[str]) -> List[str]:
    """Select known node names that appear as tokens in ``expr`` (stable order)."""
    padded = "*" + expr + "*"
    found: List[str] = []
    for name in known_names:
        if re.search(r"\W" + re.escape(name) + r"\W", padded):
            found.append(name)
    return found


def lut_to_expression(
    input_names: Sequence[str], out: Sequence[Any], style: str = "python"
) -> str:
    """Build a DNF expression for a LUT.

    Args:
        input_names: Input variable names in LUT bit order (MSB first).
        out: Truth table of length ``2**k``.
        style: ``"python"`` (``and``/``or``/``not``) or ``"symbolic"`` (``&``/``|``/``!``).
    """
    k = len(input_names)
    if k == 0:
        val = int(out[0]) if out else 0
        # Constant with no regulators: keep a boolean literal eval can handle
        # via a tautology / contradiction over a dummy — callers should avoid
        # k==0 by using identity for free inputs. Fall back to 0/1 strings that
        # parsers treat carefully; for eval we use True/False via or/and of empty.
        return "True" if val else "False"

    and_op = " and " if style == "python" else " & "
    or_op = " or " if style == "python" else " | "
    not_prefix = "not " if style == "python" else "!"

    terms: List[str] = []
    for state in range(2**k):
        if int(out[state]) != 1:
            continue
        bits = statenum_to_binstate(state, k)
        lits: List[str] = []
        for name, bit in zip(input_names, bits):
            if bit == "1":
                lits.append(name)
            else:
                lits.append(not_prefix + name)
        if len(lits) == 1:
            terms.append(lits[0])
        else:
            terms.append("(" + and_op.join(lits) + ")")

    if not terms:
        # Always false: x and not x
        x = input_names[0]
        if style == "python":
            return "({x} and not {x})".format(x=x)
        return "({x} & !{x})".format(x=x)

    if len(out) == 2**k and all(int(v) == 1 for v in out):
        x = input_names[0]
        if style == "python":
            return "({x} or not {x})".format(x=x)
        return "({x} | !{x})".format(x=x)

    return or_op.join(terms)


def permute_lut(
    out: Sequence[Any], old_inputs: Sequence[str], new_inputs: Sequence[str]
) -> List[int]:
    """Reorder a LUT from ``old_inputs`` bit order to ``new_inputs`` bit order."""
    if list(old_inputs) == list(new_inputs):
        return [int(v) for v in out]
    k = len(old_inputs)
    if k != len(new_inputs) or set(old_inputs) != set(new_inputs):
        raise ValueError("old_inputs and new_inputs must be permutations")
    name_to_old = {n: i for i, n in enumerate(old_inputs)}
    new_out = [0] * (2**k)
    for new_state in range(2**k):
        new_bits = statenum_to_binstate(new_state, k)
        old_bits = ["0"] * k
        for new_i, name in enumerate(new_inputs):
            old_bits[name_to_old[name]] = new_bits[new_i]
        old_state = binstate_to_statenum("".join(old_bits))
        new_out[new_state] = int(out[old_state])
    return new_out


def name_keyed_luts(
    network_or_logic: Any,
) -> Dict[str, Tuple[Tuple[str, ...], Tuple[int, ...]]]:
    """Return ``{name: (sorted_input_names, lut)}`` for equivalence checks.

    Accepts a :class:`~cana.boolean_network.BooleanNetwork` or a ``logic`` dict.
    Input order is canonicalized by sorting names; the LUT is permuted to match.
    """
    if hasattr(network_or_logic, "logic") and hasattr(network_or_logic, "nodes"):
        logic = network_or_logic.logic
        id_to_name = {i: network_or_logic.nodes[i].name for i in range(network_or_logic.Nnodes)}
    else:
        logic = network_or_logic
        id_to_name = {i: logic[i]["name"] for i in logic}

    result: Dict[str, Tuple[Tuple[str, ...], Tuple[int, ...]]] = {}
    for i, entry in logic.items():
        name = entry["name"]
        input_names = [id_to_name[j] for j in entry["in"]]
        sorted_names = tuple(sorted(input_names))
        lut = permute_lut(entry["out"], input_names, list(sorted_names))
        result[name] = (sorted_names, tuple(int(v) for v in lut))
    return result


def equivalent_logic(a: Any, b: Any) -> bool:
    """True if two networks / logic dicts have the same name-keyed LUTs."""
    return name_keyed_luts(a) == name_keyed_luts(b)


# ---------------------------------------------------------------------------
# Rules → logic
# ---------------------------------------------------------------------------


def rules_to_logic(
    rules: Mapping[str, str],
    extra_names: Optional[Iterable[str]] = None,
) -> LogicDict:
    """Build a CANA ``logic`` dict from ``{target: expression}``.

    Names that appear only as regulators (free inputs) receive the identity
    update ``f(x) = x``, matching BBM's booleannet representation of inputs.
    """
    # Preserve insertion order of targets; append free inputs alphabetically.
    ordered: List[str] = list(rules.keys())
    seen = set(ordered)
    mentioned: set = set()
    for expr in rules.values():
        mentioned.update(_identifiers_in_expr(normalize_operators(expr, "python")))
    if extra_names:
        mentioned.update(extra_names)
    for name in sorted(mentioned):
        if name not in seen:
            ordered.append(name)
            seen.add(name)

    name_to_id = {name: i for i, name in enumerate(ordered)}
    completed: Dict[str, str] = dict(rules)
    for name in ordered:
        if name not in completed:
            completed[name] = name  # identity

    logic: LogicDict = {}
    for i, name in enumerate(ordered):
        expr = normalize_operators(completed[name], "python")
        # True/False constants from LUT export of k==0 — treat as const outputs
        if expr in ("True", "true", "1"):
            logic[i] = {"name": name, "in": [], "out": [1]}
            continue
        if expr in ("False", "false", "0"):
            logic[i] = {"name": name, "in": [], "out": [0]}
            continue
        input_names = _inputs_for_expr(expr, ordered)
        # Identity with self-loop: ensure self is an input even if regex misses
        if expr.strip() == name and name not in input_names:
            input_names = [name]
        input_ids = [name_to_id[n] for n in input_names]
        out = output_transitions(expr, input_names)
        logic[i] = {"name": name, "in": input_ids, "out": [int(v) for v in out]}
    return logic


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def parse_booleannet(text: str) -> LogicDict:
    """Parse BooleanNet / ``*.booleannet.txt`` rules into a ``logic`` dict."""
    rules: Dict[str, str] = {}
    extras: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "*" not in line or "=" not in line:
            continue
        name = line.split("*", 1)[0].strip()
        expr = line.split("=", 1)[1].strip()
        if not name:
            continue
        rules[name] = expr
        extras.extend(_identifiers_in_expr(normalize_operators(expr, "python")))
    if not rules:
        raise ValueError("No booleannet rules found")
    return rules_to_logic(rules, extras)


def parse_bnet(text: str) -> LogicDict:
    """Parse BoolNet / PyBoolNet ``*.bnet`` (``targets,factors``) into ``logic``."""
    rules: Dict[str, str] = {}
    extras: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("targets") and "," in line:
            continue
        if "," not in line:
            continue
        name, expr = line.split(",", 1)
        name = name.strip()
        expr = expr.strip()
        if not name:
            continue
        rules[name] = expr
        extras.extend(_identifiers_in_expr(normalize_operators(expr, "python")))
    if not rules:
        raise ValueError("No bnet rules found")
    return rules_to_logic(rules, extras)


def parse_aeon(text: str) -> LogicDict:
    """Parse AEON ``*.aeon`` (regulations + ``$var: expr``) into ``logic``."""
    rules: Dict[str, str] = {}
    names: set = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m_upd = _AEON_UPDATE_RE.match(line)
        if m_upd:
            name, expr = m_upd.group(1), m_upd.group(2)
            rules[name] = expr
            names.add(name)
            names.update(_identifiers_in_expr(normalize_operators(expr, "python")))
            continue
        m_edge = _AEON_EDGE_RE.match(line)
        if m_edge:
            names.add(m_edge.group(1))
            names.add(m_edge.group(3))
            continue
        # Fallback: lines like "A -> B" with flexible arrows already covered;
        # also accept "A -| B" etc. via regex. Ignore unknown lines.
    if not rules and not names:
        raise ValueError("No aeon rules or regulations found")
    return rules_to_logic(rules, names)


# ---- SBML-qual ----

_SBML_NS = {
    "sbml": "http://www.sbml.org/sbml/level3/version1/core",
    "qual": "http://www.sbml.org/sbml/level3/version1/qual/version1",
    "math": "http://www.w3.org/1998/Math/MathML",
}


def _local(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _mathml_to_expr(node: ET.Element) -> str:
    """Convert a MathML boolean-ish subtree to a python boolean expression."""
    tag = _local(node.tag)

    if tag == "math":
        children = list(node)
        if not children:
            return "False"
        return _mathml_to_expr(children[0])

    if tag == "ci":
        return (node.text or "").strip()

    if tag == "cn":
        text = (node.text or "0").strip()
        try:
            return "True" if int(float(text)) != 0 else "False"
        except ValueError:
            return "False"

    if tag == "true":
        return "True"
    if tag == "false":
        return "False"

    if tag != "apply":
        # Unknown wrapper: descend if single child
        children = list(node)
        if len(children) == 1:
            return _mathml_to_expr(children[0])
        raise ValueError("Unsupported MathML element: {0}".format(tag))

    children = list(node)
    if not children:
        return "False"
    op = _local(children[0].tag)
    args = children[1:]

    if op == "eq":
        # typically <eq/><ci>X</ci><cn>1</cn>  → X or not X
        if len(args) != 2:
            raise ValueError("eq expects 2 arguments")
        left, right = _mathml_to_expr(args[0]), _mathml_to_expr(args[1])
        if right in ("True", "1"):
            return left if left not in ("True", "False") else right
        if right in ("False", "0"):
            if left in ("True", "False"):
                return "True" if left == "False" else "False"
            return "(not {0})".format(left)
        if left in ("True", "1"):
            return right
        if left in ("False", "0"):
            return "(not {0})".format(right)
        return "(({0} and {1}) or (not {0} and not {1}))".format(left, right)

    if op == "neq":
        if len(args) != 2:
            raise ValueError("neq expects 2 arguments")
        left, right = _mathml_to_expr(args[0]), _mathml_to_expr(args[1])
        if right in ("True", "1"):
            return "(not {0})".format(left)
        if right in ("False", "0"):
            return left
        return "(({0} and not {1}) or (not {0} and {1}))".format(left, right)

    if op == "not":
        if len(args) != 1:
            raise ValueError("not expects 1 argument")
        return "(not {0})".format(_mathml_to_expr(args[0]))

    if op in ("and", "or"):
        if not args:
            return "True" if op == "and" else "False"
        joined = (" {0} ".format(op)).join(
            "({0})".format(_mathml_to_expr(a)) for a in args
        )
        return "({0})".format(joined) if len(args) > 1 else _mathml_to_expr(args[0])

    if op == "xor":
        if len(args) != 2:
            # fold left
            expr = _mathml_to_expr(args[0])
            for a in args[1:]:
                b = _mathml_to_expr(a)
                expr = "(({0} and not {1}) or (not {0} and {1}))".format(expr, b)
            return expr
        a, b = _mathml_to_expr(args[0]), _mathml_to_expr(args[1])
        return "(({0} and not {1}) or (not {0} and {1}))".format(a, b)

    # Implicit: some files use <apply><ci>var</ci></apply> — treat as var
    if op not in ("plus", "minus", "times", "divide"):
        # First child was not a known operator — might be a bare ci as op? rare.
        pass
    raise ValueError("Unsupported MathML apply operator: {0}".format(op))


def _qual_attr(elem: ET.Element, local_name: str) -> Optional[str]:
    for key, val in elem.attrib.items():
        if _local(key) == local_name:
            return val
    return None


def parse_sbml(text: str) -> LogicDict:
    """Parse SBML-qual Boolean models into a ``logic`` dict."""
    root = ET.fromstring(text)
    species: List[str] = []
    for elem in root.iter():
        if _local(elem.tag) == "qualitativeSpecies":
            sid = _qual_attr(elem, "id") or _qual_attr(elem, "name")
            if sid:
                species.append(sid)

    rules: Dict[str, str] = {}
    for elem in root.iter():
        if _local(elem.tag) != "transition":
            continue
        outputs = [
            c
            for c in elem.iter()
            if _local(c.tag) == "output"
        ]
        target = None
        for out in outputs:
            target = _qual_attr(out, "qualitativeSpecies")
            if target:
                break
        if not target:
            continue

        # Collect functionTerms with resultLevel == 1 (ON condition)
        on_exprs: List[str] = []
        has_terms = False
        for child in elem.iter():
            if _local(child.tag) != "functionTerm":
                continue
            has_terms = True
            level = _qual_attr(child, "resultLevel")
            if level is not None and int(level) != 1:
                continue
            math = None
            for m in child:
                if _local(m.tag) == "math":
                    math = m
                    break
            if math is None:
                for m in child.iter():
                    if _local(m.tag) == "math":
                        math = m
                        break
            if math is not None:
                on_exprs.append(_mathml_to_expr(math))

        if not has_terms and not on_exprs:
            # Free input / empty transition → identity
            continue
        if not on_exprs:
            # Only defaultTerm (OFF) → always false; rare for Boolean BBM
            rules[target] = "False"
        elif len(on_exprs) == 1:
            rules[target] = on_exprs[0]
        else:
            rules[target] = "(" + " or ".join("({0})".format(e) for e in on_exprs) + ")"

    if not species and not rules:
        raise ValueError("No SBML-qual species or transitions found")
    return rules_to_logic(rules, species)


def parse_string(text: str, fmt: str) -> LogicDict:
    """Parse ``text`` of the given format into a ``logic`` dict (not cnet)."""
    fmt = normalize_format(fmt)
    if fmt == "booleannet":
        return parse_booleannet(text)
    if fmt == "bnet":
        return parse_bnet(text)
    if fmt == "aeon":
        return parse_aeon(text)
    if fmt == "sbml":
        return parse_sbml(text)
    raise ValueError(
        "parse_string does not handle format {0!r}; use BooleanNetwork.from_string_cnet".format(
            fmt
        )
    )


# ---------------------------------------------------------------------------
# Exporters
# ---------------------------------------------------------------------------


def _logic_node_order(logic: Mapping[int, Mapping[str, Any]]) -> List[int]:
    return sorted(logic.keys())


def _input_names_for(
    logic: Mapping[int, Mapping[str, Any]], node_id: int
) -> List[str]:
    return [logic[j]["name"] for j in logic[node_id]["in"]]


def logic_to_booleannet(logic: Mapping[int, Mapping[str, Any]]) -> str:
    """Export ``logic`` to BooleanNet text (``name* = expr``)."""
    lines: List[str] = []
    for i in _logic_node_order(logic):
        name = logic[i]["name"]
        inputs = _input_names_for(logic, i)
        expr = lut_to_expression(inputs, logic[i]["out"], style="python")
        lines.append("{0}* = {1}".format(name, expr))
    return "\n".join(lines) + "\n"


def logic_to_bnet(logic: Mapping[int, Mapping[str, Any]]) -> str:
    """Export ``logic`` to BoolNet ``targets,factors`` text."""
    lines = ["targets,factors"]
    for i in _logic_node_order(logic):
        name = logic[i]["name"]
        inputs = _input_names_for(logic, i)
        # Skip pure identity free-input nodes? Keep them for round-trip fidelity.
        expr = lut_to_expression(inputs, logic[i]["out"], style="symbolic")
        lines.append("{0}, {1}".format(name, expr))
    return "\n".join(lines) + "\n"


def logic_to_aeon(logic: Mapping[int, Mapping[str, Any]]) -> str:
    """Export ``logic`` to AEON text (regulations + update functions)."""
    lines: List[str] = []
    for i in _logic_node_order(logic):
        target = logic[i]["name"]
        for src_id in logic[i]["in"]:
            src = logic[src_id]["name"]
            # Unknown sign — AEON allows -?
            lines.append("{0} -? {1}".format(src, target))
    for i in _logic_node_order(logic):
        name = logic[i]["name"]
        inputs = _input_names_for(logic, i)
        expr = lut_to_expression(inputs, logic[i]["out"], style="symbolic")
        lines.append("${0}: {1}".format(name, expr))
    return "\n".join(lines) + "\n"


def _expr_to_mathml(expr: str) -> ET.Element:
    """Build MathML for a python boolean expression via its LUT expansion path.

    Instead of parsing arbitrary expr AST, callers should prefer LUT→MathML.
    This helper supports the small DNF language produced by :func:`lut_to_expression`.
    """
    # Prefer building from tokens is fragile; use a simple recursive descent.
    return _parse_python_bool_to_mathml(expr)


def _parse_python_bool_to_mathml(expr: str) -> ET.Element:
    """Parse a restricted python boolean expression into MathML ``apply`` trees."""

    def math_ci(name: str) -> ET.Element:
        el = ET.Element("ci")
        el.text = name
        return el

    def math_cn_one() -> ET.Element:
        el = ET.Element("cn")
        el.set("type", "integer")
        el.text = "1"
        return el

    def eq_level1(name: str) -> ET.Element:
        apply = ET.Element("apply")
        ET.SubElement(apply, "eq")
        apply.append(math_ci(name))
        apply.append(math_cn_one())
        return apply

    def wrap_not(inner: ET.Element) -> ET.Element:
        apply = ET.Element("apply")
        ET.SubElement(apply, "not")
        apply.append(inner)
        return apply

    def wrap_bin(op: str, parts: List[ET.Element]) -> ET.Element:
        if len(parts) == 1:
            return parts[0]
        apply = ET.Element("apply")
        ET.SubElement(apply, op)
        for p in parts:
            apply.append(p)
        return apply

    s = expr.strip()
    if s in ("True", "true", "1"):
        return ET.Element("true")
    if s in ("False", "false", "0"):
        return ET.Element("false")

    # Tokenize roughly: identifiers, not, and, or, parentheses
    tokens: List[str] = re.findall(
        r"[A-Za-z_][A-Za-z0-9_]*|\(|\)|and|or|not|&&|\|\||!|&|\|", s
    )
    # Normalize token ops
    normed: List[str] = []
    for t in tokens:
        if t in ("&&", "&"):
            normed.append("and")
        elif t in ("||", "|"):
            normed.append("or")
        elif t == "!":
            normed.append("not")
        else:
            normed.append(t)
    tokens = normed
    pos = 0

    def peek() -> Optional[str]:
        return tokens[pos] if pos < len(tokens) else None

    def eat(expected: Optional[str] = None) -> str:
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("Unexpected end of expression: {0}".format(expr))
        tok = tokens[pos]
        if expected is not None and tok != expected:
            raise ValueError("Expected {0!r}, got {1!r} in {2}".format(expected, tok, expr))
        pos += 1
        return tok

    def parse_expr() -> ET.Element:
        return parse_or()

    def parse_or() -> ET.Element:
        parts = [parse_and()]
        while peek() == "or":
            eat("or")
            parts.append(parse_and())
        return wrap_bin("or", parts)

    def parse_and() -> ET.Element:
        parts = [parse_not()]
        while peek() == "and":
            eat("and")
            parts.append(parse_not())
        return wrap_bin("and", parts)

    def parse_not() -> ET.Element:
        if peek() == "not":
            eat("not")
            return wrap_not(parse_not())
        return parse_atom()

    def parse_atom() -> ET.Element:
        tok = peek()
        if tok == "(":
            eat("(")
            node = parse_expr()
            eat(")")
            return node
        if tok is None:
            raise ValueError("Unexpected end in {0}".format(expr))
        name = eat()
        if name in ("and", "or", "not", "True", "False"):
            if name == "True":
                return ET.Element("true")
            if name == "False":
                return ET.Element("false")
            raise ValueError("Unexpected keyword {0}".format(name))
        return eq_level1(name)

    tree = parse_expr()
    if pos != len(tokens):
        raise ValueError("Trailing tokens in expression: {0}".format(expr))
    return tree


def _lut_to_mathml(input_names: Sequence[str], out: Sequence[Any]) -> ET.Element:
    """MathML for the ON condition of a Boolean LUT (resultLevel 1)."""
    expr = lut_to_expression(input_names, out, style="python")
    return _parse_python_bool_to_mathml(expr)


def logic_to_sbml(logic: Mapping[int, Mapping[str, Any]]) -> str:
    """Export ``logic`` to SBML Level 3 qualitative XML."""
    SBML = "http://www.sbml.org/sbml/level3/version1/core"
    QUAL = "http://www.sbml.org/sbml/level3/version1/qual/version1"
    MATH = "http://www.w3.org/1998/Math/MathML"

    sbml = ET.Element(
        "{%s}sbml" % SBML,
        {
            "level": "3",
            "version": "1",
            "{%s}required" % QUAL: "true",
        },
    )
    sbml.set("xmlns", SBML)
    sbml.set("xmlns:qual", QUAL)
    model = ET.SubElement(sbml, "{%s}model" % SBML)
    list_sp = ET.SubElement(model, "{%s}listOfQualitativeSpecies" % QUAL)
    for i in _logic_node_order(logic):
        name = logic[i]["name"]
        ET.SubElement(
            list_sp,
            "{%s}qualitativeSpecies" % QUAL,
            {
                "{%s}maxLevel" % QUAL: "1",
                "{%s}constant" % QUAL: "false",
                "{%s}name" % QUAL: name,
                "{%s}id" % QUAL: name,
            },
        )

    list_tr = ET.SubElement(model, "{%s}listOfTransitions" % QUAL)
    for i in _logic_node_order(logic):
        name = logic[i]["name"]
        inputs = _input_names_for(logic, i)
        tr = ET.SubElement(
            list_tr,
            "{%s}transition" % QUAL,
            {"{%s}id" % QUAL: "tr_{0}".format(name)},
        )
        loi = ET.SubElement(tr, "{%s}listOfInputs" % QUAL)
        for src in inputs:
            ET.SubElement(
                loi,
                "{%s}input" % QUAL,
                {
                    "{%s}qualitativeSpecies" % QUAL: src,
                    "{%s}transitionEffect" % QUAL: "none",
                    "{%s}sign" % QUAL: "unknown",
                    "{%s}id" % QUAL: "tr_{0}_in_{1}".format(name, src),
                },
            )
        loo = ET.SubElement(tr, "{%s}listOfOutputs" % QUAL)
        ET.SubElement(
            loo,
            "{%s}output" % QUAL,
            {
                "{%s}qualitativeSpecies" % QUAL: name,
                "{%s}transitionEffect" % QUAL: "assignmentLevel",
                "{%s}id" % QUAL: "tr_{0}_out".format(name),
            },
        )
        loft = ET.SubElement(tr, "{%s}listOfFunctionTerms" % QUAL)
        ET.SubElement(
            loft,
            "{%s}defaultTerm" % QUAL,
            {"{%s}resultLevel" % QUAL: "0"},
        )
        # Identity free input still has a proper LUT [0,1]
        ft = ET.SubElement(
            loft,
            "{%s}functionTerm" % QUAL,
            {"{%s}resultLevel" % QUAL: "1"},
        )
        math = ET.SubElement(ft, "{%s}math" % MATH)
        math.set("xmlns", MATH)
        math.append(_lut_to_mathml(inputs, logic[i]["out"]))

    # ElementTree does not always emit xmlns the way we want; serialize carefully.
    rough = ET.tostring(sbml, encoding="unicode")
    # Ensure math xmlns present
    if "xmlns:qual=" not in rough:
        rough = rough.replace(
            "<sbml ",
            '<sbml xmlns="{0}" xmlns:qual="{1}" '.format(SBML, QUAL),
            1,
        )
    return '<?xml version="1.0" encoding="UTF-8"?>' + rough


def export_string(logic: Mapping[int, Mapping[str, Any]], fmt: str) -> str:
    """Serialize ``logic`` to the requested format string (not cnet)."""
    fmt = normalize_format(fmt)
    if fmt == "booleannet":
        return logic_to_booleannet(logic)
    if fmt == "bnet":
        return logic_to_bnet(logic)
    if fmt == "aeon":
        return logic_to_aeon(logic)
    if fmt == "sbml":
        return logic_to_sbml(logic)
    raise ValueError(
        "export_string does not handle format {0!r}; use BooleanNetwork.to_cnet".format(
            fmt
        )
    )


def convert(
    text: str,
    from_type: str,
    to_type: str,
    *,
    cnet_parser=None,
    cnet_exporter=None,
) -> str:
    """Convert a model string from ``from_type`` to ``to_type`` via ``logic``.

    For ``cnet``, pass ``cnet_parser(text) -> logic`` and
    ``cnet_exporter(logic) -> str`` callables (typically bound
    :class:`~cana.boolean_network.BooleanNetwork` helpers).
    """
    src = normalize_format(from_type)
    dst = normalize_format(to_type)
    if src == "cnet":
        if cnet_parser is None:
            raise ValueError("cnet_parser required for cnet input")
        logic = cnet_parser(text)
    else:
        logic = parse_string(text, src)

    if dst == "cnet":
        if cnet_exporter is None:
            raise ValueError("cnet_exporter required for cnet output")
        return cnet_exporter(logic)
    return export_string(logic, dst)
