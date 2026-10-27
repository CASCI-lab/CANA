# -*- coding: utf-8 -*-
"""SBML-qual Boolean model import and export (MathML update functions)."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Mapping, Optional, Sequence

from cana.function_outputs.expression import lut_to_expression
from cana.network_dict import (
    NetworkDict,
    expressions_to_network_dict,
    input_names_for_node,
    sorted_node_ids,
)


def xml_local_name(tag: str) -> str:
    """Strip an XML namespace URI, leaving the local element name."""
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def qual_attribute(elem: ET.Element, local_name: str) -> Optional[str]:
    """Read a qual/SBML attribute by local name, ignoring namespace prefixes."""
    for key, val in elem.attrib.items():
        if xml_local_name(key) == local_name:
            return val
    return None


def mathml_to_python_expr(node: ET.Element) -> str:
    """Convert a MathML boolean-ish subtree to a python boolean expression."""
    tag = xml_local_name(node.tag)

    if tag == "math":
        children = list(node)
        if not children:
            return "False"
        return mathml_to_python_expr(children[0])

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
            return mathml_to_python_expr(children[0])
        raise ValueError("Unsupported MathML element: {0}".format(tag))

    children = list(node)
    if not children:
        return "False"
    op = xml_local_name(children[0].tag)
    args = children[1:]

    if op == "eq":
        # typically <eq/><ci>X</ci><cn>1</cn>  → X or not X
        if len(args) != 2:
            raise ValueError("eq expects 2 arguments")
        left, right = mathml_to_python_expr(args[0]), mathml_to_python_expr(args[1])
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
        left, right = mathml_to_python_expr(args[0]), mathml_to_python_expr(args[1])
        if right in ("True", "1"):
            return "(not {0})".format(left)
        if right in ("False", "0"):
            return left
        return "(({0} and not {1}) or (not {0} and {1}))".format(left, right)

    if op == "not":
        if len(args) != 1:
            raise ValueError("not expects 1 argument")
        return "(not {0})".format(mathml_to_python_expr(args[0]))

    if op in ("and", "or"):
        if not args:
            return "True" if op == "and" else "False"
        joined = (" {0} ".format(op)).join(
            "({0})".format(mathml_to_python_expr(a)) for a in args
        )
        return "({0})".format(joined) if len(args) > 1 else mathml_to_python_expr(args[0])

    if op == "xor":
        if len(args) != 2:
            # fold left
            expr = mathml_to_python_expr(args[0])
            for a in args[1:]:
                b = mathml_to_python_expr(a)
                expr = "(({0} and not {1}) or (not {0} and {1}))".format(expr, b)
            return expr
        a, b = mathml_to_python_expr(args[0]), mathml_to_python_expr(args[1])
        return "(({0} and not {1}) or (not {0} and {1}))".format(a, b)

    raise ValueError("Unsupported MathML apply operator: {0}".format(op))


def python_expr_to_mathml(expr: str) -> ET.Element:
    """Parse a restricted python boolean expression into MathML ``apply`` trees.

    Supports the small DNF language produced by :func:`~cana.function_outputs.expression.lut_to_expression`.
    """

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

    tokens: List[str] = re.findall(
        r"[A-Za-z_][A-Za-z0-9_]*|\(|\)|and|or|not|&&|\|\||!|&|\|", s
    )
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
            raise ValueError(
                "Expected {0!r}, got {1!r} in {2}".format(expected, tok, expr)
            )
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


def lut_to_mathml(input_names: Sequence[str], lut: Sequence[Any]) -> ET.Element:
    """MathML for the ON condition of a Boolean LUT (resultLevel 1)."""
    expr = lut_to_expression(input_names, lut, style="python")
    return python_expr_to_mathml(expr)


def parse_sbml(text: str) -> NetworkDict:
    """Parse SBML-qual Boolean models into a ``NetworkDict``."""
    root = ET.fromstring(text)
    species: List[str] = []
    for elem in root.iter():
        if xml_local_name(elem.tag) == "qualitativeSpecies":
            sid = qual_attribute(elem, "id") or qual_attribute(elem, "name")
            if sid:
                species.append(sid)

    expressions: Dict[str, str] = {}
    for elem in root.iter():
        if xml_local_name(elem.tag) != "transition":
            continue
        outputs = [
            c
            for c in elem.iter()
            if xml_local_name(c.tag) == "output"
        ]
        target = None
        for out in outputs:
            target = qual_attribute(out, "qualitativeSpecies")
            if target:
                break
        if not target:
            continue

        on_exprs: List[str] = []
        has_terms = False
        for child in elem.iter():
            if xml_local_name(child.tag) != "functionTerm":
                continue
            has_terms = True
            level = qual_attribute(child, "resultLevel")
            if level is not None and int(level) != 1:
                continue
            math = None
            for m in child:
                if xml_local_name(m.tag) == "math":
                    math = m
                    break
            if math is None:
                for m in child.iter():
                    if xml_local_name(m.tag) == "math":
                        math = m
                        break
            if math is not None:
                on_exprs.append(mathml_to_python_expr(math))

        if not has_terms and not on_exprs:
            continue
        if not on_exprs:
            expressions[target] = "False"
        elif len(on_exprs) == 1:
            expressions[target] = on_exprs[0]
        else:
            expressions[target] = (
                "(" + " or ".join("({0})".format(e) for e in on_exprs) + ")"
            )

    if not species and not expressions:
        raise ValueError("No SBML-qual species or transitions found")
    return expressions_to_network_dict(expressions, species)


def network_dict_to_sbml(network_dict: Mapping[int, Mapping[str, Any]]) -> str:
    """Export ``NetworkDict`` to SBML Level 3 qualitative XML."""
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
    for i in sorted_node_ids(network_dict):
        name = network_dict[i]["name"]
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
    for i in sorted_node_ids(network_dict):
        name = network_dict[i]["name"]
        inputs = input_names_for_node(network_dict, i)
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
        ft = ET.SubElement(
            loft,
            "{%s}functionTerm" % QUAL,
            {"{%s}resultLevel" % QUAL: "1"},
        )
        math = ET.SubElement(ft, "{%s}math" % MATH)
        math.set("xmlns", MATH)
        math.append(
            lut_to_mathml(inputs, network_dict[i]["outputs"].get_bit_lut())
        )

    rough = ET.tostring(sbml, encoding="unicode")
    if "xmlns:qual=" not in rough:
        rough = rough.replace(
            "<sbml ",
            '<sbml xmlns="{0}" xmlns:qual="{1}" '.format(SBML, QUAL),
            1,
        )
    return '<?xml version="1.0" encoding="UTF-8"?>' + rough
