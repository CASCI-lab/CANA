# -*- coding: utf-8 -*-
"""
Boolean Nodes
=================================

Commonly used boolean node functions.

"""
#   Copyright (C) 2021 by
#   Alex Gates <ajgates42@gmail.com>
#   Rion Brattig Correia <rionbr@gmail.com>
#   All rights reserved.
#   MIT license.
from .. boolean_network import BooleanNode


def AND():
    """AND boolean node.

    .. code::

        00 : 0
        01 : 0
        10 : 0
        11 : 1

    """
    return BooleanNode.from_output_list(outputs=[0, 0, 0, 1], name="AND")


def OR():
    """OR boolean node.

    .. code::

        00 : 0
        01 : 1
        10 : 1
        11 : 1

    """
    return BooleanNode.from_output_list(outputs=[0, 1, 1, 1], name="OR")


def XOR():
    """XOR boolean node.

    .. code::

        00 : 0
        01 : 1
        10 : 1
        11 : 0

    """
    return BooleanNode.from_output_list(outputs=[0, 1, 1, 0], name="XOR")


def COPYx1():
    """COPY :math:`x_1` boolean node.

    .. code::

        00 : 0
        01 : 0
        10 : 1
        11 : 1

    """
    return BooleanNode.from_output_list(outputs=[0, 0, 1, 1], name="COPY x_1")


def CONTRADICTION():
    """Contradiction boolean node.

    .. code::

        00 : 0
        01 : 0
        10 : 0
        11 : 0

    """
    return BooleanNode.from_output_list(outputs=[0, 0, 0, 0], name="CONTRADICTION")


def RULE90():
    """RULE 90 celular automata node.

    .. code::

        000 : 0
        001 : 1
        010 : 0
        011 : 1
        100 : 1
        101 : 0
        110 : 1
        111 : 0

    """
    return BooleanNode.from_output_list(outputs=[0, 1, 0, 1, 1, 0, 1, 0], name="RULE 90")


def RULE110():
    """RULE 110 celular automata node.

    .. code::

        000 : 0
        001 : 1
        010 : 1
        011 : 1
        100 : 0
        101 : 1
        110 : 1
        111 : 0

    """
    return BooleanNode.from_output_list(outputs=[0, 1, 1, 1, 0, 1, 1, 0], name="RULE 110")
