# -*- coding: utf-8 -*-
"""
Biological Boolean Networks
=================================

Some of the commonly used biological boolean networks


"""
#   Copyright (C) 2017 by
#   Alex Gates <ajgates@gmail.com>
#   Rion Brattig Correia <rionbr@gmail.com>
#   All rights reserved.
#   MIT license.
import os
import sys
from .. boolean_network import BooleanNetwork


_path = os.path.dirname(os.path.realpath(__file__))
""" Make sure we know what the current directory is """


def THALIANA():
	"""Boolean network model of the control of flower morphogenesis in Arabidobis thaliana 

	The network is defined in :cite:`Chaos:2006`.

	Returns:
		(BooleanNetwork)
	"""
	return BooleanNetwork.from_file(_path + '/thaliana.txt')

def DROSOPHILA():
	"""Drosophila Melanogaster single parameter segment

	The network is defined in :cite:`Albert:2008`.

	"""
	return BooleanNetwork.from_file(_path + '/drosophila_single_parasegment.txt')

def BUDDING_YEAST():
	""" 

	The network is defined in :cite:`Fangting:2004`.
		
	"""
	return BooleanNetwork.from_file(_path + '/yeast_cell_cycle.txt')

def MARQUESPITA():
	"""Boolean network used for the Two-Symbol schemata example.

	The network is defined in :cite:`Marques-Pita:2013`.

	"""
	return BooleanNetwork.from_file(_path + '/marques-pita_rocha.txt', keep_constants=True)


