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
	"""Boolean network model of the control of flower morphogenesis in Arabidopsis thaliana 

	The network is defined in :cite:`Chaos:2006`.

	Returns:
		(BooleanNetwork)
	"""
	return BooleanNetwork.from_file(_path + '/thaliana.txt', name="Arabidopsis Thaliana", keep_constants=True)

def DROSOPHILA(cells=1):
	"""Drosophila Melanogaster boolean model.
	This is a simplification of the original network defined in :cite:`Albert:2008`.
	In the original model, some nodes receive inputs from neighboring cells.
	In this single cell network, they are condensed (nhhnHH) and treated as constants.

	There is currently only one model available, where the original neighboring cell signals are treated as constants.
	
	Args:
		cells (int) : Which model to return.

	"""
	if cells == 1:
		return BooleanNetwork.from_file(_path + '/drosophila_single_cell.txt', name="Drosophila Melanogaster", keep_constants=True)
	else:
		raise AttributeException('Only single (1) cell drosophila boolean model currently available.')

def BUDDING_YEAST():
	""" 

	The network is defined in :cite:`Fangting:2004`.
		
	"""
	return BooleanNetwork.from_file(_path + '/yeast_cell_cycle.txt', name="Budding Yeast Cell Cycle", keep_constants=True)

def MARQUESPITA():
	"""Boolean network used for the Two-Symbol schemata example.

	The network is defined in :cite:`Marques-Pita:2013`.

	"""
	return BooleanNetwork.from_file(_path + '/marques-pita_rocha.txt', name="Marques-Pita & Rocha", keep_constants=True)


