__package__ = 'cana'
__title__ = u'CANAlization: Control & Redundancy in Boolean Networks'
__description__ = u'This package implements a series of methods used to study control, canalization and redundancy in Boolean Networks.'

__author__ = """\n""".join([
	'Alex Gates <ajgates@umail.iu.edu>',
	'Rion Brattig Correia <rionbr@gmail.com>',
	'Etienne Nzabarushimana <enzabaru@indiana.edu>',
	'Luis M. Rocha <rocha@indiana.edu>'
])

__copyright__ = u'2017, Gates, A., Correia, R. B., Rocha, L. M.'

__version__ = '0.0.2'
__release__ = '0.0.2-alpha'
#
__all__ = ['boolean_network','boolean_node']

#
from boolean_network import BooleanNetwork
from boolean_node import BooleanNode
from utils import *
#