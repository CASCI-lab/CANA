# -*- coding: utf-8 -*-
"""
BNS Attractors Interface
==========================

This module interfaces CANA with the [B]oolean [N]etworks with [S]ynchronous update (BNS) :cite:`Dubrova:2011` to compute attractors.

    BNS is a software tool for computing attractors in Boolean Networks with Synchronous update.
    Synchronous Boolean networks are used for the modeling of genetic regulatory networks.

    BNS implements the algorithm presented in which is based on a SAT-based bounded model checking.
    BNS uses much less space compared to BooleNet or other BDD-based approaches for computing attractors.
    It can handle several orders of magnitude larger networks.



.. Note::

    You must have ``bns`` compiled for you system. Alternatively, you can download the binary from the `bns website <https://people.kth.se/~dubrova/bns.html>`_ directly.
    Last updated: March 2017.

"""
#   Copyright (C) 2017 by
#   Rion Brattig Correia <rionbr@gmail.com>
#   Alex Gates <ajgates@indiana.edu>
#   All rights reserved.
#   MIT license.
import os
import subprocess
import tempfile
from cana.utils import binstate_to_statenum

_path = os.path.dirname(os.path.realpath(__file__))
""" Make sure we know what the current directory is """


def attractors(cnet, bnspath=_path, cleanup=True):
    """Makes a subprocess call to `bns` supplying a temporary file with the boolean logic.

    Args:
        cnet (file,string) : A .cnet formated string or file.
        bnspath (string) : The path to the bns binary.
        cleanup (bool) : If cnet is a string, this function creates a temporary file. This forces the removal of this temp file.

    Returns:
        list : the list of attractors
    """

    # If is file, open the file
    if os.path.isfile(cnet):
        file = cnet

    # If string, Creates a Temporary File to be supplied to BNS
    elif isinstance(cnet, str):
        tmp = tempfile.NamedTemporaryFile(delete=cleanup)
        with open(tmp.name, 'w') as openfile:
            openfile.write(cnet)
        tmp.file.close()
        file = tmp.name
    else:
        raise TypeError('The cnet input should be either a file to a .cnet file or a string containing the .cnet content')

    cmd = [os.path.join(_path, 'bns'), file]
    attractors = list()

    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        current_attractor = []
        for i, line in enumerate(p.stdout):
            # Strip line
            cleanline = line.decode('ascii').strip().replace('\n', '')

            if 'Attractor' in cleanline:
                attractors.append(current_attractor)
                current_attractor = []
            elif 'Node' in cleanline and 'assumed to be constant' in cleanline:
                pass
            elif 'Total' in cleanline:
                pass
            elif 'Start searching for all atractors.' in cleanline:
                pass
            elif 'Depth' in cleanline:
                pass
            elif 'average' in cleanline:
                pass
            elif len(cleanline) > 0:
                current_attractor.append(binstate_to_statenum(cleanline))

    except OSError:
        print("'BNS' could not be found! You must have it compiled or download the binary for your system from the 'bns' website (https://people.kth.se/~dubrova/bns.html).")

    return attractors
