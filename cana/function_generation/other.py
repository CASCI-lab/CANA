# -*- coding: utf-8 -*-
"""General / non-structured Boolean function generators.

These typically return dense :class:`~cana.function_outputs.bit_lut.BitLut`
tables rather than expression strings.
"""
from __future__ import annotations

import random

from cana.function_outputs.bit_lut import BitLut

__all__ = [
    "random_automata_table",
]


def random_automata_table(indegree, bias, bias_constraint="soft"):
    """Return a random Boolean lookup table as a :class:`BitLut`.

    Parameters
    ----------
    indegree : int
        Number of inputs (LUT has ``2**indegree`` rows).
    bias : float
        Probability of a 1 under ``soft`` / ``soft_no_constant``, or the
        target fraction of ones under ``hard``.
    bias_constraint : {"soft", "hard", "soft_no_constant"}
        Sampling regime for the output column.
    """
    if bias_constraint == "soft":
        output = [int(random.random() < bias) for b in range(2**indegree)]

    elif bias_constraint == "hard":
        n_ones = int(bias * (2**indegree))
        output = [0] * (2**indegree - n_ones) + [1] * n_ones
        random.shuffle(output)

    elif bias_constraint == "soft_no_constant":
        output = [int(random.random() < bias) for b in range(2**indegree)]
        if sum(output) == 0:
            output[0] = 1
            random.shuffle(output)
    else:
        raise ValueError("Unknown bias_constraint: {0!r}".format(bias_constraint))

    return BitLut.from_sequence(output)
