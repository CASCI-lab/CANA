# -*- coding: utf-8 -*-
#
# Regression values for five published density-classification (DCT) rules.
# The values are what CANA computes on upstream
# master at the time this file was written; any change in the canalization
# code that moves them is a behaviour change and must be explained.
#
import pytest

from cana.boolean_node import BooleanNode

# LUT convention: 128 characters, lexicographic, so character i is the output
# for the neighbourhood whose bits are the 7-digit binary expansion of i, read
# left to right as (l3, l2, l1, c, r1, r2, r3). This is CANA's default state
# ordering, with input 1 the most significant bit.
DCT_RULES = {
    # Hand-designed. Gacs, Kurdyumov & Levin (1978), Problemy Peredachi
    # Informatsii 14(3):92-98. Table as printed in Andre, Bennett & Koza
    # (1996), Genetic Programming 1996, Table 2.
    "GKL": (
        "0000000001011111000000000101111100000000010111110000000001011111"
        "0000000001011111111111110101111100000000010111111111111101011111",
        {"kr_mean": 4.65625, "kr_max": 4.75, "kr_min": 4.5625,
         "ks_mean": 1.6875, "ks_max": 1.875, "sensitivity": 1.625},
    ),
    # Genetic algorithm. Das, Mitchell & Crutchfield (1994), "A genetic
    # algorithm discovers particle-based computation in cellular automata",
    # PPSN III, LNCS 866. Table as printed in Juille & Pollack (1998),
    # Genetic Programming 1998, Table 2.
    "Das": (
        "0000011100000000000001111111111100001111000000000000111111111111"
        "0000111100000000000001111111111100001111001100010000111111111111",
        {"kr_mean": 4.138504464285714, "kr_max": 4.390625, "kr_min": 3.890625,
         "ks_mean": 1.5338541666666667, "ks_max": 2.0625, "sensitivity": 1.78125},
    ),
    # Genetic programming. Andre, Bennett & Koza (1996), "Discovery by genetic
    # programming of a cellular automata rule that is better than any known
    # rule for the majority classification problem", Genetic Programming 1996,
    # Table 2. Also tabulated as the "ABK" rule in Juille & Pollack (1998).
    "GP": (
        "0000010100000000010101010000010100000101000000000101010100000101"
        "0101010111111111010101011111111101010101111111110101010111111111",
        {"kr_mean": 4.619791666666667, "kr_max": 4.75, "kr_min": 4.5,
         "ks_mean": 1.7083333333333333, "ks_max": 1.875, "sensitivity": 1.625},
    ),
    # Coevolution. Juille & Pollack (1998), "Coevolving the 'ideal' trainer:
    # application to the discovery of cellular automata rules", Genetic
    # Programming 1998, Table 2, row "Coevolution (2)".
    "COE_2": (
        "0001010001010001001100000101110000000000010100001100111001011111"
        "0001011100010001111111110101111100001111010100111100111101011111",
        {"kr_mean": 2.96691158234127, "kr_max": 3.2578125, "kr_min": 2.734375,
         "ks_mean": 1.4994915674603175, "ks_max": 2.2109375, "sensitivity": 2.34375},
    ),
    # Gene expression programming. Ferreira (2001), "Gene expression
    # programming: a new adaptive algorithm for solving problems", Complex
    # Systems 13(2):87-129, Table 5, second rule.
    "GEP_2": (
        "0000000001010101000000000111011100000000010101010000000001110111"
        "0000111101010101000011110111011111111111010101011111111101110111",
        {"kr_mean": 4.268229166666667, "kr_max": 4.5, "kr_min": 4.046875,
         "ks_mean": 0.984375, "ks_max": 1.21875, "sensitivity": 1.78125},
    ),
}

# All values are compared exactly. The means over schemata are summed with
# math.fsum, so they are correctly rounded and do not depend on the order in
# which the prime implicants are visited.


@pytest.fixture(params=sorted(DCT_RULES), ids=sorted(DCT_RULES))
def dct_rule(request):
    lut, reference = DCT_RULES[request.param]
    assert len(lut) == 128 and set(lut) <= {"0", "1"}
    return BooleanNode.from_output_list([int(c) for c in lut], name=request.param), reference


def test_dct_input_redundancy(dct_rule):
    node, ref = dct_rule
    assert node.input_redundancy(norm=False) == ref["kr_mean"]
    assert node.input_redundancy(operator=max, norm=False) == ref["kr_max"]
    assert node.input_redundancy(operator=min, norm=False) == ref["kr_min"]


def test_dct_effective_connectivity(dct_rule):
    node, ref = dct_rule
    assert node.effective_connectivity(norm=False) == 7 - ref["kr_mean"]


def test_dct_input_symmetry(dct_rule):
    node, ref = dct_rule
    assert node.input_symmetry_mean() == ref["ks_mean"]
    assert node.input_symmetry(aggOp="mean") == ref["ks_mean"]
    assert node.input_symmetry(aggOp="max") == ref["ks_max"]


def test_dct_sensitivity(dct_rule):
    node, ref = dct_rule
    assert node.sensitivity(norm=False) == ref["sensitivity"]


def test_dct_effective_connectivity_bounds_sensitivity(dct_rule):
    # Gates, Brattig Correia, Wang & Rocha (2021), PNAS 118(12): k_e >= s.
    node, _ = dct_rule
    assert node.effective_connectivity(norm=False) >= node.sensitivity(norm=False)
