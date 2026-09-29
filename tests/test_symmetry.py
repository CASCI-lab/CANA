# -*- coding: utf-8 -*-
#
# Tests for ``symmetry.py``
#
from cana.datasets.bools import AND, OR, XOR, COPYx1
from cana.utils import isclose
from cana.boolean_node import BooleanNode


# Three famous rules.
FAMOUS_DCT_RULES = {
    "GKL": (
        "0000000001011111000000000101111100000000010111110000000001011111"
        "0000000001011111111111110101111100000000010111111111111101011111"
    ), # G ́acs, P., Kurdyumov, G. L., and Levin, L. A. (1978). Onedimensional uniform arrays that wash out finite islands. Problemy Peredachi Informatsii, 14(3):92–96.
    "GP": (
        "0000010100000000010101010000010100000101000000000101010100000101"
        "0101010111111111010101011111111101010101111111110101010111111111"
    ), # Andre, D., Bennett III, F. H., and Koza, J. R. (1996). Discovery by genetic programming of a cellular automata rule that is better than any known rule for the majority classification problem. Genetic programming, 96:3–11.
    "COMP1": (
        "0000000000000001000100110000000100010011010111110001001101011111"
        "0001001100000001111111110101111100010011010111111111111101011111"
    ), # Kari, J., & Le Gloannec, B. (2012). Modified Traffic Cellular Automaton for the Density Classification Task. Fundamenta Informaticae, 116(1–4), 141–156. https://doi.org/10.3233/FI-2012-675


}

# Expected values computed from legacy implementations and verified manually.
EXPECTED_SYMMETRY_VALUES = {
    "GKL": {"raw": 0.6642857142857145, "distinct": 0.6371323529411765},
    "GP": {"raw": 0.6363095238095241, "distinct": 0.6079044117647053},
    "COMP1": {"raw": 0.7238095238095235, "distinct": 0.6994485294117648},
}


def test_new_symmetry_matches_expected_values():
    """Check exact expected outputs for the new symmetry implementations."""
    for rule_name in ["GKL", "GP", "COMP1"]:
        node = BooleanNode.from_output_list(FAMOUS_DCT_RULES[rule_name])
        raw_value = node.raw_symmetry()
        distinct_value = node.distinct_symmetry()

        assert isclose(raw_value, EXPECTED_SYMMETRY_VALUES[rule_name]["raw"]), (
            "Raw symmetry mismatch for %s: %s != %s"
            % (rule_name, raw_value, EXPECTED_SYMMETRY_VALUES[rule_name]["raw"])
        )
        assert isclose(
            distinct_value, EXPECTED_SYMMETRY_VALUES[rule_name]["distinct"]
        ), (
            "Distinct symmetry mismatch for %s: %s != %s"
            % (
                rule_name,
                distinct_value,
                EXPECTED_SYMMETRY_VALUES[rule_name]["distinct"],
            )
        )


def test_symmetry_values_are_bounded():
    """Symmetry metrics are ratios and must stay inside [0, 1]."""
    for rule_name in ["GKL", "GP", "COMP1"]:
        node = BooleanNode.from_output_list(FAMOUS_DCT_RULES[rule_name])
        raw_value = node.raw_symmetry()
        distinct_value = node.distinct_symmetry()

        assert 0.0 <= raw_value <= 1.0
        assert 0.0 <= distinct_value <= 1.0


def test_smallest_k_behavior():
    """Verify behavior for the smallest non-trivial LUT (k=1)."""
    node = BooleanNode.from_output_list("01")

    # For k=1, each Hamming-weight bucket has one row, so raw symmetry is 1.
    assert isclose(node.raw_symmetry(), 1.0)

    # Distinct metric excludes identity permutation; with only one bit there are
    # no non-identity permutations, so each row contributes 0.
    assert isclose(node.distinct_symmetry(), 0.0)


def test_distinct_symmetry_classic_gates():
    """Check distinct symmetry values for AND/OR/XOR/COPYx1."""
    gate_factories = {
        "AND": (AND, 0.5),
        "OR": (OR, 0.5),
        "XOR": (XOR, 0.5),
        "COPYx1": (COPYx1, 0.0),
    }

    for gate_name, (gate_factory, expected_value) in gate_factories.items():
        node = gate_factory()
        observed_value = node.distinct_symmetry()
        assert isclose(observed_value, expected_value), (
            "Distinct symmetry for %s does not match, %s != %s"
            % (gate_name, observed_value, expected_value)
        )


def test_raw_symmetry_classic_gates():
    """Check raw symmetry values for AND/OR/XOR/COPYx1."""
    gate_factories = {
        "AND": (AND, 1.0),
        "OR": (OR, 1.0),
        "XOR": (XOR, 1.0),
        "COPYx1": (COPYx1, 0.75),
    }

    for gate_name, (gate_factory, expected_value) in gate_factories.items():
        node = gate_factory()
        observed_value = node.raw_symmetry()
        assert isclose(observed_value, expected_value), (
            "Raw symmetry for %s does not match, %s != %s"
            % (gate_name, observed_value, expected_value)
        )
