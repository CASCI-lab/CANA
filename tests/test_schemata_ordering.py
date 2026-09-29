# -*- coding: utf-8 -*-
#
# Schemata order and k_r/k_s must be identical across processes. Prime implicants
# are stored in sets of strings, whose iteration order depends on PYTHONHASHSEED,
# so these tests spawn subprocesses with different seeds; an in-process test
# cannot catch the problem.
import os
import subprocess
import sys

import pytest


def _assert_identical_across_hashseeds(what, per_node_expr):
    """Evaluate `per_node_expr` (an expression in `node`) for two structured k=3 rules
    and seeded random k=5 and k=7 rules, in a fresh interpreter under several
    PYTHONHASHSEED values, and assert the printed results are identical."""
    snippet = (
        "import random\n"
        "from cana.boolean_node import BooleanNode\n"
        "luts = [('maj3', '00010111'), ('par3', '01101001')]\n"
        "for k, seed in [(5, 5), (7, 7)]:\n"
        "    rng = random.Random(seed)\n"
        "    luts.append(('rand%d' % k, ''.join(rng.choice('01') for _ in range(1 << k))))\n"
        "for name, lut in luts:\n"
        "    node = BooleanNode.from_output_list(list(lut))\n"
        "    print(name, repr(" + per_node_expr + "))\n"
    )
    outputs = []
    for seed in ("0", "1", "2", "3"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        result = subprocess.run(
            [sys.executable, "-c", snippet], capture_output=True, text=True, env=env
        )
        assert result.returncode == 0, result.stderr
        outputs.append(result.stdout)
    assert len(set(outputs)) == 1, (
        what + " varied across PYTHONHASHSEED values:\n" + "\n---\n".join(outputs)
    )


def test_schemata_pi_order_deterministic_across_hashseeds():
    _assert_identical_across_hashseeds(
        "schemata_look_up_table(type='pi') row order",
        "list(node.schemata_look_up_table(type='pi').itertuples(index=False, name=None))",
    )


def test_schemata_ts_order_deterministic_across_hashseeds():
    pytest.importorskip("schematodes")
    _assert_identical_across_hashseeds(
        "schemata_look_up_table(type='ts') row order",
        "list(node.schemata_look_up_table(type='ts').itertuples(index=False, name=None))",
    )


def test_measures_bit_identical_across_hashseeds():
    _assert_identical_across_hashseeds(
        "k_r / k_e / k_s",
        "(node.input_redundancy(norm=False), node.input_redundancy(operator=max, norm=False),"
        " node.effective_connectivity(norm=False), node.input_symmetry(aggOp='mean'),"
        " node.input_symmetry_mean(), node.input_symmetry(aggOp='max'))",
    )


def test_pi_rows_are_sorted_and_content_preserved():
    """Rows are emitted in sorted order within each output group, none lost or duplicated."""
    from cana.boolean_node import BooleanNode

    node = BooleanNode.from_output_list(list("00010111"))
    df = node.schemata_look_up_table(type="pi")
    rows = list(df.itertuples(index=False, name=None))
    for out_val in (0, 1):
        group = [schemata for schemata, output in rows if output == out_val]
        assert group == sorted(group)
    assert len(rows) == len(set(rows))
