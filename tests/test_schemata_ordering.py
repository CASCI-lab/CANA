# -*- coding: utf-8 -*-
#
# Regression tests: schemata output must have a canonical, process-stable order.
#
# Prime implicants (and the two-symbol schemata derived from them) are stored as
# Python sets of pattern strings. A set of strings iterates in a different order in
# every fresh interpreter because str.__hash__ is salted per process
# (PYTHONHASHSEED). Without sorting, schemata_look_up_table() therefore returns its
# rows in a per-process-random order (the row CONTENT is always identical -- the
# prime-implicant set is mathematically unique -- only the order permutes).
#
# These tests spawn fresh subprocesses with different PYTHONHASHSEED values and
# assert byte-identical row order. Subprocesses are ESSENTIAL: within a single
# process the order is already deterministic, so an in-process test cannot catch
# the bug.
import os
import subprocess
import sys

import pytest


# LUTs exercised: two structured k=3 rules plus deterministically-generated random
# k=5 and k=7 rules. Generation uses random.Random(seed) so it is independent of
# PYTHONHASHSEED (only str hashing is salted, not the Mersenne Twister).
_LUT_SETUP = (
    "import random\n"
    "luts = [('maj3', '00010111'), ('par3', '01101001')]\n"
    "for k, seed in [(5, 5), (7, 7)]:\n"
    "    rng = random.Random(seed)\n"
    "    luts.append(('rand%d' % k, ''.join(rng.choice('01') for _ in range(1 << k))))\n"
)


def _run_across_hashseeds(snippet):
    """Run snippet under several PYTHONHASHSEED values; return the stdout list."""
    outputs = []
    for seed in ("0", "1", "2", "3"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        result = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, result.stderr
        outputs.append(result.stdout)
    return outputs


def test_schemata_pi_order_deterministic_across_hashseeds():
    snippet = (
        "from cana.boolean_node import BooleanNode\n"
        + _LUT_SETUP
        + "out = []\n"
        "for name, lut in luts:\n"
        "    node = BooleanNode.from_output_list(list(lut))\n"
        "    rows = list(node.schemata_look_up_table(type='pi')"
        ".itertuples(index=False, name=None))\n"
        "    out.append(name + ':' + repr(rows))\n"
        "print(chr(10).join(out))\n"
    )
    outputs = _run_across_hashseeds(snippet)
    assert len(set(outputs)) == 1, (
        "schemata_look_up_table(type='pi') row order varied across "
        "PYTHONHASHSEED values:\n" + "\n---\n".join(outputs)
    )


def test_schemata_ts_order_deterministic_across_hashseeds():
    # The two-symbol path needs the `schematodes` package.
    pytest.importorskip("schematodes")
    snippet = (
        "from cana.boolean_node import BooleanNode\n"
        + _LUT_SETUP
        + "out = []\n"
        "for name, lut in luts:\n"
        "    node = BooleanNode.from_output_list(list(lut))\n"
        "    rows = list(node.schemata_look_up_table(type='ts')"
        ".itertuples(index=False, name=None))\n"
        "    out.append(name + ':' + repr(rows))\n"
        "print(chr(10).join(out))\n"
    )
    outputs = _run_across_hashseeds(snippet)
    assert len(set(outputs)) == 1, (
        "schemata_look_up_table(type='ts') row order varied across "
        "PYTHONHASHSEED values:\n" + "\n---\n".join(outputs)
    )


def test_pi_rows_are_sorted_and_content_preserved():
    """In-process: rows are now emitted in canonical (sorted) order, and the set of
    rows is unchanged (content preserved, only ordering canonicalized)."""
    from cana.boolean_node import BooleanNode

    node = BooleanNode.from_output_list(list("00010111"))
    df = node.schemata_look_up_table(type="pi")
    rows = list(df.itertuples(index=False, name=None))
    # Rows are grouped by output (0 then 1); within each output group the schemata
    # are now emitted in canonical (sorted) order.
    for out_val in (0, 1):
        group = [schemata for schemata, output in rows if output == out_val]
        assert group == sorted(group)
    # Content preserved: no rows lost or duplicated.
    assert len(rows) == len(set(rows))
