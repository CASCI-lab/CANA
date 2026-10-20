# %% [markdown]
# Canalizing Map ↔ LUT roundtrip
#
# Walkthrough:
# 1. Build a Boolean node from a LUT or boolean rules.
# 2. Convert to a Canalizing Map (CM) via `BooleanNode.canalizing_map`.
# 3. Reconstruct a LUT from that CM via `cm_lut.lut_from_canalizing_map`.
# 4. Compare original vs reconstructed LUTs.
#
# The reverse step evaluates the CM modularly from node kinds + topology
# (see `cm_lut` module docstring).  It does **not** assume the forward
# `canalizing_map` builder is correct — and for CTNNB1 it is not.

# %%
from __future__ import annotations

import os
import sys

_DEV_DIR = os.path.dirname(os.path.abspath(__file__))
if _DEV_DIR not in sys.path:
    sys.path.insert(0, _DEV_DIR)

import networkx as nx
from cana.boolean_network import BooleanNetwork
from cana.boolean_node import BooleanNode

from cm_lut import (
    ROLE_POLARITY0,
    ROLE_POLARITY1,
    ROLE_SCHEMA,
    ROLE_SYMMETRY,
    connect,
    input_ids_in_order,
    lut_from_canalizing_map,
    make_fusion,
    make_input_node,
    make_output_node,
    make_threshold,
)


def summarize_roundtrip(name, original_outputs, result):
    orig = "".join(str(x) for x in original_outputs)
    recon = result.lut_string()
    exact = (
        not result.has_conflict
        and not result.is_incomplete
        and [int(x) for x in original_outputs] == result.outputs
    )
    print(f"=== {name} ===")
    print(f"  inputs     : {result.input_names}")
    print(f"  original   : {orig}")
    print(f"  from CM    : {recon}")
    print(f"  conflict   : {result.has_conflict}  states={result.conflict_states}")
    print(f"  incomplete : {result.is_incomplete}  uncovered={result.uncovered_states[:8]}"
          f"{'...' if len(result.uncovered_states) > 8 else ''}")
    print(f"  exact match: {exact}")
    return exact


# %% [markdown]
# ## Case 1 — AND / OR from LUT
#
# Simple canalizing functions.  The current forward CM builder usually gets
# these right, so the roundtrip can succeed.

# %%
and_node = BooleanNode.from_output_list(
    [0, 0, 0, 1], name="AND", inputs=[0, 1], id=2
)
and_cm = and_node.canalizing_map()
and_result = lut_from_canalizing_map(and_cm, input_order=and_node.inputs)
summarize_roundtrip("AND", and_node.outputs, and_result)

or_node = BooleanNode.from_output_list(
    [0, 1, 1, 1], name="OR", inputs=[10, 11], id=5
)
or_cm = or_node.canalizing_map()
or_result = lut_from_canalizing_map(or_cm, input_order=or_node.inputs)
summarize_roundtrip("OR", or_node.outputs, or_result)

# %% [markdown]
# ## Case 2 — CTNNB1 from boolean rules
#
# ```
# CTNNB1 = (not SRY) and (WNT4 or RSPO1) and ((not AMH) or (not SOX9))
# ```
#
# Even with a correct reverse interpreter, the reconstructed LUT will **not**
# match the true function, because `BooleanNode.canalizing_map` mis-wires
# multi-group permutation schemata (it always uses `ps[0]` for fusion fibres).

# %%
CTNNB1_BNET = (
    "CTNNB1*=(not SRY) and (WNT4 or RSPO1) and ((not AMH) or (not SOX9))\n"
    "SRY*=SRY\n"
    "WNT4*=WNT4\n"
    "RSPO1*=RSPO1\n"
    "AMH*=AMH\n"
    "SOX9*=SOX9\n"
)

N = BooleanNetwork.from_string_boolean(CTNNB1_BNET)
ct = next(n for n in N.nodes if n.name == "CTNNB1")
input_labels = [N.nodes[i].name for i in ct.inputs]
print("CTNNB1 inputs (LUT order):", list(zip(ct.inputs, input_labels)))
print("True LUT:", "".join(ct.outputs))

ct._check_compute_canalization_variables(two_symbols=True)
print("Two-symbol schemata (out=0):")
for row in ct._two_symbols[0]:
    print(" ", row)
print("Two-symbol schemata (out=1):")
for row in ct._two_symbols[1]:
    print(" ", row)

ct_cm = ct.canalizing_map()
ct_result = lut_from_canalizing_map(ct_cm, input_order=ct.inputs)
exact = summarize_roundtrip("CTNNB1 (via buggy canalizing_map)", ct.outputs, ct_result)

# Side-by-side diff of a few disagreeing states
print("\nDisagreeing / problematic states (up to 12):")
shown = 0
for i, (true, got) in enumerate(zip([int(x) for x in ct.outputs], ct_result.outputs)):
    state = format(i, "05b")
    labels = {input_labels[j]: int(state[j]) for j in range(5)}
    if state in ct_result.conflict_states:
        tag = "CONFLICT"
    elif got is None:
        tag = "UNCOVERED"
    elif got != true:
        tag = f"MISMATCH true={true} got={got}"
    else:
        continue
    print(f"  {state} {labels}  {tag}")
    shown += 1
    if shown >= 12:
        break

assert exact is False, "Expected CTNNB1 roundtrip to fail while canalizing_map is buggy"

# %% [markdown]
# ## Case 3 — Hand-correct mini-CM (fragment)
#
# A correct CM fragment for the conjunct `(WNT4 or RSPO1)` alone
# (out=1 when at least one is 1; out=0 when both are 0).
# Reverse reconstruction recovers the OR LUT exactly — showing the
# interpreter itself is sound when the graph is right.

# %%
# Label-driven helpers: ids / tids / edge types are assigned automatically.

G = nx.DiGraph(name="real_CTNNB1")
out0 = make_output_node(G, "CTNNB1", 0)
out1 = make_output_node(G, "CTNNB1", 1)

thr0 = make_threshold(G, tau=1)
thr1 = make_threshold(G, tau=2)
thr2 = make_threshold(G, tau=2)
connect(G, thr0, out0)
connect(G, thr1, out0)
connect(G, thr2, out0)
connect(G, make_input_node(G, "SRY", 1), thr0)
connect(G, make_input_node(G, "WNT4", 0), thr1)
connect(G, make_input_node(G, "RSPO1", 0), thr1)
connect(G, make_input_node(G, "AMH", 1), thr2)
connect(G, make_input_node(G, "SOX9", 1), thr2)

thr3 = make_threshold(G, tau=3)
connect(G, thr3, out1)
fus0 = make_fusion(G)
fus1 = make_fusion(G)
connect(G, fus0, thr3)
connect(G, fus1, thr3)
connect(G, make_input_node(G, "SRY", 0), thr3)
connect(G, make_input_node(G, "WNT4", 1), fus0)
connect(G, make_input_node(G, "RSPO1", 1), fus0)
connect(G, make_input_node(G, "AMH", 0), fus1)
connect(G, make_input_node(G, "SOX9", 0), fus1)

hand_result = lut_from_canalizing_map(G, input_order=input_ids_in_order(G))
summarize_roundtrip("Hand-correct CTNNB1", "00001110111011100000000000000000", hand_result)
assert hand_result.outputs == [int(bit) for bit in "00001110111011100000000000000000"]
assert not hand_result.has_conflict and not hand_result.is_incomplete

# %% [markdown]
# ## Case 4 — Canonical nested-threshold CM
#
# Same CTNNB1 function with nested schema / symmetry / polarity thresholds.
# Build order is free; thresholds do not store an ``out`` attribute.

# %%
G = nx.DiGraph(name="canonical_CTNNB1")
out0 = make_output_node(G, "CTNNB1", 0)
out1 = make_output_node(G, "CTNNB1", 1)

# out=0: three independent two-symbol schemata
out0_two0 = make_threshold(G, tau=1, role=ROLE_SCHEMA)
out0_two1 = make_threshold(G, tau=2, role=ROLE_SCHEMA)
out0_two2 = make_threshold(G, tau=2, role=ROLE_SCHEMA)
connect(G, out0_two0, out0)
connect(G, out0_two1, out0)
connect(G, out0_two2, out0)
connect(G, make_input_node(G, "SRY", 1), out0_two0)
connect(G, make_input_node(G, "WNT4", 0), out0_two1)
connect(G, make_input_node(G, "RSPO1", 0), out0_two1)
connect(G, make_input_node(G, "AMH", 1), out0_two2)
connect(G, make_input_node(G, "SOX9", 1), out0_two2)

# out=1: schema over SRY literal + two symmetry fibres (leaf-first ok)
out1_two0 = make_threshold(G, tau=3, role=ROLE_SCHEMA)
connect(G, out1_two0, out1)
out1_two0_sym0 = make_threshold(G, tau=1, role=ROLE_SYMMETRY)
out1_two0_sym1 = make_threshold(G, tau=1, role=ROLE_SYMMETRY)
connect(G, out1_two0_sym0, out1_two0)
connect(G, out1_two0_sym1, out1_two0)
out1_two0_sym0_g1 = make_threshold(G, tau=1, role=ROLE_POLARITY1)
out1_two0_sym1_g0 = make_threshold(G, tau=1, role=ROLE_POLARITY0)
connect(G, out1_two0_sym0_g1, out1_two0_sym0)
connect(G, out1_two0_sym1_g0, out1_two0_sym1)
connect(G, make_input_node(G, "SRY", 0), out1_two0)
connect(G, make_input_node(G, "WNT4", 1), out1_two0_sym0_g1)
connect(G, make_input_node(G, "RSPO1", 1), out1_two0_sym0_g1)
connect(G, make_input_node(G, "AMH", 0), out1_two0_sym1_g0)
connect(G, make_input_node(G, "SOX9", 0), out1_two0_sym1_g0)

canon_result = lut_from_canalizing_map(G, input_order=input_ids_in_order(G))
summarize_roundtrip("Canonical CTNNB1", "00001110111011100000000000000000", canon_result)
assert canon_result.outputs == [int(bit) for bit in "00001110111011100000000000000000"]
assert not canon_result.has_conflict and not canon_result.is_incomplete

# %% [markdown]
# ## Takeaway
#
# * Construction helpers are label-driven: ``connect`` infers edge types;
#   ``tid`` / input ids are automatic; thresholds do not store ``out``.
# * `lut_from_canalizing_map` evaluates nodes modularly from topology.
# * Threshold ``role`` annotates schema / symmetry / polarity0 / polarity1.
# * Roundtrip still fails for CTNNB1 via the **forward** `canalizing_map`
#   builder; hand-built fusion and nested-threshold CMs both reconstruct.
