# %% [markdown]
# Canalizing Map ↔ LUT roundtrip
#
# Walkthrough:
# 1. Build a Boolean node from a LUT or boolean rules.
# 2. Convert to a Canalizing Map (CM) via `BooleanNode.canalizing_map`.
# 3. Reconstruct a LUT from that CM via `cm_lut.lut_from_canalizing_map`.
# 4. Compare original vs reconstructed LUTs.
#
# The reverse step interprets t-units strictly from the CM graph
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
# OR of two inputs: same structure as hand_and but swapped polarities
# out=0: both literals at 0, tau=2
# out=1: fusion OR of both at 1, tau=1

G = nx.DiGraph(name="WNT4_or_RSPO1")
# inputs 0=WNT4, 1=RSPO1; node id 99 = fragment output
out0 = make_output_node(G, node_id=99, value=0, label="OR_frag")
out1 = make_output_node(G, node_id=99, value=1, label="OR_frag")

thr0 = make_threshold(G, tid=0, node_id=99, out=0, tau=2)
w0 = make_input_node(G, var_id=0, value=0, label="WNT4")
r0 = make_input_node(G, var_id=1, value=0, label="RSPO1")
G.add_edge(w0, thr0, type="literal")
G.add_edge(r0, thr0, type="literal")
G.add_edge(thr0, out0, type="out")

thr1 = make_threshold(G, tid=1, node_id=99, out=1, tau=1)
fus1 = make_fusion(G, fus=0, tid=1, node_id=99, out=1)
w1 = make_input_node(G, var_id=0, value=1, label="WNT4")
r1 = make_input_node(G, var_id=1, value=1, label="RSPO1")
G.add_edge(w1, fus1, type="fusing")
G.add_edge(r1, fus1, type="fusing")
G.add_edge(fus1, thr1, type="fused")
G.add_edge(thr1, out1, type="out")

frag_result = lut_from_canalizing_map(G, input_order=[0, 1])
summarize_roundtrip("Hand-correct (WNT4 OR RSPO1) fragment", [0, 1, 1, 1], frag_result)
assert frag_result.outputs == [0, 1, 1, 1]
assert not frag_result.has_conflict and not frag_result.is_incomplete

# %% [markdown]
# ## Takeaway
#
# * `lut_from_canalizing_map` evaluates each t-unit as
#   `score = active_literals + active_fusions`, `fire <=> score >= tau`,
#   with fusions as OR of their fusing parents.
# * Conflict / incompleteness are reported per input state.
# * Roundtrip fails for CTNNB1 today because the **forward** CM is wrong,
#   not because the reverse interpreter cannot read a well-formed CM.
