"""Tests for pinning controllability: pcstg construction, mpcf,
and minimum driver-set search.

Background
----------
The pcstg edge-loop scaffolding for limit-cycle attractors has been
in place since Alex Gates' 2018 introduction of pinning
controllability (`98792ed`): the iteration over attractor STG edges,
the ``(attsource, attsink)`` pair, and the call-site form
``pinned_step(initial, pinned_binstate=attsink, pinned_var=...)``.
What was missing was the body of ``pinned_step`` that would honor
the ``pinned_binstate`` argument — pinned positions in the output
were copied from ``initial`` instead of being written to
``pinned_binstate``. For limit-cycle attractors where the pinned
variable flips, the pcstg fragmented into one disjoint snapshot per
cycle position; the cycle's states were split across snapshots, and
``fraction_pinned_configurations`` returned 0 for every such
attractor.

A related independent fix was made in 2019 (`44a96d3`, on the
``python3-friendly`` branch) via a different mechanism — splitting
``pinned_step`` into two functions and changing the call site. That
fix shipped on its branch but didn't make it to upstream when
subsequent merges branched off pre-fix history.

This test module covers the completion of Alex's original design:
the body of ``pinned_step`` now honors ``pinned_binstate``, plus two
restored utility functions for minimum driver-set search.

Functions exercised:

- ``pinned_step`` — advance the network one Boolean step under
  pinning control (read pin from ``initial``, write pin from
  ``pinned_binstate``, advance unpinned normally).
- ``pinning_controlled_state_transition_graph`` — pcstg should be a
  single connected component for any controlled driver set.
- ``fraction_pinned_configurations`` /
  ``mean_fraction_pinned_configurations`` — return correct values
  for fixed-point and limit-cycle attractors.
- ``pinning_control_driver_nodes`` — minimum driver-set search.
- ``_signatures_distinguish_attractors`` — necessary-condition
  pre-filter (handles fixed-point and limit-cycle attractor types).

Test fixture
------------
A 3-node Boolean network designed so that all three attractor types
coexist and a single pin choice exercises each scenario:

    A *= B
    B *= A
    C *= A and B

Attractor structure (verified by ``BN.attractors(mode='stg')``):

    [[2, 4], [0], [7]]
    = [{(010), (100)}, {(000)}, {(111)}]
    = [length-2 cycle, fixed point, fixed point]

Within the cycle, A and B each *flip* between cycle states (010 ↔
100), while C stays at 0. So:

- Pinning ``A`` (or ``B``) exposes the bug — pin flips in cycle.
- Pinning ``C`` is a control case — pin constant in cycle.
- Pinning ``{A, B}`` is the unique minimum pinning-control driver
  set (3 attractors → ``ceil(log2(3)) = 2`` lower bound; only
  ``{A, B}`` distinguishes all three attractor signatures).
"""

import networkx as nx
import pytest

from cana.boolean_network import BooleanNetwork


# ---------------------------------------------------------------- #
# Fixture                                                           #
# ---------------------------------------------------------------- #


def _build_two_fp_one_cycle():
    """Build the 3-node fixture and pre-compute attractors.

    Returns a ``BooleanNetwork`` with attractors already computed
    via ``BN.attractors(mode='stg')`` so callers can use
    ``BN._attractors`` directly.
    """
    rules = "\n".join(
        [
            "A *= B",
            "B *= A",
            "C *= A and B",
        ]
    )
    BN = BooleanNetwork.from_string_boolean(rules, keep_constants=True)
    BN.attractors(mode="stg")
    return BN


# Node indices for readability in tests.
A, B, C = 0, 1, 2

# Attractor positions in BN._attractors after _build_two_fp_one_cycle().
ATT_CYCLE = 0  # [2, 4] = {(010), (100)}
ATT_FP_000 = 1  # [0]
ATT_FP_111 = 2  # [7]


# ---------------------------------------------------------------- #
# Sanity: the fixture has the expected attractor structure         #
# ---------------------------------------------------------------- #


def test_fixture_has_expected_attractors():
    BN = _build_two_fp_one_cycle()
    assert BN.Nnodes == 3
    # Order is implementation-defined but stable for this BN.
    assert BN._attractors == [[2, 4], [0], [7]]


# ---------------------------------------------------------------- #
# pinned_step — unpinned advances; pin reads from initial, writes  #
# from pinned_binstate                                              #
# ---------------------------------------------------------------- #


def test_pinned_step_pin_unchanged_keeps_pin_at_source():
    """When ``pinned_binstate`` matches ``initial[pinned_var]`` (no pin
    change), the pinned positions in the output equal those in
    ``initial``; unpinned positions advance one Boolean step computed
    from ``initial``."""
    BN = _build_two_fp_one_cycle()
    # From '010' (A=0, B=1, C=0) with A pinned to its current value '0':
    # B_next = A = 0; C_next = A and B = 0; A stays at '0'.
    # Result: '000'.
    assert BN.pinned_step("010", pinned_binstate="0", pinned_var=[A]) == "000"
    # From '110' with A pinned to its current value '1':
    # B_next = A = 1; C_next = A and B = 1; A stays at '1'.
    # Result: '111'.
    assert BN.pinned_step("110", pinned_binstate="1", pinned_var=[A]) == "111"


def test_pinned_step_pin_advanced_writes_destination_pin():
    """When ``pinned_binstate`` differs from ``initial[pinned_var]`` (pin
    is being advanced — e.g., one cycle position to the next), the
    pinned positions in the output equal ``pinned_binstate``; unpinned
    positions still advance using values from ``initial`` (the source
    pin pattern)."""
    BN = _build_two_fp_one_cycle()
    # From '010' (A=0, B=1, C=0) advancing A from '0' to '1':
    # B_next = A_initial = 0; C_next = A_initial and B = 0; A → '1'.
    # Result: '100'.
    assert BN.pinned_step("010", pinned_binstate="1", pinned_var=[A]) == "100"
    # From '100' (A=1, B=0, C=0) advancing A from '1' to '0':
    # B_next = A_initial = 1; C_next = A_initial and B = 0; A → '0'.
    # Result: '010'.
    assert BN.pinned_step("100", pinned_binstate="0", pinned_var=[A]) == "010"


def test_pinned_step_multiple_pinned_vars():
    """Pinning multiple variables: pin pattern is written in the same
    order as ``pinned_var``; unpinned positions advance from ``initial``."""
    BN = _build_two_fp_one_cycle()
    # From '010' with {A, B} pinned to '10' (A→1, B→0):
    # C_next = A_initial and B_initial = 0 and 1 = 0.
    # A → '1', B → '0', C → '0'. Result: '100'.
    assert BN.pinned_step("010", pinned_binstate="10", pinned_var=[A, B]) == "100"


def test_pinned_step_length_mismatch_raises():
    """``pinned_binstate`` length must match ``pinned_var`` length."""
    BN = _build_two_fp_one_cycle()
    with pytest.raises(AssertionError):
        BN.pinned_step("000", pinned_binstate="11", pinned_var=[A])


# ---------------------------------------------------------------- #
# pcstg construction — single-WCC invariant under pinning control   #
# ---------------------------------------------------------------- #


def test_pcstg_single_wcc_fixed_point_pin_A():
    """Fixed-point attractor: pcstg restricted to states with the
    attractor's src_pin should still be one WCC."""
    BN = _build_two_fp_one_cycle()
    pcstg_dict = BN.pinning_controlled_state_transition_graph([A])
    fp_pcstg = pcstg_dict[tuple(BN._attractors[ATT_FP_000])]
    n_wccs = sum(1 for _ in nx.weakly_connected_components(fp_pcstg))
    # All 4 states with A=0 flow into (000) → single WCC.
    assert n_wccs == 1
    assert len(fp_pcstg) == 4


def test_pcstg_single_wcc_cycle_pin_flips_post_fix():
    """Length-2 cycle, pin flips around cycle. Post-fix the pcstg
    should be a single WCC containing all 8 states (with edges that
    cross between src_pin and dst_pin layers).

    Pre-fix this WCC count is 2 — the snapshot fragmentation bug.
    Test asserts the post-fix invariant.
    """
    BN = _build_two_fp_one_cycle()
    pcstg_dict = BN.pinning_controlled_state_transition_graph([A])
    cyc_pcstg = pcstg_dict[tuple(BN._attractors[ATT_CYCLE])]
    n_wccs = sum(1 for _ in nx.weakly_connected_components(cyc_pcstg))
    assert n_wccs == 1
    assert len(cyc_pcstg) == 8


def test_pcstg_cycle_pin_constant_unaffected_by_fix():
    """Pin C: C is constant in the cycle, so pcstg construction
    behaves identically pre-fix and post-fix. The cycle's pcstg
    only contains the 4 states with C=0 (the cycle's pin value).
    """
    BN = _build_two_fp_one_cycle()
    pcstg_dict = BN.pinning_controlled_state_transition_graph([C])
    cyc_pcstg = pcstg_dict[tuple(BN._attractors[ATT_CYCLE])]
    # 4 states with C=0; the cycle is one of three WCCs in this
    # restricted pcstg (the others are FP basins for (000) and (110)).
    assert len(cyc_pcstg) == 4
    n_wccs = sum(1 for _ in nx.weakly_connected_components(cyc_pcstg))
    assert n_wccs == 3
    # The cycle states must share a WCC.
    cycle_states = set(BN._attractors[ATT_CYCLE])
    cycle_wcc = next(
        wcc for wcc in nx.weakly_connected_components(cyc_pcstg)
        if cycle_states <= wcc
    )
    assert cycle_wcc == cycle_states


# ---------------------------------------------------------------- #
# fraction_pinned_configurations + mpcf                             #
# ---------------------------------------------------------------- #


def test_pcf_pin_A_post_fix():
    """Pin A. Pre-fix the cycle attractor returns 0 (bug). Post-fix
    every attractor's basin reaches its target → all pcf = 1.0."""
    BN = _build_two_fp_one_cycle()
    pcstg_dict = BN.pinning_controlled_state_transition_graph([A])
    pcf = BN.fraction_pinned_configurations(pcstg_dict)
    # _attractors order: [cycle, fp000, fp111]
    assert pcf == [1.0, 1.0, 1.0]
    assert BN.mean_fraction_pinned_configurations(pcstg_dict) == 1.0


def test_pcf_pin_C_unchanged():
    """Pin C: pin is constant in cycle, bug does not fire. Pre- and
    post-fix should both give the same values."""
    BN = _build_two_fp_one_cycle()
    pcstg_dict = BN.pinning_controlled_state_transition_graph([C])
    pcf = BN.fraction_pinned_configurations(pcstg_dict)
    # Per-attractor: cycle has 2 states out of 4 in its WCC = 0.5;
    # fixed points each have 1 of 4 = 0.25.
    assert pcf == [0.5, 0.25, 0.25]


def test_pcf_pin_AB_post_fix():
    """Pin {A, B}: minimum pinning-control set. Both pinned vars
    flip in cycle; post-fix gives full pinning control, mpcf = 1.0."""
    BN = _build_two_fp_one_cycle()
    pcstg_dict = BN.pinning_controlled_state_transition_graph([A, B])
    pcf = BN.fraction_pinned_configurations(pcstg_dict)
    assert pcf == [1.0, 1.0, 1.0]


def test_mpcf_consistency_with_fraction_pinned():
    """``mean_fraction_pinned_configurations`` is the mean of
    ``fraction_pinned_configurations`` — keeps the contract simple."""
    BN = _build_two_fp_one_cycle()
    pcstg_dict = BN.pinning_controlled_state_transition_graph([A])
    pcf = BN.fraction_pinned_configurations(pcstg_dict)
    mpcf = BN.mean_fraction_pinned_configurations(pcstg_dict)
    assert mpcf == pytest.approx(sum(pcf) / len(pcf))


# ---------------------------------------------------------------- #
# pinning_control_driver_nodes — minimum driver-set search          #
# ---------------------------------------------------------------- #


def test_pinning_control_driver_nodes_finds_unique_minimum():
    """For our 3-attractor fixture, the minimum pinning control set
    is {A, B}. ``{A, C}`` and ``{B, C}`` fail the necessary-condition
    pre-filter (signature collision between fp000 and the cycle).
    """
    BN = _build_two_fp_one_cycle()
    result = BN.pinning_control_driver_nodes()
    # Expect exactly one minimum set: {A, B} (size 2 = ceil(log2(3))).
    assert len(result) == 1
    assert set(result[0]) == {A, B}


def test_pinning_control_driver_nodes_size_above_log2_lower_bound():
    """The search starts at the info-theoretic lower bound
    ``ceil(log2(N_attractors))`` and increments. For our fixture
    (3 attractors), the lower bound is 2 and the answer matches it,
    so size-1 candidates are not enumerated."""
    BN = _build_two_fp_one_cycle()
    result = BN.pinning_control_driver_nodes()
    for dvs in result:
        assert len(dvs) == 2


# ---------------------------------------------------------------- #
# Real-network regression: Arabidopsis thaliana (Chaos 2006)        #
# ---------------------------------------------------------------- #
#
# Thaliana exposed a sufficiency-check bug: prior to this fix the
# search used ``WCC count == 1`` per pcstg, which is necessary but
# not sufficient — multiple original attractors can collapse into a
# single pcstg in which the target is reachable but not the unique
# attracting SCC. Returns a false-positive size-5 set
# ``{AP3, UFO, AP1, LFY, WUS}``. The strict criterion (unique
# attracting SCC equal to the target) recovers Alex Gates' 2018
# slide result of three size-6 sets.
#
# Cost: ~15s. Sole network-scale regression test in this module.


def test_pinning_control_driver_nodes_thaliana_strict_criterion():
    """Search must apply the strict criterion (unique attracting SCC
    equals target), not the weaker WCC==1. Returns Alex Gates' three
    size-6 sets, not the historical size-5 false positive."""
    from cana.datasets.bio import THALIANA

    bn = THALIANA()
    result = bn.pinning_control_driver_nodes()
    names = [n.name for n in bn.nodes]
    sets = {frozenset(names[i] for i in dvs) for dvs in result}

    # Alex Gates 2018 slide-3 result (three alternative size-6 sets,
    # all sharing {UFO, WUS, AP3, AG} and extended by one of three pairs):
    expected = {
        frozenset({"UFO", "WUS", "AP3", "AG", "AP1", "LFY"}),
        frozenset({"UFO", "WUS", "AP3", "AG", "TFL1", "EMF1"}),
        frozenset({"UFO", "WUS", "AP3", "AG", "TFL1", "LFY"}),
    }
    assert sets == expected
    assert all(len(dvs) == 6 for dvs in result)


def test_thaliana_size5_false_positive_rejected_by_strict_check():
    """Direct regression on the strict criterion: the historical
    false-positive size-5 set ``{AP3, UFO, AP1, LFY, WUS}`` must fail
    the strict per-attractor check. Targets at least one attractor
    whose pcstg's unique attracting SCC differs from the target.

    Faster than the full search (~3s vs ~15s); exists so the
    regression survives even if the search algorithm changes shape.
    """
    from cana.datasets.bio import THALIANA

    bn = THALIANA()
    names = [n.name for n in bn.nodes]
    pin = sorted(names.index(n) for n in ["AP3", "UFO", "AP1", "LFY", "WUS"])
    pcstg_dict = bn.pinning_controlled_state_transition_graph(pin)

    # At least one attractor must fail the strict check. The pcstg's
    # set of attracting SCCs must not equal {set(att)}.
    failures = 0
    for att, pcstg in pcstg_dict.items():
        attracting = list(nx.attracting_components(pcstg))
        if not (len(attracting) == 1 and attracting[0] == set(att)):
            failures += 1
    assert failures > 0, (
        "Strict criterion should reject the size-5 set on Thaliana; "
        "if this assertion passes, the WCC==1-only check has been "
        "reintroduced or the bug does not manifest on Thaliana anymore."
    )


# ---------------------------------------------------------------- #
# _signatures_distinguish_attractors — necessary-condition filter   #
# ---------------------------------------------------------------- #


def test_signatures_distinguish_attractors_handles_fixed_point():
    """Two fixed points with distinct signatures on the candidate
    nodes pass the necessary-condition test."""
    from cana.boolean_network import _signatures_distinguish_attractors

    # bin_attractors[i] is a list of binary state strings for attractor i.
    fp1 = ["000"]
    fp2 = ["111"]
    assert _signatures_distinguish_attractors([0], [fp1, fp2])
    # Both signatures collapse to () when no nodes are pinned →
    # cannot distinguish; return False.
    assert not _signatures_distinguish_attractors([], [fp1, fp2])


def test_signatures_distinguish_attractors_collision_fixed_point():
    """Two fixed points sharing a signature on the candidate nodes
    cannot be distinguished by pinning."""
    from cana.boolean_network import _signatures_distinguish_attractors

    # Both attractors agree on bit 1 (B=0).
    fp1 = ["000"]
    fp2 = ["100"]
    assert not _signatures_distinguish_attractors([1], [fp1, fp2])


def test_signatures_distinguish_attractors_cycle_pin_constant():
    """A length-2 cycle whose pin is constant within the cycle is
    treated as fixed-point-like: its signature must not collide with
    any other attractor's signature on the candidate nodes.

    Setup: fixed point (000) has C=0; cycle ((010), (100)) has C
    constant=0 throughout. On candidate ``[C]`` both attractors have
    signature ``("0",)`` — collision.
    """
    from cana.boolean_network import _signatures_distinguish_attractors

    fp = ["000"]
    cycle_pin_constant = ["010", "100"]
    assert not _signatures_distinguish_attractors([C], [fp, cycle_pin_constant])


def test_signatures_distinguish_attractors_cycle_pin_flips():
    """A length-2 cycle whose pin flips within the cycle has its
    per-state signatures checked against fixed-point signatures.
    Per-state collision with a fixed-point fails the test.

    Setup: fixed point (010) has A=0; cycle ((010), (100)) has A=0 in
    one state and A=1 in the other. On candidate ``[A]`` the cycle's
    per-state signature ``("0",)`` collides with the fixed point's.
    """
    from cana.boolean_network import _signatures_distinguish_attractors

    fp = ["010"]
    cycle_pin_flips = ["010", "100"]
    assert not _signatures_distinguish_attractors([A], [fp, cycle_pin_flips])
