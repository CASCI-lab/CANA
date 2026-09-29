import itertools
from math import ceil, log2

import networkx as nx

from cana.cutils import binstate_pinned_to_binstate, statenum_to_binstate


def _signatures_distinguish_attractors(candidate_nodes, bin_attractors):
    """Necessary-condition pre-filter for pinning controllability.

    Returns ``True`` iff no obvious signature collision is detected
    among attractors on ``candidate_nodes``. The signature is the
    tuple of values the candidate nodes take in the attractor:

    - **Fixed-point attractor** (length 1): signature is the tuple of
      values at the single attractor state.
    - **Limit-cycle attractor, pin constant within the cycle**: all
      cycle states agree on the candidate nodes; the cycle is treated
      as fixed-point-like with that single signature.
    - **Limit-cycle attractor, pin flips within the cycle**: each
      cycle state contributes its own signature; each must not
      collide with another attractor's fixed-point-style signature.

    This is a *necessary* condition, not sufficient: two attractors
    that pass here may still fail the downstream PCSTG-WCC check
    (e.g., two flipping cycles whose per-state signatures happen to
    overlap on the candidate nodes — rare in practice). Sufficiency
    is verified in :func:`BooleanNetwork.pinning_control_driver_nodes`.

    Args:
        candidate_nodes (list of int): node indices to check.
        bin_attractors (list of list of str): attractors as lists of
            binary state strings; ``bin_attractors[i][j][k]`` is the
            value of node ``k`` at the ``j``-th state of attractor ``i``.

    Returns:
        bool: ``True`` if no collision is found among fixed-point /
        pin-constant signatures, and no flipping-cycle per-state
        signature collides with a fixed-point signature. ``False``
        on any detected collision (including the trivial
        ``len(candidate_nodes) == 0`` case). Note: collisions
        *among* flipping cycles are not checked here — those are
        caught by the downstream pcstg sufficiency check.
    """
    if len(candidate_nodes) == 0:
        return False
    fixed_signatures = set()
    flipping_attractors = []
    for attr in bin_attractors:
        sig = tuple(attr[0][node] for node in candidate_nodes)
        is_pin_constant = all(
            tuple(state[node] for node in candidate_nodes) == sig
            for state in attr[1:]
        )
        if len(attr) == 1 or is_pin_constant:
            if sig in fixed_signatures:
                return False
            fixed_signatures.add(sig)
        else:
            flipping_attractors.append(attr)
    for attr in flipping_attractors:
        for state in attr:
            sig = tuple(state[node] for node in candidate_nodes)
            if sig in fixed_signatures:
                return False
    return True


def pinning_control_driver_nodes(
        attractors,
        Nnodes,
        keep_constants,
        constant_nodeids,
        num2bin,
        pinning_controlled_state_transition_graph
):

    if len(attractors) == 1:
        return [()]
    
    lower_bound = ceil(log2(len(attractors)))
    nodeids = list(range(Nnodes))
    # Exclude constant nodes: they cannot distinguish attractors
    # and waste combinatorial search effort. If you need to treat
    # a constant node as a controllable driver (e.g. toggling a
    # stimulus), modify the model to make it non-constant before
    # calling this function.
    if keep_constants:
        nodeids = [nodeid for nodeid in nodeids if nodeid not in constant_nodeids]
    bin_attractors = [
        [num2bin(state) for state in attr] for attr in attractors
    ]
    result = []
    max_pin = len(nodeids)
    if lower_bound > max_pin:
        return [tuple(range(Nnodes))]
    for n_pin in range(lower_bound, max_pin + 1):
        if result:
            break
        for pvs in itertools.combinations(nodeids, n_pin):
            if not _signatures_distinguish_attractors(
                list(pvs), bin_attractors
            ):
                continue
            controlled = True
            pcstg_dict = pinning_controlled_state_transition_graph(
                list(pvs)
            )
            for att, pcstg in pcstg_dict.items():
                # Strict check: pcstg must have exactly one
                # attracting SCC, and that SCC must equal the
                # target attractor's state set. Pure
                # set/integer comparison — no floating point.
                attracting = list(nx.attracting_components(pcstg))
                if len(attracting) != 1 or attracting[0] != set(att):
                    controlled = False
                    break
            if controlled:
                result.append(pvs)
    if not result:
        return [tuple(range(Nnodes))]
    return result


def pinning_controlled_state_transition_graph(
        attractors,
        stg,
        network_name,
        driver_nodes,
        Nnodes,
        nodes,
        num2bin,
        bin2num,
        pinned_step,
):

    uncontrolled_system_size = Nnodes - len(driver_nodes)

    pcstg_dict = {}
    for att in attractors:
        # For each STG edge ``(s_src, s_dst)`` *inside* the attractor,
        # ``src_pin`` and ``dst_pin`` are the projections of those
        # states onto the pinned variables. For a fixed-point
        # attractor the self-loop gives ``src_pin == dst_pin``; for a
        # length-L cycle the L tuples have ``dst_pin`` rotated one
        # cycle-step ahead of ``src_pin``. (These are the same loop
        # variables previously named ``attsource`` and ``attsink``;
        # renamed because the old names suggested *attractor states*
        # when they actually hold the *pin-bit projections* of those
        # states.)
        dn_attractor_transitions = [
            tuple(
                "".join([num2bin(s)[dn] for dn in driver_nodes])
                for s in att_edge
            )
            for att_edge in stg.subgraph(att).edges()
        ]

        pcstg_states = [
            bin2num(
                binstate_pinned_to_binstate(
                    statenum_to_binstate(statenum, base=uncontrolled_system_size),
                    src_pin,
                    pinned_var=driver_nodes,
                )
            )
            for statenum in range(2**uncontrolled_system_size)
            for src_pin, _dst_pin in dn_attractor_transitions
        ]

        pcstg = nx.DiGraph(name="STG: " + network_name)
        pcstg.name = (
            "PC-"
            + pcstg.name
            + " ("
            + ",".join(map(str, [nodes[dv].name for dv in driver_nodes]))
            + ")"
        )

        pcstg.add_nodes_from((ps, {"label": ps}) for ps in pcstg_states)

        for src_pin, dst_pin in dn_attractor_transitions:
            for statenum in range(2**uncontrolled_system_size):
                initial = binstate_pinned_to_binstate(
                    statenum_to_binstate(statenum, base=uncontrolled_system_size),
                    src_pin,
                    pinned_var=driver_nodes,
                )
                # ``pinned_step`` advances the unpinned variables
                # using ``initial`` (which has ``src_pin`` at the
                # pinned positions) and writes ``dst_pin`` at the
                # pinned positions of the output. For fixed-point
                # attractors ``src_pin == dst_pin`` so the pinned
                # positions are unchanged; for cycles where the pin
                # flips, this is what connects positions of the
                # pcstg around the cycle.
                pcstg.add_edge(
                    bin2num(initial),
                    bin2num(
                        pinned_step(
                            initial,
                            pinned_binstate=dst_pin,
                            pinned_var=driver_nodes,
                        )
                    ),
                )

        pcstg_dict[tuple(att)] = pcstg

    return pcstg_dict


def pinned_step(
        initial,
        pinned_binstate,
        pinned_var,
        Nnodes,
        logic,
        nodes
):
    """Advance the network one Boolean step under pinning control."""
    if len(initial) != Nnodes:
        raise ValueError(
            "initial state length must equal Nnodes: "
            "expected %d, got %d" % (Nnodes, len(initial))
        )
    if len(pinned_binstate) != len(pinned_var):
        raise ValueError(
            "pinned_binstate length must match pinned_var: "
            "expected %d, got %d" % (len(pinned_var), len(pinned_binstate))
        )
    # Build a quick lookup so the comprehension is O(Nnodes)
    # rather than O(Nnodes * |pinned_var|).
    pin_map = dict(zip(pinned_var, pinned_binstate))
    return "".join(
        pin_map[i]
        if i in pin_map
        else str(node.step("".join(initial[j] for j in logic[i]["in"])))
        for i, node in enumerate(nodes, start=0)
    )


def fraction_pinned_attractors(pcstg_dict):
    """Returns the Number of Accessible Attractors

    Args:
        pcstg_dict (dict of networkx.DiGraph) : The dictionary of Pinned Controlled State-Transition-Graphs.

    Returns:
        (int) : Number of Accessible Attractors
    """
    reached_attractors = []
    for att, pcstg in pcstg_dict.items():
        pinned_att = list(nx.attracting_components(pcstg))
        print(set(att), pinned_att)
        reached_attractors.append(set(att) in pinned_att)
    return sum(reached_attractors) / float(len(pcstg_dict))

def fraction_pinned_configurations(pcstg_dict):
    """Returns the Fraction of successfully Pinned Configurations

    Args:
        pcstg_dict (dict of networkx.DiGraph) : The dictionary of Pinned Controlled State-Transition-Graphs.

    Returns:
        (list) : the Fraction of successfully Pinned Configurations to each attractor
    """
    pinned_configurations = []
    for att, pcstg in pcstg_dict.items():
        att_reached = False
        for wcc in nx.weakly_connected_components(pcstg):
            if set(att) in list(nx.attracting_components(pcstg.subgraph(wcc))):
                pinned_configurations.append(len(wcc) / len(pcstg))
                att_reached = True
        if not att_reached:
            pinned_configurations.append(0)

    return pinned_configurations

def mean_fraction_pinned_configurations(pcstg_dict):
    """Returns the mean Fraction of successfully Pinned Configurations

    Args:
        pcstg_dict (dict of networkx.DiGraph) : The dictionary of Pinned Controlled State-Transition-Graphs.

    Returns:
        (int) : the mean Fraction of successfully Pinned Configurations
    """
    return sum(fraction_pinned_configurations(pcstg_dict)) / len(pcstg_dict)