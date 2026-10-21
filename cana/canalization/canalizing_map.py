# -*- coding: utf-8 -*-
"""
Build Canalizing Map (CM) graphs from two-symbol schemata.

The legacy builder reproduces the historical ``BooleanNode.canalizing_map``
wiring.  Callers may select a builder by name via :data:`CM_BUILDERS` or pass
a callable with the same signature as :func:`build_canalizing_map_legacy`.
"""
from __future__ import annotations

from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import networkx as nx

# two_symbols entry: (schema_str, permutation_groups, same_symbol_groups)
TwoSymbols = Tuple[Sequence, Sequence]

BuilderFn = Callable[..., nx.DiGraph]


def build_canalizing_map_legacy(
    two_symbols: TwoSymbols,
    *,
    node_id: int,
    inputs: Sequence[int],
    name: str,
    constant: bool = False,
    output: Optional[int] = None,
    input_labels: Optional[Mapping[int, str]] = None,
    k: Optional[int] = None,
) -> nx.DiGraph:
    """Build a CM graph from two-symbol schemata (legacy CANA wiring).

    Parameters
    ----------
    two_symbols : tuple
        ``(ts0_list, ts1_list)`` as produced by two-symbol canalization.
        Each entry is ``(schema_str, permutation_groups, same_symbol_groups)``.
    node_id : int
        Variable id for output nodes ``var-{node_id}-out-{0|1}``.
    inputs : sequence of int
        Input variable ids in LUT order (length ``k``).
    name : str
        Output variable label.
    constant : bool, optional
        Stored on output variable nodes.
    output : int or None, optional
        If ``0`` or ``1``, only that polarity is built; ``None`` builds both.
    input_labels : mapping, optional
        Map from input variable id to display label.  Defaults to ``str(id)``.
    k : int, optional
        Number of inputs.  Defaults to ``len(inputs)``.

    Returns
    -------
    networkx.DiGraph
        Canalizing map graph.
    """
    if k is None:
        k = len(inputs)
    labels = input_labels if input_labels is not None else {}

    G = nx.DiGraph(name="CM: %s" % name)

    if output is None or output == 0:
        G.add_node(
            "var-{nid:d}-out-0".format(nid=node_id),
            **{
                "label": name,
                "type": "variable",
                "mode": "output",
                "value": 0,
                "constant": constant,
                "group": node_id,
            }
        )

    if output is None or output == 1:
        G.add_node(
            "var-{nid:d}-out-1".format(nid=node_id),
            **{
                "label": name,
                "type": "variable",
                "mode": "output",
                "value": 1,
                "constant": constant,
                "group": node_id,
            }
        )

    tid = 0
    for out, tspsss in zip([0, 1], two_symbols):
        if (not len(tspsss)) or ((output is not None) and (output != out)):
            continue

        for ts, ps, ss in tspsss:
            lits = []
            for j in range(k):
                if not any([j in group for group in ps]):
                    if ts[j] in ["0", "1"]:
                        lits.append(j)

            fibre_specs = []
            for g_idx, group in enumerate(ps):
                for polarity in (0, 1):
                    n = sum(1 for j in group if ts[j] == str(polarity))
                    if n > 0:
                        fibre_specs.append((g_idx, group, polarity, n))

            tau = len(lits) + len(fibre_specs)

            tname = "thr-{tid:d}-var-{nid:d}-out-{out:d}".format(
                tid=tid, nid=node_id, out=out
            )
            label = "{:d}".format(tau)
            G.add_node(
                tname,
                **{
                    "label": label,
                    "type": "threshold",
                    "tau": tau,
                    "group": node_id,
                }
            )

            xname = "var-{nid:d}-out-{out:d}".format(nid=node_id, out=out)
            if G.has_node(tname) and G.has_node(xname):
                G.add_edge(tname, xname, **{"type": "out"})
            else:
                raise BaseException("Adding edge to node that does not exist.")

            for lit in lits:
                in_id = inputs[lit]
                iname = "var-{nid:d}-out-{out:d}".format(
                    nid=in_id, out=int(ts[lit])
                )
                if iname not in G.nodes():
                    ilabel = labels.get(in_id, str(in_id))
                    iout = int(ts[lit])
                    G.add_node(
                        iname,
                        **{
                            "label": ilabel,
                            "type": "variable",
                            "mode": "input",
                            "value": iout,
                            "group": node_id,
                        }
                    )
                G.add_edge(iname, tname, **{"type": "literal"})

            for g_idx, group, polarity, n in fibre_specs:
                if n == 1:
                    fname = (
                        "fus-0-grp-{g:d}-thr-{thr:d}"
                        "-var-{nid:d}-out-{out:d}"
                    ).format(g=g_idx, thr=tid, nid=node_id, out=polarity)
                    G.add_node(fname, **{"type": "fusion", "group": node_id})
                    in_edge_type = "fusing"
                else:
                    fname = (
                        "thr-{tid:d}-grp-{g:d}-pol-{p:d}"
                        "-var-{nid:d}-out-{out:d}"
                    ).format(
                        tid=tid,
                        g=g_idx,
                        p=polarity,
                        nid=node_id,
                        out=out,
                    )
                    G.add_node(
                        fname,
                        **{
                            "label": "{:d}".format(n),
                            "type": "threshold",
                            "tau": n,
                            "group": node_id,
                        }
                    )
                    in_edge_type = "literal"

                for j in group:
                    in_id = inputs[j]
                    iname = "var-{nid:d}-out-{out:d}".format(
                        nid=in_id, out=polarity
                    )
                    if iname not in G.nodes():
                        ilabel = labels.get(in_id, str(in_id))
                        G.add_node(
                            iname,
                            **{
                                "label": ilabel,
                                "type": "variable",
                                "mode": "input",
                                "value": polarity,
                                "group": node_id,
                            }
                        )
                    G.add_edge(iname, fname, **{"type": in_edge_type})
                G.add_edge(fname, tname, **{"type": "fused"})

            tid += 1

    return G


CM_BUILDERS: Dict[str, BuilderFn] = {
    "legacy": build_canalizing_map_legacy,
}
"""Named CM builders.  Register future methods here (e.g. ``canonical``)."""


def build_canalizing_map(
    two_symbols: TwoSymbols,
    *,
    node_id: int,
    inputs: Sequence[int],
    name: str,
    constant: bool = False,
    output: Optional[int] = None,
    input_labels: Optional[Mapping[int, str]] = None,
    k: Optional[int] = None,
    builder: Union[str, BuilderFn] = "legacy",
) -> nx.DiGraph:
    """Build a canalizing map using a named or callable builder.

    Parameters
    ----------
    builder : str or callable, optional
        Registry key in :data:`CM_BUILDERS` (default ``'legacy'``) or a
        callable with the same signature as :func:`build_canalizing_map_legacy`.

    Other parameters
    ----------------
    Same as :func:`build_canalizing_map_legacy`.

    Returns
    -------
    networkx.DiGraph
        Canalizing map graph.

    Raises
    ------
    KeyError
        If ``builder`` is a string not present in :data:`CM_BUILDERS`.
    TypeError
        If ``builder`` is neither a string nor a callable.
    """
    if isinstance(builder, str):
        try:
            fn = CM_BUILDERS[builder]
        except KeyError:
            raise KeyError(
                "Unknown CM builder {!r}; known: {}".format(
                    builder, sorted(CM_BUILDERS)
                )
            )
    elif callable(builder):
        fn = builder
    else:
        raise TypeError(
            "builder must be a str or callable, got {!r}".format(type(builder))
        )

    return fn(
        two_symbols,
        node_id=node_id,
        inputs=inputs,
        name=name,
        constant=constant,
        output=output,
        input_labels=input_labels,
        k=k,
    )
