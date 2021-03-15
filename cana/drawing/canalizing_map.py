# -*- coding: utf-8 -*-
"""
Drawing the Canalizing Map (CM)
================================

Methods to draw the Canalizing Map.

"""
#   Copyright (C) 2021 by
#   Rion Brattig Correia <rionbr@gmail.com>
#   Alex Gates <ajgates@indiana.edu>
#   All rights reserved.
#   MIT license.
import warnings
try:
    import graphviz
except ImportError as error:
    warnings.warn("'Graphviz' could not be loaded, you won't be able to plot graphs. Try installing it first. {error:s}".format(error=error))


def draw_canalizing_map_graphviz(DG=None,
                                 on_node_fillcolor='black', on_node_fontcolor='white',
                                 off_node_fillcolor='white', off_node_fontcolor='black',

                                 input_node_bordercolor='gray',
                                 output_node_bordercolor='red',
                                 controlled_node_bordercolor='green',
                                 constant_node_bordercolor='pink',

                                 literal_edge_color='#208120', output_edge_color='#812020',
                                 fusing_edge_color='#a5a5cc', fused_edge_color='#202081',
                                 simplified_edge_color='#cca37a',

                                 fusing_edge_arrowhead='none',
                                 fused_edge_arrowhead='dot',
                                 literal_edge_arrowhead='dot',
                                 out_edge_arrowhead='normal',

                                 simplified_edge_arrowhead='normal',
                                 *args, **kwargs):
    """ Draws the Canalizing Map (CM) using the GraphViz plotting engine.

    Args:
        DG (networkx.DiGraph) : The node Canalizing Map (CM).

    Returns:
        (graphviz) : The network in graphviz dot format.
    """
    G = graphviz.Digraph(engine='neato')
    G.graph_attr.update(overlap='false')
    G.node_attr.update(fontname='helvetica', shape='circle', fontcolor='black', fontsize='12', width='.4', fixedsize='true', style='filled', color='gray', penwidth='3')
    G.edge_attr.update(arrowhead='dot', color='gray', arrowsize='1')

    # Nodes
    for n, d in DG.nodes(data=True):
        if 'type' not in d:
            raise AttributeError("Node type could not be found. Must be either 'variable', 'threshold' or 'fusion'.")

        # Variable Nodes
        if d['type'] == 'variable':

            if 'mode' in d:
                # Border Color. Dependents if 'input', 'output', 'controlled' or 'constant'
                if d['mode'] == 'input':
                    mode_bordercolor = input_node_bordercolor
                elif d['mode'] == 'output':
                    mode_bordercolor = output_node_bordercolor
                elif d['mode'] == 'constant':
                    mode_bordercolor = constant_node_bordercolor
                elif d['mode'] == 'controled':
                    mode_bordercolor = controlled_node_bordercolor
            else:
                mode_bordercolor = output_node_bordercolor

            if d['value'] == 0:
                label = d.get('label', d.get('label-tmp', 'None'))
                G.node(name=n, label=label, fontcolor=off_node_fontcolor, fillcolor=off_node_fillcolor, color=mode_bordercolor)
            elif d['value'] == 1:
                label = d.get('label', d.get('label-tmp', 'None'))
                G.node(name=n, label=label, fontcolor=on_node_fontcolor, fillcolor=on_node_fillcolor, color=mode_bordercolor)

        # Threshold Nodes
        elif d['type'] == 'threshold':
            G.node(name=n, label=d['label'], shape='diamond', style='filled,solid', fillcolor='#dae8f4', fontcolor='black', color='#b5d1e9', width='.4', height='.4')

        elif d['type'] == 'fusion':
            G.node(name=n, label='', shape='none', width='0', height='0', margin='0')

    # Edges
    for s, t, d in DG.edges(data=True):
        color = literal_edge_color
        arrowhead = out_edge_arrowhead

        if 'type' in d:
            if d['type'] == 'out':
                arrowhead = out_edge_arrowhead
                color = output_edge_color

            elif d['type'] == 'literal':
                arrowhead = literal_edge_arrowhead
                color = literal_edge_color

            elif d['type'] == 'fusing':
                arrowhead = fusing_edge_arrowhead
                color = fusing_edge_color

            elif d['type'] == 'fused':
                arrowhead = fused_edge_arrowhead
                color = fused_edge_color

            elif d['type'] == 'simplified':
                if d['mode'] == 'selfloop':
                    arrowhead = simplified_edge_arrowhead
                    color = simplified_edge_color
                elif d['mode'] == 'direct':
                    arrowhead = simplified_edge_arrowhead
                    color = simplified_edge_color

            else:
                raise AttributeError("Node type could not be found. Must be either 'out', 'literal', 'fusing', 'fused' or 'simplified'.  Got {:s}.".format(d['type']))
        G.edge(s, t, arrowhead=arrowhead, color=color)

    return G
