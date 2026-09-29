# -*- coding: utf-8 -*-
"""
Drawing the Effective graph and the Conditional Effective graph.
================================

Methods to draw the Effective graph and the Conditional Effective graph.

"""
#   Copyright (C) 2026 by
#   Yoshiaki Fujita <yfujita@binghamton.edu>
#   All rights reserved.
#   MIT license.

import warnings
import math
import io
import base64

import graphviz

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import IPython
from IPython.display import display, Markdown, HTML

import numpy as np

# ===========================
# GLOBAL VISUALIZATION CONFIG
# ===========================

VIZ_CONFIG = {
    "FONT_SIZE": "8",
    "EDGE_PENWIDTH": "4",
    "NODE_HEIGHT": "0.4",
    "WIDTH": "0.8",
    "NODE_MARGIN": ".05",
    "NODE_OUTLINE_COLOR": "black",
    "NODE_FILLCOLOR": "#edf7ed",
    "INPUT_FILL": "#edf7ed",
    "SINK_FILL": "#edf7ed",
    "NODE_FONTNAME": "Helvetica",
    "NODE_FONTCOLOR": "#000000FF",
    "NODE_FIX_0": "#FF0000FF",
    "NODE_FIX_1": "#0000FFFF",
    'ACTIONABLE': "#000000FF", 
    'CONDITIONED': "#000000FF",
    'REDUNDANT': "#4F4F4F60",
    "GRID_DX": 120.0,  # Increased from 2.0 (standard points)
    "GRID_DY": 80.0,   # Increased from 1.5
}

# ===========================
# Common functions
# ===========================

def compute_grid_layout(SG, custom_order=None):
    nodes = list(SG.nodes())
    # Sort nodes by label to keep the grid order consistent/alphabetical
    sg_label = {n: SG.nodes[n].get('label', str(n)) for n in SG.nodes()}
    nodes.sort(key=lambda n: sg_label[n].lower())

    N_total = len(nodes)
    if N_total == 0:
        return {}, set(), set(), {}

    # 1. Calculate Grid Dimensions
    # cols is the square root (e.g., sqrt(16)=4, sqrt(20)=4.47 -> 5)
    cols = math.ceil(math.sqrt(N_total))
    # rows is determined by total nodes divided by columns
    rows = math.ceil(N_total / cols)

    cfg = VIZ_CONFIG
    
    # 2. Center the grid around (0,0)
    x0 = - (cols - 1) * cfg["GRID_DX"] / 2.0
    y0 = + (rows - 1) * cfg["GRID_DY"] / 2.0

    def xy_of(r, c):
        return (x0 + c * cfg["GRID_DX"], y0 - r * cfg["GRID_DY"])

    positions = {}

    # 3. Fill the grid row-by-row
    for i, nid in enumerate(nodes):
        r = i // cols  # Current row
        c = i % cols   # Current column
        positions[nid] = xy_of(r, c)

    # Maintain compatibility with the rest of your script
    inputs = [n for n in nodes if SG.in_degree(n) == 0]
    sinks = [n for n in nodes if SG.out_degree(n) == 0]
    
    return positions, set(inputs), set(sinks), sg_label

def create_base_graph():

    cfg = VIZ_CONFIG
    g = graphviz.Digraph(engine='neato')

    # ... (graph attributes remain same)

    g.attr(
        'node',
        pin='true',
        shape='box',
        # --- ADD THESE THREE LINES ---
        fixedsize='true',          # Forces Graphviz to honor width/height
        width='0.8',               # Set your desired width in inches
        height=cfg["NODE_HEIGHT"], # This is already 0.4 in your config
        # -----------------------------
        margin=cfg["NODE_MARGIN"],
        color=cfg["NODE_OUTLINE_COLOR"],
        style='filled',
        fillcolor=cfg["NODE_FILLCOLOR"],
        fontname=cfg["NODE_FONTNAME"],
        fontcolor=cfg["NODE_FONTCOLOR"],
        fontsize=cfg["FONT_SIZE"]
    )

    g.attr('edge', arrowhead='normal', arrowsize='.5')

    stitle=""
    g.attr(
        'graph',
        label=f'\n{stitle}',
        # ... other settings ...
        overlap='false',    # Changed from 'true' to 'false'
        splines='true',     # Routes edges as curves around nodes
        sep='+10',          # Adds a small buffer/padding around each node
        esep='+2',          # Buffer specifically for edge routing
    )

    return g

# ===================================================
# Functions for the effective graph visualization
# ===================================================

def get_effective_node_color(nid, EG, norm_out, cmap):

    out_degree = EG.out_degree(nid, weight='weight')

    # --- Zero effective out-degree ---
    if out_degree == 0:
        fill = '#2ca02c'     # dark green fill
        outline = '#98df8a'  # light green border
        return fill, outline

    # --- Positive effective out-degree ---
    rgb = cmap(norm_out(out_degree))
    fill = mpl.colors.rgb2hex(rgb)

    outline = '#ff9896'  # light red border

    return fill, outline

def get_effective_legend_fig(max_outdegree=10):

    # 1. Replicate your colormap logic exactly
    cmap = LinearSegmentedColormap.from_list('custom', ['white', '#d62728'])
    cmap.set_under('#2ca02c')  # Matches your out_degree == 0 color
    
    # 2. Replicate your normalization (vmin must be slightly above 0 for 'under' to trigger)
    norm = mpl.colors.Normalize(vmin=1e-10, vmax=max_outdegree)
    
    fig = plt.figure(figsize=(1.2, 3)) 
    ax = fig.add_axes([0.05, 0.1, 0.25, 0.8])
    
    ticks = [0, 2, 4, 6, 8, 10]
    # Boundaries help define the segments in the bar
    boundaries = np.linspace(0, max_outdegree, 50).tolist()
    
    cb = mpl.colorbar.ColorbarBase(
        ax, cmap=cmap, norm=norm, 
        boundaries=boundaries,
        ticks=ticks, 
        spacing='uniform', 
        orientation='vertical',
        extend='min',  # This creates the pointed 'arrow' at the bottom for the Green color
        format='%.0f'
    )
    
    # Set the label and styling
    cb.set_label('Effective out-degree', fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=8)
    
    # Label the 'under' color specifically if you want
    ax.set_title("Zero = Green", fontsize=7, pad=10, color='#2ca02c')

    return fig

def visualize_effective_graph(bn, sModel='your model', manual_positions=None, threshold=None):
    """Visualize the effective graph (EG) of a Boolean network.
    
        Args:
            bn (cana boolean network object): Target Boolean network model.
            sModel (str, optional): The name of the model for plot titles/metadata.
                Defaults to 'your model'.
            manual_positions (dict, optional): Mapping of node IDs or names to (x,
            y) tuples
                specifying coordinate locations.
            threshold (str, optional): Display threshold value used to compute the Effective graph, EG
            (e.g., 0.0).
    """
    
    # Get (/ compute) effective graph
    if threshold == None:
        if bn._eg == None:
            # Compute effective graph
            EG = bn.effective_graph()
        else:
            # Use computed effective graph
            EG = bn._eg
    else:
        # Compute effective graph with threshold
        EG = bn.effective_graph(threshold=threshold)
    
    # Labels and layout
    sg_label = {n: EG.nodes[n].get('label', str(n)) for n in EG.nodes()}
    positions, inputs_set, sinks_set, _ = compute_grid_layout(EG)

    # Override with manual positions if provided
    if manual_positions:
        new_positions = {}
        for key, value in manual_positions.items():
            if isinstance(key, str):
                nid = next((n for n, lbl in sg_label.items() if lbl == key), None)
                if nid is None:
                    raise ValueError(f"Label '{key}' not found in network nodes.")
            else:
                nid = key
            new_positions[nid] = value
        positions.update(new_positions)

    # Base Graphviz
    pSG = create_base_graph()
    cfg = VIZ_CONFIG
    max_penwidth = float(cfg["EDGE_PENWIDTH"])

    # PRECOMPUTE NORMALIZATION FOR NODE COLORS
    out_vals = [EG.out_degree(n, weight='weight') for n in EG.nodes()]
    vmin, vmax = min(out_vals), max(out_vals)
    norm_out = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = LinearSegmentedColormap.from_list('custom', ['white', '#d62728'])
    cmap.set_under('#2ca02c')  # nodes with zero out-degree

    # Calculate the grid dimensions first to get 'cols'
    N_total = len(EG.nodes())
    cols = math.ceil(math.sqrt(N_total))

    if cols > 4:
        size_multiplier = 0.9 ** (cols - 4)
    else:
        size_multiplier = 1.0

    # Apply the multiplier to the base config
    node_w = float(VIZ_CONFIG["WIDTH"]) * size_multiplier
    node_h = float(VIZ_CONFIG["NODE_HEIGHT"]) * size_multiplier
    base_font = 10 * size_multiplier # Starting point for font

    # =====================
    # NODES Drawing
    # =====================
    for nid in EG.nodes():
        label = sg_label[nid]
        x, y = positions[nid]
        fill, outline = get_effective_node_color(nid, EG, norm_out, cmap)

    # DYNAMIC FONT CALCULATION 
        # Start with a base size (e.g., 12) and reduce it for longer labels
        # This formula shrinks the font as the character count increases
        base_size = 10
        if len(label) > 4:
            # Decrease font size by 1 point for every 2 extra characters
            dynamic_font = max(6, base_size - (len(label) - 4) // 2)
        else:
            dynamic_font = base_size
        
        pSG.node(str(nid),
                 label=label,
                 pos=f"{x:.3f},{y:.3f}!",
                 fillcolor=fill,
                 color=outline,
                 width=str(node_w),    # New Scaled Width
                 height=str(node_h),   # New Scaled Height                 
                 fontsize=str(dynamic_font),
                 fixedsize='true')

    # =====================
    # EDGES (weighted) Drawing
    # =====================
    for uid, vid, data in EG.edges(data=True):
        weight = data.get('weight', 1.0)
        penwidth = max_penwidth * float(weight)
        uid_str, vid_str = str(uid), str(vid)
        
        if uid == vid:
            # Self-loop
            color = '#636363'
            uid_draw = uid_str + ':w'
            vid_draw = vid_str + ':c'
        else:
            # Regular edge
            color = '#636363'
            uid_draw = uid_str  # Let Graphviz handle standard routing naturally
            vid_draw = vid_str

        # FIX: Pass uid_draw and vid_draw so Graphviz sees the ':w' and ':c' ports!
        pSG.edge(uid_draw, vid_draw, penwidth=str(penwidth), color=color)
       
    # =====================
    # Display 
    # =====================
    # Display the Title/Condition
    
    if threshold != None:
        title_str = f"##### Threshold Effective Graph: {sModel}"
        title_str += f"\n**Edge Weight Threshold:** {str(threshold)}"
    else:
        title_str = f"##### Effective Graph: {sModel}"
    
    display(Markdown(title_str))
    
    # Prepare Legend Image
    # Find the actual max out-degree from your data to scale the bar correctly
    current_max = max([EG.out_degree(n, weight='weight') for n in EG.nodes()])
    fig_legend = get_effective_legend_fig(current_max)
    
    buf = io.BytesIO()
    fig_legend.savefig(buf, format='png', bbox_inches='tight', dpi=150, transparent=True)
    plt.close(fig_legend)
    
    legend_base64 = base64.b64encode(buf.getbuffer()).decode("ascii")
    legend_html = f"<img src='data:image/png;base64,{legend_base64}' style='width:100px; vertical-align:top;' />"

    # Prepare Graph Image (SVG is best for Graphviz)
    graph_svg = pSG.pipe(format='svg').decode('utf-8')

    # Final Side-by-Side Layout
    display(HTML(f"""
        <div style="display: flex; align-items: flex-start; justify-content: flex-start; gap: 30px; margin-top: 20px;">
            <div style="flex: 0 1 auto; border: 1px solid #eee; padding: 10px; border-radius: 8px;">
                {graph_svg}
            </div>
            <div style="flex: 0 0 auto;">
                {legend_html}
            </div>
        </div>
    """))

# =================================================================
# Functions for the conditional effective graph visualization
# =================================================================

def get_legend_figure():
    
    # Reduced height (0.8) to keep it compact between graph and title
    fig, ax = plt.subplots(figsize=(12, 0.8)) 

    # Custom color for dark gray
    dark_gray = '#4F4F4F'
    
    handles = [
        Patch(facecolor='white', edgecolor=dark_gray, linewidth=2, label='Conditioned Node (Double Border)'),
        Patch(facecolor='white', edgecolor='red', linewidth=2, label='Node Fixed as 1'),
        Line2D([0], [0], color='red', linestyle='--', dashes=(5, 2), linewidth=2, marker='>', markersize=8, label='Signal from Fixed 1'),
        Patch(facecolor='white', edgecolor='blue', linewidth=2, label='Node Fixed as 0'),
        Line2D([0], [0], color='blue', linestyle='--', dashes=(5, 2), linewidth=2, marker='>', markersize=8, label='Signal from Fixed 0'),
        Patch(facecolor='#C0C0C0', edgecolor='white', linewidth=2, label='Fully Redundant Node'), # Updated to match your hex
        Line2D([0], [0], color='#00000040', linestyle='--', linewidth=2, marker='>', markersize=8, label='Redundant Signal'),
        Line2D([0], [0], color='#000000', linestyle='-', linewidth=2, marker='>', markersize=8, label='Viral Signal')
    ]
    
    ax.legend(handles=handles, loc='upper left', ncol=4, frameon=False, handlelength=3.0, handletextpad=0.5, fontsize=9)
    plt.axis('off')
    return fig

def visualize_conditional_effective_graph(bn, conditioned_nodes, sModel='your model', conditioned_str=None, manual_positions=None, node_attribute_map=None, category_color_map=None):
    """Visualize the conditional effective graph (EG_cn) of a Boolean network.
    
        Args:
            bn (cana boolean network object): Target Boolean network model.
            conditioned_nodes (dict): Mapping of node IDs to their conditioned binary
            states
                used to generate EG_cn (e.g., {5: 0}).
            sModel (str, optional): The name of the model for plot titles/metadata.
                Defaults to 'your model'.
            conditioned_str (str, optional): Human-readable description of the
            conditioning
                states to display on the visualization.
            manual_positions (dict, optional): Mapping of node IDs or names to (x,
            y) tuples
                specifying coordinate locations.
            node_attribute_map (dict, optional): Mapping of node IDs or names to functional
                categories (e.g., {"CycD": "HSPC"}), used for conditional coloring.
            category_color_map (dict, optional): Mapping of functional category strings
                to hex color codes (e.g., {"HSPC": "#7486F4a0"}).
        """

    # Compute conditional effective graph
    EG_cn = bn.conditional_effective_graph(conditioned_nodes=conditioned_nodes)

    # Add an attribute which discriminate conditioned nodes from non-conditioned node
    for node in EG_cn.nodes():
        if node in conditioned_nodes:
            EG_cn.nodes[node]['conditioned'] = True
        else:
            EG_cn.nodes[node]['conditioned'] = False

    fully_redundant_nodes = {}
    
    for nid in EG_cn.nodes():
        # 1. Isolate edges that go to OTHER nodes (u != v)
        external_edges = [
            (u, v, d) for u, v, d in EG_cn.out_edges(nid, data=True) 
            if u != v
        ]
        
        # 2. If it never had external edges (Sink Node), it's not 'redundant'
        if len(external_edges) == 0:
            fully_redundant_nodes[nid] = False
            continue
        
        # 3. Check if all EXTERNAL outgoing influence is dead (weight = 0)
        # This ignores whether the self-loop is active or not.
        fully_redundant_nodes[nid] = all(
            d.get('weight', 0.0) == 0 for _, _, d in external_edges
        )

    # Labels and layout
    sg_label = {n: EG_cn.nodes[n].get('label', str(n)) for n in EG_cn.nodes()}
    positions, inputs_set, sinks_set, _ = compute_grid_layout(EG_cn)

    # Override with manual positions if provided
    if manual_positions:
        new_positions = {}
        for key, value in manual_positions.items():
            # Map label to node id if needed
            if isinstance(key, str):
                nid = next((n for n, lbl in sg_label.items() if lbl == key), None)
                if nid is None:
                    raise ValueError(f"Label '{key}' not found in network nodes.")
            else:
                nid = key
            new_positions[nid] = value
        positions.update(new_positions)

    # Base Graphviz
    pSG = create_base_graph()
    
    cfg = VIZ_CONFIG
    max_penwidth = float(cfg["EDGE_PENWIDTH"])

    # Calculate the grid dimensions first to get 'cols'
    N_total = len(EG_cn.nodes())
    cols = math.ceil(math.sqrt(N_total))

    if cols > 4:
        size_multiplier = 0.9 ** (cols - 4)
    else:
        size_multiplier = 1.0

    # Apply the multiplier to the base config
    node_w = float(VIZ_CONFIG["WIDTH"]) * size_multiplier
    node_h = float(VIZ_CONFIG["NODE_HEIGHT"]) * size_multiplier
    base_font = 10 * size_multiplier # Starting point for font

    
    # =====================
    # Node drawing
    # =====================
    for nid, d in EG_cn.nodes(data=True):
            label_text = sg_label[nid]
            x, y = positions[nid]
            
            # Determine Font
            dynamic_font = max(6, 10 - (len(label_text) - 4) // 2) if len(label_text) > 4 else 10
    
            # Determine Background Color (Fill)
            # Start with default/input/sink
            node_fill = cfg["NODE_FILLCOLOR"]
            if nid in sinks_set:
                node_fill = cfg["SINK_FILL"]
            elif nid in inputs_set:
                node_fill = cfg["INPUT_FILL"]
    
            # Override with category color
            if node_attribute_map and category_color_map:
                attr = node_attribute_map.get(label_text) or node_attribute_map.get(nid)
                if attr in category_color_map:
                    node_fill = category_color_map[attr]
    
            # Apply grey if redundant (Move this up if you prefer category colors over grey)
            if fully_redundant_nodes.get(nid, False):
                node_fill = VIZ_CONFIG["REDUNDANT"]
    
            # Determine Border (Color & Style)
            border_color = "black"
            pen_w = "1"
            periph = "1"
            
            c_state = d.get('conditioned_state', None)
            is_cond = d.get('conditioned', False)
    
            if is_cond:
                periph = "2" # Double border for the source of the condition
                border_color = VIZ_CONFIG["NODE_FIX_0"] if c_state == 0 else VIZ_CONFIG["NODE_FIX_1"]
                pen_w = "1.5"
            elif c_state is not None:
                border_color = VIZ_CONFIG["NODE_FIX_0"] if c_state == 0 else VIZ_CONFIG["NODE_FIX_1"]
                pen_w = "3" # Thick border for nodes affected by the condition
    
            # ONE SINGLE CALL TO DRAW
            pSG.node(str(nid),
                     label=label_text,
                     pos=f"{x:.3f},{y:.3f}!",
                     fillcolor=node_fill,
                     color=border_color,
                     peripheries=periph,
                     penwidth=pen_w,
                     width=str(node_w),
                     height=str(node_h),
                     fontsize=str(dynamic_font),
                     fixedsize='true',
                     style='filled')

    # =====================
    # Edges drawing
    # =====================
    for uid, vid, d in EG_cn.edges(data=True):
        uid_str, vid_str = str(uid), str(vid)
        weight = d.get('weight', 0)
        penwidth = float(max_penwidth) * weight
        source_state = EG_cn.nodes[uid].get('conditioned_state', None)

        DASH_PENWIDTH = "3"

        # Determine resolved edge's attribute
        if weight==0:
            #pSG.edge(uid_str, vid_str, style='dashed', color=VIZ_CONFIG["REDUNDANT"], penwidth=DASH_PENWIDTH)
            color=VIZ_CONFIG["REDUNDANT"]
            style='dashed'
            penwidth=DASH_PENWIDTH
            #continue
        elif source_state == 0:
            #pSG.edge(uid_str, vid_str, style='dashed', color=VIZ_CONFIG["NODE_FIX_0"], penwidth=str(penwidth))
            style='dashed'
            color=VIZ_CONFIG["NODE_FIX_0"]
            penwidth=str(penwidth)
            #continue
        elif source_state == 1:
            #pSG.edge(uid_str, vid_str, style='dashed', color=VIZ_CONFIG["NODE_FIX_1"], penwidth=str(penwidth))
            style='dashed'
            color=VIZ_CONFIG["NODE_FIX_1"]
            penwidth=str(penwidth)
            #continue
        else:
            style='solid'
            color = '#000000'
            penwidth=str(penwidth)
        
        # Set viable edge attribute 
        if uid_str == vid_str: #For self-loop
            #color = '#000000'
            uid_draw = f"{uid}:w"
            vid_draw = f"{vid}:c"
        else:
            #color = '#000000'
            uid_draw = uid_str
            vid_draw = vid_str

        pSG.edge(uid_draw, vid_draw, style=style, penwidth=penwidth, color=color)

    # =====================
    # Display 
    # =====================
    # Display the Title/Condition
    title_str = f"##### Conditional Effective Graph: {sModel}"
    if conditioned_str:
        title_str += f"\n**Condition:** {conditioned_str}"
    
    display(Markdown(title_str))

    # Display the conditional effective graph
    display(pSG)

    # Display the Legend (Matplotlib)
    fig_legend = get_legend_figure()
    display(fig_legend)
    plt.close(fig_legend) # Prevents double-display in some environments
