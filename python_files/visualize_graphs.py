"""
Visualization of all 5 base graphs.

Generates a single PNG with 5 subplots (one per graph) showing:
  - Nodes colored by role: sources (blue), sinks (red), intermediates (gray)
  - Edges as solid lines
  - Sessions as dashed green arrows
  - Optimal partition groups highlighted with colored backgrounds
  - Graph name, optimal bound, and optimal partition printed

Usage:
    python visualize_graphs.py
    -> saves graph_visualization.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from fixed_graph_generation import get_all_graph_infos


# Partition group colors (pastel backgrounds)
GROUP_COLORS = [
    '#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA',
    '#E8BAFF', '#FFD9BA', '#C9FFBA', '#FFBAE8',
]


def draw_graph(ax, info, show_partition=True):
    """Draw a single graph on the given axes."""
    G = nx.Graph()
    G.add_nodes_from(info.nodes)
    G.add_edges_from(info.edges)

    # Identify node roles
    sources = set()
    sinks   = set()
    for s, t in info.sessions:
        sources.add(s)
        sinks.add(t)
    intermediates = set(info.nodes) - sources - sinks

    # Layout
    if 'grid' in info.name:
        # Use grid layout for grid graph
        pos = {
            'a': (0, 2), 'b': (1, 2), 'c': (2, 2),
            'd': (0, 1), 'e': (1, 1), 'f': (2, 1),
            'g': (0, 0), 'h': (1, 0), 'i': (2, 0),
        }
    else:
        pos = nx.spring_layout(G, seed=42, k=2.0)

    # Draw partition backgrounds if requested
    if show_partition and info.optimal_partition:
        for gi, group in enumerate(info.optimal_partition):
            group_pos = np.array([pos[n] for n in group if n in pos])
            if len(group_pos) == 0:
                continue
            color = GROUP_COLORS[gi % len(GROUP_COLORS)]
            if len(group_pos) == 1:
                circle = plt.Circle(group_pos[0], 0.15, color=color,
                                    alpha=0.4, zorder=0)
                ax.add_patch(circle)
            else:
                center = group_pos.mean(axis=0)
                radius = max(np.max(np.linalg.norm(group_pos - center, axis=1)) + 0.15, 0.2)
                circle = plt.Circle(center, radius, color=color,
                                    alpha=0.3, zorder=0)
                ax.add_patch(circle)

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#888888',
                           width=1.5, alpha=0.7)

    # Draw session arrows
    for s, t in info.sessions:
        ax.annotate("",
                     xy=pos[t], xytext=pos[s],
                     arrowprops=dict(arrowstyle='->', color='green',
                                     lw=2.0, ls='--',
                                     connectionstyle='arc3,rad=0.15'))

    # Draw nodes by role
    def draw_nodes(nodelist, color, label):
        if nodelist:
            nx.draw_networkx_nodes(G, pos, nodelist=nodelist, ax=ax,
                                   node_color=color, node_size=500,
                                   edgecolors='black', linewidths=1.5)

    draw_nodes(list(sources), '#4A90D9', 'Source')
    draw_nodes(list(sinks), '#D94A4A', 'Sink')
    draw_nodes(list(intermediates), '#AAAAAA', 'Intermediate')

    # Labels
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8,
                            font_weight='bold', font_color='white')

    # Title with info
    sessions_str = ', '.join(f'{s}->{t}' for s, t in info.sessions)
    title = (f"{info.name}\n"
             f"{len(info.nodes)}N {len(info.edges)}E | "
             f"Optimal: {info.optimal_bound:.4f} "
             f"(int={info.optimal_internal})")
    ax.set_title(title, fontsize=10, fontweight='bold', pad=10)

    # Partition annotation below
    if info.optimal_partition:
        parts = []
        for gi, group in enumerate(info.optimal_partition):
            internal = [f"{s}->{t}" for s, t in info.sessions
                        if s in set(group) and t in set(group)]
            gstr = '{' + ','.join(sorted(group)) + '}'
            if internal:
                gstr += f"[{','.join(internal)}]"
            parts.append(gstr)
        part_text = '  '.join(parts)
        ax.text(0.5, -0.05, f"Opt: {part_text}", transform=ax.transAxes,
                fontsize=7, ha='center', va='top', style='italic',
                color='#444444')

    ax.set_aspect('equal')
    ax.axis('off')


def main():
    infos = get_all_graph_infos()

    fig, axes = plt.subplots(1, 5, figsize=(28, 6))
    fig.suptitle('Network Coding Graphs — Optimal Partition Bounds',
                 fontsize=14, fontweight='bold', y=1.02)

    for ax, info in zip(axes, infos):
        draw_graph(ax, info, show_partition=True)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#4A90D9', edgecolor='black', label='Source'),
        mpatches.Patch(facecolor='#D94A4A', edgecolor='black', label='Sink'),
        mpatches.Patch(facecolor='#AAAAAA', edgecolor='black', label='Intermediate'),
        plt.Line2D([0], [0], color='green', lw=2, ls='--', label='Session'),
        mpatches.Patch(facecolor='#FFB3BA', alpha=0.4, label='Partition group'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.savefig('graph_visualization.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    print("Saved: graph_visualization.png")

    # Also print table
    print("\nGraph Summary Table:")
    print(f"{'Name':<16} {'Nodes':>5} {'Edges':>5} {'Sess':>5} "
          f"{'Trivial':>8} {'Optimal':>8} {'OptInt':>6}")
    print("-" * 65)
    for info in infos:
        trivial = len(info.edges) / len(info.sessions)
        print(f"{info.name:<16} {len(info.nodes):>5} {len(info.edges):>5} "
              f"{len(info.sessions):>5} {trivial:>8.4f} "
              f"{info.optimal_bound:>8.4f} {info.optimal_internal:>6}")


if __name__ == "__main__":
    main()