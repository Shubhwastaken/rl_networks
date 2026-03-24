"""
Training visualization — generates 7 plots from training_metrics.json.

Run after training:
    python fixed_training.py   # produces training_metrics.json
    python plot_training.py    # produces training_plots.png

Plots:
  1. Reward curve per stage (convergence)
  2. Internal sessions over training (Phase 1 learning)
  3. Per-graph optimal rate over time (rolling window)
  4. Episode length over training (reward hacking check)
  5. Action distribution per stage (strategy analysis)
  6. Per-graph bound gap boxplot (evaluation quality)
  7. Optimal hit rate bar chart (evaluation summary)
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

# Graph colors — consistent across all plots
GRAPH_COLORS = {
    'paper_7N':      '#1f77b4',
    'diamond_6N':    '#ff7f0e',
    'butterfly_8N':  '#2ca02c',
    'grid_9N':       '#d62728',
    'star_8N':       '#9467bd',
}
STAGE_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']


def rolling_mean(data, window=50):
    """Compute rolling mean with edge handling."""
    if len(data) < window:
        window = max(1, len(data) // 3)
    out = np.convolve(data, np.ones(window)/window, mode='valid')
    # Pad front to match length
    pad = np.full(len(data) - len(out), np.nan)
    return np.concatenate([pad, out])


def rolling_rate(binary_data, window=50):
    """Rolling hit rate for binary (0/1) data."""
    if len(binary_data) < window:
        window = max(1, len(binary_data) // 3)
    out = np.convolve(binary_data, np.ones(window)/window, mode='valid')
    pad = np.full(len(binary_data) - len(out), np.nan)
    return np.concatenate([pad, out]) * 100  # percentage


def load_metrics(path='training_metrics.json'):
    with open(path, 'r') as f:
        return json.load(f)


# -----------------------------------------------------------------------
# Plot 1: Reward curve per stage
# -----------------------------------------------------------------------
def plot_reward_curves(fig, axes, metrics):
    for idx, (stage, label, color) in enumerate([
        ('stage1', 'Stage 1 (Phase 2)', STAGE_COLORS[0]),
        ('stage2', 'Stage 2 (Phase 1)', STAGE_COLORS[1]),
        ('stage3', 'Stage 3 (Joint)',   STAGE_COLORS[2]),
    ]):
        ax = axes[idx]
        data = metrics[stage]['rewards']
        if not data:
            continue
        eps = np.arange(1, len(data) + 1)
        ax.plot(eps, data, alpha=0.15, color=color, linewidth=0.5)
        rm = rolling_mean(data)
        ax.plot(eps, rm, color=color, linewidth=2, label=f'Rolling avg')
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)


# -----------------------------------------------------------------------
# Plot 2: Internal sessions over training (Stages 2 & 3)
# -----------------------------------------------------------------------
def plot_internal_sessions(ax, metrics):
    # Stage 2
    s2 = metrics['stage2'].get('internals', [])
    # Stage 3
    s3 = metrics['stage3'].get('internals', [])

    combined = s2 + s3
    if not combined:
        return

    eps = np.arange(1, len(combined) + 1)
    ax.scatter(eps, combined, alpha=0.1, s=5, color='#2ca02c')
    rm = rolling_mean(combined)
    ax.plot(eps, rm, color='#2ca02c', linewidth=2, label='Rolling avg')

    # Mark stage boundary
    if s2:
        ax.axvline(x=len(s2), color='gray', linestyle='--', alpha=0.5,
                    label='Stage 2→3 boundary')

    ax.set_title('Internal Sessions Over Training', fontsize=10,
                  fontweight='bold')
    ax.set_xlabel('Episode (Stage 2 + Stage 3)')
    ax.set_ylabel('Internal Sessions Found')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


# -----------------------------------------------------------------------
# Plot 3: Per-graph optimal rate over time (Stage 3)
# -----------------------------------------------------------------------
def plot_per_graph_optimal_rate(ax, metrics):
    s3 = metrics['stage3']
    names  = s3.get('graph_names', [])
    hits   = s3.get('optimal_int_hit', [])

    if not names or not hits:
        return

    # Group by graph, maintaining order
    per_graph_hits = defaultdict(list)
    for name, hit in zip(names, hits):
        per_graph_hits[name].append(hit)

    # For each graph, compute rolling rate in its own subsequence
    for gname in sorted(per_graph_hits.keys()):
        gdata = per_graph_hits[gname]
        if len(gdata) < 5:
            continue
        window = max(10, len(gdata) // 10)
        rate = rolling_rate(gdata, window)
        eps = np.arange(1, len(rate) + 1)
        color = GRAPH_COLORS.get(gname, 'gray')
        ax.plot(eps, rate, color=color, linewidth=1.5, label=gname)

    ax.set_title('Per-Graph Optimal Partition Rate (Stage 3)', fontsize=10,
                  fontweight='bold')
    ax.set_xlabel('Episode (per graph)')
    ax.set_ylabel('Optimal Rate (%)')
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)


# -----------------------------------------------------------------------
# Plot 4: Episode length (Phase 2 steps) over training
# -----------------------------------------------------------------------
def plot_episode_length(ax, metrics):
    # Stage 1 has step counts
    s1_steps = metrics['stage1'].get('step_counts', [])
    s3_steps = metrics['stage3'].get('step_counts', [])

    combined = s1_steps + s3_steps
    if not combined:
        return

    eps = np.arange(1, len(combined) + 1)
    ax.scatter(eps, combined, alpha=0.1, s=5, color='#d62728')
    rm = rolling_mean(combined)
    ax.plot(eps, rm, color='#d62728', linewidth=2, label='Rolling avg')

    if s1_steps:
        ax.axvline(x=len(s1_steps), color='gray', linestyle='--', alpha=0.5,
                    label='Stage 1→3 boundary')

    ax.set_title('Phase 2 Episode Length', fontsize=10, fontweight='bold')
    ax.set_xlabel('Episode (Stage 1 + Stage 3)')
    ax.set_ylabel('Steps per Episode')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


# -----------------------------------------------------------------------
# Plot 5: Action distribution per stage
# -----------------------------------------------------------------------
def plot_action_distribution(ax, metrics):
    # Count total actions across all episodes per stage
    stage_actions = {}

    for stage_key, label in [('stage1', 'Stage 1'), ('stage3', 'Stage 3')]:
        counts = defaultdict(int)
        for ep_counts in metrics[stage_key].get('action_counts_per_ep', []):
            for action, count in ep_counts.items():
                counts[action] += count
        stage_actions[label] = dict(counts)

    if not stage_actions:
        return

    # Collect all action types
    all_actions = sorted(set(
        a for counts in stage_actions.values() for a in counts
    ))

    short_names = {
        "ADD_TO_ACCUMULATOR": "ADD", "APPLY_SUBMODULARITY": "SUB",
        "APPLY_PROOF2": "P2", "STORE_AND_RESET": "STO",
        "COMBINE_STORED": "CMB", "DECLARE_TERMINAL": "TRM"
    }

    x = np.arange(len(stage_actions))
    width = 0.12
    action_colors = plt.cm.Set2(np.linspace(0, 1, len(all_actions)))

    for i, action in enumerate(all_actions):
        vals = []
        for label in stage_actions:
            total = sum(stage_actions[label].values()) or 1
            vals.append(100 * stage_actions[label].get(action, 0) / total)
        short = short_names.get(action, action[:3])
        ax.bar(x + i * width, vals, width, label=short,
               color=action_colors[i])

    ax.set_title('Action Distribution (%)', fontsize=10, fontweight='bold')
    ax.set_xticks(x + width * len(all_actions) / 2)
    ax.set_xticklabels(list(stage_actions.keys()))
    ax.set_ylabel('Fraction (%)')
    ax.legend(fontsize=7, ncol=3, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')


# -----------------------------------------------------------------------
# Plot 6: Per-graph bound gap boxplot (evaluation)
# -----------------------------------------------------------------------
def plot_eval_gap_boxplot(ax, metrics):
    eval_data = metrics.get('eval', {})
    names = eval_data.get('graph_names', [])
    gaps  = eval_data.get('gaps', [])

    if not names or not gaps:
        return

    per_graph_gaps = defaultdict(list)
    for name, gap in zip(names, gaps):
        per_graph_gaps[name].append(gap)

    sorted_names = sorted(per_graph_gaps.keys())
    box_data = [per_graph_gaps[n] for n in sorted_names]
    colors = [GRAPH_COLORS.get(n, 'gray') for n in sorted_names]

    bp = ax.boxplot(box_data, tick_labels=sorted_names, patch_artist=True,
                     widths=0.5, showfliers=True,
                     flierprops=dict(marker='.', markersize=3, alpha=0.3))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Optimal')
    ax.set_title('Bound Gap Distribution (Eval)', fontsize=10,
                  fontweight='bold')
    ax.set_ylabel('RL Bound − Optimal')
    ax.tick_params(axis='x', rotation=30)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')


# -----------------------------------------------------------------------
# Plot 7: Optimal hit rate bar chart (evaluation)
# -----------------------------------------------------------------------
def plot_eval_optimal_rate(ax, metrics):
    eval_data = metrics.get('eval', {})
    names      = eval_data.get('graph_names', [])
    rl_bounds  = eval_data.get('rl_bounds', [])
    opt_bounds = eval_data.get('opt_bounds', [])

    if not names:
        return

    per_graph_hits = defaultdict(lambda: {'total': 0, 'optimal': 0})
    for name, rl_b, opt_b in zip(names, rl_bounds, opt_bounds):
        per_graph_hits[name]['total'] += 1
        if abs(rl_b - opt_b) < 0.05:
            per_graph_hits[name]['optimal'] += 1

    sorted_names = sorted(per_graph_hits.keys())
    rates  = [100 * per_graph_hits[n]['optimal'] / max(per_graph_hits[n]['total'], 1)
              for n in sorted_names]
    colors = [GRAPH_COLORS.get(n, 'gray') for n in sorted_names]

    bars = ax.bar(sorted_names, rates, color=colors, alpha=0.7, edgecolor='black')

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', fontsize=8, fontweight='bold')

    ax.set_title('Optimal Bound Hit Rate (Eval)', fontsize=10,
                  fontweight='bold')
    ax.set_ylabel('Hit Rate (%)')
    ax.set_ylim(0, max(max(rates) * 1.2, 10) if rates else 10)
    ax.tick_params(axis='x', rotation=30)
    ax.grid(True, alpha=0.3, axis='y')


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    metrics = load_metrics()

    fig = plt.figure(figsize=(22, 16))
    fig.suptitle('RL Partition Bound Training Report', fontsize=16,
                  fontweight='bold', y=0.98)

    # Layout: 3 rows
    # Row 1: 3 reward curves (one per stage)
    # Row 2: internal sessions | per-graph optimal rate | episode length
    # Row 3: action distribution | eval gap boxplot | eval optimal rate

    # Row 1
    ax1a = fig.add_subplot(3, 3, 1)
    ax1b = fig.add_subplot(3, 3, 2)
    ax1c = fig.add_subplot(3, 3, 3)
    plot_reward_curves(fig, [ax1a, ax1b, ax1c], metrics)

    # Row 2
    ax2a = fig.add_subplot(3, 3, 4)
    plot_internal_sessions(ax2a, metrics)

    ax2b = fig.add_subplot(3, 3, 5)
    plot_per_graph_optimal_rate(ax2b, metrics)

    ax2c = fig.add_subplot(3, 3, 6)
    plot_episode_length(ax2c, metrics)

    # Row 3
    ax3a = fig.add_subplot(3, 3, 7)
    plot_action_distribution(ax3a, metrics)

    ax3b = fig.add_subplot(3, 3, 8)
    plot_eval_gap_boxplot(ax3b, metrics)

    ax3c = fig.add_subplot(3, 3, 9)
    plot_eval_optimal_rate(ax3c, metrics)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('training_plots.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    print("Saved: training_plots.png")

    # Also save individual high-res plots for the review paper
    for name, plot_func, args in [
        ('plot_reward_curves', None, None),  # special handling
        ('plot_internal_sessions', plot_internal_sessions, metrics),
        ('plot_per_graph_optimal_rate', plot_per_graph_optimal_rate, metrics),
        ('plot_episode_length', plot_episode_length, metrics),
        ('plot_action_distribution', plot_action_distribution, metrics),
        ('plot_eval_gap_boxplot', plot_eval_gap_boxplot, metrics),
        ('plot_eval_optimal_rate', plot_eval_optimal_rate, metrics),
    ]:
        if plot_func is None:
            # Reward curves need 3 subplots
            fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
            plot_reward_curves(fig2, axes2, metrics)
            fig2.tight_layout()
            fig2.savefig(f'plot_reward_curves.png', dpi=150,
                          bbox_inches='tight', facecolor='white')
            plt.close(fig2)
        else:
            fig2, ax2 = plt.subplots(1, 1, figsize=(7, 5))
            plot_func(ax2, args)
            fig2.tight_layout()
            fig2.savefig(f'{name}.png', dpi=150, bbox_inches='tight',
                          facecolor='white')
            plt.close(fig2)

    print("Saved individual plots: plot_*.png")
    plt.close('all')


if __name__ == "__main__":
    main()