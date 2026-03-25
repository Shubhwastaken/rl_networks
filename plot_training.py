"""
Training visualization — generates publication-quality plots.

Run after training:
    python fixed_training.py   # produces training_metrics.json
    python plot_training.py    # produces training_plots.png + individual PNGs

6 plots that tell the complete story:

1. BOUND CONVERGENCE (full training timeline)
   - X: episode (all 3 stages concatenated)  
   - Y: bound extracted by RL agent
   - Shows: how the agent's bound quality improves from trivial → optimal
   - Per-graph colored lines with rolling average

2. PHASE 1: INTERNAL SESSION LEARNING CURVE
   - X: episode (Stages 2+3)
   - Y: rolling optimal partition rate (%) per graph
   - Shows: which graphs Phase 1 masters first

3. PHASE 2: EPISODE LENGTH OVER TRAINING
   - X: episode (Stages 1+3)
   - Y: steps per episode
   - Shows: agent learns efficient action sequences (not too short, not runaway)

4. BOUND GAP DISTRIBUTION (evaluation)
   - Per-graph violin/box plot of (RL bound - optimal)
   - Shows: final quality per graph topology

5. OPTIMAL HIT RATE (evaluation)
   - Per-graph bar chart: % of episodes where RL = optimal
   - The bottom-line metric

6. TRAINING REWARD ACROSS ALL STAGES
   - 3 panels side by side
   - Raw reward + smoothed curve per stage
   - Shows: convergence within each stage
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# Consistent colors
GRAPH_COLORS = {
    'paper_7N':      '#2166ac',
    'diamond_6N':    '#d6604d',
    'butterfly_8N':  '#4dac26',
    'grid_9N':       '#b2abd2',
    'star_8N':       '#f4a582',
}
STAGE_COLORS = ['#2166ac', '#d6604d', '#4dac26']


def rolling(data, window=100):
    """Rolling mean with nan padding."""
    if len(data) < window:
        window = max(1, len(data) // 5)
    kernel = np.ones(window) / window
    out = np.convolve(data, kernel, mode='valid')
    pad = np.full(len(data) - len(out), np.nan)
    return np.concatenate([pad, out])


def load_metrics(path='training_metrics.json'):
    with open(path) as f:
        return json.load(f)


def plot_bound_convergence(ax, metrics):
    """Plot 1: Bound quality across entire training (all stages)."""
    # Collect bounds per graph across all stages with step counts
    per_graph = defaultdict(lambda: {'eps': [], 'bounds': []})
    
    global_ep = 0
    for stage_key in ['stage1', 'stage3']:
        s = metrics.get(stage_key, {})
        bounds = s.get('bounds', [])
        names = s.get('graph_names', [])
        for b, g in zip(bounds, names):
            per_graph[g]['eps'].append(global_ep)
            per_graph[g]['bounds'].append(b)
            global_ep += 1

    # Optimal bounds for reference lines
    opt_bounds = {
        'paper_7N': 1.6667, 'diamond_6N': 1.75, 'butterfly_8N': 1.5,
        'grid_9N': 2.0, 'star_8N': 2.25
    }

    for gname in sorted(per_graph.keys()):
        data = per_graph[gname]
        eps = np.array(data['eps'])
        bounds = np.array(data['bounds'])
        color = GRAPH_COLORS.get(gname, 'gray')
        
        ax.scatter(eps, bounds, alpha=0.03, s=2, color=color)
        
        # Rolling mean within this graph's subsequence
        if len(bounds) > 20:
            window = max(20, len(bounds) // 20)
            rm = rolling(bounds, window)
            ax.plot(eps, rm, color=color, linewidth=2, label=gname)

    # Draw optimal lines
    for gname, opt in opt_bounds.items():
        color = GRAPH_COLORS.get(gname, 'gray')
        ax.axhline(y=opt, color=color, linestyle=':', alpha=0.4, linewidth=1)

    ax.set_xlabel('Episode (Stage 1 + Stage 3)', fontsize=11)
    ax.set_ylabel('Bound (lower = better)', fontsize=11)
    ax.set_title('Bound Convergence Across Training', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.2)


def plot_phase1_learning(ax, metrics):
    """Plot 2: Per-graph optimal partition rate over Stage 2+3."""
    per_graph = defaultdict(list)
    
    for stage_key in ['stage2', 'stage3']:
        s = metrics.get(stage_key, {})
        names = s.get('graph_names', [])
        hits = s.get('optimal_int_found', s.get('optimal_int_hit', []))
        for g, h in zip(names, hits):
            per_graph[g].append(h)

    for gname in sorted(per_graph.keys()):
        data = per_graph[gname]
        if len(data) < 10:
            continue
        window = max(20, len(data) // 15)
        rate = rolling(data, window) * 100
        eps = np.arange(len(rate))
        color = GRAPH_COLORS.get(gname, 'gray')
        ax.plot(eps, rate, color=color, linewidth=2, label=gname)

    ax.set_xlabel('Episode (per graph, Stage 2 + Stage 3)', fontsize=11)
    ax.set_ylabel('Optimal Partition Found (%)', fontsize=11)
    ax.set_title('Phase 1: Partition Quality Over Training', fontsize=13, fontweight='bold')
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2)


def plot_episode_length(ax, metrics):
    """Plot 3: Phase 2 steps per episode."""
    all_steps = []
    
    for stage_key in ['stage1', 'stage3']:
        s = metrics.get(stage_key, {})
        all_steps.extend(s.get('step_counts', []))

    if not all_steps:
        return

    eps = np.arange(len(all_steps))
    ax.scatter(eps, all_steps, alpha=0.05, s=3, color='#555555')
    
    rm = rolling(all_steps, max(50, len(all_steps)//30))
    ax.plot(eps, rm, color='#d62728', linewidth=2.5, label='Rolling avg')

    # Mark stage boundary
    s1_len = len(metrics.get('stage1', {}).get('step_counts', []))
    if s1_len > 0 and s1_len < len(all_steps):
        ax.axvline(x=s1_len, color='gray', linestyle='--', alpha=0.5,
                    linewidth=1, label='Stage 1→3')

    ax.set_xlabel('Episode (Stage 1 + Stage 3)', fontsize=11)
    ax.set_ylabel('Phase 2 Steps', fontsize=11)
    ax.set_title('Episode Length (Efficiency)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)


def plot_eval_gap_violin(ax, metrics):
    """Plot 4: Per-graph bound gap distribution at evaluation."""
    ev = metrics.get('eval', {})
    names = ev.get('graph_names', [])
    gaps = ev.get('gaps', [])

    if not names or not gaps:
        return

    per_graph = defaultdict(list)
    for g, gap in zip(names, gaps):
        per_graph[g].append(gap)

    sorted_names = sorted(per_graph.keys())
    data = [per_graph[n] for n in sorted_names]
    colors = [GRAPH_COLORS.get(n, 'gray') for n in sorted_names]

    parts = ax.violinplot(data, positions=range(len(sorted_names)),
                           showmeans=True, showmedians=True)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)
    parts['cmeans'].set_color('black')
    parts['cmedians'].set_color('red')

    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=25, fontsize=9)
    ax.axhline(y=0, color='green', linestyle='--', alpha=0.7, linewidth=1.5,
                label='Optimal (gap=0)')
    ax.set_ylabel('RL Bound − Optimal', fontsize=11)
    ax.set_title('Evaluation: Bound Gap per Graph', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')


def plot_eval_hit_rate(ax, metrics):
    """Plot 5: Per-graph optimal hit rate bar chart."""
    ev = metrics.get('eval', {})
    names = ev.get('graph_names', [])
    rl_bounds = ev.get('rl_bounds', [])
    opt_bounds_list = ev.get('opt_bounds', [])

    if not names:
        return

    stats = defaultdict(lambda: {'total': 0, 'hits': 0})
    for g, rl, opt in zip(names, rl_bounds, opt_bounds_list):
        stats[g]['total'] += 1
        if abs(rl - opt) < 0.05:
            stats[g]['hits'] += 1

    sorted_names = sorted(stats.keys())
    rates = [100 * stats[n]['hits'] / max(stats[n]['total'], 1) for n in sorted_names]
    colors = [GRAPH_COLORS.get(n, 'gray') for n in sorted_names]

    bars = ax.bar(range(len(sorted_names)), rates, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=0.8)

    for i, (bar, rate) in enumerate(zip(bars, rates)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{rate:.1f}%', ha='center', fontsize=10, fontweight='bold')

    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=25, fontsize=9)
    ax.set_ylabel('Optimal Bound Hit Rate (%)', fontsize=11)
    ax.set_title('Evaluation: Optimal Rate per Graph', fontsize=13, fontweight='bold')
    ax.set_ylim(0, min(max(rates) * 1.3, 100) if rates else 10)
    ax.grid(True, alpha=0.2, axis='y')


def plot_stage_rewards(axes, metrics):
    """Plot 6: Reward curves per stage (3 panels)."""
    stage_info = [
        ('stage1', 'Stage 1: Phase 2 Training', STAGE_COLORS[0]),
        ('stage2', 'Stage 2: Phase 1 Training', STAGE_COLORS[1]),
        ('stage3', 'Stage 3: Joint Fine-tuning', STAGE_COLORS[2]),
    ]

    for ax, (key, title, color) in zip(axes, stage_info):
        s = metrics.get(key, {})
        rewards = s.get('rewards', [])
        if not rewards:
            continue

        eps = np.arange(1, len(rewards) + 1)
        ax.plot(eps, rewards, alpha=0.08, color=color, linewidth=0.3)
        
        rm = rolling(rewards, max(30, len(rewards)//25))
        ax.plot(eps, rm, color=color, linewidth=2.5)

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Episode', fontsize=10)
        ax.set_ylabel('Reward', fontsize=10)
        ax.grid(True, alpha=0.2)

        # Add text with final avg
        last_n = max(len(rewards)//10, 10)
        final_avg = np.mean(rewards[-last_n:])
        ax.text(0.98, 0.05, f'Final avg: {final_avg:.3f}',
                transform=ax.transAxes, ha='right', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))


def main():
    metrics = load_metrics()

    # ============================================================
    # Combined dashboard (2 rows x 3 cols)
    # ============================================================
    fig = plt.figure(figsize=(20, 13))
    fig.suptitle('RL Partition Bound — Training Report',
                  fontsize=16, fontweight='bold', y=0.98)

    # Row 1: Convergence story
    ax1 = fig.add_subplot(2, 3, 1)
    plot_bound_convergence(ax1, metrics)

    ax2 = fig.add_subplot(2, 3, 2)
    plot_phase1_learning(ax2, metrics)

    ax3 = fig.add_subplot(2, 3, 3)
    plot_episode_length(ax3, metrics)

    # Row 2: Evaluation + stage rewards
    ax4 = fig.add_subplot(2, 3, 4)
    plot_eval_gap_violin(ax4, metrics)

    ax5 = fig.add_subplot(2, 3, 5)
    plot_eval_hit_rate(ax5, metrics)

    # Stage rewards as inset in position 6
    ax6a = fig.add_subplot(2, 3, 6)
    # Use the single panel for Stage 3 reward (most important)
    s3 = metrics.get('stage3', {})
    rewards = s3.get('rewards', [])
    if rewards:
        eps = np.arange(1, len(rewards) + 1)
        ax6a.plot(eps, rewards, alpha=0.08, color=STAGE_COLORS[2], linewidth=0.3)
        rm = rolling(rewards, max(30, len(rewards)//25))
        ax6a.plot(eps, rm, color=STAGE_COLORS[2], linewidth=2.5)
        ax6a.set_title('Stage 3: Joint Reward Curve', fontsize=11, fontweight='bold')
        ax6a.set_xlabel('Episode', fontsize=10)
        ax6a.set_ylabel('Reward', fontsize=10)
        ax6a.grid(True, alpha=0.2)
        last_n = max(len(rewards)//10, 10)
        ax6a.text(0.98, 0.05, f'Final avg: {np.mean(rewards[-last_n:]):.3f}',
                  transform=ax6a.transAxes, ha='right', fontsize=9,
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('training_plots.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("Saved: training_plots.png")
    plt.close()

    # ============================================================
    # Individual high-res plots for the review
    # ============================================================
    
    # 1. Bound convergence
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_bound_convergence(ax, metrics)
    fig.tight_layout()
    fig.savefig('plot_bound_convergence.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    # 2. Phase 1 learning
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_phase1_learning(ax, metrics)
    fig.tight_layout()
    fig.savefig('plot_phase1_learning.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    # 3. Episode length
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_episode_length(ax, metrics)
    fig.tight_layout()
    fig.savefig('plot_episode_length.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    # 4. Eval gap violin
    fig, ax = plt.subplots(figsize=(9, 6))
    plot_eval_gap_violin(ax, metrics)
    fig.tight_layout()
    fig.savefig('plot_eval_gap.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    # 5. Eval hit rate
    fig, ax = plt.subplots(figsize=(9, 6))
    plot_eval_hit_rate(ax, metrics)
    fig.tight_layout()
    fig.savefig('plot_eval_hitrate.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    # 6. All 3 stage rewards side by side
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_stage_rewards(axes, metrics)
    fig.suptitle('Reward Convergence Per Stage', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig('plot_stage_rewards.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    print("Saved individual plots: plot_*.png")


if __name__ == "__main__":
    main()