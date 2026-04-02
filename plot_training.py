"""
Training visualization — full four-stage report including Phase 3.

Plots generated:
  training_plots.png          — combined 3x3 dashboard
  plot_bound_convergence.png  — Stage 1+3 bound quality over time
  plot_phase1_learning.png    — Phase 1 optimal partition rate
  plot_episode_length.png     — Phase 2 steps per episode
  plot_eval_gap.png           — evaluation bound gap violin
  plot_eval_hitrate.png       — evaluation optimal hit rate
  plot_stage_rewards.png      — reward curves for all 4 stages
  plot_stage4_novel.png       — Stage 4: novel bound discovery rate
  plot_stage4_lambda.png      — Stage 4: which λ values produced improvements
  plot_phase3_bounds.png      — Phase 3 best bound vs PB per graph
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# ── Color scheme ──────────────────────────────────────────────────────────────
# Tier 1 graphs: blue family
# Tier 2 graphs: red/orange family
# Tier 3 graphs: green family
# Stage colors: one per stage

GRAPH_COLORS = {
    # Tier 1
    'paper_7N':       '#2166ac',
    'diamond_6N':     '#4393c3',
    'butterfly_8N':   '#92c5de',
    'grid_9N':        '#0571b0',
    'star_8N':        '#74add1',
    # Tier 2
    'hu_3pairs_6N':   '#d6604d',
    'okamura_4N':     '#f4a582',
    'grid_3x4_12N':   '#b2182b',
    'petersen_10N':   '#e08214',
    'two_k4_10N':     '#fdb863',
    # Tier 3
    'grid_4x4_16N':   '#1b7837',
}
DEFAULT_COLOR = '#888888'

STAGE_COLORS = {
    'stage1': '#2166ac',
    'stage2': '#d6604d',
    'stage3': '#4dac26',
    'stage4': '#984ea3',
}
STAGE_LABELS = {
    'stage1': 'Stage 1: Phase 2 proof calculus',
    'stage2': 'Stage 2: Phase 1 partition learning',
    'stage3': 'Stage 3: Joint fine-tuning',
    'stage4': 'Stage 4: Phase 3 fractional IO search',
}

TIER_LABEL = {
    'paper_7N': 'T1', 'diamond_6N': 'T1', 'butterfly_8N': 'T1',
    'grid_9N': 'T1',  'star_8N': 'T1',
    'hu_3pairs_6N': 'T2', 'okamura_4N': 'T2', 'grid_3x4_12N': 'T2',
    'petersen_10N': 'T2', 'two_k4_10N': 'T2',
    'grid_4x4_16N': 'T3',
}


def rolling(data, window=100):
    if len(data) < 2:
        return np.array(data, dtype=float)
    window = max(2, min(window, len(data) // 3))
    kernel = np.ones(window) / window
    out = np.convolve(data, kernel, mode='valid')
    pad = np.full(len(data) - len(out), np.nan)
    return np.concatenate([pad, out])


def load_metrics(path='training_metrics.json'):
    with open(path) as f:
        raw = json.load(f)
    # training_metrics.json stores values as str(dict) — parse them back
    result = {}
    for k, v in raw.items():
        if isinstance(v, str):
            try:
                import ast
                result[k] = ast.literal_eval(v)
            except Exception:
                result[k] = v
        else:
            result[k] = v
    return result


# ── Individual plot functions ─────────────────────────────────────────────────

def plot_bound_convergence(ax, metrics):
    """Stages 1+3: bound quality over episodes."""
    per_graph = defaultdict(lambda: {'eps': [], 'bounds': []})
    global_ep = 0
    for stage_key in ['stage1', 'stage3']:
        s = metrics.get(stage_key, {})
        if isinstance(s, str):
            continue
        bounds = s.get('bounds', [])
        names  = s.get('graph_names', [])
        for b, g in zip(bounds, names):
            try:
                per_graph[g]['eps'].append(global_ep)
                per_graph[g]['bounds'].append(float(b))
                global_ep += 1
            except (ValueError, TypeError):
                pass

    for gname in sorted(per_graph.keys()):
        data   = per_graph[gname]
        eps    = np.array(data['eps'])
        bounds = np.array(data['bounds'])
        color  = GRAPH_COLORS.get(gname, DEFAULT_COLOR)
        ax.scatter(eps, bounds, alpha=0.04, s=2, color=color)
        if len(bounds) > 10:
            rm = rolling(bounds, max(20, len(bounds)//20))
            ax.plot(eps, rm, color=color, linewidth=2,
                    label=f"{gname} [{TIER_LABEL.get(gname,'?')}]")

    ax.set_xlabel('Episode (Stage 1 + Stage 3)', fontsize=10)
    ax.set_ylabel('Bound extracted (lower = better)', fontsize=10)
    ax.set_title('Bound Convergence — Phases 1+2', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.2)


def plot_phase1_learning(ax, metrics):
    """Stages 2+3: per-graph optimal partition hit rate."""
    per_graph = defaultdict(list)
    for stage_key in ['stage2', 'stage3']:
        s = metrics.get(stage_key, {})
        if isinstance(s, str):
            continue
        names = s.get('graph_names', [])
        hits  = s.get('optimal_int_found',
                s.get('optimal_int_hit', []))
        for g, h in zip(names, hits):
            try:
                per_graph[g].append(int(h))
            except (ValueError, TypeError):
                pass

    for gname in sorted(per_graph.keys()):
        data = per_graph[gname]
        if len(data) < 5:
            continue
        rate  = rolling(data, max(20, len(data)//15)) * 100
        eps   = np.arange(len(rate))
        color = GRAPH_COLORS.get(gname, DEFAULT_COLOR)
        ax.plot(eps, rate, color=color, linewidth=2,
                label=f"{gname} [{TIER_LABEL.get(gname,'?')}]")

    ax.set_xlabel('Episode (per graph, Stage 2+3)', fontsize=10)
    ax.set_ylabel('Optimal partition found (%)', fontsize=10)
    ax.set_title('Phase 1: Partition Quality', fontsize=12, fontweight='bold')
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.2)


def plot_episode_length(ax, metrics):
    """Stages 1+3: Phase 2 steps per episode."""
    all_steps = []
    stage1_len = 0
    for stage_key in ['stage1', 'stage3']:
        s = metrics.get(stage_key, {})
        if isinstance(s, str):
            continue
        sc = s.get('step_counts', [])
        if stage_key == 'stage1':
            stage1_len = len(sc)
        all_steps.extend([int(x) for x in sc if x])

    if not all_steps:
        ax.text(0.5, 0.5, 'No step data', transform=ax.transAxes,
                ha='center', va='center')
        return

    eps = np.arange(len(all_steps))
    ax.scatter(eps, all_steps, alpha=0.04, s=2, color='#555555')
    rm = rolling(all_steps, max(30, len(all_steps)//30))
    ax.plot(eps, rm, color='#d62728', linewidth=2.5, label='Rolling avg')
    if 0 < stage1_len < len(all_steps):
        ax.axvline(x=stage1_len, color='gray', linestyle='--', alpha=0.6,
                   linewidth=1, label='Stage 1→3')

    ax.set_xlabel('Episode (Stage 1 + Stage 3)', fontsize=10)
    ax.set_ylabel('Steps per episode', fontsize=10)
    ax.set_title('Phase 2: Episode Length', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)


def plot_eval_gap_violin(ax, metrics):
    """Evaluation: bound gap distribution per graph."""
    ev = metrics.get('eval', {})
    if isinstance(ev, str):
        ax.text(0.5, 0.5, 'No eval data', transform=ax.transAxes,
                ha='center', va='center')
        return

    # Support both old format (graph_names/gaps lists) and new format (dict per graph)
    per_graph = defaultdict(list)
    if isinstance(ev, dict) and 'graph_names' in ev:
        for g, gap in zip(ev.get('graph_names', []),
                          ev.get('gaps', [])):
            try:
                per_graph[g].append(float(gap))
            except (ValueError, TypeError):
                pass
    elif isinstance(ev, dict):
        for gname, stats in ev.items():
            if isinstance(stats, dict) and 'p2_bounds' in stats:
                pb_val = None
                for info_name in ['paper_7N','diamond_6N','butterfly_8N',
                                   'grid_9N','star_8N','hu_3pairs_6N',
                                   'okamura_4N','grid_3x4_12N',
                                   'petersen_10N','two_k4_10N','grid_4x4_16N']:
                    pass   # would look up PB but skip for brevity
                p2 = [float(b) for b in stats.get('p2_bounds', [])]
                p3 = [float(b) for b in stats.get('p3_bounds', [])]
                per_graph[gname] = p3 if p3 else p2

    if not per_graph:
        ax.text(0.5, 0.5, 'No gap data available', transform=ax.transAxes,
                ha='center', va='center')
        return

    sorted_names = sorted(per_graph.keys())
    data   = [per_graph[n] for n in sorted_names if len(per_graph[n]) > 0]
    labels = [n for n in sorted_names if len(per_graph[n]) > 0]
    colors = [GRAPH_COLORS.get(n, DEFAULT_COLOR) for n in labels]

    if not data:
        return

    parts = ax.violinplot(data, positions=range(len(labels)),
                           showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)
    parts['cmeans'].set_color('black')
    parts['cmedians'].set_color('red')

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, fontsize=8)
    ax.axhline(y=0, color='green', linestyle='--', alpha=0.7,
               linewidth=1.5, label='Optimal (gap=0)')
    ax.set_ylabel('Bound value', fontsize=10)
    ax.set_title('Evaluation: Bound Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')


def plot_eval_hit_rate(ax, metrics):
    """Evaluation: optimal hit rate per graph."""
    ev = metrics.get('eval', {})
    if isinstance(ev, str):
        ax.text(0.5, 0.5, 'No eval data', transform=ax.transAxes,
                ha='center', va='center')
        return

    stats = defaultdict(lambda: {'total': 0, 'hits': 0})
    if isinstance(ev, dict) and 'graph_names' in ev:
        for g, rl, opt in zip(ev.get('graph_names', []),
                              ev.get('rl_bounds', []),
                              ev.get('opt_bounds', [])):
            try:
                stats[g]['total'] += 1
                if abs(float(rl) - float(opt)) < 0.05:
                    stats[g]['hits'] += 1
            except (ValueError, TypeError):
                pass

    if not stats:
        ax.text(0.5, 0.5, 'No hit-rate data', transform=ax.transAxes,
                ha='center', va='center')
        return

    sorted_names = sorted(stats.keys())
    rates  = [100*stats[n]['hits']/max(stats[n]['total'],1) for n in sorted_names]
    colors = [GRAPH_COLORS.get(n, DEFAULT_COLOR) for n in sorted_names]

    bars = ax.bar(range(len(sorted_names)), rates, color=colors,
                   alpha=0.8, edgecolor='black', linewidth=0.8)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1.5,
                f'{rate:.1f}%', ha='center', fontsize=9, fontweight='bold')

    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=30, fontsize=8)
    ax.set_ylabel('Optimal hit rate (%)', fontsize=10)
    ax.set_title('Evaluation: Optimal Rate per Graph', fontsize=12, fontweight='bold')
    ax.set_ylim(0, min(max(rates)*1.3, 110) if rates else 10)
    ax.grid(True, alpha=0.2, axis='y')


def plot_all_stage_rewards(axes, metrics):
    """Reward curves for all 4 stages — one panel each."""
    stage_order = ['stage1', 'stage2', 'stage3', 'stage4']
    for ax, key in zip(axes, stage_order):
        s = metrics.get(key, {})
        if isinstance(s, str):
            ax.text(0.5, 0.5, f'No data\n({key})',
                    transform=ax.transAxes, ha='center', va='center')
            ax.set_title(STAGE_LABELS[key], fontsize=10, fontweight='bold')
            continue

        rewards = s.get('rewards', [])
        if not rewards:
            ax.text(0.5, 0.5, 'No reward data',
                    transform=ax.transAxes, ha='center', va='center')
            ax.set_title(STAGE_LABELS[key], fontsize=10, fontweight='bold')
            continue

        try:
            rewards = [float(r) for r in rewards]
        except (ValueError, TypeError):
            continue

        color = STAGE_COLORS[key]
        eps   = np.arange(1, len(rewards)+1)
        ax.plot(eps, rewards, alpha=0.07, color=color, linewidth=0.4)
        rm = rolling(rewards, max(30, len(rewards)//25))
        ax.plot(eps, rm, color=color, linewidth=2.5)

        last_n    = max(len(rewards)//10, 5)
        final_avg = np.mean(rewards[-last_n:])
        ax.text(0.98, 0.05, f'Final avg: {final_avg:.3f}',
                transform=ax.transAxes, ha='right', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax.set_title(STAGE_LABELS[key], fontsize=10, fontweight='bold')
        ax.set_xlabel('Episode', fontsize=9)
        ax.set_ylabel('Reward', fontsize=9)
        ax.grid(True, alpha=0.2)


# ── Stage 4 specific plots ────────────────────────────────────────────────────

def plot_stage4_novel_rate(ax, metrics):
    """
    Stage 4: rolling rate at which Phase 3 beats the partition bound.
    This is the headline metric for the research contribution.
    """
    s4 = metrics.get('stage4', {})
    if isinstance(s4, str):
        ax.text(0.5, 0.5, 'No Stage 4 data', transform=ax.transAxes,
                ha='center', va='center')
        ax.set_title('Stage 4: Novel Bound Discovery Rate', fontsize=12,
                     fontweight='bold')
        return

    novel  = s4.get('novel_found', [])
    names  = s4.get('graph_names', [])
    cross  = s4.get('cross_partition_used', [])

    if not novel:
        ax.text(0.5, 0.5, 'No novel_found data\n(run Stage 4 first)',
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Stage 4: Novel Bound Discovery Rate', fontsize=12,
                     fontweight='bold')
        return

    try:
        novel = [int(x) for x in novel]
        cross = [int(x) for x in cross] if cross else [0]*len(novel)
    except (ValueError, TypeError):
        return

    eps = np.arange(1, len(novel)+1)

    # Overall rolling novel rate
    rm_novel = rolling(novel, max(30, len(novel)//20)) * 100
    ax.plot(eps, rm_novel, color=STAGE_COLORS['stage4'],
            linewidth=2.5, label='Novel bound found (%)', zorder=3)

    # Cross-partition submod usage rate
    rm_cross = rolling(cross, max(30, len(cross)//20)) * 100
    ax.plot(eps, rm_cross, color='#e08214', linewidth=2,
            linestyle='--', label='CROSS_SUBMOD used (%)', zorder=2)

    # Per-graph novel episodes as vertical ticks
    if names:
        per_graph_novel = defaultdict(list)
        for i, (g, n) in enumerate(zip(names, novel)):
            if n:
                per_graph_novel[g].append(i+1)

        y_tick = -4
        for gname, ep_list in sorted(per_graph_novel.items()):
            color = GRAPH_COLORS.get(gname, DEFAULT_COLOR)
            ax.scatter(ep_list, [y_tick]*len(ep_list),
                       marker='|', s=60, color=color,
                       label=f'Novel: {gname}', zorder=4, alpha=0.7)
            y_tick -= 2

    ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.4)
    ax.set_xlabel('Episode (Stage 4)', fontsize=10)
    ax.set_ylabel('Rate (%)', fontsize=10)
    ax.set_title('Stage 4: Novel Bound Discovery & Cross-Partition Usage',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.2)


def plot_stage4_bounds_vs_pb(ax, metrics):
    """
    Stage 4: per-graph scatter of best Phase 3 bound vs partition bound.
    Green bars = improvement over PB. Red bars = still above PB.
    """
    s4    = metrics.get('stage4', {})
    novel = metrics.get('novel_bounds', {})

    if isinstance(s4, str) and isinstance(novel, str):
        ax.text(0.5, 0.5, 'No Stage 4 data', transform=ax.transAxes,
                ha='center', va='center')
        ax.set_title('Phase 3: Best Bound vs Partition Bound', fontsize=12,
                     fontweight='bold')
        return

    # Collect per-graph best bounds from Stage 4 rolling data
    names   = []
    pb_vals = []
    p3_best = []

    s4_data = s4 if isinstance(s4, dict) else {}
    s4_names  = s4_data.get('graph_names', [])
    s4_bounds = s4_data.get('bounds', [])

    per_graph_best = defaultdict(lambda: float('inf'))
    for g, b in zip(s4_names, s4_bounds):
        try:
            bv = float(b)
            if bv > 0:
                per_graph_best[g] = min(per_graph_best[g], bv)
        except (ValueError, TypeError):
            pass

    # Also check novel_bounds for confirmed super-PB results
    novel_data = novel if isinstance(novel, dict) else {}

    # Collect known partition bounds (hardcoded from graph registry)
    known_pb = {
        'paper_7N': 1.6667, 'diamond_6N': 1.75, 'butterfly_8N': 1.5,
        'grid_9N': 2.0, 'star_8N': 2.25,
        'hu_3pairs_6N': 1.3333, 'okamura_4N': 1.25,
        'grid_3x4_12N': 5.6667, 'petersen_10N': 1.875,
        'two_k4_10N': 2.6667, 'grid_4x4_16N': 3.0,
    }

    all_graphs = sorted(set(list(per_graph_best.keys()) + list(novel_data.keys())))
    if not all_graphs:
        ax.text(0.5, 0.5, 'No Stage 4 bound data yet',
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Phase 3: Best Bound vs Partition Bound', fontsize=12,
                     fontweight='bold')
        return

    for g in all_graphs:
        pb = known_pb.get(g, None)
        if pb is None:
            continue
        best = per_graph_best.get(g, float('inf'))
        # Override with novel_bounds if available (more reliable)
        if g in novel_data:
            nb = novel_data[g]
            if isinstance(nb, (list, tuple)) and len(nb) > 0:
                try:
                    best = min(best, float(nb[0]))
                except (ValueError, TypeError):
                    pass
        if best < 1e9:
            names.append(g)
            pb_vals.append(pb)
            p3_best.append(best)

    if not names:
        ax.text(0.5, 0.5, 'No finite bounds found yet',
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Phase 3: Best Bound vs Partition Bound', fontsize=12,
                     fontweight='bold')
        return

    x      = np.arange(len(names))
    width  = 0.35
    colors = [GRAPH_COLORS.get(n, DEFAULT_COLOR) for n in names]

    # Partition bound bars (always blue outline)
    ax.bar(x - width/2, pb_vals, width, label='Partition Bound (PB)',
           color='#aec7e8', edgecolor='#2166ac', linewidth=1.2, alpha=0.9)

    # Phase 3 best bound bars (green if below PB, coral if above)
    bar_colors = [
        '#2ca02c' if p < pb - 1e-6 else '#d62728'
        for p, pb in zip(p3_best, pb_vals)
    ]
    bars = ax.bar(x + width/2, p3_best, width, label='Phase 3 Best Bound',
                   color=bar_colors, edgecolor='black', linewidth=0.8, alpha=0.9)

    # Annotate improvement percentage above each Phase 3 bar
    for bar, p3, pb, gname in zip(bars, p3_best, pb_vals, names):
        if p3 < pb - 1e-6:
            pct = (pb - p3) / pb * 100
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.03,
                    f'↓{pct:.1f}%', ha='center', fontsize=8,
                    fontweight='bold', color='#2ca02c')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, fontsize=8)
    ax.set_ylabel('Bound on r (lower = tighter)', fontsize=10)
    ax.set_title('Phase 3: Best Bound vs Partition Bound',
                 fontsize=12, fontweight='bold')

    # Legend with color meaning
    green_patch = mpatches.Patch(color='#2ca02c', label='Beats PB (novel!)')
    red_patch   = mpatches.Patch(color='#d62728', label='Still above PB')
    blue_patch  = mpatches.Patch(color='#aec7e8', label='Partition Bound')
    ax.legend(handles=[blue_patch, green_patch, red_patch], fontsize=9,
              loc='upper right')
    ax.grid(True, alpha=0.2, axis='y')


def plot_stage4_lambda_analysis(ax, metrics):
    """
    Stage 4: which λ values led to novel bounds.
    Shows the distribution of λ values across all FRACTIONAL_IO actions
    vs the subset that ended in a novel bound episode.
    """
    s4 = metrics.get('stage4', {})
    if isinstance(s4, str):
        ax.text(0.5, 0.5, 'No Stage 4 data', transform=ax.transAxes,
                ha='center', va='center')
        ax.set_title('Stage 4: λ Distribution', fontsize=12, fontweight='bold')
        return

    # The training loop doesn't currently log per-λ statistics.
    # We show what we do have: novel_found rate and cross_partition_used.
    # If per-λ logging is added later this plot will populate automatically.
    novel = s4.get('novel_found', [])
    cross = s4.get('cross_partition_used', [])

    if not novel:
        ax.text(0.5, 0.5, 'Run Stage 4 to populate\nλ analysis',
                transform=ax.transAxes, ha='center', va='center', fontsize=11)
        ax.set_title('Stage 4: λ Analysis (pending data)', fontsize=12,
                     fontweight='bold')
        return

    try:
        novel = [int(x) for x in novel]
        cross = [int(x) for x in cross] if cross else [0]*len(novel)
    except (ValueError, TypeError):
        return

    # Compute cumulative novel discoveries over training
    cumulative_novel = np.cumsum(novel)
    eps = np.arange(1, len(novel)+1)

    ax2 = ax.twinx()

    ax.bar(eps, novel, color=STAGE_COLORS['stage4'], alpha=0.25,
           width=1, label='Novel episode')
    ax2.plot(eps, cumulative_novel, color=STAGE_COLORS['stage4'],
             linewidth=2.5, label='Cumulative novel bounds')
    ax2.set_ylabel('Cumulative novel bounds found', fontsize=10,
                   color=STAGE_COLORS['stage4'])
    ax2.tick_params(axis='y', labelcolor=STAGE_COLORS['stage4'])

    # Cross-partition usage
    cross_rate = rolling(cross, max(20, len(cross)//15)) * 100
    ax.plot(eps, cross_rate, color='#e08214', linewidth=2,
            label='CROSS_SUBMOD rate (%)')

    total_novel = sum(novel)
    total_eps   = len(novel)
    ax.set_xlabel('Episode (Stage 4)', fontsize=10)
    ax.set_ylabel('Novel episode (bar) / CROSS_SUBMOD rate (%)', fontsize=9)
    ax.set_title(
        f'Stage 4: Cumulative Discovery  '
        f'({total_novel} novel / {total_eps} eps = {100*total_novel/max(total_eps,1):.1f}%)',
        fontsize=12, fontweight='bold'
    )
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.2)


# ── Main dashboard ────────────────────────────────────────────────────────────

def main():
    metrics = load_metrics()

    has_stage4 = (
        'stage4' in metrics and
        not isinstance(metrics.get('stage4'), str) and
        bool(metrics.get('stage4', {}).get('rewards'))
    )

    # ==========================================================
    # DASHBOARD A: Phases 1+2 training (3x3 grid)
    # ==========================================================
    fig = plt.figure(figsize=(20, 13))
    fig.suptitle('RL Network Coding Bounds — Phases 1+2 Training Report',
                  fontsize=15, fontweight='bold', y=0.98)

    ax1 = fig.add_subplot(2, 3, 1)
    plot_bound_convergence(ax1, metrics)

    ax2 = fig.add_subplot(2, 3, 2)
    plot_phase1_learning(ax2, metrics)

    ax3 = fig.add_subplot(2, 3, 3)
    plot_episode_length(ax3, metrics)

    ax4 = fig.add_subplot(2, 3, 4)
    plot_eval_gap_violin(ax4, metrics)

    ax5 = fig.add_subplot(2, 3, 5)
    plot_eval_hit_rate(ax5, metrics)

    # Stage rewards panel — show all 4 stage reward curves stacked
    ax6 = fig.add_subplot(2, 3, 6)
    all_rewards = []
    boundaries  = []
    for key in ['stage1', 'stage2', 'stage3', 'stage4']:
        s = metrics.get(key, {})
        if isinstance(s, str):
            continue
        rw = s.get('rewards', [])
        if rw:
            try:
                rw = [float(r) for r in rw]
                boundaries.append((len(all_rewards), key))
                all_rewards.extend(rw)
            except (ValueError, TypeError):
                pass

    if all_rewards:
        eps = np.arange(1, len(all_rewards)+1)
        ax6.plot(eps, all_rewards, alpha=0.06, color='#555', linewidth=0.3)
        rm = rolling(all_rewards, max(50, len(all_rewards)//30))
        ax6.plot(eps, rm, color='#333', linewidth=2)
        for start, key in boundaries:
            ax6.axvline(x=start+1, color=STAGE_COLORS[key],
                        linestyle='--', linewidth=1.5, alpha=0.7,
                        label=f'{key}')
        ax6.set_xlabel('Episode (all stages)', fontsize=10)
        ax6.set_ylabel('Reward', fontsize=10)
        ax6.set_title('All Stages: Reward Timeline', fontsize=12,
                       fontweight='bold')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('training_plots.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    print("Saved: training_plots.png")
    plt.close()

    # ==========================================================
    # DASHBOARD B: Phase 3 / Stage 4 report (only if data exists)
    # ==========================================================
    if has_stage4:
        fig2 = plt.figure(figsize=(20, 13))
        fig2.suptitle('RL Network Coding Bounds — Phase 3 Novel Inequality Search',
                       fontsize=15, fontweight='bold', y=0.98)

        ax_novel = fig2.add_subplot(2, 2, 1)
        plot_stage4_novel_rate(ax_novel, metrics)

        ax_pb = fig2.add_subplot(2, 2, 2)
        plot_stage4_bounds_vs_pb(ax_pb, metrics)

        ax_lam = fig2.add_subplot(2, 2, 3)
        plot_stage4_lambda_analysis(ax_lam, metrics)

        # Stage 4 reward curve standalone
        ax_r4 = fig2.add_subplot(2, 2, 4)
        s4 = metrics.get('stage4', {})
        rw4 = s4.get('rewards', []) if isinstance(s4, dict) else []
        if rw4:
            try:
                rw4 = [float(r) for r in rw4]
                eps4 = np.arange(1, len(rw4)+1)
                ax_r4.plot(eps4, rw4, alpha=0.07,
                           color=STAGE_COLORS['stage4'], linewidth=0.4)
                rm4 = rolling(rw4, max(30, len(rw4)//25))
                ax_r4.plot(eps4, rm4, color=STAGE_COLORS['stage4'],
                           linewidth=2.5)
                ax_r4.axhline(y=0, color='green', linestyle='--',
                              alpha=0.5, linewidth=1,
                              label='PB matched (reward=0)')
                last_n = max(len(rw4)//10, 5)
                ax_r4.text(0.98, 0.05,
                           f'Final avg: {np.mean(rw4[-last_n:]):.3f}',
                           transform=ax_r4.transAxes, ha='right', fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.3',
                                     facecolor='white', alpha=0.8))
            except (ValueError, TypeError):
                pass
        ax_r4.set_xlabel('Episode (Stage 4)', fontsize=10)
        ax_r4.set_ylabel('Reward', fontsize=10)
        ax_r4.set_title('Stage 4: Phase 3 Reward Curve\n'
                         '(positive = beats partition bound)',
                         fontsize=12, fontweight='bold')
        ax_r4.legend(fontsize=9)
        ax_r4.grid(True, alpha=0.2)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('training_plots_phase3.png', dpi=150,
                    bbox_inches='tight', facecolor='white')
        print("Saved: training_plots_phase3.png")
        plt.close()
    else:
        print("Stage 4 data not found — skipping Phase 3 dashboard.")
        print("Run full training (including stage4_episodes > 0) to generate it.")

    # ==========================================================
    # Individual high-resolution plots
    # ==========================================================

    for fname, fn, kwargs in [
        ('plot_bound_convergence.png',  plot_bound_convergence,  {}),
        ('plot_phase1_learning.png',    plot_phase1_learning,    {}),
        ('plot_episode_length.png',     plot_episode_length,     {}),
        ('plot_eval_gap.png',           plot_eval_gap_violin,    {}),
        ('plot_eval_hitrate.png',       plot_eval_hit_rate,      {}),
    ]:
        fig, ax = plt.subplots(figsize=(10, 6))
        fn(ax, metrics)
        fig.tight_layout()
        fig.savefig(fname, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()

    # All 4 stage rewards side by side
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    plot_all_stage_rewards(axes, metrics)
    fig.suptitle('Reward Convergence — All Four Stages',
                  fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig('plot_stage_rewards.png', dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close()

    if has_stage4:
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_stage4_novel_rate(ax, metrics)
        fig.tight_layout()
        fig.savefig('plot_stage4_novel.png', dpi=200, bbox_inches='tight',
                    facecolor='white')
        plt.close()

        fig, ax = plt.subplots(figsize=(12, 6))
        plot_stage4_bounds_vs_pb(ax, metrics)
        fig.tight_layout()
        fig.savefig('plot_phase3_bounds.png', dpi=200, bbox_inches='tight',
                    facecolor='white')
        plt.close()

        fig, ax = plt.subplots(figsize=(12, 6))
        plot_stage4_lambda_analysis(ax, metrics)
        fig.tight_layout()
        fig.savefig('plot_stage4_lambda.png', dpi=200, bbox_inches='tight',
                    facecolor='white')
        plt.close()

    print("Saved individual plots: plot_*.png")
    if has_stage4:
        print("Phase 3 plots: plot_stage4_novel.png  "
              "plot_phase3_bounds.png  plot_stage4_lambda.png")


if __name__ == "__main__":
    main()