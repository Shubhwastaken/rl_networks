"""
Training loop for the partition bound RL agent.

Features:
- True optimal bounds (brute-force) as baseline instead of random sampling
- Detailed per-graph logging: which graph, partition found, gap to optimal
- 3-stage curriculum: 1500 episodes each, 750 evaluation episodes
- Prints partition details whenever agent finds optimal bound
"""

import random
import numpy as np
from collections import defaultdict
from copy import deepcopy

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

from fixed_environment import PartitionBoundEnv, ActionType, Phase
from partition import generate_random_valid_partition, decode_partition
from gnn_policy import GNNPhase1Policy, GNNPhase2Policy
from fixed_base_inequality_generator import internal_per_partition
from fixed_graph_generation import (
    get_all_graph_infos, get_optimal_for_graph, identify_graph
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _partition_str(partition, sessions):
    parts = []
    for i, group in enumerate(partition):
        internal = [f"{s}->{t}" for s, t in sessions
                    if s in set(group) and t in set(group)]
        if internal:
            parts.append(f"P{i+1}={sorted(group)}[{','.join(internal)}]")
        else:
            parts.append(f"P{i+1}={sorted(group)}")
    return "  ".join(parts)


def _action_summary(action_counts):
    short = {
        "ASSIGN_NODE": "ASN", "SWAP_NODE": "SWP", "MOVE_NODE": "MOV",
        "FINALIZE_PARTITION": "FIN",
        "ADD_TO_ACCUMULATOR": "ADD", "APPLY_SUBMODULARITY": "SUB",
        "APPLY_PROOF2": "P2", "STORE_AND_RESET": "STO",
        "COMBINE_STORED": "CMB", "DECLARE_TERMINAL": "TRM"
    }
    parts = []
    for a, c in sorted(action_counts.items(), key=lambda x: -x[1]):
        s = short.get(a, a[:3])
        if c > 0:
            parts.append(f"{s}:{c}")
    return " ".join(parts)


ACTION_NAMES = {
    0: "ASSIGN_NODE", 1: "ADD_TO_ACCUMULATOR", 2: "APPLY_SUBMODULARITY",
    3: "APPLY_PROOF2", 4: "STORE_AND_RESET", 5: "COMBINE_STORED",
    6: "DECLARE_TERMINAL", 7: "SWAP_NODE", 8: "MOVE_NODE",
    9: "FINALIZE_PARTITION"
}


def _print_graph_table():
    """Print summary table of all graphs with optimal bounds."""
    infos = get_all_graph_infos()
    print(f"\n  {'Name':<16} {'N':>3} {'E':>3} {'S':>3} "
          f"{'Trivial':>8} {'Optimal':>8} {'OptInt':>6}")
    print(f"  {'-'*55}")
    for info in infos:
        trivial = len(info.edges) / len(info.sessions)
        print(f"  {info.name:<16} {len(info.nodes):>3} {len(info.edges):>3} "
              f"{len(info.sessions):>3} {trivial:>8.4f} "
              f"{info.optimal_bound:>8.4f} {info.optimal_internal:>6}")
    print()


# -----------------------------------------------------------------------
# Stage 1: Train Phase 2 on random (fixed) partitions
# -----------------------------------------------------------------------

def run_stage1(num_episodes=1500, graph_dataset_size=5):
    print("=" * 70)
    print("STAGE 1: Training Phase 2 agent on fixed partitions")
    print(f"         {num_episodes} episodes")
    print("=" * 70)
    _print_graph_table()

    env = PartitionBoundEnv(graph_dataset_size=graph_dataset_size, stage=1)

    from fixed_inequality import EntropyIndex
    max_dim = 0
    for nodes, edges, sessions in env.graph_dataset:
        chrom = generate_random_valid_partition(nodes, edges)
        part  = decode_partition(nodes, chrom)
        ix    = EntropyIndex(partitions=part, nodes=nodes,
                             edges=edges, sessions=sessions)
        max_dim = max(max_dim, ix.dim)
    coeff_dim = max_dim
    print(f"  Max coeff dim: {coeff_dim}")

    phase2_policy = GNNPhase2Policy(coeff_dim=coeff_dim)
    phase2_policy.unfreeze()
    rewards       = []
    per_graph     = defaultdict(list)

    log_interval = max(num_episodes // 15, 1)
    print(f"\n  {'Ep':>6} | {'Graph':<16} | {'AvgRew':>8} | {'BestBnd':>8} | Actions")
    print(f"  {'-'*75}")

    for episode in range(num_episodes):
        graph_tuple = random.choice(env.graph_dataset)
        nodes, edges, sessions = graph_tuple
        graph_name = identify_graph(nodes, edges, sessions)

        chrom     = generate_random_valid_partition(nodes, edges)
        partition = decode_partition(nodes, chrom)
        state     = env.reset(fixed_partition=partition, fixed_graph=graph_tuple)
        state['edges'] = edges
        done          = False
        trajectory    = []
        action_counts = defaultdict(int)
        total_reward  = 0.0

        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            action = phase2_policy.select_action(state, valid)
            aname  = ACTION_NAMES.get(int(action['type']), '?')
            action_counts[aname] += 1
            state, reward, done = env.step(action)
            state['edges'] = edges
            trajectory.append({'reward': reward})
            total_reward += reward

        phase2_policy.update(trajectory, total_reward)
        rewards.append(total_reward)
        per_graph[graph_name].append(abs(total_reward))

        if (episode + 1) % log_interval == 0:
            n    = log_interval
            avg  = np.mean(rewards[-n:])
            best = abs(min(rewards[-n:]))
            asumm = _action_summary(action_counts)
            print(f"  {episode+1:>6} | {graph_name:<16} | "
                  f"{avg:>8.4f} | {best:>8.4f} | {asumm}")

    # Per-graph summary
    print(f"\n  Per-graph avg bounds (Stage 1):")
    for gname in sorted(per_graph.keys()):
        bounds = per_graph[gname]
        opt, _ = get_optimal_for_graph(
            *next(t for t in env.graph_dataset
                  if identify_graph(*t) == gname)
        )
        avg_b = np.mean(bounds)
        best_b = min(bounds)
        print(f"    {gname:<16}: avg={avg_b:.4f}  best={best_b:.4f}  "
              f"optimal={opt:.4f}  gap={best_b - opt:+.4f}")

    print("\nStage 1 complete.\n")
    return phase2_policy, coeff_dim


# -----------------------------------------------------------------------
# Stage 2: Train Phase 1 (with SWAP/MOVE), Phase 2 frozen
# -----------------------------------------------------------------------

def run_stage2(phase2_policy, num_episodes=1500, graph_dataset_size=5):
    print("=" * 70)
    print("STAGE 2: Training Phase 1 agent (Phase 2 frozen)")
    print(f"         {num_episodes} episodes | ASSIGN + REFINE (SWAP/MOVE)")
    print("=" * 70)

    phase2_policy.freeze()
    env = PartitionBoundEnv(graph_dataset_size=graph_dataset_size, stage=2)
    phase1_policy = GNNPhase1Policy()
    rewards   = []
    internals = []
    per_graph = defaultdict(lambda: {'int': [], 'optimal_found': 0})

    log_interval = max(num_episodes // 15, 1)
    print(f"\n  {'Ep':>6} | {'Graph':<16} | {'AvgRew':>8} | {'BestBnd':>8} | "
          f"{'Int':>3} | P1 Actions")
    print(f"  {'-'*80}")

    for episode in range(num_episodes):
        env.reset()
        nodes, edges, sessions = env.nodes, env.edges, env.sessions
        graph_name = identify_graph(nodes, edges, sessions)
        opt_bound, opt_int = get_optimal_for_graph(nodes, edges, sessions)

        temperature = max(1.0, 2.0 - 2.0 * episode / max(num_episodes, 1))

        p1_traj  = []
        state    = env._get_state()
        state['edges']       = edges
        state['sessions']    = sessions
        state['temperature'] = temperature
        done = False
        p1_action_counts = defaultdict(int)

        while env.current_phase == Phase.PHASE1 and not done:
            valid  = env.get_valid_actions()
            if not valid:
                break
            state['temperature'] = temperature
            action = phase1_policy.select_action(state, valid)
            aname  = ACTION_NAMES.get(int(action.get('type', 0)), '?')
            p1_action_counts[aname] += 1
            state, reward, done = env.step(action)
            state['edges']    = edges
            state['sessions'] = sessions
            p1_traj.append({'reward': reward})

        rl_partition = [list(g) for g in env.partition] if env.partition else []
        ipp          = internal_per_partition(rl_partition, sessions)
        int_count    = sum(ipp)
        internals.append(int_count)
        per_graph[graph_name]['int'].append(int_count)

        if int_count >= opt_int:
            per_graph[graph_name]['optimal_found'] += 1

        # Phase 2 frozen
        total_reward = sum(t['reward'] for t in p1_traj)
        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            action = phase2_policy.select_action(state, valid)
            state, reward, done = env.step(action)
            state['edges']    = edges
            state['sessions'] = sessions
            total_reward += reward

        phase1_policy.update(p1_traj, total_reward)
        rewards.append(total_reward)

        # Print when optimal internal sessions found
        if int_count >= opt_int and (episode + 1) % max(log_interval // 3, 1) == 0:
            print(f"  ** Ep {episode+1}: {graph_name} OPTIMAL int={int_count} "
                  f"| {_partition_str(rl_partition, sessions)}")

        if (episode + 1) % log_interval == 0:
            n    = log_interval
            avg  = np.mean(rewards[-n:])
            best = abs(min(rewards[-n:]))
            avgi = np.mean(internals[-n:])
            asumm = _action_summary(p1_action_counts)
            print(f"  {episode+1:>6} | {graph_name:<16} | "
                  f"{avg:>8.4f} | {best:>8.4f} | {avgi:>3.1f} | {asumm}")

    # Per-graph summary
    print(f"\n  Per-graph Phase 1 summary (Stage 2):")
    for gname in sorted(per_graph.keys()):
        stats = per_graph[gname]
        total = len(stats['int'])
        avg_i = np.mean(stats['int']) if stats['int'] else 0
        opt_rate = 100 * stats['optimal_found'] / max(total, 1)
        print(f"    {gname:<16}: avg_internal={avg_i:.2f}  "
              f"optimal_rate={opt_rate:.1f}%  ({stats['optimal_found']}/{total})")

    print("\nStage 2 complete.\n")
    return phase1_policy


# -----------------------------------------------------------------------
# Stage 3: Joint fine-tuning
# -----------------------------------------------------------------------

def run_stage3(phase1_policy, phase2_policy, num_episodes=1500,
               graph_dataset_size=5):
    print("=" * 70)
    print("STAGE 3: Joint fine-tuning (both phases unfrozen)")
    print(f"         {num_episodes} episodes")
    print("=" * 70)

    phase2_policy.unfreeze()
    env = PartitionBoundEnv(graph_dataset_size=graph_dataset_size, stage=3)
    rewards   = []
    internals = []
    per_graph = defaultdict(lambda: {
        'bounds': [], 'int': [], 'optimal_bound_found': 0,
        'optimal_int_found': 0
    })

    log_interval = max(num_episodes // 15, 1)
    print(f"\n  {'Ep':>6} | {'Graph':<16} | {'AvgRew':>8} | {'BestBnd':>8} | "
          f"{'Int':>3} | P2 Actions")
    print(f"  {'-'*80}")

    for episode in range(num_episodes):
        env.reset()
        nodes, edges, sessions = env.nodes, env.edges, env.sessions
        graph_name = identify_graph(nodes, edges, sessions)
        opt_bound, opt_int = get_optimal_for_graph(nodes, edges, sessions)

        temperature = max(1.0, 1.5 - 1.0 * episode / max(num_episodes, 1))

        # Phase 1
        p1_traj  = []
        state    = env._get_state()
        state['edges']       = edges
        state['sessions']    = sessions
        state['temperature'] = temperature
        done = False
        p2_action_counts = defaultdict(int)
        total_reward     = 0.0

        while env.current_phase == Phase.PHASE1 and not done:
            valid  = env.get_valid_actions()
            if not valid:
                break
            state['temperature'] = temperature
            action = phase1_policy.select_action(state, valid)
            state, reward, done = env.step(action)
            state['edges']    = edges
            state['sessions'] = sessions
            p1_traj.append({'reward': reward})
            total_reward += reward

        rl_partition = [list(g) for g in env.partition] if env.partition else []
        ipp          = internal_per_partition(rl_partition, sessions)
        int_count    = sum(ipp)
        internals.append(int_count)

        # Phase 2
        p2_traj = []
        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            action = phase2_policy.select_action(state, valid)
            aname  = ACTION_NAMES.get(int(action.get('type', 0)), '?')
            p2_action_counts[aname] += 1
            state, reward, done = env.step(action)
            state['edges']    = edges
            state['sessions'] = sessions
            p2_traj.append({'reward': reward})
            total_reward += reward

        phase1_policy.update(p1_traj, total_reward)
        phase2_policy.update(p2_traj, total_reward)
        rewards.append(total_reward)

        rl_bound = abs(total_reward)
        per_graph[graph_name]['bounds'].append(rl_bound)
        per_graph[graph_name]['int'].append(int_count)
        if int_count >= opt_int:
            per_graph[graph_name]['optimal_int_found'] += 1
        if abs(rl_bound - opt_bound) < 0.05:
            per_graph[graph_name]['optimal_bound_found'] += 1

        # Print discoveries
        if abs(rl_bound - opt_bound) < 0.05:
            if (episode + 1) % max(log_interval // 5, 1) == 0:
                print(f"  ** Ep {episode+1}: {graph_name} OPTIMAL BOUND "
                      f"{rl_bound:.4f} int={int_count}"
                      f" | {_partition_str(rl_partition, sessions)}")

        if (episode + 1) % log_interval == 0:
            n    = log_interval
            avg  = np.mean(rewards[-n:])
            best = abs(min(rewards[-n:]))
            avgi = np.mean(internals[-n:])
            asumm = _action_summary(p2_action_counts)
            print(f"  {episode+1:>6} | {graph_name:<16} | "
                  f"{avg:>8.4f} | {best:>8.4f} | {avgi:>3.1f} | {asumm}")

    # Per-graph summary
    print(f"\n  Per-graph Stage 3 summary:")
    print(f"  {'Graph':<16} {'AvgBound':>9} {'BestBound':>10} {'Optimal':>8} "
          f"{'BndRate':>8} {'IntRate':>8} {'Episodes':>8}")
    print(f"  {'-'*70}")
    for gname in sorted(per_graph.keys()):
        stats = per_graph[gname]
        total = len(stats['bounds'])
        opt_b, _ = get_optimal_for_graph(
            *next(t for t in env.graph_dataset
                  if identify_graph(*t) == gname)
        )
        avg_b  = np.mean(stats['bounds']) if stats['bounds'] else 0
        best_b = min(stats['bounds']) if stats['bounds'] else 0
        bnd_rate = 100 * stats['optimal_bound_found'] / max(total, 1)
        int_rate = 100 * stats['optimal_int_found'] / max(total, 1)
        print(f"  {gname:<16} {avg_b:>9.4f} {best_b:>10.4f} {opt_b:>8.4f} "
              f"{bnd_rate:>7.1f}% {int_rate:>7.1f}% {total:>8}")

    print("\nStage 3 complete.\n")
    return phase1_policy, phase2_policy


# -----------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------

def evaluate(phase1_policy, phase2_policy, num_episodes=750,
             graph_dataset_size=5):
    print("=" * 70)
    print("EVALUATION")
    print(f"  {num_episodes} episodes | True optimal bounds as baseline")
    print("=" * 70)
    _print_graph_table()

    env = PartitionBoundEnv(graph_dataset_size=graph_dataset_size, stage=3)
    phase2_policy.unfreeze()

    per_graph = defaultdict(lambda: {
        'rl_bounds': [], 'internals': [], 'optimal_found': 0,
        'corrupted': 0
    })

    for episode in range(num_episodes):
        graph_tuple = random.choice(env.graph_dataset)
        nodes, edges, sessions = graph_tuple
        graph_name = identify_graph(nodes, edges, sessions)
        opt_bound, opt_int = get_optimal_for_graph(nodes, edges, sessions)

        state = env.reset(fixed_graph=graph_tuple)
        done  = False
        last_reward  = -(len(edges) / max(len(sessions), 1))
        rl_partition = None

        while env.current_phase == Phase.PHASE1 and not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            state['temperature'] = 1.0
            state['sessions']    = sessions
            state['edges']       = edges
            action = phase1_policy.select_action(state, valid)
            state, _, done = env.step(action)
            state['edges']    = edges
            state['sessions'] = sessions

        if env.partition is not None:
            rl_partition = [list(g) for g in env.partition]

        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            action = phase2_policy.select_action(state, valid)
            state, reward, done = env.step(action)
            state['edges']    = edges
            state['sessions'] = sessions
            if done:
                last_reward = reward

        rl_bound = abs(last_reward)

        # Verify bound consistency
        if rl_partition:
            ipp = internal_per_partition(rl_partition, sessions)
            denom = len(sessions) + sum(ipp)
            formula_bound = len(edges) / denom if denom > 0 else float('inf')
            if rl_bound < formula_bound - 1e-3:
                per_graph[graph_name]['corrupted'] += 1
                rl_bound = formula_bound
            int_count = sum(ipp)
        else:
            int_count = 0

        per_graph[graph_name]['rl_bounds'].append(rl_bound)
        per_graph[graph_name]['internals'].append(int_count)

        is_optimal = abs(rl_bound - opt_bound) < 0.05
        if is_optimal:
            per_graph[graph_name]['optimal_found'] += 1

        gap = rl_bound - opt_bound
        status = "OPTIMAL" if is_optimal else ""

        print(f"  Ep {episode+1:4d} | {graph_name:<16} | "
              f"RL: {rl_bound:.4f} | Opt: {opt_bound:.4f} | "
              f"Gap: {gap:+.4f} | int={int_count}/{opt_int} | {status}")

        if is_optimal and rl_partition:
            print(f"         {_partition_str(rl_partition, sessions)}")

    # ===== FINAL SUMMARY =====
    print()
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    total_eps     = 0
    total_optimal = 0
    total_corrupt = 0

    print(f"\n  {'Graph':<16} {'Eps':>5} {'AvgRL':>8} {'BestRL':>8} "
          f"{'Optimal':>8} {'AvgGap':>8} {'OptRate':>8} {'AvgInt':>7} "
          f"{'Corrupt':>7}")
    print(f"  {'-'*85}")

    for gname in sorted(per_graph.keys()):
        stats = per_graph[gname]
        eps   = len(stats['rl_bounds'])
        total_eps += eps
        total_optimal += stats['optimal_found']
        total_corrupt += stats['corrupted']

        opt_b, opt_i = get_optimal_for_graph(
            *next(t for t in env.graph_dataset
                  if identify_graph(*t) == gname)
        )

        avg_rl  = np.mean(stats['rl_bounds'])
        best_rl = min(stats['rl_bounds'])
        avg_gap = avg_rl - opt_b
        opt_rate = 100 * stats['optimal_found'] / max(eps, 1)
        avg_int = np.mean(stats['internals'])

        print(f"  {gname:<16} {eps:>5} {avg_rl:>8.4f} {best_rl:>8.4f} "
              f"{opt_b:>8.4f} {avg_gap:>+8.4f} {opt_rate:>7.1f}% "
              f"{avg_int:>7.2f} {stats['corrupted']:>7}")

    print(f"\n  TOTALS:")
    print(f"    Episodes          : {total_eps}")
    print(f"    Optimal found     : {total_optimal}/{total_eps} "
          f"({100*total_optimal/max(total_eps,1):.1f}%)")
    print(f"    Corrupted bounds  : {total_corrupt}")

    all_bounds = []
    for stats in per_graph.values():
        all_bounds.extend(stats['rl_bounds'])
    if all_bounds:
        print(f"    Overall avg bound : {np.mean(all_bounds):.4f}")
        print(f"    Overall best bound: {min(all_bounds):.4f}")

    print("=" * 70)


# -----------------------------------------------------------------------
# Training entry point
# -----------------------------------------------------------------------

def train(stage1_episodes=1500, stage2_episodes=1500,
          stage3_episodes=1500, graph_dataset_size=5):
    phase2_policy, coeff_dim = run_stage1(
        num_episodes=stage1_episodes,
        graph_dataset_size=graph_dataset_size
    )
    phase1_policy = run_stage2(
        phase2_policy=phase2_policy,
        num_episodes=stage2_episodes,
        graph_dataset_size=graph_dataset_size
    )
    phase1_policy, phase2_policy = run_stage3(
        phase1_policy=phase1_policy,
        phase2_policy=phase2_policy,
        num_episodes=stage3_episodes,
        graph_dataset_size=graph_dataset_size
    )
    return phase1_policy, phase2_policy


if __name__ == "__main__":
    phase1_policy, phase2_policy = train(
        stage1_episodes=5000,
        stage2_episodes=5000,
        stage3_episodes=5000,
        graph_dataset_size=5
    )
    evaluate(
        phase1_policy, phase2_policy,
        num_episodes=1000,
        graph_dataset_size=5
    )