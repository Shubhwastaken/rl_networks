"""
Training loop — 10k episode version.

Fixes:
- Stage 1: uses greedy session-pairing partitions (not random colorings)
  so Phase 2 trains on partitions with internal sessions from the start
- total_episodes passed to both policies for LR/entropy scheduling
- Temperature: max(1.0, 2.5 - 2.5 * ep/N)
- Episode counts: 10k/10k/10k training, 2k eval
"""

import random
import numpy as np
from collections import defaultdict
import json, time

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


def _greedy_session_partition(nodes, edges, sessions):
    """
    Greedy partition that tries to group session pairs together.
    Much better than random coloring for Stage 1 training.
    """
    adj = {n: set() for n in nodes}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    assignment = {}
    gid = 0

    # First: try to assign each session pair to the same group
    for s, t in sessions:
        if s not in assignment and t not in assignment:
            # Check if s and t are NOT adjacent (can share a group)
            if t not in adj[s]:
                assignment[s] = gid
                assignment[t] = gid
                gid += 1
            else:
                assignment[s] = gid
                gid += 1
                assignment[t] = gid
                gid += 1
        elif s in assignment and t not in assignment:
            g = assignment[s]
            # Can t join s's group?
            conflict = any(assignment.get(n) == g for n in adj[t] if n in assignment)
            if not conflict:
                assignment[t] = g
            else:
                assignment[t] = gid
                gid += 1
        elif t in assignment and s not in assignment:
            g = assignment[t]
            conflict = any(assignment.get(n) == g for n in adj[s] if n in assignment)
            if not conflict:
                assignment[s] = g
            else:
                assignment[s] = gid
                gid += 1

    # Assign remaining nodes
    for node in nodes:
        if node not in assignment:
            neighbor_groups = {assignment[n] for n in adj[node] if n in assignment}
            # Try existing groups
            placed = False
            for g in range(gid):
                if g not in neighbor_groups:
                    assignment[node] = g
                    placed = True
                    break
            if not placed:
                assignment[node] = gid
                gid += 1

    # Convert to partition list
    groups = {}
    for node, g in assignment.items():
        groups.setdefault(g, []).append(node)
    return list(groups.values())


# -----------------------------------------------------------------------
# Stage 1
# -----------------------------------------------------------------------

def run_stage1(num_episodes=10000, graph_dataset_size=5):
    print("=" * 70)
    print(f"STAGE 1: Training Phase 2 ({num_episodes} episodes)")
    print("=" * 70)
    _print_graph_table()

    env = PartitionBoundEnv(graph_dataset_size=graph_dataset_size, stage=1)

    from fixed_inequality import EntropyIndex
    max_dim = 0
    for nodes, edges, sessions in env.graph_dataset:
        # Use greedy partition to estimate max dim
        part = _greedy_session_partition(nodes, edges, sessions)
        ix = EntropyIndex(partitions=part, nodes=nodes, edges=edges, sessions=sessions)
        max_dim = max(max_dim, ix.dim)
        # Also try random to cover different partition sizes
        chrom = generate_random_valid_partition(nodes, edges)
        part2 = decode_partition(nodes, chrom)
        ix2 = EntropyIndex(partitions=part2, nodes=nodes, edges=edges, sessions=sessions)
        max_dim = max(max_dim, ix2.dim)
    coeff_dim = max_dim
    print(f"  Max coeff dim: {coeff_dim}")

    phase2_policy = GNNPhase2Policy(coeff_dim=coeff_dim, total_episodes=num_episodes)
    phase2_policy.unfreeze()
    rewards = []
    per_graph = defaultdict(list)
    metrics = {
        'rewards': [], 'bounds': [], 'graph_names': [],
        'step_counts': [], 'action_counts_per_ep': []
    }

    log_interval = max(num_episodes // 15, 1)
    print(f"\n  {'Ep':>6} | {'Graph':<16} | {'AvgRew':>8} | {'BestBnd':>8} | Actions")
    print(f"  {'-'*75}")

    for episode in range(num_episodes):
        graph_tuple = random.choice(env.graph_dataset)
        nodes, edges, sessions = graph_tuple
        graph_name = identify_graph(nodes, edges, sessions)

        # 70% greedy session-pairing, 30% random for diversity
        if random.random() < 0.7:
            partition = _greedy_session_partition(nodes, edges, sessions)
        else:
            chrom = generate_random_valid_partition(nodes, edges)
            partition = decode_partition(nodes, chrom)

        state = env.reset(fixed_partition=partition, fixed_graph=graph_tuple)
        state['edges'] = edges
        # PROOF2 forcing: 30% for first 2000 eps, anneal to 0% by 5000
        proof2_fp = max(0.0, 0.3 * (1.0 - episode / min(5000, num_episodes)))
        state['proof2_force_prob'] = proof2_fp
        done = False
        trajectory = []
        action_counts = defaultdict(int)
        total_reward = 0.0
        step_count = 0

        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            action = phase2_policy.select_action(state, valid)
            aname = ACTION_NAMES.get(int(action['type']), '?')
            action_counts[aname] += 1
            state, reward, done = env.step(action)
            state['edges'] = edges
            state['proof2_force_prob'] = proof2_fp
            trajectory.append({'reward': reward})
            total_reward += reward
            step_count += 1

        phase2_policy.update(trajectory, total_reward)
        rewards.append(total_reward)
        per_graph[graph_name].append(abs(total_reward))

        metrics['rewards'].append(total_reward)
        metrics['bounds'].append(abs(total_reward))
        metrics['graph_names'].append(graph_name)
        metrics['step_counts'].append(step_count)
        metrics['action_counts_per_ep'].append(dict(action_counts))

        if (episode + 1) % log_interval == 0:
            n = log_interval
            avg = np.mean(rewards[-n:])
            best = abs(min(rewards[-n:]))
            asumm = _action_summary(action_counts)
            print(f"  {episode+1:>6} | {graph_name:<16} | "
                  f"{avg:>8.4f} | {best:>8.4f} | {asumm}")

    print(f"\n  Per-graph avg bounds (Stage 1):")
    for gname in sorted(per_graph.keys()):
        bounds = per_graph[gname]
        opt, _ = get_optimal_for_graph(
            *next(t for t in env.graph_dataset if identify_graph(*t) == gname))
        print(f"    {gname:<16}: avg={np.mean(bounds):.4f}  best={min(bounds):.4f}  "
              f"optimal={opt:.4f}  gap={min(bounds)-opt:+.4f}")
    print("\nStage 1 complete.\n")
    return phase2_policy, coeff_dim, metrics


# -----------------------------------------------------------------------
# Stage 2
# -----------------------------------------------------------------------

def run_stage2(phase2_policy, num_episodes=10000, graph_dataset_size=5):
    print("=" * 70)
    print(f"STAGE 2: Training Phase 1 ({num_episodes} episodes, Phase 2 frozen)")
    print("=" * 70)

    phase2_policy.freeze()
    env = PartitionBoundEnv(graph_dataset_size=graph_dataset_size, stage=2)
    phase1_policy = GNNPhase1Policy(total_episodes=num_episodes)
    rewards = []
    internals = []
    per_graph = defaultdict(lambda: {'int': [], 'optimal_found': 0})
    metrics = {
        'rewards': [], 'internals': [], 'graph_names': [],
        'optimal_int_found': []
    }

    log_interval = max(num_episodes // 15, 1)
    print(f"\n  {'Ep':>6} | {'Graph':<16} | {'AvgRew':>8} | {'BestBnd':>8} | "
          f"{'Int':>3} | P1 Actions")
    print(f"  {'-'*80}")

    for episode in range(num_episodes):
        env.reset()
        nodes, edges, sessions = env.nodes, env.edges, env.sessions
        graph_name = identify_graph(nodes, edges, sessions)
        opt_bound, opt_int = get_optimal_for_graph(nodes, edges, sessions)

        temperature = max(1.0, 2.5 - 2.5 * episode / max(num_episodes, 1))

        p1_traj = []
        state = env._get_state()
        state['edges'] = edges
        state['sessions'] = sessions
        state['temperature'] = temperature
        done = False
        p1_action_counts = defaultdict(int)

        while env.current_phase == Phase.PHASE1 and not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            state['temperature'] = temperature
            action = phase1_policy.select_action(state, valid)
            aname = ACTION_NAMES.get(int(action.get('type', 0)), '?')
            p1_action_counts[aname] += 1
            state, reward, done = env.step(action)
            state['edges'] = edges
            state['sessions'] = sessions
            p1_traj.append({'reward': reward})

        rl_partition = [list(g) for g in env.partition] if env.partition else []
        ipp = internal_per_partition(rl_partition, sessions)
        int_count = sum(ipp)
        internals.append(int_count)
        per_graph[graph_name]['int'].append(int_count)
        if int_count >= opt_int:
            per_graph[graph_name]['optimal_found'] += 1

        total_reward = sum(t['reward'] for t in p1_traj)
        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            action = phase2_policy.select_action(state, valid)
            state, reward, done = env.step(action)
            state['edges'] = edges
            state['sessions'] = sessions
            total_reward += reward

        phase1_policy.update(p1_traj, total_reward)
        rewards.append(total_reward)

        metrics['rewards'].append(total_reward)
        metrics['internals'].append(int_count)
        metrics['graph_names'].append(graph_name)
        metrics['optimal_int_found'].append(1 if int_count >= opt_int else 0)

        if int_count >= opt_int and (episode + 1) % log_interval == 0:
            print(f"  >> Ep {episode+1} {graph_name}: OPTIMAL int={int_count}")
            print(f"     {_partition_str(rl_partition, sessions)}")

        if (episode + 1) % log_interval == 0:
            n = log_interval
            avg = np.mean(rewards[-n:])
            best = abs(min(rewards[-n:]))
            avgi = np.mean(internals[-n:])
            asumm = _action_summary(p1_action_counts)
            print(f"  {episode+1:>6} | {graph_name:<16} | "
                  f"{avg:>8.4f} | {best:>8.4f} | {avgi:>3.1f} | {asumm}")

    print(f"\n  Per-graph Phase 1 summary (Stage 2):")
    for gname in sorted(per_graph.keys()):
        stats = per_graph[gname]
        total = len(stats['int'])
        avg_i = np.mean(stats['int']) if stats['int'] else 0
        opt_rate = 100 * stats['optimal_found'] / max(total, 1)
        print(f"    {gname:<16}: avg_internal={avg_i:.2f}  "
              f"optimal_rate={opt_rate:.1f}%  ({stats['optimal_found']}/{total})")
    print("\nStage 2 complete.\n")
    return phase1_policy, metrics


# -----------------------------------------------------------------------
# Stage 3
# -----------------------------------------------------------------------

def run_stage3(phase1_policy, phase2_policy, num_episodes=10000,
               graph_dataset_size=5):
    print("=" * 70)
    print(f"STAGE 3: Joint fine-tuning ({num_episodes} episodes)")
    print("=" * 70)

    phase2_policy.unfreeze()
    env = PartitionBoundEnv(graph_dataset_size=graph_dataset_size, stage=3)
    rewards = []
    internals = []
    per_graph = defaultdict(lambda: {
        'bounds': [], 'int': [], 'optimal_bound_found': 0, 'optimal_int_found': 0
    })
    metrics = {
        'rewards': [], 'bounds': [], 'internals': [], 'graph_names': [],
        'step_counts': [], 'optimal_bound_hit': [], 'optimal_int_hit': []
    }

    log_interval = max(num_episodes // 15, 1)
    print(f"\n  {'Ep':>6} | {'Graph':<16} | {'AvgRew':>8} | {'BestBnd':>8} | "
          f"{'Int':>3} | P2 Actions")
    print(f"  {'-'*80}")

    for episode in range(num_episodes):
        env.reset()
        nodes, edges, sessions = env.nodes, env.edges, env.sessions
        graph_name = identify_graph(nodes, edges, sessions)
        opt_bound, opt_int = get_optimal_for_graph(nodes, edges, sessions)

        temperature = max(1.0, 2.5 - 2.5 * episode / max(num_episodes, 1))

        p1_traj = []
        state = env._get_state()
        state['edges'] = edges
        state['sessions'] = sessions
        state['temperature'] = temperature
        done = False
        p2_action_counts = defaultdict(int)
        total_reward = 0.0

        while env.current_phase == Phase.PHASE1 and not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            state['temperature'] = temperature
            action = phase1_policy.select_action(state, valid)
            state, reward, done = env.step(action)
            state['edges'] = edges
            state['sessions'] = sessions
            p1_traj.append({'reward': reward})
            total_reward += reward

        rl_partition = [list(g) for g in env.partition] if env.partition else []
        ipp = internal_per_partition(rl_partition, sessions)
        int_count = sum(ipp)
        internals.append(int_count)

        p2_traj = []
        # PROOF2 forcing for Stage 3
        proof2_fp = max(0.0, 0.3 * (1.0 - episode / min(5000, num_episodes)))
        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            state['proof2_force_prob'] = proof2_fp
            action = phase2_policy.select_action(state, valid)
            aname = ACTION_NAMES.get(int(action.get('type', 0)), '?')
            p2_action_counts[aname] += 1
            state, reward, done = env.step(action)
            state['edges'] = edges
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

        metrics['rewards'].append(total_reward)
        metrics['bounds'].append(rl_bound)
        metrics['internals'].append(int_count)
        metrics['graph_names'].append(graph_name)
        metrics['step_counts'].append(env.phase2_steps)
        metrics['optimal_bound_hit'].append(1 if abs(rl_bound - opt_bound) < 0.05 else 0)
        metrics['optimal_int_hit'].append(1 if int_count >= opt_int else 0)

        if abs(rl_bound - opt_bound) < 0.05 and (episode + 1) % max(log_interval // 3, 1) == 0:
            print(f"  >> Ep {episode+1} {graph_name}: OPTIMAL bound={rl_bound:.4f} int={int_count}")
            print(f"     {_partition_str(rl_partition, sessions)}")

        if (episode + 1) % log_interval == 0:
            n = log_interval
            avg = np.mean(rewards[-n:])
            best = abs(min(rewards[-n:]))
            avgi = np.mean(internals[-n:])
            asumm = _action_summary(p2_action_counts)
            print(f"  {episode+1:>6} | {graph_name:<16} | "
                  f"{avg:>8.4f} | {best:>8.4f} | {avgi:>3.1f} | {asumm}")

    print(f"\n  Per-graph Stage 3 summary:")
    print(f"  {'Graph':<16} {'AvgBound':>9} {'BestBound':>10} {'Optimal':>8} "
          f"{'BndRate':>8} {'IntRate':>8} {'Episodes':>8}")
    print(f"  {'-'*70}")
    for gname in sorted(per_graph.keys()):
        stats = per_graph[gname]
        total = len(stats['bounds'])
        opt_b, _ = get_optimal_for_graph(
            *next(t for t in env.graph_dataset if identify_graph(*t) == gname))
        avg_b = np.mean(stats['bounds']) if stats['bounds'] else 0
        best_b = min(stats['bounds']) if stats['bounds'] else 0
        bnd_rate = 100 * stats['optimal_bound_found'] / max(total, 1)
        int_rate = 100 * stats['optimal_int_found'] / max(total, 1)
        print(f"  {gname:<16} {avg_b:>9.4f} {best_b:>10.4f} {opt_b:>8.4f} "
              f"{bnd_rate:>7.1f}% {int_rate:>7.1f}% {total:>8}")
    print("\nStage 3 complete.\n")
    return phase1_policy, phase2_policy, metrics


# -----------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------

def evaluate(phase1_policy, phase2_policy, num_episodes=2000,
             graph_dataset_size=5):
    print("=" * 70)
    print(f"EVALUATION ({num_episodes} episodes)")
    print("=" * 70)
    _print_graph_table()

    env = PartitionBoundEnv(graph_dataset_size=graph_dataset_size, stage=3)
    phase2_policy.unfreeze()

    per_graph = defaultdict(lambda: {
        'rl_bounds': [], 'internals': [], 'optimal_found': 0, 'corrupted': 0
    })
    eval_metrics = {
        'rl_bounds': [], 'opt_bounds': [], 'gaps': [],
        'internals': [], 'opt_internals': [], 'graph_names': []
    }

    log_interval = max(num_episodes // 20, 1)

    for episode in range(num_episodes):
        graph_tuple = random.choice(env.graph_dataset)
        nodes, edges, sessions = graph_tuple
        graph_name = identify_graph(nodes, edges, sessions)
        opt_bound, opt_int = get_optimal_for_graph(nodes, edges, sessions)

        state = env.reset(fixed_graph=graph_tuple)
        done = False
        last_reward = -(len(edges) / max(len(sessions), 1))
        rl_partition = None

        while env.current_phase == Phase.PHASE1 and not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            state['temperature'] = 1.0
            state['sessions'] = sessions
            state['edges'] = edges
            action = phase1_policy.select_action(state, valid)
            state, _, done = env.step(action)
            state['edges'] = edges
            state['sessions'] = sessions

        if env.partition is not None:
            rl_partition = [list(g) for g in env.partition]

        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            action = phase2_policy.select_action(state, valid)
            state, reward, done = env.step(action)
            state['edges'] = edges
            state['sessions'] = sessions
            if done:
                last_reward = reward

        rl_bound = abs(last_reward)
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
        eval_metrics['rl_bounds'].append(rl_bound)
        eval_metrics['opt_bounds'].append(opt_bound)
        eval_metrics['gaps'].append(gap)
        eval_metrics['internals'].append(int_count)
        eval_metrics['opt_internals'].append(opt_int)
        eval_metrics['graph_names'].append(graph_name)

        status = "OPTIMAL" if is_optimal else ""
        if (episode + 1) % log_interval == 0:
            print(f"  Ep {episode+1:5d} | {graph_name:<16} | "
                  f"RL: {rl_bound:.4f} | Opt: {opt_bound:.4f} | "
                  f"Gap: {gap:+.4f} | int={int_count}/{opt_int} | {status}")
            if is_optimal and rl_partition:
                print(f"           {_partition_str(rl_partition, sessions)}")

    # Summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    total_eps = total_optimal = total_corrupt = 0

    print(f"\n  {'Graph':<16} {'Eps':>5} {'AvgRL':>8} {'BestRL':>8} "
          f"{'Optimal':>8} {'AvgGap':>8} {'OptRate':>8} {'AvgInt':>7} {'Corrupt':>7}")
    print(f"  {'-'*85}")
    for gname in sorted(per_graph.keys()):
        stats = per_graph[gname]
        eps = len(stats['rl_bounds'])
        total_eps += eps
        total_optimal += stats['optimal_found']
        total_corrupt += stats['corrupted']
        opt_b, _ = get_optimal_for_graph(
            *next(t for t in env.graph_dataset if identify_graph(*t) == gname))
        avg_rl = np.mean(stats['rl_bounds'])
        best_rl = min(stats['rl_bounds'])
        avg_gap = avg_rl - opt_b
        opt_rate = 100 * stats['optimal_found'] / max(eps, 1)
        avg_int = np.mean(stats['internals'])
        print(f"  {gname:<16} {eps:>5} {avg_rl:>8.4f} {best_rl:>8.4f} "
              f"{opt_b:>8.4f} {avg_gap:>+8.4f} {opt_rate:>7.1f}% "
              f"{avg_int:>7.2f} {stats['corrupted']:>7}")

    print(f"\n  TOTALS:")
    print(f"    Episodes: {total_eps}  Optimal: {total_optimal}/{total_eps} "
          f"({100*total_optimal/max(total_eps,1):.1f}%)  Corrupted: {total_corrupt}")
    all_bounds = [b for s in per_graph.values() for b in s['rl_bounds']]
    if all_bounds:
        print(f"    Avg bound: {np.mean(all_bounds):.4f}  Best: {min(all_bounds):.4f}")
    print("=" * 70)
    return eval_metrics


def train(stage1_episodes=10000, stage2_episodes=10000,
          stage3_episodes=10000, graph_dataset_size=5):
    phase2_policy, coeff_dim, s1_metrics = run_stage1(
        num_episodes=stage1_episodes, graph_dataset_size=graph_dataset_size)
    phase1_policy, s2_metrics = run_stage2(
        phase2_policy=phase2_policy,
        num_episodes=stage2_episodes, graph_dataset_size=graph_dataset_size)
    phase1_policy, phase2_policy, s3_metrics = run_stage3(
        phase1_policy=phase1_policy, phase2_policy=phase2_policy,
        num_episodes=stage3_episodes, graph_dataset_size=graph_dataset_size)
    return phase1_policy, phase2_policy, {'stage1': s1_metrics, 'stage2': s2_metrics, 'stage3': s3_metrics}


if __name__ == "__main__":
    t0 = time.time()
    phase1_policy, phase2_policy, train_metrics = train(
        stage1_episodes=100, stage2_episodes=100,
        stage3_episodes=100, graph_dataset_size=5)
    eval_metrics = evaluate(
        phase1_policy, phase2_policy,
        num_episodes=100, graph_dataset_size=5)

    all_metrics = {**train_metrics, 'eval': eval_metrics}
    with open('training_metrics.json', 'w') as f:
        json.dump(all_metrics, f)
    print(f"\nMetrics saved to training_metrics.json")
    print(f"TOTAL RUNTIME: {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")
    print(f"Run 'python plot_training.py' to generate plots.")