"""
Three-phase training with proper linkage.

STAGE 1 — Train Phase 2 on optimal partitions (proof calculus):
  Uses greedy session-pairing partitions as fixed input.
  Phase 2 now operates on per-node IOs and must discover combining patterns.
  Teaches the action grammar: ADD → CROSS_SUBMOD → STORE → DECLARE.

STAGE 2 — Train Phase 1 with frozen Phase 2 (partition learner):
  Phase 1 outputs partition + weight vector.
  Phase 2 (frozen) evaluates each partition's proof potential.
  Phase 1 learns to maximise internal sessions AND produce partitions
  that enable short, tight Phase 2 proofs.

STAGE 3 — Joint fine-tuning Phase 1 + Phase 2 (end-to-end):
  Both policies unfrozen, trained together on the full pipeline.
  Gradient signal flows from Phase 2 terminal reward back through
  the partition choice (via REINFORCE on Phase 1's trajectory).

STAGE 4 — Train Phase 3 (fractional IO search):
  Uses best partition + weights from Stage 3 as starting point.
  Phase 3 policy learns FRACTIONAL_IO, CROSS_SUBMOD sequences.
  Reward is ONLY positive when extracted bound < partition_bound.
  This is where novel inequalities are discovered.

Linkage mechanism:
  After each Stage 3 episode, env.partition and env.partition_weights
  are passed to Stage 4 as the starting state. Phase 3 therefore
  always starts from a good partition (not random), so it can focus
  its exploration budget on the fractional combination step.
"""

import random
import numpy as np
from collections import defaultdict
import json, time

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

from fixed_environment import PartitionBoundEnv, ActionType, Phase, _compute_partition_bound
from partition import generate_random_valid_partition, decode_partition
from gnn_policy import (
    GNNPhase1Policy, GNNPhase2Policy, GNNPhase3Policy, LAMBDA_GRID
)
from fixed_base_inequality_generator import internal_per_partition
from fixed_graph_generation import (
    get_all_graph_infos, get_optimal_for_graph, identify_graph
)
from fixed_inequality import EntropyIndex


def _partition_str(partition, sessions):
    parts = []
    for i, group in enumerate(partition):
        internal = [f"{s}->{t}" for s,t in sessions
                    if s in set(group) and t in set(group)]
        tag = f"P{i+1}={sorted(group)}"
        if internal: tag += f"[{','.join(internal)}]"
        parts.append(tag)
    return "  ".join(parts)


def _action_summary(action_counts):
    short = {
        "ASSIGN_NODE":"ASN","SWAP_NODE":"SWP","MOVE_NODE":"MOV",
        "FINALIZE_PARTITION":"FIN","ADD_TO_ACCUMULATOR":"ADD",
        "APPLY_SUBMODULARITY":"SUB","APPLY_PROOF2":"P2",
        "STORE_AND_RESET":"STO","COMBINE_STORED":"CMB",
        "DECLARE_TERMINAL":"TRM","FRACTIONAL_IO":"FIO",
        "CROSS_SUBMOD":"XSB",
    }
    parts = [f"{short.get(a,a[:3])}:{c}"
             for a, c in sorted(action_counts.items(), key=lambda x: -x[1]) if c>0]
    return " ".join(parts)


ACTION_NAMES = {
    0:"ASSIGN_NODE",1:"ADD_TO_ACCUMULATOR",2:"APPLY_SUBMODULARITY",
    3:"APPLY_PROOF2",4:"STORE_AND_RESET",5:"COMBINE_STORED",
    6:"DECLARE_TERMINAL",7:"SWAP_NODE",8:"MOVE_NODE",
    9:"FINALIZE_PARTITION",10:"FRACTIONAL_IO",11:"CROSS_SUBMOD",
}

EARLY_STOP_PATIENCE  = 2000
EARLY_STOP_MIN_EPS   = 3000


class EarlyStopper:
    def __init__(self, patience=EARLY_STOP_PATIENCE,
                 min_episodes=EARLY_STOP_MIN_EPS, window=500):
        self.patience    = patience
        self.min_episodes= min_episodes
        self.window      = window
        self.best_avg    = float('-inf')
        self.best_episode= 0
        self.rewards     = []

    def update(self, reward, episode):
        self.rewards.append(reward)
        if len(self.rewards) >= self.window:
            cur = np.mean(self.rewards[-self.window:])
            if cur > self.best_avg + 1e-4:
                self.best_avg    = cur
                self.best_episode= episode

    def should_stop(self, episode):
        if episode < self.min_episodes: return False
        return (episode - self.best_episode) >= self.patience


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
    adj = {n: set() for n in nodes}
    for u, v in edges:
        adj[u].add(v); adj[v].add(u)
    assignment = {}; gid = 0
    for s, t in sessions:
        if s not in assignment and t not in assignment:
            if t not in adj[s]:
                assignment[s] = gid; assignment[t] = gid; gid += 1
            else:
                assignment[s] = gid; gid += 1
                assignment[t] = gid; gid += 1
        elif s in assignment and t not in assignment:
            g = assignment[s]
            if not any(assignment.get(n) == g for n in adj[t] if n in assignment):
                assignment[t] = g
            else:
                assignment[t] = gid; gid += 1
        elif t in assignment and s not in assignment:
            g = assignment[t]
            if not any(assignment.get(n) == g for n in adj[s] if n in assignment):
                assignment[s] = g
            else:
                assignment[s] = gid; gid += 1
    for node in nodes:
        if node not in assignment:
            nb_groups = {assignment[n] for n in adj[node] if n in assignment}
            placed = False
            for g in range(gid):
                if g not in nb_groups:
                    assignment[node] = g; placed = True; break
            if not placed:
                assignment[node] = gid; gid += 1
    groups = {}
    for node, g in assignment.items():
        groups.setdefault(g, []).append(node)
    return list(groups.values())


# -----------------------------------------------------------------------
# Stage 1 — Train Phase 2 (proof calculus)
# -----------------------------------------------------------------------

def run_stage1(num_episodes=10000, graph_dataset_size=5):  # Tier 1 only
    print("=" * 70)
    print(f"STAGE 1: Train Phase 2 proof calculus ({num_episodes} episodes)")
    print("=" * 70)
    _print_graph_table()

    env = PartitionBoundEnv(graph_dataset_size=graph_dataset_size, stage=1)

    max_dim = 0
    for nodes, edges, sessions in env.graph_dataset:
        part = _greedy_session_partition(nodes, edges, sessions)
        ix   = EntropyIndex(partitions=part, nodes=nodes,
                            edges=edges, sessions=sessions)
        max_dim = max(max_dim, ix.dim)
        chrom = generate_random_valid_partition(nodes, edges)
        part2 = decode_partition(nodes, chrom)
        ix2   = EntropyIndex(partitions=part2, nodes=nodes,
                             edges=edges, sessions=sessions)
        max_dim = max(max_dim, ix2.dim)

    coeff_dim = max_dim
    print(f"  Max coeff dim: {coeff_dim}")

    phase2_policy = GNNPhase2Policy(coeff_dim=coeff_dim,
                                    total_episodes=num_episodes)
    phase2_policy.unfreeze()

    rewards   = []
    per_graph = defaultdict(list)
    metrics   = {'rewards':[], 'bounds':[], 'graph_names':[],
                 'step_counts':[], 'action_counts_per_ep':[]}

    log_interval = 100
    stopper = EarlyStopper()
    print(f"\n  {'Ep':>6} | {'Graph':<16} | {'AvgRew':>8} | {'BestBnd':>8} | Actions")
    print(f"  {'-'*75}")

    for episode in range(num_episodes):
        if stopper.should_stop(episode):
            print(f"\n  Early stopping at episode {episode}")
            break

        graph_tuple = random.choice(env.graph_dataset)
        nodes, edges, sessions = graph_tuple
        graph_name = identify_graph(nodes, edges, sessions)

        if random.random() < 0.7:
            partition = _greedy_session_partition(nodes, edges, sessions)
        else:
            chrom     = generate_random_valid_partition(nodes, edges)
            partition = decode_partition(nodes, chrom)

        state = env.reset(fixed_partition=partition, fixed_graph=graph_tuple)
        state['edges']   = edges
        proof2_fp = max(0.0, 0.3 * (1.0 - episode / min(5000, num_episodes)))
        state['proof2_force_prob'] = proof2_fp

        # Also pass graph info needed by Phase 3 policy (used indirectly)
        state['nodes']     = nodes
        state['sessions']  = sessions
        state['partition'] = partition

        done       = False
        trajectory = []
        action_counts = defaultdict(int)
        total_reward  = 0.0
        step_count    = 0

        while not done:
            valid  = env.get_valid_actions()
            if not valid: break
            action = phase2_policy.select_action(state, valid)
            aname  = ACTION_NAMES.get(int(action['type']), '?')
            action_counts[aname] += 1
            state, reward, done = env.step(action)
            state['edges']            = edges
            state['proof2_force_prob']= proof2_fp
            state['nodes']            = nodes
            state['sessions']         = sessions
            state['partition']        = partition
            trajectory.append({'reward': reward})
            total_reward += reward
            step_count   += 1

        phase2_policy.update(trajectory, total_reward)
        rewards.append(total_reward)
        per_graph[graph_name].append(abs(total_reward))
        stopper.update(total_reward, episode)

        metrics['rewards'].append(total_reward)
        metrics['bounds'].append(abs(total_reward))
        metrics['graph_names'].append(graph_name)
        metrics['step_counts'].append(step_count)
        metrics['action_counts_per_ep'].append(dict(action_counts))

        if (episode + 1) % log_interval == 0:
            n   = log_interval
            avg = np.mean(rewards[-n:])
            bst = abs(min(rewards[-n:]))
            print(f"  {episode+1:>6} | {graph_name:<16} | "
                  f"{avg:>8.4f} | {bst:>8.4f} | {_action_summary(action_counts)}")

    print(f"\n  Per-graph bounds (Stage 1):")
    for gname in sorted(per_graph.keys()):
        bounds = per_graph[gname]
        opt, _ = get_optimal_for_graph(
            *next(t for t in env.graph_dataset if identify_graph(*t) == gname))
        print(f"    {gname:<16}: avg={np.mean(bounds):.4f} "
              f"best={min(bounds):.4f} opt={opt:.4f}")
    print("\nStage 1 complete.\n")
    return phase2_policy, coeff_dim, metrics


# -----------------------------------------------------------------------
# Stage 2 — Train Phase 1 with frozen Phase 2
# -----------------------------------------------------------------------

def run_stage2(phase2_policy, num_episodes=10000, graph_dataset_size=5):
    print("=" * 70)
    print(f"STAGE 2: Train Phase 1 ({num_episodes} episodes, Phase 2 frozen)")
    print("=" * 70)

    phase2_policy.freeze()
    env  = PartitionBoundEnv(graph_dataset_size=graph_dataset_size, stage=2)
    phase1_policy = GNNPhase1Policy(total_episodes=num_episodes)

    rewards   = []
    internals = []
    per_graph = defaultdict(lambda: {'int':[], 'optimal_found':0})
    metrics   = {'rewards':[], 'internals':[], 'graph_names':[],
                 'optimal_int_found':[], 'partition_weights':[]}

    log_interval = 100
    stopper = EarlyStopper()
    print(f"\n  {'Ep':>6} | {'Graph':<16} | {'AvgRew':>8} | {'Int':>3} | P1 Actions")
    print(f"  {'-'*70}")

    for episode in range(num_episodes):
        if stopper.should_stop(episode):
            print(f"\n  Early stopping at episode {episode}")
            break

        env.reset()
        nodes, edges, sessions = env.nodes, env.edges, env.sessions
        graph_name = identify_graph(nodes, edges, sessions)
        opt_bound, opt_int = get_optimal_for_graph(nodes, edges, sessions)

        temperature = max(1.0, 2.5 - 2.5 * episode / max(num_episodes, 1))
        p1_traj    = []
        state      = env._get_state()
        state['edges']    = edges
        state['sessions'] = sessions
        state['temperature'] = temperature
        done = False
        p1_action_counts = defaultdict(int)

        while env.current_phase == Phase.PHASE1 and not done:
            valid = env.get_valid_actions()
            if not valid: break
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

        # Retrieve weights from Phase 1 (set during FINALIZE action)
        partition_weights = env.partition_weights

        total_reward = sum(t['reward'] for t in p1_traj)
        p2_traj = []
        while not done:
            valid = env.get_valid_actions()
            if not valid: break
            state['nodes']     = nodes
            state['sessions']  = sessions
            state['partition'] = rl_partition
            action = phase2_policy.select_action(state, valid)
            state, reward, done = env.step(action)
            state['edges']    = edges
            state['sessions'] = sessions
            p2_traj.append({'reward': reward})
            total_reward += reward

        phase1_policy.update(p1_traj, total_reward)
        rewards.append(total_reward)
        stopper.update(total_reward, episode)

        metrics['rewards'].append(total_reward)
        metrics['internals'].append(int_count)
        metrics['graph_names'].append(graph_name)
        metrics['optimal_int_found'].append(1 if int_count >= opt_int else 0)
        metrics['partition_weights'].append(partition_weights)

        if int_count >= opt_int and (episode + 1) % log_interval == 0:
            print(f"  >> Ep {episode+1} {graph_name}: OPTIMAL int={int_count}")

        if (episode + 1) % log_interval == 0:
            n    = log_interval
            avg  = np.mean(rewards[-n:])
            avgi = np.mean(internals[-n:])
            print(f"  {episode+1:>6} | {graph_name:<16} | "
                  f"{avg:>8.4f} | {avgi:>3.1f} | {_action_summary(p1_action_counts)}")

    print(f"\n  Per-graph Phase 1 (Stage 2):")
    for gname in sorted(per_graph.keys()):
        stats = per_graph[gname]
        total = len(stats['int'])
        print(f"    {gname:<16}: avg_int={np.mean(stats['int']):.2f} "
              f"opt_rate={100*stats['optimal_found']/max(total,1):.1f}%")
    print("\nStage 2 complete.\n")
    return phase1_policy, metrics


# -----------------------------------------------------------------------
# Stage 3 — Joint fine-tuning Phase 1 + Phase 2
# -----------------------------------------------------------------------

def run_stage3(phase1_policy, phase2_policy,
               num_episodes=10000, graph_dataset_size=10):  # Tier 1+2
    print("=" * 70)
    print(f"STAGE 3: Joint fine-tuning Phase 1+2 ({num_episodes} episodes)")
    print("=" * 70)

    phase2_policy.unfreeze()
    env = PartitionBoundEnv(graph_dataset_size=graph_dataset_size, stage=3)

    rewards   = []
    per_graph = defaultdict(list)
    metrics   = {'rewards':[], 'bounds':[], 'graph_names':[],
                 'best_partitions':{}}

    # Store best (partition, weights) per graph for Phase 4 handoff
    best_partitions: dict = {}   # graph_name -> (partition, weights, bound)

    log_interval = 100
    stopper = EarlyStopper()

    for episode in range(num_episodes):
        if stopper.should_stop(episode):
            print(f"\n  Early stopping at episode {episode}")
            break

        env.reset()
        nodes, edges, sessions = env.nodes, env.edges, env.sessions
        graph_name = identify_graph(nodes, edges, sessions)
        opt_bound, _ = get_optimal_for_graph(nodes, edges, sessions)

        temperature = max(1.0, 2.0 - 2.0 * episode / max(num_episodes, 1))
        p1_traj   = []
        state     = env._get_state()
        state['edges']    = edges
        state['sessions'] = sessions
        state['temperature'] = temperature
        done = False

        while env.current_phase == Phase.PHASE1 and not done:
            valid = env.get_valid_actions()
            if not valid: break
            state['temperature'] = temperature
            action = phase1_policy.select_action(state, valid)
            state, reward, done = env.step(action)
            state['edges']    = edges
            state['sessions'] = sessions
            p1_traj.append({'reward': reward})

        rl_partition     = [list(g) for g in env.partition] if env.partition else []
        partition_weights= env.partition_weights

        total_reward = sum(t['reward'] for t in p1_traj)
        p2_traj = []
        while not done:
            valid = env.get_valid_actions()
            if not valid: break
            state['nodes']     = nodes
            state['sessions']  = sessions
            state['partition'] = rl_partition
            action = phase2_policy.select_action(state, valid)
            state, reward, done = env.step(action)
            state['edges']    = edges
            state['sessions'] = sessions
            p2_traj.append({'reward': reward})
            total_reward += reward

        phase1_policy.update(p1_traj, total_reward)
        phase2_policy.update(p2_traj, total_reward)   # Pass actual trajectory

        rl_bound = abs(total_reward)
        rewards.append(total_reward)
        per_graph[graph_name].append(rl_bound)
        stopper.update(total_reward, episode)
        metrics['rewards'].append(total_reward)
        metrics['bounds'].append(rl_bound)
        metrics['graph_names'].append(graph_name)

        # Track best partition + weights for Phase 4
        if graph_name not in best_partitions or rl_bound < best_partitions[graph_name][2]:
            best_partitions[graph_name] = (rl_partition, partition_weights, rl_bound)

        if (episode + 1) % log_interval == 0:
            n   = log_interval
            avg = np.mean(rewards[-n:])
            print(f"  Ep {episode+1:>6} | {graph_name:<16} | avg={avg:.4f}")

    metrics['best_partitions'] = {
        k: {'partition': v[0], 'weights': v[1], 'bound': v[2]}
        for k, v in best_partitions.items()
    }
    print("\nStage 3 complete.\n")
    return phase1_policy, phase2_policy, metrics, best_partitions


# -----------------------------------------------------------------------
# Stage 4 — Train Phase 3 (fractional IO search for novel inequalities)
# -----------------------------------------------------------------------

def run_stage4(phase1_policy, phase2_policy, best_partitions,
               coeff_dim, num_episodes=10000, graph_dataset_size=12):  # All tiers
    """
    Phase 3 training. Uses best partitions from Stage 3 as fixed starting
    points, so the policy can focus entirely on fractional IO discovery.

    The key connection:
      Phase 1 found partition P* and weight vector w*.
      Phase 3 uses P* to determine which node pairs are cross-partition,
      and uses w* as prior λ suggestions.

      After FRACTIONAL_IO(u, v, λ), the resulting inequality has a
      coefficient of λ on the Y_ST term for u's partition and (1-λ) on
      v's partition. When these fractional inequalities are summed and
      SUBMOD is applied, the resulting Y_I coefficient may be irrational —
      which is the signature of an inequality outside the PB family.

    Reward:
      Only positive when extracted bound < partition_bound.
      Zero for matching PB (Phase 3 is not credited for reproducing Phase 2).
      Negative for worse than PB (gradient toward improvement).
    """
    print("=" * 70)
    print(f"STAGE 4: Phase 3 fractional IO search ({num_episodes} episodes)")
    print("=" * 70)

    env = PartitionBoundEnv(graph_dataset_size=graph_dataset_size, stage=4)

    phase3_policy = GNNPhase3Policy(
        coeff_dim=coeff_dim,
        total_episodes=num_episodes
    )

    rewards    = []
    per_graph  = defaultdict(list)
    novel_bounds = {}   # graph_name -> (bound, partition, weights, trace)
    metrics    = {'rewards':[], 'bounds':[], 'graph_names':[],
                  'novel_found': [], 'cross_partition_used': []}

    log_interval = 50
    stopper = EarlyStopper(patience=3000, min_episodes=2000)

    print(f"\n  Partition bounds for search:")
    for nodes, edges, sessions in env.graph_dataset:
        pb    = _compute_partition_bound(nodes, edges, sessions)
        gname = identify_graph(nodes, edges, sessions)
        print(f"    {gname:<16}: PB = {pb:.4f}")
    print()

    for episode in range(num_episodes):
        if stopper.should_stop(episode):
            print(f"\n  Early stopping at episode {episode}")
            break

        graph_tuple = random.choice(env.graph_dataset)
        nodes, edges, sessions = graph_tuple
        graph_name = identify_graph(nodes, edges, sessions)

        # Use best partition from Stage 3 if available, else greedy
        if graph_name in best_partitions:
            partition, p_weights, _ = best_partitions[graph_name]
        else:
            partition = _greedy_session_partition(nodes, edges, sessions)
            p_weights = {}

        # Set up env for Phase 3 directly
        env.nodes    = nodes
        env.edges    = edges
        env.sessions = sessions
        env.adjacency = {n: set() for n in nodes}
        env.edge_set  = set()
        for u, v in edges:
            env.adjacency[u].add(v); env.adjacency[v].add(u)
            env.edge_set.add((u,v)); env.edge_set.add((v,u))

        env.partition         = partition
        env.partition_weights = p_weights
        env.assignment        = {}
        env.num_groups        = len(partition)
        env._assignment_complete = True
        env._refinement_steps = 0
        env.prev_internal_count = 0

        env.partition_bound = _compute_partition_bound(nodes, edges, sessions)

        env._start_phase2()   # builds index, node_ios, base_inequalities
        env._start_phase3()   # initialises frac_pool, sets phase=PHASE3
        env.internal_per_part = env.internal_per_part or []

        state = env._get_state()
        state['nodes']             = nodes
        state['edges']             = edges
        state['sessions']          = sessions
        state['partition']         = partition
        state['partition_weights'] = p_weights

        done          = False
        trajectory    = []
        action_counts = defaultdict(int)
        total_reward  = 0.0
        used_cross    = False

        while not done:
            valid = env.get_valid_actions()
            if not valid:
                # Force terminal
                state, reward, done = env._extract_phase3_bound()
                total_reward += reward
                break

            action = phase3_policy.select_action(state, valid)
            aname  = ACTION_NAMES.get(int(action['type']), '?')
            action_counts[aname] += 1
            if action['type'] == ActionType.CROSS_SUBMOD:
                used_cross = True

            state, reward, done = env.step(action)
            state['nodes']             = nodes
            state['edges']             = edges
            state['sessions']          = sessions
            state['partition']         = partition
            state['partition_weights'] = p_weights
            trajectory.append({'reward': reward})
            total_reward += reward

        phase3_policy.update(trajectory, total_reward)
        rewards.append(total_reward)

        # Extract best bound from this episode
        best_b = env.frac_pool.best_bound(
            len(sessions), len(edges), env.internal_per_part
        )
        pb = env.partition_bound
        per_graph[graph_name].append(best_b)
        stopper.update(total_reward, episode)

        metrics['rewards'].append(total_reward)
        metrics['bounds'].append(best_b if best_b < 1e9 else -1)
        metrics['graph_names'].append(graph_name)
        metrics['novel_found'].append(1 if best_b < pb - 1e-8 else 0)
        metrics['cross_partition_used'].append(1 if used_cross else 0)

        # Record novel bounds
        if best_b < pb - 1e-8:
            if graph_name not in novel_bounds or best_b < novel_bounds[graph_name][0]:
                # Find the best terminal inequality for trace
                best_ineq = None
                for ineq in env.frac_pool:
                    if ineq.check_valid_terminal_form():
                        b2 = ineq.extract_bound(
                            len(sessions), len(edges), env.internal_per_part
                        )
                        if abs(b2 - best_b) < 1e-9:
                            best_ineq = ineq
                            break
                novel_bounds[graph_name] = (
                    best_b, partition, p_weights,
                    repr(best_ineq) if best_ineq else "N/A"
                )

        if (episode + 1) % log_interval == 0:
            n       = log_interval
            avg_r   = np.mean(rewards[-n:])
            novel_r = np.mean(metrics['novel_found'][-n:])
            cross_r = np.mean(metrics['cross_partition_used'][-n:])
            print(f"  Ep {episode+1:>6} | {graph_name:<16} | "
                  f"avg_r={avg_r:.4f} | novel_rate={100*novel_r:.1f}% | "
                  f"cross_used={100*cross_r:.1f}%")

            if novel_bounds:
                print(f"  ** NOVEL BOUNDS FOUND **")
                for gn, (b, part, w, trace) in sorted(novel_bounds.items()):
                    pb2 = _compute_partition_bound(
                        *next((nd,ed,ss) for nd,ed,ss in env.graph_dataset
                              if identify_graph(nd,ed,ss)==gn)
                    )
                    print(f"     {gn}: {b:.6f} < PB={pb2:.6f} "
                          f"(improvement={(pb2-b)/pb2*100:.2f}%)")
                    if trace != "N/A":
                        print(f"     Trace: {trace[:200]}")

    # Final summary
    print(f"\n{'='*70}")
    print(f"STAGE 4 COMPLETE — NOVEL INEQUALITY SEARCH RESULTS")
    print(f"{'='*70}")
    if novel_bounds:
        for gn, (b, part, w, trace) in sorted(novel_bounds.items()):
            try:
                nd, ed, ss = next((nd,ed,ss) for nd,ed,ss in env.graph_dataset
                                   if identify_graph(nd,ed,ss)==gn)
                pb2 = _compute_partition_bound(nd, ed, ss)
            except StopIteration:
                pb2 = float('inf')
            print(f"\n  Graph: {gn}")
            print(f"  Novel bound: r <= {b:.6f}")
            print(f"  Partition bound: r <= {pb2:.6f}")
            print(f"  Improvement: {(pb2-b)/pb2*100:.3f}%")
            print(f"  Partition used: {_partition_str(part, ss if "ss" in locals() else [])}")
            print(f"  Inequality: {trace[:400]}")
    else:
        print("\n  No super-PB bounds found in Stage 4.")
        print("  This does not mean none exist — increase num_episodes")
        print("  or check that CROSS_SUBMOD was used (check cross_used rate).")
        cross_total = sum(metrics['cross_partition_used'])
        print(f"  CROSS_SUBMOD used in {cross_total}/{num_episodes} episodes.")

    return phase3_policy, metrics, novel_bounds


# -----------------------------------------------------------------------
# Top-level train()
# -----------------------------------------------------------------------

def train(stage1_episodes=10000, stage2_episodes=10000,
          stage3_episodes=10000, stage4_episodes=10000,
          graph_dataset_size=5):
    """
    Run all four stages.
    graph_dataset_size controls Stage 1+2 (Tier 1 only = 5).
    Stage 3 automatically uses size=10, Stage 4 uses size=12.
    """
    phase2_policy, coeff_dim, s1 = run_stage1(stage1_episodes, graph_dataset_size)
    phase1_policy, s2             = run_stage2(phase2_policy, stage2_episodes, graph_dataset_size)
    phase1_policy, phase2_policy, s3, best_partitions = run_stage3(
        phase1_policy, phase2_policy, stage3_episodes,
        graph_dataset_size=min(10, graph_dataset_size*2)
    )
    phase3_policy, s4, novel_bounds = run_stage4(
        phase1_policy, phase2_policy, best_partitions,
        coeff_dim, stage4_episodes,
        graph_dataset_size=min(12, graph_dataset_size*3)
    )
    return (phase1_policy, phase2_policy, phase3_policy,
            {'stage1': s1, 'stage2': s2, 'stage3': s3, 'stage4': s4},
            novel_bounds)


def evaluate(phase1_policy, phase2_policy, phase3_policy,
             num_episodes=500, graph_dataset_size=5):
    """Evaluation across all phases."""
    env = PartitionBoundEnv(graph_dataset_size=graph_dataset_size, stage=4)
    results = defaultdict(lambda: {'p2_bounds':[], 'p3_bounds':[], 'novel':0})

    for episode in range(num_episodes):
        graph_tuple = random.choice(env.graph_dataset)
        nodes, edges, sessions = graph_tuple
        graph_name = identify_graph(nodes, edges, sessions)
        pb = _compute_partition_bound(nodes, edges, sessions)

        # Phase 1+2 rollout (standard bound)
        state = env.reset(fixed_graph=graph_tuple)
        state['edges'] = edges; state['sessions'] = sessions

        while env.current_phase == Phase.PHASE1:
            valid = env.get_valid_actions()
            if not valid: break
            action = phase1_policy.select_action(state, valid)
            state, _, done = env.step(action)
            state['edges'] = edges; state['sessions'] = sessions

        partition = [list(g) for g in env.partition] if env.partition else []
        state['nodes'] = nodes; state['sessions'] = sessions
        state['partition'] = partition

        while env.current_phase == Phase.PHASE2:
            valid = env.get_valid_actions()
            if not valid: break
            action = phase2_policy.select_action(state, valid)
            state, _, done = env.step(action)
            state['edges'] = edges; state['sessions'] = sessions
            if done: break

        p2_bound = abs(env._best_pool_bound() or pb * 2)
        results[graph_name]['p2_bounds'].append(p2_bound)

        # Phase 3 rollout (fractional bound)
        env.partition         = partition
        env.partition_weights = env.partition_weights or {}
        env._start_phase2()
        env._start_phase3()
        env.internal_per_part = env.internal_per_part or []

        state2 = env._get_state()
        state2['nodes']    = nodes; state2['edges'] = edges
        state2['sessions'] = sessions; state2['partition'] = partition

        while env.current_phase == Phase.PHASE3:
            valid = env.get_valid_actions()
            if not valid: break
            action = phase3_policy.select_action(state2, valid)
            state2, _, done = env.step(action)
            state2['nodes'] = nodes; state2['edges'] = edges
            state2['sessions'] = sessions; state2['partition'] = partition
            if done: break

        p3_bound = env.frac_pool.best_bound(
            len(sessions), len(edges), env.internal_per_part
        )
        if p3_bound == float('inf'): p3_bound = pb * 2
        results[graph_name]['p3_bounds'].append(p3_bound)
        if p3_bound < pb - 1e-8:
            results[graph_name]['novel'] += 1

    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Graph':<16} {'PB':>8} {'P2 avg':>8} {'P3 avg':>8} "
          f"{'P3 best':>8} {'Novel%':>8}")
    print(f"  {'-'*58}")
    for gname in sorted(results.keys()):
        r   = results[gname]
        pb2 = _compute_partition_bound(
            *next(t for t in env.graph_dataset if identify_graph(*t) == gname)
        )
        p2a = np.mean(r['p2_bounds'])
        p3a = np.mean(r['p3_bounds'])
        p3b = min(r['p3_bounds'])
        nv  = 100 * r['novel'] / max(len(r['p3_bounds']), 1)
        flag = " *** NOVEL ***" if p3b < pb2 - 1e-8 else ""
        print(f"  {gname:<16} {pb2:>8.4f} {p2a:>8.4f} {p3a:>8.4f} "
              f"{p3b:>8.4f} {nv:>7.1f}%{flag}")
    return dict(results)


if __name__ == "__main__":
    t0 = time.time()
    (phase1_policy, phase2_policy, phase3_policy,
     train_metrics, novel_bounds) = train(
        stage1_episodes=15000,   # Phase 2 proof calculus  — Tier 1 graphs (5)
        stage2_episodes=15000,   # Phase 1 partition learn — Tier 1 graphs (5)
        stage3_episodes=15000,   # Joint fine-tuning       — Tier 1+2 graphs (10)
        stage4_episodes=15000,   # Phase 3 fractional IO   — All graphs (11)
        graph_dataset_size=5
    )
    eval_results = evaluate(
        phase1_policy, phase2_policy, phase3_policy,
        num_episodes=500, graph_dataset_size=5
    )

    runtime = time.time() - t0
    print(f"\nTotal runtime: {runtime:.1f}s ({runtime/60:.1f} min)")

    all_metrics = {
        'train': {k: str(v) for k,v in train_metrics.items()},
        'eval': {k: str(v) for k,v in eval_results.items()},
        'novel_bounds': {k: str(v) for k,v in novel_bounds.items()},
        'runtime_s': runtime,
    }
    with open('training_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print("Metrics saved to training_metrics.json")

    # Summary file
    with open('training_summary.txt', 'w') as f:
        f.write(f"Training completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Runtime: {runtime:.1f}s ({runtime/60:.1f} min)\n\n")
        if novel_bounds:
            f.write("NOVEL INEQUALITIES FOUND:\n")
            for gn, (b, part, w, trace) in sorted(novel_bounds.items()):
                f.write(f"  {gn}: r <= {b:.6f}\n")
                f.write(f"  Trace: {trace[:300]}\n\n")
        else:
            f.write("No super-PB bounds found in this run.\n")
            f.write("Increase stage4_episodes or check CROSS_SUBMOD usage.\n")
    print("Summary saved to training_summary.txt")
    # --- Auto-generate plots ---
    print("\n--- Generating plots ---")
    import subprocess, sys
    try:
        subprocess.run([sys.executable, "plot_training.py"], check=True)
        print("Plots generated successfully.")
    except Exception as e:
        print(f"Plot generation failed: {e}")
        print("Run 'python plot_training.py' manually.")

    # --- Auto-generate graph visualization ---
    try:
        subprocess.run([sys.executable, "visualize_graphs.py"], check=True)
        print("Graph visualization generated.")
    except Exception as e:
        print(f"Graph visualization failed: {e}")

    # --- Auto git push ---
    print("\n--- Pushing to git ---")
    try:
        subprocess.run(["git", "add", "--all"], check=True)
        novel_tag = ""
        if novel_bounds:
            graphs_str = ", ".join(sorted(novel_bounds.keys()))
            novel_tag = f" - NOVEL BOUNDS: {graphs_str}"
        commit_msg = (
            f"Training run completed - "
            f"{time.strftime('%Y-%m-%d %H:%M')} - "
            f"runtime {runtime/60:.0f}min"
            f"{novel_tag}"
        )
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
        subprocess.run(["git", "push"], check=True)
        print("Git push completed successfully.")
    except Exception as e:
        print(f"Git push failed: {e}")
        print("Push manually with: git add --all && git commit -m 'training results' && git push")