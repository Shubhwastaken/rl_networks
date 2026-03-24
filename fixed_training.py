import random
import numpy as np
from collections import defaultdict

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

from fixed_environment import PartitionBoundEnv, ActionType, Phase
from partition import generate_random_valid_partition, decode_partition
from gnn_policy import GNNPhase1Policy, GNNPhase2Policy
from fixed_base_inequality_generator import internal_per_partition


# -----------------------------------------------------------------------
# Stronger shaped reward for Phase 1
# -----------------------------------------------------------------------

INTERNAL_SESSION_REWARD = 1.0   # was 0.01 — much stronger signal


def _partition_str(partition, sessions):
    """Returns a readable string showing partition groups and internal sessions."""
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
    """Returns compact action count string."""
    order = ["ADD_TO_ACCUMULATOR", "APPLY_SUBMODULARITY",
             "APPLY_PROOF2", "STORE_AND_RESET",
             "COMBINE_STORED", "DECLARE_TERMINAL"]
    short = {"ADD_TO_ACCUMULATOR": "ADD",
             "APPLY_SUBMODULARITY": "SUB",
             "APPLY_PROOF2": "P2",
             "STORE_AND_RESET": "STO",
             "COMBINE_STORED": "CMB",
             "DECLARE_TERMINAL": "TRM"}
    parts = []
    for a in order:
        c = action_counts.get(a, 0)
        if c > 0:
            parts.append(f"{short[a]}:{c}")
    return " ".join(parts)


# -----------------------------------------------------------------------
# Modified environment step to use stronger reward
# -----------------------------------------------------------------------

def run_episode_phase1(env, phase1_policy, sessions, print_partition=False):
    """
    Runs Phase 1 of one episode.
    Returns (state_after_p1, trajectory, partition, internal_count).
    Uses stronger shaped reward.
    """
    state        = env._get_state()
    done         = False
    trajectory   = []
    prev_internal = 0

    # inject edges into state so GNN can use real adjacency
    state["edges"] = env.edges

    while env.current_phase == Phase.PHASE1 and not done:
        valid  = env.get_valid_actions()
        action = phase1_policy.select_action(state, valid)
        state, _, done = env.step(action)

        # inject edges into every state
        state["edges"] = env.edges

        # compute stronger shaped reward
        if env.partition is not None:
            cur_internal = sum(
                internal_per_partition([list(g) for g in env.partition],
                                       sessions)
            )
        else:
            cur_internal = env._count_current_internal()

        newly  = cur_internal - prev_internal
        reward = INTERNAL_SESSION_REWARD * newly
        prev_internal = cur_internal
        trajectory.append({"reward": reward})

    partition    = [list(g) for g in env.partition] if env.partition else []
    ipp          = internal_per_partition(partition, sessions)
    int_count    = sum(ipp)

    if print_partition:
        print(f"    Partition: {_partition_str(partition, sessions)}")
        print(f"    Internal sessions: {int_count}  "
              f"Formula bound: {10/(3+int_count):.4f}")

    return state, trajectory, partition, int_count


# -----------------------------------------------------------------------
# Stage 1
# -----------------------------------------------------------------------

def run_stage1(num_episodes=100, graph_dataset_size=1):
    print("=" * 60)
    print("STAGE 1: Training Phase 2 agent on fixed partitions")
    print("=" * 60)

    env = PartitionBoundEnv(graph_dataset_size=graph_dataset_size, stage=1)

    from fixed_inequality import EntropyIndex
    sample_nodes, sample_edges, sample_sessions = env.graph_dataset[0]
    chrom     = generate_random_valid_partition(sample_nodes, sample_edges)
    partition = decode_partition(sample_nodes, chrom)
    idx       = EntropyIndex(partitions=partition, nodes=sample_nodes,
                             edges=sample_edges, sessions=sample_sessions)
    coeff_dim = idx.dim

    print(f"  Graph  : {len(sample_nodes)} nodes, "
          f"{len(sample_edges)} edges, "
          f"{len(sample_sessions)} sessions")
    print(f"  Coeff dim: {coeff_dim}  |  "
          f"Trivial bound: {len(sample_edges)/len(sample_sessions):.4f}  |  "
          f"Target: 1.6667")

    phase2_policy = GNNPhase2Policy(coeff_dim=coeff_dim)
    phase2_policy.unfreeze()
    rewards       = []
    action_totals = defaultdict(int)

    print(f"\n  {'Ep':>6} | {'AvgRew':>8} | {'BestBnd':>8} | "
          f"{'Actions (last 10%)':>30}")
    print(f"  {'-'*65}")

    for episode in range(num_episodes):
        nodes, edges, sessions = env.graph_dataset[0]
        chrom     = generate_random_valid_partition(nodes, edges)
        partition = decode_partition(nodes, chrom)
        ipp       = internal_per_partition(partition, sessions)
        state     = env.reset(fixed_partition=partition)
        state["edges"] = edges
        done          = False
        trajectory    = []
        action_counts = defaultdict(int)
        total_reward  = 0.0

        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            action = phase2_policy.select_action(state, valid)
            aname  = {1:"ADD_TO_ACCUMULATOR", 2:"APPLY_SUBMODULARITY",
                      3:"APPLY_PROOF2", 4:"STORE_AND_RESET",
                      5:"COMBINE_STORED", 6:"DECLARE_TERMINAL"}.get(
                          int(action["type"]), "?")
            action_counts[aname] += 1
            action_totals[aname] += 1
            state, reward, done = env.step(action)
            state["edges"] = edges
            trajectory.append({"reward": reward})
            total_reward += reward

        phase2_policy.update(trajectory, total_reward)
        rewards.append(total_reward)

        if (episode + 1) % max(num_episodes // 10, 1) == 0:
            n    = max(num_episodes // 10, 1)
            avg  = np.mean(rewards[-n:])
            best = abs(min(rewards[-n:]))
            asumm = _action_summary(action_counts)
            print(f"  {episode+1:>6} | {avg:>8.4f} | {best:>8.4f} | {asumm:>30}")

    print(f"\n  Total action distribution across Stage 1:")
    for a, c in sorted(action_totals.items(), key=lambda x: -x[1]):
        total = sum(action_totals.values())
        print(f"    {a:<25}: {c:5d}  ({100*c/total:.1f}%)")

    print("\nStage 1 complete.\n")
    return phase2_policy, coeff_dim


# -----------------------------------------------------------------------
# Stage 2
# -----------------------------------------------------------------------

def run_stage2(phase2_policy, num_episodes=100, graph_dataset_size=1):
    print("=" * 60)
    print("STAGE 2: Training Phase 1 agent (Phase 2 frozen)")
    print("=" * 60)
    print("  Phase 1 learns to build partitions with high internal sessions")
    print("  Target: internal sessions = 3 (all source-sink pairs grouped)\n")

    phase2_policy.freeze()
    env = PartitionBoundEnv(graph_dataset_size=graph_dataset_size, stage=2)
    phase1_policy = GNNPhase1Policy()
    rewards   = []
    internals = []

    nodes, edges, sessions = env.graph_dataset[0]

    print(f"  {'Ep':>6} | {'AvgRew':>8} | {'BestBnd':>8} | "
          f"{'AvgInt':>7} | Partition (every 10th ep)")
    print(f"  {'-'*75}")

    for episode in range(num_episodes):
        env.reset()
        show = (episode + 1) % max(num_episodes // 10, 1) == 0

        # temperature decays from 2.0 to 1.0 for exploration
        temperature = max(1.0, 2.0 - 2.0 * episode / max(num_episodes, 1))

        # Phase 1 with bound-improvement reward
        p1_traj       = []
        state         = env._get_state()
        state["edges"]       = edges
        state["sessions"]    = sessions
        state["temperature"] = temperature
        done          = False

        while env.current_phase == Phase.PHASE1 and not done:
            valid  = env.get_valid_actions()
            state["temperature"] = temperature
            action = phase1_policy.select_action(state, valid)
            state, reward, done = env.step(action)
            state["edges"]    = edges
            state["sessions"] = sessions
            p1_traj.append({"reward": reward})  # reward from env (bound improvement)

        rl_partition = [list(g) for g in env.partition] if env.partition else []
        ipp          = internal_per_partition(rl_partition, sessions)
        int_count    = sum(ipp)
        internals.append(int_count)

        # Phase 2 frozen evaluator
        total_reward = sum(t["reward"] for t in p1_traj)
        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            action = phase2_policy.select_action(state, valid)
            state, reward, done = env.step(action)
            state["edges"]    = edges
            state["sessions"] = sessions
            total_reward += reward

        phase1_policy.update(p1_traj, total_reward)
        rewards.append(total_reward)

        if show:
            n    = max(num_episodes // 10, 1)
            avg  = np.mean(rewards[-n:])
            best = abs(min(rewards[-n:]))
            avgi = np.mean(internals[-n:])
            pstr = _partition_str(rl_partition, sessions)
            print(f"  {episode+1:>6} | {avg:>8.4f} | {best:>8.4f} | "
                  f"{avgi:>7.2f} | {pstr}")

    # summary
    print(f"\n  Internal session distribution across Stage 2:")
    for v in range(4):
        cnt = internals.count(v)
        pct = 100 * cnt / len(internals)
        print(f"    {v} internal sessions: {cnt:4d} episodes ({pct:.1f}%)")

    print("\nStage 2 complete.\n")
    return phase1_policy


# -----------------------------------------------------------------------
# Stage 3
# -----------------------------------------------------------------------

def run_stage3(phase1_policy, phase2_policy, num_episodes=100,
               graph_dataset_size=1):
    print("=" * 60)
    print("STAGE 3: Joint fine-tuning (both phases unfrozen)")
    print("=" * 60)

    phase2_policy.unfreeze()
    env = PartitionBoundEnv(graph_dataset_size=graph_dataset_size, stage=3)
    rewards   = []
    internals = []

    nodes, edges, sessions = env.graph_dataset[0]

    print(f"  {'Ep':>6} | {'AvgRew':>8} | {'BestBnd':>8} | "
          f"{'AvgInt':>7} | P2 Actions (last 10%)")
    print(f"  {'-'*75}")

    for episode in range(num_episodes):
        env.reset()
        show = (episode + 1) % max(num_episodes // 10, 1) == 0

        # Phase 1
        p1_traj       = []
        prev_internal = 0
        state         = env._get_state()
        state["edges"] = edges
        done          = False
        p2_action_counts = defaultdict(int)
        total_reward  = 0.0

        temperature = max(1.0, 2.0 - 2.0 * episode / max(num_episodes, 1))

        while env.current_phase == Phase.PHASE1 and not done:
            valid  = env.get_valid_actions()
            state["temperature"] = temperature
            state["sessions"]    = sessions
            action = phase1_policy.select_action(state, valid)
            state, reward, done = env.step(action)
            state["edges"]    = edges
            state["sessions"] = sessions
            p1_traj.append({"reward": reward})
            total_reward += reward

        rl_partition = [list(g) for g in env.partition] if env.partition else []
        ipp          = internal_per_partition(rl_partition, sessions)
        int_count    = sum(ipp)
        internals.append(int_count)

        # Phase 2
        p2_traj      = []

        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            action = phase2_policy.select_action(state, valid)
            aname  = {1:"ADD_TO_ACCUMULATOR", 2:"APPLY_SUBMODULARITY",
                      3:"APPLY_PROOF2", 4:"STORE_AND_RESET",
                      5:"COMBINE_STORED", 6:"DECLARE_TERMINAL"}.get(
                          int(action["type"]), "?")
            p2_action_counts[aname] += 1
            state, reward, done = env.step(action)
            state["edges"]    = edges
            state["sessions"] = sessions
            p2_traj.append({"reward": reward})
            total_reward += reward

        phase1_policy.update(p1_traj, total_reward)
        phase2_policy.update(p2_traj, total_reward)
        rewards.append(total_reward)

        # print combination log when agent finds a notably good bound
        if abs(total_reward) <= 2.0 and int_count >= 2:
            clog = state.get('combination_log', [])
            print(f"  ** Ep {episode+1}: bound={abs(total_reward):.4f} "
                  f"internal={int_count} "
                  f"partition={_partition_str(rl_partition, sessions)}")
            for step in clog:
                if step.get('action') == 'PAIRWISE':
                    print(f"       Pairwise ineq[{step['idx_i']}]+"
                          f"ineq[{step['idx_j']}] "
                          f"-> Y_I union: {step['yi_union']}")
                elif step.get('action') == 'PROOF2':
                    print(f"       Proof2 (all-at-once)")

        if show:
            n    = max(num_episodes // 10, 1)
            avg  = np.mean(rewards[-n:])
            best = abs(min(rewards[-n:]))
            avgi = np.mean(internals[-n:])
            asumm = _action_summary(p2_action_counts)
            print(f"  {episode+1:>6} | {avg:>8.4f} | {best:>8.4f} | "
                  f"{avgi:>7.2f} | {asumm}")

    print(f"\n  Internal session distribution across Stage 3:")
    for v in range(4):
        cnt = internals.count(v)
        pct = 100 * cnt / len(internals)
        print(f"    {v} internal sessions: {cnt:4d} episodes ({pct:.1f}%)")

    print("\nStage 3 complete.\n")
    return phase1_policy, phase2_policy


# -----------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------

def _verify_bound(rl_bound, rl_partition, edges, sessions):
    if rl_partition is None:
        return False, float('inf')
    ipp   = internal_per_partition(rl_partition, sessions)
    denom = len(sessions) + sum(ipp)
    if denom <= 0:
        return False, float('inf')
    formula_bound = len(edges) / denom
    return rl_bound >= formula_bound - 1e-3, formula_bound


def evaluate(phase1_policy, phase2_policy, num_episodes=100,
             graph_dataset_size=1):
    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)

    env = PartitionBoundEnv(graph_dataset_size=graph_dataset_size, stage=3)
    phase2_policy.unfreeze()

    nodes, edges, sessions = env.graph_dataset[0]
    target_bound = 10 / 6

    rl_bounds       = []
    analytic_bounds = []
    corrupted_count = 0
    target_count    = 0

    for episode in range(num_episodes):
        best_analytic = float('inf')
        for _ in range(10):
            chrom     = generate_random_valid_partition(nodes, edges)
            partition = decode_partition(nodes, chrom)
            ipp       = internal_per_partition(partition, sessions)
            denom     = len(sessions) + sum(ipp)
            bound     = len(edges) / denom if denom > 0 else float('inf')
            if bound < best_analytic:
                best_analytic = bound
        analytic_bounds.append(best_analytic)

        state        = env.reset()
        done         = False
        last_reward  = -best_analytic
        rl_partition = None

        while env.current_phase == Phase.PHASE1 and not done:
            valid  = env.get_valid_actions()
            state["temperature"] = 1.0   # no exploration at eval time
            state["sessions"]    = sessions
            action = phase1_policy.select_action(state, valid)
            state, _, done = env.step(action)
            state["edges"]    = edges
            state["sessions"] = sessions

        if env.partition is not None:
            rl_partition = [list(g) for g in env.partition]

        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            action = phase2_policy.select_action(state, valid)
            state, reward, done = env.step(action)
            state["edges"]    = edges
            state["sessions"] = sessions
            if done:
                last_reward = reward

        rl_bound = abs(last_reward)
        is_valid, formula_bound = _verify_bound(
            rl_bound, rl_partition, edges, sessions
        )

        if not is_valid:
            corrupted_count += 1
            rl_bound = formula_bound

        rl_bounds.append(rl_bound)

        beat_analytic = rl_bound < best_analytic - 1e-6
        hit_target    = abs(rl_bound - target_bound) <= 1e-3

        if hit_target:
            target_count += 1

        status = ""
        if hit_target:
            status = "TARGET REACHED"
        elif beat_analytic:
            status = "BEAT ANALYTIC"

        ipp_str = ""
        if rl_partition:
            ipp = internal_per_partition(rl_partition, sessions)
            ipp_str = f"int={sum(ipp)}"

        print(f"  Ep {episode+1:3d} | "
              f"RL: {rl_bound:.4f} | "
              f"Analytic: {best_analytic:.4f} | "
              f"{ipp_str:6s} | {status}")

        if hit_target and rl_partition is not None:
            ipp = internal_per_partition(rl_partition, sessions)
            print(f"    {_partition_str(rl_partition, sessions)}")
            # show combination sequence that produced this bound
            clog = state.get("combination_log", [])
            if clog:
                print(f"    Combination sequence ({len(clog)} steps):")
                for step in clog:
                    if step["action"] == "PAIRWISE":
                        print(f"      Step {step['step']}: PAIRWISE "
                              f"ineq[{step['idx_i']}] + ineq[{step['idx_j']}] "
                              f"-> Y_I coeff: {step['yi_a']} + {step['yi_b']} "
                              f"-> union Y_I: {step['yi_union']}")
                    else:
                        print(f"      Step {step['step']}: PROOF2 (all-at-once)")

    avg_rl       = np.mean(rl_bounds)
    avg_analytic = np.mean(analytic_bounds)
    best_rl      = min(rl_bounds)

    print()
    print(f"  Episodes         : {num_episodes}")
    print(f"  Corrupted        : {corrupted_count}")
    print(f"  Target reached   : {target_count}/{num_episodes}")
    print(f"  Best RL bound    : {best_rl:.4f}")
    print(f"  Avg RL bound     : {avg_rl:.4f}")
    print(f"  Avg analytic     : {avg_analytic:.4f}")
    print()
    if target_count > 0:
        print(f"  RESULT: RL reached 1.67 in {target_count}/{num_episodes} episodes.")
    elif best_rl < avg_analytic - 1e-6:
        print("  RESULT: RL beats random baseline but not yet at 1.67.")
    else:
        print("  RESULT: Needs more training.")
    print("=" * 60)


def train(stage1_episodes=100, stage2_episodes=100,
          stage3_episodes=100, graph_dataset_size=1):
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
        stage1_episodes=100,
        stage2_episodes=100,
        stage3_episodes=100,
        graph_dataset_size=1
    )
    evaluate(
        phase1_policy, phase2_policy,
        num_episodes=100,
        graph_dataset_size=1
    )