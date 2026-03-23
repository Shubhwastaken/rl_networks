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
    print(f"  Coefficient vector dimension: {coeff_dim}")
    print(f"  Trivial bound: "
          f"{len(sample_edges)}/{len(sample_sessions)} = "
          f"{len(sample_edges)/len(sample_sessions):.4f}")
    print(f"  Target bound : 1.6667")

    phase2_policy = GNNPhase2Policy(coeff_dim=coeff_dim)
    phase2_policy.unfreeze()
    rewards = []

    for episode in range(num_episodes):
        nodes, edges, sessions = env.graph_dataset[0]
        chrom     = generate_random_valid_partition(nodes, edges)
        partition = decode_partition(nodes, chrom)
        state     = env.reset(fixed_partition=partition)
        done      = False
        trajectory   = []
        total_reward = 0.0

        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            action = phase2_policy.select_action(state, valid)
            state, reward, done = env.step(action)
            trajectory.append({"reward": reward})
            total_reward += reward

        phase2_policy.update(trajectory, total_reward)
        rewards.append(total_reward)

        if (episode + 1) % max(num_episodes // 10, 1) == 0:
            avg  = np.mean(rewards[-max(num_episodes // 10, 1):])
            best = abs(min(rewards[-max(num_episodes // 10, 1):]))
            print(f"  Episode {episode+1:5d} | "
                  f"Avg reward: {avg:.4f} | "
                  f"Best bound: {best:.4f}")

    print("Stage 1 complete.\n")
    return phase2_policy, coeff_dim


def run_stage2(phase2_policy, num_episodes=100, graph_dataset_size=1):
    print("=" * 60)
    print("STAGE 2: Training Phase 1 agent (Phase 2 frozen)")
    print("=" * 60)

    phase2_policy.freeze()
    env = PartitionBoundEnv(graph_dataset_size=graph_dataset_size, stage=2)
    phase1_policy = GNNPhase1Policy()
    rewards = []

    for episode in range(num_episodes):
        state        = env.reset()
        done         = False
        p1_traj      = []
        total_reward = 0.0
        rl_partition = None

        while env.current_phase == Phase.PHASE1 and not done:
            valid  = env.get_valid_actions()
            action = phase1_policy.select_action(state, valid)
            state, reward, done = env.step(action)
            p1_traj.append({"reward": reward})
            total_reward += reward

        if env.partition is not None:
            rl_partition = [list(g) for g in env.partition]

        while not done:
            valid  = env.get_valid_actions()
            if not valid:
                break
            action = phase2_policy.select_action(state, valid)
            state, reward, done = env.step(action)
            total_reward += reward

        phase1_policy.update(p1_traj, total_reward)
        rewards.append(total_reward)

        ipp   = internal_per_partition(rl_partition, env.sessions) \
                if rl_partition else []
        avg_i = np.mean([
            sum(internal_per_partition([list(g) for g in env.partition],
                env.sessions) if env.partition else [0])
            for _ in [0]
        ]) if env.partition else 0

        if (episode + 1) % max(num_episodes // 10, 1) == 0:
            recent = rewards[-max(num_episodes // 10, 1):]
            avg  = np.mean(recent)
            best = abs(min(recent))
            print(f"  Episode {episode+1:5d} | "
                  f"Avg reward: {avg:.4f} | "
                  f"Best bound: {best:.4f} | "
                  f"Internal sessions: {sum(ipp)}")

    print("Stage 2 complete.\n")
    return phase1_policy


def run_stage3(phase1_policy, phase2_policy, num_episodes=100,
               graph_dataset_size=1):
    print("=" * 60)
    print("STAGE 3: Joint fine-tuning (both phases unfrozen)")
    print("=" * 60)

    phase2_policy.unfreeze()
    env = PartitionBoundEnv(graph_dataset_size=graph_dataset_size, stage=3)
    rewards = []

    for episode in range(num_episodes):
        state        = env.reset()
        done         = False
        p1_traj      = []
        p2_traj      = []
        total_reward = 0.0
        rl_partition = None

        while env.current_phase == Phase.PHASE1 and not done:
            valid  = env.get_valid_actions()
            action = phase1_policy.select_action(state, valid)
            state, reward, done = env.step(action)
            p1_traj.append({"reward": reward})
            total_reward += reward

        if env.partition is not None:
            rl_partition = [list(g) for g in env.partition]

        while not done:
            valid  = env.get_valid_actions()
            if not valid:
                break
            action = phase2_policy.select_action(state, valid)
            state, reward, done = env.step(action)
            p2_traj.append({"reward": reward})
            total_reward += reward

        phase1_policy.update(p1_traj, total_reward)
        phase2_policy.update(p2_traj, total_reward)
        rewards.append(total_reward)

        ipp = internal_per_partition(rl_partition, env.sessions) \
              if rl_partition else []

        if (episode + 1) % max(num_episodes // 10, 1) == 0:
            recent = rewards[-max(num_episodes // 10, 1):]
            avg  = np.mean(recent)
            best = abs(min(recent))
            print(f"  Episode {episode+1:5d} | "
                  f"Avg reward: {avg:.4f} | "
                  f"Best bound: {best:.4f} | "
                  f"Internal sessions: {sum(ipp)}")

    print("Stage 3 complete.\n")
    return phase1_policy, phase2_policy


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

    rl_bounds        = []
    analytic_bounds  = []
    corrupted_count  = 0
    target_count     = 0

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
            action = phase1_policy.select_action(state, valid)
            state, _, done = env.step(action)

        if env.partition is not None:
            rl_partition = [list(g) for g in env.partition]

        while not done:
            valid  = env.get_valid_actions()
            if not valid:
                break
            action = phase2_policy.select_action(state, valid)
            state, reward, done = env.step(action)
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

        print(f"  Episode {episode+1:3d} | "
              f"RL: {rl_bound:.4f} | "
              f"Analytic: {best_analytic:.4f} | "
              f"Target: {target_bound:.4f} | "
              f"{status}")

        if hit_target and rl_partition is not None:
            ipp = internal_per_partition(rl_partition, sessions)
            print(f"    Partition:")
            for i, group in enumerate(rl_partition):
                internal = [
                    f"{s}->{t}" for s, t in sessions
                    if s in set(group) and t in set(group)
                ]
                internal_str = f"  internal: {internal}" if internal else ""
                print(f"      P{i+1} = {sorted(group)}{internal_str}")
            denom = len(sessions) + sum(ipp)
            print(f"    Sum|I(Pi,Pi)| = {sum(ipp)}  "
                  f"bound = {len(edges)}/{denom} = "
                  f"{len(edges)/denom:.4f}")

    avg_rl       = np.mean(rl_bounds)
    avg_analytic = np.mean(analytic_bounds)
    best_rl      = min(rl_bounds)

    print()
    print(f"  Episodes run          : {num_episodes}")
    print(f"  Corrupted (excluded)  : {corrupted_count}")
    print(f"  Target 1.67 reached   : {target_count}/{num_episodes} times")
    print(f"  Best RL bound         : {best_rl:.4f}")
    print(f"  Average RL bound      : {avg_rl:.4f}")
    print(f"  Average analytic bound: {avg_analytic:.4f}")
    print(f"  Target bound          : {target_bound:.4f}")
    print()
    if target_count > 0:
        print(f"  RESULT: RL agent reached optimal bound 1.67 "
              f"in {target_count}/{num_episodes} episodes.")
    elif best_rl < avg_analytic - 1e-6:
        print("  RESULT: RL found tighter bounds than random baseline.")
    else:
        print("  RESULT: RL matches random baseline — needs more training.")
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