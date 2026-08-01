"""
baseline_comparison.py
======================

The experiment that decides whether the RL is doing anything.

WHAT THIS ANSWERS
-----------------
Stage-4 novel-bound counts went 12 -> 15 -> 17 across three reward schemes
while greedy eval stayed at 0/19. Two explanations fit that trajectory
equally well:

  (A) The policy got better.
  (B) The policy explores more, so more stochastic rollouts = more lottery
      draws = more finds, with no improvement in the underlying policy.

You cannot distinguish (A) from (B) from the trained numbers alone. You
need a control. This script runs the identical Stage-4 episode loop with
the trained policy replaced by three controls, under identical seeds and
identical episode budgets, and reports the same metrics for each.

THE ARMS
--------
  trained       Phase-3 policy loaded from model_files/ckpt_stage4_phase3.pt
  untrained     Same architecture, freshly initialised, never trained.
                Separates "training did something" from "the architecture's
                inductive bias / action masking did something".
  random_typed  Uniform over valid ACTION TYPES, then uniform over the
                parameters within the chosen type. This mirrors the policy's
                own factorisation (type head -> parameter heads) and is the
                HONEST random baseline.
  random_flat   Uniform over the flat list of valid actions. This is the
                naive baseline and it is deliberately weak: the Phase-3
                valid-action list is dominated by FRACTIONAL_IO (up to 20
                cross pairs x 7 lambdas = 140 entries) versus ONE entry for
                DECLARE_TERMINAL, so a flat-uniform sampler almost never
                terminates properly. Reported for completeness. Do NOT
                use random_flat as your headline control -- beating it
                proves nothing.

THE ORACLE
----------
Critical design point. The post-episode functional-dependence oracle
(apply_all_improving_func_dep) fires on ANY episode that produced a valid
terminal form, regardless of which policy produced it. If the oracle is
what actually finds the bounds, then a random policy that stumbles into a
terminal gets exactly the same oracle treatment as the trained one, and
"novel-bound rate" would look similar for both -- for reasons that have
nothing to do with the policy.

So every episode here records TWO bounds:
    pre_oracle_bound   what the policy's own actions achieved, alone
    post_oracle_bound  after the greedy functional-dependence oracle runs

That gives the full 2x3 grid (oracle off / on) x (arm) from a SINGLE pass
per arm, not two. Reading the grid:

  pre-oracle:  trained >> random      -> the policy discovers bounds.
                                         Strongest paper. Claim it.
  pre-oracle:  trained ~= random
  post-oracle: trained >> random      -> the policy's contribution is
                                         TERMINAL CONSTRUCTION; the oracle
                                         finds the bounds. Defensible paper,
                                         but the claim is "RL-guided search",
                                         and you must say so plainly.
  post-oracle: trained ~= random      -> the RL is not contributing. Better
                                         to know this now than in review.

METRICS (per graph, per arm)
----------------------------
  terminal_rate       % of episodes that assembled a valid terminal form.
                      This is the thing the trained policy is supposed to be
                      good at and the thing greedy eval is failing at (~3%).
  novel_rate_pre      % of episodes with pre-oracle bound < PB (and >= LP)
  novel_rate_post     % of episodes with post-oracle bound < PB (and >= LP)
  oracle_fire_rate    % of episodes where the oracle strictly improved
  best_pre / best_post  best bound seen across all episodes in the arm

Rates are reported with Wilson 95% confidence intervals. A single point
estimate off 30 episodes on a ~3% event is noise, not an answer. Default
episode budget is 300/graph/arm; raise it with --episodes if the CIs
overlap on the comparison you care about.

SEEDING
-------
Before every episode, all three RNGs (random, numpy, torch) are reset to
seed = base_seed + 10_000*graph_idx + episode. That is the SAME seed for
the same (graph, episode) across all four arms, so environment-side
randomness (the 20-pair cross-pair subsample in _valid_phase3, graph
sampling, etc.) is identical across arms. Any difference between arms is
attributable to the action-selection policy, which is the only thing that
varies.

USAGE
-----
  python baseline_comparison.py                       # all 19 graphs, 300 eps/arm
  python baseline_comparison.py --episodes 1000
  python baseline_comparison.py --graphs paper_7N petersen_10N
  python baseline_comparison.py --arms trained random_typed
  python baseline_comparison.py --out config_files/baseline_comparison.json

Runtime scales as (#graphs x #arms x #episodes). Start with --episodes 100
on 2-3 graphs to sanity-check the runtime before committing to a full pass.
"""

import os
import sys
import json
import time
import math
import random
import argparse
from collections import defaultdict

import numpy as np
import torch

# NOTE: importing fixed_training pulls in its module-level seeding and the
# helper functions we deliberately REUSE rather than reimplement --
# _select_stage4_partition and _partition_bound_of in particular. Reusing
# them is the whole point: if this script picked partitions differently from
# Stage-4 training, the comparison would be against a different search space
# and would be worthless.
import fixed_training as FT
from fixed_training import _select_stage4_partition, _partition_bound_of

from fixed_environment import (
    PartitionBoundEnv, ActionType, Phase, _compute_partition_bound, MAX_DERIVED
)
from gnn_policy import GNNPhase3Policy, DEVICE
from fixed_graph_generation import get_all_graph_infos, identify_graph
from lp_lower_bound import compute_lp_lower_bound
from rl_functional_dep_integration import apply_all_improving_func_dep


# ----------------------------------------------------------------------
# Control policies
# ----------------------------------------------------------------------

class RandomTypedPolicy:
    """Uniform over valid action TYPES, then uniform over parameters within
    the chosen type.

    This mirrors the trained policy's own decision factorisation: it first
    picks an action type from a masked softmax over the 8 Phase-3 types,
    then picks parameters (node_u/node_v/lam, idx_i/idx_j, cut_idx,
    session_idx) from separate heads. A control that samples the same way
    but with uniform instead of learned distributions isolates exactly one
    variable -- the learned probabilities -- and nothing else.

    This is the baseline the trained policy must beat. If it cannot, the
    learned probabilities are worthless and no amount of framing fixes that.
    """
    name = "random_typed"

    def select_action(self, state, valid_actions, greedy=False):
        if not valid_actions:
            return {'type': ActionType.DECLARE_TERMINAL}
        by_type = defaultdict(list)
        for a in valid_actions:
            by_type[int(a['type'])].append(a)
        chosen_type = random.choice(list(by_type.keys()))
        return random.choice(by_type[chosen_type])

    def update(self, *a, **k):
        pass

    def _clear(self):
        pass


class RandomFlatPolicy:
    """Uniform over the flat valid-action list.

    Deliberately weak -- see the module docstring. Included so that nobody
    can accuse the comparison of choosing a convenient random baseline, and
    so the gap between random_flat and random_typed makes visible how much
    of any 'RL beats random' result is really 'action-type balancing beats
    an action list dominated by FRACTIONAL_IO'.
    """
    name = "random_flat"

    def select_action(self, state, valid_actions, greedy=False):
        if not valid_actions:
            return {'type': ActionType.DECLARE_TERMINAL}
        return random.choice(valid_actions)

    def update(self, *a, **k):
        pass

    def _clear(self):
        pass


def build_policy(arm, coeff_dim, ckpt_path):
    """Construct the policy for one arm."""
    if arm == "random_typed":
        return RandomTypedPolicy()
    if arm == "random_flat":
        return RandomFlatPolicy()

    pol = GNNPhase3Policy(coeff_dim=coeff_dim, total_episodes=1)
    if arm == "trained":
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Trained checkpoint not found: {ckpt_path}. "
                f"The 'trained' arm cannot run without it."
            )
        pol.net.load_state_dict(
            torch.load(ckpt_path, weights_only=True, map_location=DEVICE)
        )
        pol.net.eval()
    elif arm == "untrained":
        # Fresh init. Seeded below in run_arm so the init is reproducible.
        pol.net.eval()
    else:
        raise ValueError(f"Unknown arm: {arm}")
    return pol


# ----------------------------------------------------------------------
# One episode -- an exact mirror of the Stage-4 training episode loop,
# minus the policy update and the proof logger.
# ----------------------------------------------------------------------

def run_one_episode(env, graph_tuple, graph_name, policy,
                    best_partitions, lp_floor):
    """Run one Phase-3 episode. Returns a dict of per-episode outcomes.

    Deliberately duplicates run_stage4's inner loop (env field assignment,
    _start_phase2, _start_phase3(preseed=False), the action loop, the
    frac_pool.best_bound extraction, and the LP clamp) rather than calling
    into run_stage4, because run_stage4 is welded to training-only machinery
    (PPO updates, early stoppers, adaptive graph sampling, the proof logger,
    checkpointing). Any divergence between this loop and the training loop
    invalidates the comparison, so if run_stage4's episode body changes,
    THIS FUNCTION MUST BE UPDATED TO MATCH.
    """
    nodes, edges, sessions = graph_tuple

    partition, p_weights, opt_pb, _src = _select_stage4_partition(
        graph_name, nodes, edges, sessions, best_partitions
    )

    env.nodes    = list(nodes)
    env.edges    = list(edges)
    env.sessions = list(sessions)
    env.adjacency = {n: set() for n in nodes}
    env.edge_set  = set()
    for u, v in edges:
        env.adjacency[u].add(v); env.adjacency[v].add(u)
        env.edge_set.add((u, v)); env.edge_set.add((v, u))

    env.partition            = [list(g) for g in partition]
    env.partition_weights    = p_weights or {}
    env.assignment           = {}
    env.num_groups           = len(partition)
    env._assignment_complete = True
    env._refinement_steps    = 0
    env.prev_internal_count  = 0
    env.partition_bound      = opt_pb
    env._lp_lower_bound      = lp_floor

    env._start_phase2()
    env._start_phase3(preseed=False)
    env.internal_per_part = env.internal_per_part or []

    state = env._get_state()
    state['nodes']             = nodes
    state['edges']             = edges
    state['sessions']          = sessions
    state['partition']         = partition
    state['partition_weights'] = p_weights or {}

    done   = False
    steps  = 0
    action_counts = defaultdict(int)

    while not done:
        valid = env.get_valid_actions()
        if not valid:
            state, _reward, done = env._extract_phase3_bound()
            break
        action = policy.select_action(state, valid)
        action_counts[int(action['type'])] += 1
        state, _reward, done = env.step(action)
        state['nodes']             = nodes
        state['edges']             = edges
        state['sessions']          = sessions
        state['partition']         = partition
        state['partition_weights'] = p_weights or {}
        steps += 1

    # ---- bound extraction, pre-oracle ----
    raw_b = env.frac_pool.best_bound(
        len(sessions), len(edges), env.internal_per_part
    )
    no_terminal = (raw_b == float('inf') or raw_b >= 1e9)

    pre_b = raw_b
    # Same clamp Stage-4 applies: an LP-violating bound is invalid, not a
    # result. Treat it as PB so it cannot inflate the novel count.
    if no_terminal or pre_b < lp_floor - 1e-9:
        pre_b = opt_pb

    # ---- post-episode oracle (identical to run_stage4's block) ----
    post_b        = raw_b
    oracle_fired  = False
    if env.func_dep_actions is not None and raw_b < 1e9:
        for ineq in env.frac_pool:
            if not ineq.check_valid_terminal_form():
                continue
            fd_ineq, fd_bound, _fd_actions = apply_all_improving_func_dep(
                ineq, env.func_dep_actions, env.index,
                env.internal_per_part, len(sessions), len(edges)
            )
            if fd_bound < post_b - 1e-8 and fd_bound >= lp_floor - 1e-9:
                post_b       = fd_bound
                oracle_fired = True
                break   # one improvement per episode, matching training
    if no_terminal or post_b < lp_floor - 1e-9 or post_b >= 1e9:
        post_b = opt_pb

    novel_pre  = (pre_b  < opt_pb - 1e-8) and (pre_b  >= lp_floor - 1e-9)
    novel_post = (post_b < opt_pb - 1e-8) and (post_b >= lp_floor - 1e-9)

    # Release per-episode state so objects don't accumulate across episodes.
    env.pool          = []
    env.frac_pool     = env.frac_pool.__class__(MAX_DERIVED)
    env.accumulator   = []
    env.stored_derived = []
    policy._clear()

    return {
        'no_terminal':  bool(no_terminal),
        'pre_bound':    float(pre_b),
        'post_bound':   float(post_b),
        'novel_pre':    bool(novel_pre),
        'novel_post':   bool(novel_post),
        'oracle_fired': bool(oracle_fired),
        'pb':           float(opt_pb),
        'lp':           float(lp_floor),
        'steps':        int(steps),
        'actions':      {int(k): int(v) for k, v in action_counts.items()},
    }


# ----------------------------------------------------------------------
# Stats
# ----------------------------------------------------------------------

def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion.

    Used instead of the naive k/n +/- 1.96*sqrt(p(1-p)/n) because the normal
    approximation is badly wrong at the small proportions we care about here
    (a 3% terminal rate off 300 episodes), where it can produce negative
    lower bounds and understate the interval.
    """
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (100.0 * p, 100.0 * max(0.0, centre - half), 100.0 * min(1.0, centre + half))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Trained vs untrained vs random Phase-3 policy comparison."
    )
    ap.add_argument("--episodes", type=int, default=300,
                    help="Stochastic episodes per (graph, arm). Default 300. "
                         "At a ~3%% event rate, 30 is noise; 300 gives a "
                         "usable Wilson interval; 1000 if two arms are close.")
    ap.add_argument("--arms", nargs="+",
                    default=["trained", "untrained", "random_typed", "random_flat"],
                    choices=["trained", "untrained", "random_typed", "random_flat"])
    ap.add_argument("--graphs", nargs="+", default=None,
                    help="Restrict to named graphs. Default: all 19.")
    ap.add_argument("--seed", type=int, default=1234,
                    help="Base seed. Per-episode seed = base + 10000*graph_idx + ep, "
                         "identical across arms.")
    ap.add_argument("--ckpt", default="model_files/ckpt_stage4_phase3.pt")
    ap.add_argument("--partitions", default="model_files/ckpt_stage3_best_partitions.json")
    ap.add_argument("--coeff-dim-ckpt", default="model_files/ckpt_stage1_coeff_dim.pt")
    ap.add_argument("--out", default="config_files/baseline_comparison.json")
    args = ap.parse_args()

    t0 = time.time()

    infos = get_all_graph_infos()
    if args.graphs:
        wanted = set(args.graphs)
        missing = wanted - {i.name for i in infos}
        if missing:
            sys.exit(f"Unknown graph name(s): {sorted(missing)}")
        infos = [i for i in infos if i.name in wanted]

    coeff_dim = torch.load(args.coeff_dim_ckpt, weights_only=True,
                           map_location=DEVICE)["coeff_dim"]

    best_partitions = None
    if os.path.exists(args.partitions):
        with open(args.partitions) as f:
            raw = json.load(f)
        best_partitions = {
            k: ([tuple(p) for p in v["partition"]], v["weights"], v["bound"])
            for k, v in raw.items()
        }
        print(f"[partitions] loaded {len(best_partitions)} stored Stage-3 partitions")
    else:
        print(f"[partitions] {args.partitions} not found -- "
              f"exhaustive optimum will be used for every graph")

    print(f"[device] {DEVICE}   [coeff_dim] {coeff_dim}")
    print(f"[arms] {args.arms}")
    print(f"[graphs] {len(infos)}   [episodes/graph/arm] {args.episodes}")
    print(f"[total episodes] {len(infos) * len(args.arms) * args.episodes}\n")

    env = PartitionBoundEnv(graph_dataset_size=len(infos), stage=4)

    # LP floors, computed once.
    lp_floors = {}
    print(f"  {'Graph':<26} {'PB':>8} {'LP LB':>8} {'gap':>8}")
    print(f"  {'-'*52}")
    for info in infos:
        lp = compute_lp_lower_bound(info.nodes, info.edges, info.sessions)
        lp_floors[info.name] = lp
        print(f"  {info.name:<26} {info.optimal_bound:>8.4f} {lp:>8.4f} "
              f"{info.optimal_bound - lp:>8.4f}")
    print()

    # results[graph][arm] = list of episode dicts
    results = defaultdict(dict)

    for arm in args.arms:
        print(f"\n{'='*70}\nARM: {arm}\n{'='*70}")

        # Seed BEFORE constructing the policy so the untrained arm's random
        # init is reproducible across runs of this script.
        random.seed(args.seed); np.random.seed(args.seed)
        torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

        policy = build_policy(arm, coeff_dim, args.ckpt)
        is_net = hasattr(policy, "net")

        for gi, info in enumerate(infos):
            gt = (info.nodes, info.edges, info.sessions)
            gname = info.name
            eps = []
            for ep in range(args.episodes):
                # Identical seed for the same (graph, episode) across arms.
                s = args.seed + 10_000 * gi + ep
                random.seed(s); np.random.seed(s)
                torch.manual_seed(s); torch.cuda.manual_seed_all(s)

                if is_net:
                    with torch.no_grad():
                        r = run_one_episode(env, gt, gname, policy,
                                            best_partitions, lp_floors[gname])
                else:
                    r = run_one_episode(env, gt, gname, policy,
                                        best_partitions, lp_floors[gname])
                eps.append(r)

            results[gname][arm] = eps

            n     = len(eps)
            term  = sum(1 for e in eps if not e['no_terminal'])
            npre  = sum(1 for e in eps if e['novel_pre'])
            npost = sum(1 for e in eps if e['novel_post'])
            orc   = sum(1 for e in eps if e['oracle_fired'])
            bpre  = min(e['pre_bound']  for e in eps)
            bpost = min(e['post_bound'] for e in eps)
            print(f"  {gname:<26} term {100*term/n:5.1f}%  "
                  f"novel_pre {100*npre/n:5.1f}%  novel_post {100*npost/n:5.1f}%  "
                  f"oracle {100*orc/n:5.1f}%  best_pre {bpre:.4f}  best_post {bpost:.4f}")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    def agg(eps):
        n = len(eps)
        return {
            'n': n,
            'terminal':     sum(1 for e in eps if not e['no_terminal']),
            'novel_pre':    sum(1 for e in eps if e['novel_pre']),
            'novel_post':   sum(1 for e in eps if e['novel_post']),
            'oracle_fired': sum(1 for e in eps if e['oracle_fired']),
            'best_pre':     min(e['pre_bound']  for e in eps),
            'best_post':    min(e['post_bound'] for e in eps),
            'pb':           eps[0]['pb'],
            'lp':           eps[0]['lp'],
            'mean_steps':   sum(e['steps'] for e in eps) / n,
        }

    summary = {g: {a: agg(eps) for a, eps in per_arm.items()}
               for g, per_arm in results.items()}

    lines = []
    W = lines.append
    W("=" * 108)
    W("BASELINE COMPARISON — trained vs untrained vs random")
    W(f"episodes per (graph, arm): {args.episodes}   base seed: {args.seed}")
    W("=" * 108)

    for metric, key in (("TERMINAL-BUILD RATE", 'terminal'),
                        ("NOVEL RATE — PRE-ORACLE (policy alone)", 'novel_pre'),
                        ("NOVEL RATE — POST-ORACLE (policy + func-dep oracle)", 'novel_post')):
        W("")
        W("-" * 108)
        W(metric + "   [% of episodes, Wilson 95% CI]")
        W("-" * 108)
        W(f"  {'Graph':<26}" + "".join(f"{a:>26}" for a in args.arms))
        for g in summary:
            row = f"  {g:<26}"
            for a in args.arms:
                s = summary[g][a]
                p, lo, hi = wilson(s[key], s['n'])
                row += f"{p:>10.1f}% [{lo:4.1f},{hi:5.1f}]"
            W(row)

    W("")
    W("-" * 108)
    W("BEST BOUND FOUND   (PB = target to beat; LP = validity floor)")
    W("-" * 108)
    W(f"  {'Graph':<26}{'PB':>8}{'LP':>8}" +
      "".join(f"{a[:11]+'/pre':>18}{a[:11]+'/post':>18}" for a in args.arms))
    for g in summary:
        any_arm = summary[g][args.arms[0]]
        row = f"  {g:<26}{any_arm['pb']:>8.4f}{any_arm['lp']:>8.4f}"
        for a in args.arms:
            s = summary[g][a]
            row += f"{s['best_pre']:>18.4f}{s['best_post']:>18.4f}"
        W(row)

    # Headline: graphs where each arm found ANY novel bound in ANY episode.
    W("")
    W("-" * 108)
    W("GRAPHS WITH >=1 NOVEL BOUND (the '17/19'-style number, per arm)")
    W("-" * 108)
    for a in args.arms:
        gpre  = [g for g in summary if summary[g][a]['novel_pre']  > 0]
        gpost = [g for g in summary if summary[g][a]['novel_post'] > 0]
        W(f"  {a:<16} pre-oracle: {len(gpre):>2}/{len(summary)}    "
          f"post-oracle: {len(gpost):>2}/{len(summary)}")
    W("")
    W("READING THIS TABLE")
    W("  If trained's POST-ORACLE count is ~17/19 and random_typed's is also")
    W("  ~17/19, then 17/19 is a property of the oracle + episode budget, not")
    W("  of the policy, and the current paper claim does not survive review.")
    W("  The number that must separate is PRE-ORACLE novel rate, or failing")
    W("  that, TERMINAL-BUILD rate. If neither separates, the RL is decorative.")
    W("=" * 108)

    report = "\n".join(lines)
    print("\n" + report)

    os.makedirs("config_files", exist_ok=True)
    os.makedirs("text_files", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            'config': {
                'episodes': args.episodes, 'arms': args.arms,
                'seed': args.seed, 'graphs': [i.name for i in infos],
                'ckpt': args.ckpt,
            },
            'summary': summary,
            'runtime_s': time.time() - t0,
        }, f, indent=2)
    with open("text_files/baseline_comparison.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print(f"\nSaved: {args.out}")
    print(f"Saved: text_files/baseline_comparison.txt")
    print(f"Runtime: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()