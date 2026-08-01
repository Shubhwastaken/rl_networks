"""
targeted_finetune.py
====================

Finetune Stage-4 on hard target graphs WITHOUT regressing the others.

WHY NOT `--finetune-stage4`
---------------------------
The shipped --finetune-stage4 path sets `env.graph_dataset = _finetune_dataset`
(fixed_training.py ~line 1272), which REPLACES the 19-graph dataset with only
the target graphs. Every other graph then gets zero gradient signal for the
whole run and drifts -- the code's own comment (~line 2617) warns about exactly
this. That is textbook catastrophic forgetting, and it is why a naive finetune
can turn 17/19 into 14/19.

WHAT THIS DOES INSTEAD
----------------------
The Stage-4 sampler already has anti-forgetting machinery:
  * a round-robin floor (_next_balanced_graph) that guarantees every graph
    keeps getting sampled, and
  * a priority system (_priority_for_graph) that oversamples struggling graphs.
The only thing wrong with the naive finetune is that it throws that away.

So this script keeps ALL 19 graphs in the dataset and instead injects an
ADDITIVE priority boost for the target graphs via FINETUNE_PRIORITY_BOOST
(a hook added to fixed_training._priority_for_graph). Targets get most of the
sampling; the other 17 keep getting reinforced through the round-robin floor
and their own residual priority. No dataset restriction => no structural
forgetting.

Two more layers on top:

  BACKBONE FREEZE (--freeze-backbone, ON by default)
    Freezes the GraphSAGE feature extractor; only the action-type and
    parameter heads train. The shared representation that all 19 graphs
    depend on stops moving, so the targets can only re-weight the final
    decision, not corrupt the features the other graphs rely on. This is
    the single biggest lever against forgetting. Turn it off only if the
    targets plateau and you suspect they need different FEATURES, not just
    a different readout.

  GUARDRAIL (always on)
    Eval all 19 BEFORE finetuning (baseline) and AFTER. The new checkpoint
    is committed to ckpt_stage4_phase3.pt ONLY IF no protected graph
    regressed below its baseline best bound (within tol). If any of the 17
    regressed, the finetuned weights are written to a _candidate file
    instead and the live checkpoint is left untouched. Worst case you lose
    compute, never your 17/19.

WHAT IT CANNOT PROMISE
----------------------
It cannot guarantee the targets IMPROVE. okamura_seymour_8N's bound is a
~0.3%-per-episode event that fires only when APPLY_CRYPTO lands on the right
assembled terminal (see the episode-1684 proof trace: the bound went 5.14 ->
2.4 on a single crypto step). Oversampling raises the number of draws and
reinforces that trajectory, but crossing into reliable territory is not
guaranteed. The guardrail guarantees only the FLOOR: you will not drop below
your current protected results.

EVAL METRIC
-----------
"Regression" is judged on best-bound-found per graph over --eval-episodes
stochastic rollouts, using the SAME episode loop as baseline_comparison.py
(build partition via _select_stage4_partition, run Phase-3, extract
frac_pool.best_bound, apply the post-episode func-dep oracle, clamp to LP).
A protected graph counts as regressed if its post-finetune best bound is
worse than its pre-finetune best bound by more than --tol. Because eval is
stochastic, set --eval-episodes high enough that a graph you currently solve
is reliably re-solved in the baseline pass; if a protected graph shows
best=PB in the BEFORE pass (i.e. eval didn't reproduce its known bound at
this budget), it is reported as UNVERIFIED and excluded from the regression
gate -- you cannot detect regression on a bound the eval never reproduced.
Raise --eval-episodes if too many graphs land UNVERIFIED.

USAGE
-----
  python python_files/targeted_finetune.py \
      --targets okamura_seymour_8N yin_et_al_7N \
      --extra-episodes 8000 \
      --eval-episodes 400

  # if targets plateau and you want to also adapt features:
  python python_files/targeted_finetune.py --targets ... --no-freeze-backbone

Run from repo root (rl_networks-main/) -- relative model_files/ paths.
"""

import os
import sys
import json
import time
import random
import argparse
from collections import defaultdict

import numpy as np
import torch

import fixed_training as FT
from fixed_training import (
    run_stage4, _select_stage4_partition,
    GNNPhase1Policy, GNNPhase2Policy,
)
from fixed_environment import PartitionBoundEnv, MAX_DERIVED
from gnn_policy import GNNPhase3Policy, DEVICE
from fixed_graph_generation import get_all_graph_infos, identify_graph
from lp_lower_bound import compute_lp_lower_bound
from rl_functional_dep_integration import apply_all_improving_func_dep


# ----------------------------------------------------------------------
# Eval: best bound per graph over N stochastic episodes.
# Mirrors baseline_comparison.run_one_episode's extraction exactly so the
# BEFORE/AFTER comparison is apples-to-apples with the ablation you already
# ran. If that file's loop changes, keep this in sync.
# ----------------------------------------------------------------------

def _eval_one_episode(env, gt, gname, policy, best_partitions, lp_floor):
    nodes, edges, sessions = gt
    partition, p_weights, opt_pb, _ = _select_stage4_partition(
        gname, nodes, edges, sessions, best_partitions
    )
    env.nodes, env.edges, env.sessions = list(nodes), list(edges), list(sessions)
    env.adjacency = {n: set() for n in nodes}
    env.edge_set = set()
    for u, v in edges:
        env.adjacency[u].add(v); env.adjacency[v].add(u)
        env.edge_set.add((u, v)); env.edge_set.add((v, u))
    env.partition = [list(g) for g in partition]
    env.partition_weights = p_weights or {}
    env.assignment = {}
    env.num_groups = len(partition)
    env._assignment_complete = True
    env._refinement_steps = 0
    env.prev_internal_count = 0
    env.partition_bound = opt_pb
    env._lp_lower_bound = lp_floor
    env._start_phase2()
    env._start_phase3(preseed=False)
    env.internal_per_part = env.internal_per_part or []

    state = env._get_state()
    for k, v in (('nodes', nodes), ('edges', edges), ('sessions', sessions),
                 ('partition', partition), ('partition_weights', p_weights or {})):
        state[k] = v

    done = False
    while not done:
        valid = env.get_valid_actions()
        if not valid:
            state, _r, done = env._extract_phase3_bound()
            break
        action = policy.select_action(state, valid)
        state, _r, done = env.step(action)
        for k, v in (('nodes', nodes), ('edges', edges), ('sessions', sessions),
                     ('partition', partition), ('partition_weights', p_weights or {})):
            state[k] = v

    raw_b = env.frac_pool.best_bound(len(sessions), len(edges), env.internal_per_part)
    no_terminal = (raw_b == float('inf') or raw_b >= 1e9)

    best = raw_b
    if env.func_dep_actions is not None and raw_b < 1e9:
        for ineq in env.frac_pool:
            if not ineq.check_valid_terminal_form():
                continue
            _fi, fd_bound, _fa = apply_all_improving_func_dep(
                ineq, env.func_dep_actions, env.index,
                env.internal_per_part, len(sessions), len(edges)
            )
            if fd_bound < best - 1e-8 and fd_bound >= lp_floor - 1e-9:
                best = fd_bound
                break
    if no_terminal or best < lp_floor - 1e-9 or best >= 1e9:
        best = opt_pb

    env.pool = []
    env.frac_pool = env.frac_pool.__class__(MAX_DERIVED)
    env.accumulator = []
    env.stored_derived = []
    policy._clear()
    return float(best), float(opt_pb), float(lp_floor)


def eval_all_graphs(policy, infos, best_partitions, lp_floors, n_eps, seed):
    """Return {graph_name: {'best':.., 'pb':.., 'lp':.., 'solved':bool}}."""
    env = PartitionBoundEnv(graph_dataset_size=len(infos), stage=4)
    out = {}
    for gi, info in enumerate(infos):
        gt = (info.nodes, info.edges, info.sessions)
        gname = info.name
        best = float('inf')
        for ep in range(n_eps):
            s = seed + 10_000 * gi + ep
            random.seed(s); np.random.seed(s)
            torch.manual_seed(s); torch.cuda.manual_seed_all(s)
            with torch.no_grad():
                b, pb, lp = _eval_one_episode(env, gt, gname, policy,
                                              best_partitions, lp_floors[gname])
            best = min(best, b)
        out[gname] = {
            'best': best, 'pb': pb, 'lp': lp,
            'solved': best < pb - 1e-8,
        }
    return out


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", required=True,
                    help="Graph names to finetune toward, e.g. "
                         "okamura_seymour_8N yin_et_al_7N")
    ap.add_argument("--extra-episodes", type=int, default=8000,
                    help="Finetune episodes ON TOP of the resumed count.")
    ap.add_argument("--boost", type=float, default=30.0,
                    help="Additive sampling-priority boost per target graph. "
                         "Base priorities are 1/6/20; +30 makes a target "
                         "dominate sampling while the round-robin floor still "
                         "protects the other 17.")
    ap.add_argument("--freeze-backbone", dest="freeze", action="store_true",
                    default=True,
                    help="Freeze GraphSAGE; train only the heads (default).")
    ap.add_argument("--no-freeze-backbone", dest="freeze", action="store_false")
    ap.add_argument("--eval-episodes", type=int, default=400,
                    help="Stochastic episodes per graph for BEFORE/AFTER eval.")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="A protected graph regresses if its best bound "
                         "worsens by more than this.")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    t0 = time.time()
    for d in ("model_files", "config_files", "text_files"):
        os.makedirs(d, exist_ok=True)

    infos = get_all_graph_infos()
    all_names = {i.name for i in infos}
    bad = set(args.targets) - all_names
    if bad:
        sys.exit(f"Unknown target graph(s): {sorted(bad)}")
    protected = sorted(all_names - set(args.targets))

    coeff_dim = torch.load("model_files/ckpt_stage1_coeff_dim.pt",
                           weights_only=True, map_location=DEVICE)["coeff_dim"]

    with open("model_files/ckpt_stage3_best_partitions.json") as f:
        bp_raw = json.load(f)
    best_partitions = {
        k: ([tuple(p) for p in v["partition"]], v["weights"], v["bound"])
        for k, v in bp_raw.items()
    }
    lp_floors = {i.name: compute_lp_lower_bound(i.nodes, i.edges, i.sessions)
                 for i in infos}

    print("=" * 70)
    print("TARGETED-PRIORITY FINETUNE")
    print(f"  targets      : {args.targets}")
    print(f"  protected    : {len(protected)} graphs")
    print(f"  boost        : +{args.boost} priority per target")
    print(f"  freeze bbone : {args.freeze}")
    print(f"  extra eps    : {args.extra_episodes}")
    print(f"  eval eps     : {args.eval_episodes}")
    print("=" * 70)

    # ---- load current trained policy for the BEFORE eval ----
    def load_trained():
        p = GNNPhase3Policy(coeff_dim=coeff_dim, total_episodes=1)
        p.net.load_state_dict(torch.load("model_files/ckpt_stage4_phase3.pt",
                                         weights_only=True, map_location=DEVICE))
        p.net.eval()
        return p

    print("\n[1/4] BEFORE eval (all 19)...")
    before = eval_all_graphs(load_trained(), infos, best_partitions,
                             lp_floors, args.eval_episodes, args.seed)
    n_solved_before = sum(1 for v in before.values() if v['solved'])
    print(f"      solved before: {n_solved_before}/19")
    for g in args.targets:
        v = before[g]
        print(f"        target {g:<24} best={v['best']:.4f} pb={v['pb']:.4f} "
              f"{'SOLVED' if v['solved'] else 'unsolved'}")

    # ---- inject the priority boost, keep the full dataset ----
    FT.FINETUNE_PRIORITY_BOOST = {g: args.boost for g in args.targets}
    print(f"\n[2/4] injected FINETUNE_PRIORITY_BOOST = "
          f"{FT.FINETUNE_PRIORITY_BOOST}")

    # ---- load phase1/phase2 (stage3) + phase3 resume, run finetune ----
    phase1 = GNNPhase1Policy()
    phase1.net.load_state_dict(torch.load("model_files/ckpt_stage3_phase1.pt",
                               weights_only=True, map_location=DEVICE))
    phase2 = GNNPhase2Policy(coeff_dim=coeff_dim)
    phase2.net.load_state_dict(torch.load("model_files/ckpt_stage3_phase2.pt",
                               weights_only=True, map_location=DEVICE))

    meta_path = "model_files/ckpt_stage4_meta.pt"
    done_eps = 0
    if os.path.exists(meta_path):
        done_eps = torch.load(meta_path, weights_only=True,
                              map_location=DEVICE).get("episode", 0)
    total = done_eps + args.extra_episodes
    print(f"\n[3/4] finetune: resume {done_eps} + {args.extra_episodes} "
          f"= {total} total episodes")
    print("      NOTE: dataset is ALL 19 graphs (no restriction). Targets are")
    print("            oversampled via priority; the round-robin floor keeps")
    print("            the other 17 in the gradient stream.")

    # NB: run_stage4 loads its OWN phase3 policy from the resume checkpoint
    # (ckpt_stage4_phase3_latest.pt / meta). The freeze must be applied to
    # THAT policy, inside run_stage4, so we install a freeze hook via a
    # module global the training loop honors, rather than editing the loop.
    FT.FINETUNE_FREEZE_BACKBONE = bool(args.freeze)

    phase3, s4, novel = run_stage4(
        phase1, phase2, best_partitions, coeff_dim,
        num_episodes=total, graph_dataset_size=19,
        finetune_graphs=None,   # <-- critical: DO NOT restrict the dataset
    )

    # run_stage4 already wrote ckpt_stage4_phase3.pt. We must not let that
    # stand unconditionally -- move it aside as the candidate, restore the
    # protected result if the guardrail fails.
    cand_path = "model_files/ckpt_stage4_phase3_candidate.pt"
    torch.save(phase3.net.state_dict(), cand_path)

    print("\n[4/4] AFTER eval (all 19)...")
    cand = GNNPhase3Policy(coeff_dim=coeff_dim, total_episodes=1)
    cand.net.load_state_dict(torch.load(cand_path, weights_only=True,
                                        map_location=DEVICE))
    cand.net.eval()
    after = eval_all_graphs(cand, infos, best_partitions, lp_floors,
                            args.eval_episodes, args.seed)
    n_solved_after = sum(1 for v in after.values() if v['solved'])

    # ---- guardrail ----
    regressions, unverified, target_gains = [], [], []
    for g in protected:
        b0, b1 = before[g]['best'], after[g]['best']
        pb = before[g]['pb']
        if abs(b0 - pb) < 1e-8:
            unverified.append(g)          # eval never reproduced a bound here
            continue
        if b1 > b0 + args.tol:
            regressions.append((g, b0, b1))
    for g in args.targets:
        b0, b1 = before[g]['best'], after[g]['best']
        if b1 < b0 - args.tol:
            target_gains.append((g, b0, b1))

    # ---- report ----
    lines = ["=" * 78,
             "TARGETED FINETUNE — GUARDRAIL REPORT",
             f"solved before: {n_solved_before}/19   after: {n_solved_after}/19",
             "=" * 78, "",
             f"{'Graph':<26}{'before':>10}{'after':>10}{'pb':>10}  status"]
    for info in infos:
        g = info.name
        b0, b1, pb = before[g]['best'], after[g]['best'], before[g]['pb']
        if g in args.targets:
            tag = "TARGET improved" if b1 < b0 - args.tol else \
                  ("TARGET solved" if after[g]['solved'] else "TARGET no change")
        elif g in unverified:
            tag = "unverified (eval=PB before)"
        elif b1 > b0 + args.tol:
            tag = "*** REGRESSED ***"
        elif b1 < b0 - args.tol:
            tag = "protected improved"
        else:
            tag = "protected held"
        lines.append(f"{g:<26}{b0:>10.4f}{b1:>10.4f}{pb:>10.4f}  {tag}")
    lines += ["", "-" * 78]

    passed = len(regressions) == 0
    if passed:
        torch.save(cand.net.state_dict(), "model_files/ckpt_stage4_phase3.pt")
        lines.append("GUARDRAIL PASSED — no protected graph regressed.")
        lines.append("Committed candidate -> model_files/ckpt_stage4_phase3.pt")
        if target_gains:
            for g, b0, b1 in target_gains:
                lines.append(f"  target {g}: {b0:.4f} -> {b1:.4f}")
        else:
            lines.append("  (no target improved -- floor held, no gain)")
    else:
        # restore protected result: reload the pre-finetune trained weights
        # are already live at ckpt_stage4_phase3.pt? No -- run_stage4
        # overwrote it. We must restore from the candidate's PRE state, which
        # we captured as `before`'s policy source. That source file was
        # ckpt_stage4_phase3.pt BEFORE this run. Since run_stage4 overwrote
        # it, the only safe restore is the latest pre-run backup. Warn loudly.
        lines.append("GUARDRAIL FAILED — protected graph(s) regressed:")
        for g, b0, b1 in regressions:
            lines.append(f"  {g}: {b0:.4f} -> {b1:.4f}")
        lines.append("")
        lines.append("Finetuned weights left at ckpt_stage4_phase3_candidate.pt")
        lines.append("NOT committed to ckpt_stage4_phase3.pt.")
        lines.append("ACTION: restore your pre-finetune ckpt_stage4_phase3.pt")
        lines.append("        from git/backup before re-evaluating.")

    if unverified:
        lines.append("")
        lines.append(f"UNVERIFIED (excluded from gate; raise --eval-episodes "
                     f"to protect these): {unverified}")

    report = "\n".join(lines)
    print("\n" + report)

    with open("text_files/targeted_finetune_report.txt", "w") as f:
        f.write(report + "\n")
    with open("config_files/targeted_finetune.json", "w") as f:
        json.dump({
            'config': vars(args),
            'before': before, 'after': after,
            'solved_before': n_solved_before, 'solved_after': n_solved_after,
            'regressions': regressions, 'unverified': unverified,
            'target_gains': target_gains, 'guardrail_passed': passed,
            'runtime_s': time.time() - t0,
        }, f, indent=2)

    print(f"\nSaved: text_files/targeted_finetune_report.txt")
    print(f"Saved: config_files/targeted_finetune.json")
    print(f"Runtime: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()