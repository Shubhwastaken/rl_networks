"""
Task A: root-cause the fractional-IO false terminal.

Replays a live episode, dumping every pool/accumulator item after every
step together with an independent validity verdict from _entropy_check,
and reports the FIRST step at which an item appears that is not a valid
consequence of the node IO inequalities.
"""
import os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, torch
import fixed_training as FT
from fixed_environment import PartitionBoundEnv, Phase, ActionType
from gnn_policy import GNNPhase1Policy, GNNPhase2Policy, GNNPhase3Policy
import fixed_graph_generation as G
from _entropy_check import (max_symmetric_flow_with_arcs, entropy_vector,
                            violation, is_falsified)

FT.SUB_LP_FATAL = False
G._build_registry()
BY = {g.name: g for g in G.GRAPH_REGISTRY}
M, D = "model_files", FT.DEVICE
cd_ = torch.load(f"{M}/ckpt_stage1_coeff_dim.pt", weights_only=True, map_location=D)["coeff_dim"]
p1 = GNNPhase1Policy(); p1.net.load_state_dict(torch.load(f"{M}/ckpt_stage3_phase1.pt", weights_only=True, map_location=D))
p2 = GNNPhase2Policy(coeff_dim=cd_); p2.net.load_state_dict(torch.load(f"{M}/ckpt_stage3_phase2.pt", weights_only=True, map_location=D))
p3 = GNNPhase3Policy(coeff_dim=cd_, total_episodes=30000); p3.net.load_state_dict(torch.load(f"{M}/ckpt_stage4_phase3.pt", weights_only=True, map_location=D))
BP = json.load(open(f"{M}/ckpt_stage3_best_partitions.json"))
BP = {k: ([tuple(x) for x in v["partition"]], v["weights"], v["bound"]) for k, v in BP.items()}


def audit(name, max_trials=80, verbose_first_bad=True):
    g = BY[name]
    nodes, edges, sessions = g.nodes, g.edges, g.sessions
    r, flow = max_symmetric_flow_with_arcs(nodes, edges, sessions)
    part = BP[name][0] if name in BP else g.optimal_partition
    part = [list(p) for p in part]

    env = PartitionBoundEnv(graph_dataset_size=19, stage=4)
    random.seed(FT.SEED); np.random.seed(FT.SEED); torch.manual_seed(FT.SEED)

    for trial in range(max_trials):
        env.nodes = list(nodes); env.edges = list(edges); env.sessions = list(sessions)
        env.partition = [list(p) for p in part]; env.partition_weights = {}
        env.partition_bound = g.optimal_bound; env._lp_lower_bound = r
        env.index = None
        env.adjacency = {n: set() for n in nodes}; env.edge_set = set()
        for u, v in edges:
            env.adjacency[u].add(v); env.adjacency[v].add(u)
            env.edge_set.add((u, v)); env.edge_set.add((v, u))
        env._start_phase2(); env._start_phase3(preseed=False)
        env.internal_per_part = env.internal_per_part or []
        h = entropy_vector(env.index, nodes, edges, sessions, r, flow)

        st = env._get_state()
        st.update(nodes=nodes, edges=edges, sessions=sessions,
                  partition=part, partition_weights={})
        hist, first_bad = [], None
        step = 0
        while env.current_phase == Phase.PHASE3:
            valid = env.get_valid_actions()
            if not valid: break
            a = p3.select_action(st, valid, greedy=False)
            st, _, done = env.step(a)
            st.update(nodes=nodes, edges=edges, sessions=sessions,
                      partition=part, partition_weights={})
            step += 1
            items = ([("pool", i, x) for i, x in enumerate(env.frac_pool)] +
                     [("acc", i, x) for i, x in enumerate(env.accumulator)] +
                     [("stored", i, x) for i, x in enumerate(env.stored_derived)])
            bad = [(w, i, x) for w, i, x in items if is_falsified(x, h)]
            hist.append((step, ActionType(a.get("type")).name, a, len(items), len(bad)))
            if bad and first_bad is None:
                first_bad = (step, ActionType(a.get("type")).name, dict(a), bad[0], h)
                if verbose_first_bad:
                    return dict(graph=name, r=r, trial=trial, step=step,
                                action=ActionType(a.get("type")).name, action_raw=dict(a),
                                where=bad[0][0], idx=bad[0][1],
                                ineq=bad[0][2], viol=violation(bad[0][2], h),
                                hist=hist, env=env, h=h)
            if done: break
        env.pool = []; env.accumulator = []; env.stored_derived = []
        env.frac_pool = env.frac_pool.__class__(64)
    return None


if __name__ == "__main__":
    targets = sys.argv[1:] or ["diamond_6N"]
    out = {}
    for t in targets:
        res = audit(t)
        if res is None:
            print(f"\n{t}: no falsified pool item found in the trials run")
            out[t] = None
            continue
        print("\n" + "=" * 100)
        print(f"{t}: FIRST INVALID INEQUALITY  (achievable r = {res['r']})")
        print("=" * 100)
        print(f"  trial {res['trial']}, step {res['step']}, action {res['action']}")
        print(f"  action_raw : {res['action_raw']}")
        print(f"  located in : {res['where']}[{res['idx']}]")
        print(f"  violation  : {res['viol']:+.6f}   (>0 means FALSE)")
        print(f"  inequality : {repr(res['ineq']).encode('ascii','replace').decode()}")
        print("\n  preceding steps:")
        for s, nm, a, ni, nb in res["hist"][-12:]:
            print(f"    step {s:3d}  {nm:22s} items={ni:3d} falsified={nb}")
        out[t] = {"step": res["step"], "action": res["action"],
                  "violation": res["viol"],
                  "ineq": repr(res["ineq"]).encode("ascii", "replace").decode()}
    json.dump(out, open("config_files/_taskA.json", "w"), indent=1)
