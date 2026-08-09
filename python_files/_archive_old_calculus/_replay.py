"""
Phase 2, the part eval cannot answer: replay every claimed novel episode
under the CORRECTED proof calculus.

The cached eval_results.json shows greedy_best_bound = null for all 19
graphs BOTH before and after this fix -- the eval rollout never assembles
a terminal form. The "16/19 novel" claim comes from Stage-4 TRAINING
episodes recorded in stage4_proof_log.json, not from eval. Retraining is
forbidden, so the only faithful way to ask "does this bound survive?" is
to replay the exact logged action sequence through the corrected
environment and compare.

Each retained novel episode carries its graph, partition, and the full
ordered list of action_raw dicts. Replaying that sequence is
deterministic: no policy is consulted, so the result isolates the effect
of the operator fix from any policy or RNG difference.

Outputs, per graph, the best surviving bound over that graph's novel
episodes, the operations that produced it (1h provenance), and any
soundness assertion that tripped.
"""
import os, sys, json, traceback
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import numpy as np
import torch
import fixed_training as FT
import functional_dependence as FD
from fixed_environment import PartitionBoundEnv, Phase
from lp_lower_bound import compute_lp_lower_bound

FT.SUB_LP_FATAL = False

log = json.load(open("config_files/stage4_proof_log.json"))
novel = [e for e in log if isinstance(e, dict)
         and e.get("summary", {}).get("is_novel")]
cp = {d["name"]: d for d in json.load(open("config_files/_cpsat_pb.json"))}

env = PartitionBoundEnv(graph_dataset_size=19, stage=4)
out = defaultdict(list)

for ep in novel:
    g = ep["graph_name"]
    nodes = list(ep["nodes"])
    edges = [tuple(e) for e in ep["edges"]]
    sessions = [tuple(s) for s in ep["sessions"]]
    partition = [list(p) for p in ep["partition"]]
    claimed = ep["summary"]["best_bound"]
    lp = compute_lp_lower_bound(nodes, edges, sessions)

    rec = {"episode": ep["episode"], "claimed": claimed, "lp": lp,
           "replayed": None, "mechanisms": [], "op_counts": {},
           "assertion": None, "steps_applied": 0}
    try:
        env.nodes = list(nodes); env.edges = list(edges)
        env.sessions = list(sessions)
        env.partition = [list(p) for p in partition]
        env.partition_weights = {}
        env.partition_bound = cp[g]["pb"]
        env._lp_lower_bound = lp
        env.index = None
        env.adjacency = {n: set() for n in nodes}
        env.edge_set = set()
        for u, v in edges:
            env.adjacency[u].add(v); env.adjacency[v].add(u)
            env.edge_set.add((u, v)); env.edge_set.add((v, u))
        env._start_phase2()
        env._start_phase3(preseed=False)
        env.internal_per_part = env.internal_per_part or []

        for s in ep.get("steps", []):
            if env.current_phase != Phase.PHASE3:
                break
            a = s.get("action_raw")
            if not a:
                continue
            try:
                _, _, done = env.step(dict(a))
                rec["steps_applied"] += 1
            except FD.UnsoundDerivationError as e:
                rec["assertion"] = f"{type(e).__name__}: {str(e).splitlines()[0]}"
                break
            if done:
                break

        b, ineq = env.frac_pool.best_terminal(
            len(sessions), len(edges), env.internal_per_part)
        rec["replayed"] = None if b == float("inf") else b
        if ineq is not None:
            tr = list(getattr(ineq, "op_trace", []))
            rec["mechanisms"] = ineq.mechanisms()
            rec["op_counts"] = {m: sum(1 for x in tr if x["op"] == m)
                                for m in rec["mechanisms"]}
            rec["repr"] = repr(ineq)
    except FD.UnsoundDerivationError as e:
        rec["assertion"] = f"{type(e).__name__}: {str(e).splitlines()[0]}"
    except Exception as e:
        rec["assertion"] = f"{type(e).__name__}: {e}"
    finally:
        env.pool = []; env.accumulator = []; env.stored_derived = []
        env.frac_pool = env.frac_pool.__class__(64)

    out[g].append(rec)
    print(f"{g:26s} ep{rec['episode']:<6} claimed={claimed:.6f} "
          f"replayed={rec['replayed']} ops={rec['op_counts'] or '{}'}"
          f"{'  !! ' + rec['assertion'] if rec['assertion'] else ''}")

with open("config_files/_replay.json", "w") as f:
    json.dump(out, f, indent=1)
print("\nsaved -> config_files/_replay.json")
