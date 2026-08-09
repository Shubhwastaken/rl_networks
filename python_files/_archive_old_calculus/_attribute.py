"""
Phase 2 report (c): mechanism attribution.

eval_results.json records p3_bound but not WHICH operations produced it,
so this re-runs the greedy eval episode per graph with the same seeding
and reads the winning terminal inequality's op_trace (the 1h provenance).

Reports, in order, the operations that built the winning terminal, and
whether the bound came from the policy's own assembly or from the
post-episode func-dep oracle (apply_all_improving_func_dep).
"""
import os, sys, json, random
_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import numpy as np
import torch
import fixed_training as FT
import functional_dependence as FD
from fixed_environment import PartitionBoundEnv, Phase
from gnn_policy import GNNPhase1Policy, GNNPhase2Policy, GNNPhase3Policy
import fixed_graph_generation as G

FT.SUB_LP_FATAL = False
SEED = FT.SEED
DEVICE = FT.DEVICE
M = "model_files"

coeff_dim = torch.load(f"{M}/ckpt_stage1_coeff_dim.pt", weights_only=True,
                       map_location=DEVICE)["coeff_dim"]
p1 = GNNPhase1Policy()
p1.net.load_state_dict(torch.load(f"{M}/ckpt_stage3_phase1.pt", weights_only=True,
                                  map_location=DEVICE))
p2 = GNNPhase2Policy(coeff_dim=coeff_dim)
p2.net.load_state_dict(torch.load(f"{M}/ckpt_stage3_phase2.pt", weights_only=True,
                                  map_location=DEVICE))
p3 = GNNPhase3Policy(coeff_dim=coeff_dim, total_episodes=30000)
p3.net.load_state_dict(torch.load(f"{M}/ckpt_stage4_phase3.pt", weights_only=True,
                                  map_location=DEVICE))
with open(f"{M}/ckpt_stage3_best_partitions.json") as f:
    raw = json.load(f)
best_partitions = {k: ([tuple(p) for p in v["partition"]], v["weights"], v["bound"])
                   for k, v in raw.items()}

G._build_registry()
by_name = {g.name: g for g in G.GRAPH_REGISTRY}
targets = sys.argv[1:] or list(by_name)

from lp_lower_bound import compute_lp_lower_bound

# _run_eval_episode clears the pool before returning, so the winning
# inequality is unreachable afterwards. assert_derivation_sound is handed
# exactly that inequality immediately before the clear -- hook it.
CAPTURED = {}
_orig_assert = FT.assert_derivation_sound


def _capture(ineq, bound, graph_name, episode=None, step=None, **kw):
    CAPTURED["ineq"] = ineq
    CAPTURED["bound"] = bound
    return _orig_assert(ineq, bound, graph_name, episode, step, **kw)


FT.assert_derivation_sound = _capture

out = {}
env = PartitionBoundEnv(graph_dataset_size=19, stage=4)
for name in targets:
    g = by_name[name]
    gt = (g.nodes, g.edges, g.sessions)
    lp_bounds = {name: compute_lp_lower_bound(*gt)}
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    CAPTURED.clear()
    rec = {"bound": None, "mechanisms": [], "op_counts": {}, "op_trace": []}
    try:
        r = FT._run_eval_episode(env, gt, p1, p2, p3, best_partitions,
                                 lp_bounds, greedy=True)
    except FD.UnsoundDerivationError as e:
        rec["assertion"] = f"{type(e).__name__}: {str(e).splitlines()[0]}"
        out[name] = rec
        print(f"{name:26s} ASSERTION {rec['assertion'][:60]}")
        continue
    if r is None:
        out[name] = rec
        print(f"{name:26s} (episode returned None)")
        continue

    rec["bound"] = r[1]
    ineq = CAPTURED.get("ineq")
    if ineq is not None:
        trace = list(getattr(ineq, "op_trace", []))
        rec["mechanisms"] = ineq.mechanisms()
        rec["op_counts"] = {m: sum(1 for e in trace if e["op"] == m)
                            for m in rec["mechanisms"]}
        rec["op_trace"] = [e["op"] for e in trace]
        rec["repr"] = repr(ineq)
    out[name] = rec
    print(f"{name:26s} bound={rec['bound']}  ops={rec['op_counts'] or '{}'}")

with open("config_files/_attribution.json", "w") as f:
    json.dump(out, f, indent=1)
print("saved -> config_files/_attribution.json")
