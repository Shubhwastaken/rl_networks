"""
Task A, part 2: for every falsifying pairwise-submodularity union, decide
WHICH sub-step flipped it from valid to false:

  stage1  max/min union construction
  stage2  _collapse_to_yi_if_valid   (Y_ST -> c_min*Y_I + zero sources)
  stage3  _cancel_sources_for_node_ios (Inequality.cancel_source_terms)

Reports the distribution over all three sub-LP graphs.
"""
import os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from collections import Counter
import fixed_training as FT
import fixed_submodularity as SUB
import fixed_environment as ENV
from fixed_inequality import Inequality
from fixed_environment import PartitionBoundEnv, Phase
from gnn_policy import GNNPhase1Policy, GNNPhase2Policy, GNNPhase3Policy
import fixed_graph_generation as G
from _entropy_check import max_symmetric_flow_with_arcs, entropy_vector, violation, is_falsified

FT.SUB_LP_FATAL = False
G._build_registry(); BY = {g.name: g for g in G.GRAPH_REGISTRY}
M, D = "model_files", FT.DEVICE
cd_ = torch.load(f"{M}/ckpt_stage1_coeff_dim.pt", weights_only=True, map_location=D)["coeff_dim"]
p1 = GNNPhase1Policy(); p1.net.load_state_dict(torch.load(f"{M}/ckpt_stage3_phase1.pt", weights_only=True, map_location=D))
p2 = GNNPhase2Policy(coeff_dim=cd_); p2.net.load_state_dict(torch.load(f"{M}/ckpt_stage3_phase2.pt", weights_only=True, map_location=D))
p3 = GNNPhase3Policy(coeff_dim=cd_, total_episodes=30000); p3.net.load_state_dict(torch.load(f"{M}/ckpt_stage4_phase3.pt", weights_only=True, map_location=D))
BP = json.load(open(f"{M}/ckpt_stage3_best_partitions.json"))


def stages(a, b, index, sessions):
    """Rebuild the union in stages, exactly as apply_pairwise_submodularity does."""
    U = Inequality(index)
    aa, ab = a.active_yst(), b.active_yst()
    for i in (aa | ab):
        U.coeffs[index.yst_idx(i)] = max(a.coeffs[index.yst_idx(i)], b.coeffs[index.yst_idx(i)])
    for i in range(index.n()):
        c = max(a.coeffs[index.yi_pi_idx(i)], b.coeffs[index.yi_pi_idx(i)])
        if c > 1e-9: U.coeffs[index.yi_pi_idx(i)] = c
    ca, cb = a.coeffs[index.yi_idx()], b.coeffs[index.yi_idx()]
    c = max(ca, cb) if (aa or ab) else min(ca, cb)
    if c > 1e-9: U.coeffs[index.yi_idx()] = c
    for v in index.nodes:
        c = min(a.coeffs[index.source_idx(v)], b.coeffs[index.source_idx(v)])
        if c < -1e-9: U.coeffs[index.source_idx(v)] = c
    for e in index.edges:
        c = min(a.coeffs[index.edge_idx(e)], b.coeffs[index.edge_idx(e)])
        if c < -1e-9: U.coeffs[index.edge_idx(e)] = c
    s1 = U.copy()
    s2 = SUB._collapse_to_yi_if_valid(s1.copy(), index, sessions)
    s3 = SUB._cancel_sources_for_node_ios(s2.copy(), index, sessions)
    return s1, s2, s3


def run(name, trials=60):
    g = BY[name]; nodes, edges, sessions = g.nodes, g.edges, g.sessions
    r, flow = max_symmetric_flow_with_arcs(nodes, edges, sessions)
    part = [list(x) for x in BP[name]["partition"]] if name in BP else [list(p) for p in g.optimal_partition]
    CALLS = []
    _orig = SUB.apply_pairwise_submodularity
    def patched(a, b, index, sess):
        u, i = _orig(a, b, index, sess); CALLS.append((a.copy(), b.copy(), index, sess)); return u, i
    SUB.apply_pairwise_submodularity = patched; ENV.apply_pairwise_submodularity = patched
    env = PartitionBoundEnv(graph_dataset_size=19, stage=4)
    random.seed(FT.SEED); np.random.seed(FT.SEED); torch.manual_seed(FT.SEED)
    tally = Counter(); examples = {}
    for t in range(trials):
        env.nodes = list(nodes); env.edges = list(edges); env.sessions = list(sessions)
        env.partition = [list(p) for p in part]; env.partition_weights = {}
        env.partition_bound = g.optimal_bound; env._lp_lower_bound = r; env.index = None
        env.adjacency = {n: set() for n in nodes}; env.edge_set = set()
        for u, v in edges:
            env.adjacency[u].add(v); env.adjacency[v].add(u)
            env.edge_set.add((u, v)); env.edge_set.add((v, u))
        env._start_phase2(); env._start_phase3(preseed=False)
        env.internal_per_part = env.internal_per_part or []
        h = entropy_vector(env.index, nodes, edges, sessions, r, flow)
        st = env._get_state(); st.update(nodes=nodes, edges=edges, sessions=sessions, partition=part, partition_weights={})
        CALLS.clear()
        while env.current_phase == Phase.PHASE3:
            v_ = env.get_valid_actions()
            if not v_: break
            a = p3.select_action(st, v_, greedy=False)
            n0 = len(CALLS)
            st, _, done = env.step(a)
            st.update(nodes=nodes, edges=edges, sessions=sessions, partition=part, partition_weights={})
            for (A, B, ix, se) in CALLS[n0:]:
                if is_falsified(A, h) or is_falsified(B, h):
                    tally["input already false"] += 1; continue
                s1, s2, s3 = stages(A, B, ix, se)
                f1, f2, f3 = is_falsified(s1, h), is_falsified(s2, h), is_falsified(s3, h)
                if f1: k = "stage1 max/min union"
                elif f2: k = "stage2 collapse (Y_ST->Y_I + zero sources)"
                elif f3: k = "stage3 cancel_source_terms (node IO)"
                else: continue
                tally[k] += 1
                examples.setdefault(k, (repr(A), repr(B), repr(s3), violation(s3, h)))
            if done: break
        env.pool = []; env.accumulator = []; env.stored_derived = []
        env.frac_pool = env.frac_pool.__class__(64)
    SUB.apply_pairwise_submodularity = _orig; ENV.apply_pairwise_submodularity = _orig
    return tally, examples


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["diamond_6N", "okamura_4N", "ford_fulkerson_6N"]):
        tally, ex = run(name)
        print("\n" + "=" * 92)
        print(f"{name}: which sub-step first makes a union FALSE")
        print("=" * 92)
        if not tally: print("  (no falsifying union observed)")
        for k, v in tally.most_common():
            print(f"  {v:5d}  {k}")
        for k, (A, B, U, vi) in ex.items():
            print(f"\n  example [{k}] violation {vi:+.4f}")
            print("     A:", A.encode('ascii', 'replace').decode()[:120])
            print("     B:", B.encode('ascii', 'replace').decode()[:120])
            print("     ->", U.encode('ascii', 'replace').decode()[:120])
