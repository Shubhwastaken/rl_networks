"""Task A, part 3: producing-action distribution for the first invalid
pool item, per graph; plus a soundness unit-test of the two source-
cancellation routines."""
import os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from collections import Counter
import fixed_training as FT
import fixed_submodularity as SUB
from fixed_inequality import Inequality, EntropyIndex
from fixed_environment import PartitionBoundEnv, Phase, ActionType
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


def scan(name, trials=40):
    g = BY[name]; nodes, edges, sessions = g.nodes, g.edges, g.sessions
    r, flow = max_symmetric_flow_with_arcs(nodes, edges, sessions)
    part = [list(x) for x in BP[name]["partition"]] if name in BP else [list(p) for p in g.optimal_partition]
    env = PartitionBoundEnv(graph_dataset_size=19, stage=4)
    random.seed(FT.SEED); np.random.seed(FT.SEED); torch.manual_seed(FT.SEED)
    producers = Counter(); n_bad_eps = 0
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
        seen = False
        while env.current_phase == Phase.PHASE3:
            v_ = env.get_valid_actions()
            if not v_: break
            a = p3.select_action(st, v_, greedy=False)
            st, _, done = env.step(a)
            st.update(nodes=nodes, edges=edges, sessions=sessions, partition=part, partition_weights={})
            items = list(env.frac_pool) + list(env.accumulator) + list(env.stored_derived)
            if not seen and any(is_falsified(x, h) for x in items):
                producers[ActionType(a.get("type")).name] += 1; seen = True; n_bad_eps += 1
            if done: break
        env.pool = []; env.accumulator = []; env.stored_derived = []
        env.frac_pool = env.frac_pool.__class__(64)
    return producers, n_bad_eps, trials


print("=" * 90)
print("FIRST-INVALID-ITEM: which action produced it")
print("=" * 90)
tot = Counter()
for nm in ["diamond_6N", "okamura_4N", "ford_fulkerson_6N"]:
    pr, nb, tr = scan(nm)
    tot.update(pr)
    print(f"\n{nm}: {nb}/{tr} episodes reached a false inequality")
    for k, v in pr.most_common(): print(f"   {v:4d}  {k}")
print("\nCOMBINED:")
for k, v in tot.most_common(): print(f"   {v:4d}  {k}")

print("\n" + "=" * 90)
print("UNIT TEST: are the two source-cancellation routines individually sound?")
print("=" * 90)
g = BY["diamond_6N"]; nodes, edges, sessions = g.nodes, g.edges, g.sessions
r, flow = max_symmetric_flow_with_arcs(nodes, edges, sessions)
part = [list(x) for x in BP["diamond_6N"]["partition"]]
ix = EntropyIndex(part, nodes, edges, sessions)
h = entropy_vector(ix, nodes, edges, sessions, r, flow)
rng = np.random.default_rng(0)
bad_cancel = bad_collapse = 0
for _ in range(4000):
    q = Inequality(ix)
    q.coeffs[ix.yi_idx()] = rng.uniform(0.2, 2.0)
    for v in nodes:
        if rng.random() < 0.7: q.coeffs[ix.source_idx(v)] = -rng.uniform(0.1, 1.5)
    for e in edges: q.coeffs[ix.edge_idx(e)] = -rng.uniform(0.5, 3.0)
    if is_falsified(q, h): continue          # only test on VALID inputs
    if is_falsified(q.cancel_source_terms(), h): bad_cancel += 1
    q2 = q.copy(); q2.coeffs[ix.yst_idx(0)] = rng.uniform(0.5, 1.5)
    if is_falsified(q2, h): continue
    if is_falsified(SUB._collapse_to_yi_if_valid(q2.copy(), ix, sessions), h): bad_collapse += 1
print(f"  Inequality.cancel_source_terms   : {bad_cancel} valid inputs turned FALSE")
print(f"  _collapse_to_yi_if_valid          : {bad_collapse} valid inputs turned FALSE")
