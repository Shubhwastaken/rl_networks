"""
Task B: audit the six surviving results against the root cause, by
replaying each winning episode and validity-checking EVERY intermediate
inequality against the achievable entropy vector.

Task C: independently verify each surviving terminal inequality
numerically at the achievable routing point.

Also: isolation test of stage-1 (the max/min union) on its own.
"""
import os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import fixed_training as FT
import fixed_submodularity as SUB
from fixed_inequality import Inequality, EntropyIndex
from fixed_environment import PartitionBoundEnv, Phase, ActionType
import fixed_graph_generation as G
from _entropy_check import max_symmetric_flow_with_arcs, entropy_vector, violation, is_falsified

FT.SUB_LP_FATAL = False
G._build_registry(); BY = {g.name: g for g in G.GRAPH_REGISTRY}
SURVIVORS = ["diamond_6N", "star_8N", "ford_fulkerson_6N",
             "hu_2pairs_6N", "grid_4x4_16N", "kramer_savari_ladder_8N"]

log = json.load(open("config_files/stage4_proof_log.json"))
novel = [e for e in log if isinstance(e, dict) and e.get("summary", {}).get("is_novel")]
rep = json.load(open("config_files/_replay.json"))

# ---------------------------------------------------------------- stage 1 test
print("=" * 96)
print("ISOLATION TEST: is the stage-1 max/min union sound on its own?")
print("=" * 96)
g = BY["diamond_6N"]; nodes, edges, sessions = g.nodes, g.edges, g.sessions
r, flow = max_symmetric_flow_with_arcs(nodes, edges, sessions)
part = [list(x) for x in json.load(open("model_files/ckpt_stage3_best_partitions.json"))["diamond_6N"]["partition"]]
ix = EntropyIndex(part, nodes, edges, sessions)
h = entropy_vector(ix, nodes, edges, sessions, r, flow)
rng = np.random.default_rng(1); bad1 = tested = 0


def rand_ineq(with_yst):
    q = Inequality(ix)
    q.coeffs[ix.yi_idx()] = rng.uniform(0.1, 1.5)
    if with_yst: q.coeffs[ix.yst_idx(int(rng.integers(0, ix.n())))] = rng.uniform(0.4, 1.4)
    for v in nodes:
        if rng.random() < 0.6: q.coeffs[ix.source_idx(v)] = -rng.uniform(0.1, 1.5)
    for e in edges:
        if rng.random() < 0.8: q.coeffs[ix.edge_idx(e)] = -rng.uniform(0.5, 3.0)
    return q


def stage1(a, b):
    U = Inequality(ix); aa, ab = a.active_yst(), b.active_yst()
    for i in (aa | ab): U.coeffs[ix.yst_idx(i)] = max(a.coeffs[ix.yst_idx(i)], b.coeffs[ix.yst_idx(i)])
    for i in range(ix.n()):
        c = max(a.coeffs[ix.yi_pi_idx(i)], b.coeffs[ix.yi_pi_idx(i)])
        if c > 1e-9: U.coeffs[ix.yi_pi_idx(i)] = c
    ca, cb = a.coeffs[ix.yi_idx()], b.coeffs[ix.yi_idx()]
    c = max(ca, cb) if (aa or ab) else min(ca, cb)
    if c > 1e-9: U.coeffs[ix.yi_idx()] = c
    for v in nodes:
        c = min(a.coeffs[ix.source_idx(v)], b.coeffs[ix.source_idx(v)])
        if c < -1e-9: U.coeffs[ix.source_idx(v)] = c
    for e in edges:
        c = min(a.coeffs[ix.edge_idx(e)], b.coeffs[ix.edge_idx(e)])
        if c < -1e-9: U.coeffs[ix.edge_idx(e)] = c
    return U


for _ in range(6000):
    A, B = rand_ineq(rng.random() < .5), rand_ineq(rng.random() < .5)
    if is_falsified(A, h) or is_falsified(B, h): continue
    tested += 1
    if is_falsified(stage1(A, B), h): bad1 += 1
print(f"  stage-1 union alone : {bad1} of {tested} valid input pairs turned FALSE")

# ---------------------------------------------------------------- Task B
print("\n" + "=" * 96)
print("TASK B: per-step audit of each surviving result's winning derivation")
print("=" * 96)
env = PartitionBoundEnv(graph_dataset_size=19, stage=4)
verdicts = {}
for name in SURVIVORS:
    recs = [x for x in rep.get(name, []) if x["replayed"] is not None]
    if not recs:
        verdicts[name] = ("CANNOT DETERMINE", "no replayable episode"); continue
    best = min(recs, key=lambda x: x["replayed"])
    ep = next(e for e in novel if e["graph_name"] == name and e["episode"] == best["episode"])
    nodes, edges = list(ep["nodes"]), [tuple(x) for x in ep["edges"]]
    sessions = [tuple(x) for x in ep["sessions"]]
    part = [list(p) for p in ep["partition"]]
    r, flow = max_symmetric_flow_with_arcs(nodes, edges, sessions)
    env.nodes = list(nodes); env.edges = list(edges); env.sessions = list(sessions)
    env.partition = [list(p) for p in part]; env.partition_weights = {}
    env.partition_bound = BY[name].optimal_bound; env._lp_lower_bound = r; env.index = None
    env.adjacency = {n: set() for n in nodes}; env.edge_set = set()
    for u, v in edges:
        env.adjacency[u].add(v); env.adjacency[v].add(u)
        env.edge_set.add((u, v)); env.edge_set.add((v, u))
    env._start_phase2(); env._start_phase3(preseed=False)
    env.internal_per_part = env.internal_per_part or []
    h = entropy_vector(env.index, nodes, edges, sessions, r, flow)
    first_bad, nsteps = None, 0
    for s in ep.get("steps", []):
        if env.current_phase != Phase.PHASE3: break
        a = s.get("action_raw")
        if not a: continue
        _, _, done = env.step(dict(a)); nsteps += 1
        items = list(env.frac_pool) + list(env.accumulator) + list(env.stored_derived)
        bad = [x for x in items if is_falsified(x, h)]
        if bad and first_bad is None:
            first_bad = (nsteps, ActionType(a.get("type")).name, violation(bad[0], h), repr(bad[0]))
        if done: break
    b, ineq = env.frac_pool.best_terminal(len(sessions), len(edges), env.internal_per_part)
    term_viol = violation(ineq, h) if ineq is not None else None
    if first_bad is None:
        v = ("VERIFIED SOUND (not falsifiable at the achievable point)", f"{nsteps} steps, no false intermediate")
    elif term_viol is not None and term_viol > 1e-7:
        v = ("CONTAINS THE DEFECT", f"step {first_bad[0]} {first_bad[1]}, terminal violation {term_viol:+.4f}")
    else:
        v = ("CONTAINS THE DEFECT (not in the winning terminal)",
             f"false item at step {first_bad[0]} {first_bad[1]} (viol {first_bad[2]:+.4f}); winning terminal itself clean")
    verdicts[name] = v
    print(f"\n{name}: episode {best['episode']}, replayed bound {best['replayed']:.6f}, r={r}")
    print(f"   steps replayed        : {nsteps}")
    print(f"   first false intermediate: {'none' if first_bad is None else f'step {first_bad[0]} via {first_bad[1]} (viol {first_bad[2]:+.4f})'}")
    print(f"   winning terminal       : {repr(ineq).encode('ascii','replace').decode()[:110] if ineq is not None else 'none'}")
    print(f"   terminal violation     : {term_viol if term_viol is None else f'{term_viol:+.6f}'}")
    print(f"   VERDICT                : {v[0]}  [{v[1]}]")
    env.pool = []; env.accumulator = []; env.stored_derived = []
    env.frac_pool = env.frac_pool.__class__(64)

# ---------------------------------------------------------------- Task C
print("\n" + "=" * 96)
print("TASK C: independent numeric verification of each surviving terminal")
print("=" * 96)
print(f"{'graph':<26}{'bound':>10}{'r=LP':>8}{'sum(c*h)':>11}  verdict")
for name in SURVIVORS:
    recs = [x for x in rep.get(name, []) if x["replayed"] is not None]
    if not recs:
        print(f"{name:<26}{'-':>10}{'-':>8}{'-':>11}  CANNOT DETERMINE"); continue
    best = min(recs, key=lambda x: x["replayed"])
    ep = next(e for e in novel if e["graph_name"] == name and e["episode"] == best["episode"])
    nodes, edges = list(ep["nodes"]), [tuple(x) for x in ep["edges"]]
    sessions = [tuple(x) for x in ep["sessions"]]; part = [list(p) for p in ep["partition"]]
    r, flow = max_symmetric_flow_with_arcs(nodes, edges, sessions)
    env.nodes = list(nodes); env.edges = list(edges); env.sessions = list(sessions)
    env.partition = [list(p) for p in part]; env.partition_weights = {}
    env.partition_bound = BY[name].optimal_bound; env._lp_lower_bound = r; env.index = None
    env.adjacency = {n: set() for n in nodes}; env.edge_set = set()
    for u, v in edges:
        env.adjacency[u].add(v); env.adjacency[v].add(u)
        env.edge_set.add((u, v)); env.edge_set.add((v, u))
    env._start_phase2(); env._start_phase3(preseed=False)
    env.internal_per_part = env.internal_per_part or []
    h = entropy_vector(env.index, nodes, edges, sessions, r, flow)
    for s in ep.get("steps", []):
        if env.current_phase != Phase.PHASE3: break
        a = s.get("action_raw")
        if not a: continue
        _, _, done = env.step(dict(a))
        if done: break
    b, ineq = env.frac_pool.best_terminal(len(sessions), len(edges), env.internal_per_part)
    vv = violation(ineq, h) if ineq is not None else None
    verdict = ("FALSE - violated by an achievable code" if (vv is not None and vv > 1e-7)
               else "not falsified at the achievable point")
    print(f"{name:<26}{b:>10.4f}{r:>8.3f}{(vv if vv is not None else float('nan')):>11.4f}  {verdict}")
    env.pool = []; env.accumulator = []; env.stored_derived = []
    env.frac_pool = env.frac_pool.__class__(64)
