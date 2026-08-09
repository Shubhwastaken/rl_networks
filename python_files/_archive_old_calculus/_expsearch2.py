"""
Policy-free search, three gate configurations, so the effect of each
change is separable:

  A  OLD      : MIN_YI_COEFF >= 0.5 floor, no quality gate  (main branch)
  B  NOFLOOR  : floor removed, no quality gate              (change 1 only)
  C  QUALITY  : floor removed + terminal must have bound < PB (changes 1+2)

All three use the additive FRACTIONAL_IO and no FIO decay, so A/B/C differ
only in the terminal gate. Every accepted terminal is re-verified against
the achievable entropy vector; the reported bounds are never taken on the
gate's word.
"""
import os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import fixed_inequality as FI
from fixed_environment import PartitionBoundEnv, ActionType
import fixed_graph_generation as G
from _entropy_check import entropy_vector
from _flowpoints import optimal_flow_profiles
from _expsearch import rand_action

G._build_registry()
BY = {g.name: g for g in G.GRAPH_REGISTRY}


def search(name, mode, episodes=120, horizon=16, seed=0):
    g = BY[name]
    nodes, edges, sessions = g.nodes, g.edges, g.sessions
    part = [list(p) for p in g.optimal_partition]
    pb = g.optimal_bound
    profs = optimal_flow_profiles(nodes, edges, sessions, n=15)
    lp = profs[0][0]
    rng = random.Random(seed)

    env = PartitionBoundEnv(graph_dataset_size=19, stage=4)
    env.nodes = list(nodes); env.edges = list(edges); env.sessions = list(sessions)
    env.partition = [list(p) for p in part]; env.partition_weights = {}
    env.partition_bound = pb; env._lp_lower_bound = lp; env.index = None
    env.adjacency = {n: set() for n in nodes}; env.edge_set = set()
    for u, v in edges:
        env.adjacency[u].add(v); env.adjacency[v].add(u)
        env.edge_set.add((u, v)); env.edge_set.add((v, u))
    env._start_phase2()
    HS = [entropy_vector(env.index, nodes, edges, sessions, r, f) for r, f in profs]

    best, bseq, n_term = float('inf'), None, 0
    for ep in range(episodes):
        env._start_phase2(); env._start_phase3(preseed=False)
        env.internal_per_part = env.internal_per_part or []
        # configure the gate AFTER _start_phase3 (which arms pb_target)
        if mode == "OLD":
            FI.Inequality.pb_target = None; FI.Inequality._floor = 0.5
        elif mode == "NOFLOOR":
            FI.Inequality.pb_target = None; FI.Inequality._floor = None
        else:
            FI.Inequality.pb_target = pb;   FI.Inequality._floor = None
        seq = []
        for _ in range(horizon):
            a = rand_action(env, rng)
            try: env.step(dict(a))
            except Exception: break
            seq.append(ActionType(a['type']).name)
            b, ineq = env.frac_pool.best_terminal(
                len(sessions), len(edges), env.internal_per_part)
            if ineq is not None and b < float('inf'):
                if max(float(np.dot(ineq.coeffs, h)) for h in HS) <= 1e-7:
                    n_term += 1
                    if b < best: best, bseq = b, list(seq)
        env.pool = []; env.accumulator = []; env.stored_derived = []
        env.frac_pool = env.frac_pool.__class__(64)
    return best, bseq, n_term, pb, lp


if __name__ == "__main__":
    targets = sys.argv[1:] or [g.name for g in G.GRAPH_REGISTRY]
    rows = []
    print(f"{'graph':<26}{'PB':>8}{'LP':>7} | {'A OLD':>9}{'B NOFLR':>9}{'C QUAL':>9} | "
          f"{'A t':>5}{'B t':>5}{'C t':>5}")
    for t in targets:
        r = {}
        for mode in ("OLD", "NOFLOOR", "QUALITY"):
            try:
                b, seq, n, pb, lp = search(t, mode)
            except Exception as e:
                b, seq, n, pb, lp = float('inf'), None, 0, BY[t].optimal_bound, 0.0
            r[mode] = (b, seq, n)
        pb = BY[t].optimal_bound
        f = lambda m: (f"{r[m][0]:.4f}" if r[m][0] < float('inf') else "none")
        print(f"{t:<26}{pb:>8.3f}{lp:>7.3f} | {f('OLD'):>9}{f('NOFLOOR'):>9}{f('QUALITY'):>9} | "
              f"{r['OLD'][2]:>5}{r['NOFLOOR'][2]:>5}{r['QUALITY'][2]:>5}")
        rows.append((t, pb, lp, {m: (r[m][0], r[m][1], r[m][2]) for m in r}))
    json.dump([{ "graph": t, "pb": pb, "lp": lp,
                 "A_old": d["OLD"][0] if d["OLD"][0] < 1e18 else None,
                 "B_nofloor": d["NOFLOOR"][0] if d["NOFLOOR"][0] < 1e18 else None,
                 "C_quality": d["QUALITY"][0] if d["QUALITY"][0] < 1e18 else None,
                 "B_seq": d["NOFLOOR"][1], "terms": [d[m][2] for m in ("OLD","NOFLOOR","QUALITY")]}
               for t, pb, lp, d in rows],
              open("config_files/_expsearch2.json", "w"), indent=1)
    for lab, key in (("A OLD", "OLD"), ("B NOFLOOR", "NOFLOOR"), ("C QUALITY", "QUALITY")):
        got = sum(1 for _, _, _, d in rows if d[key][0] < float('inf'))
        beat = sum(1 for _, pb, _, d in rows if d[key][0] < pb - 1e-9)
        print(f"{lab:<12} terminals on {got}/{len(rows)} graphs, below PB on {beat}")
