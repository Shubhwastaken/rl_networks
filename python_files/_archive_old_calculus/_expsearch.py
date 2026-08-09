"""
Policy-free search over short action sequences, on the experiment branch.

Three changes under test:
  1. MIN_YI_COEFF gate deleted (scale-invariant, filtered nothing)
  2. replaced by a bound-quality gate: terminal accepted only if bound < PB
  3. FRACTIONAL_IO additive (lam, mu >= 0 independent), FIO decay disabled

No policy is involved: actions are sampled uniformly from the legal set,
with lam/mu drawn from a small grid that includes weights > 1 (impossible
under the old convex form). Every accepted terminal is independently
re-checked against the achievable entropy vector before being reported,
so a gate bug cannot manufacture a result.
"""
import os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from collections import Counter

import fixed_inequality as FI
from fixed_environment import PartitionBoundEnv, Phase, ActionType
import fixed_graph_generation as G
from _entropy_check import entropy_vector
from _flowpoints import optimal_flow_profiles

G._build_registry()
BY = {g.name: g for g in G.GRAPH_REGISTRY}

WEIGHTS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]


def rand_action(env, rng):
    """Uniform over the legal action families, with additive FIO weights."""
    nodes = list(env.node_ios.keys())
    k = rng.randrange(6)
    if k == 0 and len(nodes) >= 2:
        u, v = rng.sample(nodes, 2)
        return {'type': ActionType.FRACTIONAL_IO, 'node_u': u, 'node_v': v,
                'lam': rng.choice(WEIGHTS), 'mu': rng.choice(WEIGHTS)}
    if k == 1 and len(env.frac_pool) > 0:
        return {'type': ActionType.ADD_TO_ACCUMULATOR,
                'idx_i': rng.randrange(len(env.frac_pool))}
    if k == 2 and len(env.accumulator) >= 2:
        i, j = rng.sample(range(len(env.accumulator)), 2)
        return {'type': rng.choice([ActionType.APPLY_SUBMODULARITY,
                                    ActionType.CROSS_SUBMOD]),
                'idx_i': i, 'idx_j': j}
    if k == 3 and env.func_dep_actions is not None:
        n = env.func_dep_actions.num_crypto_cuts()
        if n:
            return {'type': ActionType.APPLY_CRYPTO, 'cut_idx': rng.randrange(n)}
    if k == 4:
        return {'type': ActionType.APPLY_DECODE,
                'session_idx': rng.randrange(len(env.sessions))}
    if len(nodes) >= 2:
        u, v = rng.sample(nodes, 2)
        return {'type': ActionType.FRACTIONAL_IO, 'node_u': u, 'node_v': v,
                'lam': rng.choice(WEIGHTS), 'mu': rng.choice(WEIGHTS)}
    return {'type': ActionType.APPLY_DECODE, 'session_idx': 0}


def run(name, episodes=400, horizon=14, seed=0):
    g = BY[name]
    nodes, edges, sessions = g.nodes, g.edges, g.sessions
    part = [list(p) for p in g.optimal_partition]
    pb = g.optimal_bound
    profs = optimal_flow_profiles(nodes, edges, sessions, n=20)
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
    ix_probe = env.index
    HS = [entropy_vector(ix_probe, nodes, edges, sessions, r, f) for r, f in profs]

    best = (float('inf'), None, None)
    n_term = 0
    for ep in range(episodes):
        env.partition_bound = pb
        env._start_phase2(); env._start_phase3(preseed=False)
        env.internal_per_part = env.internal_per_part or []
        seq = []
        for _ in range(horizon):
            a = rand_action(env, rng)
            try:
                env.step(dict(a))
            except Exception:
                break
            seq.append(ActionType(a['type']).name if not isinstance(a['type'], int)
                       else ActionType(a['type']).name)
            b, ineq = env.frac_pool.best_terminal(
                len(sessions), len(edges), env.internal_per_part)
            if ineq is not None and b < float('inf'):
                # independent re-check: never trust the gate alone
                worst = max(float(np.dot(ineq.coeffs, h)) for h in HS)
                if worst <= 1e-7:
                    n_term += 1
                    if b < best[0]:
                        best = (b, list(seq), repr(ineq))
        env.pool = []; env.accumulator = []; env.stored_derived = []
        env.frac_pool = env.frac_pool.__class__(64)
    return dict(name=name, pb=pb, lp=lp, best=best[0], seq=best[1],
                ineq=best[2], n_term=n_term, episodes=episodes)


if __name__ == "__main__":
    targets = sys.argv[1:] or [g.name for g in G.GRAPH_REGISTRY]
    out = []
    print(f"{'graph':<26}{'best':>9}{'PB':>8}{'LP':>7}{'<PB':>6}{'terms':>7}  shortest producing sequence")
    for t in targets:
        try:
            r = run(t)
        except Exception as e:
            print(f"{t:<26} ERROR {type(e).__name__}: {e}"); continue
        out.append(r)
        bs = f"{r['best']:.4f}" if r['best'] < float('inf') else "none"
        seq = " ".join(x[:5] for x in (r['seq'] or [])[:8]) if r['seq'] else "-"
        print(f"{t:<26}{bs:>9}{r['pb']:>8.3f}{r['lp']:>7.3f}"
              f"{('YES' if r['best'] < r['pb'] - 1e-9 else 'no'):>6}{r['n_term']:>7}  {seq}")
    json.dump([{k: v for k, v in r.items() if k != 'ineq'} for r in out],
              open("config_files/_expsearch.json", "w"), indent=1)
    got = [r for r in out if r['best'] < float('inf')]
    print(f"\ngraphs reaching a verified terminal : {len(got)} / {len(out)}")
    print(f"graphs with a terminal below PB     : "
          f"{sum(1 for r in got if r['best'] < r['pb'] - 1e-9)} / {len(out)}")
