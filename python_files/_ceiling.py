"""
Generator-LP ceiling for the whole registry.

ceiling(G) = the tightest bound expressible as a nonnegative combination
of the base generators (node IOs, partition IOs, corrected crypto,
corrected decode, source-independence / ST-source identities). Because an
inequality c is valid whenever c <= sum(lam_i A_i) componentwise and every
h_j is an entropy (>= 0), this is an upper limit on what ANY policy over
these operators could ever prove.

Two numbers per graph:

  ceiling_LP     the LP optimum over all nonnegative combinations
  best_single    min over individual generators of the bound each yields
                 on its own

If best_single == ceiling_LP the optimum needs no combination at all, and
the whole search apparatus is unnecessary on that graph. This is tested
directly rather than read off the LP's chosen vertex, because the LP
optimum is typically degenerate and the number of nonzero lam it happens
to return is a solver artifact.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.optimize import linprog
from fixed_inequality import EntropyIndex
from fixed_base_inequality_generator import (generate_all_node_ios,
                                             generate_base_inequalities)
from functional_dependence import (_directed_cut_edges, _sessions_separated_by_cut,
                                   _edges_into_sink, enumerate_crypto_cuts)
from _entropy_check import entropy_vector
from _flowpoints import optimal_flow_profiles
import fixed_graph_generation as G
G._build_registry()


def all_generators(index, nodes, edges, sessions, partition, max_cuts=4000):
    gens, names = [], []
    for n, io in generate_all_node_ios(partition, nodes, edges, sessions, index).items():
        gens.append(io.coeffs.copy()); names.append(f"IO({n})")
    for i, b in enumerate(generate_base_inequalities(partition, nodes, edges, sessions, index)):
        gens.append(b.coeffs.copy()); names.append(f"PIO(P{i})")
    cuts = enumerate_crypto_cuts(nodes, edges, sessions)[:max_cuts]
    for vp, sep in cuts:
        cut = _directed_cut_edges(set(vp), nodes, edges)
        if not cut or not sep: continue
        v = np.zeros(index.dim); v[index.yi_idx()] = len(sep) / len(sessions)
        okk = True
        for e in cut:
            k = f"U_{e[0]}_{e[1]}"
            if k not in index.var_to_idx: k = f"U_{e[1]}_{e[0]}"
            if k not in index.var_to_idx: okk = False; break
            v[index.var_to_idx[k]] -= 1.0
        if okk: gens.append(v); names.append(f"CRYPTO(|V'|={len(vp)},sep={len(sep)})")
    for si in range(len(sessions)):
        inc = _edges_into_sink(si, sessions, edges)
        if not inc: continue
        v = np.zeros(index.dim); v[index.yi_idx()] = 1.0 / len(sessions)
        for e in inc:
            k = f"U_{e[0]}_{e[1]}"
            if k not in index.var_to_idx: k = f"U_{e[1]}_{e[0]}"
            if k in index.var_to_idx: v[index.var_to_idx[k]] -= 1.0
        gens.append(v); names.append(f"DECODE({si})")
    v = np.zeros(index.dim); v[index.yi_idx()] = 1.0
    for nd in nodes: v[index.source_idx(nd)] = -1.0
    gens.append(v); names.append("SRC_IND(Y_I<=sumY_S)")
    v = np.zeros(index.dim); v[index.yi_idx()] = -1.0
    for nd in nodes: v[index.source_idx(nd)] = 1.0
    gens.append(v); names.append("SRC_IND(sumY_S<=Y_I)")
    for i in range(len(index.partitions)):
        srcs = {sessions[si][0] for si in index.st_sessions[i]}
        v = np.zeros(index.dim); v[index.yst_idx(i)] = 1.0
        for nd in srcs: v[index.source_idx(nd)] -= 1.0
        gens.append(v); names.append(f"ST_SRC(P{i})<=")
        v = np.zeros(index.dim); v[index.yst_idx(i)] = -1.0
        for nd in srcs: v[index.source_idx(nd)] += 1.0
        gens.append(v); names.append(f"ST_SRC(P{i})>=")
        v = np.zeros(index.dim); v[index.yi_pi_idx(i)] = 1.0; v[index.yst_idx(i)] = -1.0
        gens.append(v); names.append(f"INT<=ST(P{i})")
    return np.array(gens), names


def ceiling(name):
    g = [x for x in G.GRAPH_REGISTRY if x.name == name][0]
    nodes, edges, sessions = g.nodes, g.edges, g.sessions
    part = [list(p) for p in g.optimal_partition]
    ix = EntropyIndex(part, nodes, edges, sessions)
    profs = optimal_flow_profiles(nodes, edges, sessions, n=20)
    lp = profs[0][0]
    HS = [entropy_vector(ix, nodes, edges, sessions, r, f) for r, f in profs]
    gens, names = all_generators(ix, nodes, edges, sessions, part)
    keep = [i for i in range(len(gens))
            if max(float(np.dot(gens[i], h)) for h in HS) <= 1e-7]
    A = gens[keep]; nm = [names[i] for i in keep]
    m = len(A); nE = len(edges); nI = len(sessions)
    ipp = [sum(1 for s, t in sessions if s in set(P) and t in set(P)) for P in part]

    def denom_of(v):
        return v[ix.yi_idx()] * nI + sum(v[ix.yi_pi_idx(k)] * ipp[k] for k in range(ix.n()))

    # best single generator, solver-independent
    best_single = float("inf"); best_name = None
    for i in range(m):
        d = denom_of(A[i])
        if d <= 1e-9: continue
        cap = sum(max(0.0, -A[i][ix.edge_idx(e)]) for e in edges)
        if any(A[i][ix.yst_idx(k)] < -1e-9 for k in range(ix.n())): continue
        if any(A[i][ix.source_idx(v)] < -1e-9 for v in nodes): continue
        b = cap / d
        if b < best_single: best_single, best_name = b, nm[i]

    # LP over all nonneg combinations
    nv = m + nE
    c = np.concatenate([np.zeros(m), np.ones(nE)])
    Aub, bub = [], []
    for j, e in enumerate(edges):
        row = np.zeros(nv)
        for i in range(m): row[i] = -A[i][ix.edge_idx(e)]
        row[m + j] = -1.0
        Aub.append(row); bub.append(0.0)
    for k in range(ix.n()):
        row = np.zeros(nv)
        for i in range(m): row[i] = -A[i][ix.yst_idx(k)]
        Aub.append(row); bub.append(0.0)
    for v in nodes:
        row = np.zeros(nv)
        for i in range(m): row[i] = -A[i][ix.source_idx(v)]
        Aub.append(row); bub.append(0.0)
    den = np.zeros(nv)
    for i in range(m): den[i] = denom_of(A[i])
    res = linprog(c, A_ub=np.array(Aub), b_ub=np.array(bub),
                  A_eq=np.array([den]), b_eq=np.array([1.0]),
                  bounds=[(0, None)] * nv, method="highs")
    if not res.success:
        return dict(name=name, ok=False, lp=lp, pb=g.optimal_bound)
    used = [(nm[i], res.x[i]) for i in range(m) if res.x[i] > 1e-7]
    weights = sorted({round(w, 6) for _, w in used})
    return dict(name=name, ok=True, ceiling=res.fun, lp=lp, pb=g.optimal_bound,
                n_used=len(used), n_weights=len(weights),
                best_single=best_single, best_single_name=best_name,
                n_gens=m, used=used[:5])


if __name__ == "__main__":
    cp = {d["name"]: d for d in json.load(open("config_files/_cpsat_pb.json"))}
    targets = sys.argv[1:] or [g.name for g in G.GRAPH_REGISTRY]
    rows = []
    print(f"{'graph':<26}{'ceiling':>9}{'PB':>8}{'LP LB':>8}{'beats':>7}"
          f"{'=LP?':>6}{'#gen':>6}{'#wts':>6}{'single?':>9}")
    for t in targets:
        try: r = ceiling(t)
        except Exception as e:
            print(f"{t:<26} ERROR {type(e).__name__}: {e}"); continue
        if not r.get("ok"):
            print(f"{t:<26} infeasible"); continue
        pb = cp[t]["pb"]
        eq_lp = abs(r["ceiling"] - r["lp"]) < 1e-6
        single = abs(r["best_single"] - r["ceiling"]) < 1e-6
        rows.append((t, r, pb, eq_lp, single))
        print(f"{t:<26}{r['ceiling']:>9.4f}{pb:>8.3f}{r['lp']:>8.3f}"
              f"{('YES' if r['ceiling'] < pb - 1e-8 else 'no'):>7}"
              f"{('YES' if eq_lp else 'NO'):>6}{r['n_used']:>6}{r['n_weights']:>6}"
              f"{('YES' if single else 'NO'):>9}")
    json.dump([{k: v for k, v in r.items() if k != 'used'} for _, r, _, _, _ in rows],
              open("config_files/_ceiling.json", "w"), indent=1)
    n = len(rows)
    print(f"\nceiling == LP LB exactly : {sum(1 for *_ , e, s in rows if e)} / {n}")
    print(f"ceiling reached by ONE generator : {sum(1 for *_ , e, s in rows if s)} / {n}")
    multi = [t for t, r, pb, e, s in rows if not s]
    print(f"graphs needing a genuine combination : {multi if multi else 'NONE'}")
    beats = [t for t, r, pb, e, s in rows if r['ceiling'] < pb - 1e-8]
    print(f"ceiling beats PB on : {len(beats)} / {n}")
