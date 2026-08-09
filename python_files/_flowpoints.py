"""Several distinct optimal flow profiles at the max symmetric rate.

One optimal flow gives one achievable entropy vector. Different vertices
of the optimal face route the same rate over different edges, so testing
several is a strictly stronger falsification test than testing one.
"""
import numpy as np
from scipy.optimize import linprog


def _build(nodes, edges, sessions):
    node_list = list(nodes); ni = {n: i for i, n in enumerate(node_list)}
    adj = {i: [] for i in range(len(node_list))}
    for u, v in edges:
        adj[ni[u]].append(ni[v]); adj[ni[v]].append(ni[u])
    arcs, arc_idx = [], {}
    for u, v in edges:
        a, b = ni[u], ni[v]
        arc_idx[(a, b)] = len(arcs); arcs.append((a, b))
        arc_idx[(b, a)] = len(arcs); arcs.append((b, a))
    n_comm = len(sessions); nv = 1 + len(arcs) * n_comm
    xv = lambda a, k: 1 + a * n_comm + k
    Aub, bub = [], []
    for k, (s, _t) in enumerate(sessions):
        si = ni[s]; row = np.zeros(nv); row[0] = 1.0
        for vi in adj[si]:
            f = arc_idx.get((si, vi)); r_ = arc_idx.get((vi, si))
            if f is not None: row[xv(f, k)] -= 1.0
            if r_ is not None: row[xv(r_, k)] += 1.0
        Aub.append(row); bub.append(0.0)
    for u, v in edges:
        a, b = ni[u], ni[v]; row = np.zeros(nv)
        for k in range(n_comm):
            row[xv(arc_idx[(a, b)], k)] = 1.0; row[xv(arc_idx[(b, a)], k)] = 1.0
        Aub.append(row); bub.append(1.0)
    Aeq, beq = [], []
    for i in range(len(node_list)):
        for k, (s, t) in enumerate(sessions):
            if i == ni[s] or i == ni[t]: continue
            row = np.zeros(nv)
            for vi in adj[i]:
                f = arc_idx.get((i, vi)); r_ = arc_idx.get((vi, i))
                if f is not None: row[xv(f, k)] += 1.0
                if r_ is not None: row[xv(r_, k)] -= 1.0
            Aeq.append(row); beq.append(0.0)
    return ni, arc_idx, nv, xv, np.array(Aub), np.array(bub), \
        (np.array(Aeq) if Aeq else None), (np.array(beq) if beq else None)


def optimal_flow_profiles(nodes, edges, sessions, n=60, seed=0):
    ni, arc_idx, nv, xv, Aub, bub, Aeq, beq = _build(nodes, edges, sessions)
    c0 = np.zeros(nv); c0[0] = -1.0
    base = linprog(c0, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq,
                   bounds=[(0, None)] * nv, method="highs")
    if not base.success:
        return []
    r = max(-base.fun, 0.0)
    # pin y = r, then optimise random directions over the optimal face
    Aeq2 = np.vstack([Aeq, np.eye(nv)[0]]) if Aeq is not None else np.eye(nv)[0:1]
    beq2 = np.concatenate([beq, [r]]) if beq is not None else np.array([r])
    rng = np.random.default_rng(seed)
    out, seen = [], set()
    for t in range(n):
        if t == 0: c = np.zeros(nv)
        elif t == 1: c = np.concatenate([[0.0], np.ones(nv - 1)])
        elif t == 2: c = np.concatenate([[0.0], -np.ones(nv - 1)])
        else: c = np.concatenate([[0.0], rng.normal(size=nv - 1)])
        res = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq2, b_eq=beq2,
                      bounds=[(0, None)] * nv, method="highs")
        if not res.success: continue
        flow = {}
        for u, v in edges:
            a, b = ni[u], ni[v]
            flow[(u, v)] = sum(res.x[xv(arc_idx[(a, b)], k)] + res.x[xv(arc_idx[(b, a)], k)]
                               for k in range(len(sessions)))
        key = tuple(round(flow[e], 6) for e in edges)
        if key in seen: continue
        seen.add(key); out.append((r, flow))
    return out or [(r, {e: 0.0 for e in edges})]
