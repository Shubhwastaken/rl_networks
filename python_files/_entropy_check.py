"""
Independent validity checker for inequalities in this codebase.

Idea
----
The LP "lower bound" is the max symmetric multi-commodity FLOW. A
fractional flow is realisable by routing (time-sharing / vector routing),
so rate r = LP_LB is ACHIEVABLE. Routing with independent sources gives a
fully explicit entropy vector:

    h(Y_I)        = |I| * r                     (all sources, independent)
    h(Y_S_v)      = (#sessions sourced at v) * r
    h(Y_ST_Pk)    = |st_sessions[k]| * r
    h(Y_I_Pk)     = |internal_sessions[k]| * r
    h(U_e)        = f_e   (total flow routed across edge e, <= 1)

Any inequality that is a valid consequence of the network's constraints
must hold at this point. In this codebase's convention (positive coeff =
LHS, negative = RHS, meaning LHS <= RHS) validity requires

    sum_j coeffs[j] * h[j]  <=  0

So a strictly positive value FALSIFIES the inequality: it cannot be a
consequence of the IO constraints, however it was derived.

Direction of inference
----------------------
  value > tol   =>  the inequality is FALSE (proof, given r is achievable)
  value <= tol  =>  NOT falsified at this point. This is necessary, not
                    sufficient -- it does not prove the inequality sound.

Only the first direction is used to make claims.
"""
import numpy as np
from scipy.optimize import linprog


def max_symmetric_flow_with_arcs(nodes, edges, sessions):
    """Same LP as lp_lower_bound.compute_lp_lower_bound, but also returns
    the per-undirected-edge total flow f_e of an optimal solution."""
    node_list = list(nodes)
    ni = {n: i for i, n in enumerate(node_list)}
    n_nodes, n_comm = len(node_list), len(sessions)
    adj = {i: [] for i in range(n_nodes)}
    for u, v in edges:
        adj[ni[u]].append(ni[v]); adj[ni[v]].append(ni[u])
    arcs, arc_idx = [], {}
    for u, v in edges:
        a, b = ni[u], ni[v]
        arc_idx[(a, b)] = len(arcs); arcs.append((a, b))
        arc_idx[(b, a)] = len(arcs); arcs.append((b, a))
    nv = 1 + len(arcs) * n_comm
    xv = lambda a, k: 1 + a * n_comm + k

    c = np.zeros(nv); c[0] = -1.0
    Aub, bub = [], []
    for k, (s, _t) in enumerate(sessions):
        si = ni[s]; row = np.zeros(nv); row[0] = 1.0
        for vi in adj[si]:
            f = arc_idx.get((si, vi)); r_ = arc_idx.get((vi, si))
            if f is not None: row[xv(f, k)] -= 1.0
            if r_ is not None: row[xv(r_, k)] += 1.0
        Aub.append(row); bub.append(0.0)
    for u, v in edges:
        a, b = ni[u], ni[v]
        row = np.zeros(nv)
        for k in range(n_comm):
            row[xv(arc_idx[(a, b)], k)] = 1.0
            row[xv(arc_idx[(b, a)], k)] = 1.0
        Aub.append(row); bub.append(1.0)
    Aeq, beq = [], []
    for i in range(n_nodes):
        for k, (s, t) in enumerate(sessions):
            if i == ni[s] or i == ni[t]:
                continue
            row = np.zeros(nv)
            for vi in adj[i]:
                f = arc_idx.get((i, vi)); r_ = arc_idx.get((vi, i))
                if f is not None: row[xv(f, k)] += 1.0
                if r_ is not None: row[xv(r_, k)] -= 1.0
            Aeq.append(row); beq.append(0.0)
    res = linprog(c, A_ub=np.array(Aub), b_ub=np.array(bub),
                  A_eq=np.array(Aeq) if Aeq else None,
                  b_eq=np.array(beq) if beq else None,
                  bounds=[(0, None)] * nv, method="highs")
    if not res.success:
        return 0.0, {}
    r = max(-res.fun, 0.0)
    flow = {}
    for u, v in edges:
        a, b = ni[u], ni[v]
        tot = sum(res.x[xv(arc_idx[(a, b)], k)] + res.x[xv(arc_idx[(b, a)], k)]
                  for k in range(n_comm))
        flow[(u, v)] = tot
    return r, flow


def entropy_vector(index, nodes, edges, sessions, r, flow):
    """Entropy of every indexed variable under the routing code at rate r."""
    h = np.zeros(index.dim)
    n_I = len(sessions)
    h[index.yi_idx()] = n_I * r
    for i in range(len(index.partitions)):
        h[index.yst_idx(i)] = len(index.st_sessions[i]) * r
        h[index.yi_pi_idx(i)] = len(index.internal_sessions[i]) * r
    for v in nodes:
        h[index.source_idx(v)] = sum(1 for s, _t in sessions if s == v) * r
    for e in edges:
        h[index.edge_idx(e)] = flow.get(e, flow.get((e[1], e[0]), 0.0))
    return h


def violation(ineq, h, tol=1e-7):
    """sum(coeffs*h). > 0 means the inequality is FALSE at this point."""
    return float(np.dot(ineq.coeffs, h))


def is_falsified(ineq, h, tol=1e-7):
    return violation(ineq, h) > tol
