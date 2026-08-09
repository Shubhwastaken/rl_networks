"""
Falsification test for the joint-entropy LP.

Builds an EXPLICIT achievable routing code at rate r = LP LB, computes the
true entropy of every curated set under that code, and checks each LP
constraint at that point. The code is constructed by path decomposition:

  - solve the max symmetric multicommodity flow
  - decompose each commodity's flow into source->sink paths with weights
  - give session k the message interval [0, r); allocate disjoint
    sub-intervals to its paths in proportion to the path weights
  - arc a then carries, for commodity k, exactly the union of the
    sub-intervals of the paths that traverse a

Under independent sources the entropy of a set S is the total measure of
distinct (session, sub-interval) pieces reachable from the elements of S:

  h(S) = sum_k measure( union of intervals of session k carried by S )

Y_k contributes session k's whole interval. This is a genuine network
code (routing is a valid code), so ANY valid constraint must hold at this
point. A violated constraint is a proof that the constraint is not a
theorem, and hence that the LP bound is unsound.
"""
import numpy as np
from scipy.optimize import linprog
from _jointlp import Ground


def _flow_lp(nodes, edges, sessions):
    ni = {n: i for i, n in enumerate(nodes)}
    arcs, aidx = [], {}
    for u, v in edges:
        a, b = ni[u], ni[v]
        aidx[(a, b)] = len(arcs); arcs.append((a, b))
        aidx[(b, a)] = len(arcs); arcs.append((b, a))
    K = len(sessions); nv = 1 + len(arcs) * K
    xv = lambda a, k: 1 + a * K + k
    adj = {i: [] for i in range(len(nodes))}
    for u, v in edges:
        adj[ni[u]].append(ni[v]); adj[ni[v]].append(ni[u])
    Aub, bub, Aeq, beq = [], [], [], []
    for k, (s, _t) in enumerate(sessions):
        si = ni[s]; row = np.zeros(nv); row[0] = 1.0
        for w in adj[si]:
            row[xv(aidx[(si, w)], k)] -= 1.0
            row[xv(aidx[(w, si)], k)] += 1.0
        Aub.append(row); bub.append(0.0)
    for u, v in edges:
        a, b = ni[u], ni[v]; row = np.zeros(nv)
        for k in range(K):
            row[xv(aidx[(a, b)], k)] = 1.0; row[xv(aidx[(b, a)], k)] = 1.0
        Aub.append(row); bub.append(1.0)
    for i in range(len(nodes)):
        for k, (s, t) in enumerate(sessions):
            if i == ni[s] or i == ni[t]: continue
            row = np.zeros(nv)
            for w in adj[i]:
                row[xv(aidx[(i, w)], k)] += 1.0
                row[xv(aidx[(w, i)], k)] -= 1.0
            Aeq.append(row); beq.append(0.0)
    c = np.zeros(nv); c[0] = -1.0
    res = linprog(c, A_ub=np.array(Aub), b_ub=np.array(bub),
                  A_eq=np.array(Aeq) if Aeq else None,
                  b_eq=np.array(beq) if beq else None,
                  bounds=[(0, None)] * nv, method='highs')
    if not res.success: return None, None, None, None
    return max(-res.fun, 0.0), res.x, aidx, ni


def path_decompose(nodes, edges, sessions, x, aidx, ni, tol=1e-9):
    """Return {session: [(weight, [directed arcs])]} by repeated path peeling."""
    K = len(sessions)
    flow = {}
    for (a, b), ai in aidx.items():
        for k in range(K):
            f = x[1 + ai * K + k]
            if f > tol: flow[(a, b, k)] = f
    out = {k: [] for k in range(K)}
    for k, (s, t) in enumerate(sessions):
        si, ti = ni[s], ni[t]
        for _ in range(500):
            # find a path si->ti in the residual support of commodity k
            stack, prev = [si], {si: None}
            while stack:
                u = stack.pop()
                if u == ti: break
                for (a, b, kk), f in list(flow.items()):
                    if kk == k and a == u and f > tol and b not in prev:
                        prev[b] = (a, b); stack.append(b)
            if ti not in prev: break
            path, cur = [], ti
            while prev[cur] is not None:
                a, b = prev[cur]; path.append((a, b)); cur = a
            path.reverse()
            w = min(flow[(a, b, k)] for a, b in path)
            if w <= tol: break
            for a, b in path:
                flow[(a, b, k)] -= w
                if flow[(a, b, k)] <= tol: flow.pop((a, b, k), None)
            out[k].append((w, path))
    return out


def code_entropy(g, nodes, edges, sessions, r, paths, aidx, ni):
    """Interval allocation -> a function h(mask) giving the true entropy."""
    # session k: intervals [lo,hi) inside [0,r); one per decomposed path
    seg = {}          # (k, seg_id) -> measure
    arc_segs = {}     # directed arc elem index -> set of (k, seg_id)
    for k in range(len(sessions)):
        tot = sum(w for w, _ in paths[k])
        acc = 0.0
        for j, (w, path) in enumerate(paths[k]):
            seg[(k, j)] = w
            acc += w
            for (a, b) in path:
                u, v = nodes[a], nodes[b]
                e = g.arc[(u, v)]
                arc_segs.setdefault(e, set()).add((k, j))
        # any unrouted remainder of the message is carried by no arc
        if tot < r - 1e-9:
            seg[(k, 'rest')] = r - tot

    y_segs = {}
    for k in range(len(sessions)):
        y_segs[g.yi[k]] = {s for s in seg if s[0] == k}

    def h(mask):
        acc = set()
        for i in g.bits(mask):
            kind = g.elem[i][0]
            if kind == 'Y': acc |= y_segs.get(i, set())
            else: acc |= arc_segs.get(i, set())
        return sum(seg[s] for s in acc)
    return h


def build_code(nodes, edges, sessions):
    g = Ground(nodes, edges, sessions)
    r, x, aidx, ni = _flow_lp(nodes, edges, sessions)
    if r is None: return None, None, None
    paths = path_decompose(nodes, edges, sessions, x, aidx, ni)
    h = code_entropy(g, nodes, edges, sessions, r, paths, aidx, ni)
    return g, r, h
