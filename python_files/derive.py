"""
DERIVATION SYSTEM  --  built from scratch, exact rational arithmetic.

PRIMITIVES (both are the same family; node IO is the |P|=1 case)
---------------------------------------------------------------
For any node set P (P = {v} gives NODE IO, |P|>1 gives PARTITION IO):

    Obs(P) = ( Y_{Src(P)} , U_{In(P)} )       In(P) = arcs from outside P into P

    D_P  = sessions with source OR sink in P     (P determines all of them:
           source -> it holds the message; sink -> it decodes)

    h(Y_{D_P}) <= h(Obs(P))                       [D_P is a function of Obs(P)]
                <= h(Y_{Src(P)}) + SUM_{a in In(P)} h(U_a)     [subadditivity]

    |D_P| r <= |Src(P)| r + SUM_{a in In(P)} h(U_a)            [source indep.]

    IO(P):   (|D_P| - |Src(P)|) r  <=  SUM_{a in In(P)} h(U_a)

ACTIONS
-------
  NODE_IO(v)            create IO({v})
  PART_IO(P)            create IO(P)
  FRAC(A,B,lam,mu)      lam*A + mu*B , lam,mu >= 0   (nonneg comb of valid = valid)
  EXTRACT               apply capacity h(U_uv)+h(U_vu) <= c_uv

BOUND EXTRACTION
----------------
  final:  c*r <= SUM_a beta_a h(U_a)
  maximise RHS under capacity: each undirected edge contributes
      max(beta_{u->v}, beta_{v->u}) * c_uv
  =>  r <= SUM_edges max(beta_{u->v}, beta_{v->u}) * c_uv / c

VERIFICATION (three independent checks)
---------------------------------------
  V1 falsification : evaluate every intermediate against an explicit achievable
                     code at r = LP_LB.  Any violation = the step is false.
  V2 Farkas        : final inequality must be a nonneg combination of the base
                     IO generators, coefficients recorded and re-summed exactly.
  V3 arithmetic    : recompute the bound from the final coefficient vector with
                     Fractions, independently of the search.
"""
import sys, itertools, json
from fractions import Fraction as F
sys.path.insert(0, '/home/claude/work/rl_networks-main/python_files')
import numpy as np
from scipy.optimize import linprog
import fixed_graph_generation as G
G._build_registry()
REG = {g.name: g for g in G.GRAPH_REGISTRY}


class Net:
    def __init__(self, name):
        g = REG[name]
        self.name = name
        self.nodes = list(g.nodes)
        self.edges = [tuple(e) for e in g.edges]
        self.sess = [tuple(s) for s in g.sessions]
        self.I = len(self.sess)
        self.arcs = [(u, v) for (u, v) in self.edges] + [(v, u) for (u, v) in self.edges]
        self.ai = {a: i for i, a in enumerate(self.arcs)}

    def In(self, P):
        P = set(P)
        return [a for a in self.arcs if a[0] not in P and a[1] in P]

    def D(self, P):
        P = set(P)
        return [i for i, (s, t) in enumerate(self.sess) if s in P or t in P]

    def Src(self, P):
        P = set(P)
        return [i for i, (s, t) in enumerate(self.sess) if s in P]


class Ineq:
    """ c * r  <=  sum_a beta[a] * h(U_a)     (c, beta all Fractions) """
    def __init__(self, net, c=F(0), beta=None, prov=None):
        self.net = net
        self.c = c
        self.beta = beta if beta is not None else [F(0)] * len(net.arcs)
        self.prov = prov or {}          # generator name -> multiplier

    @staticmethod
    def IO(net, P):
        P = tuple(sorted(P))
        c = F(len(net.D(P)) - len(net.Src(P)))
        beta = [F(0)] * len(net.arcs)
        for a in net.In(P):
            beta[net.ai[a]] += F(1)
        return Ineq(net, c, beta, {f"IO{P}": F(1)})

    def frac(self, other, lam, mu):
        """lam*self + mu*other , lam,mu >= 0"""
        assert lam >= 0 and mu >= 0
        c = lam * self.c + mu * other.c
        beta = [lam * x + mu * y for x, y in zip(self.beta, other.beta)]
        prov = {}
        for k, v in self.prov.items(): prov[k] = prov.get(k, F(0)) + lam * v
        for k, v in other.prov.items(): prov[k] = prov.get(k, F(0)) + mu * v
        return Ineq(self.net, c, beta, prov)

    def extract(self):
        """apply capacity pairing; returns Fraction bound or None"""
        if self.c <= 0: return None
        tot = F(0)
        for (u, v) in self.net.edges:
            b1 = self.beta[self.net.ai[(u, v)]]
            b2 = self.beta[self.net.ai[(v, u)]]
            tot += max(b1, b2)                      # unit capacity
        return tot / self.c


# ---------- optimal weights over a candidate set-collection (LP) ----------
def best_over_sets(net, cand):
    m = len(cand)
    dvec = [len(net.D(P)) - len(net.Src(P)) for P in cand]
    if all(d <= 0 for d in dvec): return None, None
    E = net.edges; ne = len(E)
    NV = m + ne
    c = np.zeros(NV); c[m:] = 1.0
    A, b = [], []
    for k, (u, v) in enumerate(E):
        for (x, y) in ((u, v), (v, u)):          # t_e >= sum of weights charging arc x->y
            row = np.zeros(NV); row[m + k] = -1.0
            for j, P in enumerate(cand):
                if y in set(P) and x not in set(P): row[j] += 1.0
            A.append(row); b.append(0.0)
    Aeq = np.zeros((1, NV)); Aeq[0, :m] = dvec
    r = linprog(c, A_ub=np.array(A), b_ub=np.array(b), A_eq=Aeq,
                b_eq=np.array([1.0]), bounds=[(0, None)] * NV, method='highs')
    if not r.success: return None, None
    return r.fun, r.x[:m]


# ---------- V1 : explicit achievable code via multicommodity flow LP ----------
def achievable_flow(net):
    """Max concurrent multicommodity flow. Returns (r, {arc: flow}).
       In a routing code h(U_a) = total distinct information on arc a = flow_a."""
    K = len(net.sess); A = net.arcs; na = len(A)
    # vars: f[a][k]  (na*K)  then r
    NV = na * K + 1; R = NV - 1
    fi = lambda a, k: net.ai[a] * K + k
    Aeq, beq = [], []
    for k, (s, t) in enumerate(net.sess):
        for v in net.nodes:
            row = np.zeros(NV)
            for a in A:
                if a[1] == v: row[fi(a, k)] += 1.0     # into v
                if a[0] == v: row[fi(a, k)] -= 1.0     # out of v
            if v == s: row[R] += 1.0                    # net out = r
            elif v == t: row[R] -= 1.0
            Aeq.append(row); beq.append(0.0)
    Aub, bub = [], []
    for (u, v) in net.edges:
        row = np.zeros(NV)
        for k in range(K):
            row[fi((u, v), k)] = 1.0
            row[fi((v, u), k)] = 1.0
        Aub.append(row); bub.append(1.0)
    c = np.zeros(NV); c[R] = -1.0
    res = linprog(c, A_ub=np.array(Aub), b_ub=np.array(bub),
                  A_eq=np.array(Aeq), b_eq=np.array(beq),
                  bounds=[(0, None)] * NV, method='highs')
    if not res.success: return None, None
    r = -res.fun
    flow = {a: sum(res.x[fi(a, k)] for k in range(K)) for a in A}
    return r, flow


def verify(net, ineq, bound, lp_lb, verbose=True):
    ok = True
    log = []
    # V3 arithmetic recompute
    tot = F(0)
    for (u, v) in net.edges:
        tot += max(ineq.beta[net.ai[(u, v)]], ineq.beta[net.ai[(v, u)]])
    b2 = tot / ineq.c
    log.append(("V3 arithmetic", b2 == bound, f"recomputed {b2} vs {bound}"))
    ok &= (b2 == bound)
    # V2 Farkas : re-sum from provenance
    c2 = F(0); beta2 = [F(0)] * len(net.arcs)
    for gname, mult in ineq.prov.items():
        P = eval(gname[2:])
        if not isinstance(P, tuple): P = (P,)
        g = Ineq.IO(net, P)
        c2 += mult * g.c
        for i in range(len(beta2)): beta2[i] += mult * g.beta[i]
        if mult < 0: ok = False
    match = (c2 == ineq.c) and all(x == y for x, y in zip(beta2, ineq.beta))
    allpos = all(m >= 0 for m in ineq.prov.values())
    log.append(("V2 Farkas", match and allpos,
                f"resum matches={match}, all multipliers >=0: {allpos}"))
    ok &= (match and allpos)
    # V1 falsification against a genuine max-concurrent-flow code
    rr, flow = achievable_flow(net)
    if flow is None:
        log.append(("V1 falsification", False, "flow LP failed")); ok = False
    else:
        TOL = 1e-7
        capviol = max(flow[(u,v)] + flow[(v,u)] - 1.0 for (u,v) in net.edges)
        consok = abs(rr - lp_lb) < 1e-6
        lhs = float(ineq.c) * rr
        rhs = sum(float(ineq.beta[net.ai[a]]) * flow[a] for a in net.arcs)
        v1 = lhs <= rhs + TOL
        if not v1: ok = False
        log.append(("V1 falsification", v1,
                    f"r={rr:.4f} LHS {lhs:.4f} <= RHS {rhs:.4f} : {v1}"
                    f"  [cap slack {-capviol:.2e}, r matches LP_LB: {consok}]"))
        # V1b: every BASE generator must also hold at this point
        bad = []
        for gname in ineq.prov:
            P = eval(gname[2:])
            if not isinstance(P, tuple): P = (P,)
            g = Ineq.IO(net, P)
            gl = float(g.c) * rr
            gr = sum(float(g.beta[net.ai[a]]) * flow[a] for a in net.arcs)
            if gl > gr + TOL: bad.append((P, gl, gr))
        log.append(("V1b generators", not bad,
                    "all IO generators hold at the achievable point"
                    if not bad else f"VIOLATED: {bad}"))
        if bad: ok = False
    # V4 sanity: bound must be >= LP lower bound
    v4 = bound >= F(lp_lb)
    log.append(("V4 bound >= LP_LB", v4, f"{float(bound):.4f} >= {lp_lb}"))
    ok &= v4
    if verbose:
        for nm, good, msg in log:
            print(f"      [{'PASS' if good else 'FAIL'}] {nm:20s} {msg}")
    return ok


# ---------------------- run ----------------------
lpb = json.load(open('/home/claude/work/rl_networks-main/config_files/eval_results.json'))['lp_bounds']
print(f"{'graph':26s} {'BOUND':>9s} {'PB':>7s} {'LP':>6s}  {'beats PB':>8s}  sets used")
print('-' * 96)
results = []
for name, g in REG.items():
    net = Net(name)
    n = len(net.nodes)
    # candidate collection: all singletons + all connected node subsets up to size n-1
    cand = [(v,) for v in net.nodes]
    if n <= 12:
        for k in range(2, n):
            for P in itertools.combinations(net.nodes, k):
                cand.append(P)
    else:
        for k in (2, 3, n - 3, n - 2, n - 1):
            if 1 < k < n:
                for P in itertools.combinations(net.nodes, k): cand.append(P)
    cand = list(dict.fromkeys(cand))
    val, w = best_over_sets(net, cand)
    if val is None: continue
    # rebuild the derivation exactly with Fractions from the LP support
    sup = [(cand[j], F(w[j]).limit_denominator(10**6)) for j in range(len(cand)) if w[j] > 1e-9]
    acc = Ineq(net)
    for P, lam in sup:
        acc = acc.frac(Ineq.IO(net, P), F(1), lam)
    bound = acc.extract()
    pb = G.compute_optimal_bound(net.nodes, net.edges, net.sess)
    pb = pb[0] if isinstance(pb, tuple) else pb
    beat = "YES" if float(bound) < pb - 1e-9 else "no"
    ss = " + ".join(f"{lam}·IO{{{','.join(P)}}}" for P, lam in sup[:3])
    print(f"{name:26s} {float(bound):9.4f} {pb:7.4f} {lpb[name]:6.3f}  {beat:>8s}  {ss}")
    results.append((name, net, acc, bound, lpb[name], pb, beat))

print("\n" + "=" * 96)
print("END-TO-END VERIFICATION")
print("=" * 96)
allok = True
for name, net, acc, bound, lb, pb, beat in results:
    print(f"\n  {name}  bound = {bound} = {float(bound):.6f}   (PB {pb}, LP {lb})")
    allok &= verify(net, acc, bound, lb)
print(f"\n{'='*96}\nALL CHECKS PASSED: {allok}")
nb = sum(1 for r in results if r[6] == "YES")
print(f"beats PB on {nb} / {len(results)} graphs")
