"""
EXTRACT THE PROOF CERTIFICATE for  r <= 3/4  on K_2,3.

The primal LP is    max r   s.t.  A_ub x <= b_ub,  A_eq x = b_eq,  x >= 0
where x holds h(S) for each curated joint term S, plus r.

LP duality: at the optimum there exist multipliers
    y >= 0  on the inequality rows,   z (free) on the equality rows
such that the nonnegative combination

    sum_i y_i * (row_i . x)  +  sum_j z_j * (eqrow_j . x)   <=   sum_i y_i b_i

has left side >= r  (coefficient of r is 1, all other coefficients >= 0 since x >= 0).
That combination IS the proof: it is a finite list of valid inequalities with
nonnegative weights whose sum reads  r <= 0.75.

We then verify, independently of the solver:
  (V-a) every multiplier on an inequality row is >= 0
  (V-b) the weighted sum's coefficient on r is exactly 1
  (V-c) the weighted sum's coefficient on every h(S) is >= 0
  (V-d) the RHS total equals 3/4
  (V-e) every constraint carrying a nonzero multiplier is TRUE at an
        explicit achievable point (so we are not certifying with a false rule)
"""
import sys, itertools, json, collections
from fractions import Fraction as F
sys.path.insert(0, '/home/claude/work/rl_networks-main/python_files')
import numpy as np
import scipy.sparse as SPARSE
from scipy.optimize import linprog
import fixed_graph_generation as G
G._build_registry()
REG = {g.name: g for g in G.GRAPH_REGISTRY}

import os
NAME = os.environ.get("GRAPH","okamura_network_paper_5N")
BIG = os.environ.get("BIG","0")=="1"
g = REG[NAME]
nodes = list(g.nodes); edges = [tuple(e) for e in g.edges]; sess = [tuple(s) for s in g.sessions]
arcs = [(u, v) for (u, v) in edges] + [(v, u) for (u, v) in edges]
K = len(sess)
In = lambda P: [a for a in arcs if a[0] not in set(P) and a[1] in set(P)]
Out = lambda P: [a for a in arcs if a[0] in set(P) and a[1] not in set(P)]
Src = lambda P: [i for i, (s, t) in enumerate(sess) if s in set(P)]

# ---------- small curated collection so the certificate is readable ----------
C = set()
def add(Y, A): C.add((frozenset(Y), frozenset(A)))
for k in range(K): add([k], [])
for a in arcs: add([], [a])
for kk in range(1, K + 1):
    for S in itertools.combinations(range(K), kk): add(S, [])
for kk in range(1, len(nodes)):
    for P in itertools.combinations(nodes, kk):
        cut = In(P); sp = Src(P)
        dec = [i for i, (s, t) in enumerate(sess) if t in set(P) and s not in set(P)]
        add([], cut); add(sp, cut)
        if dec: add(sp + dec, cut)
if BIG:
    seeds=set(C); cur=sorted(C,key=lambda z:(len(z[0])+len(z[1])))
    new=set()
    for i in range(len(cur)):
        for j in range(i+1,len(cur)):
            new.add((cur[i][0]|cur[j][0], cur[i][1]|cur[j][1]))
            it=(cur[i][0]&cur[j][0], cur[i][1]&cur[j][1])
            if it[0] or it[1]: new.add(it)
    CAP=int(os.environ.get("CAP","900"))
    new-=seeds
    C = seeds | set(sorted(new,key=lambda z:-(len(z[0])+len(z[1])))[:max(0,CAP-len(seeds))])
C = sorted(C, key=lambda z: (len(z[0]) + len(z[1]), sorted(z[0]), sorted(map(str, z[1]))))
idx = {z: i for i, z in enumerate(C)}
NV = len(C) + 1; R = NV - 1
def nm(S):
    p = []
    if S[0]: p.append("Y{" + ",".join(str(k + 1) for k in sorted(S[0])) + "}")
    if S[1]: p.append("U{" + ",".join(f"{u}>{v}" for (u, v) in sorted(S[1])) + "}")
    return "h(" + " , ".join(p) + ")"

UR, UC, UV, bU, tU = [], [], [], [], []
def ub(pairs, b, tag):
    k = len(bU)
    for c, v in pairs: UR.append(k); UC.append(c); UV.append(v)
    bU.append(b); tU.append(tag)
ER, EC, EV, bE, tE = [], [], [], [], []
def eq(pairs, b, tag):
    k = len(bE)
    for c, v in pairs: ER.append(k); EC.append(c); EV.append(v)
    bE.append(b); tE.append(tag)

Cset = set(C)
for i, S in enumerate(C):
    for j, T in enumerate(C):
        if i >= j: continue
        if S[0] <= T[0] and S[1] <= T[1]:
            ub([(idx[S], 1.0), (idx[T], -1.0)], 0.0, f"MONO  {nm(S)} <= {nm(T)}")
        elif T[0] <= S[0] and T[1] <= S[1]:
            ub([(idx[T], 1.0), (idx[S], -1.0)], 0.0, f"MONO  {nm(T)} <= {nm(S)}")
        U = (S[0] | T[0], S[1] | T[1]); I2 = (S[0] & T[0], S[1] & T[1])
        if U in Cset:
            ub([(idx[U], 1.0), (idx[S], -1.0), (idx[T], -1.0)], 0.0,
               f"SUBADD  {nm(U)} <= {nm(S)} + {nm(T)}")
            if (I2[0] or I2[1]) and I2 in Cset:
                ub([(idx[U], 1.0), (idx[I2], 1.0), (idx[S], -1.0), (idx[T], -1.0)], 0.0,
                   f"SUBMOD  {nm(U)}+{nm(I2)} <= {nm(S)}+{nm(T)}")
for S in C:
    if len(S[0]) + len(S[1]) > 1:
        ub([(idx[S], 1.0)] + [(idx[(frozenset([k]), frozenset())], -1.0) for k in S[0]]
           + [(idx[(frozenset(), frozenset([a]))], -1.0) for a in S[1]], 0.0,
           f"SUBADD*  {nm(S)} <= sum of singletons")
for (u, v) in edges:
    ub([(idx[(frozenset(), frozenset([(u, v)]))], 1.0),
        (idx[(frozenset(), frozenset([(v, u)]))], 1.0)], 1.0,
       f"CAPACITY  h(U{{{u}>{v}}})+h(U{{{v}>{u}}}) <= 1")
for kk in range(1, K + 1):
    for S in itertools.combinations(range(K), kk):
        eq([(idx[(frozenset(S), frozenset())], 1.0), (R, -float(kk))], 0.0,
           f"SRC-INDEP  {nm((frozenset(S),frozenset()))} = {kk} r")
for v in nodes:                                     # ENCODING
    base = (frozenset(Src([v])), frozenset(In([v])))
    if not Out([v]): continue
    for S in C:
        if base[0] <= S[0] and base[1] <= S[1]:
            T = (S[0], S[1] | frozenset(Out([v])))
            if T in Cset and T != S:
                eq([(idx[T], 1.0), (idx[S], -1.0)], 0.0, f"ENCODING@{v}  {nm(T)} = {nm(S)}")
for kk in range(1, len(nodes)):                     # SET-DECODING (= crypto, on joints)
    for P in itertools.combinations(nodes, kk):
        base = (frozenset(Src(P)), frozenset(In(P)))
        dec = frozenset(i for i, (s, t) in enumerate(sess) if t in set(P) and s not in set(P))
        if not dec: continue
        for S in C:
            if base[0] <= S[0] and base[1] <= S[1]:
                T = (S[0] | dec, S[1])
                if T in Cset and T != S:
                    eq([(idx[T], 1.0), (idx[S], -1.0)], 0.0,
                       f"SETDEC P={{{','.join(P)}}}  {nm(T)} = {nm(S)}")

A_ub = SPARSE.csr_matrix((UV, (UR, UC)), shape=(len(bU), NV))
A_eq = SPARSE.csr_matrix((EV, (ER, EC)), shape=(len(bE), NV))
c = np.zeros(NV); c[R] = -1.0
res = linprog(c, A_ub=A_ub, b_ub=np.array(bU), A_eq=A_eq, b_eq=np.array(bE),
              bounds=[(0, None)] * NV, method='highs')
BOUND = -res.fun
print(f"{NAME}   curated terms={len(C)}   ub rows={len(bU)}  eq rows={len(bE)}")
print(f"PRIMAL BOUND  r <= {BOUND:.10f}\n")

# ---------------- dual multipliers ----------------
y = -np.asarray(res.ineqlin.marginals)     # >= 0 for <= rows
z = -np.asarray(res.eqlin.marginals)
TOL = 1e-9
usedU = [(i, y[i]) for i in range(len(bU)) if abs(y[i]) > 1e-8]
usedE = [(j, z[j]) for j in range(len(bE)) if abs(z[j]) > 1e-8]
print(f"CERTIFICATE uses {len(usedU)} inequality rows + {len(usedE)} equality rows "
      f"out of {len(bU)}+{len(bE)}\n")

print("--- inequality rows (multiplier > 0) ---")
fam = collections.Counter()
for i, m in sorted(usedU, key=lambda t: -abs(t[1])):
    fam[tU[i].split()[0]] += 1
for i, m in sorted(usedU, key=lambda t: -abs(t[1]))[:14]:
    print(f"  {m:+8.4f}  x  {tU[i][:88]}")
print(f"  ... families: {dict(fam)}")
print("\n--- equality rows (multiplier != 0) ---")
famE = collections.Counter()
for j, m in usedE: famE[tE[j].split()[0]] += 1
for j, m in sorted(usedE, key=lambda t: -abs(t[1]))[:12]:
    print(f"  {m:+8.4f}  x  {tE[j][:88]}")
print(f"  ... families: {dict(famE)}")

# ---------------- VERIFY the certificate ----------------
print("\n" + "=" * 80)
print("CERTIFICATE VERIFICATION (independent of the solver)")
print("=" * 80)
Au = A_ub.toarray(); Ae = A_eq.toarray()
comb = np.zeros(NV)
for i, m in usedU: comb += m * Au[i]
for j, m in usedE: comb += m * Ae[j]
rhs_tot = sum(m * bU[i] for i, m in usedU) + sum(m * bE[j] for j, m in usedE)
va = all(m >= -1e-9 for _, m in usedU)
# want: (sum y_i row_i).x <= sum y_i b_i  with coeff +1 on r and >=0 elsewhere,
# then since x>=0:  r <= r + (nonneg) <= RHS
vb = abs(comb[R] - 1.0) < 1e-6
vc = all(comb[k] >= -1e-6 for k in range(NV - 1))
vd = abs(rhs_tot - BOUND) < 1e-6
print(f"  [{'PASS' if va else 'FAIL'}] V-a  all inequality multipliers >= 0")
print(f"  [{'PASS' if vb else 'FAIL'}] V-b  combined coefficient on r = {comb[R]:+.8f} (want +1)")
print(f"  [{'PASS' if vc else 'FAIL'}] V-c  all h(S) coefficients >= 0 "
      f"(min {min(comb[:NV-1]):+.2e})")
print(f"  [{'PASS' if vd else 'FAIL'}] V-d  RHS total = {rhs_tot:.10f}  vs bound {BOUND:.10f}")
print(f"\n  => the weighted sum reads:   {comb[R]:.6f} * r  +  (nonneg terms)  <=  {rhs_tot:.6f}")
print(f"     hence  r <= {rhs_tot:.6f} = {F(rhs_tot).limit_denominator(1000)}")

# ---------------- V-e : achievable point ----------------
def flow_solution():
    na = len(arcs); ai = {a: i for i, a in enumerate(arcs)}
    NW = na * K + 1; RR = NW - 1
    fi = lambda a, k: ai[a] * K + k
    Aq, bq = [], []
    for k, (s, t) in enumerate(sess):
        for v in nodes:
            row = np.zeros(NW)
            for a in arcs:
                if a[1] == v: row[fi(a, k)] += 1.0
                if a[0] == v: row[fi(a, k)] -= 1.0
            if v == s: row[RR] += 1.0
            elif v == t: row[RR] -= 1.0
            Aq.append(row); bq.append(0.0)
    Aq2, bq2 = [], []
    for (u, v) in edges:
        row = np.zeros(NW)
        for k in range(K): row[fi((u, v), k)] = 1.0; row[fi((v, u), k)] = 1.0
        Aq2.append(row); bq2.append(1.0)
    cc = np.zeros(NW); cc[RR] = -1.0
    r2 = linprog(cc, A_ub=np.array(Aq2), b_ub=np.array(bq2), A_eq=np.array(Aq),
                 b_eq=np.array(bq), bounds=[(0, None)] * NW, method='highs')
    return -r2.fun, {(a, k): r2.x[fi(a, k)] for a in arcs for k in range(K)}

r_ach, fk = flow_solution()
pieces = []; arc_p = collections.defaultdict(set); src_p = collections.defaultdict(set)
for k in range(K):
    resid = {a: fk[(a, k)] for a in arcs}; s, t = sess[k]
    for _ in range(200):
        stack = [(s, [s], float('inf'))]; found = None
        while stack:
            v, path, bt = stack.pop()
            if v == t: found = (path, bt); break
            for a in arcs:
                if a[0] == v and resid[a] > 1e-9 and a[1] not in path:
                    stack.append((a[1], path + [a[1]], min(bt, resid[a])))
        if not found: break
        path, bt = found
        for i in range(len(path) - 1): resid[(path[i], path[i + 1])] -= bt
        pid = len(pieces); pieces.append(bt); src_p[k].add(pid)
        for i in range(len(path) - 1): arc_p[(path[i], path[i + 1])].add(pid)
def H(S):
    ps = set()
    for k in S[0]: ps |= src_p[k]
    for a in S[1]: ps |= arc_p[a]
    return sum(pieces[p] for p in ps)
xv = np.zeros(NV)
for S in C: xv[idx[S]] = H(S)
xv[R] = r_ach
badU = [tU[i] for i, m in usedU if Au[i] @ xv - bU[i] > 1e-7]
badE = [tE[j] for j, m in usedE if abs(Ae[j] @ xv - bE[j]) > 1e-7]
chain=[(m,tE[j]) for j,m in usedE if tE[j].startswith("SETDEC")]
print("\n--- SET-DECODING CHAIN (the mechanism) ---")
for m,t in sorted(chain,key=lambda z:-abs(z[0])):
    print(f"  {m:+7.4f}  {t[:150]}")
print(f"  chain length = {len(chain)}")
print(f"\n  [{'PASS' if not badU and not badE else 'FAIL'}] V-e  every constraint in the "
      f"certificate is TRUE at the achievable point (r={r_ach:.4f})")
if badU or badE: print("        violations:", (badU + badE)[:5])
print(f"  [{'PASS' if BOUND >= r_ach - 1e-9 else 'FAIL'}] V-f  bound {BOUND:.6f} >= achievable {r_ach:.6f}")
