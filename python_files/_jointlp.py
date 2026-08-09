"""
STANDALONE PROTOTYPE — curated-collection joint-entropy LP.

Independent rebuild of the formulation, sharing no code with the main
pipeline (which is untouched).

Ground set
    Y_i                 one per session
    U_{u->v}, U_{v->u}  two directed arc signals per undirected edge

Variables
    h(S) for every S in a curated collection C, plus the rate r.

Because C is a strict subset of 2^ground, the LP has fewer variables and
constraints than the full Shannon LP, so its feasible region is LARGER
and its optimum is therefore still a valid UPPER bound on r. Dropping
constraints can only loosen an upper bound, never tighten it below truth.

Sets are bitmasks over ground-element indices.
"""
import itertools
from collections import OrderedDict
import numpy as np
from scipy.sparse import coo_matrix
from scipy.optimize import linprog


class Ground:
    def __init__(self, nodes, edges, sessions):
        self.nodes = list(nodes)
        self.edges = [tuple(e) for e in edges]
        self.sessions = [tuple(s) for s in sessions]
        self.elem = []                       # index -> ('Y',i) / ('U',u,v)
        self.yi = {}                         # session -> elem index
        self.arc = {}                        # (u,v) directed -> elem index
        for i in range(len(self.sessions)):
            self.yi[i] = len(self.elem); self.elem.append(('Y', i))
        for (u, v) in self.edges:
            self.arc[(u, v)] = len(self.elem); self.elem.append(('U', u, v))
            self.arc[(v, u)] = len(self.elem); self.elem.append(('U', v, u))
        self.n = len(self.elem)
        self.In, self.Out, self.Src = {}, {}, {}
        for v in self.nodes:
            self.In[v] = self._m(self.arc[(u, w)] for (u, w) in self.arc if w == v)
            self.Out[v] = self._m(self.arc[(u, w)] for (u, w) in self.arc if u == v)
            self.Src[v] = self._m(self.yi[i] for i, (s, t) in enumerate(self.sessions) if s == v)

    @staticmethod
    def _m(idxs):
        m = 0
        for i in idxs: m |= (1 << i)
        return m

    def bits(self, mask):
        out = []
        i = 0
        while mask:
            if mask & 1: out.append(i)
            mask >>= 1; i += 1
        return out

    def in_cut(self, S):
        """{U_{u->v} : u not in S, v in S} for a node set S."""
        return self._m(self.arc[(u, v)] for (u, v) in self.arc
                       if u not in S and v in S)

    def src_of(self, S):
        m = 0
        for v in S: m |= self.Src[v]
        return m

    def sunk_in(self, S):
        """sessions sunk inside S with source outside S."""
        return self._m(self.yi[i] for i, (s, t) in enumerate(self.sessions)
                       if t in S and s not in S)


def build_collection(g, cap=700, node_subset_cap=4096, seed=0,
                     randomize_closure=False):
    """Seeds first (NEVER dropped), then pairwise union/intersection closure
    up to `cap`. Returns (list_of_masks, n_seeds)."""
    seeds = OrderedDict()

    def add(m):
        if m: seeds.setdefault(m, None)

    for i in range(g.n): add(1 << i)                       # singletons
    for v in g.nodes:                                      # per node
        add(g.In[v]); add(g.Out[v])
        add(g.In[v] | g.Src[v])
        add(g.In[v] | g.Out[v] | g.Src[v])
    for i, (s, t) in enumerate(g.sessions):                # per sink
        base = g.In[t] | g.Src[t]
        add(base); add(base | (1 << g.yi[i]))
    allY = list(g.yi.values())                             # all subsets of {Y_i}
    for k in range(1, len(allY) + 1):
        for comb in itertools.combinations(allY, k):
            add(g._m(comb))
    rng = np.random.default_rng(seed)                      # per node subset
    nodes = g.nodes; N = len(nodes)
    subsets = []
    if 2 ** N <= node_subset_cap:
        subsets = [[nodes[i] for i in range(N) if mask >> i & 1]
                   for mask in range(1, 2 ** N - 1)]
    else:
        seen = set()
        while len(subsets) < node_subset_cap:
            k = int(rng.integers(1, N))
            S = tuple(sorted(rng.choice(N, size=k, replace=False).tolist()))
            if S in seen: continue
            seen.add(S); subsets.append([nodes[i] for i in S])
    for S in subsets:
        Sset = set(S)
        cut = g.in_cut(Sset)
        if not cut: continue
        add(cut)
        add(cut | g.src_of(Sset))
        add(cut | g.src_of(Sset) | g.sunk_in(Sset))

    masks = list(seeds.keys())
    n_seeds = len(masks)

    # Pairwise closure. Seeds are retained in full and never dropped.
    #
    # randomize_closure=True shuffles the order in which closure products
    # are admitted, so two runs with the same cap but different seeds hold
    # DIFFERENT sets while holding the same NUMBER of sets. That separates
    # "sensitive to how many" from "sensitive to which" -- without it the
    # seed is inert on any graph with 2^|V| <= node_subset_cap, because
    # every node subset is enumerated deterministically.
    have = set(masks)
    frontier = list(masks)
    while len(masks) < cap and frontier:
        pairs = [(frontier[i], frontier[j])
                 for i in range(len(frontier)) for j in range(i + 1, len(frontier))]
        if randomize_closure:
            rng.shuffle(pairs)
        nxt = []
        for a, b in pairs:
            if len(masks) >= cap: break
            for m in (a | b, a & b):
                if m and m not in have:
                    have.add(m); masks.append(m); nxt.append(m)
        frontier = nxt
        if not randomize_closure and not nxt:
            break
    return masks, n_seeds


def solve(g, masks, verbose=False, connected_subsets=True):
    """LP over h(S) for S in masks, plus r. Returns (bound, n_cons)."""
    idx = {m: k for k, m in enumerate(masks)}
    NV = len(masks) + 1
    R = len(masks)                       # index of r
    rows, cols, vals, rhs, sense = [], [], [], [], []

    def con(terms, b, s):
        i = len(rhs)
        for c, v in terms:
            rows.append(i); cols.append(v); vals.append(c)
        rhs.append(b); sense.append(s)

    # h >= 0 handled by bounds. monotonicity + submodularity + subadditivity
    mlist = masks
    for a_i in range(len(mlist)):
        A = mlist[a_i]
        for b_i in range(a_i + 1, len(mlist)):
            B = mlist[b_i]
            if A & B == A and A != B:                       # A subset B
                con([(1.0, idx[A]), (-1.0, idx[B])], 0.0, 'L')
            elif A & B == B and A != B:
                con([(1.0, idx[B]), (-1.0, idx[A])], 0.0, 'L')
            U, I = A | B, A & B
            if U in idx:
                if I in idx:                                 # submodularity
                    con([(1.0, idx[U]), (1.0, idx[I]),
                         (-1.0, idx[A]), (-1.0, idx[B])], 0.0, 'L')
                else:                                        # subadditivity
                    con([(1.0, idx[U]), (-1.0, idx[A]), (-1.0, idx[B])], 0.0, 'L')

    # set-to-singleton subadditivity  h(S) <= sum_{x in S} h(x)
    for S in mlist:
        bits = g.bits(S)
        if len(bits) < 2: continue
        terms = [(1.0, idx[S])]
        ok = True
        for x in bits:
            sx = 1 << x
            if sx not in idx: ok = False; break
            terms.append((-1.0, idx[sx]))
        if ok: con(terms, 0.0, 'L')

    # capacity  h(U_uv) + h(U_vu) <= 1
    for (u, v) in g.edges:
        a, b = 1 << g.arc[(u, v)], 1 << g.arc[(v, u)]
        if a in idx and b in idx:
            con([(1.0, idx[a]), (1.0, idx[b])], 1.0, 'L')

    # source independence  h(Y_A) = |A| r
    for S in mlist:
        bits = g.bits(S)
        if bits and all(g.elem[x][0] == 'Y' for x in bits):
            con([(1.0, idx[S]), (-float(len(bits)), R)], 0.0, 'E')

    # encoding  h(S u Out(v)) = h(S)  for S superset of In(v) u Src(v)
    for v in g.nodes:
        need = g.In[v] | g.Src[v]
        for S in mlist:
            if S & need != need: continue
            T = S | g.Out[v]
            if T in idx and T != S:
                con([(1.0, idx[T]), (-1.0, idx[S])], 0.0, 'E')

    # decoding at sinks  h(S u {Y_i}) = h(S)  for S superset of In(t_i) u Src(t_i)
    for i, (s, t) in enumerate(g.sessions):
        need = g.In[t] | g.Src[t]
        yi = 1 << g.yi[i]
        for S in mlist:
            if S & need != need: continue
            T = S | yi
            if T in idx and T != S:
                con([(1.0, idx[T]), (-1.0, idx[S])], 0.0, 'E')

    # set-level decoding  h(S u D) = h(S) for S superset of cut(T) u Src(T)
    #
    # NODE-SUBSET RESTRICTION (2026-08-04). Enumerating all 2^|V| subsets is
    # what stalled n>=12: grid_4x4_16N has 44800 subsets with a non-empty
    # decodable set, each swept against |C| sets.
    #
    # Restricting to subsets that induce a CONNECTED subgraph on both sides
    # cuts that to 755 (1.2% of 2^16).
    #
    # SOUND but NOT LOSSLESS. Sound because dropping constraints only
    # enlarges the feasible region, so the optimum remains a valid upper
    # bound. Not lossless: a disconnected T can have a decodable set D
    # STRICTLY LARGER than the union of its components' decodable sets --
    # a session with source in one component and sink in another is sunk
    # inside T but inside neither component -- so its constraint is not
    # implied by the component constraints.
    #
    # Measured cost at cap=700: hu_three_session_6N 1.1429 -> 1.1429
    # (exact), yin_et_al_7N 1.2727 -> 1.2727 (exact),
    # okamura_network_paper_5N 0.7805 -> 0.7818 (+0.0013, 0.17% looser).
    for T in _decoding_subsets(g, connected_only=connected_subsets):
        need = g.in_cut(T) | g.src_of(T)
        D = g.sunk_in(T)
        if not D or not need: continue
        for S in mlist:
            if S & need != need: continue
            U = S | D
            if U in idx and U != S:
                con([(1.0, idx[U]), (-1.0, idx[S])], 0.0, 'E')

    A = coo_matrix((vals, (rows, cols)), shape=(len(rhs), NV))
    rhs = np.array(rhs)
    Lm = np.array([s == 'L' for s in sense])
    c = np.zeros(NV); c[R] = -1.0
    bounds = [(0, None)] * len(masks) + [(0, None)]
    res = linprog(c, A_ub=A.tocsr()[Lm], b_ub=rhs[Lm],
                  A_eq=A.tocsr()[~Lm] if (~Lm).any() else None,
                  b_eq=rhs[~Lm] if (~Lm).any() else None,
                  bounds=bounds, method='highs')
    if not res.success:
        return None, len(rhs)
    return -res.fun, len(rhs)


def _decoding_subsets(g, connected_only=True):
    """Node subsets used for set-level decoding constraints."""
    import networkx as nx
    N = len(g.nodes)
    GX = nx.Graph(); GX.add_nodes_from(g.nodes); GX.add_edges_from(g.edges)
    alln = set(g.nodes)
    out = []
    for mask in range(1, 2 ** N - 1):
        T = {g.nodes[i] for i in range(N) if mask >> i & 1}
        if connected_only:
            if not nx.is_connected(GX.subgraph(T)): continue
            if not nx.is_connected(GX.subgraph(alln - T)): continue
        out.append(T)
    return out


def joint_lp(nodes, edges, sessions, cap=700, seed=0, randomize_closure=False,
             connected_subsets=True):
    g = Ground(nodes, edges, sessions)
    masks, n_seeds = build_collection(g, cap=cap, seed=seed,
                                      randomize_closure=randomize_closure)
    b, ncon = solve(g, masks, connected_subsets=connected_subsets)
    return dict(bound=b, n_sets=len(masks), n_seeds=n_seeds,
                n_cons=ncon, ground=g.n)
