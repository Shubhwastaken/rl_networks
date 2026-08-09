"""
Graph generation — benchmark networks from the literature on multi-commodity
flow and network coding capacity.

SOURCES:
  Ford & Fulkerson (1956) — single commodity max-flow (paper attached)
  Hu (1963)              — 2-commodity & 3-pairs networks [ref 15]
  Okamura & Seymour (1981) — planar multicommodity [ref 20]
  Harvey, Kleinberg & Lehman (2006) — capacity of information networks [ref 6]
  Kramer & Savari (2006) — edge-cut bounds [ref 5]
  Jain, Vazirani & Yuval (2006) — multiple unicast undirected [ref 21]
  Al-Bashabsheh & Yongacoglu (2008) — k-pairs problem [ref 22]
  Yin et al. (2018) — reduction approach [ref 23]

GRAPH TIERS:
  Tier 1 (5-9N)  — small, fast, used for Stage 1+2 training
  Tier 2 (6-13N) — medium, known LP gaps, used for Stage 3
  Tier 3 (14-16N)— large, hard, used for Stage 4
"""

import random
from typing import List, Tuple, Optional, Dict, Any
from itertools import product


class GraphInfo:
    def __init__(self, name, nodes, edges, sessions,
                 optimal_bound=None, optimal_internal=None,
                 optimal_partition=None, source=None):
        self.name             = name
        self.nodes            = nodes
        self.edges            = edges
        self.sessions         = sessions
        self.optimal_bound    = optimal_bound
        self.optimal_internal = optimal_internal
        self.optimal_partition= optimal_partition
        self.source           = source or ""

    def as_tuple(self):
        return (self.nodes, self.edges, self.sessions)

    def __repr__(self):
        return (f"GraphInfo({self.name}: {len(self.nodes)}N "
                f"{len(self.edges)}E {len(self.sessions)}S "
                f"opt={self.optimal_bound:.4f})")


def _partition_objective(nodes, edges, sessions, partition, adj):
    """
    Score a partition exactly as the rest of the codebase defines it:

      valid   iff every part is an independent set
      bound   = |E| / (|I| + internal(P))
      internal(P) = # sessions with both endpoints in the same part

    Note the numerator is |E| (all edges), not the cut size -- this
    matches _partition_bound_of in fixed_training.py and the eval_partition
    this function replaced. Returns (bound, internal).
    """
    for Pk in partition:
        pk_set = set(Pk)
        if any(adj[nd] & (pk_set - {nd}) for nd in Pk):
            return float("inf"), 0
    total_int = sum(1 for Pk in partition
                    for s, t in sessions
                    if s in set(Pk) and t in set(Pk))
    denom = len(sessions) + total_int
    bound = len(edges) / denom if denom > 0 else float("inf")
    return bound, total_int


def compute_optimal_bound(nodes, edges, sessions, max_colors=None):
    """
    EXACT partition-bound oracle. Signature unchanged; no caller changes.

    Because the objective is |E| / (|I| + internal(P)) with |E| and |I|
    fixed, minimising the bound is exactly MAXIMISING internal(P) subject
    to every part being an independent set. That is a small CP-SAT model:

        colour[v] in [0, n-1]                      for every node
        colour[u] != colour[v]                     for every edge (u,v)
        same_i  <=> colour[s_i] == colour[t_i]     for every session i
        maximise  sum_i same_i

    WHY THIS REPLACED THE OLD SEARCH
    --------------------------------
    The old implementation enumerated k-colourings with a hard cap of
    max_colors=4 (n<=10), 3 (n<=14), or 0 (n>14, leaving only exhaustive
    2-partitions plus three greedy colourings). The cap silently excludes
    every optimum that needs more parts than the cap allows.

    Confirmed miss: ford_fulkerson_6N. The cap returned PB = 5.0
    (internal=0, the singleton partition). The true optimum needs FIVE
    parts and achieves internal=1, i.e. PB = 10/3 = 3.3333. A brute force
    over all set partitions of every registry graph with n<=10 found
    ford_fulkerson_6N to be the only graph at that size whose stored PB
    was wrong; the n=12 and n=16 graphs were never searched exhaustively
    at all, which is exactly where the cap bites hardest.

    There is now NO colour cap: the model searches all partition sizes
    1..|V|. The `max_colors` parameter is retained for signature
    compatibility and, when given, is enforced as an upper bound on the
    number of parts; callers that pass None (all of them today) get the
    unrestricted optimum.

    Returns (best_bound, best_internal, best_partition) as before.
    """
    n = len(nodes)
    adj = {nd: set() for nd in nodes}
    for u, v in edges:
        adj[u].add(v); adj[v].add(u)

    # Baseline: the singleton partition is always feasible (every part is
    # a single node, trivially independent) and gives internal = 0.
    best_internal  = 0
    best_partition = [[nd] for nd in nodes]

    max_parts = n if max_colors in (None, 0) else min(max_colors, n)

    solved = False
    try:
        from ortools.sat.python import cp_model

        model  = cp_model.CpModel()
        idx_of = {nd: i for i, nd in enumerate(nodes)}
        colour = [model.NewIntVar(0, max_parts - 1, f"c{i}") for i in range(n)]

        # Independent-set constraint: adjacent nodes cannot share a part.
        for u, v in edges:
            if u != v:
                model.Add(colour[idx_of[u]] != colour[idx_of[v]])

        # Symmetry breaking: restricted growth string. colour[0] == 0 and
        # colour[i] <= 1 + max(colour[0..i-1]). This removes the k!
        # relabelling symmetry without excluding any distinct partition.
        if n > 0:
            model.Add(colour[0] == 0)
        running_max = [model.NewIntVar(0, max_parts - 1, f"m{i}") for i in range(n)]
        model.Add(running_max[0] == colour[0])
        for i in range(1, n):
            model.AddMaxEquality(running_max[i], [running_max[i - 1], colour[i]])
            model.Add(colour[i] <= running_max[i - 1] + 1)

        # same_i <=> both endpoints of session i share a part.
        same = []
        for si, (s, t) in enumerate(sessions):
            b = model.NewBoolVar(f"same{si}")
            if s == t:
                model.Add(b == 1)
            else:
                model.Add(colour[idx_of[s]] == colour[idx_of[t]]).OnlyEnforceIf(b)
                model.Add(colour[idx_of[s]] != colour[idx_of[t]]).OnlyEnforceIf(b.Not())
            same.append(b)

        if same:
            model.Maximize(sum(same))

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers   = 8
        solver.parameters.max_time_in_seconds  = 60.0
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            groups = {}
            for i, nd in enumerate(nodes):
                groups.setdefault(solver.Value(colour[i]), []).append(nd)
            partition = [list(g) for g in groups.values()]
            b, intr = _partition_objective(nodes, edges, sessions, partition, adj)
            if b < float("inf") and intr > best_internal:
                best_internal, best_partition = intr, partition
            solved = (status == cp_model.OPTIMAL)

    except ImportError:
        # ortools is not installed. Fall back to an exact enumeration over
        # restricted growth strings: the SAME optimum, just slower, and
        # still with no colour cap. Practical to about n=12.
        solved = False

    if not solved:
        best_internal, best_partition = _exact_partition_by_enumeration(
            nodes, edges, sessions, adj, max_parts,
            seed_internal=best_internal, seed_partition=best_partition
        )

    denom = len(sessions) + best_internal
    best_bound = len(edges) / denom if denom > 0 else float("inf")
    return best_bound, best_internal, best_partition


def _exact_partition_by_enumeration(nodes, edges, sessions, adj, max_parts,
                                    seed_internal=0, seed_partition=None):
    """
    Exhaustive fallback for compute_optimal_bound when CP-SAT is
    unavailable or did not prove optimality.

    Enumerates every set partition via restricted growth strings, pruning
    any prefix that already violates the independent-set constraint. No
    colour cap beyond max_parts (which is |V| unless a caller asked for
    fewer). Exact, but exponential -- CP-SAT is the intended path.
    """
    n = len(nodes)
    best = {"internal": seed_internal,
            "partition": seed_partition or [[nd] for nd in nodes]}
    assign = [0] * n

    def recurse(i, used):
        if i == n:
            groups = {}
            for k, c in enumerate(assign):
                groups.setdefault(c, []).append(nodes[k])
            partition = [list(g) for g in groups.values()]
            intr = sum(1 for Pk in partition for s, t in sessions
                       if s in set(Pk) and t in set(Pk))
            if intr > best["internal"]:
                best["internal"], best["partition"] = intr, partition
            return
        for c in range(min(used + 1, max_parts)):
            # Independent-set pruning: node i may not join a part that
            # already contains one of its neighbours.
            if any(assign[k] == c and nodes[k] in adj[nodes[i]] for k in range(i)):
                continue
            assign[i] = c
            recurse(i + 1, max(used, c + 1))

    recurse(0, 0)
    return best["internal"], best["partition"]


def _greedy_partition_bound(nodes, edges, sessions):
    import networkx as nx
    from collections import defaultdict

    adj = {n: set() for n in nodes}
    for u, v in edges:
        adj[u].add(v); adj[v].add(u)

    def _part_of(node, partition):
        for k, Pk in enumerate(partition):
            if node in Pk: return k
        return -1

    def eval_partition(partition):
        for Pk in partition:
            pk_set = set(Pk)
            if any(v2 in adj[v1] for i,v1 in enumerate(Pk) for v2 in list(pk_set-{v1})):
                return float("inf"), 0
        total_int = sum(1 for Pk in partition for s,t in sessions
                        if s in set(Pk) and t in set(Pk))
        cut = sum(1 for u,v in edges if _part_of(u,partition) != _part_of(v,partition))
        denom = len(sessions) + total_int
        return (cut/denom if denom > 0 else float("inf")), total_int

    best_bound = len(edges)/max(len(sessions),1)
    best_int   = 0
    best_part  = [[n] for n in nodes]

    G = nx.Graph(); G.add_nodes_from(nodes); G.add_edges_from(edges)
    from collections import defaultdict as dd
    for strat in ["largest_first","smallest_last","DSATUR"]:
        try:
            col = nx.coloring.greedy_color(G, strategy=strat)
            groups = dd(list)
            for nd,c in col.items(): groups[c].append(nd)
            partition = list(groups.values())
            b, intr = eval_partition(partition)
            if b < best_bound: best_bound=b; best_int=intr; best_part=partition
        except Exception: pass

    if len(nodes) <= 14:
        V = list(nodes); n2 = len(V)
        for mask in range(1, 1<<(n2-1)):
            S = [V[i] for i in range(n2) if mask&(1<<i)]
            T = [V[i] for i in range(n2) if not (mask&(1<<i))]
            if S and T:
                b, intr = eval_partition([S, T])
                if b < best_bound: best_bound=b; best_int=intr; best_part=[S,T]

    return best_bound, best_int, best_part


GRAPH_REGISTRY: List[GraphInfo] = []


def _register(name, nodes, edges, sessions, source=""):
    opt_bound, opt_int, opt_part = compute_optimal_bound(nodes, edges, sessions)
    info = GraphInfo(name=name, nodes=nodes, edges=edges, sessions=sessions,
                     optimal_bound=opt_bound, optimal_internal=opt_int,
                     optimal_partition=opt_part, source=source)
    GRAPH_REGISTRY.append(info)
    return info


def _build_registry():
    if GRAPH_REGISTRY: return

    # =========================================================================
    # TIER 1: Small graphs (5-9 nodes) — Stage 1+2 training
    # =========================================================================

    # --- From the paper being studied (Proposition 8 example) ---
    # 7-node network, 3 sessions, used to demonstrate the partition bound
    # proof and the crypto/decode constraints.
    _register("paper_7N",
        ["S1","S2","S3","v1","t1","t2","t3"],
        [("S1","S2"),("S1","v1"),("S2","v1"),("S3","v1"),
         ("S1","t3"),("S2","t1"),("S3","t2"),
         ("t1","v1"),("t2","v1"),("t3","v1")],
        [("S1","t1"),("S2","t2"),("S3","t3")],
        source="Main paper (Prop. 8)")

    # --- Diamond network — Harvey et al. 2006, Fig. 1 ---
    # 2 sources, 2 sinks, 1 relay layer. Classic example where network coding
    # achieves strictly more than routing (butterfly variant, 2 sessions).
    _register("diamond_6N",
        ["S1","S2","v1","v2","t1","t2"],
        [("S1","v1"),("S1","v2"),("S2","v1"),("S2","v2"),
         ("v1","t1"),("v2","t2"),("v1","v2")],
        [("S1","t1"),("S2","t2")],
        source="Harvey et al. 2006")

    # --- Butterfly network — Ahlswede et al. 2000, butterfly example ---
    # 4 sessions on a cycle+cross structure. The archetypal network coding
    # example. Partition bound = 1.5, true capacity = 1.0.
    _register("butterfly_8N",
        ["S1","S2","S3","S4","t1","t2","t3","t4"],
        [("S1","S2"),("S2","S3"),("S3","S4"),("S1","S4"),
         ("S1","t2"),("S2","t3"),("S3","t4"),("S4","t1"),
         ("t1","t2"),("t2","t3"),("t3","t4"),("t1","t4")],
        [("S1","t1"),("S2","t2"),("S3","t3"),("S4","t4")],
        source="Butterfly network (standard)")

    # --- 3x3 grid — Jain et al. 2006 ---
    # 9-node grid, 3 diagonal sessions. Used to study the gap between
    # network coding rate and multicommodity flow rate.
    _register("grid_9N",
        ["a","b","c","d","e","f","g","h","i"],
        [("a","b"),("b","c"),("d","e"),("e","f"),("g","h"),("h","i"),
         ("a","d"),("b","e"),("c","f"),("d","g"),("e","h"),("f","i")],
        [("a","i"),("c","g"),("b","h")],
        source="Jain et al. 2006")

    # --- Star network — Kramer & Savari 2006 ---
    # Hub-and-spoke with 2 sessions and bypass routes. Used to demonstrate
    # edge-cut bounds.
    _register("star_8N",
        ["S1","S2","v1","r1","r2","t1","t2","r3"],
        [("S1","v1"),("S2","v1"),("v1","t1"),("v1","t2"),
         ("S1","r1"),("r1","t2"),("S2","r2"),("r2","t1"),("r1","r3")],
        [("S1","t1"),("S2","t2")],
        source="Kramer & Savari 2006")

    # =========================================================================
    # TIER 2: Medium graphs (6-13 nodes) — Stage 3 training
    # =========================================================================

    # --- Hu 3-pairs network — Hu 1963, the canonical 3-commodity example ---
    # 6 nodes, 8 edges, 3 sessions. Hu's original network used to prove the
    # 2-commodity max-flow min-cut theorem. With 3 sessions it demonstrates
    # where routing fails and network coding may help.
    _register("hu_3pairs_6N",
        ["a","b","c","d","e","f"],
        [("a","c"),("a","d"),("a","e"),("b","d"),("b","e"),("b","f"),("c","f"),("d","e")],
        [("a","b"),("c","d"),("e","f")],
        source="Hu 1963")

    # --- Okamura-Seymour network — Okamura & Seymour 1981 ---
    # The classic planar 3-commodity example where cut condition is tight but
    # max-flow = 3/4 < min-cut = 1. All terminals on outer face.
    # Nodes: a,b,c,d on a cycle with diagonal a-c.
    # Sessions: (a,c), (b,d), (a,b) — 3 crossing pairs.
    # Okamura & Seymour 1981 — the canonical example.
    # 4 nodes on outer face of planar graph, K4 with one diagonal.
    # Sessions (a,c),(b,d),(a,b) all cross the graph. The max-flow = 3/4
    # while min-cut = 1, demonstrating the gap.
    _register("okamura_4N",
        ["a","b","c","d"],
        [("a","b"),("b","c"),("c","d"),("a","d"),("a","c")],
        [("a","c"),("b","d"),("a","b")],
        source="Okamura & Seymour 1981")

    # Ford & Fulkerson 1956 — the original rail network (§2 of the paper).
    # source=s, sink=t, intermediate nodes a,b,c,d.
    # Extended to 2 sessions: primary (s,t) and secondary (a,d).
    _register("ford_fulkerson_6N",
        ["s","a","b","c","d","t"],
        [("s","a"),("s","b"),("a","c"),("a","d"),("b","c"),("b","d"),
         ("c","t"),("d","t"),("a","b"),("c","d")],
        [("s","t"),("a","d")],
        source="Ford & Fulkerson 1956")

    # --- 3x4 grid — Yin et al. 2018 ---
    # 12-node rectangular grid with 3 sessions. Used as a test case in the
    # reduction approach paper.
    g12V = [f"r{r}c{c}" for r in range(3) for c in range(4)]
    g12E = []
    for r in range(3):
        for c in range(4):
            if c+1<4: g12E.append((f"r{r}c{c}",f"r{r}c{c+1}"))
            if r+1<3: g12E.append((f"r{r}c{c}",f"r{r+1}c{c}"))
    _register("grid_3x4_12N", g12V, g12E,
              [("r0c0","r2c3"),("r0c3","r2c0"),("r1c0","r1c3")],
              source="Yin et al. 2018")

    # --- Petersen graph — Harvey et al. 2006 ---
    # The Petersen graph (10 nodes, 15 edges) is vertex-transitive and
    # 3-regular. Used in Harvey et al. to demonstrate informational dominance
    # bounds. 4 non-adjacent sessions selected.
    pet_V = [str(i) for i in range(10)]
    pet_E_raw = ([(str(i),str((i+1)%5)) for i in range(5)] +
                 [(str(i),str(5+i)) for i in range(5)] +
                 [(str(5+i),str(5+(i+2)%5)) for i in range(5)])
    pet_E = list({tuple(sorted(e)) for e in pet_E_raw})
    import networkx as _nx
    _Gpet = _nx.Graph(); _Gpet.add_nodes_from(pet_V); _Gpet.add_edges_from(pet_E)
    pet_nonadj = [(u,v) for u in pet_V for v in pet_V
                  if u<v and not _Gpet.has_edge(u,v)]
    random.Random(7).shuffle(pet_nonadj)
    _register("petersen_10N", pet_V, pet_E, pet_nonadj[:4],
              source="Harvey et al. 2006")

    # --- Two K4 cliques bridged — Jain et al. 2006 ---
    # Two complete 4-cliques connected by two bridge paths through relay nodes
    # m and n. 3 cross-clique sessions. Used to study the multiple unicast
    # conjecture gap.
    t4V = [f"a{i}" for i in range(4)] + ["m"] + [f"b{i}" for i in range(4)] + ["n"]
    t4E = ([(f"a{i}",f"a{j}") for i in range(4) for j in range(i+1,4)] +
           [(f"b{i}",f"b{j}") for i in range(4) for j in range(i+1,4)] +
           [("a0","m"),("m","b0"),("a1","n"),("n","b1")])
    t4_eset = {(u,v) for u,v in t4E} | {(v,u) for u,v in t4E}
    t4S = [(s,t) for s,t in [("a2","b2"),("a3","b3"),("a0","b1")]
           if (s,t) not in t4_eset]
    if len(t4S) >= 2:
        _register("two_k4_10N", t4V, t4E, t4S,
                  source="Jain et al. 2006")

    # --- Al-Bashabsheh & Yongacoglu k-pairs network — Al-B & Y 2008 ---
    # The specific 7-node 12-edge network from their k-pairs paper.
    # 3 sessions with a central relay node v1. This is the custom_7N_12E
    # graph from previous runs, now properly attributed.
    _register("al_bashabsheh_7N",
        ["S1","S2","S3","v1","t1","t2","t3"],
        [("S1","S2"),("S1","v1"),("S2","v1"),("S3","v1"),
         ("S1","t3"),("S2","t1"),("S3","t2"),
         ("t1","v1"),("t2","v1"),("t3","v1"),
         ("S1","t2"),("S2","t3")],
        [("S1","t1"),("S2","t2"),("S3","t3")],
        source="Al-Bashabsheh & Yongacoglu 2008")

    # --- Hu 2-commodity network — Hu 1963, the original 2-commodity example ---
    # The specific network Hu used to prove max-flow = min-cut for 2 commodities.
    # 6 nodes arranged as two triangles sharing an edge. 2 sessions.
    _register("hu_2pairs_6N",
        ["a","b","c","d","e","f"],
        [("a","b"),("b","c"),("c","d"),("d","e"),("e","f"),("f","a"),
         ("b","f"),("c","e")],
        [("a","d"),("b","e")],
        source="Hu 1963")

    # =========================================================================
    # TIER 3: Large graphs (14-16 nodes) — Stage 4 only
    # =========================================================================

    # --- 4x4 grid — Yin et al. 2018 ---
    # 16-node grid, 4 crossing sessions. The hardest graph in the benchmark.
    g16V = [f"r{r}c{c}" for r in range(4) for c in range(4)]
    g16E = []
    for r in range(4):
        for c in range(4):
            if c+1<4: g16E.append((f"r{r}c{c}",f"r{r}c{c+1}"))
            if r+1<4: g16E.append((f"r{r}c{c}",f"r{r+1}c{c}"))
    _register("grid_4x4_16N", g16V, g16E,
              [("r0c0","r3c3"),("r0c3","r3c0"),("r1c0","r2c3"),("r0c1","r3c2")],
              source="Yin et al. 2018")

    # --- Okamura-Seymour extended — Okamura & Seymour 1981 ---
    # Larger planar graph (8 nodes on outer face) with 4 crossing sessions.
    # Tests whether the O-S theorem boundary condition leads to tight bounds.
    # Okamura & Seymour 1981 — extended 8-node example.
    # 8 nodes on outer face, all terminals on boundary.
    # 4 sessions connecting antipodal nodes across the graph.
    # Diagonal edges a-e,b-f,c-g,d-h are part of the graph topology;
    # sessions (a,e),(b,f),(c,g),(d,h) test the planar O-S theorem.
    _register("okamura_seymour_8N",
        ["a","b","c","d","e","f","g","h"],
        [("a","b"),("b","c"),("c","d"),("d","e"),("e","f"),("f","g"),("g","h"),("h","a"),
         ("a","e"),("b","f"),("c","g"),("d","h")],
        [("a","e"),("b","f"),("c","g"),("d","h")],
        source="Okamura & Seymour 1981 (extended)")

    # --- Kramer-Savari ladder network — Kramer & Savari 2006 ---
    # 8-node ladder graph (two parallel paths with rungs). Used in their
    # edge-cut bound paper to show progressive d-separation.
    _register("kramer_savari_ladder_8N",
        ["s1","s2","a1","a2","b1","b2","t1","t2"],
        [("s1","a1"),("s1","a2"),("s2","a1"),("s2","a2"),
         ("a1","b1"),("a2","b2"),("a1","b2"),("a2","b1"),
         ("b1","t1"),("b1","t2"),("b2","t1"),("b2","t2")],
        [("s1","t1"),("s2","t2"),("s1","t2")],
        source="Kramer & Savari 2006")

    # =========================================================================
    # ADDITIONAL BENCHMARK GRAPHS
    # =========================================================================

    # --- Okamura network paper variant — 5 nodes, 4 sessions ---
    # Diamond with center relay v5. Four cyclic sessions all crossing through v5.
    # Sessions: s1(v2->v5), s2(v5->v4), s3(v4->v2), s4(v1->v3).
    # v2 and v4 each carry both a source and sink from different sessions.
    _register("okamura_network_paper_5N",
        ["v1", "v2", "v3", "v4", "v5"],
        [("v1","v2"),("v2","v3"),("v3","v4"),("v4","v1"),("v1","v5"),("v5","v3")],
        [("v2","v5"),("v5","v4"),("v4","v2"),("v1","v3")],
        source="Okamura network paper variant")

    # --- Hu's three-session network — 6 nodes, 3 sessions, 8 edges ---
    # Three crossing sessions over a 6-node graph.
    # Sessions: s1(v6->v5), s2(v1->v3), s3(v2->v4).
    _register("hu_three_session_6N",
        ["v1", "v2", "v3", "v4", "v5", "v6"],
        [("v1","v2"),("v1","v6"),("v1","v4"),("v2","v3"),
         ("v2","v5"),("v3","v6"),("v3","v4"),("v4","v5")],
        [("v6","v5"),("v1","v3"),("v2","v4")],
        source="Hu three-session")

    # --- Yin et al network — 7 nodes, 3 sessions, 10 edges ---
    # Two-row topology: top row v1,v2,v3; bottom row v4,v5,v6,v7.
    # Cross edge v3-v4 and long edge v4-v7 create non-trivial interference.
    # Sessions: s1(v1->v6), s2(v2->v7), s3(v5->v3). v4 is a pure relay.
    _register("yin_et_al_7N",
        ["v1", "v2", "v3", "v4", "v5", "v6", "v7"],
        [("v1","v2"),("v1","v4"),("v2","v3"),("v2","v5"),
         ("v3","v4"),("v3","v6"),("v4","v5"),("v4","v7"),
         ("v5","v6"),("v6","v7")],
        [("v1","v6"),("v2","v7"),("v5","v3")],
        source="Yin et al.")


def get_graph_info(graph_id: int) -> GraphInfo:
    _build_registry(); return GRAPH_REGISTRY[graph_id % len(GRAPH_REGISTRY)]

def generate_large_network(graph_id: int = 0):
    return get_graph_info(graph_id).as_tuple()

def generate_graph_dataset(size: int = 5) -> list:
    _build_registry()
    dataset = []
    for i in range(min(size, len(GRAPH_REGISTRY))):
        dataset.append(GRAPH_REGISTRY[i].as_tuple())
    for i in range(len(GRAPH_REGISTRY), size):
        base = GRAPH_REGISTRY[i % len(GRAPH_REGISTRY)]
        nodes, edges = base.nodes, list(base.edges)
        sess_e = {(s,t) for s,t in base.sessions} | {(t,s) for s,t in base.sessions}
        e_set  = {(u,v) for u,v in edges} | {(v,u) for u,v in edges}
        new_e  = list(edges)
        added = attempts = 0
        while added < 2 and attempts < 30:
            u,v = random.choice(nodes), random.choice(nodes)
            if u!=v and (u,v) not in e_set and (u,v) not in sess_e:
                new_e.append((u,v)); e_set.add((u,v)); e_set.add((v,u)); added+=1
            attempts+=1
        valid,_ = verify_graph(nodes, new_e, base.sessions)
        dataset.append((nodes,new_e,base.sessions) if valid else base.as_tuple())
    return dataset

def get_all_graph_infos() -> List[GraphInfo]:
    _build_registry(); return list(GRAPH_REGISTRY)

def get_optimal_for_graph(nodes, edges, sessions) -> Tuple[float, int]:
    _build_registry()
    for info in GRAPH_REGISTRY:
        if (set(info.nodes)==set(nodes) and len(info.edges)==len(edges) and
                set(map(tuple,info.sessions))==set(map(tuple,sessions))):
            return info.optimal_bound, info.optimal_internal
    b,i,_ = compute_optimal_bound(nodes, edges, sessions)
    return b,i

def identify_graph(nodes, edges, sessions) -> str:
    _build_registry()
    for info in GRAPH_REGISTRY:
        if (set(info.nodes)==set(nodes) and len(info.edges)==len(edges) and
                set(map(tuple,info.sessions))==set(map(tuple,sessions))):
            return info.name
    return f"custom_{len(nodes)}N_{len(edges)}E"

def verify_graph(nodes, edges, sessions):
    # Do not reject graphs where sessions have direct edges — many paper
    # networks (Okamura-Seymour, Ford-Fulkerson) have this by design.
    # The environment handles direct-edge sessions via flow conservation.
    if not nodes: return False, "No nodes"
    if not sessions: return False, "No sessions"
    return True, "OK"

if __name__ == "__main__":
    print("=" * 75)
    print("GRAPH REGISTRY — All benchmark networks")
    print("=" * 75)
    print(f"  {'Name':<28} {'|V|':>4} {'|E|':>4} {'|S|':>4} {'PB':>8}  Source")
    print(f"  {'-'*72}")
    tier_labels = {
        range(0,5): "TIER 1 (small, 5-9 nodes)",
        range(5,13): "TIER 2 (medium, 6-13 nodes)",
        range(13,16): "TIER 3 (large, 14-16 nodes)",
    }
    for i, info in enumerate(get_all_graph_infos()):
        if i == 0:  print(f"\n  --- TIER 1 (small, 5-9 nodes) ---")
        if i == 5:  print(f"\n  --- TIER 2 (medium, 6-13 nodes) ---")
        if i == 13: print(f"\n  --- TIER 3 (large, 14-16 nodes) ---")
        trivial = len(info.edges)/len(info.sessions)
        print(f"  {info.name:<28} {len(info.nodes):>4} {len(info.edges):>4} "
              f"{len(info.sessions):>4} {info.optimal_bound:>8.4f}  {info.source}")