"""
Graph generation extended to Tier 1 (6-9N), Tier 2 (10-13N), Tier 3 (14-16N).

WHY LARGER GRAPHS:
  6-9 node graphs have partition bounds that are almost always tight —
  the LP relaxation of the partition bound equals the integer bound.
  Novel inequalities need graphs where the LP bound is STRICTLY tighter,
  which requires at least 10 nodes with 4+ sessions creating crossing
  constraints that no single integer partition can handle optimally.
"""

import random
from typing import List, Tuple, Optional, Dict, Any
from itertools import product


class GraphInfo:
    def __init__(self, name, nodes, edges, sessions,
                 optimal_bound=None, optimal_internal=None,
                 optimal_partition=None):
        self.name             = name
        self.nodes            = nodes
        self.edges            = edges
        self.sessions         = sessions
        self.optimal_bound    = optimal_bound
        self.optimal_internal = optimal_internal
        self.optimal_partition= optimal_partition

    def as_tuple(self):
        return (self.nodes, self.edges, self.sessions)

    def __repr__(self):
        return (f"GraphInfo({self.name}: {len(self.nodes)}N "
                f"{len(self.edges)}E {len(self.sessions)}S "
                f"opt={self.optimal_bound:.4f})")


def compute_optimal_bound(nodes, edges, sessions, max_colors=None):
    n = len(nodes)
    if n > 10:
        return _greedy_partition_bound(nodes, edges, sessions)
    if max_colors is None:
        max_colors = min(n, 5)

    adj = {nd: set() for nd in nodes}
    for u, v in edges:
        adj[u].add(v); adj[v].add(u)

    best_bound    = float("inf")
    best_partition = None
    best_internal  = 0

    for coloring in product(range(max_colors), repeat=n):
        groups = {}
        for i, c in enumerate(coloring):
            groups.setdefault(c, []).append(nodes[i])
        partition = list(groups.values())
        valid = True
        for group in partition:
            gset = set(group)
            for nd in group:
                if adj[nd] & (gset - {nd}):
                    valid = False; break
            if not valid: break
        if not valid: continue

        total_int = sum(1 for group in partition
                        for s, t in sessions
                        if s in set(group) and t in set(group))
        denom = len(sessions) + total_int
        bound = len(edges) / denom if denom > 0 else float("inf")
        if bound < best_bound:
            best_bound = bound
            best_partition = [list(g) for g in partition]
            best_internal = total_int

    return best_bound, best_internal, best_partition


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


def _register(name, nodes, edges, sessions):
    opt_bound, opt_int, opt_part = compute_optimal_bound(nodes, edges, sessions)
    info = GraphInfo(name=name, nodes=nodes, edges=edges, sessions=sessions,
                     optimal_bound=opt_bound, optimal_internal=opt_int,
                     optimal_partition=opt_part)
    GRAPH_REGISTRY.append(info)
    return info


def _build_registry():
    if GRAPH_REGISTRY: return

    # Tier 1: Original 5 graphs (6-9 nodes)
    _register("paper_7N",
        ["S1","S2","S3","v1","t1","t2","t3"],
        [("S1","S2"),("S1","v1"),("S2","v1"),("S3","v1"),
         ("S1","t3"),("S2","t1"),("S3","t2"),
         ("t1","v1"),("t2","v1"),("t3","v1")],
        [("S1","t1"),("S2","t2"),("S3","t3")])

    _register("diamond_6N",
        ["S1","S2","v1","v2","t1","t2"],
        [("S1","v1"),("S1","v2"),("S2","v1"),("S2","v2"),
         ("v1","t1"),("v2","t2"),("v1","v2")],
        [("S1","t1"),("S2","t2")])

    _register("butterfly_8N",
        ["S1","S2","S3","S4","t1","t2","t3","t4"],
        [("S1","S2"),("S2","S3"),("S3","S4"),("S1","S4"),
         ("S1","t2"),("S2","t3"),("S3","t4"),("S4","t1"),
         ("t1","t2"),("t2","t3"),("t3","t4"),("t1","t4")],
        [("S1","t1"),("S2","t2"),("S3","t3"),("S4","t4")])

    _register("grid_9N",
        ["a","b","c","d","e","f","g","h","i"],
        [("a","b"),("b","c"),("d","e"),("e","f"),("g","h"),("h","i"),
         ("a","d"),("b","e"),("c","f"),("d","g"),("e","h"),("f","i")],
        [("a","i"),("c","g"),("b","h")])

    _register("star_8N",
        ["S1","S2","v1","r1","r2","t1","t2","r3"],
        [("S1","v1"),("S2","v1"),("v1","t1"),("v1","t2"),
         ("S1","r1"),("r1","t2"),("S2","r2"),("r2","t1"),("r1","r3")],
        [("S1","t1"),("S2","t2")])

    # Tier 2: Medium graphs (6-13 nodes) with known LP gaps
    _register("hu_3pairs_6N",
        ["a","b","c","d","e","f"],
        [("a","c"),("a","d"),("a","e"),("b","d"),("b","e"),("b","f"),("c","f"),("d","e")],
        [("a","b"),("c","d"),("e","f")])

    _register("okamura_4N",
        ["a","b","c","d"],
        [("a","b"),("b","c"),("c","d"),("a","d"),("a","c")],
        [("a","c"),("b","d"),("a","b")])

    # 3x4 grid (12 nodes)
    g12V = [f"r{r}c{c}" for r in range(3) for c in range(4)]
    g12E = []
    for r in range(3):
        for c in range(4):
            if c+1<4: g12E.append((f"r{r}c{c}",f"r{r}c{c+1}"))
            if r+1<3: g12E.append((f"r{r}c{c}",f"r{r+1}c{c}"))
    _register("grid_3x4_12N", g12V, g12E,
              [("r0c0","r2c3"),("r0c3","r2c0"),("r1c0","r1c3")])

    # Petersen graph (10 nodes, 15 edges)
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
    _register("petersen_10N", pet_V, pet_E, pet_nonadj[:4])

    # Two K4 cliques bridged (10 nodes, 4 sessions across the bridge)
    t4V = [f"a{i}" for i in range(4)] + ["m"] + [f"b{i}" for i in range(4)] + ["n"]
    t4E = ([(f"a{i}",f"a{j}") for i in range(4) for j in range(i+1,4)] +
           [(f"b{i}",f"b{j}") for i in range(4) for j in range(i+1,4)] +
           [("a0","m"),("m","b0"),("a1","n"),("n","b1")])
    t4_eset = {(u,v) for u,v in t4E} | {(v,u) for u,v in t4E}
    t4S = [(s,t) for s,t in [("a2","b2"),("a3","b3"),("a0","b1")]
           if (s,t) not in t4_eset]
    if len(t4S) >= 2:
        _register("two_k4_10N", t4V, t4E, t4S)

    # Tier 3: Large graph (16 nodes) for Stage 4
    g16V = [f"r{r}c{c}" for r in range(4) for c in range(4)]
    g16E = []
    for r in range(4):
        for c in range(4):
            if c+1<4: g16E.append((f"r{r}c{c}",f"r{r}c{c+1}"))
            if r+1<4: g16E.append((f"r{r}c{c}",f"r{r+1}c{c}"))
    _register("grid_4x4_16N", g16V, g16E,
              [("r0c0","r3c3"),("r0c3","r3c0"),("r1c0","r2c3"),("r0c1","r3c2")])


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
    e_set = {(u,v) for u,v in edges} | {(v,u) for u,v in edges}
    for s,t in sessions:
        if (s,t) in e_set: return False, f"Session ({s},{t}) has direct edge"
    return True, "OK"

if __name__ == "__main__":
    print("Graph registry:")
    for info in get_all_graph_infos():
        trivial = len(info.edges)/len(info.sessions)
        print(f"  {info.name:<22} |V|={len(info.nodes):>2} |E|={len(info.edges):>2} "
              f"|S|={len(info.sessions)} PB={info.optimal_bound:.4f} trivial={trivial:.4f}")