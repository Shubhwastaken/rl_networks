"""
Graph generation for training and evaluation.

Each graph includes:
  - nodes, edges, sessions (the graph itself)
  - name: human-readable identifier
  - optimal_bound: true optimal partition bound (brute-force verified)
  - optimal_internal: max internal sessions achievable
  - optimal_partition: one optimal partition (for reference)

All graphs maintain the invariant:
  - Every session (s,t) has NO direct edge between s and t
"""

import random
from typing import List, Tuple, Optional, Dict, Any
from itertools import product


# -----------------------------------------------------------------------
# Graph info container
# -----------------------------------------------------------------------

class GraphInfo:
    """Container for a graph with its metadata and optimal solution."""
    def __init__(self, name, nodes, edges, sessions,
                 optimal_bound=None, optimal_internal=None,
                 optimal_partition=None):
        self.name              = name
        self.nodes             = nodes
        self.edges             = edges
        self.sessions          = sessions
        self.optimal_bound     = optimal_bound
        self.optimal_internal  = optimal_internal
        self.optimal_partition = optimal_partition

    def as_tuple(self):
        return (self.nodes, self.edges, self.sessions)

    def __repr__(self):
        return (f"GraphInfo({self.name}: {len(self.nodes)}N "
                f"{len(self.edges)}E {len(self.sessions)}S "
                f"opt={self.optimal_bound:.4f})")


# -----------------------------------------------------------------------
# Brute-force optimal bound solver
# -----------------------------------------------------------------------

def compute_optimal_bound(nodes, edges, sessions, max_colors=None):
    """
    Enumerates all valid partitions (independent-set colorings)
    and returns the one that maximizes internal sessions,
    giving the tightest partition bound.

    For graphs up to ~9 nodes with 4-5 colors this runs in seconds.

    Returns: (optimal_bound, optimal_internal, optimal_partition)
    """
    n = len(nodes)
    if max_colors is None:
        max_colors = min(n, 5)

    adj = {nd: set() for nd in nodes}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    best_bound     = float('inf')
    best_partition  = None
    best_internal   = 0

    for coloring in product(range(max_colors), repeat=n):
        # Build partition from coloring
        groups = {}
        for i, c in enumerate(coloring):
            groups.setdefault(c, []).append(nodes[i])
        partition = list(groups.values())

        # Check independent set constraint
        valid = True
        for group in partition:
            gset = set(group)
            for nd in group:
                if adj[nd] & (gset - {nd}):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue

        # Count internal sessions
        total_int = 0
        for group in partition:
            gset = set(group)
            for s, t in sessions:
                if s in gset and t in gset:
                    total_int += 1

        denom = len(sessions) + total_int
        bound = len(edges) / denom if denom > 0 else float('inf')

        if bound < best_bound:
            best_bound     = bound
            best_partition = [list(g) for g in partition]
            best_internal  = total_int

    return best_bound, best_internal, best_partition


# -----------------------------------------------------------------------
# Base graph definitions (5 graphs)
# -----------------------------------------------------------------------

GRAPH_REGISTRY = []


def _register(name, nodes, edges, sessions):
    """Create GraphInfo with brute-force optimal, add to registry."""
    opt_bound, opt_int, opt_part = compute_optimal_bound(
        nodes, edges, sessions
    )
    info = GraphInfo(
        name=name, nodes=nodes, edges=edges, sessions=sessions,
        optimal_bound=opt_bound, optimal_internal=opt_int,
        optimal_partition=opt_part
    )
    GRAPH_REGISTRY.append(info)
    return info


def _build_registry():
    """Build all 5 base graphs with verified optimal bounds."""
    if GRAPH_REGISTRY:
        return  # already built

    # Graph 0: Original 7-node from the paper
    _register(
        name="paper_7N",
        nodes=['S1', 'S2', 'S3', 'v1', 't1', 't2', 't3'],
        edges=[
            ('S1', 'S2'), ('S1', 'v1'), ('S2', 'v1'), ('S3', 'v1'),
            ('S1', 't3'), ('S2', 't1'), ('S3', 't2'),
            ('t1', 'v1'), ('t2', 'v1'), ('t3', 'v1'),
        ],
        sessions=[('S1', 't1'), ('S2', 't2'), ('S3', 't3')],
    )

    # Graph 1: Diamond 6-node
    _register(
        name="diamond_6N",
        nodes=['S1', 'S2', 'v1', 'v2', 't1', 't2'],
        edges=[
            ('S1', 'v1'), ('S1', 'v2'), ('S2', 'v1'), ('S2', 'v2'),
            ('v1', 't1'), ('v2', 't2'), ('v1', 'v2'),
        ],
        sessions=[('S1', 't1'), ('S2', 't2')],
    )

    # Graph 2: Butterfly 8-node
    _register(
        name="butterfly_8N",
        nodes=['S1', 'S2', 'S3', 'S4', 't1', 't2', 't3', 't4'],
        edges=[
            ('S1', 'S2'), ('S2', 'S3'), ('S3', 'S4'), ('S1', 'S4'),
            ('S1', 't2'), ('S2', 't3'), ('S3', 't4'), ('S4', 't1'),
            ('t1', 't2'), ('t2', 't3'), ('t3', 't4'), ('t1', 't4'),
        ],
        sessions=[('S1', 't1'), ('S2', 't2'), ('S3', 't3'), ('S4', 't4')],
    )

    # Graph 3: Grid 9-node
    _register(
        name="grid_9N",
        nodes=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'],
        edges=[
            ('a', 'b'), ('b', 'c'),
            ('d', 'e'), ('e', 'f'),
            ('g', 'h'), ('h', 'i'),
            ('a', 'd'), ('b', 'e'), ('c', 'f'),
            ('d', 'g'), ('e', 'h'), ('f', 'i'),
        ],
        sessions=[('a', 'i'), ('c', 'g'), ('b', 'h')],
    )

    # Graph 4: Star extended 8-node
    _register(
        name="star_8N",
        nodes=['S1', 'S2', 'v1', 'r1', 'r2', 't1', 't2', 'r3'],
        edges=[
            ('S1', 'v1'), ('S2', 'v1'),
            ('v1', 't1'), ('v1', 't2'),
            ('S1', 'r1'), ('r1', 't2'),
            ('S2', 'r2'), ('r2', 't1'),
            ('r1', 'r3'),
        ],
        sessions=[('S1', 't1'), ('S2', 't2')],
    )


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def get_graph_info(graph_id: int) -> GraphInfo:
    """Returns GraphInfo for the given graph_id."""
    _build_registry()
    return GRAPH_REGISTRY[graph_id % len(GRAPH_REGISTRY)]


def generate_large_network(graph_id: int = 0):
    """Returns (nodes, edges, sessions) for backward compatibility."""
    return get_graph_info(graph_id).as_tuple()


def generate_graph_dataset(size: int = 5) -> list:
    """
    Returns a list of (nodes, edges, sessions) tuples.
    First 5 are the base graphs; extras are perturbations.
    """
    _build_registry()
    num_base = len(GRAPH_REGISTRY)
    dataset = []

    for i in range(min(size, num_base)):
        dataset.append(GRAPH_REGISTRY[i].as_tuple())

    for i in range(num_base, size):
        base = GRAPH_REGISTRY[i % num_base]
        nodes, edges, sessions = base.nodes, list(base.edges), base.sessions

        session_edges = set()
        for s, t in sessions:
            session_edges.add((s, t))
            session_edges.add((t, s))
        edge_set = set()
        for u, v in edges:
            edge_set.add((u, v))
            edge_set.add((v, u))

        new_edges = list(edges)
        attempts, added = 0, 0
        while added < 2 and attempts < 20:
            u = random.choice(nodes)
            v = random.choice(nodes)
            if (u != v and (u, v) not in edge_set
                    and (u, v) not in session_edges):
                new_edges.append((u, v))
                edge_set.add((u, v))
                edge_set.add((v, u))
                added += 1
            attempts += 1

        if len(new_edges) > len(sessions) + 2 and random.random() < 0.5:
            removable = [e for e in new_edges
                         if e not in [(s, t) for s, t in sessions]]
            if removable:
                new_edges.remove(random.choice(removable))

        valid, _ = verify_graph(nodes, new_edges, sessions)
        dataset.append((nodes, new_edges, sessions) if valid
                       else (nodes, edges, sessions))

    return dataset


def get_all_graph_infos() -> List[GraphInfo]:
    """Returns list of all 5 base GraphInfo objects."""
    _build_registry()
    return list(GRAPH_REGISTRY)


def get_optimal_for_graph(nodes, edges, sessions) -> Tuple[float, int]:
    """
    Looks up the optimal bound for a known graph.
    Falls back to brute-force if not in registry.
    """
    _build_registry()
    # Try to match by node count and edge count
    for info in GRAPH_REGISTRY:
        if (set(info.nodes) == set(nodes)
                and len(info.edges) == len(edges)
                and set(info.sessions) == set(sessions)):
            return info.optimal_bound, info.optimal_internal
    # Fallback: compute
    bound, internal, _ = compute_optimal_bound(nodes, edges, sessions)
    return bound, internal


def identify_graph(nodes, edges, sessions) -> str:
    """Returns the name of a known graph, or a description."""
    _build_registry()
    for info in GRAPH_REGISTRY:
        if (set(info.nodes) == set(nodes)
                and len(info.edges) == len(edges)
                and set(info.sessions) == set(sessions)):
            return info.name
    return f"custom_{len(nodes)}N_{len(edges)}E"


def verify_graph(nodes, edges, sessions):
    """Validates that no session (s,t) has a direct edge."""
    edge_set = set()
    for u, v in edges:
        edge_set.add((u, v))
        edge_set.add((v, u))
    for s, t in sessions:
        if (s, t) in edge_set:
            return False, f"Session ({s},{t}) has direct edge"
    return True, "OK"


if __name__ == "__main__":
    print("Computing optimal bounds for all 5 base graphs...\n")
    for info in get_all_graph_infos():
        print(f"{info.name}:")
        print(f"  Nodes   : {info.nodes}")
        print(f"  Edges   : {len(info.edges)}")
        print(f"  Sessions: {info.sessions}")
        print(f"  Optimal bound    : {info.optimal_bound:.4f}")
        print(f"  Optimal internal : {info.optimal_internal}")
        print(f"  Optimal partition: {info.optimal_partition}")
        print()