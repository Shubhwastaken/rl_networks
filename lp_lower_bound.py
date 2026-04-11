"""
LP-based lower bound on symmetric information rate for undirected
multiple-unicast networks.

Replicates lp_undirected.cpp using scipy.optimize.linprog.

The LP formulation:
  - Decision variables: x[u,v,k] = flow of commodity k on directed arc (u->v)
                         y        = common symmetric rate (to maximize)
  - Objective: max y
  - Constraints:
    1) Source constraint:  for each commodity k with source s_k,
       sum_{v in adj(s_k)} [ x[s_k, v, k] - x[v, s_k, k] ] >= y
    2) Non-negativity:     x[u, v, k] >= 0   for all arcs, commodities
    3) Edge capacity:      for each undirected edge {u,v},
       sum_k [ x[u,v,k] + x[v,u,k] ] <= 1
    4) Flow conservation:  for each node i that is NOT source or sink of
       commodity k:
       sum_{v in adj(i)} x[i,v,k] = sum_{v in adj(i)} x[v,i,k]

Since scipy.linprog only does minimization, we minimize -y.

Usage:
    from lp_lower_bound import compute_lp_lower_bound
    lb = compute_lp_lower_bound(nodes, edges, sessions)
"""

import numpy as np
from scipy.optimize import linprog
from typing import List, Tuple, Dict, Optional
from functools import lru_cache


def compute_lp_lower_bound(
    nodes: List,
    edges: List[Tuple],
    sessions: List[Tuple],
    verbose: bool = False,
) -> float:
    """
    Compute the maximum symmetric multi-commodity flow (LP lower bound)
    for an undirected graph with the given source-sink sessions.

    Parameters
    ----------
    nodes : list
        Node identifiers (any hashable).
    edges : list of (u, v) tuples
        Undirected edges.
    sessions : list of (source, sink) tuples
        Source-sink pairs (commodities).
    verbose : bool
        If True, print LP details.

    Returns
    -------
    float
        The LP lower bound (max symmetric flow value y*).
        Returns 0.0 if the LP is infeasible.
    """
    node_list = list(nodes)
    node_idx = {n: i for i, n in enumerate(node_list)}
    n_nodes = len(node_list)
    n_edges = len(edges)
    n_comm = len(sessions)

    # Build adjacency list (undirected -> both directed arcs)
    adj: Dict[int, List[int]] = {i: [] for i in range(n_nodes)}
    for u, v in edges:
        ui, vi = node_idx[u], node_idx[v]
        adj[ui].append(vi)
        adj[vi].append(ui)

    # Build list of all directed arcs and index them
    # Each undirected edge (u,v) gives two arcs: (ui,vi) and (vi,ui)
    arcs = []
    arc_idx: Dict[Tuple[int, int], int] = {}
    for u, v in edges:
        ui, vi = node_idx[u], node_idx[v]
        arc_idx[(ui, vi)] = len(arcs)
        arcs.append((ui, vi))
        arc_idx[(vi, ui)] = len(arcs)
        arcs.append((vi, ui))
    n_arcs = len(arcs)

    # Decision variables layout:
    #   var 0     : y  (the symmetric rate)
    #   var 1..   : x[arc, commodity] for each (arc, commodity) pair
    #                indexed as 1 + arc_id * n_comm + k
    n_vars = 1 + n_arcs * n_comm

    def x_var(arc_id: int, k: int) -> int:
        """Index of flow variable x[arc_id, k] in the decision vector."""
        return 1 + arc_id * n_comm + k

    # Objective: minimize -y  (i.e., maximize y)
    c = np.zeros(n_vars)
    c[0] = -1.0  # coefficient for y

    # Build constraints
    # We'll collect inequality (A_ub @ x <= b_ub) and
    #                equality  (A_eq @ x == b_eq)

    # --- 1) Source constraint (inequality): net outflow >= y ---
    # Rewritten as: -net_outflow + y <= 0
    # i.e.:  y - sum_{v in adj(s)} [x[s->v, k] - x[v->s, k]] <= 0
    A_ub_rows = []
    b_ub_rows = []

    for k, (src, _sink) in enumerate(sessions):
        si = node_idx[src]
        row = np.zeros(n_vars)
        row[0] = 1.0  # + y
        for vi in adj[si]:
            fwd = arc_idx.get((si, vi))
            rev = arc_idx.get((vi, si))
            if fwd is not None:
                row[x_var(fwd, k)] -= 1.0   # - x[s->v, k]
            if rev is not None:
                row[x_var(rev, k)] += 1.0   # + x[v->s, k]
        A_ub_rows.append(row)
        b_ub_rows.append(0.0)

    # --- 2) Edge capacity (inequality): for each undirected edge {u,v},
    #     sum_k [ x[u->v, k] + x[v->u, k] ] <= 1
    for u, v in edges:
        ui, vi = node_idx[u], node_idx[v]
        fwd = arc_idx[(ui, vi)]
        rev = arc_idx[(vi, ui)]
        row = np.zeros(n_vars)
        for k in range(n_comm):
            row[x_var(fwd, k)] = 1.0
            row[x_var(rev, k)] = 1.0
        A_ub_rows.append(row)
        b_ub_rows.append(1.0)

    # --- 3) Flow conservation (equality): for node i, commodity k,
    #     if i is neither source nor sink of k:
    #     sum_{v in adj(i)} x[i->v, k] == sum_{v in adj(i)} x[v->i, k]
    A_eq_rows = []
    b_eq_rows = []

    source_of = {}
    sink_of = {}
    for k, (src, snk) in enumerate(sessions):
        source_of[k] = node_idx[src]
        sink_of[k] = node_idx[snk]

    for i in range(n_nodes):
        for k in range(n_comm):
            if i == source_of[k] or i == sink_of[k]:
                continue
            row = np.zeros(n_vars)
            for vi in adj[i]:
                fwd = arc_idx.get((i, vi))
                rev = arc_idx.get((vi, i))
                if fwd is not None:
                    row[x_var(fwd, k)] += 1.0   # outflow
                if rev is not None:
                    row[x_var(rev, k)] -= 1.0   # - inflow
            A_eq_rows.append(row)
            b_eq_rows.append(0.0)

    # Assemble matrices
    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_rows) if b_ub_rows else None
    A_eq = np.array(A_eq_rows) if A_eq_rows else None
    b_eq = np.array(b_eq_rows) if b_eq_rows else None

    # Bounds: y >= 0, x[...] >= 0
    bounds = [(0, None)] * n_vars  # y >= 0, all flows >= 0

    # Solve
    result = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=bounds, method='highs'
    )

    if result.success:
        lp_bound = -result.fun  # we minimized -y, so optimal y = -result.fun
        if verbose:
            print(f"  LP lower bound: {lp_bound:.6f}")
            print(f"  Status: {result.message}")
        return max(lp_bound, 0.0)
    else:
        if verbose:
            print(f"  LP failed: {result.message}")
        return 0.0


def compute_all_lp_bounds(graph_dataset: list, verbose: bool = True) -> Dict[str, float]:
    """
    Compute LP lower bounds for all graphs in a dataset.

    Parameters
    ----------
    graph_dataset : list of (nodes, edges, sessions)
    verbose : bool

    Returns
    -------
    dict mapping graph description -> LP lower bound
    """
    from fixed_graph_generation import identify_graph

    bounds = {}
    if verbose:
        print(f"\n{'='*60}")
        print("LP LOWER BOUNDS (max symmetric multi-commodity flow)")
        print(f"{'='*60}")
        print(f"  {'Graph':<20} {'LP LB':>8} {'|V|':>4} {'|E|':>4} {'|S|':>4}")
        print(f"  {'-'*44}")

    for nodes, edges, sessions in graph_dataset:
        gname = identify_graph(nodes, edges, sessions)
        lb = compute_lp_lower_bound(nodes, edges, sessions)
        bounds[gname] = lb
        if verbose:
            print(f"  {gname:<20} {lb:>8.4f} {len(nodes):>4} "
                  f"{len(edges):>4} {len(sessions):>4}")

    if verbose:
        print()
    return bounds


def validate_bound_against_lp(
    bound: float,
    lp_lower_bound: float,
    graph_name: str = "",
    tolerance: float = 1e-6,
) -> Tuple[bool, str]:
    """
    Check whether an upper bound is valid (>= LP lower bound).

    Parameters
    ----------
    bound : float
        The upper bound produced by the RL agent (or partition bound).
    lp_lower_bound : float
        The LP lower bound for the same graph.
    graph_name : str
        For logging.
    tolerance : float
        Numerical tolerance.

    Returns
    -------
    (is_valid, message)
    """
    if bound < lp_lower_bound - tolerance:
        msg = (f"INVALID: {graph_name} upper bound {bound:.6f} < "
               f"LP lower bound {lp_lower_bound:.6f} "
               f"(violation = {lp_lower_bound - bound:.6f})")
        return False, msg
    else:
        margin = bound - lp_lower_bound
        msg = (f"VALID: {graph_name} upper bound {bound:.6f} >= "
               f"LP lower bound {lp_lower_bound:.6f} "
               f"(margin = {margin:.6f})")
        return True, msg


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from fixed_graph_generation import get_all_graph_infos, generate_graph_dataset
    from fixed_environment import _compute_partition_bound

    infos = get_all_graph_infos()
    print(f"\n{'='*70}")
    print("LP LOWER BOUND vs PARTITION BOUND (upper) COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Graph':<20} {'LP LB':>8} {'PB (UB)':>8} {'Gap':>8} {'Valid':>6}")
    print(f"  {'-'*54}")

    all_valid = True
    for info in infos:
        nodes, edges, sessions = info.nodes, info.edges, info.sessions
        lb = compute_lp_lower_bound(nodes, edges, sessions)
        pb = _compute_partition_bound(nodes, edges, sessions)
        gap = pb - lb
        valid = pb >= lb - 1e-6
        if not valid:
            all_valid = False
        print(f"  {info.name:<20} {lb:>8.4f} {pb:>8.4f} {gap:>8.4f} "
              f"{'  OK' if valid else ' FAIL':>6}")

    print(f"\n  Overall: {'ALL VALID' if all_valid else 'SOME INVALID'}")
    print(f"  (LP lower bound <= Partition bound must always hold)")
    print()
