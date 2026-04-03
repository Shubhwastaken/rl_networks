"""
Base inequality generator.

ORIGINAL (Phases 1 & 2):
  generate_base_inequalities — one IO per partition set Pi (equation 66).
  This is the partition-bound IO: it sums all nodes in Pi, so internal
  edges cancel and only boundary edges appear on the RHS.

NEW (Phase 3):
  generate_node_io — raw IO for a single node v (the atomic primitive).
    LHS: h(Y_{ST(v)}, U_{nbrs→v}, U_{v→nbrs})
    RHS: h(Y_{S(v)}, U_{nbrs→v})
    The internal edges do NOT cancel here — that only happens when you
    sum over all nodes in a partition. Keeping them separate is what
    lets Phase 3 apply fractional weights before cancellation.

  generate_fractional_io — forms λ·IO(u) + (1-λ)·IO(v) for nodes u, v
    that may be in different partition sets. Returns a FractionalInequality.
    This is the core primitive that can escape the partition bound family.

  generate_all_node_ios — returns {node: FractionalInequality} for all
    nodes, used to populate Phase 3's starting pool.
"""

from typing import List, Tuple, Dict
from fixed_inequality import (
    Inequality, FractionalInequality, EntropyIndex,
    make_fractional
)


# ---------------------------------------------------------------------------
# Original Phase 1/2 generator — unchanged
# ---------------------------------------------------------------------------

def generate_base_inequalities(
    partition  : List[List[str]],
    nodes      : List[str],
    edges      : List[Tuple[str, str]],
    sessions   : List[Tuple[str, str]],
    index      : EntropyIndex
) -> List[Inequality]:
    """One IO per partition set Pi (partition-bound proof, eq. 66)."""
    base_inequalities = []
    for i, Pi in enumerate(partition):
        Pi_set = set(Pi)
        ineq   = Inequality(index)
        ineq.set_lhs(f"Y_ST_P{i}", 1.0)
        ineq.set_lhs(f"Y_I_P{i}",  1.0)
        for v in Pi:
            ineq.set_rhs(f"Y_S_{v}", 1.0)
        for (u, v) in edges:
            if (u in Pi_set) != (v in Pi_set):   # boundary edges only
                ineq.set_rhs(f"U_{u}_{v}", 1.0)
        ineq.active_st_partitions = {i}
        base_inequalities.append(ineq)
    return base_inequalities


def count_internal_sessions(
    partition : List[List[str]],
    sessions  : List[Tuple[str, str]]
) -> int:
    count = 0
    for Pi in partition:
        Pi_set = set(Pi)
        for s, t in sessions:
            if s in Pi_set and t in Pi_set:
                count += 1
    return count


def internal_per_partition(
    partition : List[List[str]],
    sessions  : List[Tuple[str, str]]
) -> List[int]:
    result = []
    for Pi in partition:
        Pi_set = set(Pi)
        result.append(sum(1 for s, t in sessions if s in Pi_set and t in Pi_set))
    return result


def verify_base_inequality(ineq, i, partition, sessions, edges) -> bool:
    Pi     = partition[i]
    Pi_set = set(Pi)
    index  = ineq.index
    if abs(ineq.coeffs[index.yst_idx(i)] - 1.0) > 1e-9:
        return False
    if abs(ineq.coeffs[index.yi_pi_idx(i)] - 1.0) > 1e-9:
        return False
    for v in Pi:
        if abs(ineq.coeffs[index.source_idx(v)] - (-1.0)) > 1e-9:
            return False
    for (u, v) in edges:
        c           = ineq.coeffs[index.edge_idx((u, v))]
        is_boundary = (u in Pi_set) != (v in Pi_set)
        expected    = -1.0 if is_boundary else 0.0
        if abs(c - expected) > 1e-9:
            return False
    return True


# ---------------------------------------------------------------------------
# Phase 3 primitives — per-node raw IO
# ---------------------------------------------------------------------------

def _node_partition_id(node: str, partition: List[List[str]]) -> int:
    """Return which partition set index contains node (-1 if none)."""
    for i, Pi in enumerate(partition):
        if node in Pi:
            return i
    return -1


def generate_node_io(
    node       : str,
    partition  : List[List[str]],
    nodes      : List[str],
    edges      : List[Tuple[str, str]],
    sessions   : List[Tuple[str, str]],
    index      : EntropyIndex,
    adjacency  : Dict[str, List[str]]
) -> FractionalInequality:
    """
    Raw encoding constraint for a single node v:

      h(Y_{ST(v)}, {U_{u→v} : u ∈ nbrs(v)}, {U_{v→u} : u ∈ nbrs(v)})
        ≤ h(Y_{S(v)}, {U_{u→v} : u ∈ nbrs(v)})

    This is the per-node IO inequality. Internal edges are NOT cancelled
    here — cancellation only occurs when you sum over all nodes in a
    partition set. By keeping them, Phase 3 can apply fractional weights
    before combining, potentially escaping the partition-bound family.

    Returns a FractionalInequality with lam=1, source_nodes=[node],
    partition_ids=[partition_id_of_node].
    """
    pid  = _node_partition_id(node, partition)
    nbrs = adjacency.get(node, [])

    # Identify sessions that touch this node
    touching = []
    for si, (s, t) in enumerate(sessions):
        if s == node or t == node:
            touching.append(si)

    # Which sessions have this node as source
    is_source_of = [si for si, (s, t) in enumerate(sessions) if s == node]

    fi = FractionalInequality(
        index,
        lam           = 1.0,
        source_nodes  = [node],
        partition_ids = [pid]
    )

    # LHS: all sessions touching v contribute Y_ST for their partition
    # Use partition membership to place in the right Y_ST slot
    pi_set = set(partition[pid]) if pid >= 0 else set()
    for si, (s, t) in enumerate(sessions):
        if s == node or t == node:
            # Which partition does this session's Y_ST variable belong to?
            # It belongs to whichever partition set contains this node.
            if pid >= 0:
                fi.coeffs[index.yst_idx(pid)] = max(
                    fi.coeffs[index.yst_idx(pid)], 1.0
                )
                # Internal session: also add Y_I_Pi
                if s in pi_set and t in pi_set:
                    fi.coeffs[index.yi_pi_idx(pid)] = max(
                        fi.coeffs[index.yi_pi_idx(pid)], 1.0
                    )

    # RHS: source entropy of this node
    fi.set_rhs(f"Y_S_{node}", 1.0)

    # RHS: all incident edge signals (undirected — try both directions)
    for u in nbrs:
        key1 = f"U_{u}_{node}"
        key2 = f"U_{node}_{u}"
        if key1 in index.var_to_idx:
            fi.set_rhs(key1, 1.0)
        elif key2 in index.var_to_idx:
            fi.set_rhs(key2, 1.0)

    # The outgoing signals U_{v→u} are determined by v's inputs (encoding
    # constraint) and appear on the LHS. For Phase 3 purposes we leave them
    # implicit: when two node IOs are combined and SUBMOD is applied, those
    # terms will appear as h(U_v→u) coefficients that ECAP then bounds.

    return fi


def generate_all_node_ios(
    partition  : List[List[str]],
    nodes      : List[str],
    edges      : List[Tuple[str, str]],
    sessions   : List[Tuple[str, str]],
    index      : EntropyIndex
) -> Dict[str, FractionalInequality]:
    """
    Returns {node_name: FractionalInequality} for every node.
    Phase 3 starts with this as its raw material.
    """
    adjacency = {n: [] for n in nodes}
    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)

    return {
        node: generate_node_io(
            node, partition, nodes, edges, sessions, index, adjacency
        )
        for node in nodes
    }


def generate_fractional_io(
    node_u     : str,
    node_v     : str,
    lam        : float,
    partition  : List[List[str]],
    nodes      : List[str],
    edges      : List[Tuple[str, str]],
    sessions   : List[Tuple[str, str]],
    index      : EntropyIndex
) -> FractionalInequality:
    """
    Form λ·IO(u) + (1-λ)·IO(v).

    This is the key fractional IO primitive. When u and v are in different
    partition sets (cross-partition), the resulting inequality has fractional
    coefficients on the Y_ST terms that no integer partition can replicate.

    After applying submodularity to the mixed RHS, the source terms combine
    with weight λ and (1-λ) respectively — escaping the partition bound family.

    Args:
        node_u, node_v : two nodes (ideally from different partitions)
        lam            : fractional weight λ ∈ (0, 1)
        ...

    Returns FractionalInequality with:
        lam           = lam (stored for trace)
        source_nodes  = [node_u, node_v]
        partition_ids = [pid_u, pid_v]
        coeffs        = lam * IO(u).coeffs + (1-lam) * IO(v).coeffs
    """
    assert 0.0 < lam < 1.0, f"λ must be in (0,1), got {lam}"

    adjacency = {n: [] for n in nodes}
    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)

    io_u = generate_node_io(node_u, partition, nodes, edges, sessions, index, adjacency)
    io_v = generate_node_io(node_v, partition, nodes, edges, sessions, index, adjacency)

    scaled_u = io_u.scale(lam)
    scaled_v = io_v.scale(1.0 - lam)

    combined        = scaled_u.add(scaled_v)
    combined.lam    = lam
    combined.source_nodes  = [node_u, node_v]
    combined.partition_ids = [
        _node_partition_id(node_u, partition),
        _node_partition_id(node_v, partition)
    ]
    return combined