"""
Base inequality generator.

Generates one base inequality per partition set (equation 66 of paper).

For each Pi:
    LHS: +1 on Y_ST_Pi   (sessions touching Pi)
         +1 on Y_I_Pi    (sessions internal to Pi)
    RHS: -1 on Y_S_v     for each v in Pi
         -1 on U_{u}_{v} for each boundary edge of Pi
"""

from typing import List, Tuple, Set
from fixed_inequality import Inequality, EntropyIndex


def generate_base_inequalities(
    partition  : List[List[str]],
    nodes      : List[str],
    edges      : List[Tuple[str, str]],
    sessions   : List[Tuple[str, str]],
    index      : EntropyIndex
) -> List[Inequality]:

    base_inequalities = []

    for i, Pi in enumerate(partition):
        Pi_set = set(Pi)
        ineq   = Inequality(index)

        # LHS: sessions touching Pi
        ineq.set_lhs(f"Y_ST_P{i}", 1.0)

        # LHS: sessions internal to Pi
        ineq.set_lhs(f"Y_I_P{i}", 1.0)

        # RHS: source entropy at each node in Pi
        for v in Pi:
            ineq.set_rhs(f"Y_S_{v}", 1.0)

        # RHS: individual edge signal entropy for boundary edges only
        for (u, v) in edges:
            if (u in Pi_set) != (v in Pi_set):
                ineq.set_rhs(f"U_{u}_{v}", 1.0)

        ineq.active_st_partitions = {i}
        base_inequalities.append(ineq)

    return base_inequalities


def count_internal_sessions(
    partition : List[List[str]],
    sessions  : List[Tuple[str, str]]
) -> int:
    """Returns total Sigma |I(Pi,Pi)| across all partition sets."""
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
    """Returns list of |I(Pi,Pi)| for each partition set Pi."""
    result = []
    for Pi in partition:
        Pi_set = set(Pi)
        result.append(sum(
            1 for s, t in sessions
            if s in Pi_set and t in Pi_set
        ))
    return result


def verify_base_inequality(ineq, i, partition, sessions, edges) -> bool:
    """Verifies a base inequality has correct structure."""
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