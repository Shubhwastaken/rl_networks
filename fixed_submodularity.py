"""
Submodularity operations for Phase 2.

TWO PROOF STRATEGIES:

apply_n2_submodularity_all_at_once  — Proof 2:
    Applies (n,2)-way submodularity to ALL n base inequalities at once.
    Always produces the standard partition bound exactly.
    RHS uses ALL edges including internal ones.

apply_pairwise_submodularity  — Proof 1 style:
    Agent chooses exactly two inequalities to combine (idx_i, idx_j).
    Y_I collapse fires when union of active ST sets covers all sessions.
    This is what the RL agent explores.

FIX in apply_pairwise_submodularity:
    Removed the hardcoded `union_ineq.coeffs[index.yi_idx()] = 1.0`
    that was setting Y_I unconditionally before the collapse check ran.
    Y_I is now set ONLY inside _collapse_to_yi_if_valid when the
    session-coverage condition is actually met.
"""

from typing import List, Tuple, Set
from fixed_inequality import Inequality, EntropyIndex


def apply_n2_submodularity_all_at_once(
    base_inequalities : List[Inequality],
    index             : EntropyIndex,
    sessions          : List[Tuple[str, str]]
) -> Inequality:
    """
    Proof 2: applies (n,2)-way submodularity to all n base inequalities.

    Steps (equations 49-64 of paper):
    1. Sum all n base inequalities
    2. Replace n Y_ST terms with 2*h(Y_I) via submodularity
    3. Cancel one h(Y_I) with source terms via source independence
    4. Reset RHS to ALL edges with coefficient -1
    """
    if not base_inequalities:
        raise ValueError("Need at least one base inequality")

    n = index.n()

    result = base_inequalities[0].copy()
    for ineq in base_inequalities[1:]:
        result = result.add(ineq)

    all_covered = set()
    for i in range(n):
        all_covered |= index.st_sessions[i]
    if all_covered != index.all_sessions():
        raise ValueError("Partition does not cover all sessions")

    # Step 2: n Y_ST terms → 2*h(Y_I)
    for i in range(n):
        result.coeffs[index.yst_idx(i)] = 0.0
    result.coeffs[index.yi_idx()] += 2.0

    # Step 3: source independence — cancel one h(Y_I) with source terms
    result.coeffs[index.yi_idx()] -= 1.0
    for v in index.nodes:
        result.coeffs[index.source_idx(v)] = 0.0

    # Step 4: RHS = ALL edges, each with coefficient -1
    for e in index.edges:
        result.coeffs[index.edge_idx(e)] = -1.0

    return result


def apply_pairwise_submodularity(
    ineq_a   : Inequality,
    ineq_b   : Inequality,
    index    : EntropyIndex,
    sessions : List[Tuple[str, str]]
) -> Tuple[Inequality, Inequality]:
    """
    Applies h(A) + h(B) >= h(A∪B) + h(A∩B) to two chosen inequalities.

    Returns (union_ineq, intersection_ineq).
    Y_I collapse fires in union if union covers all sessions.
    """
    union_ineq        = Inequality(index)
    intersection_ineq = Inequality(index)

    active_a = ineq_a.active_yst()
    active_b = ineq_b.active_yst()

    # --- UNION ---

    # Y_ST: union of active sets
    for i in (active_a | active_b):
        union_ineq.set_lhs(f"Y_ST_P{i}", 1.0)

    # Y_I(Pi,Pi): take max coefficient
    for i in range(index.n()):
        c = max(ineq_a.coeffs[index.yi_pi_idx(i)],
                ineq_b.coeffs[index.yi_pi_idx(i)])
        if c > 1e-9:
            union_ineq.coeffs[index.yi_pi_idx(i)] = c

    # FIX: do NOT set Y_I unconditionally here.
    # Y_I is set only inside _collapse_to_yi_if_valid
    # when the session-coverage condition is verified.

    # RHS sources: union = more negative (min)
    for v in index.nodes:
        c = min(ineq_a.coeffs[index.source_idx(v)],
                ineq_b.coeffs[index.source_idx(v)])
        if c < -1e-9:
            union_ineq.coeffs[index.source_idx(v)] = c

    # RHS edges: union = more negative (min)
    for e in index.edges:
        c = min(ineq_a.coeffs[index.edge_idx(e)],
                ineq_b.coeffs[index.edge_idx(e)])
        if c < -1e-9:
            union_ineq.coeffs[index.edge_idx(e)] = c

    # Collapse Y_ST → h(Y_I) if all sessions covered
    union_ineq = _collapse_to_yi_if_valid(union_ineq, index, sessions)

    # --- INTERSECTION ---

    # Y_ST: intersection of active sets
    for i in (active_a & active_b):
        intersection_ineq.set_lhs(f"Y_ST_P{i}", 1.0)

    # Y_I(Pi,Pi): take min coefficient
    for i in range(index.n()):
        c = min(ineq_a.coeffs[index.yi_pi_idx(i)],
                ineq_b.coeffs[index.yi_pi_idx(i)])
        if c > 1e-9:
            intersection_ineq.coeffs[index.yi_pi_idx(i)] = c

    # Cross-partition edge groups: take min
    for i in range(index.n()):
        for j in range(i + 1, index.n()):
            c = min(ineq_a.coeffs[index.cross_idx(i, j)],
                    ineq_b.coeffs[index.cross_idx(i, j)])
            if c > 1e-9:
                intersection_ineq.coeffs[index.cross_idx(i, j)] = c

    # RHS sources: intersection = less negative (max)
    for v in index.nodes:
        c = max(ineq_a.coeffs[index.source_idx(v)],
                ineq_b.coeffs[index.source_idx(v)])
        if c < -1e-9:
            intersection_ineq.coeffs[index.source_idx(v)] = c

    # RHS edges: intersection = less negative (max)
    for e in index.edges:
        c = max(ineq_a.coeffs[index.edge_idx(e)],
                ineq_b.coeffs[index.edge_idx(e)])
        if c < -1e-9:
            intersection_ineq.coeffs[index.edge_idx(e)] = c

    return union_ineq, intersection_ineq


def _collapse_to_yi_if_valid(
    ineq     : Inequality,
    index    : EntropyIndex,
    sessions : List[Tuple[str, str]]
) -> Inequality:
    """
    Collapses Y_ST terms to h(Y_I) when union covers ALL sessions.

    Condition: sessions covered by active Y_ST == all sessions.
    Does NOT require all nodes to be covered — sessions only.

    Effect: n_active Y_ST terms → (n_active - 1) * h(Y_I)
            source terms for covered nodes zeroed out.
    """
    active = ineq.active_yst()
    if not active:
        return ineq

    if index.sessions_covered_by(active) != index.all_sessions():
        return ineq

    result   = ineq.copy()
    n_active = len(active)

    # Zero out all Y_ST terms
    for i in range(index.n()):
        result.coeffs[index.yst_idx(i)] = 0.0

    # (n_active - 1) * h(Y_I)
    result.coeffs[index.yi_idx()] += (n_active - 1)

    # Cancel source terms for nodes in the active partitions
    covered_nodes = set()
    for i in active:
        covered_nodes |= set(index.partitions[i])
    for v in covered_nodes:
        result.coeffs[index.source_idx(v)] = 0.0

    return result