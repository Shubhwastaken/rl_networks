"""
Submodularity operations for Phase 2.

TWO PROOF STRATEGIES:

apply_n2_submodularity_all_at_once  -- Proof 2:
    Applies (n,2)-way submodularity to ALL n base inequalities at once.
    Always produces the standard partition bound exactly.
    RHS uses ALL edges including internal ones.

apply_pairwise_submodularity  -- Proof 1 style:
    Agent chooses exactly two inequalities to combine (idx_i, idx_j).
    Y_I collapse fires when union of active ST sets covers all sessions.
    This is what the RL agent explores.

FIX: Removed cross-partition group variable (U_P{i}_P{j}) handling
     from intersection since those variables no longer exist in the index.
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

    # Step 2: n Y_ST terms -> 2*h(Y_I)
    for i in range(n):
        result.coeffs[index.yst_idx(i)] = 0.0
    result.coeffs[index.yi_idx()] += 2.0

    # Step 3: source independence -- cancel one h(Y_I) with source terms
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
    Applies h(A) + h(B) >= h(A|B) + h(A&B) to two chosen inequalities.

    Returns (union_ineq, intersection_ineq).
    Y_I collapse fires in union if union covers all sessions.
    """
    union_ineq        = Inequality(index)
    intersection_ineq = Inequality(index)

    active_a = ineq_a.active_yst()
    active_b = ineq_b.active_yst()

    # --- UNION ---

    # Y_ST: union of active sets — preserve fractional coefficients.
    # Use max(coeff_a, coeff_b) so fractional weights are not silently
    # overwritten with 1.0. For integer IOs both coeffs are 1.0 so
    # max = 1.0 and behaviour is unchanged.
    for i in (active_a | active_b):
        c = max(ineq_a.coeffs[index.yst_idx(i)],
                ineq_b.coeffs[index.yst_idx(i)])
        union_ineq.coeffs[index.yst_idx(i)] = c

    # Y_I(Pi,Pi): take max coefficient
    for i in range(index.n()):
        c = max(ineq_a.coeffs[index.yi_pi_idx(i)],
                ineq_b.coeffs[index.yi_pi_idx(i)])
        if c > 1e-9:
            union_ineq.coeffs[index.yi_pi_idx(i)] = c

    # Y_I: combining two scalar coefficients on the same variable h(Y_I).
    #
    # The correct operation depends on whether Y_ST terms are present:
    #
    # PARTITION IO PATHWAY (at least one input has Y_ST terms):
    #   max() is valid here because the Y_ST collapse step will later
    #   replace Y_ST terms with a c_min * h(Y_I) contribution via the
    #   proven weighted subadditivity theorem.  The Y_I coefficient from
    #   the non-Y_ST side rides through unchanged and the collapse adjusts
    #   it correctly.
    #
    # PURE NODE IO PATHWAY (neither input has Y_ST terms):
    #   Both inputs are scalar inequalities on the SAME variable h(Y_I).
    #   Submodularity h(A)+h(B) >= h(A∪B)+h(A∩B) applies to sets of
    #   random variables, not to scalar multiples of a single entropy term.
    #   max(c_a, c_b) * h(Y_I) is NOT a valid bound — it makes the LHS
    #   weaker than either input while the RHS sources/edges accumulate
    #   (min), creating a denominator that cancel_source_terms can shrink
    #   to near-zero, producing spuriously tight invalid bounds.
    #
    #   The only safe bound for the union of two node IOs is
    #   min(c_a, c_b) * h(Y_I) — this is the largest coefficient that is
    #   still valid for BOTH inputs, making pairwise submod on pure node
    #   IOs a no-op for terminal form generation (min Y_I + max sources
    #   means cancel_source_terms always reverts).  The agent correctly
    #   learns that add() / STORE_AND_RESET is the productive path for
    #   combining node IOs.
    c_yi_a = ineq_a.coeffs[index.yi_idx()]
    c_yi_b = ineq_b.coeffs[index.yi_idx()]
    if active_a or active_b:
        # Partition IO pathway: max() is valid, collapse handles it later
        c_yi_union = max(c_yi_a, c_yi_b)
    else:
        # Pure node IO pathway: only min() is mathematically safe
        c_yi_union = min(c_yi_a, c_yi_b)
    if c_yi_union > 1e-9:
        union_ineq.coeffs[index.yi_idx()] = c_yi_union

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

    # Collapse Y_ST -> h(Y_I) if all sessions covered
    union_ineq = _collapse_to_yi_if_valid(union_ineq, index, sessions)

    # Also try source cancellation for node IO pathway
    # (node IOs set Y_I directly, not via Y_ST, so _collapse_to_yi_if_valid
    #  is a no-op for them — this catches that case)
    union_ineq = _cancel_sources_for_node_ios(union_ineq, index, sessions)

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

    # Y_I: take min coefficient for intersection
    c_yi_inter = min(ineq_a.coeffs[index.yi_idx()],
                     ineq_b.coeffs[index.yi_idx()])
    if c_yi_inter > 1e-9:
        intersection_ineq.coeffs[index.yi_idx()] = c_yi_inter

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

    PROVEN FORMULA (Weighted Subadditivity Theorem):
      For weights c_k > 0 and random variables X_k:
        Σ_k c_k * h(X_k)  >=  c_min * h(X_1,...,X_n)
      where c_min = min_k(c_k).

      Proof:
        Step 1: Σ c_k*h(X_k) >= c_min * Σ h(X_k)   [since c_k >= c_min, h >= 0]
        Step 2: Σ h(X_k) >= h(X_1,...,X_n)           [subadditivity]
        Step 3: Combine.

      This is tight: K = c_min cannot be improved without further structure.

    The sum-min formula (sum_coeff - min_coeff) was shown to be FALSE:
      For independent X_k with equal entropy r, sum-min requires
      c_min/c_sum >= (n-1)/n which fails for fractional weights.

    Effect:
      Y_ST terms replaced by c_min * h(Y_I).
      Source terms for covered nodes zeroed (source independence).
    """
    active = ineq.active_yst()
    if not active:
        return ineq

    if index.sessions_covered_by(active) != index.all_sessions():
        return ineq

    result = ineq.copy()

    # Collect Y_ST coefficients
    yst_coeffs = [ineq.coeffs[index.yst_idx(i)] for i in active]
    min_coeff  = min(yst_coeffs)   # PROVEN valid collapse coefficient

    # Zero out all Y_ST terms
    for i in range(index.n()):
        result.coeffs[index.yst_idx(i)] = 0.0

    # Add c_min * h(Y_I) — the only proven valid Y_I coefficient
    result.coeffs[index.yi_idx()] += min_coeff

    # Zero source terms for covered nodes
    covered_nodes = set()
    for i in active:
        covered_nodes |= set(index.partitions[i])
    for v in covered_nodes:
        result.coeffs[index.source_idx(v)] = 0.0

    return result


def _cancel_sources_for_node_ios(
    ineq     : Inequality,
    index    : EntropyIndex,
    sessions : List[Tuple[str, str]]
) -> Inequality:
    """
    Cancel source terms for node-IO-derived inequalities.

    This is the node IO counterpart of _collapse_to_yi_if_valid:
      - _collapse_to_yi_if_valid handles partition IOs (Y_ST -> Y_I + zero sources)
      - This handles node IOs (Y_I already set, just zero sources and adjust Y_I)

    Only fires when:
      1. No Y_ST terms active (this is not a partition pathway)
      2. Y_I > 0 (node IOs set Y_I directly)
      3. All session source nodes have source terms (coverage complete)

    The math: if Y_S_v appears on the RHS with coefficient |c_v|, and node v
    sources n_src sessions, then h(Y_S_v) = n_src * r = (n_src/|I|) * h(Y_I).
    So we subtract sum(|c_v| * n_src_v / |I|) from the Y_I coefficient and
    zero all source coefficients.
    """
    # Skip if Y_ST terms present — partition pathway handled by _collapse
    if ineq.active_yst():
        return ineq

    return ineq.cancel_source_terms()