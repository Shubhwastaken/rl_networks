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

    # Step 4 REMOVED (2026-08-02): the RHS is now the ACCUMULATED capacity
    # from summing the n partition IOs, not a reset.
    #
    # The old line was:
    #     for e in index.edges: result.coeffs[index.edge_idx(e)] = -1.0
    # i.e. it OVERWROTE the summed capacity with -1 on every edge. That is
    # the same error class as the union (min shared capacity) and the old
    # collapse (delete without paying): it discards accumulated RHS mass.
    #
    # It tested clean on all 17 registry graphs only by coincidence -- in
    # an independent-set partition every edge is a cross edge, so it is a
    # boundary edge of exactly 2 parts and the sum gives it -2, which the
    # reset halved to -1 while the +2.0/-1.0 Y_I accounting absorbed the
    # difference. Correct by accident on the family it was written for.
    #
    # Note the accumulated form carries capacity 2 per cross edge, so the
    # bound it yields is 2|cut|/(|I|+internal) -- a factor 2 looser than
    # the |E|/(|I|+internal) the reset produced. See _partition_bound_of /
    # compute_optimal_bound, which define PB with the single-capacity
    # convention; the two conventions must not be compared directly.

    return result


def apply_pairwise_submodularity(
    ineq_a   : Inequality,
    ineq_b   : Inequality,
    index    : EntropyIndex,
    sessions : List[Tuple[str, str]]
) -> Inequality:
    """
    Combine two inequalities.  Returns a SINGLE inequality (the sum,
    followed by the exact Y_ST collapse and source cancellation).

    WHAT THIS REPLACED (2026-08-02) AND WHY
    ---------------------------------------
    This used to build a "union" by taking max() on LHS coefficients and
    min() on RHS coefficients, and a matching "intersection" with min() on
    the LHS and max() on the RHS.  Both were labelled submodularity.
    Neither is: h(A)+h(B) >= h(A|B)+h(A&B) is a statement about SETS OF
    RANDOM VARIABLES inside entropy terms, not about the coefficient
    vectors of two inequalities.

    The union counted SHARED RHS capacity once instead of twice.  On
    36253 valid inequality pairs drawn from the real generator families
    (node IOs, partition IOs, corrected crypto, corrected decode) it
    produced 66 FALSE results.  Example, both inputs valid:
        A = CRYPTO({a,b,c,s}) : Y_I <= U_a_d + U_b_d + U_c_t + U_c_d
        B = PIO(P4)           : Y_ST_P4 + Y_I_P4 <= Y_S_d + U_a_d + U_b_d
                                                    + U_d_t + U_c_d
        union                 : both LHS added, shared edges counted once
                                -> violation +1.0

    The intersection was far worse: max() on two negative RHS
    coefficients yields the LESS negative one, i.e. it SHRINKS capacity.
    It produced 12665 FALSE results out of the same 36253 pairs (35%),
    and every one went straight into the pool via pool.append(inter_ineq).
    Minimal case on diamond_6N with two ordinary node IOs whose RHS
    supports are disjoint, so the whole RHS vanishes:
        A = IO(S1) : 0.5*Y_I <= Y_S_S1 + U_S1_v1 + U_S1_v2
        B = IO(S2) : 0.5*Y_I <= Y_S_S2 + U_S2_v1 + U_S2_v2
        inter      : 0.5*Y_I <= 0                  -> violation +1.0

    There is no sound union worth keeping.  In this index every non-edge
    variable is an exact multiple of r, so submodularity on the Y_ST
    terms degenerates to inclusion-exclusion
        |st_i|r + |st_j|r == |st_i U st_j|r + |st_i & st_j|r
    -- an identity yielding nothing beyond addition.  The only repairable
    variant (max on LHS, SUM on RHS) is componentwise <= A+B and so is
    valid but strictly dominated by plain addition.

    Plain addition produced 0 FALSE results on all 36253 pairs.

    THE RULE this enforces: any operation combining two inequalities must
    ADD RHS capacity -- never min it, max it, or overwrite it.
    """
    result = ineq_a.add(ineq_b)
    result = _collapse_to_yi_if_valid(result, index, sessions)
    result = _cancel_sources_for_node_ios(result, index, sessions)
    return result


def _collapse_to_yi_if_valid(
    ineq     : Inequality,
    index    : EntropyIndex,
    sessions : List[Tuple[str, str]]
) -> Inequality:
    """
    Collapses Y_ST terms to h(Y_I) when union covers ALL sessions.

    Condition: sessions covered by active Y_ST == all sessions.

    EXACT FORMULA (source independence), replacing the old c_min bound:
        Σ_k c_k * h(Y_ST_Pk)  ==  K * h(Y_I),   K = Σ_k c_k*|st_k| / |I|
      plus a matching deduction D for the source terms it deletes.
      See the body for the derivation and for what c_min got wrong.

    The sum-min formula (sum_coeff - min_coeff) was shown to be FALSE and
    is not used. c_min was valid but lossy; K supersedes it.

    Effect:
      Y_ST terms replaced by K * h(Y_I).
      Source terms for covered nodes zeroed AND paid for via -D on Y_I.
    """
    active = ineq.active_yst()
    if not active:
        return ineq

    if index.sessions_covered_by(active) != index.all_sessions():
        return ineq

    result = ineq.copy()
    n_sessions = len(sessions)
    if n_sessions == 0:
        return ineq

    # ── Step 1: Y_ST -> Y_I, as an EXACT identity (was: lossy c_min) ────────
    # Under independent equal-rate sources every non-edge variable is an
    # exact multiple of r:  h(Y_ST_Pk) = |st_k|*r  and  h(Y_I) = |I|*r.
    # Therefore
    #     sum_k c_k*h(Y_ST_Pk) == ( sum_k c_k*|st_k| / |I| ) * h(Y_I) == K*h(Y_I)
    # which is an identity, not a bound.  c_min is a valid but strictly
    # lossy under-estimate of K.  Cross-check: when every session touches
    # two parts, sum_k |st_k| = 2|I| so K = 2 -- exactly the hard-coded
    # +2.0 in apply_n2_submodularity_all_at_once, which this generalises.
    K = sum(ineq.coeffs[index.yst_idx(i)] * len(index.st_sessions[i])
            for i in active) / n_sessions

    # ── Step 2: pay for the source deletion (this is what was missing) ─────
    # Deleting sum_v |s_v|*h(Y_S_v) removes sum_v |s_v|*n_src(v)*r from the
    # RHS, so the LHS must fall by the same amount -- i.e. the Y_I
    # coefficient by D = sum_v |s_v|*n_src(v)/|I|.
    #
    # THE BUG THIS REPLACES (2026-08-02): the old body added c_min to Y_I
    # and then zeroed the covered source terms unconditionally. The c_min
    # pays only for removing Y_ST; nothing paid for the source deletion.
    # On a pure partition IO that happened to be valid (it consumed
    # exactly the 2 units of slack such an IO carries, slack 2 -> 0), but
    # it is not a theorem: on 4000 random VALID inequalities carrying a
    # Y_ST term the old code turned 91 of them FALSE.  Once a union has
    # given the inequality Y_I mass from outside the Y_ST pathway, the
    # unpaid deletion is exactly h(Y_I) * (that outside Y_I mass) -- the
    # +0.5 violation on diamond_6N that produced 0.5*Y_I <= 0.5*U_v2_t2.
    #
    # Both steps here are exact, so the corrected collapse is SLACK
    # PRESERVING and cannot turn a valid inequality false: 0 of the same
    # 4000 turn false, with slack change 0 to machine precision.
    covered_nodes = set()
    for i in active:
        covered_nodes |= set(index.partitions[i])

    D = 0.0
    for v in covered_nodes:
        c = result.coeffs[index.source_idx(v)]
        if c < -1e-12:
            n_src = sum(1 for s, _t in sessions if s == v)
            D += abs(c) * n_src / n_sessions
            result.coeffs[index.source_idx(v)] = 0.0

    for i in range(index.n()):
        result.coeffs[index.yst_idx(i)] = 0.0
    result.coeffs[index.yi_idx()] += K - D

    result.record_op("COLLAPSE_YST", {"K": float(K), "D": float(D),
                                      "active": sorted(active)})
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