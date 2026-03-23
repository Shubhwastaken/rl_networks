"""
Mathematical verification of inequality arithmetic.

Tests the base inequality generator, submodularity, and bound
extraction against hand-derived results from the paper on a
small 3-partition network.

Test network:
    Nodes   : A, B, C
    Edges   : A-B, A-C, B-C
    Sessions: (A->B) = session 0, (B->C) = session 1
    Partition:
        P0 = [A]   ST(P0) = {0}       I(P0,P0) = {}
        P1 = [B]   ST(P1) = {0, 1}    I(P1,P1) = {}
        P2 = [C]   ST(P2) = {1}       I(P2,P2) = {}

Hand-derived base inequalities (equation 66 of paper):

    P0=[A]: h(Y_ST_P0) + h(Y_I_P0_P0)
              <= h(Y_S_A) + h(U_P0_P1) + h(U_P0_P2)

    P1=[B]: h(Y_ST_P1) + h(Y_I_P1_P1)
              <= h(Y_S_B) + h(U_P0_P1) + h(U_P1_P2)

    P2=[C]: h(Y_ST_P2) + h(Y_I_P2_P2)
              <= h(Y_S_C) + h(U_P0_P2) + h(U_P1_P2)

After submodularity on P0+P1:
    union covers sessions {0,1} = ALL sessions -> Y_I collapse
    h(Y_I) + h(Y_I_P0_P0) + h(Y_I_P1_P1)
        <= h(Y_S_A) + h(Y_S_B)
           + h(U_P0_P1) + h(U_P0_P2) + h(U_P1_P2)
    (Y_S source terms cancel with h(Y_I) via source independence)

Proof 2 all-at-once on all 3:
    h(Y_I) + sum_i h(Y_I_Pi_Pi)
        <= sum_{ALL 3 edges} h(U_e)
    bound: r <= 3 / (2 + 0) = 1.5
"""

import sys
sys.path.insert(0, '.')

from fixed_inequality import EntropyIndex, Inequality
from fixed_base_inequality_generator import (
    generate_base_inequalities,
    verify_base_inequality,
    count_internal_sessions
)
from fixed_submodularity import (
    apply_pairwise_submodularity,
    apply_n2_submodularity_all_at_once
)


def sep():
    print("=" * 60)


def run_verification():
    print()
    sep()
    print("Mathematical Verification — 3-partition network")
    print("P0=[A], P1=[B], P2=[C]")
    sep()

    nodes     = ["A", "B", "C"]
    edges     = [("A", "B"), ("A", "C"), ("B", "C")]
    sessions  = [("A", "B"), ("B", "C")]   # session 0, session 1
    partition = [["A"], ["B"], ["C"]]

    # --- build EntropyIndex ---
    index = EntropyIndex(
        partitions=partition,
        nodes=nodes,
        edges=edges,
        sessions=sessions
    )

    print(f"\nEntropyIndex built: dim={index.dim} variables")
    print("Variable layout:")
    for i in range(index.dim):
        print(f"  [{i}] {index.idx_to_var[i]}")

    # --- verify session coverage ---
    print("\nSession coverage checks:")
    all_s = index.all_sessions()
    print(f"  all_sessions() = {all_s}  (expected {{0,1}})")
    assert all_s == {0, 1}, f"FAIL: expected {{0,1}}, got {all_s}"

    cov_01 = index.sessions_covered_by({0, 1})
    print(f"  sessions_covered_by({{0,1}}) = {cov_01}  (expected {{0,1}})")
    assert cov_01 == {0, 1}

    cov_0 = index.sessions_covered_by({0})
    print(f"  sessions_covered_by({{0}})   = {cov_0}  (expected {{0}})")
    assert cov_0 == {0}

    print("  PASS\n")

    # --- generate base inequalities ---
    sep()
    print("Base inequalities:")
    base = generate_base_inequalities(
        partition, nodes, edges, sessions, index
    )
    assert len(base) == 3, f"Expected 3 base ineqs, got {len(base)}"

    all_ok = True
    for i, ineq in enumerate(base):
        ok = verify_base_inequality(ineq, i, partition, sessions, edges)
        print(f"  P{i}={partition[i]}: {ineq}")
        print(f"    verify_base_inequality: {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_ok = False
    if not all_ok:
        print("FAIL: base inequality structure incorrect")
        return False

    # --- hand-check P0 coefficients ---
    sep()
    print("Hand-checking P0=[A] base inequality coefficients:")
    b0 = base[0]
    # Base inequalities store individual edge variables (U_e_A_B)
    # not cross-partition group variables (U_P0_P1)
    # because boundary edges are tracked per-edge in the base inequality
    checks = [
        ("Y_ST_P0",  b0.coeffs[index.yst_idx(0)],           +1.0),
        ("Y_I_P0_P0",b0.coeffs[index.yi_pi_idx(0)],          +1.0),
        ("Y_S_A",    b0.coeffs[index.source_idx("A")],        -1.0),
        ("U_e_A_B",  b0.coeffs[index.edge_idx(("A","B"))],    -1.0),
        ("U_e_A_C",  b0.coeffs[index.edge_idx(("A","C"))],    -1.0),
        ("U_e_B_C",  b0.coeffs[index.edge_idx(("B","C"))],     0.0),
    ]
    all_ok = True
    for name, got, expected in checks:
        ok = abs(got - expected) < 1e-9
        print(f"  {name:12s} expected {expected:+.1f}  got {got:+.4f}  "
              f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            all_ok = False
    if not all_ok:
        print("FAIL: P0 coefficients wrong")
        return False
    print("  All coefficients correct.\n")

    # --- pairwise submodularity on P0 and P1 ---
    sep()
    print("Pairwise submodularity: base[0] (P0) and base[1] (P1)")
    union_01, intersect_01 = apply_pairwise_submodularity(
        base[0], base[1], index, sessions
    )

    print(f"\n  Union   : {union_01}")
    print(f"  Intersct: {intersect_01}")

    # union should have Y_I since ST(P0) u ST(P1) = {0,1} = all sessions
    yi_in_union = union_01.yi_coeff()
    print(f"\n  Y_I coeff in union: {yi_in_union:.4f}  (expected 1.0)")
    assert abs(yi_in_union - 1.0) < 1e-6, \
        f"FAIL: Y_I coeff should be 1.0, got {yi_in_union}"

    # intersection should have U_e_A_B (edge A-B is shared boundary)
    # stored as individual edge variable, not cross-partition group
    cross_01 = intersect_01.coeffs[index.edge_idx(("A","B"))]
    print(f"  U_e_A_B in intersect: {cross_01:.4f}  (expected -1.0, A-B is shared boundary)")
    assert abs(cross_01 - (-1.0)) < 1e-6, \
        f"FAIL: U_e_A_B coeff in intersection should be -1.0, got {cross_01}"

    # active Y_ST in union should be empty (collapsed to Y_I)
    active = union_01.active_yst()
    print(f"  Active Y_ST in union: {active}  (expected empty set after collapse)")
    assert len(active) == 0, \
        f"FAIL: Y_ST terms should have collapsed, still active: {active}"

    print("  Pairwise submodularity PASS\n")

    # --- terminal form check ---
    sep()
    print("Terminal form check:")
    print(f"  union_01 valid terminal: {union_01.check_valid_terminal_form()}")
    print(f"  base[0]  valid terminal: {base[0].check_valid_terminal_form()}")
    assert union_01.check_valid_terminal_form(), \
        "FAIL: union_01 should be valid terminal form"
    assert not base[0].check_valid_terminal_form(), \
        "FAIL: base[0] should NOT be valid terminal form"
    print("  PASS\n")

    # --- bound extraction from pairwise union ---
    sep()
    print("Bound extraction from pairwise union_01:")
    internal_per_part = [
        sum(1 for s, t in sessions
            if s in set(Pi) and t in set(Pi))
        for Pi in partition
    ]
    # union_01 has Y_I coeff=1, internal=0 (no internal sessions),
    # rhs edge sum = edges covered by union boundary
    bound_pairwise = union_01.extract_bound(
        len(sessions), len(edges), internal_per_part
    )
    print(f"  internal_per_part: {internal_per_part}")
    print(f"  Pairwise bound: r <= {bound_pairwise:.4f}")
    print(f"  (Pairwise bound may differ from analytic — agent explores these)")

    # --- Proof 2 all-at-once ---
    sep()
    print("Proof 2 — (n,2)-way submodularity all at once:")
    final = apply_n2_submodularity_all_at_once(base, index, sessions)
    print(f"  Result: {final}")
    print(f"  h(Y_I) coeff    : {final.yi_coeff():.4f}  (expected 1.0)")
    print(f"  internal sum    : {final.internal_coeff_sum():.4f}  (expected 0.0)")
    print(f"  RHS edge sum    : {final.rhs_edge_sum():.4f}  (expected {len(edges)}.0)")

    assert abs(final.yi_coeff() - 1.0) < 1e-6, \
        f"FAIL: h(Y_I) coeff should be 1.0, got {final.yi_coeff()}"
    assert abs(final.rhs_edge_sum() - len(edges)) < 1e-6, \
        f"FAIL: RHS edge sum should be {len(edges)}, got {final.rhs_edge_sum()}"
    assert final.check_valid_terminal_form(), \
        "FAIL: Proof 2 result should be valid terminal form"

    proof2_bound = final.extract_bound(
        len(sessions), len(edges), internal_per_part
    )
    internal_total = sum(internal_per_part)
    analytic = len(edges) / (len(sessions) + internal_total)
    print(f"  Extracted bound : {proof2_bound:.4f}")
    print(f"  Analytic bound  : {analytic:.4f}")
    assert abs(proof2_bound - analytic) < 1e-6, \
        f"FAIL: extracted {proof2_bound:.4f} != analytic {analytic:.4f}"
    print("  Proof 2 PASS\n")

    # --- agent control over pairwise (Issue 5) ---
    sep()
    print("Agent-controlled pairwise submodularity (Issue 5):")
    print("  Agent chooses to combine base[0] (P0) and base[2] (P2)")
    print("  skipping base[1] (P1) — non-standard combination")
    union_02, intersect_02 = apply_pairwise_submodularity(
        base[0], base[2], index, sessions
    )
    print(f"  Union P0+P2  : {union_02}")
    print(f"  Intersect    : {intersect_02}")
    # ST(P0) u ST(P2) = {0} u {1} = {0,1} = all sessions
    yi_02 = union_02.yi_coeff()
    print(f"  Y_I coeff: {yi_02:.4f}  (expected 1.0 — all sessions covered)")
    assert abs(yi_02 - 1.0) < 1e-6, \
        f"FAIL: Y_I coeff should be 1.0, got {yi_02}"
    # intersection should have U_e_A_C (edge A-C crosses both P0 and P2)
    cross_02 = intersect_02.coeffs[index.edge_idx(("A","C"))]
    print(f"  U_e_A_C in intersect: {cross_02:.4f}  (expected -1.0)")
    assert abs(cross_02 - (-1.0)) < 1e-6
    print("  Agent control PASS\n")

    sep()
    print("ALL VERIFICATION TESTS PASSED")
    print("Inequality arithmetic is mathematically correct.")
    print("Safe to proceed to GNN policy implementation.")
    return True


if __name__ == "__main__":
    ok = run_verification()
    if not ok:
        print("\nVERIFICATION FAILED — fix arithmetic before proceeding")
        sys.exit(1)
