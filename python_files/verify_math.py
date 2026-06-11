"""
Mathematical verification of inequality arithmetic.

Tests the base inequality generator, submodularity, and bound
extraction against hand-derived results from the paper on a
small 3-partition network.

Updated for the new EntropyIndex that no longer has cross-partition
group variables (U_P{i}_P{j}).
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
from functional_dependence import (
    apply_crypto_inequality_direct,
    apply_decode_substitution,
    apply_encode_substitution
)


def sep():
    print("=" * 60)


def run_verification():
    print()
    sep()
    print("Mathematical Verification -- 3-partition network")
    print("P0=[A], P1=[B], P2=[C]")
    sep()

    nodes     = ["A", "B", "C"]
    edges     = [("A", "B"), ("A", "C"), ("B", "C")]
    sessions  = [("A", "B"), ("B", "C")]
    partition = [["A"], ["B"], ["C"]]

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

    # --- verify no cross-partition variables ---
    print("\nVerify cross-partition variables removed:")
    try:
        index.cross_idx(0, 1)
        print("  FAIL: cross_idx should raise KeyError")
        return False
    except KeyError:
        print("  PASS: cross_idx raises KeyError as expected")

    # --- verify session coverage ---
    print("\nSession coverage checks:")
    all_s = index.all_sessions()
    print(f"  all_sessions() = {all_s}  (expected {{0,1}})")
    assert all_s == {0, 1}

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
    assert len(base) == 3

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
    checks = [
        ("Y_ST_P0",  b0.coeffs[index.yst_idx(0)],           +1.0),
        ("Y_I_P0",   b0.coeffs[index.yi_pi_idx(0)],         +1.0),
        ("Y_S_A",    b0.coeffs[index.source_idx("A")],      -1.0),
        ("U_A_B",    b0.coeffs[index.edge_idx(("A","B"))],   -1.0),
        ("U_A_C",    b0.coeffs[index.edge_idx(("A","C"))],   -1.0),
        ("U_B_C",    b0.coeffs[index.edge_idx(("B","C"))],    0.0),
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

    yi_in_union = union_01.yi_coeff()
    print(f"\n  Y_I coeff in union: {yi_in_union:.4f}  (expected 1.0)")
    assert abs(yi_in_union - 1.0) < 1e-6

    # intersection should have U_A_B (edge A-B is shared boundary)
    cross_01 = intersect_01.coeffs[index.edge_idx(("A","B"))]
    print(f"  U_A_B in intersect: {cross_01:.4f}  (expected -1.0)")
    assert abs(cross_01 - (-1.0)) < 1e-6

    active = union_01.active_yst()
    print(f"  Active Y_ST in union: {active}  (expected empty)")
    assert len(active) == 0

    print("  Pairwise submodularity PASS\n")

    # --- terminal form check ---
    sep()
    print("Terminal form check:")
    print(f"  union_01 valid terminal: {union_01.check_valid_terminal_form()}")
    print(f"  base[0]  valid terminal: {base[0].check_valid_terminal_form()}")
    assert union_01.check_valid_terminal_form()
    assert not base[0].check_valid_terminal_form()
    print("  PASS\n")

    # --- bound extraction ---
    sep()
    print("Bound extraction from pairwise union_01:")
    internal_per_part = [
        sum(1 for s, t in sessions if s in set(Pi) and t in set(Pi))
        for Pi in partition
    ]
    bound_pairwise = union_01.extract_bound(
        len(sessions), len(edges), internal_per_part
    )
    print(f"  internal_per_part: {internal_per_part}")
    print(f"  Pairwise bound: r <= {bound_pairwise:.4f}")

    # --- Proof 2 all-at-once ---
    sep()
    print("Proof 2 -- (n,2)-way submodularity all at once:")
    final = apply_n2_submodularity_all_at_once(base, index, sessions)
    print(f"  Result: {final}")
    print(f"  h(Y_I) coeff    : {final.yi_coeff():.4f}  (expected 1.0)")
    print(f"  internal sum    : {final.internal_coeff_sum():.4f}  (expected 0.0)")
    print(f"  RHS edge sum    : {final.rhs_edge_sum():.4f}  (expected {len(edges)}.0)")

    assert abs(final.yi_coeff() - 1.0) < 1e-6
    assert abs(final.rhs_edge_sum() - len(edges)) < 1e-6
    assert final.check_valid_terminal_form()

    proof2_bound = final.extract_bound(
        len(sessions), len(edges), internal_per_part
    )
    internal_total = sum(internal_per_part)
    analytic = len(edges) / (len(sessions) + internal_total)
    print(f"  Extracted bound : {proof2_bound:.4f}")
    print(f"  Analytic bound  : {analytic:.4f}")
    assert abs(proof2_bound - analytic) < 1e-6
    print("  Proof 2 PASS\n")

    # --- agent control: combine P0 and P2 ---
    sep()
    print("Agent-controlled pairwise: base[0] (P0) + base[2] (P2)")
    union_02, intersect_02 = apply_pairwise_submodularity(
        base[0], base[2], index, sessions
    )
    print(f"  Union P0+P2  : {union_02}")
    yi_02 = union_02.yi_coeff()
    print(f"  Y_I coeff: {yi_02:.4f}  (expected 1.0)")
    assert abs(yi_02 - 1.0) < 1e-6

    cross_02 = intersect_02.coeffs[index.edge_idx(("A","C"))]
    print(f"  U_A_C in intersect: {cross_02:.4f}  (expected -1.0)")
    assert abs(cross_02 - (-1.0)) < 1e-6
    print("  Agent control PASS\n")

    # --- verify dim is smaller without cross-partition vars ---
    sep()
    # Expected dim: 3 (Y_ST) + 3 (Y_I_Pi) + 3 (Y_S) + 3 (U_e) + 1 (Y_I) = 13
    # Old dim would have been 13 + 3 (cross-partition) = 16
    expected_dim = 3 + 3 + 3 + 3 + 1
    print(f"Dimension check: dim={index.dim}  expected={expected_dim}")
    assert index.dim == expected_dim, f"FAIL: expected {expected_dim}, got {index.dim}"
    print("  PASS\n")

    # --- verify functional dependence inequalities ---
    sep()
    print("Functional Dependence: Crypto Inequality on cut V'={A}")
    # V'={A}, separates session 0 (A->B), cut edges {A->B, A->C}
    # Start from the proof2 terminal inequality and apply crypto
    crypto_base = final.copy()
    yi_before = crypto_base.yi_coeff()
    crypto_result, was_applied = apply_crypto_inequality_direct(
        crypto_base, frozenset(["A"]), nodes, edges, sessions, index
    )
    print(f"  Applied: {was_applied}")
    assert was_applied, "Crypto inequality should apply for cut V'={{A}}"
    yi_after = crypto_result.yi_coeff()
    yi_gain = yi_after - yi_before
    print(f"  Y_I before: {yi_before:.4f}  after: {yi_after:.4f}  gain: {yi_gain:.4f}")
    # Cut {A} separates session 0 (A->B). 1 session separated out of 2.
    # So Y_I should increase by 1/2 = 0.5.
    expected_gain = 1.0 / len(sessions)  # 1 separated session / 2 total
    print(f"  Expected gain: {expected_gain:.4f}")
    assert abs(yi_gain - expected_gain) < 1e-6, f"Expected {expected_gain}, got {yi_gain}"
    # RHS edges should be UNCHANGED (the fixed crypto inequality doesn't touch RHS)
    rhs_before = final.rhs_edge_sum()
    rhs_after = crypto_result.rhs_edge_sum()
    print(f"  RHS edges before: {rhs_before:.4f}  after: {rhs_after:.4f}")
    assert abs(rhs_before - rhs_after) < 1e-6, "RHS should not change"
    print("  Crypto PASS\n")

    sep()
    print("Functional Dependence: Decode Substitution on session 0 (A->B)")
    # sink is B, incident edges: A-B, B-C
    decode_base = final.copy()
    yi_before_dec = decode_base.yi_coeff()
    decode_result, dec_applied = apply_decode_substitution(
        decode_base, 0, sessions, edges, index
    )
    print(f"  Applied: {dec_applied}")
    assert dec_applied, "Decode should apply for session 0"
    yi_after_dec = decode_result.yi_coeff()
    dec_gain = yi_after_dec - yi_before_dec
    print(f"  Y_I before: {yi_before_dec:.4f}  after: {yi_after_dec:.4f}  gain: {dec_gain:.4f}")
    # Decode adds 1/|I| = 0.5 to Y_I
    expected_dec_gain = 1.0 / len(sessions)
    print(f"  Expected gain: {expected_dec_gain:.4f}")
    assert abs(dec_gain - expected_dec_gain) < 1e-6
    print("  Decode PASS\n")

    sep()
    print("Functional Dependence: Encode Substitution on edge (B,C)")
    # Encode: h(U_{B->C}) <= h(Y_S_B) + h(U_{A->B})
    # This replaces a positive U_B_C coefficient on LHS with source+incoming
    # For this to apply, U_B_C must have a positive coefficient.
    # Create a test inequality with U_B_C on LHS
    encode_test = Inequality(index)
    encode_test.coeffs[index.edge_idx(("B","C"))] = 1.0  # put on LHS
    adjacency_test = {"A": ["B", "C"], "B": ["A", "C"], "C": ["A", "B"]}
    encode_result, enc_applied = apply_encode_substitution(
        encode_test, ("B", "C"), partition, nodes, edges, sessions, index,
        adjacency_test
    )
    print(f"  Applied: {enc_applied}")
    assert enc_applied, "Encode should apply for edge (B,C)"
    u_bc_enc = encode_result.coeffs[index.edge_idx(("B","C"))]
    ysb_enc = encode_result.coeffs[index.source_idx("B")]
    u_ab_enc = encode_result.coeffs[index.edge_idx(("A","B"))]
    print(f"  U_B_C coeff : {u_bc_enc:.4f}  (expected 0.0000 — removed from LHS)")
    print(f"  Y_S_B coeff : {ysb_enc:.4f}  (expected +1.0000 — added to LHS)")
    print(f"  U_A_B coeff : {u_ab_enc:.4f}  (expected +1.0000 — incoming edge)")
    assert abs(u_bc_enc) < 1e-6, "U_B_C should be zeroed after encode sub"
    assert abs(ysb_enc - 1.0) < 1e-6, "Y_S_B should be +1.0"
    assert abs(u_ab_enc - 1.0) < 1e-6, "U_A_B should be +1.0"
    print("  Encode PASS\n")

    sep()
    print("ALL VERIFICATION TESTS PASSED")
    print("Inequality arithmetic is mathematically correct.")
    return True


if __name__ == "__main__":
    ok = run_verification()
    if not ok:
        print("\nVERIFICATION FAILED")
        sys.exit(1)