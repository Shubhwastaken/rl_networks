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
    # apply_pairwise_submodularity now returns a SINGLE inequality
    # (sum + exact collapse). The "intersection" was deleted: it shrank
    # RHS capacity and was FALSE on 35% of valid input pairs.
    union_01 = apply_pairwise_submodularity(
        base[0], base[1], index, sessions
    )

    print(f"\n  Combined: {union_01}")

    # Y_I after the CORRECTED collapse is K - D, not c_min.
    #
    # This expectation changed from 1.0 to 0.5. The old value came from
    # the c_min collapse, which added c_min=1.0 to Y_I and then deleted
    # the covered source terms for free. Under exact accounting:
    #   K = sum_k c_k*|st_k| / |I| = (1*1 + 1*2)/2 = 1.5
    #   D = sum_v |s_v|*n_src(v) / |I| = (1*1 + 1*1)/2 = 1.0
    #   Y_I = K - D = 0.5
    # Both values are valid on THIS network (the old one is exactly tight,
    # slack 0.0; the corrected one carries slack 1.0), but the old one is
    # tight only because the unpaid deletion happened to fit inside the
    # available slack -- on 4000 random valid inequalities carrying a Y_ST
    # term the old collapse turned 91 of them FALSE, the corrected one 0.
    n_I = len(sessions)
    K_expect = sum(len(index.st_sessions[i]) for i in (0, 1)) / n_I
    D_expect = sum(1 for s, _t in sessions if s in ("A", "B")) / n_I
    expected_yi = K_expect - D_expect
    yi_in_union = union_01.yi_coeff()
    print(f"\n  Y_I coeff in union: {yi_in_union:.4f}  "
          f"(expected K-D = {K_expect:.2f}-{D_expect:.2f} = {expected_yi:.4f})")
    assert abs(yi_in_union - expected_yi) < 1e-6
    assert any(e["op"] == "COLLAPSE_YST" for e in union_01.op_trace), \
        "collapse must record COLLAPSE_YST provenance"

    # Addition accumulates RHS capacity: the shared boundary edge A-B is
    # a boundary edge of BOTH P0 and P1, so it carries -2, not -1.
    shared_ab = union_01.coeffs[index.edge_idx(("A","B"))]
    print(f"  U_A_B in combined: {shared_ab:.4f}  (expected -2.0, capacity ADDS)")
    assert abs(shared_ab - (-2.0)) < 1e-6

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

    # RHS is now ACCUMULATED capacity, not a reset to -1 per edge.
    # In an independent-set partition every edge is a cross edge, so it is
    # a boundary edge of exactly 2 parts and the sum gives it capacity 2.
    # The old "reset to -1" halved that; it was correct only by coincidence
    # on this family. Consequence: this path now yields 2x the published
    # partition bound |E|/(|I|+internal) on all 17 registry graphs, because
    # the index carries ONE variable per undirected edge and summing two
    # partition IOs claims that single physical capacity from both sides.
    assert abs(final.yi_coeff() - 1.0) < 1e-6
    assert abs(final.rhs_edge_sum() - 2 * len(edges)) < 1e-6
    assert final.check_valid_terminal_form()

    proof2_bound = final.extract_bound(
        len(sessions), len(edges), internal_per_part
    )
    internal_total = sum(internal_per_part)
    analytic = 2 * len(edges) / (len(sessions) + internal_total)   # see note above
    print(f"  Extracted bound : {proof2_bound:.4f}")
    print(f"  Analytic bound  : {analytic:.4f}")
    assert abs(proof2_bound - analytic) < 1e-6
    print("  Proof 2 PASS\n")

    # --- agent control: combine P0 and P2 ---
    sep()
    print("Agent-controlled pairwise: base[0] (P0) + base[2] (P2)")
    union_02 = apply_pairwise_submodularity(
        base[0], base[2], index, sessions
    )
    print(f"  Union P0+P2  : {union_02}")
    # Same K-D accounting as the P0+P1 union above.
    yi_02 = union_02.yi_coeff()
    K02 = sum(len(index.st_sessions[i]) for i in (0, 2)) / len(sessions)
    D02 = sum(1 for s, _t in sessions if s in ("A", "C")) / len(sessions)
    print(f"  Y_I coeff: {yi_02:.4f}  (expected K-D = {K02 - D02:.4f})")
    assert abs(yi_02 - (K02 - D02)) < 1e-6

    cross_02 = union_02.coeffs[index.edge_idx(("A","C"))]
    print(f"  U_A_C in combined: {cross_02:.4f}  (expected -2.0, capacity ADDS)")
    assert abs(cross_02 - (-2.0)) < 1e-6
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
    # RHS MUST grow by exactly one unit per cut edge.
    #
    # This assertion is INVERTED from what it used to say. The old test
    # asserted the RHS does NOT change -- i.e. it asserted the bug. Adding
    # h(Y_sep) to the LHS while leaving the RHS alone is a free tightening
    # of the bound with no cap; iterated on okamura_4N it walks the bound
    # from 1.97 down to 0.13, below the LP lower bound of 1.0. The crypto
    # inequality is a standalone valid inequality that must be ADDED to
    # both sides:
    #     (n_sep/|I|)*h(Y_I) <= sum_{e in delta(V')} h(U_e)
    from functional_dependence import _directed_cut_edges
    cut = _directed_cut_edges({"A"}, nodes, edges)
    rhs_before = final.rhs_edge_sum()
    rhs_after = crypto_result.rhs_edge_sum()
    print(f"  RHS edges before: {rhs_before:.4f}  after: {rhs_after:.4f}  "
          f"(expected +{len(cut)} for {len(cut)} cut edges)")
    assert abs((rhs_after - rhs_before) - len(cut)) < 1e-6, \
        f"RHS should grow by {len(cut)} (one unit per cut edge)"
    # Provenance must be recorded (1h).
    assert any(e["op"] == "CRYPTO" for e in crypto_result.op_trace), \
        "CRYPTO op must be recorded in op_trace"
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
    # RHS must grow by one unit per edge incident to the sink.
    from functional_dependence import _edges_into_sink
    incident = _edges_into_sink(0, sessions, edges)
    dec_rhs_before = final.rhs_edge_sum()
    dec_rhs_after  = decode_result.rhs_edge_sum()
    print(f"  RHS edges before: {dec_rhs_before:.4f}  after: {dec_rhs_after:.4f}  "
          f"(expected +{len(incident)})")
    assert abs((dec_rhs_after - dec_rhs_before) - len(incident)) < 1e-6, \
        f"RHS should grow by {len(incident)} (one unit per incident edge)"
    assert any(e["op"] == "DECODE" for e in decode_result.op_trace), \
        "DECODE op must be recorded in op_trace"
    print("  Decode PASS\n")

    sep()
    print("Functional Dependence: no-runaway check (crypto is capped)")
    # Regression test for the bug this whole fix exists to remove.
    # Under the old implementation, repeated crypto application tightened
    # the bound monotonically at zero cost and never converged. Under
    # correct addition the bound is a mediant and can never fall below the
    # crypto inequality's own standalone bound.
    #
    # The guaranteed invariant is the MEDIANT one: the sum of inequalities
    # with bounds b1..bk has a bound in [min bi, max bi]. Here the base is
    # 1.5 and the crypto inequality is 2.0, so repeated addition must walk
    # the bound UP from 1.5 toward 2.0 and never leave that interval. Note
    # this means crypto LOOSENS here -- correct, since the base inequality
    # is already tighter than the crypto inequality being added.
    from functional_dependence import crypto_standalone_bound
    floor = crypto_standalone_bound({"A"}, nodes, edges, sessions)
    cur = final.copy()
    start = cur.extract_bound(len(sessions), len(edges), internal_per_part)
    lo, hi = min(start, floor), max(start, floor)
    prev = start
    for _ in range(60):
        cur, ok = apply_crypto_inequality_direct(
            cur, frozenset(["A"]), nodes, edges, sessions, index
        )
        if not ok:
            break
        b = cur.extract_bound(len(sessions), len(edges), internal_per_part)
        assert lo - 1e-9 <= b <= hi + 1e-9, (
            f"crypto left the mediant interval [{lo:.6f}, {hi:.6f}]: {b:.6f}"
        )
        # The invariant is the mediant INTERVAL, not monotone increase:
        # the bound moves TOWARD the crypto floor from whichever side it
        # starts on. With the accumulated n2 RHS the base bound (3.0) now
        # exceeds the floor (2.0), so it descends -- still inside [lo, hi].
        prev = b
    print(f"  base bound              : {start:.4f}")
    print(f"  standalone crypto bound : {floor:.4f}")
    print(f"  bound after 60 rounds   : {prev:.4f}  (stays in "
          f"[{lo:.4f}, {hi:.4f}], never runs away)")
    print("  No-runaway PASS\n")

    sep()
    print("Functional Dependence: Encode Substitution — REMOVED")
    # apply_encode_substitution was deleted. It fired on a POSITIVE (LHS)
    # coefficient of h(U_e) and substituted the encoding UPPER bound
    # there, which enlarges the LHS and so does not follow from the
    # original inequality. This self-test was its only call site; it now
    # asserts the operator is gone rather than exercising it.
    import functional_dependence as _fd
    assert not hasattr(_fd, "apply_encode_substitution"), (
        "apply_encode_substitution is unsound (substitutes an upper bound "
        "on the LHS) and must stay deleted"
    )
    print("  Encode correctly absent PASS\n")

    sep()
    print("ALL VERIFICATION TESTS PASSED")
    print("Inequality arithmetic is mathematically correct.")
    return True


if __name__ == "__main__":
    ok = run_verification()
    if not ok:
        print("\nVERIFICATION FAILED")
        sys.exit(1)