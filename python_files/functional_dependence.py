"""
Functional Dependence Constraints Beyond the Partition Bound.

Two families of constraints that the partition-bound proof does NOT use,
but which the paper identifies as the route to tighter bounds (Section VI,
Hu's network).

==========================================================================
FAMILY 1: CRYPTO INEQUALITY  (paper eq. 32)
==========================================================================

For a cut {V', V'^c} of the bidirected graph, if the directed cut-set
cs(V', V'^c) separates session i, then:

    h(Y_{sessions separated by cs}, U_{V' ↔ V'^c}) ≤ h(U_{V' ↔ V'^c})

Equivalently (moving U to the left):

    h(Y_{sessions separated by cs}) ≤ 0   (given U across cut)

But in inequality form for the proof:

    h(Y_{sep_i}, U_{cut}) ≤ h(U_{cut})

⟹  h(Y_{sep_i}) ≤ 0   (i.e., the sessions CAN be decoded from U_{cut})

The useful form for bounding is: the crypto inequality adds extra h(Yᵢ)
terms to the LHS of an accumulated inequality when the corresponding
cut-set of edges appears in the RHS.  In Hu's network proof (Prop. 8),
this replaces:

    Σ h(U_{Pi↔Pj})  by  h(Y₃) + h(Y₃, U_{cut}) + ...  (eq. 91)

giving a tighter RHS → LHS collapse.

Concretely, the action CRYPTO(cut_edges, session_idx) does:

  Given current accumulated inequality with RHS edge set E_rhs, if
  all edges of cut_edges ⊆ E_rhs, then we may ADD h(Yᵢ) to the LHS
  (since the crypto inequality says h(Yᵢ | U_{cut}) = 0, meaning
  h(Yᵢ) ≤ h(U_{cut}) ≤ Σ h(U_e) for e ∈ cut_edges ⊆ E_rhs).

  Net effect: LHS coefficient of Y_I increases by (1/|I|) when we
  eventually normalise, making the bound TIGHTER.

==========================================================================
FAMILY 2: DECODING FUNCTIONAL DEPENDENCE  (paper eq. 17)
==========================================================================

For each session i with sink t(i), the decoding constraint gives:

    h(Yᵢ | {U_e : head(e)=t(i)}) = 0

i.e.  h(Yᵢ) ≤ h({U_e : edges into t(i)})

This allows us to substitute h(Yᵢ) ≤ Σ_{e into t(i)} h(U_e) in any
inequality where those edge entropies appear.

Action DECODE_SUB(session_idx): if the RHS of the current inequality
already has the incoming edges of t(session_idx), add h(Yᵢ) to the
LHS (borrowing from those edge terms).

==========================================================================
FAMILY 3: ENCODING FUNCTIONAL DEPENDENCE  (paper eq. 16)
==========================================================================

For each edge e=(u,v), encoding gives:

    h(U_e | Y_{S(u)}, {U_{e'} : head(e')=u}) = 0

i.e.  h(U_e) ≤ h(Y_{S(u)}) + Σ_{e' into u} h(U_{e'})

This allows substitution of h(U_e) on the LHS by source+incoming-edge
terms — useful when combining two inequalities that share an internal
edge.

==========================================================================
RL INTEGRATION
==========================================================================

Each action type is:

  CryptoAction(cut_partition_V_prime, session_idx)
    → ADD the standalone crypto inequality to the current inequality
      (both sides): +n_sep/|I| on Y_I, −1 on every cut edge.

  DecodeSubAction(session_idx)
    → ADD the standalone decode inequality to the current inequality
      (both sides): +1/|I| on Y_I, −1 on every edge incident to t(i).

==========================================================================
SOUNDNESS NOTE  (2026-08-02) — READ BEFORE TOUCHING THESE FUNCTIONS
==========================================================================

Both operations used to increment coeffs[yi_idx()] and leave the edge
(RHS) coefficients untouched.  That is UNSOUND: it tightens the extracted
bound monotonically at zero cost with no cap.  Applied in a loop to
okamura_4N's 3.200·h(Y_I) ≤ 24.83 with V'={a,d} it walks the bound down
through 1.9706 / 1.5917 / 1.3349 / 1.1495 / 1.0093 and keeps going to
0.1310 by round 60 — far below the LP lower bound of 1.0.  Every bound
ever produced with the old code is unproved.

The correct operation is ADDITION of a standalone valid inequality:

    crypto:  (n_sep/|I|)·h(Y_I) ≤ Σ_{e ∈ δ(V')} h(U_e)
    decode:  (1/|I|)·h(Y_I)     ≤ Σ_{e ∋ t(i)}  h(U_e)

Adding a valid inequality to a valid inequality yields a valid
inequality.  Because extract_bound reads numerator = Σ|edge coeffs| and
denominator = c1·|I| + …, the result is the MEDIANT of the two operand
bounds and therefore always lies between them.  The operation can still
tighten (when the standalone bound is below the current one) but it can
no longer tighten for free, and it cannot run away.
"""

from typing import List, Tuple, Set, Optional, Dict, FrozenSet
from fixed_inequality import Inequality, EntropyIndex


# ─────────────────────────────────────────────────────────────────────────────
# Soundness-check strictness (see assert_derivation_sound)
# ─────────────────────────────────────────────────────────────────────────────
#
# "strict"  — the terminal bound must be >= the standalone bound of EVERY
#             crypto/decode inequality used in its derivation.
# "mediant" — the terminal bound must be >= the MINIMUM standalone bound
#             over the crypto/decode components.
#
# DEFAULT IS "mediant", NOT "strict".  The audit asked for strict, but
# strict is demonstrably NOT a theorem — it fires on sound derivations.
# Concrete counterexample, reproducible from verify_math.py's 3-node
# network (nodes A,B,C; edges AB,AC,BC; sessions A→B, B→C):
#
#   base terminal inequality  1.000·h(Y_I) ≤ 3 edges     → bound 1.5
#   crypto inequality, V'={A}, 2 cut edges, 1 separated  → bound 2.0
#
#   adding crypto repeatedly gives the MEDIANT each time:
#     round 1  yi=1.5  edges=5   bound 1.6667
#     round 2  yi=2.0  edges=7   bound 1.7500
#     ...
#     round 60                   bound 1.9839
#
# The bound rises monotonically toward 2.0 and never reaches it — the
# runaway is gone, which is the whole point of the fix — but it sits
# BELOW the crypto floor of 2.0 the entire time, because the base
# inequality it is being added to is tighter (1.5) than the crypto
# inequality itself.  Strict mode would reject every one of those
# perfectly valid inequalities.
#
# Addition guarantees exactly this and no more: the sum of inequalities
# with bounds b₁…b_k has bound in [min bᵢ, max bᵢ].  So >= the minimum is
# the real invariant, and it is what catches the actual bug — the old
# code's unbounded descent to 0.1310 violates it immediately.
SOUNDNESS_MODE: str = "mediant"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _directed_cut_edges(
    V_prime: Set[str],
    nodes: List[str],
    edges: List[Tuple[str, str]]
) -> List[Tuple[str, str]]:
    """
    Directed cut-set cs(V', V'^c):
    All directed edges (u,v) where u ∈ V' and v ∉ V', OR u ∉ V' and v ∈ V'.
    For undirected networks, both directions of each boundary edge count.
    """
    V_c = set(nodes) - V_prime
    cut = []
    for (u, v) in edges:
        if (u in V_prime and v in V_c) or (u in V_c and v in V_prime):
            cut.append((u, v))
    return cut


def _sessions_separated_by_cut(
    V_prime: Set[str],
    nodes: List[str],
    edges: List[Tuple[str, str]],
    sessions: List[Tuple[str, str]]
) -> List[int]:
    """
    A session i is separated by cut {V', V'^c} if EVERY path from s(i) to
    t(i) crosses the cut.  For the cut to separate i, one of (s(i), t(i))
    must be in V' and the other in V'^c.

    NOTE: This is the sufficient (not necessary) condition for simple cuts.
    For the full condition we would need to check all paths.  For the
    purposes of the crypto inequality on simple partitions, the condition
    s(i) ∈ V', t(i) ∉ V' (or vice versa) is sufficient.
    """
    V_c = set(nodes) - V_prime
    separated = []
    for si, (s, t) in enumerate(sessions):
        if (s in V_prime and t in V_c) or (s in V_c and t in V_prime):
            separated.append(si)
    return separated


def _incoming_edges(
    node: str,
    edges: List[Tuple[str, str]]
) -> List[Tuple[str, str]]:
    """All edges (u, node) — incoming to node in the undirected sense."""
    return [(u, v) for (u, v) in edges if v == node or u == node]


def _edges_into_sink(
    session_idx: int,
    sessions: List[Tuple[str, str]],
    edges: List[Tuple[str, str]]
) -> List[Tuple[str, str]]:
    """Edges incident to the sink of session_idx."""
    _, t = sessions[session_idx]
    return _incoming_edges(t, edges)


# ─────────────────────────────────────────────────────────────────────────────
# Crypto Inequality Application
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_edge_indices(
    edge_list: List[Tuple[str, str]],
    index: EntropyIndex
) -> Optional[List[int]]:
    """
    Map each edge to its coefficient slot, trying U_{u}_{v} then U_{v}_{u}.

    Returns None if any edge has no slot in the index — the caller must
    then refuse to apply, since it cannot write the RHS term it owes.
    """
    out: List[int] = []
    for (u, v) in edge_list:
        key = f"U_{u}_{v}"
        if key not in index.var_to_idx:
            key = f"U_{v}_{u}"
        if key not in index.var_to_idx:
            return None
        out.append(index.var_to_idx[key])
    return out


def crypto_standalone_bound(
    V_prime: Set[str],
    nodes: List[str],
    edges: List[Tuple[str, str]],
    sessions: List[Tuple[str, str]]
) -> Optional[float]:
    """
    Bound implied by the crypto inequality for this cut ON ITS OWN:

        (n_sep/|I|)·h(Y_I) ≤ Σ_{e ∈ δ(V')} h(U_e)
        ⟹  n_sep·r ≤ |δ(V')|
        ⟹  r ≤ |δ(V')| / n_sep

    Returns None when the cut separates nothing (no inequality to state).
    Used by assert_derivation_sound as the floor this operation can
    contribute; a derivation using it can never legitimately drop below
    the minimum such floor.
    """
    cut = _directed_cut_edges(V_prime, nodes, edges)
    sep = _sessions_separated_by_cut(V_prime, nodes, edges, sessions)
    if not cut or not sep:
        return None
    return len(cut) / len(sep)


def decode_standalone_bound(
    session_idx: int,
    sessions: List[Tuple[str, str]],
    edges: List[Tuple[str, str]]
) -> Optional[float]:
    """
    Bound implied by the decode inequality for this session ON ITS OWN:

        (1/|I|)·h(Y_I) ≤ Σ_{e ∋ t(i)} h(U_e)
        ⟹  r ≤ deg(t(i))
    """
    incident = _edges_into_sink(session_idx, sessions, edges)
    if not incident:
        return None
    return float(len(incident))


def apply_crypto_inequality_direct(
    ineq: Inequality,
    V_prime: Set[str],
    nodes: List[str],
    edges: List[Tuple[str, str]],
    sessions: List[Tuple[str, str]],
    index: EntropyIndex,
    tol: float = 1e-9,
    internal_per_part: Optional[List[int]] = None,
) -> Tuple[Inequality, bool]:
    """
    Add the crypto inequality for cut {V', V'^c} to *ineq*.

    The crypto inequality (paper eq. 32) is a STANDALONE valid inequality:

        h(Y_{sep} | U_{δ(V')}) = 0
        ⟹  h(Y_{sep}) ≤ h(U_{δ(V')}) ≤ Σ_{e ∈ δ(V')} h(U_e)

    With h(Y_{sep}) = n_sep·r and h(Y_I) = |I|·r this is

        (n_sep/|I|)·h(Y_I) ≤ Σ_{e ∈ δ(V')} h(U_e)

    which in this module's sign convention (positive coeff = LHS,
    negative = RHS) is the coefficient vector

        coeffs[yi_idx()]    = +n_sep/|I|
        coeffs[edge_idx(e)] = −1.0   for each e ∈ δ(V')   (unit capacity)

    We ADD that vector to *ineq*.  Both sides move.  The sum of two valid
    inequalities is valid, so the result is proved outright — no
    precondition on the shape of *ineq* is needed for soundness.

    WHAT CHANGED AND WHY (2026-08-02):
      The old body did `coeffs[yi_idx()] += n_sep/n_sessions` and never
      touched the edge coefficients.  That is not addition of the crypto
      inequality; it is a free tightening of the denominator with the
      numerator held fixed.  Iterated, it drives the extracted bound to
      zero.  See the SOUNDNESS NOTE at the top of this module.

      Removed with it (1c): the `coeffs[source_idx(v)] < -tol` guard that
      refused to apply to per-node IOs with uncancelled source terms.
      That guard was patching a symptom of the broken accounting — under
      correct addition the operation is valid on ANY inequality, source
      terms present or not, because the added vector touches neither the
      Y_S_v nor the Y_ST_Pk slots.

      Removed with it (1d): there is no β_e ≥ 1 coefficient floor.  Under
      addition nothing is spent out of an existing budget, so no such
      precondition is required.

    RETAINED precondition: every cut edge must already carry a negative
    (RHS) coefficient.  This is NOT needed for validity — it is the
    action's gating semantics, kept so the action space is unchanged.

    Returns (new_ineq, was_applied).
    """
    cut_edges = _directed_cut_edges(V_prime, nodes, edges)
    if not cut_edges:
        return ineq.copy(), False

    cut_idx = _resolve_edge_indices(cut_edges, index)
    if cut_idx is None:
        return ineq.copy(), False

    # Gating precondition (semantics, not soundness): cut edges on RHS.
    for j in cut_idx:
        if ineq.coeffs[j] >= -tol:
            return ineq.copy(), False

    separated = _sessions_separated_by_cut(V_prime, nodes, edges, sessions)
    if not separated:
        return ineq.copy(), False

    n_sep      = len(separated)
    n_sessions = len(sessions)
    if n_sessions == 0:
        return ineq.copy(), False

    result = ineq.copy()
    # LHS: (n_sep/|I|)·h(Y_I)
    result.coeffs[index.yi_idx()] += n_sep / n_sessions
    # RHS: Σ_{e ∈ δ(V')} h(U_e), unit capacity each
    for j in cut_idx:
        result.coeffs[j] -= 1.0

    detail = {
        "V_prime":          sorted(V_prime),
        "cut_edges":        [list(e) for e in cut_edges],
        "cut_size":         len(cut_edges),
        "separated":        list(separated),
        "n_sep":            n_sep,
        "yi_delta":         n_sep / n_sessions,
        "standalone_bound": len(cut_edges) / n_sep,
    }
    _assert_rhs_changed(ineq, result, index, "CRYPTO", detail)
    _assert_mediant_step(ineq, result, detail["standalone_bound"],
                         n_sessions, internal_per_part, "CRYPTO", detail)
    result.record_op(op="CRYPTO", detail=detail)
    return result, True


# ─────────────────────────────────────────────────────────────────────────────
# Decoding Functional Dependence
# ─────────────────────────────────────────────────────────────────────────────

def apply_decode_substitution(
    ineq: Inequality,
    session_idx: int,
    sessions: List[Tuple[str, str]],
    edges: List[Tuple[str, str]],
    index: EntropyIndex,
    tol: float = 1e-9,
    internal_per_part: Optional[List[int]] = None,
) -> Tuple[Inequality, bool]:
    """
    Add the decoding functional-dependence inequality for *session_idx*.

    Decoding gives h(Yᵢ | {U_e : e incident to t(i)}) = 0, hence the
    STANDALONE valid inequality

        h(Yᵢ) ≤ Σ_{e ∋ t(i)} h(U_e)
        ⟹  (1/|I|)·h(Y_I) ≤ Σ_{e ∋ t(i)} h(U_e)

    whose coefficient vector is

        coeffs[yi_idx()]    = +1/|I|
        coeffs[edge_idx(e)] = −1.0   for each e incident to t(i)

    We ADD that vector to *ineq*.  Identical in structure to
    apply_crypto_inequality_direct with n_sep → 1 and δ(V') → the edges
    incident to t(session_idx); see that function's docstring for the full
    rationale, including why the source-term guard (1c) and the
    coefficient floor (1d) are both gone.

    WHAT CHANGED AND WHY (2026-08-02):
      The old body did `coeffs[yi_idx()] += 1.0/n_sessions` and never
      touched the edge coefficients — the same unsound free tightening
      documented in the module header.

    Returns (new_ineq, was_applied).
    """
    n_sessions = len(sessions)
    if n_sessions == 0:
        return ineq.copy(), False

    incident = _edges_into_sink(session_idx, sessions, edges)
    if not incident:
        return ineq.copy(), False

    inc_idx = _resolve_edge_indices(incident, index)
    if inc_idx is None:
        return ineq.copy(), False

    # Gating precondition (semantics, not soundness): edges on RHS.
    for j in inc_idx:
        if ineq.coeffs[j] >= -tol:
            return ineq.copy(), False

    result = ineq.copy()
    # LHS: (1/|I|)·h(Y_I) = h(Y_i)
    result.coeffs[index.yi_idx()] += 1.0 / n_sessions
    # RHS: Σ_{e ∋ t(i)} h(U_e), unit capacity each
    for j in inc_idx:
        result.coeffs[j] -= 1.0

    detail = {
        "session_idx":      session_idx,
        "sink":             sessions[session_idx][1],
        "incident_edges":   [list(e) for e in incident],
        "sink_degree":      len(incident),
        "yi_delta":         1.0 / n_sessions,
        "standalone_bound": float(len(incident)),
    }
    _assert_rhs_changed(ineq, result, index, "DECODE", detail)
    _assert_mediant_step(ineq, result, detail["standalone_bound"],
                         n_sessions, internal_per_part, "DECODE", detail)
    result.record_op(op="DECODE", detail=detail)
    return result, True


# ─────────────────────────────────────────────────────────────────────────────
# Encoding Functional Dependence — DELETED 2026-08-02
# ─────────────────────────────────────────────────────────────────────────────
#
# apply_encode_substitution() was removed.  It fired on a POSITIVE (LHS)
# coefficient of h(U_e) and replaced that term with the encoding UPPER
# bound h(Y_{S(u)}) + Σ_{e'→u} h(U_{e'}).  Substituting an upper bound for
# a term on the LHS is invalid: it makes the LHS larger, so the resulting
# inequality does not follow from the original.  (The substitution is only
# legitimate on the RHS, where enlarging a term weakens the inequality.)
#
# It had exactly one call site, the self-test in verify_math.py, which has
# been updated to assert the function is gone rather than exercise it.  No
# training, environment, or evaluation path ever called it, so no bound in
# any proof log depends on this operator.
#
# FAMILY 3 (paper eq. 16) is therefore currently unimplemented.  If it is
# reinstated it must be written as ADDITION of the standalone inequality
#     h(U_e) ≤ h(Y_{S(u)}) + Σ_{e' → u} h(U_{e'})
# in the same style as the crypto/decode operators above.


# ─────────────────────────────────────────────────────────────────────────────
# Soundness assertion (1f)
# ─────────────────────────────────────────────────────────────────────────────

class UnsoundDerivationError(AssertionError):
    """Raised when a terminal bound falls below what its derivation can prove."""


class MediantViolationError(UnsoundDerivationError):
    """Raised when one addition step leaves the [min, max] mediant interval."""


class RHSUnchangedError(UnsoundDerivationError):
    """Raised when an applied crypto/decode step did not move the RHS."""


# Every applied crypto/decode step is checked by BOTH assertions below.
# They catch different things and neither subsumes the other:
#
#   _assert_rhs_changed   — structural. The exact defect being fixed was
#       "increment Y_I, never touch the RHS". This fires on that single
#       step, unconditionally, regardless of numbers.
#
#   _assert_mediant_step  — numeric, two-sided. Addition of inequalities
#       with bounds b_before and b_addend must yield the mediant
#           (E_before + E_add) / (D_before + D_add)
#       which lies in [min(b_before, b_addend), max(b_before, b_addend)].
#       A one-sided ">= min" check would pass on a step that tightened
#       for free below b_addend but stayed above b_before; the upper
#       bound closes that hole.
#
# The mediant check alone can pass on an individual buggy step (a free
# tightening that happens to land inside the interval), which is why the
# structural check is not redundant.

def _assert_rhs_changed(before, after, index, op, detail):
    """The RHS coefficient vector must move. Byte-identical means the bug."""
    edge_slots = [index.edge_idx(e) for e in index.edges]
    if all(before.coeffs[j] == after.coeffs[j] for j in edge_slots):
        raise RHSUnchangedError(
            f"{op} reported applied=True but left the RHS byte-identical.\n"
            f"  This is precisely the defect the 2026-08-02 fix removed: "
            f"incrementing Y_I without adding the addend's edge terms is a "
            f"free tightening with no cap.\n"
            f"  detail: {detail}"
        )


def _assert_mediant_step(before, after, addend_bound, n_sessions,
                         internal_per_part, op, detail, tol=1e-6):
    """
    Two-sided mediant invariant for one addition step:

        min(b_before, b_addend) <= b_after <= max(b_before, b_addend)

    Skipped when either operand has a non-positive denominator (in which
    case extract_bound returns inf and there is no finite bound to
    compare against).
    """
    if internal_per_part is None:
        return
    b_before = before.extract_bound(n_sessions, 0, internal_per_part)
    b_after  = after.extract_bound(n_sessions, 0, internal_per_part)
    if not (b_before < float('inf') and b_after < float('inf')):
        return
    if addend_bound is None:
        return
    lo, hi = min(b_before, addend_bound), max(b_before, addend_bound)
    if not (lo - tol <= b_after <= hi + tol):
        raise MediantViolationError(
            f"{op} left the mediant interval.\n"
            f"  b_before = {b_before:.6f}\n"
            f"  b_addend = {addend_bound:.6f}\n"
            f"  b_after  = {b_after:.6f}   (must be in "
            f"[{lo:.6f}, {hi:.6f}])\n"
            f"  detail: {detail}"
        )


def assert_derivation_sound(
    ineq: Inequality,
    bound: float,
    graph_name: str,
    episode=None,
    step=None,
    tol: float = 1e-6,
) -> None:
    """
    Check a terminal bound against the crypto/decode inequalities used to
    derive it, and raise UnsoundDerivationError if it is too low.

    Two modes, selected by the module-level SOUNDNESS_MODE:

    "mediant" (default — the invariant addition actually guarantees)
        Summing inequalities produces the mediant of their bounds:
            (n₁+n₂)/(d₁+d₂) lies between n₁/d₁ and n₂/d₂.
        So the sum is >= the MINIMUM component bound, never below it.
        It may legitimately sit below an individual component's bound
        when another component is tighter.

    "strict"  (what the audit asked for; NOT a theorem)
        bound must be >= the standalone bound of EVERY crypto/decode
        inequality in the trace.  See the SOUNDNESS_MODE comment at the
        top of this module for a worked counterexample where strict mode
        rejects a valid inequality.  Available for diagnosis; do not
        leave it on for a production re-derivation.

    Either way a violation means the bound is not proved.  This raises —
    louder than the old behaviour, which silently clamped the value to PB
    (see the LP-clamp sites in fixed_training.py) and then reported zero
    violations.  Nothing is filtered and nothing is dropped.
    """
    floors = [
        (entry["op"], entry["detail"]["standalone_bound"], entry["detail"])
        for entry in getattr(ineq, "op_trace", [])
        if entry.get("op") in ("CRYPTO", "DECODE")
        and entry.get("detail", {}).get("standalone_bound") is not None
    ]
    if not floors:
        return

    if SOUNDNESS_MODE == "mediant":
        worst = min(f[1] for f in floors)
        violated = [f for f in floors if bound < worst - tol]
    else:
        violated = [f for f in floors if bound < f[1] - tol]

    if not violated:
        return

    lines = [
        f"UNSOUND DERIVATION on {graph_name} "
        f"(episode={episode}, step={step}, mode={SOUNDNESS_MODE})",
        f"  extracted bound : {bound:.6f}",
        f"  terminal ineq   : {ineq!r}",
        f"  op trace        : {len(getattr(ineq, 'op_trace', []))} recorded operations",
    ]
    for op, floor, detail in violated:
        lines.append(f"  {op} standalone bound {floor:.6f} > extracted {bound:.6f}")
        lines.append(f"      {detail}")
    raise UnsoundDerivationError("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Enumerate All Valid Crypto Actions for a Network
# ─────────────────────────────────────────────────────────────────────────────

def enumerate_crypto_cuts(
    nodes: List[str],
    edges: List[Tuple[str, str]],
    sessions: List[Tuple[str, str]]
) -> List[Tuple[FrozenSet[str], List[int]]]:
    """
    Enumerate all non-trivial cuts that separate at least one session.
    Returns list of (V_prime_frozenset, separated_session_indices).

    For efficiency, limits to cuts of size ≤ |nodes|//2 (larger cuts are
    complements of smaller ones and give the same constraint).
    """
    n = len(nodes)
    useful_cuts = []
    # Only try cuts up to 2^(n-1) to avoid duplicates
    for mask in range(1, 1 << (n - 1)):
        V_prime = frozenset(nodes[i] for i in range(n) if mask & (1 << i))
        if not V_prime:
            continue
        separated = _sessions_separated_by_cut(V_prime, nodes, edges, sessions)
        if separated:
            useful_cuts.append((V_prime, separated))

    # Deduplicate by (frozenset, frozenset(separated))
    seen = set()
    unique = []
    for (vp, sep) in useful_cuts:
        key = (vp, frozenset(sep))
        if key not in seen:
            seen.add(key)
            unique.append((vp, sep))

    return unique


def best_crypto_cuts(
    nodes: List[str],
    edges: List[Tuple[str, str]],
    sessions: List[Tuple[str, str]],
    max_cuts: int = 20
) -> List[Tuple[FrozenSet[str], List[int]]]:
    """
    Return at most max_cuts cuts, prioritised by number of separated sessions
    (more separations = better bound improvement potential).
    """
    cuts = enumerate_crypto_cuts(nodes, edges, sessions)
    cuts.sort(key=lambda x: len(x[1]), reverse=True)
    return cuts[:max_cuts]