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
    → if cut separates session AND cut edges ⊆ RHS of current ineq,
      add +1 to Y_I coeff (net: tightens bound denominator)

  DecodeSubAction(session_idx)
    → if incoming edges of t(i) ⊆ RHS,
      add +1 to Y_I coeff from that session

  EncodeSubAction(edge_idx)
    → substitute h(U_e) on LHS by encoding RHS (source + incoming)

All return an (inequality, gain) pair where gain > 0 means the
substitution strictly tightened the bound.
"""

from typing import List, Tuple, Set, Optional, Dict, FrozenSet
from fixed_inequality import Inequality, EntropyIndex


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

def apply_crypto_inequality(
    ineq: Inequality,
    V_prime: Set[str],
    nodes: List[str],
    edges: List[Tuple[str, str]],
    sessions: List[Tuple[str, str]],
    index: EntropyIndex,
    tol: float = 1e-9
) -> Tuple[Inequality, float]:
    """
    Apply the crypto inequality for cut {V', V'^c}.

    The crypto inequality (eq. 32) states:
        h(Y_{sep}, U_{cut}) ≤ h(U_{cut})

    Which means h(Y_{sep}) ≤ h(U_{cut}).

    In terms of bounding: if U_{cut} edges are already on the RHS of our
    accumulated inequality (each with negative coefficient, meaning they
    upper-bound the LHS), we can add h(Yᵢ) to the LHS for each session i
    separated by the cut, because h(Yᵢ) ≤ h(U_{cut}) ≤ Σ_{e∈cut} h(U_e).

    Concretely: add +1 to the Y_I coefficient (these sessions contribute to
    the denominator in r ≤ |E| / (|I| + internal) form).

    Returns (modified_ineq, num_sessions_added) where num_sessions_added > 0
    means a strict tightening is possible.

    Mathematical validity:
      The starting inequality has form:
        LHS_terms ≤ Σ_e c_e * h(U_e)   (after IO + submod)
      The crypto inequality says h(Y_{sep_i}) ≤ Σ_{e ∈ cut} h(U_e).
      Adding to LHS: (LHS_terms + Σᵢ h(Y_{sep_i})) ≤ Σ_e c_e * h(U_e)
        provided Σ_{e ∈ cut} h(U_e) ≤ Σ_e (c_e contribution from cut)
        which holds because cut ⊆ RHS and each h(U_e) ≤ 1 (unit capacity).
    """
    # 1. Find cut edges
    cut_edges = _directed_cut_edges(V_prime, nodes, edges)
    if not cut_edges:
        return ineq.copy(), 0.0

    # 2. Check that all cut edges appear on RHS with negative coefficient
    for e in cut_edges:
        key = f"U_{e[0]}_{e[1]}"
        if key not in index.var_to_idx:
            # Try reverse
            key = f"U_{e[1]}_{e[0]}"
        if key not in index.var_to_idx:
            return ineq.copy(), 0.0
        c = ineq.coeffs[index.var_to_idx[key]]
        if c >= -tol:  # not on RHS (RHS has negative coefficients)
            return ineq.copy(), 0.0

    # 3. Find sessions separated by this cut
    separated = _sessions_separated_by_cut(V_prime, nodes, edges, sessions)
    if not separated:
        return ineq.copy(), 0.0

    # 4. Each separated session adds h(Yᵢ) to LHS.
    #    In terminal form: h(Y_I) coeff increases by len(separated)/len(sessions)
    #    (since h(Y_I) = Σ h(Yᵢ) and each session has equal weight r).
    #    We add fractional contribution to Y_I.
    result = ineq.copy()
    n_sessions = len(sessions)

    # Add to Y_I: each session_i adds r to LHS, equivalent to +1/n to Y_I coeff
    # since h(Y_I) = n*r in terminal form.
    # More precisely: we add h(Y_{sep}) ≤ h(U_{cut}) to the LHS.
    # In normalized form (all sessions equal), this becomes:
    #   (|I| + |sep|) * r ≤ |E| → r ≤ |E| / (|I| + |sep|)
    # So we increase the Y_I coefficient by len(separated) / n_sessions.
    extra_yi = len(separated) / n_sessions
    result.coeffs[index.yi_idx()] += extra_yi

    # Reduce RHS edge capacity by the crypto inequality "cost".
    # The crypto inequality borrows h(U_{cut}) from the RHS, so
    # we subtract (len(separated) / |cut|) from each cut edge coefficient.
    # This keeps the inequality valid: we added len(separated)*r to LHS
    # and subtracted an equivalent amount from RHS.
    extra_per_edge = len(separated) / max(len(cut_edges), 1)
    for e in cut_edges:
        key = f"U_{e[0]}_{e[1]}"
        if key not in index.var_to_idx:
            key = f"U_{e[1]}_{e[0]}"
        if key in index.var_to_idx:
            # Reduce the negative RHS coefficient (make it less negative)
            # This represents "spending" some edge capacity on the crypto term
            result.coeffs[index.var_to_idx[key]] += extra_per_edge  # less negative

    return result, float(len(separated))


def apply_crypto_inequality_direct(
    ineq: Inequality,
    V_prime: Set[str],
    nodes: List[str],
    edges: List[Tuple[str, str]],
    sessions: List[Tuple[str, str]],
    index: EntropyIndex,
    tol: float = 1e-9
) -> Tuple[Inequality, bool]:
    """
    Direct crypto inequality application — Hu's network style (Section VI).

    This is the version from the paper's Proposition 8 proof:
      Given inequality of the form:
        h(Y_I) + h(U_{Pi↔Pj}) + ... ≤ RHS
      The crypto inequality says:
        h(Y_{sep_i} | U_{Pi↔Pj}) = 0
        → h(Y_{sep_i}, U_{Pi↔Pj}) = h(U_{Pi↔Pj})
        → h(U_{Pi↔Pj}) ≥ h(Y_{sep_i})

    This replaces h(U_{Pi↔Pj}) in the RHS sum:
        Σ h(U_{Pi↔Pj}) ≥ h(Y_3) + h(Y_3, U_{P1↔P23}) + h(Y_3, U_{P2↔P3})
        [via submodularity, eq. 91 in paper]

    Net effect: adds +1 to Y_I coefficient (tighter denominator).

    Returns (new_ineq, was_applied).
    The 'was_applied' flag is True if the crypto term could be added.
    """
    cut_edges = _directed_cut_edges(V_prime, nodes, edges)
    if not cut_edges:
        return ineq.copy(), False

    # Check cut edges are all on RHS
    cut_rhs_total = 0.0
    for e in cut_edges:
        key = f"U_{e[0]}_{e[1]}"
        if key not in index.var_to_idx:
            key = f"U_{e[1]}_{e[0]}"
        if key not in index.var_to_idx:
            return ineq.copy(), False
        c = ineq.coeffs[index.var_to_idx[key]]
        if c >= -tol:
            return ineq.copy(), False
        cut_rhs_total += abs(c)

    separated = _sessions_separated_by_cut(V_prime, nodes, edges, sessions)
    if not separated:
        return ineq.copy(), False

    # The paper's technique: for each separated session i, crypto inequality gives
    #   h(Y_i | U_cut) = 0  →  h(Y_i) can be added to LHS
    # This is valid as long as h(U_cut) ≥ h(Y_i), which follows from the
    # functional dependence constraint at t(i).
    # In the terminal inequality, Y_I represents h(Y_I) = |I|*r.
    # Adding h(Y_i) = r for each separated session adds n_sep * r to LHS.
    # Since Y_I coeff c1 satisfies c1 * h(Y_I) = c1 * |I| * r,
    # adding n_sep * r is equivalent to adding n_sep / |I| to c1.
    result = ineq.copy()
    n_sep = len(separated)
    n_sessions = len(sessions)

    result.coeffs[index.yi_idx()] += n_sep / n_sessions
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
    tol: float = 1e-9
) -> Tuple[Inequality, bool]:
    """
    Decoding functional dependence: h(Yᵢ | {U_e : e incident to t(i)}) = 0

    This means h(Yᵢ) ≤ Σ_{e incident to t(i)} h(U_e).

    Application: if all edges incident to t(session_idx) are on the RHS of
    `ineq` with negative coefficients, we can add h(Yᵢ) = r to the LHS,
    tightening the bound.

    The precise rule:
      If Σ_{e ∈ in(t(i))} h(U_e) ≥ h(Yᵢ) = r, and these edges are on RHS:
        LHS + r ≤ Σ_{e ∈ in(t(i))} h(U_e) + rest_of_RHS  (still valid)
      Equivalent: add +1/|I| to Y_I coefficient (for symmetric rate).

    Returns (new_ineq, was_applied).
    """
    _, t = sessions[session_idx]
    incident = _incoming_edges(t, edges)
    if not incident:
        return ineq.copy(), False

    # Check all incident edges are on RHS
    rhs_capacity = 0.0
    for e in incident:
        key = f"U_{e[0]}_{e[1]}"
        if key not in index.var_to_idx:
            key = f"U_{e[1]}_{e[0]}"
        if key not in index.var_to_idx:
            return ineq.copy(), False
        c = ineq.coeffs[index.var_to_idx[key]]
        if c >= -tol:
            return ineq.copy(), False
        rhs_capacity += abs(c)

    # The decode constraint gives h(Yᵢ) ≤ h(edges into t(i)).
    # We add +1 to Y_I (representing one extra r on the LHS) and
    # do NOT reduce RHS — h(Yᵢ) ≤ capacity ≤ rhs_capacity which is
    # already accounted for by the edge bounds.
    n_sessions = len(sessions)
    result = ineq.copy()
    result.coeffs[index.yi_idx()] += 1.0 / n_sessions   # add r/|I| = h(Yi)/|I|

    return result, True


# ─────────────────────────────────────────────────────────────────────────────
# Encoding Functional Dependence
# ─────────────────────────────────────────────────────────────────────────────

def apply_encode_substitution(
    ineq: Inequality,
    edge: Tuple[str, str],
    partition: List[List[str]],
    nodes: List[str],
    edges: List[Tuple[str, str]],
    sessions: List[Tuple[str, str]],
    index: EntropyIndex,
    adjacency: Dict[str, List[str]],
    tol: float = 1e-9
) -> Tuple[Inequality, bool]:
    """
    Encoding functional dependence: h(U_e) ≤ h(Y_{S(tail(e))}) + Σ h(U_{e'→tail(e)})

    Application: if h(U_e) appears on the LHS of `ineq` (positive coefficient),
    replace it by the encoding upper bound, potentially introducing source and
    incoming-edge terms that cancel with other parts of the inequality.

    This is useful when combining two IOs that share an internal edge: the
    internal edge appears on the LHS of one IO and the encoding constraint
    says it's bounded by source + incoming edges at its tail.

    Returns (new_ineq, was_applied).
    """
    u, v = edge
    key = f"U_{u}_{v}"
    if key not in index.var_to_idx:
        return ineq.copy(), False

    c = ineq.coeffs[index.var_to_idx[key]]
    if c <= tol:  # edge not on LHS
        return ineq.copy(), False

    # Encoding bound: h(U_{u→v}) ≤ h(Y_{S(u)}) + Σ h(U_{e'}) for e' into u
    result = ineq.copy()
    result.coeffs[index.var_to_idx[key]] = 0.0  # remove from LHS

    # Add source term for u
    src_key = f"Y_S_{u}"
    if src_key in index.var_to_idx:
        result.coeffs[index.var_to_idx[src_key]] += c  # add to LHS

    # Add incoming edge terms for u
    for (a, b) in edges:
        if b == u:  # edge into u
            e_key = f"U_{a}_{b}"
            if e_key in index.var_to_idx:
                result.coeffs[index.var_to_idx[e_key]] += c

    return result, True


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