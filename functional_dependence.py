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
from fixed_inequality import Inequality, FractionalInequality, EntropyIndex


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
# Base Inequality Generators for Functional Dependence
# ─────────────────────────────────────────────────────────────────────────────

def generate_crypto_inequality(
    V_prime: Set[str],
    nodes: List[str],
    edges: List[Tuple[str, str]],
    sessions: List[Tuple[str, str]],
    index: EntropyIndex
) -> FractionalInequality:
    r"""
    Crypto Inequality: h(Y_sep) - \sum_{e \in cut} h(U_e) <= 0.
    In terms of the terminal rate r, h(Y_sep) = |sep| * r.
    Since h(Y_I) = |I| * r, h(Y_sep) = (|sep|/|I|) * h(Y_I).
    Therefore, the base inequality is:
        (|sep| / |I|) * h(Y_I) <= \sum_{e \in cut} h(U_e)
        
    Returning this as a FractionalInequality allows it to be safely 
    added to the accumulator without mathematically invalid substitutions.
    """
    cut_edges = _directed_cut_edges(V_prime, nodes, edges)
    separated = _sessions_separated_by_cut(V_prime, nodes, edges, sessions)
    
    fi = FractionalInequality(index, lam=1.0)
    
    if not separated or not cut_edges:
        return fi
        
    n_sessions = len(sessions)
    fi.set_lhs("Y_I", len(separated) / float(n_sessions))
    
    for e in cut_edges:
        key1 = f"U_{e[0]}_{e[1]}"
        key2 = f"U_{e[1]}_{e[0]}"
        if key1 in index.var_to_idx:
            fi.set_rhs(key1, 1.0)
        elif key2 in index.var_to_idx:
            fi.set_rhs(key2, 1.0)
            
    return fi


def generate_decode_inequality(
    session_idx: int,
    sessions: List[Tuple[str, str]],
    edges: List[Tuple[str, str]],
    index: EntropyIndex
) -> FractionalInequality:
    r"""
    Decode Inequality: h(Y_i) - \sum_{e \text{ into } t(i)} h(U_e) <= 0.
    Since h(Y_i) = r = h(Y_I) / |I|:
        (1/|I|) * h(Y_I) <= \sum_{e \text{ into } t(i)} h(U_e)
    """
    incident = _edges_into_sink(session_idx, sessions, edges)
    fi = FractionalInequality(index, lam=1.0)
    
    if not incident:
        return fi
        
    n_sessions = len(sessions)
    fi.set_lhs("Y_I", 1.0 / float(n_sessions))
    
    for e in incident:
        key1 = f"U_{e[0]}_{e[1]}"
        key2 = f"U_{e[1]}_{e[0]}"
        if key1 in index.var_to_idx:
            fi.set_rhs(key1, 1.0)
        elif key2 in index.var_to_idx:
            fi.set_rhs(key2, 1.0)
            
    return fi


def generate_encode_inequality(
    edge: Tuple[str, str],
    edges: List[Tuple[str, str]],
    index: EntropyIndex
) -> FractionalInequality:
    r"""
    Encode Inequality: h(U_e) - h(Y_{S(u)}) - \sum_{e' \text{ into } u} h(U_{e'}) <= 0.
    Allows proper substitution-via-addition.
    """
    u, v = edge
    fi = FractionalInequality(index, lam=1.0)
    
    key_e1 = f"U_{u}_{v}"
    key_e2 = f"U_{v}_{u}"
    edge_key = key_e1 if key_e1 in index.var_to_idx else (key_e2 if key_e2 in index.var_to_idx else None)
    
    if edge_key is None:
        return fi
        
    fi.set_lhs(edge_key, 1.0)
    fi.set_rhs(f"Y_S_{u}", 1.0)
    
    incoming = _incoming_edges(u, edges)
    for e_in in incoming:
        k1 = f"U_{e_in[0]}_{e_in[1]}"
        k2 = f"U_{e_in[1]}_{e_in[0]}"
        k_in = k1 if k1 in index.var_to_idx else (k2 if k2 in index.var_to_idx else None)
        if k_in is not None and k_in != edge_key:
            fi.set_rhs(k_in, 1.0)
            
    return fi


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

    For efficiency, limits to cuts of size <= |nodes|//2 (larger cuts are
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