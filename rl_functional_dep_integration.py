"""
RL Environment Extension: Crypto + Decode Actions.

Adds two new action types to Phase 2 and Phase 3:

  CRYPTO(cut_id)
    Apply the crypto inequality for a precomputed cut.
    Valid when: at least one element of pool is in terminal-form and
                at least one session is separated by the cut.
    Effect: increases Y_I coefficient → tighter bound.

  DECODE(session_idx)
    Apply decoding functional dependence for session i.
    Valid when: edges into t(i) are on RHS of current accumulated inequality.
    Effect: increases Y_I coefficient → tighter bound.

==========================================================================
INTEGRATION PLAN
==========================================================================

Rather than modifying fixed_environment.py directly (which is large),
this module provides:

  1. CryptoDecodeExtension — a mixin that can wrap PartitionBoundEnv
     and override step() and get_valid_actions().

  2. precompute_functional_dep_actions() — precomputes all valid
     crypto cuts and decode actions for a given network.

  3. apply_best_functional_dep() — greedy oracle that finds the best
     single functional dependence action for a given inequality pool.
     Used to evaluate the bound improvement potential.

==========================================================================
MATHEMATICAL VALIDITY NOTES
==========================================================================

Crypto inequality (valid when cut_edges ⊆ RHS of accumulated ineq):
  Starting from:  LHS ≤ Σ_e c_e h(U_e)
  Adding crypto:  LHS + Σᵢ h(Yᵢ) ≤ Σ_e c_e h(U_e)
  Valid because:  h(Yᵢ) ≤ h(U_cut) ≤ Σ_{e∈cut} c_e h(U_e)
                  and cut ⊆ RHS.

Decode substitution (valid when in(t(i)) ⊆ RHS):
  Starting from:  LHS ≤ Σ_e c_e h(U_e)
  Adding decode:  LHS + h(Yᵢ) ≤ Σ_e c_e h(U_e)
  Valid because:  h(Yᵢ) ≤ h(edges into t(i)) ≤ Σ_{e∈in(t)} c_e h(U_e)
                  and in(t) ⊆ RHS.

Both constraints are implied by the basic inequalities and network
constraints (eqs. 16-17 in paper).
"""

from typing import List, Tuple, Set, Dict, FrozenSet, Optional
import numpy as np

from fixed_inequality import Inequality, EntropyIndex
from functional_dependence import (
    apply_crypto_inequality_direct,
    apply_decode_substitution,
    best_crypto_cuts,
    _directed_cut_edges,
    _incoming_edges,
)


# ─────────────────────────────────────────────────────────────────────────────
# Precomputed action catalogue
# ─────────────────────────────────────────────────────────────────────────────

class FuncDepActions:
    """
    Precomputed catalogue of all valid functional dependence actions for
    a given network. Created once per episode and reused for action
    validity checks and application.
    """

    def __init__(
        self,
        nodes: List[str],
        edges: List[Tuple[str, str]],
        sessions: List[Tuple[str, str]],
        max_crypto_cuts: int = 20,
    ):
        self.nodes    = nodes
        self.edges    = edges
        self.sessions = sessions

        # Precompute crypto cuts: list of (frozenset(V'), [sep_session_indices])
        self.crypto_cuts: List[Tuple[FrozenSet[str], List[int]]] = (
            best_crypto_cuts(nodes, edges, sessions, max_cuts=max_crypto_cuts)
        )

        # Precompute incoming edges for each session's sink
        self.sink_incoming: Dict[int, List[Tuple[str, str]]] = {}
        for si, (_, t) in enumerate(sessions):
            self.sink_incoming[si] = _incoming_edges(t, edges)

    def num_crypto_cuts(self) -> int:
        return len(self.crypto_cuts)

    def num_decode_actions(self) -> int:
        return len(self.sessions)

    def crypto_cut(self, idx: int) -> Tuple[FrozenSet[str], List[int]]:
        return self.crypto_cuts[idx]

    def summary(self) -> str:
        lines = [
            f"FuncDepActions: {self.num_crypto_cuts()} crypto cuts, "
            f"{self.num_decode_actions()} decode actions",
        ]
        for i, (vp, sep) in enumerate(self.crypto_cuts[:5]):
            cut_size = len(_directed_cut_edges(vp, self.nodes, self.edges))
            lines.append(
                f"  Cut {i}: V'={set(vp)}, separates {len(sep)} sessions, |cut|={cut_size}"
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Check validity of each action given current accumulated inequality
# ─────────────────────────────────────────────────────────────────────────────

def is_crypto_valid(
    ineq: Inequality,
    V_prime: FrozenSet[str],
    nodes: List[str],
    edges: List[Tuple[str, str]],
    sessions: List[Tuple[str, str]],
    index: EntropyIndex,
    tol: float = 1e-9
) -> bool:
    """True if the crypto inequality for cut V' can be applied to `ineq`."""
    cut_edges = _directed_cut_edges(set(V_prime), nodes, edges)
    if not cut_edges:
        return False
    for e in cut_edges:
        key = f"U_{e[0]}_{e[1]}"
        if key not in index.var_to_idx:
            key = f"U_{e[1]}_{e[0]}"
        if key not in index.var_to_idx:
            return False
        if ineq.coeffs[index.var_to_idx[key]] >= -tol:
            return False
    return True


def is_decode_valid(
    ineq: Inequality,
    session_idx: int,
    sink_incoming: List[Tuple[str, str]],
    index: EntropyIndex,
    tol: float = 1e-9
) -> bool:
    """True if the decode substitution for session_idx can be applied to `ineq`."""
    if not sink_incoming:
        return False
    for e in sink_incoming:
        key = f"U_{e[0]}_{e[1]}"
        if key not in index.var_to_idx:
            key = f"U_{e[1]}_{e[0]}"
        if key not in index.var_to_idx:
            return False
        if ineq.coeffs[index.var_to_idx[key]] >= -tol:
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Greedy oracle: apply all improving functional dependence actions
# ─────────────────────────────────────────────────────────────────────────────

def apply_best_functional_dep(
    ineq: Inequality,
    fda: FuncDepActions,
    index: EntropyIndex,
    internal_per_part: List[int],
    n_sessions: int,
    n_edges: int,
) -> Tuple[Inequality, float, str]:
    """
    Try all crypto and decode actions, apply the one that gives the best
    (lowest) bound.  Returns (best_ineq, best_bound, action_description).

    If no action improves the bound, returns (ineq, current_bound, 'none').
    """
    current_bound = ineq.extract_bound(n_sessions, n_edges, internal_per_part) \
        if ineq.check_valid_terminal_form() else float('inf')

    best_ineq  = ineq
    best_bound = current_bound
    best_desc  = 'none'

    # Try crypto cuts
    for ci, (vp, sep_list) in enumerate(fda.crypto_cuts):
        if not is_crypto_valid(ineq, vp, fda.nodes, fda.edges, fda.sessions, index):
            continue
        new_ineq, applied = apply_crypto_inequality_direct(
            ineq, set(vp), fda.nodes, fda.edges, fda.sessions, index
        )
        if not applied:
            continue
        if not new_ineq.check_valid_terminal_form():
            continue
        b = new_ineq.extract_bound(n_sessions, n_edges, internal_per_part)
        if b < best_bound - 1e-9:
            best_bound = b
            best_ineq  = new_ineq
            best_desc  = f"crypto_cut_{ci}(sep={len(sep_list)})"

    # Try decode substitutions
    for si in range(len(fda.sessions)):
        incoming = fda.sink_incoming[si]
        if not is_decode_valid(ineq, si, incoming, index):
            continue
        new_ineq, applied = apply_decode_substitution(
            ineq, si, fda.sessions, fda.edges, index
        )
        if not applied:
            continue
        if not new_ineq.check_valid_terminal_form():
            continue
        b = new_ineq.extract_bound(n_sessions, n_edges, internal_per_part)
        if b < best_bound - 1e-9:
            best_bound = b
            best_ineq  = new_ineq
            best_desc  = f"decode_session_{si}"

    return best_ineq, best_bound, best_desc


def apply_all_improving_func_dep(
    ineq: Inequality,
    fda: FuncDepActions,
    index: EntropyIndex,
    internal_per_part: List[int],
    n_sessions: int,
    n_edges: int,
    max_rounds: int = 5
) -> Tuple[Inequality, float, List[str]]:
    """
    Greedily apply functional dependence actions until no further
    improvement is possible (or max_rounds reached).

    Returns (final_ineq, final_bound, list_of_applied_actions).
    """
    result     = ineq
    actions    = []
    prev_bound = float('inf')

    for _ in range(max_rounds):
        new_ineq, new_bound, desc = apply_best_functional_dep(
            result, fda, index, internal_per_part, n_sessions, n_edges
        )
        if desc == 'none' or new_bound >= prev_bound - 1e-9:
            break
        result     = new_ineq
        prev_bound = new_bound
        actions.append(desc)

    final_bound = result.extract_bound(n_sessions, n_edges, internal_per_part) \
        if result.check_valid_terminal_form() else float('inf')

    return result, final_bound, actions


# ─────────────────────────────────────────────────────────────────────────────
# RL Action builders (for environment integration)
# ─────────────────────────────────────────────────────────────────────────────

ACTION_CRYPTO = 20   # new action type IDs (don't conflict with existing)
ACTION_DECODE = 21


def build_crypto_actions(fda: FuncDepActions) -> List[Dict]:
    """Return list of valid CRYPTO action dicts (one per precomputed cut)."""
    return [
        {'type': ACTION_CRYPTO, 'cut_idx': i, 'sep_count': len(sep)}
        for i, (_, sep) in enumerate(fda.crypto_cuts)
    ]


def build_decode_actions(fda: FuncDepActions) -> List[Dict]:
    """Return list of DECODE action dicts (one per session)."""
    return [
        {'type': ACTION_DECODE, 'session_idx': si}
        for si in range(len(fda.sessions))
    ]


def get_all_func_dep_actions(fda: FuncDepActions) -> List[Dict]:
    """Combined list of all functional dependence actions."""
    return build_crypto_actions(fda) + build_decode_actions(fda)