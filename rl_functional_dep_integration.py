"""
RL Environment Extension: Crypto + Decode Actions.

Adds two new action types to Phase 2 and Phase 3:

  CRYPTO(cut_id)
    Generate the crypto inequality for a precomputed cut and add it to the accumulator.
    Effect: mathematically introduces constraints on sessions separated by cuts.

  DECODE(session_idx)
    Generate decoding functional dependence inequality for session i and add it to the accumulator.
    Effect: mathematically introduces sink functional dependence constraints.

==========================================================================
INTEGRATION PLAN
==========================================================================

Rather than modifying fixed_environment.py directly (which is large),
this module provides:

  1. FuncDepActions — catalogue of precomputed cuts and sessions.
  2. Action builders to present these to the RL policy.
"""

from typing import List, Tuple, Set, Dict, FrozenSet, Optional
import numpy as np

from fixed_inequality import Inequality, EntropyIndex
from functional_dependence import (
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