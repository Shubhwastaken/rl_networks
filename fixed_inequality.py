"""
Entropy inequality representation and variable indexing.

CHANGES FOR PHASE 3 FRACTIONAL SEARCH:
  - EntropyIndex: unchanged (backward compat with Phases 1/2).
  - Inequality.scale(lam): new — returns lam * self with coeffs already
    multiplied, so all downstream ops (add, check_valid_terminal_form,
    extract_bound) are identical. This is what lets Phase 3 form
    λ·IO(u) + (1-λ)·IO(v) before any variable cancellation happens.
  - FractionalInequality: subclass that records λ, source_nodes, partition_ids
    for Phase 3 state encoding and trace logging. Mathematically identical.
  - FractionalPool: Phase 3's working set with priority eviction that
    explicitly rewards cross-partition and fractional-λ inequalities.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set
import numpy as np


@dataclass
class EntropyIndex:
    partitions : List[List[str]]
    nodes      : List[str]
    edges      : List[Tuple[str, str]]
    sessions   : List[Tuple[str, str]]

    var_to_idx : Dict[str, int] = field(default_factory=dict)
    idx_to_var : Dict[int, str] = field(default_factory=dict)
    dim        : int = 0

    st_sessions       : List[Set[int]] = field(default_factory=list)
    internal_sessions : List[Set[int]] = field(default_factory=list)

    def __post_init__(self):
        idx = 0
        n = len(self.partitions)

        for i in range(n):
            key = f"Y_ST_P{i}"
            self.var_to_idx[key] = idx
            self.idx_to_var[idx] = key
            idx += 1

        for i in range(n):
            key = f"Y_I_P{i}"
            self.var_to_idx[key] = idx
            self.idx_to_var[idx] = key
            idx += 1

        for v in self.nodes:
            key = f"Y_S_{v}"
            self.var_to_idx[key] = idx
            self.idx_to_var[idx] = key
            idx += 1

        for (u, v) in self.edges:
            key = f"U_{u}_{v}"
            self.var_to_idx[key] = idx
            self.idx_to_var[idx] = key
            idx += 1

        self.var_to_idx["Y_I"] = idx
        self.idx_to_var[idx]   = "Y_I"
        idx += 1

        self.dim = idx
        self._precompute_session_sets()

    def _precompute_session_sets(self):
        self.st_sessions       = []
        self.internal_sessions = []
        for Pi in self.partitions:
            Pi_set   = set(Pi)
            st       = set()
            internal = set()
            for si, (s, t) in enumerate(self.sessions):
                if s in Pi_set or t in Pi_set:
                    st.add(si)
                if s in Pi_set and t in Pi_set:
                    internal.add(si)
            self.st_sessions.append(st)
            self.internal_sessions.append(internal)

    def get_yi_idx(self):             return self.var_to_idx["Y_I"]
    def yi_idx(self):                 return self.get_yi_idx()
    def get_yst_idx(self, i):         return self.var_to_idx[f"Y_ST_P{i}"]
    def yst_idx(self, i):             return self.get_yst_idx(i)
    def get_yi_pi_idx(self, i):       return self.var_to_idx[f"Y_I_P{i}"]
    def yi_pi_idx(self, i):           return self.get_yi_pi_idx(i)
    def yi_partition_idx(self, i):    return self.get_yi_pi_idx(i)
    def get_source_idx(self, v):      return self.var_to_idx[f"Y_S_{v}"]
    def source_idx(self, v):          return self.get_source_idx(v)
    def get_edge_signal_idx(self, e): return self.var_to_idx[f"U_{e[0]}_{e[1]}"]
    def edge_idx(self, e):            return self.get_edge_signal_idx(e)

    def get_cross_edge_idx(self, i, j):
        lo, hi = min(i, j), max(i, j)
        key = f"U_P{lo}_P{hi}"
        if key not in self.var_to_idx:
            raise KeyError(f"Cross-partition variable {key} not in index.")
        return self.var_to_idx[key]
    def cross_idx(self, i, j):        return self.get_cross_edge_idx(i, j)

    def n(self):               return len(self.partitions)
    def all_sessions(self):    return set(range(len(self.sessions)))

    def sessions_covered_by(self, partitions: Set[int]) -> Set[int]:
        covered = set()
        for i in partitions:
            covered |= self.st_sessions[i]
        return covered


class Inequality:

    def __init__(self, index: EntropyIndex):
        self.index  = index
        self.coeffs = np.zeros(index.dim, dtype=np.float64)
        self.active_st_partitions: Set[int] = set()

    def set_lhs(self, var_key: str, value: float):
        self.coeffs[self.index.var_to_idx[var_key]] += value

    def set_rhs(self, var_key: str, value: float):
        self.coeffs[self.index.var_to_idx[var_key]] -= value

    def add(self, other: "Inequality") -> "Inequality":
        result = Inequality(self.index)
        result.coeffs = self.coeffs + other.coeffs
        result.active_st_partitions = (
            self.active_st_partitions | other.active_st_partitions
        )
        return result

    def scale(self, lam: float) -> "Inequality":
        """Return λ * self. Core primitive for fractional IO construction."""
        result = Inequality(self.index)
        result.coeffs = self.coeffs * lam
        result.active_st_partitions = set(self.active_st_partitions)
        return result

    def active_yst(self) -> Set[int]:
        return {
            i for i in range(len(self.index.partitions))
            if self.coeffs[self.index.get_yst_idx(i)] > 1e-9
        }

    def get_yi_coefficient(self) -> float:
        return float(self.coeffs[self.index.get_yi_idx()])

    def yi_coeff(self) -> float:
        return self.get_yi_coefficient()

    def get_rhs_edge_coefficient(self) -> float:
        total = 0.0
        for e in self.index.edges:
            c = self.coeffs[self.index.get_edge_signal_idx(e)]
            if c < 0:
                total += abs(c)
        return total

    def rhs_edge_sum(self) -> float:
        return self.get_rhs_edge_coefficient()

    def get_lhs_internal_coefficient(self) -> float:
        total = 0.0
        for i in range(len(self.index.partitions)):
            c = self.coeffs[self.index.get_yi_pi_idx(i)]
            if c > 0:
                total += c
        return total

    def internal_coeff_sum(self) -> float:
        return self.get_lhs_internal_coefficient()

    # Minimum Y_I coefficient for a non-degenerate terminal inequality.
    # n2-submod always produces Y_I=1.0.  A valid 2-partition fractional
    # combination via CROSS_SUBMOD should produce Y_I >= 0.5.
    # Single-edge FIOs that game the floor by tuning lam so Y_I=1/|I|
    # (giving bound=1.0=LP) are blocked — they pass the old 0.125 floor
    # by adapting lam but are still structurally vacuous.
    MIN_YI_COEFF: float = 0.5

    def check_valid_terminal_form(self, tol: float = 1e-4) -> bool:
        c1 = self.get_yi_coefficient()
        # Require Y_I >= 0.5: blocks single-node fractional IOs that tune
        # lam to game the floor while still producing bound = LP.
        if c1 < self.MIN_YI_COEFF:
            return False
        for i in range(len(self.index.partitions)):
            if self.coeffs[self.index.get_yst_idx(i)] > tol * c1:
                return False
        if self.get_rhs_edge_coefficient() <= 0:
            return False
        for v in self.index.nodes:
            if self.coeffs[self.index.get_source_idx(v)] > tol:
                return False
            if self.coeffs[self.index.get_source_idx(v)] < -tol:
                return False
        # Require at least 2 edges on RHS: a single-edge bound can never
        # exploit multi-partition graph structure and always collapses to
        # bound = 1/(c1*|I|) = LP by choosing the right lam.
        edge_count = sum(
            1 for e in self.index.edges
            if self.coeffs[self.index.get_edge_signal_idx(e)] < -tol
        )
        if edge_count < 2:
            return False
        return True

    def cancel_source_terms(self, tol: float = 1e-9) -> "Inequality":
        """
        Cancel RHS source terms by substituting source independence:
          h(Y_S_v) = n_src_v * r  and  h(Y_I) = |I| * r
          => h(Y_S_v) = (n_src_v / |I|) * h(Y_I)

        Moves source contribution from RHS to LHS by adjusting the Y_I
        coefficient.  This is the node IO counterpart of
        _collapse_to_yi_if_valid: that function handles partition IOs
        (Y_ST -> Y_I + zero sources), this one handles node IOs (Y_I
        already set, just zero sources and adjust Y_I).

        Condition: all session source nodes must have negative source
        coefficients (i.e. be present on the RHS).

        Returns self unchanged if cancellation is not possible or would
        make Y_I non-positive (destructive cancellation).
        """
        sessions   = self.index.sessions
        n_sessions = len(sessions)
        if n_sessions == 0:
            return self

        # Need Y_I > 0 to have something to cancel against
        yi = self.coeffs[self.index.yi_idx()]
        if yi <= tol:
            return self

        # Check that every session's source node has a source term on RHS
        source_nodes_needed = set(s for s, t in sessions)
        for v in source_nodes_needed:
            if self.coeffs[self.index.get_source_idx(v)] >= -tol:
                return self  # not all sources covered -> can't cancel

        result = self.copy()

        # Compute Y_I deduction:  sum |c_v| * n_src_v / |I|
        source_yi_deduction = 0.0
        for v in self.index.nodes:
            c = result.coeffs[self.index.get_source_idx(v)]
            if c < -tol:
                n_src = sum(1 for s, t in sessions if s == v)
                source_yi_deduction += abs(c) * n_src / n_sessions
                result.coeffs[self.index.get_source_idx(v)] = 0.0

        result.coeffs[self.index.yi_idx()] -= source_yi_deduction

        # If Y_I is now non-positive, cancellation was destructive -> revert
        if result.coeffs[self.index.yi_idx()] <= tol:
            return self

        return result

    def extract_bound(
        self,
        num_sessions      : int,
        num_edges         : int,
        internal_per_part : List[int]
    ) -> float:
        """
        Extract r from the terminal inequality:
          c1*h(Y_I) + sum c_k*h(Y_I_Pk) <= sum s_v*h(Y_S_v) + sum e_j*h(U_j)

        Source terms cancel EXACTLY via source independence:
          h(Y_ST_Pk) = h(sources in Pk) = h(Y_S for nodes in Pk)
        So the source coefficients on RHS match the LHS session coefficients
        and cancel exactly — leaving only edge capacity on RHS.

        The standard formula is therefore:
          (c1*|I| + weighted_internal) * r <= edge_cap
          r <= edge_cap / (c1*|I| + weighted_internal)

        This is correct for properly formed terminal inequalities where
        source terms have already been cancelled by the submodularity step.
        """
        c1       = self.get_yi_coefficient()
        edge_cap = self.get_rhs_edge_coefficient()

        weighted_internal = sum(
            self.coeffs[self.index.get_yi_pi_idx(i)] * count
            for i, count in enumerate(internal_per_part)
            if self.coeffs[self.index.get_yi_pi_idx(i)] > 1e-9
        )
        # Source terms on RHS: h(Y_S_v) <= (sessions sourced at v) * r
        # These move to LHS denominator when rearranging inequality.
        # Each source node v has exactly n_src sessions, so h(Y_S_v) <= n_src*r.
        source_coeff = 0.0
        for v in self.index.nodes:
            c = self.coeffs[self.index.get_source_idx(v)]
            if c < -1e-9:
                n_src = sum(1 for s, t in self.index.sessions if s == v)
                source_coeff += abs(c) * n_src

        denom = c1 * num_sessions + weighted_internal - source_coeff
        if denom <= 1e-9:
            return float('inf')
        return edge_cap / denom

    def copy(self) -> "Inequality":
        result = Inequality(self.index)
        result.coeffs = self.coeffs.copy()
        result.active_st_partitions = set(self.active_st_partitions)
        return result

    def __repr__(self) -> str:
        lhs, rhs = [], []
        for idx, coeff in enumerate(self.coeffs):
            if abs(coeff) > 1e-9:
                var = self.index.idx_to_var[idx]
                if coeff > 0:
                    lhs.append(f"{coeff:.3f}*{var}")
                else:
                    rhs.append(f"{abs(coeff):.3f}*{var}")
        return f"{' + '.join(lhs) or '0'} <= {' + '.join(rhs) or '0'}"


# ---------------------------------------------------------------------------
# FractionalInequality
# ---------------------------------------------------------------------------

class FractionalInequality(Inequality):
    """
    Inequality that records the fractional weight λ and which nodes / partition
    sets contributed to it.  All math goes through the parent class unchanged —
    lam is baked into .coeffs.  The extra fields exist purely for Phase 3 state
    encoding and human-readable trace output.

    Formation:
        base_u = generate_node_io(u, index)          # standard IO for node u
        base_v = generate_node_io(v, index)          # standard IO for node v
        fi = base_u.scale(lam).add(base_v.scale(1-lam))
        fi.__class__ = FractionalInequality          # or use make_fractional()
        fi.lam = lam; fi.source_nodes = [u, v]; fi.partition_ids = [pu, pv]
    """

    def __init__(self, index: EntropyIndex, lam: float = 1.0,
                 source_nodes: List[str] = None,
                 partition_ids: List[int] = None):
        super().__init__(index)
        self.lam           = lam
        self.source_nodes  = source_nodes or []
        self.partition_ids = partition_ids or []

    def copy(self) -> "FractionalInequality":
        result = FractionalInequality(
            self.index, self.lam,
            list(self.source_nodes), list(self.partition_ids)
        )
        result.coeffs = self.coeffs.copy()
        result.active_st_partitions = set(self.active_st_partitions)
        return result

    def add(self, other: "Inequality") -> "FractionalInequality":
        base = Inequality.add(self, other)
        result = FractionalInequality(self.index)
        result.coeffs = base.coeffs
        result.active_st_partitions = base.active_st_partitions
        result.source_nodes  = self.source_nodes  + getattr(other, 'source_nodes',  [])
        result.partition_ids = self.partition_ids + getattr(other, 'partition_ids', [])
        result.lam = self.lam
        return result

    def scale(self, lam: float) -> "FractionalInequality":
        base = Inequality.scale(self, lam)
        result = FractionalInequality(
            self.index, self.lam * lam,
            list(self.source_nodes), list(self.partition_ids)
        )
        result.coeffs = base.coeffs
        result.active_st_partitions = set(self.active_st_partitions)
        return result

    def is_cross_partition(self) -> bool:
        """True if nodes come from at least two different partition sets."""
        return len(set(self.partition_ids)) > 1

    def lambda_is_fractional(self) -> bool:
        """True when λ ∉ ℤ — the hallmark of an inequality outside the PB family."""
        return abs(self.lam - round(self.lam)) > 1e-6

    def __repr__(self) -> str:
        base = Inequality.__repr__(self)
        tag  = f"[λ={self.lam:.4f} src={self.source_nodes} parts={self.partition_ids}]"
        return f"{tag}  {base}"


def make_fractional(ineq: Inequality, lam: float = 1.0,
                    source_nodes: List[str] = None,
                    partition_ids: List[int] = None) -> FractionalInequality:
    """Promote a plain Inequality to FractionalInequality. Args default to empty."""
    fi = FractionalInequality(ineq.index, lam,
                              source_nodes or [], partition_ids or [])
    fi.coeffs = ineq.coeffs.copy()
    fi.active_st_partitions = set(ineq.active_st_partitions)
    return fi


# ---------------------------------------------------------------------------
# FractionalPool
# ---------------------------------------------------------------------------

class FractionalPool:
    """
    Phase 3's working inequality pool.

    Priority eviction explicitly rewards:
      - Terminal form ready       (+2.0)
      - Cross-partition mixing    (+0.5)  ← key for beating PB
      - Fractional λ coefficient  (+0.3)  ← key for beating PB
      - High yi / rhs_edge ratio  (base score)
    """

    def __init__(self, max_size: int = 50):
        self.items: List[FractionalInequality] = []
        self.max_size = max_size

    def add(self, ineq: "Inequality"):
        # Accept plain Inequality, treat as FractionalInequality with lam=1
        if not isinstance(ineq, FractionalInequality):
            ineq = make_fractional(ineq, 1.0, [], [])
        self.items.append(ineq)
        if len(self.items) > self.max_size:
            self._evict()

    def _evict(self):
        def score(ineq: FractionalInequality) -> float:
            yi  = ineq.get_yi_coefficient()
            rhs = ineq.get_rhs_edge_coefficient()
            s   = yi / max(rhs, 1e-9) if yi > 0 and rhs > 0 else 0.0
            s  += 2.0 if ineq.check_valid_terminal_form() else 0.0
            s  += 0.5 if ineq.is_cross_partition()       else 0.0
            s  += 0.3 if ineq.lambda_is_fractional()     else 0.0
            return s
        self.items.sort(key=score, reverse=True)
        self.items = self.items[:self.max_size]

    def best_bound(self, num_sessions: int, num_edges: int,
                   internal_per_part: List[int]) -> float:
        best = float('inf')
        for ineq in self.items:
            if ineq.check_valid_terminal_form():
                b = ineq.extract_bound(num_sessions, num_edges, internal_per_part)
                if b < best:
                    best = b
        return best

    def contains_equivalent(self, ineq: "FractionalInequality",
                             tol: float = 1e-6) -> bool:
        """Return True if the pool already holds an inequality whose
        coefficient vector is within *tol* of *ineq*'s coefficients.
        Used by Fix B to avoid rewarding the agent for re-adding clones."""
        if not isinstance(ineq, FractionalInequality):
            return False
        for existing in self.items:
            if existing.coeffs.shape == ineq.coeffs.shape:
                if np.max(np.abs(existing.coeffs - ineq.coeffs)) < tol:
                    return True
        return False

    def has_cross_partition(self) -> bool:
        return any(i.is_cross_partition() for i in self.items)

    def has_fractional_lambda(self) -> bool:
        return any(i.lambda_is_fractional() for i in self.items)

    def coeff_matrix(self) -> np.ndarray:
        if not self.items:
            return np.zeros((1, 1))
        return np.stack([i.coeffs for i in self.items])

    def __len__(self):          return len(self.items)
    def __getitem__(self, idx): return self.items[idx]
    def __iter__(self):         return iter(self.items)