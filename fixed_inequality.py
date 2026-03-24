"""
Entropy inequality representation and variable indexing.

FIX: Removed U_P{i}_P{j} cross-partition group variables from
     EntropyIndex. They were allocated but NEVER written to by
     any base inequality or submodularity operation. They added
     noise to the coefficient vector and wasted dimensions in
     the Transformer encoder.

     If cross-partition group variables are needed later, they
     can be re-added — but the current proof strategy only uses
     individual edge variables (U_{u}_{v}).
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

        # Y_ST_P{i}
        for i in range(n):
            key = f"Y_ST_P{i}"
            self.var_to_idx[key] = idx
            self.idx_to_var[idx] = key
            idx += 1

        # Y_I_P{i}  (internal sessions per partition)
        for i in range(n):
            key = f"Y_I_P{i}"
            self.var_to_idx[key] = idx
            self.idx_to_var[idx] = key
            idx += 1

        # FIX: REMOVED U_P{i}_P{j} cross-partition group variables.
        # They were never written to and only wasted dimensions.
        # Backward compat: cross_idx() raises KeyError if called.

        # Y_S_{v}  (source entropy per node)
        for v in self.nodes:
            key = f"Y_S_{v}"
            self.var_to_idx[key] = idx
            self.idx_to_var[idx] = key
            idx += 1

        # U_{u}_{v}  (individual edge signal entropy)
        for (u, v) in self.edges:
            key = f"U_{u}_{v}"
            self.var_to_idx[key] = idx
            self.idx_to_var[idx] = key
            idx += 1

        # Y_I  (global session entropy)
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

    # --- index accessors ---

    def get_yi_idx(self):            return self.var_to_idx["Y_I"]
    def yi_idx(self):                return self.get_yi_idx()

    def get_yst_idx(self, i):        return self.var_to_idx[f"Y_ST_P{i}"]
    def yst_idx(self, i):            return self.get_yst_idx(i)

    def get_yi_pi_idx(self, i):      return self.var_to_idx[f"Y_I_P{i}"]
    def yi_pi_idx(self, i):          return self.get_yi_pi_idx(i)
    def yi_partition_idx(self, i):   return self.get_yi_pi_idx(i)

    def get_source_idx(self, v):     return self.var_to_idx[f"Y_S_{v}"]
    def source_idx(self, v):         return self.get_source_idx(v)

    def get_edge_signal_idx(self, e): return self.var_to_idx[f"U_{e[0]}_{e[1]}"]
    def edge_idx(self, e):            return self.get_edge_signal_idx(e)

    def get_cross_edge_idx(self, i, j):
        lo, hi = min(i, j), max(i, j)
        key = f"U_P{lo}_P{hi}"
        if key not in self.var_to_idx:
            raise KeyError(
                f"Cross-partition variable {key} not in index. "
                f"Cross-partition group variables were removed in this version."
            )
        return self.var_to_idx[key]
    def cross_idx(self, i, j):       return self.get_cross_edge_idx(i, j)

    def n(self):                     return len(self.partitions)
    def all_sessions(self):          return set(range(len(self.sessions)))

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

    def active_yst(self) -> Set[int]:
        return {
            i for i in range(len(self.index.partitions))
            if self.coeffs[self.index.get_yst_idx(i)] > 1e-9
        }

    # --- coefficient accessors ---

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

    # --- terminal form check ---

    def check_valid_terminal_form(self, tol: float = 1e-4) -> bool:
        """
        Valid terminal form:
            c1*h(Y_I) + Sigma coeff_i*h(Y_I(Pi,Pi)) <= c3*Sigma h(U_e)

        Conditions:
        1. h(Y_I) coefficient c1 > 0
        2. All Y_ST coefficients negligible (< tol * c1)
        3. RHS edge sum c3 > 0
        4. No positive source entropy terms remaining on LHS
        """
        c1 = self.get_yi_coefficient()
        if c1 <= 0:
            return False

        for i in range(len(self.index.partitions)):
            if self.coeffs[self.index.get_yst_idx(i)] > tol * c1:
                return False

        if self.get_rhs_edge_coefficient() <= 0:
            return False

        for v in self.index.nodes:
            if self.coeffs[self.index.get_source_idx(v)] > tol:
                return False

        return True

    # --- bound extraction ---

    def extract_bound(
        self,
        num_sessions      : int,
        num_edges         : int,
        internal_per_part : List[int]
    ) -> float:
        """
        From: c1*h(Y_I) + Sigma_i coeff_i*h(Y_I(Pi,Pi)) <= c3*Sigma h(U_e)
        Using h(Y_i) >= r*log_b and h(U_e) <= log_b:
            r <= c3 / (c1*|I| + Sigma_i coeff_i*|I(Pi,Pi)|)

        NOTE: c3 already counts edge capacity — do NOT multiply by num_edges.
        """
        c1 = self.get_yi_coefficient()
        c3 = self.get_rhs_edge_coefficient()

        weighted_internal = sum(
            self.coeffs[self.index.get_yi_pi_idx(i)] * count
            for i, count in enumerate(internal_per_part)
            if self.coeffs[self.index.get_yi_pi_idx(i)] > 1e-9
        )

        denom = c1 * num_sessions + weighted_internal
        if denom <= 0:
            return float('inf')

        return c3 / denom

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
                    lhs.append(f"{coeff:.2f}*{var}")
                else:
                    rhs.append(f"{abs(coeff):.2f}*{var}")
        return f"{' + '.join(lhs) or '0'} <= {' + '.join(rhs) or '0'}"