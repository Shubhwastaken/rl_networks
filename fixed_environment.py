"""
RL Environment — Three-Phase Architecture.

PHASE 1 (unchanged logic, richer output):
  Agent assigns nodes to independent sets (partition).
  OUTPUT now includes:
    - partition: list of groups
    - partition_weights: dict {group_id: float} learned by policy head
  The weights become λ candidates for Phase 3.

PHASE 2 (purpose changed):
  Given a fixed partition, agent learns the PROOF CALCULUS:
  how to combine raw per-node IO inequalities via SUBMOD, SCALE, ECAP,
  FUNC_DEP into terminal form. Crucially Phase 2 now operates on
  per-node IOs (not pre-summed partition IOs), so it must discover
  the summing pattern itself. This teaches the action grammar.
  NEW ACTION: CROSS_SUBMOD — apply submodularity to two accumulator
  items that come from different partition sets. This is the action
  Phase 2 must learn to use; Phase 3 exploits it with fractional λ.

PHASE 3 (new — the actual novel-inequality search):
  Starts from Phase 1's partition + weight hints and Phase 2's learned
  proof calculus. Action space adds:
    FRACTIONAL_IO(u, v, λ): form λ·IO(u) + (1-λ)·IO(v) atomically
  Reward is ONLY positive when the extracted bound beats the partition
  bound for this graph. The partition bound is computed once at reset
  and stored as self.partition_bound.
"""

import random
from enum import IntEnum
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from fixed_graph_generation import generate_large_network, generate_graph_dataset
from partition import (
    generate_random_valid_partition,
    decode_partition,
    check_partition
)
from fixed_inequality import (
    Inequality, FractionalInequality, FractionalPool,
    EntropyIndex, make_fractional
)
from fixed_base_inequality_generator import (
    generate_base_inequalities,
    generate_node_io,
    generate_all_node_ios,
    generate_fractional_io,
    count_internal_sessions,
    internal_per_partition
)
from fixed_submodularity import (
    apply_pairwise_submodularity,
    apply_n2_submodularity_all_at_once
)


class Phase(IntEnum):
    PHASE1 = 1
    PHASE2 = 2
    PHASE3 = 3


class ActionType(IntEnum):
    # Phase 1
    ASSIGN_NODE          = 0
    SWAP_NODE            = 7
    MOVE_NODE            = 8
    FINALIZE_PARTITION   = 9
    # Phase 2 & 3
    ADD_TO_ACCUMULATOR   = 1
    APPLY_SUBMODULARITY  = 2
    APPLY_PROOF2         = 3
    STORE_AND_RESET      = 4
    COMBINE_STORED       = 5
    DECLARE_TERMINAL     = 6
    # Phase 3 only
    FRACTIONAL_IO        = 10   # λ·IO(u) + (1-λ)·IO(v)
    CROSS_SUBMOD         = 11   # submod across partition boundary


MAX_PHASE2_STEPS     = 30
MAX_PHASE3_STEPS     = 40    # Phase 3 gets more steps — it needs them
MAX_DERIVED          = 30    # larger pool for Phase 3
MAX_REFINEMENT_STEPS = 20
STEP_COST            = -0.01
STEP_COST_AFTER_TERMINAL = -0.10

# Fractional λ values offered to Phase 3 agent
LAMBDA_GRID = [0.25, 0.33, 0.40, 0.50, 0.60, 0.67, 0.75]


# ---------------------------------------------------------------------------
# Partition bound helper (brute-force for small graphs, greedy otherwise)
# ---------------------------------------------------------------------------

def _compute_partition_bound(nodes, edges, sessions) -> float:
    """Returns the tightest partition bound for this graph."""
    from itertools import combinations as _comb
    adj = {n: set() for n in nodes}
    for u, v in edges:
        adj[u].add(v); adj[v].add(u)

    def _sessions_within(S):
        Ss = set(S)
        return sum(1 for s, t in sessions if s in Ss and t in Ss)

    def _cut_edges(partition):
        part_of = {}
        for k, Pk in enumerate(partition):
            for nd in Pk: part_of[nd] = k
        return sum(1 for u, v in edges if part_of[u] != part_of[v])

    def _eval(partition):
        for Pk in partition:
            if any(adj[u] & (set(Pk) - {u}) for u in Pk):
                return float('inf')
        intra = sum(_sessions_within(Pk) for Pk in partition)
        cut   = _cut_edges(partition)
        denom = len(sessions) + intra
        return cut / denom if denom > 0 else float('inf')

    best = len(edges) / max(len(sessions), 1)

    # Greedy colorings
    import networkx as nx
    G = nx.Graph(); G.add_nodes_from(nodes); G.add_edges_from(edges)
    from collections import defaultdict
    for strat in ['largest_first', 'smallest_last', 'DSATUR']:
        try:
            col = nx.coloring.greedy_color(G, strategy=strat)
            groups = defaultdict(list)
            for nd, c in col.items(): groups[c].append(nd)
            best = min(best, _eval(list(groups.values())))
        except Exception:
            pass

    # Exhaustive 2-partitions for small graphs
    if len(nodes) <= 14:
        V = list(nodes); n = len(V)
        for mask in range(1, 1 << (n-1)):
            S = [V[i] for i in range(n) if mask & (1 << i)]
            T = [V[i] for i in range(n) if not (mask & (1 << i))]
            if S and T:
                best = min(best, _eval([S, T]))

    # Singleton partition (always valid)
    best = min(best, _eval([[v] for v in nodes]))
    return best


# ---------------------------------------------------------------------------
# Main Environment
# ---------------------------------------------------------------------------

class PartitionBoundEnv:

    def __init__(self, graph_dataset_size: int = 5, stage: int = 3):
        self.stage = stage
        print(f"Generating graph dataset ({graph_dataset_size} graphs)...")
        self.graph_dataset = generate_graph_dataset(graph_dataset_size)
        print(f"Dataset ready: {len(self.graph_dataset)} graphs.")

        # State
        self.nodes = self.edges = self.sessions = self.index = None
        self.assignment = {}
        self.num_groups = 0
        self.node_order = []
        self.current_node_idx = 0
        self.adjacency = {}
        self.edge_set  = set()

        self.base_inequalities = []
        self.node_ios: Dict[str, FractionalInequality] = {}
        self.num_base = 0
        self.pool: List[Inequality] = []
        self.frac_pool: FractionalPool = FractionalPool(MAX_DERIVED)
        self.accumulator: List[Inequality] = []
        self.stored_derived: List[Inequality] = []

        self.phase2_steps = 0
        self.phase3_steps = 0
        self.min_phase2_steps = 6
        self.partition = None
        self.partition_weights: Dict[int, float] = {}
        self.internal_per_part = None
        self.partition_bound   = float('inf')

        self.current_phase = Phase.PHASE1
        self.internal_session_count = 0
        self.prev_internal_count    = 0

        self._assignment_complete = False
        self._refinement_steps    = 0
        self._found_terminal      = False
        self._found_yi_collapse   = False
        self._proof2_used         = False

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------

    def reset(self, fixed_partition=None, fixed_graph=None,
              start_phase3=False) -> Dict[str, Any]:
        if fixed_graph is not None:
            self.nodes, self.edges, self.sessions = fixed_graph
        else:
            self.nodes, self.edges, self.sessions = random.choice(self.graph_dataset)

        self.adjacency = {n: set() for n in self.nodes}
        self.edge_set  = set()
        for u, v in self.edges:
            self.adjacency[u].add(v); self.adjacency[v].add(u)
            self.edge_set.add((u, v)); self.edge_set.add((v, u))

        self.assignment = {n: -1 for n in self.nodes}
        self.num_groups = 0
        self.node_order = self.nodes[:]
        random.shuffle(self.node_order)
        self.current_node_idx    = 0
        self.prev_internal_count = 0

        self.pool           = []
        self.frac_pool      = FractionalPool(MAX_DERIVED)
        self.accumulator    = []
        self.stored_derived = []
        self.phase2_steps   = 0
        self.phase3_steps   = 0
        self.partition      = None
        self.partition_weights = {}
        self.internal_per_part = None
        self.num_base          = 0
        self.node_ios          = {}

        self._assignment_complete = False
        self._refinement_steps    = 0
        self._found_terminal      = False
        self._found_yi_collapse   = False
        self._proof2_used         = False

        # Pre-compute partition bound for this graph (used in Phase 3 reward)
        self.partition_bound = _compute_partition_bound(
            self.nodes, self.edges, self.sessions
        )

        if fixed_partition is not None:
            self.partition = fixed_partition
            self._start_phase2()
            self.current_phase = Phase.PHASE2
        elif start_phase3:
            # Caller must have set self.partition and self.partition_weights
            self._start_phase3()
            self.current_phase = Phase.PHASE3
        else:
            self.current_phase = Phase.PHASE1

        return self._get_state()

    # -----------------------------------------------------------------------
    # Phase transitions
    # -----------------------------------------------------------------------

    def _start_phase2(self):
        """Set up Phase 2: per-node IOs as raw material."""
        self.index = EntropyIndex(
            partitions=self.partition,
            nodes=self.nodes,
            edges=self.edges,
            sessions=self.sessions
        )
        # Phase 1/2 base inequalities (partition-level, for PROOF2 action)
        self.base_inequalities = generate_base_inequalities(
            self.partition, self.nodes, self.edges,
            self.sessions, self.index
        )
        # Phase 2 now starts from per-node IOs, not partition-level IOs.
        # This is the key change: agent must discover the summing pattern.
        self.node_ios = generate_all_node_ios(
            self.partition, self.nodes, self.edges,
            self.sessions, self.index
        )
        # Pool starts with ALL per-node IOs
        self.pool     = [fi.copy() for fi in self.node_ios.values()]
        self.num_base = len(self.pool)

        self.internal_per_part     = internal_per_partition(self.partition, self.sessions)
        self.internal_session_count = sum(self.internal_per_part)

        n_parts = len(self.partition)
        self.min_phase2_steps = max(6, 2 * n_parts)

        self.current_phase = Phase.PHASE2
        self.phase2_steps  = 0
        self.combination_log = []

    def _start_phase3(self):
        """Set up Phase 3: fractional IO search starting from Phase 1/2 knowledge."""
        if self.index is None:
            self._start_phase2()   # ensure index is built

        # Populate frac_pool with all per-node IOs at weight 1.0
        self.frac_pool = FractionalPool(MAX_DERIVED)
        for fi in self.node_ios.values():
            self.frac_pool.add(fi.copy())

        # Also add partition-level IOs (from Phase 2 standard proof)
        for bi in self.base_inequalities:
            self.frac_pool.add(bi)

        self.current_phase  = Phase.PHASE3
        self.phase3_steps   = 0
        self.accumulator    = []
        self.stored_derived = []
        self._found_terminal = False
        self.combination_log = []

    # -----------------------------------------------------------------------
    # step()
    # -----------------------------------------------------------------------

    def step(self, action: Dict[str, Any]) -> Tuple[Dict, float, bool]:
        if self.current_phase == Phase.PHASE1:
            return self._step_phase1(action)
        elif self.current_phase == Phase.PHASE2:
            return self._step_phase2(action)
        else:
            return self._step_phase3(action)

    # -----------------------------------------------------------------------
    # Phase 1 step (mostly unchanged — adds weight output to FINALIZE)
    # -----------------------------------------------------------------------

    def _step_phase1(self, action):
        action_type = action.get('type', ActionType.ASSIGN_NODE)

        if action_type == ActionType.ASSIGN_NODE and not self._assignment_complete:
            node = self.node_order[self.current_node_idx]
            gid  = action['group_id']
            neighbor_groups = {
                self.assignment[n]
                for n in self.adjacency[node]
                if self.assignment[n] != -1
            }
            if gid in neighbor_groups:
                return self._get_state(), -0.1, False

            self.assignment[node] = gid
            if gid >= self.num_groups:
                self.num_groups = gid + 1
            self.current_node_idx += 1

            cur_internal = self._count_current_internal()
            reward       = self._internal_reward(cur_internal)
            self.prev_internal_count = cur_internal

            if self.current_node_idx >= len(self.nodes):
                self._assignment_complete = True

            return self._get_state(), reward, False

        elif action_type == ActionType.SWAP_NODE:
            node_a = action.get('node_a')
            node_b = action.get('node_b')
            if node_a is None or node_b is None:
                return self._get_state(), -0.05, False
            gid_a = self.assignment[node_a]
            gid_b = self.assignment[node_b]
            if gid_a == gid_b:
                return self._get_state(), -0.05, False
            self.assignment[node_a] = gid_b
            self.assignment[node_b] = gid_a
            if not self._check_assignment_valid():
                self.assignment[node_a] = gid_a
                self.assignment[node_b] = gid_b
                return self._get_state(), -0.1, False
            self._refinement_steps += 1
            cur_internal = self._count_current_internal()
            reward       = self._internal_reward(cur_internal)
            self.prev_internal_count = cur_internal
            return self._get_state(), reward, False

        elif action_type == ActionType.MOVE_NODE:
            node    = action.get('node')
            new_gid = action.get('group_id')
            if node is None or new_gid is None:
                return self._get_state(), -0.05, False
            old_gid = self.assignment[node]
            if old_gid == new_gid:
                return self._get_state(), -0.05, False
            self.assignment[node] = new_gid
            if new_gid >= self.num_groups:
                self.num_groups = new_gid + 1
            if not self._check_assignment_valid():
                self.assignment[node] = old_gid
                return self._get_state(), -0.1, False
            self._refinement_steps += 1
            cur_internal = self._count_current_internal()
            reward       = self._internal_reward(cur_internal)
            self.prev_internal_count = cur_internal
            return self._get_state(), reward, False

        elif action_type == ActionType.FINALIZE_PARTITION:
            if not self._assignment_complete:
                return self._get_state(), -0.5, False
            self.partition = self._build_partition()
            ipp   = internal_per_partition(self.partition, self.sessions)
            bonus = 0.2 * sum(ipp)

            # Store partition_weights from action (Phase 1 policy outputs them)
            self.partition_weights = action.get('weights', {})

            self._start_phase2()
            return self._get_state(), bonus, False

        return self._get_state(), 0.0, False

    # -----------------------------------------------------------------------
    # Phase 2 step (operates on per-node IOs; adds CROSS_SUBMOD)
    # -----------------------------------------------------------------------

    def _step_phase2(self, action):
        action_type = action['type']
        self.phase2_steps += 1
        worst_case  = len(self.edges) / max(len(self.sessions), 1)

        if self.phase2_steps > MAX_PHASE2_STEPS:
            return self._force_terminal_p2()

        self._cap_pool()

        if action_type == ActionType.ADD_TO_ACCUMULATOR:
            idx = action.get('idx_i', 0)
            if idx < len(self.pool):
                ineq = self.pool.pop(idx)
                if idx < self.num_base:
                    self.num_base -= 1
                self.accumulator.append(ineq)
            return self._get_state(), STEP_COST, False

        elif action_type == ActionType.APPLY_SUBMODULARITY:
            idx_i = action.get('idx_i', 0)
            idx_j = action.get('idx_j', 1)
            bonus = 0.0
            if (len(self.accumulator) >= 2
                    and idx_i < len(self.accumulator)
                    and idx_j < len(self.accumulator)
                    and idx_i != idx_j):
                a = self.accumulator[idx_i]
                b = self.accumulator[idx_j]
                union_ineq, inter_ineq = apply_pairwise_submodularity(
                    a, b, self.index, self.sessions
                )
                self.accumulator = [
                    ineq for k, ineq in enumerate(self.accumulator)
                    if k not in (idx_i, idx_j)
                ]
                self.pool.append(union_ineq)
                self.pool.append(inter_ineq)
                self.combination_log.append({
                    'step': self.phase2_steps, 'action': 'PAIRWISE',
                    'idx_i': idx_i, 'idx_j': idx_j,
                })
                if union_ineq.yi_coeff() > 0.5 and not self._found_yi_collapse:
                    self._found_yi_collapse = True
                    bonus = 0.3
            reward = STEP_COST + bonus + self._terminal_discovery_bonus_p2()
            return self._get_state(), reward, False

        elif action_type == ActionType.CROSS_SUBMOD:
            # Apply submodularity to two accumulator items from different
            # partition sets.  This is Phase 2's main learning target:
            # it must discover that summing IOs across partition boundaries
            # and then applying SUBMOD produces useful cancellations.
            idx_i = action.get('idx_i', 0)
            idx_j = action.get('idx_j', 1)
            bonus = 0.0
            if (len(self.accumulator) >= 2
                    and idx_i < len(self.accumulator)
                    and idx_j < len(self.accumulator)
                    and idx_i != idx_j):
                a = self.accumulator[idx_i]
                b = self.accumulator[idx_j]
                # Only reward if actually cross-partition
                a_parts = set(getattr(a, 'partition_ids', []))
                b_parts = set(getattr(b, 'partition_ids', []))
                is_cross = bool(a_parts and b_parts and not (a_parts & b_parts))
                union_ineq, inter_ineq = apply_pairwise_submodularity(
                    a, b, self.index, self.sessions
                )
                self.accumulator = [
                    ineq for k, ineq in enumerate(self.accumulator)
                    if k not in (idx_i, idx_j)
                ]
                self.pool.append(union_ineq)
                self.pool.append(inter_ineq)
                self.combination_log.append({
                    'step': self.phase2_steps, 'action': 'CROSS_SUBMOD',
                    'cross': is_cross,
                })
                if is_cross:
                    bonus = 0.4   # extra bonus for cross-partition
                if union_ineq.yi_coeff() > 0.5 and not self._found_yi_collapse:
                    self._found_yi_collapse = True
                    bonus += 0.3
            reward = STEP_COST + bonus + self._terminal_discovery_bonus_p2()
            return self._get_state(), reward, False

        elif action_type == ActionType.APPLY_PROOF2:
            if self._proof2_used:
                return self._get_state(), STEP_COST - 0.1, False
            self._proof2_used = True
            try:
                final = apply_n2_submodularity_all_at_once(
                    self.base_inequalities, self.index, self.sessions
                )
                self.pool.append(final)
            except Exception:
                pass
            self.combination_log.append({'step': self.phase2_steps, 'action': 'PROOF2'})
            best_bound = self._best_pool_bound()
            if best_bound is not None:
                return self._get_state(), -best_bound, True
            return self._get_state(), STEP_COST - 0.05, False

        elif action_type == ActionType.STORE_AND_RESET:
            if self.accumulator:
                combined = self.accumulator[0].copy()
                for ineq in self.accumulator[1:]:
                    combined = combined.add(ineq)
                self.stored_derived.append(combined)
                self.accumulator = []
            return self._get_state(), STEP_COST, False

        elif action_type == ActionType.COMBINE_STORED:
            idx_i = action.get('idx_i', 0)
            idx_j = action.get('idx_j', 1)
            if (idx_i < len(self.stored_derived)
                    and idx_j < len(self.stored_derived)
                    and idx_i != idx_j):
                combined = self.stored_derived[idx_i].add(self.stored_derived[idx_j])
                self.stored_derived = [
                    ineq for k, ineq in enumerate(self.stored_derived)
                    if k not in (idx_i, idx_j)
                ]
                self.pool.append(combined)
            return self._get_state(), STEP_COST, False

        elif action_type == ActionType.DECLARE_TERMINAL:
            if self.phase2_steps < self.min_phase2_steps:
                return self._get_state(), -worst_case, True
            best_bound = self._best_pool_bound()
            reward = -best_bound if best_bound is not None else -worst_case
            return self._get_state(), reward, True

        return self._get_state(), STEP_COST, False

    # -----------------------------------------------------------------------
    # Phase 3 step (fractional IO + joint search)
    # -----------------------------------------------------------------------

    def _step_phase3(self, action):
        """
        Phase 3: search for bounds that beat the partition bound.

        Reward is ONLY positive when extracted bound < partition_bound.
        Step cost is zero (agent learns episode length naturally).
        """
        action_type = action['type']
        self.phase3_steps += 1

        if self.phase3_steps > MAX_PHASE3_STEPS:
            return self._force_terminal_p3()

        if action_type == ActionType.FRACTIONAL_IO:
            # Form λ·IO(u) + (1-λ)·IO(v)
            node_u = action.get('node_u')
            node_v = action.get('node_v')
            lam    = action.get('lam', 0.5)
            if (node_u in self.node_ios and node_v in self.node_ios
                    and node_u != node_v and 0.0 < lam < 1.0):
                fi = generate_fractional_io(
                    node_u, node_v, lam,
                    self.partition, self.nodes, self.edges,
                    self.sessions, self.index
                )
                self.frac_pool.add(fi)
                # Bonus if cross-partition (escaping PB family)
                reward = 0.1 if fi.is_cross_partition() else 0.02
            else:
                reward = -0.1
            return self._get_state(), reward, False

        elif action_type == ActionType.ADD_TO_ACCUMULATOR:
            idx = action.get('idx_i', 0)
            if idx < len(self.frac_pool):
                self.accumulator.append(self.frac_pool[idx].copy())
                reward = 0.0
            else:
                reward = -0.1
            return self._get_state(), reward, False

        elif action_type in (ActionType.APPLY_SUBMODULARITY,
                             ActionType.CROSS_SUBMOD):
            idx_i = action.get('idx_i', 0)
            idx_j = action.get('idx_j', 1)
            reward = 0.0
            if (len(self.accumulator) >= 2
                    and idx_i < len(self.accumulator)
                    and idx_j < len(self.accumulator)
                    and idx_i != idx_j):
                a = self.accumulator[idx_i]
                b = self.accumulator[idx_j]
                union_ineq, inter_ineq = apply_pairwise_submodularity(
                    a, b, self.index, self.sessions
                )
                self.accumulator = [
                    ineq for k, ineq in enumerate(self.accumulator)
                    if k not in (idx_i, idx_j)
                ]
                # Promote to FractionalInequality so pool scoring works
                fu = make_fractional(
                    union_ineq, lam=1.0,
                    source_nodes  = getattr(a,'source_nodes',[]) + getattr(b,'source_nodes',[]),
                    partition_ids = getattr(a,'partition_ids',[]) + getattr(b,'partition_ids',[])
                )
                fi2 = make_fractional(
                    inter_ineq, lam=1.0,
                    source_nodes  = getattr(a,'source_nodes',[]) + getattr(b,'source_nodes',[]),
                    partition_ids = getattr(a,'partition_ids',[]) + getattr(b,'partition_ids',[])
                )
                self.frac_pool.add(fu)
                self.frac_pool.add(fi2)
                # Cross-partition bonus
                a_parts = set(getattr(a,'partition_ids',[]))
                b_parts = set(getattr(b,'partition_ids',[]))
                if a_parts and b_parts and not (a_parts & b_parts):
                    reward = 0.3   # cross-partition submod bonus
                else:
                    reward = 0.05
            return self._get_state(), reward, False

        elif action_type == ActionType.STORE_AND_RESET:
            if self.accumulator:
                combined = self.accumulator[0].copy()
                for ineq in self.accumulator[1:]:
                    combined = combined.add(ineq)
                combined_fi = make_fractional(combined, lam=1.0)
                self.frac_pool.add(combined_fi)
                self.accumulator = []
            return self._get_state(), 0.0, False

        elif action_type == ActionType.DECLARE_TERMINAL:
            return self._extract_phase3_bound()

        return self._get_state(), 0.0, False

    def _extract_phase3_bound(self) -> Tuple[Dict, float, bool]:
        """
        Extract the best bound from the fractional pool.

        Reward design (Phase 3 only):
          bound < partition_bound → POSITIVE: 5 + 20*(PB-bound)/PB
          bound = partition_bound → 1.0  (matched, no improvement)
          bound > partition_bound → negative, proportional to gap
        """
        pb = self.partition_bound
        best_bound = self.frac_pool.best_bound(
            len(self.sessions), len(self.edges), self.internal_per_part
        )
        if best_bound == float('inf'):
            # No terminal form found — heavy penalty
            return self._get_state(), -pb, True

        if best_bound < pb - 1e-8:
            # BEAT THE PARTITION BOUND
            improvement = (pb - best_bound) / pb
            reward = 5.0 + 20.0 * improvement
        elif abs(best_bound - pb) < 1e-6:
            reward = 1.0
        else:
            overshoot = (best_bound - pb) / max(pb, 1e-9)
            reward = max(-2.0, -overshoot * 2.0)

        return self._get_state(), reward, True

    # -----------------------------------------------------------------------
    # Shared helpers
    # -----------------------------------------------------------------------

    def _best_pool_bound(self) -> Optional[float]:
        best = None
        for ineq in self.pool + self.accumulator + self.stored_derived:
            if ineq.check_valid_terminal_form():
                b = ineq.extract_bound(
                    len(self.sessions), len(self.edges),
                    self.internal_per_part
                )
                if best is None or b < best:
                    best = b
        return best

    def _cap_pool(self):
        if len(self.pool) <= self.num_base + MAX_DERIVED:
            return
        base_part    = self.pool[:self.num_base]
        derived_part = self.pool[self.num_base:]

        def _score(ineq):
            c1  = ineq.get_yi_coefficient()
            c3  = ineq.get_rhs_edge_coefficient()
            if c1 > 0 and c3 > 0:
                return c1 / c3
            return 1.0 if ineq.check_valid_terminal_form() else 0.0

        derived_part.sort(key=_score, reverse=True)
        self.pool = base_part + derived_part[:MAX_DERIVED]

    def _current_step_cost(self) -> float:
        return STEP_COST_AFTER_TERMINAL if self._found_terminal else STEP_COST

    def _terminal_discovery_bonus_p2(self) -> float:
        if self._found_terminal:
            return 0.0
        for ineq in self.pool + self.accumulator + self.stored_derived:
            if ineq.check_valid_terminal_form():
                self._found_terminal = True
                return 0.5
        return 0.0

    def _force_terminal_p2(self):
        best = self._best_pool_bound()
        worst = len(self.edges) / max(len(self.sessions), 1)
        reward = -best if best is not None else -worst
        reward -= 0.5
        return self._get_state(), reward, True

    def _force_terminal_p3(self):
        state, reward, _ = self._extract_phase3_bound()
        reward -= 0.5   # timeout penalty
        return state, reward, True

    def _internal_reward(self, cur_internal):
        return 1.0 * (cur_internal - self.prev_internal_count)

    def _check_assignment_valid(self) -> bool:
        for u, v in self.edges:
            if (self.assignment.get(u, -1) != -1 and
                    self.assignment.get(v, -1) != -1 and
                    self.assignment[u] == self.assignment[v]):
                return False
        return True

    def _count_current_internal(self) -> int:
        groups = {}
        for node, gid in self.assignment.items():
            if gid != -1:
                groups.setdefault(gid, set()).add(node)
        count = 0
        for gset in groups.values():
            for s, t in self.sessions:
                if s in gset and t in gset:
                    count += 1
        return count

    def _build_partition(self) -> List[List[str]]:
        groups = {}
        for node, gid in self.assignment.items():
            groups.setdefault(gid, []).append(node)
        return list(groups.values())

    # -----------------------------------------------------------------------
    # Valid action enumeration
    # -----------------------------------------------------------------------

    def get_valid_actions(self) -> List[Dict]:
        if self.current_phase == Phase.PHASE1:
            return self._valid_phase1()
        elif self.current_phase == Phase.PHASE2:
            return self._valid_phase2()
        else:
            return self._valid_phase3()

    def _valid_phase1(self):
        valid = []
        if not self._assignment_complete:
            node = self.node_order[self.current_node_idx]
            neighbor_groups = {
                self.assignment[n]
                for n in self.adjacency[node]
                if self.assignment[n] != -1
            }
            for g in range(self.num_groups):
                if g not in neighbor_groups:
                    valid.append({'type': ActionType.ASSIGN_NODE, 'group_id': g})
            valid.append({'type': ActionType.ASSIGN_NODE, 'group_id': self.num_groups})
        else:
            if self._refinement_steps < MAX_REFINEMENT_STEPS:
                for s, t in self.sessions:
                    gid_s = self.assignment[s]; gid_t = self.assignment[t]
                    if gid_s != gid_t:
                        for n in self.nodes:
                            if self.assignment[n] == gid_s and n != s and n != t:
                                valid.append({'type': ActionType.SWAP_NODE,
                                              'node_a': t, 'node_b': n})
                            if self.assignment[n] == gid_t and n != s and n != t:
                                valid.append({'type': ActionType.SWAP_NODE,
                                              'node_a': s, 'node_b': n})
                        valid.append({'type': ActionType.MOVE_NODE, 'node': s, 'group_id': gid_t})
                        valid.append({'type': ActionType.MOVE_NODE, 'node': t, 'group_id': gid_s})
                        valid.append({'type': ActionType.MOVE_NODE, 'node': s, 'group_id': self.num_groups})
            seen = set(); unique = []
            for a in valid:
                k = str(sorted(a.items()))
                if k not in seen:
                    seen.add(k); unique.append(a)
            valid = unique
            valid.append({'type': ActionType.FINALIZE_PARTITION})
        return valid

    def _valid_phase2(self):
        valid = []
        for idx in range(len(self.pool)):
            valid.append({'type': ActionType.ADD_TO_ACCUMULATOR, 'idx_i': idx})

        k_acc = len(self.accumulator)
        for i in range(k_acc):
            for j in range(i+1, k_acc):
                valid.append({'type': ActionType.APPLY_SUBMODULARITY, 'idx_i': i, 'idx_j': j})
                # CROSS_SUBMOD: only offer if items are from different partitions
                a_parts = set(getattr(self.accumulator[i], 'partition_ids', []))
                b_parts = set(getattr(self.accumulator[j], 'partition_ids', []))
                if a_parts and b_parts and not (a_parts & b_parts):
                    valid.append({'type': ActionType.CROSS_SUBMOD, 'idx_i': i, 'idx_j': j})

        pairwise_done = sum(1 for e in self.combination_log if e.get('action') in ('PAIRWISE','CROSS_SUBMOD'))
        if pairwise_done >= 1 and not self._proof2_used:
            valid.append({'type': ActionType.APPLY_PROOF2})

        if self.phase2_steps >= 10:
            if self.accumulator:
                valid.append({'type': ActionType.STORE_AND_RESET})
            k_stored = len(self.stored_derived)
            for i in range(k_stored):
                for j in range(i+1, k_stored):
                    valid.append({'type': ActionType.COMBINE_STORED, 'idx_i': i, 'idx_j': j})

        if self.phase2_steps >= self.min_phase2_steps:
            valid.append({'type': ActionType.DECLARE_TERMINAL})
        return valid

    def _valid_phase3(self):
        """
        Phase 3 actions:
          FRACTIONAL_IO(u, v, λ): for each pair of nodes from different partitions
          ADD_TO_ACCUMULATOR(idx): from the fractional pool
          APPLY_SUBMODULARITY / CROSS_SUBMOD: on accumulator pairs
          STORE_AND_RESET: commit accumulator to pool
          DECLARE_TERMINAL: extract bound and end episode
        """
        valid = []

        # FRACTIONAL_IO: offer cross-partition pairs × lambda grid
        if len(self.partition) >= 2:
            part_of = {}
            for pid, Pi in enumerate(self.partition):
                for nd in Pi: part_of[nd] = pid

            # Sample a subset of cross-partition pairs to keep action space manageable
            cross_pairs = []
            nodes_list = list(self.nodes)
            for i, u in enumerate(nodes_list):
                for v in nodes_list[i+1:]:
                    if part_of.get(u, -1) != part_of.get(v, -1):
                        cross_pairs.append((u, v))

            # Limit to 20 pairs to avoid action explosion
            if len(cross_pairs) > 20:
                cross_pairs = random.sample(cross_pairs, 20)

            for (u, v) in cross_pairs:
                for lam in LAMBDA_GRID:
                    valid.append({
                        'type': ActionType.FRACTIONAL_IO,
                        'node_u': u, 'node_v': v, 'lam': lam
                    })

        # ADD from fractional pool
        for idx in range(len(self.frac_pool)):
            valid.append({'type': ActionType.ADD_TO_ACCUMULATOR, 'idx_i': idx})

        # SUBMOD on accumulator pairs
        k_acc = len(self.accumulator)
        for i in range(k_acc):
            for j in range(i+1, k_acc):
                valid.append({'type': ActionType.APPLY_SUBMODULARITY, 'idx_i': i, 'idx_j': j})
                a_parts = set(getattr(self.accumulator[i], 'partition_ids', []))
                b_parts = set(getattr(self.accumulator[j], 'partition_ids', []))
                if a_parts and b_parts and not (a_parts & b_parts):
                    valid.append({'type': ActionType.CROSS_SUBMOD, 'idx_i': i, 'idx_j': j})

        if self.accumulator:
            valid.append({'type': ActionType.STORE_AND_RESET})

        # Always allow terminal (Phase 3 has no step gate)
        valid.append({'type': ActionType.DECLARE_TERMINAL})
        return valid

    # -----------------------------------------------------------------------
    # State encoding
    # -----------------------------------------------------------------------

    def _get_state(self) -> Dict[str, Any]:
        state = {
            'phase'       : int(self.current_phase),
            'num_nodes'   : len(self.nodes),
            'num_edges'   : len(self.edges),
            'num_sessions': len(self.sessions),
            'stage'       : self.stage,
            'partition_bound': self.partition_bound,
        }
        if self.current_phase == Phase.PHASE1:
            state['current_node_idx']   = self.current_node_idx
            state['num_groups']         = self.num_groups
            state['assignment']         = dict(self.assignment)
            state['sessions']           = list(self.sessions)
            state['edges']              = list(self.edges)
            state['assignment_complete']= self._assignment_complete
            state['refinement_steps']   = self._refinement_steps
            state['internal_count']     = self._count_current_internal()
        elif self.current_phase == Phase.PHASE2:
            state['pool_size']          = len(self.pool)
            state['accumulator_size']   = len(self.accumulator)
            state['stored_derived_size']= len(self.stored_derived)
            state['phase2_steps']       = self.phase2_steps
            state['internal_sessions']  = self.internal_session_count
            state['combination_log']    = list(self.combination_log)
            if self.pool:
                base_part    = self.pool[:self.num_base]
                derived_part = self.pool[self.num_base:][-MAX_DERIVED:]
                pool_to_send = base_part + derived_part
                state['pool_coeffs'] = np.stack([ineq.coeffs for ineq in pool_to_send])
            if self.accumulator:
                state['accumulator_coeffs'] = np.stack(
                    [ineq.coeffs for ineq in self.accumulator[-10:]]
                )
        else:
            # Phase 3
            state['phase3_steps']       = self.phase3_steps
            state['frac_pool_size']     = len(self.frac_pool)
            state['accumulator_size']   = len(self.accumulator)
            state['has_cross_partition']= int(self.frac_pool.has_cross_partition())
            state['has_fractional_lam'] = int(self.frac_pool.has_fractional_lambda())
            state['partition_weights']  = dict(self.partition_weights)
            if len(self.frac_pool) > 0:
                state['pool_coeffs'] = self.frac_pool.coeff_matrix()
            if self.accumulator:
                state['accumulator_coeffs'] = np.stack(
                    [ineq.coeffs for ineq in self.accumulator[-10:]]
                )
            # Best bound so far in this episode
            best = self.frac_pool.best_bound(
                len(self.sessions), len(self.edges),
                self.internal_per_part or []
            )
            state['best_bound_so_far'] = best if best < 1e9 else -1.0

        return state