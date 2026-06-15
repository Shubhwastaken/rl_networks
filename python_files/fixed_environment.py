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
  NEW ACTIONS: APPLY_CRYPTO, APPLY_DECODE — functional dependence
  constraints that can tighten the bound beyond the partition bound.

PHASE 3 (new — the actual novel-inequality search):
  Starts from Phase 1's partition + weight hints and Phase 2's learned
  proof calculus. Action space adds:
    FRACTIONAL_IO(u, v, λ): form λ·IO(u) + (1-λ)·IO(v) atomically
    APPLY_CRYPTO(cut_idx):  crypto inequality for a precomputed cut
    APPLY_DECODE(session_idx): decoding functional dependence
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
from rl_functional_dep_integration import (
    FuncDepActions,
    is_crypto_valid,
    is_decode_valid,
    ACTION_CRYPTO,
    ACTION_DECODE,
)
from functional_dependence import (
    apply_crypto_inequality_direct,
    apply_decode_substitution,
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
    # Functional dependence (Phase 2 & 3)
    APPLY_CRYPTO         = 20   # crypto inequality for a precomputed cut
    APPLY_DECODE         = 21   # decoding functional dependence for a session


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
    """Returns the tightest partition bound for this graph.

    Delegates to compute_optimal_bound from fixed_graph_generation, which
    uses exhaustive k-coloring (k<=4 for n<=10, k<=3 for n<=14) plus
    exhaustive 2-partition search and greedy colorings.  This is strictly
    more thorough than the old local implementation, which only tried
    greedy colorings + exhaustive 2-partitions and therefore missed optimal
    partitions with 3 or 4 groups (e.g. paper_7N: 4-group opt PB=1.667
    was missed, returning 2.000 instead).

    The 6 affected graphs and their correct PBs:
      paper_7N            2.000 → 1.667  (4-group partition)
      grid_3x4_12N        5.667 → 2.833  (3-group partition)
      petersen_10N        2.143 → 1.875  (3-group partition)
      two_k4_10N          3.200 → 2.667  (4-group partition)
      al_bashabsheh_7N    2.400 → 2.000  (4-group partition)
      kramer_savari_ladder_8N  4.000 → 2.000  (3-group partition)

    Using the wrong (inflated) PB as the agent's baseline in Stage 4 meant:
      - env.partition_bound was set too high
      - The agent received reward=1.0 for merely matching an inflated baseline
      - Genuinely novel bounds below true PB but above inflated PB were
        never flagged as novel
      - _find_optimal_partition also returned the inflated partition, so
        the wrong starting point was used for Phase 3 exploration
    """
    from fixed_graph_generation import compute_optimal_bound
    best_bound, _, _ = compute_optimal_bound(nodes, edges, sessions)
    return best_bound


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

        # Functional dependence action catalogue (built in _start_phase2)
        self.func_dep_actions: Optional[FuncDepActions] = None

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
        self.func_dep_actions  = None

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

        # Build functional dependence action catalogue for this graph
        self.func_dep_actions = FuncDepActions(
            list(self.nodes), list(self.edges), list(self.sessions),
            max_crypto_cuts=20
        )

    def _start_phase3(self, preseed: bool = False):
        """Set up Phase 3: fractional IO search starting from Phase 1/2 knowledge.

        Args:
            preseed: If True, add a pre-built terminal-form inequality
                     (≈ PB) to frac_pool as a starting signal.
                     Defaults to False so the agent must earn a finite
                     bound by exploring FRACTIONAL_IO → STORE_AND_RESET
                     rather than immediately satisfying the stopper with
                     reward = 1.0 every episode.
        """
        if self.index is None:
            self._start_phase2()   # ensure index is built

        # Populate frac_pool with all per-node IOs at weight 1.0
        self.frac_pool = FractionalPool(MAX_DERIVED)
        for fi in self.node_ios.values():
            self.frac_pool.add(fi.copy())

        # Also add partition-level IOs (from Phase 2 standard proof)
        for bi in self.base_inequalities:
            self.frac_pool.add(bi)

        # Pre-seed with terminal-form baseline (n2 submod result).
        # Only used when explicitly requested (preseed=True).  Keeping it
        # off by default forces the agent to explore the multi-step
        # FRACTIONAL_IO → ADD_TO_ACCUMULATOR → CROSS_SUBMOD →
        # STORE_AND_RESET → DECLARE_TERMINAL sequence instead of
        # short-circuiting to reward=1.0 via the seed every episode.
        if preseed:
            try:
                from fixed_submodularity import apply_n2_submodularity_all_at_once
                terminal_baseline = apply_n2_submodularity_all_at_once(
                    self.base_inequalities, self.index, self.sessions
                )
                self.frac_pool.add(make_fractional(terminal_baseline))
            except Exception:
                pass  # non-critical — partition IOs still available

        self.current_phase  = Phase.PHASE3
        self.phase3_steps   = 0
        self.accumulator    = []
        self.stored_derived = []
        self._found_terminal = False
        self.combination_log = []
        self.phase3_used_cross_submod = False
        self.phase3_used_plain_submod = False
        self.phase3_used_crypto = False
        self.phase3_used_decode = False
        # Potential-based shaping: track best bound seen so far this episode.
        # Any improvement triggers a small reward proportional to the gap closed.
        self._phase3_best_bound = float('inf')

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
    # Phase 2 step (operates on per-node IOs; adds CROSS_SUBMOD + CRYPTO/DECODE)
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

        elif action_type == ActionType.APPLY_CRYPTO:
            # Apply crypto inequality: h(Y_sep | U_cut) = 0
            # tightens the bound when cut edges are already on RHS
            cut_idx = action.get('cut_idx', 0)
            reward  = -0.1
            if (self.func_dep_actions is not None
                    and cut_idx < self.func_dep_actions.num_crypto_cuts()):
                vp, sep_list = self.func_dep_actions.crypto_cut(cut_idx)
                # Apply to the most recently derived terminal-form inequality,
                # or the last pool item if nothing is in terminal form yet
                target = None
                for ineq in reversed(self.pool):
                    if ineq.check_valid_terminal_form():
                        target = ineq
                        break
                if target is None and self.pool:
                    target = self.pool[-1]
                if target is not None:
                    new_ineq, applied = apply_crypto_inequality_direct(
                        target, set(vp),
                        list(self.nodes), list(self.edges),
                        list(self.sessions), self.index
                    )
                    if applied:
                        self.pool.append(new_ineq)
                        self.combination_log.append({
                            'step': self.phase2_steps, 'action': 'CRYPTO',
                            'cut_idx': cut_idx, 'sep_count': len(sep_list)
                        })
                        # Stronger bonus if the new inequality is already terminal
                        reward = 0.4 if new_ineq.check_valid_terminal_form() else 0.1
                    else:
                        reward = -0.05
            return self._get_state(), reward, False

        elif action_type == ActionType.APPLY_DECODE:
            # Apply decoding substitution: h(Y_i | edges into t(i)) = 0
            # tightens the bound when sink's incident edges are on RHS
            si     = action.get('session_idx', 0)
            reward = -0.1
            if (self.func_dep_actions is not None
                    and si < len(self.sessions)):
                target = None
                for ineq in reversed(self.pool):
                    if ineq.check_valid_terminal_form():
                        target = ineq
                        break
                if target is None and self.pool:
                    target = self.pool[-1]
                if target is not None:
                    new_ineq, applied = apply_decode_substitution(
                        target, si,
                        list(self.sessions), list(self.edges), self.index
                    )
                    if applied:
                        self.pool.append(new_ineq)
                        self.combination_log.append({
                            'step': self.phase2_steps, 'action': 'DECODE',
                            'session_idx': si
                        })
                        reward = 0.3 if new_ineq.check_valid_terminal_form() else 0.08
                    else:
                        reward = -0.05
            return self._get_state(), reward, False

        elif action_type == ActionType.DECLARE_TERMINAL:
            if self.phase2_steps < self.min_phase2_steps:
                return self._get_state(), -worst_case, True
            best_bound = self._best_pool_bound()
            reward = -best_bound if best_bound is not None else -worst_case
            return self._get_state(), reward, True

        return self._get_state(), STEP_COST, False

    # -----------------------------------------------------------------------
    # Phase 3 step (fractional IO + joint search + crypto/decode)
    # -----------------------------------------------------------------------

    def _pool_improvement_bonus(self) -> float:
        """Potential-based shaping: reward proportional to best_bound improvement.

        Each time an action causes frac_pool.best_bound() to drop below the
        previous episode minimum, we give a small bonus = 0.05 * improvement_fraction.
        This gives dense credit to the multi-step sequence that progressively
        tightens the bound, without distorting the terminal reward scale.

        BUG FIX: When _phase3_best_bound is float('inf') (no valid terminal form
        found yet) the improvement fraction (inf - new_best) / pb is inf, which
        propagates into total_reward and poisons np.mean(rewards) with inf values
        in the Stage 4 logging.  Guard by returning 0.0 whenever the previous
        best was infinite — the first finite terminal form is its own reward.
        """
        if not hasattr(self, '_phase3_best_bound'):
            self._phase3_best_bound = float('inf')
        new_best = self.frac_pool.best_bound(
            len(self.sessions), len(self.edges), self.internal_per_part or []
        )
        # Guard 1: no reward when we have no finite baseline to improve upon.
        if self._phase3_best_bound >= 1e9:
            # Still update the tracker so the *next* improvement can be measured.
            if new_best < 1e9:
                self._phase3_best_bound = new_best
            return 0.0
        # Guard 2: new_best must be finite and a genuine improvement.
        if new_best < self._phase3_best_bound - 1e-9 and new_best < 1e9:
            improvement = (self._phase3_best_bound - new_best) / max(self.partition_bound, 1e-9)
            self._phase3_best_bound = new_best
            return 0.05 * min(improvement, 10.0)   # cap at 0.5 to avoid reward spikes
        return 0.0

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
                # Fix B: shaping reward for novel cross-partition FIOs.
                # A cross-partition fractional IO with a non-trivial λ is
                # the building block needed to escape the PB family.
                # Reward it only when the result is genuinely new in the
                # pool (deduplication prevents padding with clones).
                is_novel_cross = (
                    fi.is_cross_partition()
                    and not self.frac_pool.contains_equivalent(fi)
                )
                if is_novel_cross:
                    reward = 0.15   # meaningful signal; cross-partition + novel
                elif fi.is_cross_partition():
                    reward = 0.05   # cross-partition but already in pool
                else:
                    reward = 0.02   # same-partition (low value)
                self.frac_pool.add(fi)
                reward += self._pool_improvement_bonus()
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
            if action_type == ActionType.CROSS_SUBMOD:
                self.phase3_used_cross_submod = True
            else:
                self.phase3_used_plain_submod = True
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
                # Reward based on what the combination can actually produce.
                # The only productive CROSS_SUBMOD case is mixed: one input
                # has Y_ST terms (partition IO) and the other has Y_I directly
                # (node IO). This triggers _collapse_to_yi_if_valid in the
                # union, combining partition session coverage with the node
                # IO's edge subset — the one path that can produce sub-PB
                # bounds via submodularity.
                #
                # Pure cross-partition node IO pairs (both without Y_ST) are
                # a no-op after the min() fix — rewarding them misled the agent.
                a_has_yst = bool(a.active_yst())
                b_has_yst = bool(b.active_yst())
                a_parts   = set(getattr(a, 'partition_ids', []))
                b_parts   = set(getattr(b, 'partition_ids', []))
                is_mixed  = a_has_yst != b_has_yst   # one partition IO + one node IO
                is_cross  = a_parts and b_parts and not (a_parts & b_parts)
                if is_mixed:
                    reward = 0.8   # genuine collapse opportunity — doubled to compete with timeout risk
                elif is_cross:
                    reward = 0.1   # cross-partition but same type, limited value
                else:
                    reward = 0.02  # same partition, same type — minimal
                reward += self._pool_improvement_bonus()
            return self._get_state(), reward, False

        elif action_type == ActionType.STORE_AND_RESET:
            if self.accumulator:
                combined = self.accumulator[0].copy()
                for ineq in self.accumulator[1:]:
                    combined = combined.add(ineq)
                # Try source cancellation on the combined result.
                # When accumulated node IOs cover all session sources,
                # this zeros the source terms and adjusts Y_I, enabling
                # the combined inequality to reach valid terminal form.
                combined = combined.cancel_source_terms()
                combined_fi = make_fractional(combined, lam=1.0)
                self.frac_pool.add(combined_fi)

                # Fix C: shaping reward when the committed accumulator
                # contains cross-partition content.  This is the step
                # that "banks" a multi-step FRACTIONAL_IO sequence into
                # the pool — exactly the behaviour we want to encourage.
                # Check each item: if any came from different partitions
                # (has partition_ids spanning ≥2 groups), reward the commit.
                acc_part_ids = set()
                for a in self.accumulator:
                    acc_part_ids.update(getattr(a, 'partition_ids', []))
                has_cross_content = len(acc_part_ids) >= 2
                self.accumulator = []
                reward = 0.12 if has_cross_content else 0.0
                reward += self._pool_improvement_bonus()
            else:
                reward = 0.0
            return self._get_state(), reward, False

        elif action_type == ActionType.APPLY_CRYPTO:
            self.phase3_used_crypto = True
            # Apply crypto inequality to every terminal-form inequality in frac_pool
            cut_idx = action.get('cut_idx', 0)
            reward  = -0.1
            if (self.func_dep_actions is not None
                    and cut_idx < self.func_dep_actions.num_crypto_cuts()):
                vp, sep_list = self.func_dep_actions.crypto_cut(cut_idx)
                applied_any = False
                for ineq in list(self.frac_pool):
                    if not ineq.check_valid_terminal_form():
                        continue
                    new_ineq, applied = apply_crypto_inequality_direct(
                        ineq, set(vp),
                        list(self.nodes), list(self.edges),
                        list(self.sessions), self.index
                    )
                    if applied:
                        self.frac_pool.add(make_fractional(new_ineq))
                        applied_any = True
                # Reward based on whether applying crypto improved best bound
                if applied_any:
                    pb        = self.partition_bound
                    new_best  = self.frac_pool.best_bound(
                        len(self.sessions), len(self.edges), self.internal_per_part
                    )
                    if new_best < pb - 1e-8:
                        improvement = (pb - new_best) / pb
                        reward = 1.0 + 3.0 * improvement  # reduced to balance vs CROSS_SUBMOD
                    else:
                        reward = 0.2
                else:
                    reward = -0.05
            return self._get_state(), reward, False

        elif action_type == ActionType.APPLY_DECODE:
            self.phase3_used_decode = True
            # Apply decoding substitution to terminal-form inequalities in frac_pool
            si     = action.get('session_idx', 0)
            reward = -0.1
            if (self.func_dep_actions is not None
                    and si < len(self.sessions)):
                applied_any = False
                for ineq in list(self.frac_pool):
                    if not ineq.check_valid_terminal_form():
                        continue
                    new_ineq, applied = apply_decode_substitution(
                        ineq, si,
                        list(self.sessions), list(self.edges), self.index
                    )
                    if applied:
                        self.frac_pool.add(make_fractional(new_ineq))
                        applied_any = True
                if applied_any:
                    pb       = self.partition_bound
                    new_best = self.frac_pool.best_bound(
                        len(self.sessions), len(self.edges), self.internal_per_part
                    )
                    if new_best < pb - 1e-8:
                        improvement = (pb - new_best) / pb
                        reward = 1.0 + 3.0 * improvement  # reduced to balance vs CROSS_SUBMOD
                    else:
                        reward = 0.15
                else:
                    reward = -0.05
            return self._get_state(), reward, False

        elif action_type == ActionType.DECLARE_TERMINAL:
            return self._extract_phase3_bound()

        return self._get_state(), 0.0, False

    def _extract_phase3_bound(self) -> Tuple[Dict, float, bool]:
        """
        Extract the best bound from the fractional pool.

        Reward design (Phase 3 only):
          bound < lp_lower_bound  → heavy penalty (mathematically impossible)
          bound < partition_bound → POSITIVE: 5 + 20*(PB-bound)/PB
          bound = partition_bound → 1.0  (matched, no improvement)
          bound > partition_bound → negative, proportional to gap

        LP floor guard: _lp_lower_bound is set by run_stage4() on the env
        before each episode.  If not set (Stage 1-3 episodes), the guard
        is skipped via the getattr default of 0.0.  A sub-LP bound is
        mathematically impossible (it would violate the max-flow lower
        bound), so we penalise it identically to finding no terminal form
        at all.  This prevents the agent from learning to chase sub-LP
        results that produce large positive rewards but are later discarded
        by the post-hoc clamp in run_stage4(), which creates a silent
        reward poisoning problem.
        """
        pb = self.partition_bound
        best_bound = self.frac_pool.best_bound(
            len(self.sessions), len(self.edges), self.internal_per_part
        )
        if best_bound == float('inf'):
            # No terminal form found — heavy penalty
            return self._get_state(), -pb, True

        # ── LP floor guard ───────────────────────────────────────────────────
        # A bound below the LP lower bound is mathematically invalid.
        # Penalise it as heavily as finding no terminal form so the agent
        # never learns to reproduce the action sequences that cause it.
        lp_lb = getattr(self, '_lp_lower_bound', 0.0)
        if lp_lb > 0.0 and best_bound < lp_lb - 1e-9:
            return self._get_state(), -pb, True
        # ─────────────────────────────────────────────────────────────────────

        if best_bound < pb - 1e-8:
            # BEAT THE PARTITION BOUND
            improvement = (pb - best_bound) / pb
            # Bonus for using CROSS_SUBMOD chain — promotes the fractional path
            # over pure crypto/decode. Both paths are valid but CROSS_SUBMOD
            # produces structurally richer inequalities worth encouraging.
            used_cross = bool(getattr(self, 'phase3_used_cross_submod', False))
            if used_cross:
                reward = 7.0 + 20.0 * improvement  # higher bonus for cross_submod path
            else:
                reward = 5.0 + 15.0 * improvement  # still good for crypto/decode path
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

        # Crypto and decode: offer when we have at least one pool inequality
        # and the functional dependence catalogue is ready
        if self.func_dep_actions is not None and self.pool:
            # Find the best terminal-form inequality to check validity against
            target = None
            for ineq in reversed(self.pool):
                if ineq.check_valid_terminal_form():
                    target = ineq
                    break
            if target is None:
                target = self.pool[-1]

            for ci in range(self.func_dep_actions.num_crypto_cuts()):
                vp, sep_list = self.func_dep_actions.crypto_cut(ci)
                if is_crypto_valid(target, vp,
                                   list(self.nodes), list(self.edges),
                                   list(self.sessions), self.index):
                    valid.append({
                        'type': ActionType.APPLY_CRYPTO,
                        'cut_idx': ci,
                        'sep_count': len(sep_list)
                    })

            for si in range(len(self.sessions)):
                incoming = self.func_dep_actions.sink_incoming[si]
                if is_decode_valid(target, si, incoming, self.index):
                    valid.append({
                        'type': ActionType.APPLY_DECODE,
                        'session_idx': si
                    })

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
          APPLY_CRYPTO(cut_idx): crypto inequality for a precomputed cut
          APPLY_DECODE(session_idx): decode functional dependence
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
                a_has_yst = bool(self.accumulator[i].active_yst())
                b_has_yst = bool(self.accumulator[j].active_yst())
                a_parts   = set(getattr(self.accumulator[i], 'partition_ids', []))
                b_parts   = set(getattr(self.accumulator[j], 'partition_ids', []))
                is_mixed  = a_has_yst != b_has_yst  # one partition IO + one node IO
                is_cross  = a_parts and b_parts and not (a_parts & b_parts)
                # Offer CROSS_SUBMOD for the genuinely productive cases:
                # mixed (partition IO + node IO) triggers collapse pathway;
                # cross-partition pairs are worth exploring even if weaker.
                if is_mixed or is_cross:
                    valid.append({'type': ActionType.CROSS_SUBMOD, 'idx_i': i, 'idx_j': j})

        if self.accumulator:
            valid.append({'type': ActionType.STORE_AND_RESET})

        # Crypto and decode: offer once per cut/session if any pool item is valid
        if self.func_dep_actions is not None:
            seen_crypto = set()
            seen_decode = set()
            for ineq in self.frac_pool:
                if not ineq.check_valid_terminal_form():
                    continue
                for ci in range(self.func_dep_actions.num_crypto_cuts()):
                    if ci in seen_crypto:
                        continue
                    vp, sep_list = self.func_dep_actions.crypto_cut(ci)
                    if is_crypto_valid(ineq, vp,
                                       list(self.nodes), list(self.edges),
                                       list(self.sessions), self.index):
                        valid.append({
                            'type': ActionType.APPLY_CRYPTO,
                            'cut_idx': ci,
                            'sep_count': len(sep_list)
                        })
                        seen_crypto.add(ci)
                for si in range(len(self.sessions)):
                    if si in seen_decode:
                        continue
                    incoming = self.func_dep_actions.sink_incoming[si]
                    if is_decode_valid(ineq, si, incoming, self.index):
                        valid.append({
                            'type': ActionType.APPLY_DECODE,
                            'session_idx': si
                        })
                        seen_decode.add(si)

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
            state['num_crypto_cuts']    = (self.func_dep_actions.num_crypto_cuts()
                                           if self.func_dep_actions else 0)
            state['num_decode_actions'] = len(self.sessions)
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
            state['num_crypto_cuts']    = (self.func_dep_actions.num_crypto_cuts()
                                           if self.func_dep_actions else 0)
            state['num_decode_actions'] = len(self.sessions)
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
