"""
RL Environment for the partition bound problem.

MAJOR FIXES in this version:
1. Phase 1 now has SWAP_NODE and MOVE_NODE actions so the agent can
   fix mistakes after the initial assignment pass completes.
   - ASSIGN_NODE: used during the initial one-pass assignment
   - SWAP_NODE: swap two nodes between their groups (post-assignment)
   - MOVE_NODE: move a node to a different group (post-assignment)
   Phase 1 ends when agent calls FINALIZE_PARTITION.

2. Quality-based pool eviction: derived inequalities are ranked by
   their c1/c3 ratio (higher = closer to good bound), not FIFO.

3. Graph diversity: dataset now cycles through multiple graph types.

4. Consistent reward scheme throughout.

5. Seed set once at entry point, not at import time.
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
from fixed_inequality import Inequality, EntropyIndex
from fixed_base_inequality_generator import (
    generate_base_inequalities,
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


class ActionType(IntEnum):
    ASSIGN_NODE          = 0
    ADD_TO_ACCUMULATOR   = 1
    APPLY_SUBMODULARITY  = 2
    APPLY_PROOF2         = 3
    STORE_AND_RESET      = 4
    COMBINE_STORED       = 5
    DECLARE_TERMINAL     = 6
    SWAP_NODE            = 7   # NEW: swap two nodes between groups
    MOVE_NODE            = 8   # NEW: move a node to a different group
    FINALIZE_PARTITION   = 9   # NEW: end Phase 1, start Phase 2


MIN_PHASE2_STEPS     = 4
MAX_PHASE2_STEPS     = 50    # hard cap to prevent runaway episodes
MAX_DERIVED          = 15
MAX_REFINEMENT_STEPS = 20
STEP_COST            = -0.02 # small penalty per Phase 2 step


class PartitionBoundEnv:

    def __init__(self, graph_dataset_size: int = 5, stage: int = 3):
        self.stage = stage

        print(f"Generating graph dataset ({graph_dataset_size} graphs)...")
        self.graph_dataset = generate_graph_dataset(graph_dataset_size)
        print(f"Dataset ready: {len(self.graph_dataset)} graphs.")

        self.nodes    = None
        self.edges    = None
        self.sessions = None
        self.index    = None

        self.assignment       = None
        self.num_groups       = 0
        self.node_order       = None
        self.current_node_idx = 0
        self.adjacency        = None
        self.edge_set         = None

        self.base_inequalities  = None
        self.num_base           = 0
        self.pool               = None
        self.accumulator        = None
        self.stored_derived     = None
        self.phase2_steps       = 0
        self.partition          = None
        self.internal_per_part  = None

        self.current_phase          = Phase.PHASE1
        self.internal_session_count = 0
        self.prev_internal_count    = 0

        # Phase 1 refinement state
        self._assignment_complete = False
        self._refinement_steps    = 0

    def reset(self, fixed_partition=None, fixed_graph=None) -> Dict[str, Any]:
        """
        Reset the environment for a new episode.

        Args:
            fixed_partition: if provided, skip Phase 1 and use this partition
            fixed_graph: if provided, use this (nodes, edges, sessions) tuple
                         instead of random selection. MUST be used together
                         with fixed_partition to avoid graph/partition mismatch.
        """
        if fixed_graph is not None:
            self.nodes, self.edges, self.sessions = fixed_graph
        else:
            self.nodes, self.edges, self.sessions = random.choice(self.graph_dataset)

        self.adjacency = {n: set() for n in self.nodes}
        self.edge_set  = set()
        for u, v in self.edges:
            self.adjacency[u].add(v)
            self.adjacency[v].add(u)
            self.edge_set.add((u, v))
            self.edge_set.add((v, u))

        self.assignment       = {n: -1 for n in self.nodes}
        self.num_groups       = 0
        self.node_order       = self.nodes[:]
        random.shuffle(self.node_order)
        self.current_node_idx = 0
        self.prev_internal_count = 0

        self.pool           = []
        self.accumulator    = []
        self.stored_derived = []
        self.phase2_steps   = 0
        self.partition      = None
        self.internal_per_part = None
        self.num_base          = 0

        self._assignment_complete = False
        self._refinement_steps    = 0

        if fixed_partition is not None:
            self.partition = fixed_partition
            self._start_phase2()
            self.current_phase = Phase.PHASE2
        else:
            self.current_phase = Phase.PHASE1

        return self._get_state()

    def _start_phase2(self):
        self.index = EntropyIndex(
            partitions=self.partition,
            nodes=self.nodes,
            edges=self.edges,
            sessions=self.sessions
        )

        self.base_inequalities = generate_base_inequalities(
            self.partition, self.nodes, self.edges,
            self.sessions, self.index
        )

        self.pool     = [ineq.copy() for ineq in self.base_inequalities]
        self.num_base = len(self.base_inequalities)

        self.internal_per_part      = internal_per_partition(
            self.partition, self.sessions
        )
        self.internal_session_count = sum(self.internal_per_part)

        self.current_phase    = Phase.PHASE2
        self.phase2_steps     = 0
        self.combination_log  = []
        self._found_terminal  = False  # one-time bonus when first terminal form found
        self._proof2_used     = False  # limit PROOF2 to once per episode

    def _cap_pool(self):
        """
        Quality-based eviction: keeps base inequalities intact,
        ranks derived by c1/c3 ratio (higher = better), keeps top MAX_DERIVED.
        """
        if len(self.pool) <= self.num_base + MAX_DERIVED:
            return

        base_part    = self.pool[:self.num_base]
        derived_part = self.pool[self.num_base:]

        def _score(ineq):
            c1 = ineq.get_yi_coefficient()
            c3 = ineq.get_rhs_edge_coefficient()
            if c1 > 0 and c3 > 0:
                return c1 / c3
            if ineq.check_valid_terminal_form():
                return 1.0
            return 0.0

        derived_part.sort(key=_score, reverse=True)
        self.pool = base_part + derived_part[:MAX_DERIVED]

    def step(self, action: Dict[str, Any]) -> Tuple[Dict, float, bool]:
        if self.current_phase == Phase.PHASE1:
            return self._step_phase1(action)
        return self._step_phase2(action)

    def _step_phase1(self, action):
        action_type = action.get('type', ActionType.ASSIGN_NODE)

        # --- Initial assignment pass ---
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
            reward = self._internal_reward(cur_internal)
            self.prev_internal_count = cur_internal

            if self.current_node_idx >= len(self.nodes):
                self._assignment_complete = True

            return self._get_state(), reward, False

        # --- SWAP_NODE: swap two nodes between their groups ---
        elif action_type == ActionType.SWAP_NODE:
            node_a = action.get('node_a')
            node_b = action.get('node_b')
            if node_a is None or node_b is None:
                return self._get_state(), -0.05, False

            gid_a = self.assignment[node_a]
            gid_b = self.assignment[node_b]

            if gid_a == gid_b:
                return self._get_state(), -0.05, False

            # check validity after swap
            self.assignment[node_a] = gid_b
            self.assignment[node_b] = gid_a

            if not self._check_assignment_valid():
                # revert
                self.assignment[node_a] = gid_a
                self.assignment[node_b] = gid_b
                return self._get_state(), -0.1, False

            self._refinement_steps += 1
            cur_internal = self._count_current_internal()
            reward = self._internal_reward(cur_internal)
            self.prev_internal_count = cur_internal
            return self._get_state(), reward, False

        # --- MOVE_NODE: move a node to a different group ---
        elif action_type == ActionType.MOVE_NODE:
            node = action.get('node')
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
            reward = self._internal_reward(cur_internal)
            self.prev_internal_count = cur_internal
            return self._get_state(), reward, False

        # --- FINALIZE_PARTITION: end Phase 1 ---
        elif action_type == ActionType.FINALIZE_PARTITION:
            if not self._assignment_complete:
                return self._get_state(), -0.5, False

            self.partition = self._build_partition()

            # bonus for internal sessions
            ipp = internal_per_partition(self.partition, self.sessions)
            bonus = 0.5 * sum(ipp)

            self._start_phase2()
            return self._get_state(), bonus, False

        return self._get_state(), 0.0, False

    def _internal_reward(self, cur_internal):
        """Reward based on improvement in internal session count."""
        delta = cur_internal - self.prev_internal_count
        return 1.0 * delta

    def _check_assignment_valid(self) -> bool:
        """Check that no two adjacent nodes share a group."""
        for u, v in self.edges:
            if (self.assignment[u] != -1 and
                self.assignment[v] != -1 and
                self.assignment[u] == self.assignment[v]):
                return False
        return True

    def _step_phase2(self, action):
        action_type      = action['type']
        self.phase2_steps += 1
        worst_case_bound  = len(self.edges) / max(len(self.sessions), 1)

        # Force termination if over step limit
        if self.phase2_steps > MAX_PHASE2_STEPS:
            return self._force_terminal()

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
                    'step'   : self.phase2_steps,
                    'action' : 'PAIRWISE',
                    'idx_i'  : idx_i,
                    'idx_j'  : idx_j,
                    'yi_a'   : round(a.yi_coeff(), 4),
                    'yi_b'   : round(b.yi_coeff(), 4),
                    'yi_union': round(union_ineq.yi_coeff(), 4),
                })
                # One-time bonus when submodularity produces a Y_I collapse
                if union_ineq.yi_coeff() > 0.5:
                    bonus = 0.2
            reward = STEP_COST + bonus + self._terminal_discovery_bonus()
            return self._get_state(), reward, False

        elif action_type == ActionType.APPLY_PROOF2:
            if self._proof2_used:
                # Already used this episode — penalize and skip
                return self._get_state(), STEP_COST - 0.1, False
            self._proof2_used = True
            PROOF2_PENALTY = 0.3
            try:
                final = apply_n2_submodularity_all_at_once(
                    self.base_inequalities, self.index, self.sessions
                )
                self.pool.append(final)
            except Exception:
                pass
            self.combination_log.append({
                'step'  : self.phase2_steps,
                'action': 'PROOF2',
            })
            reward = STEP_COST - PROOF2_PENALTY + self._terminal_discovery_bonus()
            return self._get_state(), reward, False

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
                combined = self.stored_derived[idx_i].add(
                    self.stored_derived[idx_j]
                )
                self.stored_derived = [
                    ineq for k, ineq in enumerate(self.stored_derived)
                    if k not in (idx_i, idx_j)
                ]
                self.pool.append(combined)
            return self._get_state(), STEP_COST, False

        elif action_type == ActionType.DECLARE_TERMINAL:
            if self.phase2_steps < MIN_PHASE2_STEPS:
                return self._get_state(), -worst_case_bound, True

            best_bound = None
            for ineq in self.pool + self.accumulator + self.stored_derived:
                if ineq.check_valid_terminal_form():
                    bound = ineq.extract_bound(
                        len(self.sessions),
                        len(self.edges),
                        self.internal_per_part
                    )
                    if best_bound is None or bound < best_bound:
                        best_bound = bound

            reward = -best_bound if best_bound is not None else -worst_case_bound
            return self._get_state(), reward, True

        return self._get_state(), STEP_COST, False

    def _terminal_discovery_bonus(self) -> float:
        """One-time bonus when a valid terminal form is first discovered."""
        if self._found_terminal:
            return 0.0
        for ineq in self.pool + self.accumulator + self.stored_derived:
            if ineq.check_valid_terminal_form():
                self._found_terminal = True
                return 0.3  # one-time bonus
        return 0.0

    def _force_terminal(self):
        """Force episode end at step cap — evaluate best bound found."""
        best_bound = None
        for ineq in self.pool + self.accumulator + self.stored_derived:
            if ineq.check_valid_terminal_form():
                bound = ineq.extract_bound(
                    len(self.sessions),
                    len(self.edges),
                    self.internal_per_part
                )
                if best_bound is None or bound < best_bound:
                    best_bound = bound
        worst_case = len(self.edges) / max(len(self.sessions), 1)
        reward = -best_bound if best_bound is not None else -worst_case
        # Extra penalty for hitting the cap (should have terminated earlier)
        reward -= 0.5
        return self._get_state(), reward, True

    def get_valid_actions(self) -> List[Dict]:
        valid = []

        if self.current_phase == Phase.PHASE1:
            if not self._assignment_complete:
                # Still assigning nodes one by one
                node = self.node_order[self.current_node_idx]
                neighbor_groups = {
                    self.assignment[n]
                    for n in self.adjacency[node]
                    if self.assignment[n] != -1
                }
                for g in range(self.num_groups):
                    if g not in neighbor_groups:
                        valid.append({
                            'type': ActionType.ASSIGN_NODE,
                            'group_id': g
                        })
                valid.append({
                    'type': ActionType.ASSIGN_NODE,
                    'group_id': self.num_groups
                })
            else:
                # Assignment complete — refinement phase
                if self._refinement_steps < MAX_REFINEMENT_STEPS:
                    # SWAP_NODE actions: for each session pair (s,t),
                    # offer swaps that would bring them together
                    for s, t in self.sessions:
                        gid_s = self.assignment[s]
                        gid_t = self.assignment[t]
                        if gid_s != gid_t:
                            # try swapping t with each node in s's group
                            for n in self.nodes:
                                if (self.assignment[n] == gid_s
                                        and n != s
                                        and n != t):
                                    valid.append({
                                        'type': ActionType.SWAP_NODE,
                                        'node_a': t,
                                        'node_b': n,
                                    })
                            # try swapping s with each node in t's group
                            for n in self.nodes:
                                if (self.assignment[n] == gid_t
                                        and n != s
                                        and n != t):
                                    valid.append({
                                        'type': ActionType.SWAP_NODE,
                                        'node_a': s,
                                        'node_b': n,
                                    })

                    # MOVE_NODE actions: move session endpoints to partner's group
                    for s, t in self.sessions:
                        gid_s = self.assignment[s]
                        gid_t = self.assignment[t]
                        if gid_s != gid_t:
                            # move s to t's group
                            valid.append({
                                'type': ActionType.MOVE_NODE,
                                'node': s,
                                'group_id': gid_t,
                            })
                            # move t to s's group
                            valid.append({
                                'type': ActionType.MOVE_NODE,
                                'node': t,
                                'group_id': gid_s,
                            })
                            # move s or t to a new group (isolate)
                            valid.append({
                                'type': ActionType.MOVE_NODE,
                                'node': s,
                                'group_id': self.num_groups,
                            })

                # Deduplicate valid actions
                seen = set()
                unique_valid = []
                for a in valid:
                    key = str(sorted(a.items()))
                    if key not in seen:
                        seen.add(key)
                        unique_valid.append(a)
                valid = unique_valid

                # Always allow finalize
                valid.append({'type': ActionType.FINALIZE_PARTITION})

        else:
            # Phase 2 actions (unchanged)
            for idx in range(len(self.pool)):
                valid.append({
                    'type': ActionType.ADD_TO_ACCUMULATOR,
                    'idx_i': idx
                })

            k_acc = len(self.accumulator)
            for i in range(k_acc):
                for j in range(i + 1, k_acc):
                    valid.append({
                        'type' : ActionType.APPLY_SUBMODULARITY,
                        'idx_i': i,
                        'idx_j': j
                    })

            pairwise_done = sum(
                1 for e in self.combination_log
                if e.get('action') == 'PAIRWISE'
            )
            if pairwise_done >= 2 and not self._proof2_used:
                valid.append({'type': ActionType.APPLY_PROOF2})

            if self.accumulator:
                valid.append({'type': ActionType.STORE_AND_RESET})

            k_stored = len(self.stored_derived)
            for i in range(k_stored):
                for j in range(i + 1, k_stored):
                    valid.append({
                        'type' : ActionType.COMBINE_STORED,
                        'idx_i': i,
                        'idx_j': j
                    })

            if self.phase2_steps >= MIN_PHASE2_STEPS:
                valid.append({'type': ActionType.DECLARE_TERMINAL})

        return valid

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

    def _get_state(self) -> Dict[str, Any]:
        state = {
            'phase'       : int(self.current_phase),
            'num_nodes'   : len(self.nodes),
            'num_edges'   : len(self.edges),
            'num_sessions': len(self.sessions),
            'stage'       : self.stage
        }
        if self.current_phase == Phase.PHASE1:
            state['current_node_idx']     = self.current_node_idx
            state['num_groups']           = self.num_groups
            state['assignment']           = dict(self.assignment)
            state['sessions']             = list(self.sessions)
            state['edges']                = list(self.edges)
            state['assignment_complete']  = self._assignment_complete
            state['refinement_steps']     = self._refinement_steps
            state['internal_count']       = self._count_current_internal()
        else:
            state['pool_size']           = len(self.pool)
            state['accumulator_size']    = len(self.accumulator)
            state['stored_derived_size'] = len(self.stored_derived)
            state['phase2_steps']        = self.phase2_steps
            state['internal_sessions']   = self.internal_session_count

            state['combination_log'] = list(self.combination_log)
            if self.pool:
                base_part    = self.pool[:self.num_base]
                derived_part = self.pool[self.num_base:][-MAX_DERIVED:]
                pool_to_send = base_part + derived_part
                state['pool_coeffs'] = np.stack(
                    [ineq.coeffs for ineq in pool_to_send]
                )

            if self.accumulator:
                acc_to_send = self.accumulator[-10:]
                state['accumulator_coeffs'] = np.stack(
                    [ineq.coeffs for ineq in acc_to_send]
                )

        return state