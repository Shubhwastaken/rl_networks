"""
RL Environment for the partition bound problem.

10k-run fixes:
- MIN_PHASE2_STEPS = max(6, 2*num_partitions) — dynamic per graph
- PROOF2 gate = 1 pairwise (not 2)
- PROOF2 penalty = 0.05 (not 0.3)
- STEP_COST = -0.01
- Y_I collapse bonus = one-time only (not farmable)
- Terminal discovery bonus = 0.5, one-time
- Finalize bonus = 0.2 * internal (not 0.5)
- STORE/COMBINE gated behind step 10
- MAX_PHASE2_STEPS = 40
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
    SWAP_NODE            = 7
    MOVE_NODE            = 8
    FINALIZE_PARTITION   = 9


MAX_PHASE2_STEPS     = 20    # 40 was too generous — agent stalls
MAX_DERIVED          = 15
MAX_REFINEMENT_STEPS = 20
STEP_COST            = -0.02 # base cost per step
STEP_COST_AFTER_TERMINAL = -0.15  # heavy cost once terminal form exists — stop exploring
PROOF2_PENALTY       = 0.05


class PartitionBoundEnv:

    def __init__(self, graph_dataset_size: int = 5, stage: int = 3):
        self.stage = stage
        print(f"Generating graph dataset ({graph_dataset_size} graphs)...")
        self.graph_dataset = generate_graph_dataset(graph_dataset_size)
        print(f"Dataset ready: {len(self.graph_dataset)} graphs.")

        self.nodes = None
        self.edges = None
        self.sessions = None
        self.index = None

        self.assignment = None
        self.num_groups = 0
        self.node_order = None
        self.current_node_idx = 0
        self.adjacency = None
        self.edge_set = None

        self.base_inequalities = None
        self.num_base = 0
        self.pool = None
        self.accumulator = None
        self.stored_derived = None
        self.phase2_steps = 0
        self.min_phase2_steps = 6
        self.partition = None
        self.internal_per_part = None

        self.current_phase = Phase.PHASE1
        self.internal_session_count = 0
        self.prev_internal_count = 0

        self._assignment_complete = False
        self._refinement_steps = 0

    def reset(self, fixed_partition=None, fixed_graph=None) -> Dict[str, Any]:
        if fixed_graph is not None:
            self.nodes, self.edges, self.sessions = fixed_graph
        else:
            self.nodes, self.edges, self.sessions = random.choice(self.graph_dataset)

        self.adjacency = {n: set() for n in self.nodes}
        self.edge_set = set()
        for u, v in self.edges:
            self.adjacency[u].add(v)
            self.adjacency[v].add(u)
            self.edge_set.add((u, v))
            self.edge_set.add((v, u))

        self.assignment = {n: -1 for n in self.nodes}
        self.num_groups = 0
        self.node_order = self.nodes[:]
        random.shuffle(self.node_order)
        self.current_node_idx = 0
        self.prev_internal_count = 0

        self.pool = []
        self.accumulator = []
        self.stored_derived = []
        self.phase2_steps = 0
        self.partition = None
        self.internal_per_part = None
        self.num_base = 0

        self._assignment_complete = False
        self._refinement_steps = 0

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
        self.pool = [ineq.copy() for ineq in self.base_inequalities]
        self.num_base = len(self.base_inequalities)
        self.internal_per_part = internal_per_partition(
            self.partition, self.sessions
        )
        self.internal_session_count = sum(self.internal_per_part)

        # Dynamic MIN based on partition count
        n_parts = len(self.partition)
        self.min_phase2_steps = max(6, 2 * n_parts)

        self.current_phase = Phase.PHASE2
        self.phase2_steps = 0
        self.combination_log = []
        self._found_terminal = False
        self._found_yi_collapse = False
        self._proof2_used = False

    def _cap_pool(self):
        if len(self.pool) <= self.num_base + MAX_DERIVED:
            return
        base_part = self.pool[:self.num_base]
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

        if action_type == ActionType.ASSIGN_NODE and not self._assignment_complete:
            node = self.node_order[self.current_node_idx]
            gid = action['group_id']
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
            reward = self._internal_reward(cur_internal)
            self.prev_internal_count = cur_internal
            return self._get_state(), reward, False

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

        elif action_type == ActionType.FINALIZE_PARTITION:
            if not self._assignment_complete:
                return self._get_state(), -0.5, False

            self.partition = self._build_partition()
            ipp = internal_per_partition(self.partition, self.sessions)
            bonus = 0.2 * sum(ipp)  # reduced from 0.5

            self._start_phase2()
            return self._get_state(), bonus, False

        return self._get_state(), 0.0, False

    def _internal_reward(self, cur_internal):
        delta = cur_internal - self.prev_internal_count
        return 1.0 * delta

    def _check_assignment_valid(self) -> bool:
        for u, v in self.edges:
            if (self.assignment[u] != -1 and
                self.assignment[v] != -1 and
                self.assignment[u] == self.assignment[v]):
                return False
        return True

    def _step_phase2(self, action):
        action_type = action['type']
        self.phase2_steps += 1
        worst_case_bound = len(self.edges) / max(len(self.sessions), 1)

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
            return self._get_state(), self._current_step_cost(), False

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
                    'step': self.phase2_steps,
                    'action': 'PAIRWISE',
                    'idx_i': idx_i, 'idx_j': idx_j,
                    'yi_a': round(a.yi_coeff(), 4),
                    'yi_b': round(b.yi_coeff(), 4),
                    'yi_union': round(union_ineq.yi_coeff(), 4),
                })
                # One-time Y_I collapse bonus
                if union_ineq.yi_coeff() > 0.5 and not self._found_yi_collapse:
                    self._found_yi_collapse = True
                    bonus = 0.3

            reward = self._current_step_cost() + bonus + self._terminal_discovery_bonus()
            return self._get_state(), reward, False

        elif action_type == ActionType.APPLY_PROOF2:
            if self._proof2_used:
                return self._get_state(), self._current_step_cost() - 0.1, False
            self._proof2_used = True
            try:
                final = apply_n2_submodularity_all_at_once(
                    self.base_inequalities, self.index, self.sessions
                )
                self.pool.append(final)
            except Exception:
                pass
            self.combination_log.append({
                'step': self.phase2_steps, 'action': 'PROOF2',
            })

            # Check if PROOF2 produced a valid terminal form — if so, auto-terminate
            # This is the key fix: the agent was doing PROOF2 then continuing to
            # ADD/SUB for 30+ more steps without ever declaring terminal
            best_bound = None
            for ineq in self.pool + self.accumulator + self.stored_derived:
                if ineq.check_valid_terminal_form():
                    bound = ineq.extract_bound(
                        len(self.sessions), len(self.edges),
                        self.internal_per_part
                    )
                    if best_bound is None or bound < best_bound:
                        best_bound = bound

            if best_bound is not None:
                # Auto-terminate with the bound — small penalty for using PROOF2
                reward = -best_bound - PROOF2_PENALTY
                return self._get_state(), reward, True
            else:
                # PROOF2 failed to produce terminal form (shouldn't happen but be safe)
                reward = self._current_step_cost() - PROOF2_PENALTY
                return self._get_state(), reward, False

        elif action_type == ActionType.STORE_AND_RESET:
            if self.accumulator:
                combined = self.accumulator[0].copy()
                for ineq in self.accumulator[1:]:
                    combined = combined.add(ineq)
                self.stored_derived.append(combined)
                self.accumulator = []
            return self._get_state(), self._current_step_cost(), False

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
            return self._get_state(), self._current_step_cost(), False

        elif action_type == ActionType.DECLARE_TERMINAL:
            if self.phase2_steps < self.min_phase2_steps:
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

        return self._get_state(), self._current_step_cost(), False

    def _current_step_cost(self) -> float:
        """Escalating step cost: heavier once a valid terminal form exists."""
        if self._found_terminal:
            return STEP_COST_AFTER_TERMINAL
        return STEP_COST

    def _terminal_discovery_bonus(self) -> float:
        if self._found_terminal:
            return 0.0
        for ineq in self.pool + self.accumulator + self.stored_derived:
            if ineq.check_valid_terminal_form():
                self._found_terminal = True
                return 0.5
        return 0.0

    def _force_terminal(self):
        best_bound = None
        for ineq in self.pool + self.accumulator + self.stored_derived:
            if ineq.check_valid_terminal_form():
                bound = ineq.extract_bound(
                    len(self.sessions), len(self.edges),
                    self.internal_per_part
                )
                if best_bound is None or bound < best_bound:
                    best_bound = bound
        worst_case = len(self.edges) / max(len(self.sessions), 1)
        reward = -best_bound if best_bound is not None else -worst_case
        reward -= 0.5
        return self._get_state(), reward, True

    def get_valid_actions(self) -> List[Dict]:
        valid = []

        if self.current_phase == Phase.PHASE1:
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
                        gid_s = self.assignment[s]
                        gid_t = self.assignment[t]
                        if gid_s != gid_t:
                            for n in self.nodes:
                                if self.assignment[n] == gid_s and n != s and n != t:
                                    valid.append({'type': ActionType.SWAP_NODE,
                                                  'node_a': t, 'node_b': n})
                            for n in self.nodes:
                                if self.assignment[n] == gid_t and n != s and n != t:
                                    valid.append({'type': ActionType.SWAP_NODE,
                                                  'node_a': s, 'node_b': n})
                    for s, t in self.sessions:
                        gid_s = self.assignment[s]
                        gid_t = self.assignment[t]
                        if gid_s != gid_t:
                            valid.append({'type': ActionType.MOVE_NODE,
                                          'node': s, 'group_id': gid_t})
                            valid.append({'type': ActionType.MOVE_NODE,
                                          'node': t, 'group_id': gid_s})
                            valid.append({'type': ActionType.MOVE_NODE,
                                          'node': s, 'group_id': self.num_groups})

                seen = set()
                unique_valid = []
                for a in valid:
                    key = str(sorted(a.items()))
                    if key not in seen:
                        seen.add(key)
                        unique_valid.append(a)
                valid = unique_valid
                valid.append({'type': ActionType.FINALIZE_PARTITION})

        else:
            # Phase 2 actions
            for idx in range(len(self.pool)):
                valid.append({'type': ActionType.ADD_TO_ACCUMULATOR, 'idx_i': idx})

            k_acc = len(self.accumulator)
            for i in range(k_acc):
                for j in range(i + 1, k_acc):
                    valid.append({'type': ActionType.APPLY_SUBMODULARITY,
                                  'idx_i': i, 'idx_j': j})

            pairwise_done = sum(
                1 for e in self.combination_log
                if e.get('action') == 'PAIRWISE'
            )
            if pairwise_done >= 1 and not self._proof2_used:
                valid.append({'type': ActionType.APPLY_PROOF2})

            # Gate STORE/COMBINE behind step 10 to reduce early confusion
            if self.phase2_steps >= 10:
                if self.accumulator:
                    valid.append({'type': ActionType.STORE_AND_RESET})
                k_stored = len(self.stored_derived)
                for i in range(k_stored):
                    for j in range(i + 1, k_stored):
                        valid.append({'type': ActionType.COMBINE_STORED,
                                      'idx_i': i, 'idx_j': j})

            if self.phase2_steps >= self.min_phase2_steps:
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
            'phase': int(self.current_phase),
            'num_nodes': len(self.nodes),
            'num_edges': len(self.edges),
            'num_sessions': len(self.sessions),
            'stage': self.stage
        }
        if self.current_phase == Phase.PHASE1:
            state['current_node_idx'] = self.current_node_idx
            state['num_groups'] = self.num_groups
            state['assignment'] = dict(self.assignment)
            state['sessions'] = list(self.sessions)
            state['edges'] = list(self.edges)
            state['assignment_complete'] = self._assignment_complete
            state['refinement_steps'] = self._refinement_steps
            state['internal_count'] = self._count_current_internal()
        else:
            state['pool_size'] = len(self.pool)
            state['accumulator_size'] = len(self.accumulator)
            state['stored_derived_size'] = len(self.stored_derived)
            state['phase2_steps'] = self.phase2_steps
            state['internal_sessions'] = self.internal_session_count
            state['combination_log'] = list(self.combination_log)
            if self.pool:
                base_part = self.pool[:self.num_base]
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