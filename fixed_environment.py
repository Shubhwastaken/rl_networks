import random
import numpy as np
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

"""
RL Environment for the partition bound problem.
"""

import random
from enum import IntEnum
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from fixed_graph_generation import generate_large_network
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
    ASSIGN_NODE         = 0
    ADD_TO_ACCUMULATOR  = 1
    APPLY_SUBMODULARITY = 2
    APPLY_PROOF2        = 3
    STORE_AND_RESET     = 4
    COMBINE_STORED      = 5
    DECLARE_TERMINAL    = 6


MIN_PHASE2_STEPS = 4
MAX_DERIVED      = 15   # max derived inequalities kept in pool beyond base


class PartitionBoundEnv:

    def __init__(self, graph_dataset_size: int = 100, stage: int = 3):
        self.stage = stage

        print(f"Generating graph dataset ({graph_dataset_size} graphs)...")
        self.graph_dataset = []
        for _ in range(graph_dataset_size):
            nodes, edges, sessions = generate_large_network()
            self.graph_dataset.append((nodes, edges, sessions))
        print("Dataset ready.")

        self.nodes    = None
        self.edges    = None
        self.sessions = None
        self.index    = None

        self.assignment       = None
        self.num_groups       = 0
        self.node_order       = None
        self.current_node_idx = 0
        self.adjacency        = None

        self.base_inequalities  = None
        self.num_base           = 0     # number of base inequalities in pool
        self.pool               = None
        self.accumulator        = None
        self.stored_derived     = None
        self.phase2_steps       = 0
        self.partition          = None
        self.internal_per_part  = None

        self.current_phase          = Phase.PHASE1
        self.internal_session_count = 0
        self.prev_internal_count    = 0

    def reset(self, fixed_partition=None) -> Dict[str, Any]:
        self.nodes, self.edges, self.sessions = random.choice(self.graph_dataset)

        self.adjacency = {n: set() for n in self.nodes}
        for u, v in self.edges:
            self.adjacency[u].add(v)
            self.adjacency[v].add(u)

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

        # pool starts with base inequalities only
        self.pool     = [ineq.copy() for ineq in self.base_inequalities]
        self.num_base = len(self.base_inequalities)

        self.internal_per_part      = internal_per_partition(
            self.partition, self.sessions
        )
        self.internal_session_count = sum(self.internal_per_part)

        self.current_phase    = Phase.PHASE2
        self.phase2_steps     = 0
        self.combination_log  = []   # records each combination step

    def _cap_pool(self):
        """
        Keeps ALL base inequalities intact.
        Caps only the derived inequalities (union/intersection results)
        at MAX_DERIVED to prevent unbounded memory growth.

        Base inequalities are always at pool[0 : num_base].
        Derived inequalities are at pool[num_base : ].
        """
        if len(self.pool) > self.num_base + MAX_DERIVED:
            base_part    = self.pool[:self.num_base]
            derived_part = self.pool[self.num_base:][-MAX_DERIVED:]
            self.pool    = base_part + derived_part

    def step(self, action: Dict[str, Any]) -> Tuple[Dict, float, bool]:
        if self.current_phase == Phase.PHASE1:
            return self._step_phase1(action)
        return self._step_phase2(action)

    def _step_phase1(self, action):
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
        # FIX: reward = improvement in partition bound from this assignment
        # prev_bound - curr_bound is positive when bound tightened
        # this gives a meaningful gradient signal at every step
        num_sessions = len(self.sessions)
        num_edges    = len(self.edges)
        prev_bound = (num_edges / (num_sessions + self.prev_internal_count)
                      if num_sessions + self.prev_internal_count > 0
                      else num_edges / max(num_sessions, 1))
        curr_bound = (num_edges / (num_sessions + cur_internal)
                      if num_sessions + cur_internal > 0
                      else num_edges / max(num_sessions, 1))
        reward = prev_bound - curr_bound   # positive when bound improved
        self.prev_internal_count = cur_internal

        if self.current_node_idx >= len(self.nodes):
            self.partition = self._build_partition()
            self._start_phase2()

        return self._get_state(), reward, False

    def _step_phase2(self, action):
        action_type      = action['type']
        self.phase2_steps += 1
        worst_case_bound  = len(self.edges) / max(len(self.sessions), 1)

        # cap derived inequalities after every step
        self._cap_pool()

        if action_type == ActionType.ADD_TO_ACCUMULATOR:
            idx = action.get('idx_i', 0)
            if idx < len(self.pool):
                ineq = self.pool.pop(idx)
                # if we removed a base inequality, decrement num_base
                if idx < self.num_base:
                    self.num_base -= 1
                self.accumulator.append(ineq)
            return self._get_state(), self._compute_ratio_reward(), False

        elif action_type == ActionType.APPLY_SUBMODULARITY:
            idx_i = action.get('idx_i', 0)
            idx_j = action.get('idx_j', 1)
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
                # derived inequalities go to the end of pool
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
            return self._get_state(), self._compute_ratio_reward(), False

        elif action_type == ActionType.APPLY_PROOF2:
            # Small penalty to discourage always taking the easy Proof 2 route.
            # Agent still gets the bound via Proof2 but pays a cost.
            # Pairwise combinations that find 1.67 get full reward with no penalty.
            PROOF2_PENALTY = 0.2
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
            reward = self._compute_ratio_reward() - PROOF2_PENALTY
            return self._get_state(), reward, False

        elif action_type == ActionType.STORE_AND_RESET:
            if self.accumulator:
                combined = self.accumulator[0].copy()
                for ineq in self.accumulator[1:]:
                    combined = combined.add(ineq)
                self.stored_derived.append(combined)
                self.accumulator = []
            return self._get_state(), 0.0, False

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
            return self._get_state(), 0.0, False

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

        return self._get_state(), 0.0, False

    def get_valid_actions(self) -> List[Dict]:
        valid = []

        if self.current_phase == Phase.PHASE1:
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
            for idx in range(len(self.pool)):
                valid.append({'type': ActionType.ADD_TO_ACCUMULATOR, 'idx_i': idx})

            k_acc = len(self.accumulator)
            for i in range(k_acc):
                for j in range(i + 1, k_acc):
                    valid.append({
                        'type' : ActionType.APPLY_SUBMODULARITY,
                        'idx_i': i,
                        'idx_j': j
                    })

            # FIX: PROOF2 only available after at least 2 pairwise steps
            # forces agent to explore pairwise combinations first
            pairwise_done = sum(
                1 for e in self.combination_log
                if e.get('action') == 'PAIRWISE'
            )
            if pairwise_done >= 2:
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

    def _compute_ratio_reward(self) -> float:
        best_ratio = 0.0
        for ineq in self.pool + self.accumulator + self.stored_derived:
            c1 = ineq.get_yi_coefficient()
            c3 = ineq.get_rhs_edge_coefficient()
            if c1 > 0 and c3 > 0:
                best_ratio = max(best_ratio, c1 / c3)
        return 0.005 * best_ratio

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
            state['current_node_idx'] = self.current_node_idx
            state['num_groups']       = self.num_groups
            state['assignment']       = dict(self.assignment)
            # FIX: expose sessions and edges so GNN can reason about
            # which nodes are session partners
            state['sessions'] = list(self.sessions)
            state['edges']    = list(self.edges)
        else:
            state['pool_size']           = len(self.pool)
            state['accumulator_size']    = len(self.accumulator)
            state['stored_derived_size'] = len(self.stored_derived)
            state['phase2_steps']        = self.phase2_steps
            state['internal_sessions']   = self.internal_session_count

            state['combination_log'] = list(self.combination_log)
            if self.pool:
                # send base inequalities + last MAX_DERIVED derived ones
                base_part    = self.pool[:self.num_base]
                derived_part = self.pool[self.num_base:][-MAX_DERIVED:]
                pool_to_send = base_part + derived_part
                state['pool_coeffs'] = np.stack(
                    [ineq.coeffs for ineq in pool_to_send]
                )

            if self.accumulator:
                # cap accumulator slice sent to GNN
                acc_to_send = self.accumulator[-10:]
                state['accumulator_coeffs'] = np.stack(
                    [ineq.coeffs for ineq in acc_to_send]
                )

        return state