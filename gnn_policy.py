"""
GNN-based policies for Phase 1 and Phase 2.
Plain PyTorch only -- no torch_geometric required.
GPU support via DEVICE.

MAJOR FIXES:
1. Phase 1 policy handles three sub-modes:
   - ASSIGN mode (initial one-pass assignment)
   - REFINE mode (SWAP_NODE, MOVE_NODE, FINALIZE_PARTITION)
   Each valid action is scored via a unified action scorer.

2. Phase 2: BOTH pointer indices are now learned (not random for j).
   Second pointer is conditioned on the first selection.

3. Both policies use GAE (lambda=0.95) for lower variance.

4. Phase 1 GNN features:
   - is_assigned, group_id, is_current_node, degree
   - is_session_source, is_session_sink
   - session_partner_group, session_partner_assigned
   - num_internal_sessions_in_group (how many sessions internal to this group)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Tuple
from enum import IntEnum

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


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


def compute_gae(rewards: List[float], values: List[float],
                gamma: float = 0.99, lam: float = 0.95) -> List[float]:
    """Generalized Advantage Estimation for lower variance."""
    advantages = []
    gae = 0.0
    n = len(rewards)
    for t in reversed(range(n)):
        next_val = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    returns = [a + v for a, v in zip(advantages, values[:n])]
    return returns, advantages


# -----------------------------------------------------------------------
# Phase 1 -- GraphSAGE with session-aware features
# -----------------------------------------------------------------------

class SAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W_self  = nn.Linear(in_dim, out_dim, bias=False)
        self.W_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.norm    = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        deg   = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        neigh = (adj @ x) / deg
        return F.relu(self.norm(self.W_self(x) + self.W_neigh(neigh)))


class Phase1GNN(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_layers: int = 3,
                 max_groups: int = 32):
        super().__init__()
        # Input: 10 features per node (up from 6)
        input_dim   = 10
        dims        = [input_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList([
            SAGELayer(dims[i], dims[i + 1]) for i in range(num_layers)
        ])

        # Policy heads
        # For ASSIGN: score each group for the current node
        self.assign_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, max_groups)
        )

        # For REFINE actions: score each valid action as a flat distribution
        # We encode each action as a small feature vector and score it
        self.refine_action_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 4, hidden_dim), nn.ReLU(),
        )
        self.refine_scorer = nn.Linear(hidden_dim, 1)

        # Value head: global graph -> scalar
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.max_groups = max_groups

    def forward(self, x, adj):
        h = x
        for layer in self.layers:
            h = layer(h, adj)
        return h  # (n_nodes, hidden_dim)

    def assign_logits(self, h, cur_idx):
        return self.assign_head(h[cur_idx])

    def global_value(self, h):
        return self.value_head(h.mean(dim=0))


class GNNPhase1Policy:

    def __init__(self, hidden_dim: int = 64, num_layers: int = 3,
                 max_groups: int = 32, lr: float = 3e-4,
                 entropy_coeff: float = 0.1):
        self.net           = Phase1GNN(hidden_dim, num_layers,
                                        max_groups).to(DEVICE)
        self.optimizer     = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.max_groups    = max_groups
        self.hidden_dim    = hidden_dim
        self.entropy_coeff = entropy_coeff
        self._log_probs : List[torch.Tensor] = []
        self._values    : List[float]        = []
        self._rewards   : List[float]        = []
        self._entropies : List[torch.Tensor] = []

    def select_action(self, state: dict, valid_actions: List[dict]) -> dict:
        x, adj, node_map = self._build_tensors(state)
        h = self.net(x, adj)

        value = self.net.global_value(h).squeeze()
        self._values.append(value.item())

        assignment_complete = state.get('assignment_complete', False)

        if not assignment_complete:
            # ASSIGN mode
            return self._select_assign(state, valid_actions, h, value)
        else:
            # REFINE mode
            return self._select_refine(state, valid_actions, h, value, node_map)

    def _select_assign(self, state, valid_actions, h, value):
        cur_idx = state['current_node_idx']
        logits  = self.net.assign_logits(h, cur_idx)

        valid_gids = [a['group_id'] for a in valid_actions
                      if int(a.get('type', 0)) == ActionType.ASSIGN_NODE]
        mask = torch.full((self.max_groups,), float('-inf')).to(DEVICE)
        for gid in valid_gids:
            if gid < self.max_groups:
                mask[gid] = 0.0

        temperature = state.get('temperature', 1.0)
        probs = F.softmax((logits + mask) / temperature, dim=-1)
        dist  = torch.distributions.Categorical(probs)
        gid_t = dist.sample()

        self._log_probs.append(dist.log_prob(gid_t))
        self._rewards.append(0.0)
        self._entropies.append(dist.entropy())

        return {'type': ActionType.ASSIGN_NODE, 'group_id': gid_t.item()}

    def _select_refine(self, state, valid_actions, h, value, node_map):
        """
        Score each valid refine action and sample from the distribution.
        Each action is encoded as (node_a_embedding, node_b_embedding, features).
        For FINALIZE, we use the graph mean as both embeddings.
        """
        if not valid_actions:
            return {'type': ActionType.FINALIZE_PARTITION}

        graph_mean = h.mean(dim=0)
        action_scores = []
        temperature = state.get('temperature', 1.0)

        for a in valid_actions:
            atype = int(a.get('type', ActionType.FINALIZE_PARTITION))
            if atype == ActionType.SWAP_NODE:
                na = a['node_a']
                nb = a['node_b']
                idx_a = node_map.get(na, 0)
                idx_b = node_map.get(nb, 0)
                feat = torch.tensor([1.0, 0.0, 0.0, 0.0]).to(DEVICE)
                enc = torch.cat([h[idx_a], h[idx_b], feat])
            elif atype == ActionType.MOVE_NODE:
                nd = a['node']
                idx_nd = node_map.get(nd, 0)
                gid = a['group_id']
                feat = torch.tensor([0.0, 1.0, gid / max(state.get('num_groups', 1), 1), 0.0]).to(DEVICE)
                enc = torch.cat([h[idx_nd], graph_mean, feat])
            elif atype == ActionType.FINALIZE_PARTITION:
                feat = torch.tensor([0.0, 0.0, 0.0, 1.0]).to(DEVICE)
                enc = torch.cat([graph_mean, graph_mean, feat])
            else:
                feat = torch.tensor([0.0, 0.0, 0.0, 0.0]).to(DEVICE)
                enc = torch.cat([graph_mean, graph_mean, feat])

            score = self.net.refine_scorer(
                F.relu(self.net.refine_action_encoder(enc))
            ).squeeze()
            action_scores.append(score)

        scores = torch.stack(action_scores)
        probs  = F.softmax(scores / temperature, dim=-1)
        dist   = torch.distributions.Categorical(probs)
        idx    = dist.sample()

        self._log_probs.append(dist.log_prob(idx))
        self._rewards.append(0.0)
        self._entropies.append(dist.entropy())

        return valid_actions[idx.item()]

    def update(self, trajectory: list, final_reward: float):
        if not self._log_probs:
            return

        n = min(len(self._log_probs), len(trajectory))
        if n == 0:
            self._clear()
            return

        for i, t in enumerate(trajectory[:n]):
            self._rewards[i] = t.get('reward', 0.0)
        if self._rewards:
            self._rewards[n - 1] += final_reward

        # GAE
        returns, advantages = compute_gae(
            self._rewards[:n], self._values[:n]
        )
        ret_t = torch.tensor(returns, dtype=torch.float32).to(DEVICE)
        adv_t = torch.tensor(advantages, dtype=torch.float32).to(DEVICE)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        lps_t = torch.stack(self._log_probs[:n])
        ent_t = torch.stack(self._entropies[:n])

        if not lps_t.requires_grad and lps_t.grad_fn is None:
            self._clear()
            return

        vals_t = torch.tensor(self._values[:n],
                               dtype=torch.float32).to(DEVICE)

        pg_loss  = -(lps_t * adv_t.detach()).mean()
        vf_loss  = F.mse_loss(vals_t, ret_t.detach())
        ent_loss = -self.entropy_coeff * ent_t.mean()
        loss     = pg_loss + 0.5 * vf_loss + ent_loss

        if loss.grad_fn is None:
            self._clear()
            return

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.optimizer.step()
        self._clear()

    def _clear(self):
        self._log_probs = []
        self._values    = []
        self._rewards   = []
        self._entropies = []

    def _build_tensors(self, state):
        assignment = state['assignment']
        nodes      = list(assignment.keys())
        n          = len(nodes)
        num_groups = max(state.get('num_groups', 1), 1)

        edges = state.get('edges', [])
        sessions = state.get('sessions', [])
        node_to_i = {nd: i for i, nd in enumerate(nodes)}

        adj = torch.zeros(n, n)
        for u, v in edges:
            if u in node_to_i and v in node_to_i:
                i, j = node_to_i[u], node_to_i[v]
                adj[i][j] = 1.0
                adj[j][i] = 1.0
        adj = adj + torch.eye(n)

        # Build session maps
        source_of = {}  # node -> session partner (if node is source)
        sink_of   = {}  # node -> session partner (if node is sink)
        for s, t in sessions:
            source_of[s] = t
            sink_of[t]   = s

        # Count internal sessions per group
        groups = {}
        for nd, gid in assignment.items():
            if gid != -1:
                groups.setdefault(gid, set()).add(nd)
        internal_per_group = {}
        for gid, gset in groups.items():
            cnt = sum(1 for s, t in sessions if s in gset and t in gset)
            internal_per_group[gid] = cnt

        cur_idx = state.get('current_node_idx', 0)

        # 10 features per node
        feats = torch.zeros(n, 10)
        for i, nd in enumerate(nodes):
            gid = assignment[nd]
            # 0: is_assigned
            feats[i, 0] = 1.0 if gid != -1 else 0.0
            # 1: normalized group_id
            feats[i, 1] = gid / num_groups if gid != -1 else -1.0
            # 2: is_current_node (during assignment phase)
            if not state.get('assignment_complete', False) and i == cur_idx:
                feats[i, 2] = 1.0
            # 3: normalized degree
            feats[i, 3] = adj[i].sum().item() / n
            # 4: is_session_source
            feats[i, 4] = 1.0 if nd in source_of else 0.0
            # 5: is_session_sink
            feats[i, 5] = 1.0 if nd in sink_of else 0.0
            # 6: session partner's group (source's partner = sink)
            partner = source_of.get(nd, sink_of.get(nd, None))
            if partner is not None:
                p_gid = assignment.get(partner, -1)
                feats[i, 6] = p_gid / num_groups if p_gid != -1 else -1.0
            else:
                feats[i, 6] = -1.0
            # 7: is partner assigned?
            if partner is not None:
                feats[i, 7] = 1.0 if assignment.get(partner, -1) != -1 else 0.0
            # 8: same_group_as_partner (1.0 if in same group = internal session)
            if partner is not None and gid != -1:
                p_gid = assignment.get(partner, -1)
                feats[i, 8] = 1.0 if gid == p_gid else 0.0
            # 9: internal sessions in this node's group
            if gid != -1:
                feats[i, 9] = internal_per_group.get(gid, 0) / max(len(sessions), 1)

        feats = feats.to(DEVICE)
        adj   = adj.to(DEVICE)
        return feats, adj, node_to_i


# -----------------------------------------------------------------------
# Phase 2 -- Transformer encoder + dual pointer network
# -----------------------------------------------------------------------

class InequalityEncoder(nn.Module):
    def __init__(self, coeff_dim: int, token_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(coeff_dim, token_dim)
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.norm(self.proj(x)))


class Phase2Net(nn.Module):
    def __init__(self, coeff_dim: int, token_dim: int = 128,
                 num_heads: int = 4, num_layers: int = 2, ff_dim: int = 256):
        super().__init__()
        self.encoder  = InequalityEncoder(coeff_dim, token_dim)
        enc_layer     = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=num_heads,
            dim_feedforward=ff_dim, batch_first=True,
            norm_first=False
        )
        self.transformer  = nn.TransformerEncoder(enc_layer, num_layers)
        self.token_dim    = token_dim

        self.action_type_head = nn.Sequential(
            nn.Linear(token_dim, token_dim), nn.ReLU(),
            nn.Linear(token_dim, 7)
        )

        # Dual pointer: first and second selection
        self.ptr_query_1 = nn.Linear(token_dim, token_dim)
        self.ptr_key_1   = nn.Linear(token_dim, token_dim)

        # Second pointer conditioned on first selection
        self.ptr_query_2 = nn.Linear(token_dim * 2, token_dim)
        self.ptr_key_2   = nn.Linear(token_dim, token_dim)

        self.value_head = nn.Sequential(
            nn.Linear(token_dim, token_dim), nn.ReLU(),
            nn.Linear(token_dim, 1)
        )

    def encode(self, coeffs: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(coeffs)
        tokens = self.transformer(tokens.unsqueeze(0))
        return tokens.squeeze(0)

    def pointer_scores_1(self, tokens: torch.Tensor,
                         global_tok: torch.Tensor) -> torch.Tensor:
        q = self.ptr_query_1(global_tok.unsqueeze(0))
        k = self.ptr_key_1(tokens)
        return (q @ k.T).squeeze(0) / (self.token_dim ** 0.5)

    def pointer_scores_2(self, tokens: torch.Tensor,
                         global_tok: torch.Tensor,
                         selected_tok: torch.Tensor) -> torch.Tensor:
        """Second pointer conditioned on first selection."""
        context = torch.cat([global_tok, selected_tok])
        q = self.ptr_query_2(context.unsqueeze(0))
        k = self.ptr_key_2(tokens)
        return (q @ k.T).squeeze(0) / (self.token_dim ** 0.5)


class GNNPhase2Policy:

    def __init__(self, coeff_dim: int = 256, token_dim: int = 128,
                 num_heads: int = 4, num_layers: int = 2, lr: float = 3e-4,
                 entropy_coeff: float = 0.1):
        self.net           = Phase2Net(coeff_dim, token_dim, num_heads,
                                        num_layers).to(DEVICE)
        self.optimizer     = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.coeff_dim     = coeff_dim
        self.entropy_coeff = entropy_coeff
        self._frozen       = False
        self._log_probs : List[torch.Tensor] = []
        self._values    : List[float]        = []
        self._rewards   : List[float]        = []
        self._entropies : List[torch.Tensor] = []

    def freeze(self):
        for p in self.net.parameters():
            p.requires_grad = False
        self._frozen = True

    def unfreeze(self):
        for p in self.net.parameters():
            p.requires_grad = True
        self._frozen = False

    def select_action(self, state: dict, valid_actions: List[dict]) -> dict:
        if not valid_actions:
            return {'type': ActionType.DECLARE_TERMINAL}

        tokens, global_tok = self._encode(state)

        type_logits = self.net.action_type_head(global_tok)
        valid_types = {int(a['type']) for a in valid_actions}
        type_mask   = torch.full((7,), float('-inf')).to(DEVICE)
        for t in valid_types:
            if t < 7:
                type_mask[t] = 0.0

        type_probs = F.softmax(type_logits + type_mask, dim=-1)
        type_dist  = torch.distributions.Categorical(type_probs)
        atype_t    = type_dist.sample()
        atype      = ActionType(atype_t.item())
        lp_type    = type_dist.log_prob(atype_t)
        entropy    = type_dist.entropy()

        value   = self.net.value_head(global_tok).squeeze()
        self._values.append(value.item())

        lp_idx  = torch.tensor(0.0).to(DEVICE)
        action  = {'type': atype}

        if atype == ActionType.ADD_TO_ACCUMULATOR:
            idxs = [a['idx_i'] for a in valid_actions
                    if int(a['type']) == int(atype)]
            if idxs and len(tokens) > 0:
                # Remap pool indices to token positions.
                # pool_coeffs may be a truncated view of the full pool,
                # so some action indices may exceed len(tokens).
                # Build a mapping: token_position -> pool_index
                n_tokens = len(tokens)
                mappable = [i for i in idxs if i < n_tokens]
                if mappable:
                    idx, lp_i = self._pointer_1(tokens, global_tok, mappable)
                    action['idx_i'] = idx
                    lp_idx = lp_i
                else:
                    # All indices out of token range — pick randomly
                    pick = idxs[np.random.randint(len(idxs))]
                    action['idx_i'] = pick
                    lp_idx = torch.tensor(-np.log(max(len(idxs), 1))).to(DEVICE)

        elif atype == ActionType.APPLY_SUBMODULARITY:
            pairs = [(a['idx_i'], a['idx_j']) for a in valid_actions
                     if int(a['type']) == int(atype)]
            if pairs:
                acc = state.get('accumulator_coeffs', None)
                if acc is not None and len(acc) > 0:
                    acc_t   = self._pad(
                        torch.tensor(acc, dtype=torch.float32).to(DEVICE)
                    )
                    acc_tok = self.net.encode(acc_t)
                    n_acc_tokens = len(acc_tok)

                    # The accumulator_coeffs sent in state may be truncated
                    # (last 10 items). Remap: the state sends items
                    # [acc_size-10 : acc_size], so token 0 = acc item (acc_size-10).
                    acc_size = state.get('accumulator_size', n_acc_tokens)
                    offset   = max(0, acc_size - n_acc_tokens)

                    # First pointer: select idx_i from remapped positions
                    all_i = sorted(set(p[0] for p in pairs))
                    all_i_remapped = [i - offset for i in all_i
                                      if 0 <= i - offset < n_acc_tokens]

                    if all_i_remapped:
                        i_tok, lp_i = self._pointer_1(
                            acc_tok, acc_tok.mean(dim=0), all_i_remapped
                        )
                        i_sel = i_tok + offset  # remap back to accumulator index
                    else:
                        # Fallback: pick from original indices randomly
                        i_sel = all_i[np.random.randint(len(all_i))]
                        lp_i  = torch.tensor(-np.log(max(len(all_i), 1))).to(DEVICE)

                    # Second pointer: select idx_j conditioned on i_sel
                    valid_j = sorted(set(p[1] for p in pairs if p[0] == i_sel))
                    if not valid_j:
                        # i_sel may have changed due to remapping; find any valid pair
                        valid_j = sorted(set(p[1] for p in pairs))
                    if not valid_j:
                        valid_j = [pairs[0][1]]

                    if len(valid_j) == 1:
                        j_sel = valid_j[0]
                        lp_j  = torch.tensor(0.0).to(DEVICE)
                    else:
                        valid_j_remapped = [j - offset for j in valid_j
                                            if 0 <= j - offset < n_acc_tokens]
                        if valid_j_remapped:
                            i_tok_clamped = min(max(i_sel - offset, 0),
                                                n_acc_tokens - 1)
                            selected_tok = acc_tok[i_tok_clamped]
                            j_tok, lp_j = self._pointer_2(
                                acc_tok, acc_tok.mean(dim=0),
                                selected_tok, valid_j_remapped
                            )
                            j_sel = j_tok + offset
                        else:
                            j_sel = valid_j[np.random.randint(len(valid_j))]
                            lp_j  = torch.tensor(
                                -np.log(max(len(valid_j), 1))
                            ).to(DEVICE)

                    lp_idx = lp_i + lp_j
                else:
                    i_sel = pairs[0][0]
                    j_sel = pairs[0][1]
                    lp_idx = torch.tensor(0.0).to(DEVICE)

                action['idx_i'] = i_sel
                action['idx_j'] = j_sel

        elif atype == ActionType.COMBINE_STORED:
            pairs = [(a['idx_i'], a['idx_j']) for a in valid_actions
                     if int(a['type']) == int(atype)]
            if pairs:
                pi = np.random.randint(len(pairs))
                action['idx_i'] = pairs[pi][0]
                action['idx_j'] = pairs[pi][1]
                lp_idx = torch.tensor(-np.log(max(len(pairs), 1))).to(DEVICE)

        self._log_probs.append(lp_type + lp_idx)
        self._rewards.append(0.0)
        self._entropies.append(entropy)

        return action

    def update(self, trajectory: list, final_reward: float):
        if self._frozen or not self._log_probs:
            self._clear()
            return

        n = min(len(self._log_probs), len(trajectory))
        if n == 0:
            self._clear()
            return

        for i, t in enumerate(trajectory[:n]):
            self._rewards[i] = t.get('reward', 0.0)
        if self._rewards:
            self._rewards[n - 1] += final_reward

        # GAE
        returns, advantages = compute_gae(
            self._rewards[:n], self._values[:n]
        )
        ret_t = torch.tensor(returns, dtype=torch.float32).to(DEVICE)
        adv_t = torch.tensor(advantages, dtype=torch.float32).to(DEVICE)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        lps_t = torch.stack(self._log_probs[:n])
        ent_t = torch.stack(self._entropies[:n])

        if not lps_t.requires_grad and lps_t.grad_fn is None:
            self._clear()
            return

        vals_t = torch.tensor(self._values[:n],
                               dtype=torch.float32).to(DEVICE)

        pg_loss  = -(lps_t * adv_t.detach()).mean()
        vf_loss  = F.mse_loss(vals_t, ret_t.detach())
        ent_loss = -self.entropy_coeff * ent_t.mean()
        loss     = pg_loss + 0.5 * vf_loss + ent_loss

        if loss.grad_fn is None:
            self._clear()
            return

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.optimizer.step()
        self._clear()

    def _clear(self):
        self._log_probs = []
        self._values    = []
        self._rewards   = []
        self._entropies = []

    def _encode(self, state):
        pool = state.get('pool_coeffs', None)
        if pool is not None and len(pool) > 0:
            t = self._pad(
                torch.tensor(pool, dtype=torch.float32).to(DEVICE)
            )
        else:
            t = torch.zeros(1, self.coeff_dim).to(DEVICE)
        tokens     = self.net.encode(t)
        global_tok = tokens.mean(dim=0)
        return tokens, global_tok

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        if d == self.coeff_dim:
            return x
        if d < self.coeff_dim:
            pad = torch.zeros(*x.shape[:-1], self.coeff_dim - d).to(DEVICE)
            return torch.cat([x, pad], dim=-1)
        return x[..., :self.coeff_dim]

    def _pointer_1(self, tokens, global_tok, valid_idxs):
        """First pointer selection."""
        scores = self.net.pointer_scores_1(tokens, global_tok)
        mask   = torch.full((len(tokens),), float('-inf')).to(DEVICE)
        any_valid = False
        for vi in valid_idxs:
            if vi < len(tokens):
                mask[vi] = 0.0
                any_valid = True
        if not any_valid:
            # Fallback: unmask all positions to avoid nan
            mask = torch.zeros(len(tokens)).to(DEVICE)
        probs = F.softmax(scores + mask, dim=-1)
        # Safety: replace nan with uniform if softmax failed
        if torch.isnan(probs).any():
            probs = torch.ones_like(probs) / len(probs)
        dist  = torch.distributions.Categorical(probs)
        idx_t = dist.sample()
        return idx_t.item(), dist.log_prob(idx_t)

    def _pointer_2(self, tokens, global_tok, selected_tok, valid_idxs):
        """Second pointer selection conditioned on first."""
        scores = self.net.pointer_scores_2(tokens, global_tok, selected_tok)
        mask   = torch.full((len(tokens),), float('-inf')).to(DEVICE)
        any_valid = False
        for vi in valid_idxs:
            if vi < len(tokens):
                mask[vi] = 0.0
                any_valid = True
        if not any_valid:
            mask = torch.zeros(len(tokens)).to(DEVICE)
        probs = F.softmax(scores + mask, dim=-1)
        if torch.isnan(probs).any():
            probs = torch.ones_like(probs) / len(probs)
        dist  = torch.distributions.Categorical(probs)
        idx_t = dist.sample()
        return idx_t.item(), dist.log_prob(idx_t)