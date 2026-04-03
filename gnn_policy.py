"""
GNN policies for all three phases.

PHASE 1 (GNNPhase1Policy) — unchanged architecture, NEW output:
  select_action now returns actions with an optional 'weights' dict
  when FINALIZE_PARTITION is chosen. The weight head outputs one
  scalar per partition group, softmax-normalised, representing the
  suggested λ distribution over partition sets.

PHASE 2 (GNNPhase2Policy) — unchanged architecture.
  Now sees CROSS_SUBMOD as a valid action type (type=11).
  The action_type_head logit dimension is extended to 12.

PHASE 3 (GNNPhase3Policy) — new.
  Input: graph features (from Phase1GNN) + pool embedding (Transformer).
  Action heads:
    (a) action_type: {FRACTIONAL_IO, ADD_TO_ACC, SUBMOD, CROSS_SUBMOD,
                      STORE, DECLARE_TERMINAL}
    (b) node_u pointer, node_v pointer (for FRACTIONAL_IO)
    (c) lambda selector (discrete over LAMBDA_GRID)
    (d) pool index pointer (for ADD_TO_ACC / SUBMOD)
  Value head: scalar baseline.

All policies use GAE + vanilla PG (policy gradient) with cosine LR
and entropy annealing. Phase 3 uses a slightly higher entropy coefficient
to maintain exploration pressure during the harder search.
"""

import random as _random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Tuple
from enum import IntEnum

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    try:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    except Exception:
        pass  # GPU present but name unavailable


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
    FRACTIONAL_IO        = 10
    CROSS_SUBMOD         = 11


LAMBDA_GRID = [0.25, 0.33, 0.40, 0.50, 0.60, 0.67, 0.75]


def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0.0
    n = len(rewards)
    for t in reversed(range(n)):
        next_val = values[t+1] if t+1 < len(values) else 0.0
        delta    = rewards[t] + gamma * next_val - values[t]
        gae      = delta + gamma * lam * gae
        advantages.insert(0, gae)
    returns = [a + v for a, v in zip(advantages, values[:n])]
    return returns, advantages


# ═══════════════════════════════════════════════════════════════════
# Phase 1 — GraphSAGE (partition learner + weight head)
# ═══════════════════════════════════════════════════════════════════

class SAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_self  = nn.Linear(in_dim,  out_dim, bias=False)
        self.W_neigh = nn.Linear(in_dim,  out_dim, bias=False)
        self.norm    = nn.LayerNorm(out_dim)

    def forward(self, x, adj):
        deg   = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        neigh = (adj @ x) / deg
        return F.relu(self.norm(self.W_self(x) + self.W_neigh(neigh)))


class Phase1GNN(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=3, max_groups=32):
        super().__init__()
        input_dim = 10
        dims = [input_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList([
            SAGELayer(dims[i], dims[i+1]) for i in range(num_layers)
        ])
        self.assign_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, max_groups)
        )
        self.refine_action_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 4, hidden_dim), nn.ReLU(),
        )
        self.refine_scorer = nn.Linear(hidden_dim, 1)
        self.value_head    = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # NEW: weight head — outputs one score per group, used at FINALIZE
        self.weight_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, max_groups)
        )
        self.max_groups = max_groups

    def forward(self, x, adj):
        h = x
        for layer in self.layers:
            h = layer(h, adj)
        return h

    def assign_logits(self, h, cur_idx):
        return self.assign_head(h[cur_idx])

    def global_value(self, h):
        return self.value_head(h.mean(dim=0))

    def group_weights(self, h, num_groups):
        """Returns softmax-normalised weights over groups (for FINALIZE)."""
        global_h = h.mean(dim=0)
        raw = self.weight_head(global_h)[:num_groups]
        return F.softmax(raw, dim=-1)


class GNNPhase1Policy:
    def __init__(self, hidden_dim=64, num_layers=3, max_groups=32,
                 lr=3e-4, entropy_coeff_start=0.15, entropy_coeff_end=0.01,
                 total_episodes=10000):
        self.net = Phase1GNN(hidden_dim, num_layers, max_groups).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(total_episodes, 1), eta_min=1e-5
        )
        self.max_groups          = max_groups
        self.hidden_dim          = hidden_dim
        self.entropy_coeff_start = entropy_coeff_start
        self.entropy_coeff_end   = entropy_coeff_end
        self.entropy_coeff       = entropy_coeff_start
        self.total_episodes      = total_episodes
        self._episode_count      = 0
        self._log_probs: List[torch.Tensor] = []
        self._values:   List[float]         = []
        self._rewards:  List[float]         = []
        self._entropies:List[torch.Tensor]  = []
        # Store last computed weights for FINALIZE action
        self._last_weights: Dict[int, float] = {}

    def reset_scheduler(self, total_episodes=None):
        if total_episodes is not None:
            self.total_episodes = total_episodes
        self._episode_count = 0
        self.entropy_coeff  = self.entropy_coeff_start
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(self.total_episodes, 1), eta_min=1e-5
        )

    def select_action(self, state, valid_actions):
        x, adj, node_map = self._build_tensors(state)
        h = self.net(x, adj)
        value = self.net.global_value(h).squeeze()
        self._values.append(value.item())

        if not state.get('assignment_complete', False):
            return self._select_assign(state, valid_actions, h)
        else:
            return self._select_refine(state, valid_actions, h, node_map)

    def _select_assign(self, state, valid_actions, h):
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

        lp = dist.log_prob(gid_t)
        self._log_probs.append(lp)
        self._rewards.append(0.0)
        self._entropies.append(dist.entropy())
        return {'type': ActionType.ASSIGN_NODE, 'group_id': gid_t.item()}

    def _select_refine(self, state, valid_actions, h, node_map):
        if not valid_actions:
            return {'type': ActionType.FINALIZE_PARTITION}

        graph_mean  = h.mean(dim=0)
        temperature = state.get('temperature', 1.0)
        action_scores = []

        for a in valid_actions:
            atype = int(a.get('type', ActionType.FINALIZE_PARTITION))
            if atype == ActionType.SWAP_NODE:
                idx_a = node_map.get(a['node_a'], 0)
                idx_b = node_map.get(a['node_b'], 0)
                feat  = torch.tensor([1., 0., 0., 0.]).to(DEVICE)
                enc   = torch.cat([h[idx_a], h[idx_b], feat])
            elif atype == ActionType.MOVE_NODE:
                idx_nd = node_map.get(a['node'], 0)
                gid    = a['group_id']
                feat   = torch.tensor([0., 1., gid / max(state.get('num_groups',1),1), 0.]).to(DEVICE)
                enc    = torch.cat([h[idx_nd], graph_mean, feat])
            elif atype == ActionType.FINALIZE_PARTITION:
                feat   = torch.tensor([0., 0., 0., 1.]).to(DEVICE)
                enc    = torch.cat([graph_mean, graph_mean, feat])
            else:
                feat   = torch.tensor([0., 0., 0., 0.]).to(DEVICE)
                enc    = torch.cat([graph_mean, graph_mean, feat])

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

        chosen = valid_actions[idx.item()]

        # When FINALIZE is chosen, attach the weight suggestion
        if int(chosen.get('type', -1)) == ActionType.FINALIZE_PARTITION:
            num_groups  = state.get('num_groups', 1)
            w_tensor    = self.net.group_weights(h, num_groups)
            weights     = {g: w_tensor[g].item() for g in range(num_groups)}
            self._last_weights = weights
            chosen = dict(chosen)
            chosen['weights'] = weights

        return chosen

    def update(self, trajectory, final_reward):
        if not self._log_probs:
            return
        self._episode_count += 1
        progress = min(self._episode_count / max(self.total_episodes, 1), 1.0)
        self.entropy_coeff = (self.entropy_coeff_start +
                              progress * (self.entropy_coeff_end - self.entropy_coeff_start))

        n = min(len(self._log_probs), len(trajectory))
        if n == 0:
            self._clear(); return

        for i, t in enumerate(trajectory[:n]):
            self._rewards[i] = t.get('reward', 0.0)
        if self._rewards:
            self._rewards[n-1] += final_reward

        returns, advantages = compute_gae(self._rewards[:n], self._values[:n])
        ret_t = torch.tensor(returns,    dtype=torch.float32).to(DEVICE)
        adv_t = torch.tensor(advantages, dtype=torch.float32).to(DEVICE)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        lps_t = torch.stack(self._log_probs[:n])
        ent_t = torch.stack(self._entropies[:n])
        if not lps_t.requires_grad and lps_t.grad_fn is None:
            self._clear(); return

        vals_t = torch.tensor(self._values[:n], dtype=torch.float32).to(DEVICE)

        # PPO clipped surrogate (epsilon=0.2)
        # _old_log_probs stores detached log_probs from the forward pass at selection time.
        # On first update after an episode these equal lps_t so ratio=1 (identical to PG).
        # On subsequent epochs (if called multiple times) the clip kicks in.
        # Use detached log probs from action-selection time for PPO ratio.
        # Sliced to [:n] to match lps_t length exactly.
        # Use PREVIOUS episode log_probs as frozen reference for PPO ratio.
        # _prev_log_probs are set in _clear() at end of last episode.
        _prev = getattr(self, '_prev_log_probs', [])
        if _prev and len(_prev) >= n:
            old_lps_t = torch.tensor([x.detach().item() for x in _prev[:n]],
                                     dtype=torch.float32).to(DEVICE)
        else:
            old_lps_t = lps_t.detach()  # first episode: ratio=1, no clip
        ratio    = torch.exp(lps_t - old_lps_t)
        pg_loss  = -torch.min(
            ratio * adv_t.detach(),
            torch.clamp(ratio, 0.8, 1.2) * adv_t.detach()
        ).mean()
        vf_loss  = F.mse_loss(vals_t, ret_t.detach())
        ent_loss = -self.entropy_coeff * ent_t.mean()
        loss     = pg_loss + 0.5 * vf_loss + ent_loss

        if loss.grad_fn is None:
            self._clear(); return

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        self._clear()

    def _clear(self):
        # Save current log_probs as old for next episode PPO ratio
        # (not cleared — they become the frozen reference policy)
        self._prev_log_probs = list(self._log_probs)
        self._log_probs  = []; self._values = []
        self._rewards    = []; self._entropies = []
        # _old_log_probs stays as-is — it will be updated at start of next episode

    def _build_tensors(self, state):
        assignment = state['assignment']
        nodes      = list(assignment.keys())
        n          = len(nodes)
        num_groups = max(state.get('num_groups', 1), 1)
        edges      = state.get('edges', [])
        sessions   = state.get('sessions', [])
        node_to_i  = {nd: i for i, nd in enumerate(nodes)}

        adj = torch.zeros(n, n)
        for u, v in edges:
            if u in node_to_i and v in node_to_i:
                i, j = node_to_i[u], node_to_i[v]
                adj[i][j] = adj[j][i] = 1.0
        adj = adj + torch.eye(n)

        source_of = {}; sink_of = {}
        for s, t in sessions:
            source_of[s] = t; sink_of[t] = s

        groups = {}
        for nd, gid in assignment.items():
            if gid != -1:
                groups.setdefault(gid, set()).add(nd)
        internal_per_group = {
            gid: sum(1 for s, t in sessions if s in gset and t in gset)
            for gid, gset in groups.items()
        }
        cur_idx = state.get('current_node_idx', 0)

        feats = torch.zeros(n, 10)
        for i, nd in enumerate(nodes):
            gid = assignment[nd]
            feats[i,0] = 1.0 if gid != -1 else 0.0
            feats[i,1] = gid / num_groups if gid != -1 else -1.0
            if not state.get('assignment_complete', False) and i == cur_idx:
                feats[i,2] = 1.0
            feats[i,3] = adj[i].sum().item() / n
            feats[i,4] = 1.0 if nd in source_of else 0.0
            feats[i,5] = 1.0 if nd in sink_of   else 0.0
            partner = source_of.get(nd, sink_of.get(nd, None))
            if partner is not None:
                p_gid = assignment.get(partner, -1)
                feats[i,6] = p_gid / num_groups if p_gid != -1 else -1.0
                feats[i,7] = 1.0 if p_gid != -1 else 0.0
                if gid != -1:
                    feats[i,8] = 1.0 if gid == p_gid else 0.0
            else:
                feats[i,6] = -1.0
            if gid != -1:
                feats[i,9] = internal_per_group.get(gid, 0) / max(len(sessions),1)

        return feats.to(DEVICE), adj.to(DEVICE), node_to_i


# ═══════════════════════════════════════════════════════════════════
# Phase 2 — Transformer on inequality pool (extended for CROSS_SUBMOD)
# ═══════════════════════════════════════════════════════════════════

class InequalityEncoder(nn.Module):
    def __init__(self, coeff_dim, token_dim=128):
        super().__init__()
        self.proj = nn.Linear(coeff_dim, token_dim)
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, x):
        return F.relu(self.norm(self.proj(x)))


class Phase2Net(nn.Module):
    def __init__(self, coeff_dim, token_dim=128, num_heads=4, num_layers=2, ff_dim=256):
        super().__init__()
        self.encoder = InequalityEncoder(coeff_dim, token_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=num_heads,
            dim_feedforward=ff_dim, batch_first=True, norm_first=False
        )
        self.transformer    = nn.TransformerEncoder(enc_layer, num_layers)
        self.token_dim      = token_dim
        # Extended to 12 action types (includes FRACTIONAL_IO=10, CROSS_SUBMOD=11)
        self.action_type_head = nn.Sequential(
            nn.Linear(token_dim, token_dim), nn.ReLU(),
            nn.Linear(token_dim, 12)
        )
        self.ptr_query_1    = nn.Linear(token_dim, token_dim)
        self.ptr_key_1      = nn.Linear(token_dim, token_dim)
        self.ptr_query_2    = nn.Linear(token_dim * 2, token_dim)
        self.ptr_key_2      = nn.Linear(token_dim, token_dim)
        self.value_head     = nn.Sequential(
            nn.Linear(token_dim, token_dim), nn.ReLU(),
            nn.Linear(token_dim, 1)
        )

    def encode(self, coeffs):
        tokens = self.encoder(coeffs)
        tokens = self.transformer(tokens.unsqueeze(0))
        return tokens.squeeze(0)

    def pointer_scores_1(self, tokens, global_tok):
        q = self.ptr_query_1(global_tok.unsqueeze(0))
        k = self.ptr_key_1(tokens)
        return (q @ k.T).squeeze(0) / (self.token_dim ** 0.5)

    def pointer_scores_2(self, tokens, global_tok, selected_tok):
        context = torch.cat([global_tok, selected_tok])
        q       = self.ptr_query_2(context.unsqueeze(0))
        k       = self.ptr_key_2(tokens)
        return (q @ k.T).squeeze(0) / (self.token_dim ** 0.5)


class GNNPhase2Policy:
    def __init__(self, coeff_dim=256, token_dim=128, num_heads=4, num_layers=2,
                 lr=3e-4, entropy_coeff_start=0.15, entropy_coeff_end=0.01,
                 total_episodes=10000):
        self.net = Phase2Net(coeff_dim, token_dim, num_heads, num_layers).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(total_episodes, 1), eta_min=1e-5
        )
        self.coeff_dim           = coeff_dim
        self.entropy_coeff_start = entropy_coeff_start
        self.entropy_coeff_end   = entropy_coeff_end
        self.entropy_coeff       = entropy_coeff_start
        self.total_episodes      = total_episodes
        self._episode_count      = 0
        self._frozen             = False
        self._log_probs:  List[torch.Tensor] = []
        self._values:     List[float]         = []
        self._rewards:    List[float]         = []
        self._entropies:  List[torch.Tensor]  = []

    def freeze(self):
        for p in self.net.parameters(): p.requires_grad = False
        self._frozen = True

    def unfreeze(self):
        for p in self.net.parameters(): p.requires_grad = True
        self._frozen = False

    def reset_scheduler(self, total_episodes=None):
        if total_episodes is not None:
            self.total_episodes = total_episodes
        self._episode_count = 0
        self.entropy_coeff  = self.entropy_coeff_start
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(self.total_episodes, 1), eta_min=1e-5
        )

    def select_action(self, state, valid_actions):
        if not valid_actions:
            return {'type': ActionType.DECLARE_TERMINAL}

        tokens, global_tok = self._encode(state)

        # Action type selection (12 types now)
        type_logits = self.net.action_type_head(global_tok)
        valid_types = {int(a['type']) for a in valid_actions}
        type_mask   = torch.full((12,), float('-inf')).to(DEVICE)
        for t in valid_types:
            if t < 12:
                type_mask[t] = 0.0

        type_probs = F.softmax(type_logits + type_mask, dim=-1)
        type_dist  = torch.distributions.Categorical(type_probs)
        atype_t    = type_dist.sample()
        atype      = ActionType(atype_t.item())
        lp_type    = type_dist.log_prob(atype_t)
        entropy    = type_dist.entropy()

        # PROOF2 forcing (annealed)
        proof2_fp = state.get('proof2_force_prob', 0.0)
        if (proof2_fp > 0 and ActionType.APPLY_PROOF2 in valid_types
                and _random.random() < proof2_fp):
            atype   = ActionType.APPLY_PROOF2
            atype_t = torch.tensor(int(atype)).to(DEVICE)
            lp_type = type_dist.log_prob(atype_t)

        value = self.net.value_head(global_tok).squeeze()
        self._values.append(value.item())

        lp_idx = torch.tensor(0.0).to(DEVICE)
        action = {'type': atype}

        # Index selection (pointer network) for actions that need one
        if atype in (ActionType.ADD_TO_ACCUMULATOR,):
            idxs = [a['idx_i'] for a in valid_actions if int(a['type']) == int(atype)]
            if idxs and len(tokens) > 0:
                mappable = [i for i in idxs if i < len(tokens)]
                if mappable:
                    idx, lp_idx = self._pointer_1(tokens, global_tok, mappable)
                    action['idx_i'] = idx
                else:
                    action['idx_i'] = idxs[0]
            elif idxs:
                action['idx_i'] = idxs[0]

        elif atype in (ActionType.APPLY_SUBMODULARITY, ActionType.CROSS_SUBMOD):
            pairs = [(a['idx_i'], a['idx_j']) for a in valid_actions
                     if int(a['type']) == int(atype)]
            if pairs:
                acc = state.get('accumulator_coeffs', None)
                if acc is not None and len(acc) > 0:
                    acc_t   = self._pad(torch.tensor(acc, dtype=torch.float32).to(DEVICE))
                    acc_tok = self.net.encode(acc_t)
                    n_tok   = len(acc_tok)
                    offset  = max(0, state.get('accumulator_size', n_tok) - n_tok)
                    all_i   = sorted(set(p[0] for p in pairs))
                    map_i   = [i - offset for i in all_i if 0 <= i - offset < n_tok]
                    if map_i:
                        i_tok, lp_i = self._pointer_1(acc_tok, acc_tok.mean(0), map_i)
                        i_sel = i_tok + offset
                    else:
                        i_sel = all_i[0]; lp_i = torch.tensor(0.0).to(DEVICE)
                    valid_j = sorted(set(p[1] for p in pairs if p[0] == i_sel)) or [pairs[0][1]]
                    if len(valid_j) == 1:
                        j_sel = valid_j[0]; lp_j = torch.tensor(0.0).to(DEVICE)
                    else:
                        map_j = [j - offset for j in valid_j if 0 <= j - offset < n_tok]
                        if map_j:
                            i_clamped = min(max(i_sel - offset, 0), n_tok - 1)
                            j_tok, lp_j = self._pointer_2(acc_tok, acc_tok.mean(0), acc_tok[i_clamped], map_j)
                            j_sel = j_tok + offset
                        else:
                            j_sel = valid_j[0]; lp_j = torch.tensor(0.0).to(DEVICE)
                    lp_idx = lp_i + lp_j
                else:
                    i_sel, j_sel = pairs[0]; lp_idx = torch.tensor(0.0).to(DEVICE)
                action['idx_i'] = i_sel
                action['idx_j'] = j_sel

        elif atype == ActionType.COMBINE_STORED:
            pairs = [(a['idx_i'], a['idx_j']) for a in valid_actions
                     if int(a['type']) == int(atype)]
            if pairs:
                action['idx_i'] = pairs[0][0]
                action['idx_j'] = pairs[0][1]

        total_lp = lp_type + lp_idx
        self._log_probs.append(total_lp)
        self._rewards.append(0.0)
        self._entropies.append(entropy)
        return action

    def update(self, trajectory, final_reward):
        if self._frozen or not self._log_probs:
            self._clear(); return

        self._episode_count += 1
        progress = min(self._episode_count / max(self.total_episodes, 1), 1.0)
        self.entropy_coeff = (self.entropy_coeff_start +
                              progress * (self.entropy_coeff_end - self.entropy_coeff_start))

        n = min(len(self._log_probs), len(trajectory))
        if n == 0:
            self._clear(); return

        for i, t in enumerate(trajectory[:n]):
            self._rewards[i] = t.get('reward', 0.0)
        if self._rewards:
            self._rewards[n-1] += final_reward

        returns, advantages = compute_gae(self._rewards[:n], self._values[:n])
        ret_t = torch.tensor(returns,    dtype=torch.float32).to(DEVICE)
        adv_t = torch.tensor(advantages, dtype=torch.float32).to(DEVICE)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        lps_t = torch.stack(self._log_probs[:n])
        ent_t = torch.stack(self._entropies[:n])
        if not lps_t.requires_grad and lps_t.grad_fn is None:
            self._clear(); return

        vals_t = torch.tensor(self._values[:n], dtype=torch.float32).to(DEVICE)
        # PPO clipped surrogate (epsilon=0.2)
        # _old_log_probs stores detached log_probs from the forward pass at selection time.
        # On first update after an episode these equal lps_t so ratio=1 (identical to PG).
        # On subsequent epochs (if called multiple times) the clip kicks in.
        # Use detached log probs from action-selection time for PPO ratio.
        # Sliced to [:n] to match lps_t length exactly.
        # Use PREVIOUS episode log_probs as frozen reference for PPO ratio.
        # _prev_log_probs are set in _clear() at end of last episode.
        _prev = getattr(self, '_prev_log_probs', [])
        if _prev and len(_prev) >= n:
            old_lps_t = torch.tensor([x.detach().item() for x in _prev[:n]],
                                     dtype=torch.float32).to(DEVICE)
        else:
            old_lps_t = lps_t.detach()  # first episode: ratio=1, no clip
        ratio    = torch.exp(lps_t - old_lps_t)
        pg_loss  = -torch.min(
            ratio * adv_t.detach(),
            torch.clamp(ratio, 0.8, 1.2) * adv_t.detach()
        ).mean()
        vf_loss  = F.mse_loss(vals_t, ret_t.detach())
        ent_loss = -self.entropy_coeff * ent_t.mean()
        loss     = pg_loss + 0.5 * vf_loss + ent_loss

        if loss.grad_fn is None:
            self._clear(); return

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        self._clear()

    def _clear(self):
        # Save current log_probs as old for next episode PPO ratio
        # (not cleared — they become the frozen reference policy)
        self._prev_log_probs = list(self._log_probs)
        self._log_probs  = []; self._values = []
        self._rewards    = []; self._entropies = []
        # _old_log_probs stays as-is — it will be updated at start of next episode

    def _encode(self, state):
        pool = state.get('pool_coeffs', None)
        if pool is not None and len(pool) > 0:
            t = self._pad(torch.tensor(pool, dtype=torch.float32).to(DEVICE))
        else:
            t = torch.zeros(1, self.coeff_dim).to(DEVICE)
        tokens     = self.net.encode(t)
        global_tok = tokens.mean(dim=0)
        return tokens, global_tok

    def _pad(self, x):
        d = x.shape[-1]
        if d == self.coeff_dim:  return x
        if d < self.coeff_dim:
            pad = torch.zeros(*x.shape[:-1], self.coeff_dim - d).to(DEVICE)
            return torch.cat([x, pad], dim=-1)
        return x[..., :self.coeff_dim]

    def _pointer_1(self, tokens, global_tok, valid_idxs):
        scores = self.net.pointer_scores_1(tokens, global_tok)
        mask   = torch.full((len(tokens),), float('-inf')).to(DEVICE)
        any_v  = False
        for vi in valid_idxs:
            if vi < len(tokens):
                mask[vi] = 0.0; any_v = True
        if not any_v:
            mask = torch.zeros(len(tokens)).to(DEVICE)
        probs = F.softmax(scores + mask, dim=-1)
        if torch.isnan(probs).any():
            probs = torch.ones_like(probs) / len(probs)
        dist  = torch.distributions.Categorical(probs)
        idx_t = dist.sample()
        return idx_t.item(), dist.log_prob(idx_t)

    def _pointer_2(self, tokens, global_tok, selected_tok, valid_idxs):
        scores = self.net.pointer_scores_2(tokens, global_tok, selected_tok)
        mask   = torch.full((len(tokens),), float('-inf')).to(DEVICE)
        any_v  = False
        for vi in valid_idxs:
            if vi < len(tokens):
                mask[vi] = 0.0; any_v = True
        if not any_v:
            mask = torch.zeros(len(tokens)).to(DEVICE)
        probs = F.softmax(scores + mask, dim=-1)
        if torch.isnan(probs).any():
            probs = torch.ones_like(probs) / len(probs)
        dist  = torch.distributions.Categorical(probs)
        idx_t = dist.sample()
        return idx_t.item(), dist.log_prob(idx_t)


# ═══════════════════════════════════════════════════════════════════
# Phase 3 — Joint graph+pool policy for fractional IO search
# ═══════════════════════════════════════════════════════════════════

class Phase3Net(nn.Module):
    """
    Encodes both the graph structure (via SAGELayer) and the fractional
    pool (via Transformer) then combines them for action selection.

    Action heads:
      action_type_head : 6 logits (subset of ActionType for Phase 3)
      node_scorer      : scores each node for FRACTIONAL_IO u/v selection
      lambda_head      : 7 logits over LAMBDA_GRID
      pool_ptr         : pointer over pool for ADD_TO_ACCUMULATOR
      value_head       : scalar baseline
    """
    def __init__(self, graph_hidden=64, coeff_dim=256, token_dim=128,
                 num_sage_layers=3, max_nodes=32):
        super().__init__()
        self.max_nodes = max_nodes

        # Graph encoder (reuse SAGE from Phase 1)
        dims = [10] + [graph_hidden] * num_sage_layers
        self.sage_layers = nn.ModuleList([
            SAGELayer(dims[i], dims[i+1]) for i in range(num_sage_layers)
        ])

        # Pool encoder (stable MLP — Transformer replaced due to GPU instability)
        self.pool_proj = nn.Linear(coeff_dim, token_dim)
        self.pool_norm = nn.LayerNorm(token_dim)
        # pool_transformer removed — encode_pool now uses mean-pool MLP

        # Combined representation
        comb_dim = graph_hidden + token_dim
        self.combiner = nn.Sequential(
            nn.Linear(comb_dim, 256), nn.ReLU(),
            nn.Linear(256, 128)
        )

        # Action type: FRACTIONAL_IO, ADD, SUBMOD, CROSS_SUBMOD, STORE, DECLARE
        self.action_type_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 6)
        )
        # Node scorer (for u and v selection in FRACTIONAL_IO)
        self.node_scorer = nn.Linear(graph_hidden, 1)

        # Lambda selector
        self.lambda_head = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, len(LAMBDA_GRID))
        )

        # Pool pointer
        self.pool_ptr_q = nn.Linear(128, token_dim)
        self.pool_ptr_k = nn.Linear(token_dim, token_dim)

        # Value
        self.value_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.token_dim    = token_dim
        self.graph_hidden = graph_hidden
        self.coeff_dim    = coeff_dim

    def encode_graph(self, x, adj):
        h = x
        for layer in self.sage_layers:
            h = layer(h, adj)
        return h   # (n_nodes, graph_hidden)

    def encode_pool(self, pool_coeffs):
        """
        Stable pool encoding: normalize → per-row MLP.
        Returns (per_row_tokens, mean_token):
          per_row_tokens: (n_ineqs, token_dim) — used by pool pointer
          mean_token:     (token_dim,)          — used by combined()
        """
        pool_coeffs = pool_coeffs.clamp(-10.0, 10.0)
        mu  = pool_coeffs.mean(dim=0, keepdim=True)
        std = pool_coeffs.std(dim=0, keepdim=True).clamp(min=1e-4)
        pc  = (pool_coeffs - mu) / std
        pc  = torch.nan_to_num(pc, nan=0.0, posinf=1.0, neginf=-1.0)
        tokens = F.relu(self.pool_norm(self.pool_proj(pc)))  # (n, token_dim)
        mean   = tokens.mean(dim=0)                          # (token_dim,)
        return tokens, mean

    def combined(self, h_graph, h_pool_mean):
        """h_pool_mean: (token_dim,) mean token from encode_pool."""
        g = h_graph.mean(dim=0)   # (graph_hidden,)
        p = h_pool_mean if h_pool_mean.dim() == 1 else h_pool_mean.squeeze(0)
        return self.combiner(torch.cat([g, p]))   # (128,)


PHASE3_ACTION_TYPES = [
    ActionType.FRACTIONAL_IO,
    ActionType.ADD_TO_ACCUMULATOR,
    ActionType.APPLY_SUBMODULARITY,
    ActionType.CROSS_SUBMOD,
    ActionType.STORE_AND_RESET,
    ActionType.DECLARE_TERMINAL,
]


class GNNPhase3Policy:
    """
    Phase 3 policy: learns the fractional IO proof search.

    Key design decisions:
    1. Separate node scorer for u/v selection in FRACTIONAL_IO so the
       policy learns which cross-partition pairs are promising.
    2. Lambda head learns which weight to use — starts with high entropy
       (all lambdas equally likely) and sharpens over training.
    3. Pool pointer for ADD_TO_ACCUMULATOR learns which inequality to
       select from the fractional pool.
    4. High entropy coefficient (0.2 → 0.02) — Phase 3 needs more
       exploration than Phases 1/2 because the search space is larger.
    """

    def __init__(self, coeff_dim=256, graph_hidden=64, token_dim=128,
                 lr=3e-5, entropy_coeff_start=0.10, entropy_coeff_end=0.01,
                 total_episodes=10000):
        self.net = Phase3Net(
            graph_hidden=graph_hidden, coeff_dim=coeff_dim,
            token_dim=token_dim
        ).to(DEVICE)
        # AdamW with weight decay prevents weight explosion on GPU
        self.optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=lr, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(total_episodes, 1), eta_min=1e-5
        )
        self.coeff_dim           = coeff_dim
        self.entropy_coeff_start = entropy_coeff_start
        self.entropy_coeff_end   = entropy_coeff_end
        self.entropy_coeff       = entropy_coeff_start
        self.total_episodes      = total_episodes
        self._episode_count      = 0
        self._log_probs:  List[torch.Tensor] = []
        self._values:     List[float]         = []
        self._rewards:    List[float]         = []
        self._entropies:  List[torch.Tensor]  = []

    def reset_scheduler(self, total_episodes=None):
        if total_episodes is not None:
            self.total_episodes = total_episodes
        self._episode_count = 0
        self.entropy_coeff  = self.entropy_coeff_start
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(self.total_episodes, 1), eta_min=1e-5
        )

    def select_action(self, state, valid_actions):
        if not valid_actions:
            return {'type': ActionType.DECLARE_TERMINAL}

        h_graph, adj, node_map = self._build_graph_tensors(state)
        h_g = self.net.encode_graph(h_graph, adj)

        pool_coeffs = state.get('pool_coeffs', None)
        if pool_coeffs is not None and len(pool_coeffs) > 0:
            pc = self._pad_pool(torch.tensor(pool_coeffs, dtype=torch.float32).to(DEVICE))
            h_p_tokens, h_p_mean = self.net.encode_pool(pc)
        else:
            h_p_tokens = torch.zeros(0, self.net.token_dim).to(DEVICE)
            h_p_mean   = torch.zeros(self.net.token_dim).to(DEVICE)

        h_comb = self.net.combined(h_g, h_p_mean)
        value   = self.net.value_head(h_comb).squeeze()
        self._values.append(value.item())

        # Action type selection
        valid_types = {int(a['type']) for a in valid_actions}
        type_logits = self.net.action_type_head(h_comb)
        # Map PHASE3_ACTION_TYPES index → logit
        type_mask = torch.full((6,), float('-inf')).to(DEVICE)
        for k, at in enumerate(PHASE3_ACTION_TYPES):
            if int(at) in valid_types:
                type_mask[k] = 0.0
        # Safety: if nothing mapped (shouldn't happen) allow all to prevent NaN
        if (type_mask == float('-inf')).all():
            type_mask = torch.zeros(6).to(DEVICE)

        type_probs = F.softmax(type_logits + type_mask, dim=-1)
        type_dist  = torch.distributions.Categorical(type_probs)
        k_t        = type_dist.sample()
        atype      = PHASE3_ACTION_TYPES[k_t.item()]
        lp_type    = type_dist.log_prob(k_t)
        entropy    = type_dist.entropy()

        lp_extra = torch.tensor(0.0).to(DEVICE)
        action   = {'type': atype}

        if atype == ActionType.FRACTIONAL_IO:
            # Select node_u, node_v (from different partitions), and λ
            fio_actions = [a for a in valid_actions
                           if int(a['type']) == ActionType.FRACTIONAL_IO]
            if fio_actions:
                # Score nodes using graph embedding
                node_scores = self.net.node_scorer(h_g).squeeze(-1)  # (n_nodes,)
                nodes_list  = list(node_map.keys())

                # Select u
                u_mask = torch.full((len(nodes_list),), float('-inf')).to(DEVICE)
                u_nodes = list({a['node_u'] for a in fio_actions if a['node_u'] in node_map})
                for nd in u_nodes:
                    if nd in node_map: u_mask[node_map[nd]] = 0.0
                u_probs = F.softmax(node_scores + u_mask, dim=-1)
                u_dist  = torch.distributions.Categorical(u_probs)
                u_idx_t = u_dist.sample()
                u_node  = nodes_list[u_idx_t.item()]
                lp_u    = u_dist.log_prob(u_idx_t)

                # Select v (different partition from u)
                v_mask  = torch.full((len(nodes_list),), float('-inf')).to(DEVICE)
                v_nodes = list({a['node_v'] for a in fio_actions
                                if a['node_u'] == u_node and a['node_v'] in node_map})
                if not v_nodes:
                    v_nodes = list({a['node_v'] for a in fio_actions if a['node_v'] in node_map})
                for nd in v_nodes:
                    if nd in node_map: v_mask[node_map[nd]] = 0.0
                v_probs = F.softmax(node_scores + v_mask, dim=-1)
                v_dist  = torch.distributions.Categorical(v_probs)
                v_idx_t = v_dist.sample()
                v_node  = nodes_list[v_idx_t.item()]
                lp_v    = v_dist.log_prob(v_idx_t)

                # Select λ
                lam_logits = self.net.lambda_head(h_comb)
                lam_probs  = F.softmax(lam_logits, dim=-1)
                lam_dist   = torch.distributions.Categorical(lam_probs)
                lam_idx_t  = lam_dist.sample()
                lam        = LAMBDA_GRID[lam_idx_t.item()]
                lp_lam     = lam_dist.log_prob(lam_idx_t)

                action['node_u'] = u_node
                action['node_v'] = v_node
                action['lam']    = lam
                lp_extra = lp_u + lp_v + lp_lam
            else:
                action = {'type': ActionType.DECLARE_TERMINAL}

        elif atype == ActionType.ADD_TO_ACCUMULATOR:
            add_actions = [a for a in valid_actions
                           if int(a['type']) == ActionType.ADD_TO_ACCUMULATOR]
            if add_actions and h_p_tokens.shape[0] > 0:
                idxs   = [a['idx_i'] for a in add_actions]
                n_pool = h_p_tokens.shape[0]
                valid_mapped = [i for i in idxs if i < n_pool]
                if valid_mapped:
                    q      = self.net.pool_ptr_q(h_comb).unsqueeze(0)
                    k      = self.net.pool_ptr_k(h_p_tokens)
                    scores = (q @ k.T).squeeze(0) / (self.net.token_dim ** 0.5)
                    mask   = torch.full((n_pool,), float('-inf')).to(DEVICE)
                    for i in valid_mapped: mask[i] = 0.0
                    probs  = F.softmax(scores + mask, dim=-1)
                    dist   = torch.distributions.Categorical(probs)
                    idx_t  = dist.sample()
                    action['idx_i'] = idx_t.item()
                    lp_extra = dist.log_prob(idx_t)
                else:
                    action['idx_i'] = add_actions[0]['idx_i']
            elif add_actions:
                action['idx_i'] = add_actions[0]['idx_i']

        elif atype in (ActionType.APPLY_SUBMODULARITY, ActionType.CROSS_SUBMOD):
            pairs = [(a['idx_i'], a['idx_j']) for a in valid_actions
                     if int(a['type']) == int(atype)]
            if pairs:
                action['idx_i'] = pairs[0][0]
                action['idx_j'] = pairs[0][1]

        total_lp = lp_type + lp_extra
        self._log_probs.append(total_lp)
        self._rewards.append(0.0)
        self._entropies.append(entropy)
        return action

    def update(self, trajectory, final_reward):
        if not self._log_probs:
            return
        self._episode_count += 1
        progress = min(self._episode_count / max(self.total_episodes, 1), 1.0)
        self.entropy_coeff = (self.entropy_coeff_start +
                              progress * (self.entropy_coeff_end - self.entropy_coeff_start))

        n = min(len(self._log_probs), len(trajectory))
        if n == 0:
            self._clear(); return

        for i, t in enumerate(trajectory[:n]):
            self._rewards[i] = t.get('reward', 0.0)
        if self._rewards:
            self._rewards[n-1] += final_reward

        # Clip rewards tightly — GPU runs faster, larger gradients per step
        # Normalise to [-1, +1] to keep loss scale stable across all graphs
        raw = [max(-5.0, min(25.0, r)) for r in self._rewards]
        r_min, r_max = min(raw), max(raw)
        r_range = max(r_max - r_min, 1e-6)
        self._rewards = [2.0 * (r - r_min) / r_range - 1.0 for r in raw]
        returns, advantages = compute_gae(self._rewards[:n], self._values[:n])
        ret_t = torch.tensor(returns,    dtype=torch.float32).to(DEVICE)
        adv_t = torch.tensor(advantages, dtype=torch.float32).to(DEVICE)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        lps_t = torch.stack(self._log_probs[:n])
        ent_t = torch.stack(self._entropies[:n])
        if not lps_t.requires_grad and lps_t.grad_fn is None:
            self._clear(); return

        vals_t = torch.tensor(self._values[:n], dtype=torch.float32).to(DEVICE)
        # PPO clipped surrogate (epsilon=0.2)
        # _old_log_probs stores detached log_probs from the forward pass at selection time.
        # On first update after an episode these equal lps_t so ratio=1 (identical to PG).
        # On subsequent epochs (if called multiple times) the clip kicks in.
        # Use detached log probs from action-selection time for PPO ratio.
        # Sliced to [:n] to match lps_t length exactly.
        # Use PREVIOUS episode log_probs as frozen reference for PPO ratio.
        # _prev_log_probs are set in _clear() at end of last episode.
        _prev = getattr(self, '_prev_log_probs', [])
        if _prev and len(_prev) >= n:
            old_lps_t = torch.tensor([x.detach().item() for x in _prev[:n]],
                                     dtype=torch.float32).to(DEVICE)
        else:
            old_lps_t = lps_t.detach()  # first episode: ratio=1, no clip
        ratio    = torch.exp(lps_t - old_lps_t)
        pg_loss  = -torch.min(
            ratio * adv_t.detach(),
            torch.clamp(ratio, 0.8, 1.2) * adv_t.detach()
        ).mean()
        vf_loss  = F.mse_loss(vals_t, ret_t.detach())
        ent_loss = -self.entropy_coeff * ent_t.mean()
        loss     = pg_loss + 0.5 * vf_loss + ent_loss

        if loss.grad_fn is None:
            self._clear(); return

        self.optimizer.zero_grad()
        loss.backward()

        # Check for NaN gradients BEFORE clipping/stepping
        has_nan_grad = any(
            p.grad is not None and torch.isnan(p.grad).any()
            for p in self.net.parameters()
        )
        if has_nan_grad:
            # Skip this update entirely — NaN gradient means the loss
            # computation produced an invalid result. Clearing and moving
            # on is safer than trying to clip NaN values.
            self.optimizer.zero_grad()
            self._clear()
            return

        nn.utils.clip_grad_norm_(self.net.parameters(), 0.2)
        self.optimizer.step()
        self.scheduler.step()

        # Post-step NaN check — should now be rare with the above guard
        for name, param in self.net.named_parameters():
            if torch.isnan(param).any():
                nn.init.xavier_uniform_(param.data) if param.dim() >= 2 else nn.init.zeros_(param.data)

        self._clear()

    def _clear(self):
        # Save current log_probs as old for next episode PPO ratio
        # (not cleared — they become the frozen reference policy)
        self._prev_log_probs = list(self._log_probs)
        self._log_probs  = []; self._values = []
        self._rewards    = []; self._entropies = []
        # _old_log_probs stays as-is — it will be updated at start of next episode

    def _pad_pool(self, x):
        d = x.shape[-1]
        if d == self.coeff_dim: return x
        if d < self.coeff_dim:
            pad = torch.zeros(*x.shape[:-1], self.coeff_dim - d).to(DEVICE)
            return torch.cat([x, pad], dim=-1)
        return x[..., :self.coeff_dim]

    def _build_graph_tensors(self, state):
        """Build graph features from Phase 3 state (reuse Phase 1 logic)."""
        nodes    = state.get('nodes', [])
        edges    = state.get('edges', [])
        sessions = state.get('sessions', [])
        n        = len(nodes)
        if n == 0:
            return (torch.zeros(1, 10).to(DEVICE),
                    torch.eye(1).to(DEVICE), {})

        node_to_i = {nd: i for i, nd in enumerate(nodes)}
        adj = torch.zeros(n, n)
        for u, v in edges:
            if u in node_to_i and v in node_to_i:
                i, j = node_to_i[u], node_to_i[v]
                adj[i][j] = adj[j][i] = 1.0
        adj = adj + torch.eye(n)

        feats = torch.zeros(n, 10)
        part_of = {}
        partition = state.get('partition', [])
        for pid, Pi in enumerate(partition):
            for nd in Pi: part_of[nd] = pid

        n_parts = max(len(partition), 1)
        pw = state.get('partition_weights', {})
        source_of = {}; sink_of = {}
        for s, t in sessions:
            source_of[s] = t; sink_of[t] = s

        for i, nd in enumerate(nodes):
            pid = part_of.get(nd, -1)
            feats[i,0] = 1.0 if pid >= 0 else 0.0
            feats[i,1] = pid / n_parts if pid >= 0 else -1.0
            feats[i,2] = pw.get(pid, 0.0)
            feats[i,3] = adj[i].sum().item() / n
            feats[i,4] = 1.0 if nd in source_of else 0.0
            feats[i,5] = 1.0 if nd in sink_of   else 0.0

        return feats.to(DEVICE), adj.to(DEVICE), node_to_i