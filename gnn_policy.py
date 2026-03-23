"""
GNN-based PPO policies for Phase 1 and Phase 2.
Plain PyTorch only — no torch_geometric required.
GPU support via DEVICE — automatically uses CUDA if available.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Tuple
from enum import IntEnum

# -----------------------------------------------------------------------
# Device setup — automatically uses GPU if available
# -----------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


class ActionType(IntEnum):
    ASSIGN_NODE         = 0
    ADD_TO_ACCUMULATOR  = 1
    APPLY_SUBMODULARITY = 2
    APPLY_PROOF2        = 3
    STORE_AND_RESET     = 4
    COMBINE_STORED      = 5
    DECLARE_TERMINAL    = 6


# -----------------------------------------------------------------------
# Shared utility
# -----------------------------------------------------------------------

def compute_returns(rewards: List[float], gamma: float = 0.99) -> List[float]:
    """Simple discounted returns."""
    returns, R = [], 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


# -----------------------------------------------------------------------
# Phase 1 — GraphSAGE-style GNN
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
                 max_groups: int = 128):
        super().__init__()
        dims        = [3] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList([
            SAGELayer(dims[i], dims[i+1]) for i in range(num_layers)
        ])
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, max_groups)
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.max_groups = max_groups

    def forward(self, x, adj):
        h = x
        for layer in self.layers:
            h = layer(h, adj)
        return self.policy_head(h), self.value_head(h)


class GNNPhase1Policy:

    def __init__(self, hidden_dim: int = 64, num_layers: int = 3,
                 max_groups: int = 128, lr: float = 1e-4):
        self.net        = Phase1GNN(hidden_dim, num_layers, max_groups).to(DEVICE)
        self.optimizer  = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.max_groups = max_groups
        self._log_probs : List[torch.Tensor] = []
        self._values    : List[torch.Tensor] = []
        self._rewards   : List[float]        = []

    def select_action(self, state: dict, valid_actions: List[dict]) -> dict:
        x, adj, cur_idx = self._build_tensors(state)
        logits, value   = self.net(x, adj)
        cur_logits      = logits[cur_idx]

        valid_gids = [a["group_id"] for a in valid_actions]
        mask       = torch.full((self.max_groups,), float('-inf')).to(DEVICE)
        for gid in valid_gids:
            if gid < self.max_groups:
                mask[gid] = 0.0

        probs    = F.softmax(cur_logits + mask, dim=-1)
        dist     = torch.distributions.Categorical(probs)
        gid_t    = dist.sample()

        self._log_probs.append(dist.log_prob(gid_t))
        self._values.append(value[cur_idx].squeeze())
        self._rewards.append(0.0)

        return {"type": ActionType.ASSIGN_NODE, "group_id": gid_t.item()}

    def update(self, trajectory: list, final_reward: float):
        if not self._log_probs:
            return

        n = min(len(self._log_probs), len(trajectory))
        if n == 0:
            self._clear()
            return

        for i, t in enumerate(trajectory[:n]):
            self._rewards[i] = t.get("reward", 0.0)
        if self._rewards:
            self._rewards[n - 1] += final_reward

        returns = compute_returns(self._rewards[:n])
        ret_t   = torch.tensor(returns, dtype=torch.float32).to(DEVICE)
        vals_t  = torch.stack(self._values[:n])
        lps_t   = torch.stack(self._log_probs[:n])

        if not lps_t.requires_grad and lps_t.grad_fn is None:
            self._clear()
            return

        adv  = (ret_t - vals_t.detach())
        adv  = (adv - adv.mean()) / (adv.std() + 1e-8)

        pg_loss = -(lps_t * adv).mean()
        vf_loss = F.mse_loss(vals_t, ret_t)
        loss    = pg_loss + 0.5 * vf_loss

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

    def _build_tensors(self, state):
        assignment = state["assignment"]
        nodes      = list(assignment.keys())
        n          = len(nodes)
        cur_idx    = state["current_node_idx"]
        num_groups = max(state["num_groups"], 1)

        feats = torch.zeros(n, 3)
        for i, nd in enumerate(nodes):
            gid = assignment[nd]
            if gid != -1:
                feats[i, 0] = 1.0
                feats[i, 1] = gid / num_groups
            else:
                feats[i, 1] = -1.0
            if i == cur_idx:
                feats[i, 2] = 1.0

        adj = torch.eye(n)

        # move to GPU
        feats = feats.to(DEVICE)
        adj   = adj.to(DEVICE)
        return feats, adj, cur_idx


# -----------------------------------------------------------------------
# Phase 2 — Transformer encoder + pointer network
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
        self.ptr_query  = nn.Linear(token_dim, token_dim)
        self.ptr_key    = nn.Linear(token_dim, token_dim)
        self.value_head = nn.Sequential(
            nn.Linear(token_dim, token_dim), nn.ReLU(),
            nn.Linear(token_dim, 1)
        )

    def encode(self, coeffs: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(coeffs)
        tokens = self.transformer(tokens.unsqueeze(0))
        return tokens.squeeze(0)

    def pointer_scores(self, tokens: torch.Tensor,
                       global_tok: torch.Tensor) -> torch.Tensor:
        q = self.ptr_query(global_tok.unsqueeze(0))
        k = self.ptr_key(tokens)
        return (q @ k.T).squeeze(0) / (self.token_dim ** 0.5)


class GNNPhase2Policy:

    def __init__(self, coeff_dim: int = 256, token_dim: int = 128,
                 num_heads: int = 4, num_layers: int = 2, lr: float = 1e-4):
        self.net       = Phase2Net(coeff_dim, token_dim, num_heads,
                                   num_layers).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.coeff_dim = coeff_dim
        self._frozen   = False
        self._log_probs : List[torch.Tensor] = []
        self._values    : List[torch.Tensor] = []
        self._rewards   : List[float]        = []

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
            return {"type": ActionType.DECLARE_TERMINAL}

        tokens, global_tok = self._encode(state)

        type_logits = self.net.action_type_head(global_tok)
        valid_types = {int(a["type"]) for a in valid_actions}
        type_mask   = torch.full((7,), float('-inf')).to(DEVICE)
        for t in valid_types:
            type_mask[t] = 0.0

        type_probs = F.softmax(type_logits + type_mask, dim=-1)
        type_dist  = torch.distributions.Categorical(type_probs)
        atype_t    = type_dist.sample()
        atype      = ActionType(atype_t.item())
        lp_type    = type_dist.log_prob(atype_t)

        value  = self.net.value_head(global_tok).squeeze()
        lp_idx = torch.tensor(0.0).to(DEVICE)
        action = {"type": atype}

        if atype == ActionType.ADD_TO_ACCUMULATOR:
            idxs = [a["idx_i"] for a in valid_actions
                    if int(a["type"]) == int(atype)]
            if idxs and len(tokens) > 0:
                idx, lp_i = self._pointer(tokens, idxs)
                action["idx_i"] = idx
                lp_idx = lp_i

        elif atype == ActionType.APPLY_SUBMODULARITY:
            pairs = [(a["idx_i"], a["idx_j"]) for a in valid_actions
                     if int(a["type"]) == int(atype)]
            if pairs:
                acc = state.get("accumulator_coeffs", None)
                if acc is not None and len(acc) > 0:
                    acc_t   = self._pad(
                        torch.tensor(acc, dtype=torch.float32).to(DEVICE)
                    )
                    acc_tok = self.net.encode(acc_t)
                    all_i   = list({p[0] for p in pairs})
                    i_sel, lp_i = self._pointer(acc_tok, all_i)
                else:
                    i_sel = pairs[0][0]
                    lp_i  = torch.tensor(0.0).to(DEVICE)
                valid_j = [p[1] for p in pairs if p[0] == i_sel] or [pairs[0][1]]
                j_sel   = valid_j[np.random.randint(len(valid_j))]
                lp_idx  = lp_i - np.log(max(len(valid_j), 1))
                action["idx_i"] = i_sel
                action["idx_j"] = j_sel

        elif atype == ActionType.COMBINE_STORED:
            pairs = [(a["idx_i"], a["idx_j"]) for a in valid_actions
                     if int(a["type"]) == int(atype)]
            if pairs:
                pi = np.random.randint(len(pairs))
                action["idx_i"] = pairs[pi][0]
                action["idx_j"] = pairs[pi][1]
                lp_idx = torch.tensor(-np.log(len(pairs))).to(DEVICE)

        self._log_probs.append(lp_type + lp_idx)
        self._values.append(value)
        self._rewards.append(0.0)

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
            self._rewards[i] = t.get("reward", 0.0)
        if self._rewards:
            self._rewards[n - 1] += final_reward

        returns = compute_returns(self._rewards[:n])
        ret_t   = torch.tensor(returns, dtype=torch.float32).to(DEVICE)
        vals_t  = torch.stack(self._values[:n])
        lps_t   = torch.stack(self._log_probs[:n])

        if not lps_t.requires_grad and lps_t.grad_fn is None:
            self._clear()
            return

        adv  = (ret_t - vals_t.detach())
        adv  = (adv - adv.mean()) / (adv.std() + 1e-8)

        pg_loss = -(lps_t * adv).mean()
        vf_loss = F.mse_loss(vals_t, ret_t)
        entropy = -(torch.exp(lps_t.detach()) * lps_t).mean()
        loss    = pg_loss + 0.5 * vf_loss - 0.01 * entropy

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

    def _encode(self, state):
        pool = state.get("pool_coeffs", None)
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

    def _pointer(self, tokens, valid_idxs):
        global_tok = tokens.mean(dim=0)
        scores     = self.net.pointer_scores(tokens, global_tok)
        mask       = torch.full((len(tokens),), float('-inf')).to(DEVICE)
        for vi in valid_idxs:
            if vi < len(tokens):
                mask[vi] = 0.0
        probs = F.softmax(scores + mask, dim=-1)
        dist  = torch.distributions.Categorical(probs)
        idx_t = dist.sample()
        return idx_t.item(), dist.log_prob(idx_t)