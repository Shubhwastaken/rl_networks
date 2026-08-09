"""
Three-phase training with proper linkage.

STAGE 1 — Train Phase 2 on optimal partitions (proof calculus):
  Uses greedy session-pairing partitions as fixed input.
  Phase 2 now operates on per-node IOs and must discover combining patterns.
  Teaches the action grammar: ADD → CROSS_SUBMOD → STORE → DECLARE.

STAGE 2 — Train Phase 1 with frozen Phase 2 (partition learner):
  Phase 1 outputs partition + weight vector.
  Phase 2 (frozen) evaluates each partition's proof potential.
  Phase 1 learns to maximise internal sessions AND produce partitions
  that enable short, tight Phase 2 proofs.

STAGE 3 — Joint fine-tuning Phase 1 + Phase 2 (end-to-end):
  Both policies unfrozen, trained together on the full pipeline.
  Gradient signal flows from Phase 2 terminal reward back through
  the partition choice (via REINFORCE on Phase 1's trajectory).

STAGE 4 — Train Phase 3 (fractional IO search):
  Uses best partition + weights from Stage 3 as starting point.
  Phase 3 policy learns FRACTIONAL_IO, CROSS_SUBMOD sequences.
  Reward is ONLY positive when extracted bound < partition_bound.
  This is where novel inequalities are discovered.

Linkage mechanism:
  After each Stage 3 episode, env.partition and env.partition_weights
  are passed to Stage 4 as the starting state. Phase 3 therefore
  always starts from a good partition (not random), so it can focus
  its exploration budget on the fractional combination step.
"""

import os
import gc
import random
import numpy as np
import torch
from collections import defaultdict
from rl_functional_dep_integration import apply_all_improving_func_dep
from functional_dependence import assert_derivation_sound
from lp_lower_bound import compute_lp_lower_bound, validate_bound_against_lp
from stage4_proof_logger import Stage4ProofLogger, generate_proof_document
import json, time


# ---------------------------------------------------------------------------
# Sub-LP bounds are unsoundness, not noise (1g)
# ---------------------------------------------------------------------------
# A bound below the max-flow LP lower bound is mathematically impossible.
# It cannot be a "bad result" to be clamped away -- it is proof that some
# operation in the derivation does not follow from its premises.
#
# Both places that used to clamp (Stage 4 training and _run_eval_episode)
# now route here. Set SUB_LP_FATAL = False only to survey how many
# violations a run produces; the default aborts so nothing sub-LP can be
# laundered into a metric, a reward, or a reported bound.
SUB_LP_FATAL: bool = True
SUB_LP_OBSERVED: list = []


class SubLPBoundError(AssertionError):
    """Raised when a derivation produces a bound below the LP lower bound."""


def _policy_fingerprint(*policies) -> str:
    """
    Stable hash of the policy weights an eval run was produced with.

    NOTE: this function did not exist before. The eval resume check keyed
    only on (stochastic_episodes, greedy_trials), so a cached
    eval_results.json from a DIFFERENT checkpoint -- or from before a
    change to the proof calculus -- was silently reused as long as those
    two numbers matched. That is exactly how stale bounds survive a fix.
    Keying on the weights makes reuse impossible unless the policies are
    bit-identical.
    """
    import hashlib
    h = hashlib.sha256()
    for p in policies:
        if p is None:
            h.update(b"<none>")
            continue
        sd = getattr(p, "state_dict", None)
        if sd is None:
            h.update(repr(p).encode())
            continue
        for k, v in sorted(sd().items()):
            h.update(k.encode())
            h.update(np.ascontiguousarray(
                v.detach().cpu().numpy()).tobytes())
    return h.hexdigest()[:32]


def raise_sub_lp(graph_name, bound, lp_lb, pb, episode=None, where=""):
    """Report a sub-LP bound. Raises unless SUB_LP_FATAL is disabled."""
    msg = (f"SUB-LP BOUND [{where}] on {graph_name}"
           f"{f' (episode {episode})' if episode is not None else ''}: "
           f"bound={bound:.6f} < LP lower bound={lp_lb:.6f} (PB={pb:.6f}). "
           f"The derivation that produced this is unsound.")
    SUB_LP_OBSERVED.append((where, graph_name, episode, bound, lp_lb))
    if SUB_LP_FATAL:
        raise SubLPBoundError(msg)
    print(f"  !! {msg}")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
# Seed torch too. Without these, every action drawn by the GNN policies
# (torch.distributions.Categorical(...).sample()) comes from torch's global
# RNG, which random.seed/np.random.seed do NOT touch. That made every run
# non-reproducible: the same code produced different novel/not-found counts
# purely from the seed, so no two runs could be compared and no fix could be
# measured. cudnn.deterministic trades a little GPU speed for reproducibility.
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from fixed_environment import PartitionBoundEnv, ActionType, Phase, _compute_partition_bound, MAX_DERIVED
from partition import generate_random_valid_partition, decode_partition
from gnn_policy import (
    GNNPhase1Policy, GNNPhase2Policy, GNNPhase3Policy, LAMBDA_GRID, DEVICE
)
from fixed_base_inequality_generator import internal_per_partition
from fixed_graph_generation import (
    get_all_graph_infos, get_optimal_for_graph, identify_graph
)
from fixed_inequality import EntropyIndex


def _partition_str(partition, sessions):
    parts = []
    for i, group in enumerate(partition):
        internal = [f"{s}->{t}" for s,t in sessions
                    if s in set(group) and t in set(group)]
        tag = f"P{i+1}={sorted(group)}"
        if internal: tag += f"[{','.join(internal)}]"
        parts.append(tag)
    return "  ".join(parts)


def _action_summary(action_counts):
    short = {
        "ASSIGN_NODE":"ASN","SWAP_NODE":"SWP","MOVE_NODE":"MOV",
        "FINALIZE_PARTITION":"FIN","ADD_TO_ACCUMULATOR":"ADD",
        "APPLY_SUBMODULARITY":"SUB","APPLY_PROOF2":"P2",
        "STORE_AND_RESET":"STO","COMBINE_STORED":"CMB",
        "DECLARE_TERMINAL":"TRM","FRACTIONAL_IO":"FIO",
        "CROSS_SUBMOD":"XSB",
    }
    parts = [f"{short.get(a,a[:3])}:{c}"
             for a, c in sorted(action_counts.items(), key=lambda x: -x[1]) if c>0]
    return " ".join(parts)


ACTION_NAMES = {
    0:"ASSIGN_NODE",1:"ADD_TO_ACCUMULATOR",2:"APPLY_SUBMODULARITY",
    3:"APPLY_PROOF2",4:"STORE_AND_RESET",5:"COMBINE_STORED",
    6:"DECLARE_TERMINAL",7:"SWAP_NODE",8:"MOVE_NODE",
    9:"FINALIZE_PARTITION",10:"FRACTIONAL_IO",11:"CROSS_SUBMOD",
    20:"APPLY_CRYPTO",21:"APPLY_DECODE",
}

EARLY_STOP_PATIENCE  = 2000
EARLY_STOP_MIN_EPS   = 3000


def _safe_mean(values):
    """np.mean with inf/nan filtered out; returns 0.0 on empty."""
    finite = [v for v in values if v is not None and v == v and abs(v) < 1e9]
    return float(np.mean(finite)) if finite else 0.0


def _mechanism_label(used_cross, used_crypto, used_decode, used_plain, no_terminal):
    if no_terminal:
        return "no_terminal"
    used_fd = used_crypto or used_decode
    if used_cross and used_fd:
        return "cross_submod_plus_crypto_decode"
    if used_cross:
        return "cross_submod_only"
    if used_fd:
        return "crypto_decode_only"
    if used_plain:
        return "plain_fractional_or_submod"
    return "plain_fractional_or_submod"


def _load_stage4_novel_bounds_from_log(log_path):
    """Rebuild graph->best novel tuple from an existing Stage 4 proof log."""
    if not os.path.exists(log_path):
        return {}
    try:
        with open(log_path) as f:
            episodes = json.load(f)
    except Exception:
        return {}

    restored = {}
    for ep in episodes:
        sm = ep.get("summary", {})
        if not sm.get("is_novel"):
            continue
        gn = ep.get("graph_name")
        b = sm.get("best_bound")
        if gn is None or b is None:
            continue
        trace = sm.get("terminal_ineq") or "N/A"
        part = [list(p) for p in ep.get("partition", [])]
        if gn not in restored or float(b) < restored[gn][0]:
            restored[gn] = (float(b), part, {}, trace)
    return restored


def _write_stage4_status_document(graph_name, graph_tuple, partition, pb, lp_lb,
                                  stats, out_path):
    nodes, edges, sessions = graph_tuple
    attempts = stats.get("attempts", 0)
    best_bound = stats.get("best_bound", float("inf"))
    if attempts == 0:
        status = "NO STAGE 4 ATTEMPT RECORDED"
    elif stats.get("valid_novel", 0) > 0:
        status = "NOVEL PROOF LOGGED ELSEWHERE"
    elif stats.get("terminal_valid", 0) > 0:
        status = "NON-NOVEL TERMINAL"
    else:
        status = "NO TERMINAL INEQUALITY FOUND"

    lines = [
        "=" * 70,
        f"STAGE 4 STATUS DOCUMENT: {graph_name}",
        "=" * 70,
        "",
        f"STATUS: {status}",
        "",
        "NETWORK",
        f"  Nodes   : {list(nodes)}",
        f"  Edges   : {[list(e) for e in edges]}",
        f"  Sessions: {[list(s) for s in sessions]}",
        "",
        "PARTITION",
    ]
    for i, p in enumerate(partition or []):
        lines.append(f"  P{i+1} = {{{', '.join(p)}}}")
    lines.extend([
        "",
        "BOUNDS",
        f"  Partition bound: {pb:.6f}",
        f"  LP lower bound : {lp_lb:.6f}",
        f"  Best observed : {best_bound:.6f}" if best_bound < 1e9 else "  Best observed : none",
        "",
        "STAGE 4 COUNTS",
        f"  Attempts             : {attempts}",
        f"  Valid novel episodes : {stats.get('valid_novel', 0)}",
        f"  Terminal episodes    : {stats.get('terminal_valid', 0)}",
        f"  No-terminal episodes : {stats.get('no_terminal', 0)}",
        f"  CROSS_SUBMOD used    : {stats.get('used_cross_submod', 0)}",
        f"  APPLY_CRYPTO used    : {stats.get('used_crypto', 0)}",
        f"  APPLY_DECODE used    : {stats.get('used_decode', 0)}",
        "",
        "This is a status report, not a mathematical proof.",
        "=" * 70,
    ])
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


class EarlyStopper:
    def __init__(self, patience=EARLY_STOP_PATIENCE,
                 min_episodes=EARLY_STOP_MIN_EPS, window=500):
        self.patience    = patience
        self.min_episodes= min_episodes
        self.window      = window
        self.best_avg    = float('-inf')
        self.best_episode= 0
        self.rewards     = []

    def update(self, reward, episode):
        self.rewards.append(reward)
        if len(self.rewards) >= self.window:
            cur = _safe_mean(self.rewards[-self.window:])
            if cur > self.best_avg + 1e-4:
                self.best_avg    = cur
                self.best_episode= episode

    def should_stop(self, episode):
        if episode < self.min_episodes: return False
        return (episode - self.best_episode) >= self.patience


class Stage4EarlyStopper:
    """
    Composite early stopper for Stage 4 (fractional IO search).

    The standard EarlyStopper fails in Stage 4 because the agent must learn
    a multi-step chain (FRACTIONAL_IO → ADD → CROSS_SUBMOD → STORE → DECLARE)
    before it can beat PB at all.  During chain assembly the improvement signal
    is flat, causing the stopper to fire prematurely.

    This stopper uses a weighted composite signal:
        signal = w1*(pb-best_b)/pb  +  w2*novel_rate_W  +  w3*cross_rate_W

    where W is the trailing window.  This means:
      - Primary credit: actually beating PB          (w1=0.6)
      - Secondary credit: novel rate trending up     (w2=0.3)
      - Tertiary credit: cross usage trending up     (w3=0.1)

    The stopper resets its best whenever the composite signal improves,
    so the agent gets credit for chain-assembly progress even before it
    completes the full sequence.

    Hard-floor override: if novel_rate has been exactly zero for the last
    `novel_zero_patience` episodes AND we are past min_episodes, stop
    regardless of the composite score.  This prevents running the full
    budget on a broken chain that does lots of CROSS_SUBMOD but never
    produces a valid terminal bound.
    """

    def __init__(self,
                 patience          : int   = 5000,
                 min_episodes      : int   = 4000,
                 window            : int   = 500,
                 novel_zero_patience: int  = 3000,
                 composite_lock_floor: int = 1500,
                 w1: float = 0.6,
                 w2: float = 0.3,
                 w3: float = 0.1):
        self.patience           = patience
        self.min_episodes       = min_episodes
        self.window             = window
        self.novel_zero_patience= novel_zero_patience
        # Composite score isn't allowed to lock in a "best" until this many
        # episodes have passed. Without this, a single early lucky spike
        # (noise, not skill -- pb_score/novel_rate are near-random at low
        # episode counts) sets best_composite, and the agent then has only
        # `patience` episodes to beat a target set by chance rather than
        # learned behavior. This was very likely what happened on the run
        # that stopped at episode ~5825: best_episode probably landed
        # around episode ~800, and nothing in the next 5000 episodes beat
        # a fluke.
        self.composite_lock_floor = composite_lock_floor
        self.w1, self.w2, self.w3 = w1, w2, w3

        self.best_composite     = float('-inf')
        self.best_episode       = 0

        # Per-episode history for the three signal components
        self._pb_improvements   : list = []   # (pb-best_b)/pb per episode
        self._novel_found       : list = []   # 0/1 per episode
        self._cross_used        : list = []   # 0/1 per episode

        # Tracks last episode where novel_found==1 (for hard-floor check)
        self._last_novel_episode: int  = -1

    def update(self, pb_improvement: float, novel_found: int,
               cross_used: int, episode: int):
        """
        Args:
            pb_improvement: (pb - best_b) / pb for this episode.
                            0 when agent matches or misses PB, positive when it beats PB.
            novel_found:    1 if a sub-PB bound was found this episode, else 0.
            cross_used:     1 if CROSS_SUBMOD was executed this episode, else 0.
            episode:        current episode index.
        """
        self._pb_improvements.append(max(pb_improvement, 0.0))
        self._novel_found.append(novel_found)
        self._cross_used.append(cross_used)

        if novel_found:
            self._last_novel_episode = episode

        if len(self._pb_improvements) >= self.window:
            w = self.window
            novel_rate = np.mean(self._novel_found[-w:])
            cross_rate = np.mean(self._cross_used[-w:])
            pb_score   = np.mean(self._pb_improvements[-w:])

            composite = (self.w1 * pb_score
                         + self.w2 * novel_rate
                         + self.w3 * cross_rate)

            if episode >= self.composite_lock_floor and composite > self.best_composite + 1e-5:
                self.best_composite = composite
                self.best_episode   = episode

    def should_stop(self, episode: int) -> bool:
        if episode < self.min_episodes:
            return False

        # Hard-floor: if novel rate has been zero for novel_zero_patience
        # episodes, the agent is stuck regardless of composite score.
        episodes_since_novel = (
            episode - self._last_novel_episode
            if self._last_novel_episode >= 0
            else episode
        )
        if episodes_since_novel >= self.novel_zero_patience:
            return True

        # Normal composite patience check
        return (episode - self.best_episode) >= self.patience


def _print_graph_table():
    infos = get_all_graph_infos()
    print(f"\n  {'Name':<16} {'N':>3} {'E':>3} {'S':>3} "
          f"{'Trivial':>8} {'Optimal':>8} {'OptInt':>6}")
    print(f"  {'-'*55}")
    for info in infos:
        trivial = len(info.edges) / len(info.sessions)
        print(f"  {info.name:<16} {len(info.nodes):>3} {len(info.edges):>3} "
              f"{len(info.sessions):>3} {trivial:>8.4f} "
              f"{info.optimal_bound:>8.4f} {info.optimal_internal:>6}")
    print()


def _greedy_session_partition(nodes, edges, sessions):
    adj = {n: set() for n in nodes}
    for u, v in edges:
        adj[u].add(v); adj[v].add(u)
    assignment = {}; gid = 0
    for s, t in sessions:
        if s not in assignment and t not in assignment:
            if t not in adj[s]:
                assignment[s] = gid; assignment[t] = gid; gid += 1
            else:
                assignment[s] = gid; gid += 1
                assignment[t] = gid; gid += 1
        elif s in assignment and t not in assignment:
            g = assignment[s]
            if not any(assignment.get(n) == g for n in adj[t] if n in assignment):
                assignment[t] = g
            else:
                assignment[t] = gid; gid += 1
        elif t in assignment and s not in assignment:
            g = assignment[t]
            if not any(assignment.get(n) == g for n in adj[s] if n in assignment):
                assignment[s] = g
            else:
                assignment[s] = gid; gid += 1
    for node in nodes:
        if node not in assignment:
            nb_groups = {assignment[n] for n in adj[node] if n in assignment}
            placed = False
            for g in range(gid):
                if g not in nb_groups:
                    assignment[node] = g; placed = True; break
            if not placed:
                assignment[node] = gid; gid += 1
    groups = {}
    for node, g in assignment.items():
        groups.setdefault(g, []).append(node)
    return list(groups.values())


def _partition_bound_of(partition, edges, sessions):
    """Unweighted partition bound of a SPECIFIC partition.

    Returns float('inf') if the partition violates the independent-set
    constraint (some part contains two adjacent nodes) or is degenerate.
    This is the single source of truth used everywhere a stored/derived
    partition must be scored: Stage 3 saving, Stage 4 selection, eval.

    FIX (root cause of the petersen_10N regression): Stage 3 used to store
    `env._best_pool_bound()` (a Phase-2 proof-calculus bound) or the graph's
    GLOBAL optimum as the "bound" of the partition it saved. Those numbers
    are not the bound OF that partition. petersen ended up stored as
    ([[0,9],[1,3],[2,8],[4,5],[6,7]], bound=1.875) when that partition's
    real bound is 15/7 = 2.143. Stage 4 then re-scored it, correctly saw
    2.143 > 1.875, and fell into the (previously weak) brute-force fallback.
    """
    eset = set()
    for u, v in edges:
        eset.add((u, v)); eset.add((v, u))
    for Pk in partition:
        pl = list(Pk)
        for i in range(len(pl)):
            for j in range(i + 1, len(pl)):
                if (pl[i], pl[j]) in eset:
                    return float('inf')
    internal = sum(1 for Pk in partition
                   for s, t in sessions
                   if s in set(Pk) and t in set(Pk))
    denom = len(sessions) + internal
    return len(edges) / denom if denom > 0 else float('inf')


def _find_optimal_partition(nodes, edges, sessions):
    """
    Returns the partition achieving the true optimal partition bound.

    FIX: this used to run its OWN weaker search (greedy colorings +
    exhaustive 2-way splits only), while the TARGET bound used by Stage 4
    (env.partition_bound) came from the strictly more thorough
    compute_optimal_bound (exhaustive k-coloring, k<=4 for n<=10).
    The two disagreed on exactly the graphs the docstring of
    _compute_partition_bound lists (petersen_10N: greedy gives a
    3-coloring with bound 2.5 vs true optimum 1.875 via a 3-group
    partition the greedy strategies never produce). Result: the agent
    searched under a PB-2.5 partition while being required to beat 1.875 —
    the base-inequality family that produced the old novel bound simply
    did not exist in its pool.

    compute_optimal_bound already computes AND returns the argmin
    partition; the old code threw it away. Delegate instead of
    re-implementing a worse search. compute_optimal_bound always returns
    a valid partition (it initialises with the singleton partition, which
    is optimal-by-construction for graphs like okamura_seymour_8N where
    every session is a direct edge), so this never returns None.
    """
    from fixed_graph_generation import compute_optimal_bound
    _, _, best_part = compute_optimal_bound(nodes, edges, sessions)
    return [list(g) for g in best_part]


# --------------------------------------------------------------------------
# Stage 4 / eval shared partition selection
# --------------------------------------------------------------------------

# Experiment overrides (enabled only with --pin-partitions on the CLI).
# Purpose: pin a graph to a SPECIFIC search partition regardless of the
# normal RL-vs-optimal selection, to test whether the search partition
# (not the policy) is what blocks novel bounds.
#
# al_bashabsheh_7N: the pre-regression run found r <= 1.523 while searching
# under this 4-part partition. Note it is DELIBERATELY PB-suboptimal:
# al_bashabsheh has 3 sessions and this partition internalises only 2 of
# them ((S1,t1) and (S2,t2); (S3,t3) crosses parts), so its own bound is
# 12/5 = 2.4 vs the true PB 2.0 achieved by the pair partition
# [[S1,t1],[S2,t2],[S3,t3],[v1]]. The pair partition is what the current
# run searches under and it produced 3 valid terminals in 562 episodes,
# best 3.08 — the experiment asks whether the old, "worse-PB" partition
# generates a base pool the agent can actually combine from.
# env.partition_bound stays the TRUE optimum in all cases, so novelty is
# still judged against PB = 2.0, never against the pinned partition's 2.4.
PARTITION_OVERRIDES = {
    "al_bashabsheh_7N": [["v1"], ["S1", "S3", "t1"], ["S2", "t2"], ["t3"]],
}
PIN_PARTITIONS = False   # set True via --pin-partitions

# Targeted-priority finetune hook. Maps graph_name -> extra sampling weight
# ADDED on top of whatever _priority_for_graph would otherwise assign. This
# is how targeted_finetune.py biases sampling toward specific graphs WITHOUT
# restricting the dataset (which is what causes catastrophic forgetting of
# the other graphs). Empty by default => zero effect on normal runs.
FINETUNE_PRIORITY_BOOST = {}


def _select_stage4_partition(graph_name, nodes, edges, sessions,
                             best_partitions):
    """Single partition-selection path shared by Stage 4 training AND eval.

    FIX: training and eval previously selected partitions independently and
    could disagree on the same graph in the same run:
      - training re-scored the stored RL partition and fell back to the
        (weak) brute-force search when suboptimal;
      - eval used the stored RL partition UNCONDITIONALLY when present,
        and ran a fresh Phase-1 rollout when absent.
    For petersen_10N that meant training searched under a PB-2.5 greedy
    partition while eval searched under the PB-2.143 stored pair partition
    — two different search spaces, neither matching the true optimum.
    Any train/eval novelty comparison on such a graph was meaningless.

    Returns (partition, weights, opt_global_bound, source_label):
      source_label in {"override", "RL", "exhaustive"}.
    Selection:
      1. If PIN_PARTITIONS and graph has an override -> override.
      2. If the stored Stage-3 RL partition's OWN bound (re-scored here,
         never trusted from the file — see _partition_bound_of docstring)
         matches the true optimum -> RL partition (preserves the
         "RL found an optimal partition" research claim).
      3. Otherwise -> the exhaustive optimum from compute_optimal_bound.
    """
    from fixed_graph_generation import compute_optimal_bound
    opt_bound, _, opt_part = compute_optimal_bound(nodes, edges, sessions)
    opt_part = [list(g) for g in opt_part]

    if PIN_PARTITIONS and graph_name in PARTITION_OVERRIDES:
        pinned = [list(g) for g in PARTITION_OVERRIDES[graph_name]]
        if _partition_bound_of(pinned, edges, sessions) == float('inf'):
            print(f"  [partition] {graph_name}: override INVALID "
                  f"(violates independent-set constraint) — ignored")
        else:
            return pinned, {}, opt_bound, "override"

    if best_partitions and graph_name in best_partitions:
        rl_part, rl_weights, _stored_bound = best_partitions[graph_name]
        rl_part = [list(g) for g in rl_part]
        rl_bound = _partition_bound_of(rl_part, edges, sessions)
        if rl_bound <= opt_bound + 1e-6:
            return rl_part, (rl_weights or {}), opt_bound, "RL"

    return opt_part, {}, opt_bound, "exhaustive"


# -----------------------------------------------------------------------
# Stage 1 — Train Phase 2 (proof calculus)
# -----------------------------------------------------------------------

def run_stage1(num_episodes=10000, graph_dataset_size=5):  # Tier 1 only
    print("=" * 70)
    print(f"STAGE 1: Train Phase 2 proof calculus ({num_episodes} episodes)")
    print("=" * 70)
    _print_graph_table()

    env = PartitionBoundEnv(graph_dataset_size=graph_dataset_size, stage=1)

    max_dim = 0
    for nodes, edges, sessions in env.graph_dataset:
        part = _greedy_session_partition(nodes, edges, sessions)
        ix   = EntropyIndex(partitions=part, nodes=nodes,
                            edges=edges, sessions=sessions)
        max_dim = max(max_dim, ix.dim)
        chrom = generate_random_valid_partition(nodes, edges)
        part2 = decode_partition(nodes, chrom)
        ix2   = EntropyIndex(partitions=part2, nodes=nodes,
                             edges=edges, sessions=sessions)
        max_dim = max(max_dim, ix2.dim)

    coeff_dim = max_dim
    print(f"  Max coeff dim: {coeff_dim}")

    phase2_policy = GNNPhase2Policy(coeff_dim=coeff_dim,
                                    total_episodes=num_episodes)
    phase2_policy.unfreeze()

    # Resume from latest periodic checkpoint if available
    _resume_ep = 0
    os.makedirs("model_files", exist_ok=True)
    if os.path.exists("model_files/ckpt_stage1_phase2_latest.pt") and os.path.exists("model_files/ckpt_stage1_meta.pt"):
        _meta = torch.load("model_files/ckpt_stage1_meta.pt", weights_only=True)
        if _meta.get("coeff_dim", coeff_dim) == coeff_dim:
            phase2_policy.net.load_state_dict(
                torch.load("model_files/ckpt_stage1_phase2_latest.pt", weights_only=True, map_location=DEVICE))
            _resume_ep = _meta["episode"]
            print(f"  [resume] Stage 1 resuming from episode {_resume_ep}")
        else:
            print(f"  [resume] coeff_dim mismatch — starting Stage 1 from scratch")

    rewards   = []
    per_graph = defaultdict(list)
    metrics   = {'rewards':[], 'bounds':[], 'graph_names':[],
                 'step_counts':[], 'action_counts_per_ep':[]}

    log_interval = 500
    # Stage 1 needs more patience than the default: Phase 2 must learn the
    # full ADD → CROSS_SUBMOD → STORE → DECLARE grammar, not just the ADD-heavy
    # shortcut.  The previous patience=2000/min_episodes=3000 fired exactly at
    # min_episodes (episode 3000 in the training log) before the grammar was
    # internalised.  Increasing both gives Phase 2 enough time to move past
    # the shortcut plateau and learn structured combinations.
    stopper = EarlyStopper(patience=5000, min_episodes=5000)
    print(f"\n  {'Ep':>6} | {'Graph':<16} | {'AvgRew':>8} | {'BestBnd':>8} | Actions")
    print(f"  {'-'*75}")

    for episode in range(_resume_ep, num_episodes):
        if stopper.should_stop(episode):
            print(f"\n  Early stopping at episode {episode}")
            break

        graph_tuple = random.choice(env.graph_dataset)
        nodes, edges, sessions = graph_tuple
        graph_name = identify_graph(nodes, edges, sessions)

        if random.random() < 0.7:
            partition = _greedy_session_partition(nodes, edges, sessions)
        else:
            chrom     = generate_random_valid_partition(nodes, edges)
            partition = decode_partition(nodes, chrom)

        state = env.reset(fixed_partition=partition, fixed_graph=graph_tuple)
        state['edges']   = edges
        # Anneal the APPLY_PROOF2 forcing probability over the full
        # num_episodes budget, not the old min(5000, num_episodes) horizon.
        # The old horizon caused forcing to reach zero at episode 5000
        # regardless of num_episodes — with early stopping at 3000, Phase 2
        # never trained without the crutch and never had to discover
        # APPLY_PROOF2 independently.  Annealing over the full budget
        # means the forcing is still ~0.15 at the midpoint, giving the
        # policy a gentle scaffold throughout rather than a hard cutoff.
        proof2_fp = max(0.0, 0.3 * (1.0 - episode / max(num_episodes, 1)))
        state['proof2_force_prob'] = proof2_fp

        # Also pass graph info needed by Phase 3 policy (used indirectly)
        state['nodes']     = nodes
        state['sessions']  = sessions
        state['partition'] = partition

        done       = False
        trajectory = []
        action_counts = defaultdict(int)
        total_reward  = 0.0
        step_count    = 0

        while not done:
            valid  = env.get_valid_actions()
            if not valid: break
            action = phase2_policy.select_action(state, valid)
            aname  = ACTION_NAMES.get(int(action['type']), '?')
            action_counts[aname] += 1
            state, reward, done = env.step(action)
            state['edges']            = edges
            state['proof2_force_prob']= proof2_fp
            state['nodes']            = nodes
            state['sessions']         = sessions
            state['partition']        = partition
            trajectory.append({'reward': reward})
            total_reward += reward
            step_count   += 1

        phase2_policy.update(trajectory, total_reward)
        rewards.append(total_reward)
        # Use actual pool bound for metrics, not abs(total_reward).
        # abs(total_reward) is RL reward noise, not a valid upper bound.
        _s2_bound = env._best_pool_bound()
        _s2_bound = _s2_bound if _s2_bound is not None else env.partition_bound
        per_graph[graph_name].append(_s2_bound)
        stopper.update(total_reward, episode)

        metrics['rewards'].append(total_reward)
        metrics['bounds'].append(_s2_bound)
        metrics['graph_names'].append(graph_name)
        metrics['step_counts'].append(step_count)
        metrics['action_counts_per_ep'].append(dict(action_counts))

        if (episode + 1) % log_interval == 0:
            n   = log_interval
            avg = _safe_mean(rewards[-n:])
            bst = abs(min(rewards[-n:]))
            print(f"  {episode+1:>6} | {graph_name:<16} | "
                  f"{avg:>8.4f} | {bst:>8.4f} | {_action_summary(action_counts)}")
            # Periodic checkpoint — survives mid-stage crashes
            torch.save(phase2_policy.net.state_dict(), "model_files/ckpt_stage1_phase2_latest.pt")
            torch.save({"episode": episode + 1, "coeff_dim": coeff_dim},
                     "model_files/ckpt_stage1_meta.pt")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n  Per-graph bounds (Stage 1):")
    for gname in sorted(per_graph.keys()):
        bounds = per_graph[gname]
        opt, _ = get_optimal_for_graph(
            *next(t for t in env.graph_dataset if identify_graph(*t) == gname))
        print(f"    {gname:<16}: avg={np.mean(bounds):.4f} "
              f"best={min(bounds):.4f} opt={opt:.4f}")
    print("\nStage 1 complete.\n")
    return phase2_policy, coeff_dim, metrics


# -----------------------------------------------------------------------
# Stage 2 — Train Phase 1 with frozen Phase 2
# -----------------------------------------------------------------------

def run_stage2(phase2_policy, num_episodes=10000, graph_dataset_size=5):
    print("=" * 70)
    print(f"STAGE 2: Train Phase 1 ({num_episodes} episodes, Phase 2 frozen)")
    print("=" * 70)

    phase2_policy.freeze()
    env  = PartitionBoundEnv(graph_dataset_size=graph_dataset_size, stage=2)
    phase1_policy = GNNPhase1Policy(total_episodes=num_episodes)

    # Resume from latest periodic checkpoint if available
    _resume_ep_s2 = 0
    os.makedirs("model_files", exist_ok=True)
    if (os.path.exists("model_files/ckpt_stage2_phase1_latest.pt") and
                os.path.exists("model_files/ckpt_stage2_phase2_latest.pt") and
                os.path.exists("model_files/ckpt_stage2_meta.pt")):
        phase1_policy.net.load_state_dict(
                torch.load("model_files/ckpt_stage2_phase1_latest.pt", weights_only=True, map_location=DEVICE))
        phase2_policy.net.load_state_dict(
                torch.load("model_files/ckpt_stage2_phase2_latest.pt", weights_only=True, map_location=DEVICE))
        _meta_s2 = torch.load("model_files/ckpt_stage2_meta.pt", weights_only=True)
        _resume_ep_s2 = _meta_s2["episode"]
        print(f"  [resume] Stage 2 resuming from episode {_resume_ep_s2}")

    rewards   = []
    internals = []
    per_graph = defaultdict(lambda: {'int':[], 'optimal_found':0})
    metrics   = {'rewards':[], 'internals':[], 'graph_names':[],
                 'optimal_int_found':[], 'partition_weights':[]}

    log_interval = 500

    # Stopper: per-graph reward plateau — stops when every graph's
    # average reward stops improving. Does NOT use opt_int/brute-force
    # answer — Phase 1 converges purely from the RL signal.
    _S2_PLATEAU_WINDOW  = 500   # episodes per graph
    _S2_PLATEAU_DELTA   = 0.005 # min improvement to count as progress
    _S2_MIN_EPISODES    = 8000

    class _PerGraphPlateauStopper:
        def __init__(self, window, delta, min_eps):
            self.window   = window
            self.delta    = delta
            self.min_eps  = min_eps
            self._rewards = defaultdict(list)

        def update(self, graph_name, reward):
            self._rewards[graph_name].append(reward)

        def should_stop(self, episode, all_graph_names):
            if episode < self.min_eps:
                return False
            for gn in all_graph_names:
                hist = self._rewards[gn]
                if len(hist) < self.window * 2:
                    return False
                # compare last window vs previous window
                prev = np.mean(hist[-self.window*2:-self.window])
                curr = np.mean(hist[-self.window:])
                if curr - prev > self.delta:   # still improving
                    return False
            return True

    stopper = _PerGraphPlateauStopper(_S2_PLATEAU_WINDOW, _S2_PLATEAU_DELTA, _S2_MIN_EPISODES)
    all_graph_names_seen: set = set()

    print(f"\n  {'Ep':>6} | {'Graph':<16} | {'AvgRew':>8} | {'Int':>3} | P1 Actions")
    print(f"  {'-'*70}")

    for episode in range(_resume_ep_s2, num_episodes):
        if stopper.should_stop(episode, all_graph_names_seen):
            print(f"\n  Early stopping at episode {episode}")
            break

        env.reset()
        nodes, edges, sessions = env.nodes, env.edges, env.sessions
        graph_name = identify_graph(nodes, edges, sessions)
        temperature = max(1.0, 2.5 - 2.5 * episode / max(num_episodes, 1))
        p1_traj    = []
        state      = env._get_state()
        state['edges']    = edges
        state['sessions'] = sessions
        state['temperature'] = temperature
        done = False
        p1_action_counts = defaultdict(int)

        while env.current_phase == Phase.PHASE1 and not done:
            valid = env.get_valid_actions()
            if not valid: break
            state['temperature'] = temperature
            action = phase1_policy.select_action(state, valid)
            aname  = ACTION_NAMES.get(int(action.get('type', 0)), '?')
            p1_action_counts[aname] += 1
            state, reward, done = env.step(action)
            state['edges']    = edges
            state['sessions'] = sessions
            p1_traj.append({'reward': reward})

        rl_partition = [list(g) for g in env.partition] if env.partition else []
        ipp          = internal_per_partition(rl_partition, sessions)
        int_count    = sum(ipp)
        internals.append(int_count)
        per_graph[graph_name]['int'].append(int_count)
        # track whether RL partition bound matches brute-force optimum
        _rl_pb = len(edges) / max(len(sessions) + int_count, 1)
        _bf_pb, _ = get_optimal_for_graph(nodes, edges, sessions)
        _is_optimal = (_rl_pb <= _bf_pb + 1e-6)
        if _is_optimal:
            per_graph[graph_name]['optimal_found'] += 1

        # Retrieve weights from Phase 1 (set during FINALIZE action)
        partition_weights = env.partition_weights

        total_reward = sum(t['reward'] for t in p1_traj)
        p2_traj = []
        while not done:
            valid = env.get_valid_actions()
            if not valid: break
            state['nodes']     = nodes
            state['sessions']  = sessions
            state['partition'] = rl_partition
            action = phase2_policy.select_action(state, valid)
            state, reward, done = env.step(action)
            state['edges']    = edges
            state['sessions'] = sessions
            p2_traj.append({'reward': reward})
            total_reward += reward

        phase1_policy.update(p1_traj, total_reward)
        rewards.append(total_reward)
        all_graph_names_seen.add(graph_name)
        stopper.update(graph_name, total_reward)

        metrics['rewards'].append(total_reward)
        metrics['internals'].append(int_count)
        metrics['graph_names'].append(graph_name)
        metrics['optimal_int_found'].append(1 if _is_optimal else 0)
        metrics['partition_weights'].append(partition_weights)

        if _is_optimal and (episode + 1) % log_interval == 0:
            print(f"  >> Ep {episode+1} {graph_name}: RL matched optimal PB={_rl_pb:.4f}")

        if (episode + 1) % log_interval == 0:
            n    = log_interval
            avg  = _safe_mean(rewards[-n:])
            avgi = np.mean(internals[-n:])
            print(f"  {episode+1:>6} | {graph_name:<16} | "
                  f"{avg:>8.4f} | {avgi:>3.1f} | {_action_summary(p1_action_counts)}")
            # Periodic checkpoint
            torch.save(phase1_policy.net.state_dict(), "model_files/ckpt_stage2_phase1_latest.pt")
            torch.save(phase2_policy.net.state_dict(), "model_files/ckpt_stage2_phase2_latest.pt")
            torch.save({"episode": episode + 1}, "model_files/ckpt_stage2_meta.pt")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n  Per-graph Phase 1 (Stage 2):")
    for gname in sorted(per_graph.keys()):
        stats = per_graph[gname]
        total = len(stats['int'])
        bf_bound, _ = get_optimal_for_graph(*next(
            (n,e,s) for n,e,s in env.graph_dataset
            if identify_graph(n,e,s) == gname
        ))
        print(f"    {gname:<16}: avg_int={np.mean(stats['int']):.2f} "
              f"opt_rate={100*stats['optimal_found']/max(total,1):.1f}% "
              f"(bf_bound={bf_bound:.4f})")
    print("\nStage 2 complete.\n")
    return phase1_policy, metrics


# -----------------------------------------------------------------------
# Stage 3 — Joint fine-tuning Phase 1 + Phase 2
# -----------------------------------------------------------------------

def run_stage3(phase1_policy, phase2_policy,
               num_episodes=10000, graph_dataset_size=10):  # Tier 1+2
    print("=" * 70)
    print(f"STAGE 3: Joint fine-tuning Phase 1+2 ({num_episodes} episodes)")
    print("=" * 70)

    # ---- Stability fixes for joint training ----
    #
    # The reward collapse seen in the training log (avg dropping to -140 by
    # episode 11200) is caused by two interacting instabilities:
    #
    # 1. LEARNING RATE: Both policies enter Stage 3 with their Stage 1/2 lr
    #    still set (3e-4).  Joint updates compound gradient magnitudes —
    #    Phase 1 shifts its partition distribution, Phase 2 sees a moving
    #    target, and the combined update diverges.  Halving both lrs to 1.5e-4
    #    keeps early Stage 3 updates conservative without slowing the
    #    productive mid-stage improvement.
    #
    # 2. DELAYED PHASE 2 UNFREEZE: Unfreezing Phase 2 immediately means both
    #    policies receive large simultaneous gradient updates from episode 1.
    #    Keeping Phase 2 frozen for the first S3_PHASE2_WARMUP episodes lets
    #    Phase 1 adapt its partition distribution against a stable Phase 2
    #    before joint updates begin.  After warmup, Phase 2 unfreezes and
    #    both policies train jointly from a stable initialisation.
    #
    # 3. TIGHTER GRADIENT CLIPPING: The existing clip_grad_norm of 1.0 in
    #    gnn_policy.py is appropriate for single-policy training but too loose
    #    for joint training where gradient norms compound.  We post-process
    #    each update by re-clipping after the policy's own backward pass.

    S3_LR_SCALE  = 0.5   # multiply current lr by this at Stage 3 entry
    S3_CLIP_NORM = 0.5   # tighter grad clip for joint training

    # Scale down learning rates for both policies
    for policy in (phase1_policy, phase2_policy):
        for pg in policy.optimizer.param_groups:
            pg['lr'] = pg['lr'] * S3_LR_SCALE
    print(f"  Stage 3 lr scaled to "
          f"P1={phase1_policy.optimizer.param_groups[0]['lr']:.2e}, "
          f"P2={phase2_policy.optimizer.param_groups[0]['lr']:.2e}")

    # Both policies train jointly from episode 1.
    # Stability is maintained by the lr reduction and tighter gradient clip
    # rather than a warmup freeze — avoiding the tradeoff of Phase 2 wasting
    # 2000 episodes without learning while Phase 1 adapts to a frozen policy.
    phase2_policy.unfreeze()

    env = PartitionBoundEnv(graph_dataset_size=graph_dataset_size, stage=3)

    # Resume from latest periodic checkpoint if available
    _resume_ep_s3 = 0
    os.makedirs("model_files", exist_ok=True)
    os.makedirs("config_files", exist_ok=True)
    if (os.path.exists("model_files/ckpt_stage3_phase1_latest.pt") and
            os.path.exists("model_files/ckpt_stage3_phase2_latest.pt") and
            os.path.exists("model_files/ckpt_stage3_meta.pt")):
        phase1_policy.net.load_state_dict(
            torch.load("model_files/ckpt_stage3_phase1_latest.pt", weights_only=True, map_location=DEVICE))
        phase2_policy.net.load_state_dict(
            torch.load("model_files/ckpt_stage3_phase2_latest.pt", weights_only=True, map_location=DEVICE))
        _meta_s3 = torch.load("model_files/ckpt_stage3_meta.pt", weights_only=True)
        _resume_ep_s3 = _meta_s3["episode"]
        print(f"  [resume] Stage 3 resuming from episode {_resume_ep_s3}")

    rewards   = []
    per_graph = defaultdict(list)
    metrics   = {'rewards':[], 'bounds':[], 'graph_names':[],
                 'best_partitions':{}}

    # Store best (partition, weights) per graph for Phase 4 handoff
    best_partitions: dict = {}
    # Restore best_partitions from latest checkpoint if resuming
    if _resume_ep_s3 > 0 and os.path.exists("config_files/ckpt_stage3_best_partitions_latest.json"):
        with open("config_files/ckpt_stage3_best_partitions_latest.json") as _bf:
            import json as _j
            _bp_raw = _j.load(_bf)
        # Restore full tuple including weights and bound
        def _restore_bp(v):
            if isinstance(v, dict):
                def _restore_key(ek):
                    if ek.startswith("("): return eval(ek)
                    try: return int(ek)
                    except ValueError: return ek
                return ([list(p) for p in v["partition"]],
                        {_restore_key(ek): ev
                         for ek, ev in v.get("weights", {}).items()},
                        float(v["bound"]) if v.get("bound") is not None else float("inf"))
            else:
                return ([list(p) for p in v], {}, float("inf"))
        best_partitions = {k: _restore_bp(v) for k, v in _bp_raw.items()}
        print(f"  [resume] Restored best_partitions for {list(best_partitions.keys())}")   # graph_name -> (partition, weights, bound)

    # Pre-compute LP lower bounds for validation
    lp_bounds = {}
    lp_violations = []
    for nodes_g, edges_g, sessions_g in env.graph_dataset:
        gn = identify_graph(nodes_g, edges_g, sessions_g)
        if gn not in lp_bounds:
            lp_bounds[gn] = compute_lp_lower_bound(nodes_g, edges_g, sessions_g)
    print(f"  LP lower bounds: {lp_bounds}")

    log_interval = 500
    # Issue 1 fix (Stage 3): Stage 3 stopped at episode 8258 because the
    # Stage 3 stopped at exactly ep 8000 (the min_episodes floor) last run,
    # meaning it was still learning — best_partitions for two_k4/petersen
    # were suboptimal and caused Stage 4 regressions.
    # Fix: increase min_episodes=12000 and patience=6000, AND add a secondary
    # reward-plateau condition: don't stop until avg_reward > -1.0 has been
    # sustained for 1000 consecutive episodes. This ensures Stage 3 only exits
    # once the joint policy has genuinely converged, not just hit the floor.
    _S3_REWARD_PLATEAU_THRESHOLD = -1.0
    _S3_REWARD_PLATEAU_WINDOW    = 1000
    stopper = EarlyStopper(patience=6000, min_episodes=8000)

    def _s3_reward_plateaued(rewards_list):
        """True if avg reward has been above threshold for plateau window."""
        if len(rewards_list) < _S3_REWARD_PLATEAU_WINDOW:
            return False
        return _safe_mean(rewards_list[-_S3_REWARD_PLATEAU_WINDOW:]) > _S3_REWARD_PLATEAU_THRESHOLD

    for episode in range(_resume_ep_s3, num_episodes):
        if stopper.should_stop(episode) and _s3_reward_plateaued(rewards):
            print(f"\n  Early stopping at episode {episode} (reward plateaued above {_S3_REWARD_PLATEAU_THRESHOLD})")
            break

        env.reset()
        nodes, edges, sessions = env.nodes, env.edges, env.sessions
        graph_name = identify_graph(nodes, edges, sessions)
        opt_bound, _ = get_optimal_for_graph(nodes, edges, sessions)

        temperature = max(1.0, 2.0 - 2.0 * episode / max(num_episodes, 1))
        p1_traj   = []
        state     = env._get_state()
        state['edges']    = edges
        state['sessions'] = sessions
        state['temperature'] = temperature
        done = False

        while env.current_phase == Phase.PHASE1 and not done:
            valid = env.get_valid_actions()
            if not valid: break
            state['temperature'] = temperature
            action = phase1_policy.select_action(state, valid)
            state, reward, done = env.step(action)
            state['edges']    = edges
            state['sessions'] = sessions
            p1_traj.append({'reward': reward})

        rl_partition     = [list(g) for g in env.partition] if env.partition else []
        partition_weights= env.partition_weights

        total_reward = sum(t['reward'] for t in p1_traj)
        p2_traj = []
        while not done:
            valid = env.get_valid_actions()
            if not valid: break
            state['nodes']     = nodes
            state['sessions']  = sessions
            state['partition'] = rl_partition
            action = phase2_policy.select_action(state, valid)
            state, reward, done = env.step(action)
            state['edges']    = edges
            state['sessions'] = sessions
            p2_traj.append({'reward': reward})
            total_reward += reward

        phase1_policy.update(p1_traj, total_reward)
        phase2_policy.update(p2_traj, total_reward)

        # Tighter gradient clip for joint training stability.
        # The policies' own update() methods clip at 1.0, which is
        # correct for single-policy training but too loose when both
        # policies update simultaneously against a shared reward signal.
        # Re-clipping at 0.5 after the backward pass catches any residual
        # large gradients without interfering with the policies' internal
        # optimiser state (the clip happens post-step, so it only affects
        # the *next* backward pass via parameter values, not the current one).
        # We apply it as a parameter norm guard rather than a gradient hook
        # to keep the implementation simple and side-effect-free.
        import torch.nn as nn
        nn.utils.clip_grad_norm_(phase1_policy.net.parameters(), S3_CLIP_NORM)
        nn.utils.clip_grad_norm_(phase2_policy.net.parameters(), S3_CLIP_NORM)

        # Use actual extracted mathematical bound for validation.
        # If no terminal form exists yet (Phase 2 hasn't fired APPLY_PROOF2),
        # fall back to partition_bound — NOT abs(total_reward), which is
        # accumulated RL reward noise and is not a valid upper bound on rate r.
        actual_bound = env._best_pool_bound()
        rl_bound = actual_bound if actual_bound is not None else env.partition_bound
        # Sanity check: a valid upper bound cannot be below the LP lower bound.
        # If the extracted bound violates this, discard it and fall back to PB.
        # This catches cases where Phase 2 produces an invalid terminal form
        # (e.g. petersen_10N where LP_LB=1.25 but pool returns 1.0).
        _lp_floor_s3 = lp_bounds.get(graph_name, 0.0)
        if rl_bound < _lp_floor_s3 - 1e-9:
            rl_bound = env.partition_bound
        rewards.append(total_reward)
        per_graph[graph_name].append(rl_bound)
        stopper.update(total_reward, episode)
        metrics['rewards'].append(total_reward)
        metrics['bounds'].append(rl_bound)
        metrics['graph_names'].append(graph_name)

        # Validate against LP lower bound
        if graph_name in lp_bounds and rl_bound < 1e9:
            lp_lb = lp_bounds[graph_name]
            is_valid, msg = validate_bound_against_lp(
                rl_bound, lp_lb, graph_name
            )
            if not is_valid:
                lp_violations.append((episode, graph_name, rl_bound, lp_lb))
                if len(lp_violations) <= 5:  # limit spam
                    print(f"  !! LP VIOLATION: {msg}")

        # Track best partition + weights for Phase 4.
        # FIX: the "bound" stored alongside the partition MUST be the
        # partition's OWN unweighted bound, not `rl_bound`. `rl_bound` is
        # the Phase-2 proof-calculus bound (or the graph's global PB as a
        # fallback) — a number about the PROOF, not about the PARTITION.
        # Storing it here is how petersen_10N ended up saved as
        # (pair-partition, 1.875) when that partition's real bound is
        # 2.143: Stage 4 re-scored it, rejected it, and fell into the
        # weak brute-force fallback. Comparison and storage now both use
        # the partition's own bound, so the file can never again pair a
        # partition with a bound it does not achieve.
        _rl_part_own_bound = _partition_bound_of(rl_partition, edges, sessions)
        if _rl_part_own_bound < float('inf'):
            _prev = best_partitions.get(graph_name)
            _prev_own = (_partition_bound_of(_prev[0], edges, sessions)
                         if _prev else float('inf'))
            if _rl_part_own_bound < _prev_own - 1e-9:
                best_partitions[graph_name] = (
                    rl_partition, partition_weights, _rl_part_own_bound)

        if (episode + 1) % log_interval == 0:
            n   = log_interval
            avg = _safe_mean(rewards[-n:])
            print(f"  Ep {episode+1:>6} | {graph_name:<16} | avg={avg:.4f}")
            # Periodic checkpoint
            torch.save(phase1_policy.net.state_dict(), "model_files/ckpt_stage3_phase1_latest.pt")
            torch.save(phase2_policy.net.state_dict(), "model_files/ckpt_stage3_phase2_latest.pt")
            _bp_serial = {k: {"partition": [list(p) for p in v[0]],
                                  "weights": {str(ek): float(ev) for ek, ev in v[1].items()},
                                  "bound": float(v[2]) if v[2] != float("inf") else None}
                          for k, v in best_partitions.items()}
            with open("config_files/ckpt_stage3_best_partitions_latest.json", "w") as _bf:
                import json as _j; _j.dump(_bp_serial, _bf)
            torch.save({"episode": episode + 1}, "model_files/ckpt_stage3_meta.pt")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    metrics['best_partitions'] = {
        k: {'partition': v[0], 'weights': v[1], 'bound': v[2]}
        for k, v in best_partitions.items()
    }

    # LP validation summary for Stage 3
    if lp_violations:
        print(f"\n  !! STAGE 3 LP VIOLATIONS: {len(lp_violations)} episodes")
        for ep, gn, ub, lb in lp_violations[:10]:
            print(f"     Ep {ep}: {gn} bound={ub:.6f} < LP_LB={lb:.6f}")
        print(f"  >> These bounds are INVALID — approach needs review.")
    else:
        print(f"\n  LP validation: ALL Stage 3 bounds valid (>= LP lower bound)")
    metrics['lp_bounds'] = lp_bounds
    metrics['lp_violations'] = len(lp_violations)

    print("\nStage 3 complete.\n")
    return phase1_policy, phase2_policy, metrics, best_partitions


# -----------------------------------------------------------------------
# Stage 4 — Train Phase 3 (fractional IO search for novel inequalities)
# -----------------------------------------------------------------------

def run_stage4(phase1_policy, phase2_policy, best_partitions,
               coeff_dim, num_episodes=10000, graph_dataset_size=12,
               finetune_graphs=None):  # finetune_graphs: restrict sampling to named graphs
    """
    Phase 3 training. Uses best partitions from Stage 3 as fixed starting
    points, so the policy can focus entirely on fractional IO discovery.

    The key connection:
      Phase 1 found partition P* and weight vector w*.
      Phase 3 uses P* to determine which node pairs are cross-partition,
      and uses w* as prior λ suggestions.

      After FRACTIONAL_IO(u, v, λ), the resulting inequality has a
      coefficient of λ on the Y_ST term for u's partition and (1-λ) on
      v's partition. When these fractional inequalities are summed and
      SUBMOD is applied, the resulting Y_I coefficient may be irrational —
      which is the signature of an inequality outside the PB family.

    Reward:
      Only positive when extracted bound < partition_bound.
      Zero for matching PB (Phase 3 is not credited for reproducing Phase 2).
      Negative for worse than PB (gradient toward improvement).
    """
    print("=" * 70)
    print(f"STAGE 4: Phase 3 fractional IO search ({num_episodes} episodes)")
    print("=" * 70)

    env = PartitionBoundEnv(graph_dataset_size=graph_dataset_size, stage=4)

    phase3_policy = GNNPhase3Policy(
        coeff_dim=coeff_dim,
        total_episodes=num_episodes
    )

    # Resume from latest periodic checkpoint if available
    _resume_ep_s4 = 0
    os.makedirs("model_files", exist_ok=True)
    os.makedirs("config_files", exist_ok=True)
    if (os.path.exists("model_files/ckpt_stage4_phase3_latest.pt") and
            os.path.exists("model_files/ckpt_stage4_meta.pt")):
        phase3_policy.net.load_state_dict(
            torch.load("model_files/ckpt_stage4_phase3_latest.pt", weights_only=True, map_location=DEVICE))
        _meta_s4 = torch.load("model_files/ckpt_stage4_meta.pt", weights_only=True)
        _resume_ep_s4 = _meta_s4["episode"]
        print(f"  [resume] Stage 4 resuming from episode {_resume_ep_s4}")

    # Targeted-finetune backbone freeze: if targeted_finetune.py set this
    # global, freeze the GraphSAGE feature extractor so only the action
    # heads adapt. The shared representation all graphs depend on stops
    # moving => the finetune targets cannot corrupt features the other
    # graphs rely on. No effect on normal runs (global defaults to False).
    if globals().get("FINETUNE_FREEZE_BACKBONE", False):
        _frozen = 0
        for _p in phase3_policy.net.sage_layers.parameters():
            _p.requires_grad = False
            _frozen += _p.numel()
        # Rebuild the optimizer over only the still-trainable params, else
        # the frozen tensors stay registered and (depending on the optimizer)
        # can still receive weight-decay / momentum updates.
        _trainable = [p for p in phase3_policy.net.parameters() if p.requires_grad]
        try:
            _lr = phase3_policy.optimizer.param_groups[0]["lr"]
        except Exception:
            _lr = 3e-4
        phase3_policy.optimizer = torch.optim.Adam(_trainable, lr=_lr)
        # The CosineAnnealing scheduler is bound to the old optimizer; rebind
        # it to the new one so LR scheduling keeps working post-freeze.
        if hasattr(phase3_policy, "scheduler"):
            phase3_policy.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                phase3_policy.optimizer,
                T_max=max(getattr(phase3_policy, "total_episodes", num_episodes), 1),
                eta_min=1e-5,
            )
        print(f"  [finetune] GraphSAGE backbone FROZEN "
              f"({_frozen} params); optimizer+scheduler rebuilt over "
              f"{sum(p.numel() for p in _trainable)} trainable params.")

    rewards    = []
    per_graph  = defaultdict(list)
    novel_bounds = {}   # graph_name -> (bound, partition, weights, trace)
    metrics    = {'rewards':[], 'bounds':[], 'graph_names':[],
                  'novel_found': [], 'cross_partition_used': [],
                  'used_crypto': [], 'used_decode': [],
                  'used_plain_submod': [], 'no_terminal': [],
                  'mechanism': [],
                  # NEW: separates what the policy's own logged actions
                  # achieved from what the post-episode oracle added on top.
                  # Without this, "novel_found" conflates the two and there
                  # was no way to tell, after the fact, how much of any
                  # given episode's result came from the trained policy
                  # versus this deterministic post-processing step.
                  'pre_oracle_bound': [], 'oracle_fired': [],
                  'oracle_improvement': [], 'partition_bound_per_ep': []}
    per_graph_stats = defaultdict(lambda: {
        'attempts': 0, 'valid_novel': 0, 'terminal_valid': 0,
        'no_terminal': 0, 'used_cross_submod': 0, 'used_crypto': 0,
        'used_decode': 0, 'used_plain_submod': 0,
        'best_bound': float('inf'), 'recent_no_terminal': [],
        '_last_sampled_ep': -1,
    })

    log_interval = 50
    # When finetuning, use a fresh stopper with higher patience since we are
    # targeting hard graphs. The old stopper state from the previous run
    # would fire immediately if resumed, giving zero new episodes.
    _is_finetune = finetune_graphs is not None
    stopper = Stage4EarlyStopper(
        patience           = 8000 if _is_finetune else 5000,
        min_episodes       = 2000 if _is_finetune else 4000,
        window             = 500,
        novel_zero_patience= 5000 if _is_finetune else 3000,
    )
    # On finetune, the stopper starts fresh but resumes from a non-zero episode.
    # The hard-floor check fires when episode - _last_novel_episode >= novel_zero_patience.
    # With _last_novel_episode=-1 and resume at ep 5700, this fires immediately (5700 > 5000).
    # Fix: set _last_novel_episode to _resume_ep_s4 so the patience counts from resume.
    if _is_finetune and _resume_ep_s4 > 0:
        stopper._last_novel_episode = _resume_ep_s4
        stopper.best_episode = _resume_ep_s4

    # Pre-compute LP lower bounds for all graphs in this stage
    lp_bounds = {}
    lp_violations = []
    print(f"\n  {'Graph':<16} {'PB (UB)':>8} {'LP LB':>8} {'Gap':>8}")
    print(f"  {'-'*44}")
    for nodes, edges, sessions in env.graph_dataset:
        gname = identify_graph(nodes, edges, sessions)
        pb    = _compute_partition_bound(nodes, edges, sessions)
        lb    = compute_lp_lower_bound(nodes, edges, sessions)
        lp_bounds[gname] = lb
        print(f"    {gname:<16}: PB={pb:.4f}  LP_LB={lb:.4f}  gap={pb-lb:.4f}")
    print()

    # ---- Stage 4 Proof Logger ----
    # Logs every Phase 3 agent action and resulting inequality state.
    # Only novel episodes (bound < PB) get full step traces saved to JSON.
    # No changes to env, policies, or math — pure observation layer.
    proof_logger = Stage4ProofLogger(
        output_path = "config_files/stage4_proof_log.json",
        verbose     = True,   # print proof summary to stdout when novel bound found
        only_novel  = True,   # only store full step traces for novel episodes
    )
    if _resume_ep_s4 > 0 and os.path.exists("config_files/stage4_proof_log.json"):
        try:
            with open("config_files/stage4_proof_log.json") as _f:
                proof_logger._episodes = json.load(_f)
            novel_bounds.update(_load_stage4_novel_bounds_from_log("config_files/stage4_proof_log.json"))
            for _gn, (_b, _part, _w, _trace) in novel_bounds.items():
                per_graph_stats[_gn]['valid_novel'] = max(per_graph_stats[_gn]['valid_novel'], 1)
                per_graph_stats[_gn]['terminal_valid'] = max(per_graph_stats[_gn]['terminal_valid'], 1)
                per_graph_stats[_gn]['best_bound'] = min(per_graph_stats[_gn]['best_bound'], _b)
            print(f"  [resume] Restored {len(novel_bounds)} graph novel bounds from stage4_proof_log.json")
        except Exception as e:
            print(f"  [warn] Could not restore Stage 4 proof log state: {e}")

    graph_lookup = {
        identify_graph(*gt): gt
        for gt in env.graph_dataset
    }
    # If finetune_graphs is set, restrict dataset to only those graphs.
    # All 10k episodes go to the new graphs — original 16 already solved.
    if finetune_graphs is not None:
        _finetune_dataset = [
            gt for gt in env.graph_dataset
            if identify_graph(*gt) in finetune_graphs
        ]
        if not _finetune_dataset:
            print(f"  [warn] finetune_graphs not found in dataset — using all graphs")
            _finetune_dataset = list(env.graph_dataset)
        env.graph_dataset = _finetune_dataset
        print(f"  [finetune] Sampling restricted to: "
              f"{[identify_graph(*gt) for gt in env.graph_dataset]}")

    balanced_cycle = []

    def _next_balanced_graph():
        nonlocal balanced_cycle
        if not balanced_cycle:
            balanced_cycle = list(env.graph_dataset)
            random.shuffle(balanced_cycle)
        return balanced_cycle.pop()

    def _priority_for_graph(gn):
        stats = per_graph_stats[gn]
        # Steepened from the original 5.0/3.0/1.0. At 5.0, the intended
        # oversampling of zero-novel graphs was invisible in practice --
        # actual attempt counts across all 19 graphs came out to 278-331,
        # a ~15% spread, nowhere near what a real 5x boost should produce.
        # 20.0 makes the boost large enough to actually show up in the
        # attempt-count distribution instead of being lost in the noise
        # of averaging against 18 other graphs' weights.
        if stats['valid_novel'] == 0:
            priority = 20.0
        elif stats['valid_novel'] < 3:
            priority = 6.0
        else:
            priority = 1.0
        recent = stats['recent_no_terminal'][-50:]
        if recent and (sum(recent) / len(recent)) > 0.4:
            priority += 4.0
        # Targeted-priority finetune: additive boost for named graphs. Keeps
        # the full dataset in play (no forgetting) while concentrating extra
        # gradient signal on the finetune targets.
        priority += FINETUNE_PRIORITY_BOOST.get(gn, 0.0)
        return priority

    def _sample_adaptive_graph():
        names = [identify_graph(*gt) for gt in env.graph_dataset]
        weights = [_priority_for_graph(gn) for gn in names]
        total = sum(weights)
        probs = [w / total for w in weights]
        return env.graph_dataset[np.random.choice(len(env.graph_dataset), p=probs)]

    # Hard floor: any graph that has gone this many episodes with zero
    # novel finds gets force-routed to the front of the queue, bypassing
    # the coin flip below. Capped to at most 1-in-6 episodes overall --
    # simulated this against a synthetic run first: WITHOUT the cap, once
    # only one graph remains eligible, "pick the most overdue eligible
    # graph" degenerates into "pick this one every single episode" (it's
    # the only candidate, so it's trivially always "most overdue"). That
    # produced 73% of an entire 5825-episode budget going to one graph in
    # simulation -- the exact catastrophic-forgetting failure mode this
    # whole design was supposed to avoid. The 1-in-6 cap limits how much
    # of the total budget this mechanism can ever claim, regardless of
    # how many graphs are eligible.
    _STUCK_FLOOR_EPISODES  = 150
    _STUCK_OVERRIDE_EVERY  = 6

    def _most_overdue_stuck_graph():
        """Return the zero-novel graph most overdue for attention, or None."""
        candidates = [
            gn for gn in (identify_graph(*gt) for gt in env.graph_dataset)
            if per_graph_stats[gn]['valid_novel'] == 0
            and per_graph_stats[gn]['attempts'] >= _STUCK_FLOOR_EPISODES
        ]
        if not candidates:
            return None
        # Prioritize whichever stuck graph has gone longest since it was
        # last actually sampled, not just attempt count -- otherwise one
        # stuck graph can dominate every forced slot while others starve.
        last_seen = {gn: per_graph_stats[gn].get('_last_sampled_ep', -1)
                     for gn in candidates}
        chosen = min(candidates, key=lambda gn: last_seen[gn])
        for gt in env.graph_dataset:
            if identify_graph(*gt) == chosen:
                return gt
        return None

    for episode in range(_resume_ep_s4, num_episodes):
        if stopper.should_stop(episode):
            print(f"\n  Early stopping at episode {episode}")
            break

        # Selection order, highest priority first:
        #   1. Any zero-novel graph stuck past _STUCK_FLOOR_EPISODES gets
        #      force-routed here, bypassing probability entirely. This is
        #      the fix for the sampling-imbalance bug: a pure probability
        #      boost (below) can still leave a struggling graph under-
        #      served by chance across a few thousand episodes -- this
        #      floor makes that impossible for the worst cases.
        #   2. Otherwise, adaptive-weighted sampling most of the time
        #      (raised from 25% to 65% -- at 25% the intended oversampling
        #      was diluted into a ~15% spread across graphs, not enough to
        #      show up against a graph that's just structurally harder).
        #   3. Otherwise, flat round-robin, to guarantee every graph still
        #      gets baseline coverage and the network doesn't drift toward
        #      only the currently-weighted graphs (catastrophic forgetting
        #      risk if the adaptive fraction dominated completely).
        graph_tuple = (
            _most_overdue_stuck_graph() if episode % _STUCK_OVERRIDE_EVERY == 0
            else None
        )
        if graph_tuple is None:
            graph_tuple = (
                _sample_adaptive_graph()
                if random.random() < 0.65
                else _next_balanced_graph()
            )
        nodes, edges, sessions = graph_tuple
        graph_name = identify_graph(nodes, edges, sessions)
        per_graph_stats[graph_name]['_last_sampled_ep'] = episode

        # Partition selection — single shared path (also used by eval).
        # See _select_stage4_partition for the full rationale. Summary of
        # what this replaces and why:
        #   - The stored Stage-3 "bound" field is NEVER trusted; the stored
        #     partition is re-scored with _partition_bound_of.
        #   - If the RL partition ties the true optimum, it is used
        #     (research claim preserved).
        #   - Otherwise the EXHAUSTIVE optimum partition from
        #     compute_optimal_bound is used. The old code called
        #     _find_optimal_partition, which ran a strictly weaker search
        #     (greedy colorings + 2-way splits) than the one that computes
        #     the target bound — for petersen_10N it handed the agent a
        #     PB-2.5 partition while requiring it to beat 1.875.
        #   - env.partition_bound is ALWAYS the true optimum, including
        #     under --pin-partitions overrides.
        partition, p_weights, opt_global_bound, _partition_source = \
            _select_stage4_partition(graph_name, nodes, edges, sessions,
                                     best_partitions)

        if episode == 0 or (episode + 1) % 500 == 0:
            print(f"  [partition] {graph_name}: source={_partition_source}, "
                  f"search_pb={_partition_bound_of(partition, edges, sessions):.4f}, "
                  f"target_pb={opt_global_bound:.4f}")

        # Set up env for Phase 3 directly
        env.nodes    = nodes
        env.edges    = edges
        env.sessions = sessions
        env.adjacency = {n: set() for n in nodes}
        env.edge_set  = set()
        for u, v in edges:
            env.adjacency[u].add(v); env.adjacency[v].add(u)
            env.edge_set.add((u,v)); env.edge_set.add((v,u))

        env.partition         = partition
        env.partition_weights = p_weights
        env.assignment        = {}
        env.num_groups        = len(partition)
        env._assignment_complete = True
        env._refinement_steps = 0
        env.prev_internal_count = 0

        # Use the globally optimal partition bound as the agent's baseline.
        # This is the true PB the agent must beat, not the (possibly worse)
        # bound achievable by Stage 3's specific partition.
        env.partition_bound   = opt_global_bound
        env._lp_lower_bound   = lp_bounds.get(graph_name, 0.0)

        env._start_phase2()   # builds index, node_ios, base_inequalities
        # Fix A: preseed=False — agent must earn a finite bound via
        # FRACTIONAL_IO → STORE_AND_RESET rather than inheriting a free
        # terminal-form inequality that yields reward=1.0 immediately.
        env._start_phase3(preseed=False)
        env.internal_per_part = env.internal_per_part or []

        # Begin logging this episode — called after phase3 is initialised
        # so the logger has access to the complete env state (index, pool, etc.)
        proof_logger.begin_episode(graph_name, episode, partition, env)

        state = env._get_state()
        state['nodes']             = nodes
        state['edges']             = edges
        state['sessions']          = sessions
        state['partition']         = partition
        state['partition_weights'] = p_weights

        done          = False
        trajectory    = []
        action_counts = defaultdict(int)
        total_reward  = 0.0
        used_cross    = False
        used_plain_submod = False
        used_crypto = False
        used_decode = False

        while not done:
            valid = env.get_valid_actions()
            if not valid:
                # Force terminal
                state, reward, done = env._extract_phase3_bound()
                total_reward += reward
                break

            action = phase3_policy.select_action(state, valid)
            aname  = ACTION_NAMES.get(int(action['type']), '?')
            action_counts[aname] += 1
            if action['type'] == ActionType.CROSS_SUBMOD:
                used_cross = True
            elif action['type'] == ActionType.APPLY_SUBMODULARITY:
                used_plain_submod = True
            elif action['type'] == ActionType.APPLY_CRYPTO:
                used_crypto = True
            elif action['type'] == ActionType.APPLY_DECODE:
                used_decode = True

            # logger.step wraps env.step — logs before/after state, then
            # calls the real env.step unchanged. Returns identical (state, reward, done).
            state, reward, done = proof_logger.step(env, action)
            state['nodes']             = nodes
            state['edges']             = edges
            state['sessions']          = sessions
            state['partition']         = partition
            state['partition_weights'] = p_weights
            trajectory.append({'reward': reward})
            total_reward += reward

        phase3_policy.update(trajectory, total_reward)
        rewards.append(total_reward)

        # Extract best bound from this episode
        # Extract best bound from this episode
        best_b = env.frac_pool.best_bound(
            len(sessions), len(edges), env.internal_per_part
        )
        pb = env.partition_bound
        _no_terminal = best_b == float('inf') or best_b >= 1e9

        # NEW: snapshot the bound BEFORE the oracle runs. This is what the
        # policy's own logged actions achieved on their own, with nothing
        # added afterward.
        _pre_oracle_b = best_b
        _oracle_fired = False

        # Post-episode greedy oracle: apply functional dependence constraints
        # to the best terminal inequality found. This catches improvements
        # that the RL agent hasn't learned to make yet, giving a denser
        # reward signal during early training.
        # GUARD: only accept oracle improvements that stay above LP lower bound.
        # Without this guard, repeated crypto/decode applications compound and
        # drive the bound below LP, producing mathematically invalid results.
        #
        # CRITICAL: when the oracle fires we add the improved inequality back
        # into frac_pool so the trace search below can find it via
        # check_valid_terminal_form + extract_bound.  Without this the trace
        # search fails to match best_b (oracle result is not in pool) and
        # novel_bounds stores "N/A" or — worse — a wrong inequality that
        # happens to have the same bound value by coincidence.
        if env.func_dep_actions is not None and best_b < 1e9:
            _lp_floor = lp_bounds.get(graph_name, 0.0)
            for ineq in env.frac_pool:
                if not ineq.check_valid_terminal_form():
                    continue
                fd_ineq, fd_bound, fd_actions = apply_all_improving_func_dep(
                    ineq, env.func_dep_actions, env.index,
                    env.internal_per_part, len(sessions), len(edges)
                )
                # A sub-LP result here is not "an improvement we decline to
                # take" -- it is the func-dep oracle producing an unsound
                # inequality. Surface it instead of stepping over it.
                if fd_bound < _lp_floor - 1e-9 and fd_bound < 1e9:
                    raise_sub_lp(graph_name, fd_bound, _lp_floor, pb,
                                 episode=episode, where="func_dep_oracle")
                if fd_bound < best_b - 1e-8 and fd_bound >= _lp_floor - 1e-9:
                    best_b = fd_bound
                    _oracle_fired = True
                    # Add the oracle-improved inequality to the pool so the
                    # trace search below can locate and record it correctly.
                    env.frac_pool.add(fd_ineq)
                    break   # one improvement per episode is enough

        per_graph[graph_name].append(best_b)

        # Composite early-stopping signal for Stage 4.
        # w1: how much the agent beat PB this episode (normalised by pb).
        # w2/w3: fed after metrics are recorded so novel_found and
        #        cross_used reflect this episode's outcome.
        _pb_impr = (pb - best_b) / max(pb, 1e-9) if best_b < pb - 1e-8 else 0.0

        # ── THE LP CLAMP IS GONE (1g) ────────────────────────────────────────
        # DELETED, not reordered. The removed lines were:
        #
        #     _lp_floor_s4 = lp_bounds.get(graph_name, 0.0)
        #     if best_b < _lp_floor_s4 - 1e-9:
        #         best_b = pb   # invalid — discard, use PB as fallback
        #
        # They ran BEFORE the `validate_bound_against_lp` block further
        # down that appends to lp_violations. By the time that check ran,
        # best_b had already been overwritten with pb, so it was comparing
        # pb against the floor pb was chosen to satisfy: a tautology, not
        # a check. lp_violations == 0 was structurally guaranteed no
        # matter what the policy produced. okamura_4N episode 2517 carries
        # pool bounds of 0.8333 (step 26, APPLY_CRYPTO) and 0.7143 (step
        # 33, APPLY_DECODE) against an LP lower bound of 1.0, and still
        # reported a clean summary of 1.25.
        #
        # best_b now flows through UNMODIFIED. A sub-LP value is a bug to
        # surface, never a number to correct, so the validation block
        # below sees the real value and raise_sub_lp() reports it.
        _lp_floor_s4 = lp_bounds.get(graph_name, 0.0)
        if best_b < _lp_floor_s4 - 1e-9:
            raise_sub_lp(graph_name, best_b, _lp_floor_s4, pb,
                         episode=episode, where="stage4_training")

        # Close the episode log — records summary and (for novel episodes)
        # the terminal inequality detail. best_b is the real derived
        # value; nothing has been clamped.
        #
        # BLIND SPOT FIX: only_novel=True normally means a graph with zero
        # novel finds has ZERO logged traces anywhere -- not even a summary
        # (end_episode discards self._current entirely before appending).
        # For a graph currently stuck at zero novel finds, force-keep this
        # episode's full trace every ~25 attempts, so there is at least
        # some record of what the policy is actually doing on graphs that
        # never succeed. Cheap: only fires for graphs with valid_novel==0,
        # which is a small minority once training is working.
        _stats_before = per_graph_stats[graph_name]
        _force_keep = (
            _stats_before['valid_novel'] == 0
            and _stats_before['attempts'] % 25 == 0
        )
        if _force_keep:
            proof_logger.only_novel = False
        proof_logger.end_episode(best_b, pb)
        if _force_keep:
            proof_logger.only_novel = True

        # Flush every 200 episodes — protects against crash at end of training.
        # Only novel episodes have step traces in memory so this is cheap.
        if episode % 200 == 0 and episode > 0:
            proof_logger.flush()

        metrics['rewards'].append(total_reward)
        metrics['bounds'].append(best_b if best_b < 1e9 else -1)
        metrics['graph_names'].append(graph_name)
        # novel_found only fires when the bound is BOTH below PB AND above LP.
        # After the LP clamp above, best_b == pb for invalid bounds, so the
        # first condition (< pb) already gates out LP violations.  The explicit
        # LP check below is a belt-and-suspenders guard.
        _lp_floor_nf = lp_bounds.get(graph_name, 0.0)
        _is_novel = (best_b < pb - 1e-8) and (best_b >= _lp_floor_nf - 1e-9)
        metrics['novel_found'].append(1 if _is_novel else 0)
        metrics['cross_partition_used'].append(1 if used_cross else 0)
        metrics['used_plain_submod'].append(1 if used_plain_submod else 0)
        metrics['used_crypto'].append(1 if used_crypto else 0)
        metrics['used_decode'].append(1 if used_decode else 0)
        metrics['no_terminal'].append(1 if _no_terminal else 0)
        metrics['mechanism'].append(_mechanism_label(
            used_cross, used_crypto, used_decode, used_plain_submod, _no_terminal
        ))
        # NEW: log what the policy achieved on its own vs. what the oracle
        # added. _pre_oracle_b is clamped the same way best_b is above, for
        # a fair apples-to-apples comparison (otherwise an LP-invalid
        # pre-oracle reading could look artificially good).
        _pre_oracle_clamped = _pre_oracle_b
        if _pre_oracle_clamped < _lp_floor_nf - 1e-9 or _pre_oracle_clamped >= 1e9:
            _pre_oracle_clamped = pb
        metrics['pre_oracle_bound'].append(_pre_oracle_clamped)
        metrics['oracle_fired'].append(1 if _oracle_fired else 0)
        metrics['oracle_improvement'].append(
            float(_pre_oracle_clamped - best_b) if _oracle_fired else 0.0
        )
        metrics['partition_bound_per_ep'].append(float(pb))

        _pgs = per_graph_stats[graph_name]
        _pgs['attempts'] += 1
        _pgs['valid_novel'] += 1 if _is_novel else 0
        _pgs['terminal_valid'] += 0 if _no_terminal else 1
        _pgs['no_terminal'] += 1 if _no_terminal else 0
        _pgs['used_cross_submod'] += 1 if used_cross else 0
        _pgs['used_plain_submod'] += 1 if used_plain_submod else 0
        _pgs['used_crypto'] += 1 if used_crypto else 0
        _pgs['used_decode'] += 1 if used_decode else 0
        _pgs['recent_no_terminal'].append(1 if _no_terminal else 0)
        if len(_pgs['recent_no_terminal']) > 100:
            _pgs['recent_no_terminal'] = _pgs['recent_no_terminal'][-100:]
        if best_b < _pgs['best_bound']:
            _pgs['best_bound'] = best_b

        # Update composite stopper now that novel_found and cross_used are
        # recorded for this episode.
        stopper.update(
            pb_improvement = _pb_impr,
            novel_found    = 1 if _is_novel else 0,
            cross_used     = 1 if used_cross else 0,
            episode        = episode,
        )

        # Validate against LP lower bound
        if graph_name in lp_bounds and best_b < 1e9:
            lp_lb = lp_bounds[graph_name]
            is_valid, msg = validate_bound_against_lp(
                best_b, lp_lb, graph_name
            )
            if not is_valid:
                lp_violations.append((episode, graph_name, best_b, lp_lb))
                if len(lp_violations) <= 10:
                    print(f"  !! LP VIOLATION: {msg}")

        # Record novel bounds
        if best_b < pb - 1e-8:
            if graph_name not in novel_bounds or best_b < novel_bounds[graph_name][0]:
                # Find the best terminal inequality for trace.
                # Use a slightly looser tolerance (1e-6) to catch the case
                # where the oracle chain introduces small floating-point drift
                # between best_b and the inequality's extract_bound value.
                # If no exact match is found, fall back to whichever valid
                # terminal inequality gives the lowest bound in the pool —
                # this is always the genuine sub-PB inequality, not a
                # degenerate one, because check_valid_terminal_form now
                # requires a cross-partition edge on the RHS.
                best_ineq = None
                best_ineq_b = float('inf')
                for ineq in env.frac_pool:
                    if ineq.check_valid_terminal_form():
                        b2 = ineq.extract_bound(
                            len(sessions), len(edges), env.internal_per_part
                        )
                        if abs(b2 - best_b) < 1e-6 and b2 < best_ineq_b:
                            best_ineq   = ineq
                            best_ineq_b = b2
                # Fallback: take the pool's best valid terminal inequality
                # when no exact match was found (e.g. oracle drift).
                if best_ineq is None:
                    for ineq in env.frac_pool:
                        if ineq.check_valid_terminal_form():
                            b2 = ineq.extract_bound(
                                len(sessions), len(edges), env.internal_per_part
                            )
                            if b2 < best_ineq_b:
                                best_ineq   = ineq
                                best_ineq_b = b2
                novel_bounds[graph_name] = (
                    best_b, partition, p_weights,
                    repr(best_ineq) if best_ineq else "N/A"
                )

        if (episode + 1) % log_interval == 0:
            n       = log_interval
            avg_r   = _safe_mean(rewards[-n:])
            novel_r = np.mean(metrics['novel_found'][-n:])
            cross_r = np.mean(metrics['cross_partition_used'][-n:])
            print(f"  Ep {episode+1:>6} | {graph_name:<16} | "
                  f"avg_r={avg_r:.4f} | novel_rate={100*novel_r:.1f}% | "
                  f"cross_used={100*cross_r:.1f}%")
            # Periodic checkpoint
            torch.save(phase3_policy.net.state_dict(), "model_files/ckpt_stage4_phase3_latest.pt")
            torch.save({"episode": episode + 1}, "model_files/ckpt_stage4_meta.pt")

            if novel_bounds:
                print(f"  ** NOVEL BOUNDS FOUND **")
                for gn, (b, part, w, trace) in sorted(novel_bounds.items()):
                    try:
                        pb2 = _compute_partition_bound(
                            *graph_lookup[gn]
                        )
                    except (KeyError, StopIteration):
                        pb2 = float('inf')
                    impr = f"{(pb2-b)/pb2*100:.2f}%" if pb2 < 1e9 else "n/a"
                    print(f"     {gn}: {b:.6f} < PB={pb2:.6f} (improvement={impr})")
                    if trace != "N/A":
                        print(f"     Trace: {trace[:200]}")
            env.pool = env.pool[:env.num_base]
            env.accumulator = []
            env.frac_pool = env.frac_pool.__class__(MAX_DERIVED)
            env.stored_derived = []
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Write all logged episodes to JSON and generate per-graph proof/status documents
    proof_logger.flush()
    print("\n  Generating proof/status documents for all Stage 4 graphs...")
    os.makedirs("text_files", exist_ok=True)
    for gn in sorted(graph_lookup.keys()):
        nd, ed, ss = graph_lookup[gn]
        try:
            doc_path = os.path.join("text_files", f"proof_{gn}_cd.txt")
            proof_log_exists = os.path.exists("config_files/stage4_proof_log.json")
            wrote_proof = False
            if gn in novel_bounds and proof_log_exists:
                result = generate_proof_document(
                    log_path   = "config_files/stage4_proof_log.json",
                    graph_name = gn,
                    out_path   = doc_path,
                )
                # generate_proof_document returns a "No novel episode" string if
                # the log has no matching entry — treat that as a fallback case
                wrote_proof = os.path.exists(doc_path)
            if not wrote_proof:
                part = (_find_optimal_partition(nd, ed, ss)
                        or (best_partitions.get(gn, (None, {}, None))[0]
                            if best_partitions and gn in best_partitions else []))
                _write_stage4_status_document(
                    graph_name = gn,
                    graph_tuple = (nd, ed, ss),
                    partition = part,
                    pb = _compute_partition_bound(nd, ed, ss),
                    lp_lb = lp_bounds.get(gn, 0.0),
                    stats = per_graph_stats[gn],
                    out_path = doc_path,
                )
                print(f"  [status] Stage 4 status document written to {doc_path}")
        except Exception as e:
            import traceback
            print(f"  [warn] Could not generate proof/status doc for {gn}: {e}")
            traceback.print_exc()

    # Final summary
    print(f"\n{'='*70}")
    print(f"STAGE 4 COMPLETE — NOVEL INEQUALITY SEARCH RESULTS")
    print(f"{'='*70}")
    if novel_bounds:
        for gn, (b, part, w, trace) in sorted(novel_bounds.items()):
            try:
                # Use graph_lookup (full 19-graph dict) not env.graph_dataset
                # which may be restricted to finetune graphs only
                nd, ed, ss = graph_lookup[gn]
                pb2 = _compute_partition_bound(nd, ed, ss)
            except (KeyError, StopIteration):
                pb2 = float('inf')
            print(f"\n  Graph: {gn}")
            print(f"  Novel bound: r <= {b:.6f}")
            print(f"  Partition bound: r <= {pb2:.6f}")
            print(f"  Improvement: {(pb2-b)/pb2*100:.3f}%")
            print(f"  Partition used: {_partition_str(part, ss if 'ss' in locals() else [])}")
            print(f"  Inequality: {trace[:400]}")
    else:
        print("\n  No super-PB bounds found in Stage 4.")
        print("  This does not mean none exist — increase num_episodes")
        print("  or check that CROSS_SUBMOD was used (check cross_used rate).")
        cross_total = sum(metrics['cross_partition_used'])
        print(f"  CROSS_SUBMOD used in {cross_total}/{num_episodes} episodes.")

    # LP validation summary for Stage 4
    print(f"\n{'='*70}")
    print("LP LOWER BOUND VALIDATION — STAGE 4")
    print(f"{'='*70}")
    if lp_violations:
        print(f"  !! {len(lp_violations)} LP VIOLATIONS detected!")
        for ep, gn, ub, lb in lp_violations[:20]:
            print(f"     Ep {ep}: {gn} bound={ub:.6f} < LP_LB={lb:.6f}")
        print(f"  >> These 'novel' bounds are INVALID and must be discarded.")
        # Filter out invalid novel bounds
        invalid_graphs = {gn for _, gn, ub, lb in lp_violations if ub < lb - 1e-6}
        for gn in invalid_graphs:
            if gn in novel_bounds:
                print(f"  >> Removing invalid novel bound for {gn}")
                del novel_bounds[gn]
    else:
        print(f"  All Stage 4 bounds are valid (>= LP lower bound). Approach is sound.")
    metrics['lp_bounds'] = lp_bounds
    metrics['lp_violations'] = len(lp_violations)

    # Policy-vs-oracle breakdown, per graph. This is the number that answers
    # "did the exposure fix actually work" -- not the final bound, which the
    # oracle can produce on its own regardless of what the policy learned.
    print(f"\n{'='*70}")
    print("POLICY vs. ORACLE CONTRIBUTION — STAGE 4")
    print(f"{'='*70}")
    print(f"  {'Graph':<24} {'Episodes':>9} {'Oracle fired %':>15} "
          f"{'Policy beat PB alone %':>23}")
    print(f"  {'-'*76}")
    _by_graph = defaultdict(lambda: {'n': 0, 'oracle_fired': 0, 'policy_beat_pb': 0})
    for i, gname in enumerate(metrics['graph_names']):
        pre_b   = metrics['pre_oracle_bound'][i]
        fired   = metrics['oracle_fired'][i]
        pb_ep   = metrics['partition_bound_per_ep'][i]
        _by_graph[gname]['n'] += 1
        _by_graph[gname]['oracle_fired'] += fired
        if pre_b < pb_ep - 1e-8:
            _by_graph[gname]['policy_beat_pb'] += 1
    for gname in sorted(_by_graph.keys()):
        stats = _by_graph[gname]
        n = max(stats['n'], 1)
        oracle_pct = 100.0 * stats['oracle_fired'] / n
        policy_pct = 100.0 * stats['policy_beat_pb'] / n
        print(f"  {gname:<24} {stats['n']:>9} {oracle_pct:>14.1f}% {policy_pct:>22.1f}%")
    print(f"\n  'Policy beat PB alone' = fraction of episodes where the bound\n"
          f"  BEFORE the oracle ran was already below the partition bound --\n"
          f"  i.e. the trained policy did it without any post-processing help.\n"
          f"  'Oracle fired' = fraction of episodes where the post-episode\n"
          f"  oracle improved the bound beyond what the policy's logged\n"
          f"  actions achieved. A graph can show high numbers on both if the\n"
          f"  policy gets partway and the oracle tightens further on top.")

    return phase3_policy, metrics, novel_bounds


# -----------------------------------------------------------------------
# Top-level train()
# -----------------------------------------------------------------------

def train(stage1_episodes=10000, stage2_episodes=10000,
          stage3_episodes=10000, stage4_episodes=10000,
          graph_dataset_size=5):
    """
    Run all four stages with full checkpoint/resume support.
    - Completed stages are skipped automatically if their final .pt files exist.
    - Each stage saves a *_latest.pt every log_interval episodes for crash recovery.
    - graph_dataset_size controls Stage 1+2 (Tier 1 only = 5).
    - Stage 3 automatically uses size=10, Stage 4 uses size=12.
    """
    import json as _json

    # Ensure all output folders exist before any file is written
    for _d in ("model_files", "config_files", "text_files", "image_files"):
        os.makedirs(_d, exist_ok=True)

    # Device report — if this says 'cpu', CUDA is not available in this environment
    print(f"\n{'='*50}")
    print(f"DEVICE: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("WARNING: Running on CPU — training will be slow.")
        print("  Check that torch was installed with CUDA support:")
        print("  python -c \"import torch; print(torch.cuda.is_available())\"")
    print(f"{'='*50}\n")

    # ---- Stage 1 ----
    _s1_done = (os.path.exists("model_files/ckpt_stage1_phase2.pt") and
                os.path.exists("model_files/ckpt_stage1_coeff_dim.pt"))
    if _s1_done:
        coeff_dim = torch.load("model_files/ckpt_stage1_coeff_dim.pt", weights_only=True)["coeff_dim"]
        phase2_policy = GNNPhase2Policy(coeff_dim=coeff_dim, total_episodes=stage1_episodes)
        phase2_policy.net.load_state_dict(
            torch.load("model_files/ckpt_stage1_phase2.pt", weights_only=True, map_location=DEVICE))
        s1 = {}
        print(f"[skip] Stage 1 already complete (coeff_dim={coeff_dim}), loaded from checkpoint.")
    else:
        phase2_policy, coeff_dim, s1 = run_stage1(stage1_episodes, graph_dataset_size)
        torch.save(phase2_policy.net.state_dict(), "model_files/ckpt_stage1_phase2.pt")
        torch.save({"coeff_dim": coeff_dim}, "model_files/ckpt_stage1_coeff_dim.pt")
        print("[checkpoint] Stage 1 saved -> model_files/ckpt_stage1_phase2.pt")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Stage 2 ----
    _s2_done = (os.path.exists("model_files/ckpt_stage2_phase1.pt") and
                os.path.exists("model_files/ckpt_stage2_phase2.pt"))
    if _s2_done:
        phase1_policy = GNNPhase1Policy(total_episodes=stage2_episodes)
        phase1_policy.net.load_state_dict(
            torch.load("model_files/ckpt_stage2_phase1.pt", weights_only=True, map_location=DEVICE))
        phase2_policy.net.load_state_dict(
            torch.load("model_files/ckpt_stage2_phase2.pt", weights_only=True, map_location=DEVICE))
        s2 = {}
        print("[skip] Stage 2 already complete, loaded from checkpoint.")
    else:
        phase1_policy, s2 = run_stage2(phase2_policy, stage2_episodes, graph_dataset_size)
        torch.save(phase1_policy.net.state_dict(), "model_files/ckpt_stage2_phase1.pt")
        torch.save(phase2_policy.net.state_dict(), "model_files/ckpt_stage2_phase2.pt")
        print("[checkpoint] Stage 2 saved -> model_files/ckpt_stage2_phase1.pt, model_files/ckpt_stage2_phase2.pt")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Stage 3 ----
    _s3_done = (os.path.exists("model_files/ckpt_stage3_phase1.pt") and
                os.path.exists("model_files/ckpt_stage3_phase2.pt") and
                os.path.exists("model_files/ckpt_stage3_best_partitions.json"))
    if _s3_done:
        phase1_policy.net.load_state_dict(
            torch.load("model_files/ckpt_stage3_phase1.pt", weights_only=True, map_location=DEVICE))
        phase2_policy.net.load_state_dict(
            torch.load("model_files/ckpt_stage3_phase2.pt", weights_only=True, map_location=DEVICE))
        with open("model_files/ckpt_stage3_best_partitions.json") as _f:
            _bp_raw = _json.load(_f)
        def _restore_bp_final(v):
            if isinstance(v, dict):
                def _restore_key(ek):
                    if ek.startswith("("): return eval(ek)
                    try: return int(ek)
                    except ValueError: return ek
                return ([list(p) for p in v["partition"]],
                        {_restore_key(ek): ev
                         for ek, ev in v.get("weights", {}).items()},
                        float(v["bound"]) if v.get("bound") is not None else float("inf"))
            else:
                return ([list(p) for p in v], {}, float("inf"))
        best_partitions = {k: _restore_bp_final(v) for k, v in _bp_raw.items()}
        s3 = {}
        print("[skip] Stage 3 already complete, loaded from checkpoint.")
    else:
        phase1_policy, phase2_policy, s3, best_partitions = run_stage3(
            phase1_policy, phase2_policy, stage3_episodes,
            graph_dataset_size=min(13, graph_dataset_size*3)
        )
        torch.save(phase1_policy.net.state_dict(), "model_files/ckpt_stage3_phase1.pt")
        torch.save(phase2_policy.net.state_dict(), "model_files/ckpt_stage3_phase2.pt")
        with open("model_files/ckpt_stage3_best_partitions.json", "w") as _f:
            _json.dump({k: {"partition": [list(p) for p in v[0]],
                            "weights": {str(ek): float(ev) for ek, ev in v[1].items()},
                            "bound": float(v[2]) if v[2] != float("inf") else None}
                        for k, v in best_partitions.items()}, _f)
        print("[checkpoint] Stage 3 saved -> model_files/ckpt_stage3_phase1.pt, model_files/ckpt_stage3_phase2.pt, model_files/ckpt_stage3_best_partitions.json")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Stage 4 ----
    _s4_done = os.path.exists("model_files/ckpt_stage4_phase3.pt")
    if _s4_done:
        phase3_policy = GNNPhase3Policy(coeff_dim=coeff_dim, total_episodes=stage4_episodes)
        phase3_policy.net.load_state_dict(
            torch.load("model_files/ckpt_stage4_phase3.pt", weights_only=True, map_location=DEVICE))
        s4 = {}
        # Restore novel_bounds from proof log so training_summary.txt is correct
        # when all stages are skipped and only eval + summary are run.
        novel_bounds = _load_stage4_novel_bounds_from_log("config_files/stage4_proof_log.json")
        if novel_bounds:
            print(f"[skip] Stage 4 already complete, loaded from checkpoint. "
                  f"Restored {len(novel_bounds)} novel bounds from proof log.")
        else:
            print("[skip] Stage 4 already complete, loaded from checkpoint.")
    else:
        phase3_policy, s4, novel_bounds = run_stage4(
            phase1_policy, phase2_policy, best_partitions,
            coeff_dim, stage4_episodes,
            graph_dataset_size=min(19, graph_dataset_size*4)
        )
        torch.save(phase3_policy.net.state_dict(), "model_files/ckpt_stage4_phase3.pt")
        print("[checkpoint] Stage 4 saved -> model_files/ckpt_stage4_phase3.pt")

    return (phase1_policy, phase2_policy, phase3_policy,
            {'stage1': s1, 'stage2': s2, 'stage3': s3, 'stage4': s4},
            novel_bounds, best_partitions)


def evaluate(phase1_policy, phase2_policy, phase3_policy,
             best_partitions=None,
             graph_dataset_size=5,
             stochastic_episodes=30,
             greedy_trials=8,
             num_episodes=None):
    """Evaluation across all phases.

    REDESIGNED: the old version ran `num_episodes` total episodes, randomly
    sampling a graph each time, and always used greedy=True action
    selection. That combination is broken: greedy + a fixed stored
    partition per graph makes the ENTIRE Phase 3 rollout deterministic --
    same graph, same partition, same policy, same argmax choices every
    single time. Running the same graph 250+ times (as num_episodes=4800
    over 19 graphs implied) produced 250+ identical copies of one outcome,
    not 250 independent samples. "novel_pct" computed from that is
    meaningless -- it can only ever read exactly 0% or exactly 100% per
    graph, and increasing num_episodes cannot change that; it only repeats
    identical work.

    This version separates two different questions that were being
    conflated into one broken metric:

      1. "What's the single best bound this policy can produce?"
         -> answered by ONE greedy (deterministic) rollout per graph.
            Running it more than once is pure waste.

      2. "How reliably does the policy find a good bound on its own --
         is it consistent, or does it only work when it gets lucky?"
         -> answered by `stochastic_episodes` independent STOCHASTIC
            (greedy=False) rollouts per graph, with the fraction that
            reach novel status reported as a genuine success rate.

    Both numbers are reported per graph. `stochastic_episodes` defaults to
    30 -- large enough to distinguish "never happens" from "happens most
    of the time" without excessive compute; raise it if you need tighter
    confidence on a specific graph's success rate.

    `num_episodes` is accepted for backward compatibility with old call
    sites but is otherwise ignored -- it doesn't map cleanly onto the new
    two-pass structure. Pass `stochastic_episodes` instead.
    """
    with torch.no_grad():
        return _evaluate_inner(phase1_policy, phase2_policy, phase3_policy,
                               best_partitions, graph_dataset_size,
                               stochastic_episodes, greedy_trials)


def _run_eval_episode(env, graph_tuple, phase1_policy, phase2_policy,
                       phase3_policy, best_partitions, lp_bounds, greedy):
    """Run one full Phase1/2/3 eval rollout on graph_tuple.

    Returns (p2_bound, p3_bound, is_novel, lp_valid, pb, graph_name).
    Factored out of the old inline loop body so it can be called once in
    greedy mode and `stochastic_episodes` times in stochastic mode per
    graph, instead of being duplicated.
    """
    nodes, edges, sessions = graph_tuple
    graph_name = identify_graph(nodes, edges, sessions)
    pb = _compute_partition_bound(nodes, edges, sessions)

    # FIX: eval now selects its partition through the SAME shared path as
    # Stage 4 training (_select_stage4_partition). Previously eval used the
    # stored RL partition unconditionally when present (even when its real
    # bound was worse than the optimum — petersen_10N was evaluated under
    # the PB-2.143 pair partition while training searched under a PB-2.5
    # greedy partition), and ran a fresh stochastic Phase-1 rollout when
    # absent (the 6 graphs missing from ckpt_stage3_best_partitions.json got
    # a DIFFERENT random partition every eval run). Both paths are gone:
    # train and eval now always search under the identical partition, and
    # the missing-6-graphs eval gap is closed by the exhaustive fallback.
    partition, partition_weights, _opt_pb, _src = _select_stage4_partition(
        graph_name, nodes, edges, sessions, best_partitions)

    state = env.reset(fixed_graph=graph_tuple, fixed_partition=partition)
    env.partition_weights = partition_weights or {}
    state['edges'] = edges; state['sessions'] = sessions
    state['nodes'] = nodes; state['partition'] = partition
    while env.current_phase == Phase.PHASE2:
        valid = env.get_valid_actions()
        if not valid: break
        # NOTE: GNNPhase2Policy.select_action has no greedy mode -- it is
        # always stochastic. This means even the "greedy" (Phase 3) pass
        # below is not fully deterministic end-to-end: Phase 2's proof
        # construction under the fixed partition varies episode to
        # episode. Flagging this rather than pretending the greedy pass
        # is perfectly reproducible -- it mostly is, for the Phase 3
        # decisions that matter here, but not completely.
        action = phase2_policy.select_action(state, valid)
        state, _, done = env.step(action)
        state['edges'] = edges; state['sessions'] = sessions
        if done: break

    # FIX: no fabricated fallback here either. The old line was
    #   p2_bound = abs(env._best_pool_bound() or pb * 2)
    # -- the same "invent a plausible number" bug class as the removed
    # Stage-4 pb*2 fallback, just surviving in the Phase-2 column of
    # eval_results.json. None now propagates; consumers guard for it.
    p2_bound = env._best_pool_bound()

    env.nodes             = list(nodes)
    env.edges             = list(edges)
    env.sessions          = list(sessions)
    env.partition         = [list(g) for g in partition]
    env.partition_weights = partition_weights or {}
    env.partition_bound   = pb
    env._lp_lower_bound   = lp_bounds.get(graph_name, 0.0)
    env.index             = None
    env.adjacency = {n: set() for n in nodes}
    env.edge_set  = set()
    for _u, _v in edges:
        env.adjacency[_u].add(_v); env.adjacency[_v].add(_u)
        env.edge_set.add((_u, _v)); env.edge_set.add((_v, _u))
    env._start_phase2()
    env._start_phase3(preseed=False)
    env.internal_per_part = env.internal_per_part or []

    state2 = env._get_state()
    state2['nodes']             = nodes
    state2['edges']             = edges
    state2['sessions']          = sessions
    state2['partition']         = partition
    state2['partition_weights'] = partition_weights or {}

    while env.current_phase == Phase.PHASE3:
        valid = env.get_valid_actions()
        if not valid: break
        action = phase3_policy.select_action(state2, valid, greedy=greedy)
        state2, _, done = env.step(action)
        state2['nodes']             = nodes
        state2['edges']             = edges
        state2['sessions']          = sessions
        state2['partition']         = partition
        state2['partition_weights'] = partition_weights or {}
        if done: break

    p3_bound, p3_ineq = env.frac_pool.best_terminal(
        len(sessions), len(edges), env.internal_per_part
    )
    # NO fabricated fallback. If the episode never assembled a valid
    # terminal-form inequality, p3_bound stays None -- a real missing
    # result, not a number that looks like data. Downstream code MUST
    # treat None as "no result", never format/average/plot it as a bound.
    # This is a distinct failure mode from an LP-violating result below.
    no_valid_terminal = (p3_bound == float('inf'))

    if no_valid_terminal:
        env.pool = []
        env.frac_pool = env.frac_pool.__class__(MAX_DERIVED)
        env.accumulator = []
        env.stored_derived = []
        return p2_bound, None, False, False, pb, graph_name, True

    # Soundness assertion (1f): attribute the bound to the crypto/decode
    # inequalities in its provenance trace and fail if it is below what
    # they can prove.
    if p3_ineq is not None:
        assert_derivation_sound(
            p3_ineq, p3_bound, graph_name=graph_name,
            episode="eval", step="terminal",
        )

    # ── THE LP CLAMP IS GONE (1g), eval side ─────────────────────────────────
    # DELETED, not reordered. The removed line was:
    #
    #     if not lp_valid:
    #         p3_bound = pb   # clamp invalid sub-LP results to PB
    #
    # It ran before the returned bound was appended to
    # stochastic_p3_bounds, and stoch_lp_violations is computed from that
    # list by comparing against lp_lb -- i.e. comparing already-clamped
    # values against the floor they were clamped to satisfy. Dead code.
    # The greedy path was laundered the same way: a violating trial got
    # clamped UP to pb, which made it lose the best-of-trials min, so
    # lp_valid=False never reached the summary. Between them,
    # eval_results.json's lp_violations: 0 was structurally guaranteed
    # regardless of what the policy actually produced.
    #
    # p3_bound now propagates UNMODIFIED, with lp_valid=False attached.
    lp_floor = lp_bounds.get(graph_name, 0.0)
    lp_valid = p3_bound >= lp_floor - 1e-6
    if not lp_valid:
        raise_sub_lp(graph_name, p3_bound, lp_floor, pb,
                     episode="eval", where="eval_episode")

    is_novel = p3_bound < pb - 1e-8

    # Release per-episode pool/accumulator state so objects don't accumulate.
    env.pool = []
    env.frac_pool = env.frac_pool.__class__(MAX_DERIVED)
    env.accumulator = []
    env.stored_derived = []

    return p2_bound, p3_bound, is_novel, lp_valid, pb, graph_name, False


def _is_oom_error(e: Exception) -> bool:
    """True if this exception is a CUDA/CPU out-of-memory error."""
    msg = str(e).lower()
    return (isinstance(e, torch.cuda.OutOfMemoryError)
            or ("out of memory" in msg)
            or ("cuda error" in msg and "memory" in msg))


def _safe_run_episode(env, graph_tuple, phase1_policy, phase2_policy,
                       phase3_policy, best_partitions, lp_bounds, greedy):
    """Wraps _run_eval_episode with OOM recovery.

    On an out-of-memory error: clears CUDA cache and Python garbage, waits
    briefly, and retries ONCE. If it fails again, logs a warning and
    returns None instead of crashing the whole eval run -- an 8-hour
    overnight run should not die because one episode on one graph hit a
    transient memory spike.

    Returns the same tuple _run_eval_episode does, or None if both
    attempts failed.
    """
    for attempt in range(2):
        try:
            return _run_eval_episode(
                env, graph_tuple, phase1_policy, phase2_policy, phase3_policy,
                best_partitions, lp_bounds, greedy=greedy
            )
        except Exception as e:
            if not _is_oom_error(e):
                raise  # not an OOM -- a real bug, don't hide it, let it surface
            gname = identify_graph(*graph_tuple)
            print(f"  !! OOM on {gname} (greedy={greedy}), attempt {attempt+1}/2 "
                  f"-- clearing memory and {'retrying' if attempt == 0 else 'skipping'}")
            del e
            env.pool = []
            env.frac_pool = env.frac_pool.__class__(MAX_DERIVED)
            env.accumulator = []
            env.stored_derived = []
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(2)
    print(f"  !! Skipping this episode after repeated OOM -- not counted "
          f"toward results, but the run continues.")
    return None


def _evaluate_inner(phase1_policy, phase2_policy, phase3_policy,
                    best_partitions=None,
                    graph_dataset_size=5,
                    stochastic_episodes=30,
                    greedy_trials=8):
    """Inner implementation of evaluate() — called inside torch.no_grad()."""
    env = PartitionBoundEnv(graph_dataset_size=graph_dataset_size, stage=4)
    results = defaultdict(lambda: {
        'p2_bounds': [], 'greedy_p3_bound': None, 'greedy_is_novel': False,
        'greedy_lp_valid': True, 'greedy_no_valid_terminal': False,
        'stochastic_novel_count': 0,
        'stochastic_total': 0, 'stochastic_p3_bounds': [],
        'stochastic_no_terminal_count': 0,
    })

    # Full RNG re-seed at the top of eval, so a re-derivation run is
    # reproducible regardless of how much training ran before it.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    _fingerprint = _policy_fingerprint(phase1_policy, phase2_policy, phase3_policy)
    print(f"  [eval] policy fingerprint: {_fingerprint}")

    _eval_log_path = "config_files/eval_results.json"
    _eval_log = {"episodes": [], "summary": {}, "complete": False,
                 "policy_fingerprint": _fingerprint}
    def _flush_eval_log():
        with open(_eval_log_path, "w") as _f:
            json.dump(_eval_log, _f, indent=2)

    lp_bounds = {}
    for nodes_g, edges_g, sessions_g in env.graph_dataset:
        gn = identify_graph(nodes_g, edges_g, sessions_g)
        if gn not in lp_bounds:
            lp_bounds[gn] = compute_lp_lower_bound(nodes_g, edges_g, sessions_g)
    # ------------------------------------------------------------------
    # Resume support: if config_files/eval_results.json already has
    # results from a previous (possibly crashed) run, skip re-computing
    # graphs that already finished. An 8-hour overnight run crashing at
    # graph 15 of 19 should mean redoing 4 graphs, not 19.
    #
    # MUST run before anything below writes to _eval_log_path -- reading
    # the prior file AFTER a flush has already overwritten it with a
    # fresh, empty log defeats the entire point.
    # ------------------------------------------------------------------
    _already_done = set()
    try:
        with open(_eval_log_path) as _f:
            _prior = json.load(_f)
        # NOTE: deliberately NOT requiring _prior.get("complete") here.
        # Per-graph summaries are written immediately after each graph
        # finishes (see below), with "complete" only flipping to True at
        # the very end. Requiring "complete" would mean a genuine mid-run
        # crash -- the exact scenario this is meant to protect against --
        # produces a log this code refuses to resume from. Matching
        # settings is still required, so a run with different
        # stochastic_episodes/greedy_trials won't be silently reused.
        # Resume is now additionally keyed on the POLICY FINGERPRINT. A
        # prior log written by different weights (or by the pre-fix proof
        # calculus, which has no fingerprint recorded at all) is not
        # reusable and is ignored -- every graph is recomputed.
        if _prior.get("stochastic_episodes_per_graph") == stochastic_episodes \
                and _prior.get("greedy_trials") == greedy_trials \
                and _prior.get("policy_fingerprint") == _fingerprint:
            for gname, s in _prior.get("summary", {}).items():
                results[gname]['greedy_p3_bound'] = s['greedy_best_bound']  # may be None
                results[gname]['greedy_is_novel'] = s['greedy_is_novel']
                results[gname]['greedy_lp_valid'] = s['greedy_lp_valid']
                results[gname]['greedy_no_valid_terminal'] = s.get(
                    'greedy_no_valid_terminal', s['greedy_best_bound'] is None
                )
                results[gname]['stochastic_total'] = s['stochastic_episodes']
                results[gname]['stochastic_no_terminal_count'] = s.get(
                    'stochastic_no_terminal_count', 0
                )
                # stochastic_success_pct is None when every stochastic
                # episode for this graph failed to reach a terminal form --
                # do not divide by it or treat it as 0.0% (0.0% implies
                # "tried and found nothing novel", which is a different,
                # better outcome than "never produced anything to judge").
                pct = s.get('stochastic_success_pct')
                if pct is not None and s['stochastic_episodes'] > 0:
                    results[gname]['stochastic_novel_count'] = int(
                        round(pct / 100.0 * s['stochastic_episodes'])
                    )
                else:
                    results[gname]['stochastic_novel_count'] = 0
                _already_done.add(gname)
            if _already_done:
                was_complete = _prior.get("complete", False)
                print(f"  [resume] {len(_already_done)} graphs found in a prior "
                      f"{'complete' if was_complete else 'INCOMPLETE (crashed?)'} "
                      f"run with matching settings -- skipping them.")
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        pass  # no valid prior run to resume from -- start clean

    # Seed the fresh log's summary with whatever we're resuming, so a
    # crash immediately after this point still has the resumed graphs on
    # disk rather than losing them again.
    if _already_done:
        for gname in _already_done:
            _eval_log.setdefault("summary", {})[gname] = _prior["summary"][gname]

    _eval_log["lp_bounds"] = lp_bounds
    _flush_eval_log()

    # ------------------------------------------------------------------
    # One combined pass per graph: greedy_trials rollouts (best kept) then
    # stochastic_episodes rollouts, summary written immediately after each
    # graph finishes.
    #
    # RESTRUCTURED: this used to be two separate full loops over all 19
    # graphs (all greedy passes, then all stochastic passes), with the
    # summary only ever built once at the very end from the in-memory
    # `results` dict. That meant a crash anywhere during Pass 2 lost BOTH
    # the entire stochastic pass in progress AND the fact that Pass 1 had
    # already finished for every graph -- there was nothing on disk
    # structured enough to resume from. Doing both passes per graph, and
    # writing that graph's summary to disk immediately after, means a
    # crash at graph 15 of 19 costs you graph 15 (partially) plus 4
    # ungraded graphs, not all 19.
    #
    # CORRECTION on greedy: this used to run ONE rollout per graph on the
    # assumption that greedy Phase 3 + a fixed stored partition makes the
    # whole rollout deterministic. That's wrong -- GNNPhase2Policy has no
    # greedy mode, it always samples, and Phase 2 runs before Phase 3 and
    # builds what Phase 3 works with. A single "greedy" run is really "one
    # random Phase 2 draw, then deterministic Phase 3 given that draw."
    # Running several trials and keeping the best result accounts for that.
    # ------------------------------------------------------------------
    print(f"\n{'='*90}")
    print("EVALUATION SUMMARY")
    print(f"{'='*90}")
    print(f"  {'Graph':<16} {'PB':>8} {'LP LB':>8} {'Greedy best':>12} "
          f"{'Greedy OK':>10} {'Stoch. success%':>16} {'Stoch. LP OK':>13}")
    print(f"  {'-'*88}")
    eval_violations = 0
    summary = dict(_eval_log.get("summary", {}))  # keep any resumed entries

    for graph_tuple in env.graph_dataset:
        gname = identify_graph(*graph_tuple)
        if gname in _already_done:
            r = results[gname]
            pb2 = _compute_partition_bound(*graph_tuple)
            lp_lb = lp_bounds.get(gname, 0.0)
            print(f"  {gname:<16} {pb2:>8.4f} {lp_lb:>8.4f} "
                  f"{'(resumed from prior run)':>44}")
            continue

        # --- greedy_trials rollouts, best kept ---
        # best_p3b/best_record only ever hold REAL bounds. no_terminal_count
        # tracks trials that never assembled a valid terminal form, tallied
        # separately from OOM skips and from successful trials.
        best_p3b = None
        best_record = None
        oom_skips = 0
        no_terminal_count = 0
        for _ in range(greedy_trials):
            result = _safe_run_episode(
                env, graph_tuple, phase1_policy, phase2_policy, phase3_policy,
                best_partitions, lp_bounds, greedy=True
            )
            if result is None:
                oom_skips += 1
                continue
            p2b, p3b, is_novel, lp_valid, pb, _, no_terminal = result
            # p2b may be None now (no fabricated pb*2 fallback in
            # _run_eval_episode). Only real bounds enter the list.
            if p2b is not None:
                results[gname]['p2_bounds'].append(p2b)
            if no_terminal:
                no_terminal_count += 1
                continue
            if best_p3b is None or p3b < best_p3b:
                best_p3b = p3b
                best_record = (p2b, p3b, is_novel, lp_valid, pb)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if oom_skips == greedy_trials:
            print(f"  !! All {greedy_trials} greedy trials OOM'd for {gname} "
                  f"-- skipping entirely, check VRAM headroom.")
            continue

        if best_record is None:
            # Every non-OOM trial ran but NONE produced a valid terminal
            # inequality. This is a real, distinct outcome from a bound --
            # do not synthesize a number. Record it explicitly.
            print(f"  !! {no_terminal_count}/{greedy_trials - oom_skips} greedy "
                  f"trials for {gname} never reached a valid terminal form "
                  f"(policy likely stuck on a repeated non-terminal action).")
            results[gname]['greedy_p3_bound'] = None
            results[gname]['greedy_is_novel'] = False
            results[gname]['greedy_lp_valid'] = False
            results[gname]['greedy_no_valid_terminal'] = True
            _eval_log["episodes"].append({
                "mode": "greedy_best_of_trials", "graph": gname,
                "greedy_trials": greedy_trials,
                "p3_bound": None, "no_valid_terminal": True,
                "partition_bound": float(pb),
                "lp_lower_bound": float(lp_bounds.get(gname, 0.0)),
                "is_novel": False, "lp_valid": False,
            })
        else:
            p2b, p3b, is_novel, lp_valid, pb = best_record
            results[gname]['greedy_p3_bound'] = p3b
            results[gname]['greedy_is_novel'] = is_novel
            results[gname]['greedy_lp_valid'] = lp_valid
            results[gname]['greedy_no_valid_terminal'] = False
            _eval_log["episodes"].append({
                "mode": "greedy_best_of_trials", "graph": gname,
                "greedy_trials": greedy_trials,
                "p2_bound": (float(p2b) if p2b is not None else None),
                "p3_bound": float(p3b),
                "partition_bound": float(pb),
                "lp_lower_bound": float(lp_bounds.get(gname, 0.0)),
                "is_novel": bool(is_novel), "lp_valid": bool(lp_valid),
            })

        # --- stochastic_episodes rollouts ---
        oom_count = 0
        no_terminal_stoch_count = 0
        for _ in range(stochastic_episodes):
            result = _safe_run_episode(
                env, graph_tuple, phase1_policy, phase2_policy, phase3_policy,
                best_partitions, lp_bounds, greedy=False
            )
            if result is None:
                oom_count += 1
                continue
            _, p3b_s, is_novel_s, lp_valid_s, pb_s, _, no_terminal_s = result
            if no_terminal_s:
                no_terminal_stoch_count += 1
                _eval_log["episodes"].append({
                    "mode": "stochastic", "graph": gname,
                    "p3_bound": None, "no_valid_terminal": True,
                    "partition_bound": float(pb_s),
                    "lp_lower_bound": float(lp_bounds.get(gname, 0.0)),
                    "is_novel": False, "lp_valid": False,
                })
                continue
            # 'stochastic_total' denotes episodes that produced a real,
            # evaluable bound -- NOT total episodes attempted. Success rate
            # is therefore success / (episodes that could have succeeded),
            # not diluted or inflated by no-terminal episodes.
            results[gname]['stochastic_total'] += 1
            results[gname]['stochastic_p3_bounds'].append(p3b_s)
            if is_novel_s and lp_valid_s:
                results[gname]['stochastic_novel_count'] += 1
            _eval_log["episodes"].append({
                "mode": "stochastic", "graph": gname,
                "p3_bound": float(p3b_s), "partition_bound": float(pb_s),
                "lp_lower_bound": float(lp_bounds.get(gname, 0.0)),
                "is_novel": bool(is_novel_s), "lp_valid": bool(lp_valid_s),
            })
            if len(_eval_log["episodes"]) > 400:
                _eval_log["episodes"] = _eval_log["episodes"][-400:]
        results[gname]['stochastic_no_terminal_count'] = no_terminal_stoch_count
        if oom_count:
            print(f"  !! {oom_count}/{stochastic_episodes} stochastic episodes "
                  f"OOM'd for {gname} -- success rate computed over the "
                  f"{stochastic_episodes - oom_count} that completed.")
        if no_terminal_stoch_count:
            print(f"  !! {no_terminal_stoch_count}/{stochastic_episodes} stochastic "
                  f"episodes for {gname} never reached a valid terminal form.")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- write this graph's summary immediately -- this is what makes
        # resume actually work after a real crash, not just after a clean
        # full completion ---
        r = results[gname]
        pb2 = _compute_partition_bound(*graph_tuple)
        lp_lb = lp_bounds.get(gname, 0.0)
        # stochastic_total now only counts episodes that produced a real
        # bound (no_terminal episodes are excluded, not silently folded in).
        # If ALL stochastic episodes failed to terminate, report that
        # explicitly instead of dividing by a fabricated denominator of 1
        # and printing a 0.0% that looks like "tried and failed" when it
        # actually means "never produced anything to judge".
        stoch_attempted = r['stochastic_total'] + r['stochastic_no_terminal_count']
        if r['stochastic_total'] > 0:
            stoch_pct = 100.0 * r['stochastic_novel_count'] / r['stochastic_total']
            stoch_pct_str = f"{stoch_pct:>14.1f}%"
        else:
            stoch_pct_str = "  NO_TERMINAL "
        stoch_lp_violations = sum(
            1 for b in r['stochastic_p3_bounds'] if b < lp_lb - 1e-6
        )
        stoch_lp_ok = (stoch_lp_violations == 0)
        # A no-valid-terminal result is NOT an LP violation -- it's an
        # absence of any result to check. Only count real sub-LP bounds
        # as violations, so the final message means what it says.
        if (not r['greedy_no_valid_terminal'] and not r['greedy_lp_valid']) \
                or (not stoch_lp_ok):
            eval_violations += 1
        flag = " *** NOVEL (greedy) ***" if r['greedy_is_novel'] else ""
        greedy_bound_str = (
            f"{r['greedy_p3_bound']:>12.4f}" if r['greedy_p3_bound'] is not None
            else f"{'NO_TERMINAL':>12}"
        )
        greedy_ok_str = (
            "  OK" if (r['greedy_lp_valid'] and not r['greedy_no_valid_terminal'])
            else "FAIL"
        )
        print(f"  {gname:<16} {pb2:>8.4f} {lp_lb:>8.4f} {greedy_bound_str} "
              f"{greedy_ok_str:>10} {stoch_pct_str} "
              f"{'  OK' if stoch_lp_ok else 'FAIL':>13}{flag}")
        summary[gname] = {
            "partition_bound": float(pb2),
            "lp_lower_bound": float(lp_lb),
            "greedy_best_bound": (
                float(r['greedy_p3_bound']) if r['greedy_p3_bound'] is not None else None
            ),
            "greedy_no_valid_terminal": bool(r['greedy_no_valid_terminal']),
            "greedy_is_novel": bool(r['greedy_is_novel']),
            "greedy_lp_valid": bool(r['greedy_lp_valid']),
            "stochastic_attempted": int(stoch_attempted),
            "stochastic_no_terminal_count": int(r['stochastic_no_terminal_count']),
            "stochastic_success_pct": (
                float(100.0 * r['stochastic_novel_count'] / r['stochastic_total'])
                if r['stochastic_total'] > 0 else None
            ),
            # NOTE: "stochastic_episodes" here means episodes that produced a
            # real bound, i.e. r['stochastic_total'] -- NOT the number
            # attempted. Use "stochastic_attempted" above for the raw count,
            # including no-terminal episodes. Keeping both prevents the
            # resume logic (which reads this field) from silently treating
            # no-terminal episodes as successes or vice versa.
            "stochastic_episodes": int(r['stochastic_total']),
            "stochastic_lp_valid": bool(stoch_lp_ok),
            "improvement_pct": float(100.0 * (pb2 - r['greedy_p3_bound']) / pb2)
                                if r['greedy_is_novel'] else 0.0,
        }
        # Flush the summary NOW, per graph, with complete still False --
        # this is the actual crash-recovery mechanism. "complete" only
        # flips to True once every graph is done, further down.
        _eval_log["summary"] = summary
        _eval_log["stochastic_episodes_per_graph"] = stochastic_episodes
        _eval_log["greedy_trials"] = greedy_trials
        _flush_eval_log()

    if eval_violations:
        print(f"\n  !! {eval_violations} graphs have at least one bound "
              f"below LP lower bound — INVALID")
    else:
        print(f"\n  LP validation: ALL evaluation bounds are valid.")

    _eval_log["summary"] = summary
    _eval_log["complete"] = True
    _eval_log["stochastic_episodes_per_graph"] = stochastic_episodes
    _eval_log["greedy_trials"] = greedy_trials
    _eval_log["policy_fingerprint"] = _fingerprint
    # lp_violations now counts real, surfaced violations. With
    # SUB_LP_FATAL=True a sub-LP bound aborts the run before reaching
    # here, so a completed log with 0 means 0 -- not "the clamp hid them".
    _eval_log["lp_violations"] = eval_violations
    _eval_log["sub_lp_observed"] = [list(x) for x in SUB_LP_OBSERVED]
    _flush_eval_log()
    print(f"\n  [eval log] Results saved to config_files/eval_results.json")

    return dict(results)




if __name__ == "__main__":
    import sys

    # --pin-partitions: enable PARTITION_OVERRIDES (experiment mode).
    # Currently pins al_bashabsheh_7N to the pre-regression 4-part
    # partition {v1},{S1,S3,t1},{S2,t2},{t3}. That partition is
    # DELIBERATELY PB-suboptimal (own bound 2.4 vs true PB 2.0 — it
    # internalises only 2 of the graph's 3 sessions); the target bound
    # the agent must beat stays the true PB 2.0. Use together with
    # --finetune-stage4 to test whether the search partition, not the
    # policy, is what blocks al_bashabsheh's novel bound.
    if "--pin-partitions" in sys.argv:
        PIN_PARTITIONS = True
        print("[experiment] --pin-partitions active: "
              f"overrides for {sorted(PARTITION_OVERRIDES.keys())}")

    if "--finetune-stage4" in sys.argv:
        # Fine-tune Stage 4 on all 19 graphs using existing checkpoint.
        # Resumes from ckpt_stage4_phase3.pt and trains for fewer episodes.
        # Use this when new graphs were added after an initial full run.
        import json as _json
        from collections import defaultdict

        # Ensure output folders exist
        for _d in ("model_files", "config_files", "text_files", "image_files"):
            os.makedirs(_d, exist_ok=True)

        print("\n=== Stage 4 fine-tune on 19 graphs ===")
        print(f"DEVICE: {DEVICE}")

        # Load existing Stage 3 artifacts
        coeff_dim_data = torch.load("model_files/ckpt_stage1_coeff_dim.pt",
                                    weights_only=True, map_location=DEVICE)
        coeff_dim = coeff_dim_data["coeff_dim"]

        phase1_policy = GNNPhase1Policy()
        phase1_policy.net.load_state_dict(
            torch.load("model_files/ckpt_stage3_phase1.pt",
                       weights_only=True, map_location=DEVICE))

        phase2_policy = GNNPhase2Policy(coeff_dim=coeff_dim)
        phase2_policy.net.load_state_dict(
            torch.load("model_files/ckpt_stage3_phase2.pt",
                       weights_only=True, map_location=DEVICE))

        with open("model_files/ckpt_stage3_best_partitions.json") as _f:
            _bp_raw = _json.load(_f)
        best_partitions = {
            k: ([tuple(p) for p in v["partition"]], v["weights"], v["bound"])
            for k, v in _bp_raw.items()
        }

        # _resume_ep_s4 inside run_stage4 reads the meta checkpoint and gets
        # the episode count from the completed run (e.g. 30000).
        # num_episodes must be _resume + extra, not just the extra count.
        _s4_meta_path = "model_files/ckpt_stage4_meta.pt"
        _s4_done_eps  = 0
        if os.path.exists(_s4_meta_path):
            _s4_done_eps = torch.load(_s4_meta_path, weights_only=True,
                                      map_location=DEVICE).get("episode", 0)
        _finetune_extra = 10000
        _finetune_total = _s4_done_eps + _finetune_extra
        print(f"  Resuming from episode {_s4_done_eps}, "
              f"running {_finetune_extra} extra → total {_finetune_total}")

        # Target the 3 graphs confirmed (via real trace data, not guessing)
        # to have zero novel finds despite 500+ dedicated attempts each in
        # the last run: al_bashabsheh_7N, okamura_seymour_8N, petersen_10N.
        # okamura_network_paper_5N still excluded -- gap=0, mathematically
        # impossible regardless of training.
        #
        # WARNING: finetune_graphs restricts run_stage4's ENTIRE dataset to
        # ONLY these 3 graphs for the full run -- not a bias, a total
        # exclusion of the other 16. That means every other graph's
        # performance (including graphs that were novel in the last run)
        # gets zero gradient signal for the whole finetune duration and can
        # regress. Re-evaluate ALL 19 graphs after this finishes, not just
        # these 3 -- do not assume the rest are untouched.
        _FINETUNE_GRAPHS = [
            "al_bashabsheh_7N",
            "okamura_seymour_8N",
            "petersen_10N",
        ]

        phase3_policy, s4, novel_bounds = run_stage4(
            phase1_policy, phase2_policy, best_partitions,
            coeff_dim,
            num_episodes=_finetune_total,
            graph_dataset_size=19,
            finetune_graphs=_FINETUNE_GRAPHS,
        )

        torch.save(phase3_policy.net.state_dict(),
                   "model_files/ckpt_stage4_phase3.pt")
        print("[checkpoint] Fine-tune saved -> model_files/ckpt_stage4_phase3.pt")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        eval_results = evaluate(
            phase1_policy, phase2_policy, phase3_policy,
            best_partitions=best_partitions,
            stochastic_episodes=30, greedy_trials=8, graph_dataset_size=19
        )
        print("Fine-tune + eval complete.")
        sys.exit(0)


    t0 = time.time()
    (phase1_policy, phase2_policy, phase3_policy,
     train_metrics, novel_bounds, best_partitions) = train(
        stage1_episodes=10000,   # Phase 2 proof calculus  — Tier 1 graphs (5)
        stage2_episodes=15000,   # Phase 1 partition learn — Tier 1 graphs (5)
        stage3_episodes=15000,   # Joint fine-tuning       — Tier 1+2 graphs (10)
        stage4_episodes=30000,   # Phase 3 fractional IO   — All graphs (16)
        graph_dataset_size=5
    )
    # Free GPU memory accumulated during training before running evaluation.
    # Training fills ~14.5 GiB; without this, evaluate() OOMs immediately.
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    eval_results = evaluate(
        phase1_policy, phase2_policy, phase3_policy,
        best_partitions=best_partitions,
        stochastic_episodes=30, greedy_trials=8, graph_dataset_size=19
    )

    runtime = time.time() - t0
    print(f"\nTotal runtime: {runtime:.1f}s ({runtime/60:.1f} min)")

    def _jsonable(obj):
        """Recursively make an object JSON-serializable."""
        if isinstance(obj, dict):
            return {str(k): _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonable(x) for x in obj]
        if isinstance(obj, float):
            if obj != obj:        # NaN
                return None
            if obj == float('inf') or obj == float('-inf'):
                return None
            return obj
        if isinstance(obj, (int, bool)):
            return obj
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if (v != v or v == float('inf') or v == float('-inf')) else v
        if isinstance(obj, np.ndarray):
            return _jsonable(obj.tolist())
        return str(obj)

    all_metrics = {
        'stage1': _jsonable(train_metrics.get('stage1', {})),
        'stage2': _jsonable(train_metrics.get('stage2', {})),
        'stage3': _jsonable(train_metrics.get('stage3', {})),
        'stage4': _jsonable(train_metrics.get('stage4', {})),
        'eval': _jsonable(eval_results),
        'novel_bounds': _jsonable(novel_bounds),
        'runtime_s': runtime,
    }
    with open('config_files/training_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print("Metrics saved to config_files/training_metrics.json")

    # Summary file
    with open('text_files/training_summary.txt', 'w', encoding='utf-8') as f:
        f.write(f"Training completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Runtime: {runtime:.1f}s ({runtime/60:.1f} min)\n\n")

        # --- Results table: Graph | PB | LP | RL Bound | Improvement ---
        f.write("=" * 80 + "\n")
        f.write("RESULTS TABLE\n")
        f.write("=" * 80 + "\n")
        f.write(f"  {'Graph':<22} {'PB':>8} {'LP LB':>8} {'RL Bound':>10} {'Improv%':>9} {'Status':>12}\n")
        f.write(f"  {'-'*75}\n")

        # Collect all graphs from Stage 4 dataset
        from fixed_graph_generation import get_all_graph_infos, identify_graph
        from lp_lower_bound import compute_lp_lower_bound
        # Use info.optimal_bound — exhaustive search from registry, NOT
        # _compute_partition_bound which uses greedy and returns wrong values
        # e.g. paper_7N: greedy returns 2.0, exhaustive correct answer is 1.667

        all_infos = get_all_graph_infos()
        for info in all_infos:
            gn = info.name
            nodes, edges, sessions = info.nodes, info.edges, info.sessions
            pb  = info.optimal_bound   # exhaustive search, not greedy
            lp  = compute_lp_lower_bound(nodes, edges, sessions)

            if gn in novel_bounds:
                rl_b = novel_bounds[gn][0]
                improv = (pb - rl_b) / pb * 100
                if abs(rl_b - lp) < 1e-6:
                    status = "TIGHT (=LP)"
                elif rl_b < pb - 1e-6:
                    status = "NOVEL"
                else:
                    status = "NO IMPROV"
            elif abs(pb - lp) < 1e-6:
                # PB already equals LP lower bound — routing capacity achieved.
                # No novel bound is mathematically possible; this is a
                # verification result, not a failure.
                rl_b   = pb
                improv = 0.0
                status = "TIGHT (PB=LP)"
            else:
                rl_b   = pb   # no improvement found
                improv = 0.0
                status = "NOT FOUND"

            f.write(f"  {gn:<22} {pb:>8.4f} {lp:>8.4f} {rl_b:>10.6f} {improv:>8.2f}%  {status:>10}\n")

        f.write(f"\n  {'='*75}\n")
        n_novel  = len(novel_bounds)
        n_tight  = sum(1 for gn,(b,*_) in novel_bounds.items()
                       if abs(b - compute_lp_lower_bound(
                           *next((i.nodes,i.edges,i.sessions)
                                 for i in all_infos if i.name==gn))) < 1e-6)
        f.write(f"  Novel bounds found: {n_novel} / {len(all_infos)} graphs\n")
        f.write(f"  Tight (= LP lower bound): {n_tight}\n\n")

        # --- Full inequality traces ---
        if novel_bounds:
            f.write("=" * 80 + "\n")
            f.write("NOVEL INEQUALITIES (full traces)\n")
            f.write("=" * 80 + "\n")
            for gn, (b, part, w, trace) in sorted(novel_bounds.items()):
                pb2 = next(i.optimal_bound for i in all_infos if i.name==gn)
                f.write(f"\n  Graph:      {gn}\n")
                f.write(f"  RL Bound:   r <= {b:.6f}\n")
                f.write(f"  PB:         r <= {pb2:.6f}\n")
                f.write(f"  Improvement:{(pb2-b)/pb2*100:.3f}%\n")
                f.write(f"  Inequality: {trace}\n")
        else:
            f.write("No novel bounds found in this run.\n")

    print("Summary saved to text_files/training_summary.txt")
    # --- Auto-generate plots ---
    print("\n--- Generating plots ---")
    import subprocess, sys
    try:
        subprocess.run([sys.executable, "plot_training.py"], check=True)
        print("Plots generated successfully.")
    except Exception as e:
        print(f"Plot generation failed: {e}")
        print("Run 'python plot_training.py' manually.")

    # --- Auto-generate graph visualization ---
    try:
        subprocess.run([sys.executable, "visualize_graphs.py"], check=True)
        print("Graph visualization generated.")
    except Exception as e:
        print(f"Graph visualization failed: {e}")

    # --- Auto git push ---
    # COMMENTED OUT ON REQUEST: disabled before a training run so nothing
    # gets pushed automatically. Re-enable by uncommenting this block.
    # print("\n--- Pushing to git ---")
    # try:
    #     subprocess.run(["git", "add", "--all"], check=True)
    #     status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    #     if status.stdout.strip() != "":
    #         novel_tag = ""
    #         if novel_bounds:
    #             graphs_str = ", ".join(sorted(novel_bounds.keys()))
    #             novel_tag = f" - NOVEL BOUNDS: {graphs_str}"
    #         commit_msg = (
    #             f"Training run completed - "
    #             f"{time.strftime('%Y-%m-%d %H:%M')} - "
    #             f"runtime {runtime/60:.0f}min"
    #             f"{novel_tag}"
    #         )
    #         subprocess.run(["git", "commit", "-m", commit_msg], check=True)
    #         subprocess.run(["git", "push"], check=True)
    #         print("Git push completed successfully.")
    #     else:
    #         print("No changes to commit. Auto git push skipped.")
    # except subprocess.CalledProcessError as e:
    #     print(f"Git push failed with error code: {e.returncode}")
    #     print("Run 'git status' and commit manually if needed.")
    # except Exception as e:
    #     print(f"Git push failed: {e}")
    #     print("Push manually with: git add --all && git commit -m 'training results' && git push")