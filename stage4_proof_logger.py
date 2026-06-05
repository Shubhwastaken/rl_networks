"""
stage4_proof_logger.py
======================
Drop-in logger for Stage 4 (Phase 3) of the RL training loop.

PURPOSE
-------
Captures every action the agent takes during a Phase 3 episode and the
exact inequality state before and after each action.  The output is a
self-contained JSON file that can be replayed step-by-step to reconstruct
the proof for any graph where a novel bound was found.

USAGE
-----
Import and call wrap_stage4_env(env) immediately after _start_phase3() in
the Stage 4 loop in fixed_training.py.  Then call logger.flush() after
each episode to write the log.  No other code changes needed.

    from stage4_proof_logger import Stage4ProofLogger

    logger = Stage4ProofLogger(output_path="stage4_proof_log.json")

    # inside Stage 4 episode loop, after env._start_phase3():
    logger.begin_episode(graph_name, episode, partition, env)

    # inside action loop, replace:
    #   state, reward, done = env.step(action)
    # with:
    #   state, reward, done = logger.step(env, action)

    # after episode ends:
    logger.end_episode(best_b, pb)

    # after all episodes:
    logger.flush()

WHAT GETS LOGGED PER ACTION
----------------------------
Every step records:
  - step_number
  - action_type name and raw action dict
  - accumulator contents BEFORE the action (full inequality repr + YI coeff + edge coeffs)
  - frac_pool size BEFORE the action
  - result inequality (union) repr after CROSS_SUBMOD / APPLY_SUBMODULARITY
  - result inequality repr after STORE_AND_RESET
  - for CROSS_SUBMOD: inputs A and B, whether collapse fired, c_min used, YI coeff of result
  - reward received
  - whether pool's best_bound improved

For CROSS_SUBMOD specifically (the step your professor flagged as unclear):
  We log the FULL derivation:
    - input A: full repr, partition_ids, has_yst, YI coeff
    - input B: full repr, partition_ids, has_yst, YI coeff
    - union result: full repr, YI coeff, edge coeffs, is_terminal
    - collapse_fired: bool
    - c_min: the minimum YST coefficient used in collapse
    - covered_sessions: which sessions the union covers
    - internal_sessions: YI_Pi terms surviving in result
"""

import json
import copy
from typing import Any, Dict, List, Optional
from fixed_inequality import Inequality, FractionalInequality, EntropyIndex

# Action type names (mirrors ActionType in fixed_environment.py)
_ACTION_NAMES = {
    0:  "ASSIGN_NODE",
    1:  "SWAP_NODE",
    2:  "MOVE_NODE",
    3:  "FINALIZE_PARTITION",
    4:  "STORE_AND_RESET",
    5:  "COMBINE_STORED",
    6:  "DECLARE_TERMINAL",
    10: "FRACTIONAL_IO",
    11: "CROSS_SUBMOD",
    20: "APPLY_CRYPTO",
    21: "APPLY_DECODE",
}


# ---------------------------------------------------------------------------
# Helpers to snapshot an inequality into a plain dict
# ---------------------------------------------------------------------------

def _snap_ineq(ineq, index: EntropyIndex, label: str = "") -> Dict:
    """Produce a human-readable + machine-readable snapshot of an inequality."""
    if ineq is None:
        return {"label": label, "repr": "None"}

    snap = {
        "label":          label,
        "repr":           repr(ineq),
        "yi_coeff":       round(ineq.coeffs[index.yi_idx()], 6),
        "partition_ids":  list(getattr(ineq, "partition_ids", [])),
        "source_nodes":   list(getattr(ineq, "source_nodes",  [])),
        "has_yst":        bool(ineq.active_yst()),
        "active_yst":     sorted(ineq.active_yst()),
        "yi_pi_coeffs":   {},
        "edge_coeffs":    {},
    }

    # Y_I_Pi terms (internal sessions per partition)
    for i in range(len(index.partitions)):
        c = ineq.coeffs[index.yi_pi_idx(i)]
        if abs(c) > 1e-9:
            snap["yi_pi_coeffs"][f"P{i}"] = round(c, 6)

    # Edge coefficients on RHS (stored as negative in coeffs)
    for e in index.edges:
        c = ineq.coeffs[index.edge_idx(e)]
        if abs(c) > 1e-9:
            key = f"U_{e[0]}_{e[1]}"
            snap["edge_coeffs"][key] = round(abs(c), 6)

    snap["edge_total"] = round(sum(snap["edge_coeffs"].values()), 6)
    return snap


def _snap_pool(env, label: str = "") -> Dict:
    """Snapshot the frac_pool: size + best terminal inequality."""
    size = len(env.frac_pool)
    best_repr = "none"
    best_b    = None
    for ineq in env.frac_pool:
        if ineq.check_valid_terminal_form():
            b = ineq.extract_bound(
                len(env.sessions), len(env.edges), env.internal_per_part
            )
            if best_b is None or b < best_b:
                best_b    = b
                best_repr = repr(ineq)
    return {
        "label":              label,
        "pool_size":          size,
        "best_bound":         round(best_b, 6) if best_b is not None else None,
        "best_terminal_repr": best_repr,
    }


def _snap_accumulator(env, index: EntropyIndex, label: str = "") -> List[Dict]:
    """Snapshot every item currently in the accumulator."""
    return [
        _snap_ineq(ineq, index, label=f"acc[{i}]")
        for i, ineq in enumerate(env.accumulator)
    ]


# ---------------------------------------------------------------------------
# Cross-submod detailed snapshot
# ---------------------------------------------------------------------------

def _snap_cross_submod(a, b, union_ineq, inter_ineq, index: EntropyIndex) -> Dict:
    """
    Full derivation snapshot for a CROSS_SUBMOD / APPLY_SUBMODULARITY step.

    Records:
      - inputs A and B (full snapshots)
      - union and intersection results
      - whether the YST->YI collapse fired in the union
      - the c_min coefficient used in the collapse (= min of YST weights)
      - which sessions are covered by the union
      - the net YI gain from the collapse
    """
    # Was collapse triggered in union? Collapse fires if no YST terms remain
    # but YI coeff went up. We detect it by checking: union has no active YST
    # and YI > max(a.yi, b.yi).
    a_yi = a.coeffs[index.yi_idx()]
    b_yi = b.coeffs[index.yi_idx()]
    u_yi = union_ineq.coeffs[index.yi_idx()]
    collapse_fired = (not union_ineq.active_yst()) and (u_yi > max(a_yi, b_yi) + 1e-9)

    # Reconstruct c_min: min of the YST coefficients of the two inputs
    # (only meaningful if collapse fired)
    yst_coeffs = {}
    c_min = None
    if collapse_fired:
        a_ysts = {i: a.coeffs[index.yst_idx(i)] for i in a.active_yst()}
        b_ysts = {i: b.coeffs[index.yst_idx(i)] for i in b.active_yst()}
        all_ysts = {**a_ysts, **b_ysts}
        yst_coeffs = {f"YST_P{k}": round(v, 6) for k, v in all_ysts.items()}
        c_min = round(min(all_ysts.values()), 6) if all_ysts else None

    # Sessions covered by the union of the two ST sets
    a_sessions = set()
    b_sessions = set()
    for i in a.active_yst():
        a_sessions |= index.st_sessions[i]
    for i in b.active_yst():
        b_sessions |= index.st_sessions[i]
    union_sessions = sorted(a_sessions | b_sessions)
    all_sessions   = sorted(index.all_sessions())
    full_cover     = (set(union_sessions) == set(all_sessions))

    return {
        "input_A":           _snap_ineq(a,         index, label="A"),
        "input_B":           _snap_ineq(b,         index, label="B"),
        "union_result":      _snap_ineq(union_ineq, index, label="union"),
        "inter_result":      _snap_ineq(inter_ineq, index, label="inter"),
        "collapse_fired":    collapse_fired,
        "c_min_used":        c_min,
        "yst_coeffs_before": yst_coeffs,
        "sessions_in_A":     sorted(a_sessions),
        "sessions_in_B":     sorted(b_sessions),
        "sessions_in_union": union_sessions,
        "all_sessions":      all_sessions,
        "full_cover":        full_cover,
        "yi_gain":           round(u_yi - max(a_yi, b_yi), 6),
        "is_terminal_form":  union_ineq.check_valid_terminal_form(),
    }


# ---------------------------------------------------------------------------
# Main logger class
# ---------------------------------------------------------------------------

class Stage4ProofLogger:
    """
    Logs every Phase 3 agent action and the resulting inequality states.

    Output format: JSON file with one entry per episode.
    Only episodes where a novel bound (< partition bound) was found are
    written in full detail. All other episodes record only the summary.

    Parameters
    ----------
    output_path : str
        Path to the output JSON file.
    verbose : bool
        If True, also print a human-readable trace to stdout during training
        for episodes where a novel bound is found.
    only_novel : bool
        If True (default), write full step-by-step traces only for novel
        episodes. All episodes get a summary entry regardless.
    """

    def __init__(
        self,
        output_path: str = "stage4_proof_log.json",
        verbose:     bool = True,
        only_novel:  bool = True,
    ):
        self.output_path  = output_path
        self.verbose      = verbose
        self.only_novel   = only_novel

        self._episodes:    List[Dict] = []   # full log
        self._current:     Optional[Dict] = None
        self._env         = None
        self._index       = None
        self._prev_best_b = None

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def begin_episode(self, graph_name: str, episode: int,
                      partition: List, env) -> None:
        """Call immediately after env._start_phase3()."""
        self._env   = env
        self._index = env.index

        self._prev_best_b = env.frac_pool.best_bound(
            len(env.sessions), len(env.edges), env.internal_per_part
        )

        self._current = {
            "episode":      episode,
            "graph_name":   graph_name,
            "partition":    [list(p) for p in partition],
            "sessions":     list(env.sessions),
            "edges":        [list(e) for e in env.edges],
            "nodes":        list(env.nodes),
            "partition_bound": round(env.partition_bound, 6),
            "steps":        [],
            "summary":      {},
        }

    def step(self, env, action: Dict):
        """
        Replacement for env.step(action).  Logs before/after state, then
        calls the real env.step.

        Returns the same (state, reward, done) tuple as env.step.
        """
        if self._current is None:
            return env.step(action)

        index   = self._index
        atype   = int(action.get("type", -1))
        aname   = _ACTION_NAMES.get(atype, f"UNKNOWN_{atype}")

        # ---- snapshot BEFORE ----
        pre_acc  = _snap_accumulator(env, index, label="before")
        pre_pool = _snap_pool(env, label="before")

        # ---- special pre-capture for CROSS_SUBMOD ----
        cross_detail = None
        if atype in (11, 4):  # CROSS_SUBMOD or APPLY_SUBMODULARITY
            idx_i = action.get("idx_i", 0)
            idx_j = action.get("idx_j", 1)
            if (len(env.accumulator) >= 2
                    and idx_i < len(env.accumulator)
                    and idx_j < len(env.accumulator)
                    and idx_i != idx_j):
                # Import here to avoid circular at module level
                from fixed_submodularity import apply_pairwise_submodularity
                a = env.accumulator[idx_i]
                b = env.accumulator[idx_j]
                # Run submod on copies — does NOT mutate env
                u, inter = apply_pairwise_submodularity(
                    a.copy(), b.copy(), env.index, env.sessions
                )
                cross_detail = _snap_cross_submod(a, b, u, inter, index)

        # ---- special pre-capture for STORE_AND_RESET ----
        store_detail = None
        if atype == 4:  # STORE_AND_RESET
            if env.accumulator:
                combined = env.accumulator[0].copy()
                for ineq in env.accumulator[1:]:
                    combined = combined.add(ineq)
                combined = combined.cancel_source_terms()
                store_detail = _snap_ineq(combined, index, label="combined_after_cancel")

        # ---- call real env step ----
        state, reward, done = env.step(action)

        # ---- snapshot AFTER ----
        post_pool = _snap_pool(env, label="after")

        # Did pool best bound improve?
        new_best_b = post_pool["best_bound"]
        improved   = (
            new_best_b is not None
            and (self._prev_best_b is None or new_best_b < self._prev_best_b - 1e-9)
        )
        if improved:
            self._prev_best_b = new_best_b

        # ---- build step record ----
        step_record = {
            "step":         len(self._current["steps"]) + 1,
            "action_type":  aname,
            "action_raw":   {k: (v if not hasattr(v, 'item') else float(v))
                             for k, v in action.items()},
            "reward":       round(float(reward), 6),
            "done":         bool(done),
            "pool_improved": improved,
            "pre_accumulator":  pre_acc,
            "pre_pool":         pre_pool,
            "post_pool":        post_pool,
        }

        if cross_detail is not None:
            step_record["cross_submod_detail"] = cross_detail

        if store_detail is not None:
            step_record["store_and_reset_combined"] = store_detail

        # FRACTIONAL_IO detail
        if atype == 10:
            step_record["fractional_io"] = {
                "node_u": action.get("node_u"),
                "node_v": action.get("node_v"),
                "lambda": round(float(action.get("lam", 0.5)), 4),
            }

        self._current["steps"].append(step_record)

        # ---- verbose stdout for novel episodes ----
        if self.verbose and improved and new_best_b is not None:
            pb = self._current["partition_bound"]
            if new_best_b < pb - 1e-8:
                self._print_step(step_record, graph_name=self._current["graph_name"])

        return state, reward, done

    def end_episode(self, best_b: float, partition_bound: float) -> None:
        """Call after the episode loop ends."""
        if self._current is None:
            return

        is_novel = best_b < partition_bound - 1e-8

        # Build summary
        action_counts = {}
        for s in self._current["steps"]:
            action_counts[s["action_type"]] = action_counts.get(s["action_type"], 0) + 1

        # Find the terminal inequality if novel
        terminal_repr = None
        terminal_detail = None
        if is_novel and self._env is not None:
            for ineq in self._env.frac_pool:
                if ineq.check_valid_terminal_form():
                    b2 = ineq.extract_bound(
                        len(self._env.sessions),
                        len(self._env.edges),
                        self._env.internal_per_part
                    )
                    if abs(b2 - best_b) < 1e-5:
                        terminal_repr   = repr(ineq)
                        terminal_detail = _snap_ineq(ineq, self._index, label="terminal")
                        break

        self._current["summary"] = {
            "best_bound":      round(best_b, 6),
            "partition_bound": round(partition_bound, 6),
            "is_novel":        is_novel,
            "improvement_pct": round((partition_bound - best_b) / partition_bound * 100, 3)
                               if is_novel else 0.0,
            "total_steps":     len(self._current["steps"]),
            "action_counts":   action_counts,
            "terminal_ineq":   terminal_repr,
            "terminal_detail": terminal_detail,
        }

        # Only keep full step trace for novel episodes (saves memory)
        if self.only_novel and not is_novel:
            self._current["steps"] = []

        self._episodes.append(self._current)
        self._current = None

        # Print human-readable proof summary for novel episodes
        if self.verbose and is_novel:
            self._print_episode_summary(self._episodes[-1])

    def flush(self) -> None:
        """Write all logged episodes to the output JSON file."""
        with open(self.output_path, "w") as f:
            json.dump(self._episodes, f, indent=2)
        print(f"\n[Stage4ProofLogger] Written {len(self._episodes)} episodes "
              f"to {self.output_path}")
        novel = sum(1 for ep in self._episodes if ep["summary"].get("is_novel"))
        print(f"[Stage4ProofLogger] Novel episodes with full traces: {novel}")

    # ------------------------------------------------------------------
    # Human-readable print helpers
    # ------------------------------------------------------------------

    def _print_step(self, s: Dict, graph_name: str = "") -> None:
        print(f"\n  [{graph_name}] Step {s['step']} — {s['action_type']}  "
              f"reward={s['reward']:.4f}  pool_improved={s['pool_improved']}")

        if "cross_submod_detail" in s:
            d = s["cross_submod_detail"]
            print(f"    INPUT A: {d['input_A']['repr']}")
            print(f"    INPUT B: {d['input_B']['repr']}")
            print(f"    UNION  : {d['union_result']['repr']}")
            print(f"    collapse_fired={d['collapse_fired']}  "
                  f"c_min={d['c_min_used']}  full_cover={d['full_cover']}")
            print(f"    YI gain: {d['yi_gain']:.4f}  "
                  f"is_terminal={d['is_terminal_form']}")

        if "store_and_reset_combined" in s:
            d = s["store_and_reset_combined"]
            print(f"    COMBINED (after cancel): {d['repr']}")

        if s["pool_improved"] and s["post_pool"]["best_bound"] is not None:
            print(f"    ** pool best_bound improved to "
                  f"{s['post_pool']['best_bound']:.6f}")

    def _print_episode_summary(self, ep: Dict) -> None:
        sm = ep["summary"]
        print(f"\n{'='*65}")
        print(f"NOVEL BOUND FOUND: {ep['graph_name']}")
        print(f"  r <= {sm['best_bound']:.6f}  (PB = {sm['partition_bound']:.6f}, "
              f"improvement = {sm['improvement_pct']:.2f}%)")
        print(f"  Steps: {sm['total_steps']}  |  Actions: {sm['action_counts']}")
        if sm["terminal_ineq"]:
            print(f"  Terminal inequality: {sm['terminal_ineq']}")
        if sm["terminal_detail"]:
            td = sm["terminal_detail"]
            print(f"  YI coeff: {td['yi_coeff']}  |  Edge total: {td['edge_total']}")
            print(f"  Edge coefficients:")
            for e, c in td["edge_coeffs"].items():
                print(f"    {e}: {c:.4f}")

        # Print step-by-step proof trace
        print(f"\n  --- Step-by-step proof trace ---")
        for s in ep["steps"]:
            line = f"  [{s['step']:>2}] {s['action_type']:<22} reward={s['reward']:>8.4f}"
            if s["pool_improved"]:
                line += f"  ** bound -> {s['post_pool']['best_bound']}"
            print(line)

            if "cross_submod_detail" in s:
                d = s["cross_submod_detail"]
                print(f"       A: {d['input_A']['repr'][:80]}")
                print(f"       B: {d['input_B']['repr'][:80]}")
                print(f"       union: {d['union_result']['repr'][:80]}")
                print(f"       collapse={d['collapse_fired']}  "
                      f"c_min={d['c_min_used']}  "
                      f"sessions_covered={d['full_cover']}")

            if "store_and_reset_combined" in s:
                print(f"       combined: "
                      f"{s['store_and_reset_combined']['repr'][:80]}")

        print(f"{'='*65}")


# ---------------------------------------------------------------------------
# Convenience: generate a human-readable proof document from a log file
# ---------------------------------------------------------------------------

def generate_proof_document(log_path: str, graph_name: str,
                             out_path: str = None) -> str:
    """
    Given a JSON log file and a graph name, generate a step-by-step
    human-readable proof document string.

    The output mirrors the structure of a handwritten proof:
      1. Network definition
      2. Partition
      3. Per-node inequalities (from the best episode's terminal inequality)
      4. Each agent step with full arithmetic
      5. Terminal inequality
      6. Rate extraction

    Returns the document as a string and optionally writes it to out_path.
    """
    with open(log_path) as f:
        episodes = json.load(f)

    # Find best novel episode for this graph
    candidates = [
        ep for ep in episodes
        if ep["graph_name"] == graph_name
        and ep["summary"].get("is_novel")
        and ep["steps"]
    ]
    if not candidates:
        return f"No novel episode found for {graph_name} in {log_path}"

    best_ep = min(candidates, key=lambda e: e["summary"]["best_bound"])
    sm      = best_ep["summary"]

    lines = []
    lines.append("=" * 70)
    lines.append(f"PROOF DOCUMENT: {graph_name}")
    lines.append(f"Generated from Stage 4 action log — episode {best_ep['episode']}")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"NETWORK")
    lines.append(f"  Nodes   : {best_ep['nodes']}")
    lines.append(f"  Edges   : {best_ep['edges']}")
    lines.append(f"  Sessions: {best_ep['sessions']}")
    lines.append("")
    lines.append(f"PARTITION (from RL Phase 1)")
    for i, p in enumerate(best_ep["partition"]):
        lines.append(f"  P{i+1} = {{{', '.join(p)}}}")
    lines.append("")
    lines.append(f"RESULT")
    lines.append(f"  r <= {sm['best_bound']:.6f}  "
                 f"(partition bound = {sm['partition_bound']:.6f}, "
                 f"improvement = {sm['improvement_pct']:.2f}%)")
    lines.append("")
    lines.append(f"TERMINAL INEQUALITY")
    lines.append(f"  {sm['terminal_ineq'] or 'N/A'}")
    lines.append("")
    lines.append(f"STEP-BY-STEP AGENT PROOF TRACE")
    lines.append("-" * 70)

    for step in best_ep["steps"]:
        lines.append(f"\nStep {step['step']}: {step['action_type']}  "
                     f"(reward = {step['reward']:.4f})")

        # Pre-accumulator state
        if step["pre_accumulator"]:
            lines.append("  Accumulator before:")
            for item in step["pre_accumulator"]:
                lines.append(f"    {item['label']}: {item['repr']}")
                lines.append(f"      YI coeff = {item['yi_coeff']},  "
                             f"has_YST = {item['has_yst']},  "
                             f"edge_total = {item['edge_total']}")

        # CROSS_SUBMOD full derivation
        if "cross_submod_detail" in step:
            d = step["cross_submod_detail"]
            lines.append("")
            lines.append("  CROSS-PARTITION SUBMODULARITY APPLICATION")
            lines.append(f"  Input A (partition_ids={d['input_A']['partition_ids']}):")
            lines.append(f"    {d['input_A']['repr']}")
            lines.append(f"    YI_coeff = {d['input_A']['yi_coeff']}  |  "
                         f"has_YST = {d['input_A']['has_yst']}  |  "
                         f"active_YST = {d['input_A']['active_yst']}")
            lines.append("")
            lines.append(f"  Input B (partition_ids={d['input_B']['partition_ids']}):")
            lines.append(f"    {d['input_B']['repr']}")
            lines.append(f"    YI_coeff = {d['input_B']['yi_coeff']}  |  "
                         f"has_YST = {d['input_B']['has_yst']}  |  "
                         f"active_YST = {d['input_B']['active_yst']}")
            lines.append("")
            lines.append(f"  Applying h(A) + h(B) >= h(A∪B) + h(A∩B):")
            lines.append(f"    Sessions covered by A : {d['sessions_in_A']}")
            lines.append(f"    Sessions covered by B : {d['sessions_in_B']}")
            lines.append(f"    Sessions in union     : {d['sessions_in_union']}")
            lines.append(f"    Full session coverage : {d['full_cover']}")
            lines.append("")
            if d["collapse_fired"]:
                lines.append(f"  YST -> YI COLLAPSE FIRED:")
                lines.append(f"    YST coefficients before collapse: "
                             f"{d['yst_coeffs_before']}")
                lines.append(f"    c_min = {d['c_min_used']}  "
                             f"(Weighted Subadditivity Theorem)")
                lines.append(f"    Collapsed: sum(c_k * h(YST_Pk)) >= "
                             f"{d['c_min_used']} * h(Y_I)")
                lines.append(f"    YI gain = {d['yi_gain']:.4f}")
            else:
                lines.append("  YST collapse did NOT fire "
                             "(union does not cover all sessions)")
            lines.append("")
            lines.append(f"  Union result:")
            lines.append(f"    {d['union_result']['repr']}")
            lines.append(f"    YI_coeff = {d['union_result']['yi_coeff']}  |  "
                         f"edge_total = {d['union_result']['edge_total']}  |  "
                         f"is_terminal = {d['union_result']['is_terminal_form']}")
            if d["union_result"]["yi_pi_coeffs"]:
                lines.append(f"    YI_Pi terms: {d['union_result']['yi_pi_coeffs']}")

        # STORE_AND_RESET detail
        if "store_and_reset_combined" in step:
            d = step["store_and_reset_combined"]
            lines.append("")
            lines.append("  STORE_AND_RESET — accumulator summed + source terms cancelled:")
            lines.append(f"    {d['repr']}")
            lines.append(f"    YI_coeff = {d['yi_coeff']}  |  "
                         f"edge_total = {d['edge_total']}")

        # FRACTIONAL_IO detail
        if "fractional_io" in step:
            fi = step["fractional_io"]
            lines.append(f"  FRACTIONAL IO: {fi['lambda']:.4f} * IO({fi['node_u']}) "
                         f"+ {1-fi['lambda']:.4f} * IO({fi['node_v']})")

        # Pool state after
        post = step["post_pool"]
        lines.append(f"  Pool after: size={post['pool_size']}  "
                     f"best_bound={post['best_bound']}")
        if step["pool_improved"]:
            lines.append(f"  ** BOUND IMPROVED to {post['best_bound']}")

    lines.append("")
    lines.append("-" * 70)
    lines.append("TERMINAL INEQUALITY VERIFICATION")
    if sm["terminal_detail"]:
        td = sm["terminal_detail"]
        lines.append(f"  {td['repr']}")
        lines.append(f"  YI_coeff  = {td['yi_coeff']}")
        lines.append(f"  Edge coefficients:")
        for e, c in td["edge_coeffs"].items():
            lines.append(f"    {e} : {c:.4f}")
        lines.append(f"  Edge total = {td['edge_total']}")
        lines.append("")
        lines.append(f"RATE EXTRACTION")
        n_sess = len(best_ep["sessions"])
        yi_c   = td["yi_coeff"]
        e_tot  = td["edge_total"]
        # Account for YI_Pi internal session terms
        int_add = sum(
            td["yi_pi_coeffs"].get(f"P{i}", 0.0)
            * sum(1 for s, t in best_ep["sessions"]
                  if all(n in best_ep["partition"][i]
                         for n in [s, t]))
            for i in range(len(best_ep["partition"]))
        )
        denom = yi_c * n_sess + int_add
        lines.append(f"  h(Y_I) = {n_sess}r * log2(b)")
        lines.append(f"  {yi_c} * {n_sess}r <= {e_tot}")
        if abs(int_add) > 1e-9:
            lines.append(f"  (+ internal session contribution: {int_add:.4f})")
            lines.append(f"  {denom:.4f} * r <= {e_tot}")
        lines.append(f"  r <= {e_tot} / {denom:.4f} = {e_tot/denom:.6f}")
        lines.append(f"  r <= {sm['best_bound']:.6f}  ✓")

    lines.append("=" * 70)

    doc = "\n".join(lines)
    if out_path:
        with open(out_path, "w") as f:
            f.write(doc)
        print(f"[Stage4ProofLogger] Proof document written to {out_path}")
    return doc