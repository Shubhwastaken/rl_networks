"""
Phase 2 re-derivation driver.

Loads the EXISTING trained Stage-4 checkpoint (no training, no finetune),
fully seeds the RNG, and re-runs evaluation on all 19 graphs under the
corrected proof calculus.

SUB_LP_FATAL is set False for this run so a single sub-LP result records
and continues instead of aborting all 19 graphs. Nothing is clamped
either way -- the real bound propagates and every violation lands in
SUB_LP_OBSERVED for the report. Assertion trips (mediant / RHS-unchanged)
are caught per graph and recorded, never swallowed.
"""
import os, sys, json, random, traceback
_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(_HERE))   # repo root: eval writes config_files/ relative
sys.path.insert(0, _HERE)

import numpy as np
import torch

import fixed_training as FT
import functional_dependence as FD

# Record-and-continue so one bad graph cannot hide the other eighteen.
FT.SUB_LP_FATAL = False

SEED = FT.SEED
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from gnn_policy import GNNPhase1Policy, GNNPhase2Policy, GNNPhase3Policy

DEVICE = FT.DEVICE
M = "model_files"

coeff_dim = torch.load(f"{M}/ckpt_stage1_coeff_dim.pt", weights_only=True,
                       map_location=DEVICE)["coeff_dim"]

p1 = GNNPhase1Policy()
p1.net.load_state_dict(torch.load(f"{M}/ckpt_stage3_phase1.pt",
                                  weights_only=True, map_location=DEVICE))
p2 = GNNPhase2Policy(coeff_dim=coeff_dim)
p2.net.load_state_dict(torch.load(f"{M}/ckpt_stage3_phase2.pt",
                                  weights_only=True, map_location=DEVICE))
p3 = GNNPhase3Policy(coeff_dim=coeff_dim, total_episodes=30000)
p3.net.load_state_dict(torch.load(f"{M}/ckpt_stage4_phase3.pt",
                                  weights_only=True, map_location=DEVICE))

with open(f"{M}/ckpt_stage3_best_partitions.json") as f:
    raw = json.load(f)
best_partitions = {k: ([tuple(p) for p in v["partition"]], v["weights"], v["bound"])
                   for k, v in raw.items()}

print("=" * 78)
print("PHASE 2 RE-DERIVATION -- existing checkpoint, no training")
print(f"  fingerprint: {FT._policy_fingerprint(p1, p2, p3)}")
print(f"  SOUNDNESS_MODE = {FD.SOUNDNESS_MODE}   SUB_LP_FATAL = {FT.SUB_LP_FATAL}")
print("=" * 78)

ASSERTION_TRIPS = []
_orig_run = FT._run_eval_episode


def _guarded(env, graph_tuple, *a, **kw):
    """Catch soundness assertions per episode and record them."""
    try:
        return _orig_run(env, graph_tuple, *a, **kw)
    except FD.UnsoundDerivationError as e:
        ASSERTION_TRIPS.append({
            "graph": FT.identify_graph(*graph_tuple),
            "type": type(e).__name__,
            "message": str(e),
        })
        return None


FT._run_eval_episode = _guarded

results = FT.evaluate(p1, p2, p3, best_partitions=best_partitions,
                      graph_dataset_size=19,
                      stochastic_episodes=30, greedy_trials=8)

out = {
    "assertion_trips": ASSERTION_TRIPS,
    "sub_lp_observed": [list(x) for x in FT.SUB_LP_OBSERVED],
}
with open("config_files/_rederive_diagnostics.json", "w") as f:
    json.dump(out, f, indent=1)

print(f"\nassertion trips : {len(ASSERTION_TRIPS)}")
print(f"sub-LP observed : {len(FT.SUB_LP_OBSERVED)}")
print("diagnostics -> config_files/_rederive_diagnostics.json")
