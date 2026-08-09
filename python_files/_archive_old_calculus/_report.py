"""Final Phase 2 + Phase 3 report."""
import json, os, sys
from collections import defaultdict
_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(_HERE))

cp = {d["name"]: d for d in json.load(open("config_files/_cpsat_pb.json"))}
sys.path.insert(0, _HERE)
from lp_lower_bound import compute_lp_lower_bound
import fixed_graph_generation as _G
_G._build_registry()
LP = {g.name: compute_lp_lower_bound(g.nodes, g.edges, g.sessions)
      for g in _G.GRAPH_REGISTRY}
rep = json.load(open("config_files/_replay.json"))
ev = json.load(open("config_files/eval_results.json"))
diag = json.load(open("config_files/_rederive_diagnostics.json")) if \
    os.path.exists("config_files/_rederive_diagnostics.json") else \
    {"assertion_trips": [], "sub_lp_observed": []}
m = json.load(open("config_files/training_metrics.json"))
s4 = m.get("stage4", m)

OLD = {"paper_7N":1.039474,"diamond_6N":1.000000,"butterfly_8N":1.090909,
 "grid_9N":1.714286,"star_8N":1.181818,"hu_3pairs_6N":1.000000,"okamura_4N":1.055675,
 "ford_fulkerson_6N":2.333333,"grid_3x4_12N":1.321997,"petersen_10N":1.260986,
 "two_k4_10N":0.708824,"al_bashabsheh_7N":1.891837,"hu_2pairs_6N":1.000000,
 "grid_4x4_16N":1.012658,"okamura_seymour_8N":None,"kramer_savari_ladder_8N":1.000000,
 "okamura_network_paper_5N":None,"hu_three_session_6N":1.180370,"yin_et_al_7N":None}
OLDPB = {"paper_7N":1.6667,"diamond_6N":1.75,"butterfly_8N":1.5,"grid_9N":2.0,
 "star_8N":2.25,"hu_3pairs_6N":1.3333,"okamura_4N":1.25,"ford_fulkerson_6N":5.0,
 "grid_3x4_12N":2.8333,"petersen_10N":1.875,"two_k4_10N":2.6667,"al_bashabsheh_7N":2.0,
 "hu_2pairs_6N":2.0,"grid_4x4_16N":3.0,"okamura_seymour_8N":3.0,
 "kramer_savari_ladder_8N":2.0,"okamura_network_paper_5N":0.75,
 "hu_three_session_6N":1.3333,"yin_et_al_7N":1.6667}
order = list(OLDPB)

best = {}
for g, recs in rep.items():
    ok = [r for r in recs if r["replayed"] is not None]
    best[g] = min(ok, key=lambda r: r["replayed"]) if ok else None

# policy vs oracle
gn, of, nf = s4.get("graph_names"), s4.get("oracle_fired"), s4.get("novel_found")
attr = defaultdict(lambda: [0, 0])
for i, g in enumerate(gn or []):
    if nf[i]:
        attr[g][0] += 1
        if of[i]: attr[g][1] += 1

W = 128
print("=" * W)
print("PHASE 2 — RE-DERIVATION UNDER THE CORRECTED PROOF CALCULUS")
print(f"eval fingerprint {ev.get('policy_fingerprint','ABSENT')}   complete={ev.get('complete')}   "
      f"graphs={len(ev.get('summary',{}))}/19")
print("=" * W)
print(f"{'graph':<26}{'old bound':>11}{'new bound':>11}{'PB(CP-SAT)':>12}{'LP LB':>8}"
      f"{'beats PB':>10}  producing mechanism")
print("-" * W)
nbeat = 0
lost, kept = [], []
for g in order:
    pb = cp[g]["pb"]
    ob = OLD[g]
    s = ev.get("summary", {}).get(g, {})
    lp = LP[g]
    b = best.get(g)
    if b is None:
        nb, mech = None, "no surviving derivation"
    else:
        nb = b["replayed"]
        mech = ", ".join(f"{k}x{v}" for k, v in b["op_counts"].items()) or "submod/fractional only"
    obs = f"{ob:.6f}" if ob is not None else "NOT_FOUND "
    if nb is None:
        nbs, bt = "NO_TERMINAL", "no"
    else:
        nbs = f"{nb:.6f}"
        good = nb < pb - 1e-8
        bt = "YES" if good else "no"
        if good: nbeat += 1; kept.append(g)
    if ob is not None and (nb is None or nb >= pb - 1e-8):
        lost.append(g)
    print(f"{g:<26}{obs:>11}{nbs:>11}{pb:>12.4f}{lp:>8.4f}{bt:>10}  {mech}")
print("-" * W)
print(f"still beating PB (corrected CP-SAT PB): {nbeat} / 19   was 16 / 19")
print(f"lost their result                     : {len(lost)}")

print()
print("=" * W); print("(c) ATTRIBUTION — policy assembly vs post-episode oracle"); print("=" * W)
print(f"{'graph':<26}{'novel eps':>10}{'oracle':>8}{'policy':>8}   surviving winner's operation order")
for g in order:
    if g not in attr: continue
    t, o = attr[g]
    b = best.get(g)
    seq = " -> ".join(b["mechanisms"]) if b and b.get("mechanisms") else "(submodularity / fractional IO only)"
    print(f"{g:<26}{t:>10}{o:>8}{t-o:>8}   {seq}")

print()
print("=" * W); print("(d) LOSSES — which operation was responsible"); print("=" * W)
print(f"{'graph':<26}{'claimed':>10}{'corrected':>11}{'PB':>9}  responsible")
for g in lost:
    b = best.get(g)
    if b is None:
        print(f"{g:<26}{OLD[g]:>10.4f}{'NO_TERM':>11}{cp[g]['pb']:>9.4f}  no terminal survives")
        continue
    ops = [k for k in ("CRYPTO", "DECODE") if k in b["op_counts"]]
    t, o = attr.get(g, [0, 0])
    why = ("APPLY_" + "/APPLY_".join(ops)) if ops else "-"
    if not ops and o == t and t > 0:
        why = "post-episode func-dep oracle (all novel eps oracle-produced)"
    elif not ops:
        why = "pool composition changed once crypto/decode stopped inflating"
    print(f"{g:<26}{OLD[g]:>10.4f}{b['replayed']:>11.4f}{cp[g]['pb']:>9.4f}  {why}")

print()
print("=" * W); print("(b) ASSERTION FAILURES"); print("=" * W)
t_, s_ = diag["assertion_trips"], diag["sub_lp_observed"]
print(f"mediant / RHS-unchanged trips during eval : {len(t_)}")
print(f"sub-LP bounds observed during eval        : {len(s_)}")
for t in t_[:10]:
    print(f"  [{t['type']}] {t['graph']}: {t['message'].splitlines()[0]}")
for x in s_[:10]:
    print(f"  [SubLP] {x[0]} {x[1]} ep={x[2]} bound={x[3]:.6f} lp={x[4]:.6f}")
if not t_ and not s_:
    print("  none — every surviving derivation satisfied both assertions and the LP floor.")

print()
print("=" * W); print("PHASE 3 — PARTITION ORACLE"); print("=" * W)
print(f"{'graph':<26}{'old PB':>9}{'CP-SAT PB':>11}{'int':>5}{'parts':>7} {'diff':>5}  optimal partition")
nd = 0
for g in order:
    d = cp[g]; o = OLDPB[g]
    df = abs(d["pb"] - o) > 1e-4
    nd += df
    p = " | ".join("{" + ",".join(map(str, x)) + "}" for x in d["partition"])
    if len(p) > 44: p = p[:41] + "..."
    print(f"{g:<26}{o:>9.4f}{d['pb']:>11.4f}{d['internal']:>5}{len(d['partition']):>7} "
          f"{'***' if df else '':>5}  {p}")
print(f"\ndifferences: {nd} / 19")
