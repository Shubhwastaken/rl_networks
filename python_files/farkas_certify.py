"""
FARKAS CERTIFICATE CHECKER.

certify() is the reusable piece and is REPRESENTATION-FREE: it takes a
target coefficient vector c and a matrix of known-valid generator vectors,
and asks whether c is dominated by a nonnegative combination of them.

    Every h_j is an entropy, so h >= 0. Therefore if
        c <= d   componentwise, and d is valid (sum d_j h_j <= 0),
    then c is valid too.

    So c is CERTIFIED valid whenever there exist lam >= 0 with
        c  <=  sum_i lam_i * A_i     componentwise.

    That is an LP feasibility problem, and unlike a numeric point check
    (which can only falsify) a feasible answer is a genuine proof.

=============================================================================
STATUS FOR THE CHAIN-BASED APPROACH (2026-08-05)
=============================================================================

KEEP AND USE:  certify(c, gens)
    Pure linear algebra over whatever coefficient space you hand it.
    Works unchanged against the new (A, S) joint-term representation.

NEEDS REWRITE: generators(...)          <-- see the marker on the function
    Built entirely on the OLD EntropyIndex aggregated vector (~25 slots:
    Y_ST_Pk / Y_I_Pk / Y_S_v / U_e / Y_I). It cannot name a joint term
    h(Y_A, U_S) for arbitrary (A, S), which is exactly what the new
    approach needs. Rebuild it over the new ground set (sessions Y_i plus
    directed arcs U_{u->v}, two per undirected edge) before use.

NEEDS REWRITE: the __main__ block below
    Reads config_files/_survivor_terminals.json and _replay.json, both of
    which now live in config_files/_archive_old_calculus/. It is retained
    only as a worked example of how certify() was called.

The old-calculus caveat that motivated this file still stands: a
certificate is only as strong as its generator set. A "NO CERTIFICATE"
answer means "not provable from THESE generators", never "false".
"""

import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.optimize import linprog
import fixed_graph_generation as G
from fixed_inequality import EntropyIndex, Inequality
from fixed_base_inequality_generator import generate_all_node_ios
from functional_dependence import (_directed_cut_edges, _sessions_separated_by_cut,
                                   _edges_into_sink, best_crypto_cuts)
from _entropy_check import max_symmetric_flow_with_arcs, entropy_vector, violation, is_falsified
from _flowpoints import optimal_flow_profiles

G._build_registry(); BY = {g.name: g for g in G.GRAPH_REGISTRY}


def generators(index, nodes, edges, sessions, partition):
    """
    !!! NEEDS REWRITE FOR THE NEW GROUND SET -- DO NOT USE AS IS !!!

    Built on the OLD EntropyIndex aggregated representation. Every vector
    here is indexed by index.yi_idx() / yst_idx() / source_idx() /
    edge_idx(), i.e. the fixed ~25-slot layout the chain-based approach
    replaces. Retained so the generator FAMILIES are documented (node IOs,
    partition IOs, corrected crypto, corrected decode, subadditivity,
    source-independence, ST-source, INT<=ST) -- rebuild each of these over
    joint terms h(Y_A, U_S) rather than porting the code.

    Known-valid inequality vectors: node IOs + crypto + decode.
    """
    gens, names = [], []
    for n, io in generate_all_node_ios(partition, nodes, edges, sessions, index).items():
        gens.append(io.coeffs.copy()); names.append(f"IO({n})")
    for vp, sep in best_crypto_cuts(nodes, edges, sessions, max_cuts=60):
        cut = _directed_cut_edges(set(vp), nodes, edges)
        if not cut or not sep: continue
        v = np.zeros(index.dim)
        v[index.yi_idx()] = len(sep) / len(sessions)
        for e in cut:
            k = f"U_{e[0]}_{e[1]}"
            if k not in index.var_to_idx: k = f"U_{e[1]}_{e[0]}"
            if k in index.var_to_idx: v[index.var_to_idx[k]] -= 1.0
        gens.append(v); names.append(f"CRYPTO({sorted(vp)})")
    for si in range(len(sessions)):
        inc = _edges_into_sink(si, sessions, edges)
        if not inc: continue
        v = np.zeros(index.dim)
        v[index.yi_idx()] = 1.0 / len(sessions)
        for e in inc:
            k = f"U_{e[0]}_{e[1]}"
            if k not in index.var_to_idx: k = f"U_{e[1]}_{e[0]}"
            if k in index.var_to_idx: v[index.var_to_idx[k]] -= 1.0
        gens.append(v); names.append(f"DECODE({si})")
    # partition-level base IOs (the Y_ST pathway the collapse uses)
    from fixed_base_inequality_generator import generate_base_inequalities
    for i, b in enumerate(generate_base_inequalities(partition, nodes, edges, sessions, index)):
        gens.append(b.coeffs.copy()); names.append(f"PIO(P{i})")
    # subadditivity: h(Y_I) <= sum_k h(Y_ST_Pk) when the union covers all sessions
    v = np.zeros(index.dim); v[index.yi_idx()] = 1.0
    for i in range(len(index.partitions)): v[index.yst_idx(i)] = -1.0
    gens.append(v); names.append("SUBADD(Y_I<=sum Y_ST)")
    # source independence, both directions: h(Y_I) == sum_v h(Y_S_v)
    v = np.zeros(index.dim); v[index.yi_idx()] = 1.0
    for nd in nodes: v[index.source_idx(nd)] = -1.0
    gens.append(v); names.append("SRC_IND(Y_I<=sum Y_S)")
    v = np.zeros(index.dim); v[index.yi_idx()] = -1.0
    for nd in nodes: v[index.source_idx(nd)] = 1.0
    gens.append(v); names.append("SRC_IND(sum Y_S<=Y_I)")
    # Y_ST_Pk vs its own sources, both directions; and Y_I_Pk <= Y_ST_Pk
    for i in range(len(index.partitions)):
        srcs = {sessions[si][0] for si in index.st_sessions[i]}
        v = np.zeros(index.dim); v[index.yst_idx(i)] = 1.0
        for nd in srcs: v[index.source_idx(nd)] -= 1.0
        gens.append(v); names.append(f"ST_SRC(P{i}) <=")
        v = np.zeros(index.dim); v[index.yst_idx(i)] = -1.0
        for nd in srcs: v[index.source_idx(nd)] += 1.0
        gens.append(v); names.append(f"ST_SRC(P{i}) >=")
        v = np.zeros(index.dim); v[index.yi_pi_idx(i)] = 1.0; v[index.yst_idx(i)] = -1.0
        gens.append(v); names.append(f"INT<=ST(P{i})")
    return np.array(gens), names


def certify(c, gens):
    """Feasible lam >= 0 with c <= gens^T lam  (componentwise)?  -> valid."""
    m = gens.shape[0]
    # constraint: c_j - sum_i lam_i gens[i,j] <= 0  for every j
    A_ub = -gens.T                      # (dim, m)
    b_ub = -c
    res = linprog(np.zeros(m), A_ub=A_ub, b_ub=b_ub,
                  bounds=[(0, None)] * m, method="highs")
    return res.success, (res.x if res.success else None)


if __name__ == "__main__":
    # DEMO ONLY. These inputs now live in
    # config_files/_archive_old_calculus/ -- this block is retained as a
    # worked example of how certify() was called, not as a live entry point.
    rep = json.load(open("config_files/_replay.json"))
    log = json.load(open("config_files/stage4_proof_log.json"))
    novel = [e for e in log if isinstance(e, dict) and e.get("summary", {}).get("is_novel")]
    TERMS = json.load(open("config_files/_survivor_terminals.json"))

    print("=" * 104)
    print("TASK C (strengthened): terminal tested at MULTIPLE achievable points, plus Farkas certificate")
    print("=" * 104)
    print(f"{'graph':<26}{'bound':>9}{'r':>6}{'pts':>5}{'max sum(c*h)':>14}{'numeric':>12}  {'certificate':>12}")
    for name, rec in TERMS.items():
        g = BY[name]
        nodes = rec["nodes"]; edges = [tuple(e) for e in rec["edges"]]
        sessions = [tuple(s) for s in rec["sessions"]]; part = [list(p) for p in rec["partition"]]
        ix = EntropyIndex(part, nodes, edges, sessions)
        c = np.array(rec["coeffs"])
        q = Inequality(ix); q.coeffs = c.copy()
        profiles = optimal_flow_profiles(nodes, edges, sessions, n=12)
        worst = -1e18
        for r, flow in profiles:
            h = entropy_vector(ix, nodes, edges, sessions, r, flow)
            worst = max(worst, float(np.dot(c, h)))
        gens, gnames = generators(ix, nodes, edges, sessions, part)
        ok, lam = certify(c, gens)
        num = "FALSE" if worst > 1e-7 else "not falsified"
        cert = "CERTIFIED" if ok else "NO CERTIFICATE"
        print(f"{name:<26}{rec['bound']:>9.4f}{profiles[0][0]:>6.2f}{len(profiles):>5}"
              f"{worst:>14.4f}{num:>12}  {cert:>12}")
        if ok:
            used = [(gnames[i], lam[i]) for i in range(len(gnames)) if lam[i] > 1e-7]
            print("      certificate: " + ", ".join(f"{n}x{v:.3f}" for n, v in used[:8]))
