# Archive — old-calculus audit tooling

Archived 2026-08-05. **Nothing here is dead code that was abandoned.** These
scripts are the audit trail for the soundness review of the pre-fix proof
calculus. They located five independent soundness failures, and the numbers
quoted in the review report come from these files. They are kept as evidence
and as fallback.

They are archived rather than deleted because the project has switched from
the aggregated `EntropyIndex` representation to a chain-based derivation over
joint entropy terms. **These scripts will not run against the new
representation** — see "What they run against" below.

---

## Why these exist: the five bugs

The review asked whether the bounds the RL system had been reporting were
provable. They were not. Five independent defects were found, all of the same
error class: *an operation that combines two inequalities must ADD right-hand-side
capacity, never re-use it, delete it, min it, max it, or overwrite it.*

| # | Defect | Where | Effect |
|---|---|---|---|
| 1 | crypto RHS accounting | `apply_crypto_inequality_direct` | incremented `Y_I`, never touched edge coefficients |
| 2 | decode RHS accounting | `apply_decode_substitution` | same shape as #1 |
| 3 | source-zeroing collapse | `_collapse_to_yi_if_valid` | deleted source terms without paying for them |
| 4 | max/min union | `apply_pairwise_submodularity` | counted shared RHS capacity once instead of twice |
| 5 | intersection | `apply_pairwise_submodularity` | took `max` of two negative RHS coefficients, i.e. shrank capacity |

---

## Which file produced which finding

### `_taskA.py` — first invalid inequality, live
Replays live episodes and, after every step, validity-checks every pool /
accumulator / stored item against an achievable entropy vector. Reports the
**first** step at which a false inequality appears, with the producing action.

- **diamond_6N**: first false item at step 38, action `APPLY_SUBMODULARITY`,
  violation **+0.330**, inequality `0.500*Y_I <= 0.670*U_S2_v2 + 1.000*U_v1_v2`.
- **okamura_4N**: step 23, `APPLY_SUBMODULARITY`, violation **+0.670**.
- **ford_fulkerson_6N**: step 32, `CROSS_SUBMOD`, violation **+1.250**.

This is what pointed the investigation at `apply_pairwise_submodularity`
rather than at crypto/decode, which had been the initial suspicion.

### `_taskA2.py` — bisecting which sub-step flips a union false
Rebuilds each pairwise-submodularity union in three stages and reports which
stage first makes a valid pair produce a false result:

- stage 1, max/min union construction
- stage 2, `_collapse_to_yi_if_valid` (Y_ST → Y_I **plus** source zeroing)
- stage 3, `_cancel_sources_for_node_ios`

**Isolated defect #3 to `_collapse_to_yi_if_valid`.** On diamond_6N the two
inputs were individually valid (slack +2.0 and +1.5), stage 1 preserved
validity (slack +1.5), and stage 2 produced **violation +0.5**. Splitting
stage 2 further showed the collapse WITHOUT source zeroing kept slack +1.5,
so the source deletion alone was responsible.

The quantified mechanism, from a sweep in the same file:

```
Y_I mass contributed from outside the Y_ST pathway:  0.000  0.125  0.250  0.500  1.000
violation after the collapse:                        0.000  0.250  0.500  1.000  2.000
```

i.e. **violation = h(Y_I) × (Y_I mass from outside the Y_ST pathway)** exactly.
The `c_min` the collapse added paid only for removing `Y_ST`; nothing paid for
the source deletion.

### `_taskA3.py` — producer distribution and cancellation unit tests
Two results.

1. **Producer attribution.** Across 120 episodes on three graphs, the action
   producing the first false inequality was `APPLY_SUBMODULARITY` 14 times and
   `CROSS_SUBMOD` 11 times — **100% through `apply_pairwise_submodularity`**,
   0% from any other action.

2. **The 177/4000 count.** Randomised unit test over 4000 *valid* inequalities
   carrying a `Y_ST` term:
   - `Inequality.cancel_source_terms` — **0** valid inputs turned false (sound;
     it removes equal mass from both sides)
   - `_collapse_to_yi_if_valid` — **177** valid inputs turned false

   A later sweep with a different sampling distribution gave **91/4000** for
   the same function; both are non-zero and either refutes the claim that the
   collapse was a theorem. The corrected collapse (`K − D`) gives **0/4000**
   with slack change zero to machine precision.

### `_taskBC.py` — survivor audit and independent numeric verification
Per-step audit of the six results that survived the first round of fixes,
plus numeric verification of each terminal at achievable points.

- **stage-1 union isolation test**: **0 of 5756** valid input pairs turned
  false by the max/min union alone — which is why defect #4 was initially
  missed and recorded as "unresolved".
- Verdicts: `diamond_6N` and `kramer_savari_ladder_8N` **CONTAINS THE DEFECT**
  with terminal violations **+1.00** and **+2.00** despite both passing the LP
  floor. `star_8N`, `ford_fulkerson_6N`, `hu_2pairs_6N`, `grid_4x4_16N` showed
  no false intermediate at the single test point.
- `hu_2pairs_6N` was later falsified (**+1.00**) once multiple optimal flow
  profiles were used — a single test point was not enough.

### The 12,665 / 36,253 intersection count
Produced by the combining-operation audit (driver code lives in the review
transcript; it reuses `_entropy_check` and `_flowpoints`, both of which stayed
at top level). Over **36,253** valid inequality pairs drawn from the real
generator families across 17 graphs:

| operation | false results |
|---|---|
| max/min union (defect #4) | **66** |
| **intersection (defect #5)** | **12,665 (35%)** |
| plain addition `A+B` | **0** |
| max-LHS + sum-RHS | **0** |
| `apply_n2_submodularity_all_at_once` | 0 |

The intersection was by far the worst and was not inert: every result went
straight into the pool via `pool.append(inter_ineq)` in both Phase 2 and
Phase 3. Minimal counterexample, two ordinary node IOs with disjoint RHS
support so the whole RHS vanishes:

```
A = IO(S1) : 0.5*Y_I <= Y_S_S1 + U_S1_v1 + U_S1_v2
B = IO(S2) : 0.5*Y_I <= Y_S_S2 + U_S2_v1 + U_S2_v2
INTER      : 0.5*Y_I <= 0                              violation +1.0
```

After the fixes, the replacement (`A.add(B)` + corrected collapse) produced
**0 false results out of 47,065** valid pairs.

### `_replay.py` — what the fixes cost
Replays all 59 retained novel episodes through the corrected environment.
The decisive separation:

- **33 episodes** whose winning terminal used CRYPTO/DECODE → **33 loosened,
  0 improved**
- **16 episodes** whose winning terminal used neither → **reproduced the
  claimed bound exactly**

That separation is the strongest evidence the fixes touched precisely the
unsound operators and nothing else. Net effect: **16/19 claimed novel results
dropped to 6**, and after the collapse fix to **2** that were both
base-certified and unfalsified.

### `_rederive.py`, `_report.py`, `_attribute.py`
Full evaluation driver (RNG seeding, policy-fingerprint gating, diagnostics
capture), the report generator, and a mechanism-attribution pass. `_rederive.py`
surfaced **7 sub-LP violations** that the pre-fix code reported as
`lp_violations: 0`, because the LP clamp ran before the violation counter and
made that counter a tautology.

### `_expsearch.py`, `_expsearch2.py` — gate experiments
Policy-free random search under three terminal-gate configurations, to test
whether `MIN_YI_COEFF` was the binding constraint. It was not:

- Removing the floor changed the best bound on **0 of 19** graphs and unblocked
  **0 of the 11** graphs it was supposed to be blocking.
- The rejection tally on diamond_6N over 60 episodes was
  `leftover_source_neg` **12,716**, `yi_nonpositive` **10,919**,
  `leftover_yst` **56** — the Y_I floor never appears.
- The real blocker is source cancellation, which needs every session source
  node present and is only reachable through submodularity.

---

## What they run against

All of these import `fixed_inequality.EntropyIndex` and operate on its **fixed
flat coefficient vector**: one slot each for `Y_ST_Pk`, `Y_I_Pk`, `Y_S_v`,
`U_e`, plus a single aggregate `Y_I`, and **one variable per undirected edge**.

The chain-based approach replaces this with joint terms `h(Y_A, U_S)` over a
ground set of sessions plus **directed** arcs (two per undirected edge). There
is no way to name an arbitrary joint term in the old layout, so **these scripts
cannot be ported by editing — they would have to be rewritten.**

Two further reasons they will not run as-is:

1. Their input JSONs moved to `config_files/_archive_old_calculus/`.
2. They import `_entropy_check` / `_flowpoints` by bare name, which resolved
   when they sat in `python_files/`. From this subdirectory the import path
   differs.

Both are deliberate: these are evidence, not live tools.

---

## What is still reusable

**The methodology, which transfers to any representation:**

1. **Build an independent falsification oracle first.** The LP lower bound is
   max symmetric multicommodity flow, so `r = LP_LB` is achievable by routing.
   Routing with independent sources gives a fully explicit entropy vector. Any
   valid inequality must satisfy `Σ c_j h_j ≤ 0` at that point; a strictly
   positive value is a *proof* of falsity, independent of how the inequality
   was derived. (`_entropy_check.py`, kept at top level.)

2. **Test at multiple achievable points, not one.** `hu_2pairs_6N` passed at
   one optimal flow profile and failed at another. (`_flowpoints.py`, kept.)

3. **Locate the first invalid inequality, then bisect which sub-step flips
   it.** Not "which action was running" — decompose the operation into stages
   and find the stage that turns a valid input into a false output. This is
   what separated the collapse defect from the union defect when both were
   firing through the same action.

4. **Sample from the real generator families, never random coefficient
   vectors.** The 0/5756 result that hid defect #4 came from random vectors;
   the counterexample appeared immediately once pairs were drawn from actual
   node IOs, partition IOs, crypto and decode inequalities.

5. **A certificate beats a point check.** A numeric test can only falsify.
   Farkas domination — `c ≤ Σ λ_i A_i` with `λ ≥ 0` — is a genuine proof when
   feasible. Kept and generalised as `python_files/farkas_certify.py`.

**Caveat carried forward:** a certificate is only as strong as its generator
set, and the generator set is only as strong as its representation. The old
generator LP returned a ceiling of 1.0 for K₂,₃ where the published IO+crypto
proof gives 0.75 — not because an inequality was missing, but because the
aggregated variable space could not express the argument. The joint-entropy LP
reaches 0.75000 exactly at cap=1400. Any "no certificate" result means "not
provable from these generators in this representation", never "false".
