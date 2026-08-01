# S1 Report — Frozen-Model Branch-Carry Loop: Built, Validated, and Reachability-Closed

**Date:** 2026-06-17 · **Phase:** M+N / S1 (instrumented baseline of the real branch loop) ·
**Model:** Ouro-RLTT local (`models/ouro_rltt_local`, 48 layers × 4 UT loops, hidden 2048, `total_ut_steps=4`, `early_exit_threshold=1.0`) · **Hardware:** single GPU, ~12 GiB usable.

---

## 0. Executive summary

S1 set out to build the never-before-assembled in-transformer basal-ganglia branch loop
(inject → carry-KV → prune → loop-back → terminal) on the **frozen** Ouro-RLTT model, validate its
mechanism, and measure whether frozen-model branching can reach correct answers the base model misses.

**Key positive result:** S1 converts the branch-carry idea from an architectural sketch into a
**validated measurement harness** — a correct, gate-verified instrument we can now use to test whether
*training* makes branching useful. The capability result on the frozen model is negative, but the
instrument is real and reusable.

**Three results:**

1. **The mechanism is built and bit-exact correct.** Five gates pass (re-derive plumbing, α>0
   multi-locus chaining, live tap-prune integration, and two structural lineage invariants). The full
   12-locus reference loop runs end-to-end with **zero correctness loss**.

2. **Frozen branching is reachability-neutral — LOCALLY CLOSED under tested regimes.** Under the tested
   α/token_range/decode regimes (α ∈ {.02,.05,.10} × token_range ∈ {last, last-8, second-half} × loop-1
   loci + a loop-4 sentinel, plus the chained α=0.02 loop), frozen deterministic injection/carry does
   not beat sampling: the greedy (deterministic) fork yields **0.0** new-correct everywhere, and a
   K-matched plain-sampling baseline **exceeds** fork+sampling (0.75 vs 0.611 oracle). The apparent
   "wins" were stochastic decoding, not the injection mechanism. *This is a local decision verdict — not
   a claim that no frozen branch regime can ever work.*

3. **The existing content tap is not a reliable winner selector for generated branch candidates.** When
   branching/sampling produces new correct candidates on base-missed tasks, the current CoreContent-style
   selector does not reliably rank them first. (Tap macro ≈ 0.67 globally — not useless; it is simply not
   adequate as a top-1 selector over *these generated branch candidates* in this setup.) This is a
   distinct wall from reachability.

**Implication:** the two levers map exactly onto S3's two jobs — **M: make internal branches
outcome-distinct and useful**, and **N: make the model/selector choose the useful branch**. Local frozen
*perturbation games* are done; S3 is a training-time integration test, not stronger perturbations.

---

## 1. What was built and validated (the substrate)

The branch loop reuses two validated primitives: the **KV branch-carry + suffix-recompute splice**
(`build_spliced_branch`, bit-exact, compute-saving for K≥2) and the **hidden-state taps** (DualAnchor =
validity/survival; CoreContent_v2 = content selection — both antisymmetric pairwise:
`layernorm(stateᵢ−stateⱼ)·w`). On top of these, S1 built and gated the per-survivor branch-specific
re-derivation needed for a correct multi-locus loop.

| Gate | What it proves | Result |
|---|---|---|
| **1** — α=0 re-derive plumbing | `branch_specific_root_pack` (root prefill with lineage hooks active) + fork + continue is identity-correct | **bit-exact** — first-logit maxabs 0.0, tokens identical 48/48 |
| **2a** — single-locus α>0 | splice fork (`build_spliced_branch`) ≡ hook replay (`full_perturbed_prefill`) at the last-token range | **bit-exact** — maxabs 0.0, 48/48 |
| **2b** — two-locus chaining α>0 | re-derive through an α>0 ancestor (`LayerOutputPerturbHook(last-token)`) → capture next boundary → splice ≡ all-hooks-live | **bit-exact** — maxabs 0.0, 48/48 |
| **3** — live prune integration | branch text → `[3,4,2048]` pooled tap features → DualAnchor (6 ch) + CoreContent_v2 (3 ch) tournament scores, finite & varying | **pass** — gibberish branch correctly bottomed (CoreContent −0.81 vs +0.30/+0.12) |
| **4** — no-original-root-reuse | a survivor's full lineage is always re-derived (never a clean root after divergence) | **held** across 12 loci × 4 tasks (no AssertionError) |
| **5** — lineage/splice-stack sanity | ancestor loci strictly precede the fork locus; appended entry == fork; full lineage strictly increasing | **held** across 12 loci × 4 tasks |

**Fork mechanism (locked):** single primitive `apply_boundary_perturbation` /
`LayerOutputPerturbHook(token_range)`; canonical reference `token_range` = last live token
`(seq_len−1, seq_len)`; causal-suffix-safe only; lineage replay reproduces the exact perturbation that
created the branch (token_range stored per lineage entry).

**Prune is the trained pairwise tap on branch *text*** (a teacher/scaffold), not live-KV-trajectory
features: each fork decodes a candidate continuation, which is re-encoded through
`BGTransformerFeatureExtractor` and scored. Loci 24/36/47 are injection/divergence boundaries.

---

## 2. The reference loop (S1.4)

**Config:** 4 tasks (1/domain: math, reasoning, logic, coding), K=2, BUDGET=4, α=0.02, last-token,
greedy decode, rank-only prune (DualAnchor diagnostic → CoreContent rank → budget), answer-only scoring,
single shared eager model, ~37 min. Full inject→carry→prune→loop-back→terminal, 12 loci, terminal L4_47.

**Result — correctness-preserving, zero loss, no gain:**

```
oracle_over_survivors 0.25  ==  base_acc 0.25  ==  selected_acc 0.25
mean terminal survivors 4.0 · defer 0
```

`selected_acc = 0.25` here matches base because the *only* correct task (math) was already base-correct
**and** its branches reconverged so every survivor was correct — selection was trivial, not a
demonstration that the selector works. On the 3 base-wrong tasks the diverged branches never reached a
new correct answer; on math branches diverged mid-loop then **reconverged** to the correct final. Branch
divergence at α=0.02 is real but task-dependent:

| task | loci diverged | terminal cc spread | base→survivors |
|---|---|---|---|
| logic | 0/12 (full collapse) | 0.0 | wrong → wrong |
| math | 10/12 (reconverged) | 0.0 | right → right |
| reasoning | 9/12 | 2.31 | wrong → wrong |
| coding | 11/12 | 0.81 | wrong → wrong |

The loop is a **working measurement instrument**; it does not, on the frozen model, exceed base.

---

## 3. Fork-parameter screen (S1.4a)

Bounded **single-locus fork probe** (NOT the chained loop): for each cell, fork each of loop-1's three
loci (L24/L36/L47) *independently from the clean root*, K=4 branches, MNT=96, **prompt+answer scoring**,
+ a loop-4 sentinel (α=0.10, second-half, sample at L4_36/L4_47). 18 cells, ~183 min.

**Decisive pattern — the fork adds nothing; decode does everything:**

| decode | new_correct@base_missed | loci_diverged | reconvergence |
|---|---|---|---|
| **greedy** (all 9 cells, every α/token_range) | **0.0** | 0.25–0.5 | 0.29–0.5 |
| **sample** (all 9 cells) | **0.5–0.75** | 1.0 | 0.0 |
| sentinel (loop-4, sample) | 0.5 | 1.0 | 0.0 |

Sample-cell results are **~identical regardless of α or token_range** — the sampling RNG dominates, the
fork parameters do not move the needle. `selected_acc ≈ 0` everywhere (one cell 0.25).

**Two confounds flagged at the time:** (i) the apparent reachability is *sampling*, not the fork
(greedy = 0 everywhere); (ii) MNT=96 truncated math's base before "FINAL ANSWER", falsely marking it
base-missed (it is base-correct at MNT=160). Per protocol, both required the K-matched baseline.

---

## 4. K-matched sampling baseline (S1.4b)

**Mandated confound check.** No fork; N=12 plain samples/task (= 3 loci × K=4, matching the fork arm's
branch budget); temp 0.7 / top_p 0.95; MNT=96; base decoded greedy at **both** MNT=160 and MNT=96 to
remove the truncation confound. ~9 min.

```
base_acc           0.25 @MNT160   (math correct)      0.0 @MNT96 (math truncated)
plain sampling oracle@12           0.75
  new-correct vs true base (160)   0.667   (logic, reasoning reached; coding not)
  new-correct vs base (96)         0.75
plain sampling selected_acc        0.0
mean unique finals 11.25 · parse-invalid 0.0
```

| arm | new-correct on base-missed | oracle |
|---|---|---|
| greedy fork (deterministic injection) | 0.0 | 0.0 |
| sample fork (fork + sampling) | 0.611 | 0.611 |
| **plain sampling (no fork, K-matched)** | — | **0.75** |

`sample_fork − plain_sampling = −0.139`: **plain sampling exceeds fork+sampling.** The fork is
neutral-to-slightly-negative. Combined with greedy-fork = 0, the verdict is unambiguous:

> **FROZEN_FORK_CLOSED_LOCAL — under tested α/token_range/decode regimes, frozen injection/carry does
> not beat sampling — S3 is the lever.**

(Per-task: logic oracle@12=1 [2 correct], math=1 [1], reasoning=1 [5], coding=0. Note `selected_acc=0.0`
here: where sampling produced a correct candidate on a base-missed task, the content tap did not rank it
first.)

*Note:* the script's auto-verdict initially mis-fired (`abs(diff)≤TOL` treated a negative diff as
"escalate"); a negative diff (sampling ≥ fork) is pro-*closed*. Logic fixed; artifact decision recomputed.

---

## 5. Consolidated findings — two cleanly separated walls

These map directly onto S3's two distinct jobs (M and N) and must not collapse into one vague
"branching failed" story.

**Wall A — generation / reachability (→ S3-M).** Deterministic frozen injection/carry does not create
new correct branches (greedy fork = 0.0 new-correct across the chained α=0.02 loop and every tested
single-locus regime: α ≤ 0.10 × {last, last-8, second-half} × loop-1 + loop-4 sentinel). Stochastic
decoding (plain sampling) *does* create some new correct candidates (logic, reasoning), but **fork +
sampling does not beat K-matched plain sampling** (0.611 vs 0.75 oracle) — the carry adds nothing on the
frozen model. Consistent with S1.2 (injection ≤ sampling), now confirmed by the full chained loop and a
matched baseline. **Scoped:** local, under tested regimes.

**Wall B — selection / conversion (→ S3-N).** Even when correct candidates exist in the pool, the
current CoreContent-style selector does not reliably rank them first. In the reference loop
`selected_acc=0.25` only because the lone correct task was base-correct and reconverged (trivial
selection); in the sampling/fork regimes where new correct candidates appeared on base-missed tasks,
`selected_acc=0.0` — the selector failed to convert them. So the claim is precise: **the existing content
tap is not a reliable winner selector for generated branch candidates** (not "useless" — its macro is
≈0.67 — but inadequate as top-1 over these candidates in this setup). The survivor-set handoff (locked
safe stance) still preserves a correct answer *if present*; forced top-1 by the current tap does not.

**What is NOT in question:** the loop's correctness. Gates 1–5 hold; the reference loop is zero-loss. The
mechanism is sound — and the larger positive result is that **S1 delivers a validated measurement harness**
for branch-carry, ready to test whether training (S3) makes branching outcome-distinct (M) and
selector-readable (N).

---

## 6. Caveats and limits

- **N = 4 (1/domain)** — a shakedown, not a reachability study. 0.25 = one task. No statistical claims.
- **Scoring format** — S1.4 reference used answer-only; S1.4a/b used prompt+answer (the dominant tap
  training format). Cross-domain numbers should standardize on prompt+answer.
- **Single-locus scope (S1.4a/b)** — a single strong fork tests local perturbation magnitude, not
  accumulated branch history or later-loop semantic maturity. The null is scoped to *single-locus*
  regimes; the loop-4 sentinel partially probes later-loop maturity (also null beyond sampling).
  Combined with the chained α=0.02 null, this is strong enough to stop **local** spend — it is **not** a
  proof that no chained regime anywhere could differ.
- **Hardware** — ~12 GiB usable forced batched sampling and the expensive per-survivor re-derive
  (~9 min/task). Scaling N or chained ablations needs the phase-two optimized carried-KV path and/or
  cloud GPUs.
- **Taps** — frozen CoreContent_v2 macro ≈ 0.67; not an oracle selector by construction.

---

## 7. Implications — S3 is a training-time integration test (not stronger perturbations)

S1 already says local frozen *perturbation games* are done. S3 trains the backbone / loop dynamics so
that branching is a **learned operation**, not an accidental hidden-state injury. Its two jobs map onto
the two walls:

- **S3-M — outcome-distinct branch training (Wall A).** Train so an injected internal branch lands on a
  *distinct, reachable* trajectory — something sampling doesn't already provide and the frozen carry
  can't. Requires external/cloud GPUs; weight backup verified (S0).

- **S3-N — branch-correctness selector training (Wall B).** Train a selector that puts a correct branch
  on top of the candidate pool: refit/retrain on the loop's own branch-text distribution, prompt+answer
  consistent, toward a relational-preference objective aligned to verifier outcomes. Without it, even an
  S3-M reachability gain won't convert under forced top-1 (survivor-set handoff stays the safe object).

**Do not** escalate to the chained-loop-at-winner run: the fork does not exceed sampling, so there is no
"winning regime" whose carry is worth the ~37-min/cell chained cost.

---

## 8. Artifact index

All under `artifacts/reports/probes/mpn_s1_baseline_2026-06-13/`:

| File | Contents |
|---|---|
| `s1_4_gate1_rederive.json` | Gate 1 (α=0 re-derive) — bit-exact |
| `s1_4_gate2_alpha_chain.json` | Gate 2a/2b (α>0 chaining) — bit-exact |
| `s1_4_gate3_prune_integration.json` | Gate 3 (live tap prune) — pass |
| `s1_4_reference_loop.json` | Reference loop (gates 4+5 held; 0.25 zero-loss) |
| `s1_4a_fork_param_screen.json` | 18-cell fork screen + loop-4 sentinel |
| `s1_4b_kmatched_sampling.json` | K-matched plain-sampling baseline — FROZEN_FORK_CLOSED |
| `s1_report_2026-06-17.md` | this report |

Scripts: `utilities/tests/manual/mpn_s1_4_rederive_gate.py`, `mpn_s1_4_gate2_alpha_chain.py`,
`mpn_s1_4_gate3_prune_integration.py`, `mpn_s1_4_reference_loop.py`, `mpn_s1_5_divergence_ablation.py`,
`mpn_s1_4b_kmatched_sampling.py`. Logs under `artifacts/logs/mpn_s1/`.

---

## 9. Recommended next steps

1. **Close S1 as the frozen-baseline phase** (done: mechanism validated, frozen branching locally
   reachability-closed under tested regimes, both walls characterized, measurement harness delivered).
2. **S3 = training-time integration test.** Question: *can branch/carry become useful when the model is
   trained so injected internal branches become outcome-distinct and selector-readable?* Three threads:
   - **S3A — outcome-distinct branch training** (Wall A / M).
   - **S3B — branch-correctness selector training** (Wall B / N).
   - **S3C — integrated branch/carry loop eval** (reuse the S1 harness as the instrument).
3. **Before any backbone training:** re-confirm the verified weight backup (S0) and the hard-constraint
   ledger (backbone training conditionally permitted only with verified backup).
