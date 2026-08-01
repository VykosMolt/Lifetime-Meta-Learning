# S3 Closure & Šokac/Sokač Validity-Confirmation Bundle

**Project:** Ouro-RLTT evaluator / hidden-state taps / in-transformer branch-carry loop (M+N program)
**Authored:** 2026-06-22 · **Work dated:** 2026-06-17 (S1 baseline + S3B threads) · **Phase:** S3 closure / external-review readiness
**Model:** Ouro-2.6B-Thinking + RLTT post-train, local (`models/ouro_rltt_local`, 48 layers × 4 UT loops, hidden 2048, `total_ut_steps=4`, `early_exit_threshold=1.0`)
**Hardware:** single local GPU (~12–16 GiB usable) — harness/prototype only; backbone training is a cloud job.

> **Scope note (read first).** S1 / S2 / S3 are *internal engineering/research phases*, **not** the Šokac proof
> framework. The Šokac/Sokač review is a **separate external-validity confirmation checklist** (§7) whose only job is
> to certify that the current claims are valid, bounded, reproducible, and not artifacts. This bundle does **not**
> start proto-introspection, does **not** write the paper, does **not** run or simulate S3A cloud training, and does
> **not** claim any capability ("Jormungandr") improvement. Every claim below carries an explicit status tag.

---

## 1. Executive summary

The in-transformer basal-ganglia **branch → carry → prune → loop-back → terminal** mechanism is built and
**mechanically validated**: five gates pass bit-exactly (α=0 re-derive identity, α>0 single- and two-locus splice
equivalence, live tap-prune integration, and two structural lineage invariants), and the full 12-locus reference
loop runs end-to-end with **zero correctness loss**. On the **frozen** backbone, deterministic internal branching is
**locally reachability-neutral**: the greedy (deterministic) fork produces **0.0** new-correct answers under every
tested α/token-range/decode regime, and a K-matched plain-sampling baseline **exceeds** fork+sampling (oracle 0.75 vs
0.611) — so the apparent "fork+sampling gains" are explained by **stochastic decoding, not the injection mechanism**.
Separately, the existing in-distribution hidden-state taps **do not transfer as direct correctness rankers** on
*generated* branch pools (correct-vs-incorrect separability is near chance, 0.49–0.57; CoreContent_v2 drops from
0.6691 in-distribution to 0.417 on generated pools), a **generated-distribution transfer failure** — not a global
tap failure. The three readout roles must stay **distinct and uncollapsed**: **DualAnchor = validity/survivability**
(never scored on correctness here, so *not evaluated on its primary role*), **CoreContent = relational content
quality** (its content→correctness *alignment* breaks out-of-distribution, but it is not globally broken), and the
**external verifier = correctness truth**. The next decisive capability experiment is **S3A: RLTT-style continuation
on verifier-labeled branch tournaments** — *training-time loop-dynamics integration*, not inference-time steering —
which cannot be run honestly on the local 12 GiB GPU and is therefore delivered here as a complete protocol, not an
execution. Local engineering is now **bounded enough to close**: the mechanism is validated, the frozen control case
is closed under tested regimes, the walls are cleanly separated, and the only remaining levers are training-time
(cloud) — so it is safe to **return to proto-introspection evidence and paper writing** after this closure, with the
optional local support tasks in §11 explicitly *not* gating that return.

---

## 2. Current claim table

Status legend: **PROVEN** (positive, validated) · **NEGATIVE_RESULT** (validated null/negative) · **UNDERPOWERED**
(direction observed, N too small) · **PROTOCOL_ONLY** (designed, not executed) · **CLOUD_REQUIRED** (needs
backbone/multi-GPU) · **NOT_TESTED** (out of scope of what was run).

| # | Claim | Status | Evidence artifact | Caveat | What would strengthen / falsify |
|---|---|---|---|---|---|
| 1 | Hidden-state pairwise evaluator/taps read **useful signals in-distribution** | **PROVEN** | `bg_corecontent_dataset_expansion_refit_v2_2026-06-04/` heldout (macro 0.6691 [0.6448–0.6902]); `mpn_s3b_2026-06-17/s3b0_refit_sanity.json` (refit path sane) | Realistic edge over broad baseline is single-digit pts on real negatives (+0.06–0.09); coding tap is partly a corruption detector | Powered relevance-negative eval; cross-seed refit CIs |
| 2 | The **mechanical branch loop works** (inject→carry→prune→loop-back→terminal) | **PROVEN** | `s1_4_reference_loop.json` (12 loci × 4 tasks, zero-loss, defer 0); gates 1–5 | N=4 shakedown; rank-only prune (no live thresholds); ~37 min/run | Scale N; run the gated pipeline; phase-two carried-KV path |
| 3 | **Re-derive / fork / carry path is valid** (identity + α>0 equivalence) | **PROVEN** | `s1_4_gate1_rederive.json`, `s1_4_gate2_alpha_chain.json` (maxabs 0.0, tokens 48/48) | bf16; last-token suffix range; tolerance is exact at prefill | Wider token-range ablations; longer chains |
| 4 | **Lineage invariants prevent clean-root misuse** | **PROVEN** | `s1_4_reference_loop.json` (gates 4+5 asserted inline, no AssertionError across 12 loci × 4 tasks) | Structural assertion, not a statistical test | Adversarial lineage fuzzing |
| 5 | **Live prune/tap path executes** on branch candidates | **PROVEN** | `s1_4_gate3_prune_integration.json` (features [3,4,2048]; DualAnchor 6ch + CoreContent 3ch finite/varying; gibberish branch bottomed −0.81) | Prune scores re-encoded *text* (teacher/scaffold), not the carried KV in-place | A no-rollout internal-state selector |
| 6 | **Frozen deterministic branching gives no reachability gain** (tested regimes) | **NEGATIVE_RESULT** | `s1_4a_fork_param_screen.json` (greedy new-correct 0.0 / 9 cells); `s1_4_reference_loop.json` | Local; tested α∈{.02,.05,.10} × {last,last-8,second-half} × loop-1 + loop-4 sentinel + chained α=0.02 | A regime where greedy fork > base — none found locally |
| 7 | **Fork+sampling gains are explained by K-matched sampling** | **NEGATIVE_RESULT** | `s1_4b_kmatched_sampling.json` (plain 0.75 ≥ sample-fork 0.611; diff −0.139; greedy-fork 0.0) | N=4; temp 0.7/top_p 0.95; budget N=12=3loci×K4 | Fork arm beating matched-budget sampling at scale |
| 8 | **Existing taps fail direct generated-branch correctness transfer** | **NEGATIVE_RESULT** (+ **UNDERPOWERED**) | `s3b1_corrected_addendum_2026-06-17.md`; `s3b1_loop_pool_transfer.json` (separability 0.49–0.57; CoreContent 0.417 < random 0.583) | Only **8 usable oracle-present pools** (math 2, reasoning 4, logic 2; coding 0/4) | Powered S3B-2 task-heldout selector; does generated-state hold a *trainable* correctness signal? |
| 9 | **DualAnchor not evaluated on its primary validity role** by direct correctness ranking | **NOT_TESTED** | `s3b1_corrected_addendum_2026-06-17.md` §2,§4 | DualAnchor was scored against correctness (not its job); 0.417/0.557 here is uninformative about validity | Survivor-retention / false-prune / oracle-retained-after-gate eval |
| 10 | **CoreContent is not globally broken**, but **content→correctness transfer fails on generated branches** | in-dist **PROVEN** / transfer **NEGATIVE_RESULT** | `s3b0_refit_sanity.json` (0.6691 in-dist); `s3b1_*` (0.417 generated, carried only by math 1.0; reasoning 0.25, logic 0.0) | The OOD drop is an *alignment* break, not loss of content signal | Generated-pool content-ranking eval among survivors (not correctness) |
| 11 | **Full ordered DualAnchor → CoreContent pipeline** is **not yet fully evaluated** | **NOT_TESTED** | `s1_4_reference_loop.json` `prune_mode = rank_only_content_plus_budget (validity diagnostic only)`; `s3b1_corrected_addendum` §1,§4 | The reference loop ran rank-only (thresholds = −1e9); the gated validity→content→survivor-set pipeline was never run | Run the ordered gated pipeline; headline survivor-set retention, top-1 diagnostic only |
| 12 | **S3A requires training-time loop-dynamics integration** (not inference-time steering) | **CLOUD_REQUIRED** / **PROTOCOL_ONLY** | §9 design; `mpn-run-design` S0/S1 ledger; frozen-steering closed (prior runs) | World_size_4 FSDP ⇒ multi-GPU; local is harness-only | Executed S3A on cloud with S3C eval gates |
| 13 | **Current local evidence does not prove Jormungandr capability improvement** | **NOT_TESTED** (no positive capability shown) | This bundle; `s1_report_2026-06-17.md` | The only local capability test (frozen loop) was reachability-neutral; the lever (S3A) is cloud-required | A passing S3C comparison after S3A training |

---

## 3. S1 mechanical proof (each gate, plain + technical)

**Plain-language headline:** *We proved the branch machinery does not corrupt the computation by itself.* When the
loop is told to do nothing (α=0), it reproduces the base model bit-for-bit; when it forks at one or two points, the
fork is exactly equivalent to running the same perturbation directly; the pruning/scoring path runs on real branch
candidates; and the bookkeeping that keeps each branch's history separate is enforced structurally. The full
12-point loop runs start-to-finish without losing a correct answer that was reachable.

| Gate | Plain | Technical | Result |
|---|---|---|---|
| **α=0 identity** | Loop set to "no change" = base model exactly | `branch_specific_root_pack` = `root_prefill_with_boundary` with the branch's lineage hooks active, then `build_spliced_branch`; empty-lineage @L1_24 and α=0-lineage @L1_36 vs clean | **bit-exact**: first-logit maxabs **0.0**, tokens **48/48** (`s1_4_gate1_rederive.json`) |
| **α>0 single-locus equivalence** | One fork = the same edit applied directly | splice fork (`build_spliced_branch`) ≡ hook replay (`full_perturbed_prefill` / `LayerOutputPerturbHook(token_range=last)`), α=0.02 | **bit-exact**: maxabs **0.0**, **48/48** (`s1_4_gate2_alpha_chain.json` 2a) |
| **Two-locus chaining equivalence** | Two stacked forks via per-branch re-derivation = doing both at once | re-derive through an α>0 ancestor as `LayerOutputPerturbHook(last-token)` → capture next boundary @L1_36 → splice ≡ all-hooks-live in one prefill | **bit-exact**: maxabs **0.0**, **48/48** (`s1_4_gate2_alpha_chain.json` 2b) |
| **Live prune path** | The scorer actually runs on branch text and ranks sensibly | branch text → `BGTransformerFeatureExtractor.encode_text_to_pooled_features` → `[3 layers(24/36/47) × 4 loops × 2048]` → `policy_candidate_scores` tournament; DualAnchor 6ch, CoreContent 3ch | **pass**: finite/varying; degenerate/gibberish branch correctly bottomed (CoreContent **−0.81** vs +0.30/+0.12) (`s1_4_gate3_prune_integration.json`) |
| **Structural lineage invariants** | A branch never "borrows" the clean starting point after it has diverged | Gate 4: a survivor's full lineage is always re-derived (no clean-root reuse after divergence). Gate 5: ancestor loci strictly precede the fork locus; appended entry == fork; lineage strictly increasing | **held** across **12 loci × 4 tasks** (no AssertionError) (`s1_4_reference_loop.json`) |
| **12-locus reference loop** | The whole loop runs and keeps reachable-correct answers | inject@24/L1 → perturb@24/36/47 every loop → DualAnchor (diagnostic) → CoreContent rank → budget → KV-carry loop-back @L*_47 → terminal @L4_47 | **runs end-to-end, zero loss**: `oracle_over_survivors 0.25 == base_acc 0.25 == selected_acc 0.25`, 4.0 survivors, defer 0 |

**Locked mechanism facts.** Single fork primitive = `apply_boundary_perturbation` / `LayerOutputPerturbHook(token_range)`;
canonical reference `token_range` = **last live token** `(seq_len−1, seq_len)`, causal-suffix-safe only. Root
re-derivation replays the **branch-specific lineage** (each branch owns its KV/cache, generated_ids, attention_mask,
position_ids, cache_position, boundary metadata, lineage). **No clean-root reuse** after divergence (gate 4). Decode
runs with **no hooks** (the one-time fork is baked into the cache). The KV-carry + suffix-recompute splice is itself
bit-exact and compute-saving for K≥2 (validated earlier in `partial_cache_splice_v2`).

---

## 4. Frozen reachability & sampling deconfound

**Definitions.**
- **base_acc** — base model greedy correctness (decode-length sensitive: **0.25 @MNT160** vs **0.0 @MNT96**; math is truncated before "FINAL ANSWER" at 96 — this is the MNT confound).
- **oracle_over_survivors** — does *any* terminal survivor hold a correct answer (upper bound on what selection could recover).
- **selected_acc** — does the *chosen top-1* survivor hold a correct answer (the forced-top-1 *diagnostic*).
- **deterministic fork result** — greedy decode after the injection: across all 9 greedy cells of the screen, **new-correct on base-missed = 0.0**.
- **fork-parameter screen (S1.4a)** — 18 cells = α{.02,.05,.10} × token_range{last, last-8, second-half} × decode{greedy, sample}, single-locus from clean root, K=4, + a loop-4 sentinel. Sample cells: new-correct **0.5–0.75**, ~identical regardless of α/token_range (sampling RNG dominates). `selected_acc ≈ 0` throughout.
- **K-matched sampling baseline (S1.4b)** — no fork; N=12 plain samples (= 3 loci × K4) per task, temp 0.7/top_p 0.95; base decoded greedy at both MNT 160 and 96.

**Why sampling explains the apparent gains.**

| arm | new-correct on base-missed | oracle |
|---|---|---|
| greedy fork (deterministic injection) | **0.0** | 0.0 |
| sample fork (fork + sampling) | 0.611 | 0.611 |
| **plain sampling (no fork, K-matched)** | — | **0.75** |

`sample_fork − plain_sampling = −0.139` → plain sampling **exceeds** fork+sampling. Combined with greedy-fork = 0.0,
the injection mechanism adds **nothing** beyond stochastic decoding; the "wins" are sampling diversity (plain sampling
reached new-correct on 2/3 true-base-missed tasks — logic + reasoning — but not coding). Per-task correct counts in
the matched baseline: logic 2, math 1, reasoning 5, coding 0; mean unique finals 11.25; parse-invalid 0.0.

**Why local frozen perturbation sweeps should stop.** The greedy fork is 0.0 across every tested α/token_range/loop,
the loop-4 sentinel is also null beyond sampling, the chained α=0.02 loop is null, and matched sampling beats
fork+sampling. There is no "winning regime" whose carry justifies the ~37-min/cell chained cost. The lever is
training-time, not stronger perturbations.

> **Verdict:** `FROZEN_BRANCHING_LOCALLY_CLOSED_UNDER_TESTED_REGIMES`.
> This is a **local decision verdict** — it does **not** claim that *no* frozen branch regime can ever work anywhere.

---

## 5. Selector / tap-transfer closure

- The original evaluator and the DualAnchor/CoreContent taps read **hidden-state features** (Ouro digests labeled
  text; the tap reads only pooled loop hidden states `[3×4×2048]`; antisymmetric by construction). They are real
  readouts, not text classifiers.
- **CoreContent worked in-distribution** (heldout core macro **0.6691**; refit path reproduced this, `S3B0_REFIT_SANE`).
- **Generated branch pools are out-of-distribution** for these taps: they were trained on curated, often
  constructed-negative tournaments; generated pools are plausible-but-varied model continuations.
- On generated pools the existing taps are **near chance as direct correctness rankers**: correct-vs-incorrect
  **separability 0.49–0.57** (random 0.46, oracle 1.0); CoreContent_v2 **0.417** (below random 0.583), carried
  entirely by math (1.0), collapsing on reasoning (0.25) and logic (0.0).
- This is a **generated-distribution transfer failure for direct correctness ranking**. It is **not** a global tap
  failure (in-distribution macro ≈ 0.67 stands), **not** a DualAnchor validity failure (validity was never tested),
  and **not** a failure of the ordered pipeline (never run). It is also **underpowered** — 8 usable pools.

---

## 6. Mandatory role separation

Three axes, never to be collapsed: **validity** (can a branch survive) vs **content** (which surviving branch is
better) vs **correctness** (is it actually right, per external verifier).

| Component | Role | Correct metric | Incorrect / misleading metric |
|---|---|---|---|
| **DualAnchor** | **Validity / survivability** (the first prune; CoreContent cannot judge this) | survivor retention, false-prune rate, oracle retained after gate | direct forced top-1 correctness *as the sole judgment* |
| **CoreContent_v2** | **Relational content quality** (content selection + content-threshold prune among valid branches) | content ranking among survivors, top-k retention, in-distribution ranking | treating it as a correctness oracle out-of-distribution (alignment may break OOD) |
| **Verifier / gold** | **Correctness truth** | exact answer, unit tests, parser/gold key, z3, sympy `math_equal_robust` | — (it is the label; never a tap output) |
| **Branch-correctness selector** | **Future / generated-distribution correctness selector** (separate axis; S3B-2/S3-N) | `selected_correct_when_oracle_present`, pairwise correct>incorrect separability, chance-normalized oracle conversion | reusing DualAnchor or CoreContent as if they were this |
| **Forced top-1** | **Diagnostic** unless the architecture explicitly commits to forced top-1 | report alongside survivor-set retention | reporting it as the *only* architecture metric |
| **Survivor-set retention** | **Correct headline** for branch-survival pipeline tests | `oracle_over_survivors`, top2/top4 retention | — |

---

## 7. Šokac / Sokač validity-confirmation checklist

> **Provenance:** no prior Šokac/Sokač-specific checklist artifact exists in the repo (searched; only unrelated
> browser-catalog files matched the string). **This checklist is reconstructed from current project state** per the
> task's fallback instruction. It is a *separate* external-validity review, **not** S1/S2/S3.

| ID | Confirmation item | Status | Artifact | Explanation | Remaining action |
|---|---|---|---|---|---|
| **A** | Artifact integrity & reproducibility | **PARTIAL** | `mpn_s1_baseline_2026-06-13/*`, `mpn_s3b_2026-06-17/*`, this dir | All key reports + JSON summaries saved; scripts under `utilities/tests/manual/mpn_s1_*`, `mpn_s3b*`; logs under `artifacts/logs/mpn_s1`. Configs (α, MNT, temp/top_p, K, budget) recorded in JSON. **Gaps:** explicit RNG **seeds not pinned** in sampling artifacts; **no dedicated `artifacts/logs/mpn_s3b` dir** (S3B evidence is the JSONs). | Pin/record seeds for any powered re-run; add an mpn_s3b log path |
| **B** | No hidden training of Ouro during readout/tap tests | **PASS** | `S0_RELOAD_VERIFY.json`, `s3b0_refit_sanity.json` | Backbone **frozen** for all S1/S3B readout/tap work; only a tiny selector head was trained (S3B-0), saved separately (`s3b0_trained_selector.pt`); **no base/tokenizer overwrite**; S0 backup reload-verified (finite logits, FSDP 533 entries, taps load). | None for closure; re-affirm before any S3A run |
| **C** | Data split integrity | **PARTIAL** | `s3b1_corrected_addendum_2026-06-17.md`, `corecontent_v2` heldout | In-distribution heldout preserved; S3B-1 is a one-shot diagnostic (no training ⇒ no candidate leakage). The **mandated task-level heldout for a trained generated-branch selector is not yet built** (that is S3B-2). | Build powered S3B-2 with **task-level** (not candidate-level) heldout |
| **D** | Pair orientation / antisymmetry controls | **PASS** | `s3b1_corrected_addendum` §3; tap construction | Taps are **exactly antisymmetric** by construction (`layernorm(i−j)·w`, sign-flip 0). ORACLE/RANDOM **excluded** from "best real selector." Flip behavior known (paper ρ≈−0.94). | None |
| **E** | Feature-cache / provenance integrity | **PASS** | `s3b1_loop_pools.pt`, `s3b1_pool_texts.json` | Features `[3,4,2048]` correspond to the correct prompt+answer inputs; layer/loop positions (24/36/47 × 4 loops, `force_all_loops`) recorded; pools + texts persisted; no stale cache silently reused. | Standardize scoring **format** (see F) across sub-runs |
| **F** | Text-shortcut controls | **PARTIAL** | `s1_4_reference_loop.json` `score_format_caveat`; corecontent v2 followups | Heads train on **hidden states** (Ouro digests text; tap never reads raw text) — the core shortcut is controlled. **Unresolved:** prompt/length/domain-style shortcuts not fully ablated; scoring format varied (S1.4 answer-only vs S1.4a/b prompt+answer); coding tap shown to be a *corruption detector*, not a relevance ranker. | Length/style/domain shortcut ablation; standardize on prompt+answer |
| **G** | Baselines present | **PASS** | `s1_4b_kmatched_sampling.json`, `s3b1_*` | random, oracle, base greedy, K-matched sampling, frozen branch/carry, existing taps, mixed heads (MIX_HH/MIX_OBJECTIVE), DualAnchor (diagnostic only) all present. | None |
| **H** | Sampling deconfound | **PASS** | `s1_4b_kmatched_sampling.json` | fork+sampling compared to K-matched plain sampling; apparent fork gains attributed to sampling (diff −0.139, greedy-fork 0.0); compute/sample budget reported (N=12=3loci×K4, ~9 min). | None |
| **I** | Mechanism correctness | **PASS** | `s1_4_gate1/2/3*.json`, `s1_4_reference_loop.json` | α=0 identity, α>0 fork equivalence, two-locus chaining, lineage invariants (gates 4+5), no clean-root misuse, full 12-locus loop — all pass/held. | None |
| **J** | Role separation | **PASS** | `s3b1_corrected_addendum_2026-06-17.md` | DualAnchor=validity (not correctness), CoreContent=content (not correctness), verifier/gold=correctness, forced top-1=diagnostic — enforced and corrected in the addendum. | Keep enforced in all future write-ups |
| **K** | Statistical power / underpowered areas | **PARTIAL** | `s1_report_2026-06-17.md` §6; `s3b1_corrected_addendum` §4,§5 | N=4 shakedown caveats stated; 8-usable-pool S3B-1 caveat stated; coding underpowered (0 oracle-present pools) and **excluded** from oracle-present macros; zero-oracle domains never averaged in. **Controls are correct; the underlying power is low.** | Powered N for any capability claim |
| **L** | Negative-result honesty | **PASS** | `s1_report_2026-06-17.md`, this bundle | frozen branching reachability-neutral (stated); S3A not run locally (stated); **no Jormungandr capability claim**; **proto-introspection not claimed proven** by S3. | None |
| **M** | External-review readiness | **PASS** | this bundle §8 | What can/can't be shown to an ML reviewer, and the one experiment that would change the conclusion, are stated explicitly. | None |

**Checklist roll-up:** 8 PASS, 4 PARTIAL, 0 FAIL, 0 NOT_TESTED-blocking. The PARTIALs (A seeds/logs, C task-heldout,
F shortcut ablation, K power) are **honest limitations of an early-stage frozen-baseline program**, not validity
breaks — none invalidates a stated claim; each maps to a powered follow-up (S3B-2 / paper-grade re-run), not to a
correction.

---

## 8. What is complete enough for Šokac / Sokač

- **What exactly has been proven?** The in-transformer branch-carry loop is mechanically correct (5 gates bit-exact /
  structurally held; 12-locus loop zero-loss), the hidden-state taps carry real in-distribution signal (CoreContent
  macro 0.6691; refit sane), and the sampling deconfound is clean (fork adds nothing beyond K-matched sampling).
- **What exactly failed?** On the **frozen** backbone, deterministic internal branching is reachability-neutral
  (greedy fork 0.0 new-correct; fork+sampling < plain sampling), and existing taps do **not** transfer as direct
  correctness rankers on generated branch pools (separability near chance; CoreContent 0.417 < random).
- **What is bounded but not solved?** Whether *trained* loop dynamics make injected branches **outcome-distinct**
  (Wall A / S3-M) and whether a **branch-correctness selector** can be trained on generated pools (Wall B / S3-N).
  Both are bounded as the two open levers; neither is solved locally.
- **What local artifacts support the claims?** The S1 gate JSONs, the reference-loop JSON, the fork-screen and
  K-matched-sampling JSONs, the CoreContent v2 heldout, the S3B-0 refit sanity, and the S3B-1 corrected addendum +
  pools (full index in §13).
- **What validity concerns remain?** (from §7) unpinned seeds, task-level heldout not yet built for a trained
  generated-branch selector, incomplete text-shortcut ablation, and low statistical power (N=4 / 8 pools).
- **The one experiment that turns mechanism into capability:** **S3A** — RLTT-style trajectory-credit continuation on
  verifier-labeled branch tournaments — followed by the **S3C** integrated comparison (does trained branch/carry beat
  frozen branch/carry **and** K-matched sampling, with a trained selector converting oracle to selected?).
- **Why a professional ML person should take this seriously:** the mechanism is bit-exact and gated (not hand-waved);
  the negative results are deconfounded against the obvious alternative (sampling); the role separation is enforced;
  and there is a verified weight backup (S0) and a real RLTT checkpoint+optimizer to continue from — the next
  experiment is well-posed and well-resourced (on cloud).
- **Why they should not overread it:** every capability number here is on a **frozen** model at **shakedown N**;
  nothing here demonstrates a capability gain, proto-introspection, or "Jormungandr" improvement; the taps' transfer
  failure is shown on **8 pools**; and S3A has **not** been run.

---

## 9. S3A design — branch-tournament RLTT continuation (the S3 mainline)

> **Status:** `READY_AS_CLOUD_BACKBONE_TRAINING_PROTOCOL_NOT_LOCALLY_RUN`. This section is a **protocol**. It was not
> run, and **must not** be run on the 12 GiB local GPU (world_size_4 FSDP ⇒ multi-GPU/cloud). No fabricated numbers.

- **Objective.** Train injected/internal branches to become **outcome-distinct and verifier-useful** — i.e. a bounded
  injection at a locus lands on a *distinct, reachable* trajectory that sampling does not already provide, and that
  the loop's carry/prune can act on. This directly targets **Wall A** (the frozen null). It is **not** a better text
  generator and **not** inference-time steering.
- **Data.**
  - `branch_set_dpo` (66.3k offline preference pairs; oracle/overbranch/underbranch/budget views) — eval-aligned `train_v5` substrate.
  - **Math** branch tournaments (verifier-labeled).
  - **Code** branch tournaments (unit-test-labeled; use reasoned-code data to avoid the terse-wrong-code degeneration seen offline).
  - **Generated branch pools** if/when powered (S3B-2 output) — usable, but verifier-labeled only.
- **Labels.** External **verifier / gold correctness only** (`label_branch` / `verify_final` / `verify_code_unit_tests`
  / `math_equal_robust` / MCQ keys / z3). **Never** tap/evaluator scores as correctness labels. Alignment (no
  objective verifier) uses the antisymmetrized pairwise tap margin `(f(A,B)−f(B,A))/2`, marked **model-internal**, never external truth.
- **Update mode.**
  - LoRA/adapters — **optional pilot/debug** only (offline LoRA was reachability-neutral; use as a smoke test).
  - **RLTT-style continuation — mainline.** Fixed loop depth T=4, **trajectory-level credit across loops**
    (`Σ ωₜ ∇log P^(t)·Â`, the local model exposes `OuroForCausalLMRLTT.forward(return_per_loop_logits=True)` +
    `per_loop_exit_gates`), binary correctness reward, **no reward model**, KL to the pre-RL (backed-up) policy.
    Diversity credited only when ≥1 branch is verifier-positive (`offline_reward_v2`).
  - Full backbone update — **high-capacity implementation if compute allows**; continue the real recipe from the
    pristine world_size_4 FSDP **model + optimizer + extra_state** shards.
- **Safety.** **S0 backup confirmed** (`COMPLETE_VERIFIED`, reload-verified); train against the **v2 root**; **no
  base/tokenizer overwrite**; **no destructive mutation**; regression-baseline tripwire (CoreContent heldout 0.6691 /
  DualAnchor 0.3787; survival retention 0.9848 / terminal 1.0) re-run after training to detect de-calibration.
- **Why this is not steering.** Frozen steering was **inference-time perturbation** of a fixed model (closed:
  readout-geometry ≠ success-geometry ≠ logit-control-geometry). **S3A is training-time loop-dynamics integration** —
  it changes how the model *branches*, then freezes and evaluates; the control action remains discrete keep/prune
  over materialized KV branches, not a steering vector.

---

## 10. S3C integrated evaluation protocol

> **Status:** `INTEGRATED_EVALUATION_PROTOCOL_SPECIFIED_NOT_EXECUTED`. Reuses the validated S1 harness as the instrument.

**Arms to compare (equal compute/sample budget):**
1. base greedy,
2. K-matched plain sampling,
3. frozen S1 branch/carry,
4. S3A-trained branch/carry,
5. S3A-trained branch/carry **+ branch-correctness selector** (S3-N / S3B-2),
6. S3A-trained branch/carry **+ ordered DualAnchor → CoreContent pipeline** (validity gate → content rank → survivor-set).

**Metrics:** `base_acc`, `oracle_over_survivors`, `selected_acc`, `selected_correct_when_oracle_present`,
`new_correct_on_base_missed`, branch diversity (distinct outcomes), parse/invalid rate, **carry gain over sampling**
(arm 4/5/6 vs arm 2 at matched budget), compute budget, and **domain breakdown** (oracle-present domains only;
zero-oracle domains excluded from oracle-present macros; ORACLE/RANDOM excluded from "best real selector").

**Decision rule (pre-registered):** a capability claim requires arm 4 (and/or 5/6) to **beat both** arm 1 (base) and
arm 2 (K-matched sampling) on `new_correct_on_base_missed` **with** a selector that converts oracle to selected
(`selected_correct_when_oracle_present` ≫ random), on a **task-heldout** set at powered N. Otherwise: no capability claim.

---

## 11. Optional remaining local work (support, not a blocker)

These are **optional support tasks**, explicitly **not** required before returning to proto-introspection (the
artifact review found **no hard validity gap** that gates the return):

- Powered **S3B-2** generated-branch dataset (task-heldout, verifier labels).
- **Branch-correctness selector** refit on generated pools (the S3-N axis; distinct from DualAnchor/CoreContent).
- **Ordered DualAnchor → CoreContent survivor-retention** test (the untested gated pipeline; headline = survivor-set, top-1 diagnostic only).
- Artifact/table cleanup: pin seeds, add `artifacts/logs/mpn_s3b`, standardize prompt+answer scoring.

---

## 12. Final verdict constants

```text
S1_MECHANISM_VERDICT                     = BRANCH_CARRY_LOOP_MECHANICALLY_VALIDATED
FROZEN_REACHABILITY_VERDICT              = FROZEN_BRANCHING_LOCALLY_REACHABILITY_NEUTRAL
SAMPLING_DECONFOUND_VERDICT              = APPARENT_FORK_SAMPLE_GAINS_EXPLAINED_BY_K_MATCHED_SAMPLING
S3B1_DIRECT_CORRECTNESS_TRANSFER_VERDICT = EXISTING_TAPS_NEAR_CHANCE_ON_GENERATED_BRANCH_CORRECTNESS
S3B1_DUALANCHOR_INTERPRETATION           = NOT_EVALUATED_ON_PRIMARY_VALIDITY_ROLE
S3B1_CORECONTENT_INTERPRETATION          = CONTENT_TO_CORRECTNESS_TRANSFER_FAILS_ON_GENERATED_BRANCHES
S3A_TRAINING_DESIGN_VERDICT              = READY_AS_CLOUD_BACKBONE_TRAINING_PROTOCOL_NOT_LOCALLY_RUN
S3C_PROTOCOL_VERDICT                     = INTEGRATED_EVALUATION_PROTOCOL_SPECIFIED_NOT_EXECUTED
SOKAC_VALIDITY_BUNDLE_VERDICT            = CURRENT_LOCAL_CLAIMS_AUDITED_WITH_REMAINING_LIMITATIONS_EXPLICIT
FINAL_LOCAL_CLOSURE_VERDICT              = MECHANISM_VALIDATED_FROZEN_CONTROL_CLOSED_TRAINING_TIME_INTEGRATION_REQUIRED
NEXT_PHASE_RECOMMENDATION                = RETURN_TO_PROTO_INTROSPECTION_EVIDENCE_AND_PAPER_WRITING
```

---

## 13. Artifact index

**Reports (Markdown)**
- `artifacts/reports/probes/mpn_s1_baseline_2026-06-13/s1_report_2026-06-17.md` — consolidated S1 report.
- `artifacts/reports/probes/mpn_s3b_2026-06-17/s3b1_corrected_addendum_2026-06-17.md` — S3B-1 strict role-separation correction.
- `artifacts/reports/probes/mpn_s3_closure_2026-06-17/s3_closure_and_sokac_validity_bundle.md` — **this bundle**.
- `/home/moloch/ouro_backups/mpn_s0_pre_run_2026-06-13/S0_REPORT.md` + `regression_baseline_reference/` — S0 backup + tripwire reference.

**Result JSONs**
- S1 gates: `s1_4_gate1_rederive.json`, `s1_4_gate2_alpha_chain.json`, `s1_4_gate3_prune_integration.json` (under `mpn_s1_baseline_2026-06-13/`).
- S1 loop/screen: `s1_4_reference_loop.json`, `s1_4a_fork_param_screen.json`, `s1_4b_kmatched_sampling.json`, `s1_4_onetime_carry.json`, plus S1.2 series (`s1_2_*`), `s1_0_tripwire.json`, `s1_1_task_set.json`.
- S3B: `mpn_s3b_2026-06-17/s3b0_refit_sanity.json`, `s3b1_loop_pool_transfer.json`, `s3b1_pool_texts.json`.
- S0: `/home/moloch/ouro_backups/mpn_s0_pre_run_2026-06-13/S0_BACKUP_MANIFEST.json`, `S0_RELOAD_VERIFY.json`.
- Closure verdicts: `artifacts/reports/probes/mpn_s3_closure_2026-06-17/s3_closure_verdicts.json`.

**Scripts** (all under `utilities/tests/manual/`, run via `venv/bin/python -u`)
- `mpn_s1_4_rederive_gate.py`, `mpn_s1_4_gate2_alpha_chain.py`, `mpn_s1_4_gate3_prune_integration.py`,
  `mpn_s1_4_reference_loop.py`, `mpn_s1_5_divergence_ablation.py` (→ fork screen), `mpn_s1_4b_kmatched_sampling.py`,
  `mpn_s3b0_refit_sanity.py`, `mpn_s3b1_loop_pool_transfer.py`.
- Shared primitives: `bg_autoregressive_cache_common_v1.py`, `bg_partial_cache_splice_v2_common.py`,
  `branch_training_v1_common.py`, `bg_corecontent_v2_*` ; reward `utilities/branch_training/offline_reward_v2.py`;
  S0 tooling `utilities/tools/mpn_s0_backup.py`, `mpn_s0_verify_reload.py`.

**Logs** — `artifacts/logs/mpn_s1/`, `artifacts/logs/mpn_s0/`.
`MISSING: artifacts/logs/mpn_s3b/ — consequence: S3B-0/S3B-1 have no dedicated run log (the JSONs are the artifacts of record); reproducibility relies on re-running the scripts. Low impact.`

**Generated datasets / pools / feature dirs**
- `artifacts/reports/probes/mpn_s3b_2026-06-17/s3b1_loop_pools.pt` (15 MB, the generated branch pools + verifier labels for S3B-2).
- `data/corecontent_v2/features/` (65 shards, ~4.87 GB) — frozen-Ouro pooled features for tap refit/re-run.
- `data/branch_training_logic_expansion_v1/` + `train_v2/`…`train_v5/` — offline branch-set / DPO / reward views (S3A data substrate).

**Model / tap / checkpoint assets**
- Operational backbone: `models/ouro_rltt_local/` (frozen; never overwritten).
- Pristine RLTT FSDP: `/home/moloch/Downloads/RLTT/Downloads_RLTT/RLTT/` — `model_/optim_/extra_state_world_size_4_rank_0..3.pt` + `fsdp_config.json` + `consolidated_clean.pt` (S3A continuation source).
- S0 backup: `/home/moloch/ouro_backups/mpn_s0_pre_run_2026-06-13/` (33 GB; backbone_operational + backbone_rltt_fsdp_source + all taps + manifest + reload-verify). **Caveat: same-disk; no offsite copy yet.**
- Taps: DualAnchor MIX heads, CoreContent_v2 policy (`corecontent_v2_policy.pt`), S3B-0 trained selector (`s3b0_trained_selector.pt`), head registry, `pairwise_epoch2.pt` (defunct reference).

`MISSING: offsite/cloud copy of S0 backup — consequence: a single-disk failure would lose the only verified pre-S3A weight backup; weakens the S3A safety precondition until an offsite copy exists.`

---

*End of bundle. This is an engineering-closure + external-validity artifact. It does not start proto-introspection,
does not write the paper, and does not claim any capability improvement.*
