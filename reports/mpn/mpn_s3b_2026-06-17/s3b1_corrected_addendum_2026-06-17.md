# S3B-1 Corrected Addendum — Strict Role Separation

**Date:** 2026-06-17 · Supersedes the interpretation/verdict in the original `s3b1_loop_pool_transfer.json`
run output. The raw per-pool measurements are unchanged; the framing, aggregation, and verdict are corrected.

This addendum enforces a strict separation of three axes that the original write-up conflated, and corrects a
macro-aggregation artifact (domains with **zero** oracle-present pools were averaged in as 0.0, deflating every
selector — including ORACLE, which read 0.75 instead of 1.0).

---

## 1. Corrected executive summary

S3B-1 is a **direct correctness-ranking diagnostic** on *generated branch pools*: for each task we generated a
pool of branches (sampled fork @loop1_L24 + plain samples), labeled each branch by the **external verifier**
(answer/unit-test/gold), and asked several existing scorers, in one shot, to rank the verifier-correct branches
to the top. It is **not** the full architecture pipeline (validity gate → content ranking → survivor-set
handoff), and forced top-1 is **not** the architecture's success metric.

Finding: **existing in-distribution content/preference taps do not transfer as direct correctness rankers on
the generated branch distribution.** The cleanest signal is correct-vs-incorrect **separability**, which is
near chance for every real tap (0.49–0.57; random 0.46; oracle 1.0). Top-1 correctness selection is only weakly
discriminating here because the oracle-present pools are correctness-rich (random top-1 already ≈ 0.58).
CoreContent_v2, strong in-distribution (0.6691, S3B-0), drops to 0.4167 on generated pools — below random.

This is a **generated-distribution transfer failure for direct correctness ranking**. It is *not* a statement
that DualAnchor failed at validity, that CoreContent failed at content ranking in its intended in-distribution
role, or that the ordered pipeline failed (none of those were tested here).

---

## 2. Role separation (the three axes)

| Axis | Owner | What it decides | Tested correctly here? |
| --- | --- | --- | --- |
| **Validity / survivability** | **DualAnchor** | Is a branch viable enough to keep alive? A valid branch can still be wrong. | **No** — DualAnchor was scored against correctness, which is not its job. |
| **Relational content quality** | **CoreContent_v2** | Rank branch/candidate content relationally among alternatives. Correlates with correctness *in its training distribution* (correctness-labeled tournaments); that correlation is empirical and can break OOD. | Partially — its content→correctness *alignment* was tested, not "content ranking" per se. |
| **Correctness** | **External verifier** | Exact answer / unit tests / gold preference. A label, **not** a tap output. | Yes — this is the label axis. |
| **Branch-correctness selector** | **S3B (new)** | A selector trained explicitly to rank verifier-correct branches. **Not identical to DualAnchor or CoreContent.** | Not yet built (that is S3B-2). |

---

## 3. What the numbers legitimately show

Macros over **oracle-present domains only** (math, reasoning, logic; coding excluded — 0 oracle-present pools).
ORACLE and RANDOM are baselines and are **excluded** from "best real selector."

| selector | sel@oracle | separability (correct>incorrect) | top2 ret | top4 ret | regret |
| --- | --- | --- | --- | --- | --- |
| ORACLE *(baseline)* | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| RANDOM *(baseline)* | 0.583 | 0.463 | 0.583 | 0.750 | 0.250 |
| **mixedhead_MIX_HH_OBJECTIVE** *(best real)* | **0.667** | **0.569** | 0.750 | 0.833 | 0.250 |
| MIX_OBJECTIVE_ALL_only | 0.500 | 0.544 | 0.500 | 0.667 | 0.333 |
| S3B0_pairwise_blockwise (refit) | 0.500 | 0.508 | 0.750 | 0.833 | 0.333 |
| S3B0_listwise_blockwise (refit) | 0.417 | 0.500 | 0.750 | 0.833 | 0.417 |
| CoreContent_v2_blockwise | 0.417 | 0.490 | 0.583 | 0.833 | 0.417 |
| DualAnchor *(diagnostic only)* | 0.417 | 0.557 | 0.500 | 0.583 | 0.417 |

- **Best real selector** (excluding ORACLE/RANDOM): MIX_HH at sel@oracle 0.667, separability 0.569.
- **Separability is near chance for every real tap** (0.49–0.57 vs random 0.46) — the robust diagnostic.
- **CoreContent_v2 collapses**: 0.6691 in-distribution (S3B-0) → 0.417 on generated pools (below random 0.583).
  Per-domain it is carried entirely by math (1.0) and collapses on reasoning (0.25) and logic (0.0).
- Top-1 `sel@oracle` is **weakly discriminating** here: oracle-present pools are correctness-rich (e.g. some
  pools had 7–8 of 10 correct), so random top-1 already scores ≈ 0.58. Separability and per-domain breakdown are
  the more honest reads.
- Survivor-set retention (top2/top4) for the better real taps (0.75/0.83) is modestly above random (0.58/0.75)
  and below oracle (1.0) — i.e. the survivor-*set* degrades less than forced top-1, consistent with the locked
  "survivor-set handoff, not forced top-1" stance (but on tiny N).

Pool counts: **16 tasks → 8 oracle-present pools = 8 reward-diverse pools = 8 usable pools** (math 2,
reasoning 4, logic 2; **coding 0/4** — sampling never produced a correct coding branch).

---

## 4. What the numbers do NOT show

- They do **not** show DualAnchor is bad at branch validity. DualAnchor was scored on correctness, which is not
  its role; its 0.417/0.557 here is uninformative about validity/survivability.
- They do **not** show the ordered **DualAnchor → CoreContent → survivor-set** pipeline fails — that pipeline
  was never run.
- They do **not** show CoreContent has no content signal generally — only that its content→correctness
  *alignment* (an empirical correlation from correctness-labeled training) does not transfer to generated
  branches under this one-shot ranking test.
- They do **not** settle whether generated-branch hidden states contain a *trainable* correctness signal: only
  8 usable pools across 3 domains, far too few for a powered train/heldout selector result.

---

## 5. Corrected metrics (panel)

```text
oracle_present_pool_count           : 8   (of 16 tasks)
reward_diverse_pool_count           : 8
usable_pool_count                   : 8   (math 2, reasoning 4, logic 2; coding 0)
oracle_present_domains              : math, reasoning, logic   (coding excluded from macros)

selected_correct_when_oracle_present (macro over oracle-present domains):
  ORACLE (baseline)                 : 1.000
  RANDOM (baseline)                 : 0.583
  best_real_selector (MIX_HH)       : 0.667     <- excludes ORACLE/RANDOM
  CoreContent_v2_blockwise          : 0.417     (in-distribution was 0.6691)
  DualAnchor (diagnostic only)      : 0.417

pairwise_separability (correct>incorrect), real taps : 0.490 - 0.569  (random 0.463, oracle 1.0)
top2_oracle_retention  : best real 0.750 (random 0.583, oracle 1.0)
top4_oracle_retention  : best real 0.833 (random 0.750, oracle 1.0)
regret                 : best real 0.250 (random 0.250, oracle 0.0)

per-domain selected_correct_when_oracle_present:
  CoreContent_v2 : math 1.0  reasoning 0.25  logic 0.0
  MIX_HH         : math 1.0  reasoning 0.50  logic 0.5
  RANDOM         : math 0.5  reasoning 0.75  logic 0.5
```

---

## 6. Corrected conclusion

> S3B-1 shows that existing in-distribution content/preference taps do not directly transfer as correctness
> rankers on generated branch pools. The strongest interpretation is a **generated-distribution transfer
> failure**. This is distinct from DualAnchor's validity role and distinct from CoreContent's in-distribution
> content-ranking role. The next decisive test is **S3B-2**: train a branch-correctness selector on generated
> branch pools with verifier labels and evaluate on **task-heldout** generated pools.

---

## 7. S3B-2 decision

`S3B2_RECOMMENDATION = GENERATE_POWERED_BRANCH_POOL_DATASET_BEFORE_REFIT`

The current 8 usable oracle-present/reward-diverse pools are **too few** for a meaningful train/heldout selector
result. A powered S3B-2 must first generate a **larger** labeled generated-branch dataset.

**S3B-2 design constraints:**
- Larger labeled generated-branch dataset; **task-heldout** splits (never candidate-heldout).
- Labels = verifier/gold correctness; **never** evaluator/tap scores as labels.
- Train a **new** branch-correctness selector, separate from DualAnchor and CoreContent.
- Compare against: frozen CoreContent_v2, S3B-0 refits, MIX_HH, MIX_OBJECTIVE_ALL, **DualAnchor (diagnostic
  baseline only, not a correctness selector)**, RANDOM, ORACLE.
- Primary metric: `selected_correct_when_oracle_present`. Secondary: pairwise separability, top2/top4 retention,
  regret, domain breakdown.

**Separate (optional) pipeline test** — run the actual ordered architecture: DualAnchor validity gate →
CoreContent ranking among survivors → survivor-set retention. **Headline survivor-set / oracle retention**, not
forced top-1; forced top-1 is diagnostic only.

---

## Top-line verdict constants

```text
S3B1_DIRECT_CORRECTNESS_TRANSFER_VERDICT = EXISTING_TAPS_NEAR_CHANCE_ON_GENERATED_BRANCH_CORRECTNESS
S3B1_DUALANCHOR_INTERPRETATION          = NOT_EVALUATED_ON_PRIMARY_VALIDITY_ROLE
S3B1_CORECONTENT_INTERPRETATION         = CONTENT_TO_CORRECTNESS_TRANSFER_FAILS_ON_GENERATED_BRANCHES
S3B1_PIPELINE_VERDICT                   = FULL_DUALANCHOR_TO_CORECONTENT_PIPELINE_NOT_TESTED
S3B2_RECOMMENDATION                     = GENERATE_POWERED_BRANCH_POOL_DATASET_BEFORE_REFIT
```

---

## What changed from the previous report

1. **Role separation enforced.** Validity (DualAnchor), content (CoreContent), and verifier correctness are now
   three distinct axes. DualAnchor is no longer described as a failed correctness selector; CoreContent is no
   longer described as globally broken.
2. **Verdict narrowed.** "Selector wall / selectors cannot pick winners / DualAnchor & CoreContent fail / all
   selectors useless" → the five scoped constants above (direct-correctness-transfer failure; DualAnchor not
   tested on validity; CoreContent content→correctness transfer fails; full pipeline not tested).
3. **Aggregation artifact fixed.** Macros now exclude domains with 0 oracle-present pools (coding). Previously
   coding contributed 0.0, deflating every selector — including ORACLE (0.75 → 1.0). Corrected: random
   sel@oracle 0.4375 → 0.583; best real (MIX_HH) → 0.667; CoreContent → 0.417.
4. **ORACLE/RANDOM excluded from "best real selector."** Best real selector is now reported separately
   (MIX_HH), and the previous bug (ORACLE's 1.0 separability counted as "best separability") is removed.
5. **Primary signal reframed.** Near-chance **separability** (0.49–0.57) is the robust diagnostic; forced top-1
   `sel@oracle` is flagged as weakly discriminating on correctness-rich pools and is no longer the headline.
6. **CoreContent transfer stated precisely.** In-distribution 0.6691 → generated 0.417 (below random), carried
   only by math; reframed as a content→correctness *alignment* failure OOD, not a global tap failure.
7. **Pipeline scope clarified.** S3B-1 is a one-shot correctness-ranking diagnostic, explicitly **not** the
   ordered DualAnchor → CoreContent → survivor-set pipeline; that remains untested.
