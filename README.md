# Lifetime Meta-Learning

Research repository for the **lifetime meta-learning direction** on Princeton
Ouro-RLTT looped transformers: the program that moves from *reading* frozen
loop-state signals to *writing* them — asking whether the model's recurrent
hidden states are a causally reachable substrate that training could
eventually exploit across a model's lifetime.

> The repository is named for the direction's goal, not its current
> contents: what lives here today is the groundwork that goal depends on —
> the O1 oracle-reachability program (is the substrate causally writable at
> all?) plus the M+N backbone-training and branch-training reports.
> Meta-learning experiments proper begin only if O1 returns a positive,
> preregistered result.

The flagship experiment is **O1 (oracle reachability)**: a preregistered,
precommitted intervention study injecting frozen, hash-sealed axis directions
into the L3 (physical layer 24) residual stream at the final prompt token and
asking whether task outcomes (H_reach) move — with common random numbers,
a Latin-square action-to-stream design, sealed parsing/verification, and
transport measured (diagnostic only) at the L4_47 loop boundary.

## Position in the research program

This is the third repository of the program. The sibling repos hold the
earlier stages, and nothing published there is duplicated here:

| Stage | Repository | Contents |
|---|---|---|
| Readout (RPE) | [Hidden-State-Evaluator](https://github.com/VykosMolt/Hidden-State-Evaluator) | pairwise / relational preference evaluator line (arXiv:2604.09870 + erratum), CLT notes |
| Probes & branch selection (OPI) | [Branching-Looped-Transformer](https://github.com/VykosMolt/Branching-Looped-Transformer) | evaluator probes, proto-introspection results, branch-survival / terminal-selection scaffolding, kirin2026 papers, M+N probe code |
| Write access / lifetime meta-learning (this repo) | — | the O1 oracle-reachability program end to end, M+N stage reports, branch-training-toward-internal-branching reports |
| Agent track | [Hunter-Seeker-v2](https://github.com/VykosMolt/Hunter-Seeker-v2) (and [v1](https://github.com/VykosMolt/Hunter-Seeker-v1), frozen) | transactional non-LLM ARC-AGI-3 agent; consumes Ouro loop-state taps as typed sensors |

## Contents

- **`o1_packages/`** — the sealed O1 implementation lineage:
  `O1_oracle_reachability_v1.5.3` → `v2.0.0` (superseded by audit) →
  `v2.1.0` (verified; zip `a8b0571c…`), plus the runner packages
  `O1_B200_RUNNER_v0.1.0` → `v0.2.0` → `O1_B300_RUNNER_v0.3.1`
  (zip `89cf2092…`).
- **`o1_runs/`** — the scientific record: `O1_REAL_001` (v1.5.3 real attempt,
  failed axis reconstruction, preserved), `AXIS_DIAGNOSIS_V2`, and
  `O1_V2_AXIS_BANK_REDESIGN` — frozen axis package (A1–A4 + Gram-matched
  random bank), calibration cohorts (96 tasks) and 2,400-task confirmatory
  candidate pool, seed matrix, freeze manifest, calibration precommit
  (`e819aeeb…`), external chronology, interpretation addendum, and the
  append-only `ATTEMPT_LEDGER.jsonl`.
- **`o1_failure_snapshots/`, `o1_diagnostics/`** — preserved failures and
  diagnostics; nothing is deleted, everything is hashed.
- **`o1_b200/`** — the complete hardware-independent execution stack
  (three backends behind one sealed contract, batched engine, non-O1
  validation corpus, equivalence/benchmark harnesses, frozen selection
  policy, budget watchdog, zero-touch state machine, deploy environment,
  runbook) with its full local test/equivalence/failure-injection reports and
  the preserved, aborted 403-row 2026-07-31 calibration attempt
  (`NO ROWS REUSED`).  From v0.3.x the target is a single **preemptible**
  accelerator — B300 primary, B200 an explicit honest fallback — acquired
  through the pinned RunPod GraphQL spot surface, with eviction treated as a
  first-class lifecycle state (row-boundary resume for O1, atomic-checkpoint
  resume for the Foundation Learner) inside one hard dollar budget.
- **`foundation_learner/`** — the Foundation Learner campaign that runs on the
  same rented accelerator *after* O1 closes: session supervisor, ladder,
  ecology/episode generation, evaluation and training code, with a hard
  isolation guard that refuses every O1 root.  It runs inside the O1 container
  and borrows its venv; from image v0.3.1 its source is baked into that image
  so a combined session can actually start.  The pregenerated episode corpus
  is not in Git — it is fetched on the pod and every shard is re-hashed
  against `SHARD_SUMS.json` before the ladder begins.
- **`reports/mpn/`** — M+N backbone-training stage reports (S1 baseline,
  S3/S3b closure); the probe *code* lives in Branching-Looped-Transformer.
- **`reports/branch_training/`** — branch-training-logic expansion and
  offline verifier/generator v2 reports (toward model-internal branching).

## Status (2026-08-18)

- O1 v2.1 design is sealed and externally precommitted; the first real
  calibration attempt was operator-stopped at 403/4608 rows and formally
  **aborted for backend redesign** (ledgered; no rows reused).
- The runner is **B300 PREEMPTIBLE SOFTWARE COMPLETE / HARDWARE
  UNVALIDATED**. v0.2.0 added the RunPod pre-rental stack: the production REST
  API v2 adapter (pinned OpenAPI snapshot, typed fail-closed models),
  transport-level read-only enforcement, the eleven-condition live-mutation
  interlock, redundant termination (adapter + independent watchdog process),
  decimal budget engine (USD 45/40/5) and mock-server contract matrix.
  v0.3.1 migrates the target to a preemptible B300 (B200 explicit fallback)
  over a pinned GraphQL spot surface — REST v2 exposes no interruptible
  capacity at all — rebuilds the container on torch 2.12.1+cu130 / CUDA 13.0
  (the cu128 reference build cannot launch on current hosts), repairs a
  credential-isolation defect that let an operator key reach mock hosts,
  delivers the Foundation Learner into the image, and proves the single
  `HF_TOKEN` in both directions before any multi-gigabyte fetch. 256/256
  local checks pass across 19 modules, plus 1700 in the Foundation Learner
  suite. No pod created, no cloud spend; **no GPU has ever executed this
  stack**, and real calibration and confirmatory generation have **not**
  run.
- No scientific claim about oracle reachability is made in this repository:
  the calibration and confirmatory phases have not been executed.

## Provenance

The externally verified commitment chronology of record for the O1 v2.1
precommit was made on the `o1-v2-axis-bank-redesign` /
`o1-v2-b200-runner` branches of
[Hidden-State-Evaluator](https://github.com/VykosMolt/Hidden-State-Evaluator)
(precommit sha256 `e819aeeb…` committed at `5469cab7`, remote-verified by
blob re-fetch). This repository carries byte-identical copies of those sealed
artifacts; every package directory ships its own `SHA256SUMS`.

## License

Apache-2.0 (see `LICENSE`), matching the sibling repositories.
