# Lifetime Meta-Learning

Research repository for the **lifetime meta-learning direction** on Princeton
Ouro-RLTT looped transformers: the program that moves from *reading* frozen
loop-state signals to *writing* them — asking whether the model's recurrent
hidden states are a causally reachable substrate that training could
eventually exploit across a model's lifetime.

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
| Agent track | (forthcoming) | Hunter-Seeker will receive its own repository |

## Contents

- **`o1_packages/`** — the sealed O1 implementation lineage:
  `O1_oracle_reachability_v1.5.3` → `v2.0.0` (superseded by audit) →
  `v2.1.0` (verified; zip `a8b0571c…`), plus the B200 runner package
  `O1_B200_RUNNER_v0.1.0` (zip `884041e8…`).
- **`o1_runs/`** — the scientific record: `O1_REAL_001` (v1.5.3 real attempt,
  failed axis reconstruction, preserved), `AXIS_DIAGNOSIS_V2`, and
  `O1_V2_AXIS_BANK_REDESIGN` — frozen axis package (A1–A4 + Gram-matched
  random bank), calibration cohorts (96 tasks) and 2,400-task confirmatory
  candidate pool, seed matrix, freeze manifest, calibration precommit
  (`e819aeeb…`), external chronology, interpretation addendum, and the
  append-only `ATTEMPT_LEDGER.jsonl`.
- **`o1_failure_snapshots/`, `o1_diagnostics/`** — preserved failures and
  diagnostics; nothing is deleted, everything is hashed.
- **`o1_b200/`** — the complete hardware-independent B200 execution stack
  (three backends behind one sealed contract, batched engine, non-O1
  validation corpus, equivalence/benchmark harnesses, frozen selection
  policy, budget watchdog, zero-touch state machine, deploy environment,
  runbook) with its full local test/equivalence/failure-injection reports and
  the preserved, aborted 403-row 2026-07-31 calibration attempt
  (`NO ROWS REUSED`).
- **`reports/mpn/`** — M+N backbone-training stage reports (S1 baseline,
  S3/S3b closure); the probe *code* lives in Branching-Looped-Transformer.
- **`reports/branch_training/`** — branch-training-logic expansion and
  offline verifier/generator v2 reports (toward model-internal branching).

## Status (2026-08-01)

- O1 v2.1 design is sealed and externally precommitted; the first real
  calibration attempt was operator-stopped at 403/4608 rows and formally
  **aborted for backend redesign** (ledgered; no rows reused).
- The B200 runner is **software-complete and hardware-unvalidated**. v0.2.0
  adds the complete RunPod pre-rental stack: the production RunPod REST API
  v2 adapter (pinned OpenAPI snapshot, typed fail-closed models),
  transport-level read-only enforcement, the eleven-condition live-mutation
  interlock, redundant termination (adapter + independent watchdog process),
  decimal budget engine (USD 45/40/5), mock-server contract matrix, and the
  locally built production container (torch cu128 with sm_100, transformers
  4.54.1 exact). 155/155 local checks pass; no pod created, no cloud spend;
  real calibration and confirmatory generation have **not** run.
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
