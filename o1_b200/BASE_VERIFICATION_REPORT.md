# O1 B200 Runner — Base Package Verification Report

Date: 2026-08-01 (UTC)
Task: build the complete O1 v2 B200 execution stack locally
(SOFTWARE CONSTRUCTION AND LOCAL VALIDATION ONLY — no B200 access, no spend).

## Source identity

| item | value |
|---|---|
| canonical repository | /home/moloch/ouro_project |
| durable O1 worktree | /home/moloch/ouro_worktrees/o1-v2-axis-bank-redesign |
| durable worktree HEAD at fork | `6db215fa52ee315ad1011e14cf7a12560107da56` ("pause O1 v2.1 calibration at 403/4608 rows") |
| remote branch at fork | `origin/o1-v2-axis-bank-redesign` = `0526c45919a50f62e8b327dca4e8a2674417203e` (local HEAD is 2 commits ahead: de901b2, 6db215f) |
| new branch | `o1-v2-b200-runner`, created from `6db215f` |
| new worktree | /home/moloch/ouro_worktrees/o1-v2-b200-runner |

Note: the task text referenced `/home/moloch/ouro/worktrees/...`; that path does
not exist. The actual worktree root used by this repository is
`/home/moloch/ouro_worktrees/`, and the new worktree follows that convention.

The durable worktree's tracked files were clean at fork time. The only
untracked content was the stopped calibration attempt's runtime artifacts
(records, boundaries, log, progress), which are preserved — see
`preserved_attempts/calibration_20260731_operator_stop/`.

## Base package identity (most recent verified O1 implementation)

| item | value |
|---|---|
| base package | `O1_oracle_reachability_v2.1.0` |
| package zip SHA-256 | `a8b0571cc67584b862db3e42fb015f47ae7d2887d7ce59605e3c273591fe7892` (verified against `.sha256` sidecar: OK) |
| package internal SHA256SUMS | verified: all files OK |
| source commit | `6db215fa52ee315ad1011e14cf7a12560107da56` |
| calibration precommit SHA-256 | `e819aeebc642fefde769f398b385f23f98dc3980412eeb67274d0f5891c5457d` (externally committed at `5469cab762c52eb33219565c7ec49ba02c2f22c8`, remote-verified per EXTERNAL_CHRONOLOGY ledger entry) |
| axis-package tree hash | `8c3b34a3574fbca9e63de23a88831e076e317fec33236405fb3af72f0ff96fdd` |
| structured tensor SHA-256 (`axes_l3_24.npy`) | `72107b8eec817930652016c768fa90529a5989d82129b3d4a44f8d0b593946db` |
| random tensor SHA-256 (`random_axes_l3_24.npy`) | `1ff455edd3f2982ea45b6c11085887e46ece392e4e5eafab7916262f6fd7660b` |
| checkpoint tree hash (`models/ouro_rltt_local`) | `a701f7a75300ddf57098572fef3894bef59d5179580ec7eae7cd561a36056889` |
| transformers version (canonical venv, asserted by sealed code) | `4.54.1` (exact) |
| torch / numpy / python (canonical venv) | 2.12.0.dev20260407+cu128 / 2.4.4 / 3.14.6 |

## Hostile-audit mechanical fixes — presence confirmed

All v2.1 audit fixes are present in the base package and covered by its tests:

1. **Strict final-answer parsing, no A/B/C prefix collision** — `o1_answer_parser_v2.py`
   grammar v2.1 accepts only a complete answer line (`FINAL ANSWER: A|B|C` or
   `<A>|<B>|<C>`, optional single period); atomic group prevents the
   backtracking stall; 58/58 adversarial parser fixtures pass.
2. **generated_text in the sealed record** — orchestrator `_score_fields`
   stores `generated_text`, `generated_text_sha256`, `generated_token_ids`,
   `generated_token_ids_sha256`.
3. **Parser/verifier/R recomputation from generated text** —
   `verify_stored_scoring`, `verify_token_text_binding`,
   `verify_record_binding.py`; stored booleans never override recomputation.
4. **Sealed generation-to-record orchestration** — `run_o1_v2_orchestrator.py`
   is the only producer of analysis-valid rows.
5. **Sealed transport-rho computation** — `o1_transport_v2.py` (frozen L4_47
   semantics, zero-alpha bitwise assertion, finite/nonnegative requirements).
6. **Deterministic post-calibration cohort construction** —
   `build_confirmatory_cohort.py` + cohort builder fixtures (17/17 pass).
7. **Medium-band upper edge [0.55, 0.70)** — `calibration_analysis.py` lines
   10–12 and band code; half-open on both strata.
8. **Explicit infeasible budget-MDE handling** — `budget_mde` returns `None`
   surfaced as `NO_DETECTABLE_EFFECT_AT_G` (`o1_analysis.py`,
   `calibration_analysis.py:746`).
9. **Interpretation addendum** — `O1_V2_INTERPRETATION_ADDENDUM.md` present in
   the run directory; sha256 bound in the attempt ledger
   (`f0e2279159a387e5db2ad680dc6da9fd1a41418a5feffec4942e6ea46cf239cd`).
10. **Durable runtime-artifact path handling** — `RUNTIME_ARTIFACT_PATHS.json`
    lives in the durable worktree run directory (not /tmp) and every artifact
    is hash-verified by `verify_artifact_paths` at orchestrator start.

## Inherited test suite

`run_all_tests.py` executed in the canonical venv on 2026-08-01:
**ALL O1 v2.1 CHECKS PASSED** (JSON templates; 12 CLI parsers; 53 confirmatory,
24 calibration, 15 integration, 7 CLI-workflow, 58 parser-adversarial,
15 orchestrator, 17 cohort-builder fixtures; axis verifier self-test 15/15;
exact McNemar planning minima 324/817/2238/337; no stale labels; no bytecode).

## Verdict

The B200 runner is built on `O1_oracle_reachability_v2.1.0` at source commit
`6db215f` — the most recent verified implementation including every hostile-audit
mechanical fix. The known-defective v2.0.0 package is not used. The B200 runner
imports the sealed v2.1.0 modules **byte-hash-verified at import time**
(`runner/sealed_import.py`); it never copies or re-implements their scientific
behavior.
