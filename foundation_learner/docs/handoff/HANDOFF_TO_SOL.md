# HANDOFF: Foundation Learner B200 V0 — final phase

You ("Sol", the Codex orchestrating agent) are taking over the FINAL PHASE of
a large, nearly-complete build. The previous orchestrator (Claude) built,
validated, and packaged the campaign through two full independent gate
rounds; you finish the last fix batch, revalidate, re-release, run one
confirmation review, and write the final report. The user directed this
handoff to conserve their Claude usage. All rules below are USER RULES and
bind you and every subagent you spawn.

## 0. Read these first, in order

1. This file.
2. `foundation_learner/docs/handoff/W6_STATE_OF_WORK.md` — exact state of the
   in-flight fix batch you are completing.
3. `foundation_learner/docs/handoff/REVIEWER_CONFIRMATION_N1.md` — the
   findings your fix batch must close (N1 CRITICAL, N2, N3, N4, Defect C).
4. `foundation_learner/docs/handoff/GATE_HISTORY.md` and
   `VERIFIER_CONFIRMATION.md` — what has already been verified (do not redo).
5. `foundation_learner/docs/IMPLEMENTATION_CONTRACT.md` +
   `docs/CONTRACT_AMENDMENTS.md` (Amendments 1–15 exist; you write 16) +
   `FOUNDATION_LEARNER_V0_PREREGISTRATION.md` — the frozen scientific
   contract. NOTHING in them may be weakened.

## 1. Hard rules (non-negotiable)

- NO cloud spend of any kind. NO B200/RunPod rental. NO pod creation.
  Rental confirmation is NOT AUTHORIZED. Cloud spend for this task: USD 0.00.
- NEVER modify anything under `/home/moloch/ouro_worktrees/o1-v2-b200-runner`
  or the canonical repo `/home/moloch/ouro_project` (except read-only use of
  its venv and the frozen checkpoint at
  `/home/moloch/ouro_project/models/ouro_rltt_local`).
- The SEALED_TEST data stays unopened. Never derive the sealed key, never
  read sealed plaintext, never use sealed outcomes for anything.
- Never weaken tests, gates, thresholds, fixtures, or acceptance criteria to
  obtain a pass. Hostile fixtures are permanent. No mocks/placeholders in
  place of capability. Report failures honestly; INCONCLUSIVE stays
  INCONCLUSIVE.
- Work ONLY in `/home/moloch/ouro_worktrees/foundation-learner-b200-v0`
  (branch `foundation-learner-b200-v0`), plus scratch under /tmp.
- Python: `/home/moloch/ouro_project/venv/bin/python` (transformers 4.54.1
  exact). Local GPU (RTX 5070 Ti 12 GB) may be used; the real 2.6B checkpoint
  may be loaded ONLY by `scripts/local_smoke_test.py` and the (≤15 min)
  determinism probe pattern — never a full training arm locally.
- Separation of roles: implementation, verification, and the confirmation
  review must be SEPARATE fresh contexts (spawn subagents / fresh
  `codex exec` sessions). Never let the implementer self-confirm. Do not
  rush a running reviewer; missing output is never approval.
- If a required step fails twice seriously, STOP and write the final report
  with status BLOCKED for that item — do not improvise around it.
- Commits: keep the repo's style (imperative subject + wrapped body). Add
  `Co-Authored-By: Codex <noreply@openai.com>` (attribute yourself honestly;
  do not impersonate the previous orchestrator).

## 2. Remaining work, in order

1. **Complete the N-batch fix** per `REVIEWER_CONFIRMATION_N1.md` §Required
   repair (6 sub-items) + N2 + N3 + N4-recording + Defect C, exactly as
   specified there, starting from `W6_STATE_OF_WORK.md`. Record everything as
   Amendment 16 in `docs/CONTRACT_AMENDMENTS.md` (append-only; supersede
   Amendment 12.16, do not edit it). Include the independent corroboration
   note (1600 cache-incompatibility warnings; two gates converged).
2. **Scoped tests** for every file you touched, then the **full suite**:
   `venv/bin/python -m pytest foundation_learner/tests -q` → must be 0
   failed. Then `python -m foundation_learner.scripts.run_all_tests` (full
   mode) → ALL PASS (its package-checksum suite will fail until step 5 —
   run it before packaging expecting that one suite to be the only failure,
   or run it after step 5; the committed TEST_REPORT.json must come from a
   run where everything passes).
3. **Rehearsal + smoke**: `python -m foundation_learner.scripts.dress_rehearsal`
   → PASS (wall-clock advisory per N3);
   `python -m foundation_learner.scripts.local_smoke_test` on GPU → PASS.
4. **Commit** the fix batch (clean tree required for packaging).
5. **Re-release**: `python -m foundation_learner.scripts.package_release`
   (writes new SHA256SUMS + `FOUNDATION_LEARNER_B200_V0.1.0.zip` + sidecar —
   the hash WILL change; that is correct), then
   `python -m foundation_learner.scripts.make_manifest`, commit the metadata,
   then a full-pass `run_all_tests` and commit its TEST_REPORT.
6. **Push** to `origin foundation-learner-b200-v0`. Note: GitHub reports the
   origin repo was renamed (`ouro_project` → `Hidden-State-Evaluator`); the
   redirect works — record this fact in the final report, do not "fix" it.
7. **Fresh-clone verification**: shallow-clone the pushed branch into /tmp,
   `pregenerate_all` (regenerates all data from code), `package_release` —
   the zip hash must EXACTLY match the released one. Record both hashes.
8. **Confirmation review (required before ACCEPT)**: spawn a FRESH,
   independent session (no shared context with your implementer) whose brief
   is: adversarially verify each of the 6 N1 sub-repairs + N2 + Defect C
   against `REVIEWER_CONFIRMATION_N1.md`, by running its own probes (e.g.
   decode(train-mode) refusal; post-training arm generations == fresh
   eval-mode load; rehearsal log free of cache-incompatibility warnings from
   decode paths; Amendment 16 supersedes 12.16 honestly), plus regression
   spot-checks of the sealed two-phase gate and eval-set family coverage.
   Verdict must be explicit ACCEPT or REJECT with evidence. On REJECT:
   repair, re-run affected gates, re-review (max two repair cycles, then
   BLOCKED).
9. **Final report**: write
   `foundation_learner/docs/handoff/FINAL_REPORT.md` covering, with concrete
   values: source commit(s); branch/worktree; Ouro-RLTT checkpoint identity
   (path + tree hash); the 12 task families; the frozen split; episode
   design (EPISODE_STRUCTURE_V0, K=6); each FL0–FL8 rung and its exact
   implementation; core training objectives (incl. FL3 weight table); FL4
   value-target definition; FL5 fast-state architecture (incl. ON/OFF/ON_S0
   triad); FL7 fast-parameter architecture; FL8 consolidation; the 14 frozen
   + 4 added metrics; development selection policy; promotion rules;
   compute-matching rules; the B200 affordability rule; time-aware ladder
   behavior; all leakage/adversarial checks and gate history (cite the
   handoff gate files); final local test results; smoke result; package
   version + zip hash + SHA256SUMS hash; commit/ref verification incl. the
   fresh-clone hash match and the repo-rename note; the exact unresolved
   B200-only fields (7 of them — see the campaign manifest). Source most
   content from the prereg/contract; keep it factual, no success narrative.
   END the report with EXACTLY this block (choose each value honestly):

```
FOUNDATION LEARNER V0 SOFTWARE: COMPLETE or BLOCKED
EXACT-VERIFIER TASK ECOLOGY: COMPLETE or BLOCKED
TRAIN/DEV/SEALED-TEST FAMILY SPLIT: SEALED or BLOCKED
FL0–FL8 LADDER IMPLEMENTATION: COMPLETE or BLOCKED
LOCAL HOSTILE VALIDATION: PASS or FAIL
OURO-RLTT BACKBONE: HASH-BOUND
B200 HARDWARE EXECUTION: NOT STARTED
FOUNDATION LEARNER TRAINING: NOT STARTED
SEALED TEST OUTCOMES: UNOPENED
O1 CALIBRATION DATA USED BY FOUNDATION LEARNER: NO
CLOUD SPEND FOR THIS TASK: USD 0.00
```

10. **Done marker**: as your VERY LAST action write the single word
    `COMPLETE` or `BLOCKED` to
    `/home/moloch/ouro_worktrees/foundation-learner-b200-v0/SOL_DONE`
    (this file is gitignored-irrelevant; do not commit it). The previous
    orchestrator wakes on this marker and will read FINAL_REPORT.md.

## 3. Facts you would otherwise have to rediscover

- Checkpoint tree SHA-256 (frozen): a701f7a75300ddf57098572fef3894bef59d5179580ec7eae7cd561a36056889.
- Current release (pre-your-fix): zip 439fe915… @ commit e27edd3 — your fix
  obsoletes it; expect a new hash.
- Split manifest digest chain ends at
  6cf330b47af9406edf8022564ea330f3f164fbaef22618710a11fa3b2f40d9b5; the
  split ASSIGNMENT never changed. Full pregen at `artifacts_fl/pregen` is
  current for the committed generators; your fix batch touches no
  generators, so DO NOT regenerate (the fresh-clone step regenerates in the
  clone as verification).
- Full suite baseline before your fixes: 1646 passed / 8 skipped / 0 failed.
- The dress rehearsal writes reports under `foundation_learner/reports/`
  (gitignored `local_runs/`); TEST_REPORT.json and SMOKE_REPORT.json are
  committed evidence.
- `pytest` is in the venv. Suite takes ~5–30 min depending on load.
