# O1_B300_RUNNER v0.3.1

Status: **B300 PREEMPTIBLE SOFTWARE COMPLETE / HARDWARE UNVALIDATED**

The complete hardware-independent execution stack for running the sealed
O1 v2.1 calibration on a future, separately authorized single-GPU RunPod
Secure Cloud INTERRUPTIBLE (spot) session. Primary accelerator profile:
NVIDIA B300 SXM6 AC (Blackwell Ultra, sm_103, compute capability 10.3,
288 GB HBM3e); explicit fallback profile NVIDIA B200 (sm_100, CC 10.0,
180 GB HBM3e), selected only when B300 is refused, with the refusal reason
recorded in the quote/report — never silent. The rental authorization
already commits to BOTH profiles' canonical bodies; the session config's
`profile_preference` (default `["B300", "B200"]`) only orders which one is
*tried first* and can be set to `["B200", "B300"]` when B300 demand makes it
unobtainable — a first-choice B200 is then reported as a deliberate
`operator_preference_applied: true` selection, not a fallback, and dropping
a profile from the list never removes it as a fallback. Built on the verified
scientific package `O1_oracle_reachability_v2.1.0` (imported
byte-hash-verified, never modified). This package is development
infrastructure — it is NOT:

- B300/B200 hardware validated,
- B300/B200 benchmarked,
- calibration ready,
- confirmatory ready,
- zero-touch production ready.

Those statuses require real hardware and a selected provider. Nothing in this
package spends money, contacts a cloud provider, runs a hardware benchmark,
runs real O1 calibration, or runs confirmatory generation. The scientific
design is frozen and untouched (see `docs/ARCHITECTURE.md`).

## Layout

- `runner/` — backends (REFERENCE_SERIAL / B200_REPLICA / B200_BATCHED, dual
  B300-primary/B200-fallback profile aware), batched engine + intervention +
  transport, records, persistence, RNG, corpus, equivalence + benchmark
  harnesses, budget watchdog, provider adapters (pinned GraphQL spot
  acquisition + pinned REST v2 lifecycle), zero-touch state machine,
  templates.
- `policies/` — frozen benchmark order, frozen backend-selection rule, budget
  policy template (USD 45/40/5), environment-report template, calibration-
  precommit template.
- `deploy/` — `Dockerfile.b300`, pinned lock (`requirements.b300.lock`,
  `transformers==4.54.1` exact) + frozen local wheel set
  (`WHEELS_B300.sha256`), entrypoint `start_b300.sh` ->
  `runner/production_entry.py`, environment/artifact verification, transfer
  manifest, output packaging, checksums.
- `corpus/` — non-O1 validation corpus (mechanically disjoint from all O1
  task populations; outcomes never usable for O1 decisions).
- `tests/` — full local suite incl. hostile tests
  (`tests/run_all_b200_tests.py`).
- `docs/` — `B200_ACCESS_RUNBOOK.md` (the exact later order; documentation
  only), `ARCHITECTURE.md`.
- `reports/` — test report, equivalence report, dress-rehearsal report,
  local smoke report.
- `preserved_attempts/` — the hashed, frozen 403-row stopped laptop attempt
  (ABORTED_BEFORE_CALIBRATION_ANALYSIS; NO ROWS REUSED).

## Current validation state

Aggregate runner suite: 228 checks / 0 failed across 18 modules. Foundation
Learner suite: 1662 passed / 0 failed / 8 skipped. Mocked dress rehearsal:
COMPLETE. 21-point failure injection: all-terminal. Master pre-rental
readiness: PASS. Live GET-only preflight: PASS, with B300 and B200 both
showing Low stock (B300 $7.89/h, B200 $6.79/h secure spot; Secure-filtered
minimum bids equal to those figures). Container image
`o1-b300-runner:v0.3.1`, local id
`sha256:37f749847fed72456c95a397ccfb9fc60c92223dfe1cee1c59f911481953dd9b`,
registry digest UNRESOLVED until the operator pushes.

## Base identity

See `BASE_VERIFICATION_REPORT.md`: source commit `6db215f`, package zip
`a8b0571c…`, sealed calibration precommit `e819aeeb…` (externally verified),
axis package `8c3b34a3…`, checkpoint tree `a701f7a7…`.
