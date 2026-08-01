# O1_B200_RUNNER v0.1.0

Status: **B200_SOFTWARE_COMPLETE_HARDWARE_UNVALIDATED**

The complete hardware-independent execution stack for running the sealed
O1 v2.1 calibration on a future, separately authorized single-B200 session.
Built on the verified scientific package `O1_oracle_reachability_v2.1.0`
(imported byte-hash-verified, never modified). This package is development
infrastructure — it is NOT:

- B200 validated,
- B200 benchmarked,
- calibration ready,
- confirmatory ready,
- zero-touch production ready.

Those statuses require real hardware and a selected provider. Nothing in this
package spends money, contacts a cloud provider, runs a B200 benchmark, runs
real O1 calibration, or runs confirmatory generation. The scientific design
is frozen and untouched (see `docs/ARCHITECTURE.md`).

## Layout

- `runner/` — backends (REFERENCE_SERIAL / B200_REPLICA / B200_BATCHED),
  batched engine + intervention + transport, records, persistence, RNG,
  corpus, equivalence + benchmark harnesses, budget watchdog, provider
  adapters, zero-touch state machine, templates.
- `policies/` — frozen benchmark order, frozen backend-selection rule, budget
  policy template (USD 45/40/5), environment-report template, B200
  calibration-precommit template.
- `deploy/` — Dockerfile, pinned lock (`transformers==4.54.1` exact),
  entrypoint, environment/artifact verification, transfer manifest, output
  packaging, checksums.
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

## Base identity

See `BASE_VERIFICATION_REPORT.md`: source commit `6db215f`, package zip
`a8b0571c…`, sealed calibration precommit `e819aeeb…` (externally verified),
axis package `8c3b34a3…`, checkpoint tree `a701f7a7…`.
