# B200 Access Runbook

**Documentation only. Do not execute.** B200 access does **not** currently
exist; no provider is selected; no spend is authorized. This runbook states
the exact order of the later, separate B200-access task. The later run must
require no source-code development and no interactive debugging.

Budget authority: `policies/B200_BUDGET_POLICY.template.json` — USD 45.00
total, USD 40.00 compute ceiling, no extension, no override, no auto-reload,
`confirmation_authorized: false`.

Steps requiring **explicit user authorization** are marked **[USER AUTH]**.

1. **Provider chosen and frozen: RunPod / Pod / Secure Cloud / on-demand /
   exactly one NVIDIA B200 / max USD 5.89/h GPU rate.** See
   `provider/runpod/policy.py`.
2. **DONE (2026-08-01): the production RunPod API v2 adapter is implemented
   and mock-contract-tested** (`provider/runpod/`, pinned OpenAPI snapshot
   `provider/runpod/openapi/`, full test suite in
   `tests/test_runpod_*.py`). Remaining live validation is GET-only:
   `scripts/runpod_pre_rental_readonly_check.sh` once `RUNPOD_API_KEY`
   exists.
3. **Verify the quoted hourly rate against the USD 45 policy.** Populate the
   budget policy's unresolved fields; compute
   `floor(40.00 / rate * 3600)` seconds as the hard runtime limit.
4. **Stage artifacts before billing starts** wherever the provider allows
   (object storage upload etc.). Artifacts and hashes:
   `deploy/TRANSFER_MANIFEST.json`. The checkpoint travels out of band; it is
   never in Git.
5. **[USER AUTH] Provision exactly one B200.** Start both watchdogs (in
   process + external) immediately at the billing start reference.
6. **Verify GPU, environment, artifacts.** `validate_environment.py`;
   `verify_artifacts.py`; fill `ENVIRONMENT_REPORT.template.json`; validate
   via `runner/env_report.py::validate_b200_report` (transformers must equal
   4.54.1; eager attention; deterministic flags; BF16; single GPU).
7. **Run the non-O1 equivalence suite** (`compare_o1_backends.py`, b200
   mode): REFERENCE_SERIAL vs B200_REPLICA workers 2/4/8/12/16 (where memory
   allows) and B200_BATCHED batch 1/2/4/8/16/32/64 (where memory allows), on
   the validation corpus only. Report levels A (structural, must be exact),
   B (numerical, frozen tolerances), C (stochastic trajectory identity).
   Never downgrade a failure.
8. **Run the bounded non-O1 benchmark** in the frozen staged order
   (`policies/BENCHMARK_ORDER.json`): serial → replica 2/4/8 → batched
   4/8/16; extensions only under the predeclared cost rule; stop on OOM or
   integrity failure; stop launching at 95% budget.
9. **Apply the frozen backend-selection policy**
   (`policies/B200_BACKEND_SELECTION_POLICY.json`, executable form
   `runner/selection.py`). Eligibility gates first; then highest verified
   rows/hour; deterministic tie-breaks. No O1 outcome may influence
   selection.
10. **Populate the final B200 calibration precommit** from
    `policies/CALIBRATION_PRECOMMIT_B200.template.json` — hardware facts
    only; everything scientific is already frozen. Finalization refuses
    unresolved fields (`runner/precommit_template.py`).
11. **[USER AUTH] Commit and push the finalized precommit externally.**
12. **Independently verify the remote commit/ref** (fetch the blob back from
    the remote and hash it, as the v2.1 chronology did).
13. **Run the affordability gate**
    (`runner/budget.py::affordability_gate`): projected calibration time
    × 1.30 + verification/transfer reserve + termination reserve must fit in
    the remaining authorized runtime; otherwise the system refuses to start
    calibration and proceeds to verified termination.
14. **Only then run real calibration** (4,608 sealed rows; resumable; sealed
    v2.1 semantics through the selected backend; NO rows reused from the
    aborted 2026-07-31 laptop attempt).
15. **Verify and transfer records** (full recomputation; canonical merge;
    `package_outputs.sh`; hash-verify after download).
16. **Terminate the B200 and confirm termination** (billing off is confirmed
    independently, not assumed).
17. **Perform calibration analysis locally** (sealed
    `calibration_analysis.py`), on the local machine, after the instance is
    gone.
18. **Do not run confirmation.** Confirmatory generation is a separate,
    later authorization; `confirmation_authorized` is frozen false and the
    state machine refuses it.

Failure at any step: the zero-touch state machine aborts forward progress,
preserves completed rows, quarantines invalid ones, terminates the instance,
and writes a machine-readable final status. If termination cannot be
confirmed, the status says so explicitly — that is the one condition that
requires immediate operator attention (a provider-side billing stop).
