# B300 Access Runbook

**Documentation only. Do not execute.** GPU access does **not** currently
exist; no pod is running; no spend is authorized. This runbook states the
exact order of the later, separate access task. The later run must require
no source-code development and no interactive debugging.

Budget authority: hard USD 45/40/5 (total/compute ceiling/reserve), enforced
by the session-cumulative `SpendTracker`, no extension, no override, no
auto-reload, `confirmation_authorized: false`.

Steps requiring **explicit user authorization** are marked **[USER AUTH]**.

1. **Provider chosen and frozen: RunPod / Pod / Secure Cloud / exactly one
   GPU / purchase mode INTERRUPTIBLE (spot).** Primary profile NVIDIA B300
   SXM6 AC (Blackwell Ultra, sm_103, CC 10.3, 288 GB HBM3e); explicit
   fallback profile NVIDIA B200 (sm_100, CC 10.0, 180 GB HBM3e), selected
   only when B300 is refused, with the refusal reason recorded in the
   quote/report — never silent. There is no static price cap; the live
   secure spot quote (bid) is authoritative and is refused only by the
   mechanical budget-viability rule (`MIN_VIABLE_SESSION_SECONDS=7200` at
   `MAX_COMPUTE_USD=40`, an effective ceiling near $20.00/h — at live rates
   ~5.07 h of runtime on B300 and ~5.89 h on B200). See
   `provider/runpod/policy.py`. The session config's `profile_preference`
   field (default `["B300", "B200"]`) only orders which profile is tried
   first — both remain committed by the rental authorization regardless of
   order, so it is not a new capability. Set `["B200", "B300"]` to prefer
   the B200, e.g. when B300 demand makes it unobtainable; a first-choice
   B200 is then reported as `operator_preference_applied: true`, not as a
   fallback, and a `fallback_reason` still appears only when an
   earlier-preference profile was actually refused.
2. **DONE: the production RunPod adapter is implemented and
   mock-contract-tested.** Spot acquisition uses the pinned GraphQL surface
   (`provider/runpod/graphql_spot.py`, contract pinned in
   `provider/runpod/openapi/GRAPHQL_SPOT_CONTRACT.json`) because the REST v2
   API has no interruptible surface. All other lifecycle operations
   (status, logs, billing, stop, terminate/DELETE) stay on the pinned REST
   v2 contract (re-pinned 2026-08-17); full test suite in
   `tests/test_runpod_*.py`. Remaining live validation is GET-only:
   `scripts/runpod_pre_rental_readonly_check.sh` once `RUNPOD_API_KEY`
   exists.
3. **Obtain a live secure spot quote and verify it against the USD 45/40/5
   budget** via the mechanical budget-viability rule
   (`MIN_VIABLE_SESSION_SECONDS=7200` at `MAX_COMPUTE_USD=40`). Populate the
   session config's unresolved fields (schema
   `o1b300.runpod_session_config.v2`).
4. **Stage artifacts before billing starts** wherever the provider allows
   (object storage upload etc.). Artifacts and hashes:
   `deploy/TRANSFER_MANIFEST.json`. The checkpoint travels out of band; it is
   never in Git.
5. **[USER AUTH] Provision exactly one GPU (spot).** Start both watchdogs
   (in process + external) immediately at the billing start reference.
6. **Verify GPU, environment, artifacts.** The hardware gate
   (`deploy/hardware_gate.py`) enforces identity/CC/HBM/BF16/driver/arch and
   runs representative real workloads (BF16 GEMM, Ouro-RLTT
   forward/generation, backward+optimizer, O1 intervention hook + transport
   capture, checkpoint save/load) before anything scientific;
   `validate_environment.py`; `verify_artifacts.py`; fill
   `ENVIRONMENT_REPORT.template.json` (transformers must equal 4.54.1 exact;
   eager attention only, no FP8/FP4/quantization/vLLM/TensorRT-LLM/SGLang;
   deterministic flags; BF16; single GPU; host driver ≥ r580 / CUDA 13
   runtime).
7. **Run the non-O1 equivalence suite** (`compare_o1_backends.py`):
   REFERENCE_SERIAL vs B200_REPLICA workers 2/4/8/12/16 (where memory
   allows) and B200_BATCHED batch 1/2/4/8/16/32/64 (where memory allows), on
   the validation corpus only. Report levels A (structural, must be exact),
   B (numerical, frozen tolerances), C (stochastic trajectory identity).
   Structural equivalence (level A) is measured for EVERY benchmark
   configuration (per `config_id`), not per backend; a configuration
   without its own equivalence verdict is ineligible for selection — so the
   deepest batch, which selection would otherwise prefer on throughput, can
   never be chosen on a verdict measured for a different batch size. Never
   downgrade a failure.
8. **Run the bounded non-O1 benchmark** in the frozen staged order
   (`policies/BENCHMARK_ORDER.json`): serial → replica 2/4/8 → batched
   4/8/16; extensions only under the predeclared cost rule; stop on OOM or
   integrity failure; stop launching at 95% budget.
9. **Apply the frozen backend-selection policy**
   (executable form `runner/selection.py`). Eligibility gates first; then
   highest verified rows/hour; deterministic tie-breaks. No O1 outcome may
   influence selection.
10. **Populate the final calibration precommit** — hardware facts only
    (including which profile, B300 or B200, was actually provisioned and,
    if B200, the recorded B300 refusal reason); everything scientific is
    already frozen. Finalization refuses unresolved fields
    (`runner/precommit_template.py`). The sealed-format precommit is minted
    ONCE per session and reused by every later pod from the durable store,
    because the sealed record sink binds every row to the precommit's
    SHA-256 and refuses to mix bindings; per-pod hardware commitments are
    stored under per-digest keys so an earlier pod's commitment is never
    overwritten.
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
    aborted 2026-07-31 laptop attempt). Because this is spot/interruptible
    capacity, eviction is expected and normal, not a failure: the pod goes
    EXITED (stopped) with roughly 5 seconds of SIGTERM notice; the session
    terminates the remnant, carries spend forward in the session-cumulative
    `SpendTracker`, re-quotes fresh (profile re-selected, B300 first, B200
    fallback if refused), and reacquires — bounded by the hard USD 45/40/5
    budget, by `max_pod_creations` in the v2 rental authorization (schema
    `o1b300.rental_authorization.v2`, template
    `B300_RENTAL_AUTHORIZATION.template.json`), by
    `MAX_POD_ACQUISITIONS=4`, and by a zero-progress repeat-failure guard.
    Committed O1 rows and the records file stream continuously to the
    result destination (private HF repo or local store,
    `runner/durability.py`); on reacquisition the sealed orchestrator
    resumes from the next missing canonical row. FL checkpoints mirror the
    same way.
15. **Verify and transfer records** (full recomputation; canonical merge;
    `package_outputs.sh`; hash-verify after download).
16. **Terminate the GPU pod and confirm termination** (billing off is
    confirmed independently, not assumed).
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
