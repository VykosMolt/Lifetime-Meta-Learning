# O1 B300 Runner — Architecture

Package: `O1_B300_RUNNER_v0.3.1`, status
`B300 PREEMPTIBLE SOFTWARE COMPLETE / HARDWARE UNVALIDATED`. Targets a
RunPod Pod, Secure Cloud, exactly one GPU, purchase mode INTERRUPTIBLE
(spot): primary profile NVIDIA B300 SXM6 AC (Blackwell Ultra, sm_103, CC
10.3, 288 GB HBM3e), explicit fallback profile NVIDIA B200 (sm_100, CC 10.0,
180 GB HBM3e) when B300 is refused, with the refusal reason always recorded.
The `profile_preference` session-config field (default `["B300", "B200"]`)
only orders which of the two authorization-committed profiles is tried
first — set `["B200", "B300"]` to prefer the B200 (useful when B300 demand
makes it unobtainable) without adding or removing capability; unknown
profile names are refused, and a first-choice B200 is reported as
`operator_preference_applied: true`, not as a fallback.
Separate from the verified scientific package
`O1_oracle_reachability_v2.1.0`, which it imports byte-hash-verified and
never modifies.

## Layering

```
o1_b200/runner/
  sealed_import.py     hash-verified import bridge to the sealed v2.1.0 modules
  identity.py          canonical JSON + domain-tagged SHA-256 (+ sealed digests)
  row_specs.py         immutable RowSpec + canonical enumerations + manifest hash
  row_rng.py           sealed CRN/step-seed mapping, exposed + auditable
  records.py           canonical record schema, build, FULL recomputation
  persistence.py       atomic row commits, quarantine, resume identity, merge
  backend_interface.py Backend ABC + RuntimeConfig + RunBundle
  intervention.py      batched one-shot L3_24 edit (per-row [1,2048] slices)
  engine_batched.py    left-padded batched prefill/decode with sealed sampling
  backends.py          REFERENCE_SERIAL / B200_REPLICA / B200_BATCHED
  synthetic_runtime.py deterministic elementwise test model (sealed call surface)
  validation_corpus.py non-O1 corpus builder + mechanical disjointness proof
  runbuild.py          validation-run bundle construction (non-O1 only)
  compare_o1_backends.py  3-level equivalence harness (A/B/C)
  benchmark_o1_b200.py    frozen-order benchmark harness (local rehearsal only)
  selection.py         executable frozen backend-selection rule
  budget.py            watchdog (95%/100%), affordability gate, mock clock
  provider_adapter.py  adapter contract + Mock + Local (production = refused)
  state_machine.py     zero-touch PRECHECK→…→TERMINATE→COMPLETE
  env_report.py        environment template validation (+ labeled local report)
  precommit_template.py  refuses finalization with unresolved hardware facts
  transfer_package.py  manifests, deterministic archives, SHA256SUMS, secrets deny-list
```

## Sealed semantics — where they live and how they are preserved

| frozen element | authority | runner treatment |
|---|---|---|
| generation loop, sampler, stopping | `run_o1_v2_generation.py` (sealed) | REFERENCE_SERIAL and B200_REPLICA call `generate_branch` unmodified; B200_BATCHED replays the identical per-row semantics and imports the sealed `_sample_top_p`, parser, verifier |
| intervention locus L3/loop-2/layer-24/`layers[23]`, one-shot prefill, final non-padding token | sealed hook | batched hook edits position −1 under a LEFT-padding invariant (asserted), per-row `[1,2048]` slices, identity path for baseline/zero-alpha |
| transport at normalized L4_47 | `o1_transport_v2.py` (sealed) | called per row on per-row boundaries; zero-alpha bitwise assertion intact |
| CRN seeds, Latin square, K=8, alpha grid | `o1_analysis.py` (sealed) | `derive_stream_seed`/`action_index` imported; enumeration mirrors the sealed loops; RowSpec seeds re-verified |
| parser/verifier/R | sealed modules | recomputed at build AND at every resume scan AND at finalize; stored booleans can never override |

## Row-specific randomness

The sealed sampler constructs a fresh generator per token seeded
`seed + step`; the seed is the sealed CRN function of (master_seed, task_id,
stream). Therefore no shared RNG state exists across rows: batch size, worker
count, scheduling, neighbor completion, restart, and resume cannot remap any
row's stream. Arm/direction/sign enter through the sealed Latin-square
action→stream map — CRN pairing across arms is the design and is preserved.
Locally proven: exact stream identity across serial/replica/batched, batch
sizes 1–32, shuffled scheduling, mixed completion lengths, and resume (on the
synthetic runtime, which is constructed to be bitwise batch-invariant).
Cross-GPU token identity is NOT claimed — that is a B300/B200 hardware
equivalence question, addressed on the target profile by the hardware gate
(`deploy/hardware_gate.py`).

## Records and persistence

One canonical schema (strict v2.1 superset; identical scientific field names)
with a domain-tagged `record_hash` over the full record. Per-row atomic
commits (tmp+fsync+rename), append-only attempt log, duplicate-row rejection,
corrupt/stale/foreign-row quarantine, exact resume-identity refusal on 16
fields, deterministic canonical merge in `exec_index` order that refuses gaps.

## Budget derivation

The committed budget is unchanged: USD 45.00 total authorized / 40.00 max
compute / 5.00 reserved non-compute. The obsolete static $5.89/h price cap is
replaced by a mechanical viability rule — a quote is refused only when the
$40 compute allocation cannot buy at least `MIN_VIABLE_SESSION_SECONDS`
(7200 s = 2 h), an effective ceiling near $20.00/h. At the live rates that is
~5.07 h of runtime on B300 ($7.89/h) and ~5.89 h on B200 ($6.79/h). Every
per-pod deadline — the independent in-process watchdog and RunPod's
provider-side `terminateAfter` — is armed from the REMAINING allocation net
of what earlier evicted pods already spent, so an eviction/reacquisition
sequence can never authorize more total compute than the $40 allocation even
if the driving orchestrator dies.

## Billing boundaries

Spend is metered from POD CREATION (RunPod bills from provisioning, so a pod
evicted before RUNNING is not free) and frozen only AFTER termination is
confirmed (a pod keeps billing while terminating).

## HF offline scoping

The pod keeps `HF_HUB_OFFLINE=1` for the whole scientific process so
model/tokenizer loading provably cannot reach the network. Durability,
external commitment, and result transfer reach the Hugging Face hub ONLY
through isolated subprocess helpers (`runner/hf_transfer.py`; the Foundation
Learner package carries its own copy) which are spawned with the offline
flags stripped and refuse loudly if the flag survives into the subprocess.
Every transfer is SHA-256 verified end to end.

## Deployment identity

The artifact source and result destination are part of the deployment
identity committed by the authorization — they decide what the pod fetches
and where it publishes. Launch-variable values (acquired profile, authorized
seconds, hourly rate, `HF_TOKEN`, launch nonce) and the datacenter are NOT
identity and are injected at launch. The deployment hash is re-derived from
the rendering's contents before it is compared with the authorization, so a
forged rendering cannot deploy.

## Sealed precommit across evictions

The sealed-format calibration precommit is minted ONCE per session and
reused by every later pod from the durable store — required because the
sealed record sink binds every row to the precommit's SHA-256 and refuses to
mix bindings. Per-pod hardware commitments are stored under per-digest keys
so an earlier pod's commitment is never overwritten.

## Equivalence per configuration

Structural equivalence (level A) is measured for EVERY benchmark
configuration (per `config_id`), not per backend. A configuration without
its own equivalence verdict is ineligible for selection — so the deepest
batch, which selection would otherwise prefer on throughput, can never be
chosen on a verdict measured for a different batch size.

## Deliberate non-claims

- No B300/B200 hardware validation, benchmark, or backend selection has
  occurred.
- The synthetic-runtime equivalence results prove SOFTWARE correctness only.
- Compaction, torch.compile, CUDA graphs, non-eager attention: ineligible.
- Real O1 calibration/confirmation cannot be launched from this package's
  local modes; the state machine hard-refuses O1 bindings and confirmation.
- The B300 wheel-set embeds no PTX, so silent JIT fallback to a different
  SASS target is impossible; sm_103 native execution rests on NVIDIA's
  documented same-major/higher-minor SASS forward-compatibility rule plus 59
  sm_103a arch-tuned kernels, verified at runtime by the hardware gate
  (`deploy/hardware_gate.py`), not assumed.
