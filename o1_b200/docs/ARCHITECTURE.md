# O1 B200 Runner — Architecture

Package: `O1_B200_RUNNER_v0.1.0`, status
`B200_SOFTWARE_COMPLETE_HARDWARE_UNVALIDATED`. Separate from the verified
scientific package `O1_oracle_reachability_v2.1.0`, which it imports
byte-hash-verified and never modifies.

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

| frozen element | authority | B200-runner treatment |
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
Cross-GPU token identity is NOT claimed — that is a B200 equivalence question.

## Records and persistence

One canonical schema (strict v2.1 superset; identical scientific field names)
with a domain-tagged `record_hash` over the full record. Per-row atomic
commits (tmp+fsync+rename), append-only attempt log, duplicate-row rejection,
corrupt/stale/foreign-row quarantine, exact resume-identity refusal on 16
fields, deterministic canonical merge in `exec_index` order that refuses gaps.

## Deliberate non-claims

- No B200 hardware validation, benchmark, or backend selection has occurred.
- The synthetic-runtime equivalence results prove SOFTWARE correctness only.
- Compaction, torch.compile, CUDA graphs, non-eager attention: ineligible.
- No production provider adapter exists (interface + mock + local only).
- Real O1 calibration/confirmation cannot be launched from this package's
  local modes; the state machine hard-refuses O1 bindings and confirmation.
