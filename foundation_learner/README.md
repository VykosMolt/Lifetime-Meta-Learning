# FOUNDATION_LEARNER_B200_V0 0.1.0

Status: **PRE_RENTAL_BUILD** — software complete, hardware unvalidated.

The complete first B200 campaign for the Foundation Learner / persistent
adaptation research programme (Pilot 0), built so that when the accelerator is
available the remaining work is *execution*, not design or software
development.

Research question:

> Can Ouro-RLTT be trained over complete learning histories so that it becomes
> better at **learning** a previously unseen task family from attempts and
> feedback, rather than merely becoming better at the training tasks
> themselves?

The primary object is the LEARNING DYNAMICS (the trajectory R_0..R_K), not
static accuracy. The strongest evaluation is a whole-generator-family holdout.
A null result is useful; no success narrative is pre-written.

## This package is NOT

- **not** B200 validated, benchmarked, or throughput-characterised;
- **not** a campaign that has run — no arm, no evaluation, and no sealed
  opening has produced a scientific result;
- **not** rental-authorized: RunPod rental confirmation remains
  **NOT AUTHORIZED** and cloud spend for the work that produced this package is
  **USD 0.00**;
- **not** a claim about "recursive self-improvement" or a "generally
  self-improving system" — that vocabulary is forbidden everywhere in this
  programme, in code, docs and reports;
- **not** an O1 modification: nothing under `o1_b200/`, `o1_packages/` or
  `o1_runs/` is read, written, imported, or changed by this campaign.

The local validation (`scripts/run_all_tests.py`) runs on the TINY
NONSCIENTIFIC Ouro model with tiny pools, a stub O1 command and a mocked clock.
It says nothing about hardware and nothing about the research question.
`scripts/local_smoke_test.py` is the only path that touches the real 2.6 B
checkpoint; it is explicitly nonscientific, 15-minute watchdog-limited, and
produces `SMOKE_REPORT.json` marked `NONSCIENTIFIC`.

## Layout

| path | contents |
|---|---|
| `docs/IMPLEMENTATION_CONTRACT.md` | the FROZEN contract; implementation follows it exactly |
| `docs/CONTRACT_AMENDMENTS.md` | every recorded deviation, with its reason |
| `FOUNDATION_LEARNER_V0_PREREGISTRATION.md` | the preregistration |
| `FOUNDATION_LEARNER_V0_MANIFEST.template.json` | campaign manifest template (B200-derived fields left UNRESOLVED) |
| `ecology/` | the 12 exact-verifier generator families, the frozen family split, surface remapping, poison, manifests |
| `episodes/` | `EPISODE_STRUCTURE_V0`: event schema, assembly, scripted attempt policies, rendering with exact span maps, the answer grammar |
| `data/` | deterministic pre-generation, plain-JSONL shards + `SHARD_SUMS.json`, sealed-shard enciphering |
| `training/` | frozen backbone loading, tokenisation + loss masks, the arms, objectives, custom LoRA, checkpointing, compute accounting, stability, the tiny model |
| `mechanisms/` | FL4 value head, FL5 fast state, FL6 value gating, FL7 fast adapter, FL8 consolidation |
| `evaluation/` | manual greedy decode, teacher-forced scoring, the online learning-curve walker, context reset, interference, poison, remap, family holdout, the 14 frozen metrics |
| `campaign/` | scheduler + affordability, promotion, dev-grid selection, the sealed gate, O1 isolation, the session supervisor, the stage table, the result verifier, `FL_BUDGET_POLICY.json` |
| `analysis/` | clustered bootstrap statistics and reporting (LOCAL, post-transfer) |
| `deploy/` | pod entry, environment lock, `INTEGRATION.md` |
| `scripts/` | pre-generation, the aggregate test runner, the dress rehearsal, the local smoke test, packaging, manifest filling |
| `tests/`, `tests/hostile/` | the unit suite and the permanent hostile regression fixtures |
| `reports/` | `TEST_REPORT.json`, rehearsal and smoke reports (`reports/local_runs/` is git-ignored) |

## The rules that bind the campaign

- **The model is frozen.** Every arm starts from a fresh load of the same
  checkpoint (`tree sha256 a701f7a7…`); the base artefact is never modified.
- **Exact verifiers only.** No LLM judge anywhere in a principal outcome.
- **The sealed set opens once.** The cipher primitives themselves live in
  `ecology/base.py` (they are what `data/shards.py` enciphers the sealed shards
  with at pre-generation time, and `data/shards.read_shard` will apply an
  explicitly supplied key — Amendments 1, 7 and 8). What is unique is the
  **campaign path**: `campaign/sealed_gate.py` is the only module that DERIVES
  `K_seal` and the only route by which campaign code reads sealed data. It
  refuses until `DEV_DECISIONS_FROZEN.json` exists; the opening is two-phase,
  so the single-use `SEALED_OPENED` entry in the append-only hash-chained
  ledger is written only once the evaluation records exist, and a failed
  attempt is recorded as `SEALED_OPENING_ABORTED` and leaves exactly one retry.
  A second committed opening refuses, results are written read-only, and the
  sealed evaluation runs the promoted arm restored from its recorded
  checkpoint. SEALED_TEST is never consulted for model selection,
  hyperparameters, promotion, checkpoint selection or early stopping — the
  promotion API takes an object that structurally cannot hold a sealed record.
  The protection is PROCEDURAL (gate, ledger, hostile fixtures): the key comes
  from a public digest, and no cryptographic unopenability is claimed.
- **O1 has absolute priority and absolute isolation.** FL runs only after the
  O1 records are verified, transferred and the O1 process closed; every FL path
  is realpath-guarded against the O1 roots.
- **Time is bounded mechanically.** `projected × 1.25 + reserve ≤ remaining`,
  a 1200 s transfer reserve that is never consumed, and a `U` chosen from the
  frozen ladder by measurement — never by outcome.
- **No hyperparameter invention.** The development grid is exactly two frozen
  learning rates on FL3 at 25 % of `U`; the winner is locked for FL1/FL2/FL3.

## Running the local validation

```bash
python -m foundation_learner.scripts.run_all_tests          # unit + hostile + rehearsal
python -m foundation_learner.scripts.run_all_tests --fast   # skips the slow walkers
python -m foundation_learner.scripts.dress_rehearsal        # offline miniature campaign
python -m foundation_learner.scripts.local_smoke_test --tiny  # smoke-test self-test
```

The pre-generated data is not tracked in Git (it is large and bitwise
reproducible); its manifests are, under `artifacts_fl/pregen*/MANIFESTS/`.
Regenerate it with:

```bash
python -m foundation_learner.scripts.pregenerate_all --out artifacts_fl/pregen
```

**Order matters when packaging**: *validate → package → manifest*.

```bash
python -m foundation_learner.scripts.run_all_tests          # writes reports/
python -m foundation_learner.scripts.package_release        # writes SHA256SUMS + zip
python -m foundation_learner.scripts.make_manifest          # fills the manifest LAST
```

The manifest is written **after** the zip because it binds the zip hash, and
`package_release.py` therefore excludes `FOUNDATION_LEARNER_V0_MANIFEST.json`
from the bundle: a file cannot describe the digest of an archive that contains
it. `SHA256SUMS` likewise covers every other bundled file and is written and
added last, so it never covers itself.

The bundle contains only **git-tracked** files under `foundation_learner/` plus
`artifacts_fl/pregen/**` — so a fresh clone of the pushed branch reproduces it
byte for byte. Consequently a release build **refuses a dirty work tree**:
under a tracked-content policy an uncommitted module would be silently missing
from the zip, so commit first (a `--dry-run` build may proceed and says so). `foundation_learner/reports/**` is deliberately *not* in the
zip: those evidence JSONs stay in git (reviewable, history-tracked), while the
zip is the accelerator bundle. That also removes the old ordering trap in which
re-running the validation invalidated a checksum snapshot the zip had already
taken.
