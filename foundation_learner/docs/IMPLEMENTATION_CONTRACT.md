# FOUNDATION LEARNER B200 V0 — FROZEN IMPLEMENTATION CONTRACT

Status: FROZEN before implementation. Workers implement exactly this. Deviations
require a contract amendment recorded in `docs/CONTRACT_AMENDMENTS.md` with
reason; no silent drift.

This is a BUILD / VALIDATION / PACKAGING / PRE-STAGING task.
NO B200 rental. NO RunPod pod. NO real campaign. NO real O1 calibration.
NO cloud spend. Cloud spend for this task: USD 0.00.

## 0. Objective

Build the complete first B200 campaign for the Foundation Learner / persistent
adaptation research programme, so that when the accelerator is available the
remaining work is execution, not design or software development.

Research question (Pilot 0):

> Can Ouro-RLTT be trained over complete learning histories so that it becomes
> better at learning a previously unseen task family from attempts and
> feedback, rather than merely becoming better at the training tasks
> themselves?

Primary object: LEARNING DYNAMICS (the trajectory R_0..R_K), not static
accuracy. Strongest evaluation: whole-generator-family holdout.

This is Foundation Learner Pilot 0. A null result is useful. No success
narrative is pre-written. Never call any result "recursive self-improvement" or
the model "a generally self-improving system".

## 1. Frozen backbone binding (MODEL IS FROZEN)

| item | value |
|---|---|
| checkpoint path (local canonical) | `/home/moloch/ouro_project/models/ouro_rltt_local` |
| checkpoint tree SHA-256 (O1 `sha256_tree` convention) | `a701f7a75300ddf57098572fef3894bef59d5179580ec7eae7cd561a36056889` |
| matches O1 v2.1 binding | YES — byte-identical base shared with the O1 programme |
| architecture | `OuroForCausalLM` via `auto_map` (`trust_remote_code` local files) |
| model_type | `ouro` |
| physical hidden layers (`num_hidden_layers`) | 48 |
| recurrent-loop configuration (`total_ut_steps`) | 4 |
| hidden_size / heads / kv heads | 2048 / 16 / 16 |
| intermediate_size | 5632 |
| vocab_size | 49152 |
| max_position_embeddings | 65536 |
| dtype | bfloat16 (`torch_dtype: bfloat16`) |
| early_exit_threshold | 1.0 |
| RLTT config keys | `rltt_logprob_chunk_size: 2048`, `rltt_loop_level_checkpointing: true` |
| config.json SHA-256 | `7d6764dbc8210d023c8d83da4620910808ac5a450532b15550e57d1ef0e4f741` |
| tokenizer.json SHA-256 | `fcb808fe5e7642f5299be28aea07fc7f6d4f4364c3ac5e408e15a772cbc8fa8d` |
| tokenizer_config.json SHA-256 | `7936619f224ec48539a38f8d7dbc64b3c1ba397f4c4b753397d52116cf71dcd8` |
| vocab.json SHA-256 | `7b9de3f47796abf8d00ab96be299fea0dc9afdf1827f34e7e0b9fb44593efe5c` |
| merges.txt SHA-256 | `0b54e8aa4e53d5383e2e4bc635a56b43f9647f7b13832d5d9ecd8f82dac4f510` |
| special_tokens_map.json SHA-256 | `aadabc9bd7e3f4738bc1160ef0aa932ba09401c1f11271bd269a21ba987b353b` |
| configuration_ouro.py SHA-256 | `4c7c6138715351f7b673eed4a8e7553ccccb8a1f1ab93e82e9773ad96c6ee7d6` |
| modeling_ouro.py SHA-256 | `bcd27ff6a18578feaec168695d70dc76509c57e4d4c377347c9d05d6266d9e82` |
| chat_template.jinja SHA-256 | `609d03c963c8d8d0519c9212df62280bc6dd561013259c3b6cd0ee43f7910f58` |
| model.safetensors.index.json SHA-256 | `e1842668e1ba1568364a4ae7227a5ab80ab5403c95baba37b92205d2cb22a001` |
| transformers | `4.54.1` EXACT (assert at runtime; hard fail otherwise) |
| torch (canonical local venv) | `2.12.0.dev20260407+cu128` |
| peft / accelerate / numpy | `0.19.1` / `1.13.0` / `2.4.4` |
| local venv | `/home/moloch/ouro_project/venv` |

Forbidden: any other checkpoint (no other Ouro, no smaller proxy for science,
no Huginn/Qwen/Gemma, no other recurrent backbone, no API model as the trained
learner). Every training arm starts from a fresh load of this identical frozen
checkpoint. The base checkpoint artifact is never modified. No O1 intervention
axis, O1 calibration result, or O1-altered model state enters training.

`sha256_tree` convention (identical to O1 `deploy/verify_artifacts.py`):
sorted `os.walk`, per file update `relpath.replace(os.sep,'/')` UTF-8 bytes,
`\0`, raw digest bytes of file SHA-256, `\n`.

## 2. Repository facts

- Canonical repo: `/home/moloch/ouro_project` (dirty; DO NOT work there).
- Work happens ONLY in worktree `/home/moloch/ouro_worktrees/foundation-learner-b200-v0`,
  branch `foundation-learner-b200-v0`, forked from `main` @ `1b858d0`.
- Push remote: `origin` = `https://github.com/VykosMolt/ouro_project.git`.
- O1 B200 runner (sealed; read-only reference): worktree
  `/home/moloch/ouro_worktrees/o1-v2-b200-runner`, branch `o1-v2-b200-runner`
  @ `add95c8`. NOTHING under `o1_b200/`, `o1_packages/`, `o1_runs/` may be
  modified by this campaign. The FL package must integrate with the existing
  B200 container/runtime WITHOUT changing the O1 scientific package.
- All new code lives under `foundation_learner/` in the FL worktree.
- Historical OPI/O1 artifacts are never modified.

## 3. Package layout (frozen)

```
foundation_learner/
  __init__.py  VERSION(=0.1.0)  STATUS
  FOUNDATION_LEARNER_V0_PREREGISTRATION.md
  FOUNDATION_LEARNER_V0_MANIFEST.template.json
  ecology/
    __init__.py  base.py  surface_remap.py  poison.py  split.py  manifests.py
    families/__init__.py + 12 family modules (section 4)
  episodes/
    __init__.py  schema.py  assemble.py  attempt_policy.py  render.py  parse.py
  data/
    __init__.py  generate_shards.py  shards.py  pools.py
  training/
    __init__.py  tokenization.py  arms.py  trainer.py  objectives.py
    peft_modes.py  compute_accounting.py  checkpointing.py  model_loading.py
    tiny_model.py  stability.py
  mechanisms/
    __init__.py  value_head.py  fast_state.py  value_gating.py
    fast_adapter.py  consolidation.py
  evaluation/
    __init__.py  fl0_base.py  learning_curve.py  context_reset.py
    interference.py  poison_eval.py  remap_eval.py  family_holdout.py
    metrics.py
  campaign/
    __init__.py  scheduler.py  promotion.py  dev_selector.py  sealed_gate.py
    o1_isolation.py  session_supervisor.py  stage_definitions.py
    result_verifier.py  affordability.py
  analysis/
    __init__.py  stats.py  report.py
  deploy/
    fl_b200_entry.sh  environment_lock.json  INTEGRATION.md
  scripts/
    run_all_tests.py  package_release.py  make_manifest.py
    pregenerate_all.py  local_smoke_test.py  dress_rehearsal.py
  tests/
    test_*.py
    hostile/test_hostile_*.py
  docs/
    IMPLEMENTATION_CONTRACT.md (this file)  CONTRACT_AMENDMENTS.md
```

Python: stdlib + numpy + torch + transformers==4.54.1 + peft + safetensors
only. No new third-party dependencies. All randomness from explicit
`numpy.random.Generator(numpy.random.PCG64(seed))` or `torch.Generator` with
derived seeds; NEVER global RNG, NEVER wall-clock in scientific paths.

Determinism/hash conventions: reuse O1 conventions —
`canonical_json(obj) = json.dumps(obj, sort_keys=True, separators=(",",":"), ensure_ascii=False, allow_nan=False)`;
`domain_sha256(domain, obj) = sha256(domain.encode() + b"\0" + canonical_json(obj).encode())`;
`sha256_file` (1 MiB chunks); `sha256_tree` (section 1). Implement once in
`foundation_learner/ecology/base.py` (module `flhash` functions) and import
everywhere. Seeds for substreams: `derive_seed(root_seed, *tags) =
int(domain_sha256("FL_V0_SEED", [root_seed, *tags])[:16], 16) % 2**63`.

## 4. Task ecology — 12 exact-verifier generator families (frozen)

Every family module implements the `TaskFamily` interface in `ecology/base.py`:

```python
class TaskFamily:
    family_id: str                    # frozen id below
    generator_version: str            # "1.0.0"
    def sample_rule(self, rng) -> Rule                    # fresh latent rule
    def rule_spec(self, rule) -> dict                     # exact serializable latent-rule spec
    def sample_instance(self, rule, rng, difficulty) -> TaskInstance
    def related_instance(self, rule, rng, difficulty) -> TaskInstance   # same rule, new instance
    def transfer_instance(self, rule, rng, difficulty) -> TaskInstance  # same rule, shifted distribution
    def verify(self, rule, instance, answer_text) -> VerifierResult     # EXACT; no LLM judge ever
    def structured_feedback(self, rule, instance, answer_text, rng) -> str  # informative, does NOT reveal the answer
    def corrupted_feedback(self, rule, instance, answer_text, rng) -> str   # poisoned variant (env-flagged)
    def surface_remap(self, rule, instance, remap_id) -> TaskInstance   # semantics-preserving resurface
    def difficulty_levels(self) -> list[int]              # >= 3 levels
```

`TaskInstance`: `{family_id, generator_version, rule_id, seed, difficulty,
surface_map_id, prompt_text, answer_canonical, instance_id}` where
`rule_id = domain_sha256("FL_V0_RULE", rule_spec)` and
`instance_id = domain_sha256("FL_V0_INSTANCE", {family_id, rule_id, seed,
difficulty, surface_map_id, prompt_text, answer_canonical})`.

`VerifierResult`: `{correct: bool, canonical_answer: str, parsed: str|None}`.
Verification = strict grammar parse of a final `ANSWER: <payload>` line
(grammar in `episodes/parse.py`: the LAST line matching
`^ANSWER:\s*(\S.*?)\s*$`; family-specific canonicalization of `<payload>`,
e.g. whitespace/case normalization declared per family), then exact match
against `answer_canonical`. Mirrors the O1 lesson: strict grammar, no
substring credit.

The 12 families (frozen `family_id`s — the split rule in section 5 operates on
exactly these strings):

1. `boolean_rule` — hidden Boolean function over k∈{3,4,5} variables as a
   bounded DNF (2–4 terms, term width ≤3). Task: evaluate on an assignment.
   Answer: `0|1`. Structured feedback: identifies one variable whose flip
   changes the output for the queried assignment (never states the output).
2. `propositional_transform` — hidden transformation on propositional
   formulas: one of a parameterized family of rule-sets (e.g. swap ∧/∨ under
   negation, distribute, push negation with variable-specific exceptions).
   Task: given formula, output transformed formula string. Feedback: first
   syntax-tree position where the attempt diverges from the target.
3. `modular_arithmetic` — hidden f(x)=(a·x²+b·x+c) mod m (degree ∈{1,2},
   m∈{7,11,13,17,23}). Task: compute f(x). Feedback: sign/absolute residue
   distance band between attempt and truth mod m (never the residue itself).
4. `sequence_transform` — hidden composition of 2–3 ops from {reverse,
   rotate_k, swap_adjacent_pairs, drop_every_k, duplicate_first, map_symbol}
   applied to symbol sequences. Task: transform a sequence. Feedback: length
   correctness + first mismatched position.
5. `string_rewrite` — hidden set of 2–3 non-overlapping rewrite productions
   (pattern→replacement over a small alphabet) applied leftmost-innermost for
   a fixed number of passes. Task: rewrite a string. Feedback: number of
   correctly rewritten positions (Hamming-style count on aligned prefix).
6. `finite_state_transducer` — hidden Mealy machine (3–5 states, alphabet
   size 3–4, output alphabet 2–4). Task: transduce an input string.
   Feedback: longest correct output prefix length.
7. `permutation_composition` — hidden permutation σ ∈ S_n (n∈{5,6,7}) given by
   generator names; tasks: apply σ or σ∘σ to a list of symbols. Feedback:
   number of fixed points the attempt got right (count only).
8. `set_operations` — hidden set expression over named base sets (depth ≤3
   over ∪, ∩, ∖ with 2–4 base sets). Task: output the resulting set
   membership for a queried universe (sorted element list). Feedback:
   symmetric-difference cardinality between attempt set and truth.
9. `graph_edge_semantics` — small digraph whose edge labels carry a hidden
   semantics (e.g. label X means traversable, Y blocked, Z reverses
   direction; assignment hidden). Task: is node b reachable from a?
   Answer `YES|NO`. Feedback: the true out-degree (under hidden semantics) of
   one named node on the queried path (never the reachability bit).
10. `dsl_execution` — hidden short program (3–5 instructions) in a fixed tiny
    register DSL {INC r, DEC r, ADD r s, SWAP r s, JNZ r offset(bounded)}
    executed with bounded steps on 2–3 registers. Task: given initial
    registers, output final registers. Feedback: which register(s) are
    incorrect (names only, not values).
11. `constraint_rules` — hidden pairwise constraint template over 3–5 slots
    (hidden comparator set, e.g. slot_i < slot_j, slot_k ≠ v). Task: decide
    whether a full assignment satisfies the hidden constraints
    (`SAT|UNSAT`). Feedback: the number of violated constraints (count only).
12. `grammar_classification` — hidden regular language over {a,b,c} from a
    parameterized family (e.g. "even count of a AND ends with b", conjunction
    of 2 atomic predicates from a pool of 8). Task: classify string
    `IN|OUT`. Feedback: a length-preserving property hint (which atomic
    predicate family the string fails, by opaque predicate index).

These are genuinely distinct computational mechanisms — NOT formatting
variants: truth-table evaluation, tree rewriting, modular algebra, sequence
ops, string rewriting, automata transduction, group action, set algebra,
graph traversal under hidden semantics, program execution, constraint
checking, language membership.

Difficulty controls: each family exposes ≥3 integer difficulty levels
(sizes/depths/lengths as parameterized above). Default campaign difficulty:
level 1 (middle) for all arms; frozen.

Surface remapping (`surface_remap.py`, applied via family `surface_remap`):
deterministic from `remap_id`: symbol/vocabulary renaming (bijective token
maps), label permutation where semantics allow (e.g. 0/1 → F/T with legend in
prompt so semantics preserved), format changes (separator/prefix variants),
irrelevant distractor sentences (from a fixed pool, inserted at deterministic
positions), reordered description clauses. Latent rule unchanged. Remapped
instances must verify identically after inverse-mapping the answer
(`surface_remap` returns instance with its own `answer_canonical` in the
remapped surface; verifier operates in the remapped surface).

Structured feedback must NEVER contain the canonical answer string for the
pending item (hostile-tested). Correctness-only feedback: `CORRECT` /
`INCORRECT` exactly.

Poison (`poison.py`): a corrupted variant of structured feedback (wrong
position/count/hint computed by a deterministic corruption of the true
feedback, guaranteed ≠ true feedback content). Presentation channels
(section 6) guarantee: certified channel is NEVER corrupted; poison appears
only in the uncertified HINT channel. The environment records
`poison=True` internally; the model is never told.

## 5. Family split — deterministic public rule (frozen BEFORE computation)

Domain string: `FOUNDATION_LEARNER_V0_FAMILY_SPLIT`.
For each of the 12 frozen `family_id` strings:
`h(fid) = sha256(b"FOUNDATION_LEARNER_V0_FAMILY_SPLIT\0" + fid.encode())`
hex digest. Sort the 12 families by `h(fid)` ascending (lexicographic hex).
Positions 0–5 → TRAIN (6), positions 6–8 → DEVELOPMENT (3), positions 9–11 →
SEALED_TEST (3). No manual placement, no re-rolls, no post-hoc adjustment.
`ecology/split.py` implements this; `manifests.py` writes
`family_split_manifest.json` with the hashes, ordering, assignment, generator
source hashes (sha256 of each family module file), and
`split_manifest_sha256 = domain_sha256("FL_V0_SPLIT_MANIFEST", manifest)`.

Zero overlap requirements (validated + hostile-tested):
- family-level: a family is in exactly one split;
- exact-task: no `instance_id` appears in more than one of train/dev/test
  shard sets;
- latent-rule: no `rule_id` is shared across splits (fresh rules per episode;
  collision check enforced at generation).

SEALED_TEST family shards are sealed (section 12) and never opened during
model selection, hyperparameter choice, promotion, checkpoint selection, or
early stopping.

## 6. Episode format (frozen: `EPISODE_STRUCTURE_V0`)

Event roles (enum in `episodes/schema.py`):
`TASK, SUPPORT_OBS, MODEL_ATTEMPT, FEEDBACK, HINT, REVEAL, ADAPT_EVENT,
RELATED_TASK, QUERY_TASK, TRANSFER_TASK`.
Nothing is flattened into undifferentiated next-token data: every event
carries `{role, text, item_id, interaction_index, certified: bool,
poison: bool (env-only), reveal: bool}` and the renderer
(`episodes/render.py`) produces both the text and an exact character/token
span map from events to positions (used for loss masks and tap points).

Feedback channels:
- `FEEDBACK` (certified): verifier-computed, ALWAYS truthful. Correctness-only
  (`CORRECT`/`INCORRECT`) or certified structured feedback.
- `HINT` (uncertified): structured feedback content; may be poisoned in poison
  conditions. Truthful hints and poisoned hints are rendered identically
  (modulo content), so poison is not detectable from the channel tag.
- `REVEAL` (certified): the correct answer for a SUPPORT item only; present
  only in conditions that explicitly define supervised feedback (FL7 inner
  loop; optional reveal step in the standard episode — see structure below).
  Hidden QUERY/TRANSFER labels are NEVER exposed to the model.

Standard episode (one latent rule per episode; interaction indices frozen):

```
 idx  event
  -   TASK P1 (problem statement, support problem 1)
  0   MODEL_ATTEMPT on P1            -> R_0
  -   FEEDBACK(P1 attempt0) [+ HINT in structured conditions]
  1   MODEL_ATTEMPT on P1 (revision) -> R_1
  -   FEEDBACK(P1 attempt1)
  -   TASK P2 (support problem 2, same rule)
  2   MODEL_ATTEMPT on P2            -> R_2
  -   FEEDBACK(P2 attempt0) [+ HINT]
  3   MODEL_ATTEMPT on P2 (revision) -> R_3
  -   FEEDBACK(P2 attempt1) [+ REVEAL(P1,P2) only in reveal conditions]
  -   RELATED_TASK P3 (same rule, fresh instance)
  4   MODEL_ATTEMPT on P3            -> R_4
  -   FEEDBACK(P3)
  5   QUERY_TASK Q1..Q3 (fresh instances; answered without further feedback)
                                     -> R_5 = mean success over Q1..Q3
  6   TRANSFER_TASK T1..T2 (shifted distribution) -> R_6
```

K = 6 (interaction indices 0..6). Configurable in code
(`EpisodeStructure` dataclass) but FROZEN at these values for every arm of
this campaign. Rendering: plain-text protocol with fixed markers
(`TASK:`, `ATTEMPT:`, `FEEDBACK:`, `HINT:`, `REVEALED ANSWER (P#):`,
`RELATED TASK:`, `QUERY:`, `TRANSFER:`) and answers as final
`ANSWER: <payload>` lines. NO chat template (deterministic, tokenizer-simple;
the RLTT chat template is not used in this campaign).

Training histories are OFF-POLICY and pre-generated: model attempts in
training data are produced by frozen scripted attempt policies
(`episodes/attempt_policy.py`), parameterized per family with error models:
attempt0 wrong with p=0.7 (drawn corruption of the true answer from a
family-specific plausible-error sampler), revision wrong with p=0.25, related
wrong with p=0.30; queries/transfers carry TRUE answers in the target slots
(used or masked per arm). Deterministic from seed. This is a declared v0
design decision (no on-policy rollout generation in v0 training data).

Evaluation episodes run ONLINE: real greedy model generations at each
MODEL_ATTEMPT slot (max_new_tokens 64, temperature 0 / greedy, frozen), exact
verifier computes real feedback, and R_k is measured per interaction index.

## 7. The ladder FL0–FL8 (frozen definitions)

Common: every independent arm starts from a fresh load of the frozen base
checkpoint. Root seed 20260809 (primary); second predeclared seed 20260810
(run only if affordable). Sequence budget: max_seq_len 2048 tokens per
episode rendering (generation-side truncation forbidden — episodes are sized
to fit; assembly asserts fit).

- **FL0** — base model, no training. Learning-curve eval on DEVELOPMENT
  families (online, structured-feedback condition and correctness-only
  condition), plus context-reset eval. Establishes R_0, spontaneous
  improvement from context alone, baseline transfer. Not a treatment.
- **FL1** — static baseline. Data: isolated (TASK → ANSWER) pairs drawn from
  the same TRAIN-family pools (support/related/query problem statements as
  independent items, no history, no feedback). Loss on answer tokens only.
- **FL2** — successful-history imitation. Data: complete successful episodes
  (scripted policies constrained so final revisions and queries are correct).
  Objective: next-token NLL on ALL MODEL_ATTEMPT answer tokens (attempts,
  revisions, related, queries, transfers) — pure behavior cloning of a
  successful learner; no future-competence weighting.
- **FL3** — ordered-feedback meta-training (CORE TREATMENT). Data: complete
  ordered episodes INCLUDING imperfect attempts and real (scripted) feedback.
  Objective `L_meta`: weighted NLL restricted to answer-token spans with
  frozen weights — QUERY 1.0, TRANSFER 1.0, revision attempts (index 1,3)
  0.25, attempt0 spans 0.0, all TASK/FEEDBACK/HINT/REVEAL context 0.0
  (masked). Per-episode weighted token-mean, then batch mean. This is
  explicitly direct optimization of post-feedback future competence; no
  "learning-progress reward" novelty is claimed.
- **FL4** — future-competence value head. Head: MLP on stop-gradient hidden
  state — input = final-layer, final-loop hidden state at the last token of
  feedback/hint item j; arch Linear(2048→256) + GELU + Linear(256→1),
  bf16→fp32 head. Target (frozen): realized learning value
  `y_j = [mean per-token log-likelihood of correct QUERY answers with item j
  present] − [same with item j ablated from the context]`, computed with the
  final FL3 checkpoint, teacher-forced, TRAIN families only. Candidate
  actions supported: use item A / use item B / skip redundant / update from
  informative / reject corrupted — realized as presence/ablation variants
  pre-generated per episode. Loss: pairwise within-episode ranking hinge
  (primary) + MSE on z-scored targets (auxiliary, weight 0.5). DEV eval:
  Spearman, pairwise ranking accuracy, calibration slope/intercept, top-1
  regret vs oracle selection, vs random, vs surface heuristics (item length,
  item type, lexical overlap). NOT a correctness predictor of the current
  answer.
- **FL5** — persistent fast state. `s ∈ R^1024`, init 0 at episode start.
  Update on each FEEDBACK/HINT/REVEAL event: `u_j = W_in · mean(final-layer,
  final-loop hidden states over item-j tokens)` (2048→1024);
  `s ← GRUCell(u_j, s)`. Injection: `P = reshape(W_p · LayerNorm(s), (8,
  2048))` → 8 prefix vectors prepended via `inputs_embeds`; per-vector norm
  clamped to ρ = 2 × median base input-embedding row norm (computed once from
  the frozen checkpoint; recorded constant). Fully differentiable (BPTT
  within episode), resettable (`s=0`), checkpointed, auditable (GRUCell
  equations), ≪ 2.6B (≈ 2048·1024 + 3·(1024²+1024·1024) + 1024·8·2048 ≈ 25M
  params), no access to hidden test labels. Arms: `FAST_STATE_ON` vs
  `FAST_STATE_OFF` — identical FL3 objective, data order, seeds; OFF disables
  injection AND state usage (no prefix). Context-reset: textual history
  removed, s retained, query presented with s-derived prefix only.
- **FL6** — value-gated fast adaptation. Gate: frozen FL4 head applied to
  item j; incorporate (apply the FL5 state update / FL7 inner update) iff
  `v̂_j > 0`; else skip. Threshold 0 FROZEN before sealed evaluation.
  Compare unconditional vs gated under poison-laden episodes.
- **FL7** — fast parameter adaptation (later rung; runs only if justified and
  affordable). Bounded resettable low-rank fast adapter: LoRA r=8, α=16,
  dropout 0, on ALL attention `q_proj` and `v_proj` Linear modules of the 48
  physical layers (exact module names bound from the loaded model; recorded
  in manifest). Inner loop: per feedback-supervised SUPPORT item, 4 SGD steps
  lr 1e-3 on NLL of the REVEALED support answer; total fast-delta Frobenius
  norm clipped to β=1.0; per-episode reset to zero; deterministic checkpoint
  format; base checkpoint artifact untouched; persists across context reset;
  ungated and value-gated variants. FL7 does not replace FL5 — different
  claims.
- **FL8** — consolidation (prepared, not required). Sequence: fast-adapt on
  family A episodes → fast-adapt on family B → consolidate: merge selected
  per-episode fast deltas (selection = value gate; comparisons: none /
  indiscriminate all / value-gated) into a separate slow LoRA bank with scale
  η=0.5 → clear fast state → test A, B, and unrelated families. Metrics:
  retained competence, interference, unrelated degradation, A→B→A recovery.
  Only counts as consolidation because it consumes the selected learning
  history/fast deltas; plain fine-tuning is not consolidation.

## 8. Core comparison + compute matching (frozen)

Headline: FL3 vs FL1 vs FL2 on whole-family-held-out learning curves.
FL4–FL8 are mechanistic extensions and can never replace a failed core
comparison in the headline.

Compute matching policy (declared: option B — normalize and report):
all core arms (FL1, FL2, FL3) run the SAME number of optimizer updates U and
the SAME per-update token budget (`max_tokens_per_batch`, packed episodes /
items), i.e. FLOP-matched at equal updates. Loss-token counts intrinsically
differ across objectives; they are RECORDED and REPORTED, never hidden.
`training/compute_accounting.py` records per arm: optimizer updates, training
(loss) tokens, forward tokens, backward tokens, wall time, GPU seconds,
examples, episodes. U is a B200-derived mechanical quantity (throughput ×
allocation via the affordability rule); everything else is frozen.

Trainable-parameter modes: `PEFT_MODE` (LoRA r=16, α=32, dropout 0.0, on all
attention q_proj/v_proj + MLP down_proj; predeclared) and `FULL_MODEL_MODE`.
The SAME mode for all of FL1/FL2/FL3 (never mixed). Selection is mechanical
(section 11 affordability rule), never outcome-based. FL4–FL8 add only their
declared small modules.

Optimizer (frozen): AdamW β=(0.9, 0.95), eps 1e-8, weight_decay 0.01, cosine
schedule, 3% warmup, grad-norm clip 1.0, bf16 compute with fp32 master where
standard.

Development grid (frozen; committed before accelerator access):
- PEFT_MODE LRs: {1e-4, 3e-4}; FULL_MODEL_MODE LRs: {1e-5, 3e-5}.
- One interaction horizon (EPISODE_STRUCTURE_V0). One optimizer family. One
  scheduler family.
- Grid runs on FL3 only (2 configs, shortened predeclared step count = 25% of
  core U). Selection rule: higher DEVELOPMENT macro-AULC; tie → lower LR.
  The winning LR is then used for FL1, FL2, FL3 core runs (same scope).
  Locked after selection; no new candidates after seeing dev performance;
  SEALED_TEST opened once at the end.

## 9. Evaluations (all exact-verifier; frozen)

- Learning-curve harness (`learning_curve.py`): online episodes, R_0..R_6 per
  episode; per-family aggregation.
- Context-reset (`context_reset.py`) — LOAD-BEARING: (1) run learning
  interactions; (2) let the mechanism update; (3) REMOVE textual learning
  history (verified empty prompt prefix except current query + any s-derived
  prefix); (4) related query; (5) measure retained improvement. FL0/FL3
  history-only behavior is expected to collapse; FL5/FL7 tested for retained
  gains. Hostile test asserts no textual leakage into the reset prompt.
- Interference (`interference.py`): deterministic A→B→A episode chains;
  retention ratio, interference cost, recovery interactions;
  family-balanced reporting.
- Poison (`poison_eval.py`): conditions {correct-informative, correct-
  redundant, irrelevant, partially-misleading, corrupted}; certified channel
  never corrupted; environment tracks poison flags; measures blind
  incorporation (FL3), predicted value (FL4), gating behavior (FL6), fast-
  param protection (FL7).
- Surface-remap (`remap_eval.py`): learn under one surface, evaluate
  queries/transfers under remapped surfaces; latent rule fixed.
- Whole-family generalization (`family_holdout.py`): reports UNSEEN-INSTANCE
  (known generator) and UNSEEN-FAMILY (held-out generator) SEPARATELY;
  instance-level generalization is never described as transferable learning.

Metrics (`metrics.py`, frozen): (1) macro-AULC = mean over families of mean
verifier success over interaction indices 0..6; (2) ΔAULC vs FL1; (3) ΔAULC
vs FL2; (4) R_0; (5) R_K; (6) improvement slope (OLS over interaction index);
(7) interactions-to-threshold (first index with family success ≥ 0.5, where
identifiable); (8) related-task transfer (R_4); (9) whole-family transfer;
(10) context-reset persistence (retained fraction of in-context gain);
(11) A→B→A retention/interference; (12) poison robustness (gap poisoned vs
clean AULC); (13) remap robustness (gap remapped vs canonical AULC);
(14) FL4/FL6 value ranking/calibration/regret. Macro (family-level) reporting
everywhere; task/family-clustered uncertainty (section 14); no single-family
domination of aggregates.

## 10. Promotion ladder (frozen rules)

Stage order: FL0, FL1, FL2, FL3, FL4, FL5, FL6, FL7, FL8. Not all must run.
`campaign/promotion.py` implements:
- FL3 → extensions when: training stable (no unresolved stability events) AND
  DEV macro-AULC(FL3) ≥ DEV macro-AULC(FL1) + 0.02 AND positive mean
  improvement slope evidence (slope > 0 on DEV macro curve), not merely R_0
  gain.
- FL4 runs when FL3 finished and ≥ 200 realized episodes with ≥ 2 scoreable
  feedback items each exist for target computation.
- FL5 runs whenever the core comparison is complete and scheduler admits it.
- FL6 requires FL4 DEV pairwise ranking accuracy ≥ 0.55.
- FL7 requires scheduler admission AND stable fast-update implementation
  (local dress-rehearsal flag) — independently predeclared, may run even if
  FL5 shows null, but is skipped if FL6-dependent gating is its only variant
  remaining and FL4 failed.
- FL8 requires evidence that persistent adaptation exists at all (FL5 or FL7
  context-reset persistence > 0 with CI excluding 0 on DEV).
- SEALED_TEST is NEVER used for promotion.
Fallbacks (frozen): FL3 null on dev → complete matched baselines, run second
predeclared seed if affordable, run frozen context/surface/feedback
diagnostics; never invent objectives live. FL4 null → preserve null, skip
FL6, proceed FL5. FL5 fail → skip dependent parts of FL6; FL7 only as
independently predeclared. FL7 fail → skip FL8. All remaining time goes to
PREDECLARED work only.

## 11. Time-aware scheduler + affordability (frozen)

`campaign/scheduler.py` accepts `available_foundation_learner_seconds` as
input; NEVER assumes ownership of the whole rental. O1 has absolute
priority; FL runs only after O1 close (section 13). Frozen constants:
SAFETY_FACTOR = 1.25; FINAL_TRANSFER_RESERVE = 1200 s; checkpoint cadence =
every 600 s or 200 optimizer steps (whichever first), atomic (tmp + fsync +
rename). Admission rule: start stage only if
`projected_stage_seconds × 1.25 + remaining_reserve ≤ remaining_authorized`.
Projections come from the measured throughput benchmark (stage `BENCH`, runs
first, on a non-evaluation training shard). Minimum-runtime table per rung
derived mechanically from BENCH.

Affordability rule (frozen): FULL_MODEL_MODE is eligible iff
`projected_core_comparison_seconds(FULL, U_min=600 updates/arm, grid +
3 arms + evals) × 1.25 + FINAL_TRANSFER_RESERVE ≤
available_foundation_learner_seconds`; otherwise PEFT_MODE for all core arms.
U itself = largest step count from the frozen ladder
{600, 1200, 2400, 4800} updates/arm that fits the same inequality (mechanical).

Session priority within FL allocation: 1 runtime/throughput validation (BENCH),
2 FL0, 3 FL1/FL2/FL3 core matched comparison, 4 FL4, 5 FL5, 6 FL6, 7 FL7,
8 FL8, 9 additional predeclared seeds. Core comparison outranks every later
rung. USD 45 RunPod budget remains authoritative; rental confirmation remains
NOT AUTHORIZED; transfer/termination reserve is never consumed.

## 12. Sealed-test access control (frozen)

Sealed shards are stored ENCIPHERED: bytes XOR a SHA-256 counter keystream
with key `K_seal = sha256(b"FL_V0_SEALED_KEY\0" + split_manifest_sha256_hex)`
(derivable at pre-generation time; see Amendment 1). The decipher path lives
ONLY in `sealed_gate.py`, which refuses to derive/apply `K_seal` unless
`DEV_DECISIONS_FROZEN.json` exists and its hash has been written to the
opening ledger. Therefore no package code path can read sealed data before
the development decisions are frozen. `sealed_gate.py`:
(a) loader refuses SEALED families without an unlock record; (b) unlock
writes an append-only `SEALED_OPENING_LEDGER.jsonl` entry (dev decisions
hash, stage states, timestamp); (c) single opening — a second unlock refuses;
(d) results written to immutable (read-only chmod + hash-ledgered) files.
No model modification may be justified from sealed outcomes; a future cycle
requires a NEW sealed set.

## 13. O1 isolation + session supervisor (frozen)

Directories on the accelerator: `/workspace/o1_calibration/` (O1; exact O1
result roots bound from the O1 transfer manifests at supervisor start) and
`/workspace/foundation_learner/` (FL). `campaign/o1_isolation.py`: every
input path is `os.path.realpath`-resolved and REFUSED if under any O1 root
(config-listed + discovered from O1 manifests). O1 outputs never imported as
features, targets, task selection, or training data. FL never reads O1
calibration outcomes, transport results, difficulty selection, or analysis.
After O1 closes, FL starts from a FRESH reload of the pristine checkpoint
(tree-hash verified at load), never from an O1-mutated process state.

`campaign/session_supervisor.py` (state machine; does NOT modify the sealed
O1 runner — it invokes the existing O1 zero-touch entry as an opaque
subprocess): START_SESSION → RUN_O1_CALIBRATION → O1_HALT_OR_COMPLETE →
VERIFY_O1_RECORDS → TRANSFER_O1_RECORDS → CLOSE_O1_PROCESS →
RELOAD_PRISTINE_OURO (tree-hash check) → COMPUTE_REMAINING_AUTHORIZED_TIME →
RUN_FL_LADDER → CHECKPOINT_AND_VERIFY → TRANSFER_FL_ARTIFACTS →
TERMINATE_ACCELERATOR. FL never starts before O1 artifacts are transferred
and verified. O1 analysis is not performed on the accelerator beyond its
sealed pipeline. Each transition writes a fsync'd journal record; crash →
resume at last completed state; the two workloads share only checkpoint,
runtime, and accelerator.

## 14. Statistics (local, post-transfer; frozen)

`analysis/stats.py`: per-family means; macro averages; clustered bootstrap
(resample families with replacement, then episodes within family; 10,000
replicates; identical resample indices across arms for paired differences);
learning curves; retention/interference; value-head rank metrics; poison and
remap robustness. The accelerator produces only raw generations, checkpoints,
and exact verifier records; all statistics run locally after termination.

## 15. Checkpoint policy (frozen)

Every arm checkpoint records: base checkpoint tree hash, training config
hash, optimizer state (core arms), trained module / full-model state, stage,
step count, token count, RNG states (numpy + torch + python), family-split
hash, training-shard hashes, code commit, environment digest. Atomic writes.
Keep: initial, best-DEV (frozen rule: max DEV macro-AULC at scheduled evals),
final, pre-promotion. No redundant multi-GB checkpoint spam that endangers
transfer reserve (cadence in section 11).

## 16. Data pre-generation (local, before rental)

`scripts/pregenerate_all.py` produces, shards (JSONL + zstd? NO — plain JSONL
+ per-shard SHA-256 in `SHARD_SUMS.json`; no new deps), and checksums:
episode manifests, exact labels, verifier metadata, surface-remapped
variants, poisoned-feedback schedules, train/dev/test episode seeds,
algorithmic baseline histories, FL4 ablation variants. Pool sizes (frozen):
TRAIN 2,000 episodes/family (12,000 total), DEV 300/family, SEALED 300/family,
plus per-episode remap and poison variants, and 200 A→B→A chains per ordered
family pair used by interference eval (DEV pairs). Training shards may be
reused per frozen sampling policy (uniform without replacement per epoch,
reshuffle by derived seed per epoch). DEV/SEALED episodes immutable. B200
never generates ordinary data.

## 17. Stability monitoring (frozen)

`training/stability.py` detects: NaN/Inf loss, grad-norm explosion (>50 for
3 consecutive steps), optimizer overflow, loss divergence (>2× trailing
median for 200 steps), memory growth, OOM, throughput collapse (<40% of
BENCH for 5 min), corrupted checkpoint (hash fail on load), repeated zero
grad, fast-state norm explosion. Bounded automatic responses: exactly one
retry from last valid checkpoint for transient infrastructure error; NO
hyperparameter improvisation after scientific training begins; scientific
instability → record failure and move on per scheduler.

## 18. Local validation + hostile suite (required before packaging)

Unit tests: generator determinism (bitwise repeat), verifier correctness
(positive/negative/adversarial answers), split disjointness, task-content
disjointness, sealed-access denial, episode assembly, feedback ordering,
surface remap semantic preservation, poison schedule, metric computation,
all trainers on TINY synthetic Ouro model (`training/tiny_model.py`: tiny
OuroForCausalLM built from the checkpoint's own configuration/modeling code,
random init, hidden 64 / 2 layers / 2 ut steps), value-head loss, fast-state
update/reset, context-reset persistence machinery, fast-adapter reset/norm
bound, gating, consolidation mechanics, A→B→A evaluator, checkpoint/resume
(bitwise RNG restoration), scheduler admission math, promotion rules,
sealed-opening gate, O1-isolation refusal, result hashing, supervisor state
machine (with a stub O1 subprocess).

Hostile suite (`tests/hostile/`) — every attack from the task list becomes a
permanent regression fixture, including: family leakage, test-task in train
shard, dev selector reading sealed data, feedback containing hidden answer,
task-ID collisions, remap leakage, poisoned item labeled certified, static
arm receiving history, imitation arm receiving future-query objective, FL3
reducing to static examples (mask check), value head seeing future answers,
fast state surviving episode reset, fast state leaking across episodes,
context reset preserving text, fast adapter not reset / exceeding norm bound,
value gate using ground-truth future results, consolidation reading sealed
performance, checkpoint selection using sealed data, hidden unequal compute,
different starting checkpoints between core arms, promotion from sealed
outcomes, FL runner reading O1 paths, supervisor starting FL before O1
transfer, hard-reserve violation.

Dress rehearsal (`scripts/dress_rehearsal.py`): full miniature campaign
(tiny model, tiny pools, seconds-scale budgets) through the real supervisor,
scheduler, trainers, evaluators, packaging — end to end, offline.

`LOCAL_NONSCIENTIFIC_OURO_SMOKE_TEST` (`scripts/local_smoke_test.py`): real
checkpoint, local GPU (RTX 5070 Ti 12 GB), ≤ 15 minutes wall-clock enforced:
load + tree-hash verify, one forward, one greedy generation on a
non-evaluation synthetic example, one PEFT optimizer step on a throwaway
shard, finiteness asserts, parse round-trip. NOT a training arm; produces
`SMOKE_REPORT.json` marked NONSCIENTIFIC.

## 19. Packaging + git (frozen)

Deterministic package `FOUNDATION_LEARNER_B200_V0.1.0.zip` (fixed mtimes
1980-01-01, sorted entries, no compression timestamps) + `.sha256` sidecar +
internal `SHA256SUMS` over all package files; built by
`scripts/package_release.py`. Environment lock (`deploy/environment_lock.json`)
pins transformers 4.54.1, torch, peft, numpy, python, CUDA, and binds the O1
B200 container reference (Dockerfile.b200 digest) — the FL package RUNS IN
the existing O1 container and changes nothing about the runtime it borrows:
no package installation, no dependency change, no venv change. From image
`o1-b300-runner:v0.3.1` the FL *source* is baked at
`/opt/foundation_learner/foundation_learner` (one `Dockerfile.b300` COPY,
recorded as `foundation_learner_source_sha256`), because an image without it
refused every combined session at the O1→FL handover with exit 78. The
~480 MB `artifacts_fl/pregen` corpus is still not baked: `fetch_pregen.py`
materialises it on the pod and re-hashes every shard first. Commit on
`foundation-learner-b200-v0`; push to origin; independently verify remote ref
and package hash from a fresh clone/fetch. No modification of O1/OPI
artifacts anywhere in history.

## 20. Forbidden shortcuts (non-negotiable)

- No LLM judge for principal outcomes; exact verifiers only.
- No weakening tests/gates/thresholds to obtain a pass; hostile fixtures are
  permanent.
- No mocks/hard-coded outputs/silent fallbacks standing in for capability.
- No sealed-test peeking anywhere; no promotion from sealed outcomes.
- No mixing PEFT/FULL across core arms; no unequal hidden compute.
- No new checkpoint, no proxy model in scientific paths (tiny model is for
  mechanics tests only and is clearly non-scientific).
- No hyperparameter invention outside the frozen grid.
- No wall-clock/global-RNG nondeterminism in scientific data paths.
- No work in the dirty canonical repo; no O1 package modification.
- No B200/RunPod/cloud spend of any kind in this task.

## 22. Runtime integration facts (bound from O1-runner survey; frozen)

Container: the FL package runs INSIDE the existing O1 B200 container
(`o1-b200-runner:v0.2.0`; base `python:3.14-slim-bookworm` by digest; venv
`/opt/venv`; torch `2.12.0.dev20260408+cu128` cu128 wheel + triton 3.7.0;
build-asserted transformers `4.54.1`, numpy `2.4.4`; env
`TRANSFORMERS_OFFLINE=1`, `HF_HUB_OFFLINE=1`,
`CUBLAS_WORKSPACE_CONFIG=":4096:8"`, `PYTHONDONTWRITEBYTECODE=1`; no open
ports; container disk 60 GB; checkpoint mounted at
`/artifacts/ouro_rltt_local`, never baked). The FL package changes NOTHING in
the image. Torch note: local venv is `dev20260407`, image is `dev20260408`
(documented O1 pin drift); FL asserts transformers exactly and RECORDS torch,
tolerating the 0407/0408 split (environment_lock.json lists both with roles).

NO peft on the B200 (not in `requirements.b200.lock`): all LoRA-like
functionality is implemented in `training/lora.py` (custom, tested; peft
0.19.1 is used LOCALLY ONLY as a numerical oracle in unit tests, lazily
imported, skipped when absent). See Amendment 3.

Frozen backbone loading recipe (`training/model_loading.py`):
`AutoTokenizer/AutoModelForCausalLM.from_pretrained(checkpoint_dir,
trust_remote_code=True, local_files_only=True, torch_dtype=bfloat16,
low_cpu_mem_usage=True, attn_implementation="eager")`; then set
`model.config.total_ut_steps = 4`, `model.config.early_exit_threshold = 1.0`
(and `model.model.total_ut_steps = 4`); assert
`transformers.__version__ == "4.54.1"` (hard fail); optional tree-hash
verification against §1. Eval paths additionally set
`torch.use_deterministic_algorithms(True)`, cudnn deterministic. Training
uses gradient checkpointing + `enable_input_require_grads()`.

Generation for evaluation: NEVER `model.generate()`; a manual left-padded
greedy decode loop (O1 engine pattern; known `UniversalTransformerCache`
batched-decode mask hazard). Gate: batched greedy must be exactly equivalent
to unbatched greedy on the tiny model; if the gate fails at any point, eval
falls back to unbatched decode (throughput recorded, correctness preserved).

Accelerator directory layout: FL owns `/workspace/foundation_learner/` (runs,
outputs, pylib). O1-forbidden roots (frozen defaults for `o1_isolation.py`,
plus any roots discovered from O1 manifests at runtime): `/outputs`,
`/opt/o1_b200`, `/artifacts/AXIS_PACKAGE_V2`, `/artifacts/COHORTS`,
`/artifacts/calibration_seed_matrix.json`, `/artifacts/policies`,
`/workspace/o1_calibration`, plus the O1 results HF repo. Allowed shared
READ-ONLY inputs: `/artifacts/ouro_rltt_local`,
`/artifacts/TOKENIZER_BINDING.json`. Everything resolved via realpath before
use; mutation of shared inputs forbidden.

Session topology (frozen): the O1 zero-touch rental as currently sealed
terminates its pod unconditionally after O1 transfer, freezes its pod spec
hash into the rental authorization, and exposes no channel for a second
process. Therefore the combined two-workload session REQUIRES a new
session-level configuration whose pod entry is the FL
`session_supervisor` (§13), which invokes the O1 pod-side entry as an OPAQUE
configured command (`o1_entry_command`), waits for its declared completion
artifacts, verifies + transfers + closes O1, and only then runs the FL
ladder. The supervisor copies patterns from the O1 provider machinery but
never edits O1 files; O1 sealed modules are never imported by FL (the two
hash conventions are re-implemented identically instead — 23 lines of
stdlib). Pre-existing recorded gap, NOT repaired here (out of scope): the O1
package's own pod entrypoint `start_b200.sh` is currently a refusing stub;
`o1_entry_command` is therefore an operator-bound field in the session
config, marked UNRESOLVED in the manifest template. FL's budget module
(`campaign/scheduler.py` + its own policy file `FL_BUDGET_POLICY.json`) is
separate from O1's `runner/budget.py` (which hard-asserts O1's exact 45/40/5
policy and cannot be reused); the USD 45 total session budget remains
authoritative at session level; rental confirmation remains NOT AUTHORIZED.

Packaging conventions mirrored from O1: deterministic zip (sorted entries,
ZipInfo date_time=(1980,1,1,0,0,0), external_attr 0o644<<16, ZIP_DEFLATED 9,
secret-pattern deny-list), SHA256SUMS exact-coverage (bijection, no
bytecode), single-line VERSION/STATUS, machine-readable JSON with
`"schema": "flb200.<name>.v1"` keys written `indent=2, sort_keys=True` +
trailing newline, `.gitignore` for `reports/local_runs/`.

## 23. Frozen inter-module interfaces (workers implement/consume EXACTLY)

```python
# episodes/schema.py (W1)
class Role(str, Enum): TASK SUPPORT_OBS MODEL_ATTEMPT FEEDBACK HINT REVEAL ADAPT_EVENT RELATED_TASK QUERY_TASK TRANSFER_TASK
@dataclass Event: role; text: str; item_id: str; interaction_index: int|None; certified: bool; poison: bool; reveal: bool; meta: dict
@dataclass Episode: episode_id; family_id; split; rule_id; seed; condition; events: list[Event]; structure="EPISODE_STRUCTURE_V0"
# episodes/render.py (W1)
render_episode(ep, include_event_indices: list[int]|None=None, upto_event: int|None=None) -> Rendered   # Rendered.text, Rendered.spans: per included event (char_start, char_end)
render_reset_query(ep, query_event_index: int) -> Rendered      # instruction header + the query only (context-reset)
# data/shards.py (W1)
read_shard(path, sealed_key: bytes|None=None) -> list[dict]; episode_from_json(d) -> Episode; write_shard(...)
# training/model_loading.py (W2)
load_frozen_backbone(checkpoint_dir: str, device: str, verify_tree_hash: bool=False) -> ModelBundle   # .model .tokenizer .device .identity(dict)
# training/tiny_model.py (W2)
build_tiny_model(seed: int, device: str="cpu") -> ModelBundle   # real modeling_ouro code, hidden 64, 2 layers, 2 ut steps, random init
# training/lora.py (W2)
attach_lora(model, spec: LoraSpec) -> LoraHandles; zero_lora_(h); lora_state_dict(h); load_lora_state_(h, sd)
clip_lora_frobenius_(h, max_norm) -> float; merge_scaled_(dst_h, src_h, scale)
# training/tokenization.py (W2)
episode_to_training_example(ep, tokenizer, arm: str) -> {input_ids, loss_mask(float weights), spans_meta}
# training/trainer.py (W2)
run_training_arm(cfg: ArmConfig, bundle, examples_iter, out_dir, hooks) -> ArmResult
# evaluation/generation.py (W4)
greedy_generate(bundle, prompts: list[str], max_new_tokens: int=64, prefix_embeds=None) -> list[str]
# evaluation/scoring.py (W4)
answer_logprob(bundle, text: str, answer_char_span: tuple[int,int], prefix_embeds=None) -> float  # teacher-forced mean per-token logprob
# mechanisms/fast_state.py (W3)
FastStateModule(torch.nn.Module): reset_state(batch)->s0; update(s, h_item)->s'; prefix_embeds(s)->Tensor[8,2048]-clamped
```

Rule for cross-worker drift: consumers adapt to the producer's actual
implementation; nobody edits another worker's files; genuine interface
conflicts go to CONTRACT_AMENDMENTS.md.

## 21b. Acceptance criteria

All 40 deliverables from the task statement; `scripts/run_all_tests.py`
passes everything (unit + hostile + dress rehearsal); smoke test within
budget or explicitly reported as skipped with reason; split manifest sealed;
preregistration + manifest template complete with only B200-derived fields
open (measured throughput, affordable U, FULL vs PEFT eligibility outcome);
deterministic zip + hashes; branch pushed; remote ref + hash independently
verified; final report in the exact required format.
