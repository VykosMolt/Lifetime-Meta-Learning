# Contract amendments

## Amendment 1 — sealed-shard key derivation (2026-08-09, pre-implementation)

Original §12 derived the sealed-shard cipher key from
`dev_decisions_sha256`. That is impossible: sealed shards are enciphered at
local pre-generation time, before any development decisions exist.

Amended rule: `K_seal = sha256(b"FL_V0_SEALED_KEY\0" +
split_manifest_sha256_hex_utf8)`. Enciphering happens at shard-write time in
`data/shards.py` using the shared util in `ecology/base.py`. Deciphering code
exists ONLY in `campaign/sealed_gate.py` and refuses to run unless
`DEV_DECISIONS_FROZEN.json` exists and the single-use opening ledger entry has
been written. Enforcement of the sealed-test policy is therefore procedural
(gate + ledger + hostile fixtures) with the cipher preventing accidental or
casual plaintext access; this is recorded honestly rather than claiming
cryptographic unopenability.

Recorded before any implementation began. No data or results existed at
amendment time.

## Amendment 2 — family split computed (2026-08-09, pre-implementation)

The §5 rule was executed once, mechanically, immediately after freezing:
TRAIN = {grammar_classification, set_operations, dsl_execution,
string_rewrite, finite_state_transducer, modular_arithmetic};
DEV = {sequence_transform, graph_edge_semantics, boolean_rule};
SEALED_TEST = {constraint_rules, permutation_composition,
propositional_transform}. `ecology/split.py` must reproduce exactly this
assignment from the frozen rule; a regression test pins it. No re-rolls
occurred; no performance information existed at computation time.

## Amendment 3 — no peft at runtime; custom LoRA (2026-08-09, pre-implementation)

The B200 container lock (`o1_b200/deploy/requirements.b200.lock`) contains no
`peft`. Rather than adding runtime dependencies or modifying the O1 image,
all low-rank adapter functionality (PEFT_MODE for core arms, FL7 fast
adapter, FL8 slow bank) is implemented in `training/lora.py` (custom,
norm-controllable, resettable — properties FL7 needs anyway). peft 0.19.1
(present in the local venv only) serves as a numerical oracle in a local unit
test (lazily imported; skipped where absent). Recorded before W2 began.

## Amendment 4 — §21 renumbered to §21b after §22/§23 insertion (editorial).

## Amendment 5 — training-core interface reality vs §23 (2026-08-09, W2)

Recorded by W2 (training core) after reading the implemented
`episodes/schema.py` and `episodes/render.py`. Per the §23 drift rule the
CONSUMER adapted; no producer file was edited. Nothing here changes an
objective, a weight, a metric, or a gate.

1. **FL1 is one episode to MANY examples.** §7 defines FL1 data as *isolated*
   (TASK -> ANSWER) pairs, so a single episode yields one example per
   task-bearing item. The §23 signature
   `episode_to_training_example(ep, tokenizer, arm)` still exists and behaves
   exactly as specified (it gains a keyword-only `item_index=0` selecting one
   FL1 pair, and remains the whole-episode example for FL2/FL3);
   `episode_to_training_examples(...)` (plural) returns all of them. The pool
   path `isolated_pair_to_training_example(prompt, answer, tokenizer)` is the
   primary FL1 route when items come from `data/pools.py` rather than episodes.

2. **FL3 weight for interaction index 4 (the related-task attempt).** §7 lists
   QUERY 1.0, TRANSFER 1.0, revisions (1, 3) 0.25, attempt0 0.0 and "all
   TASK/FEEDBACK/HINT/REVEAL context 0.0" but does not name index 4. It is
   frozen at **0.0** ("everything else 0.0"). Recorded before any training run.

3. **`Event` has no `meta` field.** §23 lists `meta: dict` on `Event`;
   `episodes/schema.py` carries `slot` instead and keeps per-item data in
   `Episode.items[item_id]["instance"]`. `training/tokenization.py` reads the
   item table first and per-event `meta` second, so both shapes work, and FL1
   still refuses (hard error) any item whose canonical answer cannot be
   established.

4. **`render_episode` signature and span shape.** §23 specifies
   `render_episode(ep, include_event_indices=None, upto_event=None)` returning
   `.spans` as `(char_start, char_end)` pairs. The implementation is
   `render_episode(episode, include_header=True)` plus
   `render_subset(episode, keep_indices, include_header=True)`, returning
   `EventSpan(event_index, role, item_id, interaction_index, start, end)`
   objects. The training core normalises all of these shapes and needs no
   subsetting, so no producer change is required.

5. **Answer spans are located through the event's own text.** The renderer
   emits `ATTEMPT: ANSWER: <payload>` on ONE line, i.e. the `ANSWER:` marker is
   not at a line start of the rendered text (the verifier parses the event
   text, where it is). The loss-mask builder therefore locates the event text
   inside its rendered block (it must occur exactly once) and applies the §4
   line grammar to that text. The resulting character span can be straddled by
   a single byte-level token (`ĠANSWER`); the frozen policy
   `on_straddle="expand_whitespace"` includes such a token when everything it
   contributes outside the span is whitespace — which is exactly the separator
   the model must emit — and hard-fails on any other straddle. No mask is ever
   approximated silently.

## Amendment 6 — stability thresholds left open by §17 (2026-08-09, W2)

§17 fixes the grad-norm rule (>50 for 3 consecutive steps), the divergence
rule (>2x trailing median over 200 steps) and the throughput rule (<40 % of
BENCH for 5 min), but states no threshold for "memory growth", "repeated zero
grad" or "fast-state norm explosion". `training/stability.py` declares them
now, before any run, as IMPLEMENTATION CHOICES exposed on `StabilityConfig`
and recorded inside every failure record:

- memory growth: allocated memory > 1.25x its value 200 steps earlier
  (severity OBSERVATION, not a scientific failure);
- repeated zero grad: grad norm exactly 0.0 on 10 consecutive steps;
- fast-state norm: ||s|| > 1000.0 (FL5 owns the tighter mechanism-specific
  bound and may pass its own value).

These are detector thresholds only. They can stop an arm and write a failure
record; they can never change a hyperparameter, an objective, or a gate.

## Amendment 7 — ecology / episodes / data implementation decisions (2026-08-09, W1)

Recorded by W1 (task ecology, episode format, data pre-generation) after
implementing §4, §5, §6 and §16. None of these changes an objective, a weight,
a metric, a gate, a pool size, the split, or the sealed-test policy.

1. **Instance kind lives in the seed.** `TaskInstance` has exactly the nine §4
   fields, so support / related / query / transfer cannot be a tenth field. The
   kind is carried in the two low bits of `seed`
   (`ecology/base.encode_seed` / `decode_seed`). Consequence, and the reason for
   the choice: every instance is a pure function of
   `(rule, seed, difficulty, surface_map_id)`, so `verify`,
   `structured_feedback` and `surface_remap` can reconstruct the full latent
   state of an item from the instance alone, with no side channel.

2. **`TaskFamily` gains `plausible_error` (and the `wrong_answer` wrapper).**
   §6 requires "a family-specific plausible-error sampler" for the scripted
   attempt policy; only the family knows what a plausible wrong answer is. The
   §4 interface is implemented in full and unchanged; this is an addition.
   `wrong_answer` wraps it and hard-fails rather than ever returning the truth.

3. **Structured feedback is deterministic; its `rng` argument is not consumed.**
   §4 passes `rng` to `structured_feedback` / `corrupted_feedback`, but a
   stochastic hint makes the §4 guarantee "corrupted feedback differs from the
   true feedback" unverifiable across calls (an early smoke run produced exactly
   that failure). Feedback content is therefore a pure function of
   `(rule, instance, attempt)`; any family tie-break draws from a seed derived
   from the instance. The `rng` parameter is retained for signature
   compatibility and deliberately unused.

4. **Hints are opaque, digit-free letter codes.** Every structured hint renders
   as `TAG_<CODE>` fields, contains no digits, and identifies items by their
   index in an ordering the prompt displays (registers, nodes, predicates). This
   is what makes "feedback never contains the pending canonical answer"
   machine-checkable (`ecology/base.answer_leak`, asserted on every call and
   hostile-tested). Two structural preconditions are enforced in code:
   `MAX_HINT_OPTIONS = 128` bounds the code alphabet, and no answer label may
   coincide with a code (a `("F","T")` boolean label variant was found and
   removed by the hostile fixture for exactly this reason).

5. **Episode identity excludes the poison condition.** `make_episode_id`
   covers family, split, rule, seed, difficulty, mode, structure, structured,
   reveal and arm tag but NOT the §9 poison condition, and a mode-independent
   `make_episode_key` derives items and attempt draws. The five poison
   conditions are therefore alternative HINT bodies over one and the same
   episode, and FL2's `successful` and FL3's `scripted` variants share items and
   first-attempt draws. Both contrasts are controlled rather than confounded
   with a different item sample.

6. **Certified feedback is correctness-only.** §6 permits "`CORRECT`/`INCORRECT`
   or certified structured feedback"; the implementation always uses the former,
   so all structured content lives in the uncertified HINT channel where poison
   is defined. Truthful and poisoned hints render identically.

7. **REVEAL is supported but not pre-generated.** §6 makes REVEAL present only
   in conditions that define supervised feedback (FL7). `assemble_episode`
   builds REVEAL events on demand from the stored support answers
   (`reveal=True`); the standard pools are generated with `reveal=False` rather
   than doubling every pool. The correctness-only condition is likewise the
   standard episode rendered without its HINT events
   (`episodes/render.render_subset`).

8. **§23 reconciliation for W1-owned interfaces.** §23 was added to the
   contract after W1 began; the W1 modules were brought to it ADDITIVELY, with
   no signature removed and no consumer broken:
   - `Event` gains the §23 `meta: dict` side table (env-only, never rendered,
     defaulting to `{}`; `Event.from_dict` tolerates records without it). The
     rendering label `slot` is retained, so Amendment 5 item 3 remains valid.
   - `render_episode` gains the §23 `include_event_indices` / `upto_event`
     selectors (composable) and `Rendered.char_spans` returns the §23
     `(char_start, char_end)` pairs; the richer `EventSpan` objects and
     `render_subset` remain.
   - `render_reset_query(episode, query_event_index)` is added and REFUSES any
     event that is not a query-style task, so a context-reset prompt cannot
     silently retain history.
   - `read_shard` gains the §23 `sealed_key` parameter and
     `episode_from_json(d)` is added. `read_shard` never DERIVES a key: without
     an explicit key it still refuses enciphered bytes. Because §23 places a
     decipher parameter in `data/shards.py`, the hostile fixture's invariant is
     restated where it now bites — no module under `ecology/`, `episodes/`,
     `data/` or `scripts/` performs a sealed read, derives a key for reading, or
     ships a decipher CLI — and Amendment 1's honest position is unchanged: the
     protection is procedural, the key comes from a public digest, and no
     cryptographic unopenability is claimed.

9. **Uniqueness rule made precise.** §5 requires that no `instance_id` crosses
   train/dev/test and no `rule_id` is shared across splits. Generation enforces
   the stronger operational rule that an `instance_id` may be claimed only by
   one `(split, family, episode key)` — which permits exactly the intended
   sharing between the two attempt-policy modes of one episode and nothing else
   — and that a `rule_id` is fresh within a family (deterministic redraw, hard
   failure after `RULE_DRAW_ATTEMPTS = 256`; measured worst case over both
   predeclared seeds and the full pools is 12).

## Amendment 8 — evaluation-side interface reality vs §23 (2026-08-09, W4; renumbered from a duplicate "7" by the integrator, content untouched)

The evaluation and analysis modules consume the producers' ACTUAL APIs (the §23
drift rule: consumers adapt, nobody edits another worker's files). The
deviations from the §23 sketch, and how they are absorbed, are:

1. **`render_reset_query(ep, query_event_index)` does not exist.**
   `episodes/render.py` provides the same capability as
   `render_subset(episode, keep_indices)` (its docstring names the
   context-reset evaluation as a motivating caller).
   `evaluation/learning_curve.resolve_reset_renderer` therefore accepts
   `render_reset_query` FIRST and falls back to `render_subset(ep, [idx])`,
   recording the resolved spelling in every episode record
   (`reset_render_source`). The reset prompt is header + the current query
   block + the renderer's own attempt cue, and is additionally VERIFIED
   against every prior event text (`context_reset.LeakCheckingWalker`).

2. **`render_episode` takes `include_header`, not `include_event_indices` /
   `upto_event`; spans are `EventSpan` objects, not pairs.** The walker builds
   its prompts by rendering a LIVE episode (the plan with the model's real
   attempts and the real computed feedback substituted) and locating the
   pending attempt slot with a sentinel, so it needs no `upto_event`
   parameter; `rendered_spans` accepts `.start`/`.end` objects, dicts and
   pairs.

3. **`data/shards.read_shard(path)` takes no sealed key and there is no
   `episode_from_json`.** `evaluation/fl0_base.load_episodes` calls
   `read_shard(path)` and rebuilds episodes with `Episode.from_dict`. Sealed
   shards remain unreadable from the evaluation package by construction:
   deciphering lives only in `campaign/sealed_gate.py`.

4. **Poison-condition ids.** §9 spells the untouched condition
   "correct-informative" and metric 12 calls it "clean". The data layer's
   frozen ids (`ecology/poison.POISON_CONDITIONS`) are
   `("clean", "correct-redundant", "irrelevant", "partially-misleading",
   "corrupted")`. `evaluation/metrics` imports that tuple and aliases the §9
   spellings onto it; `CLEAN_CONDITION = "clean"`. No condition is added,
   removed or renamed.

5. **`greedy_generate` and `answer_logprob` keep their §23 signatures
   exactly**; the extra capabilities they need (batch policy, detailed
   records, batch scoring) are keyword-only additions and separate functions.

Two ONLINE-evaluator policies that §6 leaves open are frozen here, and are
documented in `evaluation/learning_curve.py`:

- **Channel policy.** In BOTH the `structured` and `correctness_only`
  conditions the certified `FEEDBACK` event carries the verifier's verdict
  (`CORRECT`/`INCORRECT`) computed from the model's REAL attempt, and the
  uncertified `HINT` event carries the family's structured content, selected
  per slot by the data layer's poison schedule through
  `ecology.poison.hint_for_condition`. `correctness_only` drops HINT events
  entirely (§6: hints exist "in structured conditions"). The certified channel
  is never corrupted, and a certified event flagged `poison` is a hard error.
- **Leak-check formulation.** The context-reset check removes exactly ONE
  occurrence of the current query's own text before probing the prompt for
  prior-event text, because two items of one family can carry byte-identical
  surfaces (a 4-variable Boolean assignment repeats within an episode) and a
  naive substring probe would flag the query itself. A renderer that really
  keeps the history is still caught; a hostile fixture proves it.

## Amendment 9 — FL4–FL8 mechanism decisions (2026-08-09, W3)

Recorded by W3 (mechanisms) after implementing §7's FL4–FL8 against the
producers' ACTUAL modules (§23 drift rule: consumers adapt, nobody edits
another worker's files). Numbering starts at 9 because §23's amendment stream
already contains two independently authored Amendment 7s (W1 and W4). None of
the items below changes an objective, a weight, a metric, a gate, a threshold,
a pool size, the split, or the sealed-test policy.

1. **Final-loop hidden states come from `return_per_loop_hidden_states`.**
   §7 binds "final-layer, final-loop hidden state" without naming an API.
   `mechanisms/hidden_states.py` uses
   `OuroForCausalLM(..., return_per_loop_hidden_states=True)
   .per_loop_hidden_states[-1]`, which is the ONLY way to obtain that tensor
   AND the logits from one differentiable forward (FL5 training needs both).
   `modeling_ouro.OuroModel.forward` appends the post-`self.norm` state of each
   recurrent loop to `hidden_states_list` and returns
   `last_hidden_state = hidden_states_list[-1]`, so this tap is IDENTICAL to
   the one `evaluation/learning_curve._final_loop_hidden_states` uses online;
   a unit test pins the bitwise equality of the two. The inner-module fallback
   is used only when a wrapper hides the RLTT fields, and it REFUSES to serve a
   request that also needs logits rather than substituting another quantity.

2. **FL4 head-input contexts end at item j.** `head_input_hidden` renders
   events `0..j` with `render_subset` and refuses any context containing a
   `QUERY_TASK`/`TRANSFER_TASK` block or a query/transfer attempt. This is the
   structural form of "the head never sees future answers"; a hostile fixture
   attacks it with a feedback event moved behind the query block.

3. **FL4 target span convention.** The target is the mean per-token logprob of
   the correct QUERY answers, averaged over the three query ITEMS (each item's
   own per-token mean), both halves computed with W4's
   `evaluation.scoring.answer_logprob_batch`. Because that function requires a
   STRICTLY token-aligned character span while W2's loss masks use the frozen
   `expand_whitespace` straddle policy (`ĠANSWER` carries the separator space),
   `value_head.token_aligned_answer_span` selects exactly the tokens W2 would
   put loss on and hands the scorer THEIR character boundaries. The FL4 target
   therefore measures the log-likelihood of exactly the tokens FL3 trains on,
   with no re-alignment inside the scorer. Events after the last query attempt
   are dropped from the scored rendering: under a causal LM they cannot change
   the query log-probabilities, so the score is numerically identical and the
   forward is shorter.

4. **FL4 constants not fixed by §7** (all frozen here, before any run, and
   documented in `mechanisms/value_head.py`): ranking-hinge margin 1.0 (the
   `torch.nn.MarginRankingLoss` default; predictions are unbounded, so the
   margin is a scale convention and the auxiliary z-scored MSE fixes the
   scale); targets z-scored with the TRAINING-set mean/std, stored in the head
   as buffers so DEV predictions share the scale (calibration slope/intercept
   are therefore reported against the z-scored target); head optimizer AdamW
   with the §8 campaign constants and LR 1e-3 (§8's grid binds the BACKBONE
   arms); default 40 epochs, 8 episode-groups per batch (both caller-settable
   and recorded per run).

5. **FL5 module parameterisation.** `W_in` and `W_p` are BIAS-FREE (matching
   §7's equations and its ~25 M parameter estimate); `GRUCell` keeps its
   standard biases. `hidden_size` is a constructor parameter (2048 for the
   frozen checkpoint, 64 for the tiny mechanics model). `rho` must be supplied
   EXACTLY ONCE — either as an embedding matrix (rho = 2 x median row L2 norm,
   computed once and recorded in `config()`/`config_hash()`) or as an explicit
   value for checkpoint reload — so it can never be silently defaulted. The
   per-vector clamp is `P_i * min(1, rho/||P_i||)`, differentiable and exact at
   the boundary. FL5's mechanism-specific fast-state norm bound (Amendment 6
   left the value to FL5) is 64.0, passed through W2's `fast_state_norm_max`.

6. **FL5 training is a segmented forward, and BOTH arms use it.** §7 says
   FAST_STATE_ON and FAST_STATE_OFF share the FL3 objective, data order and
   seeds and that OFF "disables injection AND state usage". `fl5_training.py`
   renders and tokenises the episode ONCE, partitions its tokens into segments
   cut after each run of feedback events (each token belongs to the segment
   containing its start offset, so the partition is exact), and forwards each
   segment as `[prefix; segment tokens]`. Consequence, and the reason for the
   choice: the ONLY channel from an earlier interaction to a later segment is
   the state `s`, so the ON/OFF contrast is exactly "the state carried the
   learning". The FL3-weighted NLL is accumulated over ALL segments (revision
   spans contribute from their own segments) and reduced as `OBJ_EPISODE_MEAN`,
   i.e. the objective is identical to FL3's, not merely similar. A weighted
   token at segment-local position 0 is a hard error, as in W2.

7. **FL5 plugs into W2 by reuse, not by copy.** `run_training_arm` is built
   around `examples_to_batch` plus one whole-sequence forward and cannot express
   a stateful segmented forward through its external step hook, so FL5 runs its
   own loop and REUSES W2's `build_optimizer`, `build_scheduler`, `epoch_order`,
   `ComputeLedger`, `StabilityMonitor` and `save_checkpoint`. `FL5TrainConfig`
   is duck-typed for `build_optimizer`/`build_scheduler` (they read only
   `learning_rate`, `betas`, `eps`, `weight_decay`, `warmup_steps`, `updates`)
   rather than registering a fake core arm in `training/arms.py`.

8. **FL7 inner-loop supervision is enforced, not assumed.** The supervised
   example is the SUPPORT task event plus an answer event carrying the REVEALED
   canonical answer, rendered through W1's renderer and masked by W2's
   `episode_to_training_examples(..., arm="FL1")` (loss on the `ANSWER:` line
   only). `supervised_example` REFUSES any item without a `REVEAL` event or
   with a non-support role, so query/transfer labels cannot enter the inner
   loop. Because the fast delta lives in the model, exactly one FL7 episode may
   be in flight at a time (`attach_lora` refuses a second attachment, so the
   mistake cannot be made silently).

9. **FL8's "separate slow bank" is W2's merged-component list.** `merge_scaled_`
   appends a FROZEN low-rank component to a layer; `zero_lora_` explicitly does
   not touch those components. "Merge into a separate slow bank, then clear the
   fast state" is therefore `merge_scaled_` followed by `zero_lora_`, and
   `slow_bank_report` reports the bank alone. A recorded per-episode delta is
   carried by `FastDelta`, which exposes exactly the three attributes
   `merge_scaled_` reads from a source handle; restoring a delta with
   `load_lora_state_` instead would CLEAR the destination's merged components
   (i.e. wipe the bank), which is why the delta is carried separately. The
   merge arithmetic itself is W2's function, unmodified.

10. **FL8 value-gated selection rule.** §7 says "selection = value gate" but
    not how per-item gate decisions become a per-episode selection. Frozen
    rule: an episode's delta is merged iff the gate ADMITTED AT LEAST ONE
    update in that episode (`n_admitted > 0`). A delta from zero admitted
    updates is exactly the zero delta, so the rule only makes the bookkeeping
    explicit. Sealed families are refused at every consolidation entry point,
    including after a record has already been accepted.

11. **Numerical tolerance in the FL7 norm-bound fixture.**
    `clip_lora_frobenius_` rescales float32 `B` factors in place, so the
    re-measured global norm carries float32 rounding (~1e-8 relative; float32
    eps is 1.2e-7). The hostile fixture allows `1e-6 * beta` for that rounding
    and separately asserts that the PRE-clip norms exceed beta by orders of
    magnitude, so a real violation cannot hide inside the tolerance. The bound
    itself is not relaxed.

## Amendment 10 — campaign / deploy / scripts implementation decisions (2026-08-09, W5)

Recorded by W5 (campaign layer, deploy, packaging scripts) after implementing
§10–§13, §15, §18 and §19 against the producers' ACTUAL modules (§23 drift
rule: consumers adapt, nobody edits another worker's files). Numbering
continues at 10 because the amendment stream already contains two
independently authored Amendment 7s (W1 and W4). None of the items below
changes an objective, a weight, a metric, a promotion threshold, a pool size,
the split, the safety factor, the transfer reserve, or the sealed-test policy.

1. **The stage table contains two operational stages that are not §7 rungs.**
   `BENCH` is §11's "runtime/throughput validation" (priority 1) and
   `DEV_GRID` is §8's mandatory two-learning-rate FL3 grid, which must complete
   before any core arm because it is what DEFINES the core learning rate. Both
   sit inside the core-comparison allocation that §11's affordability rule
   already projects ("grid + 3 arms + evals"). A `SECOND_SEED` stage carries
   §11 priority 9 (additional predeclared seeds), and `SEALED_EVAL` — the
   single §12 opening — is LAST, so it cannot precede a development decision.

2. **BENCH is admitted against a DECLARED budget, not a measured one.** Every
   other stage is projected from BENCH's measurements; BENCH cannot be, because
   it IS the measurement. `FL_BUDGET_POLICY.json` therefore carries
   `bench_declared_budget_seconds = 900` and BENCH is admitted against it under
   the same `× 1.25 + reserve` inequality. This is the ONE projection in the
   campaign that is declared rather than measured, and it is labelled
   `BENCH_DECLARED` in the journal.

3. **BENCH measures the evaluation cost on TRAIN episodes.** §11 requires the
   throughput benchmark to run "on a non-evaluation training shard". The
   affordability projection also needs a per-episode EVALUATION cost, so BENCH
   times the real W4 walker on TRAIN episodes as well. No DEVELOPMENT or
   SEALED_TEST episode is consumed by the benchmark. The evaluation cost is
   scope-independent (the walker only runs greedy inference), so it is measured
   once and attached to every scope's measurement.

4. **FULL_MODEL_MODE is INELIGIBLE when unmeasured.** §11's rule needs a FULL
   projection to decide. BENCH measures both scopes when both are requested; if
   a FULL measurement is absent, the rule returns "ineligible" rather than
   estimating one, so an unmeasured scope can never be selected.

5. **Per-stage evaluation maxima are frozen in `stage_definitions.py`.** The
   preregistration leaves "evaluation batch size and resulting eval episode
   counts" B200-derived. Only the BATCH SIZE is genuinely B200-derived (it is
   the output of the §22 equivalence gate); the AMOUNT of evidence is frozen
   here (`EVAL_EPISODES_PER_STAGE`) so it cannot drift with available time.

6. **`max_new_tokens` is frozen at 64 and the override is rehearsal-only.**
   `_generation_config` refuses a shortened decode budget unless the context is
   explicitly marked `rehearsal`; the dress rehearsal uses 8 and records both
   values in its report.

7. **The dress rehearsal uses a labelled miniature ladder.** §18 sanctions
   "tiny model, tiny pools, seconds-scale budgets", which the frozen U ladder
   {600, 1200, 2400, 4800} cannot express on a CPU tiny model.
   `plan_core_comparison(ladder=…)` REFUSES any non-frozen ladder unless
   `rehearsal=True` is passed explicitly, stamps the plan
   `REHEARSAL_LADDER_OVERRIDE`, and every rehearsal arm configuration is tagged
   `STAGE_SMOKE` ("local mechanics only; never a scientific result"). The
   frozen ladder and the frozen learning-rate grid remain fully enforced on the
   CORE/GRID stage tags, which the unit suite exercises directly. The
   rehearsal's nominal authorized budget is 3600 s rather than "seconds-scale"
   because the 1200 s transfer reserve is NOT scaled down; the wall-clock cost
   of the rehearsal itself is ~4 minutes.

8. **`VERIFY_O1_RECORDS` runs through a separate hash-only custodian.** §13
   requires FL to refuse every O1 path AND to verify O1's records against O1's
   own manifests. Those are in tension, so `session_supervisor.O1RecordCustodian`
   is a deliberately crippled reader: it accepts ONLY paths under the declared
   O1 roots, exposes no method that returns file content, and parses nothing
   but path/digest pairs. The FL isolation guard continues to refuse those
   paths for all FL work. Hashing bytes is not reading outcomes; every path the
   custodian touches is journalled, and a unit test pins the absence of a
   content-returning method.

9. **The sealed opening ledger carries an audit nonce.** Two openings in the
   same wall-clock second would otherwise produce byte-identical entries, so a
   deleted-and-recreated ledger would be indistinguishable from the original.
   `opening_nonce` (16 random bytes) makes the opening-entry hash recorded
   INSIDE every read-only sealed result a real cross-check. It is an audit
   value only; no code path branches on it, so §20's ban on nondeterminism in
   scientific paths is untouched. Amendment 1's honest position is unchanged:
   the protection is procedural, and no cryptographic unopenability is claimed.

10. **Checkpoint best-DEV selection lives in `campaign/promotion.py`.** §15
    fixes the rule (max DEV macro-AULC at scheduled evaluations) but not its
    home. `select_best_dev_checkpoint` consumes the same `DevMetrics` object as
    every promotion rule, which structurally cannot hold a sealed record, and
    breaks ties toward the EARLIER evaluation so the choice is deterministic.

11. **The secret deny-list scrub covers the whole "tokeniz…" word family.** The
    O1 packager scrubs the word `tokenizer` before matching (a tokenizer is a
    model asset, not a credential); this package also contains
    `tokenization.py`, so the scrub is `tokeniz(er|ation|e|ing|…)`. A bare
    `token` in a file name (`hf_token`, `api_token`) still trips the deny-list.

12. **Pregenerated data is git-ignored; its manifests are not.** The raw shards
    (~360 MB) are bitwise reproducible from `scripts/pregenerate_all.py`, so the
    repository-root `.gitignore` ignores `artifacts_fl/pregen*/*` and
    re-includes `artifacts_fl/pregen*/MANIFESTS/`.
    `scripts/package_release.py` mirrors `PREGEN_MANIFEST.json`,
    `SHARD_SUMS.json` and `family_split_manifest.json` into that directory,
    hash-verified on every run, so packaging, manifest filling and review can
    always find them. `foundation_learner/.gitignore` cannot express this
    itself (the data lives outside the package directory) and says so.

13. **`SHA256SUMS` covers the whole content manifest, rooted at the repository
    root.** It lists `foundation_learner/**` and `artifacts_fl/pregen/**`
    (minus `reports/local_runs/`, `__pycache__/`, `*.pyc`, `*.tmp` and the
    release artefacts) in the O1 two-space format, with an EXACT-coverage
    bijection check in both directions. When the data is absent from a checkout,
    `scripts/run_all_tests.py` reports the checksum suite as SKIPPED with that
    reason rather than as a pass.

14. **`scripts/run_all_tests.py --fast` names what it omits.** The measured
    slow modules (the long model walkers) are listed in `SLOW_NODE_IDS` and are
    recorded in `TEST_REPORT.json` as `skipped_modules`, so a fast run can never
    be mistaken for a full one. No test is weakened or deleted to obtain speed.

15. **`campaign/stage_definitions.py` names the FL4–FL8 entry points as dotted
    path strings** (`foundation_learner.mechanisms.<module>:run_fl<N>_stage`),
    resolved lazily at stage time. A MISSING mechanisms module is recorded as
    `SKIPPED_MECHANISMS`; a PRESENT module without the named entry point is an
    integration ERROR, never a silent skip. This keeps the campaign layer
    independent of the mechanism API while it is still being written, at the
    cost of binding those five entry-point names — which W3 may satisfy with a
    thin adapter if its internal API differs.

## Amendment 11 — FL4–FL8 campaign stage adapters (2026-08-09, W3)

Recorded by W3 after wiring the mechanism rungs into W5's declarative stage
table. The table resolves each rung lazily through a dotted entry point
(`foundation_learner.mechanisms.<module>:run_fl<N>_stage`) with the signature
`run_flN_stage(ctx: StageContext, stage: StageDefinition) -> dict`; W3 supplies
those five adapters plus `mechanisms/stage_support.py`, which holds everything
they share. No campaign file was edited. None of the items below changes an
objective, a weight, a metric, a promotion threshold, the split, or the
sealed-test policy.

1. **FL4's DEVELOPMENT evaluation needs DEVELOPMENT targets.** §7 fixes the
   head's TRAINING targets as "TRAIN families only" and, in the same paragraph,
   requires a DEV evaluation of the head (Spearman, pairwise ranking accuracy,
   calibration, top-1 regret, versus surface heuristics) — which is only
   computable if the same realized-value quantity is computed on DEVELOPMENT
   episodes. `assert_target_split` therefore replaces the previous all-or-
   nothing check: the default admits TRAIN only, the FL4 stage's DEV evaluation
   passes `allowed_splits=TARGET_SPLITS_DEV_EVAL`, and SEALED_TEST is filtered
   out of ANY caller-supplied list, so it is refused under every setting. This
   strictly tightens the old `require_train_split=False` path, which performed
   no split check at all.

2. **FL4 targets are computed with the restored FL3 checkpoint.** §7 says the
   targets are computed "with the final FL3 checkpoint", but `StageContext`
   hands out a FRESH BASE bundle by design (§7's per-arm fresh-load rule).
   `stage_support.arm_checkpoint_state` restores the arm's `final` checkpoint
   into that fresh bundle with W2's own `load_checkpoint` +
   `load_lora_state_` / `load_state_dict`. When no FL3 checkpoint exists in the
   session (a rehearsal, or FL3 skipped) the base checkpoint is used and the
   payload records `target_model = BASE_CHECKPOINT_NO_TRAINED_ARM` with the
   reason — it is never passed off as the FL3 model.

3. **Payload → promotion field mapping (frozen).** `stage_definitions.fl6_entry`
   reads `ctx.results["FL4"]["pairwise_ranking_accuracy"]`, so FL4 surfaces that
   exact key at the TOP LEVEL of its payload; `fl7_entry` reads
   `status == "COMPLETE"`; `promote_fl4`'s data-sufficiency evidence is
   published as `ctx.extra["fl4_scoreable_items_per_episode"]`; `fl8_entry`
   reads `ctx.extra["persistence_evidence"]["point"]` / `["ci"]`, which FL5
   (and FL7, when FL5 published nothing defined) writes.

4. **The FL8 persistence estimand.** §10 admits FL8 on "FL5 or FL7 context-reset
   persistence > 0 with CI excluding 0 on DEV" without naming the statistic.
   Frozen here: the per-episode retained improvement
   `mean(R_k, k >= reset_from_index) - R_0` under the context-reset condition,
   compared between the mechanism arm and its MATCHED no-mechanism arm on the
   SAME episodes, with §14's paired family-clustered bootstrap (one shared
   resample plan, 10,000 replicates). The difference form is what makes the
   claim about the mechanism rather than about the task.

5. **FL7's supervised episodes are re-assembled with `reveal=True`.** The
   standard pools are pre-generated with `reveal=False` (Amendment 7 item 7),
   but FL7's inner loop is REVEAL-supervised. `stage_support.reveal_variant`
   calls W1's `assemble_episode` with the episode's own family, rule spec,
   seed, difficulty, mode, condition and structured flag and `reveal=True`;
   since assembly is a pure function of exactly those, the result carries the
   SAME rule, items and scripted attempts plus the REVEAL events. Nothing is
   fabricated.

6. **A missing FL4 head is a recorded skip, never a fabricated gate.** FL6 (and
   FL7's gated variant, and FL8's `value_gated` mode) needs the FROZEN FL4
   head. FL4 publishes it in-process (`ctx.extra["fl4_value_head"]`) and
   persists its tensors next to the report through the guard. When it is
   absent, FL6 returns `status = SKIPPED_MISSING_INPUT`, FL7 runs its ungated
   variant only, and FL8's `value_gated` mode records `gate_available = false`;
   an untrained head is never substituted, because a random gate presented as a
   mechanism would be a fabricated capability.

7. **Rehearsal-scale reductions are declared, never silent.** Every adapter
   takes its scale from `ctx.eval_episode_cap` / `ctx.train_example_cap` /
   `ctx.rehearsal`, and every payload carries `rehearsal` plus the note "DRESS
   REHEARSAL: mechanics only, never a scientific result". Two rehearsal
   defaults are declared in code: mechanism training arms use 2 updates
   (`stage_support.REHEARSAL_UPDATES`), and FL6 walks the two ENDS of the frozen
   §9 condition list (`clean`, `corrupted`) instead of all five
   (`value_gating.REHEARSAL_POISON_CONDITIONS`). A real run uses `ctx.updates`
   (required; it is never guessed) and every frozen condition present in the
   data. `ctx.extra` keys (`fl4_train_episodes`, `fl5_updates`,
   `fl6_conditions`, `fl8_chains`, `fl8_modes`, ...) let the campaign set any of
   these explicitly, and the chosen values are recorded in the payload.

8. **FL7 runs one episode at a time.** The fast delta lives IN the model, so
   several FL7 episodes cannot share a bundle; `attach_lora` refuses a second
   attachment, so the mistake cannot be made silently. FL7 therefore evaluates
   episode by episode and uses one fresh bundle per variant plus one for the
   matched no-adapter baseline. This is a real throughput property of
   parameter-level adaptation, not an implementation shortcut.

9. **Frozen campaign conventions are imported, not re-implemented.** The
   adapters use `campaign.stage_definitions`'s guarded episode loader, its
   exact-verifier `env_factory` and its `_generation_config` (the frozen
   64-token decode budget, rehearsal-overridable). Duplicating those rules
   would let two copies of a FROZEN constant drift; if the campaign renames the
   helper the stage raises loudly instead of decoding with a different budget.
   All imports are made INSIDE the adapter functions, so importing
   `mechanisms` never pulls in the campaign package and the lazy stage table
   stays lazy.

## Amendment 12 — campaign-runtime and release repairs after adversarial review (2026-08-09, W6)

Recorded by W6 after an adversarial review (verdict REJECT) and an independent
verification (verdict FAILED on packaging) of the complete package. Every item
below repairs an implementation defect or records a decision the contract left
open; NONE of them changes an objective, a loss weight, a metric definition, a
promotion threshold, the split, a pool size, the safety factor, the transfer
reserve, or the sealed-test policy. Everything here is frozen BEFORE any
accelerator use, before any training run, with the sealed set unopened.

1. **The campaign has a production entry (`campaign/entry.py`).**
   `session_supervisor.main()` previously built a supervisor whose
   `context_factory` and `ladder_runner` were `None`, so the pod entry reached
   `RUN_FL_LADDER` and refused: only the dress rehearsal and the tests ever
   built a `StageContext`. `campaign/entry.py` is now the single place a stage
   context is built, from the session configuration:
   `bundle_factory = load_frozen_backbone(checkpoint_dir, device,
   verify_tree_hash=True)` (§13's PRISTINE reload), the pre-generation root,
   the isolation guard, the scope, the root seed and the token budget. The
   REHEARSAL path is the SAME code with ONE difference — `fl_tiny_model: true`
   substitutes the tiny NONSCIENTIFIC bundle, which is refused outside a
   session explicitly marked `rehearsal`, and evaluation caps
   (`fl_eval_episode_cap`, `fl_train_example_cap`) are likewise rehearsal-only,
   because they shorten the frozen per-stage evaluation maxima.

2. **§22 determinism is applied and RECORDED at campaign start.**
   `training/model_loading.set_deterministic_eval()` existed but was never
   called. `campaign.entry.configure_determinism()` calls it for real sessions
   AND rehearsals, then records the flags as torch reports them
   (`flb200.determinism_record.v1`) in the session journal
   (`CAMPAIGN_ENTRY_WIRED`) and in `SESSION_FINAL_STATUS.json`. If
   `torch.use_deterministic_algorithms(True)` does not take effect the entry
   refuses rather than proceeding quietly.

3. **A failed session always closes out.** `SessionSupervisor.run` wraps the
   state machine in `try/finally`. On any abort — a failed state, an exception
   outside a state, a KeyboardInterrupt — the supervisor attempts
   `TRANSFER_FL_ARTIFACTS` (best effort; skipped when no ladder output exists,
   because an empty archive would misdescribe the session) and ALWAYS attempts
   `TERMINATE_ACCELERATOR`, records both in `close_out` and in the journal, and
   always writes `SESSION_FINAL_STATUS.json`. Termination is attempted even
   when the transfer failed: losing artefacts is bad, paying for an abandoned
   accelerator is worse.

4. **Resume rebuilds `state_results` from the journal.** A resumed session
   restores every `STATE_COMPLETED` result payload, so
   `COMPUTE_REMAINING_AUTHORIZED_TIME` is no longer skipped WITHOUT its result
   (which previously made `RUN_FL_LADDER` refuse for want of an FL allowance —
   a mid-session crash silently cost the entire FL half of the rental). The
   restored allowance is REDUCED by the wall-clock gap between the journalled
   record and the resume; the pod billed for that gap. This is an operational
   budget decision, not a scientific path, and it can only ever SHRINK the
   allowance (§20 is untouched).

5. **Evaluation sets are capped PER FAMILY and their coverage is asserted.**
   Every evaluation set was assembled as `load_episodes(...)[:cap]` over
   shard-ordered episodes, which returns the first family's episodes and
   silently turns every family-macro, family-clustered metric (§9, §14) into a
   single-family metric. The cap is now divided across the split's families and
   applied per shard (`eval_plan`, `load_balanced_eval_episodes`), the sealed
   evaluation slices per SHARD, `mechanisms/stage_support.dev_episodes`
   delegates to the same assembler, and `assert_eval_family_coverage` fails
   loudly wherever a set is assembled. A hostile fixture
   (`test_hostile_eval_set_family_collapse.py`) attacks it from four
   directions. Consequence for projections: the projection uses the PLANNED set
   size (`planned_eval_episodes`), never the raw cap.

6. **A->B->A chains resolve against the split pool and stay pair-balanced.**
   `load_chain_specs` moved into the campaign (`stage_definitions`) and
   `mechanisms/stage_support.load_chain_specs` delegates to it. It completes the
   episode pool from the split's own shards when a chain names an episode the
   caller does not hold (previously most chains vanished silently) and selects
   chains ROUND-ROBIN over the ordered family pairs (previously a `limit`
   returned chains of one pair, which is not the family-balanced interference
   metric §9 defines).

7. **The sealed evaluation runs the FROZEN PROMOTED CANDIDATE.** It previously
   evaluated a fresh BASE bundle and reported it as the campaign's sealed
   result. `promoted_arm_bundle` restores the promoted arm's `final` checkpoint
   through `mechanisms.stage_support.arm_checkpoint_state` and the report
   records the arm, the checkpoint manifest hash and the restoration record; a
   missing checkpoint REFUSES. The promoted arm is mechanical and frozen before
   any sealed access: the core arm pinned in the development decisions, else
   the frozen core treatment FL3 (`DEFAULT_PROMOTED_ARM`). A new entry
   condition `sealed_eval_entry` requires the COMPLETED core comparison
   (FL1/FL2/FL3), the frozen `DEV_DECISIONS_FROZEN.json`, and the promoted
   arm's checkpoint. The sealed report also carries the §9 family-holdout
   report (UNSEEN-INSTANCE / UNSEEN-FAMILY separated) and the §13.1 claim-scope
   statement, and records are split-tagged through the evaluation layer's own
   annotation path.

8. **The sealed opening is TWO-PHASE (§12c, restated).** `open_sealed` now
   grants PROVISIONAL access and writes NOTHING; `commit()` appends the single
   `SEALED_OPENED` entry once the evaluation records exist; anything that
   escapes before the commit appends `SEALED_OPENING_ABORTED` and permits
   EXACTLY ONE retry (`MAX_OPENING_ATTEMPTS = 2`); a third attempt refuses and
   both attempts stay in the append-only hash-chained ledger forever. A
   provisional unlock may READ sealed shards but may NOT write a sealed result.
   `sealed_opening(...)` is the context manager the campaign uses; it never
   swallows the exception — the abort is a RECORD, not a recovery. Rationale: a
   seal burned by the FIRST line of an evaluation is not robust; the EVIDENCE
   is what consumes the opening. Nothing is weakened: the opening is still
   single-use, still ledgered, still irreversible, and every attempt is
   permanent.

9. **The core comparison is audited in production (`campaign/core_matching.py`).**
   `training.arms.assert_core_arms_matched`,
   `training.compute_accounting.compare_core_ledgers` and the base-identity
   check were invoked nowhere outside the test suite. A new `CORE_MATCHING`
   stage (§11 priority band 5, immediately after the three core arms and before
   any extension consumes them) runs all three against the real `ArmConfig`
   objects, the realized compute ledgers and the arms' checkpoint manifests,
   and writes `flb200.core_matching_report.v1`. A mismatch RAISES: the
   scheduler records a failed stage with its predeclared fallback rather than
   letting an unmatched comparison become the headline.

10. **Metrics 11, 12 and 13 now have producing stages.** Three UNCONDITIONAL
    diagnostics are added to the frozen table: `REMAP_DIAG` (metric 13, over
    the pre-generated DEVELOPMENT remap records — 1,800 of the 3,600 total; the
    SEALED_TEST half is never touched outside the single opening),
    `INTERFERENCE_DIAG` (metric 11, over the pre-generated A->B->A chains) and
    `POISON_DIAG` (metric 12, over the five frozen §9 conditions, independent of
    FL6, which measures GATING rather than blind incorporation). They run on the
    promoted arm when its checkpoint exists and record the fallback to the base
    checkpoint when it does not. **Frozen order**: they carry §11 priority band
    11 — after the FL4-FL8 admission attempts (bands 6-10) and BEFORE additional
    predeclared seeds (12) and the sealed opening (13). They have NO entry
    condition, so they also run under the §10 FL3-null fallback, which names
    exactly these diagnostics as predeclared work. `SECOND_SEED` moves to band
    12 and `SEALED_EVAL` to 13; the relative order of every pre-existing stage
    is unchanged.

11. **`fl0_work` passes the real TRAIN family list.** It passed `[]`, which
    makes §9's UNSEEN-INSTANCE / UNSEEN-FAMILY partition vacuous. FL0 is
    untrained, so every DEVELOPMENT family is an unseen FAMILY; the list fixes
    the partition and makes FL0 comparable with the trained arms, and the
    payload records that reasoning.

12. **Projections model what the stages really run (§11).** Previously every
    stage was projected as one training arm plus one evaluation pass.
    `STAGE_EVAL_MULTIPLIER` now carries the structural cell counts (FL0 x4 =
    2 feedback conditions x {history, context_reset}; FL5 x4 eval and
    `STAGE_TRAIN_MULTIPLIER` x2 train; FL6 x10 = 2 gate variants x 5 poison
    conditions; FL7 x3 = ungated + gated + matched baseline; FL8 x9 = 3
    consolidation modes x 3 chain positions; REMAP_DIAG x3; INTERFERENCE_DIAG
    x3; POISON_DIAG x5), `STAGE_BUNDLE_LOADS` adds a per-stage FRESH-LOAD term
    (§7 reloads the checkpoint per arm), and FL4 is projected from its TARGET
    COMPUTATION — `FL4_TARGET_FORWARDS_PER_EPISODE = 8` teacher-forced forwards
    per episode — rather than from optimizer updates it does not run. Both new
    inputs are MEASURED in BENCH (`model_load_seconds`,
    `forward_seconds_per_episode`); a missing measurement BLOCKS the stage
    instead of being assumed to be zero. `CORE_MATCHING` uses a declared
    bookkeeping budget (`AUDIT_STAGE_SECONDS = 60`), the second declared (rather
    than measured) projection in the ladder after BENCH, and it loads no model.

13. **Stage watchdog.** A stage is ABORTED when its elapsed time exceeds
    `max(projected x safety_factor, stage_watchdog_minimum_seconds)` or as soon
    as the remaining authorized time falls to the FINAL TRANSFER RESERVE. The
    outcome is `STAGE_ABORTED_OVERRUN` with the stage's predeclared fallback,
    and the ladder continues (§10). `stage_watchdog_minimum_seconds = 900` is a
    NEW frozen field of `FL_BUDGET_POLICY.json`, declared because a projection
    derived from a short BENCH probe cannot bound a stage below its own
    measurement noise; the reserve condition is the hard money guarantee and
    has NO floor. HONEST LIMIT: the watchdog is checked at every phase boundary
    of a stage and on every trainer step; it cannot interrupt a single in-flight
    model call.

14. **`bench_declared_budget_seconds` is policy-pinned.** The one declared
    projection in the ladder may only be overridden by an explicitly labelled
    dress rehearsal; a real campaign takes it from the frozen policy file.

15. **`affordability.load_policy` routes its read through the isolation guard**
    (the default guard when none is supplied), so no campaign file read
    bypasses §13's realpath refusal (V-Obs7).

16. **Layer-level gradient checkpointing is recorded honestly.**
    `OuroModel` declares `supports_gradient_checkpointing = True` and
    initialises `self.gradient_checkpointing = False`, but its forward never
    consults the flag: `gradient_checkpointing_enable()` is a NO-OP on this
    architecture, and only the loop-level RLTT checkpointing
    (`torch.utils.checkpoint` around `_run_single_ut_loop`) is a real memory
    saving. `enable_training_memory_savings` now separates REQUESTED from
    ACTIVE and DETECTS the active value
    (`layer_checkpointing_is_consulted`, by source inspection, reporting
    `UNKNOWN` when the source is unavailable) instead of asserting it from the
    API call succeeding. Recording a no-op as an active memory-saving path put a
    false statement into every arm result and into the manifest.

17. **`SECOND_SEED` semantics.** `PREDECLARED_NOT_RUN` is a DECLARATION, not a
    result and not a silent skip. The whole campaign — pre-generated data,
    split manifest, episode seeds, RNG substreams — is a pure function of the
    root seed, so the second predeclared seed is a second CAMPAIGN RUN launched
    with `--seed 20260810` against its own pre-generated root, never a stage
    that mutates this run's frozen seed midway. The stage stays in the table so
    that §11 priority "additional predeclared seeds" is explicit, the admission
    arithmetic covers it, and the journal records whether the time for it
    existed; it now writes its declaration as an artefact.

18. **Release content policy (contract §19).** The zip is the ACCELERATOR
    BUNDLE. It contains exactly: every GIT-TRACKED file under
    `foundation_learner/` (`git ls-files`, which makes fresh-clone
    reproducibility definitional — an uncommitted local scratch file can no
    longer enter a release and make its hash unverifiable) plus
    `artifacts_fl/pregen/**` (bitwise reproducible). It EXCLUDES
    `foundation_learner/reports/**` — the evidence JSONs stay in GIT, where
    they are reviewable and history-tracked; the zip is not the evidence
    archive, and excluding them also removes the ordering trap in which
    re-running the validation invalidated a checksum snapshot the zip had
    already taken. It EXCLUDES `FOUNDATION_LEARNER_V0_MANIFEST.json`, which
    BINDS the zip hash and therefore cannot be inside the archive it describes.
    `SHA256SUMS` is not content: it covers every other bundled file and is the
    LAST file written and added, so it never covers itself. Packaging outside a
    git work tree REFUSES, and a RELEASE build (as opposed to a labelled
    throwaway dry run) REFUSES on a dirty work tree: under a git-tracked
    content policy an uncommitted module is silently ABSENT from the bundle, so
    "commit, then package" is not advice but an enforced precondition. Build
    order, corrected in `README.md`: validate -> package_release ->
    make_manifest (the manifest LAST, never covered by the zip).

19. **A seventh operator-bound unresolved field.**
    `b200_derived_unresolved.container_registry_digest_ref =
    UNRESOLVED_OPERATOR_BOUND_REGISTRY_PUSH`: the B200 image digest does not
    exist until the operator pushes the image, mirroring the O1 record
    (V-Obs6).

20. **The dress rehearsal's own wall-clock bound is raised to 1200 s.**
    The rehearsal now walks four more stages (`CORE_MATCHING` and the three
    unconditional diagnostics) and family-balanced evaluation sets, which
    triples the episode walks of the pre-existing stages. Measured on a 24-core
    CPU: 673 s total, of which REMAP_DIAG 218 s, POISON_DIAG 188 s,
    INTERFERENCE_DIAG 33 s, CORE_MATCHING 0.002 s. Contract §18 fixes no
    rehearsal wall-clock number (the 600 s figure was the rehearsal's own bound
    from Amendment 10 item 7); the bound is raised to cover the larger
    rehearsal rather than shrinking the mechanics it walks, and the measured
    elapsed time is reported in `REHEARSAL_REPORT.json` either way.

21. **README sealed-path claim aligned with Amendments 1/7/8.** The cipher
    primitives live in `ecology/base.py` and `data/shards.read_shard` will apply
    an explicitly supplied key; what is unique is the CAMPAIGN path —
    `campaign/sealed_gate.py` is the only module that DERIVES `K_seal` and the
    only route by which campaign code reads sealed data. The protection is
    procedural; no cryptographic unopenability is claimed.

## Amendment 13 — scientific-validity repairs (2026-08-09, W7, PRE-RUN)

Recorded by W7 after an adversarial review of the built package. Every item
below is frozen BEFORE any campaign run, on data that has not been produced by
the model under study. Nothing here weakens a test, a gate, a threshold or a
promotion rule; no frozen value of §7–§11 changes; no metric is removed or
replaced. Two things DO change by necessity and are stated explicitly: the
`graph_edge_semantics` generator (with its version, its hashes and its
regenerated data) and the eval-time sequence allowance (a new constant, not a
change to the §7 rendering budget).

### 1. Eval-time online context allowance = 4096 tokens (ENGINEERING FREEZE)

§7 fixes `max_seq_len` 2048 "per episode rendering" and forbids
generation-side truncation. That budget was applied to the ONLINE evaluator as
well, which is a category error: an online episode is not the rendering it
walks. Every one of the ten `MODEL_ATTEMPT` slots is replaced by up to 64 REAL
generated tokens, so the last prompt of a long episode projects to
`scripted_tokens + (n_attempt_slots + 1) x 64`. Measured on the tiny
pre-generation with the real Ouro tokenizer: longest scripted rendering 1740
tokens, longest projected online context 2444 — i.e. the current data really
does exceed 2048 online, confirming the reviewer's ~2334-2448 projection.

Frozen here, before any run:

- `evaluation.learning_curve.EVAL_ONLINE_MAX_SEQ_LEN = 4096` is the EVAL-TIME
  allowance and the default of `LearningCurveConfig.max_seq_len`;
- `MAX_SEQ_LEN = 2048` remains the §7 TRAINING-side rendering budget and is
  untouched (`data/pools.MAX_SEQ_LEN`, `training/tokenization`,
  `campaign/stage_definitions._arm_config` all keep it);
- the constant is duplicated in `data/pools.py` because nothing under `data/`
  may import the torch-dependent evaluation package; a test pins the two
  together.

This is an engineering constant, not a scientific one: the frozen backbone's
`max_position_embeddings` is 65536 (§1), 2048 was a data-generation constraint,
and 4096 covers the worst case present in the data with margin. Truncation
remains forbidden in every path.

### 2. Per-episode isolation of an over-long online episode

`evaluation/learning_curve.run_episodes` raised `SequenceBudgetError` on the
first offending prompt, which aborted the WHOLE batch — one long episode
destroyed every other episode's evidence, invisibly (no records were returned
at all). Repaired:

- `EpisodeWalker.abort_online_budget` stops THAT episode; the batch continues;
- its record carries `status = "ONLINE_BUDGET_EXCEEDED"`, `R = {}`,
  `aulc = None`, `R_missing_reason`, `partial_interaction_indices` (audit only)
  and `online_budget_event` (prompt tokens, projection, allowance,
  `truncated: false`, message);
- a partial curve is NEVER averaged: `R` is marked missing rather than
  reported short, so macro-AULC cannot silently average over a different index
  grid;
- `evaluation.metrics.excluded_records` reports the count, the statuses, the
  families and the episode ids, and `summarize` always includes it;
- `analysis/report.py` prints an "Episodes recorded but EXCLUDED" section.

Every episode record now also carries `status` and `eval_online_allowance`.
Records written before this amendment (there are none from a real run) are
treated as `COMPLETE`.

Hostile fixture: `tests/hostile/test_hostile_online_budget_overflow.py` builds
a real batch containing one over-long episode, injects an allowance measured
from the real renderings, and asserts batch survival, the recorded status, the
missing `R`, the absence of truncation, the exclusion count, and — the failure
mode it exists for — that the batch does NOT come back one record short.

### 3. Pre-generation asserts the ONLINE projection as well

`data/generate_shards.TokenBudget.check` now takes a required
`n_attempt_slots` and hard-fails when
`scripted_tokens + (n_attempt_slots + 1) x 64 > 4096`, for the canonical
surface AND every remapped variant. `PREGEN_MANIFEST.json` records
`eval_online_allowance`, `eval_max_new_tokens` and
`max_online_projected_seen`. Consequence: data that the evaluator would have
to exclude cannot be shipped in the first place.

### 4. `graph_edge_semantics` hint-node selection is answer-independent
(generator 1.0.0 -> 1.1.0)

`_feedback` drew the hinted node from the queried PATH when one existed and
from the SOURCE otherwise (§4's "one named node on the queried path"). The
SELECTION therefore carried the reachability bit outright. Measured over 2160
sampled items on generator 1.0.0:

| feature (computable from the prompt) | branch | n | P(answer) |
|---|---|---|---|
| hinted node is the source | False | 382 | YES = 1.000 |
| hinted node is the target | True | 291 | YES = 1.000 |

Generator 1.1.0 selects the hinted node uniformly over the DISPLAYED node list
with a generator seeded ONLY by displayed data (edges, source, target, node
count) — nothing that depends on the hidden edge semantics — and keeps the hint
content exactly as §4 requires (the TRUE out-degree under the hidden semantics
of the named node). It remains a pure function of the instance, so two
different attempts on one item get the same hinted node.

This DEVIATES from §4's wording "one named node on the queried path": that
phrase is precisely what made the selection answer-dependent (a path exists iff
the answer is YES), so it cannot be satisfied without the leak. Measured after
the repair: every model-computable branch sits within 0.05 of the base rate
(strongest 0.7225 vs base 0.6778 for `outdeg = 0`), against 1.000 before.

Consequences, all intended and none silent:

- `generator_version` "1.0.0" -> "1.1.0" for this family ONLY, which changes
  its `rule_id`, `instance_id` and episode ids;
- the family module's source hash changes, so `family_split_manifest.json` and
  `split_manifest_sha256` change, and therefore `K_seal` and the sealed shard
  bytes change; the pre-generated data must be REGENERATED by the integrator
  and `SHARD_SUMS.json` / `SHA256SUMS` / the release zip re-made;
- the SPLIT ITSELF DOES NOT CHANGE: the §5 rule hashes `family_id` strings, and
  no family id changes. Amendment 2's assignment stands.
- `tests/test_families_generation.py` pins the per-family version table, so a
  generator can never change again without its version moving.

### 5. Hint-decodability is now measured for every family

New hostile fixture `tests/hostile/test_hostile_hint_selection_leak.py` samples
2160 items per family (120 rules x 18 items), computes
`P(correct answer | branch)` for every discrete hint-derived feature, and
classifies each feature as `content` (what the hint says), `selection` (WHICH
displayed entity it names, expressed against the entities the QUESTION names —
computable by a reader from the prompt alone) or `diagnostic` (requires the
HIDDEN state to evaluate, therefore not a decode channel). It writes the full
measured table to a JSON under the test scratch directory. Assertions:

1. NO `selection` branch (n >= 30) is deterministic, for any family. The
   fixture proves by construction that this fires on generator 1.0.0: it
   re-runs the identical measurement against the old selection rule and
   requires the violation to appear.
2. The deterministic `content` branches are EXACTLY the frozen set
   `{(constraint_rules, violated), (grammar_classification, fails)}` — new
   entries fail, and a listed entry disappearing also fails.
3. Per-family measured decodability stays under pinned ceilings.

Measured per-family maximum over MODEL-COMPUTABLE branches (purity =
P(most likely answer | branch); base = that answer's base rate):

| family | base | max purity | max lift | deterministic |
|---|---|---|---|---|
| boolean_rule | 0.628 | 0.8934 (`pivot=ABSENT`) | 0.2652 | no |
| propositional_transform | 0.001 | 0.0556 | 0.0546 | no |
| modular_arithmetic | 0.093 | 0.0931 | 0.0000 | no |
| sequence_transform | 0.002 | 0.0333 | 0.0329 | no |
| string_rewrite | 0.033 | 0.0627 | 0.0299 | no |
| finite_state_transducer | 0.018 | 0.2353 | 0.2172 | no |
| permutation_composition | 0.001 | 0.0061 | 0.0056 | no |
| set_operations | 0.241 | 0.5823 | 0.3416 | no |
| graph_edge_semantics (1.1.0) | 0.678 | 0.7225 | 0.0447 | no |
| dsl_execution | 0.012 | 0.0257 | 0.0141 | no |
| constraint_rules | 0.884 | 1.0000 | 0.8838 | YES (by §4 content) |
| grammar_classification | 0.550 | 1.0000 | 0.5500 | YES (by §4 content) |

`boolean_rule`'s `pivot=ABSENT` branch (0.893 vs a 0.628 base rate) is RECORDED
AND PERMITTED: probabilistic evidence is the legitimate content of feedback,
and the fixture asserts that this branch stays both non-deterministic AND
informative. Only DETERMINISTIC decode branches are defects.

RECORDED LIMITATION, NOT REPAIRED HERE. `constraint_rules` (§4.11, "the number
of violated constraints") and `grammar_classification` (§4.12, "which atomic
predicate family the string fails") are answer-determining through their
CONTENT, not through a selection artefact: a violation count of zero IS
satisfaction, and failing neither predicate implies membership under both
admissible connectives. Removing that would change §4's frozen feedback
definitions for two families, which is a scientific-design decision outside
this repair's authority; it is pinned, measured and visible instead of unknown.
`constraint_rules` is a SEALED_TEST family and `grammar_classification` a TRAIN
family, so neither touches the DEVELOPMENT-side promotion decision; the effect
is that on those families a correct revision after INCORRECT feedback needs no
rule inference — which is exactly what item 7's `flip_attributable_success_rate`
measures and reports.

### 6. `answer_line_rate` (metric 15) and the FORMAT_NONCOMPLIANT annotation

The base model may never emit `ANSWER:` inside the frozen 64-token decode
budget (chain-of-thought preamble), in which case FL0 floors at 0 for a reason
that is not about learning. `evaluation/metrics.py` adds:

- `answer_line_rate` — the macro fraction of `MODEL_ATTEMPT` generations
  containing a parseable `ANSWER:` line, computed from the `parsed` field the
  walker already writes (the SAME strict grammar the verifier scores against),
  reported per arm, per family and per interaction index;
- `finish_reason_counts` — the `answer_line` / `eos` / `max_new_tokens` /
  `nonfinite_logits` breakdown, i.e. the direct evidence of "the answer never
  arrived inside 64 tokens";
- `format_compliance` / `format_flag` with the frozen threshold
  `ANSWER_LINE_RATE_FLAG_THRESHOLD = 0.5`.

`analysis/report.py` prints the answer-line rate next to every arm's macro-AULC
and annotates any cell below the threshold `FORMAT_NONCOMPLIANT`. The
annotation NEVER replaces, hides or adjusts the number. Prompts, headers and
`max_new_tokens` are frozen and untouched.

### 7. Flip-immune AULC (metric 16) and flip attribution (metric 17)

On a two-label family, "emit the other label after INCORRECT" earns credit at
the revision indices with no rule inference at all. Frozen additions:

- `POST_FEEDBACK_FRESH_INDICES = (4, 5, 6)` and
  `aulc_post_feedback_fresh` — macro-AULC restricted to the fresh items, where
  a flip cannot pay. Reported with its own clustered interval next to the
  headline macro-AULC, and as a paired delta next to every headline delta.
- `flip_attributable_success_rate` — over the binary-answer families
  (`BINARY_ANSWER_FAMILIES = boolean_rule, constraint_rules,
  grammar_classification, graph_edge_semantics`, pinned and checked against the
  real generators by a test), the share of successes at indices 1 and 3 that
  followed an INCORRECT verdict on the same item AND changed the emitted label.
  Computed from the transcripts already in the records.

### 8. FL2 comparison reported with and without interaction index 0 (R-M10)

FL2 is trained to imitate attempt0 behaviour, so a delta against FL2 that
includes index 0 mixes an imitation handicap into a claim about learning from
feedback. `analysis.stats.arm_comparison` gains an `indices` parameter (same
paired machinery, same shared resample plan) and `analysis/report.py` now emits,
for EVERY headline comparison, both companions:
`INDICES_EXCLUDING_ZERO = (1, 2, 3, 4, 5, 6)` and the fresh
`{4, 5, 6}`. The headline delta is unchanged and still the frozen metric 2/3.

### 9. FL5: the `FAST_STATE_ON_S0` control and the corrected claim (R-M5)

`fl5_training.py` claimed the ON/OFF contrast was "exactly 'the state carried
the learning' and nothing else". That OVERCLAIMED. The docstring is corrected in
place to the accurate segment-to-segment statement and names the three
confounds: (1) FAST_STATE_OFF is an impossible-task control (segmentation
removes the history and OFF has no state, so it cannot use any earlier
interaction at all); (2) ON trains ~25M parameters OFF does not have; (3)
`prefix_embeds(0)` is a LEARNED STATIC PREFIX, i.e. ordinary prefix tuning.

New EVAL-ONLY control arm `FAST_STATE_ON_S0` (`FastStateS0Callbacks`): the
TRAINED ON module evaluated on the same DEVELOPMENT episodes with `s` pinned to
0 and never updated. There is NO third training run. `run_fl5_stage` runs it on
the ON arm's own bundle and publishes `state_update_contribution`
(ON - ON_S0, the state-update effect) and `static_prefix_contribution`
(ON_S0 - OFF, the static prefix plus the extra parameters), both through the
same §14 paired family-clustered bootstrap. Confounds (1) and (2) are recorded,
not removed.

Every FL5 payload, every per-arm entry and every `FL5ArmResult` now carries
`comparable_to_fl3: false` with the reason string
(`fl5_training.FL5_NOT_COMPARABLE_REASON`), so an FL5 number cannot be read off
next to an FL3 number without the disclaimer travelling with it.

### 10. `hint_for_condition` applies the answer-leak invariant (minor a)

`ecology.poison.hint_for_condition` was the one hint-rendering path with no
`answer_leak` check: a corruption, an irrelevant-pool body or a redundant
restatement could have carried the answer with nothing firing. It now takes a
REQUIRED keyword-only `answer_canonical` and applies exactly the check
`TaskFamily.feedback_record` / `corrupted_feedback` apply, raising
`AnswerLeakError`. Required rather than optional because an
optional-and-omitted check is the failure mode being repaired; the three
callers (`episodes/assemble.py` x2, `evaluation/learning_curve.py`) pass the
pending item's canonical answer.

### 11. Episode records carry the fields the new metrics need (minor b)

`finish_reason` and `parsed` were already written for every attempt;
`tests/test_evaluation_learning_curve.py` now asserts it, so the format metric
can never silently lose its input.

### Cross-worker consequences (for the integrator)

1. REGENERATE the pre-generated data (`scripts/pregenerate_all.py`) after this
   change: `graph_edge_semantics` ids, the family-split manifest, the sealed
   key and all shard sums move. The split assignment itself does not move.
2. Re-make `SHARD_SUMS.json`, `SHA256SUMS`, the release zip and the filled
   manifest.
3. The preregistration should state metrics 15–17 and the two restricted
   deltas as pre-run additions (the prereg text is the integrator's file).

## Amendment 14 — the last two answer-decoding hints are REPAIRED, not excused (2026-08-09, W7, PRE-RUN)

Amendment 13 item 5 pinned two families as a recorded limitation:
`constraint_rules` and `grammar_classification` decoded the label through their
frozen §4 feedback CONTENT. That carve-out is withdrawn. `constraint_rules` is
a SEALED_TEST family, and a permanent deterministic-decode exemption on the
sealed set would undermine the principal claim; nothing has run, the sealed set
is unopened and no development decision exists, so this is a legitimate
preregistration-time repair rather than a post-hoc adjustment. Amendments 12
and 13 are unmodified; this amendment supersedes Amendment 13 item 5's
"RECORDED LIMITATION" paragraph and its allow-list.

### 1. `constraint_rules` 1.0.0 -> 1.1.0: candidate-constraint probe

§4.11 froze the feedback as "the number of violated constraints (count only)".
That count IS the answer: zero violations is satisfaction and any positive
count is violation, so every branch decoded the label (measured P = 1.000 on
all seven branches with n >= 30).

1.1.0 reports instead ONE CANDIDATE CONSTRAINT plus the displayed assignment's
relation to it:

* `CANDIDATE_POOL` (24 entries) enumerates, over the first THREE displayed
  slots, every `(i, j, relation)` with relation in {LT, GT, NEQ} and every
  "slot i must not equal v" for v in 1..5. The candidate NEED NOT be an active
  constraint of the hidden rule.
* The hint is `probe=CND_<code>; status=STA_MET|STA_UNMET` — one opaque
  positional code plus the true status of the DISPLAYED assignment under that
  candidate.
* The probe is selected by a generator seeded from DISPLAYED data only (the
  assignment values and the displayed slot count), never from the hidden
  constraint set and never from the answer, and is a pure function of the
  instance (both attempts on an item see the same probe).

Why an inactive candidate is the whole point: a candidate that the assignment
violates may not be in the rule (so the item can still be SAT), and a candidate
that holds says nothing about the constraints that were not probed. No branch
implies the label.

Two design details that are load-bearing, both discovered by measurement:

* the pool is INSTANCE-INDEPENDENT (`MAX_PROBE_SLOTS = 3`; every rule has at
  least three slots, so these candidates always apply). A pool whose SIZE
  varied with the displayed slot count made the opaque code disclose that
  count, and the displayed slot count is itself strongly predictive of the
  label (more slots -> more constraints -> almost always UNSAT). With a
  slot-count-dependent pool, 13 of 55 code branches predicted UNSAT with
  certainty for exactly that reason. Fixing the pool removed all of them.
* the candidate is carried by ONE code, not by several fields. A multi-field
  encoding let the deterministic corruption emit a structurally impossible
  record (`left == right` under `LT`), and a malformed hint would disclose that
  the hint is poisoned, which §6 forbids. With one code, corruption can only
  move the probe to another well-formed candidate and/or flip the status; it is
  still guaranteed to differ from the truth by the existing generic machinery.

Status codes are `MET`/`UNMET` because they may not collide with any answer
label of this family (`SAT`/`UNSAT`, `VALID`/`INVALID`, `HOLDS`/`FAILS`,
`PASS`/`REJECT`) under any surface — the ("F","T") lesson of Amendment 7.4.

### 2. `grammar_classification` 1.0.0 -> 1.1.0: pool-predicate probe

§4.12 froze the feedback as "which atomic predicate family the string fails, by
opaque predicate index". Two of its four branches decoded the label: failing
NEITHER conjunct implies membership under both admissible connectives, and
failing BOTH implies non-membership under both (measured P = 1.000 at n = 541
and n = 576).

1.1.0 reports `probe=PRD_<code>; status=STA_PASS|STA_FAIL`, where the code
indexes `PREDICATE_POOL` — a frozen, rule-independent enumeration of all eight
predicate schemas over the alphabet POSITIONS (50 entries; positions, not
literal symbols, so one code denotes the same predicate under every surface
remap). The probe is seeded from the displayed string only, never from the rule
and never from the answer. A pool predicate outside the rule can fail while the
string is IN, so no branch implies the label.

### 3. Informativeness, stated honestly

Both hints stay informative in the sense the campaign needs: the probe identity
is a STABLE POSITIONAL code into a frozen pool, so across feedback rounds a
learner accumulates (candidate, status, certified verdict) triples over many
items and eliminates the candidate rules that cannot explain the verdicts —
which is the rule identification these families exist to test. The module
docstrings say this, and also say what is given up: the probe STATUS is
computable from the displayed prompt, so the hint's value is that it points at
a hypothesis from the family's own template pool and pre-computes it, not that
it discloses hidden state.

The obvious stronger design — also reporting whether the probed candidate is
ACTIVE in the hidden rule — was REJECTED: "active AND violated" implies UNSAT
with certainty, and "is a conjunct AND fails" implies OUT under AND with
certainty, i.e. it would reintroduce exactly the defect being repaired. A hint
that discloses hidden state without ever determining the label is not available
in these two families; the probe design is the informative maximum that is
provably leak-free.

### 4. The hostile fixture has NO allow-list any more

`tests/hostile/test_hostile_hint_selection_leak.py`:

* `DETERMINISTIC_CONTENT_BY_DESIGN` is now EMPTY and asserted empty, so
  re-introducing an exemption is a visible edit to a named object;
* the primary assertion is that NO branch of ANY model-computable feature
  (`content` or `selection`) of ANY of the twelve families is deterministic.
  Features that require the HIDDEN state to evaluate (`graph_edge_semantics`'s
  "the hinted node lies on the true path") stay classified `diagnostic`: a
  reader cannot compute them without having already solved the task, so they
  are measured and recorded but are not a decode channel;
* non-vacuity is proved by construction for ALL THREE repairs — the identical
  measurement is re-run against a frozen inline reimplementation of each
  family's OLD hint rule, and the violation is REQUIRED to appear (graph's
  `at_is_source` / `at_is_target`, constraint_rules' `violated` count including
  the SAT-decoding zero branch, grammar_classification's `BOTH` and `ABSENT`
  branches);
* new tests pin that each probe reports the TRUE status of the named candidate,
  that it is a pure function of the instance (not of the attempt), and that the
  probe code RANGE does not vary with any instance covariate;
* the sampling plan is raised to 360 rules x 18 items = 6480 items per family.
  This is a POWER requirement, not a relaxation: at `constraint_rules`' 0.893
  base rate a 32-item branch is accidentally pure about 2 % of the time, so
  small branches make "P = 1.0" ambiguous. The fixture now ASSERTS its own
  power (`base_rate_top ** smallest_branch < 1e-4`, smallest measured branch =
  57 items) and records each perfect branch's chance probability. The criterion
  itself is unchanged and fail-closed: a perfect branch fails the fixture
  whatever its chance probability says.

### 5. Measured decodability after the repair (6480 items/family)

purity = P(most likely answer | branch) over model-computable branches;
base = that answer's base rate; no family has a deterministic branch.

| family | ver | base | max purity | max lift | smallest branch |
|---|---|---|---|---|---|
| boolean_rule | 1.0.0 | 0.6332 | 0.8861 | 0.2529 | 263 |
| propositional_transform | 1.0.0 | 0.0009 | 0.0476 | 0.0468 | 105 |
| modular_arithmetic | 1.0.0 | 0.0843 | 0.0843 | 0.0000 | 6480 |
| sequence_transform | 1.0.0 | 0.0008 | 0.0198 | 0.0195 | 101 |
| string_rewrite | 1.0.0 | 0.0295 | 0.2024 | 0.1729 | 66 |
| finite_state_transducer | 1.0.0 | 0.0137 | 0.1553 | 0.1431 | 62 |
| permutation_composition | 1.0.0 | 0.0003 | 0.0027 | 0.0026 | 368 |
| set_operations | 1.0.0 | 0.2483 | 0.5828 | 0.5477 | 57 |
| graph_edge_semantics | 1.1.0 | 0.6682 | 0.7405 | 0.0723 | 247 |
| dsl_execution | 1.0.0 | 0.0110 | 0.0225 | 0.0116 | 3150 |
| constraint_rules | 1.1.0 | 0.8934 | 0.9408 | 0.0474 | 180 |
| grammar_classification | 1.1.0 | 0.5048 | 0.6045 | 0.1093 | 99 |

Before the repairs the same measurement gave P = 1.000 for
`graph_edge_semantics` (selection), `constraint_rules` (all seven count
branches) and `grammar_classification` (`BOTH`, `ABSENT`).
`boolean_rule`'s `pivot=ABSENT` branch (0.886 against a 0.633 base rate) remains
RECORDED AND PERMITTED: probabilistic evidence is the legitimate content of
feedback, and the fixture asserts that this branch stays both
non-deterministic and informative.

### 6. Consequences for the integrator

`split_manifest_sha256` changes AGAIN (three family module hashes moved):
`983c4ed1fc4ea240…` (original) -> `1f4901219c3a4e26…` (Amendment 13) ->
`5e91a4548428663c…` (this amendment). Therefore `K_seal` and every sealed shard
change again. REGENERATE THE FULL PREGEN ONCE, after this amendment, and re-make
`SHARD_SUMS.json`, `SHA256SUMS`, the release zip and the filled manifest. The
tiny pre-generation has already been regenerated and verified against
`5e91a4548428663c…`. Instance ids, rule ids and episode ids move for
`constraint_rules` and `grammar_classification` as well as
`graph_edge_semantics`; the SPLIT ITSELF STILL DOES NOT CHANGE, because §5
hashes `family_id` strings and no family id changed (Amendment 2's assignment
stands).

### 7. Observation recorded, NOT repaired: `constraint_rules` label imbalance

The measurement incidentally shows that `constraint_rules` items are 89.3 %
UNSAT and `boolean_rule` 63.3 % one label. That is a property of the frozen §4
item samplers, not of any hint, and it means a constant-answer policy scores
0.893 on that family's R_k. It is NOT touched here (changing item distributions
is a different, larger scientific decision than removing a decode channel), but
it is recorded so that no aggregate is read without it. `constraint_rules` is a
SEALED_TEST family.

## Amendment 15 — `constraint_rules` label balance and the constant-answer floor (2026-08-09, W7, PRE-RUN)

Amendment 14 item 7 recorded, without repairing, that `constraint_rules` items
were 89.3 % UNSAT: a constant "always answer UNSAT" policy scored 0.893 on that
family's R_k, so no learning curve on it could be read. That is repaired here.
`boolean_rule`'s 0.633 imbalance is RECORDED AND ACCEPTED unchanged — a hidden
Boolean function is not expected to be balanced over assignments, and 0.633 is
not a degenerate floor. Amendments 12, 13 and 14 are unmodified. This is the
LAST generator change on the science side.

### 1. `constraint_rules` 1.1.0 -> 1.2.0: label-balanced item sampler

Two conditions, both deterministic, both leaving every instance a pure function
of `(rule, seed, difficulty, surface_map_id)`:

* **per item (the sampler).** `_item` draws a FAIR TARGET LABEL from the item's
  own generator and then REJECTION-SAMPLES assignments from the same generator
  until one carries that label, with a budget of
  `LABEL_DRAW_ATTEMPTS = 512`; the fallback is the FIRST draw, so an item is
  always well defined. Assignments are still drawn uniformly from the value
  grid, so a balanced item carries no shape signature: a SAT item's assignment
  is a uniform draw from the satisfying set, and nothing about the order or
  magnitude of the values encodes the label. Applies to support, related, query
  and transfer generation alike, and at every difficulty.
* **per rule (the necessary precondition).** 14 % of 1.1.0 rules were outright
  UNSATISFIABLE at the frozen campaign difficulty and a further 14 % were
  satisfiable with probability < 0.005, so for those NO per-item budget can
  produce a SAT item. `sample_rule` now redraws from the SAME frozen §4.11 rule
  space until `satisfiability_rate(rule) >= RULE_MIN_SAT_RATE = 0.005`
  (estimated by a 256-draw Monte Carlo seeded FROM THE RULE SPEC, so the
  estimate and hence the rule stay pure functions). Exhausting the bounded
  redraw budget is a hard error, never a silent acceptance.

The rule condition is deliberately the mildest one that works: it keeps ~72 % of
the 1.1.0 rule space and removes only the degenerate tail. (Conditioning rules
into a satisfiability BAND of [0.2, 0.8] would have kept 19 % and would have
changed the latent-rule distribution far more.) Rule diversity remains ample:
40 sequential draws give 40 distinct rules, 3000 draws give 2464 distinct rule
ids, and the deterministic uniqueness loop needed at most 3 redraws to place a
full 300-rule SEALED pool.

Predicted rates without the rule condition are recorded for honesty: per-item
rejection ALONE would have reached only ~0.40 SAT at a 512 budget and ~0.43 at
8192, because the unsatisfiable tail is unreachable by any budget.

**Achieved rate (measured, 2400 items over 200 rules, all four instance kinds):
P(SAT) = 0.4904** — by kind: support 0.5067, related 0.4650, query 0.5083,
transfer 0.4817; by difficulty: 0.4958 / 0.5028 / 0.4972. The constant-answer
floor on this family therefore falls from **0.893 to ~0.50**.
`tests/test_families_generation.py::test_label_answer_families_are_not_degenerate`
pins it over >= 2000 items with the honest finite-sample tolerance
[0.35, 0.65], and also pins `graph_edge_semantics`, `grammar_classification`
and (at a wider, status-quo bound) `boolean_rule`.

### 2. Metric 18: the constant-answer floor is reported next to every family

`evaluation/metrics.constant_answer_baseline` computes, per family, the AULC
that the best "always answer X" policy would score, using the metric module's
OWN aggregation (per candidate constant, the mean over interaction indices of
the fraction of that index's attempts whose canonical answer is that constant,
averaged over episodes), so it is directly comparable to `per_family_aulc`.
`aulc_above_constant_answer_floor` reports the per-family difference.
Open-answer families (more than `MAX_LABEL_ANSWERS = 4` distinct canonical
answers) report `None` together with the observed number of distinct answers,
so the omission is visible rather than silent; families whose records carry no
canonical answers likewise report `None` with the reason.

`analysis/report.py` prints a "Per-family results against the constant-answer
floor" table (arm, family, macro-AULC, floor, above-floor) and annotates any
cell at or below its floor `AT-OR-BELOW-FLOOR`. This is PURE REPORTING: no
gate, threshold, promotion rule or frozen value consumes it, and no number is
replaced or adjusted.

### 3. Hint-decodability re-measured after balancing

The balanced sampler changes `constraint_rules`' answer distribution, so the
hostile fixture's table was regenerated. New row (6480 items, 24 branches, no
allow-list):

| family | ver | base | max purity | max lift | smallest branch | deterministic |
|---|---|---|---|---|---|---|
| constraint_rules | 1.2.0 | 0.5034 | 0.5734 | 0.0768 | 202 | none |

Its pinned ceiling tightens from 0.96 to 0.65 accordingly. The power assertion
still holds with the new base rate (0.5034 ** 202 is vanishing), and the other
eleven rows are unchanged from Amendment 14. No family has a deterministic
model-computable branch.

### 4. Consequences for the integrator

`split_manifest_sha256` moves once more — `983c4ed1fc4ea240…` (original) ->
`1f4901219c3a4e26…` (Amd 13) -> `5e91a4548428663c…` (Amd 14) ->
**`6cf330b47af9406edf8022564ea330f3f164fbaef22618710a11fa3b2f40d9b5`** (this
amendment) — so `K_seal` and every sealed shard change again. This is the FINAL
science-side generator change: regenerate the FULL pregen ONCE now, then re-make
`SHARD_SUMS.json`, `SHA256SUMS`, the release zip and the filled manifest. The
tiny pre-generation has already been regenerated and verified against
`6cf330b4…`. `constraint_rules` instance ids, rule ids and episode ids all move;
the SPLIT ITSELF STILL DOES NOT CHANGE (§5 hashes `family_id` strings, and no
family id changed).

### 5. What is NOT changed

`boolean_rule` (0.633 majority label) is unchanged by decision; its floor is
now printed next to its results by metric 18 rather than being invisible. No
other family's item sampler is touched, and no frozen threshold, weight, pool
size, promotion rule or split moves.

## Amendment 16 — train-mode decoding, and a correction to Amendment 12 item 16 (2026-08-09, W6)

Recorded by W6 after the reviewer's confirmation pass, which verified the
Amendment 12 repairs and found one CRITICAL defect that the batch itself had
hardened. Frozen before any accelerator use, before any training run, with the
sealed set unopened. Nothing here changes an objective, a weight, a metric, a
threshold, the split, or the sealed-test policy.

1. **CORRECTION of Amendment 12 item 16 (do not read that item as current).**
   Amendment 12 item 16 stated that layer-level gradient checkpointing is a
   NO-OP on this architecture and recorded
   `layer_level_checkpointing_detection = NOT_CONSULTED_BY_FORWARD`. **That was
   wrong.** `OuroDecoderLayer` inherits
   `transformers.modeling_layers.GradientCheckpointingLayer`, whose `__call__`
   reads the flag:

       if self.gradient_checkpointing and self.training:
           ... kwargs["use_cache"] = False; kwargs["past_key_values"] = None
           return self._gradient_checkpointing_func(...)

   The detector had inspected only each concrete class's own source
   (`modeling_ouro.py`) and never its base classes, so it missed the one place
   the flag is actually read. Layer-level checkpointing is therefore ACTIVE
   during training on this backbone. `layer_checkpointing_is_consulted` now
   walks the whole MRO; verified on the tiny model: the detector returns
   `True`, and the record says `CONSULTED_BY_FORWARD`.

1b. **Two independent gates converged on this.** The adversarial reviewer's
   confirmation pass raised it as N1 and the independent verifier raised the
   same defect as its Defect A, having measured **1,600** "Caching is
   incompatible with gradient checkpointing" warnings in a single rehearsal at
   the pristine commit — one per decoded layer-call in train mode. The verifier
   independently confirmed the detector blind spot (Defect B) that Amendment 12
   item 16 rested on. Two independent gates reaching the same conclusion from
   different evidence is why this is recorded as a correction rather than as a
   judgement call.

2. **The defect the correction exposes (CRITICAL).** The same base-class branch
   DROPS `use_cache` and `past_key_values` in train mode. The package's
   evaluation decoder is a manual greedy loop that maintains its OWN KV cache
   (§22), so a model left in train mode with the flag set silently recomputes
   from scratch and emits DIFFERENT text. Measured on the tiny model with
   identical weights and prompts: eval-mode decode
   `[' eclipse eclipse eclipse …', 'itization gaining embry …']` versus
   train-mode decode
   `[' eclipseimore quantum Cambridge …', 'itization voila Coin topsoil …']`.
   Nothing raised; the only trace was the log line "Caching is incompatible
   with gradient checkpointing", of which the rehearsal log was full. Every
   trained arm is dev-evaluated immediately after training, so this affected
   the FL1/FL2/FL3 development metrics, the grid selection that consumes them,
   and every mechanism rung's evaluation.

3. **Repairs, defence in depth.**
   * `training/trainer.py` — `run_training_arm` now RESTORES eval mode and
     disables layer-level checkpointing before returning, and records the
     transition in `result.trainable["post_training_mode"]`. The post-condition
     is documented in its docstring: the returned model is valid to decode.
   * `mechanisms/fl5_training.py` — the same post-condition for FL5's own
     training loop (its arms are evaluated immediately afterwards).
   * `campaign/stage_definitions.py` — `prepare_bundle_for_evaluation()` is
     applied at every evaluation boundary: `_run_dev_eval`, `fl0_work`,
     `promoted_arm_bundle` (which restores trained weights and is used by the
     sealed evaluation and all three diagnostics), and the BENCH evaluation
     probe, so the MEASURED evaluation cost is measured in the state evaluation
     really runs in.
   * `mechanisms/stage_support.py` — `dev_records` enforces the same state,
     and `prepare_for_evaluation()` exposes it to the rungs.
   * `mechanisms/consolidation.py` + `mechanisms/fast_adapter.py` — the guard
     immediately caught a SECOND live instance: FL8 walks its A->B->A chains
     through `run_interference_eval` directly (not through `dev_records`) after
     the fast-adaptation inner loop has put the model in train mode, and
     `FastAdapter.inner_update` restored "whatever the mode was" rather than
     the evaluation state — which preserves an invalid state whenever the
     bundle arrived in train mode (every fresh tiny bundle does). The inner
     loop now ALWAYS returns to eval, and the FL8 stage prepares the bundle
     before the interference walk. This is what a structural refusal is for:
     the defect was found by the guard, not by a reviewer.
   * `evaluation/generation.py` — `assert_decodable(model)` at the decode entry
     (`_decode_group`, i.e. on every path through `greedy_generate` and
     `greedy_generate_detailed`) REFUSES a train-mode model and refuses a model
     with layer-level checkpointing still enabled. This is what makes the
     invariant structural rather than a convention: a missed `eval()` is now
     impossible to ignore instead of invisible in the numbers.
   * `training/model_loading.py` — `set_evaluation_mode()` /
     `disable_gradient_checkpointing_()` are the single implementation every
     caller uses, and the identity record now states
     `cache_disabled_while_training: true` plus
     `evaluation_requires_eval_mode`, instead of denying the mechanism.

4. **Permanent regression fixture.**
   `tests/hostile/test_hostile_train_mode_decode.py` pins the defect itself
   (bypassing the guard to show train-mode decoding still differs — so the
   guard can never be "simplified" away on the belief that it guards nothing),
   the refusals, and the end-to-end property: a trained arm's generations are
   IDENTICAL to those of a fresh eval-mode bundle carrying the same weights.

5. **N2 — a timing abort may not consume a sealed attempt.**
   `ctx.checkpoint("sealed:walk")` sat INSIDE the
   `with sealed_gate.sealed_opening(...)` block, so a stage-watchdog overrun
   raised there would have been recorded as one of the two permanent opening
   attempts for a reason unrelated to the sealed evaluation. The watchdog
   checkpoint is now `sealed:before_opening`, immediately before the opening;
   no watchdog check remains inside the opened block.

6. **N3 — the dress rehearsal's wall-clock bound is ADVISORY and host-scaled.**
   The same rehearsal measured 673 s and 813 s on the same machine depending on
   what else was running, so a hard wall-clock gate makes an unrelated process
   a failed validation. `within_wall_clock_limit` is now listed in
   `ADVISORY_CHECKS`: it is measured, reported, and printed as an advisory, but
   it no longer decides the verdict. The budget is additionally scaled by a
   measured single-core benchmark (`measure_host_speed`) so a slow host is
   judged against itself. The FIFTEEN substantive checks (states walked, O1
   ordering, every stage's outcome, dev decisions frozen, sealed ledger,
   transfer archive, result-manifest verification, determinism configured,
   core matching, diagnostics, outcome complete) remain HARD.

7. **Defect C — a fresh clone must not fail on absent DATA.**
   `tests/test_campaign_entry.py::test_build_stage_context_binds_the_session_configuration`
   builds a real `StageContext`, which refuses a missing pre-generation root
   (that refusal is itself a test, with an explicitly nonexistent path). In a
   fresh clone, where `artifacts_fl/pregen_tiny` has not been staged yet, it
   therefore HARD-FAILED while its siblings correctly skipped. Both tests that
   need the staged data now carry the same `requires_tiny_pregen` marker.
   Verified by pointing the module at a nonexistent pregeneration root:
   `7 passed, 2 skipped`, no failures. Absent data is a skip; absent behaviour
   is a failure.

8. **RESIDUAL RISK, recorded rather than repaired (N4).** The two-phase sealed
   opening leaves a crash window between `commit()` (the `SEALED_OPENED` entry)
   and the immutable result writes. A crash inside that window consumes the
   opening while leaving the results unwritten, and the retry budget does not
   apply because the seal is committed. This is inherent to "evidence commits
   the seal": the alternative — committing after the writes — reopens the
   larger hole that a write failure leaves an unrecorded read of sealed data.
   The window is now milliseconds of local file writes rather than the entire
   evaluation, the records exist in memory at that point, and the ledger's
   `records_read` / `evaluation` payload documents what was read even if the
   result files are missing. Not repaired; recorded.
