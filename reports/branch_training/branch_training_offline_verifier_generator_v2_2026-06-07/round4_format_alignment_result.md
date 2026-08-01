# Round 4 (C_r4 on v5 eval-aligned data) — result note (2026-06-13)

ROUND4_VERDICT = FORMAT_ALIGNMENT_FIXES_DEGENERATION_BUT_NO_REACHABILITY_GAIN
OFFLINE_VIEWS_V5_VERDICT = V5_EVAL_ALIGNED_READY
K_CANARY_SELECTION_VERDICT = NO_ADAPTER_IMPROVES_ON_CANARY (all 10 arms, 4 rounds)

## Diagnosis carried in from rounds 1–3

Rounds 1–3 repaired the **content** channel domain by domain
(`round2_executed_data_result.md`, `round3_multidomain_result.md`): logic executed renders
−0.233 → 0.000; math rationales −0.53 → −0.20 → −0.13; canonical-solution coding *backfired*
−0.067 → −0.467 (terse "just write the function" style → confident wrong code). Macro stayed
negative and selection stayed `NO_ADAPTER_IMPROVES_ON_CANARY` for all 9 arms / 3 rounds.

What no round touched is the **format** channel. The canary and the real pool generator both
render `B1.build_prompt(tok, task, scaffold)` — GEN_SYSTEM(/MATH/CODE) posture, the tokenizer
chat template, and (for coding) the MBPP function-name note — and sample **one single solution
per (scaffold, sample)**. Training instead fed the *bare* task prompt through the trainer's
`_fmt`, which wrapped it under the tokenizer default system line ("You are a helpful
assistant.") with **multi-branch "Branch 1/2/3 … FINAL ANSWER" completions**. So every prior
round trained a prompt/response shape the model never sees at generation time. The scorer only
reads the first branch anyway (`trim_generation` cuts at the first `FINAL ANSWER:`,
`extract_code` takes the first ```python block), so the extra branches were pure wasted budget.

## Round-4 fix (format alignment, subsumes the content fixes)

`build_eval_aligned_views_v5.py` → `data/.../train_v5/`:

1. **Prompt = exact eval rendering.** Every row's prompt is `B1.build_prompt(tok, task,
   scaffold)`. Proven byte-for-byte identical to what the canary's `_gen_branches` builds for
   the same task (coding *with* fn-note, and math) — see validation below.
2. **Single-solution completions**, no "Branch i:" prefixes. Gen-pool rows use the branch's
   **own** scaffold (inverse `SCAFFOLD_FMT`), matching how that text was generated and how the
   deployment generator samples scaffolds; dataset rationales use "direct" (matching the canary).
3. **Coding = verified reasoned code** from the 60 gen-pool groups (unit-test-labeled),
   verbatim fenced ```python, **no appended FINAL ANSWER line** (which would break
   `extract_code`'s def-to-end fallback). This is the reasoned-code substrate round 3 named as
   missing — deliberation + implementation, not bare canonical functions.
4. **Math** = GSM8K/Hendrycks rationales with `<<…>>` calculator markup **stripped** (never done
   before) + `FINAL ANSWER: <gold>`; DPO adds **reasoning_error** negatives (corrupted mid-work
   number + matching wrong final) alongside show_work and wrong_final.

Trainer patch (`train_offline_branch_generator_v2.py:_fmt`): a prompt already containing
`<|im_start|>` is used **verbatim** and the turn is closed with `<|im_end|>`, instead of being
re-wrapped through `apply_chat_template` (which would nest the whole ChatML string inside a
fresh user turn under the default system line — the very mismatch being fixed). DPO needs no
trainer change: trl concatenates raw prompt+completion and appends EOS.

## Built view counts (`eval_aligned_views_v5.md`)

- branch_set_rejection_sft **21,817** (logic 12,000 · math 8,893 · coding 924 = 77 unique × 12 dup)
- branch_set_dpo **28,172** (oracle 12,172 · wrong_final 6,000 · reasoning_error 6,000 · show_work 4,000)
- one_branch_sft **13,137**

Scope cuts, grounded: **reasoning excluded** — gen shards did not persist MCQ options, so those
prompts cannot be re-rendered faithfully (and reasoning never regressed). **v3/v4 rows not
carried** — they are the misaligned format round 4 removes.

## Validation evidence (pre-training, all green)

- **Prompt parity:** v5 prompt == `V.build_prompt(tok, canary_task, "direct")` byte-for-byte for
  both coding (incl. the `MUST be named` fn-note) and math. This is the round-4 contract, proven
  not asserted.
- **Format contract** (200-row sample): prompts chat-templated / carry GEN_SYSTEM / end with
  `<|im_start|>assistant`; completions never start with "Branch "; non-coding end with FINAL
  ANSWER; coding fenced with no FINAL ANSWER. All True.
- **No train-on-test:** canary↔train_v5 prompt overlap handled by normalized-prompt exclusion —
  50 GSM8K/Hendrycks rationale rows that coincide with canary math tasks were dropped; gen-pool
  overlap is genuinely 0/240; coding 0/60.
- **Patched `_fmt`:** produces a single-turn ChatML exchange, ends with `<|im_end|>`, no
  "helpful assistant" double-wrap.
- **DPO well-formedness:** 28,172 rows, 0 malformed, 0 chosen==rejected.
- **Token gates:** SFT prompt+completion ≤ 1000 (< trainer max_length 1024) so FINAL ANSWER is
  never truncated; DPO prompt ≤ 640 / total ≤ 1020 (matching DPOConfig). Drops: sft 229, dpo 4.

## Arm + pipeline

`configs/C_offline_verifier_dpo_r4.json` (clone of r3; `data_root: train_v5`; SFT+DPO, 500 steps,
bf16 LoRA r16, base/tokenizer untouched, STOP-pausable). Pipeline:
`run_offline_v2_round4_pipeline.sh` (build → train → canary → combined selection).

## Result (training: SFT 300 steps loss 0.308 · DPO 200 steps loss 0.067; adapter 121 MB)

Paired canary (`evaluate_offline_branch_generator_canary_checkpoint_v2.py`, seed 1234,
30/domain + full 110 alignment; identical tasks for base and every arm). C-lineage deltas vs base:

| arm | macro | math | coding | logic | reasoning | chars_vs_base | degen_alarm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | :--: |
| C (r1, meta-text) | −0.244 | −0.533 | −0.500 | −0.233 | +0.067 | 0.264 | **True** |
| C_r2 (executed logic) | −0.052 | −0.200 | −0.067 | 0.000 | +0.033 | 0.401 | **True** |
| C_r3 (+rationale math, canonical code) | −0.112 | −0.133 | **−0.467** | 0.000 | +0.067 | 0.337 | **True** |
| **C_r4 (eval-aligned format)** | **−0.059** | **−0.067** | **−0.067** | −0.067 | −0.067 | **0.697** | **False** |

Base (same subset): coding 0.500 · logic 0.833 · math 0.867 · reasoning 0.833 · alignment 0.536.
r4 absolute: coding 0.433 · logic 0.767 · math 0.800 · reasoning 0.767 · alignment 0.509.

Per-domain mean branch chars (r4 vs base): coding 1611/1746 = **0.92×**, logic 735/1091 = 0.67×,
reasoning 495/839 = 0.59×, math 353/863 = 0.41×. r4 parse_ok: logic/math/reasoning 1.00, coding 0.91.

## Reading

1. **Format alignment fixed the degeneration.** r4 is the **first C-lineage arm to clear the
   degeneration alarm** (chars 0.70× base vs 0.26–0.40× for r1–r3). The output-length collapse
   that defined rounds 1–3 was substantially a **train/eval format-mismatch artifact**, not proof
   that offline training is intrinsically harmful. Math at 0.41× is *concision* (tight rationale
   style, parse 1.0, reachability only −0.067), not *collapse* (r1's bare-answer −0.533).
2. **Reasoned-code data fixed the coding catastrophe.** Round 3's canonical-solution coding
   crushed coding to −0.467 at 0.34× length; r4's gen-pool **reasoned** code (deliberation +
   fenced implementation) holds coding at **0.92× length and −0.067 reachability** — the single
   biggest round-4 win, confirming round 3's hypothesis that the coding substrate had to be
   deliberation-bearing.
3. **Uniform, within-noise neutrality, not a measured regression.** r4 lost exactly ~2/30 tasks
   in every gen domain (−0.067 each); at n=30 the Wilson intervals overlap base heavily, so r4 is
   statistically **≈ base on every domain with no collapse anywhere** — the cleanest "no-harm"
   profile of the series, achieved uniformly rather than by the lucky cancellation that gave r2 a
   similar macro while still degenerating.
4. **Still no improvement → external baseline stands.** macro −0.059 < 0, so selection is
   `NO_ADAPTER_IMPROVES_ON_CANARY` for the 10th arm / 4th round. Base reachability is already high
   where offline data is strong (logic/math/reasoning 0.83–0.87, little headroom); the weak spots
   (coding 0.50, alignment 0.54) are exactly where offline data is hardest (reasoned code at
   scale; no alignment verifier). r4 ranked 2nd of 10 (selected with A_format_control and r2 for
   the ≤3 Part-L slate), but selection promotes only positive-macro arms, and none exist.

## Policy

`KEEP_EXTERNAL_DUALANCHOR_CORECONTENT_BASELINE` — unchanged across 10 arms / 4 rounds. The
offline-bounded series ends here with a **refined** conclusion: the rounds 1–3 "training degrades
the generator" verdict was largely a **format-mismatch + bad-coding-data confound**; with the
confound removed, offline SFT+DPO is **reachability-neutral and non-degenerate**, but does not add
reachability on canary tasks where base is already strong. The converged **M (teacher
distillation) + N (verifier-reward RL)** run inherits a validated, eval-aligned, non-degenerate
substrate (`train_v5` + the `_fmt` patch), and its real levers are the low-base domains: reasoned
coding at scale and verifier-reward RL — not more SFT on already-saturated logic/math/reasoning.

Constraints honored: external verifiers the only correctness labels; no Ouro base/tokenizer/
checkpoint overwrite (adapter only under the v2 model root); no steering; no git; DualAnchor =
policy teacher and CoreContent_v2 = soft ranker, neither a correctness label.
