"""Batched engine: mixed lengths, stopping, batch invariance, compaction,
batch-slot identity attacks."""
from __future__ import annotations

import numpy as np
import torch

from _h import CORPUS_DIR, Runner

from o1_b200.runner.engine_batched import generate_batch
from o1_b200.runner.runbuild import build_validation_bundle
from o1_b200.runner.synthetic_runtime import SynthCache, SyntheticOuroModel

ARTIFACT = {"kind": "synthetic", "device": "cpu", "seed_tag": 0}


def _tasks():
    bundle = build_validation_bundle(CORPUS_DIR, ARTIFACT)
    return bundle, {t["task_id"]: t for t in bundle.tasks_by_id.values()}


def run() -> Runner:
    r = Runner("engine_batched")
    model = SyntheticOuroModel(device="cpu")
    bundle, tasks = _tasks()
    tid = sorted(tasks)
    # variable classes and prompt lengths, deliberately shuffled
    mix_ids = [tid[0], tid[3], tid[4], tid[5], tid[8], tid[10], tid[7], tid[9]]
    mix = [tasks[t] for t in mix_ids]
    seeds = [7000 + i for i in range(len(mix))]

    serial = [generate_batch(model, model.tokenizer, [t], [s], [None], [0.0])[0]
              for t, s in zip(mix, seeds)]

    def batch_invariance():
        for B in (2, 4, 8):
            got = []
            for i in range(0, len(mix), B):
                got.extend(generate_batch(
                    model, model.tokenizer, mix[i:i + B], seeds[i:i + B],
                    [None] * len(mix[i:i + B]), [0.0] * len(mix[i:i + B])))
            for a, b in zip(serial, got):
                assert a["generated_token_ids"] == b["generated_token_ids"]
                assert a["finish_reason"] == b["finish_reason"]
                assert a["generated_text"] == b["generated_text"]
                assert np.array_equal(a["_prefill_l4_47"], b["_prefill_l4_47"])
    r.check("token/stop/text/boundary identity at batch sizes 1,2,4,8 "
            "with mixed prompt lengths and completion lengths",
            batch_invariance)

    def shuffled_scheduling():
        order = [5, 2, 7, 0, 3, 6, 1, 4]
        got = generate_batch(model, model.tokenizer,
                             [mix[i] for i in order], [seeds[i] for i in order],
                             [None] * 8, [0.0] * 8)
        for slot, i in enumerate(order):
            assert got[slot]["generated_token_ids"] == \
                serial[i]["generated_token_ids"], (slot, i)
    r.check("shuffled scheduling order cannot change any row's trajectory",
            shuffled_scheduling)

    def stopping_reasons_covered():
        reasons = {res["finish_reason"] for res in serial}
        assert "parseable_final_answer" in reasons
        assert "eos_without_commitment" in reasons
        assert "max_new_tokens" in reasons
    r.check("EOS, parser-stop, and max-token termination all exercised",
            stopping_reasons_covered)

    def repetition_flagged():
        rep_task = tasks[[t for t in tid if "repeat_max" in t][0]]
        res = generate_batch(model, model.tokenizer, [rep_task], [1], [None],
                             [0.0])[0]
        assert res["catastrophic_repetition"] is True
        assert res["finish_reason"] == "max_new_tokens"
    r.check("catastrophic repetition is produced and mechanically flagged",
            repetition_flagged)

    def prefix_attack_never_commits():
        atk = tasks[[t for t in tid if "prefix_attack" in t][0]]
        res = generate_batch(model, model.tokenizer, [atk], [1], [None],
                             [0.0])[0]
        assert res["well_formed"] is False
        assert "FINAL ANSWER:" in res["generated_text"]
        assert res["finish_reason"] == "eos_without_commitment"
    r.check("HOSTILE prefix-parser attack: 'FINAL ANSWER: About...' does not "
            "stop or parse", prefix_attack_never_commits)

    def compaction_matches_noncompaction():
        a = generate_batch(model, model.tokenizer, mix, seeds, [None] * 8,
                           [0.0] * 8, compaction=False)
        b = generate_batch(model, model.tokenizer, mix, seeds, [None] * 8,
                           [0.0] * 8, compaction=True)
        for x, y in zip(a, b):
            assert x["generated_token_ids"] == y["generated_token_ids"]
            assert x["finish_reason"] == y["finish_reason"]
            assert np.array_equal(x["_prefill_l4_47"], y["_prefill_l4_47"])
    r.check("optional compaction (default OFF) reproduces non-compacting "
            "results bitwise: identity moves with the row",
            compaction_matches_noncompaction)

    def slot_swap_is_detectable():
        # a deliberately sabotaged cache (two rows' states swapped) must
        # change trajectories — proving slot identity is load-bearing and the
        # serial-vs-batched comparison would catch a swap defect
        t2 = [tasks[t] for t in (tid[8], tid[9])]  # two stochastic tasks
        s2 = [4242, 4243]
        enc = [model.tokenizer(__import__("o1_prompt_template_v2").render_prompt(t),
                               return_tensors="pt")["input_ids"][0]
               for t in t2]
        from o1_b200.runner.engine_batched import _left_pad
        ids, mask = _left_pad([[int(x) for x in e] for e in enc], 2, "cpu")
        with torch.inference_mode():
            out = model(input_ids=ids, attention_mask=mask, use_cache=True,
                        logits_to_keep=1, return_per_loop_hidden_states=True)
        cache = out.past_key_values
        swapped = SynthCache([s.flip(0).clone() for s in cache.states],
                             list(reversed(cache.rows)))
        logits = out.logits[:, -1, :]
        from o1_b200.runner.row_rng import sample_token
        tok_honest = [sample_token(logits[b], s2[b], 0) for b in range(2)]
        m1 = torch.cat([mask, torch.ones(2, 1, dtype=mask.dtype)], dim=1)
        nxt = torch.tensor([[tok_honest[0]], [tok_honest[1]]])
        with torch.inference_mode():
            honest = model(input_ids=nxt, attention_mask=m1,
                           past_key_values=cache, use_cache=True,
                           logits_to_keep=1)
            forged = model(input_ids=nxt, attention_mask=m1,
                           past_key_values=swapped, use_cache=True,
                           logits_to_keep=1)
        assert not torch.equal(honest.logits, forged.logits), \
            "slot swap was undetectable — cache carries no row identity"
    r.check("HOSTILE batch-slot identity swap changes outputs (detectable)",
            slot_swap_is_detectable)

    def stale_cache_after_bad_compaction_detectable():
        t2 = [tasks[t] for t in (tid[8], tid[9])]
        honest = generate_batch(model, model.tokenizer, t2, [11, 12],
                                [None, None], [0.0, 0.0], compaction=True)
        # sabotage: monkeypatch selection to keep the WRONG slot
        import o1_b200.runner.engine_batched as eb
        orig = eb._select_cache
        def bad_select(cache, keep):  # keeps slot 0 regardless
            return orig(cache, [0 for _ in keep])
        eb._select_cache = bad_select
        try:
            forged = generate_batch(model, model.tokenizer, t2, [11, 12],
                                    [None, None], [0.0, 0.0], compaction=True)
        finally:
            eb._select_cache = orig
        same = all(h["generated_token_ids"] == f["generated_token_ids"]
                   for h, f in zip(honest, forged))
        assert not same or all(
            h["finish_reason"] == "eos_without_commitment" for h in honest), \
            "stale KV cache after compaction was undetectable"
    r.check("HOSTILE stale KV cache after compaction diverges from honest run",
            stale_cache_after_bad_compaction_detectable)

    return r


if __name__ == "__main__":
    raise SystemExit(run().report())
