"""Tensor-level intervention and transport tests (batched semantics)."""
from __future__ import annotations

import numpy as np
import torch

from _h import CORPUS_DIR, Runner

from o1_b200.runner.engine_batched import generate_batch
from o1_b200.runner.intervention import (
    BatchedInterventionError, BatchedOneShotResidualEdit,
)
from o1_b200.runner.runbuild import build_validation_bundle, load_frozen_axes
from o1_b200.runner.sealed_import import ensure_sealed_path
from o1_b200.runner.synthetic_runtime import SyntheticOuroModel

ensure_sealed_path()
from o1_transport_v2 import TransportError, downstream_delta_rms  # noqa: E402

ARTIFACT = {"kind": "synthetic", "device": "cpu", "seed_tag": 0}


def _mixed_batch(model, axes, alphas_signs):
    """Run a mixed batch over corpus tasks; returns per-row results."""
    bundle = build_validation_bundle(CORPUS_DIR, ARTIFACT)
    tasks_all = list(bundle.tasks_by_id.values())
    # mixed prompt lengths: short + long tasks interleaved
    tasks = [tasks_all[0], tasks_all[3], tasks_all[1], tasks_all[7],
             tasks_all[2], tasks_all[9]][: len(alphas_signs)]
    seeds = [1000 + i for i in range(len(tasks))]
    directions, signed = [], []
    for kind, d, s, a in alphas_signs:
        if kind == "baseline":
            directions.append(None)
            signed.append(0.0)
        else:
            directions.append(np.asarray(axes["structured"][d - 1],
                                         dtype=np.float64))
            signed.append(s * a)
    return tasks, seeds, directions, signed, generate_batch(
        model, model.tokenizer, tasks, seeds, directions, signed)


def run() -> Runner:
    r = Runner("intervention_transport")
    model = SyntheticOuroModel(device="cpu")
    axes = load_frozen_axes()

    plan = [("baseline", None, None, 0.0),
            ("int", 1, 1, 0.04),
            ("int", 2, -1, 0.08),
            ("baseline", None, None, 0.0),
            ("int", 3, 1, 0.005),
            ("int", 4, -1, 0.02)]
    tasks, seeds, directions, signed, results = _mixed_batch(model, axes, plan)

    def per_row_injected_rms():
        for (kind, d, s, a), res in zip(plan, results):
            if kind == "baseline":
                assert res["injected_rms"] == 0.0
                assert res["realized_delta_rms"] is None
            else:
                assert res["injected_rms"] > 0.0
                assert res["realized_delta_rms"] is not None
                # intended vs realized are close but stored separately
                assert abs(res["realized_delta_rms"] - res["injected_rms"]) \
                    <= 0.02 * res["injected_rms"] + 1e-12
    r.check("mixed batch: per-row intended and realized delta RMS recorded",
            per_row_injected_rms)

    def baseline_rows_untouched_in_mixed_batch():
        # rows with direction None inside an intervention batch must equal a
        # pure-baseline run bitwise (no delta on another row's position)
        solo = generate_batch(model, model.tokenizer, [tasks[0]], [seeds[0]],
                              [None], [0.0])
        assert np.array_equal(solo[0]["_prefill_l4_47"],
                              results[0]["_prefill_l4_47"])
        assert solo[0]["generated_token_ids"] == results[0]["generated_token_ids"]
    r.check("HOSTILE cross-row bleed: baseline row inside an intervention "
            "batch is bitwise identical to a solo baseline run",
            baseline_rows_untouched_in_mixed_batch)

    def intervention_moves_own_boundary_only():
        base0 = generate_batch(model, model.tokenizer, [tasks[1]], [seeds[1]],
                               [None], [0.0])[0]["_prefill_l4_47"]
        assert not np.array_equal(base0, results[1]["_prefill_l4_47"]), \
            "intervention row boundary did not move"
    r.check("intervention changes the intervened row's boundary",
            intervention_moves_own_boundary_only)

    def zero_alpha_identity():
        za = generate_batch(model, model.tokenizer, [tasks[0]], [seeds[0]],
                            [None], [0.0])
        base = generate_batch(model, model.tokenizer, [tasks[0]], [seeds[0]],
                              [None], [0.0])
        rho = downstream_delta_rms(za[0]["_prefill_l4_47"],
                                   base[0]["_prefill_l4_47"], 0.0,
                                   is_zero_alpha=True)
        assert rho == 0.0
    r.check("zero-alpha rows keep the bitwise identity path (rho exactly 0)",
            zero_alpha_identity)

    def right_padding_attack_refused():
        mask = torch.ones((2, 5), dtype=torch.long)
        mask[1, -1] = 0  # right-padded row: final position is padding
        try:
            BatchedOneShotResidualEdit(
                [np.asarray(axes["structured"][0]), None], [0.04, 0.0], mask)
        except BatchedInterventionError:
            return
        raise AssertionError("right-padding (wrong final position) accepted")
    r.check("HOSTILE wrong final prompt position (right padding) is refused",
            right_padding_attack_refused)

    def one_shot_only():
        # the hook must fire exactly once (prefill, loop 2) and never during
        # decode: generate_batch already asserts applied == 1; additionally
        # prove no hook remains registered on the model afterwards
        layer = model.model.layers[23]
        assert not layer._forward_hooks, "hook left registered after batch"
    r.check("HOSTILE repeated decode-time injection impossible "
            "(one-shot hook, removed after prefill)", one_shot_only)

    def mismatched_direction_alpha_refused():
        mask = torch.ones((1, 4), dtype=torch.long)
        try:
            BatchedOneShotResidualEdit([None], [0.5], mask)
        except BatchedInterventionError:
            return
        raise AssertionError("null direction with nonzero alpha accepted")
    r.check("null direction with nonzero alpha is refused",
            mismatched_direction_alpha_refused)

    def analytic_transport():
        h_base = np.zeros(2048)
        delta = np.full(2048, 3.0)
        rho = downstream_delta_rms(h_base + delta, h_base, 1.5)
        assert abs(rho - 2.0) < 1e-12, rho
    r.check("transport rho matches the analytic value on a known delta",
            analytic_transport)

    def transport_rejects_nan_and_negative():
        h = np.zeros(2048)
        try:
            downstream_delta_rms(h + np.nan, h, 1.0)
        except TransportError:
            pass
        else:
            raise AssertionError("NaN boundary accepted")
        try:
            downstream_delta_rms(h + 1.0, h, 0.0)
        except TransportError:
            return
        raise AssertionError("zero injected_rms accepted on intervention row")
    r.check("transport enforces finite/nonnegative requirements",
            transport_rejects_nan_and_negative)

    def transport_not_mixed_between_rows():
        # per-row rho from the mixed batch equals the solo-run rho bitwise
        for i, (kind, d, s, a) in enumerate(plan):
            if kind == "baseline":
                continue
            solo = generate_batch(
                model, model.tokenizer, [tasks[i]], [seeds[i]],
                [np.asarray(axes["structured"][d - 1], dtype=np.float64)],
                [s * a])[0]
            base = generate_batch(model, model.tokenizer, [tasks[i]],
                                  [seeds[i]], [None], [0.0])[0]
            rho_solo = downstream_delta_rms(
                solo["_prefill_l4_47"], base["_prefill_l4_47"],
                solo["injected_rms"])
            rho_mixed = downstream_delta_rms(
                results[i]["_prefill_l4_47"], base["_prefill_l4_47"],
                results[i]["injected_rms"])
            assert rho_solo == rho_mixed, (i, rho_solo, rho_mixed)
    r.check("HOSTILE transport mixing: per-row rho identical solo vs mixed batch",
            transport_not_mixed_between_rows)

    return r


if __name__ == "__main__":
    raise SystemExit(run().report())
