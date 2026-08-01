"""Sealed O1 v2 transport-rho computation.

v2.0 exposed the L4_47 boundary state from the generator but computed rho in
unsealed operator code.  v2.1 freezes the computation here; the orchestrator is
its only caller and the analysis gates consume the recorded value.

Frozen measurement semantics (identical to the sealed manifest transport
section — this module is the executable form of that prose):

  * intervention locus: zero-based loop 2 / paper L3; physical layer 24 =
    ``model.model.layers[23]`` (zero-based module index 23); post-layer
    residual output; final non-padding prompt position; one-shot during
    prefill.
  * downstream measurement state: ``OuroModel`` ``per_loop_hidden_states[3]``
    — the loop-3 (zero-based) = paper-L4 boundary AFTER the final decoder
    module (zero-based module index 47) and after ``model.norm`` — at the
    final non-padding prompt position, captured from the prefill forward pass
    BEFORE any stochastic decoding.  The prose label "L4_47" uses the
    zero-based final-module index 47; the intervention label "L3_24" uses the
    one-based physical layer 24.  Both conventions are stated here once so
    neither is silently reinterpreted.
  * baseline pairing: ``h_base`` is the same-task baseline prefill boundary
    state.  Prefill is sampling-free, so ``h_base`` is stream-independent; the
    orchestrator asserts bitwise equality across baseline streams.
  * numerator: RMS over the 2048 hidden coordinates of ``h_int - h_base``.
  * denominator: the intended injected-delta RMS recorded by the generation
    hook (``injected_rms``).  NOTE: the hook computes the intended
    ``|s*alpha*r*d|`` in float32 before the bfloat16 cast; the realized in-graph
    delta deviates elementwise by bf16 rounding (O(0.3%)).  The intended value
    is the frozen denominator.
  * measurement dtype: float64 from the float32 captured states.
  * requirements: rho must be finite and non-negative; a NaN poisons every
    downstream summary and is rejected here, not in analysis.
  * zero-alpha convention: rho is exactly 0.0, and this module EARNS it by
    asserting the captured intervention boundary equals the baseline boundary
    bitwise before returning 0.0.  A zero-alpha row with a nonzero boundary
    difference is a pipeline defect and raises.
  * baseline rows carry a null transport field and never call this module.
  * transport remains DIAGNOSTIC ONLY: no endpoint conditions on it and no
    axis is removed or reweighted because of it (manifest prohibitions).

The dead-transport threshold (0.0001) is an interpretation constant frozen in
the manifest, not a numerical tolerance used here.
"""
from __future__ import annotations

import numpy as np

TRANSPORT_LOOP_ZERO_BASED = 3      # per_loop_hidden_states index (paper L4)
TRANSPORT_FINAL_MODULE_ZERO_BASED = 47
HIDDEN = 2048


class TransportError(RuntimeError):
    """A transport measurement violated its frozen contract."""


def _as_state(vec, name: str) -> np.ndarray:
    arr = np.asarray(vec)
    if arr.shape != (HIDDEN,):
        raise TransportError(f"{name} must have shape ({HIDDEN},), got {arr.shape}")
    arr = arr.astype(np.float64)
    if not np.isfinite(arr).all():
        raise TransportError(f"{name} contains non-finite values")
    return arr


def rms_2048(vec, name: str = "state") -> float:
    """RMS over the 2048 hidden coordinates, float64."""
    arr = _as_state(vec, name)
    return float(np.sqrt(np.mean(arr * arr)))


def downstream_delta_rms(
    h_int, h_base, injected_rms: float, *, is_zero_alpha: bool = False,
) -> float:
    """rho = RMS(h_int - h_base) / injected_rms at the frozen L4_47 boundary."""
    if is_zero_alpha:
        a = np.asarray(h_int)
        b = np.asarray(h_base)
        if a.shape != b.shape or a.shape != (HIDDEN,):
            raise TransportError(
                f"zero-alpha boundary shapes {a.shape} vs {b.shape} != ({HIDDEN},)")
        if float(injected_rms) != 0.0:
            raise TransportError(
                f"zero-alpha row carries injected_rms={injected_rms!r}, must be 0.0")
        if not np.array_equal(a, b):
            raise TransportError(
                "zero-alpha boundary state differs bitwise from baseline; the "
                "documented identity path is broken — halt for pipeline repair")
        return 0.0
    inj = float(injected_rms)
    if not np.isfinite(inj) or inj <= 0.0:
        raise TransportError(
            f"injected_rms must be finite and > 0 on intervention rows, got {inj!r}")
    ai = _as_state(h_int, "h_int")
    ab = _as_state(h_base, "h_base")
    diff = ai - ab
    rho = float(np.sqrt(np.mean(diff * diff)) / inj)
    if not np.isfinite(rho) or rho < 0.0:
        raise TransportError(f"transport rho {rho!r} is not finite and non-negative")
    return rho
