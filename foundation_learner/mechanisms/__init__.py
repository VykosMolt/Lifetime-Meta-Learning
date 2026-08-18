"""Foundation Learner V0 — FL4-FL8 mechanisms (contract §7, §9, §23).

* ``value_head``     FL4 future-competence value head (targets, training,
                     DEV rank/calibration/regret helpers).
* ``fast_state``     FL5 persistent fast state ``s`` + its eval-side callbacks.
* ``fl5_training``   FL5 segmented-episode training under the FL3 objective.
* ``value_gating``   FL6 gate: frozen FL4 head, incorporate iff ``v > 0``.
* ``fast_adapter``   FL7 bounded resettable LoRA fast adapter + callbacks.
* ``consolidation``  FL8 merge of selected fast deltas into a slow bank.
* ``hidden_states``  the shared final-layer, FINAL-LOOP hidden-state tap.

Invariants these modules enforce structurally (all hostile-tested):

1. The FL4 head input path renders events ``0..j`` only and REFUSES a context
   containing a QUERY / TRANSFER block or attempt.
2. FL4 target computation refuses non-TRAIN families and refuses any ablation
   plan that would remove a non-feedback event.
3. FL5 state resets to exact zero at episode start and cannot cross episodes
   unless a caller explicitly asks for chain carry.
4. The FL6 gate consumes the head prediction and nothing else.
5. FL7 resets its delta per episode, bounds it at Frobenius 1.0, and never
   writes a base weight.
6. FL8 refuses SEALED_TEST material anywhere.

Nothing here uses the global RNG or the wall clock in a scientific path.
"""

from __future__ import annotations

VERSION = "0.1.0"

__all__ = [
    "VERSION",
    "value_head",
    "fast_state",
    "fl5_training",
    "value_gating",
    "fast_adapter",
    "consolidation",
    "hidden_states",
]
