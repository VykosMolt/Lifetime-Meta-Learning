"""Row-specific randomness: the auditable mapping from sealed identity to
sampling stream.

FROZEN SCIENTIFIC FACTS (from the sealed v2.1 generator — the semantic
authority; nothing here may change them):

  * The stream seed is the sealed CRN derivation
    ``derive_stream_seed(master_seed, task_id, stream_index)``
    (sha256_domain_uint64_be_v1).  Common random numbers are BY DESIGN: two
    rows of the same task and stream share the same seed across arms, so the
    arm contrast is paired.  Arm / bank / direction / sign therefore enter the
    row's randomness through the sealed action-to-stream Latin square
    (stream = (action_index + rank) mod 8), NOT as extra entropy.
  * At generation step ``t`` the sealed sampler constructs a FRESH
    ``torch.Generator`` seeded ``int(seed) + t`` and draws exactly one
    multinomial sample from it (``run_o1_v2_generation._sample_top_p``).

Consequences, which the tests in ``tests/test_row_rng.py`` prove mechanically:

  * The sampling stream of a row is a pure function of
    (master_seed, task_id, stream_index, step) — nothing else.
  * No global CUDA RNG is consumed per sample: batch size, worker count,
    scheduling order, neighbor completion, restart, and resume cannot remap a
    row's stream, because no cross-row RNG state exists to be remapped.
  * Given bitwise-identical logits, the sampled token at (row, step) is
    identical under serial, replica, and batched execution.

This module exposes the mapping (delegating to sealed code), plus an audit
record that binds the full sealed row identity to its derived seeds so the
mapping can be re-verified offline.

Cross-GPU token identity is NOT claimed here: logits equality across hardware
is a future B200 equivalence question.  Locally we prove exact seed and
sampling-stream ASSIGNMENT identity, and (on the synthetic runtime) exact
stream identity.
"""
from __future__ import annotations

from . import sealed_import
from .identity import derive_stream_seed, domain_sha256
from .row_specs import RowSpec

sealed_import.ensure_sealed_path()


def step_generator_seed(row_seed: int, step: int) -> int:
    """The exact per-step generator seed used by the sealed sampler."""
    if not (0 <= int(step)):
        raise ValueError("step must be non-negative")
    return int(row_seed) + int(step)


def sample_token(logits, row_seed: int, step: int) -> int:
    """Draw the (row, step) token via the SEALED sampler, unmodified."""
    from run_o1_v2_generation import _sample_top_p  # sealed, hash-verified
    return _sample_top_p(logits, step_generator_seed(row_seed, step))


def verify_spec_seed(spec: RowSpec, master_seed: int) -> None:
    """Recompute the sealed CRN seed for a RowSpec and require equality."""
    want = derive_stream_seed(int(master_seed), spec.task_id, spec.stream_index)
    if int(spec.seed) != int(want):
        raise ValueError(
            f"row {spec.row_id[:16]}: seed {spec.seed} disagrees with the "
            f"sealed CRN derivation {want}")


def rng_audit_record(spec: RowSpec, n_steps: int = 8) -> dict:
    """Auditable binding of the sealed row identity to its sampling seeds.

    Binds task ID, bank, arm, direction, sign, stream, seed, and generation
    step (the full sealed identity) to the derived per-step generator seeds.
    A pure function of the RowSpec: recomputable by any auditor.
    """
    steps = {str(t): step_generator_seed(spec.seed, t) for t in range(n_steps)}
    payload = {
        "identity": spec.identity_dict(),
        "row_id": spec.row_id,
        "scheme": "sealed sha256_domain_uint64_be_v1 stream seed; "
                  "per-step generator seed = seed + step (sealed sampler)",
        "step_generator_seeds": steps,
    }
    payload["audit_sha256"] = domain_sha256("o1b200.rng_audit.v1", {
        "identity": payload["identity"], "steps": steps})
    return payload
