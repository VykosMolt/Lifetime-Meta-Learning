"""Foundation Learner V0 — campaign layer (contract §10–§13, §15, §18–§19).

The campaign layer owns everything that decides *what runs when* on the
accelerator, and everything that keeps the sealed test set and the O1
calibration programme out of the Foundation Learner's reach:

* ``o1_isolation``      — the realpath guard through which EVERY campaign file
                          access passes (contract §13/§22).
* ``affordability``     — the frozen §11 arithmetic (safety factor 1.25, final
                          transfer reserve 1200 s, FULL-vs-PEFT rule, U ladder).
* ``FL_BUDGET_POLICY.json`` — the frozen policy those constants live in.
* ``scheduler``         — monotonic-clock stage admission, checkpoint cadence,
                          journalling, and the BENCH-first priority order.
* ``stage_definitions`` — the declarative FL0..FL8 + BENCH stage table with
                          lazily resolved work functions and predeclared
                          fallback work.
* ``promotion``         — the §10 promotion rules over DEVELOPMENT metrics that
                          structurally cannot carry a sealed record.
* ``dev_selector``      — the frozen 2-LR grid on FL3 at 25 % U and the
                          ``DEV_DECISIONS_FROZEN.json`` writer.
* ``sealed_gate``       — the ONLY campaign decipher path for sealed shards,
                          with the two-phase, single-use, append-only opening
                          ledger (Amendment 12).
* ``core_matching``     — the production audit of the core comparison: compute
                          matching, realized ledgers, and one shared frozen base.
* ``result_verifier``   — run-output hashing and the deterministic transfer
                          archive.
* ``session_supervisor``— the O1-first session state machine.
* ``entry``             — the PRODUCTION campaign entry: §22 determinism, the
                          real ``StageContext`` factory, and the ladder runner
                          that ``session_supervisor.main()`` wires on.

Nothing in this package spends money, contacts a provider, rents an
accelerator, or opens sealed data outside ``sealed_gate``.
"""
from __future__ import annotations

CAMPAIGN_VERSION = "0.2.0"

__all__ = ["CAMPAIGN_VERSION"]
