"""B200 calibration-precommit template resolution.

The template freezes everything scientific by hash and leaves UNRESOLVED
exactly the facts that require the future B200 session.  ``finalize``
REFUSES while any required field is unresolved, and the result of a mock
resolution is stamped so it can never be mistaken for (or used as) a real
precommit.
"""
from __future__ import annotations

import json
import os

from .identity import domain_sha256

TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                             "policies",
                             "CALIBRATION_PRECOMMIT_B200.template.json")


class PrecommitTemplateError(RuntimeError):
    pass


def load_template() -> dict:
    with open(os.path.abspath(TEMPLATE_PATH), encoding="utf-8") as fh:
        return json.load(fh)


def unresolved_fields(doc: dict) -> list[str]:
    return sorted(k for k, v in doc.get("unresolved_hardware_facts", {}).items()
                  if isinstance(v, str) and v.startswith("UNRESOLVED"))


def resolve(doc: dict, hardware_facts: dict, *, mock: bool) -> dict:
    out = json.loads(json.dumps(doc))  # deep copy
    facts = out.get("unresolved_hardware_facts", {})
    unknown = sorted(set(hardware_facts) - set(facts))
    if unknown:
        raise PrecommitTemplateError(
            f"unknown hardware facts supplied: {unknown}")
    for k, v in hardware_facts.items():
        facts[k] = v
    if mock:
        out["status"] = ("MOCK_DRESS_REHEARSAL_ONLY — populated by the local "
                         "mocked state machine; NOT a calibration precommit; "
                         "any attempt to run calibration against it must be "
                         "refused")
        out["mock"] = True
    return out


def finalize(doc: dict) -> dict:
    """Produce the finalization envelope; refuses unresolved fields.

    Even a successfully finalized MOCK document remains unusable for real
    calibration: the mock flag survives finalization and the calibration
    entry gate refuses mock precommits.
    """
    left = unresolved_fields(doc)
    if left:
        raise PrecommitTemplateError(
            f"finalization refused: unresolved fields remain: {left}")
    body = json.loads(json.dumps(doc))
    envelope = {
        "finalized": True,
        "mock": bool(body.get("mock", False)),
        "document": body,
        "document_sha256": domain_sha256("o1b200.b200_precommit.v1", body),
        "next_required_step": (
            "external commit + independent remote verification BEFORE any "
            "calibration (runbook steps 11-13)"),
    }
    return envelope
