"""Hash-verified import bridge to the sealed O1 v2.1.0 package.

The B200 runner never re-implements sealed scientific behavior.  Every sealed
module it uses is imported from the verified v2.1.0 source directory, and the
byte content of each file is SHA-256 checked against the pinned values below
(taken from the package's own verified SHA256SUMS) BEFORE the module becomes
importable.  A modified sealed file fails closed.
"""
from __future__ import annotations

import hashlib
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# o1_b200/runner/ -> worktree root
WORKTREE_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
SEALED_DIR = os.path.join(
    WORKTREE_ROOT, "o1_packages", "O1_oracle_reachability_v2.1.0_source", "o1_v210")

# Pinned from the verified package SHA256SUMS (zip sha256
# a8b0571cc67584b862db3e42fb015f47ae7d2887d7ce59605e3c273591fe7892).
SEALED_MODULE_SHA256 = {
    "o1_answer_parser_v2.py":
        "9c8bac6cdfdae49138e144c64171ccdbabedbc59bf760e19290c69f549e25bfb",
    "o1_prompt_template_v2.py":
        "18132c41a0fd2d118f5f1913822ec60fdb8abad657072e77a347381d68cd0a90",
    "o1_transport_v2.py":
        "7939194b85029b76ccc6873e0ec851bc5c856c2f4fa8dd50587cfce9900f8ef8",
    "o1_truth_table_verifier_v2.py":
        "d97497c64dd182e9b87b24dc47dcd9ec28b9d98d2ecd8ddf7bb48b2297cd2e68",
    "run_o1_v2_generation.py":
        "878dd4c465b884bf11ce10c107bd59ed44f290fa8583ee81a787b91e9f2d0097",
    "run_o1_v2_orchestrator.py":
        "1bef2b9e75aa4a07a1a6d5fd976bddf7e1423f3e2906c5eda082623bf29d04ae",
    "o1_analysis.py":
        "feb5334cdcbe47d334837e829d31286f2de53c01dc0023b682326efe698612f2",
    "calibration_analysis.py":
        "d507f719e6fe1ed485a1a7b006d8e1c9dd2106fccd8f8556121d09fde240af44",
    "build_run_provenance.py":
        "2052da51334b101945aa9b18907db7a8ff434107c3082458b4620372f177221c",
    "verify_axis_artifact.py":
        "ce6e79c6c39847144955b0ff4a725ddb4006d5ffb893644eb7b34861441107ac",
    "horizon_logic_generator_v2.py":
        "7fbfbdf79309f79855e7631e4b9923a6eae607741a1aed0cd96904b6651dfb4d",
}

BASE_PACKAGE = "O1_oracle_reachability_v2.1.0"
BASE_PACKAGE_ZIP_SHA256 = (
    "a8b0571cc67584b862db3e42fb015f47ae7d2887d7ce59605e3c273591fe7892")
SOURCE_COMMIT = "6db215fa52ee315ad1011e14cf7a12560107da56"


class SealedImportError(RuntimeError):
    """A sealed module failed byte verification; nothing was imported."""


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_VERIFIED = False


def verify_sealed_modules(sealed_dir: str = SEALED_DIR) -> dict:
    """Byte-verify every pinned sealed module.  Returns {name: sha256}."""
    out = {}
    for name, want in sorted(SEALED_MODULE_SHA256.items()):
        path = os.path.join(sealed_dir, name)
        if not os.path.isfile(path):
            raise SealedImportError(f"sealed module missing: {path}")
        got = _sha256_file(path)
        if got != want:
            raise SealedImportError(
                f"sealed module {name} hashes {got[:16]}... but the verified "
                f"package pins {want[:16]}...; refusing to import")
        out[name] = got
    # Refuse stale bytecode shadowing the verified sources.
    pycache = os.path.join(sealed_dir, "__pycache__")
    if os.path.isdir(pycache) and os.listdir(pycache):
        raise SealedImportError(
            f"stale bytecode present in {pycache}; delete it before running "
            f"(sealed sources must be the only executable form)")
    return out


def ensure_sealed_path() -> str:
    """Verify the sealed package then place it on sys.path (idempotent)."""
    global _VERIFIED
    if not _VERIFIED:
        verify_sealed_modules()
        sys.dont_write_bytecode = True
        _VERIFIED = True
    if SEALED_DIR not in sys.path:
        sys.path.insert(0, SEALED_DIR)
    return SEALED_DIR
