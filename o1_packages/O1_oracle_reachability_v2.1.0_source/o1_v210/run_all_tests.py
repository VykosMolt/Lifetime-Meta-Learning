"""Run every shipped O1 v2.1 verification suite and static package check."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

sys.dont_write_bytecode = True
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

MODULES = [
    "o1_analysis.py",
    "calibration_analysis.py",
    "calibration_precommit.py",
    "build_run_provenance.py",
    "build_seed_matrix.py",
    "pregeneration_seal.py",
    "run_o1_primary.py",
    "axis_reconstruction.py",
    "verify_axis_artifact.py",
    "horizon_logic_generator_v2.py",
    "build_o1_v2_cohorts.py",
    "o1_prompt_template_v2.py",
    "o1_answer_parser_v2.py",
    "o1_truth_table_verifier_v2.py",
    "run_o1_v2_generation.py",
    "o1_transport_v2.py",
    "run_o1_v2_orchestrator.py",
    "cohort_allocation.py",
    "build_confirmatory_cohort.py",
    "verify_record_binding.py",
]

SUITES = [
    ("confirmatory fixtures", [sys.executable, "test_o1_fixtures.py"], "53/53 passed"),
    ("calibration fixtures", [sys.executable, "test_calibration_fixtures.py"], "24/24 passed"),
    ("integration fixtures", [sys.executable, "test_v20_integrity.py"], "15/15 passed"),
    ("CLI workflow fixtures", [sys.executable, "test_cli_workflow.py"], "7/7 passed"),
    ("parser adversarial fixtures", [sys.executable, "test_parser_fixtures.py"], "58/58 passed"),
    ("orchestrator fixtures", [sys.executable, "test_orchestrator_fixtures.py"], "15/15 passed"),
    ("cohort builder fixtures", [sys.executable, "test_cohort_builder_fixtures.py"], "17/17 passed"),
    ("axis verifier self-test", [sys.executable, "verify_axis_artifact.py", "--selftest"], "15/15 axis-artifact checks behaved as specified"),
]

CLI_MODULES = [
    "run_o1_v2_orchestrator.py",
    "verify_record_binding.py",
    "build_confirmatory_cohort.py",
    "calibration_precommit.py",
    "calibration_analysis.py",
    "build_seed_matrix.py",
    "build_run_provenance.py",
    "pregeneration_seal.py",
    "run_o1_primary.py",
    "axis_reconstruction.py",
    "build_o1_v2_cohorts.py",
    "run_o1_v2_generation.py",
]


def run(cmd: list[str]) -> str:
    # Use a real temporary file rather than a PIPE. One fixture deliberately
    # launches a flattened child suite; a file avoids inherited-pipe lifetime
    # surprises while still retaining the complete diagnostic log. Bytecode
    # generation is disabled in every child so verification never mutates the
    # package or creates an unhashed executable shadow.
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with tempfile.TemporaryFile(mode="w+t") as fh:
        cp = subprocess.run(cmd, cwd=HERE, text=True, stdout=fh,
                            stderr=subprocess.STDOUT, env=env)
        fh.seek(0)
        text = fh.read()
    if cp.returncode != 0:
        print(text)
        raise SystemExit(f"FAILED ({cp.returncode}): {' '.join(cmd)}")
    return text


def reject_bytecode_artifacts(root: Path = HERE) -> None:
    """Reject every packaged Python bytecode artifact, including empty caches.

    A ``.pyc`` can be made timestamp/size-valid and shadow reviewed source. It
    is therefore forbidden rather than ignored or listed in ``SHA256SUMS``.
    """
    bad: list[str] = []
    for path in root.rglob("*"):
        rel = path.relative_to(root).as_posix()
        if path.is_dir() and path.name == "__pycache__":
            bad.append(rel + "/")
        elif path.is_file() and path.suffix.lower() in {".pyc", ".pyo"}:
            bad.append(rel)
    if bad:
        raise RuntimeError(
            "forbidden executable bytecode artifacts present: "
            + ", ".join(sorted(bad)[:20])
        )


def verify_package_checksums(root: Path = HERE) -> int:
    """Verify SHA256SUMS values and complete package-file coverage.

    ``SHA256SUMS`` deliberately does not list itself. Every other regular file
    must be listed exactly once. Python bytecode and ``__pycache__`` directories
    are forbidden outright and are checked before coverage.
    """
    reject_bytecode_artifacts(root)
    sums_path = root / "SHA256SUMS"
    if not sums_path.is_file():
        raise RuntimeError("SHA256SUMS is missing")

    declared: dict[str, str] = {}
    for lineno, raw in enumerate(sums_path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise RuntimeError(f"SHA256SUMS line {lineno} is malformed")
        digest, name = parts
        name = name.lstrip("* ")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
            raise RuntimeError(f"SHA256SUMS line {lineno} has invalid digest")
        if name in declared:
            raise RuntimeError(f"SHA256SUMS lists {name!r} more than once")
        declared[name] = digest.lower()

    actual_files: set[str] = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel == "SHA256SUMS":
            continue
        actual_files.add(rel)

    missing = sorted(actual_files - set(declared))
    extra = sorted(set(declared) - actual_files)
    if missing or extra:
        raise RuntimeError(
            "SHA256SUMS coverage mismatch: "
            f"unlisted={missing[:10]} extra={extra[:10]}"
        )

    for name, expected in sorted(declared.items()):
        path = root / name
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        got = h.hexdigest()
        if got != expected:
            raise RuntimeError(
                f"SHA256SUMS mismatch for {name}: expected {expected}, got {got}"
            )
    return len(declared)


def main() -> int:
    print("O1 v2.1 full verification", flush=True)
    print("=" * 72, flush=True)

    n_hashed = verify_package_checksums(HERE)
    print(f"PASS  SHA256SUMS integrity and coverage: {n_hashed} files", flush=True)

    for name in MODULES:
        source = (HERE / name).read_text(encoding="utf-8")
        compile(source, str(HERE / name), "exec")
    print(f"PASS  Python syntax compilation: {len(MODULES)} modules", flush=True)

    for name in (
        "FREEZE_MANIFEST.template.json",
        "CALIBRATION_METADATA.template.json",
        "RUNTIME_ARTIFACT_PATHS.template.json",
        "AXIS_SOURCE_BUNDLE.template.json",
    ):
        with open(HERE / name) as fh:
            json.load(fh)
    manifest = json.load(open(HERE / "FREEZE_MANIFEST.template.json"))
    assert manifest["prereg_version"] == "2.1"
    assert list(manifest["action_space"]["structured_axes"]["axes"]) == [
        "A1_READABLE", "A2_WRITABLE", "A3_CAUSAL_MEAN", "A4_SEQUENCE_MEAN"]
    assert manifest["action_space"]["structured_axes"]["axes"]["A3_CAUSAL_MEAN"][
        "pc_extraction"] is False
    assert manifest["action_space"]["structured_axes"]["axes"]["A4_SEQUENCE_MEAN"][
        "pc_extraction"] is False
    assert manifest["action_space"]["alpha_selection_rule"] == "largest_passing"
    assert manifest["cohorts"]["seed_derivation_scheme"] == "sha256_domain_uint64_be_v1"
    assert manifest["budget"]["min_completed_task_fraction"] == 1.0
    boundary = manifest["action_space"]["writable_boundary"]
    assert boundary["physical_layer_one_based"] == 24
    assert boundary["module_index_zero_based"] == 23
    assert boundary["first_affected_cache_slot"] == [2, 24]
    print("PASS  JSON templates and fixed policy fields", flush=True)

    for name in CLI_MODULES:
        run([sys.executable, name, "--help"])
    print(f"PASS  CLI parsers: {len(CLI_MODULES)} commands", flush=True)

    for label, cmd, marker in SUITES:
        out = run(cmd)
        if marker not in out:
            print(out)
            raise SystemExit(f"FAILED: {label} did not emit expected marker {marker!r}")
        print(f"PASS  {label}: {marker}", flush=True)

    from o1_analysis import required_G

    cases = ((0.25, 0.08, 324), (0.25, 0.05, 817),
             (0.25, 0.03, 2238), (0.10, 0.05, 337))
    got = []
    for q, delta, want in cases:
        value = required_G(q, delta, G_hi=3000)
        if value != want:
            raise SystemExit(f"FAILED exact power q={q} delta={delta}: {value} != {want}")
        got.append(value)
    print("PASS  Exact McNemar planning minima: " + " / ".join(map(str, got)), flush=True)

    forbidden = []
    for name in ("README.md", "PREREGISTRATION.md", "FREEZE_MANIFEST.template.json"):
        p = HERE / name
        text = p.read_text(errors="replace")
        if ("Preregistration v1.4" in text or "50/50 passed" in text or
                "52/52" in text or "v1.5.2" in text or
                "d4_components_l3_24.npy" in text or
                "d2_empirical_outcome" in text or
                "17/17 calibration" in text or
                '"prereg_version": "2.0"' in text):
            forbidden.append(name)
    if forbidden:
        raise SystemExit("FAILED stale version/count text in: " + ", ".join(forbidden))
    print("PASS  No stale package labels or obsolete test counts", flush=True)

    reject_bytecode_artifacts(HERE)
    print("PASS  No packaged or test-generated Python bytecode", flush=True)

    print("=" * 72)
    print("ALL O1 v2.1 CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
