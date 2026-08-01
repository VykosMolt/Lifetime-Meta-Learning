"""Templates, sealed-import integrity, transfer tooling, bytecode rejection."""
from __future__ import annotations

import json
import os
import shutil

from _h import Runner, fresh_dir

from o1_b200.runner import sealed_import
from o1_b200.runner.env_report import (
    EnvReportError, load_template as env_template, unresolved_fields as env_unresolved,
    validate_b200_report,
)
from o1_b200.runner.precommit_template import (
    PrecommitTemplateError, finalize, load_template, resolve, unresolved_fields,
)
from o1_b200.runner.transfer_package import (
    TransferError, build_manifest, deterministic_zip, verify_sha256sums,
    write_sha256sums,
)

_FULL_FACTS = {
    "provider": "X", "instance_id": "i-1", "gpu_uuid": "GPU-1",
    "driver_runtime_report_sha256": "0" * 64,
    "container_image_digest_as_deployed": "sha256:" + "0" * 64,
    "selected_eligible_backend": "B200_REPLICA",
    "selected_worker_batch_configuration": "B200_REPLICA_w4_b1",
    "measured_benchmark_throughput_rows_per_hour": 4000.0,
    "actual_hourly_rate_usd": 2.99,
    "computed_hard_runtime_seconds": 48160,
    "environment_digest_sha256": "1" * 64,
    "final_backend_benchmark_report_sha256": "2" * 64,
}


def _resolved_env():
    rep = env_template()
    rep.update({
        "gpu_name": "NVIDIA B200", "gpu_uuid": "GPU-XYZ", "gpu_count": 1,
        "hbm_capacity_bytes": 192 * 2**30, "nvidia_driver": "580.00",
        "cuda_runtime": "12.8", "cuda_toolkit": "12.8", "cudnn": "9.8",
        "nccl": "2.27", "pytorch_version": "2.12.0.dev20260407+cu128",
        "transformers_version": "4.54.1", "numpy_version": "2.4.4",
        "python_version": "3.14.6", "linux_kernel": "6.8.0",
        "container_image_digest": "sha256:" + "a" * 64,
        "attention_backend": "eager",
        "deterministic_settings": {
            "torch_use_deterministic_algorithms": True,
            "cudnn_deterministic": True, "cudnn_benchmark": False},
        "cublas_workspace_config": ":4096:8",
        "tf32_state": {"matmul_allow_tf32": False, "cudnn_allow_tf32": False},
        "bf16_support": True, "compile_state": "OFF",
        "cuda_graph_state": "OFF"})
    return rep


def run() -> Runner:
    r = Runner("templates_integrity")

    r.expect_raises(
        "precommit template refuses finalization while unresolved",
        PrecommitTemplateError, lambda: finalize(load_template()))

    def partial_resolution_still_refused():
        doc = resolve(load_template(), {"provider": "X"}, mock=True)
        try:
            finalize(doc)
        except PrecommitTemplateError as exc:
            assert "instance_id" in str(exc)
            return
        raise AssertionError("partially resolved template finalized")
    r.check("partially resolved precommit template still refuses",
            partial_resolution_still_refused)

    def full_mock_resolution_finalizes_but_stays_mock():
        doc = resolve(load_template(), dict(_FULL_FACTS), mock=True)
        env = finalize(doc)
        assert env["finalized"] and env["mock"] is True
        assert "MOCK" in env["document"]["status"]
    r.check("fully resolved MOCK precommit finalizes but remains marked mock",
            full_mock_resolution_finalizes_but_stays_mock)

    r.expect_raises("unknown hardware fact refused", PrecommitTemplateError,
                    lambda: resolve(load_template(), {"gpu_color": "green"},
                                    mock=True))

    def template_scientific_hashes_frozen():
        t = load_template()
        f = t["frozen_scientific_identity"]
        assert f["sealed_calibration_precommit_sha256"] == \
            "e819aeebc642fefde769f398b385f23f98dc3980412eeb67274d0f5891c5457d"
        assert f["package_zip_sha256"] == \
            sealed_import.BASE_PACKAGE_ZIP_SHA256
        assert f["source_commit"] == sealed_import.SOURCE_COMMIT
        assert f["transformers_version"] == "4.54.1"
        assert len(unresolved_fields(t)) == 12
    r.check("precommit template freezes all scientific hashes; exactly the "
            "12 hardware facts are unresolved", template_scientific_hashes_frozen)

    def env_template_unresolved():
        missing = env_unresolved(env_template())
        assert len(missing) >= 20
    r.check("environment template: every hardware field is unresolved",
            env_template_unresolved)

    def env_validation_rules():
        ok = validate_b200_report(_resolved_env())
        assert ok["valid"]
        for field, value in (("transformers_version", "4.55.0"),
                             ("gpu_count", 2),
                             ("attention_backend", "sdpa"),
                             ("cublas_workspace_config", ":16:8"),
                             ("compile_state", "ON")):
            bad = _resolved_env()
            bad[field] = value
            try:
                validate_b200_report(bad)
            except EnvReportError:
                continue
            raise AssertionError(f"bad {field}={value!r} accepted")
    r.check("HOSTILE stale/altered environment values are refused "
            "(transformers, GPU count, attention, CUBLAS, compile)",
            env_validation_rules)

    def sealed_modules_verify_and_tamper_detected():
        sealed_import.verify_sealed_modules()
        d = fresh_dir("sealed_tamper")
        for name in sealed_import.SEALED_MODULE_SHA256:
            shutil.copy2(os.path.join(sealed_import.SEALED_DIR, name),
                         os.path.join(d, name))
        with open(os.path.join(d, "o1_answer_parser_v2.py"), "a") as fh:
            fh.write("\n# tampered\n")
        try:
            sealed_import.verify_sealed_modules(d)
        except sealed_import.SealedImportError:
            return
        raise AssertionError("tampered sealed module accepted")
    r.check("HOSTILE tampered sealed module is refused at import time",
            sealed_modules_verify_and_tamper_detected)

    def bytecode_rejected():
        d = fresh_dir("sealed_bytecode")
        for name in sealed_import.SEALED_MODULE_SHA256:
            shutil.copy2(os.path.join(sealed_import.SEALED_DIR, name),
                         os.path.join(d, name))
        os.makedirs(os.path.join(d, "__pycache__"))
        with open(os.path.join(d, "__pycache__", "x.pyc"), "wb") as fh:
            fh.write(b"\x00")
        try:
            sealed_import.verify_sealed_modules(d)
        except sealed_import.SealedImportError:
            return
        raise AssertionError("stale bytecode accepted")
    r.check("HOSTILE stale bytecode alongside sealed sources is refused",
            bytecode_rejected)

    def sha256sums_roundtrip_and_tamper():
        d = fresh_dir("sums")
        files = []
        for i in range(3):
            p = os.path.join(d, f"f{i}.txt")
            with open(p, "w") as fh:
                fh.write(f"content {i}\n")
            files.append(p)
        sums = os.path.join(d, "SHA256SUMS")
        write_sha256sums(files, d, sums)
        assert verify_sha256sums(sums, d)["checked"] == 3
        with open(files[1], "a") as fh:
            fh.write("tamper")
        try:
            verify_sha256sums(sums, d)
        except TransferError:
            return
        raise AssertionError("tampered file passed SHA256SUMS")
    r.check("SHA256SUMS write/verify roundtrip; tampering detected",
            sha256sums_roundtrip_and_tamper)

    def deterministic_zip_stable():
        d = fresh_dir("zips")
        files = []
        for i in range(3):
            p = os.path.join(d, f"f{i}.txt")
            with open(p, "w") as fh:
                fh.write(f"data {i}\n")
            files.append(p)
        h1 = deterministic_zip(files, d, os.path.join(d, "a.zip"))
        h2 = deterministic_zip(list(reversed(files)), d, os.path.join(d, "b.zip"))
        assert h1 == h2, "archive is not deterministic"
    r.check("deterministic archive: byte-identical regardless of input order",
            deterministic_zip_stable)

    def secrets_refused():
        d = fresh_dir("secrets")
        p = os.path.join(d, "id_rsa")
        with open(p, "w") as fh:
            fh.write("PRIVATE")
        try:
            build_manifest({"bad": {"path": p, "kind": "file"}},
                           os.path.join(d, "m.json"))
        except TransferError:
            return
        raise AssertionError("secret-looking file packaged")
    r.check("HOSTILE secret material is refused by the transfer tooling",
            secrets_refused)

    def manifest_hash_mismatch_refused():
        d = fresh_dir("manifest")
        p = os.path.join(d, "artifact.bin")
        with open(p, "wb") as fh:
            fh.write(b"payload")
        try:
            build_manifest({"a": {"path": p, "kind": "file",
                                  "expected_sha256": "0" * 64}},
                           os.path.join(d, "m.json"))
        except TransferError:
            return
        raise AssertionError("hash-mismatched artifact accepted")
    r.check("HOSTILE corrupted artifact (hash mismatch) refused by manifest "
            "builder", manifest_hash_mismatch_refused)

    return r


if __name__ == "__main__":
    raise SystemExit(run().report())
