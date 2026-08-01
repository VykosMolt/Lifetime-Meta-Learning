#!/usr/bin/env python3
"""Environment validation script (deployment side).

Asserts the pinned software stack (transformers==4.54.1 exactly, torch BF16,
deterministic settings capable), collects a runtime report, and REFUSES to
declare a B200 environment: GPU-identity fields are recorded verbatim from
the machine it runs on and validated against the template requirements by
runner/env_report.py in the future session.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys


def collect(strict_versions: bool = True) -> dict:
    report = {"schema": "o1b200.deploy_env_check.v1"}
    import numpy
    report["python_version"] = sys.version.split()[0]
    report["numpy_version"] = numpy.__version__
    report["linux_kernel"] = platform.release()
    try:
        import transformers
        report["transformers_version"] = transformers.__version__
    except ImportError:
        report["transformers_version"] = None
    if report["transformers_version"] != "4.54.1":
        raise SystemExit(
            f"FATAL: transformers must equal 4.54.1, got "
            f"{report['transformers_version']!r}")
    import torch
    report["pytorch_version"] = torch.__version__
    report["cuda_available"] = torch.cuda.is_available()
    report["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if report["cublas_workspace_config"] != ":4096:8":
        raise SystemExit("FATAL: CUBLAS_WORKSPACE_CONFIG must be ':4096:8'")
    if torch.cuda.is_available():
        report["gpu_count"] = torch.cuda.device_count()
        report["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        report["hbm_capacity_bytes"] = props.total_memory
        report["bf16_support"] = torch.cuda.is_bf16_supported()
        report["cuda_runtime"] = torch.version.cuda
        if not report["bf16_support"]:
            raise SystemExit("FATAL: BF16 is required")
    report["tf32_state"] = {
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    report["deterministic_settings"] = {
        "torch_use_deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }
    report["is_b200_environment_claim"] = False
    report["note"] = ("This report validates the pinned stack on the machine "
                      "it ran on. It is NOT a B200 hardware validation.")
    return report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    report = collect()
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({k: report[k] for k in
                      ("transformers_version", "pytorch_version",
                       "cuda_available")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
