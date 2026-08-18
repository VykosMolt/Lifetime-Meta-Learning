"""Environment-report template handling and validation.

The template (policies/ENVIRONMENT_REPORT.template.json) contains ONLY
unresolved fields for facts that require the actual B200 session.  This
module validates a filled report: completeness, hard requirements
(transformers==4.54.1, single GPU, eager attention, deterministic flags,
CUBLAS workspace, BF16, compile/graphs off).  It can also collect a LOCAL
report for the local machine — clearly labeled, never a B200 report.
"""
from __future__ import annotations

import json
import os
import platform
import sys

from .identity import domain_sha256

TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                             "policies", "ENVIRONMENT_REPORT.template.json")

REQUIRED_FIELDS = (
    "gpu_name", "gpu_uuid", "gpu_count", "hbm_capacity_bytes",
    "nvidia_driver", "cuda_runtime", "cuda_toolkit", "cudnn", "nccl",
    "pytorch_version", "transformers_version", "numpy_version",
    "python_version", "linux_kernel", "container_image_digest",
    "attention_backend", "deterministic_settings", "cublas_workspace_config",
    "tf32_state", "bf16_support", "compile_state", "cuda_graph_state",
)


class EnvReportError(RuntimeError):
    pass


def load_template() -> dict:
    with open(os.path.abspath(TEMPLATE_PATH), encoding="utf-8") as fh:
        return json.load(fh)


def unresolved_fields(report: dict) -> list[str]:
    out = []
    for key in REQUIRED_FIELDS:
        v = report.get(key)
        if v is None:
            out.append(key)
        elif isinstance(v, str) and v.startswith("UNRESOLVED"):
            out.append(key)
        elif isinstance(v, dict):
            for sub, sv in v.items():
                if isinstance(sv, str) and sv.startswith("UNRESOLVED"):
                    out.append(f"{key}.{sub}")
    return out


def validate_b200_report(report: dict) -> dict:
    """Validate a fully-resolved future B200 environment report."""
    missing = unresolved_fields(report)
    if missing:
        raise EnvReportError(
            f"environment report has unresolved fields: {missing}")
    if report["transformers_version"] != "4.54.1":
        raise EnvReportError(
            f"transformers must equal 4.54.1, got "
            f"{report['transformers_version']!r}")
    if int(report["gpu_count"]) != 1:
        raise EnvReportError("exactly one GPU is required")
    if report["attention_backend"] != "eager":
        raise EnvReportError(
            "attention backend must be 'eager' (sealed generator) unless a "
            "separately committed validation authorizes otherwise")
    det = report["deterministic_settings"]
    if not (det.get("torch_use_deterministic_algorithms") is True
            and det.get("cudnn_deterministic") is True
            and det.get("cudnn_benchmark") is False):
        raise EnvReportError("deterministic settings are not as sealed")
    if report["cublas_workspace_config"] != ":4096:8":
        raise EnvReportError("CUBLAS_WORKSPACE_CONFIG must be ':4096:8'")
    if report["bf16_support"] is not True:
        raise EnvReportError("BF16 support is required")
    for key in ("compile_state", "cuda_graph_state"):
        if report[key] not in ("OFF", "off", False):
            raise EnvReportError(f"{key} must be OFF (unvalidated optimization)")
    return {"valid": True,
            "environment_digest_sha256": domain_sha256(
                "o1b200.environment_report.v1", report)}


def collect_pod_report(container_image_digest: str = "UNKNOWN") -> dict:
    """Fully-resolved environment report on the ACCELERATOR pod.

    Sets and records the sealed deterministic runtime configuration, then
    resolves every template field with live values; validate_b200_report
    enforces the sealed expectations (transformers 4.54.1, eager attention,
    deterministic flags, CUBLAS workspace, BF16, compile/cuda-graph OFF).
    """
    import platform as _platform
    import sys as _sys

    import numpy
    import torch
    import transformers

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    props = torch.cuda.get_device_properties(0)
    return {
        "schema": "o1b200.environment_report.v1",
        "status": "RESOLVED_ON_POD",
        "gpu_name": props.name,
        "gpu_uuid": str(props.uuid),
        "gpu_count": torch.cuda.device_count(),
        "hbm_capacity_bytes": int(props.total_memory),
        "nvidia_driver": _driver_version(),
        "cuda_runtime": torch.version.cuda,
        "cuda_toolkit": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nccl": ".".join(str(x) for x in torch.cuda.nccl.version()),
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "numpy_version": numpy.__version__,
        "python_version": _sys.version.split()[0],
        "linux_kernel": _platform.release(),
        "container_image_digest": container_image_digest,
        "attention_backend": "eager",
        "deterministic_settings": {
            "torch_use_deterministic_algorithms": True,
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        },
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG", ""),
        "tf32_state": {
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        },
        "bf16_support": bool(torch.cuda.is_bf16_supported()),
        "compile_state": "OFF",
        "cuda_graph_state": "OFF",
    }


def _driver_version() -> str:
    try:
        import ctypes
        lib = ctypes.CDLL("libnvidia-ml.so.1")
        lib.nvmlInit_v2()
        buf = ctypes.create_string_buffer(80)
        lib.nvmlSystemGetDriverVersion(buf, 80)
        out = buf.value.decode()
        lib.nvmlShutdown()
        return out
    except Exception:  # noqa: BLE001
        return "UNKNOWN"


def collect_local_report() -> dict:
    """LOCAL machine report — labeled, never a B200 report."""
    import numpy
    import torch
    report = {
        "label": "LOCAL_MACHINE_REPORT_NOT_B200",
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "numpy_version": numpy.__version__,
        "linux_kernel": platform.release(),
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import transformers
        report["transformers_version"] = transformers.__version__
    except ImportError:
        report["transformers_version"] = None
    if torch.cuda.is_available():
        report["gpu_name"] = torch.cuda.get_device_name(0)
        report["gpu_count"] = torch.cuda.device_count()
    return report
