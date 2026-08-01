"""Common sealed backend contract for the O1 B200 execution stack.

One explicit interface; three implementations (REFERENCE_SERIAL, B200_REPLICA,
B200_BATCHED).  Backends consume immutable RowSpecs and emit the one canonical
record schema (records.py).  No backend-specific field may alter scientific
analysis; backend metadata is descriptive only.
"""
from __future__ import annotations

import abc
import dataclasses
import platform
import sys
from typing import Any

import numpy as np

from .identity import domain_sha256
from .row_specs import RowSpec


BACKEND_IDS = ("REFERENCE_SERIAL", "B200_REPLICA", "B200_BATCHED")


@dataclasses.dataclass(frozen=True)
class RuntimeConfig:
    backend_id: str
    run_dir: str
    model_artifact: dict          # {"kind": "synthetic"|"ouro_rltt", ...}
    worker_count: int = 1
    batch_size: int = 1
    compaction: bool = False      # optional optimization, default OFF
    device: str = "cpu"
    # optional performance modes — ALL default OFF and scientifically
    # ineligible until they pass future B200 equivalence testing:
    torch_compile: bool = False
    cuda_graphs: bool = False
    attention_override: str | None = None

    def config_sha256(self) -> str:
        return domain_sha256("o1b200.backend_config.v1", dataclasses.asdict(self))

    def validate(self) -> None:
        if self.backend_id not in BACKEND_IDS:
            raise ValueError(f"unknown backend {self.backend_id!r}")
        if self.backend_id != "B200_BATCHED" and self.batch_size != 1:
            raise ValueError("batch_size > 1 requires B200_BATCHED")
        if self.backend_id != "B200_REPLICA" and self.worker_count != 1:
            raise ValueError("worker_count > 1 requires B200_REPLICA")
        if self.compaction and self.backend_id != "B200_BATCHED":
            raise ValueError("compaction requires B200_BATCHED")
        if self.torch_compile or self.cuda_graphs or self.attention_override:
            raise ValueError(
                "optional performance modes are locked OFF until a future "
                "B200 equivalence pass explicitly authorizes them")


@dataclasses.dataclass(frozen=True)
class RunBundle:
    """Everything a backend needs, sealed before execution."""
    tasks_by_id: dict
    specs: list                    # list[RowSpec], canonical order
    artifact_ids: dict             # package/source/axis/model/tokenizer ids
    axes: dict                     # {"structured": np.ndarray|None, "random": ...}
    row_spec_manifest_sha256: str
    binding_key: str
    binding_sha256: str
    task_manifest_sha256: str
    seed_matrix_sha256: str
    seal_sha256: str

    def resume_identity(self, config: RuntimeConfig,
                        environment_sha256: str) -> dict:
        return {
            "source_commit": self.artifact_ids["source_commit"],
            "package_zip_sha256": self.artifact_ids["package_zip_sha256"],
            "model_artifact_sha256": self.artifact_ids["model_artifact_sha256"],
            "tokenizer_sha256": self.artifact_ids["tokenizer_sha256"],
            "axis_package_sha256": self.artifact_ids["axis_package_sha256"],
            "backend_id": config.backend_id,
            "backend_config_sha256": config.config_sha256(),
            "batch_size": config.batch_size,
            "worker_count": config.worker_count,
            "runtime_environment_sha256": environment_sha256,
            "binding_key": self.binding_key,
            "binding_sha256": self.binding_sha256,
            "seal_sha256": self.seal_sha256,
            "task_manifest_sha256": self.task_manifest_sha256,
            "seed_matrix_sha256": self.seed_matrix_sha256,
            "row_spec_manifest_sha256": self.row_spec_manifest_sha256,
        }


def runtime_environment_sha256(extra: dict | None = None) -> str:
    import torch
    payload = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "deterministic_algorithms": True,
    }
    try:
        import transformers
        payload["transformers"] = transformers.__version__
    except ImportError:
        payload["transformers"] = None
    if extra:
        payload.update(extra)
    return domain_sha256("o1b200.runtime_env.v1", payload)


class Backend(abc.ABC):
    """The sealed backend contract."""

    def __init__(self):
        self._initialized = False

    @abc.abstractmethod
    def initialize(self, runtime_config: RuntimeConfig, bundle: RunBundle) -> None:
        """Bind configuration; open the row store; verify resume identity."""

    @abc.abstractmethod
    def load_model(self, model_artifact: dict) -> None:
        """Load (or connect) the model instance(s)."""

    @abc.abstractmethod
    def validate_artifacts(self) -> dict:
        """Hash-verify every consumed artifact; return the verification map."""

    @abc.abstractmethod
    def execute_rows(self, row_specs: list) -> dict:
        """Execute the given sealed rows (skipping verified-complete ones)."""

    @abc.abstractmethod
    def checkpoint(self) -> dict:
        """Report durable progress (rows committed are already atomic)."""

    @abc.abstractmethod
    def resume(self) -> dict:
        """Verify identity + committed rows; return remaining row specs."""

    @abc.abstractmethod
    def finalize_records(self) -> dict:
        """Canonical deterministic merge; refuses on missing rows."""

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Release resources; safe to call multiple times."""
