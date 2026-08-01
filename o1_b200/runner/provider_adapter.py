"""Provider adapter boundary.

The provider has NOT been chosen; no cloud API is guessed here.  This module
defines the strict interface every future adapter must implement, plus:

  * MockProviderAdapter — fully scriptable, used by all local tests and the
    dress rehearsal; can inject failure at every operation;
  * LocalProviderAdapter — dress rehearsal only: the "instance" is the local
    machine, the billing rate is 0.0, uploads are file copies, launch runs a
    callable.  It exists to exercise the zero-touch state machine end to end.

A PRODUCTION provider adapter is a DELIBERATE FUTURE HARDWARE-ACCESS
DEPENDENCY: it may only be implemented once a provider is selected and its
current API contract is in hand, and it must be tested before any instance is
started (runbook step 2).  Nothing in this package performs a real provider
API call, and instantiating the production placeholder raises.
"""
from __future__ import annotations

import abc
import os
import shutil
from typing import Any, Callable


class ProviderError(RuntimeError):
    pass


class ProviderAdapter(abc.ABC):
    """Strict future-provider contract (all methods required)."""

    @abc.abstractmethod
    def quote_instance(self) -> dict:
        """Return {'instance_type', 'gpu', 'hourly_rate_usd', 'region'}."""

    @abc.abstractmethod
    def validate_single_gpu(self) -> dict:
        """Confirm the quote is exactly one B200; raise otherwise."""

    @abc.abstractmethod
    def start_instance(self) -> str:
        """Provision and return the instance reference."""

    @abc.abstractmethod
    def get_instance_status(self, instance_ref: str) -> str:
        """'pending' | 'running' | 'terminated' | 'error'."""

    @abc.abstractmethod
    def get_billing_rate(self, instance_ref: str) -> float:
        """The actually-billed hourly USD rate."""

    @abc.abstractmethod
    def upload_artifacts(self, instance_ref: str, manifest: dict) -> dict:
        """Transfer artifacts; return per-artifact verification hashes."""

    @abc.abstractmethod
    def launch_job(self, instance_ref: str, job: dict) -> dict:
        """Start the remote job; return {'job_id', ...}."""

    @abc.abstractmethod
    def download_results(self, instance_ref: str, dest_dir: str) -> dict:
        """Fetch outputs; return {'files': [...]}."""

    @abc.abstractmethod
    def terminate_instance(self, instance_ref: str) -> None:
        """Request termination (idempotent)."""

    @abc.abstractmethod
    def confirm_terminated(self, instance_ref: str) -> bool:
        """Independently confirm the instance is terminated (billing off)."""


class ProductionProviderAdapter(ProviderAdapter):
    """Placeholder that refuses to exist until a provider is selected."""

    def __init__(self, *_a, **_kw):
        raise ProviderError(
            "No production provider adapter exists: the provider has not "
            "been selected. Implementing one is a declared future "
            "hardware-access dependency (see B200_ACCESS_RUNBOOK.md step 2).")

    quote_instance = validate_single_gpu = start_instance = None  # type: ignore
    get_instance_status = get_billing_rate = upload_artifacts = None  # type: ignore
    launch_job = download_results = terminate_instance = None  # type: ignore
    confirm_terminated = None  # type: ignore


class MockProviderAdapter(ProviderAdapter):
    """Scriptable mock: tests inject failures per operation."""

    def __init__(self, *, hourly_rate_usd: float = 2.99,
                 gpu: str = "MOCK-B200", gpu_count: int = 1,
                 fail_on: dict[str, Exception] | None = None):
        self.hourly_rate_usd = float(hourly_rate_usd)
        self.gpu = gpu
        self.gpu_count = gpu_count
        self.fail_on = dict(fail_on or {})
        self.calls: list[str] = []
        self.instance = None
        self.terminated = False
        self.jobs: list[dict] = []
        self.uploaded: dict | None = None

    def _op(self, name: str):
        self.calls.append(name)
        if name in self.fail_on:
            raise self.fail_on.pop(name)

    def quote_instance(self) -> dict:
        self._op("quote_instance")
        return {"instance_type": "mock.b200.1x", "gpu": self.gpu,
                "gpu_count": self.gpu_count,
                "hourly_rate_usd": self.hourly_rate_usd, "region": "mock-1"}

    def validate_single_gpu(self) -> dict:
        self._op("validate_single_gpu")
        if self.gpu_count != 1:
            raise ProviderError(
                f"quote is for {self.gpu_count} GPUs; exactly 1 is required")
        return {"gpu": self.gpu, "gpu_count": 1}

    def start_instance(self) -> str:
        self._op("start_instance")
        self.instance = "mock-instance-0001"
        self.terminated = False
        return self.instance

    def get_instance_status(self, instance_ref: str) -> str:
        self._op("get_instance_status")
        if self.instance != instance_ref:
            return "error"
        return "terminated" if self.terminated else "running"

    def get_billing_rate(self, instance_ref: str) -> float:
        self._op("get_billing_rate")
        return self.hourly_rate_usd

    def upload_artifacts(self, instance_ref: str, manifest: dict) -> dict:
        self._op("upload_artifacts")
        self.uploaded = dict(manifest)
        return {"verified": True, "artifacts": sorted(manifest)}

    def launch_job(self, instance_ref: str, job: dict) -> dict:
        self._op("launch_job")
        self.jobs.append(dict(job))
        return {"job_id": f"job-{len(self.jobs):04d}"}

    def download_results(self, instance_ref: str, dest_dir: str) -> dict:
        self._op("download_results")
        os.makedirs(dest_dir, exist_ok=True)
        return {"files": []}

    def terminate_instance(self, instance_ref: str) -> None:
        self._op("terminate_instance")
        self.terminated = True

    def confirm_terminated(self, instance_ref: str) -> bool:
        self._op("confirm_terminated")
        return self.terminated


class LocalProviderAdapter(ProviderAdapter):
    """Dress rehearsal only: 'instance' is this machine; rate is 0.0 USD.

    launch_job runs a supplied callable synchronously.  Exists so the
    zero-touch state machine can be exercised end to end with the synthetic
    runtime and the non-O1 corpus, with zero spend and zero network use.
    """

    def __init__(self, work_dir: str,
                 job_runner: Callable[[dict], dict] | None = None):
        self.work_dir = work_dir
        self.job_runner = job_runner
        self.instance = None
        self.terminated = False
        self.results: dict = {}

    def quote_instance(self) -> dict:
        return {"instance_type": "local.dress-rehearsal", "gpu": "LOCAL",
                "gpu_count": 1, "hourly_rate_usd": 0.0, "region": "local"}

    def validate_single_gpu(self) -> dict:
        return {"gpu": "LOCAL", "gpu_count": 1}

    def start_instance(self) -> str:
        os.makedirs(self.work_dir, exist_ok=True)
        self.instance = "local-instance"
        self.terminated = False
        return self.instance

    def get_instance_status(self, instance_ref: str) -> str:
        return "terminated" if self.terminated else "running"

    def get_billing_rate(self, instance_ref: str) -> float:
        return 0.0

    def upload_artifacts(self, instance_ref: str, manifest: dict) -> dict:
        out = {}
        for name, src in manifest.items():
            dest = os.path.join(self.work_dir, os.path.basename(str(src)))
            if os.path.isdir(src):
                if not os.path.exists(dest):
                    shutil.copytree(src, dest)
            else:
                shutil.copy2(src, dest)
            out[name] = dest
        return {"verified": True, "artifacts": out}

    def launch_job(self, instance_ref: str, job: dict) -> dict:
        if self.job_runner is None:
            raise ProviderError("LocalProviderAdapter has no job runner bound")
        self.results = self.job_runner(job)
        return {"job_id": "local-job", "result": self.results}

    def download_results(self, instance_ref: str, dest_dir: str) -> dict:
        os.makedirs(dest_dir, exist_ok=True)
        return {"files": [], "result": self.results}

    def terminate_instance(self, instance_ref: str) -> None:
        self.terminated = True

    def confirm_terminated(self, instance_ref: str) -> bool:
        return self.terminated
