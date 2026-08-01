"""The three sealed backends: REFERENCE_SERIAL, B200_REPLICA, B200_BATCHED.

REFERENCE_SERIAL wraps the sealed one-row-at-a-time generator unchanged; it is
the executable semantic reference, the local regression oracle, the future
B200 equivalence reference, and the fallback backend.

B200_REPLICA is pure replica parallelism: N independent worker processes, one
model instance each, one scientific row at a time per worker, unchanged
single-row semantics, deterministic task-affine row allocation, atomic record
output, deterministic canonical merge, safe worker restart.

B200_BATCHED executes multiple independent rows per forward/decode batch via
engine_batched (left padding, per-row hook, per-row sealed sampling, per-row
stopping/transport/records).  Non-compacting active-mask execution is the
correctness baseline; compaction is optional and OFF by default.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from typing import Any, Callable

import numpy as np

from . import sealed_import
from .backend_interface import Backend, RunBundle, RuntimeConfig, runtime_environment_sha256
from .engine_batched import generate_batch
from .identity import domain_sha256, sha256_file
from .persistence import PersistenceError, RowStore
from .records import RecordError, build_record, utc_now, verify_record
from .row_specs import RowSpec

sealed_import.ensure_sealed_path()

from o1_transport_v2 import downstream_delta_rms  # noqa: E402 (sealed)
from run_o1_v2_generation import generate_branch  # noqa: E402 (sealed)
from o1_prompt_template_v2 import render_prompt   # noqa: E402 (sealed)


class BackendError(RuntimeError):
    pass


# ==========================================================================
# shared helpers
# ==========================================================================


def load_model_artifact(model_artifact: dict):
    kind = model_artifact.get("kind")
    if kind == "synthetic":
        from .synthetic_runtime import load_synthetic
        return load_synthetic(model_artifact)
    if kind == "ouro_rltt":
        from run_o1_v2_generation import _load_model  # sealed
        import pathlib
        return _load_model(pathlib.Path(model_artifact["checkpoint"]))
    raise BackendError(f"unknown model artifact kind {kind!r}")


def model_artifact_sha256(model_artifact: dict) -> str:
    if model_artifact.get("kind") == "synthetic":
        return domain_sha256("o1b200.model_artifact.synthetic.v1", model_artifact)
    if model_artifact.get("kind") == "ouro_rltt":
        # precomputed tree hash must be supplied and is re-checked at
        # artifact-verification time on the target machine
        return model_artifact["checkpoint_tree_sha256"]
    raise BackendError("unknown model artifact kind")


def make_verifier(bundle: RunBundle) -> Callable[[dict], None]:
    specs_by_id = {s.row_id: s for s in bundle.specs}

    def _verify(record: dict) -> None:
        task = bundle.tasks_by_id.get(record.get("task_id"))
        if task is None:
            raise RecordError(
                f"task {record.get('task_id')!r} is not in the sealed manifest")
        verify_record(record, task=task, spec=specs_by_id.get(record.get("row_id")),
                      expected_artifact_ids=bundle.artifact_ids)
        if record.get("row_id") not in specs_by_id:
            raise RecordError("row_id is not in the sealed row-spec manifest")

    return _verify


def prompt_token_ids_for(tokenizer, task: dict) -> list[int]:
    enc = tokenizer(render_prompt(task), return_tensors="pt", truncation=True,
                    max_length=1536, padding=False)
    return [int(t) for t in enc["input_ids"][0]]


def rho_for(spec: RowSpec, gen: dict, h_base: np.ndarray) -> float | None:
    boundary = np.asarray(gen["_prefill_l4_47"])
    if boundary.ndim == 2 and boundary.shape[0] == 1:
        boundary = boundary[0]
    if spec.arm == "baseline":
        return None
    if spec.arm == "zero_alpha":
        return downstream_delta_rms(boundary, h_base, 0.0, is_zero_alpha=True)
    return downstream_delta_rms(boundary, h_base, float(gen["injected_rms"]))


def group_specs_by_task(specs: list[RowSpec]) -> list[tuple[str, list[RowSpec]]]:
    order: list[str] = []
    groups: dict[str, list[RowSpec]] = {}
    for s in sorted(specs, key=lambda x: x.exec_index):
        if s.task_id not in groups:
            groups[s.task_id] = []
            order.append(s.task_id)
        groups[s.task_id].append(s)
    return [(tid, groups[tid]) for tid in order]


def _boundary_of(gen: dict) -> np.ndarray:
    b = np.asarray(gen["_prefill_l4_47"])
    if b.ndim == 2 and b.shape[0] == 1:
        b = b[0]
    return b


def run_serial_core(specs: list[RowSpec], bundle: RunBundle, store: RowStore,
                    model, tokenizer, *, backend_id: str, worker_id: str,
                    environment_sha256: str,
                    generate_one: Callable | None = None) -> dict:
    """Sealed one-row-at-a-time execution over a spec subset.

    generate_one(task, seed, direction, signed_alpha) defaults to the SEALED
    generate_branch — its scientific behavior is not rewritten here.
    """
    if generate_one is None:
        def generate_one(task, seed, direction, signed_alpha):
            return generate_branch(model, tokenizer, task, seed,
                                   direction, signed_alpha)
    done = store.load_completed()
    n_generated = 0
    for task_id, task_specs in group_specs_by_task(specs):
        task = bundle.tasks_by_id[task_id]
        baselines = [s for s in task_specs if s.arm == "baseline"]
        others = [s for s in task_specs if s.arm != "baseline"]
        if all(s.row_id in done for s in task_specs):
            continue
        h_base = None
        for s in baselines:
            if s.row_id in done:
                continue
            gen = generate_one(task, s.seed, None, 0.0)
            boundary = _boundary_of(gen)
            if h_base is None:
                h_base = boundary
            elif not np.array_equal(boundary, h_base):
                raise BackendError(
                    f"task {task_id}: baseline prefill boundary differs "
                    f"across streams — prefill must be deterministic")
            _commit(store, bundle, s, gen, None, tokenizer, task,
                    backend_id=backend_id, worker_id=worker_id,
                    batch_id="serial", batch_size=1,
                    environment_sha256=environment_sha256)
            n_generated += 1
        if h_base is None and any(s.row_id not in done for s in others):
            # every baseline pre-exists (or lives outside this spec subset):
            # replay stream 0 and require exact token reproduction before
            # trusting the boundary (sealed policy)
            all_base = [s for s in bundle.specs
                        if s.task_id == task_id and s.arm == "baseline"]
            if not all_base:
                raise BackendError(
                    f"task {task_id}: no baseline rows exist in the sealed "
                    f"design; h_base cannot be established")
            s0 = min(all_base, key=lambda s: s.stream_index)
            gen = generate_one(task, s0.seed, None, 0.0)
            stored = done.get(s0.row_id)
            if stored is None or list(stored["generated_token_ids"]) != [
                    int(t) for t in gen["generated_token_ids"]]:
                raise BackendError(
                    f"task {task_id}: baseline replay does not reproduce the "
                    f"stored row; boundary cannot be re-established safely")
            h_base = _boundary_of(gen)
        for s in others:
            if s.row_id in done:
                continue
            direction = None
            if s.bank is not None:
                bank = bundle.axes[s.bank]
                direction = np.asarray(bank[s.direction_id - 1], dtype=np.float64)
            gen = generate_one(task, s.seed, direction, s.signed_alpha())
            rho = rho_for(s, gen, h_base)
            _commit(store, bundle, s, gen, rho, tokenizer, task,
                    backend_id=backend_id, worker_id=worker_id,
                    batch_id="serial", batch_size=1,
                    environment_sha256=environment_sha256)
            n_generated += 1
    return {"n_generated": n_generated}


def _commit(store: RowStore, bundle: RunBundle, spec: RowSpec, gen: dict,
            rho: float | None, tokenizer, task: dict, *, backend_id: str,
            worker_id: str, batch_id: str, batch_size: int,
            environment_sha256: str) -> None:
    started = gen.get("_started_utc", utc_now())
    prompt_ids = gen.get("_prompt_token_ids")
    if prompt_ids is None:
        prompt_ids = prompt_token_ids_for(tokenizer, task)
    record = build_record(
        spec, gen, rho=rho, prompt_token_ids=prompt_ids,
        backend_id=backend_id, worker_id=worker_id, batch_id=batch_id,
        batch_size=batch_size, artifact_ids=bundle.artifact_ids,
        runtime_environment_sha256=environment_sha256,
        row_spec_manifest_sha256=bundle.row_spec_manifest_sha256,
        started_utc=started, finished_utc=utc_now(),
        realized_delta_rms=gen.get("realized_delta_rms"))
    store.commit_row(record)


# ==========================================================================
# REFERENCE_SERIAL
# ==========================================================================


class ReferenceSerialBackend(Backend):
    backend_id = "REFERENCE_SERIAL"

    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.store = None
        self.config = None
        self.bundle = None
        self.env_sha = None

    def initialize(self, runtime_config: RuntimeConfig, bundle: RunBundle) -> None:
        runtime_config.validate()
        if runtime_config.backend_id != self.backend_id:
            raise BackendError("config backend_id mismatch")
        self.config = runtime_config
        self.bundle = bundle
        self.env_sha = runtime_environment_sha256(
            {"device": runtime_config.device})
        self.store = RowStore(
            runtime_config.run_dir,
            bundle.resume_identity(runtime_config, self.env_sha),
            make_verifier(bundle))
        self._initialized = True

    def load_model(self, model_artifact: dict) -> None:
        want = self.bundle.artifact_ids["model_artifact_sha256"]
        got = model_artifact_sha256(model_artifact)
        if got != want:
            raise BackendError(
                f"model artifact hashes {got[:16]}... but the run identity "
                f"pins {want[:16]}...")
        self.model, self.tokenizer = load_model_artifact(model_artifact)

    def validate_artifacts(self) -> dict:
        out = {"specs": len(self.bundle.specs)}
        for name, arr in self.bundle.axes.items():
            if arr is None:
                continue
            if arr.shape[1] != 2048 or not np.isfinite(arr).all():
                raise BackendError(f"{name} axis tensor invalid")
            rms = np.sqrt(np.mean(arr * arr, axis=1))
            if not np.allclose(rms, 1.0, rtol=0.0, atol=1e-9):
                raise BackendError(f"{name} axis rows are not unit RMS")
            out[f"axes.{name}"] = list(arr.shape)
        return out

    def execute_rows(self, row_specs: list) -> dict:
        self.store.log("EXECUTE_ROWS", backend=self.backend_id,
                       n_specs=len(row_specs))
        return run_serial_core(
            row_specs, self.bundle, self.store, self.model, self.tokenizer,
            backend_id=self.backend_id, worker_id="w0",
            environment_sha256=self.env_sha)

    def checkpoint(self) -> dict:
        return {"completed": len(self.store.load_completed())}

    def resume(self) -> dict:
        done = self.store.load_completed()
        remaining = [s for s in self.bundle.specs if s.row_id not in done]
        return {"completed": len(done), "remaining": len(remaining),
                "remaining_specs": remaining}

    def finalize_records(self) -> dict:
        return self.store.finalize(self.bundle.specs)

    def shutdown(self) -> None:
        self.model = None
        self.tokenizer = None


# ==========================================================================
# B200_REPLICA
# ==========================================================================


def _replica_worker(payload: dict) -> dict:
    """Top-level worker entrypoint (spawn-safe)."""
    from .backend_interface import RunBundle, RuntimeConfig
    bundle = payload["bundle"]
    config: RuntimeConfig = payload["config"]
    env_sha = payload["environment_sha256"]
    store = RowStore(config.run_dir, payload["resume_identity"],
                     make_verifier(bundle))
    model, tokenizer = load_model_artifact(payload["model_artifact"])
    crash_after = payload.get("crash_after_rows")
    counter = {"n": 0}

    generate_one = None
    if crash_after is not None:
        def generate_one(task, seed, direction, signed_alpha):
            if counter["n"] >= crash_after:
                raise SystemExit(97)  # injected crash for tests
            counter["n"] += 1
            return generate_branch(model, tokenizer, task, seed,
                                   direction, signed_alpha)

    return run_serial_core(
        payload["specs"], bundle, store, model, tokenizer,
        backend_id="B200_REPLICA", worker_id=payload["worker_id"],
        environment_sha256=env_sha, generate_one=generate_one)


def _worker_main(payload, queue):
    try:
        out = _replica_worker(payload)
        queue.put(("ok", payload["worker_id"], out))
    except SystemExit as exc:
        queue.put(("crash", payload["worker_id"], int(exc.code or 1)))
        raise
    except BaseException as exc:  # noqa: BLE001 - report then die
        queue.put(("error", payload["worker_id"], repr(exc)))
        raise


class ReplicaBackend(Backend):
    backend_id = "B200_REPLICA"

    def __init__(self):
        super().__init__()
        self.config = None
        self.bundle = None
        self.store = None
        self.env_sha = None
        self.model_artifact = None
        self.crash_plan: dict[str, int] = {}   # worker_id -> crash_after (tests)

    def initialize(self, runtime_config: RuntimeConfig, bundle: RunBundle) -> None:
        runtime_config.validate()
        if runtime_config.backend_id != self.backend_id:
            raise BackendError("config backend_id mismatch")
        if runtime_config.worker_count < 1:
            raise BackendError("worker_count must be >= 1")
        self.config = runtime_config
        self.bundle = bundle
        self.env_sha = runtime_environment_sha256(
            {"device": runtime_config.device})
        self.store = RowStore(
            runtime_config.run_dir,
            bundle.resume_identity(runtime_config, self.env_sha),
            make_verifier(bundle))
        self._initialized = True

    def load_model(self, model_artifact: dict) -> None:
        want = self.bundle.artifact_ids["model_artifact_sha256"]
        got = model_artifact_sha256(model_artifact)
        if got != want:
            raise BackendError("model artifact identity mismatch")
        self.model_artifact = dict(model_artifact)

    def validate_artifacts(self) -> dict:
        return ReferenceSerialBackend.validate_artifacts(self)

    def partition(self, row_specs: list) -> list[list]:
        """Deterministic task-affine allocation: task g -> worker g mod W.

        Task affinity keeps a task's baseline bank and its intervention rows
        in one worker, so the sealed h_base pairing never crosses processes.
        """
        groups = group_specs_by_task(row_specs)
        parts: list[list] = [[] for _ in range(self.config.worker_count)]
        for g, (_task_id, specs) in enumerate(groups):
            parts[g % self.config.worker_count].extend(specs)
        return parts

    def execute_rows(self, row_specs: list, max_restarts: int = 2) -> dict:
        self.store.log("EXECUTE_ROWS", backend=self.backend_id,
                       n_specs=len(row_specs),
                       worker_count=self.config.worker_count)
        parts = self.partition(row_specs)
        ctx = mp.get_context("spawn")
        results: dict[str, Any] = {}
        restarts = 0
        pending = {f"w{i}": part for i, part in enumerate(parts) if part}
        while pending:
            procs = {}
            queue = ctx.Queue()
            for worker_id, part in pending.items():
                payload = {
                    "bundle": self.bundle,
                    "config": self.config,
                    "specs": part,
                    "worker_id": worker_id,
                    "model_artifact": self.model_artifact,
                    "environment_sha256": self.env_sha,
                    "resume_identity": self.bundle.resume_identity(
                        self.config, self.env_sha),
                    "crash_after_rows": self.crash_plan.pop(worker_id, None),
                }
                p = ctx.Process(target=_worker_main, args=(payload, queue))
                p.start()
                procs[worker_id] = p
            for worker_id, p in procs.items():
                p.join()
            reported = {}
            import queue as queue_mod
            while True:
                try:
                    kind, worker_id, data = queue.get(timeout=1.0)
                except queue_mod.Empty:
                    break
                reported[worker_id] = (kind, data)
            failed = {}
            for worker_id, part in pending.items():
                kind, data = reported.get(worker_id, ("error", "no report"))
                if kind == "ok":
                    results[worker_id] = data
                else:
                    failed[worker_id] = part
                    self.store.log("WORKER_FAILED", worker=worker_id,
                                   kind=kind, detail=str(data))
            if failed:
                restarts += 1
                if restarts > max_restarts:
                    raise BackendError(
                        f"workers kept failing after {max_restarts} restarts: "
                        f"{sorted(failed)}")
                # safe restart: committed rows are atomic; the rescan inside
                # run_serial_core regenerates only missing rows
                pending = failed
            else:
                pending = {}
        return {"workers": results,
                "n_generated": sum(r["n_generated"] for r in results.values()),
                "restarts": restarts}

    def checkpoint(self) -> dict:
        return {"completed": len(self.store.load_completed())}

    def resume(self) -> dict:
        done = self.store.load_completed()
        remaining = [s for s in self.bundle.specs if s.row_id not in done]
        return {"completed": len(done), "remaining": len(remaining),
                "remaining_specs": remaining}

    def finalize_records(self) -> dict:
        return self.store.finalize(self.bundle.specs)

    def shutdown(self) -> None:
        self.model_artifact = None


# ==========================================================================
# B200_BATCHED
# ==========================================================================


class BatchedBackend(Backend):
    backend_id = "B200_BATCHED"

    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.store = None
        self.config = None
        self.bundle = None
        self.env_sha = None
        self._batch_counter = 0

    def initialize(self, runtime_config: RuntimeConfig, bundle: RunBundle) -> None:
        runtime_config.validate()
        if runtime_config.backend_id != self.backend_id:
            raise BackendError("config backend_id mismatch")
        if runtime_config.batch_size < 1:
            raise BackendError("batch_size must be >= 1")
        self.config = runtime_config
        self.bundle = bundle
        self.env_sha = runtime_environment_sha256(
            {"device": runtime_config.device})
        self.store = RowStore(
            runtime_config.run_dir,
            bundle.resume_identity(runtime_config, self.env_sha),
            make_verifier(bundle))
        self._initialized = True

    def load_model(self, model_artifact: dict) -> None:
        want = self.bundle.artifact_ids["model_artifact_sha256"]
        got = model_artifact_sha256(model_artifact)
        if got != want:
            raise BackendError("model artifact identity mismatch")
        self.model, self.tokenizer = load_model_artifact(model_artifact)

    def validate_artifacts(self) -> dict:
        return ReferenceSerialBackend.validate_artifacts(self)

    def _next_batch_id(self) -> str:
        self._batch_counter += 1
        return f"b{self._batch_counter:06d}"

    def _run_batch(self, specs: list[RowSpec], h_bases: dict[str, np.ndarray],
                   collect_boundary: dict | None = None) -> None:
        tasks = [self.bundle.tasks_by_id[s.task_id] for s in specs]
        seeds = [s.seed for s in specs]
        directions = []
        alphas = []
        for s in specs:
            if s.bank is None:
                directions.append(None)
                alphas.append(0.0)
            else:
                bank = self.bundle.axes[s.bank]
                directions.append(np.asarray(bank[s.direction_id - 1],
                                             dtype=np.float64))
                alphas.append(s.signed_alpha())
        batch_id = self._next_batch_id()
        gens = generate_batch(self.model, self.tokenizer, tasks, seeds,
                              directions, alphas,
                              compaction=self.config.compaction)
        for s, gen in zip(specs, gens):
            if collect_boundary is not None and s.arm == "baseline":
                collect_boundary.setdefault(s.task_id, []).append(
                    (s.stream_index, _boundary_of(gen)))
            rho = None
            if s.arm != "baseline":
                rho = rho_for(s, gen, h_bases[s.task_id])
            _commit(self.store, self.bundle, s, gen, rho, self.tokenizer,
                    self.bundle.tasks_by_id[s.task_id],
                    backend_id=self.backend_id, worker_id="w0",
                    batch_id=batch_id, batch_size=len(specs),
                    environment_sha256=self.env_sha)

    def execute_rows(self, row_specs: list) -> dict:
        self.store.log("EXECUTE_ROWS", backend=self.backend_id,
                       n_specs=len(row_specs),
                       batch_size=self.config.batch_size,
                       compaction=self.config.compaction)
        done = self.store.load_completed()
        groups = group_specs_by_task(row_specs)
        B = self.config.batch_size

        # phase 1: all missing baseline rows, batched across tasks
        missing_base = [s for _, specs in groups for s in specs
                        if s.arm == "baseline" and s.row_id not in done]
        boundaries: dict[str, list] = {}
        for i in range(0, len(missing_base), B):
            self._run_batch(missing_base[i:i + B], {}, collect_boundary=boundaries)

        h_bases: dict[str, np.ndarray] = {}
        for task_id, pairs in boundaries.items():
            ref = pairs[0][1]
            for stream, b in pairs[1:]:
                if not np.array_equal(b, ref):
                    raise BackendError(
                        f"task {task_id}: baseline prefill boundary differs "
                        f"across streams (stream {stream}) — prefill must be "
                        f"deterministic")
            h_bases[task_id] = ref

        # phase 1b: replay stream-0 baselines for tasks whose baselines all
        # pre-exist but which still need intervention rows
        need_replay = []
        for task_id, specs in groups:
            if task_id in h_bases:
                continue
            others_missing = [s for s in specs
                              if s.arm != "baseline" and s.row_id not in done]
            if not others_missing:
                continue
            all_base = [s for s in self.bundle.specs
                        if s.task_id == task_id and s.arm == "baseline"]
            if not all_base:
                raise BackendError(
                    f"task {task_id}: no baseline rows exist in the sealed "
                    f"design; h_base cannot be established")
            need_replay.append(min(all_base, key=lambda s: s.stream_index))
        for i in range(0, len(need_replay), B):
            chunk = need_replay[i:i + B]
            tasks = [self.bundle.tasks_by_id[s.task_id] for s in chunk]
            gens = generate_batch(
                self.model, self.tokenizer, tasks, [s.seed for s in chunk],
                [None] * len(chunk), [0.0] * len(chunk),
                compaction=self.config.compaction)
            for s, gen in zip(chunk, gens):
                stored = done.get(s.row_id)
                if stored is None or list(stored["generated_token_ids"]) != [
                        int(t) for t in gen["generated_token_ids"]]:
                    raise BackendError(
                        f"task {s.task_id}: baseline replay does not "
                        f"reproduce the stored row; boundary cannot be "
                        f"re-established safely")
                h_bases[s.task_id] = _boundary_of(gen)

        # phase 2: all missing non-baseline rows, batched across tasks/arms
        missing_other = [s for _, specs in groups for s in specs
                         if s.arm != "baseline" and s.row_id not in done]
        n = 0
        for i in range(0, len(missing_other), B):
            chunk = missing_other[i:i + B]
            for s in chunk:
                if s.task_id not in h_bases:
                    raise BackendError(
                        f"internal: no h_base for task {s.task_id}")
            self._run_batch(chunk, h_bases)
            n += len(chunk)
        return {"n_generated": len(missing_base) + n}

    def checkpoint(self) -> dict:
        return {"completed": len(self.store.load_completed())}

    def resume(self) -> dict:
        done = self.store.load_completed()
        remaining = [s for s in self.bundle.specs if s.row_id not in done]
        return {"completed": len(done), "remaining": len(remaining),
                "remaining_specs": remaining}

    def finalize_records(self) -> dict:
        return self.store.finalize(self.bundle.specs)

    def shutdown(self) -> None:
        self.model = None
        self.tokenizer = None


BACKENDS = {
    "REFERENCE_SERIAL": ReferenceSerialBackend,
    "B200_REPLICA": ReplicaBackend,
    "B200_BATCHED": BatchedBackend,
}
