"""Stability monitoring (contract §17).

Detectors (thresholds frozen by the contract where the contract states one):

===========================  ==============================================
detector                     rule
===========================  ==============================================
``NAN_LOSS``                 loss (or grad norm) is NaN/Inf
``GRAD_NORM_EXPLOSION``      grad norm > 50 on 3 CONSECUTIVE steps
``OPTIMIZER_OVERFLOW``       non-finite gradients seen at the update
``LOSS_DIVERGENCE``          loss > 2x the median of the trailing 200 steps
``MEMORY_GROWTH``            allocated memory > 1.25x the 200-step-old value
``OOM``                      CUDA out-of-memory raised by the step
``THROUGHPUT_COLLAPSE``      < 40 % of the BENCH tokens/s over a 5-minute window
``CORRUPTED_CHECKPOINT``     checkpoint hash verification failed on load
``REPEATED_ZERO_GRAD``       grad norm exactly 0 on 10 consecutive steps
``FAST_STATE_NORM``          ||s|| above the configured bound (FL5)
===========================  ==============================================

Contract §17 leaves three thresholds unspecified (memory growth, repeated zero
grad, fast-state norm).  The values above are IMPLEMENTATION CHOICES, exposed
on :class:`StabilityConfig` and recorded in every failure record; they are
declared here rather than tuned later (see docs/CONTRACT_AMENDMENTS.md,
Amendment 6).

Response policy (contract §17, deliberately bounded):

* transient INFRASTRUCTURE error -> EXACTLY ONE retry from the last valid
  checkpoint, executed by the trainer;
* scientific instability -> structured failure record, stop the arm, move on
  per the scheduler;
* NO hyperparameter improvisation of any kind.  Nothing in this module can
  mutate a learning rate, batch size, schedule, or objective; a unit test pins
  that the arm configuration is unchanged after a fatal event.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

STABILITY_EVENT_SCHEMA = "flb200.stability_event.v1"
STABILITY_FAILURE_SCHEMA = "flb200.stability_failure.v1"

SEVERITY_OBSERVATION = "OBSERVATION"
SEVERITY_TRANSIENT_INFRA = "TRANSIENT_INFRA"
SEVERITY_SCIENTIFIC_INSTABILITY = "SCIENTIFIC_INSTABILITY"

KIND_NAN_LOSS = "NAN_LOSS"
KIND_GRAD_NORM_EXPLOSION = "GRAD_NORM_EXPLOSION"
KIND_OPTIMIZER_OVERFLOW = "OPTIMIZER_OVERFLOW"
KIND_LOSS_DIVERGENCE = "LOSS_DIVERGENCE"
KIND_MEMORY_GROWTH = "MEMORY_GROWTH"
KIND_OOM = "OOM"
KIND_THROUGHPUT_COLLAPSE = "THROUGHPUT_COLLAPSE"
KIND_CORRUPTED_CHECKPOINT = "CORRUPTED_CHECKPOINT"
KIND_REPEATED_ZERO_GRAD = "REPEATED_ZERO_GRAD"
KIND_FAST_STATE_NORM = "FAST_STATE_NORM"
KIND_TRANSIENT_INFRA_RETRY = "TRANSIENT_INFRA_RETRY"


class StabilityAbort(RuntimeError):
    """Raised by the trainer when a fatal stability event ends an arm."""

    def __init__(self, message: str, record: dict[str, Any]):
        super().__init__(message)
        self.record = record


@dataclass(frozen=True)
class StabilityConfig:
    # Contract-frozen thresholds.
    grad_norm_threshold: float = 50.0
    grad_norm_consecutive: int = 3
    loss_divergence_factor: float = 2.0
    loss_divergence_window: int = 200
    throughput_collapse_fraction: float = 0.40
    throughput_collapse_window_s: float = 300.0
    # Implementation choices (contract silent; see module docstring).
    memory_growth_factor: float = 1.25
    memory_growth_window: int = 200
    zero_grad_consecutive: int = 10
    fast_state_norm_max: float = 1000.0
    # BENCH throughput (tokens/s) supplied by the scheduler's BENCH stage.
    bench_tokens_per_second: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StabilityEvent:
    step: int
    kind: str
    severity: str
    detail: str
    values: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "schema": STABILITY_EVENT_SCHEMA,
            "step": int(self.step),
            "kind": self.kind,
            "severity": self.severity,
            "detail": self.detail,
            "values": dict(self.values),
        }


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


class StabilityMonitor:
    """Per-arm detector state machine.  Observation only; never mutates config."""

    def __init__(self, config: StabilityConfig | None = None, arm_id: str = "") -> None:
        self.config = config or StabilityConfig()
        self.arm_id = arm_id
        self.events: list[StabilityEvent] = []
        self.fatal_event: StabilityEvent | None = None
        self._losses: list[float] = []
        self._grad_streak = 0
        self._zero_grad_streak = 0
        self._memory: list[float] = []
        self._throughput: list[tuple[float, float]] = []  # (elapsed_s, tokens)

    # -- helpers -----------------------------------------------------------
    def _emit(
        self, step: int, kind: str, severity: str, detail: str, **values: Any
    ) -> StabilityEvent:
        event = StabilityEvent(
            step=step, kind=kind, severity=severity, detail=detail, values=values
        )
        self.events.append(event)
        if severity == SEVERITY_SCIENTIFIC_INSTABILITY and self.fatal_event is None:
            self.fatal_event = event
        return event

    @property
    def fatal(self) -> bool:
        return self.fatal_event is not None

    def scientific_events(self) -> list[StabilityEvent]:
        return [
            e for e in self.events if e.severity == SEVERITY_SCIENTIFIC_INSTABILITY
        ]

    # -- detectors ---------------------------------------------------------
    def observe_step(
        self,
        *,
        step: int,
        loss: float,
        grad_norm: float | None = None,
        tokens: int = 0,
        step_seconds: float = 0.0,
        memory_bytes: float | None = None,
        grads_finite: bool = True,
    ) -> list[StabilityEvent]:
        cfg = self.config
        new: list[StabilityEvent] = []
        before = len(self.events)

        if not _finite(loss):
            self._emit(
                step,
                KIND_NAN_LOSS,
                SEVERITY_SCIENTIFIC_INSTABILITY,
                f"loss is not finite ({loss!r})",
                loss=str(loss),
            )
        if grad_norm is not None and not _finite(grad_norm):
            self._emit(
                step,
                KIND_NAN_LOSS,
                SEVERITY_SCIENTIFIC_INSTABILITY,
                f"gradient norm is not finite ({grad_norm!r})",
                grad_norm=str(grad_norm),
            )
        if not grads_finite:
            self._emit(
                step,
                KIND_OPTIMIZER_OVERFLOW,
                SEVERITY_SCIENTIFIC_INSTABILITY,
                "non-finite gradients at the optimizer update",
            )

        if grad_norm is not None and _finite(grad_norm):
            if grad_norm > cfg.grad_norm_threshold:
                self._grad_streak += 1
                if self._grad_streak >= cfg.grad_norm_consecutive:
                    self._emit(
                        step,
                        KIND_GRAD_NORM_EXPLOSION,
                        SEVERITY_SCIENTIFIC_INSTABILITY,
                        f"grad norm > {cfg.grad_norm_threshold} on "
                        f"{self._grad_streak} consecutive steps",
                        grad_norm=float(grad_norm),
                        consecutive=self._grad_streak,
                    )
            else:
                self._grad_streak = 0
            if grad_norm == 0.0:
                self._zero_grad_streak += 1
                if self._zero_grad_streak >= cfg.zero_grad_consecutive:
                    self._emit(
                        step,
                        KIND_REPEATED_ZERO_GRAD,
                        SEVERITY_SCIENTIFIC_INSTABILITY,
                        f"gradient norm exactly zero on "
                        f"{self._zero_grad_streak} consecutive steps",
                        consecutive=self._zero_grad_streak,
                    )
            else:
                self._zero_grad_streak = 0

        if _finite(loss):
            if len(self._losses) >= cfg.loss_divergence_window:
                window = self._losses[-cfg.loss_divergence_window :]
                median = statistics.median(window)
                if median > 0 and loss > cfg.loss_divergence_factor * median:
                    self._emit(
                        step,
                        KIND_LOSS_DIVERGENCE,
                        SEVERITY_SCIENTIFIC_INSTABILITY,
                        f"loss {loss:.4f} > {cfg.loss_divergence_factor}x the "
                        f"median ({median:.4f}) of the trailing "
                        f"{cfg.loss_divergence_window} steps",
                        loss=float(loss),
                        trailing_median=float(median),
                    )
            self._losses.append(float(loss))

        if memory_bytes is not None:
            self._memory.append(float(memory_bytes))
            if len(self._memory) > cfg.memory_growth_window:
                past = self._memory[-cfg.memory_growth_window - 1]
                if past > 0 and memory_bytes > cfg.memory_growth_factor * past:
                    self._emit(
                        step,
                        KIND_MEMORY_GROWTH,
                        SEVERITY_OBSERVATION,
                        f"allocated memory grew by more than "
                        f"{cfg.memory_growth_factor}x over "
                        f"{cfg.memory_growth_window} steps",
                        memory_bytes=float(memory_bytes),
                        reference_bytes=float(past),
                    )

        if tokens and step_seconds > 0:
            self._throughput.append((float(step_seconds), float(tokens)))
            total_time = sum(t for t, _ in self._throughput)
            while total_time > cfg.throughput_collapse_window_s and len(self._throughput) > 1:
                dropped_t, _ = self._throughput.pop(0)
                total_time -= dropped_t
            if (
                cfg.bench_tokens_per_second
                and total_time >= cfg.throughput_collapse_window_s
            ):
                total_tokens = sum(n for _, n in self._throughput)
                rate = total_tokens / total_time
                if rate < cfg.throughput_collapse_fraction * cfg.bench_tokens_per_second:
                    self._emit(
                        step,
                        KIND_THROUGHPUT_COLLAPSE,
                        SEVERITY_OBSERVATION,
                        f"throughput {rate:.1f} tok/s < "
                        f"{cfg.throughput_collapse_fraction:.0%} of BENCH "
                        f"({cfg.bench_tokens_per_second:.1f} tok/s) over "
                        f"{total_time:.0f}s",
                        tokens_per_second=float(rate),
                    )
        new = self.events[before:]
        return new

    def observe_fast_state_norm(self, step: int, norm: float) -> list[StabilityEvent]:
        before = len(self.events)
        if not _finite(norm) or norm > self.config.fast_state_norm_max:
            self._emit(
                step,
                KIND_FAST_STATE_NORM,
                SEVERITY_SCIENTIFIC_INSTABILITY,
                f"fast-state norm {norm!r} exceeds bound "
                f"{self.config.fast_state_norm_max}",
                norm=str(norm),
            )
        return self.events[before:]

    def record_corrupted_checkpoint(self, step: int, path: str, detail: str):
        return self._emit(
            step,
            KIND_CORRUPTED_CHECKPOINT,
            SEVERITY_SCIENTIFIC_INSTABILITY,
            f"checkpoint verification failed for {path}: {detail}",
            path=path,
        )

    def record_oom(self, step: int, detail: str):
        return self._emit(
            step, KIND_OOM, SEVERITY_TRANSIENT_INFRA, f"CUDA OOM: {detail}"
        )

    def record_transient_infra(self, step: int, detail: str):
        return self._emit(
            step, KIND_TRANSIENT_INFRA_RETRY, SEVERITY_TRANSIENT_INFRA, detail
        )

    # -- exception classification -----------------------------------------
    @staticmethod
    def classify_exception(exc: BaseException) -> str:
        """Classify an exception as transient infrastructure vs fatal.

        Only a small, explicit set counts as transient (contract §17 allows
        EXACTLY ONE retry for those).  Everything else is fatal: a scientific
        failure must be recorded, not retried into a different result.
        """
        name = type(exc).__name__
        text = f"{name}: {exc}".lower()
        if name in ("OutOfMemoryError", "OutOfMemoryError_"):
            return SEVERITY_TRANSIENT_INFRA
        transient_markers = (
            "cuda out of memory",
            "out of memory",
            "nccl",
            "cuda error: unspecified launch failure",
            "cuda driver",
            "connection reset",
            "transport endpoint",
            "input/output error",
            "no space left on device",
        )
        if any(marker in text for marker in transient_markers):
            return SEVERITY_TRANSIENT_INFRA
        return SEVERITY_SCIENTIFIC_INSTABILITY

    # -- reporting ---------------------------------------------------------
    def failure_record(
        self,
        *,
        arm_id: str,
        step: int,
        reason: str,
        config_snapshot: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = {
            "schema": STABILITY_FAILURE_SCHEMA,
            "arm_id": arm_id or self.arm_id,
            "step": int(step),
            "reason": reason,
            "response": (
                "RECORDED_AND_STOPPED — no hyperparameter improvisation "
                "(contract §17)"
            ),
            "thresholds": self.config.to_dict(),
            "events": [e.to_dict() for e in self.events],
            "arm_config": config_snapshot,
        }
        if extra:
            record.update(extra)
        return record

    def events_as_dicts(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.events]


def summarise_events(events: Sequence[StabilityEvent]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        counts[event.kind] = counts.get(event.kind, 0) + 1
    return counts
