"""Transport-neutral video benchmark measurements and comparison runner."""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

from ...core.clocks import Clock, MonotonicClock
from ...core.errors import ModelValidationError
from ...core.types import EncodedFrame, ProcessedFrame
from .base import VideoEncoder, VideoTransport


@dataclass(frozen=True, slots=True)
class BenchmarkInput:
    """One processed frame plus when it entered the measured path."""

    frame: ProcessedFrame
    queued_timestamp_ns: int


@dataclass(frozen=True, slots=True)
class DeliveryReceipt:
    """Optional receiver/display timestamps for actual loss and E2E latency."""

    sequence: int
    received_timestamp_ns: int
    displayed_timestamp_ns: int | None = None
    wire_bytes: int | None = None


class DeliveryProbe(Protocol):
    """Deployment-specific acknowledgement source used by real benchmarks."""

    def wait_for(
        self,
        sequence: int,
        timeout_s: float,
    ) -> DeliveryReceipt | None:
        """Return a matching remote receipt or ``None`` on loss/timeout."""


@dataclass(frozen=True, slots=True)
class LatencySummary:
    count: int
    minimum_ns: int | None
    p50_ns: int | None
    p95_ns: int | None
    p99_ns: int | None
    maximum_ns: int | None
    mean_ns: float | None


def summarize_latencies(values: Iterable[int]) -> LatencySummary:
    """Summarize non-negative durations with deterministic nearest-rank values."""

    ordered = sorted(values)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in ordered
    ):
        raise ModelValidationError(
            "latency values must be non-negative integers"
        )
    if not ordered:
        return LatencySummary(0, None, None, None, None, None, None)

    def percentile(percent: int) -> int:
        index = max(0, (percent * len(ordered) + 99) // 100 - 1)
        return ordered[index]

    return LatencySummary(
        count=len(ordered),
        minimum_ns=ordered[0],
        p50_ns=percentile(50),
        p95_ns=percentile(95),
        p99_ns=percentile(99),
        maximum_ns=ordered[-1],
        mean_ns=sum(ordered) / len(ordered),
    )


@dataclass(frozen=True, slots=True)
class VideoBenchmarkResult:
    """Comparable result for legacy JPEG, WebRTC H.264, or RTP H.264."""

    path_name: str
    frames_attempted: int
    frames_encoded: int
    frames_submitted: int
    frame_errors: int
    component_drops: int
    encoded_bytes: int
    wire_bytes: int | None
    duration_ns: int
    encoded_bitrate_bps: float
    wire_bitrate_bps: float | None
    delivery_loss_rate: float | None
    cpu_time_ns: int
    cpu_utilization_percent: float
    gpu_utilization_percent: float | None
    queue_delay: LatencySummary
    encode_latency: LatencySummary
    send_latency: LatencySummary
    receive_latency: LatencySummary
    end_to_end_latency: LatencySummary

    def to_mapping(self) -> dict:
        """Return JSON-serializable nested primitives."""

        return dataclasses.asdict(self)


@dataclass(slots=True)
class VideoBenchmarkPath:
    """One independently constructed encoder/transport pair."""

    name: str
    encoder: VideoEncoder
    transport: VideoTransport


GpuSampler = Callable[[], float | None]
CpuClock = Callable[[], int]


def _start(component: object) -> None:
    start = getattr(component, "start", None)
    if callable(start):
        start()


def _close(component: object) -> None:
    close = getattr(component, "close", None)
    if callable(close):
        close()


def _component_drops(component: object) -> int:
    metrics = getattr(component, "metrics", None)
    if metrics is None:
        return 0
    names = (
        "dropped_input",
        "dropped_output",
        "dropped_latest",
        "dropped_stale",
        "dropped_late",
    )
    return sum(
        int(getattr(metrics, name, 0))
        for name in names
    )


def _wire_bytes(component: object) -> int | None:
    metrics = getattr(component, "metrics", None)
    value = None if metrics is None else getattr(metrics, "bytes_sent", None)
    return value if isinstance(value, int) else None


class VideoBenchmarkRunner:
    """Run a finite set of already processed frames through one video path."""

    def __init__(
        self,
        *,
        clock: Clock | None = None,
        cpu_clock: CpuClock = time.process_time_ns,
        gpu_sampler: GpuSampler | None = None,
        receipt_timeout_s: float = 0.25,
    ) -> None:
        timeout = float(receipt_timeout_s)
        if not math.isfinite(timeout) or timeout < 0:
            raise ModelValidationError(
                "receipt_timeout_s must be non-negative"
            )
        self._clock = clock or MonotonicClock()
        self._cpu_clock = cpu_clock
        self._gpu_sampler = gpu_sampler
        self._receipt_timeout_s = timeout

    def run(
        self,
        path: VideoBenchmarkPath,
        inputs: Iterable[BenchmarkInput],
        *,
        delivery_probe: DeliveryProbe | None = None,
    ) -> VideoBenchmarkResult:
        if not isinstance(path.name, str) or not path.name.strip():
            raise ModelValidationError("benchmark path name must be non-empty")
        prepared = tuple(inputs)
        if any(not isinstance(item, BenchmarkInput) for item in prepared):
            raise ModelValidationError(
                "benchmark inputs must contain BenchmarkInput values"
            )

        queue_delays: list[int] = []
        encode_latencies: list[int] = []
        send_latencies: list[int] = []
        receive_latencies: list[int] = []
        end_to_end_latencies: list[int] = []
        frames_encoded = 0
        frames_submitted = 0
        frame_errors = 0
        encoded_bytes = 0
        receipts = 0
        receipt_losses = 0
        receipt_wire_bytes = 0
        receipt_wire_bytes_known = True
        gpu_samples: list[float] = []

        wall_start = self._clock.now_ns()
        cpu_start = self._cpu_clock()
        started_encoder = False
        started_transport = False
        cleanup_errors: list[Exception] = []
        try:
            _start(path.encoder)
            started_encoder = True
            _start(path.transport)
            started_transport = True
            for item in prepared:
                encode_start = self._clock.now_ns()
                queue_delays.append(
                    max(0, encode_start - item.queued_timestamp_ns)
                )
                try:
                    encoded = path.encoder.encode(item.frame)
                    encode_end = self._clock.now_ns()
                    encode_latencies.append(max(0, encode_end - encode_start))
                    frames_encoded += 1
                    encoded_bytes += len(encoded.data)
                    send_start = self._clock.now_ns()
                    path.transport.send(encoded)
                    send_end = self._clock.now_ns()
                    send_latencies.append(max(0, send_end - send_start))
                    frames_submitted += 1
                    if delivery_probe is not None:
                        receipt = delivery_probe.wait_for(
                            encoded.sequence,
                            self._receipt_timeout_s,
                        )
                        if receipt is None:
                            receipt_losses += 1
                        else:
                            self._record_receipt(
                                encoded,
                                receipt,
                                receive_latencies,
                                end_to_end_latencies,
                            )
                            receipts += 1
                            if receipt.wire_bytes is None:
                                receipt_wire_bytes_known = False
                            else:
                                receipt_wire_bytes += receipt.wire_bytes
                    if self._gpu_sampler is not None:
                        sample = self._gpu_sampler()
                        if sample is not None:
                            gpu_samples.append(float(sample))
                except Exception:
                    frame_errors += 1
        finally:
            if started_transport:
                try:
                    _close(path.transport)
                except Exception as exc:
                    cleanup_errors.append(exc)
            if started_encoder:
                try:
                    _close(path.encoder)
                except Exception as exc:
                    cleanup_errors.append(exc)

        wall_end = self._clock.now_ns()
        cpu_end = self._cpu_clock()
        if cleanup_errors:
            raise cleanup_errors[0]
        duration_ns = max(1, wall_end - wall_start)
        cpu_time_ns = max(0, cpu_end - cpu_start)
        transport_wire_bytes = _wire_bytes(path.transport)
        wire_bytes = (
            receipt_wire_bytes
            if (
                delivery_probe is not None
                and receipts
                and receipt_wire_bytes_known
            )
            else transport_wire_bytes
        )
        return VideoBenchmarkResult(
            path_name=path.name,
            frames_attempted=len(prepared),
            frames_encoded=frames_encoded,
            frames_submitted=frames_submitted,
            frame_errors=frame_errors,
            component_drops=_component_drops(path.transport),
            encoded_bytes=encoded_bytes,
            wire_bytes=wire_bytes,
            duration_ns=duration_ns,
            encoded_bitrate_bps=encoded_bytes * 8_000_000_000 / duration_ns,
            wire_bitrate_bps=(
                None
                if wire_bytes is None
                else wire_bytes * 8_000_000_000 / duration_ns
            ),
            delivery_loss_rate=(
                None
                if delivery_probe is None
                else receipt_losses / max(1, receipts + receipt_losses)
            ),
            cpu_time_ns=cpu_time_ns,
            cpu_utilization_percent=cpu_time_ns * 100.0 / duration_ns,
            gpu_utilization_percent=(
                None
                if not gpu_samples
                else sum(gpu_samples) / len(gpu_samples)
            ),
            queue_delay=summarize_latencies(queue_delays),
            encode_latency=summarize_latencies(encode_latencies),
            send_latency=summarize_latencies(send_latencies),
            receive_latency=summarize_latencies(receive_latencies),
            end_to_end_latency=summarize_latencies(end_to_end_latencies),
        )

    @staticmethod
    def _record_receipt(
        encoded: EncodedFrame,
        receipt: DeliveryReceipt,
        receive_latencies: list[int],
        end_to_end_latencies: list[int],
    ) -> None:
        if receipt.sequence != encoded.sequence:
            raise ModelValidationError(
                "delivery receipt sequence does not match encoded frame"
            )
        receive_latencies.append(
            max(0, receipt.received_timestamp_ns - encoded.encoded_timestamp_ns)
        )
        if receipt.displayed_timestamp_ns is not None:
            end_to_end_latencies.append(
                max(
                    0,
                    receipt.displayed_timestamp_ns
                    - encoded.source_timestamp_ns,
                )
            )


def compare_video_paths(
    paths: Iterable[VideoBenchmarkPath],
    inputs: Iterable[BenchmarkInput],
    *,
    runner: VideoBenchmarkRunner | None = None,
) -> tuple[VideoBenchmarkResult, ...]:
    """Run independently constructed paths over the same immutable inputs."""

    selected = tuple(paths)
    names = [path.name for path in selected]
    if len(set(names)) != len(names):
        raise ModelValidationError("benchmark path names must be unique")
    prepared = tuple(inputs)
    benchmark = runner or VideoBenchmarkRunner()
    return tuple(benchmark.run(path, prepared) for path in selected)
