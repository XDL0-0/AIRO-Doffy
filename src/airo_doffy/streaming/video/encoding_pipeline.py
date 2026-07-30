"""Bounded drop-oldest worker around a synchronous video encoder."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

from ...config.models import VideoStreamingConfig
from ...core.buffers import is_newer_sequence
from ...core.clocks import Clock, MonotonicClock
from ...core.errors import LifecycleError, ModelValidationError
from ...core.types import EncodedFrame, ProcessedFrame
from .base import VideoEncoder


@dataclass(frozen=True, slots=True)
class VideoEncodingMetrics:
    """Snapshot of bounded pipeline throughput and failure counters."""

    submitted: int
    encoded: int
    dropped_input: int
    dropped_output: int
    rejected_stale: int
    errors: int
    bytes_encoded: int
    total_encode_ns: int
    last_encode_ns: int


class LatestVideoEncodingPipeline:
    """Encode on one worker while dropping oldest queued frames on overload."""

    def __init__(
        self,
        encoder: VideoEncoder,
        config: VideoStreamingConfig,
        *,
        clock: Clock | None = None,
    ) -> None:
        self._encoder = encoder
        self._clock = clock or MonotonicClock()
        self._input_capacity = config.input_queue_capacity
        self._output_capacity = config.output_queue_capacity
        self._condition = threading.Condition()
        self._input: deque[ProcessedFrame] = deque()
        self._output: deque[EncodedFrame] = deque()
        self._thread: threading.Thread | None = None
        self._started = False
        self._closed = False
        self._stop = False
        self._health_error: Exception | None = None
        self._last_submitted_sequence: int | None = None
        self._submitted = 0
        self._encoded = 0
        self._dropped_input = 0
        self._dropped_output = 0
        self._rejected_stale = 0
        self._errors = 0
        self._bytes_encoded = 0
        self._total_encode_ns = 0
        self._last_encode_ns = 0

    @property
    def health_error(self) -> Exception | None:
        with self._condition:
            return self._health_error

    @property
    def metrics(self) -> VideoEncodingMetrics:
        with self._condition:
            return VideoEncodingMetrics(
                submitted=self._submitted,
                encoded=self._encoded,
                dropped_input=self._dropped_input,
                dropped_output=self._dropped_output,
                rejected_stale=self._rejected_stale,
                errors=self._errors,
                bytes_encoded=self._bytes_encoded,
                total_encode_ns=self._total_encode_ns,
                last_encode_ns=self._last_encode_ns,
            )

    def start(self) -> None:
        with self._condition:
            if self._closed:
                raise LifecycleError("cannot start a closed video encoding pipeline")
            if self._started:
                raise LifecycleError("video encoding pipeline is already started")
        start_encoder = getattr(self._encoder, "start", None)
        if callable(start_encoder):
            start_encoder()
        with self._condition:
            self._started = True
            self._stop = False
            self._thread = threading.Thread(
                target=self._worker,
                name="airo-doffy-video-encoder",
                daemon=True,
            )
            try:
                self._thread.start()
            except Exception:
                self._thread = None
                self._started = False
                close_encoder = getattr(self._encoder, "close", None)
                if callable(close_encoder):
                    close_encoder()
                raise

    def _require_started(self) -> None:
        if self._closed:
            raise LifecycleError("video encoding pipeline is closed")
        if not self._started:
            raise LifecycleError("video encoding pipeline has not been started")

    def submit(self, frame: ProcessedFrame) -> bool:
        if not isinstance(frame, ProcessedFrame):
            raise ModelValidationError("frame must be a ProcessedFrame")
        with self._condition:
            self._require_started()
            if (
                self._last_submitted_sequence is not None
                and not is_newer_sequence(
                    frame.sequence,
                    self._last_submitted_sequence,
                )
            ):
                self._rejected_stale += 1
                return False
            self._last_submitted_sequence = frame.sequence
            self._submitted += 1
            if len(self._input) >= self._input_capacity:
                self._input.popleft()
                self._dropped_input += 1
            self._input.append(frame)
            self._condition.notify()
            return True

    def _worker(self) -> None:
        while True:
            with self._condition:
                while not self._input and not self._stop:
                    self._condition.wait()
                if self._stop:
                    return
                frame = self._input.popleft()
            started_ns = self._clock.now_ns()
            try:
                encoded = self._encoder.encode(frame)
            except Exception as exc:
                with self._condition:
                    self._health_error = exc
                    self._errors += 1
                    self._stop = True
                    self._condition.notify_all()
                return
            elapsed_ns = max(0, self._clock.now_ns() - started_ns)
            with self._condition:
                if len(self._output) >= self._output_capacity:
                    self._output.popleft()
                    self._dropped_output += 1
                self._output.append(encoded)
                self._encoded += 1
                self._bytes_encoded += len(encoded.data)
                self._total_encode_ns += elapsed_ns
                self._last_encode_ns = elapsed_ns
                self._condition.notify_all()

    def read_latest(self) -> EncodedFrame | None:
        with self._condition:
            self._require_started()
            return self._output[-1] if self._output else None

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            if not self._started:
                self._closed = True
                close_encoder = getattr(self._encoder, "close", None)
                if callable(close_encoder):
                    close_encoder()
                return
            self._stop = True
            self._dropped_input += len(self._input)
            self._input.clear()
            thread = self._thread
            self._condition.notify_all()
        if thread is not None:
            thread.join(timeout=5.0)
            if thread.is_alive():
                raise LifecycleError(
                    "video encoder worker did not stop; pipeline was not invalidated"
                )
        close_encoder = getattr(self._encoder, "close", None)
        cleanup_error: Exception | None = None
        if callable(close_encoder):
            try:
                close_encoder()
            except Exception as exc:
                cleanup_error = exc
        with self._condition:
            self._started = False
            self._closed = True
            self._thread = None
            self._condition.notify_all()
        if cleanup_error is not None:
            raise LifecycleError("video encoder cleanup failed") from cleanup_error
