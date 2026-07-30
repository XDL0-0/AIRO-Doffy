"""Recording behavior composed onto the existing teleoperation loop."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock

from ..core.errors import (
    CommandRejectedError,
    LifecycleError,
    ModelValidationError,
)
from ..core.events import RuntimeCommand
from ..recording import (
    Episode,
    EpisodeRollback,
    EpisodeState,
    EpisodeStateMachine,
    EpisodeWriter,
    ExportQueueFullError,
    ExportTaskKind,
    ExportTicket,
    ExportWorker,
    RecordingSample,
    RecordingSchema,
    SampleBuffer,
)
from .session import TeleopCycle, TeleopSession

SampleFactory = Callable[[TeleopCycle], RecordingSample | None]


@dataclass(frozen=True, slots=True)
class DataCollectionStatus:
    """Immutable recording extension status."""

    state: EpisodeState
    next_episode_index: int
    active_samples: int
    pending_episode_index: int | None
    queued_exports: int
    export_busy: bool
    last_error: str | None


class RecordingCycleExtension:
    """Build and export episodes from cycles without owning the teleop loop."""

    def __init__(
        self,
        *,
        schema: RecordingSchema,
        task: str,
        sample_factory: SampleFactory,
        writer: EpisodeWriter,
        rollback: EpisodeRollback | None = None,
        next_episode_index: int = 0,
        export_queue_capacity: int = 2,
        sample_capacity: int | None = None,
    ) -> None:
        if not isinstance(schema, RecordingSchema):
            raise ModelValidationError("schema must be a RecordingSchema")
        if not isinstance(task, str) or not task.strip():
            raise ModelValidationError("task must be a non-empty string")
        if not callable(sample_factory):
            raise ModelValidationError("sample_factory must be callable")
        if not isinstance(writer, EpisodeWriter):
            raise ModelValidationError("writer must satisfy EpisodeWriter")
        if rollback is not None and not isinstance(rollback, EpisodeRollback):
            raise ModelValidationError("rollback must satisfy EpisodeRollback")
        self._schema = schema
        self._task = task
        self._sample_factory = sample_factory
        self._state = EpisodeStateMachine(
            next_episode_index=next_episode_index
        )
        self._worker = ExportWorker(
            writer,
            rollback=rollback,
            capacity=export_queue_capacity,
        )
        self._sample_capacity = sample_capacity
        self._buffer: SampleBuffer | None = None
        self._pending_ticket: ExportTicket | None = None
        self._pending_episode: Episode | None = None
        self._started = False
        self._closed = False
        self._lock = RLock()

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise LifecycleError("cannot start closed recording extension")
            if self._started:
                raise LifecycleError("recording extension is already started")
        self._worker.start()
        with self._lock:
            self._started = True

    def on_cycle(self, cycle: object) -> None:
        if not isinstance(cycle, TeleopCycle):
            raise ModelValidationError("recording extension requires TeleopCycle")
        self.poll()
        with self._lock:
            if self._state.snapshot().state is not EpisodeState.RECORDING:
                return
            buffer = self._buffer
        assert buffer is not None
        sample = self._sample_factory(cycle)
        if sample is None:
            return
        buffer.append(sample)
        self._state.note_sample()

    def start_recording(self, _command: RuntimeCommand | None = None) -> str:
        with self._lock:
            self._require_started()
            self.poll()
            index = self._state.start_episode()
            self._buffer = SampleBuffer(
                self._schema,
                capacity=self._sample_capacity,
            )
        return f"recording episode {index} started"

    def stop_recording(self, _command: RuntimeCommand | None = None) -> str:
        with self._lock:
            self._require_started()
            self.poll()
            try:
                index = self._state.request_finish()
            except LifecycleError as exc:
                raise CommandRejectedError(str(exc)) from exc
            assert self._buffer is not None
            episode = self._buffer.seal(index=index, task=self._task)
            self._buffer = None
            self._pending_episode = episode
            try:
                self._pending_ticket = self._worker.submit_export(episode)
            except ExportQueueFullError as exc:
                self._state.export_failed(index, str(exc))
                raise CommandRejectedError(str(exc)) from exc
        return f"episode {index} queued for export"

    def rollback_last_episode(
        self,
        _command: RuntimeCommand | None = None,
    ) -> str:
        with self._lock:
            self._require_started()
            self.poll()
            try:
                request = self._state.request_rollback()
            except LifecycleError as exc:
                raise CommandRejectedError(str(exc)) from exc
            if request.discard_active:
                self._buffer = None
                return "discarded active episode"
            if request.episode_index is None:
                raise CommandRejectedError("no completed episode to rollback")
            try:
                self._pending_ticket = self._worker.submit_rollback(
                    request.episode_index
                )
            except (ExportQueueFullError, LifecycleError) as exc:
                self._state.rollback_failed(request.episode_index, str(exc))
                raise CommandRejectedError(str(exc)) from exc
        return f"episode {request.episode_index} queued for rollback"

    def retry_failed_export(self) -> str:
        with self._lock:
            self._require_started()
            self.poll()
            if self._pending_episode is None:
                raise CommandRejectedError("no failed episode is available")
            try:
                index = self._state.retry_export()
                self._pending_ticket = self._worker.submit_export(
                    self._pending_episode
                )
            except (ExportQueueFullError, LifecycleError) as exc:
                if self._state.snapshot().state is EpisodeState.EXPORT_PENDING:
                    self._state.export_failed(index, str(exc))
                raise CommandRejectedError(str(exc)) from exc
        return f"episode {index} export retried"

    def discard_failed_export(self) -> int:
        with self._lock:
            self._require_started()
            self.poll()
            index = self._state.discard_failed_export()
            self._pending_episode = None
            return index

    def poll(self) -> None:
        with self._lock:
            ticket = self._pending_ticket
            if ticket is None or not ticket.done:
                return
            result = ticket.result(0)
            self._pending_ticket = None
            if ticket.kind is ExportTaskKind.WRITE:
                if result.succeeded:
                    self._state.export_succeeded(ticket.episode_index)
                    self._pending_episode = None
                else:
                    error = result.error or RuntimeError("recording export failed")
                    self._state.export_failed(
                        ticket.episode_index,
                        f"{type(error).__name__}: {error}",
                    )
            elif result.succeeded:
                self._state.rollback_succeeded(ticket.episode_index)
            else:
                error = result.error
                self._state.rollback_failed(
                    ticket.episode_index,
                    None
                    if error is None
                    else f"{type(error).__name__}: {error}",
                )

    def status(self) -> DataCollectionStatus:
        self.poll()
        lifecycle = self._state.snapshot()
        worker = self._worker.metrics()
        return DataCollectionStatus(
            state=lifecycle.state,
            next_episode_index=lifecycle.next_episode_index,
            active_samples=lifecycle.active_samples,
            pending_episode_index=lifecycle.pending_episode_index,
            queued_exports=worker.queued,
            export_busy=worker.busy,
            last_error=lifecycle.last_error,
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            started = self._started
        if started:
            self._worker.close(drain=True)
            self.poll()
        self._state.close()
        with self._lock:
            self._started = False
            self._closed = True

    def _require_started(self) -> None:
        if self._closed:
            raise LifecycleError("recording extension is closed")
        if not self._started:
            raise LifecycleError("recording extension has not been started")


class DataCollectionSession:
    """Recording-capable facade that delegates the one teleop loop."""

    def __init__(
        self,
        teleop: TeleopSession,
        recording: RecordingCycleExtension,
    ) -> None:
        if not isinstance(teleop, TeleopSession):
            raise ModelValidationError("teleop must be a TeleopSession")
        if not isinstance(recording, RecordingCycleExtension):
            raise ModelValidationError(
                "recording must be a RecordingCycleExtension"
            )
        if recording not in teleop.extensions:
            raise ModelValidationError(
                "recording extension must be composed into TeleopSession"
            )
        self.teleop = teleop
        self.recording = recording

    def start(self) -> None:
        self.teleop.start()

    def run(self, *, max_cycles: int | None = None) -> None:
        self.teleop.run(max_cycles=max_cycles)

    def request_stop(self) -> None:
        self.teleop.request_stop()

    def close(self) -> None:
        self.teleop.close()
