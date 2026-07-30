"""Latest-only threaded visualization consumer."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Lock, Thread

from ..core.buffers import LatestValueBuffer
from ..core.errors import BufferClosedError, LifecycleError, ModelValidationError
from .base import SnapshotRenderer
from .models import VisualizationSnapshot


@dataclass(frozen=True, slots=True)
class VisualizationMetrics:
    """Small immutable consumer health snapshot."""

    running: bool
    accepting: bool
    published: int
    rejected: int
    rendered: int
    renderer_closed: bool
    last_error: str | None


class TypedSnapshotConsumer:
    """Render only the newest typed snapshot on a dedicated thread."""

    def __init__(
        self,
        renderer: SnapshotRenderer,
        *,
        wait_timeout_s: float = 0.1,
        thread_name: str = "visualization-consumer",
    ) -> None:
        if wait_timeout_s <= 0:
            raise ModelValidationError("wait_timeout_s must be positive")
        if not isinstance(thread_name, str) or not thread_name:
            raise ModelValidationError("thread_name must be a non-empty string")
        self._renderer = renderer
        self._wait_timeout_s = float(wait_timeout_s)
        self._thread_name = thread_name
        self._latest: LatestValueBuffer[VisualizationSnapshot] = (
            LatestValueBuffer()
        )
        self._stop = Event()
        self._lock = Lock()
        self._thread: Thread | None = None
        self._accepting = False
        self._published = 0
        self._rejected = 0
        self._rendered = 0
        self._renderer_closed = False
        self._last_error: str | None = None

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                raise LifecycleError("visualization consumer is already started")
            if self._stop.is_set():
                raise LifecycleError("closed visualization consumer cannot restart")
            self._accepting = True
            self._thread = Thread(
                target=self._run,
                name=self._thread_name,
                daemon=True,
            )
            self._thread.start()

    def publish(self, snapshot: VisualizationSnapshot) -> bool:
        if not isinstance(snapshot, VisualizationSnapshot):
            raise ModelValidationError(
                "visualization consumer accepts only VisualizationSnapshot"
            )
        with self._lock:
            accepting = self._accepting
        if not accepting:
            return False
        try:
            accepted = self._latest.publish(snapshot)
        except BufferClosedError:
            return False
        with self._lock:
            if accepted:
                self._published += 1
            else:
                self._rejected += 1
        return accepted

    def metrics(self) -> VisualizationMetrics:
        with self._lock:
            thread = self._thread
            return VisualizationMetrics(
                running=thread is not None and thread.is_alive(),
                accepting=self._accepting,
                published=self._published,
                rejected=self._rejected,
                rendered=self._rendered,
                renderer_closed=self._renderer_closed,
                last_error=self._last_error,
            )

    def close(self, *, timeout_s: float = 2.0) -> None:
        if timeout_s < 0:
            raise ModelValidationError("timeout_s must be non-negative")
        with self._lock:
            thread = self._thread
            self._accepting = False
            self._stop.set()
            self._latest.close()
        if thread is None:
            self._close_renderer()
            return
        thread.join(timeout_s)
        if thread.is_alive():
            raise LifecycleError("visualization consumer did not stop before timeout")
        with self._lock:
            self._thread = None

    def _run(self) -> None:
        last_sequence: int | None = None
        try:
            self._renderer.start()
            while not self._stop.is_set():
                snapshot = self._latest.wait_for_new(
                    after_sequence=last_sequence,
                    timeout=self._wait_timeout_s,
                )
                if snapshot is None:
                    continue
                keep_open = self._renderer.render(snapshot)
                last_sequence = snapshot.sequence
                with self._lock:
                    self._rendered += 1
                if not keep_open:
                    with self._lock:
                        self._renderer_closed = True
                    break
        except BaseException as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                self._accepting = False
            self._stop.set()
            self._latest.close()
            self._close_renderer()

    def _close_renderer(self) -> None:
        try:
            self._renderer.close()
        except BaseException as exc:
            with self._lock:
                if self._last_error is None:
                    self._last_error = f"{type(exc).__name__}: {exc}"
