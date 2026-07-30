"""Recording-specific failure types."""

from ..core.errors import AiroDoffyError


class RecordingError(AiroDoffyError, RuntimeError):
    """Base class for recording lifecycle and persistence failures."""


class RecordingSchemaMismatchError(RecordingError):
    """An existing dataset does not match the configured schema."""


class ExportQueueFullError(RecordingError):
    """A bounded export queue rejected new work."""
