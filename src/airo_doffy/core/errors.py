"""Shared exception hierarchy for hardware-independent package contracts."""


class AiroDoffyError(Exception):
    """Base class for errors raised by the v2 package."""


class ModelValidationError(AiroDoffyError, ValueError):
    """A domain value violates a shape, range, or immutability contract."""


class BufferClosedError(AiroDoffyError, RuntimeError):
    """A producer attempted to publish after a latest-value buffer closed."""


class LifecycleError(AiroDoffyError, RuntimeError):
    """A component lifecycle transition is invalid or failed."""


class OptionalDependencyError(AiroDoffyError, ImportError):
    """An explicitly selected adapter is missing its optional dependency."""
