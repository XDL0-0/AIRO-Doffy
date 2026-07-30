"""Robot capabilities, hardware adapters, and command executors."""

from .base import RobotBackend
from .executor import ExecutorSnapshot, LatestActionExecutor
from .mock import InjectedRobotError, MockRobotBackend

__all__ = [
    "ExecutorSnapshot",
    "InjectedRobotError",
    "LatestActionExecutor",
    "MockRobotBackend",
    "RobotBackend",
]
