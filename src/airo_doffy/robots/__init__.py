"""Robot capabilities, hardware adapters, and command executors."""

from .base import RobotBackend
from .executor import ExecutorSnapshot, LatestActionExecutor
from .factory import create_gripper, create_robot_backend
from .mock import InjectedRobotError, MockRobotBackend
from .realman import RealManRobotBackend, create_realman_backend
from .realman_executor import RealManCanfdExecutor, RealManExecutorSnapshot
from .ur import URRobotBackend, create_ur_backend

__all__ = [
    "ExecutorSnapshot",
    "InjectedRobotError",
    "LatestActionExecutor",
    "MockRobotBackend",
    "RealManRobotBackend",
    "RealManCanfdExecutor",
    "RealManExecutorSnapshot",
    "RobotBackend",
    "URRobotBackend",
    "create_realman_backend",
    "create_gripper",
    "create_robot_backend",
    "create_ur_backend",
]
