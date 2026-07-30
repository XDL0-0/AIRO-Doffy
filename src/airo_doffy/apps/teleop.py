"""Thin teleoperation application entry point."""

from .common import entrypoint as _entrypoint


def entrypoint() -> None:
    _entrypoint(mode="teleop")


if __name__ == "__main__":
    entrypoint()
