"""Shared CLI/config/session handoff for thin applications."""

from __future__ import annotations

import argparse
import importlib
import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..config import AiroDoffyConfig, load_config
from ..core.errors import ModelValidationError, OptionalDependencyError

logger = logging.getLogger(__name__)

SessionFactory = Callable[[AiroDoffyConfig], "ApplicationSession"]


@runtime_checkable
class ApplicationSession(Protocol):
    """Lifecycle/run surface shared by teleop and data collection sessions."""

    def start(self) -> None:
        """Start owned resources."""

    def run(self) -> None:
        """Run until stopped or interrupted."""

    def request_stop(self) -> None:
        """Request cooperative loop shutdown."""

    def close(self) -> None:
        """Close resources idempotently."""


def build_parser(*, mode: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"airo-doffy-{mode}",
        description=f"Run the AIRO-Doffy {mode} session",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="base YAML configuration",
    )
    parser.add_argument("--robot-config", help="optional robot YAML layer")
    parser.add_argument("--experiment-config", help="optional experiment YAML layer")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.FIELD=VALUE",
        help="repeatable highest-precedence config override",
    )
    parser.add_argument(
        "--session-factory",
        default=None,
        metavar="MODULE:SYMBOL",
        help=(
            "composition factory; defaults to "
            f"AIRO_DOFFY_{mode.upper()}_SESSION_FACTORY"
        ),
    )
    return parser


def parse_overrides(values: Sequence[str]) -> dict[str, str]:
    result = {}
    for raw in values:
        name, separator, value = raw.partition("=")
        if not separator or not name or "." not in name:
            raise ModelValidationError(
                f"override must use SECTION.FIELD=VALUE syntax: {raw!r}"
            )
        if name in result:
            raise ModelValidationError(f"duplicate CLI override: {name}")
        result[name] = value
    return result


def resolve_session_factory(target: str) -> SessionFactory:
    module_name, separator, symbol = target.partition(":")
    if not separator or not module_name or not symbol:
        raise ModelValidationError(
            f"session factory must use module:symbol syntax: {target!r}"
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise OptionalDependencyError(
            f"cannot import session factory module {module_name!r}"
        ) from exc
    try:
        factory = getattr(module, symbol)
    except AttributeError as exc:
        raise ModelValidationError(
            f"session factory target does not exist: {target!r}"
        ) from exc
    if not callable(factory):
        raise ModelValidationError(f"session factory is not callable: {target!r}")
    return factory


def run_session(session: ApplicationSession) -> int:
    if not isinstance(session, ApplicationSession):
        raise ModelValidationError(
            "session factory result must satisfy ApplicationSession"
        )
    try:
        session.start()
        session.run()
    except KeyboardInterrupt:
        session.request_stop()
    finally:
        session.close()
    return 0


def run_application(
    *,
    mode: str,
    argv: Sequence[str] | None = None,
    environment: dict[str, str] | None = None,
) -> int:
    parser = build_parser(mode=mode)
    args = parser.parse_args(argv)
    source = os.environ if environment is None else environment
    target = args.session_factory or source.get(
        f"AIRO_DOFFY_{mode.upper()}_SESSION_FACTORY"
    )
    if target is None:
        parser.error(
            "--session-factory or the matching AIRO_DOFFY_*_SESSION_FACTORY "
            "environment variable is required"
        )
    config = load_config(
        Path(args.config),
        robot_path=args.robot_config,
        experiment_path=args.experiment_config,
        environment=source,
        cli_overrides=parse_overrides(args.set),
    )
    factory = resolve_session_factory(target)
    return run_session(factory(config))


def entrypoint(*, mode: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        code = run_application(mode=mode)
    except Exception:
        logger.exception("%s session failed", mode)
        code = 1
    raise SystemExit(code)
