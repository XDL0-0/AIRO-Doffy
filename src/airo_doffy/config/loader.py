"""Layered configuration loading without importing hardware dependencies."""

from __future__ import annotations

import copy
import dataclasses
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..core.errors import ModelValidationError, OptionalDependencyError
from .models import (
    AiroDoffyConfig,
    CameraConfig,
    CommandTransportConfig,
    NetworkConfig,
    RecordingConfig,
    RobotConfig,
    RuntimeConfig,
    StateTransportConfig,
    TactileConfig,
    TeleopConfig,
    VideoStreamingConfig,
    VisualizationConfig,
    VRConfig,
    WrenchConfig,
)

ENV_PREFIX = "AIRO_DOFFY__"

_SECTION_TYPES = {
    "network": NetworkConfig,
    "robot": RobotConfig,
    "camera": CameraConfig,
    "vr": VRConfig,
    "teleop": TeleopConfig,
    "tactile": TactileConfig,
    "recording": RecordingConfig,
    "visualization": VisualizationConfig,
    "video": VideoStreamingConfig,
    "state_transport": StateTransportConfig,
    "command_transport": CommandTransportConfig,
    "wrench": WrenchConfig,
    "runtime": RuntimeConfig,
}


def deep_merge(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a recursive mapping merge without mutating either input."""

    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        current = result.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge(current, value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML mapping, using stdlib JSON for JSON-compatible YAML."""

    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise OptionalDependencyError(
                "non-JSON YAML requires the 'config' optional dependency: "
                "pip install 'airo-doffy[config]'"
            ) from exc
        try:
            loaded = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ModelValidationError(f"invalid YAML in {config_path}: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise ModelValidationError(f"configuration root in {config_path} must be a mapping")
    return copy.deepcopy(dict(loaded))


def _parse_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def environment_overrides(
    environment: Mapping[str, str] | None = None,
    *,
    prefix: str = ENV_PREFIX,
) -> dict[str, Any]:
    """Translate ``AIRO_DOFFY__SECTION__FIELD`` variables to a nested mapping."""

    source = os.environ if environment is None else environment
    result: dict[str, Any] = {}
    for name, value in source.items():
        if not name.startswith(prefix):
            continue
        parts = name[len(prefix) :].lower().split("__")
        if len(parts) != 2 or not all(parts):
            raise ModelValidationError(
                f"environment override {name!r} must be {prefix}SECTION__FIELD"
            )
        section, field_name = parts
        result.setdefault(section, {})[field_name] = _parse_value(value)
    return result


def cli_override_mapping(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    """Translate dotted ``section.field`` CLI keys to a nested mapping."""

    result: dict[str, Any] = {}
    for name, value in (overrides or {}).items():
        parts = name.lower().split(".")
        if len(parts) != 2 or not all(parts):
            raise ModelValidationError(f"CLI override {name!r} must be section.field")
        section, field_name = parts
        result.setdefault(section, {})[field_name] = _parse_value(value)
    return result


def config_from_mapping(values: Mapping[str, Any]) -> AiroDoffyConfig:
    """Validate a nested mapping and create immutable section models."""

    unknown_sections = set(values) - set(_SECTION_TYPES)
    if unknown_sections:
        names = ", ".join(sorted(unknown_sections))
        raise ModelValidationError(f"unknown configuration section(s): {names}")

    sections: dict[str, Any] = {}
    for name, section_type in _SECTION_TYPES.items():
        raw_section = values.get(name, {})
        if not isinstance(raw_section, Mapping):
            raise ModelValidationError(f"configuration section {name!r} must be a mapping")
        field_names = {field.name for field in dataclasses.fields(section_type)}
        unknown_fields = set(raw_section) - field_names
        if unknown_fields:
            fields = ", ".join(sorted(unknown_fields))
            raise ModelValidationError(f"unknown field(s) in section {name!r}: {fields}")
        try:
            sections[name] = section_type(**dict(raw_section))
        except TypeError as exc:
            raise ModelValidationError(f"invalid values in section {name!r}: {exc}") from exc
    return AiroDoffyConfig(**sections)


def config_to_mapping(config: AiroDoffyConfig) -> dict[str, Any]:
    """Convert a typed configuration into serialization-friendly containers."""

    return dataclasses.asdict(config)


def load_config(
    default_path: str | Path,
    *,
    robot_path: str | Path | None = None,
    experiment_path: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
) -> AiroDoffyConfig:
    """Load default, robot, experiment, environment, then CLI layers."""

    merged: dict[str, Any] = read_yaml(default_path)
    for path in (robot_path, experiment_path):
        if path is not None:
            merged = deep_merge(merged, read_yaml(path))
    merged = deep_merge(merged, environment_overrides(environment))
    merged = deep_merge(merged, cli_override_mapping(cli_overrides))
    return config_from_mapping(merged)
