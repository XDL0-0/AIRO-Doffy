"""Root launchers keep imports and patch points on the packaged implementation."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


ROOT_IMPLEMENTATIONS = (
    ("main", "doffy_teleop.runtime.classic_entrypoint"),
    ("realman_teleop", "doffy_teleop.runtime.realman_cli"),
    ("realman_teachcollect", "doffy_teleop.runtime.teachcollect"),
    ("realman_recollect", "doffy_teleop.runtime.recollect"),
)


@pytest.mark.parametrize("root_name,canonical_name", ROOT_IMPLEMENTATIONS)
def test_root_import_and_patch_share_the_implementation(
    root_name, canonical_name, monkeypatch,
):
    root = importlib.import_module(root_name)
    implementation = importlib.import_module(canonical_name)

    assert root is implementation
    assert root.main.__globals__ is implementation.__dict__

    dispatch = Mock(return_value=17)
    monkeypatch.setattr(root, "main", dispatch)
    assert implementation.main() == 17
    dispatch.assert_called_once_with()


def test_classic_root_config_patch_reaches_the_helper(monkeypatch):
    root = importlib.import_module("main")
    implementation = importlib.import_module("doffy_teleop.runtime.classic_entrypoint")
    monkeypatch.setattr(
        root, "cfg", SimpleNamespace(CONTROLLER_RESET_TRIGGER_THRESHOLD=0.8),
    )

    assert not implementation.controller_reset_requested(
        [None, {"Joystick_Press": True, "IndexTrigger": 0.7}],
    )
    assert implementation.controller_reset_requested(
        [None, {"Joystick_Press": True, "IndexTrigger": 0.8}],
    )


def test_realman_root_factory_patch_reaches_composition(monkeypatch):
    root = importlib.import_module("realman_teleop")
    implementation = importlib.import_module("doffy_teleop.runtime.realman_cli")
    entrypoint = importlib.import_module("doffy_teleop.runtime.realman_entrypoint")
    factory = object()
    dispatch = Mock()
    monkeypatch.setattr(root, "Config", factory)
    monkeypatch.setattr(entrypoint, "main", dispatch)

    implementation.main()

    dispatch.assert_called_once()
    assert dispatch.call_args.kwargs["config_factory"] is factory


def test_body_root_launcher_uses_the_packaged_entrypoint():
    root = importlib.import_module("teleop_body_visualizer")
    implementation = importlib.import_module("doffy_teleop.runtime.body_visualizer")

    assert root.main is implementation.main
