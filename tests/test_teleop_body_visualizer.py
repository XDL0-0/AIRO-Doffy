"""Launcher lifecycle checks that never construct a robot runtime."""

from __future__ import annotations

from pathlib import Path
import signal
import subprocess
import sys
from types import ModuleType

import pytest

from doffy_teleop.runtime import body_visualizer as launcher


ROOT_ENTRY = Path(__file__).resolve().parents[1] / "teleop_body_visualizer.py"


class FakeProcess:
    def __init__(self, *, returncode=None, timeouts=0):
        self.returncode = returncode
        self.timeouts = timeouts
        self.events = []

    def poll(self):
        return self.returncode

    def send_signal(self, signum):
        self.events.append(("signal", signum))

    def terminate(self):
        self.events.append(("terminate",))

    def kill(self):
        self.events.append(("kill",))

    def wait(self, timeout):
        self.events.append(("wait", timeout))
        if self.timeouts:
            self.timeouts -= 1
            raise subprocess.TimeoutExpired("fake teleoperation", timeout)
        self.returncode = 0
        return self.returncode


def _visualizer(monkeypatch, run):
    module = ModuleType("doffy_teleop.runtime.body_viewer")
    module.run_visualizer = run
    monkeypatch.setitem(sys.modules, module.__name__, module)


@pytest.mark.parametrize("module_entry", [False, True])
def test_help_without_site_packages(tmp_path, module_entry):
    entry = ["-m", "doffy_teleop.runtime.body_visualizer"] if module_entry else [str(ROOT_ENTRY)]
    result = subprocess.run(
        [sys.executable, "-S", *entry, "--help"],
        cwd=ROOT_ENTRY.parent if module_entry else tmp_path,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0
    assert "--visualization-only" in result.stdout
    assert "--teleop {realman,classic}" in result.stdout
    assert "default: visualization only" in " ".join(result.stdout.split())
    assert "8015" in result.stdout
    assert result.stderr == ""


def test_visualizer_preflight_precedes_robot_and_close_interrupts_child(monkeypatch):
    child = FakeProcess()
    events = []
    calls = []

    def popen(command, **kwargs):
        events.append("robot-started")
        calls.append((command, kwargs))
        return child

    def run(**options):
        events.append("preflight-complete")
        options["on_ready"]()
        assert options["should_stop"]() is False

    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    _visualizer(monkeypatch, run)
    assert launcher.main(["--teleop", "realman"]) == 0
    assert events == ["preflight-complete", "robot-started"]
    assert calls == [(
        [sys.executable, str(launcher.REPOSITORY_ROOT / "realman_teleop.py")],
        {"cwd": str(launcher.REPOSITORY_ROOT), "start_new_session": True},
    )]
    assert child.events == [
        ("signal", signal.SIGINT), ("wait", launcher.INTERRUPT_TIMEOUT)
    ]


def test_preflight_failure_does_not_launch_robot(monkeypatch, capsys):
    def run(**options):
        raise OSError("body UDP port is already in use")

    def forbidden_popen(*args, **kwargs):
        pytest.fail("robot must not start before visualization preflight")

    monkeypatch.setattr(launcher.subprocess, "Popen", forbidden_popen)
    _visualizer(monkeypatch, run)
    assert launcher.main(["--teleop", "realman"]) == 1
    assert "already in use" in capsys.readouterr().err


@pytest.mark.parametrize("args", [
    [],
    ["--visualization-only"],
    ["--demo"],
    ["--save-preview", "body.png"],
    ["--demo", "--save-preview", "body.png"],
    ["--teleop", "realman", "--demo"],
    ["--teleop", "realman", "--save-preview", "body.png"],
    ["--teleop", "classic", "--demo"],
    ["--teleop", "classic", "--save-preview", "body.png"],
])
def test_read_only_paths_never_launch_robot(monkeypatch, args):
    captured = []

    def run(**options):
        captured.append(options)
        options["on_ready"]()
        assert options["should_stop"]() is False

    def forbidden_popen(*args, **kwargs):
        pytest.fail("this visualization mode must not start a robot")

    monkeypatch.setattr(launcher.subprocess, "Popen", forbidden_popen)
    _visualizer(monkeypatch, run)
    assert launcher.main(args) == 0
    assert captured[0]["demo"] == ("--demo" in args)
    assert captured[0]["bind_ip"] == "0.0.0.0"
    assert captured[0]["port"] == 8015
    if "--save-preview" in args:
        assert captured[0]["save_preview"] == "body.png"


def test_classic_and_receiver_options_are_forwarded(monkeypatch):
    child = FakeProcess()
    commands = []
    captured = []

    def popen(command, **kwargs):
        commands.append(command)
        return child

    def run(**options):
        captured.append(options)
        options["on_ready"]()

    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    _visualizer(monkeypatch, run)
    assert launcher.main([
        "--teleop", "classic", "--bind-ip", "127.0.0.1", "--body-port", "8123",
        "--hz", "20", "--stale-after", "1.5",
    ]) == 0
    assert commands == [[sys.executable, str(launcher.REPOSITORY_ROOT / "main.py")]]
    assert captured[0]["bind_ip"] == "127.0.0.1"
    assert captured[0]["port"] == 8123
    assert captured[0]["hz"] == 20.0
    assert captured[0]["stale_after"] == 1.5


@pytest.mark.parametrize("returncode, expected", [(17, 17), (-9, 137), (0, 0)])
def test_early_child_exit_closes_visualizer_and_propagates_status(
    monkeypatch, capsys, returncode, expected
):
    child = FakeProcess(returncode=returncode)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **kw: child)

    def run(**options):
        options["on_ready"]()
        assert options["should_stop"]() is True
        assert options["should_stop"]() is True

    _visualizer(monkeypatch, run)
    assert launcher.main(["--teleop", "realman"]) == expected
    assert child.events == []
    assert capsys.readouterr().err.count("Teleoperation exited") == 1


def test_ctrl_c_interrupts_child_and_returns_130(monkeypatch):
    child = FakeProcess()
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **kw: child)

    def run(**options):
        options["on_ready"]()
        raise KeyboardInterrupt

    _visualizer(monkeypatch, run)
    assert launcher.main(["--teleop", "realman"]) == 130
    assert child.events[0] == ("signal", signal.SIGINT)


def test_visualizer_error_still_interrupts_child(monkeypatch, capsys):
    child = FakeProcess()
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **kw: child)

    def run(**options):
        options["on_ready"]()
        raise RuntimeError("renderer failed")

    _visualizer(monkeypatch, run)
    assert launcher.main(["--teleop", "realman"]) == 1
    assert child.events[0] == ("signal", signal.SIGINT)
    assert "renderer failed" in capsys.readouterr().err


def test_second_ctrl_c_escalates_to_kill_without_leaving_traceback(monkeypatch):
    child = FakeProcess()
    original_wait = child.wait

    def interrupted_wait(timeout):
        child.wait = original_wait
        raise KeyboardInterrupt

    child.wait = interrupted_wait
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **kw: child)
    _visualizer(monkeypatch, lambda **options: options["on_ready"]())
    assert launcher.main(["--teleop", "realman"]) == 130
    assert child.events == [
        ("signal", signal.SIGINT), ("kill",), ("wait", launcher.KILL_TIMEOUT)
    ]


def test_failed_child_start_is_reported(monkeypatch, capsys):
    def popen(*args, **kwargs):
        raise OSError("interpreter unavailable")

    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    _visualizer(monkeypatch, lambda **options: options["on_ready"]())
    assert launcher.main(["--teleop", "realman"]) == 1
    assert "interpreter unavailable" in capsys.readouterr().err


def test_shutdown_escalates_only_after_timeouts():
    child = FakeProcess(timeouts=2)
    launcher._shutdown_teleop(child)
    assert child.events == [
        ("signal", signal.SIGINT), ("wait", launcher.INTERRUPT_TIMEOUT),
        ("terminate",), ("wait", launcher.TERMINATE_TIMEOUT),
        ("kill",), ("wait", launcher.KILL_TIMEOUT),
    ]


def test_unresponsive_child_cleanup_returns_failure(monkeypatch, capsys):
    child = FakeProcess(timeouts=3)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **kw: child)
    _visualizer(monkeypatch, lambda **options: options["on_ready"]())
    assert launcher.main(["--teleop", "realman"]) == 1
    assert "did not stop after SIGKILL" in capsys.readouterr().err


@pytest.mark.parametrize("args", [
    ["--body-port", "0"], ["--body-port", "65536"],
    ["--hz", "0"], ["--hz", "nan"], ["--hz", "inf"],
    ["--stale-after", "-1"],
    ["--visualization-only", "--teleop", "realman"],
    ["--teleop", "classic", "--visualization-only"],
])
def test_invalid_options_fail_before_visualizer_import(args):
    with pytest.raises(SystemExit) as exc:
        launcher.main(args)
    assert exc.value.code == 2
