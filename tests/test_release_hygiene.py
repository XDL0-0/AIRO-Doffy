"""Offline regression checks; no robot SDK, Unity installation or network needed."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

from scripts.check_release_hygiene import check_text, scan_repository

ROOT = Path(__file__).resolve().parents[1]


class ReleaseHygieneTests(unittest.TestCase):
    def test_home_paths_on_linux_macos_and_windows(self):
        samples = ["/" + "home/operator/work", "/" + "Users/operator/work",
                   "C:" + "\\Users\\operator\\work", "C:" + "/" + "Users/operator/work"]
        for text in samples:
            with self.subTest(text=text):
                self.assertTrue(check_text("new.py", text))

    def test_site_addresses_and_reviewed_exception_scope(self):
        for octets in [(10, 5, 4, 3), (172, 20, 4, 3), (192, 168, 2, 40)]:
            self.assertTrue(check_text("new.md", ".".join(map(str, octets))))
        robot = ".".join(map(str, (192, 168, 1, 18)))
        self.assertFalse(check_text("doffy_teleop/config.py", robot))
        self.assertTrue(check_text("unreviewed.py", robot))

    def test_device_ids_without_republishing_real_identifiers(self):
        for sample in ["2G0" + "A1B2C3D4E5F", "serial `" + "123456789012`",
                       '"serial_number": "' + 'ABCD12345678"', "adb -s " + "ABC123456789"]:
            self.assertTrue(check_text("validation.md", sample), sample)

    def test_documentation_addresses_hashes_and_redactions_are_allowed(self):
        sample = ('192.0.2.10 198.51.100.20 203.0.113.30 127.0.0.1 0.0.0.0 '
                  '/path/to/unity ./datasets/example $HOME/work '
                  'Quest 3 `QUEST_SERIAL_REDACTED` serial `REALSENSE_SERIAL_REDACTED` '
                  'sha256 ' + 'a1b2c3d4' * 8)
        self.assertEqual(check_text("example.md", sample), [])

    def test_scans_tracked_text_including_dotfiles_but_not_local_ignored_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / ".gitignore").write_text(".env\n")
            bad = "/" + "home/operator/work"
            (root / ".env").write_text(bad)
            (root / ".tracked-settings").write_text(bad)
            (root / "binary.bin").write_bytes(b"\0" + bad.encode())
            subprocess.run(["git", "add", ".gitignore", ".tracked-settings", "binary.bin"], cwd=root, check=True)
            count, findings = scan_repository(root)
            self.assertEqual(count, 2)
            self.assertEqual(len(findings), 1)
            self.assertIn(".tracked-settings:1", findings[0])

    def test_cli_paths_are_required_before_any_external_work(self):
        runners = [
            "scripts/teleop_refactor/compile_unity_sources.py",
            "scripts/teleop_refactor/audit_unity_project.py",
            "scripts/teleop_refactor/run_csharp_protocol_checks.py",
            "scripts/teleop_refactor/run_csharp_session_checks.py",
            "scripts/teleop_refactor/run_csharp_recording_checks.py",
            "scripts/teleop_refactor/run_bracelet_drag_checks.py",
            "scripts/teleop_refactor/run_wrist_detents_checks.py",
            "scripts/teleop_refactor/run_wrist_visibility_checks.py",
            "scripts/teleop_refactor/organize_unity_sources.py",
            "docs/body_visualization/quest/check_body_tracking_quality.py",
            "docs/body_visualization/quest/check_body_wire_format.py",
        ]
        env = {key: value for key, value in os.environ.items() if not key.startswith("DOFFY_")}
        for runner in runners:
            with self.subTest(runner=runner):
                result = subprocess.run([sys.executable, str(ROOT / runner)], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("required", result.stderr)
                help_result = subprocess.run([sys.executable, str(ROOT / runner), "--help"], env=env, capture_output=True, text=True)
                self.assertEqual(help_result.returncode, 0, help_result.stderr)

    def test_cli_overrides_environment_and_resolves_relative_paths(self):
        # The audit reads the selected project; an empty fixture must fail there,
        # never report a successful audit or fall back to a workstation project.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("cli-project", "env-project", "packages"):
                (root / name).mkdir()
            env = {**os.environ, "DOFFY_UNITY_PROJECT": str(root / "env-project"),
                   "DOFFY_UNITY_PACKAGE_CACHE": str(root / "packages")}
            command = [sys.executable, str(ROOT / "scripts/teleop_refactor/audit_unity_project.py")]
            result = subprocess.run(command + ["--project", "cli-project"], cwd=root, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(str(root / "cli-project/ProjectSettings/EditorBuildSettings.asset"), result.stderr)
            result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(str(root / "env-project/ProjectSettings/EditorBuildSettings.asset"), result.stderr)

    def test_repository_is_clean(self):
        count, findings = scan_repository(ROOT)
        self.assertGreater(count, 100)
        self.assertEqual(findings, [])


if __name__ == "__main__":
    unittest.main()
