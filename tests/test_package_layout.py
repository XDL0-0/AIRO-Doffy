"""Phase 1 structural and compatibility checks."""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class PackageLayoutTest(unittest.TestCase):
    def test_base_package_import_is_side_effect_free(self) -> None:
        code = """
import sys
import airo_doffy
blocked = ("torch", "cv2", "pyrealsense2", "aiortc", "serial")
loaded = [name for name in sys.modules if name.startswith(blocked)]
assert not loaded, loaded
print(airo_doffy.__version__)
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_supported_sources_do_not_import_deprecated_code(self) -> None:
        roots = [REPO_ROOT / "src", REPO_ROOT / "main.py", REPO_ROOT / "inference.py"]
        violations: list[str] = []
        for root in roots:
            paths = root.rglob("*.py") if root.is_dir() else [root]
            for path in paths:
                tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
                for node in ast.walk(tree):
                    module = None
                    if isinstance(node, ast.ImportFrom):
                        module = node.module
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name == "deprecated" or alias.name.startswith("deprecated."):
                                violations.append(f"{path}:{node.lineno}")
                    if module == "deprecated" or (
                        module is not None and module.startswith("deprecated.")
                    ):
                        violations.append(f"{path}:{node.lineno}")
        self.assertEqual(violations, [])

    def test_main_runtime_keeps_only_ble4_tactile_backend(self) -> None:
        source = (REPO_ROOT / "main.py").read_text(encoding="utf-8-sig")
        self.assertIn("from tactile_4point import FourPointTactileBleReader", source)
        self.assertNotIn("from tactile import", source)
        self.assertTrue(
            (REPO_ROOT / "deprecated/tactile/magtouch_ilias_41taxel.py").is_file()
        )

    def test_legacy_safety_wrappers_are_importable(self) -> None:
        for module_name in (
            "dataset_tool.tag_HF",
            "test_tool.ForceMode",
            "test_tool.freedrive",
        ):
            with self.subTest(module_name=module_name):
                importlib.import_module(module_name)

    def test_robot_backend_facade_defers_legacy_sdk_imports(self) -> None:
        code = """
import sys
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import robot_backend
blocked = ("numpy", "scipy", "airo_robots", "ur_analytic_ik")
loaded = [name for name in sys.modules if name.startswith(blocked)]
assert not loaded, loaded
assert "airo_doffy.robots.legacy" not in sys.modules
assert "make_robot_backend" in robot_backend.__all__
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_visualizer_rollback_uses_typed_runtime_command(self) -> None:
        visualizer = (REPO_ROOT / "visualizer.py").read_text(encoding="utf-8-sig")
        runtime = (REPO_ROOT / "main.py").read_text(encoding="utf-8-sig")
        self.assertIn("RuntimeCommand(", visualizer)
        self.assertIn(
            "command.kind is RuntimeCommandType.ROLLBACK_LAST_EPISODE",
            runtime,
        )


if __name__ == "__main__":
    unittest.main()
