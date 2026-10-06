"""Static architecture guards for the PC teleop module split."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

IMPLEMENTATION_MODULES = (
    "doffy_teleop/control/canfd_loop.py",
    "doffy_teleop/control/canfd_setpoints.py",
    "doffy_teleop/control/contracts.py",
    "doffy_teleop/control/input_mapping.py",
    "doffy_teleop/control/realman_qp.py",
    "doffy_teleop/control/target_policy.py",
    "doffy_teleop/protocol/body.py",
    "doffy_teleop/media/body.py",
    "doffy_teleop/visualization/body_geometry.py",
    "doffy_teleop/visualization/body.py",
    "doffy_teleop/visualization/body_demo.py",
    "doffy_teleop/visualization/body_pose.py",
    "doffy_teleop/runtime/body_viewer.py",
    "doffy_teleop/runtime/body_visualizer.py",
    "doffy_teleop/runtime/classic.py",
    "doffy_teleop/runtime/classic_config.py",
    "doffy_teleop/runtime/classic_helpers.py",
    "doffy_teleop/runtime/publisher.py",
    "doffy_teleop/runtime/realman.py",
    "doffy_teleop/runtime/realman_entrypoint.py",
    "doffy_teleop/runtime/state.py",
    "doffy_teleop/robots/backends.py",
    "doffy_teleop/robots/contracts.py",
    "doffy_teleop/robots/factory.py",
    "doffy_teleop/robots/gripper.py",
    "doffy_teleop/robots/legacy/control.py",
    "doffy_teleop/robots/legacy/hand.py",
    "doffy_teleop/robots/legacy/runtime.py",
    "doffy_teleop/robots/legacy/state.py",
    "doffy_teleop/robots/legacy/step.py",
)


def test_implementation_modules_stay_within_reviewable_size() -> None:
    sizes = {
        path: len((ROOT / path).read_text().splitlines())
        for path in IMPLEMENTATION_MODULES
    }
    oversized = {path: size for path, size in sizes.items() if size > 500}
    assert not oversized, f"implementation modules exceed 500 lines: {oversized}"


ROOT_LAUNCHERS = {
    "main.py", "realman_teleop.py", "realman_teachcollect.py",
    "realman_recollect.py", "teleop_body_visualizer.py",
}

# These ignored launchers can remain in a developer's complete workspace.
# A public source checkout contains only ROOT_LAUNCHERS.
LOCAL_ONLY_ROOT_LAUNCHERS = {
    "inference.py", "eval_policy.py", "seahorse_teleop.py",
}


def test_root_entries_are_facades() -> None:
    root_files = {path.name for path in ROOT.glob("*.py")}
    assert ROOT_LAUNCHERS <= root_files, f"missing public launchers: {ROOT_LAUNCHERS - root_files}"
    unexpected = root_files - ROOT_LAUNCHERS - LOCAL_ONLY_ROOT_LAUNCHERS
    assert not unexpected, f"unexpected root Python modules: {unexpected}"
    for name in ROOT_LAUNCHERS:
        source = (ROOT / name).read_text()
        assert len(source.splitlines()) <= 20
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            for node in ast.walk(ast.parse(source))
        ), f"implementation must live inside doffy_teleop: {name}"


def test_package_does_not_import_root_modules() -> None:
    forbidden = {
        "config", "utils", "visualizer", "visualizer_config", "dataset",
        "data_recording", "data_schema", "brainco_hand", "beaver", "beaver_reader",
        "tactile", "tactile_4point", "force_filter", "realsense_camera", "wrm_akm",
        "parse_vr", "udp_comms", "udp", "WebRTC_udp", "robot_backend",
        "robot_teleop", "ur_teleop", "main", "realman_teleop", "realman_teachcollect",
        "realman_recollect", "eval_policy", "eval_config", "inference", "seahorse_teleop",
        "dataset_tool", "test_tool",
    }
    for path in (ROOT / "doffy_teleop").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                assert node.module.split(".")[0] not in forbidden, (path, node.module)
            elif isinstance(node, ast.Import):
                assert not forbidden.intersection(alias.name.split(".")[0] for alias in node.names), path


def test_body_root_entry_is_a_thin_launcher() -> None:
    path = ROOT / "teleop_body_visualizer.py"
    source = path.read_text()
    assert len(source.splitlines()) <= 20
    tree = ast.parse(source)
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in ast.walk(tree)
    ), "BODY implementations belong in doffy_teleop, not the root launcher"
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "doffy_teleop.runtime.body_visualizer"
        and any(alias.name == "main" for alias in node.names)
        for node in tree.body
    ), "the root BODY launcher must import the packaged entry point"


def test_body_compatibility_facade_contains_only_exports() -> None:
    path = ROOT / "doffy_teleop/body_visualization.py"
    for node in ast.parse(path.read_text()).body:
        is_docstring = (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
        is_export_list = (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "__all__"
            and isinstance(node.value, (ast.Tuple, ast.List))
            and all(
                isinstance(item, ast.Constant) and isinstance(item.value, str)
                for item in node.value.elts
            )
        )
        assert is_docstring or isinstance(node, ast.ImportFrom) or is_export_list, (
            "BODY compatibility facade must only re-export packaged symbols",
            ast.dump(node),
        )


def test_body_runtime_and_visualization_do_not_import_old_entries() -> None:
    forbidden = {"doffy_teleop.body_visualization", "teleop_body_visualizer"}
    for directory in ("doffy_teleop/runtime", "doffy_teleop/visualization"):
        for path in (ROOT / directory).rglob("*.py"):
            package = path.relative_to(ROOT).with_suffix("").parts[:-1]
            for node in ast.walk(ast.parse(path.read_text())):
                imported: set[str] = set()
                if isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    prefix = package[:len(package) - node.level + 1] if node.level else ()
                    suffix = tuple(node.module.split(".")) if node.module else ()
                    module = ".".join(prefix + suffix)
                    imported.add(module)
                    imported.update(
                        ".".join((module, alias.name)) for alias in node.names
                        if alias.name != "*"
                    )
                assert not forbidden.intersection(imported), (path, imported)



def test_robot_implementation_does_not_depend_on_compatibility_facades() -> None:
    """Facade imports must not create cycles in the reusable robot layer."""
    forbidden = {"realman_teleop", "robot_backend", "robot_teleop", "main"}
    for path in (ROOT / "doffy_teleop/robots").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                assert node.module not in forbidden, (path, node.module)
            elif isinstance(node, ast.Import):
                assert not forbidden.intersection(alias.name for alias in node.names), path
