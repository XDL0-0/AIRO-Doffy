"""Compile and run the Unity-free teleop session coordinators under Mono."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


def find_source(project: Path, filename: str) -> Path:
    """Find a production source even when Unity assets have been reorganized."""
    candidates = sorted(project.rglob(filename))
    if not candidates:
        raise FileNotFoundError(f"could not find {filename} below {project}")

    # Prefer the refactored Core copy when an old backup or compatibility copy
    # is present elsewhere in the Unity project.
    core = [path for path in candidates if "Teleop" in path.parts and "Core" in path.parts]
    return core[0] if core else candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("/home/yuyuan/UNITY_Project/Codex"),
        help="Unity project root containing Assets/",
    )
    parser.add_argument(
        "--mono-bin",
        type=Path,
        default=Path(
            "/home/yuyuan/Unity/Hub/Editor/6000.5.6f1/Editor/Data/MonoBleedingEdge/bin"
        ),
        help="Unity Mono bin directory containing mcs and mono",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        help="optionally keep the compiled SessionHarness.exe",
    )
    args = parser.parse_args()

    source_names = [
        "TeleopSessionState.cs",
        "TeleopRuntimeSettings.cs",
        "TeleopConnectionValidator.cs",
        "TeleopSettingsStore.cs",
        "TeleopSessionStateMachine.cs",
        "TeleopVideoSessionCoordinator.cs",
        "TeleopSessionCoordinator.cs",
        "TeleopCoreTests.cs",
    ]
    sources = [find_source(args.project, name) for name in source_names]
    harness = Path(__file__).parent / "csharp/SessionHarness.cs"
    if not harness.is_file():
        raise FileNotFoundError(f"missing harness: {harness}")

    mcs = args.mono_bin / "mcs"
    mono = args.mono_bin / "mono"
    if not mcs.is_file() or not mono.is_file():
        raise FileNotFoundError(f"Unity Mono tools not found below {args.mono_bin}")

    with tempfile.TemporaryDirectory(prefix="doffy-session-") as directory:
        binary = Path(directory) / "SessionHarness.exe"
        compile_command = [str(mcs), "-out:" + str(binary), *map(str, sources), str(harness)]
        print("Compiling session harness from:")
        for source in sources:
            print(f"  {source}")
        subprocess.run(compile_command, check=True, cwd=str(args.project))

        result = subprocess.run(
            [str(mono), str(binary)],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(args.project),
        )
        if result.stdout:
            print(result.stdout.rstrip())
        if result.stderr:
            print(result.stderr.rstrip())

        if args.export_dir:
            args.export_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(binary, args.export_dir / binary.name)
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
