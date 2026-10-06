"""Compile and run the Unity-free wrist-ring detent tracker checks under Mono."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        type=lambda value: Path(value).expanduser().resolve(),
        default=os.environ.get("DOFFY_UNITY_PROJECT"), required=not os.environ.get("DOFFY_UNITY_PROJECT"),
        help="Unity project root containing Assets/",
    )
    parser.add_argument(
        "--mono-bin",
        type=lambda value: Path(value).expanduser().resolve(),
        default=os.environ.get("DOFFY_MONO_BIN"), required=not os.environ.get("DOFFY_MONO_BIN"),
        help="Unity Mono bin directory containing mcs and mono",
    )
    args = parser.parse_args()

    tracker = args.project / "Assets/Teleop/UI/WristDetentTracker.cs"
    harness = Path(__file__).parent / "csharp/WristDetentsHarness.cs"
    mcs = args.mono_bin / "mcs"
    mono = args.mono_bin / "mono"
    for path in (tracker, harness, mcs, mono):
        if not path.is_file():
            parser.error(f"required file not found: {path}")

    with tempfile.TemporaryDirectory(prefix="doffy-wrist-detents-") as directory:
        binary = Path(directory) / "WristDetentsHarness.exe"
        subprocess.run(
            [str(mcs), "-out:" + str(binary), str(tracker), str(harness)],
            check=True,
        )
        result = subprocess.run(
            [str(mono), str(binary)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            print(result.stdout.rstrip())
        if result.stderr:
            print(result.stderr.rstrip())
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
