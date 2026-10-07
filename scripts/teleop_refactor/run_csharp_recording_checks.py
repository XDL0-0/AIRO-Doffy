"""Compile and exercise the production recording and hand-sender C# sources under Mono."""

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
        help="Unity project containing the production teleop scripts",
    )
    parser.add_argument(
        "--mono-bin",
        type=lambda value: Path(value).expanduser().resolve(),
        default=os.environ.get("DOFFY_MONO_BIN"), required=not os.environ.get("DOFFY_MONO_BIN"),
        help="Unity Mono bin directory containing mcs and mono",
    )
    args = parser.parse_args()

    sources = [
        args.project / "Assets/Teleop/Core/RecordingController.cs",
        args.project / "Assets/Teleop/Input/HandTrackingSender.cs",
        args.project / "Assets/Teleop/Protocol/TeleopWireFormat.cs",
        Path(__file__).parent / "csharp/RecordingAndHandSenderHarness.cs",
    ]
    missing = [str(source) for source in sources if not source.is_file()]
    if missing:
        parser.error("missing source file(s): " + ", ".join(missing))

    mcs = args.mono_bin / "mcs"
    mono = args.mono_bin / "mono"
    if not mcs.is_file() or not mono.is_file():
        parser.error("Unity Mono tools are not available below " + str(args.mono_bin))

    with tempfile.TemporaryDirectory(prefix="doffy-recording-") as directory:
        binary = Path(directory) / "RecordingAndHandSenderHarness.exe"
        compile_result = subprocess.run(
            [str(mcs), "-langversion:latest", "-out:" + str(binary), *map(str, sources)],
            check=False,
            cwd=str(args.project),
        )
        if compile_result.returncode:
            return compile_result.returncode
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
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
