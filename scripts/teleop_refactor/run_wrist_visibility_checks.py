"""Run production wrist-pose hysteresis with no Unity runtime or XR hardware."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=os.environ.get("DOFFY_UNITY_PROJECT"),
                        required=not os.environ.get("DOFFY_UNITY_PROJECT"))
    parser.add_argument("--mono-bin", type=Path, default=os.environ.get("DOFFY_MONO_BIN"),
                        required=not os.environ.get("DOFFY_MONO_BIN"))
    args = parser.parse_args()
    mono = args.mono_bin.expanduser().resolve()
    source = args.project.expanduser().resolve() / "Assets/Teleop/UI/WristVisibilityGate.cs"
    harness = Path(__file__).parent / 'csharp/WristVisibilityHarness.cs'
    with tempfile.TemporaryDirectory(prefix='wrist-visibility-') as directory:
        binary = Path(directory) / 'checks.exe'
        subprocess.run([str(mono / 'mcs'), '-out:' + str(binary), str(source), str(harness)], check=True)
        subprocess.run([str(mono / 'mono'), str(binary)], check=True)


if __name__ == '__main__':
    main()
