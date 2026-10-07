"""Run continuous bracelet geometry/contact regression checks against production C#."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=lambda value: Path(value).expanduser().resolve(), default=os.environ.get("DOFFY_UNITY_PROJECT"), required=not os.environ.get("DOFFY_UNITY_PROJECT"))
    parser.add_argument('--mono-bin', type=lambda value: Path(value).expanduser().resolve(), default=os.environ.get("DOFFY_MONO_BIN"), required=not os.environ.get("DOFFY_MONO_BIN"))
    args = parser.parse_args()
    sources = [args.project / 'Assets/Teleop/UI' / name for name in ('BraceletDragMath.cs', 'WristDetentTracker.cs')]
    sources.append(Path(__file__).parent / 'csharp/BraceletDragHarness.cs')
    with tempfile.TemporaryDirectory(prefix='doffy-bracelet-checks-') as temporary:
        binary = Path(temporary) / 'BraceletDragHarness.exe'
        subprocess.run([str(args.mono_bin / 'mcs'), '-out:' + str(binary), *map(str, sources)], check=True)
        return subprocess.run([str(args.mono_bin / 'mono'), str(binary)], check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
