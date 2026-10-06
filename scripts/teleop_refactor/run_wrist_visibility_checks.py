"""Run production wrist-pose hysteresis with no Unity runtime or XR hardware."""
from pathlib import Path
import subprocess
import tempfile


def main():
    mono = Path('/home/yuyuan/Unity/Hub/Editor/6000.5.6f1/Editor/Data/MonoBleedingEdge/bin')
    source = Path('/home/yuyuan/UNITY_Project/Codex/Assets/Teleop/UI/WristVisibilityGate.cs')
    harness = Path(__file__).parent / 'csharp/WristVisibilityHarness.cs'
    with tempfile.TemporaryDirectory(prefix='wrist-visibility-') as directory:
        binary = Path(directory) / 'checks.exe'
        subprocess.run([str(mono / 'mcs'), '-out:' + str(binary), str(source), str(harness)], check=True)
        subprocess.run([str(mono / 'mono'), str(binary)], check=True)


if __name__ == '__main__':
    main()
