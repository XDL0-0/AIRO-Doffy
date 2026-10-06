"""Integration test for the production Unity signaling client without Unity/native WebRTC."""

from pathlib import Path
import os
import subprocess
import sys
import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/teleop_refactor/run_webrtc_csharp_checks.py"
UNITY = Path(os.environ["DOFFY_UNITY_PROJECT"]).expanduser().resolve() if os.environ.get("DOFFY_UNITY_PROJECT") else None
EDITOR = Path(os.environ["DOFFY_UNITY_EDITOR_DATA"]).expanduser().resolve() if os.environ.get("DOFFY_UNITY_EDITOR_DATA") else None


def test_real_video_signaling_client_against_aiohttp_server():
    if UNITY is None or EDITOR is None or not UNITY.is_dir() or not (EDITOR / "MonoBleedingEdge/bin-linux64/mcs").is_file():
        pytest.skip("Optional Unity C# integration requires the Quest source and Unity SDK; configure DOFFY_UNITY_PROJECT/DOFFY_UNITY_EDITOR_DATA")
    result = subprocess.run(
        [sys.executable, str(RUNNER)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "REAL_WEBRTC_SIGNALING_CHECKS_PASS" in result.stdout
