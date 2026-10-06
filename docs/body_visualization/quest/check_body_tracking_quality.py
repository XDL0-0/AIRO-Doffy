#!/usr/bin/env python3
"""Exercise the deployed BODY sender under pure C# adapters using UDP loopback.

Checks the high-confidence boundary, focus/pause loss, unchanged WRM diagnostics,
and valid-but-untracked SDK joint forwarding. This does not run Unity or hardware.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path


UNITY_PROJECT = Path("/home/yuyuan/UNITY_Project/CodexBracelet")
MONO_BIN = Path("/home/yuyuan/Unity/Hub/Editor/6000.5.6f1/Editor/Data/MonoBleedingEdge/bin")
HERE = Path(__file__).resolve().parent
EXPECTED = {
    "high": (True, 0.75),
    "boundary": (False, 0.5),
    "low": (False, 0.25),
    "zero": (False, 0.0),
    "recovered": (True, 0.5001),
    "paused": (False, 0.875),
    "unfocused": (False, 0.875),
    "resumed": (True, 0.875),
    "sdk_invalid": (False, 0.75),
    "nan": (False, 0.0),
    "infinite": (False, 0.0),
    "out_of_range": (False, 1.0),
    "valid_untracked": (True, 0.875),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=UNITY_PROJECT)
    parser.add_argument("--mono-bin", type=Path, default=MONO_BIN)
    args = parser.parse_args()
    sources = [
        args.project / "Assets/Teleop/UpperLimb/BodyPoseTelemetrySender.cs",
        args.project / "Assets/Teleop/Protocol/BodyPoseWireFormat.cs",
        HERE / "BodyPoseTelemetrySenderHarness.cs",
    ]
    with tempfile.TemporaryDirectory(prefix="body-tracking-quality-") as temp_dir:
        executable = Path(temp_dir) / "BodyPoseTelemetrySenderHarness.exe"
        subprocess.run(
            [str(args.mono_bin / "mcs"), "-langversion:latest", f"-out:{executable}", *map(str, sources)],
            check=True,
        )
        result = subprocess.run(
            [str(args.mono_bin / "mono"), str(executable)], check=True, capture_output=True, text=True
        )

    packets = {}
    for line in result.stdout.splitlines():
        label, raw = line.split("\t", 1)
        assert label in EXPECTED and label not in packets, f"Unexpected or duplicate case: {label}"
        packet = json.loads(raw)
        valid, confidence = EXPECTED[label]
        assert packet["tracking_valid"] is valid, f"{label}: incorrect tracking quality gate"
        assert math.isclose(packet["confidence"], confidence, abs_tol=1e-6), f"{label}: diagnostic confidence changed"
        assert packet["wrm"] == {
            "elbow_alpha": 0.25, "confidence": 0.75, "enabled": True, "calibrated": True
        }, f"{label}: BODY quality gate changed WRM diagnostics"
        assert len(packet["joints"]) == 70, f"{label}: joint count changed"
        for index, joint in enumerate(packet["joints"]):
            assert joint["position_valid"] is valid and joint["orientation_valid"] is valid
            assert joint["position_tracked"] is (valid and label != "valid_untracked")
            assert joint["orientation_tracked"] is (valid and label != "valid_untracked")
            assert joint["position"] == ([index + 0.25, 1.5, -2.5] if valid else None)
            assert joint["rotation"] == ([0, 0, 0, 1] if valid else None)
        packets[label] = packet
    assert set(packets) == set(EXPECTED), "Missing BODY tracking-quality scenario"
    assert [packets[label]["frame_id"] for label in EXPECTED] == list(range(1, 14))
    print("PASS: 13 deployed sender cases; confidence boundary/loss/recovery; pause/focus; actual tracked bits; WRM unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
