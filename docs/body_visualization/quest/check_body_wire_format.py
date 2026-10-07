#!/usr/bin/env python3
"""Compile the deployed BODY serializer and check its JSON wire contract.

BODY v1 adds position_tracked/orientation_tracked without changing existing fields
or the version. Older consumers can ignore these additive fields. An old producer
that omits them retains its legacy validity-based tracking semantics; omission is
not an explicit untracked signal. This checker requires both fields from the new
serializer and verifies that valid-but-untracked estimates retain their values.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
HARNESS = HERE / "BodyPoseWireFormatHarness.cs"
MAX_UDP_PAYLOAD_BYTES = 65507
EXPECTED_COUNTS = {100: 70, 101: 84, 102: 84, 103: 70, 104: 84, 105: 70, 106: 70, 107: 70}


def fail(message: str) -> None:
    raise AssertionError(message)


def parse_constant(token: str) -> None:
    fail(f"non-JSON numeric constant emitted: {token}")


def load_json_line(line: str, index: int) -> dict[str, Any]:
    try:
        value = json.loads(line, parse_constant=parse_constant)
    except (json.JSONDecodeError, AssertionError) as error:
        fail(f"packet line {index} is not strict JSON: {error}")
    if not isinstance(value, dict):
        fail(f"packet line {index} is not a JSON object")
    return value


def extract_wire_names(source_text: str) -> list[str]:
    match = re.search(
        r"private\s+static\s+readonly\s+string\[\]\s+JointNames\s*=\s*\{(.*?)\};",
        source_text,
        re.DOTALL,
    )
    if not match:
        fail("could not locate JointNames in BodyPoseWireFormat source")
    names = re.findall(r'"([A-Za-z0-9_]+)"', match.group(1))
    if len(names) != 84:
        fail(f"expected 84 canonical names in source, found {len(names)}")
    return names


def find_bone_id_enum(ovr_text: str) -> str:
    match = re.search(
        r"\bpublic\s+enum\s+BoneId\s*\{(.*?)^\s*\}",
        ovr_text,
        re.DOTALL | re.MULTILINE,
    )
    if not match:
        fail("could not locate public enum BoneId in installed OVRPlugin.cs")
    return match.group(1)


def enum_value(expression: str) -> int:
    expression = expression.strip()
    if re.fullmatch(r"-?\d+", expression):
        return int(expression)
    match = re.fullmatch(r"(?:Body_Start|FullBody_Start)\s*\+\s*(\d+)", expression)
    if match:
        return int(match.group(1))
    fail(f"unsupported BoneId enum expression: {expression}")


def sdk_joint_names(ovr_text: str, prefix: str, limit: int) -> list[str]:
    enum_body = find_bone_id_enum(ovr_text)
    indexed: dict[int, str] = {}
    for line in enum_body.splitlines():
        clean = line.split("//", 1)[0].strip().rstrip(",").strip()
        match = re.match(r"([A-Za-z_]\w*)\s*=\s*(.+)$", clean)
        if not match or not match.group(1).startswith(prefix):
            continue
        sdk_name = match.group(1)[len(prefix) :]
        if sdk_name in ("Start", "End"):
            continue
        value = enum_value(match.group(2))
        if 0 <= value < limit:
            if value in indexed:
                fail(f"duplicate {prefix}BoneId index {value} in installed SDK")
            indexed[value] = sdk_name
    missing = sorted(set(range(limit)) - set(indexed))
    if missing:
        fail(f"installed SDK enum is missing {prefix} joints at indices {missing}")
    return [indexed[index] for index in range(limit)]


def check_sdk_mapping(names: list[str], ovr_text: str) -> None:
    upper_sdk = sdk_joint_names(ovr_text, "Body_", 70)
    full_sdk = sdk_joint_names(ovr_text, "FullBody_", 84)
    if names[:70] != upper_sdk:
        for index, (wire_name, sdk_name) in enumerate(zip(names[:70], upper_sdk)):
            if wire_name != sdk_name:
                fail(f"upper-body joint {index}: wire={wire_name}, SDK={sdk_name}")
    if names != full_sdk:
        for index, (wire_name, sdk_name) in enumerate(zip(names, full_sdk)):
            if wire_name != sdk_name:
                fail(f"full-body joint {index}: wire={wire_name}, SDK={sdk_name}")
    if names[7] != "Head" or names[19] != "LeftHandWrist" or names[45] != "RightHandWrist":
        fail("head or wrist canonical indices do not match SDK ordering (7, 19, 45)")
    if names[70:84] != [
        "LeftUpperLeg",
        "LeftLowerLeg",
        "LeftFootAnkleTwist",
        "LeftFootAnkle",
        "LeftFootSubtalar",
        "LeftFootTransverse",
        "LeftFootBall",
        "RightUpperLeg",
        "RightLowerLeg",
        "RightFootAnkleTwist",
        "RightFootAnkle",
        "RightFootSubtalar",
        "RightFootTransverse",
        "RightFootBall",
    ]:
        fail("full-body leg ordering at indices 70..83 differs from the expected SDK mapping")


def check_joint_records(packet: dict[str, Any], names: list[str], expected_count: int) -> None:
    joints = packet.get("joints")
    if not isinstance(joints, list) or len(joints) != expected_count:
        fail(f"frame {packet.get('frame_id')}: expected {expected_count} joint records")
    for index, joint in enumerate(joints):
        if not isinstance(joint, dict):
            fail(f"frame {packet.get('frame_id')} joint {index} is not an object")
        if joint.get("id") != index or joint.get("name") != names[index]:
            fail(f"frame {packet.get('frame_id')} joint {index} has an unexpected id/name mapping")
        for field, valid_field, tracked_field, size in (
            ("position", "position_valid", "position_tracked", 3),
            ("rotation", "orientation_valid", "orientation_tracked", 4),
        ):
            value = joint.get(field)
            valid = joint.get(valid_field)
            tracked = joint.get(tracked_field)
            if not isinstance(valid, bool):
                fail(f"frame {packet.get('frame_id')} joint {index} {valid_field} is not boolean")
            if not isinstance(tracked, bool):
                fail(f"frame {packet.get('frame_id')} joint {index} {tracked_field} is not boolean")
            if tracked and (not valid or not packet["tracking_valid"]):
                fail(f"frame {packet.get('frame_id')} joint {index} {tracked_field} requires valid tracking")
            if packet["frame_id"] != 107 or index >= 5:
                if tracked != valid:
                    fail(f"frame {packet.get('frame_id')} joint {index} legacy tracked default differs from validity")
            if valid:
                if not isinstance(value, list) or len(value) != size:
                    fail(f"frame {packet.get('frame_id')} joint {index} has invalid {field} data")
                if any(not isinstance(number, (int, float)) or isinstance(number, bool) for number in value):
                    fail(f"frame {packet.get('frame_id')} joint {index} {field} contains non-numeric data")
                if any(not math.isfinite(number) for number in value):
                    fail(f"frame {packet.get('frame_id')} joint {index} {field} contains non-finite data")
            elif value is not None:
                fail(f"frame {packet.get('frame_id')} joint {index} invalid {field} must be null")


def check_tracking_metadata(packet: dict[str, Any]) -> None:
    frame_id = packet["frame_id"]
    if not isinstance(packet.get("tracking_valid"), bool):
        fail(f"frame {frame_id} tracking_valid is not boolean")
    wrm = packet.get("wrm")
    if not isinstance(wrm, dict):
        fail(f"frame {frame_id} wrm is not an object")
    for field in ("enabled", "calibrated"):
        if not isinstance(wrm.get(field), bool):
            fail(f"frame {frame_id} wrm.{field} is not boolean")
    for label, value in (
        ("confidence", packet.get("confidence")),
        ("wrm.elbow_alpha", wrm.get("elbow_alpha")),
        ("wrm.confidence", wrm.get("confidence")),
    ):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            fail(f"frame {frame_id} {label} is not a finite number")
        if not 0 <= value <= 1:
            fail(f"frame {frame_id} {label} is outside the unit interval")


def check_json_roundtrip(packet: dict[str, Any], line_number: int) -> None:
    try:
        roundtrip = load_json_line(json.dumps(packet, allow_nan=False), line_number)
    except ValueError as error:
        fail(f"packet line {line_number} cannot roundtrip as strict JSON: {error}")
    if roundtrip != packet:
        fail(f"packet line {line_number} changed during strict JSON roundtrip")


def assert_invalid_joint(joint: dict[str, Any], index: int, position: bool, orientation: bool) -> None:
    expected = (position, orientation)
    actual = (joint.get("position_valid"), joint.get("orientation_valid"))
    if actual != expected:
        fail(f"edge-case joint {index} validity={actual}; expected {expected}")
    if (joint.get("position") is not None) != position:
        fail(f"edge-case joint {index} position nullability disagrees with validity")
    if (joint.get("rotation") is not None) != orientation:
        fail(f"edge-case joint {index} rotation nullability disagrees with validity")


def run_checks(source: Path, ovr_plugin: Path, mcs: Path, mono: Path) -> None:
    if not source.is_file():
        fail(f"serializer source does not exist: {source}")
    if not ovr_plugin.is_file():
        fail(f"installed OVRPlugin.cs does not exist: {ovr_plugin}")
    if not HARNESS.is_file():
        fail(f"C# harness is missing: {HARNESS}")

    names = extract_wire_names(source.read_text(encoding="utf-8"))
    check_sdk_mapping(names, ovr_plugin.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix="body-wire-format-") as temp_dir:
        executable = Path(temp_dir) / "BodyPoseWireFormatHarness.exe"
        compile_result = subprocess.run(
            [str(mcs), "-langversion:latest", f"-out:{executable}", str(source), str(HARNESS)],
            check=False,
            capture_output=True,
            text=True,
        )
        if compile_result.returncode:
            fail("Mono compilation failed:\n" + compile_result.stdout + compile_result.stderr)

        run_result = subprocess.run(
            [str(mono), str(executable)],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if run_result.returncode:
            fail("Mono harness failed:\n" + run_result.stdout + run_result.stderr)

        raw_lines = [line for line in run_result.stdout.splitlines() if line.strip()]
        if len(raw_lines) != len(EXPECTED_COUNTS):
            fail(f"expected {len(EXPECTED_COUNTS)} JSON packet lines, received {len(raw_lines)}")
        packets_by_frame: dict[int, dict[str, Any]] = {}
        raw_by_frame: dict[int, str] = {}
        for line_number, line in enumerate(raw_lines, 1):
            packet = load_json_line(line, line_number)
            frame_id = packet.get("frame_id")
            if not isinstance(frame_id, int) or frame_id in packets_by_frame:
                fail(f"packet line {line_number} has a missing or duplicate frame_id")
            if frame_id not in EXPECTED_COUNTS:
                fail(f"unexpected frame_id {frame_id}")
            if len(line.encode("utf-8")) >= MAX_UDP_PAYLOAD_BYTES:
                fail(f"frame {frame_id} exceeds the strict UDP payload limit")
            if packet.get("type") != "BODY" or packet.get("version") != 1:
                fail(f"frame {frame_id} has an unexpected packet type or version")
            if packet.get("timestamp_ns") != 1728123456789012345:
                fail(f"frame {frame_id} timestamp did not survive JSON serialization")
            if packet.get("coordinate_space") != "unity_world":
                fail(f"frame {frame_id} coordinate_space is unexpected")
            expected_set = "full_body" if EXPECTED_COUNTS[frame_id] == 84 else "upper_body"
            if packet.get("joint_set") != expected_set:
                fail(f"frame {frame_id} joint_set should be {expected_set}")
            check_tracking_metadata(packet)
            check_joint_records(packet, names, EXPECTED_COUNTS[frame_id])
            check_json_roundtrip(packet, line_number)
            packets_by_frame[frame_id] = packet
            raw_by_frame[frame_id] = line
            print(f"frame_id={frame_id} utf8_bytes={len(line.encode('utf-8'))}/{MAX_UDP_PAYLOAD_BYTES}")

        if set(packets_by_frame) != set(EXPECTED_COUNTS):
            fail("one or more expected frame ids are missing")

        upper = packets_by_frame[100]
        if upper.get("tracking_valid") is not True or upper.get("confidence") != 0.75:
            fail("valid upper-body packet tracking metadata differs from expected values")
        if "\"position\":[0.25,-0.5,0]" not in raw_by_frame[100]:
            fail("fr-FR culture test did not retain invariant decimal-dot JSON formatting")
        if upper["joints"][0].get("orientation_valid") is not True:
            fail("finite nonzero quaternion should be serialized as valid")

        full = packets_by_frame[101]
        if full.get("tracking_valid") is not True or len(full["joints"]) != 84:
            fail("valid full-body packet tracking metadata differs from expected values")
        if [joint["name"] for joint in full["joints"][70:84]] != names[70:84]:
            fail("full-body packet did not include the canonical lower-body joints")

        tracking_lost = packets_by_frame[102]
        if tracking_lost.get("tracking_valid") is not False or not math.isclose(
            tracking_lost["confidence"], 0.9, abs_tol=1e-6
        ):
            fail("globally invalid tracking must retain sanitized confidence for diagnostics")
        for index, joint in enumerate(tracking_lost["joints"]):
            assert_invalid_joint(joint, index, False, False)

        edge = packets_by_frame[103]["joints"]
        assert_invalid_joint(edge[0], 0, True, True)
        if edge[0]["position"] != [0, 0, 0] or edge[0]["rotation"] != [0, 0, 0, 1]:
            fail("zero-valued valid position or identity quaternion was changed")
        assert_invalid_joint(edge[1], 1, False, True)
        assert_invalid_joint(edge[2], 2, False, True)
        assert_invalid_joint(edge[3], 3, False, True)
        assert_invalid_joint(edge[4], 4, True, False)
        assert_invalid_joint(edge[5], 5, True, False)
        assert_invalid_joint(edge[6], 6, True, False)
        assert_invalid_joint(edge[7], 7, False, False)
        assert_invalid_joint(edge[8], 8, False, False)

        stress = packets_by_frame[104]
        if stress["joints"][0]["position"][0] < 3.0e38:
            fail("large finite stress position was not retained")
        if stress["joints"][0]["position"][1] > -3.0e38:
            fail("large negative finite stress position was not retained")
        if stress["joints"][0]["rotation"][0] < 3.0e38:
            fail("large finite stress quaternion component was not retained")
        if stress["wrm"].get("elbow_alpha") != 0 or stress["wrm"].get("confidence") != 0:
            fail("non-finite/out-of-range stress telemetry was not clamped to JSON-safe values")

        for frame_id in (105, 106):
            for index, joint in enumerate(packets_by_frame[frame_id]["joints"]):
                if frame_id == 106 and index < 2:
                    continue
                assert_invalid_joint(joint, index, False, False)

        tracking_bits = packets_by_frame[107]["joints"]
        expected_bits = [(False, False), (True, False), (False, True), (False, True), (True, False)]
        for index, expected in enumerate(expected_bits):
            joint = tracking_bits[index]
            actual = (joint["position_tracked"], joint["orientation_tracked"])
            if actual != expected:
                fail(f"explicit tracking joint {index} tracked={actual}; expected {expected}")
            position_valid = index != 3
            orientation_valid = index != 4
            assert_invalid_joint(joint, index, position_valid, orientation_valid)
            if position_valid and joint["position"] != [index + 0.25, -0.5 - index, index * 0.125]:
                fail(f"valid-but-untracked joint {index} position was changed")
            if orientation_valid and joint["rotation"] != [0, 0, 0.25, 0.75]:
                fail(f"valid-but-untracked joint {index} rotation was changed")

    print(
        f"PASS: {len(EXPECTED_COUNTS)} strict JSON packets and roundtrips; 70/84 joint counts; "
        "OVRPlugin BoneId mapping; fr-FR decimal invariance; validity, tracking-bit, "
        "and finite-number rules; tracking-loss confidence; UDP byte limits"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=os.environ.get("DOFFY_BODY_WIRE_SOURCE"), required=not os.environ.get("DOFFY_BODY_WIRE_SOURCE"), help="BodyPoseWireFormat.cs to compile")
    parser.add_argument("--ovr-plugin", type=Path, default=os.environ.get("DOFFY_OVR_PLUGIN"), required=not os.environ.get("DOFFY_OVR_PLUGIN"), help="installed OVRPlugin.cs enum source")
    parser.add_argument("--mcs", type=Path, default=os.environ.get("DOFFY_MCS"), required=not os.environ.get("DOFFY_MCS"), help="Unity bundled Mono C# compiler")
    parser.add_argument("--mono", type=Path, default=os.environ.get("DOFFY_MONO"), required=not os.environ.get("DOFFY_MONO"), help="Unity bundled Mono runtime")
    args = parser.parse_args()
    try:
        run_checks(args.source.expanduser().resolve(), args.ovr_plugin.expanduser().resolve(), args.mcs.expanduser().resolve(), args.mono.expanduser().resolve())
    except (AssertionError, OSError, subprocess.SubprocessError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
