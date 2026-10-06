"""BODY remains usable without the optional teleop and rendering stacks."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import textwrap


REPOSITORY = Path(__file__).resolve().parents[1]


def _run_without_site(source: str) -> None:
    result = subprocess.run(
        [sys.executable, "-S", "-c", textwrap.dedent(source)],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_body_imports_and_parsing_need_only_the_standard_library() -> None:
    _run_without_site("""
        import json
        import socket
        import sys

        def unexpected_socket(*args, **kwargs):
            raise AssertionError("BODY imports must not open sockets")

        socket.socket = unexpected_socket
        from doffy_teleop.protocol import (
            BodyFrame, BodyJoint, JOINT_IDS, JOINT_NAMES, is_finger, parse_body_packet,
        )
        from doffy_teleop.media import BodyReceiver
        from doffy_teleop.protocol import body as protocol_body
        from doffy_teleop.media import body as media_body

        assert sys.flags.no_site == 1
        for name in protocol_body.__all__:
            assert globals()[name] is getattr(protocol_body, name)
        assert BodyReceiver is media_body.BodyReceiver
        assert len(JOINT_NAMES) == 84
        assert JOINT_IDS["Head"] == 7
        payload = {
            "type": "BODY", "version": 1, "coordinate_space": "unity_world",
            "joint_set": "upper_body", "frame_id": 1, "timestamp_ns": 10,
            "confidence": 0.9, "tracking_valid": True,
            "joints": [{
                "id": 7, "name": "Head", "position": [0, 1.7, 0],
                "rotation": [0, 0, 0, 1],
                "position_valid": True, "orientation_valid": True,
            }],
        }
        frame = parse_body_packet(json.dumps(payload), received_ns=123)
        assert isinstance(frame, BodyFrame)
        assert isinstance(frame.joints[0], BodyJoint)
        assert frame.received_ns == 123
        assert frame.joints[0].position == (0, 1.7, 0)
        payload["joints"][0]["rotation"] = [0, 0, 0, 0]
        assert parse_body_packet(json.dumps(payload)) is None

        forbidden = (
            "numpy", "cv2", "parse_vr", "utils", "matplotlib", "scipy",
            "doffy_teleop.protocol.parse_vr", "doffy_teleop.utils", "doffy_teleop.config",
            "av", "aiortc", "camera_manager", "robot_backend", "robot_teleop",
            "realman_teleop", "doffy_teleop.protocol.control", "doffy_teleop.protocol.jpeg",
            "doffy_teleop.protocol.vr", "doffy_teleop.media.camera", "doffy_teleop.media.frames",
            "doffy_teleop.media.udp", "doffy_teleop.media.webrtc", "doffy_teleop.media.webrtc_peer",
            "doffy_teleop.media.webrtc_signaling", "doffy_teleop.robots", "doffy_teleop.cameras",
            "doffy_teleop.runtime", "doffy_teleop.visualization",
        )
        assert not [name for name in sys.modules if any(
            name == prefix or name.startswith(prefix + ".") for prefix in forbidden
        )]
    """)


def test_package_exports_preserve_legacy_names_and_resolve_lazily() -> None:
    # Stubs keep this API compatibility check independent of installed camera,
    # codec and VR dependencies. The defining module paths are the old API.
    _run_without_site("""
        import importlib
        import sys
        import types

        legacy_exports = {
            "doffy_teleop.protocol": {
                "jpeg": ["HD_HEADER_FMT", "HD_HEADER_SIZE", "JpegChunkAssembler",
                         "JpegChunkSender", "JpegDatagram"],
                "control": ["LegacyControlState", "RecordControl"],
                "vr": ["LegacyVRPacketDecoder"],
                "signaling": ["SignalingMessage", "parse_signaling_message"],
            },
            "doffy_teleop.media": {
                "camera": ["CameraCaptureService", "CameraFrameProvider",
                           "CameraVideoTrack", "FrameStore"],
                "udp": ["UDPManagerCore"],
                "webrtc_peer": ["WebRTCSession"],
                "webrtc_signaling": ["WebRTCSignalingServer"],
                "webrtc": ["WebRTCUDPManagerCore"],
                "frames": ["center_zoom", "prepare_rgb_frame"],
            },
        }
        body_exports = {
            "doffy_teleop.protocol": {
                "BodyFrame", "BodyJoint", "JOINT_IDS", "JOINT_NAMES",
                "is_finger", "parse_body_packet",
            },
            "doffy_teleop.media": {"BodyReceiver"},
        }
        for package_name, modules in legacy_exports.items():
            package = importlib.import_module(package_name)
            names = {name for exports in modules.values() for name in exports}
            assert set(package.__all__) == names | body_exports[package_name]
            assert names <= set(dir(package))
            assert not names & package.__dict__.keys()
            for module_name, exports in modules.items():
                qualified = package_name + "." + module_name
                assert qualified not in sys.modules
                module = types.ModuleType(qualified)
                for name in exports:
                    setattr(module, name, object())
                sys.modules[qualified] = module
                namespace = {}
                exec("from " + package_name + " import " + ", ".join(exports), namespace)
                for name in exports:
                    assert namespace[name] is getattr(module, name)
                    assert getattr(package, name) is namespace[name]
            try:
                getattr(package, "does_not_exist")
            except AttributeError:
                pass
            else:
                raise AssertionError("unknown package export did not raise AttributeError")
    """)
