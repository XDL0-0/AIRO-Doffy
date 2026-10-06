#!/usr/bin/env python3
"""Real local media loopbacks for the legacy Unity/PC wire contracts.

The default run uses real UDP sockets and exercises malformed, duplicate,
out-of-order, incomplete/expired, and restarted JPEG frames.  ``--csharp``
also starts the supplied Unity C# harness, which uses the same
``JpegFrameAssembler`` source as the Unity client.  ``--realsense`` is an
optional capture-only check; it never imports or operates a robot.
"""

from __future__ import annotations

import argparse
import cv2
import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

import numpy as np

# Keep the diagnostic directly executable from a source checkout where the
# package has not been installed editable yet.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from doffy_teleop.protocol.jpeg import JpegChunkAssembler, JpegChunkSender


MONO_DEFAULT = Path(
    os.environ.get("DOFFY_MONO") or shutil.which("mono") or "mono"
)
CSHARP_HARNESS_DEFAULT = Path("/tmp/teleop-protocol-tests.exe")


def _decode_size(encoded: bytes) -> list[int] | None:
    frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    return list(frame.shape[:2]) if frame is not None else None


def _send_and_receive(
    tx: socket.socket,
    rx: socket.socket,
    packets: list[bytes],
    *,
    assembler: JpegChunkAssembler,
    duplicate: bool = False,
) -> bytes:
    address = rx.getsockname()
    if duplicate and packets:
        tx.sendto(packets[-1], address)
    for packet in reversed(packets):
        tx.sendto(packet, address)
    rx.settimeout(0.25)
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            packet, _peer = rx.recvfrom(65_535)
        except socket.timeout:
            continue
        image = assembler.feed(packet)
        if image is not None:
            return image
    raise TimeoutError("JPEG loopback did not complete before its deadline")


def run_python_udp_loopback(*, chunk_size: int = 180, quality: int = 88) -> dict[str, Any]:
    """Exercise the Python sender and assembler through real UDP sockets."""

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sender_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    assembler = JpegChunkAssembler(frame_timeout_s=0.03)
    try:
        image = np.random.default_rng(20260916).integers(
            0, 256, size=(96, 128, 3), dtype=np.uint8
        )
        sender = JpegChunkSender(chunk_size=chunk_size, quality=quality)
        expected = sender.encode(image, quality=quality)
        _frame_id, packets = sender.packets(image, quality=quality)
        sender_socket.sendto(b"malformed", receiver.getsockname())
        assembled = _send_and_receive(
            sender_socket,
            receiver,
            packets,
            assembler=assembler,
            duplicate=True,
        )
        first_stats = dict(assembler.stats)

        # Deliberately omit the final datagram and expire the partial frame.
        _frame_id, missing_packets = sender.packets(image, quality=quality)
        for packet in missing_packets[:-1]:
            sender_socket.sendto(packet, receiver.getsockname())
        receiver.settimeout(0.25)
        for _ in missing_packets[:-1]:
            packet, _peer = receiver.recvfrom(65_535)
            assembler.feed(packet)
        time.sleep(0.05)
        expired = assembler.expire()

        # A restarted sender starts at frame id zero.  Resetting the receiver
        # is explicit and prevents a prior session's completed-id cache from
        # rejecting the new first frame.
        restarted = JpegChunkSender(chunk_size=chunk_size, quality=quality)
        _frame_id, restart_packets = restarted.packets(image, quality=quality)
        assembler.reset()
        restarted_bytes = _send_and_receive(
            sender_socket,
            receiver,
            restart_packets,
            assembler=assembler,
        )
        if assembled != expected or restarted_bytes != expected:
            raise AssertionError("Python JPEG loopback bytes changed during reassembly")
        return {
            "status": "passed",
            "transport": "python-udp",
            "shape": _decode_size(assembled),
            "jpeg_sha256": hashlib.sha256(assembled).hexdigest(),
            "chunks": len(packets),
            "malformed": first_stats["malformed"],
            "duplicates": first_stats["duplicates"],
            "expired": expired,
            "restarts": assembler.stats["restarts"],
        }
    finally:
        sender_socket.close()
        receiver.close()


def run_unity_csharp_loopback(
    *,
    mono: Path = MONO_DEFAULT,
    harness: Path = CSHARP_HARNESS_DEFAULT,
    chunk_size: int = 180,
    quality: int = 88,
    image: np.ndarray | None = None,
) -> dict[str, Any]:
    """Send Python JPEG chunks to the Unity C# receiver harness."""

    if not mono.exists() or not harness.exists():
        return {
            "status": "skipped",
            "transport": "python-to-unity-csharp-udp",
            "reason": f"missing {mono} or {harness}",
        }
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    port = receiver.getsockname()[1]
    receiver.close()
    output_path = Path(tempfile.mktemp(prefix="teleop-csharp-", suffix=".jpg"))
    process = subprocess.Popen(
        [str(mono), str(harness), "receive", str(port), str(output_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        if process.stdout is None:
            raise RuntimeError("C# harness stdout was not opened")
        ready, _write, _error = select.select([process.stdout], [], [], 5.0)
        if not ready:
            raise TimeoutError("Unity C# receiver did not announce READY")
        ready_line = process.stdout.readline().strip()
        if not ready_line.startswith("READY"):
            raise RuntimeError(f"unexpected Unity C# receiver output: {ready_line}")

        if image is None:
            image = np.random.default_rng(20260916).integers(
                0, 256, size=(72, 96, 3), dtype=np.uint8
            )
        sender = JpegChunkSender(chunk_size=chunk_size, quality=quality)
        expected = sender.encode(image, quality=quality)
        _frame_id, packets = sender.packets(image, quality=quality)
        for packet in reversed(packets):
            tx.sendto(packet, ("127.0.0.1", port))
        process.communicate(timeout=10.0)
        if process.returncode != 0:
            raise RuntimeError(f"Unity C# receiver returned {process.returncode}")
        actual = output_path.read_bytes()
        if actual != expected:
            raise AssertionError("Unity C# reassembly bytes differ from Python sender")
        return {
            "status": "passed",
            "transport": "python-to-unity-csharp-udp",
            "shape": _decode_size(actual),
            "jpeg_sha256": hashlib.sha256(actual).hexdigest(),
            "chunks": len(packets),
            "ready": ready_line,
        }
    finally:
        tx.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3.0)
        output_path.unlink(missing_ok=True)


def run_realsense_capture(*, frames: int = 3, chunk_size: int = 180, csharp_harness: Path | None = None, mono: Path = MONO_DEFAULT) -> dict[str, Any]:
    """Capture a few frames from the first visible RealSense, if available."""

    try:
        import pyrealsense2 as rs
        from airo_camera_toolkit.cameras.realsense.realsense import Realsense
    except ImportError as exc:
        return {"status": "skipped", "transport": "realsense-capture", "reason": str(exc)}
    devices = list(rs.context().query_devices())
    if not devices:
        return {"status": "skipped", "transport": "realsense-capture", "reason": "no device"}
    serial = devices[0].get_info(rs.camera_info.serial_number)
    camera = None
    stage = "camera_init"
    captured_frames = 0
    try:
        stage = "camera_init"
        camera = Realsense(
            fps=30,
            resolution=(640, 480),
            enable_depth=False,
            enable_pointcloud=False,
            enable_hole_filling=False,
            serial_number=serial,
        )
        sender = JpegChunkSender(chunk_size=chunk_size, quality=88)
        stage = "udp_bind"
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(("127.0.0.1", 0))
        # A 180-byte stress chunk produces hundreds of datagrams for one
        # camera frame.  Drain concurrently and enlarge the kernel receive
        # queue so a fast sender cannot make the camera check fail because of
        # its own burst before recvfrom() starts.
        receiver.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        assembler = JpegChunkAssembler(frame_timeout_s=0.5)
        shapes: list[list[int] | None] = []
        try:
            for _ in range(max(1, int(frames))):
                stage = "camera_capture"
                camera.grab_images()
                rgb = camera.retrieve_rgb_image()
                captured_frames += 1
                stage = "jpeg_encode"
                _frame_id, packets = sender.packets(
                    cv2.cvtColor(
                        np.clip(rgb * 255, 0, 255).astype(np.uint8)
                        if rgb.dtype != np.uint8
                        else rgb,
                        cv2.COLOR_RGB2BGR,
                    ),
                )
                address = receiver.getsockname()
                send_errors: list[BaseException] = []

                def send_packets() -> None:
                    try:
                        for packet in packets:
                            tx.sendto(packet, address)
                    except BaseException as exc:  # surface sender errors below
                        send_errors.append(exc)

                stage = "udp_loopback"
                sender_thread = threading.Thread(target=send_packets, daemon=True)
                sender_thread.start()
                receiver.settimeout(1.0)
                complete = None
                deadline = time.monotonic() + 2.0
                while (sender_thread.is_alive() or complete is None) and time.monotonic() < deadline:
                    try:
                        packet, _peer = receiver.recvfrom(65_535)
                    except socket.timeout:
                        continue
                    complete = assembler.feed(packet)
                sender_thread.join(timeout=1.0)
                if send_errors:
                    raise OSError(f"camera UDP sender failed: {send_errors[0]!r}")
                if complete is None:
                    raise TimeoutError("RealSense UDP capture frame did not complete")
                shapes.append(_decode_size(complete))
        finally:
            tx.close()
            receiver.close()
        csharp = None
        if csharp_harness is not None:
            stage = "camera_to_csharp_udp"
            bgr = cv2.cvtColor(np.clip(rgb * 255, 0, 255).astype(np.uint8) if rgb.dtype != np.uint8 else rgb, cv2.COLOR_RGB2BGR)
            csharp = run_unity_csharp_loopback(mono=mono, harness=csharp_harness, image=bgr, chunk_size=1200)
            if csharp["status"] != "passed":
                raise RuntimeError(f"Camera to C# did not pass: {csharp}")
        return {
            "status": "passed",
            "transport": "realsense-to-python-udp",
            "serial": serial,
            "frames": len(shapes),
            "captured_frames": captured_frames,
            "shapes": shapes,
            "unity_csharp": csharp,
        }
    except Exception as exc:
        # A camera may be reserved by another process.  Report the concrete
        # failure while leaving robot and other hardware untouched.
        return {
            "status": "unavailable",
            "transport": "realsense-capture",
            "serial": serial,
            "stage": stage,
            "captured_frames": captured_frames,
            "reason": repr(exc),
        }
    finally:
        if camera is not None:
            try:
                pipeline = getattr(camera, "pipeline", None)
                if pipeline is not None and hasattr(pipeline, "stop"):
                    pipeline.stop()
                elif hasattr(camera, "close"):
                    camera.close()
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csharp", action="store_true", help="also run the Unity C# receiver harness")
    parser.add_argument("--realsense", action="store_true", help="capture a few frames only from the first RealSense")
    parser.add_argument("--frames", type=int, default=3, help="RealSense frames to capture")
    parser.add_argument("--harness", type=Path, default=CSHARP_HARNESS_DEFAULT, help="compiled C# ProtocolHarness.exe path")
    parser.add_argument("--mono", type=Path, default=MONO_DEFAULT, help="Mono runtime (or set DOFFY_MONO)")
    args = parser.parse_args()
    results: list[dict[str, Any]] = [run_python_udp_loopback()]
    if args.csharp:
        results.append(run_unity_csharp_loopback(mono=args.mono.expanduser().resolve(), harness=args.harness))
    if args.realsense:
        results.append(run_realsense_capture(frames=args.frames, csharp_harness=args.harness if args.csharp else None, mono=args.mono.expanduser().resolve()))
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if all(result["status"] in {"passed", "skipped"} for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
