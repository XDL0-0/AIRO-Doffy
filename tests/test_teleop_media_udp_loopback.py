from __future__ import annotations

import socket
import hashlib
import os
from pathlib import Path
import select
import shutil
import subprocess
import time
import unittest

import cv2
import numpy as np

from doffy_teleop.protocol.jpeg import JpegChunkAssembler, JpegChunkSender


class RealUdpJpegLoopbackTests(unittest.TestCase):
    def test_out_of_order_duplicate_bad_packet_and_restart(self) -> None:
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sender_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(("127.0.0.1", 0))
        address = receiver.getsockname()
        receiver.settimeout(0.25)
        try:
            image = np.random.default_rng(42).integers(
                0, 256, size=(96, 128, 3), dtype=np.uint8
            )
            sender = JpegChunkSender(chunk_size=180, quality=88)
            _frame_id, packets = sender.packets(image, cam_idx=0)
            self.assertGreater(len(packets), 2)
            assembler = JpegChunkAssembler(frame_timeout_s=0.2)

            sender_socket.sendto(b"not-a-jpeg-datagram", address)
            # Duplicate the first received chunk before the remaining chunks,
            # then reverse the order to exercise real kernel datagram delivery.
            sender_socket.sendto(packets[-1], address)
            sender_socket.sendto(packets[-1], address)
            for packet in reversed(packets[:-1]):
                sender_socket.sendto(packet, address)

            assembled = None
            deadline = time.monotonic() + 2.0
            while assembled is None and time.monotonic() < deadline:
                try:
                    packet, _ = receiver.recvfrom(65_535)
                except socket.timeout:
                    continue
                assembled = assembler.feed(packet)
            self.assertIsNotNone(assembled)
            decoded = cv2.imdecode(np.frombuffer(assembled, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertIsNotNone(decoded)
            self.assertEqual(decoded.shape[:2], image.shape[:2])
            self.assertGreaterEqual(assembler.stats["malformed"], 1)
            self.assertGreaterEqual(assembler.stats["duplicates"], 1)

            # Drop one chunk on a second frame and let the real-socket receive
            # path expire the partial frame.
            _frame_id, missing_packets = sender.packets(image, cam_idx=0)
            missing_assembler = JpegChunkAssembler(frame_timeout_s=0.03)
            for packet in missing_packets[:-1]:
                sender_socket.sendto(packet, address)
            for _ in missing_packets[:-1]:
                packet, _ = receiver.recvfrom(65_535)
                missing_assembler.feed(packet)
            time.sleep(0.05)
            missing_assembler.expire()
            self.assertEqual(missing_assembler.stats["expired"], 1)

            # A new sender starts at frame id zero.  reset() marks an explicit
            # transport restart and permits the same id to be accepted again.
            restarted_sender = JpegChunkSender(chunk_size=180, quality=88)
            _frame_id, restarted_packets = restarted_sender.packets(image, cam_idx=0)
            assembler.reset()
            for packet in reversed(restarted_packets):
                sender_socket.sendto(packet, address)
            restarted = None
            deadline = time.monotonic() + 2.0
            while restarted is None and time.monotonic() < deadline:
                packet, _ = receiver.recvfrom(65_535)
                restarted = assembler.feed(packet)
            self.assertIsNotNone(restarted)
            self.assertEqual(assembler.stats["restarts"], 1)
        finally:
            sender_socket.close()
            receiver.close()

    @unittest.skipUnless(
        Path("/tmp/teleop-protocol-tests.exe").exists()
        and Path(
            os.environ.get("DOFFY_MONO") or shutil.which("mono") or "mono"
        ).exists(),
        "Unity C# protocol harness is not installed",
    )
    def test_python_sender_to_unity_csharp_assembler(self) -> None:
        """Exercise the exact C# assembler used by the Unity client."""

        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(("127.0.0.1", 0))
        port = receiver.getsockname()[1]
        receiver.close()
        output_path = Path(f"/tmp/teleop-python-to-csharp-{os.getpid()}.jpg")
        mono = (
            os.environ.get("DOFFY_MONO") or shutil.which("mono") or "mono"
        )
        process = subprocess.Popen(
            [mono, "/tmp/teleop-protocol-tests.exe", "receive", str(port), str(output_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        sender_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.assertIsNotNone(process.stdout)
            ready, _write, _error = select.select([process.stdout], [], [], 5.0)
            self.assertTrue(ready, "Unity C# receiver did not announce READY")
            ready_line = process.stdout.readline().strip()
            self.assertTrue(ready_line.startswith("READY"), ready_line)

            image = np.random.default_rng(123).integers(
                0, 256, size=(72, 96, 3), dtype=np.uint8
            )
            sender = JpegChunkSender(chunk_size=160, quality=87)
            expected = sender.encode(image, quality=87)
            _frame_id, packets = sender.packets(image, cam_idx=0, quality=87)
            for packet in reversed(packets):
                sender_socket.sendto(packet, ("127.0.0.1", port))
            process.communicate(timeout=10.0)
            self.assertEqual(process.returncode, 0)
            actual = output_path.read_bytes()
            self.assertEqual(hashlib.sha256(actual).digest(), hashlib.sha256(expected).digest())
            decoded = cv2.imdecode(np.frombuffer(actual, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertIsNotNone(decoded)
            self.assertEqual(decoded.shape[:2], image.shape[:2])
        finally:
            sender_socket.close()
            if process.poll() is None:
                process.kill()
                process.wait(timeout=3.0)
            output_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
