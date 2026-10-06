"""Real UDP interoperability: production C# encoding ↔ production PC protocol code."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import select
import socket
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from doffy_teleop.protocol.parse_vr import parse_data
from doffy_teleop.control.contracts import pack_quest_tcp_state_packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--harness', type=Path, default=Path('/tmp/doffy-csharp-checks/ProtocolHarness.exe'))
    parser.add_argument('--mono', type=Path, default=Path('/home/yuyuan/Unity/Hub/Editor/6000.5.6f1/Editor/Data/MonoBleedingEdge/bin/mono'))
    args = parser.parse_args()
    command = [str(args.mono), str(args.harness)]
    results = []
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(('127.0.0.1', 0)); receiver.settimeout(3)
        subprocess.run(command + ['controller', str(receiver.getsockname()[1])], check=True, timeout=5)
        packet, _ = receiver.recvfrom(65535)
        left, right = parse_data(packet.decode())
        assert left['FrameId'] == 42 and left['Position'] == (1.25, -2.5, 3)
        assert right['Position'] == (-1.25, 2.5, -3)
        assert left['GripTrigger'] == 1 and right['GripTrigger'] == 0
        assert right['Button_AX'] == 1 and right['Joystick_Press'] == 1
        results.append({'transport': 'C#→UDP→Python controller decoder', 'status': 'passed', 'locale': 'fr-FR', 'fields': len(packet.split(b','))})
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(('127.0.0.1', 0)); port = probe.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix='doffy-tcp-') as directory:
        output = Path(directory) / 'sample.json'
        process = subprocess.Popen(command + ['tcp', str(port), str(output)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            if not select.select([process.stdout], [], [], 5)[0]:
                raise TimeoutError('C# TCP receiver did not become ready')
            assert process.stdout.readline().startswith('READY')
            pose = np.eye(4); pose[:3, 3] = [1, 2, 3]
            axes = np.array([[0, -1, 0], [0, 0, 1], [1, 0, 0]])
            packet = pack_quest_tcp_state_packet(pose, np.array([10, 20, 30, 0, 0, 0]), axes)
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
                sender.sendto(packet, ('127.0.0.1', port))
            _, error = process.communicate(timeout=6)
            assert process.returncode == 0, error
            decoded = json.loads(output.read_text())['rightTCP']
            assert decoded['position'] == [-2, 3, 1]
            assert decoded['rotation'] == [1, 0, 0, 0]
            assert decoded['force'] == [-20, 30, 10]
            results.append({'transport': 'Python TCP encoder→UDP→C# decoder', 'status': 'passed', 'position': decoded['position'], 'force': decoded['force'], 'quaternion_order': 'wxyz'})
        finally:
            if process.poll() is None:
                process.kill(); process.wait(timeout=3)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
