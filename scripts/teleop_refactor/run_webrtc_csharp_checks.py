"""Run real VideoSignalingClient checks against a local aiohttp WebSocket server.

The runner compiles the production signaling client and its transport helpers with Unity's
embedded Mono compiler, then drives that binary from a local server. It deliberately avoids
Unity Editor, native WebRTC, cameras, robots, and hardware.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap


REPO = Path(__file__).resolve().parents[2]
UNITY = Path(os.environ.get("DOFFY_UNITY_PROJECT", "/home/yuyuan/UNITY_Project/Codex"))
EDITOR_DATA = Path(os.environ.get("DOFFY_UNITY_EDITOR_DATA", "/home/yuyuan/Unity/Hub/Editor/6000.5.6f1/Editor/Data"))
REFERENCE = Path(os.environ.get("DOFFY_UNITY_REFERENCE_PROJECT", "/home/yuyuan/UNITY_Project/classic"))
NEWTONSOFT = next(
    (UNITY / "Library").glob("PackageCache/com.unity.nuget.newtonsoft-json*/Runtime/Newtonsoft.Json.dll"),
    next((REFERENCE / "Library").glob("PackageCache/com.unity.nuget.newtonsoft-json*/Runtime/Newtonsoft.Json.dll"), None),
)


SERVER = r'''
import asyncio
import json
import struct
import sys
from aiohttp import web, WSMsgType

state = {"concurrent": 0, "connection": 0}
unicode_message = json.dumps(
    {"type": "unicode", "session_id": "session-a",
     "payload": {"text": "分片🙂"}}, ensure_ascii=False, separators=(",", ":")
).encode("utf-8")


def frame(payload, opcode, final):
    first = opcode | (0x80 if final else 0)
    size = len(payload)
    if size < 126:
        return bytes((first, size)) + payload
    if size < 65536:
        return bytes((first, 126)) + struct.pack("!H", size) + payload
    return bytes((first, 127)) + struct.pack("!Q", size) + payload


async def fragmented_text(ws, data):
    # Split immediately before the final byte of a multi-byte UTF-8 sequence and put
    # the continuation bytes in the next WebSocket frame.
    marker = data.index("🙂".encode("utf-8")) + 1
    transport = ws._writer.transport
    transport.write(frame(data[:marker], 0x1, False))
    transport.write(frame(data[marker:], 0x0, True))
    await asyncio.sleep(0)


async def handler(request):
    ws = web.WebSocketResponse(max_msg_size=2 * 1024 * 1024)
    await ws.prepare(request)
    state["connection"] += 1
    try:
        async for message in ws:
            if message.type != WSMsgType.TEXT:
                continue
            item = json.loads(message.data)
            kind = item.get("type")
            if kind == "concurrent":
                state["concurrent"] += 1
                if state["concurrent"] == 64:
                    await fragmented_text(ws, unicode_message)
            elif kind == "close":
                await ws.close()
                break
            elif kind == "oversize":
                payload = {"type": "oversize", "session_id": "session-b",
                           "payload": {"blob": "x" * (1024 * 1024 + 128)}}
                await ws.send_str(json.dumps(payload, separators=(",", ":")))
    except Exception as exc:
        print("server handler:", repr(exc), file=sys.stderr, flush=True)
    return ws


async def main():
    app = web.Application()
    app.router.add_get("/", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    print(f"PORT {port}", flush=True)
    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()


asyncio.run(main())
'''


STUB = """
using System;
namespace UnityEngine {
    public static class Debug {
        public static void LogException(Exception exception) {
            Console.Error.WriteLine(exception.ToString());
        }
    }
}
"""


def source(name: str) -> Path:
    preferred = UNITY / "Assets/Scripts/New" / name
    if preferred.exists():
        return preferred
    matches = sorted(UNITY.glob(f"Assets/**/{name}"))
    if not matches:
        raise FileNotFoundError(name)
    return matches[0]


def compile_harness(output: Path, work: Path) -> Path:
    mcs = EDITOR_DATA / "MonoBleedingEdge/bin-linux64/mcs"
    mono = EDITOR_DATA / "MonoBleedingEdge/bin-linux64/mono"
    if not mcs.exists() or not mono.exists() or NEWTONSOFT is None:
        raise RuntimeError("Unity embedded Mono or cached Newtonsoft assembly is unavailable")
    stub = work / "UnityEngineStub.cs"
    harness = REPO / "scripts/teleop_refactor/csharp/webrtc_signaling_harness.cs"
    stub.write_text(textwrap.dedent(STUB))
    out = work / "WebRtcSignalingHarness.exe"
    websockets = EDITOR_DATA / "MonoBleedingEdge/lib/mono/4.7.1-api/Facades/System.Net.WebSockets.dll"
    websockets_client = EDITOR_DATA / "MonoBleedingEdge/lib/mono/4.7.1-api/Facades/System.Net.WebSockets.Client.dll"
    netstandard = EDITOR_DATA / "MonoBleedingEdge/lib/mono/4.7.1-api/Facades/netstandard.dll"
    args = [
        str(mcs), "-sdk:4.7.1-api", "-langversion:latest", "-target:exe", f"-out:{out}",
        f"-r:{NEWTONSOFT}", f"-r:{websockets}", f"-r:{websockets_client}", f"-r:{netstandard}",
        str(stub), str(source("VideoSignalingClient.cs")),
        str(UNITY / "Assets/Teleop/Media/SignalingEnvelopeCodec.cs"),
        str(UNITY / "Assets/Teleop/Media/WebSocketMessageReader.cs"), str(harness),
    ]
    result = subprocess.run(args, cwd=work, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError("C# harness compile failed:\n" + result.stdout + result.stderr)
    shutil.copy2(NEWTONSOFT, work / "Newtonsoft.Json.dll")
    return mono


async def run_server(python: Path):
    process = await asyncio.create_subprocess_exec(
        str(python), "-c", SERVER,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    line = await asyncio.wait_for(process.stdout.readline(), timeout=8)
    if not line.startswith(b"PORT "):
        stderr = await process.stderr.read()
        process.kill()
        raise RuntimeError("aiohttp server failed: " + stderr.decode(errors="replace"))
    return process, int(line.split()[1])


async def main() -> int:
    with tempfile.TemporaryDirectory(prefix="airo-webrtc-csharp-") as directory:
        work = Path(directory)
        mono = compile_harness(work / "WebRtcSignalingHarness.exe", work)
        server, port = await run_server(Path(sys.executable))
        try:
            result = await asyncio.create_subprocess_exec(
                str(mono), str(work / "WebRtcSignalingHarness.exe"), "127.0.0.1", str(port),
                cwd=str(work), env={**os.environ, "MONO_PATH": str(work)},
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=45)
            if stdout:
                print(stdout.decode(errors="replace"), end="")
            if result.returncode:
                print(stderr.decode(errors="replace"), file=sys.stderr, end="")
                return result.returncode
            return 0
        finally:
            server.terminate()
            try:
                await asyncio.wait_for(server.wait(), timeout=5)
            except asyncio.TimeoutError:
                server.kill()
                await server.wait()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
