"""Run the actual Quest protocol code under Mono without starting Unity."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=lambda value: Path(value).expanduser().resolve(), default=os.environ.get("DOFFY_UNITY_PROJECT"), required=not os.environ.get("DOFFY_UNITY_PROJECT"))
    parser.add_argument("--mono-bin", type=lambda value: Path(value).expanduser().resolve(), default=os.environ.get("DOFFY_MONO_BIN"), required=not os.environ.get("DOFFY_MONO_BIN"))
    parser.add_argument("--package-cache", type=lambda value: Path(value).expanduser().resolve(), default=os.environ.get("DOFFY_UNITY_PACKAGE_CACHE"), required=not os.environ.get("DOFFY_UNITY_PACKAGE_CACHE"))
    parser.add_argument("--export-dir", type=lambda value: Path(value).expanduser().resolve(), help="keep compiled harness for cross-language socket checks")
    args = parser.parse_args()
    sources = [args.project / "Assets/Teleop/Protocol/JpegFrameAssembler.cs",
               args.project / "Assets/Teleop/Networking/DatagramReceiver.cs",
               args.project / "Assets/Teleop/Protocol/TcpStatePacket.cs",
               args.project / "Assets/Teleop/Protocol/TeleopWireFormat.cs",
               args.project / "Assets/Teleop/Protocol/UpperLimbHeightMapping.cs",
               Path(__file__).parent / "csharp/ProtocolHarness.cs"]
    json_library = next(args.package_cache.glob("com.unity.nuget.newtonsoft-json*/Runtime/Newtonsoft.Json.dll"))
    netstandard = args.mono_bin.parent / "lib/mono/4.5/Facades/netstandard.dll"
    with tempfile.TemporaryDirectory(prefix="doffy-csharp-") as directory:
        binary = Path(directory) / "ProtocolHarness.exe"
        subprocess.run([str(args.mono_bin / "mcs"), "-out:" + str(binary),
                        "-r:" + str(json_library),
                        "-r:" + str(netstandard),
                        *map(str, sources)], check=True)
        shutil.copy2(json_library, binary.parent / json_library.name)
        result = subprocess.run([str(args.mono_bin / "mono"), str(binary)], check=True,
                                capture_output=True, text=True)
        print(result.stdout.strip())
        if args.export_dir:
            args.export_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(binary, args.export_dir / binary.name)
            shutil.copy2(json_library, args.export_dir / json_library.name)
        guard = Path(directory) / "TrackingGuardHarness.exe"
        guard_source = next((args.project / "Assets").rglob("TeleopTrackingGuard.cs"))
        subprocess.run([str(args.mono_bin / "mcs"), "-out:" + str(guard), str(guard_source),
                        str(Path(__file__).parent / "csharp/TrackingGuardHarness.cs")], check=True)
        subprocess.run([str(args.mono_bin / "mono"), str(guard)], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
