"""Compile rebuilt C# against cached SDK assemblies; does not run Unity Editor.

This is a source/API check, not an Android build or a license substitute.
Unity's actual Editor/build validation is a separate required check.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=lambda value: Path(value).expanduser().resolve(), default=os.environ.get("DOFFY_UNITY_PROJECT"), required=not os.environ.get("DOFFY_UNITY_PROJECT"))
    parser.add_argument("--reference", type=lambda value: Path(value).expanduser().resolve(), default=os.environ.get("DOFFY_UNITY_REFERENCE_PROJECT"), required=not os.environ.get("DOFFY_UNITY_REFERENCE_PROJECT"))
    parser.add_argument("--editor-data", type=lambda value: Path(value).expanduser().resolve(), default=os.environ.get("DOFFY_UNITY_EDITOR_DATA"), required=not os.environ.get("DOFFY_UNITY_EDITOR_DATA"))
    parser.add_argument("--include-editor", action="store_true", help="also check the scene validator and build entry point")
    parser.add_argument("--output", type=lambda value: Path(value).expanduser().resolve(), default=Path("/tmp/airo-teleop-csharp"))
    args = parser.parse_args()
    candidates = list((args.reference / "Library/Bee/artifacts").glob("*E.dag/Assembly-CSharp.rsp"))
    if not candidates:
        parser.error("No cached Editor C# reference response file is available")
    template = max(candidates, key=lambda path: path.stat().st_mtime)
    lines = []
    for line in template.read_text().splitlines():
        if line.startswith(('"Assets/', '-out:', '-refout:', '/additionalfile:', '-analyzer:')):
            continue
        lines.append(line)
    args.output.mkdir(parents=True, exist_ok=True)
    lines.append(f'-out:"{args.output / "Assembly-CSharp.dll"}"')
    sources = sorted(path for path in (args.project / "Assets").rglob("*.cs")
                     if "Editor" not in path.parts and "Tests" not in path.parts)
    lines.extend(f'"{path}"' for path in sources)
    response = args.output / "runtime.rsp"
    response.write_text("\n".join(lines) + "\n")
    compilers = sorted((args.editor_data / "DotNetSdk/sdk").glob("*/Roslyn/bincore/csc.dll"))
    result = subprocess.run([str(args.editor_data / "DotNetSdk/dotnet"),
                             str(compilers[-1]), "@" + str(response)], cwd=args.reference)
    print(f"C# source/API check: {len(sources)} source files, exit {result.returncode}.")
    if result.returncode == 0 and args.include_editor:
        editor_template = template.with_name("Assembly-CSharp-Editor.rsp")
        editor_lines = []
        for line in editor_template.read_text().splitlines():
            if line.startswith(('"Assets/', '-out:', '-refout:', '/additionalfile:', '-analyzer:')):
                continue
            if line.startswith('-r:') and line.rstrip('"').endswith('/Assembly-CSharp.dll'):
                continue
            editor_lines.append(line)
        editor_lines.append(f'-r:"{args.output / "Assembly-CSharp.dll"}"')
        editor_lines.append(f'-out:"{args.output / "Assembly-CSharp-Editor.dll"}"')
        editor_sources = sorted(path for path in (args.project / "Assets").rglob("*.cs")
                                if "Editor" in path.parts and "Tests" not in path.parts)
        editor_lines.extend(f'"{path}"' for path in editor_sources)
        editor_response = args.output / "editor.rsp"
        editor_response.write_text("\n".join(editor_lines) + "\n")
        result = subprocess.run([str(args.editor_data / "DotNetSdk/dotnet"),
                                 str(compilers[-1]), "@" + str(editor_response)], cwd=args.reference)
        print(f"Editor tooling source/API check: {len(editor_sources)} files, exit {result.returncode}.")
    print("This check does not validate scene execution or Android packaging.")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
