"""Check serialized script GUIDs and enabled scene paths without running Unity."""
from __future__ import annotations

import argparse
import os
from collections import defaultdict
import json
from pathlib import Path
import re

GUID = re.compile(r"^guid: ([0-9a-f]{32})$", re.M)
SCRIPT = re.compile(r"m_Script: \{fileID: (-?\d+), guid: ([0-9a-f]{32}), type: \d+\}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=lambda value: Path(value).expanduser().resolve(), default=os.environ.get("DOFFY_UNITY_PROJECT"), required=not os.environ.get("DOFFY_UNITY_PROJECT"))
    parser.add_argument("--package-cache", type=lambda value: Path(value).expanduser().resolve(), default=os.environ.get("DOFFY_UNITY_PACKAGE_CACHE"), required=not os.environ.get("DOFFY_UNITY_PACKAGE_CACHE"))
    args = parser.parse_args()
    local = defaultdict(list)
    package_guids = set()
    errors = []
    for path in (args.project / "Assets").rglob("*.meta"):
        match = GUID.search(path.read_text(errors="replace"))
        if match:
            local[match[1]].append(str(path.relative_to(args.project)))
            if not Path(str(path)[:-5]).exists():
                errors.append({"orphan_meta": str(path.relative_to(args.project))})
    for path in (args.project / "Assets/Teleop").rglob("*"):
        if path.suffix != ".meta" and not Path(str(path) + ".meta").exists():
            errors.append({"missing_meta": str(path.relative_to(args.project))})
    for path in args.package_cache.rglob("*.meta"):
        match = GUID.search(path.read_text(errors="replace"))
        if match:
            package_guids.add(match[1])
    for guid, paths in local.items():
        if len(paths) > 1:
            errors.append({"duplicate_guid": guid, "paths": paths})
    references = 0
    for path in (args.project / "Assets").rglob("*"):
        if path.suffix not in {".unity", ".prefab", ".asset"}:
            continue
        for file_id, guid in SCRIPT.findall(path.read_text(errors="replace")):
            if guid.startswith("0000000000000000"):
                continue  # Unity built-in resources, not external script assets.
            references += 1
            if guid not in local and guid not in package_guids:
                errors.append({"missing_script": guid, "file": str(path.relative_to(args.project))})
    build = (args.project / "ProjectSettings/EditorBuildSettings.asset").read_text()
    scenes = re.findall(r"- enabled: 1\s+path: (.+)\s+guid: ([0-9a-f]{32})", build)
    if not scenes:
        errors.append({"build": "No enabled scene"})
    for name, guid in scenes:
        path = args.project / name
        if not path.is_file() or guid not in local:
            errors.append({"missing_build_scene": name, "guid": guid})
    manifest = json.loads((args.project / "Packages/manifest.json").read_text())
    lock = json.loads((args.project / "Packages/packages-lock.json").read_text())
    meta_version = manifest["dependencies"]["com.meta.xr.sdk.all"]
    for name, value in lock["dependencies"].items():
        if name.startswith("com.meta.xr") and name not in {"com.meta.xr.sdk.audio", "com.meta.xr.sdk.voice"}:
            if value["version"] != meta_version:
                errors.append({"mixed_meta_version": name, "version": value["version"]})
    result = {"script_references_checked": references, "local_guids": len(local),
              "enabled_scenes": [name for name, _ in scenes], "meta_sdk": meta_version,
              "errors": errors, "note": "Static reference audit; Editor import and XR runtime validation are separate."}
    print(json.dumps(result, indent=2))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
