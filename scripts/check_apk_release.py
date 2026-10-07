#!/usr/bin/env python3
"""Check APK identity and its public Unity source pin (Python 3.10+, stdlib only)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
from urllib.request import urlopen
import zipfile


SOURCE_REPOSITORY = "XDL0-0/AIRO-DOFFY-APP"
# The published Editor entrypoint overrides saved package/code settings. Bind this
# reviewed interpretation to the entire script, rather than guessing with C# regexes.
# Any script change requires reviewing its effective build settings again.
BUILD_SCRIPT = "Assets/Teleop/Editor/TeleopBuild.cs"
REVIEWED_BUILD_PROFILES = {
    ("Doffy.Editor.TeleopBuild.BuildMetaUpdateArm64Only",
     "4689f6ec958e184cb9705ad1f99e52ab4805ff3e5fbd26b37698e237ab777fa6"): {
        "package": "com.AIROLab.AIRODOFFY", "version_name": "0.9.7",
        "version_code": 18, "abi": "arm64-v8a",
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def android_identity(data: bytes) -> dict:
    """Read only the root manifest attributes from Android's binary XML."""
    require(struct.unpack_from("<H", data)[0] == 3, "not Android binary XML")
    pos = struct.unpack_from("<H", data, 2)[0]
    strings = []

    def length(offset: int, wide: bool = False) -> tuple[int, int]:
        unit, mask = (2, 0x8000) if wide else (1, 0x80)
        first = int.from_bytes(data[offset:offset + unit], "little")
        offset += unit
        if first & mask:
            second = int.from_bytes(data[offset:offset + unit], "little")
            return ((first & (mask - 1)) << (unit * 8)) | second, offset + unit
        return first, offset

    while pos < len(data):
        kind, header_size, size = struct.unpack_from("<HHI", data, pos)
        require(size >= header_size >= 8 and pos + size <= len(data), "bad XML chunk")
        if kind == 1:  # RES_STRING_POOL_TYPE
            count, _, flags, start, _ = struct.unpack_from("<IIIII", data, pos + 8)
            for index in range(count):
                offset = pos + start + struct.unpack_from("<I", data, pos + header_size + index * 4)[0]
                count_chars, offset = length(offset, not bool(flags & 0x100))
                if flags & 0x100:
                    count_bytes, offset = length(offset)
                    strings.append(data[offset:offset + count_bytes].decode("utf-8"))
                else:
                    strings.append(data[offset:offset + count_chars * 2].decode("utf-16-le"))
        elif kind == 0x102:  # RES_XML_START_ELEMENT_TYPE
            ext = pos + header_size
            _, name, start, stride, count = struct.unpack_from("<IIHHH", data, ext)
            if strings[name] == "manifest":
                attrs = {}
                require(stride >= 20, "bad XML attribute size")
                for index in range(count):
                    offset = ext + start + index * stride
                    _, name, raw, _, _, value_type, value = struct.unpack_from("<IIIHBBI", data, offset)
                    attrs[strings[name]] = (strings[raw] if raw != 0xFFFFFFFF
                                            else strings[value] if value_type == 3 else value)
                return {"package": attrs["package"], "version_name": attrs["versionName"],
                        "version_code": int(attrs["versionCode"])}
        pos += size
    raise ValueError("APK has no root manifest element")


def apk_identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    with zipfile.ZipFile(path) as apk:
        result = android_identity(apk.read("AndroidManifest.xml"))
        abis = sorted({name.split("/")[1] for name in apk.namelist()
                       if name.startswith("lib/") and name.endswith(".so")})
        require(len(abis) == 1, "expected one native ABI")
        header = apk.read("assets/bin/Data/globalgamemanagers")[:128]
        version = re.search(rb"\x00(\d+\.\d+\.\d+[abfp]\d+)\x00", header)
        require(version is not None, "cannot identify Unity version in APK")
        result.update(abi=abis[0], unity_version=version.group(1).decode("ascii"))
    result.update(file=path.name, bytes=path.stat().st_size, sha256=digest.hexdigest())
    return result


def fetch_source_file(repository: str, revision: str, path: str) -> str:
    # Repository is allowlisted and revision validated before any network request.
    url = f"https://raw.githubusercontent.com/{repository}/{revision}/{path}"
    with urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8")


def check_source(manifest: dict, fetch=fetch_source_file) -> None:
    source, build = manifest["source"], manifest["build"]
    require(source.get("status") == "pinned", "source status must be pinned for a release")
    revision = source.get("revision")
    require(isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision) is not None,
            "source.revision must be a full 40-character commit SHA; branch/tag names are insufficient")
    require(source.get("repository") == SOURCE_REPOSITORY, "unexpected Unity source repository")
    require(build.get("meta_xr_version_status") == "source_lock_verified",
            "Meta XR version must be checked against the pinned packages-lock.json")
    require(re.fullmatch(r"\d+\.\d+\.\d+", str(build.get("meta_xr_all_version"))) is not None,
            "missing exact Meta XR All-in-One version")
    project_path = source.get("project_path", "")
    require(isinstance(project_path, str) and (project_path == "" or
            all(re.fullmatch(r"[A-Za-z0-9_.-]+", part) and part not in (".", "..")
                for part in project_path.split("/"))), "invalid source.project_path")
    prefix = project_path + "/" if project_path else ""
    read = lambda path: fetch(source["repository"], revision, prefix + path)
    project_version = read("ProjectSettings/ProjectVersion.txt")
    require(f"m_EditorVersion: {build['unity_version']}" in project_version.splitlines(),
            "Unity source version differs from APK")
    if build.get("entrypoint"):
        require(build.get("script") == BUILD_SCRIPT, "unexpected build script path")
        script = read(BUILD_SCRIPT)
        digest = hashlib.sha256(script.encode("utf-8")).hexdigest()
        profile = REVIEWED_BUILD_PROFILES.get((build["entrypoint"], digest))
        require(profile is not None, "unreviewed build entrypoint/script: review effective build settings")
        require(build.get("script_sha256") == digest, "build script hash differs from manifest")
        for key, expected in profile.items():
            require(manifest.get(key) == expected, f"build entrypoint {key} differs from APK")
    else:
        settings = read("ProjectSettings/ProjectSettings.asset")
        for key, expected in (("bundleVersion", manifest["version_name"]),
                              ("AndroidBundleVersionCode", str(manifest["version_code"]))):
            match = re.search(rf"^\s*{key}:\s*(.*?)\s*$", settings, re.MULTILINE)
            require(match is not None and match.group(1).strip("\"'") == expected,
                    f"Unity source {key} differs from APK")
    packages = json.loads(read("Packages/manifest.json"))["dependencies"]
    locked = json.loads(read("Packages/packages-lock.json"))["dependencies"]
    version = build["meta_xr_all_version"]
    require(packages.get("com.meta.xr.sdk.all") == version, "Meta XR manifest version mismatch")
    require(locked.get("com.meta.xr.sdk.all", {}).get("version") == version,
            "Meta XR locked version mismatch")


def validate(root: Path, fetch=fetch_source_file) -> str:
    manifest = json.loads((root / "apk/manifest.json").read_text())
    filename = manifest["file"]
    require(isinstance(filename, str) and Path(filename).name == filename
            and filename.endswith(".apk"), "manifest.file must be an APK basename")
    path = root / "apk" / filename
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    apk_paths = {p for p in tracked if p.lower().endswith(".apk")}
    apk_paths.update(p.relative_to(root).as_posix() for p in (root / "apk").iterdir()
                     if p.suffix.lower() == ".apk")
    require(apk_paths == {f"apk/{filename}"}, "unmanifested APK: publish exactly the manifest's APK")
    actual = apk_identity(path)
    for key in ("file", "package", "version_name", "version_code", "abi", "bytes", "sha256"):
        require(manifest.get(key) == actual[key], f"APK {key} differs from manifest")
    require(manifest["build"]["unity_version"] == actual["unity_version"], "APK Unity version mismatch")
    check_source(manifest, fetch)
    return "PASS: APK identity and public Unity source metadata agree (not a reproducible-build proof)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        print(validate(args.root.resolve()))
    except (ValueError, KeyError, IndexError, TypeError, OSError, struct.error,
            zipfile.BadZipFile, subprocess.CalledProcessError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
