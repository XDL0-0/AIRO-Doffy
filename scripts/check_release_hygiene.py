#!/usr/bin/env python3
"""Check tracked UTF-8 text for workstation paths, device IDs and site addresses.

This is a source-tree regression guard, not a credential or APK/history audit.
Run after staging new files so they are included in git ls-files.
"""
from __future__ import annotations

import argparse
import ipaddress
from pathlib import Path
import re
import subprocess


HOME_PATH = re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+(?:/|\b)|[A-Za-z]:[\\/]+Users[\\/]+[A-Za-z0-9_.-]+")
QUEST_SERIAL = re.compile(r"\b[12]G0[A-Z0-9]{11}\b")
LABELED_SERIAL = re.compile(
    r"(?:serial(?:[ _-]?(?:number|id))?|adb\s+-s)\s*[`\"']?\s*[:=]?\s*[`\"']?([A-Za-z0-9_-]{8,})\b",
    re.IGNORECASE,
)
IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
PRIVATE_NETWORKS = tuple(ipaddress.ip_network(parts) for parts in (
    (int(ipaddress.IPv4Address(bytes([10, 0, 0, 0]))), 8),
    (int(ipaddress.IPv4Address(bytes([172, 16, 0, 0]))), 12),
    (int(ipaddress.IPv4Address(bytes([192, 168, 0, 0]))), 16),
))
# Reviewed exceptions are scoped to an exact address AND file. No subnet exemptions.
# Keep the configured RM75 endpoint; do not rewrite robot connection defaults.
# The host .100 is the documented wired-interface example, with an env override.
REVIEWED_PRIVATE_IPS = {
    "192.168.1.18": {
        "README.md", "doffy_teleop/config.py", "doffy_teleop/recording/replay.py",
        "test_tool/freedrive.py", "test_tool/revo2_keyboard_teleop_single_send.py",
        "tests/test_realman_canfd.py", "docs/teleop_refactor/python-audit-20260928.md",
        "docs/teleop_refactor/python-fixes-20260928.md", "docs/teleop_refactor/v2-reference-inventory.md",
        "docs/release-hygiene.md", "scripts/check_release_hygiene.py",
    },
    "192.168.1.100": {
        "README.md", "doffy_teleop/config.py", "docs/teleop_refactor/python-fixes-20260928.md",
        "docs/release-hygiene.md", "scripts/check_release_hygiene.py",
    },
}


def check_text(path: str, text: str) -> list[str]:
    findings = []
    for number, line in enumerate(text.splitlines(), 1):
        rules = []
        if HOME_PATH.search(line):
            rules.append("personal absolute home path")
        if QUEST_SERIAL.search(line):
            rules.append("Quest device identifier")
        for match in LABELED_SERIAL.finditer(line):
            value = match[1]
            if any(c.isdigit() for c in value) and not value.upper().endswith("_REDACTED"):
                rules.append("literal device serial; use a redacted placeholder")
                break
        for match in IPV4.finditer(line):
            value = match[0]
            try:
                address = ipaddress.ip_address(value)
            except ValueError:
                continue
            if any(address in network for network in PRIVATE_NETWORKS):
                if path not in REVIEWED_PRIVATE_IPS.get(value, set()):
                    rules.append("site IPv4 address without a reviewed file-specific exception")
                    break
        findings.extend(f"{path}:{number}: {rule}" for rule in rules)
    return findings


def scan_repository(root: Path) -> tuple[int, list[str]]:
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    findings = []
    count = 0
    for name in filter(None, names):
        path = root / name
        if not path.is_file() or path.is_symlink():
            continue
        with path.open("rb") as stream:
            prefix = stream.read(8192)
            if b"\0" in prefix:
                continue
            data = prefix + stream.read()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        count += 1
        findings.extend(check_text(name, text))
    return count, findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    count, findings = scan_repository(args.root)
    for finding in findings:
        print(finding)
    print(f"Release hygiene: {count} tracked text files, {len(findings)} finding(s).")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
