"""Exercise the release gate against the real APK and controlled source records."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import struct
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

from scripts.check_apk_release import BUILD_SCRIPT, apk_identity, check_source, validate


ROOT = Path(__file__).resolve().parents[1]


def fixture_apk(path, utf8=True):
    """Minimal Android binary XML, not a runnable or replacement application."""
    values = ["manifest", "package", "com.example.fixture", "versionName", "0.9.7", "versionCode"]
    offsets, strings = [], b""
    for value in values:
        offsets.append(len(strings))
        if utf8:
            strings += bytes([len(value), len(value)]) + value.encode() + b"\0"
        else:
            strings += struct.pack("<H", len(value)) + value.encode("utf-16-le") + b"\0\0"
    strings += b"\0" * (-len(strings) % 4)
    start = 28 + 4 * len(values)
    pool = struct.pack("<HHIIIIII", 1, 28, start + len(strings), len(values), 0,
                       0x100 if utf8 else 0, start, 0)
    pool += struct.pack("<" + "I" * len(offsets), *offsets) + strings
    attrs = b"".join(struct.pack("<IIIHBBI", 0xFFFFFFFF, name, raw, 8, 0, kind, value)
                     for name, raw, kind, value in ((1, 2, 3, 2), (3, 4, 3, 4),
                                                    (5, 0xFFFFFFFF, 0x10, 18)))
    node = struct.pack("<HHIII", 0x102, 16, 36 + len(attrs), 1, 0xFFFFFFFF)
    node += struct.pack("<IIHHHHHH", 0xFFFFFFFF, 0, 20, 20, 3, 0, 0, 0) + attrs
    binary_xml = struct.pack("<HHI", 3, 8, 8 + len(pool) + len(node)) + pool + node
    with zipfile.ZipFile(path, "w") as apk:
        apk.writestr("AndroidManifest.xml", binary_xml)
        apk.writestr("assets/bin/Data/globalgamemanagers", b"\0" * 48 + b"6000.5.6f1\0")
        apk.writestr("lib/arm64-v8a/libfixture.so", b"fixture")


class ApkReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        (self.root / "apk").mkdir()
        self.apk = self.root / "apk/fixture.apk"
        fixture_apk(self.apk)
        identity = apk_identity(self.apk)
        self.manifest = {key: value for key, value in identity.items() if key != "unity_version"}
        self.manifest.update(source={"repository": "XDL0-0/AIRO-DOFFY-APP", "status": "unavailable",
                                     "revision": None, "reason": "Controlled test fixture"},
                             build={"unity_version": "6000.5.6f1", "meta_xr_all_version": "205.0.0",
                                    "meta_xr_version_status": "reported_unverified"})
        self.write_manifest()

    def write_manifest(self):
        (self.root / "apk/manifest.json").write_text(json.dumps(self.manifest))

    def pinned(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["source"].update(status="pinned", revision="a" * 40)
        manifest["build"]["meta_xr_version_status"] = "source_lock_verified"
        files = {
            "ProjectSettings/ProjectVersion.txt": "m_EditorVersion: 6000.5.6f1\n",
            "ProjectSettings/ProjectSettings.asset":
                "PlayerSettings:\n  bundleVersion: 0.9.7\n  AndroidBundleVersionCode: 18\n",
            "Packages/manifest.json": json.dumps({"dependencies": {"com.meta.xr.sdk.all": "205.0.0"}}),
            "Packages/packages-lock.json": json.dumps({"dependencies": {
                "com.meta.xr.sdk.all": {"version": "205.0.0"}}}),
        }
        fetch = Mock(side_effect=lambda repo, sha, path: files[path])
        return manifest, files, fetch

    def test_real_binary_metadata(self):
        manifest = json.loads((ROOT / "apk/manifest.json").read_text())
        identity = apk_identity(ROOT / "apk" / manifest["file"])
        for key in ("file", "sha256", "bytes", "package", "version_name", "version_code", "abi"):
            self.assertEqual(identity[key], manifest[key])
        self.assertEqual(identity["unity_version"], manifest["build"]["unity_version"])

    def test_utf16_string_pool(self):
        fixture_apk(self.apk, utf8=False)
        identity = apk_identity(self.apk)
        self.assertEqual(identity["package"], "com.example.fixture")
        self.assertEqual((identity["version_name"], identity["version_code"]), ("0.9.7", 18))

    def test_strict_release_rejects_unavailable_source(self):
        with self.assertRaisesRegex(ValueError, "source status must be pinned"):
            validate(self.root)

    def test_binary_manifest_mismatches_fail(self):
        original = copy.deepcopy(self.manifest)
        for field, value in (("sha256", "0" * 64), ("bytes", 1), ("version_code", 19),
                             ("version_name", "0.9.8"), ("package", "com.example.wrong")):
            with self.subTest(field=field):
                self.manifest = copy.deepcopy(original)
                self.manifest[field] = value
                self.write_manifest()
                with self.assertRaisesRegex(ValueError, field):
                    validate(self.root)

    def test_extra_apk_outside_apk_directory_is_rejected(self):
        (self.root / "unrecorded.APK").write_bytes(b"unrecorded release")
        subprocess.run(["git", "add", "unrecorded.APK"], cwd=self.root, check=True)
        with self.assertRaisesRegex(ValueError, "unmanifested APK"):
            validate(self.root)

    def test_changed_binary_still_requires_a_source_pin(self):
        # A ZIP/APK can carry a trailing byte; metadata remains readable while hash changes.
        target = self.root / "apk" / self.apk.name
        with target.open("ab") as stream:
            stream.write(b"x")
        identity = apk_identity(target)
        self.manifest.update({key: identity[key] for key in ("sha256", "bytes")})
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "source status must be pinned"):
            validate(self.root)

    def test_pinned_source_success_uses_exact_revision(self):
        self.manifest, _, fetch = self.pinned()
        self.write_manifest()
        self.assertIn("PASS", validate(self.root, fetch=fetch))
        self.assertEqual(fetch.call_count, 4)
        for call in fetch.call_args_list:
            self.assertEqual(call.args[:2], ("XDL0-0/AIRO-DOFFY-APP", "a" * 40))

    def test_missing_short_and_moving_refs_fail_before_network(self):
        for revision in (None, "main", "v0.9.7", "994a672"):
            with self.subTest(revision=revision):
                manifest, _, fetch = self.pinned()
                manifest["source"]["revision"] = revision
                with self.assertRaisesRegex(ValueError, "40-character"):
                    check_source(manifest, fetch)
                fetch.assert_not_called()

    def test_old_source_settings_fail(self):
        manifest, files, fetch = self.pinned()
        files["ProjectSettings/ProjectSettings.asset"] = "  bundleVersion: 0.5.0\n  AndroidBundleVersionCode: 1\n"
        with self.assertRaisesRegex(ValueError, "bundleVersion"):
            check_source(manifest, fetch)

    def test_wrong_unity_or_package_lock_fails(self):
        for path, content, error in (
            ("ProjectSettings/ProjectVersion.txt", "m_EditorVersion: 6000.0.1f1\n", "Unity source"),
            ("Packages/packages-lock.json", '{"dependencies": {}}', "locked version"),
        ):
            with self.subTest(path=path):
                manifest, files, fetch = self.pinned()
                files[path] = content
                with self.assertRaisesRegex(ValueError, error):
                    check_source(manifest, fetch)

    def test_source_project_subdirectory(self):
        manifest, files, _ = self.pinned()
        manifest["source"]["project_path"] = "AIRO-Doffy"
        nested = {"AIRO-Doffy/" + path: text for path, text in files.items()}
        fetch = Mock(side_effect=lambda repo, sha, path: nested[path])
        check_source(manifest, fetch)
        self.assertTrue(all(call.args[2].startswith("AIRO-Doffy/") for call in fetch.call_args_list))

    def test_invalid_project_paths_fail_before_network(self):
        for path in ("../other", "/AIRO-Doffy", "AIRO-Doffy//", "foo?ref=main"):
            with self.subTest(path=path):
                manifest, _, fetch = self.pinned()
                manifest["source"]["project_path"] = path
                with self.assertRaisesRegex(ValueError, "project_path"):
                    check_source(manifest, fetch)
                fetch.assert_not_called()

    def test_reviewed_entrypoint_overrides_saved_code(self):
        manifest, files, fetch = self.pinned()
        files["ProjectSettings/ProjectSettings.asset"] = "  bundleVersion: 0.9.7\n  AndroidBundleVersionCode: 16\n"
        script = "// controlled reviewed build-profile fixture\n"
        files[BUILD_SCRIPT] = script
        method = "Doffy.Editor.TeleopBuild.BuildMetaUpdateArm64Only"
        digest = hashlib.sha256(script.encode()).hexdigest()
        manifest["build"].update(entrypoint=method, script=BUILD_SCRIPT, script_sha256=digest)
        expected = {key: manifest[key] for key in ("package", "version_name", "version_code", "abi")}
        with patch("scripts.check_apk_release.REVIEWED_BUILD_PROFILES", {(method, digest): expected}):
            check_source(manifest, fetch)
            manifest["version_code"] = 19
            with self.assertRaisesRegex(ValueError, "entrypoint version_code"):
                check_source(manifest, fetch)
            manifest["version_code"] = 18
            files[BUILD_SCRIPT] += "// changed build settings\n"
            with self.assertRaisesRegex(ValueError, "unreviewed build"):
                check_source(manifest, fetch)

    def test_unknown_entrypoint_is_rejected(self):
        manifest, files, fetch = self.pinned()
        manifest["build"].update(entrypoint="Unknown.Build", script=BUILD_SCRIPT)
        files[BUILD_SCRIPT] = "// unknown profile"
        with self.assertRaisesRegex(ValueError, "unreviewed build"):
            check_source(manifest, fetch)

    def test_unreachable_source_fails_closed(self):
        manifest, _, _ = self.pinned()
        with self.assertRaises(OSError):
            check_source(manifest, Mock(side_effect=OSError("source unreachable")))


if __name__ == "__main__":
    unittest.main()
