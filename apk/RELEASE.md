# AIRO Doffy v0.9.7 / code 18 — release provenance

**Status: public source snapshot pinned; source/binary metadata consistent.**
The complete project was published after the APK. Its original build checkout
and a bit-for-bit rebuild have not been verified; this record does not claim either.

## Published artifact and source

| Field | Value / evidence |
| --- | --- |
| Distribution commit | [`a3d1233c53d82f35394f68f0c8d2faa2a4857c81`](https://github.com/XDL0-0/AIRO-Doffy/commit/a3d1233c53d82f35394f68f0c8d2faa2a4857c81), 2026-10-06 |
| APK | [AIRO_Doffy_v0.9.7_arm64_code18.apk](https://github.com/XDL0-0/AIRO-Doffy/blob/a3d1233c53d82f35394f68f0c8d2faa2a4857c81/apk/AIRO_Doffy_v0.9.7_arm64_code18.apk) |
| SHA256 | `3a1e95322c83c865ba6729a5317bcc06b60875241c78c03d8e28752db6d24a4e` — recomputed from the binary |
| Size | 92,549,501 bytes |
| Android package | `com.AIROLab.AIRODOFFY` — decoded from the APK's binary AndroidManifest.xml |
| Version / code | `0.9.7` / `18` — decoded from the APK |
| Native ABI | `arm64-v8a` — checked in the APK |
| Public source revision | [`811d0b4f9e9d6368a7fb2402943b41eb324c7167`](https://github.com/XDL0-0/AIRO-DOFFY-APP/tree/811d0b4f9e9d6368a7fb2402943b41eb324c7167/AIRO-Doffy), published 2026-10-07 |
| Unity project path | **`AIRO-Doffy/`** inside AIRO-DOFFY-APP; the repository root is a separate historical project |
| Build entrypoint | `Doffy.Editor.TeleopBuild.BuildMetaUpdateArm64Only` |
| Unity | `6000.5.6f1` — APK serialized-file header and pinned `ProjectSettings/ProjectVersion.txt` agree |
| Meta XR All-in-One | `205.0.0` — confirmed in the pinned project's `Packages/manifest.json` and `Packages/packages-lock.json` |
| OVRPlugin | `1.205.0` string observed in APK `lib/arm64-v8a/libOVRPlugin.so`; not a complete embedded package lock |
| Original APK build revision | **Unknown / null**; the source pin above is a post-release publication |
| APK VCS record | `META-INF/version-control-info.textproto`: `generate_error_reason: NO_SUPPORTED_VCS_FOUND` |

Machine-readable record: [manifest.json](manifest.json). The signed APK was not
modified or re-signed. Existing signature/install/launch claims remain historical;
they were not re-tested. See the [device validation record](../docs/body_visualization/validation.md#097-发布-apk).

## Source publication and build selection

The [new project README](https://github.com/XDL0-0/AIRO-DOFFY-APP/blob/811d0b4f9e9d6368a7fb2402943b41eb324c7167/AIRO-Doffy/README.md)
identifies this as the v0.9.7/code18 implementation. The committed tree includes
scripts, scenes, resources, original `.meta` files, Editor build code, project
settings and a package lock. The three BODY component files below are byte-for-byte
identical to the copies in the original APK distribution commit:

| Source under `AIRO-Doffy/Assets/Teleop/` | SHA256 |
| --- | --- |
| `Protocol/BodyPoseWireFormat.cs` | `80ef95fbd5d426843359ec89fbf3a45f43f4b1f8f5c9836816bf8cc554f1ab22` |
| `UpperLimb/BodyPoseTelemetrySender.cs` | `0c3615a29fb5d99dc506b40a89163bc6d84d4fa5279994565394b19be110e07d` |
| `UpperLimb/UpperLimbAkmManager.cs` | `a577397698d2493210ab71ce0c2873cb71ff484eb8f9aa8c099b4b3fb490c1fa` |

The saved project settings have package `org.airolab.doffy.bracelet` and code
**16**. They are not the effective settings for the released Meta build.
[`TeleopBuild.cs`](https://github.com/XDL0-0/AIRO-DOFFY-APP/blob/811d0b4f9e9d6368a7fb2402943b41eb324c7167/AIRO-Doffy/Assets/Teleop/Editor/TeleopBuild.cs)
uses this call chain:

1. `BuildMetaUpdateArm64Only` calls `BuildMetaUpdateApk(MetaArm64Apk, 18, true)`.
2. That wrapper selects `com.AIROLab.AIRODOFFY` and the ARM64 asset filter.
3. `BuildQuestApk` sets version `0.9.7`, the supplied version code, ARM64 and IL2CPP.
4. The wrapper restores the saved package/code after the build.

Use **Tools → DOFFY → Build Meta ARM64-only update APK**, not the default Quest
or code16 Meta menu. The exact reviewed build script has SHA256
`4689f6ec958e184cb9705ad1f99e52ab4805ff3e5fbd26b37698e237ab777fa6`.
Its settings agree with the published APK. These checks establish a documented
association to the maintainer-published source snapshot, not the original build
checkout or binary reproducibility. The new README reports a clean Unity import
and scene validation, and explicitly says it did not build a new APK; this audit
did not repeat Unity or headset execution.

## Audit history

At the initial audit on 2026-10-07, only `main`/`v0.6.0` at
`994a672175af58fc4f14b92f2ca10b4595c8e0a9` and its two ancestors were public in
AIRO-DOFFY-APP. All four PC-repository branches, four PR heads and their 126
reachable commits contained only partial C# sources/harnesses, not the complete
Unity project. No v0.9.7 GitHub Release or source tag existed. The historical root
project's settings are version `0.5.0` / code1 despite the `v0.6.0` tag name.

The later `811d0b4` publication supersedes the **source unavailable** finding.
The old missing-source README proposal is obsolete. The complete source can now
be linked by SHA and subdirectory; the APK's lack of original VCS/build provenance
remains a separate limitation. No new tag or GitHub Release was invented to hide it.

## Check and next release

No Unity, robot hardware or third-party Python dependencies are needed:

```bash
python3 scripts/check_apk_release.py
python3 -m unittest discover -s tests -p test_apk_release.py -v
```

The check reads the real APK, verifies hash/size/package/version/code/ABI/Unity,
requires a full public source SHA and safe project subdirectory, retrieves the
pinned Unity/package files and compares effective build metadata. For the custom
Editor entrypoint, its complete script hash must match the reviewed profile;
unknown or changed build scripts fail closed and need a new review of overrides.
Direct project-settings builds continue to compare saved version/code.

All PR, push, tag and manual runs now use the same strict check. **The historical
missing-source exception has been removed.** The `release` event is post-publication
detection, not a blocker for GitHub's manual Publish button. No branch protection
or release permissions were changed.

For the next APK, commit the complete project and package lock, build from that
clean revision, retain the build log/source SHA, and record the actual entrypoint.
Update the manifest, README and this note with the same immutable source link and
APK identity. If the Editor build script changes, review its settings and update
`REVIEWED_BUILD_PROFILES`; do not relax the digest check. Run the strict preflight
before publishing. Metadata agreement remains a consistency check, not proof of a
bit-for-bit rebuild or of which checkout produced a historical binary.
