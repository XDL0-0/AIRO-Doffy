# AIRO Doffy v0.9.7 / code 18 — release provenance

**Status: binary available; complete Unity source revision unavailable.** This
APK cannot currently be reproduced from the two public repositories. No source
commit has been guessed, reconstructed, or inferred from the distribution commit.

## Published artifact

| Field | Value / evidence |
| --- | --- |
| Distribution commit | [`a3d1233c53d82f35394f68f0c8d2faa2a4857c81`](https://github.com/XDL0-0/AIRO-Doffy/commit/a3d1233c53d82f35394f68f0c8d2faa2a4857c81), 2026-10-06 |
| APK | [AIRO_Doffy_v0.9.7_arm64_code18.apk](https://github.com/XDL0-0/AIRO-Doffy/blob/a3d1233c53d82f35394f68f0c8d2faa2a4857c81/apk/AIRO_Doffy_v0.9.7_arm64_code18.apk) |
| SHA256 | `3a1e95322c83c865ba6729a5317bcc06b60875241c78c03d8e28752db6d24a4e` — recomputed from the downloaded binary |
| Size | 92,549,501 bytes |
| Android package | `com.AIROLab.AIRODOFFY` — decoded from the APK's binary AndroidManifest.xml |
| Version / code | `0.9.7` / `18` — decoded from the APK |
| Native ABI | `arm64-v8a` — checked in the APK |
| Unity | `6000.5.6f1` — serialized-file header in `assets/bin/Data/globalgamemanagers` |
| Meta XR All-in-One | **Reported: `205.0.0`; exact build package lock unverified** — [local integration notes](../docs/teleop_refactor/unity-integration-validation.md) and [refactor notes](../docs/teleop_refactor/README.md) |
| OVRPlugin | `1.205.0` string observed in `lib/arm64-v8a/libOVRPlugin.so`; this does not establish all Meta XR package versions |
| Unity source commit | **Unknown / null**, not the distribution commit above |
| APK VCS record | `META-INF/version-control-info.textproto`: `generate_error_reason: NO_SUPPORTED_VCS_FOUND` |

Machine-readable record: [manifest.json](manifest.json). The signed APK was not
modified or re-signed during this audit. Existing signature/install/launch claims
remain historical; they were not re-tested. See the [device validation record](../docs/body_visualization/validation.md#097-发布-apk).

## Public-source audit, 2026-10-07 (Europe/Brussels)

- **AIRO-DOFFY-APP:** all advertised refs, releases, PRs and all three reachable
  commits were checked. Only `main` and `v0.6.0` were found, both pointing to
  [`994a672175af58fc4f14b92f2ca10b4595c8e0a9`](https://github.com/XDL0-0/AIRO-DOFFY-APP/tree/994a672175af58fc4f14b92f2ca10b4595c8e0a9)
  (2026-08-12); no PRs or later source refs were available. The other commits are
  `efc65da66383b262d9b56aba67ea1ca653c47efa` and
  `e05a60f534f7e80b768a4301d8431561b4d36146`. Searches of their C#, JSON, XML,
  project settings and README found no `BODY`, `0.9.7` or `versionCode` matches.
  The tip's actual Unity settings are `bundleVersion: 0.5.0` and
  `AndroidBundleVersionCode: 1`, despite the historical `v0.6.0` tag. Do not infer
  binary provenance from that tag name. Its Unity and Meta versions happen to
  match the reported toolchain; that alone is not a source-to-APK link.
- **AIRO-Doffy:** all four public branch tips and four PR head refs were fetched;
  their 126 reachable commits' trees were inspected for Unity project settings,
  package files and C# sources. `main` and `feature/realman-policy-training-updates`
  point to `a3d1233`; `AIRO-DOFFY-v2.0` to
  `9274594dadea06bc4d26f1de744971259a697a42`; `forceflowpp-policy` to
  `99ea6d3c13ba01a7ef08273201bc47082ae91021`. C# history contains component copies
  and test harnesses, not a complete Unity project. There were no tags,
  GitHub Releases or open PRs at audit time. The October artifact is a repository
  file; this document is its release note, not a newly created GitHub Release.
- **Partial BODY source exists:** [component copies at the distribution SHA](https://github.com/XDL0-0/AIRO-Doffy/tree/a3d1233c53d82f35394f68f0c8d2faa2a4857c81/docs/body_visualization/quest)
  include `BodyPoseTelemetrySender.cs`, `BodyPoseWireFormat.cs`, and
  `UpperLimbAkmManager.cs`, plus harnesses. They do not supply the full scene,
  UI, Editor build scripts, project settings or the exact resolved package lock.
- **Local-only lead:** [historical build notes](../docs/body_visualization/validation.md)
  refer to `/home/yuyuan/UNITY_Project/CodexBracelet`. That workstation was not
  accessible for this audit. This is evidence of a reported local build path,
  not confirmation that a matching clean Git commit still exists there.

Scope is public advertised refs and their reachable history. Deleted/unreferenced
commits, private refs and local files cannot be ruled out. Recover the original
build tree and records before assigning a source SHA. If the old APK cannot be
linked reliably, keep this gap and publish a newly versioned, source-pinned build.

## Checks and next release

No Unity, robot hardware or third-party Python dependencies are needed:

```bash
# Repository audit: passes only the exact known historical binary, with a warning.
python3 scripts/check_apk_release.py --allow-known-unmapped

# Release preflight: intentionally FAILS for current v0.9.7 because source is absent.
python3 scripts/check_apk_release.py

python3 -m unittest discover -s tests -p test_apk_release.py -v
```

For the next APK:

1. Publish the complete Unity build project (including scene, `.meta`, build
   scripts, `ProjectSettings`, `Packages/manifest.json` and `packages-lock.json`).
   Commit the actual build version/code settings. Build from that clean revision;
   retain the build log and source revision with the resulting APK.
2. Set `source.repository`, `source.revision` (full 40-character SHA), and
   `source.status: "pinned"` in `manifest.json`. A tag may be mentioned in release
   notes, but never substitutes for the SHA. Remove obsolete gap-only fields.
3. Update APK hash, bytes, version/code, ABI and build versions. Set
   `build.meta_xr_version_status: "source_lock_verified"` only after reading the
   pinned package lock. Update the APK allowlist in `.gitignore` when needed.
4. Update README, this release note and distribution metadata with the same
   artifact and source links. Run the strict preflight **before publishing**.
   It checks actual binary metadata/hash, rejects extra APKs, requires a full
   source SHA, retrieves public Unity files at that SHA and compares version/code,
   Unity and Meta XR All-in-One versions. Unreachable source fails closed.

The workflow audits PRs and branch pushes with an exception restricted to the
single SHA256 above. Tags and manual preflight use the strict check. Its `release`
event is a **post-publication detection**, not a blocker for GitHub's manual
Publish button. No branch protection or release permissions were changed.
Matching metadata is a consistency check, not proof of a bit-for-bit rebuild or
proof that an APK was actually compiled from the claimed revision.
