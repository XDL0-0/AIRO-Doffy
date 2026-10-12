# Teleoperation rebuild — 2026-09-16

## Completed source work

- Classic/v1/v2 inventory completed: 21 features and 7 automatic runtime creation mechanisms; the current scene and historical optional capabilities are recorded separately.
- The old Codex project was archived, the archive was verified, and the project was then deleted; it was rebuilt from Classic's valid scene/rig/assets. The reference project remains read-only.
- Meta XR All-in-One is pinned to the latest stable version, 205.0.0, with its required Audio/Voice dependencies preserved; an Android ARM64/IL2CPP build entry point is provided.
- Rebuilt the six-page spatial workbench, numeric keyboard, and persistent session/recording actions; preserved video windows, force/TCP, calibration, upper-limb functionality, and the original XR rig. A clearly labeled SVG design preview is provided.
- 52 original custom scripts were migrated to feature directories with their GUIDs preserved; protocol, network, Core, media, and UI modules were added; inherited stale assets were cleaned up and archived separately.
- The PC code was split into protocol/media/control/robots/runtime modules, with root CLI/import entry points and replacement hooks preserved. Robot constraints, state/timeouts, CAN-FD ownership, and the recording schema are preserved.
- Final UI review fixes: startup preserves an explicit manual reference frame, and base editing is locked by default with a cancel action available; source/API compilation and session regression checks passed again.
- Fixed locale-dependent number formatting, UDP chunking/PC restart recovery, stale feedback, asynchronous video cancellation/callbacks, initial DataChannel messages, and send gates for tracking and reference-frame editing.

## Validation results

- Full repository: 698 passed, 95 subtests passed, 1 failed. The only failure is the pre-refactor policy ACTION_STEPS configuration of 4 / test expectation of 8; unrelated policy configuration was not changed.
- C#: API compilation of 82 runtime + 1 Editor files, 0 errors/warnings; static audit of 846 script references, 0 errors.
- Production pure C#: 47 protocol/math/lifecycle checks, 10 tracking event checks, and 25 asynchronous session behavior checks passed; Core settings/state tests passed.
- Real C# ClientWebSocket↔Python aiohttp, Python aiortc WebRTC loopback, and bidirectional controller/TCP UDP all passed.
- Real D435 three-frame capture and transmission of the final frame to the production C# JPEG assembler passed: 640×480, 38 chunks, with the JPEG matching the original before transmission.

## Incomplete hardware acceptance

Real Unity Editor acceptance returned exit 198: no valid Editor license; ADB had no connected Quest. Editor import/Play, APK build, Quest texture/interaction/wireless latency, and end-to-end robot acceptance remain incomplete. Source compilation and local transport evidence do not substitute for these acceptance checks, and release acceptance has not been marked as passed.

Once a valid local Unity license and a Quest are available, follow `unity-integration-validation.md`.

## Backups

- `/path/to/unity/.backups/codex-before-refactor-20260916.tar.gz`
- `/path/to/unity/.backups/pc-teleop-baseline-20260916.tar.gz`
- `/path/to/unity/.backups/codex-inherited-stale-assets-20260916`

The PC baseline comes from the working tree at the start of the task, rather than git HEAD; it preserves the user's existing experiment, evaluation, data, and paper changes. No robot control or policy training tasks were started.
