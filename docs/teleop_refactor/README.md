# Airo-Doffy teleoperation refactor

Local paths and network addresses in this documentation are anonymized examples;
see [local setup and validation paths](../release-hygiene.md) before running checks.

New Unity project: `/path/to/unity/Codex`. PC project: `/path/to/AIRO-Doffy`.

The old Codex project was archived before deletion; the new project was rebuilt from Classic's XR rig, valid scene, and protocols. The Classic and AIRO-DOFFY-v2 reference projects were not modified. Meta XR All-in-One is pinned to **205.0.0**, the latest stable version verified in the official registry; Audio 85.0.0 and Voice 85.0.1 are dependencies specified by that version, rather than missed upgrades.

## Features and interface

The [complete Classic feature inventory](classic-feature-inventory.md) distinguishes 21 features and 7 runtime automatic creation mechanisms, as well as the current scene and historical optional features. [Architecture and communication contracts](architecture.md) describes module boundaries, ports, and coordinate conventions on both sides. [Migration acceptance by feature](feature-parity.md) identifies what has been verified and what still requires actual device acceptance checks.

The current spatial workspace has five pages: Teleop Config, Camera, WRM Setting, System Setting, and Exit APP; Teleop Config contains Connection & input and Alignment. The main button is Start/Stop Teleop, and a separate recording panel provides Start recording, Stop recording, and Undo episode → Confirm undo. The numeric keypad supports PC addresses and camera ports; camera windows can be moved, resized, arranged, and closed. The new workspace supports Meta ray/poke and retains the original rig's hand and controller capabilities. When robot feedback becomes stale, it is marked STALE and stale force arrows are hidden; this indication is not a robot emergency stop.

Layout previews: [Session page](workspace-session.svg), [Cameras page](workspace-cameras.svg). These SVGs are design previews, **not Unity/Quest runtime screenshots**. The actual workspace is built by `WorkspaceShell` when entering Play/running the scene.

## PC structure

The project name is **Airo-Doffy**; the Python module name uses an underscore: `doffy_teleop`.

- `doffy_teleop/config.py`, `doffy_teleop/utils.py`: central configuration, filtering, safety checks, and general helpers.
- `doffy_teleop/protocol/`: Classic C/H/HB, recording, video control, JPEG fragmentation, and signaling envelopes.
- `doffy_teleop/media/`: camera frames, UDP video, WebRTC peers, WebSocket signaling, and media orchestration.
- `doffy_teleop/sensors/`: Beaver, serial/BLE MagTouch, and six-axis force filtering.
- `doffy_teleop/control/`: input mapping, target constraints, CAN-FD loops, and remote IK.
- `doffy_teleop/robots/`: robot backends, grippers, BrainCo hands, and the Classic control implementation.
- `doffy_teleop/recording/`: data schemas, HDF5/LeRobot recording services, trajectory replay, and the recollection interface.
- `doffy_teleop/runtime/`: startup, thread lifecycles, and publishing for Classic, RealMan, teaching collection, recollection, and the BODY viewer; local `seahorse.py` is outside the upload scope.
- `doffy_teleop/visualization/`: force/tactile/camera/data dashboards, BODY plotting, and display configuration.
- `doffy_teleop/body_visualization.py`: compatibility exports for the BODY viewer API.

The public root directory retains five CLI launchers: `main.py`, `realman_teleop.py`, `realman_teachcollect.py`, `realman_recollect.py`, and `teleop_body_visualizer.py`. These commands and arguments remain available; library implementations use paths within `doffy_teleop`, and old root-level library imports need updating. `dataset_tool/` retains dataset tool CLIs, while replay and recollection UI implementations reside in `doffy_teleop/recording/`. See the [module layout guide](../module-layout.md) for directory responsibilities, migration mappings, and `python -m` commands. Teleop runtime default arguments, recording schemas, robot constraints, and existing data formats are retained.

The upload scope covers UR/RealMan teleoperation, WRM, BrainCo, Beaver, VR/cameras, BODY visualization, dataset tools, and related tests, protocol source, and documentation. Seahorse, policy training/inference/evaluation, Jev, and standalone experiments remain in the local research working tree and are outside this upload.

## Validation and reproduction

**Public source update (2026-10-07)**: the complete Unity project is now in the [`AIRO-Doffy/` subdirectory at pinned commit `811d0b4`](https://github.com/XDL0-0/AIRO-DOFFY-APP/tree/811d0b4f9e9d6368a7fb2402943b41eb324c7167/AIRO-Doffy), including scenes, `.meta` files, build scripts, and the package lock. The commands below are historical local validation commands; replace their paths with your actual checkout paths before use. code18 corresponds to `Doffy.Editor.TeleopBuild.BuildMetaUpdateArm64Only`, rather than the default code16 build. See the [release notes](../../apk/RELEASE.md) for version and build information.

Python acceptance results for this uploaded subset: **433 tests and 12 subtests passed, 3 skipped**. Tests were selected for the retained teleop, BODY, dataset tools, and related tests; no external simulation directory was added to the import path. This result is separate from the historical full working-tree regression record below and does not replace Unity/Quest or robot hardware acceptance checks.

The following commands reproduce the historical refactor-specific validation. The original validation used the workstation's existing dependency environment; pytest was installed separately in `/tmp/airo-teleop-qa` at the time, without changing the existing conda environment.

```bash
# Production C# protocols, filtering, and event lifecycles; SDK events use a test adapter.
python3 scripts/teleop_refactor/run_csharp_protocol_checks.py --export-dir /tmp/doffy-csharp-checks

# Core state, preference persistence, and 25 asynchronous session timing checks.
python3 scripts/teleop_refactor/run_csharp_session_checks.py

# Actual bidirectional UDP: controller data, TCP coordinates, and force.
/tmp/airo-teleop-qa/bin/python scripts/teleop_refactor/csharp_interop.py

# JPEG out-of-order/packet-loss loopback, production C# receiver; RealSense opens only when explicitly requested.
/tmp/airo-teleop-qa/bin/python scripts/teleop_refactor/media_loopback.py \
  --csharp --harness /tmp/doffy-csharp-checks/ProtocolHarness.exe --realsense --frames 3

# Source/API compilation and static reference checks against existing Unity/Meta cached assemblies.
python3 scripts/teleop_refactor/compile_unity_sources.py --include-editor
python3 scripts/teleop_refactor/audit_unity_project.py
```

Historical full working-tree regression record: 698 tests and 95 subtests passed, with 1 existing policy configuration/test mismatch failure (`ACTION_STEPS` 4 versus the expected 8); that value was already present in the pre-refactor snapshot. This historical record includes local research tests outside the current upload scope and cannot replace the acceptance results for this uploaded subset above. C# checks include 47 protocol/math/lifecycle, 10 tracking event, and 25 asynchronous session checks; Core settings/state tests also passed.

See the evidence in [PC](pc-refactor-validation.md), [media](media-validation.md), [independent acceptance review](acceptance-review.md), and [Unity integration](unity-integration-validation.md).

## Unity / Quest acceptance entry point

Open the new project with Unity **6000.5.6f1**, open `Assets/Scenes/Teleoperation.unity`, and enter Play to view the workspace. The `Tools → DOFFY → Validate scene` menu checks the scene; `Build Meta ARM64-only update APK` builds a v0.9.7/code18 Android ARM64 / IL2CPP package. The regular `Build Quest APK` uses a different package name and code16.

During early refactor validation, the local Unity Editor reported a missing license and ADB detected no Quest. The currently released [v0.9.7/code18 ARM64 APK](../../apk/AIRO_Doffy_v0.9.7_arm64_code18.apk) has passed signature verification and Quest 3 installation/launch checks and includes BODY sending code that defaults to off and must be enabled in System Setting; see [BODY validation](../body_visualization/validation.md#097-release-apk). Acceptance checks for the actual Quest headset interface, wireless video, ray/hand interaction, real BODY poses, and robot hardware are still incomplete. Source/API compilation, local transport, and installation/launch checks do not replace these acceptance checks; incomplete items are not marked as passed.

## Backups and existing work

- Old Codex: `/path/to/unity/.backups/codex-before-refactor-20260916.tar.gz`
- PC working-tree source/test baseline: `/path/to/unity/.backups/pc-teleop-baseline-20260916.tar.gz`
- Cleanup of invalid assets inherited from Classic: [details](inherited-reference-cleanup.json), with the original content archived separately.
- The pre-refactor git status is retained locally in `pre-refactor-git-status.txt` to distinguish existing training, evaluation, data, and paper changes; it is not uploaded with the source repository. These existing changes were neither reverted nor incorporated into this refactor.
