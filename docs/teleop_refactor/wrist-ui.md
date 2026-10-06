# Wrist UI implementation

Active Unity project: `/home/yuyuan/UNITY_Project/Codex`, Unity 6000.5.6f1, Meta XR 205.0.0.
Scene: `Assets/Scenes/Teleoperation.unity`. The classic project and PC robot protocols are unchanged.

Interaction uses the existing [Meta canvas integration](https://developers.meta.com/horizon/documentation/unity/unity-isdk-canvas-integration/) and the installed SDK APIs.
Pre-change sources/scene: `/home/yuyuan/UNITY_Project/.backups/wrist-ui-20260928`.

## Existing architecture and preserved behavior

`WorkspaceShell` previously created a 1.12 m × 0.76 m world-space panel 1.3 m in front of the headset and 18 cm below it. It delegated its six feature pages to `WorkspacePages` / `WorkspaceFeaturePages`, and its buttons to the existing `WorkspaceWidgets`, theme and rounded graphics. `PointableCanvas` / `PointableCanvasModule` translated the existing Meta ray/poke interactions to Unity UI events. `WorkspaceKeypad` edited IP/port fields; the older `UpperLimbVrKeyboard` remained available for legacy fields.

The new shell reuses these pages, widgets, colors and callbacks. Six page buttons, session start/stop, recording and recalibration sit around the left wrist. Drag the outer rim with the right trigger to rotate the controls; labels stay upright. The centre is transparent and the annular ray surface rejects centre hits. Detailed pages expand beside the wrist; selecting the same page again or Close collapses them. Display → Reset wrist ring resets the dial and attachment smoothing. Floating video windows remain independent.

## Files/components

Paths below are relative to the Unity project unless otherwise noted.

- Modified `Assets/Teleop/UI/WorkspaceShell.cs`, `WorkspacePages.cs`, `WorkspaceFeaturePages.cs`: wrist ring, retained pages, updated help.
- Added `Assets/Teleop/UI/WristUISettings.cs` and `Assets/Teleop/Resources/WristUISettings.asset`: persistent tuning.
- Added `WristMount.cs`, `WristVisibilityGate.cs`: follow and pose hysteresis, immediate active-control hide.
- Added `WristInteractionSource.cs`, `WristCanvasSurface.cs`: reuse existing Meta input references; right-only controller ray / hand poke filtering, surface lifecycle.
- Added `WristRingGraphic.cs`, `WristRingSurface.cs`: annular rendering/hit testing.
- Added `WristRingRotator.cs`, `WristDetentTracker.cs`: drag angle, detent crossings and short right-controller pulses.
- Modified `Assets/Teleop/UI/WorkspaceKeypad.cs`, `Assets/Teleop/Legacy/UI/UpperLimbVrKeyboard.cs`: direct poke surfaces, press feedback, retained text callbacks.
- Modified `Assets/Teleop/Core/AppManager.cs`, `Assets/Teleop/Input/TeleopTrackingGuard.cs`: derived active-control property and hand system-gesture pause through the existing session state.
- Modified `Assets/Scenes/Teleoperation.unity`: serialized wrist shell, input source, settings, Meta ray/poke references and legacy presentation references.
- Added `Assets/Teleop/Editor/WristUIValidation.cs`: native geometry/configuration and idle Play smoke checks.
- PC repository: `scripts/teleop_refactor/run_wrist_visibility_checks.py`, `csharp/WristVisibilityHarness.cs`, `run_wrist_detents_checks.py`, `csharp/WristDetentsHarness.cs`, and this guide.
- New Unity assets include stable `.meta` files.

## Inspector tuning

Select `Assets/Teleop/Resources/WristUISettings.asset` outside Play mode. Defaults:

| Setting | Default |
| --- | --- |
| Button-centre radius | 0.13 m |
| Button width / height | 0.072 / 0.044 m |
| Rotary rim width | 0.022 m |
| Controller wrist offset | (0, 0.055, -0.035) m |
| Controller orientation | (55, 0, 0) degrees |
| Hand wrist offset | (0, 0.035, -0.025) m |
| Hand orientation | (70, 0, 0) degrees |
| Follow sharpness | 24 (exponential, time-based) |
| Rotation sensitivity | 1 |
| Detent spacing | 10 degrees |
| Haptic amplitude / duration | 0.25 / 0.015 s |
| Show / hide facing angles | 65 / 82 degrees |
| Show / hide downward angles | 35 / 55 degrees below horizontal |
| Show / hide dwell | 0.18 / 0.25 s |
| Detail page scale | 0.0007 m per UI unit |
| Detail page top-left offset | (0.22, 0.22) m in wrist plane |

Scene references are wired; no new packages, input actions or hand interactors need to be installed. Settings are applied when the shell builds on entering Play; stop/re-enter Play after changing layout settings. Device comfort and hand orientation should be tuned on the Quest.

## Input and priority

The UI detects available Meta controller/hand tracking, independently of the explicitly selected robot sender mode. Existing controller and hand references identify handedness. The right controller ray can hit the wrist UI in controller mode. Hand mode uses the existing right index `PokeInteractor`; hand rays and left pointers are rejected on these surfaces. Other application surfaces keep their existing input policy. A mode switch removes the old interactable before enabling the new one, cancelling any old selection.

`AppManager.IsTeleoperationActive` is a read-only derivation of `CanSendTeleopData` plus hands/either controller grip/active WRM clutch. It introduces no parallel session state. While it is true, the wrist presentation and its interactables/colliders are disabled immediately, with no pose dwell. Session-start notifications also trigger immediate hiding. Controller grip release allows the ring to return in a viewing pose; the existing left menu button still stops the session.

Hand control streams continuously, so the guard recognizes Meta's system gesture and pauses through existing `HandleTrackingLost`. This closes the same send gate used by every sender and requires explicit recalibration to resume. Use the Quest system gesture, then raise the wrist to stop the session, record, or recalibrate. Automatic tracking recovery never resumes robot control.

Outside active control, visibility uses wrist forward direction relative to gravity and UI front normal relative to the eyes, not world height. Separate show/hide thresholds and dwell suppress boundary jitter. Lost tracking/focus or missing required input fails closed. Hand pose uses a normalized wrist/finger frame; follow smoothing never smooths the robot state gate.

## Keyboard and rotary feedback

The keypad has its own finite Meta ray/poke surface on its visible plane. Its detail-page background interaction is disabled while typing. Button colors/press feedback come through normal Unity pointer events; digits, dot, Back, Clear, Done and existing end-edit callbacks are retained. Closing or hiding disables the keypad synchronously before destroying it, so an old touch cannot click an underlying control during the delayed destruction frame. The legacy keyboard receives equivalent direct-touch support.

The dial samples the existing right-ray intersection with the wrist plane, unwraps the angle across ±180°, and applies sensitivity to angular movement. A hysteretic detent counter emits crossings in either direction. Short pulses are explicitly stopped and separated by silence; the pulse queue is bounded to prevent a long vibration tail during fast sweeps. Releasing, hiding, losing tracking/focus, or switching to hands cancels the drag and haptics. No stationary frame produces a new detent.

## Acceptance on Quest

1. Raise/turn the left controller; check ring readability and clear centre. Lower it naturally and test both threshold boundaries without flicker.
2. Right trigger-drag the rim clockwise/counterclockwise, through angle wrap and with a stationary pointer. Check discrete clicks and immediate cancellation on release.
3. Start a controller session, grip either controller: ring vanishes and cannot receive input. Release and raise wrist: it returns. Check left-menu stop.
4. Put controllers down; poke buttons and the keypad with the right index. Check left-hand and hand-ray rejection, single key events, press feedback, digits/Back/Clear/Done.
5. In a hand-control session, verify system gesture pauses the existing send gate and the wrist UI returns; verify deliberate recalibration is required.
6. Check all six pages, recording, alignment, WRM and floating video windows; reconnect/switch modes while touching or dragging.

Local verification results are recorded below. Headset comfort, physical poke reliability and haptic feel require this device acceptance pass.

## Local verification — 2026-09-28

- Final production C# / SDK compile: **91 runtime files + 2 Editor files, exit 0, no warnings**.
- Unity 6000.5.6f1 native import/compile and `Doffy.Editor.WristUIValidation.Validate`: **passed**, including serialized rig references and transformed annular hit/miss geometry.
- Idle native Play smoke: **passed** UI construction, all six page constructors, clear/type/backspace callbacks, tracking-unavailable hidden state and no robot session started. This is a UI smoke pass, not a clean XR runtime pass: this Linux host emitted missing `UnityOpenXR`, `OVRPlugin`, `MetaXRAudioUnity` and `libdl.so` native-library exceptions. Those SDK limitations prevent claiming live controller/hand validation.
- Serialized asset audit: **849 script references, 0 errors**.
- Production detent tracker: **30 checks passed**, including direction, wrap, jitter, reset, multiple crossings and no residual pulses after a large jump.
- Production visibility gate: **15 checks passed**, including dwell, hysteresis, immediate block and no reveal at rest after control stops.
- Existing session regression suite: **TeleopCoreTests.RunAll + 25 coordinator checks passed**.

Reproduce from `/home/yuyuan/AIRO-Doffy`:

```bash
python3 scripts/teleop_refactor/compile_unity_sources.py --reference /home/yuyuan/UNITY_Project/Codex --include-editor
python3 scripts/teleop_refactor/audit_unity_project.py --package-cache /home/yuyuan/UNITY_Project/Codex/Library/PackageCache
python3 scripts/teleop_refactor/run_wrist_detents_checks.py
python3 scripts/teleop_refactor/run_wrist_visibility_checks.py
python3 scripts/teleop_refactor/run_csharp_session_checks.py
```

Native validation logs: `/tmp/doffy-wrist-editor-validation.log`, `/tmp/doffy-wrist-play-smoke.log`. Run the Editor entry points above using Unity `-batchmode -nographics -projectPath /home/yuyuan/UNITY_Project/Codex -executeMethod ... -logFile ...`; `Validate` uses `-quit`, while `RunPlaySmoke` exits itself. The smoke check is intended for an idle host without a headset.

The initial source-validation pass did not build an APK. The subsequent device deployment is recorded below. No additional scene wiring or package installation is required.


## Quest deployment — 2026-09-28

Built Android ARM64/IL2CPP successfully and installed with `adb install -r` on the connected Quest 3, preserving app data. OpenXR Android validation initially failed because additive interaction features had no base controller profile; enabled the existing `OculusTouchControllerProfile Android` in `Assets/XR/Settings/OpenXR Package Settings.asset` and rebuilt successfully. The previous settings asset and APK are backed up under `/home/yuyuan/UNITY_Project/.backups/wrist-ui-20260928/`.

- APK: `/home/yuyuan/UNITY_Project/App_output/AIRO_Doffy_refactored.apk` (93,376,271 bytes).
- Package: `org.airolab.doffy.teleoperation`, version 0.8.0 / code 8.
- Installed APK SHA-256 verified equal to fresh build: `46bedcbb87086877894153ae9cc1dcdfbaf7d8b942fd0b202a9548e6918ac678`.
- Device package last update: 2026-09-28 13:31:26.
- Launched `com.unity3d.player.UnityPlayerGameActivity`: Status ok; process remained running. No wrist UI exceptions or fatal startup errors found in captured process log.
- Build log: `/tmp/doffy-wrist-quest-build.log`; startup log: `/tmp/doffy-wrist-quest-startup.log`.

Installation and startup are verified. Physical wrist pose, poke reliability and detent feel still require the on-headset acceptance steps above.
