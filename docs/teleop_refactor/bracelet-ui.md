# DOFFY Bracelet variant

> Public copy: local paths, device identifiers and site addresses are anonymized; recorded results and version/hash data are unchanged.

The bracelet variant lives in `/path/to/unity/CodexBracelet`. The previous flat wrist-ring project remains in `/path/to/unity/Codex`; its saved APK is `/path/to/unity/App_output/wrist-ring-v1/AIRO_Doffy_wrist_ring_v1.apk` (SHA256 `46bedcbb87086877894153ae9cc1dcdfbaf7d8b942fd0b202a9548e6918ac678`).

The bracelet is a separate Android application: `org.airolab.doffy.bracelet`, display name **DOFFY Bracelet**, version 0.9.7 / code 16. It can coexist with `org.airolab.doffy.teleoperation`. Each package has its own Android app data and connection preferences.

## Behavior

The cuff is a hollow cylinder around the left wrist. Its local Z axis follows the wrist/fingers; its position and full orientation follow the left controller or tracked wrist. Six outward-facing control tiles are ordered **Start Teleop**, **Teleop Config**, **Camera**, **WRM Setting**, **System Setting**, and **Exit APP**. Teleop Config contains **Connection & input** and **Alignment** tabs; reference reset and manual alignment are available there. Recording is available only from the session panel. The segmented strip beside the tiles is the drag surface. The cuff is hidden when the wrist points down, tracking/focus is lost, or a session starts. Wrist roll does not hide the cuff.

Use the right controller pointer and trigger on the strip to rotate. In hand tracking, press the strip with the right index fingertip and keep touching while sliding. Rotation accumulates without end stops in either direction; releasing and grabbing again preserves the angle. Only the visual quaternion wraps at 360 degrees. Existing detent settings are retained: 10 degrees, amplitude 0.25, duration 15 ms, frequency 0.5, bounded short pulses rather than constant vibration. Hand mode uses direct poke and does not drive controller vibration.

From **Start Teleop** until teleop stops, the bracelet, details and keyboard stay hidden and non-interactive, including during video negotiation or released grips. The separate small panel has **Start recording** / **Stop recording**, **Quit teleop**, and **Undo episode**. Undo requires a second selection within four seconds. Quit calls the existing session stop path, including stopping recording. The teleop panel and camera drag bars remain interactive during teleop; the wrist settings remain hidden. Tracking pause keeps Quit and any active recording stop available; failed startup returning to Idle permits the wrist UI again. Recording state remains a local request, as in the existing app; the UI does not claim a robot-side acknowledgement.

The Teleop Config page has an **Edit IP** button beside the address. Its in-world keypad suppresses native keyboard activation, updates the address directly, and commits once on close. **Done** returns to the page; the header **Close** has a dedicated ray/poke surface while the keypad is open and closes the details directly.

Settings/detail panels follow left-wrist translation and absolute horizontal yaw, starting on the opening frame. They retain a 45-degree pitch and zero roll, independently of bracelet rotation. The panel's position offset uses that same yaw; neither head position nor viewing direction contributes a yaw offset. Near-vertical hand directions hold the last reliable yaw (world-forward if no reliable sample exists); the next reliable horizontal heading restores absolute alignment immediately.

The session panel starts 0.43 m ahead of the headset and 0.22 m below it, then keeps its world position. Hold the unlabelled bottom drag bar with the right controller trigger to translate it in 3D; right-index poke slides it along its contact plane, and lifting the finger releases it. The bar preserves the initial contact offset and freezes rotation while held. After release, only horizontal facing follows the head with exponential damping; the panel stays upright and keeps its scale. Session end resets the opening placement for the next session. Focus loss, input tracking loss, modality changes and hiding cancel drag capture.

Alignment's **Reset reference** records headset horizontal heading and world position and replaces manual alignment. **Place / replace base** uses the right stick to confirm origin and direction; **Adjust axes** toggles manual axes editing. Regeneration reuses the existing gizmo, and only the active edit preview or committed axes is displayed at once. Cancel restores the prior committed axes visibility.

System Setting groups passthrough, force vectors, robot axes, tactile display, body data, diagnostics and bracelet reset. Available toggles show their actual ON/OFF state. Exit APP opens an explicit confirmation with **Cancel** and **Exit APP**; opening or cancelling that page does not quit.

UDP views and every WebRTC single/dual/tri camera view have independent text-free bottom drag bars with hover/selection highlighting. Each view owns a root Canvas and PointableCanvas so Meta projects input along that view at any yaw, including 180 degrees. The camera layout group lives under a canvas-free floating container; transport visibility still targets the same serialized view objects and Arrange views moves the shared group. They use the same controller ray and hand-poke capture behavior as the teleop panel, keep their chosen world position, and face the headset horizontally after release. Camera controls retain their existing port, close, zoom and transport functions. Camera input depends on the selected tracked right input and XR focus, independently of the left wrist. UDP port entry opens a camera-scoped keypad; camera interaction and facing pause while typing, and Done commits once and returns to the camera.

UDP close and port controls use centered, bold single-line text sized to their actual rectangles. The close X is 42 points; the port input and placeholder are 32 points with a 60-unit input height and 6×4-unit viewport insets. These settings are reapplied after the general palette so its ellipsis policy cannot hide a glyph or truncate a port number.

## Configuration

`Assets/Teleop/Resources/WristUISettings.asset` controls the variant. Base/controller defaults: cuff radius 55 mm, tile width 36 mm, tile height 38 mm, drag strip width 20 mm. In hand mode the cuff uses an additional 0.8 scale (44 mm radius, approximately 29 × 30 mm tiles). Controller details width is 468 mm; hand-mode details width is 378 mm, including the nested keyboard. Interaction geometry is updated to the same actual metre dimensions when input mode changes. The tracked-hand offset is `(0, 0, -0.035)` metres; the fallback controller offset is `(0, 0, -0.105)` metres. Both orientation offsets are zero. The controller offset applies only when no controller-driven wrist skeleton is available.

In controller mode, the cuff is centered on the rendered left hand's wrist joint. Its axis points toward the middle knuckle, and its dorsal direction comes from the index and pinky knuckles. The scene explicitly binds the same `HandVisual` used by Meta's controller-driven hand model. The cuff follows each skeleton update directly, including render-time updates, without separate position/rotation smoothing. If the skeleton is unavailable, the controller-anchor wrist estimate is `(0, 0, -0.105)` metres; the previous 15 mm vertical offset is removed. Hand-tracking attachment and smoothing retain their existing settings.

The existing Meta Interaction rig provides the right ray and index poke. Runtime mode selection, tracking checks and focus checks remain in `WristInteractionSource`. Controller vibration uses existing OVR input APIs. No alternate XR input stack is added.

## Source map

- `WorkspaceShell.cs`, `WorkspacePages.cs`, `WorkspaceFeaturePages.cs`: six-item cuff navigation, nested teleop configuration, system toggles, exit confirmation and separate recorder root.
- `BraceletBody.cs` and `Resources/BraceletBody.shader`: hollow cuff mesh and stereo-compatible material.
- `BraceletRotator.cs`, `BraceletDragHandle.cs`: continuous rotary capture and existing detent haptics.
- `WristMount.cs`, `WristUISettings.cs`: wrist pose, input-specific sizes and roll-independent visibility.
- `WristDetailAnchor.cs`: absolute wrist yaw and its attachment offset, fixed 45-degree reading pitch and zero roll.
- `TeleopRecordPanel.cs`, `TeleopPanelDragHandle.cs`, `CameraPanelInteraction.cs`, `CameraPanelContainer.cs`, `WristInteractionSource.cs`, `WristCanvasSurface.cs`: three-button teleop panel, shared captured drag behavior, independently draggable camera views and scoped Meta input filters.
- `WorkspaceKeypad.cs`, `WorkspacePages.cs`: explicit IP entry, isolated keypad input, and single-commit close behavior.
- `Editor/TeleopBuild.cs`, `ProjectSettings/ProjectSettings.asset`: independent package identity and output.
- `Editor/WristUIValidation.cs`, `Editor/WristInputRoutingSmoke.cs`: native geometry/configuration, yaw, and SDK input-routing smoke checks.

## Build

```bash
/path/to/Unity/Hub/Editor/6000.5.6f1/Editor/Unity \
  -batchmode -nographics \
  -projectPath /path/to/unity/CodexBracelet \
  -buildTarget Android -executeMethod Doffy.Editor.TeleopBuild.BuildQuest \
  -quit -logFile /tmp/doffy-bracelet-quest-build.log
```

Output: `/path/to/unity/App_output/AIRO_Doffy_bracelet.apk`.

In-headset acceptance requires checking cuff fit, several full rotations in both directions, continuous poke across strip segments, wrist-roll following, and recording toggles during a connected robot session. Automated construction and device startup checks do not replace these physical interaction checks.

## Menu and camera interaction update — 2026-10-06

Version remains **0.9.7 / code 16**. Runtime source/API compilation passed for 102 files, Editor compilation passed for five files, and the static audit checked 850 serialized script references with no missing references. Native Play smoke passed repeated axes reuse, six ordered menu controls, nested alignment and exit cancellation, all six WebRTC views and the real UDP prefab's ray/poke routes at 180-degree yaw and after hide/re-enable, camera-facing/capture cancellation, and camera keypad single commit/bar bounds. Existing controller-wrist, teleop panel, BODY telemetry and IP keypad regressions also passed. No robot session, recording, undo or camera transport was started by these fixtures.

Android ARM64 build and replacement installation succeeded on Quest 3 `QUEST_SERIAL_REDACTED`; installed version is **0.9.7 / code 16**. Archived APK: `/path/to/unity/App_output/teleop-menu-camera-v0.9.7/AIRO_Doffy_bracelet.apk`, SHA256 `04f2b15938bec0fec1a2d2062a4c9ec7ba4d970bf6b11d1aee27a3a840d26e5f`. Play log: `/tmp/doffy-menu-camera-v097-play-smoke-r4.log`; build log: `/tmp/doffy-menu-camera-v097-build.log`. Actual controller/hand feel and visual arrangement remain headset acceptance checks.

## UDP control readability update — 2026-10-06

The actual UDP prefab was rendered under the native Editor and checked for fully visible X and port values `1`, `8000`, and `65535`. Runtime API compilation (102 files) and the existing full Play smoke passed, including camera ray/poke routing, port keypad, repeated initialization, drag/facing and controller-wrist regressions. The temporary render fixture was removed before building. No robot or camera transport was started.

Android ARM64 build and replacement installation succeeded on Quest 3 `QUEST_SERIAL_REDACTED`; installed version remains **0.9.7 / code 16**. Archived APK: `/path/to/unity/App_output/udp-labels-v0.9.7/AIRO_Doffy_bracelet.apk`, SHA256 `70317b6c121ff459898fc8c7b3d1745cbe59602245ca1ad56c6e586a3764e1ed`. The same folder contains `udp-label-preview.png`. Play log: `/tmp/doffy-udp-label-v097-play-smoke.log`; build log: `/tmp/doffy-udp-label-v097-build.log`. Readability in the headset remains a physical acceptance check.

## Verification for the initial 0.9.0 release

- C# API compilation: 95 runtime files and 2 Editor files passed.
- Scene reference audit: 849 script references, no missing references.
- Detent regression: 30 checks passed. Additional wrapped-input check passed 500 positive turns, re-grab, and 1000 negative turns without an end stop.
- Existing teleoperation core and 25 session coordinator behavior checks passed.
- Native Unity Play smoke passed: hollow cuff, 24 drag strips, nine controls, recording-only surface isolation, six retained pages, and keypad editing. This headless Linux smoke logs unavailable Meta native XR/audio plugins and a Unity Search cache warning; physical XR validation is separate.

## Initial 0.9.0 installation — 2026-09-28

The Android ARM64 build completed and was installed successfully on the connected Quest 3 (`QUEST_SERIAL_REDACTED`). The new activity launched successfully and remained running; the startup process log contained no exception/crash matches. No robot session was started as part of verification.

- New APK: `AIRO_Doffy_bracelet.apk`, 93,389,709 bytes; SHA256 `84a489c7bdbafe3ba77406c369a2695f2987b2d24f137f89a24d6c55170c308e`. Device-installed APK hash matches.
- Both package names are present on the headset. The installed original teleoperation APK still hashes to `46bedcbb87086877894153ae9cc1dcdfbaf7d8b942fd0b202a9548e6918ac678`.
- Build log: `/tmp/doffy-bracelet-quest-build.log`; Play smoke log: `/tmp/doffy-bracelet-play-smoke.log`; device startup log: `/tmp/doffy-bracelet-device-startup.log`.

## 0.9.1 changes

The cuff and details now have separate hand/controller sizes. Starting a session immediately requests the persistent three-button session panel, even while the grip is released. The wrist recording tile is removed. Controller attachment position and orientation were deliberately kept unchanged pending discussion of wrist alignment (user issue 5). The 0.9.0 bracelet APK is archived at `/path/to/unity/App_output/bracelet-v0.9.0/AIRO_Doffy_bracelet.apk`.

Direct poke acquires from fingertip contact with the drag strip, retains capture across segment boundaries, and releases after finger withdrawal. Hand rotation sensitivity is 1.3. Brief right-input tracking loss freezes capture for 100 ms and rebases it on return; longer loss requires lifting before reacquisition. Controller ray mapping selects the cylinder intersection nearest the previous angle, preserving direction at symmetric intersections and continuous rotation at grazing angles. The detent vibration settings are unchanged.

In 0.9.1, detail pages and their keyboard latched a user-facing horizontal direction when opened, with a fixed 45-degree book-like tilt. World orientation and offset remained fixed while the wrist/controller rotated. The user revised this behavior in 0.9.2 to follow whole-hand yaw.

## 0.9.1 verification

- C# API compilation: 97 runtime and 2 Editor source files passed. Scene audit: 849 references, no missing references.
- Production drag math: 25 checks passed, including multi-turn clockwise/counterclockwise motion, seam crossing, symmetric ray roots, grazing rays, strip contact/release and tracking-loss reacquisition. Existing 30 detent checks and 25 session coordinator checks also passed.
- Native Unity Play smoke passed: eight wrist controls; three session-panel actions; session visibility across start, video connection, streaming, tracking pause and stop; hand/controller size switching; fixed 45-degree orientation through parent/wrist rotation; translation following; all six retained detail pages; Edit IP → Clear/Back → `192.0.2.15` → Done (one commit); reopening and direct header Close.
- The headless Linux Play test reports unavailable Meta/OpenXR native plugins and the existing Unity Search index warning. The UI assertions completed successfully; no robot session, recording or undo operation was issued.
- Play smoke log: `/tmp/doffy-bracelet-v091-play-smoke.log`.

## 0.9.1 installation — 2026-09-28

The Android ARM64 build completed and `adb install -r` succeeded on Quest 3 `QUEST_SERIAL_REDACTED`. The installed package reports version **0.9.1**, code **10**, and the activity launch returned `Status: ok`. The application process remained present during follow-up verification. Both application packages remain installed and the original teleoperation APK hash is unchanged.

- APK size: 93,396,161 bytes; SHA256 `6377f5e499077de350eb734cbf57ecdf35fa5306ce16ea61b439197a0f545b5f`. The installed APK has the same hash.
- Original installed teleoperation APK SHA256: `46bedcbb87086877894153ae9cc1dcdfbaf7d8b942fd0b202a9548e6918ac678`.
- The startup log contains a Unity `ClassNotFoundException` probe for `com.google.android.play.core.assetpacks.AssetPackManager`, followed by successful OpenXR initialization. No fatal crash or UI `NullReferenceException`/`MissingReferenceException` appeared in the captured process log. The log also shows headset focus/pause transitions; this verifies installation/launch, not physical hand/controller interaction.
- Build log: `/tmp/doffy-bracelet-v091-quest-build.log`; device log: `/tmp/doffy-bracelet-v091-device-startup.log`.

## Controller wrist alignment — pending discussion

Controller mode currently derives the cuff from `OVRCameraRig.leftControllerAnchor` with a fixed local offset. That controller pose is not a measured wrist pose, so grip and hand-model differences can place the cuff above the wrist. No attachment offset or rotation was changed in 0.9.1. For a virtual hand, first evaluate its actual wrist-bone or hand-on-controller pose; for the real wrist, prefer a one-time position/orientation calibration saved in controller-local coordinates. A tracked wrist pose may be a better source when simultaneously available, but that availability must be verified on this headset/runtime before relying on it.

The user confirmed the mismatch is against the real wrist in passthrough. Recommend a one-time position/orientation adjustment stored relative to the controller, rather than substituting a virtual hand bone. The offset remains unchanged in 0.9.2 pending that discussion.

## 0.9.2 input fixes and verification

The user reported that 0.9.1's visible session panel could not be selected and its IP keyboard still did not accept input. The earlier smoke directly invoked button callbacks and did not cover the SDK-to-graphic routing path.

- **Session filter:** Meta stores injected interactor filters as UnityEngine.Objects and rebuilds the interface list in Awake. The plain C# recording-scope adapter became null when the initially inactive panel activated. `WristCanvasSurface` now implements `IGameObjectFilter` itself and delegates to the normal/session source rules, preserving a valid component reference through Awake and re-enable.
- **IP keyboard:** the keyboard's nested Canvas had its own PointableCanvas, but Meta accepts graphic hits only when the candidate graphic's rootCanvas matches the registered event Canvas. The keyboard now shares the host root Canvas/PointableCanvas and keeps its own bounded ray/poke surface. Header Close stays separately selectable.
- **Yaw:** 0.9.2 added whole-hand yaw following while retaining the head-oriented opening offset. Version 0.9.3 removes that offset for absolute alignment. Wrist pitch, roll and independent bracelet rotation cannot tilt the panel. The session panel stays in front of the user.
- C# API check passed for 97 runtime and 3 Editor files. Native Unity Play smoke passed, including session-filter identity after inactive construction/Awake/re-enable, ray-collider and poke-patch coverage at button centers, Meta PointerEvents → PointableCanvasModule → GraphicRaycaster → real Unity Button clicks, all three session buttons through both event paths, complete `192.0.2.15` entry using alternating ray/poke events, Back, Done (one commit), reopen and header Close.
- The fixture freezes hardware-driven presentation and sends synthetic Meta events. App/session callbacks stay gated off, so no robot session, recording or undo command is issued. It validates event routing and geometry, not physical tracking or hand feel. The Linux test still reports unavailable Meta/OpenXR native plugins and the existing Unity Search index warning.
- Play log: `/tmp/doffy-bracelet-v092-play-smoke.log`. Previous 0.9.1 APK snapshot: `/path/to/unity/App_output/bracelet-v0.9.1/AIRO_Doffy_bracelet.apk`.

## 0.9.2 installation — 2026-09-28

Android ARM64 build and `adb install -r` succeeded on Quest 3 `QUEST_SERIAL_REDACTED`. The installed package reports **0.9.2 / code 11**. Cold activity launch returned `Status: ok`. The APK is 93,403,317 bytes with SHA256 `751f1226a96d042a1a8b2118b4a1fcdab5dda987537db42ff5607a11c8aa927b`; the installed APK hash matches. Both package names remain installed and the original teleoperation APK still hashes to `46bedcbb87086877894153ae9cc1dcdfbaf7d8b942fd0b202a9548e6918ac678`.

Build log: `/tmp/doffy-bracelet-v092-quest-build.log`; device startup log: `/tmp/doffy-bracelet-v092-device-startup.log`. The startup log retains the Unity AssetPackManager class probe reported in 0.9.1; headset interaction acceptance remains a physical check.

## 0.9.3 initial yaw correction

Removed the head-to-wrist opening direction and stored yaw offset introduced in 0.9.2. Opening and reopening a settings/detail panel now directly uses the horizontal projection of the tracked controller/hand's forward direction. Both the panel rotation and horizontal position offset use that absolute heading. Fixed pitch (45 degrees), zero roll, and the independent session-panel placement remain unchanged. Near-vertical recovery resumes absolute yaw without rebasing a persistent offset. Previous APK: `/path/to/unity/App_output/bracelet-v0.9.2/AIRO_Doffy_bracelet.apk`.

The 97 runtime source files passed the C# API check. Native Unity Play smoke passed with new assertions for initial +65-degree yaw despite an off-axis head position, the matching position offset, head movement/removal, independent parent/bracelet rotation, wrist pitch/roll, translation and yaw changes, near-vertical hold, immediate absolute resynchronization, and reopening at -145 degrees. The existing session-panel and IP-keypad Meta ray/poke event-routing regression also passed. No robot control/record/undo commands were issued. Log: `/tmp/doffy-bracelet-v093-play-smoke.log`.

Android ARM64 build and installation on Quest 3 `QUEST_SERIAL_REDACTED` succeeded. The installed application reports **0.9.3 / code 12**, and its cold activity launch returned `Status: ok`. APK size: 93,397,777 bytes; SHA256 `ae16302df2e31caa5b4abe4160883995f1251c8419bcce56f6ced95b789899be`, matching the installed APK. The original teleoperation application's installed SHA256 remains `46bedcbb87086877894153ae9cc1dcdfbaf7d8b942fd0b202a9548e6918ac678`. Build log: `/tmp/doffy-bracelet-v093-quest-build.log`; startup log: `/tmp/doffy-bracelet-v093-device-startup.log`.
