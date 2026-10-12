# Validation record

> Public copy: local paths, device identifiers and site addresses are anonymized; recorded test results and version information are retained.

Historical implementation record date: 2026-10-05; release APK information updated on 2026-10-06. The implementation and installation records below are retained; see **0.9.7 release APK** at the end for the latest release APK and **0.9.6 freezing on tracking loss** for the BODY tracking-loss freezing implementation record. Robot teleop was not run.

## Software checks

- Python: `./.venv/bin/python -m pytest -q tests/test_body_visualization.py tests/test_teleop_body_visualizer.py`: **103 passed**. Covers strict protocol parsing, actual local UDP, invalid/missing joints, stream loss, angles and projection, controls, failures before visualization startup, teleop subprocess monitoring, and exit cleanup.
- GUI: the TkAgg synthetic animation opens and closes through the exit callback; the generated local `output/body_visualization/preview.png` uses clearly labeled synthetic data and is not uploaded with the source. The four views, bottom angles, and control layout were inspected.
- C# runtime: `scripts/teleop_refactor/compile_unity_sources.py --project /path/to/unity/CodexBracelet --reference /path/to/unity/CodexBracelet --output /tmp/codex-bracelet-body-compile`: **99 runtime source files compiled successfully**.
- Protocol: `docs/body_visualization/quest/check_body_wire_format.py`: **7 strict JSON scenarios passed**, checking the installed Meta SDK's 70/84 joint names/IDs, fr-FR decimal serialization, tracking loss, non-finite numbers, valid zero coordinates, and datagram sizes of 8610–18601 bytes.
- Interoperability: the **7 datagrams** output by the production C# encoder above were sent through actual localhost UDP to the production Python `BodyReceiver`; all were accepted, with 0 rejections.

## Android build

The existing `Doffy.Editor.TeleopBuild.BuildQuest` completed the Unity Android ARM64 build and scene checks with process exit code 0.

- APK: `/path/to/unity/App_output/body-visualization/AIRO_Doffy_bracelet_body_visualization.apk`
- Package name: `org.airolab.doffy.bracelet`, version `0.9.4` / versionCode `13`, retaining the existing project version settings.
- Size: 93,415,049 bytes.
- Log: `/tmp/doffy-bracelet-body-visualization-build.log`, containing `DOFFY Quest build succeeded`.
- The APK manifest was confirmed to contain `com.oculus.permission.BODY_TRACKING` and `arm64-v8a`.

Install on the connected Quest to run the new sender:

```bash
adb install -r /path/to/unity/App_output/body-visualization/AIRO_Doffy_bracelet_body_visualization.apk
```

## Actual device checks still required

Wear the Quest, grant body tracking permission, set the local PC IP on the connection screen, and start the PC with `--visualization-only`. Confirm LIVE status and a receive rate near 25 Hz, and inspect arm raising, elbow bending, finger movement, turning, and walking individually; losing focus/removing the headset should show INVALID or subsequently STALE. Full-body poses can pass acceptance checks only after valid leg coordinates are actually received. Robot integration and real-time CAN-FD timing have not yet undergone hardware acceptance checks.

## 0.9.5 revision: independent viewing, large skeleton, and Session sending switch

- The default command runs only body UDP reception and the GUI; robot teleop requires explicit `--teleop realman|classic`. Actual TkAgg window checks confirmed no robot module imports and no teleop subprocess launch.
- A single large skeleton is the default; actual mouse events verified four-view/single-skeleton, 3D/Front/Side/Top, Zoom +/-, and Fit pose controls. The upper body fits the actual joint extent; 2D projections also display joint names and orientation axes.
- Python: the two test groups above reported **113 passed**, including new display buttons, projection labels/orientation axes, and checks that robot startup is disabled by default.
- C#: **99 runtime + 4 Editor files** compiled successfully, and 7 BODY wire checks passed.
- Unity native Play: `WristUIValidation.RunPlaySmoke` exited with code 0; log: `/tmp/doffy-bracelet-v095-body-play-smoke.log`. Actual Meta ray/poke events were routed through Canvas to the new Session button, checking default OFF with no socket/data, ON with actual local UDP, OFF immediately closing the socket with no heartbeat, re-enabling, closing/reopening the page, disabling the component, and rebuilding it. Robot session, WRM, calibration, and tracking mode retained their original states. This headless check still contains the existing logs about missing Meta native libraries and cannot replace physical interaction acceptance checks in the headset.
- Android ARM64: the Unity build exited with code 0; log: `/tmp/doffy-bracelet-v095-body-build.log`.
- APK from that build: `/path/to/unity/App_output/body-visualization-v0.9.5/AIRO_Doffy_bracelet_body_visualization.apk`, version **0.9.5 / code 14**, 93,413,329 bytes.
- Installed on Quest 3 `QUEST_SERIAL_REDACTED` using `adb install -r`, which returned Success; the device query confirmed 0.9.5/code14 and BODY_TRACKING permission granted=true.
- Cold application launch with `am start -W` returned Status ok, and the process remained running; the inspected startup logs contained no BODY sender NullReference/MissingReference or fatal exceptions.

To use it, first set and Apply the local PC IP in the headset's Session page, then click **Body data: OFF → ON**. Viewing poses alone does not require Start session. After the BODY switch is turned off, the PC displays STALE within the default 0.5 s. Real body poses and the tactile feel of the button while wearing the headset still require actual checks.

## PC modularization

- The implementation was moved by responsibility into `protocol`, `media`, `visualization`, and `runtime` in the current `doffy_teleop` package; see [module responsibilities](README.md#pc-module-responsibilities). The root-level `teleop_body_visualizer.py` is an 8-line entry point; the original `doffy_teleop.body_visualization` is a 23-line compatibility export module. The original root-level command and default independent viewing behavior remain unchanged.
- BODY, startup lifecycle, architecture boundaries, and original media/protocol/compatibility import tests: **140 passed, 1 skipped**. The skipped test is the existing Unity JPEG C# loopback test, because the `/tmp/teleop-protocol-tests.exe` test program was missing.
- New independent import tests: **2 passed**. Under `python -S`, BODY parsing/reception modules do not open sockets or load NumPy, Matplotlib, camera, WebRTC, or robot dependencies; original public protocol/media exports remain available. All **27 actual public exports** were also confirmed to point to their original definitions in the complete teleop environment.
- Running the root script from another directory and `--help` for the package entry point both passed under `python -S`. Startup tests directly verified the migrated implementation modules and original teleop script paths.
- Actual TkAgg independent window startup/close checks passed: no teleop subprocess or robot/camera/WebRTC module imports, and the window and BODY socket were cleaned up normally. The root entry point successfully generated a DEMO PNG preview.

## 0.9.6 freezing on tracking loss

- Valid SDK coordinates do not mean active tracking continues. The APP's BODY messages add `position_tracked` / `orientation_tracked`, retaining the actual SDK flags; BODY display quality uses Meta `OVRBody`'s threshold, `confidence > 0.5`. Invalidity notifications and diagnostic confidence continue to be sent on invalid tracking; WRM sampling, calibration, and robot control logic were not changed.
- The PC's `visualization/body_pose.py` caches the displayed pose. It retains history on tracking loss, low confidence, empty positions, and stream loss; when previously tracked key arm joints become untracked, it freezes the entire arm. Updates resume automatically on recovery, and the interface separately shows new-message diagnostics and `HOLDING` / `STALE`. The view stays blank until the first valid pose; different senders or joint_set values do not inherit history.
- Six Python test groups reported **170 passed**: BODY, tracked fields, freezing/recovery, startup lifecycle, independent imports, and module boundaries. Scenarios include low-confidence drooping coordinates, repeated invalidity, fallback of previously tracked arms, missing joints, projection views/zoom, recovery, and continued updates of inferred full-body legs.
- C#: **99 runtime + 4 Editor** files compiled successfully; **8 scenarios** passed the wire checks, with a maximum datagram size of 22,885 bytes; **13 scenarios** passed the production sender's pure C# adapter tests, covering confidence boundaries, focus loss/pause, recovery, abnormal numeric values, valid but untracked SDK data, and WRM diagnostics.
- The **13 datagrams** output by the production C# sender above were passed through actual localhost UDP to Python `BodyReceiver`; all were accepted, with 0 rejections, then passed to the production `BodyPoseHold` to verify freezing and recovery.
- Combined actual TkAgg window and localhost UDP checks passed: after a valid pose was received, a low-confidence drooping fallback was received; window coordinates remained unchanged and `HOLDING` was displayed. After high-quality data returned, the window returned to `LIVE` and updated coordinates; resources were released normally on exit.
- The local freezing preview `output/body_visualization/tracking_hold_preview.png` is clearly marked DEMO and is not uploaded with the source; the frozen skeleton, status, raw confidence, and frame-age displays were inspected.
- The Android ARM64 build exited with code 0; log `/tmp/doffy-bracelet-v096-body-hold-build.log` contains `DOFFY Quest build succeeded`.
- APK: `/path/to/unity/App_output/body-visualization-v0.9.6/AIRO_Doffy_bracelet_body_visualization.apk`, package name `org.airolab.doffy.bracelet`, **0.9.6 / code 15**, 93,417,833 bytes, with the BODY_TRACKING permission declaration present.
- Installation was not performed: after the build, `adb devices` did not detect a connected Quest. The last confirmed device installation record is 0.9.5/code14; the 0.9.6 APK is ready and requires connecting the device, installing it, and actually observing the SDK's arm tracking flags.

## 0.9.7 release APK

> See the [release notes](../../apk/RELEASE.md) for release versions and the Unity project entry point. The following are historical device check records; these checks were not rerun during this documentation cleanup.

- Release file: [AIRO_Doffy_v0.9.7_arm64_code18.apk](../../apk/AIRO_Doffy_v0.9.7_arm64_code18.apk), Android ARM64, version **0.9.7 / code 18**, package name `com.AIROLab.AIRODOFFY`.
- Size: 92,549,501 bytes. See the [APK manifest](../../apk/manifest.json) for machine-readable information.
- Includes BODY sending code, passed signature verification, and installed and launched successfully on Quest 3.
- BODY sending defaults to **OFF** on each launch. Set and Apply the PC address in **Teleop Config → Connection & input**, then switch **Body data: OFF → ON** in **System Setting**; independent body pose viewing does not require Start Teleop.
- The actual headset interface, interaction, and real BODY poses have not yet passed acceptance checks. Successful signature verification, installation, and launch cannot replace these checks or robot integration and CAN-FD timing acceptance checks.

Install from the repository root:

```bash
adb install -r apk/AIRO_Doffy_v0.9.7_arm64_code18.apk
```
