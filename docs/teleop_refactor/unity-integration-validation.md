# Unity integration acceptance record

Date: 2026-09-16. New project: `/path/to/unity/Codex`; reference: `/path/to/unity/classic`. Source/API compilation used the reference project's cached Unity 6000.5.6f1, Meta XR 205.0.0, and WebRTC 3.0.0 assemblies; Editor licensing was not bypassed.

## Evidence obtained

| Validation layer | Result | Boundary |
|---|---|---|
| Runtime C# source/API | 82 source files compiled successfully, 0 errors, 0 warnings | Roslyn + existing SDK assemblies; Unity native APIs were not executed |
| Editor tool source/API | 1 build/scene-check file compiled successfully | Does not constitute an actual Editor import or Android build |
| Static asset references | 846 script references, 362 local GUIDs; 0 missing/duplicate/orphaned reference errors | Scene/prefab serialization audit; does not establish exception-free startup |
| Pure C# session coordinators | Core tests + 25 asynchronous behavior checks passed | Actual production coordinators, with test adapters for underlying services |
| Pure C# protocol/math/lifecycle | 47 checks passed | Production JPEG/TCP/wire/height mapping/socket source; excludes Meta sampling |
| Tracking guard | 10 event/lifecycle checks passed | Actual guard source, with SDK events triggered by test adapters |
| Cross-language controller UDP | C# fr-FR formatting→Python production parsing passed | 31 fields, axis values, buttons, frame; nothing sent to a robot |
| Cross-language TCP/force UDP | Python production packaging→C# production parsing passed | Known coordinate/force transforms and wxyz order match; excludes Unity Transform rendering |
| Real D435 images→C# | Three 640×480 frames captured successfully; final frame sent through production UDP chunking→production C# reassembly with the JPEG matching the original before transmission | Real local camera, real UDP, C# executed on Mono; excludes the Quest wireless link |
| Python WebRTC | Real aiortc negotiation, video decoding, and message round trips passed | localhost; excludes the Unity native peer |
| C# WebSocket | Real ClientWebSocket↔aiohttp passed | 64 concurrent sends, fragmented Chinese text/emoji, reconnection, 1 MiB limit; excludes native WebRTC |

Raw camera results are in [camera-interop-result.json](camera-interop-result.json); signaling implementation and reproduction are in [WebRTC acceptance](unity-webrtc-validation.md). Pure Core and asynchronous session validation are in [Core acceptance](unity-core-validation.md).

## Actual Editor/headset checks

The real Editor scene acceptance command was run:

```bash
/path/to/Unity/Hub/Editor/6000.5.6f1/Editor/Unity \
  -batchmode -nographics -quit \
  -projectPath /path/to/unity/Codex \
  -executeMethod Doffy.Editor.TeleopBuild.ValidateScene \
  -logFile /tmp/doffy-unity-final-validation.log
```

Result: exit code **198**. Log: `No valid Unity Editor license found. Please activate your license.`; Editor did not execute scene acceptance. ADB `devices -l` returned only the header, with no connected devices. Therefore, **Editor import, APK build, Quest execution, and end-to-end robot acceptance remain incomplete**.

This is unrelated to safety approval and does not require approval of code changes again. Continuing this part requires a valid local Unity Editor license and a connectable Quest.

## Hardware acceptance steps

1. Activate a valid Editor license, open the project and wait for package imports, then run `Tools/DOFFY/Validate scene`; in Play mode, check the new workbench and unique runtime managers, with no missing scripts/NullReference errors.
2. `Build Quest APK` and install the new APK; confirm hand/body tracking permissions, passthrough, controller ray, hand pinch/poke, keyboard, dragging, and Compact view. The SVG preview cannot substitute for this step.
3. Put the PC and Quest on the same network: switch between UDP 1→5 windows and WebRTC 1/2/3 streams at least 20 times continuously; stop/restart the PC and confirm video recovery, release of old ports, and no old frames overwriting the new session.
4. Check IP/stream-count persistence; invalid addresses must not change the destination; changing the host during an active session must first stop recording and require recalibration. Separately verify PC reception of start/stop and recording Start/Stop/Undo messages.
5. Capture C/H/HB/WRM data with a test receiver and validate sampling frequency, fields, buttons, and coordinates; initially, do not connect the robot loop that issues motion commands.
6. Send known TCP/force data to confirm position, orientation, hiding at zero force, and stale-data indicators; perform Mirror/View and manual calibration, and confirm that Confirm & resume preserves the manual reference frame.
7. Simulate headset removal, loss of focus, pause, and tracking loss individually, and confirm that pose and WRM transmission stop; recovery and successful video connection must not resume control automatically, and control must resume only after explicit calibration/confirmation.
8. Use the prepared real robot for final acceptance of control, stopping, upper-limb mapping, camera latency, and recording playback; record the devices, network, frame rate/latency, and results.

Source integrity and local transport evidence are currently available; the release status remains **a refactored version awaiting Editor/Quest acceptance**.
