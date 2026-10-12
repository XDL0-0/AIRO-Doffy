# Classic feature migration and acceptance matrix

The baseline is the [read-only Classic inventory](classic-feature-inventory.md), rather than inferred features of the old Codex project. The new project is at `/path/to/unity/Codex`; its main scene is `Assets/Scenes/Teleoperation.unity`. All code directories in the tables below are relative to `Assets/Teleop`. `Retained` means that the source/scene path exists, **not that it has passed acceptance checks on an actual Quest**.

| ID | Migration result / new entry point | Evidence and pending acceptance checks |
|---|---|---|
| F-001 Session | `Core` state machine and coordinator; persistent Start/Stop session | State/cancellation/video failure isolation regressions; packet capture after stopping on the actual device remains pending |
| F-002 Configuration/IP/UDP | `Core` settings/store/validation; `Networking/UdpSocket`; Session page | IPv4 validation, legacy PlayerPrefs keys, port compatibility; actual cross-language UDP tested |
| F-003 Dual controllers | `Input/DualControllerSender`; 100 Hz, 31-field C protocol retained | French-locale C# formatting → actual Python parsing passed; OVR sampling frequency and physical buttons await actual device checks |
| F-004 Hands | `Input/HandTrackingSender`; H/HB, 26 bones, tracking filtering retained | Endianness/floating-point and PC hand protocol tests; actual bone sampling awaits actual device checks |
| F-005 Input mode | `Input/TrackingModeManager`; two modes on Session page | Persistence, mutually exclusive senders; Hands disables WRM; recalibration after switching |
| F-006 Reference-frame calibration | `Calibration`; Alignment page and original A+X/trigger shortcuts | Original coordinate implementation retained, PC coordinate regressions passed; Quest gizmo/pose comparison pending |
| F-007 Mirror/View/base | `Calibration`; Session/Alignment pages | Original two-stage placement, axes, and reset suppression retained; actual device drag-and-drop/coordinate comparison pending |
| F-008 WRM/AKM | `UpperLimb`; Upper limb page, original hold-A calibration | Two-stage calibration, 30 Hz, confidence/grip/recenter protocol retained; height mapping endpoints/filtering/holding on tracking loss tested; Meta body tracking awaits actual device checks |
| F-009 TCP/force | `Feedback/TCPPoseReceiver` + `ForceArrow`; Display page | Actual Python TCP packing → C# UDP/parsing passed; wxyz, finite-value/length checks; main-thread display and arrow direction await actual device checks |
| F-010 Legacy force receiver | `Feedback/ForceSensorReceiver`, inactive by default, 8013 | Source and old scene references retained; standalone sensor binary/JSON and visuals await hardware checks |
| F-011 WebRTC | `Media/WebRTC`; 1/2/3 streams on Cameras page | Actual negotiation and video decoding with Python aiortc; C# SDK API compilation; Quest native peer/textures pending, see WebRTC report |
| F-012 UDP video windows | `Media/UDP`; Cameras page Add/Close/Arrange, per-window port/resize/drag | Five windows and port allocation retained; out-of-order/duplicate/timeout/restart parsing tested; actual D435 → production C# reassembly produced an identical JPEG; Quest textures and interaction pending |
| F-013 Focus/zoom | Per-window local UV zoom; WebRTC DataChannel resolution control | Legacy 8005 resolution/focus is sent only when the old UI is explicitly bound and WRM is off, avoiding conflicts on the shared channel |
| F-014 41-point tactile | `Visualization/Tactile` + `Networking/TactileArrayStream` | Original 41-point layout, arrows, and TensorPrefab retained; already disabled in the V0.7 baseline. Requires old scene wiring and a separate port; default UI shows unavailable; does not occupy 8012, currently used by TCP |
| F-015 Recording | `Core/RecordingController`; persistent recording button, Undo on Help page | Explicit state replaces reading button text; still sends Start/Stop/Undo on 8003; Stop/host changes first stop recording at the old target; no ACK protocol, so no “PC confirmed” indication |
| F-016 Collapse/manual/style | `UI/WorkspaceShell`; Compact view, Help page | Persistent session actions; dynamic camera windows share the theme; old collapse/manual classes retained in `Legacy/Presentation` |
| F-017 Numeric keypad | `UI/WorkspaceKeypad`; IP and camera port input | Large buttons, backspace, submit/close; old keypad classes retained; Quest ray/poke input pending |
| F-018 Tracking safety | `Input/TeleopTrackingGuard` + Core send gate | 10 actual guard event/lifecycle tests (SDK event adapter); multiple loss signals latched independently, current SDK state read at initialization; recovery does not automatically clear the recalibration requirement |
| F-019 XR interaction/movement | Original Meta rig/building blocks; Pointable/Ray/Poke on new Canvas | Original left/right active/inactive configuration retained; new UI SDK205 API compilation; controllers, pinch, poke, and movement gestures await actual device checks |
| F-020 Exit/passthrough/debugging | Help/Display pages; `Diagnostics` | Passthrough action connects to the existing actual layer; exit API retained; old custom passthrough component with null references remains inactive |
| F-021 Virtual robot | `Visualization/VirtualRobot` | Original VRJS, joint order, actual/command, and Robotiq mimic retained; no active instance in the baseline, so 8011 is not occupied by default; URDF scene pending |

| Automatically created item | Handling in the new project |
|---|---|
| R-001 Control mode/World/tools | Original idempotent creation chain retained, called by Core composition root |
| R-002 TrackingGuard | Core ensures and initializes a single instance at startup |
| R-003 Upper limb | Original bootstrap retained; existing scene components are reused |
| R-004 VR keyboard | New workspace creates it on demand; old keyboard discovery by name retained for compatibility UI |
| R-005 TCP objects | Receiver's automatic creation of TCP child objects retained |
| R-006 Force arrows | ForceArrow's automatic creation of marker/head/shaft retained |
| R-007 anchor/axis/preview | Created through View/calibration/placement paths; startup does not force a change to the Mirror reference frame |

## Intentional behavior changes

- The old Codex project was backed up before deletion. The new project retains Classic's XR rig, valid prefabs, and script GUIDs, while rebuilding the application layer and UI.
- WRM and other controls share the session/tracking safety gate; enabling WRM before Start session does not send control packets. Video recovery and changes to host/input mode/reference frame do not bypass the recalibration requirement.
- UDP zoom now actually changes the displayed UV; this completes the old scene's behavior, which only changed labels. Reception can recover when the video frame counter resets to zero after the PC restarts; stale TCP feedback hides force arrows and displays STALE.
- The recording button uses explicit state, and ending the session automatically stops recording; Undo requires a second click to confirm. Protocol strings remain unchanged.
- Confirm & resume after manual adjustment, and the subsequent Start session, both retain the manual reference frame; Recalibrate explicitly switches back to the automatic reference frame. Base editing now requires explicit entry and can be exited with Cancel base placement.
- Default configuration is explicitly bound to the scene, retaining Classic's actual default IP `192.0.2.11`, 60 Hz text hand protocol, and priority for old preferences.
- Historical invalid URP/XR/tactile resources were archived to the backup directory; before/after cleanup records are in the [reference cleanup inventory](inherited-reference-cleanup.json). Current active Classic features and valid tactile resources are retained.

UI SVGs are layout previews and cannot serve as acceptance evidence for actual Unity rendering or Quest interaction. See [Unity integration acceptance](unity-integration-validation.md) for the final hardware acceptance steps.
