# Teleoperation rebuild

## Boundaries

- `UNITY_Project/classic` and `AIRO-DOFFY-v2` are read-only references.
- `UNITY_Project/Codex` is the rebuilt Quest app. Preserve the proven Meta XR rig, passthrough and asset GUIDs while replacing the app shell and reorganizing behavior by feature.
- The active PC repository is `AIRO-Doffy`. Keep the eight existing CLI commands working while moving implementations into `doffy_teleop`; library imports follow the [module migration guide](../module-layout.md). Preserve current robot limits, state validation, recording schema, hand and WRM behavior.
- Policy training, checkpoints, datasets and manuscript work are outside this refactor.

## Communication contract

The default profile remains compatible with the working classic app and PC runtime:

| Direction | Port | Content |
|---|---:|---|
| Quest → PC | UDP 8001 | Controller `C,...`, hand `H,...` / `HB,...` input |
| Quest → PC | UDP 8003 | Legacy recording `Start`, `Stop`, `Undo` |
| Quest → PC | UDP 8005 | WRM JSON; legacy video controls when WRM is disabled |
| PC → Quest | UDP 8000, 8002, 8004, 8006, 8008 | JPEG chunks: big-endian `uint32 frame, uint16 chunk, uint16 count, uint32 size` |
| PC → Quest | UDP 8012 | TCP pose and wrench JSON; quaternion order w,x,y,z |
| PC → Quest | UDP 8011 | Optional virtual robot joint states |
| Both | WebSocket 8765 | Existing WebRTC session envelope, ordered camera tracks, control DataChannel |

Additions must not silently replace or collide with these channels. New command acknowledgements must report applied state; legacy UDP send completion is never reported as PC confirmation. Network IP settings must flow to every sender. Keep retry cancellation and start/stop idempotent. Video failure must not corrupt recording state or auto-resume paused robot control.

## UI direction — DOFFY / workspace

A calm graphite workspace with ivory typography, restrained mint active states, amber attention states and red reserved for stopping/recording. A stable world-space panel with large targets and explicit text labels. Keep the passthrough environment visible and movable camera windows outside the controls.

- Persistent rail: Session, Cameras, Alignment, Upper limb, Display, Help.
- Persistent status strip: connection, input tracking, video state, recording state.
- Primary session action and recording control remain accessible; use explicit state instead of reading button labels as state.
- Session: host input with validation, controller/hand selection, stream start/stop, recalibration recovery.
- Cameras: UDP/WebRTC selection, one/two/three WebRTC tracks, UDP windows, zoom and movable views.
- Alignment: clear step-by-step reference calibration and robot base placement; preserve existing shortcuts.
- Upper limb: WRM enable, down/horizontal calibration instructions and progress, confidence/alpha, clutch/recenter guidance.
- Display: passthrough, force/tactile visibility, debug details and UI placement.
- Help: contextual shortcuts and troubleshooting without hiding essential controls.

Use the existing Meta ray interaction integration. Preserve hand/controller interaction and provide panel repositioning; avoid forced head-follow behavior. Do not change the scale of calibrated world objects when resizing UI.

Design references: [Meta hands UI](https://developers.meta.com/horizon/design/hands-ui-best-practices/), [MR design considerations](https://developers.meta.com/horizon/design/mr-design-guideline/), [Unity UI optimization](https://create.unity3d.com/Unity-UI-optimization-tips).

## Validation layers

1. Existing PC behavior tests before and after extraction; meaningful parser/lifecycle regressions.
2. Actual localhost sockets for controller, hand, WRM, wrench and video fragmentation, including malformed/out-of-order packets and repeated start/stop.
3. Actual WebRTC negotiation and frame decoding with generated camera frames; camera capture test when accessible.
4. C# compile against installed Unity/Meta assemblies and scene GUID/reference audit.
5. Licensed Unity Editor compile, scene validation, Android build and Quest camera roundtrip. This layer requires an active Editor license and an available headset.

Only layer-specific evidence may be claimed. A source compile is not an Editor/Android build; a synthetic video roundtrip is not Quest hardware verification.
