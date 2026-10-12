# Unity session core acceptance

New project: `/path/to/unity/Codex`. Classic remains read-only. Original `.meta` GUIDs were preserved when moving the scripts to `Assets/Teleop`; the migration manifest is [unity-source-migration.json](unity-source-migration.json).

## Implementation

`Core/AppManager` preserves the legacy scene's serialized fields and public calls, and handles scene composition. Separate modules manage runtime configuration, IPv4/stream-count validation, legacy PlayerPrefs reads/writes, the state machine, the session lifecycle, the video lifecycle, and legacy UI compatibility bindings. The new workbench reads public state; recording and session logic no longer read button labels.

New workbench interfaces include `ConfigureConnection(host, tracks)`, `SetTrackingMode`, `SetWebRTCTrackCount`, `SetDebugVisible`, `StatusText`, `LastError`, `Changed`, and `ConfirmAlignment`.

- 8001/8003/8005, legacy PlayerPrefs keys, and existing text/binary protocols are preserved. Changing the host during an active session first stops recording at the old destination, then updates all network consumers and closes the control gate while awaiting recalibration.
- Session and video are managed separately. Stale asynchronous starts, video connection success/failure, and signaling recovery cannot clear a tracking pause. Stopping is idempotent and cleans up the reference frame, recording, and video resources.
- `CanSendTeleopData` checks session state, independent HMD/focus/tracking/pause flags, and coordinate editing state together. Starting/recalibrating also checks actual guard availability; an existing pause=false configuration cannot bypass a tracking pause.
- Controllers/Hands are mutually exclusive. Switching input or the control reference frame pauses the session; Hands disables WRM; WRM and C/H/HB are all restricted by the send gate.
- Manual coordinate editing and two-stage base placement are mutually exclusive. **Confirm & resume** on the Alignment page accepts the current manual reference frame without overwriting it with automatic head yaw. **Recalibrate** still performs explicit automatic calibration. Session startup preserves the configured manual reference frame; automatic calibration can still explicitly overwrite it. The base tool defaults to Locked and enters editing only after an explicit Place robot base action, with Cancel base placement available, preventing implicit editing from blocking startup after saving as View.
- `RecordingController` explicitly stores the requested state and sends the original Start/Stop/Undo strings; there is no PC ACK, so the UI state does not indicate remote confirmation.

## Actual C# tests

```bash
python3 scripts/teleop_refactor/run_csharp_session_checks.py
python3 scripts/teleop_refactor/run_csharp_protocol_checks.py
```

Results:

```text
PASS TeleopCoreTests.RunAll + 25 session coordinator behavior checks
PASS 10 tracking event/lifecycle checks (SDK event adapters)
PASS 47 C# protocol/lifecycle checks
```

`TeleopCoreTests.RunAll` covers invalid input, persistence of legacy preferences, and video/tracking states not overwriting each other. The 25 coordinator checks compile and execute the production `TeleopSessionCoordinator` / `TeleopVideoSessionCoordinator`, using TaskCompletionSource to control real asynchronous timing: stopping during startup, immediately replacing a startup, late video results, tracking pauses, calibration failure, recording stop/reference-frame cleanup, and Stop idempotency. Test adapters only replace the underlying video/record/frame services.

The 10 guard checks execute production `TeleopTrackingGuard`, with SDK events triggered through adapters, validating independent loss reasons, event subscription/unsubscription, and loss-of-focus state already present before the component subscribes.

The final source/API check reports **82 runtime source files + 1 Editor tool file, 0 errors, 0 warnings**; the static audit reports **846 script references, 0 errors**. These are source and local behavior evidence and cannot substitute for native execution in Unity Editor. UI, Meta sampling, the actual pose of the manual reference frame, Quest networking, and Android builds still require the steps in [Integration acceptance](unity-integration-validation.md).
