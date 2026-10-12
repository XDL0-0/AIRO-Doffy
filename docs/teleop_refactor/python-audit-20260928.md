# Python teleoperation audit (2026-09-28)

> Public copy: local paths, device identifiers and site addresses are anonymized; recorded test results and version information are retained.

> This document preserves the findings and reproduction records from before the fixes. For subsequent code fixes, network restoration, and validation results, see
> [Fix validation record](python-fixes-20260928.md); the old addresses, failure results, and test counts below do not describe the state after the fixes.

This audit examined the current workspace code, covering Python teleoperation, the shared recording service, sensor interfaces, and their correspondence with the current `CodexBracelet` Unity protocol. It did not start real robot control, delete existing datasets, or modify production logic.

**Conclusion: the core separation and local communication are largely effective, but the current system cannot be accepted as having “all features working.”** The relevant existing tests were run in three groups, with 129 unique tests passing; additional integration checks found issues absent from the original coverage, including a UDP recording deadlock and incorrect LeRobot statistics after Undo. The real-device network was also not yet connected.

## Validation scope and environment

- Runtime environment: `/path/to/source-env/bin/python`, Python 3.10.0.
- Test environment: `/tmp/airo-teleop-qa/bin/python`, inheriting site-packages from the environment above.
- Unity protocol source: `/path/to/unity/CodexBracelet/Assets/Teleop`.
- Robot control tests used a fake backend; recording/deletion tests used temporary directories only.
- Real-hardware checks covered Quest network diagnostics and RealSense frame acquisition; no robot reset, motion, gripper, or dexterous-hand actions were performed.

## Completed validation

| Scope | Result | What it actually establishes |
|---|---|---|
| Core control, mapping, startup dependency injection, module boundaries, BrainCo algorithms, camera interface, visualization | 100 tests passed | Logic verified using simulated hardware; does not establish a successful real-robot closed loop |
| Protocol, UDP image chunking, WebRTC components, basic UDP interface, C# source contracts | 12 tests passed | Existing communication tests pass; lifecycle issues remain outside the original test coverage |
| UDP control/controllers/hands | Local loopback passed | Actual UDP sockets; C, H, HB, Start/Stop/Undo, three zoom ports |
| Python → current C# state packets | Local loopback passed | `rightTCP` position, wxyz quaternion, and force data accepted by the current C# parser |
| WebSocket + WebRTC video | Local loopback passed | Production Python signaling service completed hello/ack, offer/answer, real aiortc connection and video decoding, and stop_video cleanup |
| Current Unity C# image frame assembly | Local loopback passed | 37 JPEG chunks sent by Python reconstructed by the current C# assembler, with the image matching the sent result |
| HDF5 / LeRobot normal Undo path | Partially passed | Files/episode metadata/numbering and continued recording pass; LeRobot aggregate statistics still have a confirmed error, described below |
| Real RealSense D435 | Passed | 640×480 RGB uint8; 10 samples produced 10 distinct acquisition timestamps; all frame acquisition threads exited after close |
| Major Python module imports | Passed | The base environment can import main, realman_teleop, udp, WebRTC_udp, dataset, brainco_hand, and beaver |
| Current Quest ↔ PC network | Not established | No response to bidirectional ping; TCP probes from Quest to both actual PC interface addresses timed out, and no UDP marker packets arrived |
| Current RealMan API network | Not established | Connection to configured address `192.168.1.18:8080` timed out; no robot commands were sent |
| Real Beaver / BrainCo / BLE tactile hardware | Incomplete | No Beaver USB serial port identified, robot API disconnected, BLE not tested on hardware; algorithm/parser tests do not replace actual hardware validation |

The local WebRTC test negotiated a DataChannel but did not send a zoom payload through it. Hand H/HB validation used the 26-joint layout defined by the current source; the real hand acquisition sender that depends on the OVR SDK was not exercised. Current Unity can generate packets with fewer than 26 joints, while Python strictly requires 26; whether such frames occur on actual devices still requires testing.

## Current environment blockers

`config.py:11-18` configures PC as `192.0.2.16`, VR as `192.0.2.17`, and robot as `192.168.1.18:8080`. The actual PC has `192.0.2.12/22` and `192.0.2.10/22`; Quest is `192.0.2.14/21`.

Directly attempting to bind `Config.PC_IP` returned **`OSError: [Errno 99] Cannot assign requested address`**. This confirms that the current default configuration cannot start a receiving socket normally on this computer. Different subnets alone do not necessarily imply loss of connectivity; actual bidirectional ping and probes from Quest to temporary TCP/UDP listeners on both PC interfaces also failed. Authorized USB ADB establishes only that the USB debugging connection is available.

First restore a routable network between devices, then configure the Python PC/VR addresses and the target PC address on Quest. The robot controller must be able to reach the configured PC address to send realtime state pushes. End-to-end communication cannot currently be declared fully established.

## Reproduced code issues

### P1: continuing recording after Undo in the default LeRobot format produces incomplete statistics

`dataset.py:867-870` deletes `meta/stats.json` without rebuilding statistics from the retained episodes. When statistics are absent, the currently installed LeRobot uses the statistics of the next saved episode directly as those of the entire dataset.

Reproduced with the real LeRobot API, a temporary directory, and 7-dimensional state/action: save one frame per episode, first value 0, then value 10; Undo deletes value 10; then save value 20. The final retained data consists of two frames, values 0 and 20:

| Field | Expected | Actual |
|---|---|---|
| `info.total_frames` | 2 | 2 |
| state/action `stats.count` | 2 | 1 |
| state/action `stats.mean` (per dimension) | 10 | 20 |
| state/action `stats.min` (per dimension) | 0 | 20 |
| state/action `stats.std` (per dimension) | 10 | 0 |

Thus, “saving still works after Undo” and “stats.json reappeared” are insufficient evidence of dataset correctness. This affects training normalization that relies on those statistics. Rebuild aggregate statistics from the remaining episodes' statistics or data, and add a regression test for “Undo → continue recording → compare statistics across all retained data.”

### P1: recording control in UDP transport mode deadlocks by acquiring the same lock twice

- `realsense_camera.py:31` creates a non-reentrant `threading.Lock()`.
- `doffy_teleop/media/udp.py:124` uses it as the manager lock; the state property setters at `:150-158` acquire it again.
- `ManagerRecordingControl` in `data_recording.py:112-145` holds that same lock before writing these properties.

Using real `UDPManagerCore` with camera/socket stubs disconnected from hardware, `_apply_record_control('Undo')` first simulated receipt of a Quest command, followed by a real HDF5 rollback. The last file was deleted and the count decreased, but `process_pending_once()` stalled in `clear_rollback()`. The thread remained alive after a bounded wait, and neither the rollback flag nor `pause_event` was cleared. `start_recording()` reproduces the same deadlock.

Default WebRTC uses RLock when creating its own camera, so this deadlock was not reproduced there; injecting a camera object using a plain Lock still poses the same risk. A fix should define lock ownership, operate through atomic control-state methods, or consistently use reentrant locks, avoiding reliance on a stub happening to use RLock.

### P2: recording commands use only boolean flags, so new commands may be lost during I/O

`data_recording.py:272-305` reads request flags, performs file operations, then unconditionally clears the flags. An event barrier blocked the first rollback while a second Undo was submitted; only one deletion occurred, and the second request was cleared. Start/Stop/Undo in `doffy_teleop/protocol/control.py:91-106` can also overwrite each other's flags.

Use an ordered command queue or numbered state transitions, define in-progress, completed, and failed states, and allow Quest to receive processing results. Local button state alone cannot be treated as server state.

### Python and current Quest recording states have a synchronization gap in the source

Undo in the current `CodexBracelet/Assets/Teleop/Core/RecordingController.cs:32-35` only sends the command without clearing `IsRecording`; the session panel still allows Undo during recording (`UI/TeleopRecordPanel.cs:170-175`). Python stops acquisition upon receiving Undo, so Undo during recording may leave the UI displaying Stop recording. This conclusion comes from the current source on both ends; the operation was not performed through headset interaction. It needs to be fixed together with Python command state/acknowledgments.

### The WebRTC path ignores entrypoint configuration

- `doffy_teleop/runtime/publisher.py:15-24`
- `doffy_teleop/runtime/classic.py:50-54`
- `WebRTC_udp.py:90-99`

Both entrypoints call `WebRTCUDPManager()`, although its constructor accepts `config`. Mock validation captured `call()` in both cases, rather than the supplied configuration instance. Consequently, temporary changes to entrypoint IP, resolution, or other settings cause WebRTC to create a fresh default `Config()`, potentially differing from the settings used by the control/recording modules. The UDP branch already passes `config=cfg`.

Classic `RobotTeleop` also creates its own `Config()` (`doffy_teleop/robots/legacy/runtime.py:37-46`). All components should share the entrypoint configuration instance.

### The UDP receive thread cannot exit normally

`udp_comms.py:49-53` only closes the socket; the receive loop at `:93-98` has no exit condition, and `:84-90` swallows EBADF. An isolated-process check showed `_rx_thread.is_alive()` still True after `close()`. Add a stopped state, make blocking receives interruptible, and wait for the thread to terminate.

### Classic startup failures bypass cleanup

Resource initialization in `doffy_teleop/runtime/classic.py:44-143` precedes the main `try/finally`. Simulating a timeout raised by `cu_manager.test_connection()` confirmed **0** calls to `cu_manager.close()`. If camera, network, or subsequent device initialization fails, already-created resources receive no unified cleanup. Initialization in the new RealMan entrypoint is protected by cleanup and can serve as a reference for a unified lifecycle.

## Actual Undo semantics and normal-path validation

Current Quest `RecordingController.Undo()` sends the literal `Undo` to UDP 8003; Python socket_1 receives it and sets a rollback request. The `DataRecordingService` shared by Classic/RealMan calls `DatasetRecorder.rollback_last_episode()`.

- **No episodes:** no files are deleted, and the count remains 0.
- **Recording with unsaved frames:** discard the current unsaved buffer and retain previously saved episodes.
- **Saved HDF5 episode:** delete the last HDF5 file and its description row, decrease the count, and retain the preceding file; the number can be reused next time.
- **Saved LeRobot episode:** two episodes with 64×48 RGB/H264 video were actually written in a temporary directory. After Undo, data parquet and episode metadata retain only episode 0, and info counts agree; recording then successfully continues at number 1.
- **Shared video file:** when two episodes share `file-000.mp4`, Undo of the latter retains the entire video file because episode 0 still references it. Episode metadata and data rows are deleted, but the undone segment is not physically cut from the shared video. Continued recording creates `file-001.mp4`.
- **Statistical correctness:** the normal save/continued-recording checks above pass, but aggregate-statistics validation fails; see the P1 issue.

These results come from actual files and the LeRobot library; lock and concurrent-state issues were reproduced separately in isolated simulated environments. The complete real Quest click → Python execution → UI confirmation chain was not accepted because the current network was disconnected.

## Modularity and code reduction

The separation has substantive effects: the four compatibility entrypoints total 277 lines; the five major `doffy_teleop` subpackages in scope contain 39 Python files and 7,336 lines, with the largest file, `media/webrtc.py`, at 487 lines. The scan found no direct import cycles. Responsibilities are now clear:

| Subpackage | Responsibility |
|---|---|
| protocol | Packet parsing, control messages, signaling formats |
| media | Camera frames, UDP image transport, WebRTC |
| control | VR mapping, target policies, CAN-FD control loop |
| robots | Robot backends, Classic control implementation |
| runtime | Classic / RealMan startup, operation, and shutdown |

Further reduction should prioritize improvements to state and dependency boundaries:

1. **Unify configuration and lifecycle.** The configuration forwarding and startup cleanup gaps above remain.
2. **Make required mixin state explicit.** The four Classic mixins and the RealMan mapping/policy/lifecycle mixins depend on many `self` attributes initialized in other files; the split files still share implicit state. Gradually define explicit state objects or Protocol interfaces.
3. **Reduce package dependencies back into root compatibility entrypoints.** `runtime` still imports facades such as `robot_teleop`, `udp`, and `WebRTC_udp`, as well as root configuration, recording, and parsing utilities. Inject implementations through the composition entrypoint while retaining compatible external imports.
4. **Merge overlapping components.** The acquisition loops in `realsense_camera.py` and `media/camera.py` overlap, but the former also provides warmup and timestamp history, which should be preserved; the two image-downscaling functions and two dummy controllers can be shared. No references to `beaver_reader.py` were found; confirm external users before removing it.
5. **Organize runtime dependencies.** There is currently no separate installable-package configuration; requirements combine teleop and training dependencies without version constraints. Create validated constraints/lock files and optional hardware dependency groups for teleop.

The current structure supports continued maintenance, but modularization and reduction cannot yet be considered complete. File-length limits and compatibility-import tests do not establish correct thread lifecycle, concurrent recording, or real-hardware behavior.

## Dependency check

`python -m pip check` in the base `airo-doffy` environment returned 4 conflicts:

| Consumer | Declared range | Currently installed |
|---|---|---|
| lerobot 0.4.4 → av | >=15, <16 | 17.1.0 |
| lerobot 0.4.4 → huggingface-hub | >=0.34.2, <0.36 | 1.29.0 |
| lerobot 0.4.4 → packaging | >=24.2, <26 | 26.3 |
| datasets 4.8.5 → fsspec | >=2023.1.0, <=2026.2.0 | 2026.7.0 |

Successful imports of major modules do not guarantee compatibility of these version combinations. Preserve the currently working environment first, then validate and pin compatible combinations. The QA overlay additionally reports setuptools 57.4.0 below LeRobot's declared requirement; this does not appear in the base-environment check above.

## Reproducible existing test commands

```bash
MPLBACKEND=Agg /tmp/airo-teleop-qa/bin/python -m pytest -q \
  tests/test_realman_teleop_loop.py tests/test_wrm_akm.py \
  tests/test_realsense_camera.py tests/test_vr_coordinate_mapping.py \
  tests/test_teleop_refactor_imports.py tests/test_teleop_refactor_boundaries.py \
  tests/test_brainco_hand.py tests/test_wrm_visualizer.py tests/test_visualizer_layout.py
# 100 passed in 2.96s

PYTHONWARNINGS=ignore /tmp/airo-teleop-qa/bin/python -m pytest -q \
  tests/test_teleop_media_protocol.py tests/test_teleop_media_udp_loopback.py \
  tests/test_teleop_media_webrtc.py tests/test_udp.py tests/test_webrtc_csharp_source.py
# 12 passed in 13.27s

/tmp/airo-teleop-qa/bin/python -m pytest -q \
  tests/test_teleop_media_protocol.py::LegacyProtocolTests::test_control_state_preserves_record_and_zoom_semantics \
  tests/test_beaver_recording.py tests/test_dataset_recording_status.py \
  tests/test_realman_teleop_loop.py::RealManTeleopTest::test_episode_recorder_collects_cached_robot_and_camera_data \
  tests/test_realman_teleop_loop.py::RealManTeleopTest::test_episode_recorder_vr_export_and_visualizer_rollback \
  tests/test_realman_teachcollect.py::RealManTeachCollectorTests::test_force_dropout_during_replay_rolls_back_without_stopping_collector \
  tests/test_realman_teachcollect.py::RealManTeachCollectorTests::test_visualizer_commands_teach_then_replay_collect
# 20 passed; 3 overlap with the first two groups, giving 129 unique tests in total.
```

The current bracelet-version source was also checked with `DOFFY_UNITY_PROJECT=/path/to/unity/CodexBracelet` using `tests/test_webrtc_csharp_source.py`; 1 test passed, covering concurrent sends, UTF-8 fragmentation, reconnection, and oversized-message rejection in the C# signaling client.

Additional temporary validation code is stored in `/tmp/doffy-python-audit/comms` and `/tmp/doffy-python-audit/undo`; it is not production implementation. The C# protocol harness passed 47 checks; the combined check script subsequently exited 1 in the separate TrackingGuard compilation stage because an `OVRHand` reference was missing, so the entire combined C# check command was not counted as passed. Subsequent protocol interoperability checks of the current source passed separately.

Manual hardware scripts such as `tests/test_realman_canfd.py` were not run; they control real robots and cannot be executed directly as ordinary unit tests.

## Recommended fix order

1. Fix LeRobot Undo aggregate statistics and the UDP recording deadlock, and add regression tests using actual persisted data and a plain Lock.
2. Unify recording command ordering and acknowledgments, and update Quest's local recording state.
3. Fix UDP receive-thread termination, Classic startup-failure cleanup, and shared configuration forwarding.
4. Restore device networking and test integration with the real headset, robot, dexterous hand, and sensors.
5. Then refine explicit state interfaces, shared camera implementation, and dependency pinning.
