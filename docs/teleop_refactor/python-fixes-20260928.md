# Teleoperation fixes and validation (2026-09-28)

> Public copy: local paths, device identifiers and site addresses are anonymized; recorded test results and version information are retained.

This round fixed the recording deadlock, lost concurrent commands, incorrect LeRobot statistics after Undo, missing configuration forwarding, resource leaks on startup failure, and non-terminating UDP receive thread reproduced in the [previous audit](python-audit-20260928.md). Real-hardware validation also found and corrected the state-push interval unit error. Python-related tests: **232 passed, plus 3 parameter subtests passed**. The bracelet version was built and installed as **0.9.4 / versionCode 13**, retaining the separate package name `org.airolab.doffy.bracelet`.

## Fixes

| Issue | Implementation and validation |
|---|---|
| UDP recording control reacquired the camera's plain Lock, stalling Start / Undo | The separate `doffy_teleop/recording_control.py` owns the recording-state lock; both production managers share it directly. A real manager, plain Lock, and temporary HDF5 verified completed export and Undo, with pause state restored. |
| Requests arriving during export/Undo were cleared by the old operation | Stop / Undo use numbered FIFO requests, completing only the claimed request each time; a later Start resumes acquisition after all pending file operations finish. Validated consecutive Undo, Stop→Undo→Start, export exceptions, and queue draining on exit. |
| The receive thread discarded the first recording message at startup | Both transports retain commands received before startup and process the receive queue one command at a time. |
| Normalization statistics after LeRobot Undo and continued recording contained only the new episode | Rebuild aggregate statistics from retained episodes' feature statistics before deletion; first generate serializable replacement files, then perform rollback. Reject deletion when statistics, latest-episode metadata, or valid data references are missing. Remove the fallback that guessed filenames from episode numbers. |
| Risk of incorrectly deleting shared LeRobot video files | Delete only video files no longer referenced by retained episodes. Validated retention of shared video, continued recording after Undo, and rerecording after undoing all episodes. |
| Quest still displayed recording after Undo | `RecordingController.Undo()` clears local `IsRecording`, updates the button, and sends only Undo; it does not send an extra Stop that would first save content intended for discard. |
| Incomplete Quest hand-skeleton packets did not match Python's 26-joint protocol | Validate and read all 26 joints together before sending H / HB; do not send incomplete frames. |
| WebRTC and Classic robot used separate default configurations | Pass the same configuration instance from the entrypoint; retain compatibility with the original simple factory interface and correct the Classic loop to use the actual configured control frequency. |
| Resources leaked when Classic or media initialization failed partway through | Include initialization in the cleanup scope; close resources successfully created earlier when camera creation or socket allocation fails. UDP / WebRTC share a socket allocator. |
| Failure of a queued operation during exit skipped remaining requests and dataset closure | Record each operation error and continue draining accepted requests; close the dataset even if final saving fails, and propagate the first error after cleanup. Background exit failures are written to error state. |
| The UDP receive thread kept running after close | Stop event, finite receive timeout, socket closure, and bounded join; repeated close works, and invalid UTF-8 does not kill the receive thread. |
| PC used different network interfaces for the robot and Quest | Add `REALMAN_STATE_PUSH_IP`, separating the robot telemetry receive address from Quest's `PC_IP`; None retains the compatible fallback. |
| Incorrect unit for the realtime state-push interval | Keep configuration in milliseconds and convert with `cycle_ms // 5` at the SDK boundary; 5 / 10 / 25 ms send SDK values 1 / 2 / 5 respectively, consistently in enable and disable paths. See real-device retest below. |
| LeRobot / WebRTC dependency conflicts | Create a validated teleop venv, pin compatible related versions, and pass `pip check`; leave the original shared Conda environment unchanged. |

The LeRobot statistics regression test actually saves values 0 and 10, undoes 10, then saves 20; the result retains two frames with state/action count=2, mean=10, min=0, max=20, std=10. The test also covers Beaver features and video statistics. All HDF5 / LeRobot deletion checks use temporary directories.

The push-interval issue was found during this round's real-hardware validation: passing SDK `cycle=5` measured approximately 40 Hz. The [official RealMan protocol](https://develop.realman-robotics.com/robot4th/json/udpConfig/) specifies `cycle` in 5 ms units, so the original 5 ms configuration should pass `cycle=1`; passing 5 directly actually means 25 ms.

## Devices and network

| Purpose | Current address |
|---|---|
| PC communication with Quest, `PC_IP` | `192.0.2.10` |
| Quest, `VR_IP` | `192.0.2.20` |
| PC robot network interface, `REALMAN_STATE_PUSH_IP` | `192.168.1.100` |
| RealMan API | `192.168.1.18:8080` |

- Ping, an actual TCP connection, and UDP marker packets from Quest to PC all succeeded.
- Start / Stop / Undo were sent from Quest over the real network to production `UDPManagerCore`, connected to real `DataRecordingService` and temporary HDF5: two frames saved successfully, Undo deleted the newly saved episode, the count returned to zero, state and pause flags recovered, and the receive thread exited.
- The RealMan SDK successfully connected and read the current 7-joint and TCP state, returning code 0, then closed the handle normally.
- RealMan → PC `192.168.1.100:8098` realtime pushes: before correction, `cycle=5` produced 63/63 valid callbacks at 39.999 Hz; after correction, `cycle=1` produced 321/321 valid callbacks at 200.005 Hz. Callbacks came from the robot address with `errCode=0`, and valid 7-joint and TCP xyz/RPY data. Both tests backed up push configuration beforehand, restored it afterward, and read back every field to verify equality; handle deletion and SDK destruction returned 0, with no listener left on the port.
- The previous audit verified 640×480 RGB acquisition and thread termination on a real RealSense D435; camera acquisition validation was not repeated this round.

The Quest recording check here sent commands on the headset through ADB and used simulated frame data; it is not described as full validation by wearing the headset and clicking the UI. No robot reset, motion, gripper, or dexterous-hand actions were performed. Real Beaver / BrainCo / BLE hardware integration still awaits on-site validation.

Telemetry measurements before and after the fix are stored locally in `/tmp/doffy-python-audit/realman-push-before-result.json` and `/tmp/doffy-python-audit/realman-push-result.json`, respectively.

## Python validation

The environment is `.venv`; see the [environment notes](teleop-environment.md) for specific dependencies, reconstruction, and inheritance scope. This environment inherits SDK / Torch and other packages installed locally; the constraints file describes local compatibility, rather than a complete cross-platform lockfile.

```bash
HF_HUB_OFFLINE=1 MPLBACKEND=Agg \
DOFFY_UNITY_PROJECT=/path/to/unity/CodexBracelet \
./.venv/bin/python -m pytest -q \
  tests/test_realman_teleop_loop.py tests/test_wrm_akm.py \
  tests/test_realsense_camera.py tests/test_vr_coordinate_mapping.py \
  tests/test_teleop_refactor_imports.py tests/test_teleop_refactor_boundaries.py \
  tests/test_brainco_hand.py tests/test_wrm_visualizer.py tests/test_visualizer_layout.py \
  tests/test_teleop_media_protocol.py tests/test_teleop_media_udp_loopback.py \
  tests/test_teleop_media_webrtc.py tests/test_udp.py tests/test_webrtc_csharp_source.py \
  tests/test_beaver_recording.py tests/test_dataset_recording_status.py \
  tests/test_realman_teachcollect.py tests/test_realman_recollect.py \
  tests/test_replay_realman_lerobot.py tests/test_dataset_rollback.py \
  tests/test_recording_commands.py tests/test_udp_lifecycle.py \
  tests/test_runtime_lifecycle.py tests/test_media_startup_cleanup.py
# 232 passed, 3 subtests passed in 17.22s
```

The result XML is stored locally in `/tmp/doffy-python-audit/fixed-teleop-tests.xml`. This suite includes real localhost UDP / WebRTC, real temporary files, and the LeRobot API; robot motion uses a fake backend. Manual test scripts that operate hardware are excluded from the automated tests.

## Quest build and installation

- Project: `/path/to/unity/CodexBracelet`.
- APK: `/path/to/unity/App_output/AIRO_Doffy_bracelet.apk`.
- Previous version saved to: `/path/to/unity/App_output/bracelet-v0.9.3/AIRO_Doffy_bracelet.apk`.
- C# validation: 18 recording/hand-sending checks, 10 tracking-guard checks, and 47 protocol/lifecycle checks; 97 runtime and 3 editor source files compiled, and the Unity Android build succeeded.
- `adb install -r` succeeded; device package information confirmed version 0.9.4 / 13.

OpenXR initialized after application launch, but the application paused soon afterward without reaching the `AppManager.Start()` log. The application's saved server IP was therefore not confirmed, and the Unity UI→Python→robot closed loop while wearing the headset was not completed. Before use, confirm in the headset settings that the target PC IP is `192.0.2.10`.

## Usage and limitations

Start from the repository directory:

```bash
cd /path/to/AIRO-Doffy
./.venv/bin/python realman_teleop.py
```

Undo semantics are preserved: if there are currently unsaved frames, discard the current recording; otherwise, delete the last saved episode. If LeRobot episodes share a video container, a container referenced by retained episodes is not deleted, and undone segments within it are not physically cut out.

The recording queue guarantees processing order for requests Python has received; the existing UDP wire protocol still has no request IDs, server acknowledgment, or retries, so local button state cannot be treated as confirmation that data was saved to disk. Actual export/Undo results remain determined by Python logs and dataset state.

For modularity, this round centralized recording-control state and socket allocation in separate modules, unified configuration forwarding, and completed resource lifecycle handling. Making all mixin state explicit, merging camera acquisition implementations, and providing a complete installable package with optional dependency groups, as recommended by the original audit, remain subsequent structural improvements; this round did not perform a broad rewrite unrelated to the failures.
