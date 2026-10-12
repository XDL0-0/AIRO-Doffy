# AIRO-DOFFY-v2 reference repository inventory

> Public copy: local paths, device identifiers and site addresses are anonymized; recorded test results and version information are retained.

Audit date: 2026-09-16\
Objective: provide an inventory of reusable code, protocols, tests, and gaps for the full-featured Unity + PC refactor of `AIRO-Doffy`.\
Scope: read-only examination of the current working tree, documentation, and tests in `/path/to/AIRO-DOFFY-v2`; large datasets, weights, and `outputs/` contents were not examined, and v2 was not modified.

## Key conclusions

- The most valuable reusable parts of v2 are the immutable data models and ports in `src/airo_doffy/core`, protocol decoding and latest-only receivers in `devices/vr`, the transform/mapping/safety chain in `teleop`, video/state/command protocols in `streaming`, schema/rollback/asynchronous export in `recording`, and lifecycle and a single control loop in `runtime`. All have hardware-independent fake/mock tests.
- The current working tree is not a stable reference ready for release: the branch is `AIRO-DOFFY-v2.0`, HEAD is `9274594`, and `git status` shows deletions, modifications, or untracked changes in approximately 110 files. The current working tree deletes old root modules, `deprecated/`, `dataset_tool/`, `test_tool/`, and old compatibility entrypoints, while new Beaver, policy evaluation, UDP VR, and Matplotlib files remain untracked. Pin a commit and review these differences individually before referencing implementations.
- The built-in `airo-doffy-teleop` currently covers only the basic Quest UDP pose → UR/RealMan robot path; cameras, video, tactile, wrench, state channels, reliable commands, recording, and grippers are not connected in this factory. `airo-doffy-collect` still requires composition through an external `--session-factory`.
- v2 has no Unity/C# client, Unity project, scenes, protocol fixtures, or cross-language golden bytes; `unity/README.md` explicitly states it is a placeholder. The Unity+PC refactor must implement its own client and interoperability validation.
- Code-level protocols, mock runtime, and most adapters are fairly complete; real UR/RealMan/RealSense/Quest/BLE4, PyAV/aiortc, real HDF5/LeRobot, Unity RTP/WebRTC, and full workcell composition have not completed hardware or network end-to-end validation.

## 1. Reference tree baseline and entrypoints

`README.md` (current working tree) reports version `2.0.0.dev0` and lists capabilities including UR3e/UR5e, RealMan RM75, RealSense, Quest, four-point BLE4 MagTouch, video transport, state/command channels, and HDF5/LeRobot. `docs/refactor/phase_1_report.md` through `phase_15_report.md` and `docs/release_checklist.md` preserve evidence from each phase, but some describe earlier working-tree snapshots; for example, historical reports still assume root compatibility wrappers exist, while the current `README.md`, `tests/test_package_layout.py`, and filesystem have removed them.

| Entrypoint/location | Current behavior | Refactor value and limitations |
|---|---|---|
| `src/airo_doffy/apps/common.py` | Configuration parsing, `module:symbol` factory, lifecycle, and Ctrl-C cleanup | Can serve as the PC process entrypoint boundary; the factory only composes components and does not replace full deployment orchestration |
| `src/airo_doffy/apps/teleop.py` | `airo-doffy-teleop`, defaulting to `airo_doffy.apps.deployment:build_teleop_session` | Starts basic UDP Quest + robot; cameras/recording/commands/state channels/grippers excluded by default |
| `src/airo_doffy/apps/collect.py` | `airo-doffy-collect`, requiring `--session-factory` or an environment variable | Clear interface, but no directly runnable full-featured collection composition |
| `src/airo_doffy/apps/deployment.py` (untracked) | Basic UR/RealMan deployment; optional Matplotlib; also an RM75 policy-evaluation composition | Suitable experimental starting point; the implementation remains a basic path, delegates RealMan joint IK to the CAN-FD executor, and does not connect all v2 components |
| `src/airo_doffy/apps/evaluate.py` (untracked) | Validates a checkpoint, loads an external `beaver_policies` adapter, then runs RM75 evaluation | A separate evaluation tool, not generic Unity teleop; external policy/Beaver dependencies and hardware still need deployment validation |
| `configs/default.yaml`, `configs/robots/*.yaml` | Currently JSON-compatible YAML; supports layering, environment variables, and `--set` | Reusable configuration mechanism, but current defaults contain `192.0.2.13` and `192.0.2.21`, and the RealMan profile contains `192.168.1.18`; this conflicts with the documented “no addresses by default,” so remove deployment-specific addresses from the new PC configuration |
| `scripts/` | Diagnostics, dataset conversion/replay, video and trajectory benchmarks | Migrate CLIs as needed; the current tree deletes several old scripts, so v1 script compatibility cannot be assumed |
| `unity/` | Only `unity/README.md` | No client implementation; protocol boundary notes only |

Configuration models are centralized in `src/airo_doffy/config/models.py`, currently aggregating 15 categories: network, robot, camera, VR, teleop, tactile, Beaver, policy evaluation, recording, visualization, video, state transport, command transport, wrench, and runtime. `src/airo_doffy/config/loader.py` performs deep merging, type conversion, environment-variable overrides via `AIRO_DOFFY__SECTION__FIELD`, and `--set` overrides, rejecting unknown fields; `config/factories.py` provides eight narrow factories for robot/camera/encoder/video/VR/tactile/recorder/visualizer. These are useful dependency-injection boundaries, but do not automatically build a multi-device session.

## 2. Modular implementation inventory

Here, “software complete” means interfaces, error handling, and hardware-independent tests are largely in place, not that real-device acceptance has passed.

| Module | Key file locations | Assessment | Reusable parts/main gaps |
|---|---|---|---|
| Core models and infrastructure | `src/airo_doffy/core/types.py`, `interfaces.py`, `buffers.py`, `clocks.py`, `errors.py`, `events.py` | **Complete, directly reusable** | Frozen/slots samples, shape/finite checks, clocks/sequence numbers, latest buffer, errors, and events. Few dependencies; suitable as domain models for the Unity PC process; Unity still needs a separate C# representation |
| Cameras | `devices/cameras/base.py`, `mock.py`, `realsense.py` | **Software complete, unverified on hardware** | Lazy RealSense SDK import, acquisition thread, latest frame and health status; mock supports failure/delay/staleness. Current deployment does not create a camera/video pipeline |
| VR input | `devices/vr/protocol.py`, `binary_v2.py`, `receiver.py`, `mock.py`, `udp.py` | **Protocols and receivers complete, directly reusable** | Historical text, `AVR2` binary v2, dual-hand aggregation, duplicates/out-of-order/uint32 wrap, staleness, and latest-only; UDP transport is an untracked addition in the current tree. Compatibility parsers still have permissive behavior (HB trailing bytes/base64, recognition of old negative timestamps); new Unity should use strict binary v2 |
| Tactile | `devices/tactile/base.py`, `source.py`, `magtouch_ble4.py`, `filters.py`, `mock.py` | **Software complete, devices unverified** | BLE4 `(4,3)` samples, calibration, deadband/EMA/Kalman/drift, and disconnect semantics; old serial/legacy files are deleted in the current tree, and README says they are no longer supported. Not connected in deployment |
| wrench | `devices/wrench/base.py`, `robot_source.py`, `compensation.py`, `filters.py`, `pipeline.py` | **Processing chain complete, sources/wiring insufficient** | Gravity/bias compensation, mean/low-pass/deadband/clamp are pure logic; currently no separate physical wrench adapter, and not part of basic teleop composition |
| Robots and executors | `robots/base.py`, `mock.py`, `ur.py`, `realman.py`, `executor.py`, `realman_executor.py`, `grippers/*` | **Software adapters largely complete, unverified on hardware** | Lazy UR/RealMan SDK imports, Mock, generic latest executor, RealMan CAN-FD high-follow executor, Robotiq/Null gripper. Must observe state-source thread ownership constraints in `docs/architecture.md`: session and executor must not concurrently read/write a thread-affine SDK; current basic RealMan composition still needs hardware confirmation |
| Transforms/mapping/safety | `teleop/transforms/*`, `teleop/mappings/*`, `teleop/safety/*` | **Complete, directly reusable** | Axis mapping, quaternions, local/world rotation, reference/rebase, joint/TCP/IK/gripper mapping, workspace/joint/velocity/acceleration/rate, freshness/watchdog; deterministic tests. Coordinate calibration must be rechecked using v1 recorded trajectories and the real workcell |
| Video | `streaming/video/frame_processor.py`, `h264_encoder.py`, `encoding_pipeline.py`, `legacy_jpeg_udp.py`, `rtp_h264_udp.py`, `webrtc_transport.py` | **Components complete, production chain not composed/actual transport unverified** | Processing, drop-oldest/latest, PyAV/NVENC→x264 selection, compatible JPEG chunks, RFC6184 RTP/FU-A, aiortc transport. Real codecs, SDP/ICE, MTU, Quest/Unity receiver, and reconnection unverified; legacy JPEG 12B `!IHHI` header retains compatibility, but 60,000B chunks may cause IP fragmentation |
| State/command channels | `streaming/state/{protocol,channels}.py`, `streaming/commands/{protocol,channels,router}.py` | **Protocols/fake transport complete, runtime not connected** | Latest-only binary state, WebRTC unordered/unreliable adapter, ordered reliable commands, ACK/timeouts/deduplication/router all tested; basic deployment creates no channel/handler, and the Unity side is absent |
| Recording and export | `recording/schema.py`, `samples.py`, `state.py`, `writers/{hdf5,lerobot,rollback}.py`, `export_worker.py` | **Boundaries and state machines complete, real libraries unverified** | Immutable samples, episode numbering, rollback/reuse, bounded worker, visible failures/retry/discard; `DataCollectionSession` extends the same teleop loop. Current collect has no built-in composition; comparison using real h5py/LeRobot storage incomplete |
| runtime | `runtime/lifecycle.py`, `session.py`, `data_collection.py`, `ports.py` | **Control loop and lifecycle complete** | Startup rollback, reverse close, worker health, HOLD/STOP, single-loop TeleopCycle, recording extension. Does not automatically discover or start camera/video/tactile/state/command; requires higher-level composition |
| Visualization | `visualization/models.py`, `consumer.py`, `mock.py`, `commands.py`; untracked `extension.py`, `matplotlib.py`, `opencv_eval.py` | **Headless boundaries complete, GUI/wiring unverified** | Typed latest-only snapshots, error isolation, memory renderer usable in CI; Matplotlib/evaluation renderers and multisensor snapshots unverified on real desktops/full sessions; Unity UI absent |
| policy/Beaver | Untracked `policies/base.py`, `policies/forceflowpp/*`, `devices/beaver/*`, `runtime/policy_evaluation.py` | **New experimental path, unsuitable as a stable baseline** | Beaver serial decoding/reconnect/staleness, adapter contract, and RM75 dry-run/execute safety have tests; depends on external `beaver_policies`, checkpoints, and hardware, and does not form generic teleop |

For design rules and extension constraints, read `docs/architecture.md` and `docs/extension_guide.md`: constructors perform no I/O, `start()` acquires resources, data crossing boundaries must be immutable, background queues are bounded with a drop policy, and the deployment factory explicitly handles SDK thread affinity.

## 3. Protocols and required Unity implementation

| Channel | v2 implementation/specification | Reusable tests | Current interoperability status |
|---|---|---|---|
| VR input | `docs/protocols/vr_binary_v2.md`, `devices/vr/binary_v2.py`; little-endian `AVR2` v2, 24B header, controller/hand entity, float32, quaternion XYZW, uint32 sequence | `tests/unit/test_vr_binary_v2.py`, `test_vr_protocol.py`, `test_vr_receiver.py` | Internal Python round-trip/malformed packet/order checks pass; Unity/C# packing has only a documentation outline, no fixtures |
| realtime state | `docs/protocols/state_channel_v1.md`, `streaming/state/protocol.py`; 20B envelope, magic `0xAD20`, VR/robot payload, latest-only | `tests/unit/test_state_protocol.py`, `test_state_channels.py` | Fake channels verified; WebRTC data channel and Unity decoder unverified; not connected in basic deployment |
| runtime commands | `docs/protocols/reliable_commands_v1.md`, `streaming/commands/*`; strict canonical JSON v1, ordered reliable, ACK, command-ID deduplication | `tests/unit/test_command_protocol.py`, `test_command_channels.py`, `test_command_router.py` | Python fake/retry/duplicate checks pass; no Unity serializer/ACK fixtures/real WebRTC acceptance |
| video | `docs/communication.md`; WebRTC H.264, RTP H.264, old JPEG chunks; default RTP MTU 1200, JPEG compatibility header `!IHHI` | `test_frame_processor.py`, `test_h264_encoder.py`, `test_video_encoding_pipeline.py`, `test_legacy_jpeg_udp.py`, `test_rtp_h264_udp.py`, `test_webrtc_transport.py` | Packetizer/fake encoder verified; actual aiortc/PyAV/Unity receiver/ICE/reconnection unverified |

Ports and labels are in `docs/communication.md`: old base/pose/control use `8000/8001/8005`, signaling `8765`, RTP `5004`, diagnostic state `5005`, and the RealMan realtime state-push example uses `8098`. The new PC/Unity design should make these configurable instead of hardcoding them in the client; state and commands must use separate DataChannels (state unordered/unreliable, commands ordered/reliable).

`docs/communication.md` and `docs/protocols/vr_binary_v2.md` describe C# field order, little-endian, float32, row-major 4×4, quaternion XYZW, uint64 timestamps, sequence wrap, and related guidance, but provide no cross-language byte samples for automated Unity regression. The minimum Unity deliverable should include separate pack/unpack classes, golden bytes shared with Python (normal/truncated/wrong version/invalid entity/sequence wrap), WebRTC channel labels/reliability settings, command ACK/deduplication, and RTP/H.264 receive logs.

## 4. Directly verifiable hardware-free paths

In the current v2 working tree, explicitly set `PYTHONPATH=src` when the editable package is not installed. The following were executed and passed:

```bash
cd /path/to/AIRO-DOFFY-v2
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /path/to/source-env/bin/python -m unittest discover -s tests -q
# Ran 316 tests in 1.462s
# OK (skipped=7)

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /path/to/source-env/bin/python -m compileall -q src tests
# exit 0
```

Entrypoint smoke checks that can also be executed directly and return 0:

```bash
PYTHONPATH=src python -m airo_doffy.apps.teleop --help
PYTHONPATH=src python -m airo_doffy.apps.collect --help
PYTHONPATH=src python -m airo_doffy.apps.evaluate --help
```

Key hardware-independent coverage:

- Core/models/buffers: `tests/unit/test_core_types.py`, `test_core_buffers.py`, `test_domain_interfaces.py`.
- config/entrypoints/factories: `test_config_loader.py`, `test_config_models.py`, `test_config_factories.py`, `test_app_entrypoints.py`, `tests/test_package_layout.py`.
- Mock sessions/lifecycle/recording: `tests/integration/test_mock_session.py`, `test_runtime_lifecycle.py`, `test_runtime_session.py`, `test_runtime_data_collection.py`, and `test_recording_*`.
- Robots/teleop: `test_mock_robot.py`, `test_robot_executor.py`, `test_realman_executor.py`, `test_ur_robot.py`, `test_realman_robot.py`, `test_teleop_{transforms,mappings,safety,watchdog}.py`.
- Cameras/tactile/VR/wrench: `test_mock_camera.py`, `test_realsense_camera.py`, `test_mock_tactile.py`, `test_magtouch_ble4.py`, `test_tactile_filters.py`, `test_mock_vr.py`, `test_vr_*.py`, `test_wrench_*.py`.
- Networking/encoding/state/commands: `test_{h264_encoder,video_encoding_pipeline,legacy_jpeg_udp,rtp_h264_udp,webrtc_transport,state_protocol,state_channels,command_protocol,command_channels,command_router}.py`.
- Newly added evaluation path: `test_beaver_source.py`, `test_policy_evaluation.py`, `test_policy_eval_cli.py`; because these are untracked, first confirm whether they will enter a formal commit.

`tests/hardware/test_devices.py` skips by default; read `tests/hardware/README.md` before enabling it. Environment variables are `AIRO_DOFFY_TEST_UR_IP`, `AIRO_DOFFY_TEST_REALMAN_IP`, `AIRO_DOFFY_TEST_REALSENSE=1` (optional `AIRO_DOFFY_TEST_REALSENSE_SERIAL`), and `AIRO_DOFFY_TEST_BLE4=1`. Robot smoke tests read state once and close without actively submitting motion actions, but close may send the adapter's normal stop; run them at a supervised safe workcell.

The current environment has Python 3.13 and `ruff`; no `pytest` or `pyright` command was found. `docs/release_checklist.md` also lists pytest/ruff/pyright/pre-commit, HDF5, Unity, hardware, network E2E, and performance as unchecked. Thus, the 316 unittests establish only pure Python/Mock paths, not production deployment.

## 5. Specific gaps for v1 use cases

1. **A full-featured PC composition does not yet exist.** Actual v1 use combines VR, robot, cameras/video, tactile/wrench, visualization, recording, and control in one teleop workflow; v2 separates the modules, but `build_teleop_session()` only composes basic VR + robot + mapping/safety/executor. Compose a single control loop and multiple acquisition/stream/export workers through the `runtime` ports, then explicitly attach `RecordingCycleExtension`, visualization, and the state/command router.
2. **The Unity input/display/control side is empty.** v2 has protocol documentation only; cross-language bytes, clocks/coordinates, DataChannel establishment order, ACK/deduplication, H.264 reception, and disconnect/reconnect have no Unity validation. Unity is the first new project in this refactor, not a folder that can be copied from v2.
3. **v2 channel implementations are not yet part of the application path.** State/command protocols and fake adapters exist, but current deployment does not construct them or route reset/recording/recalibrate/mode/safe-hold commands into runtime.
4. **Thread ownership poses a production risk.** `docs/architecture.md` explicitly states that `TeleopSession` and executor must not concurrently read/write thread-affine vendor SDKs; RealMan should provide `RobotStateSource` through the executor owner thread or a realtime state-push/cache. Current deployment directly uses the backend as state source; confirm SDK constraints on a real RM75 before finalizing it.
5. **The current working tree deletes the old compatibility surface.** Root `main.py`, `realman_teleop.py`, `robot_backend.py`, `parse_vr.py`, `WebRTC_udp.py`, `dataset.py`, `tactile*.py`, as well as `deprecated/`, `dataset_tool/`, and `test_tool/`, are all currently deletions. `docs/migration_v2.md`/README choose “removed roots unsupported,” but this breaks external v1 imports and script calls; if this refactor must preserve the v1 workflow, add separate thin wrappers/migration commands and tests.
6. **Real media/storage/devices remain unverified.** RealSense, BLE4, UR/RealMan SDKs, aiortc/PyAV/NVENC, Unity RTP, h5py/LeRobot shared storage, real network packet loss, and stop/reconnect remain incomplete release-checklist items.
7. **Configuration has drifted toward local deployment settings.** Current `configs/default.yaml` and the RealMan profile contain actual LAN addresses inconsistent with the safe defaults claimed by documentation; do not directly copy this working tree's configuration to a new PC or commit it to a public repository.
8. **New policy/Beaver paths are not generic teleop.** They are untracked evaluation paths requiring external packages and specific RM75/data; they can serve as v1 policy/eval references, but must not become mandatory dependencies of basic Unity teleop.

## 6. Recommendations for architecture and integration

- Treat v2 as a reference for ports, data models, protocols, and safety algorithms, and first pin a clean commit; individually decide which currently untracked implementations (especially `deployment.py`, Beaver, policy evaluation, Matplotlib) enter the baseline.
- Compose the PC side along the dependency direction `core → devices/robots → teleop → runtime`: VR/camera/tactile/wrench each latest-only; video encoding/transport, state sending, reliable commands, recording export, and visualization as bounded workers/extensions; the control loop only consumes snapshots and submits safe actions to the unique robot-owner executor.
- Freeze protocol fixtures before writing Unity C#: establish Python↔C# golden tests for VR binary v2, state v1, command JSON/ACK, and RTP packetization; logs must record sequence gaps, staleness, ACK timeouts, codec/ICE, and stop reasons.
- Build a hardware-free full session with MockRobot/MockVR/MockCamera/MockTactile/MemoryRenderer; then localhost UDP/WebRTC loopback; finally device acceptance using the supervised matrix in `tests/hardware/README.md`. Cover watchdog HOLD, STOP, disconnect, reconnect, duplicate commands, and bounded-queue frame drops at every step.
- If the product still needs v1 root scripts/data tools, retain an explicit compatibility layer and migration notes; if compatibility is intentionally broken, document the behavior differences between current deletions and new CLIs as an acceptance checklist, avoiding confusion between an uncommitted working tree and a completed migration.
- Move actual addresses, serial ports, dataset paths, and runtime switches out of default configuration; defaults should be importable, mockable templates without hardware side effects. Do not label v2 a release candidate before Unity + network E2E, real storage, and hardware stop validation are complete.
