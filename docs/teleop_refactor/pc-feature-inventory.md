# PC teleoperation feature inventory and refactor baseline

This is a read-only inventory of the PC runtime in `AIRO-Doffy`, prepared for a
modular refactor that keeps the classic Unity application's behavior. It does
not inspect large datasets, papers, or policy weights. The source tree already
contains extensive uncommitted work; this inventory is the only file intended
to be added by this analysis.

The line counts below are a useful measure of the current coupling (the files
were inspected on 2026-09-16):

| File | Lines | Main concern |
| --- | ---: | --- |
| `realman_teleop.py` | 2,867 | RM75 app wiring, mapping, state callbacks, QP adapter, CAN-FD loop, recording and dashboard publishing in one module |
| `visualizer.py` | 1,254 | UI model, Matplotlib layout, tactile/Beaver panels and process transport |
| `dataset.py` | 1,044 | Schema construction, HDF5, LeRobot writer, rollback and finalization |
| `robot_teleop.py` | 963 | Legacy generic teleop mapping and data publication |
| `realman_teachcollect.py` | 922 | Local drag-teach state machine, replay and recording |
| `WebRTC_udp.py` | 791 | WebRTC signaling/video plus a second camera and VR receiver implementation |
| `beaver.py` | 728 | Binary protocol, incremental decoder, serial discovery and reader |
| `wrm_akm.py` | 628 | WRM packet receiver and RM75 arm-angle IK |
| `robot_backend.py` | 661 | Common robot interface and UR/RealMan/torque implementations |
| `config.py` | 561 | All network, robot, sensor, dataset and timing defaults/validation |
| `data_recording.py` | 546 | Threaded collection/export service and RealMan frame adapter |
| `main.py` | 502 | Classic full-feature entrypoint and tactile bridge |
| `udp.py` | 411 | UDP camera chunks, VR/control receiver and tactile sender |
| `realsense_camera.py` | 214 | Local camera capture and timestamped nearest-frame buffer |
| `parse_vr.py` | 232 | Controller, text-hand and binary-hand packet parsing |

## Runtime entry points and ownership

The current runtime has several deliberately different paths. They should stay
separate after extraction because they have different hardware and timing
requirements.

| Command/module | Role | Hardware/network opened by the normal path | Compatibility requirement |
| --- | --- | --- | --- |
| `python main.py` | Classic Unity VR teleoperation, camera streaming, tactile transfer, Beaver, recording and dashboard | UR or RealMan backend; RealSense; UDP or WebRTC; optional tactile and Beaver | Preserve the full classic behavior and module-level helpers |
| `python realman_teleop.py` | RealMan-only high-follow teleoperation and recording | RM75; RealSense; VR transport; optional WRM UDP, BrainCo hand, force state and Beaver | Preserve the high-rate CAN-FD path and all safety gates |
| `robot_teleop.py:RobotTeleop` | Historical robot-agnostic teleop implementation used by `main.py` | Robot backend during construction; no standalone CLI | Preserve class, `make_robot`, and `FastRobotiq2F85` imports |
| `python realman_teachcollect.py` | Local RealMan drag-teach, replay and synchronized collection; no VR | RM75; local RealSense; optional Beaver; dashboard | Preserve CLI and teach FSM; it must not acquire VR sockets |
| `python realman_recollect.py` or `python -m dataset_tool.recollect_realman` | Replay an existing RealMan LeRobot episode, optionally replace it by drag-teaching, then collect again | Dry-run: dataset only. Live mode: RM75, RealSense, Beaver, local web UI | Preserve dry-run as hardware-free and keep output/manifest rules |
| `python inference.py --policy ...` | Live policy deployment | RealSense; robot backend; optional tactile; model/checkpoint loader | Keep action-type aliases and inference options |
| `python eval_policy.py ...` | Live RM75 policy evaluation with Beaver and monitor | RM75; RealSense; optional Beaver; OpenCV monitor | Keep policy selection, safety, latency and verdict CLI |
| `python -m dataset_tool.replay_realman_lerobot ...` | Offline schema/trajectory validation or live RealMan replay | `--dry-run`: no robot; otherwise RealMan and optional camera alignment | Keep dry-run separate from the hardware replay path |
| `python test_tool/vr_data.py [--visualize]` | Standalone VR receiver and hand landmark visualizer | UDP VR input; optional Matplotlib | Preserve as a parser/transport smoke tool |
| `python test_tool/camera_test.py` | Camera/video transport smoke test | RealSense plus UDP/WebRTC | Keep clearly classified as hardware-only |

The central dependency flow is:

```text
Classic path:
main.py
  -> Config + VisualizerConfig
  -> UDPManager or WebRTCUDPManager
       -> RealSense capture + parse_vr + UDP control/record state
  -> RobotTeleop -> robot_backend factory -> UR/RealMan adapter
  -> DataRecordingService -> DatasetRecorder -> HDF5 or LeRobot
  -> optional BeaverReader, tactile reader and visualizer process

RealMan high-rate path:
realman_teleop.main
  -> transport/camera manager + optional WrmUdpReceiver + BeaverReader
  -> RealManTeleop
       -> RealManBackend
       -> optional RealManRemoteIkSolver and Rm75ArmAngleIk
       -> CanfdCommandLoop (dedicated high-follow clock)
  -> QuestTcpStateSender, RealManEpisodeRecorder and visualizer publisher

Offline/deployment boundary:
policies/realman_beaver/{dataset,configuration,modeling,train,checkpoint,
eval_registry,offline_eval_*}
  -> offline datasets/checkpoints and model evaluation
inference.py / eval_policy.py
  -> live camera/robot/Beaver adapters and policy actions
```

`Config` is currently a process-wide policy for most callers, but the paths do
not all inject it consistently. `main.py` and `realman_teleop._create_camera_manager`
pass a config to `UDPManager`, while `WebRTCUDPManager.__init__` always creates
`Config()` and ignores the caller's values. That is both a testing seam and a
source of runtime drift. The first extraction should make config injection
explicit while retaining a no-argument compatibility constructor.

## Feature and dependency inventory

### Configuration and schema

`config.py:Config` owns the defaults and validation for robot type/IP/port,
VR axes, TCP tool, command mode, RealMan CAN-FD/QP limits, RealSense, network,
video transport, rates, force/torque, tactile, Beaver, dataset and inference.
`__post_init__` normalizes robot/tracking/tool/data aliases, chooses RealMan's
seven-joint defaults, builds `TCP_TRANSFORM`, and rejects unsafe or inconsistent
values. It imports `airo_spatial_algebra` during configuration validation, so a
"hardware-free" test still needs the geometry dependency unless configuration
validation is split from vendor imports.

`data_schema.py` is the stable representation contract:

* `qpos`, `joint`, and `joint_configuration` normalize to joint state/action.
* `tcp`, `tcp_quat`, and `eef` normalize to `[qx,qy,qz,qw,x,y,z]` state/action.
* `delta_tcp` stores joint state and six-dimensional translation/rotation-vector
  action.
* `both` stores joint state/action and `extra.tcp_pose`.

`normalize_data_type`, `state_representation`, `action_representation`,
`should_store_extra_tcp_pose`, `build_data_schema`, and `DataSchema` should be
kept in a dependency-light `recording/contracts.py` (or re-exported from
`data_schema.py`) because `Config`, `DatasetRecorder`, `RobotTeleop`,
`RealManEpisodeRecorder`, inference and policy I/O all depend on these exact
aliases and dimensions.

### VR parsing and network transport

`parse_vr.py` is pure parsing and should not know about sockets. It recognizes
the legacy controller CSV and the newer `C,...` controller form, text hand
packets, and base64/binary hand packets. The controller result is a two-entry
list with `Position`, `Rotation`, `Joystick`, triggers and button fields;
hand parsing returns a side, frame/timestamp and 26-joint landmark payload.
Malformed packets return `None` rather than changing the last valid state.

`udp.py:UDPManager` currently combines three concerns:

1. Composes a `RealSenseCameraManager` and aliases its lock/image/depth/timestamp
   dictionaries for existing callers.
2. Sends up to five JPEG camera streams as a 12-byte big-endian header
   (`frame_id`, `chunk_index`, `total_chunks`, `total_bytes`) plus payload.
3. Receives VR pose/hand packets, record commands (`Start`, `Stop`, `Undo`,
   `Rollback`, `DeleteLast`), and legacy zoom controls, while optionally sending
   tactile bytes.

It exposes a mutable drop-in state surface: `data`, `hand_data`,
`camera_images`, `camera_data`, depth/timestamp dictionaries, `_lock`,
`data_collecting_state`, `data_export_state`, `data_rollback_state`,
`tactile_data`, `tactile_byte`, `tactile_timestamp_ns`,
`vr_input_timestamp_ns`, `camera_zoom`, `test_connection`,
`start_comms_threads`, `send_and_receive_data`, `is_movement_exist`, and
`close`.

`WebRTC_udp.py:WebRTCUDPManager` is intended as a drop-in replacement but
duplicates camera discovery/creation, capture threads, zoom conversion, VR
packet reception, control flags and tactile sending. It adds a WebSocket
signaling server on `SIGNALING_PORT`, one `RealsenseCameraTrack` per camera and
the `control` DataChannel. Its VR and tactile channels remain UDP. It has the
same state/method surface as `UDPManager`, but currently constructs cameras
itself and has a no-argument `__init__`. Extract common `VrInputState`,
`RecordControl`, `CameraSnapshot` and a transport protocol before sharing
implementation; preserve each wire transport's behavior during the first
move.

`udp_comms.py:UdpComms` is the small socket wrapper. It should remain below the
transport adapters, with socket creation injectable for tests.

The effective classic UDP port layout with `IP_PORT=8000` is:

| Channel | PC TX | PC RX | Meaning |
| --- | ---: | ---: | --- |
| `socket_0` | 8000 | 8001 | Camera 0 TX and Quest controller/hand pose RX |
| `socket_1` | 8002 | 8003 | Camera 1 TX and Quest record-control RX |
| `socket_2` | 8004 | 8005 | Camera 2 TX and Quest zoom/control RX; WRM uses this RX port when enabled |
| further sockets | 8006, 8008, ... | 8007, 8009, ... | Additional camera TX/RX pairs, up to five streamed cameras |
| tactile | `TACTILE_PORT` (default 8012) | none | Optional PC-to-Quest tactile bytes |

The transport protocol is part of the Unity compatibility surface. The existing
Quest app expects JPEG chunk fields in network byte order, controller/hand
packets on the pose RX channel, the legacy record strings, and the WebRTC
signaling envelope. Do not renumber ports or silently change quaternion order.

### Local camera capture

`realsense_camera.py:RealSenseCameraManager` is intentionally local-only: it
detects all RealSense serials, creates one `Realsense` wrapper per device,
captures RGB and optional depth at `REALSENSE_FPS`, and stores bounded
timestamped frame buffers. `snapshot_nearest(reference_timestamp_ns)` returns
copies of the nearest RGB/depth frames and timestamps. `start()` is idempotent,
warms up for up to 20 seconds, and `close()` joins threads and stops pipelines.
This is the correct single owner for local capture. WebRTC should consume it
through an adapter rather than maintaining a second capture implementation.

### Robot backends

`robot_backend.py` defines the common `RobotBackend` contract and keeps vendor
imports mostly lazy in `make_robot_backend`:

* `NullGripper` and `FastRobotiq2F85` cover tool compatibility.
* `PositionManipulatorBackend` adapts an airo position manipulator and provides
  joint/TCP/force reads, IK, servo and reset operations.
* `URPositionBackend` adds UR analytic IK and freedrive behavior.
* `RealManBackend` adapts `RealmanControl`, converts units, retries transient
  read error `-2`, exposes force/freedrive and performs ordered cleanup.
* `URTorqueBackend` owns the cached torque worker and torque/freedrive cleanup.
* `make_robot_backend(cfg)` is the new factory; `make_robot(...)` is the old
  raw-UR factory still used by replay scripts.

The minimal split is `robot/contracts.py`, `robot/gripper.py`,
`robot/backends/base.py`, `robot/backends/position.py`,
`robot/backends/realman.py`, `robot/backends/ur.py`, and `robot/factory.py`.
Leave `robot_backend.py` as a re-export shim until all repository imports have
migrated. Backend methods must retain radians at the project boundary,
`tcp_transform` semantics, force availability, safe target checks, reset and
freedrive lifecycle.

### RealMan teleoperation control path

`realman_teleop.py` is the largest and most safety-sensitive module. Its
current classes have distinct responsibilities even though they share a file:

| Current symbol | Responsibility | Recommended destination |
| --- | --- | --- |
| `pack_quest_tcp_state_packet` | RealMan base pose/wrench to Unity display axes and `rightTCP` JSON; quaternion is `[w,x,y,z]` | `teleop/transport/quest_state.py` |
| `CanfdLoopSnapshot` and `RealManStateSnapshot` | Immutable diagnostics/state payloads | `teleop/contracts.py` |
| `QuestTcpStateSender` | Fresh-state UDP sender to `(VR_IP, FORCE_PORT)`; pauses on stale/error | `teleop/transport/quest_state.py` |
| `RealManRemoteIkSolver` | Vendor continuous QP IK adapter, units and solver safety parameters | `teleop/control/realman_qp.py` |
| `CanfdCommandLoop` | Dedicated 180–200 Hz command clock, joint/TCP setpoints, speed/acceleration interpolation, heartbeat and rate watchdog | `teleop/control/canfd_loop.py` |
| `RealManTeleop` | VR-to-robot mapping, clutch/reset/hand behavior, WRM/QP resolution, sensor cache, realtime callback, lifecycle | `teleop/runtime/realman.py` plus `teleop/control/mapping.py` |
| `_create_camera_manager`, image reducers, `visualizer_publish_loop` | Process wiring and UI snapshot publication | `teleop/runtime/entrypoint.py` and `ui/publisher.py` |
| `main` | Hardware ordering, startup gates and cleanup/quarantine | `teleop/runtime/realman_entrypoint.py` |

The first move should extract `CanfdCommandLoop`, the QP adapter and packet
sender without changing `RealManTeleop`. Then extract pure mapping helpers and
only afterwards move orchestration. This keeps timing-sensitive behavior
reviewable.

The following invariants are acceptance criteria for the extracted control
layer:

* `REALMAN_CTRL_RATE` is strictly above `REALMAN_MIN_CANFD_RATE` (default
  180 Hz versus 100 Hz); a startup window must pass before motion is accepted.
* A packet gap or SDK call over 10 ms is a runtime failure after verification;
  the heartbeat threshold is 50 ms by default. The loop holds/quarantines
  safely and exposes its error in the snapshot.
* Joint, linear and angular target changes use the configured speed and
  acceleration limits. The loop sends `rm_movej_canfd` or `rm_movep_canfd`
  continuously in high-follow mode.
* Controller/TCP mode, `FREEZE_ROTATION`, clutch/reference refresh, stale VR
  input, reset edge, BrainCo grab/release edge behavior and wrist joystick bias
  remain unchanged.
* Joint mode retains optional RealMan QP IK. Optional WRM maps elbow confidence
  and alpha to a redundant arm-angle target; human elbow data never becomes a
  direct joint command. QP/WRM outputs still pass joint-limit, elbow and speed
  gates.
* Realtime state push remains preferred, with the polling fallback and force
  frame validation/filtering. State/action/timestamp snapshots are coherent
  for the recorder and dashboard.
* `FORCE_ENABLE` sends exactly one `rightTCP` object with display-axis position,
  orientation and force; do not reuse the tactile byte channel when force
  streaming is enabled.

### WRM / arm-angle mapping

`wrm_akm.py` is already mostly independent and should become two small layers:

* `tracking/wrm_protocol.py`: `WrmTrackingSample`, JSON/CSV parsing,
  `WrmUdpReceiver`, diagnostics and the legacy controller-shaped view.
* `tracking/rm75_arm_angle.py`: `Rm75AkmSettings`,
  `Rm75ArmAngleIk`, DH elbow geometry, vendor API calls, seed continuity,
  confidence/staleness freeze, singularity/height/limit checks and TCP-Z
  coupling.

Keep `wrm_akm.py` as a compatibility shim. It owns the optional UDP bind at
`PC_IP:CONTROL_PORT` (normally 8005), so the entrypoint must continue closing
or reserving the legacy `socket_2` before starting WRM to avoid a bind collision.

### Beaver sensing

`beaver.py` contains both a pure protocol and a hardware reader. The pure part
(`grid_width_from_flags`, `sensor_size`, `empty_snapshot`, `parse_sensor`,
`parse_frame`, `FrameDecoder`, `BeaverSnapshot`) handles 4x4/8x8, legacy 8-bit
and current 16-bit distances, partial USB frames, boot text, sensor metadata,
sequence/loss and stale diagnostics. `find_port(s)`, `open_port` and
`BeaverReader` add serial discovery, reconnect, latest snapshots and bounded
timestamp history. Split these into `sensing/beaver_protocol.py` and
`sensing/beaver_reader.py`, preserving the exact dtype/shape/layout and
`visualizer_payload` fields. The reader must remain non-blocking with respect
to the robot command deadline.

### Recording and dataset boundary

`data_recording.py` is the right service boundary already:

* `RecordingFrame` is the atomic multimodal frame contract.
* `RecordingControl` is a local thread-safe state machine;
  `ManagerRecordingControl` adapts UDP/WebRTC `Start`/`Stop`/`Undo` flags.
* `DataRecordingService` serializes collection, export, rollback and close on
  worker threads. It owns dataset mutations and propagates background errors.
* `RealManEpisodeRecorder` adapts RealMan snapshots and nearest camera/Beaver
  data to the schema. `tcp`, `delta_tcp`, joint and `both` semantics must not
  move into the robot control thread.

`dataset.py:DatasetRecorder` should be split behind its existing API into
`recording/hdf5.py`, `recording/lerobot.py`, and `recording/recorder.py`, with
`recording/contracts.py` for the frame/schema/timestamp features. Keep
`DatasetRecorder` and `normalize_lerobot_fps` re-exported from `dataset.py`.
Preserve:

* state/action dimensions and names from `data_schema.py`;
* camera/depth, force/torque, tactile, Beaver and `extra.tcp_pose` feature keys;
* monotonic timestamp vector names (`collect`, `robot_state`, `robot_action`,
  `vr_input`, `tactile`, optional `beaver`, then cameras);
* HDF5 episode numbering and LeRobot episode/video/parquet metadata;
* in-progress episode rollback, latest-episode rollback and finalization;
* optional `PUSH_TO_HUB` behavior only after a successful close.

`DatasetRecorder` may delete a LeRobot root during schema mismatch when
`recreate_on_schema_mismatch=True`; hardware-free tests must use a temporary
directory and must never point this constructor at a user's dataset.

### Visualization and tactile

`visualizer.py` has no robot ownership. `TeleopSample` is the UI snapshot,
`VisualizerHandle` is the bounded data/command queue, `TeleopDashboard` is the
Matplotlib UI and `start_visualizer` is the multiprocessing launcher. Move the
protocol to `ui/contracts.py`, the process/queue wrapper to `ui/visualizer.py`,
and panels/layout to `ui/panels.py` only after the sample keys are frozen.
Keep old optional arguments to `start_visualizer`, including camera count,
rollback/record/teach controls, tactile and Beaver options; other entrypoints
and tests call them directly.

`main.py` also owns the optional tactile reader selection (BLE 4-point or
serial), callback holder and 100 Hz bridge into the camera manager. Tactile
device readers (`tactile.py`, `tactile_4point.py`) should sit below a small
`sensing/tactile.py` adapter. Keep tactile transfer, visualizer-only tactile,
recalibration on reset, and the existing byte payload independent from force
JSON.

## Training and evaluation boundary

The policy package and live PC runtime have a useful existing boundary that
the refactor should make stricter:

| Area | Current modules | Allowed dependencies after refactor | Keep out |
| --- | --- | --- | --- |
| Offline training/model code | `policies/realman_beaver/{configuration.py,dataset.py,modeling*.py,train*.py,checkpoint.py}` | PyTorch/LeRobot, policy config, dataset readers, W&B/training utilities | robot drivers, RealSense, UDP/WebRTC, tactile and live UI |
| Offline metrics/checkpoint inspection | `policies/realman_beaver/offline_eval_*`, `offline_metrics.py`, `eval_registry.py` | checkpoint/config/dataset/model code | live robot or camera setup |
| Deployment inference | `inference.py` | policy loader, `data_schema`, robot factory, camera adapter, optional tactile | training entrypoint side effects; hard-coded live globals where injectable config is possible |
| Live policy evaluation | `eval_policy.py`, `eval_config.py` | inference/model policy, robot backend, camera, Beaver, monitor and verdict logic | dependence from policy training into hardware modules |
| Dataset validation/replay | `dataset_tool/replay_realman_lerobot.py` | dataset parser and trajectory safety checks; optional hardware replay adapter | opening hardware in `--dry-run` |
| Teach/recollect | `realman_teachcollect.py`, `realman_recollect.py` | local camera, robot, Beaver, recorder and UI | VR transport in teach-collect |

Training and offline evaluation should import a pure `policy_io`/schema layer;
the live adapters may import that layer, never the other way around. Keep all
checkpoint discovery and policy variant validation under `policies/` and
`eval_policy.py`. This refactor does not require touching policy weights or
training runs.

## Public imports and CLI surface to preserve

These names are used by current tests, replay tools, examples or downstream
scripts. During migration, re-export them from the old module path and emit no
behavioral change.

| Old module | Names/surface |
| --- | --- |
| `realman_teleop` | `pack_quest_tcp_state_packet`, `CanfdLoopSnapshot`, `RealManStateSnapshot`, `QuestTcpStateSender`, `RealManRemoteIkSolver`, `CanfdCommandLoop`, `RealManTeleop`, `RealManEpisodeRecorder` (currently imported into the module), `visualizer_publish_loop`, `_UNCLOSED_REALMAN_TELEOPS`, `_create_camera_manager`, `main` |
| `robot_backend` | `NullGripper`, `FastRobotiq2F85`, `CommandResult`, `RobotBackend`, `PositionManipulatorBackend`, `URPositionBackend`, `RealManBackend`, `URTorqueBackend`, `make_robot_backend`, `make_robot` |
| `robot_teleop` | `RobotTeleop`, `FastRobotiq2F85`, `make_robot`; preserve the historical class name |
| `udp` | `UDPManager`, `HD_HEADER_FMT`, `HD_HEADER_SIZE`, `MAX_STREAM_CAMERAS`, `STREAM_FPS`, and the mutable manager attributes/methods listed above |
| `WebRTC_udp` | `WebRTCUDPManager`, `RealsenseCameraTrack`, `MAX_CAMERAS`, and the same drop-in state/method surface |
| `realsense_camera` | `RealSenseCameraManager`, including `camera_list`, timestamp dictionaries, `camera_frame_buffers`, `snapshot_nearest`, `start`, `close` |
| `parse_vr` | `detect_packet_type`, `parse_data`, `parse_hand_data`, controller field names and hand payload shape |
| `beaver` | protocol constants, `BeaverSnapshot`, `empty_snapshot`, `parse_sensor`, `parse_frame`, `FrameDecoder`, `find_port(s)`, `open_port`, `BeaverReader` |
| `wrm_akm` | `WrmTrackingSample`, `parse_wrm_unity_packet`, `WrmUdpReceiver`, `Rm75AkmSettings`, `Rm75ArmAngleIk` |
| `data_recording` | `RecordingFrame`, `RecordingControlProtocol`, `RecordingControl`, `ManagerRecordingControl`, `DataRecordingService`, `RealManEpisodeRecorder` |
| `dataset` / `data_schema` | `DatasetRecorder`, `normalize_lerobot_fps`, `DataSchema` and all schema helper functions |
| `visualizer` | `TeleopSample`, `VisualizerHandle`, `TeleopDashboard`, `start_visualizer` and compatible keyword arguments |

CLI entrypoints and flags to retain:

* `python main.py` and `python realman_teleop.py` remain no-argument launches
  whose defaults are read from `Config`.
* `realman_teachcollect.py`: `--robot-ip`, `--port`, `--dataset-dir`, `--task`,
  `--fps`, `--beaver-simulate-8bit`.
* `realman_recollect.py`: `--dataset-root`, `--output-dataset`, `--episodes`,
  `--from-episode`, `--to-episode`, `--source`, `--robot-ip`, `--port`,
  `--initial-speed`, `--max-joint-speed`, `--joint-jump-policy`, `--task`,
  `--beaver-port`, `--beaver-grid-width`, `--host`, `--ui-port`, `--no-browser`,
  `--dry-run`.
* `dataset_tool.replay_realman_lerobot`: `--dataset-dir`, `--robot-ip`,
  `--port`, `--episodes`, `--from-episode`, `--to-episode`, `--control-mode`,
  `--source`, `--initial-speed`, `--initial-timeout`, `--max-joint-speed`,
  `--max-linear-speed`, `--max-angular-speed`, `--dry-run`, `--align-camera`,
  `--yes`.
* `inference.py`: required `--policy`; preserve `--device`, `--fps`,
  `--episodes`, `--max-steps`, `--horizon`, `--action_horizon`,
  `--n_obs_steps`, `--action_type`/`--action-type` and action aliases.
* `eval_policy.py`: repeatable `--policy`, `--device`, `--train_steps`/`--train-steps`,
  `--checkpoint-root`, `--fps`, `--latency-steps`, `--action-steps`, WRM wrap
  overrides, `--episodes`, `--max-steps`, `--output-dir`, `--no-video`,
  `--no-log`, `--no-monitor`, `--no-ask-success`, `--no-prompt`, vision
  disturbance/disable flags, Beaver 8-bit flag and scheduler/inference-step
  flags.

Do not make `_`-prefixed helpers the only implementation path without leaving
the old module alias: tests currently import `_UNCLOSED_REALMAN_TELEOPS`, and
entrypoint code relies on `_create_camera_manager` and visualizer helper
behavior.

## Recommended minimal module layering

The following package layout is enough to remove the long files while keeping
the feature set intact. It is intentionally smaller than a full rewrite.

```text
config.py                         # compatibility facade over config/contracts
data_schema.py                    # compatibility facade over recording/contracts
robot_backend.py                  # compatibility facade over robot/*
robot/
  contracts.py                    # RobotBackend protocol, CommandResult
  gripper.py                      # NullGripper, Robotiq and BrainCo tool adapters
  backends/{base,position,realman,ur}.py
  factory.py
sensing/
  camera.py                       # RealSenseCameraManager and CameraSnapshot
  beaver_protocol.py              # pure frame/snapshot decoder
  beaver_reader.py                # serial owner/reconnect/latest snapshot
  tactile.py                      # BLE/serial adapter boundary
transport/
  vr_parser.py                    # parse_vr implementation
  vr_state.py                     # parsed input + timestamps + control commands
  udp_video.py                    # UDPManager compatibility adapter
  webrtc_video.py                 # WebRTC manager/signaling adapter
  udp_socket.py                   # UdpComms alias/factory
tracking/
  wrm_protocol.py
  rm75_arm_angle.py
recording/
  contracts.py                    # DataSchema, RecordingFrame, timestamps
  service.py                      # DataRecordingService/control adapters
  recorder.py                     # DatasetRecorder facade
  hdf5.py
  lerobot.py
teleop/
  contracts.py                    # CAN-FD/state/sender diagnostic snapshots
  control/canfd_loop.py
  control/realman_qp.py
  control/mapping.py              # VR/hand delta, clutch, filters, target policy
  runtime/realman.py              # RealManTeleop lifecycle and sensor cache
  transport/quest_state.py
  runtime/publisher.py
  runtime/realman_entrypoint.py
ui/
  contracts.py                    # TeleopSample and command names
  visualizer.py                   # process/queue/UI facade
main.py                           # stable classic composition facade
realman_teleop.py                 # stable RealMan composition facade
```

The migration order should be:

1. Add pure contracts and compatibility re-exports without changing imports.
2. Extract parser/protocol functions and test them with bytes/strings only.
3. Extract `RealSenseCameraManager` ownership and inject it into both video
   adapters; make WebRTC accept `config=None` while preserving no-arg use.
4. Extract robot backends behind the existing factory and run the backend
   lifecycle tests with fake vendor objects.
5. Extract `CanfdCommandLoop`, `RealManRemoteIkSolver` and Quest state sender;
   compare snapshots and timing diagnostics against the existing tests.
6. Extract WRM/Beaver readers and the recording service/dataset backends.
7. Move `RealManTeleop` mapping/sensor lifecycle, then reduce the two large
   entrypoints to composition and cleanup only.
8. Only after parity, migrate imports in tests/tools and leave shims for one
   release. Keep classic `main.py` and RealMan `realman_teleop.py` as separate
   compositions; do not make the classic path depend on high-rate CAN-FD.

## Tests and hardware-free verification

The focused mock baseline reported for the current source is **68 tests
passing** across:

```bash
python -m pytest -q \
  tests/test_realman_teleop_loop.py \
  tests/test_udp.py \
  tests/test_wrm_akm.py \
  tests/test_realsense_camera.py \
  tests/test_vr_coordinate_mapping.py
```

This baseline covers the command loop and RealMan lifecycle with fake arms and
backends, UDP chunking/control with fake sockets/cameras, WRM packet and IK
behavior with fake vendor API, RealSense capture/timestamp buffers with fake
devices, and VR coordinate transforms. It is the first parity gate after each
extraction. `tests/test_realman_teleop_loop.py` imports several symbols directly
from `realman_teleop`, which is why the compatibility shim is required.

The rest of the test inventory is:

| Test group | Files | What it exercises |
| --- | --- | --- |
| Protocol/recording | `test_beaver_recording.py`, `test_dataset_recording_status.py` | Beaver binary/reader snapshots, frame service, dataset status/rollback seams |
| Hand/tool | `test_brainco_hand.py`, `test_vr_brainco_hand_teleop.py` | OpenXR mapping, BrainCo driver/motion; the latter is an integration script with `--dry-run` |
| UI | `test_visualizer_layout.py`, `test_wrm_visualizer.py`, `test_eval_closure_visualizer.py` | Agg Matplotlib layout, controls, WRM/Beaver status payloads |
| Teach/recollect/replay | `test_realman_teachcollect.py`, `test_realman_recollect.py`, `test_replay_realman_lerobot.py` | FSM, synchronization, trajectory repair and dry-run replay |
| Policy/deployment | `test_eval_policy_latency.py`, `test_eval_consensus.py`, `test_eval_train_steps.py`, `test_eval_vision_disturbance.py`, `test_consensus_*`, `test_dp_*`, `test_tightness_annotator.py` | Offline model/eval logic and monitor metrics; keep out of the teleop control extraction |
| Additional visual/data tools | `test_visualize_lerobot_rerun.py`, `test_visualizer_layout.py` and dataset-tool tests | Offline visualization/annotation behavior |

Useful selective commands, all from the repository root:

```bash
# Core PC runtime parity (mock vendor/socket/camera objects)
python -m pytest -q tests/test_realman_teleop_loop.py tests/test_udp.py \
  tests/test_wrm_akm.py tests/test_realsense_camera.py \
  tests/test_vr_coordinate_mapping.py

# Recording and sensor protocols (use temporary directories in tests)
python -m pytest -q tests/test_beaver_recording.py \
  tests/test_dataset_recording_status.py tests/test_brainco_hand.py

# Teach/recollect/replay seams
python -m pytest -q tests/test_realman_teachcollect.py \
  tests/test_realman_recollect.py tests/test_replay_realman_lerobot.py

# Policy/evaluation logic without starting a robot
python -m pytest -q tests/test_eval_policy_latency.py \
  tests/test_eval_consensus.py tests/test_eval_train_steps.py \
  tests/test_eval_vision_disturbance.py

# Dataset validation; this reads only the selected local dataset and opens no robot
python -m dataset_tool.replay_realman_lerobot \
  --dataset-dir ./datasets/<dataset> --episodes 0 --dry-run
```

For a no-hardware test of a newly extracted layer:

* Replace vendor arms, `UdpComms`, `pyrealsense2.context`, `Realsense`, serial
  ports and WebRTC peer objects with fakes at the adapter boundary.
* Use `Config(ROBOT_TYPE="realman", ROBOT_IP="192.0.2.1")` only after the
  geometry dependency is installed; no socket or robot is opened by `Config`.
* Exercise parser/IK/schema functions directly with deterministic bytes,
  arrays and fake SDK return values.
* Use `tempfile.TemporaryDirectory()` for HDF5/LeRobot test roots. Never use
  `./datasets` for a destructive schema-mismatch or rollback test.
* Set `MPLBACKEND=Agg` for dashboard tests and test `VisualizerHandle` queues
  without spawning a visible window.
* Call `--dry-run` on replay/recollect tools before any live run. `replay` and
  `recollect` without `--dry-run`, `main.py`, `realman_teleop.py`,
  `test_tool/camera_test.py`, and non-dry-run hand/force/tactile scripts are
  hardware/integration paths and must not be used as unit-test commands.

Two scripts deserve explicit classification: `tests/test_realman_canfd.py`
opens a real RM75 and sends a 200 Hz oscillation, and
`tests/test_vr_brainco_hand_teleop.py` can drive the BrainCo hand. They are
manual hardware checks rather than safe default test-suite members; only run
them with an operator, configured IP/ports and a clear workspace. The source
tree's current unit baseline should remain mock-only.

## Refactor acceptance checklist

Before deleting or renaming a legacy module, verify:

* `main.py` still streams all enabled cameras in UDP and WebRTC modes, receives
  controller and hand input, honors Start/Stop/Undo/rollback and tactile state,
  records the same schema, and closes every worker on Ctrl-C.
* `realman_teleop.py` still starts only after VR/camera/dataset readiness and a
  clean CAN-FD rate/heartbeat gate; stale input, reset, QP/WRM, force sender,
  recording and quarantine behavior are unchanged.
* Old imports listed above and all CLI flags produce the same object shapes,
  wire fields, dtype/order and status keys.
* UDP packet fragmentation, WebRTC signaling/data-channel control, force JSON,
  hand formats and Beaver 8/16-bit frames have byte-level regression coverage.
* Training/offline modules remain importable without robot, camera, network,
  tactile or UI initialization, while deployment/evaluation retain explicit
  hardware boundaries.
* The focused 68-test baseline and the relevant recording/UI/policy suites pass
  before broadening to live camera or robot verification.
