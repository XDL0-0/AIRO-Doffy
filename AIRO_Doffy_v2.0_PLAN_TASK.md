# AIRO-Doffy v2.0 Refactor Plan and Task List

Repository: `XDL0-0/AIRO-Doffy`

Target release: `v2.0`

Primary goal: redesign AIRO-Doffy as a modular, low-latency, reusable teleoperation framework for UR and RealMan manipulators, VR controller or hand-tracking input, multi-camera streaming, tactile sensing, dataset recording, visualization, and policy evaluation.

This document is intended to be executed incrementally by Codex or another coding agent. Do not perform the entire refactor in one change. Complete and validate each phase before moving to the next one.

---

# 1. Overall Objectives

AIRO-Doffy v2.0 must:

1. Preserve current supported behavior unless a change is explicitly approved.
2. Reduce the number of implementation files in the repository root.
3. Use a standard installable Python package with a `src/` layout.
4. Separate every major behavior into an independently reusable component.
5. Avoid all-in-one modules such as the current `camera_udp.py`, `WebRTC_udp.py`, `main.py`, or large teleoperation managers.
6. Keep only the 4-taxel BLE tactile implementation as the supported tactile backend.
7. Move unused legacy code into `deprecated/` instead of deleting it.
8. Support UR3e, UR5e, and RealMan RM75 through isolated robot backends.
9. Support VR controller and hand-tracking input through a common typed input interface.
10. Support low-latency video streaming through replaceable transports.
11. Use binary, low-overhead protocols for high-frequency state data.
12. Separate unreliable real-time state updates from reliable control commands.
13. Allow every hardware-dependent component to be replaced by a mock.
14. Keep dataset schemas and network protocols backward compatible unless explicitly approved.
15. Add architecture documentation, automated tests, performance instrumentation, and migration notes.

---

# 2. Non-Negotiable Architecture Principles

## 2.1 One Module, One Primary Responsibility

Every implementation module must have one clearly stated primary responsibility.

A module must not combine several of the following behaviors:

- Hardware discovery or acquisition.
- Network socket management.
- Protocol parsing.
- State storage.
- Teleoperation mapping.
- Coordinate transforms.
- Safety checking.
- Robot command execution.
- Dataset buffering.
- Dataset serialization.
- Visualization.
- Runtime orchestration.
- Configuration loading.
- Command-line parsing.

Moving a large multi-purpose class into a new directory does not count as decoupling.

Each source module must include a short module-level docstring stating its responsibility.

## 2.2 Separate Data Producers from Data Consumers

Hardware and communication components must produce typed data only.

Examples:

- A camera produces `CameraFrame`.
- A tactile sensor produces `TactileSample`.
- A VR receiver produces `VRInputState`.
- A robot backend produces `RobotState`.
- A wrench source produces `WrenchSample`.

These producers must not directly:

- Record datasets.
- Update visualizations.
- Send robot commands.
- Access unrelated hardware.
- Modify arbitrary attributes on another object.

Consumers must receive samples through explicit methods, typed queues, immutable snapshots, or narrow callback interfaces.

## 2.3 Separate Transport from Protocol

Network transport and message semantics must be independent.

Required pattern:

```text
socket or data channel
    -> raw bytes
protocol decoder
    -> typed message
runtime router or teleoperation mapping
    -> system behavior
```

A socket class must not interpret controller semantics.

A protocol parser must not create sockets.

A teleoperation mapping must not know whether data arrived through UDP, WebRTC, a file, or a mock source.

## 2.4 Separate Acquisition, Processing, Encoding, and Transport

Required video pipeline:

```text
CameraSource
    -> CameraFrame
FrameProcessor
    -> ProcessedFrame
VideoEncoder
    -> EncodedFrame
VideoTransport
```

A camera source must not:

- Start UDP or WebRTC servers.
- Parse VR controller packets.
- Store tactile samples.
- Control dataset recording.
- Update a visualizer.
- Access robot state.

A video transport must not:

- Discover or initialize RealSense cameras.
- Process teleoperation commands.
- Own robot or tactile state.
- Manage episode recording.

## 2.5 Separate Mapping, Safety, and Execution

Required teleoperation pipeline:

```text
VRInputState
    -> TeleopMapping
RobotAction
    -> ActionFilter
SafeRobotAction
    -> RobotExecutor
```

`TeleopMapping` is responsible only for:

- Controller or hand interpretation.
- Coordinate transforms.
- Relative or absolute pose mapping.
- Scaling.
- Gripper target mapping.
- Target action generation.

`ActionFilter` is responsible only for:

- Workspace limits.
- Joint limits.
- Velocity limits.
- Acceleration limits.
- Command freshness.
- Command validation, clipping, or rejection.

`RobotExecutor` is responsible only for:

- Sending validated commands to a robot backend.
- Respecting command rates.
- Handling backend command errors.
- Holding or stopping safely when commands become stale.

## 2.6 Orchestration Is Not Implementation

Runtime classes may coordinate components, but must not implement component internals.

Allowed:

```python
vr_state = vr_source.get_latest()
robot_state = robot.read_state()
action = mapping.map(vr_state, robot_state)
safe_action = safety.apply(action, robot_state)
executor.execute(safe_action)
recorder.append(observation, safe_action)
```

Not allowed inside runtime classes:

- Parsing raw UDP strings.
- Computing BLE tactile filtering.
- Encoding JPEG or H.264 packets.
- Performing backend-specific inverse kinematics.
- Writing HDF5 fields directly.
- Drawing visualization panels.
- Implementing coordinate transformation mathematics.

## 2.7 Avoid God Classes and God Managers

Do not introduce broad replacements such as:

- `SystemManager`
- `TeleopManager`
- `CameraManager`
- `CommunicationManager`
- `DataManager`
- `HardwareManager`

A class requires architectural review when it:

- Has more than approximately 300 lines.
- Has more than approximately 10 public methods.
- Owns more than one unrelated hardware device.
- Owns both hardware and network resources.
- Owns both control and recording logic.
- Requires unrelated configuration sections.
- Manages both protocol parsing and business logic.

These are warning indicators, not automatic hard limits. Any exception must be justified in `docs/architecture.md`.

## 2.8 Dependencies Must Point Inward

Required dependency direction:

```text
apps
  -> runtime
      -> teleop / recording / visualization
          -> core interfaces and data types
              <- devices / robots / streaming adapters
```

Forbidden dependencies include:

```text
devices -> runtime
devices -> visualization
devices -> recording
robots -> VR
streaming -> camera acquisition
recording -> visualization
teleop mappings -> UDP or WebRTC
core -> hardware SDKs
```

Concrete implementations must be injected at the application composition root.

## 2.9 Configuration Contains Values Only

Configuration models must not:

- Open sockets.
- Connect to robots.
- Start threads.
- Discover cameras.
- Import optional hardware SDKs.
- Instantiate runtime components.

Object creation belongs in factories used by the application entry point.

---

# 3. Target Repository Structure

```text
AIRO-Doffy/
├── pyproject.toml
├── README.md
├── CHANGELOG.md
├── configs/
│   ├── default.yaml
│   ├── robots/
│   │   ├── ur3e.yaml
│   │   ├── ur5e.yaml
│   │   └── realman_rm75.yaml
│   └── experiments/
│       ├── collect_ur3e.yaml
│       ├── collect_rm75.yaml
│       └── vr_hand_tracking.yaml
├── src/
│   └── airo_doffy/
│       ├── __init__.py
│       ├── core/
│       │   ├── types.py
│       │   ├── interfaces.py
│       │   ├── events.py
│       │   ├── buffers.py
│       │   ├── clocks.py
│       │   └── errors.py
│       ├── config/
│       │   ├── models.py
│       │   ├── loader.py
│       │   └── factories.py
│       ├── devices/
│       │   ├── cameras/
│       │   │   ├── base.py
│       │   │   ├── realsense.py
│       │   │   └── mock.py
│       │   ├── tactile/
│       │   │   ├── base.py
│       │   │   ├── magtouch_ble4.py
│       │   │   └── mock.py
│       │   ├── vr/
│       │   │   ├── base.py
│       │   │   ├── protocol.py
│       │   │   ├── receiver.py
│       │   │   ├── controller_types.py
│       │   │   ├── hand_types.py
│       │   │   └── mock.py
│       │   └── wrench/
│       │       ├── filters.py
│       │       └── compensation.py
│       ├── robots/
│       │   ├── base.py
│       │   ├── ur.py
│       │   ├── realman.py
│       │   ├── mock.py
│       │   ├── executors.py
│       │   └── grippers/
│       │       ├── base.py
│       │       ├── robotiq_2f85.py
│       │       └── mock.py
│       ├── teleop/
│       │   ├── actions.py
│       │   ├── mappings/
│       │   │   ├── base.py
│       │   │   ├── controller_pose.py
│       │   │   ├── hand_pose.py
│       │   │   └── gripper.py
│       │   ├── transforms/
│       │   │   ├── coordinate_frames.py
│       │   │   └── pose_delta.py
│       │   └── safety/
│       │       ├── base.py
│       │       ├── workspace.py
│       │       ├── joint_limits.py
│       │       ├── velocity_limits.py
│       │       └── watchdog.py
│       ├── streaming/
│       │   ├── video/
│       │   │   ├── base.py
│       │   │   ├── frame_processor.py
│       │   │   ├── encoder.py
│       │   │   ├── webrtc_transport.py
│       │   │   ├── rtp_udp_transport.py
│       │   │   └── legacy_jpeg_udp.py
│       │   ├── state/
│       │   │   ├── protocol.py
│       │   │   ├── webrtc_channel.py
│       │   │   └── udp_transport.py
│       │   └── commands/
│       │       ├── protocol.py
│       │       └── webrtc_channel.py
│       ├── recording/
│       │   ├── schema.py
│       │   ├── episode.py
│       │   ├── recorder.py
│       │   ├── hdf5_writer.py
│       │   ├── lerobot_writer.py
│       │   └── rollback.py
│       ├── visualization/
│       │   ├── dashboard.py
│       │   ├── models.py
│       │   └── publisher.py
│       ├── policies/
│       │   ├── inference.py
│       │   └── evaluation.py
│       └── runtime/
│           ├── lifecycle.py
│           ├── command_router.py
│           ├── teleop_session.py
│           └── data_collection_session.py
├── apps/
│   ├── teleop.py
│   ├── collect.py
│   ├── inference.py
│   └── evaluate.py
├── scripts/
│   ├── calibration/
│   ├── dataset/
│   ├── diagnostics/
│   └── benchmarks/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── hardware/
├── unity/
├── docs/
│   ├── architecture.md
│   ├── communication.md
│   ├── extension_guide.md
│   └── refactor/
└── deprecated/
    ├── README.md
    └── tactile/
        └── magtouch_ilias_41taxel.py
```

The structure may be adjusted after repository inspection, but every deviation must be documented before implementation.

---

# 4. Core Data Models

Create explicit typed domain models instead of passing large mutable dictionaries.

At minimum, evaluate and implement:

- `CameraFrame`
- `ProcessedFrame`
- `EncodedFrame`
- `TactileSample`
- `WrenchSample`
- `ControllerState`
- `HandState`
- `VRInputState`
- `RobotState`
- `RobotAction`
- `Observation`
- `RuntimeCommand`

Requirements:

- Prefer frozen dataclasses.
- Include source timestamps and sequence numbers in high-frequency samples.
- Validate stable array shapes at module boundaries.
- Do not expose mutable internal buffers.
- Do not create one massive mutable global state object.

Suggested action types:

```python
class RobotCommandType(Enum):
    JOINT_POSITION = auto()
    TCP_POSE = auto()
    JOINT_VELOCITY = auto()
    TCP_TWIST = auto()
    HOLD = auto()
    STOP = auto()
```

---

# 5. Core Interfaces

Define narrow interfaces in domain-specific `base.py` modules.

Required interface categories:

- `CameraSource`
- `FrameProcessor`
- `VideoEncoder`
- `VideoTransport`
- `VRInputSource`
- `TactileSensor`
- `TeleopMapping`
- `ActionFilter`
- `RobotBackend`
- `EpisodeRecorder`

Example:

```python
class CameraSource(Protocol):
    def start(self) -> None: ...
    def read_latest(self) -> CameraFrame | None: ...
    def close(self) -> None: ...


class VideoTransport(Protocol):
    def start(self) -> None: ...
    def send(self, frame: EncodedFrame) -> None: ...
    def close(self) -> None: ...


class TeleopMapping(Protocol):
    def map(
        self,
        input_state: VRInputState,
        robot_state: RobotState,
    ) -> RobotAction | None: ...


class RobotBackend(Protocol):
    def connect(self) -> None: ...
    def read_state(self) -> RobotState: ...
    def execute(self, action: RobotAction) -> None: ...
    def close(self) -> None: ...
```

Base interfaces must not import hardware SDKs.

---

# 6. Communication Architecture

## 6.1 Video Path

Default v2.0 video architecture:

```text
RealSense camera
    -> latest-frame buffer
FrameProcessor
    -> processed RGB frame
H.264 low-latency encoder
    -> encoded frame
WebRTC video track
    -> Quest 3 / Unity receiver
```

Requirements:

- Use H.264 as the primary codec.
- Prefer NVIDIA NVENC when available.
- Support a software fallback.
- Disable B-frames.
- Disable encoder lookahead.
- Use low-latency or zero-latency encoder settings.
- Use bounded queues with depth 1 or another explicitly justified small value.
- Drop old frames instead of accumulating delay.
- Keep camera acquisition independent from transport.
- Support multiple camera tracks.
- Add optional RTP/UDP H.264 for controlled local-network benchmarking.
- Keep JPEG UDP only as a temporary compatibility implementation.
- Do not rely on large UDP datagrams that require IP fragmentation.

Suggested starting configuration:

```yaml
streaming:
  mode: webrtc
  codec: h264
  encoder: nvenc
  fps: 60
  queue_depth: 1
  drop_oldest: true
  b_frames: 0
  lookahead: 0
  gop_size: 30
  target_bitrate_mbps: 4
```

All values must remain configurable.

## 6.2 Real-Time State Path

Use a dedicated binary real-time state channel for:

- VR controller poses.
- VR hand-tracking joints.
- Optional robot joint states sent to the headset.
- Optional low-rate diagnostic status.

Preferred transport:

```text
WebRTC DataChannel
ordered = false
maxRetransmits = 0
```

Requirements:

- Use binary serialization.
- Do not use CSV, JSON, Base64, or comma-separated text in the primary high-frequency path.
- Include protocol version, message type, sequence number, source timestamp, payload length, and flags.
- Reject duplicate or stale sequence numbers.
- Keep only the latest valid state.
- Prefer `float32` payloads.
- Keep serialization independent from transport.

Suggested binary header:

```text
magic          uint16
version        uint8
message_type   uint8
sequence       uint32
timestamp_ns   uint64
payload_size   uint16
flags          uint16
```

## 6.3 Reliable Command Path

Use a separate reliable and ordered channel for:

- Start recording.
- Stop recording.
- Roll back the latest episode.
- Recalibrate tactile baseline.
- Reset wrench baseline.
- Change teleoperation mode.
- Change camera zoom or resolution.
- Request a safe hold.
- Request a controlled stop.

Preferred transport:

```text
WebRTC DataChannel
ordered = true
reliable = true
```

Do not send reliable commands through the real-time state channel.

Define explicit command enums. The protocol parser produces commands, while `CommandRouter` executes the routing.

## 6.4 Robot State Acquisition

For UR robots:

- Continue using RTDE for local robot state acquisition.
- Keep RTDE reading separate from VR communication.
- Store the latest robot state in a thread-safe latest-value buffer.
- Use local high-rate state for control.
- Record at the configured dataset rate.
- Downsample before sending robot state to the headset.

Suggested rates:

```text
UR state acquisition: 250-500 Hz when supported
control loop: backend-dependent
dataset recording: experiment rate, commonly 50 Hz
Quest visualization: 60-90 Hz
dashboard text status: 10-30 Hz
```

For RealMan:

- Isolate RealMan API behavior in `robots/realman.py`.
- Return the same `RobotState` model used by UR.
- Do not expose RealMan SDK types outside the backend.

## 6.5 Time and Latency Instrumentation

Every high-frequency sample must include:

- Sequence number.
- Source timestamp.
- Receive timestamp when useful.
- Processing timestamps when useful.

Add metrics for:

- Camera capture time.
- Frame processing time.
- Encoding time.
- Transmission time.
- Receive time.
- Display time when Unity can report it.
- Packet loss.
- Out-of-order packets.
- Dropped frames.
- Queue depth.
- Bitrate.
- Robot control-loop period.
- End-to-end control latency.
- End-to-end video latency estimate.

Use monotonic clocks for local measurements.

## 6.6 Watchdogs

Add explicit freshness checks for:

- VR input.
- Robot state.
- Video pipeline health.
- Tactile connection.

If control input becomes stale:

1. Stop updating the active target.
2. Send or maintain a safe hold command.
3. Stop velocity-based commands immediately.
4. Require a valid recovery condition before resuming.

The robot hardware safety system and emergency stop remain independent and mandatory.

---

# 7. Deprecated Code Policy

Create:

```text
deprecated/
├── README.md
└── tactile/
    └── magtouch_ilias_41taxel.py
```

Move the legacy tactile implementation using `git mv`.

Requirements:

- Do not delete the file.
- Do not add `deprecated/__init__.py`.
- Supported code must never import from `deprecated/`.
- Deprecated code is excluded from package installation.
- Record the reason, replacement, date, and migration note.

Use this table:

```markdown
| File | Original purpose | Deprecation reason | Replacement | Date |
|------|------------------|--------------------|-------------|------|
```

---

# 8. Refactor Phases and Tasks

# Phase 0: Repository Audit

Do not modify source behavior in this phase.

## Task 0.1: Generate Repository Inventory

Create `docs/refactor/repository_inventory.md`.

For every tracked file, record:

- Path.
- Purpose.
- Main classes and functions.
- Known importers.
- Runtime participation.
- Classification: module, app, script, test, example, or legacy.
- Recommended destination.
- Move risk: low, medium, or high.
- Compatibility wrapper requirement.

Acceptance criteria:

- Every root-level Python file is included.
- Every tactile-related file is identified.
- Every runtime entry point is identified.
- Duplicate responsibilities are documented.
- No source files are changed.

## Task 0.2: Build Dependency Overview

Create `docs/refactor/dependency_overview.md`.

Include:

- Text dependency graph.
- Modules with excessive responsibilities.
- Circular or bidirectional dependencies.
- Hardware modules depending on runtime modules.
- Objects mutating external shared state.
- Global configuration dependencies.
- Optional dependencies imported too early.
- Recommended dependency direction.

Explicitly analyze:

- `main.py`
- `camera_udp.py`
- `WebRTC_udp.py`
- `robot_teleop.py`
- Tactile readers.
- Recording and visualization coupling.

## Task 0.3: Baseline Current Behavior

Create `docs/refactor/behavior_baseline.md`.

Record:

- Current run commands.
- Current configuration values.
- UDP packet formats.
- WebRTC signaling behavior.
- VR text and binary formats.
- HDF5 schema.
- LeRobot schema.
- Robot command rates.
- Camera settings.
- Tactile shape and filtering.
- Shutdown behavior.

Add characterization tests where possible without changing behavior.

## Task 0.4: Produce Migration Map

Create `docs/refactor/migration_map.md`.

For each current file, define:

- Target file or files.
- Move, split, wrapper, deprecation, or later removal strategy.
- Required tests before migration.
- Protocol compatibility concerns.
- Hardware validation requirements.

Do not continue to Phase 1 until the migration map is reviewed.

---

# Phase 1: Package Skeleton and Safe Reorganization

## Task 1.1: Add `src` Package Layout

Create the target package skeleton and `pyproject.toml`.

Acceptance criteria:

- `pip install -e .` succeeds.
- `python -c "import airo_doffy"` succeeds.
- Optional hardware dependencies remain optional.
- Existing run commands remain functional.

## Task 1.2: Add Deprecation Policy

Create `deprecated/README.md` and document unsupported-code rules.

## Task 1.3: Move Legacy Tactile Reader

Use:

```bash
git mv tactile.py deprecated/tactile/magtouch_ilias_41taxel.py
```

Acceptance criteria:

- Legacy code remains in Git history.
- Main runtime no longer imports it.
- 4-taxel tactile support still works.
- No supported module imports `deprecated`.

## Task 1.4: Organize Scripts and Assets

Move:

- Dataset tools -> `scripts/dataset/`
- Calibration tools -> `scripts/calibration/`
- Manual hardware checks -> `scripts/diagnostics/`
- Performance tests -> `scripts/benchmarks/`
- Automated tests -> `tests/`
- Unity/C# code -> `unity/`

Do not classify manual hardware scripts as unit tests.

---

# Phase 2: Core Types, Interfaces, Buffers, and Events

## Task 2.1: Add Core Typed Models

Implement typed, immutable domain models and shape validation.

## Task 2.2: Add Narrow Interfaces

Implement interfaces without importing hardware SDKs.

## Task 2.3: Add Latest-Value Buffers

Create a thread-safe latest-value buffer with:

- Constant memory usage.
- Sequence-aware stale rejection.
- Optional blocking wait for new data.
- Safe shutdown.
- No unbounded real-time queues.

## Task 2.4: Add Runtime Commands and Events

Replace arbitrary runtime strings with typed commands and events.

---

# Phase 3: Robot Backend Refactor

## Task 3.1: Split Robot Backends

Split current robot code into:

- `robots/base.py`
- `robots/ur.py`
- `robots/realman.py`
- `robots/mock.py`
- `robots/grippers/robotiq_2f85.py`

Requirements:

- Isolate hardware SDK imports.
- Return a common `RobotState`.
- Keep gripper behavior separate from arm backends.

## Task 3.2: Add Robot Executors

Separate command scheduling and execution from mapping.

Support:

- Joint position.
- TCP pose.
- Optional velocity modes.
- Hold.
- Stop.

## Task 3.3: Add Mock Robot Backend

Support configurable state, captured commands, artificial latency, and injected failures.

---

# Phase 4: Tactile and Wrench Refactor

## Task 4.1: Add Tactile Interface

Define `TactileSensor` and `TactileSample` with supported shape `(4, 3)`.

## Task 4.2: Refactor 4-Taxel BLE MagTouch

Move:

```text
tactile_4point.py
-> src/airo_doffy/devices/tactile/magtouch_ble4.py
```

Remove external-holder mutation.

Requirements:

- Preserve BLE communication.
- Preserve filtering and baseline calibration.
- Use private thread-safe state.
- Add idempotent `close()`.
- Do not import VR, camera, recording, or visualization modules.

## Task 4.3: Add Mock Tactile Sensor

Support fixed, random, periodic, disconnected, and delayed modes.

## Task 4.4: Refactor Wrench Processing

Separate:

- Raw acquisition.
- Gravity compensation.
- Moving average.
- Low-pass filtering.
- Deadband.
- Baseline reset.

---

# Phase 5: Camera Acquisition Refactor

## Task 5.1: Extract RealSense Camera Source

Responsibilities only:

- Discovery.
- Serial selection.
- Initialization.
- Frame acquisition.
- Timestamping.
- Shutdown.

It must not create transports, parse VR, store tactile data, control recording, access the robot, or update visualization.

## Task 5.2: Add Mock Camera Source

Support static frames, generated frames, video playback, dropped frames, and artificial delay.

## Task 5.3: Add Frame Processor

Separate color conversion, resize, crop, zoom, rotation, and encoding preparation.

---

# Phase 6: VR Input and Protocol Refactor

## Task 6.1: Extract VR Protocol Parsing

Move parsing into pure functions.

Support current legacy formats and the new binary v2 format during migration.

## Task 6.2: Extract VR Receiver

Responsibilities:

- Receive raw messages.
- Timestamp receive events.
- Decode messages.
- Store latest valid state.
- Report malformed or stale packets.

It must run without cameras, robots, tactile sensors, or recording.

## Task 6.3: Add Binary State Protocol v2

Implement versioning, sequence numbers, timestamps, `float32` payloads, and stale-packet rejection.

Add cross-language documentation for Unity.

## Task 6.4: Add Mock VR Input Source

Support scripted trajectories, buttons, stale input, delay, packet loss, and reordering.

---

# Phase 7: Video Streaming Refactor

## Task 7.1: Define Video Interfaces

Implement independent frame processing, encoding, and transport interfaces.

## Task 7.2: Implement Low-Latency H.264 Encoder

Support:

- NVENC.
- Software fallback.
- No B-frames.
- No lookahead.
- Configurable bitrate and GOP.
- Bounded queues.
- Drop-oldest overload behavior.
- Encoding metrics.

## Task 7.3: Refactor WebRTC Video Transport

Responsibilities only:

- Signaling.
- Peer connection lifecycle.
- Video tracks.
- Codec negotiation.
- Connection state.
- Graceful shutdown.

It must not initialize cameras or own dataset, tactile, robot, or mapping state.

## Task 7.4: Add RTP/UDP H.264 Benchmark Transport

Implement as an optional experimental local-network transport with bounded jitter and late-packet dropping.

## Task 7.5: Isolate Legacy JPEG UDP

Move it to `streaming/video/legacy_jpeg_udp.py` and mark it deprecated after compatibility validation.

## Task 7.6: Add Video Benchmark Tool

Compare:

1. Legacy JPEG UDP.
2. WebRTC H.264.
3. RTP/UDP H.264.

Measure latency, loss, queue delay, bitrate, CPU, GPU, and end-to-end display delay where available.

---

# Phase 8: State and Command Channels

## Task 8.1: Add Real-Time State Transport

Default:

- WebRTC DataChannel.
- Unordered.
- No retransmission.
- Latest-only semantics.

Add independent serializer, stale rejection, sequence metrics, and an optional UDP diagnostic transport.

## Task 8.2: Add Reliable Command Transport

Default:

- WebRTC DataChannel.
- Ordered.
- Reliable.

Add duplicate handling, idempotency where needed, acknowledgements, timeouts, and error reporting.

## Task 8.3: Add Command Router

Route typed commands without embedding component implementation.

---

# Phase 9: Teleoperation Mapping and Safety

## Task 9.1: Split Teleoperation Mapping

Do not preserve a large all-in-one `RobotTeleop` class.

Separate:

- Controller pose mapping.
- Hand pose mapping.
- Coordinate transforms.
- Pose delta computation.
- Translation scaling.
- Rotation mapping.
- Gripper mapping.
- Command mode selection.

The mapping must be testable as a deterministic operation without hardware or sockets.

## Task 9.2: Extract Safety Filters

Implement composable filters for:

- Workspace bounds.
- Joint limits.
- Velocity limits.
- Acceleration limits.
- Action freshness.
- IK rejection when applicable.
- Rate limiting.

## Task 9.3: Add Watchdog and Hold Behavior

Detect stale VR input and robot state, stop velocity commands, transition to hold, and require a valid recovery condition.

---

# Phase 10: Recording Refactor

## Task 10.1: Separate Recording State from Serialization

Split:

- Episode state machine.
- Sample buffering.
- HDF5 writer.
- LeRobot writer.
- Rollback implementation.
- Export worker.

Recording must not depend on visualization. Serialization must not depend on hardware.

## Task 10.2: Preserve Existing Schemas

Characterize and test all current paths, shapes, dtypes, representations, numbering, and rollback semantics.

## Task 10.3: Add Non-Blocking Export

Use a bounded export queue. Do not block the control loop on disk or dataset conversion.

---

# Phase 11: Visualization Refactor

## Task 11.1: Convert Visualizer to a Consumer

Visualization receives typed snapshots only.

It must not read hardware directly, mutate recording state directly, or own teleoperation behavior.

## Task 11.2: Preserve Mock Mode

The dashboard must work with mock data and with individual sensor types disabled.

Closing it must not crash the runtime.

---

# Phase 12: Runtime Orchestration

## Task 12.1: Create Lifecycle Management

Responsibilities:

- Start resources in dependency order.
- Track successfully started resources.
- Shut down in reverse order.
- Handle partial initialization failures.
- Make repeated `close()` safe.

## Task 12.2: Create `TeleopSession`

Coordinate VR input, robot state, mapping, safety, execution, optional video, tactile, visualization, commands, and watchdogs.

Do not implement component internals.

Acceptance criteria:

- Works with all mocks.
- Works with optional components disabled.
- Ctrl+C shuts down cleanly.
- No background thread remains alive after shutdown.

## Task 12.3: Create `DataCollectionSession`

Add recording behavior by composition. Do not duplicate the teleoperation loop.

## Task 12.4: Add Thin Application Entry Points

Each app may only:

1. Parse CLI arguments.
2. Load configuration.
3. Construct concrete components.
4. Create a session.
5. Run it.
6. Handle top-level errors and shutdown.

Keep a temporary root `main.py` compatibility wrapper.

---

# Phase 13: Configuration Refactor

## Task 13.1: Add Typed Configuration Models

Suggested sections:

- `NetworkConfig`
- `RobotConfig`
- `CameraConfig`
- `VRConfig`
- `TeleopConfig`
- `TactileConfig`
- `RecordingConfig`
- `VisualizationConfig`
- `VideoStreamingConfig`
- `StateTransportConfig`
- `CommandTransportConfig`
- `WrenchConfig`
- `RuntimeConfig`

Components receive only the sections they need.

## Task 13.2: Add YAML Configuration

Precedence:

1. Default YAML.
2. Robot YAML.
3. Experiment YAML.
4. Environment variables.
5. CLI overrides.

Do not commit secrets or mandatory machine-specific paths.

## Task 13.3: Add Factories

Add focused factories for robot, camera, encoder, transport, VR source, tactile sensor, recorder, and visualizer construction.

---

# Phase 14: Tests and Quality

## Task 14.1: Add Unit Tests

Cover:

- VR text and binary protocols.
- State and command serialization.
- Sequence rejection.
- Latest-value buffers.
- Teleoperation mappings.
- Coordinate transforms.
- Safety filters.
- Watchdogs.
- Tactile and wrench processing.
- Frame processing.
- Encoder configuration.
- UDP packetization.
- Dataset schema and writers.
- Rollback.
- Runtime lifecycle.

No physical hardware is allowed in unit tests.

## Task 14.2: Add Integration Tests with Mocks

Verify a complete short session with mock robot, VR, camera, video, tactile, recorder, and visualizer.

## Task 14.3: Add Hardware Tests

Place hardware-dependent checks in `tests/hardware/` and skip them by default.

## Task 14.4: Add Quality Tools

Configure:

- `ruff`
- `pytest`
- `pyright` or `mypy`
- `pre-commit`

Base package imports must work without optional hardware SDKs.

---

# Phase 15: Documentation and Release

## Task 15.1: Rewrite README

Document architecture, robots, VR modes, transports, channels, tactile support, installation, configuration, operation, recording, inference, evaluation, diagnostics, and deprecation policy.

## Task 15.2: Add Architecture Document

Create `docs/architecture.md` with module responsibilities, dependency direction, data flow, thread ownership, lifecycle, shutdown, errors, and extension points.

## Task 15.3: Add Communication Document

Create `docs/communication.md` with WebRTC signaling, video tracks, H.264 settings, binary state protocol, reliable command protocol, timestamps, sequences, compatibility, and Unity notes.

## Task 15.4: Add Extension Guide

Document how to add a robot, gripper, tactile sensor, camera, video transport, teleoperation mapping, safety filter, and dataset writer.

## Task 15.5: Prepare v2.0 Release

Requirements:

- Update `CHANGELOG.md`.
- Document breaking changes.
- Document compatibility wrappers.
- Remove wrappers only when approved.
- Tag a tested release candidate before final `v2.0`.

---

# 9. Thread and Loop Ownership

Each continuous loop must have one responsibility.

Expected loops:

- Camera acquisition loop.
- Video encoding loop.
- Video transmission loop.
- VR receive loop.
- Robot state acquisition loop.
- Robot control loop.
- Tactile acquisition loop.
- Dataset export loop.
- Visualization update loop.

Each loop must document:

- Input.
- Output.
- Target frequency.
- Owned resources.
- Buffer type.
- Queue capacity.
- Drop policy.
- Shutdown condition.
- Error behavior.
- Watchdog behavior.

Do not create one loop that captures frames, receives VR input, controls the robot, records data, updates the UI, and sends network packets.

---

# 10. Required Decoupling Acceptance Criteria

AIRO-Doffy v2.0 is not complete until all of the following are true:

1. RealSense acquisition runs without UDP or WebRTC.
2. Frame processing runs from a NumPy array without hardware.
3. H.264 encoding runs without a camera.
4. UDP video transport is testable without RealSense.
5. WebRTC transport imports without starting cameras.
6. VR input works without video streaming.
7. VR packets parse without opening sockets.
8. Teleoperation mapping runs from recorded input data.
9. Teleoperation mapping runs with a fake robot state.
10. Safety filters run independently.
11. Robot backends do not import VR modules.
12. Robot backends do not import visualization.
13. Tactile sensing runs without camera or VR objects.
14. Tactile data records without visualization.
15. Visualization runs entirely from mock samples.
16. Recording runs without visualization.
17. HDF5 writing runs without robot hardware.
18. LeRobot writing runs without robot hardware.
19. UDP and WebRTC share camera acquisition code.
20. UDP and WebRTC share VR protocol code where applicable.
21. No component writes arbitrary attributes into another component.
22. No supported module imports from `deprecated/`.
23. Runtime classes coordinate behavior but do not implement device internals.
24. Every background loop has one documented responsibility.
25. Optional hardware dependencies are imported only by their adapters.
26. Every component can be replaced by a mock through an interface.
27. Shutdown of one component does not require unrelated component knowledge.
28. There is no replacement equivalent to the previous all-in-one camera manager.
29. There is no replacement equivalent to the previous all-in-one teleoperation manager.
30. Real-time state uses latest-only behavior.
31. Reliable commands use a separate ordered reliable path.
32. Old video frames are dropped instead of queued indefinitely.
33. Stale control input causes a safe hold.
34. Existing dataset formats remain compatible.
35. Existing Unity protocol behavior remains compatible until v2 migration is complete.

---

# 11. Behavioral Compatibility Constraints

Do not change the following without explicit approval:

1. HDF5 schema.
2. LeRobot schema.
3. Unity message formats during compatibility phases.
4. UDP packet formats before replacement validation.
5. Robot safety limits.
6. Control-loop frequencies.
7. Initial joint configurations.
8. Controller coordinate mappings.
9. Gripper mapping.
10. Tactile output shape.
11. Force and torque conventions.
12. Dataset episode numbering.
13. Episode rollback semantics.

When a behavior must change:

- Document the old behavior.
- Document the new behavior.
- Add migration notes.
- Add compatibility flags where practical.
- Test both modes during migration.

---

# 12. Performance Targets

## Video

- No unbounded frame queues.
- Queue depth normally 1.
- No B-frames.
- No encoder lookahead.
- Hardware encoding when available.
- Stable multi-camera streaming at configured FPS.
- Graceful frame dropping under overload.
- End-to-end latency instrumentation.

## Real-Time State

- Binary packets.
- No string parsing in the primary high-frequency path.
- Latest-state-only buffering.
- Sequence-aware stale rejection.
- No retransmission for pose or joint state.
- Avoid packet fragmentation.

## Reliable Commands

- Ordered reliable delivery.
- Duplicate handling.
- Clear acknowledgement or failure reporting where needed.

## Control

- No network or disk I/O in the hard control path.
- Explicit watchdog.
- Safe hold on stale commands.
- Control-loop timing metrics.

Measure before setting stricter numerical requirements.

---

# 13. Commit Plan

Use focused commits similar to:

1. `docs: add v2 repository inventory`
2. `docs: add dependency and behavior baselines`
3. `build: add src package layout`
4. `chore: add deprecated code policy`
5. `refactor: move legacy tactile reader to deprecated`
6. `refactor: organize scripts tests and unity assets`
7. `refactor: add core typed models`
8. `refactor: add interfaces and latest-value buffers`
9. `refactor: split robot backend implementations`
10. `refactor: add robot executors and mock backend`
11. `refactor: introduce tactile sensor interface`
12. `refactor: decouple four-taxel tactile reader`
13. `refactor: split wrench processing stages`
14. `refactor: extract realsense camera acquisition`
15. `refactor: add frame processor and mock camera`
16. `refactor: extract vr protocol parsing`
17. `refactor: add binary realtime state protocol`
18. `refactor: add independent vr receiver`
19. `refactor: add video encoder interface`
20. `refactor: add low-latency h264 encoder`
21. `refactor: extract webrtc video transport`
22. `refactor: add optional rtp udp video transport`
23. `refactor: isolate legacy jpeg udp transport`
24. `refactor: add realtime state data channel`
25. `refactor: add reliable command channel`
26. `refactor: add runtime command router`
27. `refactor: split teleoperation mappings`
28. `refactor: add composable safety filters`
29. `refactor: add watchdog and safe hold`
30. `refactor: split episode recorder and writers`
31. `refactor: convert visualizer to snapshot consumer`
32. `refactor: introduce runtime lifecycle management`
33. `refactor: add teleop session`
34. `refactor: add data collection session`
35. `refactor: add typed yaml configuration`
36. `test: add mock runtime integration tests`
37. `perf: add communication benchmarks`
38. `docs: add architecture communication and extension guides`
39. `release: prepare airo-doffy v2.0`

After every commit:

- Run unit tests.
- Run import checks.
- Run static checks.
- Report moved and modified files.
- Report behavior changes.
- Report remaining coupling.
- Report unresolved hardware risks.
- Do not continue if the previous commit leaves failing tests.

---

# 14. Required Report After Every Phase

After each phase, produce a report with:

## Components Changed

List every created, moved, split, or modified component.

## Responsibility

Describe each component in one sentence.

## Dependencies

List its internal project dependencies.

## Coupling Removed

Explain which previous cross-component dependencies were removed.

## Remaining Coupling

Identify coupling still requiring work.

## Behavior Preservation

State whether runtime behavior, protocol formats, dataset formats, safety settings, or control frequencies changed.

## Tests

List tests proving components work independently.

## Hardware Validation

List hardware checks still required.

A phase is not complete merely because files were moved.

---

# 15. Codex Execution Rules

Codex must follow these rules:

1. Start with Phase 0 only.
2. Do not perform large-scale edits before completing the audit.
3. Do not delete legacy code without explicit approval.
4. Use `git mv` whenever practical.
5. Preserve existing behavior during structural phases.
6. Add characterization tests before changing complicated behavior.
7. Keep changes focused and reviewable.
8. Do not combine unrelated tasks in one commit.
9. Do not introduce broad manager classes.
10. Do not move a multi-purpose file unchanged and call it refactored.
11. Prefer dependency injection.
12. Prefer typed immutable samples.
13. Prefer bounded latest-value buffers for real-time data.
14. Keep protocol and transport independent.
15. Keep hardware SDKs isolated.
16. Keep optional dependencies optional.
17. Report uncertain classifications instead of guessing.
18. Stop and document the issue if compatibility cannot be preserved.
19. Do not change safety behavior without explicit approval.
20. Do not modify Unity protocols silently.
21. Do not change dataset schemas silently.
22. Do not optimize before adding measurement.
23. Do not hide failures with broad exception handling.
24. Do not leave background threads alive after tests.
25. Do not add unnecessary dependencies.

---

# 16. First Codex Task

Execute only Phase 0.

Do not modify implementation files.

Create:

- `docs/refactor/repository_inventory.md`
- `docs/refactor/dependency_overview.md`
- `docs/refactor/behavior_baseline.md`
- `docs/refactor/migration_map.md`

Then provide a summary answering:

1. Which files should be moved into `deprecated/`?
2. Which root files should become package modules?
3. Which current files contain multiple unrelated behaviors?
4. Which responsibilities are duplicated between UDP and WebRTC implementations?
5. Which modules are most tightly coupled?
6. Which objects mutate unrelated external state?
7. Which optional dependencies are imported too early?
8. Which protocol and dataset compatibility constraints require tests?
9. Which proposed target modules should be merged or split differently after inspecting the real repository?
10. What is the safest revised implementation order?

Do not continue to Phase 1 until the Phase 0 documents have been reviewed.
