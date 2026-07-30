# AIRO-Doffy v2 Extension Guide

## General contract

Add the smallest component that satisfies one v2 protocol. Do not subclass a
multi-purpose manager or copy a complete runtime loop.

Every adapter should follow these rules:

1. Constructor validates configuration but performs no I/O.
2. `start()` acquires resources and may load optional SDKs.
3. Data methods return immutable v2 types.
4. `close()` is idempotent and releases only owned resources.
5. Background work has an explicit owner, stop signal, bounded join, health
   status, and documented queue/drop policy.
6. Vendor exceptions are wrapped only when a v2 error adds useful context.
7. Unit tests use injected fake SDK/transport objects and no physical hardware.
8. Add an opt-in smoke test under `tests/hardware/` when a device is supported.

Optional dependencies belong in a named `pyproject.toml` extra and must not load
when importing `airo_doffy`.

## Factory targets

Focused factories resolve a callable using `module:symbol`. The callable should
return an unstarted component:

| Target kind | Callable signature |
|---|---|
| robot | `(RobotConfig) -> RobotBackend` |
| camera | `(CameraConfig) -> CameraSource` |
| encoder | `(VideoStreamingConfig) -> VideoEncoder` |
| video transport | `(VideoStreamingConfig, NetworkConfig) -> VideoTransport` |
| VR source | `(VRConfig, NetworkConfig) -> VRInputSource` |
| tactile | `(TactileConfig) -> TactileSensor` |
| recorder compatibility | `(RecordingConfig) -> EpisodeRecorder` |
| visualizer | `(VisualizationConfig) -> SnapshotConsumer` |

Example:

```python
def create_my_camera(config: CameraConfig) -> MyCamera:
    return MyCamera(config)
```

Configuration can then select:

```text
my_package.camera:create_my_camera
```

The factory performs a runtime structural-protocol check after construction.

## Add a robot backend

Implement `RobotBackend` from `airo_doffy.robots.base`:

```python
class MyRobotBackend:
    @property
    def name(self) -> str: ...

    @property
    def dof(self) -> int: ...

    def start(self) -> None: ...

    def read_state(self) -> RobotState: ...

    def apply_action(self, action: RobotAction) -> None: ...

    def close(self) -> None: ...
```

Requirements:

- `dof` is stable and matches returned joints.
- joint angles are radians;
- TCP pose is a row-major 4 by 4 homogeneous transform;
- wrench order is `Fx, Fy, Fz, Tx, Ty, Tz`;
- gripper width is metres;
- all samples have increasing sequence and a truthful clock domain;
- unsupported action types fail explicitly;
- HOLD and STOP semantics are documented;
- constructor and factory do not connect or move the robot.

Keep cadence and target repetition out of the backend. Use
`LatestActionExecutor` for generic robots or implement a narrow executor when
the vendor has high-follow timing requirements.

If the SDK is thread-affine, the executor thread must own all SDK calls.
Provide a cached/state-push `RobotStateSource` to the session rather than reading
the shared SDK handle concurrently.

Tests should cover:

- start/read/action/close lifecycle;
- 6/7-DOF validation;
- every supported command type;
- unit and coordinate conversion;
- stale action rejection in the executor;
- injected SDK failure and cleanup;
- HOLD, STOP, disconnect, and repeated close.

## Add a gripper

Implement `Gripper`:

```python
class MyGripper:
    def start(self) -> None: ...
    def read_width_m(self) -> float | None: ...
    def command_width_m(self, width_m: float) -> None: ...
    def close(self) -> None: ...
```

Keep gripper connection and state independent from the arm backend. Validate and
clamp only according to an explicitly documented compatibility rule. Test unit
conversion, limits, reconnect behavior, partial reads/writes, and idempotent
close with a fake socket or SDK.

Use `NullGripper` when a deployment has no gripper.

## Add a tactile sensor

Implement `TactileSensor`:

```python
class MyTactileSensor:
    def start(self) -> None: ...
    def read_latest(self) -> TactileSample | None: ...
    def recalibrate(self) -> None: ...
    def close(self) -> None: ...
```

Acquisition, baseline calibration, and signal filtering should be separate
objects when they can vary independently. Do not write results onto a camera,
robot, visualizer, or recorder.

Document:

- stable sample shape and units;
- device/callback thread ownership;
- calibration sample count;
- disconnect and reconnection behavior;
- latest buffer and drop policy.

Tests must include fixed values, noise, calibration, disconnect, recalibration,
shape validation, and operation without camera or VR components.

## Add a camera

Implement `CameraSource`:

```python
class MyCameraSource:
    def start(self) -> None: ...
    def read_latest(self) -> CameraFrame | None: ...
    def close(self) -> None: ...
```

The source acquires raw color/depth frames only. Cropping, resizing, color
conversion, encoding, UDP, WebRTC, and visualization belong elsewhere.

`CameraFrame` must detach bytes from SDK buffers and specify:

- sequence and timestamps;
- stable stream ID;
- packed shape;
- pixel format.

For threaded acquisition, publish into a latest-only buffer and expose
connection/error metrics. Unit tests inject a fake SDK camera and verify
discovery, retry thresholds, frame detachment, drop behavior, worker failure,
and shutdown.

## Add a frame processor

Implement the pure `FrameProcessor.process(CameraFrame) -> ProcessedFrame`
contract. It must not access a camera or network.

Test all geometry and pixel-format operations using small deterministic byte
arrays. H.264 preparation must ensure even dimensions and reject depth input
unless a dedicated depth codec is provided.

## Add a video encoder

Implement `VideoEncoder.encode(ProcessedFrame) -> EncodedFrame`.

The encoder must not:

- own camera capture;
- create an unbounded queue;
- send network packets;
- silently change geometry mid-stream.

If initialization is lazy, expose the selected codec/backend. Document
B-frames, lookahead, GOP, bitrate, time base, and hardware-to-software fallback.
Use `LatestVideoEncodingPipeline` for bounded asynchronous encoding.

Tests inject a fake codec builder and cover configuration, fallback, empty
output, geometry changes, error propagation, queue pressure, and cleanup.

## Add a video transport

Implement:

```python
class MyVideoTransport:
    def start(self) -> None: ...
    def send(self, frame: EncodedFrame) -> None: ...
    def close(self) -> None: ...
```

The transport consumes encoded access units; it does not discover cameras or
encode pixels. Specify supported codecs, stream IDs, MTU/fragmentation,
ordering, retransmission, stale rejection, and metrics.

Tests use fake sockets, peer connections, or event loops. They must not require
RealSense. Verify exact packet bytes when a compatibility format is involved.

## Add a VR mode or raw transport

New wire formats belong in a decoder/encoder module and should produce
`VRInputState`. Network code implements `RawVRTransport` and delivers one
complete message at a time to `VRReceiver`.

Preserve:

- side and entity-count validation;
- metres and XYZW quaternion convention;
- device timestamp and sequence;
- version and reserved-field checks;
- unsigned sequence wrap handling.

Do not add video or robot behavior to a VR receiver.

## Add a teleoperation mapping

Implement:

```python
class MyMapping:
    def map_input(
        self,
        vr_input: VRInputState,
        robot_state: RobotState,
        dt_s: float,
    ) -> RobotAction: ...
```

A mapping may own reference/rebase state but performs no I/O. Coordinate
transforms should be pure functions under `teleop.transforms`. Keep candidate
generation separate from safety limits.

Tests should use recorded or constructed VR/state values and cover:

- initial reference;
- controller/hand release and re-engagement;
- axis mapping and rotation composition;
- gripper behavior;
- sequence/timestamp propagation;
- 6/7-DOF differences;
- invalid or unavailable IK.

## Add a safety filter

Implement:

```python
class MyFilter:
    def apply(
        self,
        action: RobotAction,
        robot_state: RobotState,
        now_ns: int,
    ) -> RobotAction | None: ...
```

Return:

- a safe action when accepted or clamped;
- `None` when the action must be rejected.

Filters must not submit commands or read networks. Add them to
`SafetyFilterChain` in an explicit order. For stateful rate/acceleration limits,
document reset and first-sample behavior.

Tests cover boundaries, non-finite values, stale state, command-type handling,
clamp versus reject policy, metrics, and independent operation.

## Add a dataset writer

Implement `EpisodeWriter`:

```python
class MyEpisodeWriter:
    def write(self, episode: Episode) -> Path | None: ...
    def close(self) -> None: ...
```

Optionally implement `EpisodeRollback.rollback(index) -> bool`.

The writer receives a sealed immutable episode and never reads hardware.
Schema mapping belongs in the writer; episode state and queue ownership remain
in `RecordingCycleExtension` and `ExportWorker`.

Requirements:

- validate the complete schema before partial storage mutation;
- use temporary/staging output and atomic publish where practical;
- preserve explicit episode index;
- make close idempotent;
- define rollback of missing, partial, and latest data;
- do not perform uploads unless configuration explicitly enables them.

Tests use a temporary directory and cover schema fields, dtypes, shapes,
numbering, failure cleanup, rollback, and retry. Tests for optional storage
libraries should skip with a clear dependency message rather than silently
mocking the serializer.

## Add a visualizer

Prefer implementing a `SnapshotRenderer` and reuse `TypedSnapshotConsumer`.
Renderers consume only `VisualizationSnapshot` and return whether the UI should
stay open.

User actions are emitted as `RuntimeCommand` through
`VisualizationCommandOutbox`; a visualizer must not mutate recorder, camera,
robot, or tactile instances directly.

Test with `MemorySnapshotRenderer` and snapshots where every optional sensor is
absent as well as complete snapshots.

## Add a session extension

Implement `SessionExtension`:

```python
class MyExtension:
    def start(self) -> None: ...
    def on_cycle(self, cycle: object) -> None: ...
    def close(self) -> None: ...
```

`on_cycle` runs on the session thread and must stay bounded. Disk, network, heavy
encoding, and blocking UI work should be submitted to an owned bounded worker.
If the extension has background health, expose `check_health()`.

Do not combine camera capture, video encoding, transport, sensor acquisition,
recording, and UI into a production all-in-one extension.

## Compose an application

A deployment module exports a factory:

```python
from airo_doffy.apps.common import ApplicationSession
from airo_doffy.config import AiroDoffyConfig


def build_teleop(config: AiroDoffyConfig) -> ApplicationSession:
    robot = build_robot(config.robot)
    state_source = build_sdk_safe_state_source(robot, config.robot)
    executor = build_executor(robot, config.robot, config.teleop)
    vr = build_vr(config.vr, config.network)
    mapping = build_mapping(config.teleop, robot.dof)
    safety = build_safety(config.teleop, config.robot)
    extensions = build_extensions(config)
    return build_session(
        config,
        vr=vr,
        state_source=state_source,
        executor=executor,
        mapping=mapping,
        safety=safety,
        extensions=extensions,
    )
```

The helper names are deployment-owned placeholders, not library globals. This
is intentional: IPs, force frames, safety limits, tool geometry, cameras,
writers, and emergency-stop procedures are workcell policy.

Run:

```bash
airo-doffy-teleop \
  --config configs/default.yaml \
  --session-factory my_deployment.sessions:build_teleop
```

## Pull-request checklist

- [ ] New component targets one narrow protocol.
- [ ] Constructor performs no I/O.
- [ ] Optional dependency is lazy and declared in an extra.
- [ ] Lifecycle and thread ownership are documented.
- [ ] Queue capacity and drop/reject policy are explicit.
- [ ] Immutable types preserve units, timestamps, and sequences.
- [ ] No import from `deprecated/`.
- [ ] Unit tests run without physical hardware.
- [ ] Compatibility bytes/schema are protected by golden tests.
- [ ] Hardware smoke test is opt-in and has safety instructions.
- [ ] Architecture, communication, or migration docs are updated if behavior
      changes.
