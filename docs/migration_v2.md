# Migrating to AIRO-Doffy v2

## Migration strategy

Version 2 is available alongside root compatibility modules. Migrate one
boundary at a time:

1. install/import the `airo_doffy` package;
2. move configuration to layered typed YAML;
3. replace legacy data holders with immutable v2 samples;
4. replace device/transport managers with narrow adapters;
5. move mapping and safety into the v2 teleoperation pipeline;
6. compose a `TeleopSession` or `DataCollectionSession`;
7. switch the executable to the installed v2 CLI;
8. remove deployment imports of compatibility wrappers;
9. run protocol, dataset, mock integration, and supervised hardware checks.

Do not remove root wrappers merely because one deployment has migrated. Wrapper
removal requires an explicitly approved release change.

## Module mapping

| Legacy module or responsibility | v2 location | Status |
|---|---|---|
| `config.py` | `airo_doffy.config` + `configs/` | new path ready; root retained |
| `robot_backend.py` | `airo_doffy.robots` | root is lazy compatibility facade |
| robot cadence in teleop classes | `robots.executor` / `realman_executor` | split |
| `tactile_4point.py` | `devices.tactile.magtouch_ble4` | root facade retained |
| old serial tactile | compatibility/deprecated path | not a supported v2 backend |
| `force_filter.py` | `devices.wrench` | pure stages available |
| camera acquisition in `camera_udp.py` | `devices.cameras` | split |
| JPEG/UDP transfer in `camera_udp.py` | `streaming.video.legacy_jpeg_udp` | wire frozen |
| WebRTC camera manager | `streaming.video.webrtc_transport` | transport split |
| `parse_vr.py` | `devices.vr` | typed parser/receiver |
| state UDP helpers | `streaming.state` | binary latest-state path added |
| control strings | `streaming.commands` | reliable typed path added |
| `data_schema.py` / `dataset.py` | `recording` | schema/state/writers split |
| `visualizer.py` | `visualization` | typed consumer added; root retained |
| `main.py` | `apps` + `runtime` + deployment factory | root runtime retained |
| `realman_teleop.py` | robot/runtime components | compatibility runtime retained |
| `inference.py` | future policy adapter + v2 executor | compatibility runtime retained |

## Configuration migration

### Before

Legacy code imports a mutable/root `Config` class and relies on embedded IPs or
module attributes:

```python
from config import Config

config = Config()
robot_ip = config.ROBOT_IP
```

### After

Load immutable sections before constructing components:

```python
from airo_doffy.config import load_config

config = load_config(
    "configs/default.yaml",
    robot_path="configs/robots/ur3e.yaml",
    experiment_path="configs/experiments/collect_ur3e.yaml",
    cli_overrides={"robot.ip": '"192.0.2.10"'},
)
robot_config = config.robot
```

Important differences:

- device addresses default to `None`;
- unknown fields fail instead of becoming arbitrary attributes;
- list-like values normalize to immutable tuples;
- components receive only their relevant section;
- environment/CLI override names are explicit and validated.

Precedence is default, robot, experiment, environment, then CLI.

## Lifecycle migration

### Before

Construction may connect devices, start threads, or discover cameras. Shutdown
can depend on unrelated manager internals.

### After

```python
component = factory(config_section)  # no I/O
component.start()                    # explicit acquisition
try:
    ...
finally:
    component.close()                # idempotent release
```

Use `LifecycleManager` for ordered startup and reverse shutdown. Use
`ManagedWorker` for caller-owned stoppable threads. Never discard a thread
handle or rely on daemon exit for normal cleanup.

## Data-model migration

Replace mutable arrays/dictionaries crossing subsystem boundaries with:

- `RobotState`
- `RobotAction`
- `VRInputState`
- `CameraFrame`
- `ProcessedFrame`
- `EncodedFrame`
- `TactileSample`
- `WrenchSample`
- `Observation`
- `RecordingSample`
- `VisualizationSnapshot`

Migration considerations:

- sequences are non-negative and monotonic within a stream;
- timestamps are integer nanoseconds;
- clock domain is explicit;
- joint values are radians;
- position is metres;
- quaternions are XYZW;
- TCP transforms are row-major 4 by 4 tuples;
- data bytes are detached from SDK buffers.

Do not add mutable runtime fields to these values or attach state to a different
component.

## Robot migration

1. Select `URRobotBackend`, `RealManRobotBackend`, or a custom `RobotBackend`.
2. Select an action executor and explicit cadence.
3. Provide a `RobotStateSource` that respects SDK ownership.
4. Add mapping, safety filters, and watchdog outside the backend.
5. Keep gripper lifecycle separate.

The v2 UR adapter does not automatically move to configured initial joints.
The v2 RealMan backend translates one action; high-follow cadence, slew limits,
rate gate, and heartbeat belong to `RealManCanfdExecutor`.

For thread-affine SDKs, do not pass the same live backend to independent reader
and writer threads. Use state push or an executor-owned cache.

## VR and teleoperation migration

Legacy controller CSV, hand text, and base64 hand payloads remain accepted.
Move parsing to `devices.vr` and pass only `VRInputState` to mappings.

Candidate mapping and safety are intentionally separate:

```text
VRInputState + RobotState
    -> TeleopMapping
    -> RobotAction candidate
    -> ActionFilter chain
    -> TeleopWatchdog
    -> ActionExecutor
```

Existing coordinate axes, initial reference behavior, rotation composition,
gripper mapping, limits, and watchdog thresholds must be copied into typed
configuration and verified by recorded-input tests before switching a robot.

## Video migration

Split the old camera manager into:

```text
CameraSource
  -> FrameProcessor
  -> bounded VideoEncodingPipeline
  -> VideoTransport
```

Keep `packetize_legacy_jpeg` when the Unity client still expects the frozen
12-byte header. Prefer H.264/WebRTC or explicitly select RTP/UDP for new
deployments.

Do not instantiate one camera per transport. UDP and WebRTC should consume the
same acquisition/processing output in a given deployment.

## State and command migration

Do not send reliable operations on the realtime state channel.

- state: `realtime_state`, unordered, `maxRetransmits=0`, latest-only binary;
- commands: `commands`, ordered and reliable, strict JSON with ACK/dedupe.

During Unity migration, run new paths as opt-in and preserve legacy ports. Switch
only after packet-golden, loss/reordering, retry, and end-to-end tests pass.

## Tactile and wrench migration

The supported v2 tactile shape is `(4, 3)` from the BLE4 MagTouch path. Replace
external attribute mutation with `read_latest()`. Visualization and recording
consume copied typed samples.

Wrench stages are composed in this order when enabled:

```text
source -> gravity compensation -> calibration -> moving average
       -> low pass -> deadband -> delta/absolute clamp
```

Preserve the established `Fx, Fy, Fz, Tx, Ty, Tz` convention and explicitly
select the vendor force frame.

## Recording migration

Build one immutable `RecordingSchema`, convert each cycle to a detached
`RecordingSample`, and let `RecordingCycleExtension` own episode state.

Behavior intentionally preserved:

- ACT/HDF5 field paths and shapes;
- LeRobot feature mapping;
- highest-index-plus-one numbering;
- rollback deletes an explicit episode and reuses its index.

Behavior changed:

- export runs on a bounded worker;
- queue-full and serializer errors are visible;
- failed episodes remain available for retry/discard;
- recording observes the teleop loop instead of owning another loop.

Run the real HDF5 tests with `h5py` installed before migrating stored datasets.

## Visualization migration

Build `VisualizationSnapshot` outside the renderer and publish it to a
`SnapshotConsumer`. Every sensor field is optional.

UI actions become `RuntimeCommand` values. A renderer must not directly call
recorder rollback, camera resolution, tactile calibration, or robot methods.

## Application migration

Create a deployment composition factory:

```python
def build_teleop(config: AiroDoffyConfig) -> ApplicationSession:
    ...
```

Then run:

```bash
airo-doffy-teleop \
  --config configs/default.yaml \
  --session-factory my_deployment.sessions:build_teleop
```

Use `airo-doffy-collect` for a factory that returns a data-collection session.
Ctrl+C is translated into cooperative stop and reverse-order close.

There is deliberately no built-in production factory that guesses workcell
safety or hardware topology.

## Breaking changes

The following are breaking for code that moves directly to v2:

- constructors no longer imply connection/start;
- configuration fields are lowercase, typed, immutable, and sectioned;
- IPs and serial numbers have no machine-specific defaults;
- component results are immutable typed values rather than shared attributes;
- camera acquisition is not bundled with video/VR transport;
- state and reliable commands are separate protocols;
- realtime paths reject stale sequences;
- recording export is asynchronous and bounded;
- unsupported tactile backends fail instead of being selected implicitly;
- application CLIs require an explicit composition factory;
- optional SDK failures occur when selecting/starting an adapter, not at base
  package import.

## Compatibility guarantees

Until separately approved:

- legacy Unity VR messages remain parseable;
- JPEG/UDP packet bytes remain frozen;
- WebRTC signaling envelope shape remains compatible;
- HDF5 and LeRobot schemas remain compatible;
- robot safety limits, axes, initial joints, gripper behavior, tactile shape,
  wrench convention, episode numbering, and rollback semantics do not change;
- root compatibility modules remain present.

Supported `src/airo_doffy` code does not depend on `deprecated/`.

## Verification checklist

- [ ] Base package imports without hardware extras.
- [ ] Layered configuration contains no secrets or mandatory machine paths.
- [ ] Mock unit and integration suites pass.
- [ ] HDF5 tests pass with recording extras.
- [ ] Protocol golden and loss/reordering tests pass.
- [ ] Dataset schema/numbering/rollback is checked against existing data.
- [ ] Selected Unity build negotiates video and both DataChannels.
- [ ] VR axes, reference, rotation, gripper, and safety limits are replay-tested.
- [ ] UR/RealMan state reads use an SDK-safe owner/cache.
- [ ] Supervised device smoke tests pass.
- [ ] HOLD, STOP, disconnect, startup rollback, and Ctrl+C shutdown are tested.
- [ ] End-to-end video and control latency meets the workcell target.
- [ ] No compatibility wrapper is removed without release approval.
