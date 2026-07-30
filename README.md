# AIRO-Doffy 2.0

AIRO-Doffy is a modular VR teleoperation and data-collection runtime for UR and
RealMan manipulators. Version 2 separates hardware acquisition, protocol
decoding, teleoperation mapping, safety, robot execution, video transport,
recording, visualization, and application orchestration behind typed interfaces.

The repository is currently `2.0.0.dev0`. The v2 package and compatibility
wrappers are implemented and covered by hardware-free tests. A production
release candidate has not been tagged, and physical UR, RealMan, RealSense,
Quest, and BLE4 validation remains an explicit deployment step.

## What v2 provides

- Immutable, timestamped robot, VR, camera, tactile, wrench, action, and
  observation models.
- UR3e/UR5e, RealMan RM75, mock robot, Robotiq 2F-85, RealSense, Quest VR, and
  four-taxel MagTouch adapters.
- Controller and hand-tracking input with legacy text compatibility and a
  versioned binary protocol.
- Pure pose transforms, replaceable controller/hand mappings, composable safety
  filters, stale-input watchdogs, and backend-independent action executors.
- Independent frame processing, bounded latest-frame encoding, legacy JPEG/UDP,
  RTP/H.264 UDP, and WebRTC video transports.
- Separate binary latest-only state and reliable ordered command channels.
- Typed HDF5 and LeRobot schemas, bounded asynchronous export, rollback, and
  explicit episode state.
- A typed, latest-only visualization consumer that can run entirely from mock
  snapshots.
- Thin teleoperation and collection applications with deterministic lifecycle
  and cooperative shutdown.

## Architecture at a glance

```text
devices -> immutable samples -> mapping -> safety -> action executor -> robot
   |                              |
   +-> frame/video pipelines      +-> immutable TeleopCycle
   +-> tactile/wrench                  |-> recording extension
                                       |-> visualization consumer

unreliable latest state  !=  reliable ordered runtime commands
```

The dependency direction points inward toward `airo_doffy.core`. Hardware SDKs
are imported only inside adapters and only when their lifecycle starts. Runtime
classes coordinate components but do not parse packets, process images, access
SDKs, or serialize datasets.

See:

- [Architecture](docs/architecture.md)
- [Communication protocols](docs/communication.md)
- [Extension guide](docs/extension_guide.md)
- [Migration guide](docs/migration_v2.md)
- [Changelog](CHANGELOG.md)

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
python -m pip install -e .
```

The base installation has no mandatory third-party runtime dependency and is
safe to import without hardware SDKs. Install only the extras needed by a
deployment:

```bash
python -m pip install -e ".[config]"
python -m pip install -e ".[robot-ur]"
python -m pip install -e ".[robot-realman]"
python -m pip install -e ".[camera-realsense]"
python -m pip install -e ".[video-h264,video-webrtc]"
python -m pip install -e ".[tactile-ble4]"
python -m pip install -e ".[recording,visualization]"
python -m pip install -e ".[policy]"
python -m pip install -e ".[dev]"
```

Relevant extras are declared in `pyproject.toml`. Some robotics libraries may
require their own platform-specific installation steps.

## Configuration

The v2 configuration is immutable and split into network, robot, camera, VR,
teleoperation, tactile, recording, visualization, video, state transport,
command transport, wrench, and runtime sections.

Precedence is:

1. `configs/default.yaml`
2. optional robot profile from `configs/robots/`
3. optional experiment profile from `configs/experiments/`
4. `AIRO_DOFFY__SECTION__FIELD` environment variables
5. `--set section.field=value` CLI overrides

The committed default contains no device IP, camera serial, secret, or
machine-specific absolute path. Deployment values must be supplied explicitly.

Example:

```bash
airo-doffy-teleop \
  --robot-config configs/robots/ur3e.yaml \
  --experiment-config configs/experiments/vr_hand_tracking.yaml \
  --set robot.ip=192.0.2.10 \
  --set network.vr_ip=192.0.2.20 \
  --session-factory my_deployment.sessions:build_teleop
```

On PowerShell, environment overrides look like:

```powershell
$env:AIRO_DOFFY__ROBOT__IP = '"192.0.2.10"'
$env:AIRO_DOFFY__NETWORK__VR_IP = '"192.0.2.20"'
```

JSON scalars and arrays are parsed before falling back to plain text. Unknown
sections, fields, and malformed values fail before hardware construction.

## Applications and composition

Installed entry points:

```bash
airo-doffy-teleop --help
airo-doffy-collect --help
```

Both applications require a composition factory through `--session-factory` or
`AIRO_DOFFY_TELEOP_SESSION_FACTORY` /
`AIRO_DOFFY_COLLECT_SESSION_FACTORY`. A factory is a callable that accepts one
`AiroDoffyConfig` and returns an object with `start()`, `run()`,
`request_stop()`, and `close()`.

This explicit boundary prevents the library from guessing which robot, cameras,
ports, optional sensors, writer, and safety policy are appropriate for a
workcell. The [extension guide](docs/extension_guide.md) contains a composition
example.

The root `main.py`, `realman_teleop.py`, and `inference.py` remain compatibility
entry points during migration. They still use legacy configuration and should
not be used as examples for new integrations.

## Supported components

### Robots and grippers

| Component | v2 adapter | Notes |
|---|---|---|
| UR3e / UR5e | `URRobotBackend` | 6 DOF; position or optional torque mode |
| RealMan RM75 | `RealManRobotBackend` | 7 DOF; CAN-FD executor available |
| Hardware-free | `MockRobotBackend` | latency and failure injection |
| Robotiq 2F-85 | `Robotiq2F85Gripper` | isolated gripper lifecycle |
| Hardware-free gripper | `NullGripper` | deterministic state |

Robot backends expose state and atomic actions. Cadence, latest-command
scheduling, watchdogs, and shutdown belong to executors and runtime sessions.

### VR

- `controller`: exactly one left and one right controller.
- `hand`: one or two hands, each with 26 joints and optional wrist pose.
- Legacy text controller/hand messages remain parseable.
- Binary v2 messages carry an explicit magic, version, flags, sequence, timestamp,
  payload length, and CRC32.

Packet parsing does not open sockets. `VRReceiver` accepts an injected raw
transport and rejects stale sequences before publishing the latest typed state.

### Video and cameras

RealSense acquisition publishes camera frames independently of transport.
`PackedFrameProcessor` performs crop, zoom, nearest-neighbor resize, rotation,
color conversion, and even-dimension preparation without camera SDKs.

Video choices:

- legacy JPEG/UDP for existing Unity receivers;
- RTP/H.264 over UDP with RFC 6184 packetization and jitter handling;
- WebRTC video tracks with signaling and latest-only submission.

Encoding queues are bounded, default to depth 1, and drop old frames under
overload. Hardware H.264 is preferred when configured and available; software
fallback is explicit and observable.

### Tactile and wrench

The supported new tactile path is the four-taxel BLE4 MagTouch adapter with a
fixed `(4, 3)` output. Filtering, calibration, disconnect handling, mocks, and
recording are independent of camera and VR objects. Older serial tactile code is
retained only through compatibility modules.

Wrench collection, gravity compensation, calibration, moving average, low-pass,
deadband, and clamp stages are separate. The canonical order is
`Fx, Fy, Fz, Tx, Ty, Tz`.

## State and command channels

High-frequency state and runtime commands intentionally use different delivery
semantics:

| Channel | Delivery | Buffering | Typical contents |
|---|---|---|---|
| state | unordered, unreliable | latest only | pose, joints, gripper, wrench |
| commands | ordered, reliable | bounded queue + dedupe | start/stop recording, rollback, recalibrate, stop |

Both paths use versioned binary envelopes, timestamps, sequence numbers, payload
length validation, and CRC32. See [communication](docs/communication.md) for
wire details and Unity migration notes.

## Recording

`RecordingCycleExtension` observes the existing teleoperation loop; it does not
create a second control loop. Samples are validated against an immutable
`RecordingSchema`, detached from live buffers, and exported by a bounded worker.

Supported writers:

- ACT-compatible HDF5;
- LeRobot features and episodes.

Episode numbering uses the highest persisted index plus one. Rollback is
explicit, serialized with export operations, and reuses the removed index.
Failed export remains visible and can be retried or discarded.

## Policy inference and evaluation

The `policy` optional dependency group contains the packages needed by the
legacy policy path. The current `inference.py` remains a compatibility entry
point while policy adapters are moved behind v2 interfaces:

```bash
python inference.py --policy ./checkpoints/my_policy --device cuda --fps 10
```

Do not place policy inference, model downloads, or disk I/O in the robot control
loop. A v2 deployment should publish bounded actions into the same safety and
executor path used by teleoperation.

## Diagnostics and observability

Components expose immutable metrics for accepted/rejected sequences, dropped
frames, queue pressure, encode latency, command acknowledgements, export state,
worker health, and last error. Failures are surfaced through typed errors and
`check_health()` rather than hidden inside background threads.

The repository includes a video benchmark runner and mock renderers. Production
dashboards should consume `VisualizationSnapshot`; they must not read mutable
robot, camera, recorder, or tactile internals.

## Testing

The default hardware-free suite is:

```bash
python -m unittest discover -s tests
```

It includes unit tests, package-boundary checks, a complete mock integration
session, default-skipped hardware checks, and compatibility characterization.

Device-specific smoke tests live under `tests/hardware/` and require explicit
environment variables. Read [their safety notes](tests/hardware/README.md)
before enabling them.

With development dependencies installed:

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m pyright
python -m pre_commit run --all-files
```

## Repository layout

```text
src/airo_doffy/
  apps/            thin CLI and configuration handoff
  config/          typed sections, layered loader, lazy factories
  core/            immutable types, clocks, buffers, errors, interfaces
  devices/         camera, VR, tactile, and wrench adapters
  robots/          robot/gripper adapters and action executors
  teleop/          transforms, mappings, safety filters, watchdog
  streaming/       video, state, and reliable command channels
  recording/       schema, samples, state, writers, export worker
  visualization/   typed snapshots, consumer, commands, mock renderer
  runtime/         lifecycle, teleop session, data-collection composition
configs/           default, robot, and experiment YAML layers
docs/              architecture, protocols, migration, and phase evidence
tests/unit/        hardware-free component tests
tests/integration/ complete mock session
tests/hardware/    explicitly enabled device smoke tests
deprecated/        unsupported legacy code retained for migration only
```

## Deprecation policy

- Existing root imports and entry points remain compatibility wrappers during
  the v2 migration.
- Supported `src/airo_doffy` modules do not import from `deprecated/`.
- Compatibility protocol and dataset behavior remains frozen unless a change is
  explicitly approved and documented with tests and migration notes.
- New integrations should import from `airo_doffy`, use typed configuration,
  and inject dependencies through interfaces.
- Wrapper removal requires a separately approved release change; it is not part
  of the current development release.

## License and release status

No `v2.0` release tag is created by this refactor. Before tagging a release
candidate, run the dev toolchain, HDF5 tests, applicable supervised hardware
checks, and deployment-specific end-to-end latency measurements.
