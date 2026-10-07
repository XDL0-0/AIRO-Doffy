# doffy-teleop: VR Teleoperation for Robot Manipulators

> **Latest VR APK**: [AIRO Doffy v0.9.7, code 18, Android ARM64](apk/AIRO_Doffy_v0.9.7_arm64_code18.apk), package `com.AIROLab.AIRODOFFY`. This build includes BODY telemetry; enable it in the Session page as described below.
>
> **Unity source**: [AIRO-Doffy project at `811d0b4`](https://github.com/XDL0-0/AIRO-DOFFY-APP/tree/811d0b4f9e9d6368a7fb2402943b41eb324c7167/AIRO-Doffy), published on 2026-10-07 with the v0.9.7/BODY implementation. Use `Doffy.Editor.TeleopBuild.BuildMetaUpdateArm64Only` for the code18 Meta package. This is a post-release source snapshot; the original APK build revision and bit-for-bit reproducibility remain unverified. See [release provenance](apk/RELEASE.md). The repository root / `v0.6.0` tag is historical and has no BODY telemetry.

The project is named **doffy-teleop**; its Python module uses an underscore, `doffy_teleop`. The teleoperation implementation lives in `doffy_teleop/`; the five public root Python files are CLI launchers for teleop, teach/recollect and BODY workflows. Library imports use the packaged paths listed in the [module layout and migration guide](docs/module-layout.md). The pinned Unity source uses Unity 6000.5.6f1 and locks Meta XR All-in-One to 205.0.0. See the [feature inventory, UI previews and validation status](docs/teleop_refactor/README.md); headset display, interaction and robot hardware acceptance remain pending.

A high-performance codebase for controlling robot manipulators (UR3e, UR5e, RealMan, or compatible backends) using VR controllers or hand tracking via UDP. It features camera streaming to the VR headset via **HD chunked UDP** or **WebRTC** (aiortc), low-latency robot control, tactile sensing integration, and dataset recording (HDF5 & LeRobot formats).

## Repository contents

The published source covers UR/RealMan teleoperation, WRM input mapping,
BrainCo hand control, Beaver and other teleop sensors, VR communication,
camera streaming, BODY visualization, recording/replay/recollection, dataset
tools and their related tests. It includes five root CLI launchers, the
`doffy_teleop` modules used by these workflows, packaged UI assets, Beaver
firmware, Quest protocol sources, documentation and dependency files.

Seahorse, deployment, policy training/inference/evaluation, Jev and standalone experiments
are local research work outside this upload scope.

Datasets, trained checkpoints, videos, run outputs, manuscripts, local cluster
jobs (`.gpulab/`), machine audit records (`evidence/`) and workstation settings
stay local. `.gitignore` excludes these files, including uppercase video
extensions. Supply datasets separately before running dataset tools that
require them.

The selected Python upload subset passed **433 tests and 12 subtests**, with
**3 skipped**, without adding an external simulation directory to the import
path. Unity/Quest and robot hardware acceptance remain separate; see the
[validation records](docs/teleop_refactor/README.md#验证与复现).

## Key Features
- **Low-Latency Teleoperation**: Real-time VR controller tracking to robot end-effector mapping with backend-specific IK/control and safety limits.
- **Hand Tracking Support**: Receive and visualize 24-bone hand skeleton data from Meta Quest hand tracking (text and binary protocols).
- **HD Video Streaming**: Two transport options:
  - **UDP** (`doffy_teleop/media/udp_manager.py`): Chunked JPEG transfer, compatible with `UdpSocketMultiHD.cs`.
  - **WebRTC** (`doffy_teleop/media/webrtc_manager.py`): aiortc-based multi-track video (H.264/VP8) with WebSocket signaling + DataChannel for control. Lower bandwidth, NAT-friendly.
- **Fast Gripper Control**: Custom non-blocking TCP socket implementation for the Robotiq 2F-85 gripper.
- **Tactile & Force Integration**: Supports serial MagTouch and 4-taxel BLE MagTouch readers, UR force/torque readings, gravity compensation, baseline reset, and configurable wrench filtering.
- **Live Teleop Visualizer**: Optional multiprocessing matplotlib dashboard for force/torque, camera previews, TCP/joint status, tactile bubbles, dataset status, and last-episode rollback.
- **Dataset Recording**: Save robotic trajectories directly in ACT (HDF5) or Hugging Face `lerobot` formats.
- **Dataset Rollback**: Delete the latest recorded ACT/HDF5 or LeRobot episode from the VR record-control channel or the visualizer.
- **BrainCo Hand Control**: Control Revo2 presets from the controller or all six motors from hand tracking, including separate thumb flexion and rotation channels.
- **BODY Visualization**: Inspect Meta body telemetry in an independent skeleton viewer, with optional RealMan or Classic teleop.
- **Hand Visualizer**: Real-time matplotlib 3D hand skeleton visualization with finger bone connections and dynamic axis scaling.

## Installation

### 1. Prerequisites
- **Python 3.10+** (Recommended: Conda environment `airo-mono`)
- **Robot**: Compatible robot backend, such as UR3e/UR5e with RTDE enabled or RealMan over its network API.
- **Cameras**: Intel RealSense Cameras.
- **VR Setup**: Meta Quest running the [v0.9.7 ARM64 APK](apk/AIRO_Doffy_v0.9.7_arm64_code18.apk). Unity project: [AIRO-Doffy at the pinned source SHA](https://github.com/XDL0-0/AIRO-DOFFY-APP/tree/811d0b4f9e9d6368a7fb2402943b41eb324c7167/AIRO-Doffy); open this subfolder, not the historical project at the repository root.

### 2. Install the VR APK

With a Quest connected through ADB and USB debugging authorized, install the
downloaded APK from the repository root:

```bash
adb install -r apk/AIRO_Doffy_v0.9.7_arm64_code18.apk
```

The APK's signature has been verified, and installation and application startup
on Quest 3 succeeded. Headset display, interaction and real BODY pose acceptance
remain pending. Version, size and SHA256 are recorded in
[the APK manifest](apk/manifest.json) and [the validation record](docs/body_visualization/validation.md#097-发布-apk).

The binary was published by commit [`a3d1233`](https://github.com/XDL0-0/AIRO-Doffy/commit/a3d1233c53d82f35394f68f0c8d2faa2a4857c81), which identifies this PC/APK repository, **not the Unity build source**. Unity `6000.5.6f1` is verified from the APK. The subsequently published source is pinned to `811d0b4f9e9d6368a7fb2402943b41eb324c7167` and its package lock confirms Meta XR `205.0.0`. Its code18 build entrypoint overrides the saved code16 settings. Source/binary metadata agree, but the original build checkout and a bit-for-bit rebuild have not been verified. The [release record](apk/RELEASE.md) explains the evidence and the release check.

BODY sending is **OFF on every app launch**. In **Session**, set and Apply the
PC address, then switch **Body data: OFF → ON** to send BODY data to UDP 8015.
BODY viewing does not require starting a robot session. See the
[BODY setup guide](docs/body_visualization/README.md) for PC viewer commands.

### 3. Dependencies
Install the required Python packages:
```bash
pip install -c requirements-teleop-constraints.txt -r requirements.txt
```

Or install individually:
```bash
pip install numpy opencv-python pyrealsense2 scipy h5py torch matplotlib loguru pyserial ruckig pyav huggingface-hub

# Additional dependencies for WebRTC streaming mode:
pip install aiortc aiohttp av
```

Additionally, this project depends on custom robotic libraries. Ensure the following are installed in your environment:
- `airo-robots[realman,ur]` (UR RTDE, Robotiq control, and the optional RealMan SDK)
- `airo-camera-toolkit` (RealSense wrappers)
- `airo-spatial-algebra` (SE3 containers)
- `ur_analytic_ik` (Analytic Inverse Kinematics for UR)
- `lerobot` (For Hugging Face dataset creation and replay)
- `sensor_comm_dds` (For tactile sensor communication, optional)

## Configuration
The system uses [`doffy_teleop/config.py`](doffy_teleop/config.py) as its central configuration. Key settings:

### Network & Robot
| Parameter | Description | Default |
|---|---|---|
| `ROBOT_TYPE` | Robot backend (`ur3e` / `ur5e` / `realman`) | `realman` |
| `ROBOT_IP` | Robot controller IP; UR types fall back to `UR_IP` when this is set to `None` | `192.168.1.18` |
| `UR_IP` | UR robot IP address fallback | `10.42.0.162` |
| `REALMAN_PORT` | RealMan API port | `8080` |
| `REALMAN_READ_RETRIES` | Attempts for transient RealMan state-read timeouts | `3` |
| `REALMAN_RETRY_DELAY` | Delay between RealMan state-read retries in seconds | `0.05` |
| `PC_IP` | Host PC interface reachable by the VR headset | `10.10.130.209` |
| `VR_IP` | VR headset IP address | `10.10.131.245` |
| `REALMAN_STATE_PUSH_IP` | Host PC interface reachable by the robot; `None` falls back to `PC_IP` | `192.168.1.100` |
| `TELEOP_COMMAND_MODE` | Teleoperation command path (`joint` / `tcp`) | `joint` |
| `FREEZE_ROTATION` | Keep the TCP orientation fixed while mapping controller translation | `False` |
| `VR_ROTATION_AXIS_SIGNS` | Controller rotation signs in Unity-local `[pitch, yaw, roll]`; RealMan reverses pitch and roll | `[-1, 1, -1]` |
| `GRIPPER` | Connect/control the gripper and include it in recorded state/actions | `False` |

### RealMan CAN-FD

| Parameter | Description | Default |
|---|---|---|
| `REALMAN_CTRL_RATE` | Dedicated CAN-FD setpoint rate; must remain strictly above 100 Hz | `200` |
| `REALMAN_MIN_CANFD_RATE` | Minimum measured rate accepted by the runtime watchdog | `100.0` |
| `REALMAN_RATE_CHECK_WINDOW` | Seconds per measured-rate window | `1.0` |
| `REALMAN_RATE_FAILURE_WINDOWS` | Consecutive failed timing windows allowed before the startup gate aborts | `3` |
| `REALMAN_CANFD_HEARTBEAT_TIMEOUT` | Maximum time without a completed CAN-FD SDK call before the health check fails | `0.05` |
| `REALMAN_MAX_JOINT_SPEED` | Per-joint CAN-FD interpolation limit, capped by controller-reported limits, in rad/s | `0.5` |
| `REALMAN_MAX_JOINT_ACCELERATION` | Host and QP per-joint acceleration limit in rad/s² | `1.0` |
| `REALMAN_MAX_LINEAR_SPEED` | TCP translation interpolation limit in m/s | `0.1` |
| `REALMAN_MAX_LINEAR_ACCELERATION` | TCP translation acceleration limit in m/s² | `0.2` |
| `REALMAN_MAX_ANGULAR_SPEED` | TCP rotation interpolation limit in rad/s | `0.5` |
| `REALMAN_MAX_ANGULAR_ACCELERATION` | TCP rotation acceleration limit in rad/s² | `1.0` |
| `REALMAN_QP_IK_ENABLE` | Use RealMan's continuous teleoperation IK/QP solver in joint mode | `True` |
| `REALMAN_QP_DQ_WEIGHT` | Per-joint QP speed multiplier; lower values trade tracking accuracy for smoother motion | `0.5` |
| `REALMAN_QP_LIMIT_HOLDON` | Hold the QP solution at a joint limit instead of pushing through it | `True` |
| `REALMAN_QP_ELBOW_MARGIN_DEG` | Keep J4 (7-DoF) or J3 (6-DoF) this far from the straight-elbow singularity | `3.0` |
| `WRM_TCP_Z_DROP_M` | Maximum reference-relative TCP Z decrease as WRM elbow progress moves from high to horizontal | `0.05` |
| `REALMAN_REALTIME_STATE_PUSH` | Receive joint, TCP, and force state through the controller's realtime UDP push | `True` |
| `REALMAN_STATE_PUSH_CYCLE_MS` | Realtime state-push cycle; must be a positive multiple of 5 ms | `5` |
| `REALMAN_STATE_PUSH_PORT` | PC UDP port on which realtime state packets are received | `8098` |
| `REALMAN_STATE_PUSH_TIMEOUT` | Startup wait for the first valid realtime state packet, in seconds | `2.0` |
| `REALMAN_FORCE_COORDINATE` | Force frame requested from the controller: `0` sensor, `1` work, `2` tool | `0` |
| `REALMAN_SENSOR_RATE` | Synchronous joint/TCP/force polling rate when realtime state push is disabled | `30.0` |
| `REALMAN_VR_TIMEOUT` | Hold the last target after this many seconds without a VR packet | `0.25` |
| `RESET_JOINT_SPEED` | Speed of the reset/home motion in rad/s (teach-collect initial pose and teleop reset); position backends only | `1.0` |

### Tracking & Streaming
| Parameter | Description | Default |
|---|---|---|
| `TRACKING_MODE` | VR arm input mode: `"controller"` or `"hand"` | `controller` |
| `REALSENSE_RESOLUTION` | Camera resolution `(width, height)` | `(640, 480)` |
| `REALSENSE_FPS` | Camera framerate | `60` |
| `JPEG_QUALITY` | JPEG encoding quality for VR streaming (1-100) | `100` |
| `HD_CHUNK_SIZE` | Max payload bytes per UDP chunk (UDP mode only) | `60000` |
| `SIGNALING_PORT` | WebSocket port for WebRTC signaling (WebRTC mode) | `8765` |

### Dataset
| Parameter | Description | Default |
|---|---|---|
| `DATASET_DIR` | Dataset base path shared by teleop and teach collection | `./datasets/WRM_grasp` |
| `DATASET_TYPE` | `"a"` = ACT/HDF5, `"l"` = LeRobot | `l` |
| `DATA_TYPE` | State/action representation (`qpos`, `both`, `tcp`, `delta_tcp`) | `both` |
| `TEACH_ACTION_MODE` | Teach-replay action label: next measured joints (`next_joint`) or current taught target (`command`) | `next_joint` |
| `SENSOR_SYNC_BUFFER_SIZE` | Recent timestamped camera/Beaver frames retained for nearest-time matching | `8` |
| `BEAVER_SIMULATE_8BIT` | Legacy opt-in: quantize Beaver distances to 10 mm steps; keep `False` for raw 16-bit data | `False` |
| `TEACH_INITIAL_DISCARD_FRAMES` | Teaching samples discarded before trajectory edge trimming | `40` |
| `LEROBOT_IMAGE_WRITER_PROCESSES` | Background processes used for image compression | `1` |
| `LEROBOT_IMAGE_WRITER_THREADS` | Background threads used for image compression | `1` |
| `LEROBOT_VIDEO_CODEC` | Video codec used during episode save | `h264` |
| `LEROBOT_ENCODER_THREADS` | Threads allowed for each video encoder | `1` |
| `TACTILE_TRANSFER` | Enable tactile sensor data | `False` |
| `FORCE_COLLECT` | Record TCP force `[Fx, Fy, Fz]` when available | `False` |
| `TORQUE_COLLECT` | Record TCP torque `[Tx, Ty, Tz]` when available | `False` |

### Force, Tactile & Visualizer
| Parameter | Description | Default |
|---|---|---|
| `GRAVITY_COMP` | Enable tool gravity compensation for TCP wrench readings | `False` |
| `GRAVITY_COMP_FILTER_ALPHA` | Low-pass alpha used inside gravity compensation | `0.15` |
| `FORCE_MOVING_AVERAGE_WINDOW` | Moving-average window applied to 6D wrench readings | `8` |
| `FORCE_LOW_PASS_ALPHA` | Final wrench low-pass alpha after moving average/deadband | `0.15` |
| `FORCE_ENABLE` | Stream combined RealMan TCP pose and force JSON to Quest | `True` |
| `FORCE_PORT` | Quest `TCPPoseReceiver` main UDP port | `8012` |
| `FORCE_SEND_RATE` | RealMan TCP-state packet rate in Hz | `30.0` |
| `TCP_DISPLAY_AXES` | Map RealMan base-frame position, rotation, and force into Unity display axes | RealMan `(X forward, Y left, Z up)` to Unity `(X right, Y up, Z forward)` |
| `TACTILE_ENABLE` | Start tactile hardware when VR transfer or visualizer needs it | `False` |
| `TACTILE_READER` | Tactile reader backend (`ble4` / `serial`) | `ble4` |
| `TACTILE_SHAPE` | Stored tactile sample shape | `(4, 3)` |
| `TACTILE_FILTER_ALPHA` | BLE tactile exponential filter alpha | `0.75` |

[`doffy_teleop/visualization/config.py`](doffy_teleop/visualization/config.py) controls the live dashboard:

| Parameter | Description | Default |
|---|---|---|
| `ENABLED` | Start the shared dashboard from a teleoperation entry point | `True` |
| `HZ` | Visualizer refresh/publish rate | `30.0` |
| `WINDOW_S` | Plot history window in seconds | `8.0` |
| `FORCE_PANEL_RANGE` | Force plot and Fx/Fy panel +/- range in newtons | `30.0` |

## How to Run

Run the following commands from the repository root in the environment for the selected runtime. The root launchers retain their existing arguments; equivalent `python -m doffy_teleop...` commands are listed in the [module layout guide](docs/module-layout.md#running-the-entry-points).

### 1. Data Collection & Teleoperation
Initiate the main teleoperation and dataset recording loop:
```bash
python main.py
```
- Real-time camera streams will appear in the VR headset automatically.
- Controller movements dictate the robot pose and, when `GRIPPER=True`, the gripper aperture.
- Squeeze trigger & buttons to start / stop dataset recording.
- If `VisualizerConfig.ENABLED` is true, a dashboard opens with wrench plots, tactile bubbles, camera previews, robot status, dataset counters, and a rollback button.
- A VR record-control value of `Undo`, `Rollback`, or `DeleteLast` removes the latest saved episode and reuses its index.
- Pressing the reset trigger combination recalibrates force/tactile baselines when those sensors are enabled.

### Human pose visualization with teleop

Use the independent Meta body viewer without connecting to a robot:

```bash
/home/yuyuan/.venvs/airo-teleop/bin/python teleop_body_visualizer.py
# Opt in to RealMan teleop when needed:
/home/yuyuan/.venvs/airo-teleop/bin/python teleop_body_visualizer.py --teleop realman
# Synthetic preview without Quest or robot hardware:
/home/yuyuan/.venvs/airo-teleop/bin/python teleop_body_visualizer.py --demo
# Equivalent packaged entry point, run from this repository:
/home/yuyuan/.venvs/airo-teleop/bin/python -m doffy_teleop.runtime.body_visualizer
```

The viewer defaults to one enlarged skeleton, with view selection, four-view
and zoom buttons, joint angles, tracking validity and confidence on UDP 8015.
BODY sending defaults to OFF in the [v0.9.7 APK](apk/AIRO_Doffy_v0.9.7_arm64_code18.apk).
Enable **Body data: ON** in its Session page to send
BODY telemetry; no robot session is required. The current
default is upper-body tracking; legs are displayed only when valid full-body
joints arrive. The root command remains a thin launcher for the modular
`doffy_teleop` implementation; `doffy_teleop.body_visualization` provides
compatibility exports for the BODY viewer API. See
[setup, usage and module ownership](docs/body_visualization/README.md).

### 2. RealMan CAN-FD Teleoperation

For RealMan high-rate teleoperation and dataset recording without gripper or
tactile hardware:

```bash
python realman_teleop.py
```

On the verified workstation, use
`/home/yuyuan/.venvs/airo-teleop/bin/python realman_teleop.py` from the repository
root. See the [teleop environment notes](docs/teleop_refactor/teleop-environment.md)
for the tested dependency constraints and setup.

Set `ROBOT_TYPE="realman"`, choose `TELEOP_COMMAND_MODE="joint"` or `"tcp"`,
and set `TRACKING_MODE="controller"` or `"hand"`, `GRIPPER=False`, and
`TACTILE_TRANSFER=False`. Configure `DATASET_DIR`, `DATASET_TYPE`, `DATA_TYPE`,
and `COLLECT_RATE` as usual. Use the VR app's built-in buttons to start and save
episodes. The visualizer provides only **Undo episode**, matching `main.py`.
VR rollback messages use the same rollback path. Camera, cached robot
state/action, timestamps, and optional force/torque fields are collected outside
the deadline thread.
Set `FORCE_COLLECT=True` and/or `TORQUE_COLLECT=True` to save the integrated
sensor readings.
Set `FORCE_ENABLE=True` to stream the measured RealMan TCP pose and latest
filtered force to `(VR_IP, FORCE_PORT)` at `FORCE_SEND_RATE`. Each UTF-8 JSON
datagram has one `rightTCP` object with `position` in metres, `rotation` in
`[w,x,y,z]` quaternion order, and `force` as `[Fx,Fy,Fz]` in newtons. Port 8012
is the main `TCPPoseReceiver` port; a separate force-only receiver is not used.
`TCP_DISPLAY_AXES` applies the same RealMan-to-Unity basis change to position,
orientation, and force so the streamed TCP state stays aligned with the Quest
scene after calibration.

With `TCP_TOOL="Hand"` and `BRAINCO_HAND_ENABLE=True`, controller tracking
enables the BrainCo Revo2 presets: push the right joystick forward to run the staged **grab** motion and
pull it backward to **release**. Each direction is edge-triggered; return the
stick to neutral before intentionally repeating the same motion. The threshold
is configured by `BRAINCO_HAND_JOYSTICK_THRESHOLD`.
The horizontal axis remains assigned to the robot wrist: move the right
joystick left or right while holding the grip trigger to rotate the final joint.

For simultaneous wrist and finger tracking, set the following in `doffy_teleop/config.py`
and select hand tracking in the VR app:

```python
ROBOT_TYPE = "realman"
TCP_TOOL = "Hand"
BRAINCO_HAND_ENABLE = True
TRACKING_MODE = "hand"
```

The right-hand wrist pose controls the arm TCP while the OpenXR skeleton
controls all six Revo2 motors in the order `[thumb flex, index, middle, ring,
pinky, thumb rotation]`. Thumb flexion and opposition/rotation have separate
channels. Start with an open hand: the first valid frame calibrates the thumb
rotation's open endpoint, and `BRAINCO_THUMB_ROTATE_PROGRESS_RANGE` controls
its sensitivity. `BRAINCO_HAND_MAX_SEND_HZ` limits hand commands independently
of the arm's CAN-FD rate. Set `BRAINCO_HAND_ENABLE=False` to skip the hand
connection and commands while retaining arm teleoperation and the configured
tool TCP. This integration currently uses RealMan RM_ARM+.

This entry point keeps camera streaming, the robot's integrated six-axis force
sensor, and the shared visualizer. A dedicated thread targets
`REALMAN_CTRL_RATE` and continuously sends `rm_movej_canfd` or
`rm_movep_canfd`. Joint, linear, and angular target changes are interpolated on
that configured command clock using the matching `REALMAN_MAX_*_SPEED` and
`REALMAN_MAX_*_ACCELERATION` limits. The acceleration limiter ramps velocity
and applies braking as a target is approached instead of immediately jumping
to the configured speed.
In joint mode, `REALMAN_QP_IK_ENABLE=True` keeps the latest VR Cartesian target
in RealMan's teleoperation solver and advances it on the same fixed command clock.
The adapter converts this project's radians to the solver's degrees, converts
the joint acceleration limit to the solver's RPM/s units, and keeps the elbow
on its initial side of the configured nonzero singularity margin.

Press the right joystick while holding the right index trigger past
`CONTROLLER_RESET_TRIGGER_THRESHOLD` to return to the startup pose. The reset
is edge-triggered, uses the active CAN-FD stream, and requires releasing the
grip trigger before controller motion resumes.

Before VR motion is accepted, the camera, dataset workers, and visualizer are
started, then the sender must complete a fresh clean timing window under that
final system load at a measured rate strictly above
`REALMAN_MIN_CANFD_RATE=100` Hz. Packet gaps
and SDK calls above 10 ms count as timing violations; after startup verification,
one such violation stops the command path immediately. The command loop also
records a successful-command heartbeat;
`REALMAN_CANFD_HEARTBEAT_TIMEOUT=0.05` seconds is the health-check threshold for
a CAN-FD call that stops completing. Startup aborts instead of enabling motion
when the rate gate or heartbeat check fails. The dashboard reports the measured
rate, packet gaps, total control-step duration (QP solve plus CAN-FD call), and
errors.

Realtime robot state should normally use the controller's UDP push:

```python
REALMAN_REALTIME_STATE_PUSH = True
REALMAN_STATE_PUSH_CYCLE_MS = 5
REALMAN_STATE_PUSH_PORT = 8098
REALMAN_STATE_PUSH_TIMEOUT = 2.0
REALMAN_FORCE_COORDINATE = 0
```

Set `PC_IP` to the PC interface that the headset can reach and
`REALMAN_STATE_PUSH_IP` to the interface reachable by the RealMan controller.
They can differ when the headset uses Wi-Fi and the robot uses Ethernet.
The controller sends UDP state packets to
`REALMAN_STATE_PUSH_IP:REALMAN_STATE_PUSH_PORT` (falling back to `PC_IP` when
the separate address is `None`); allow inbound UDP on that port in the PC
firewall, ensure both hosts have a valid route, and make sure another process is
not already using the port. Startup waits up to `REALMAN_STATE_PUSH_TIMEOUT` for
the first valid packet and fails with a connection diagnostic if none arrives.
The default 5 ms cycle provides joint, TCP, and integrated force state without
placing synchronous state reads in the CAN-FD command path.
`REALMAN_STATE_PUSH_CYCLE_MS` is in milliseconds; the API adapter converts it
to the SDK's 5 ms units (`5` ms becomes `cycle=1`).

`REALMAN_FORCE_COORDINATE` selects the reported wrench frame: `0` is the force
sensor frame, `1` the active work frame, and `2` the active tool frame. The
script consumes the controller's zeroed force values. RealMan
`zero_force_data` is already controller-compensated, so this entry point does
not apply the repository's additional software gravity compensation.

For an older SDK or controller setup that cannot provide realtime state push,
set `REALMAN_REALTIME_STATE_PUSH=False`. This schedules synchronous state and
force reads at `REALMAN_SENSOR_RATE` on the same thread that owns the RealMan
SDK command calls, avoiding concurrent use of the SDK handle. It is an explicit
fallback, not an automatic downgrade: those reads consume CAN-FD timing budget,
so the same startup gate and runtime watchdog remain active and will refuse or
stop motion if the measured command timing is no longer valid.

### RealMan Teach, Replay, and Collect

Collect measured robot state (seven joint angles and TCP pose), the integrated
six-axis force/torque sensor, and RGB observations from every detected RealSense
camera, without VR, tactile sensing, or gripper data. The detected camera count
is used to initialize both the LeRobot dataset and visualizer.
Camera capture is local-only and does not create UDP sockets, WebRTC signaling,
or VR receiver threads. When tactile collection is disabled, the visualizer also
omits the tactile panel. Freedrive is only enabled while teaching, with RealMan's
drag-teach sensitivity set to `99`:

```bash
python realman_teachcollect.py \
    --robot-ip 192.168.1.18 \
    --task "freedrive demonstration" \
    --fps 10
```

Both RealMan teleoperation and teach collection use `Config.DATASET_DIR` by
default. Teach collection also accepts `--dataset-dir` as an explicit override.
The output uses LeRobot format and is written to the selected dataset base path
with the repository's `_lero` suffix. The visualizer workflow is:

1. Optionally press **Initial pose** to move to `Config.INITIAL_JOINT`.
2. Press **Teach**, drag the robot through the desired path, then press
   **End Teach**. Teaching stores joint waypoints in memory but does not write a
   dataset episode. Every sample is retained while teaching. When teaching ends,
   the first `Config.TEACH_INITIAL_DISCARD_FRAMES` samples (40 by default) are
   discarded to remove startup shake. The stationary prefix and suffix of the
   remaining samples are then trimmed; slow motion and pauses inside the
   demonstrated path are preserved.
3. Press **Teach** while a trajectory is ready to clear it and immediately
   begin teaching a replacement. Press **Reteach** to clear the path and wait
   before starting again.
4. Once teaching ends, **Replay collect** is enabled. It moves to the first
   taught waypoint, replays the path, records synchronized robot, force,
   camera, depth, and enabled Beaver data, and exports one dataset episode.
   After export completes and recording is disabled, the robot automatically
   returns to `Config.INITIAL_JOINT`; that return motion is not recorded.

The observation is always the measured joint state during replay.
`Config.TEACH_ACTION_MODE="next_joint"` follows Reactive Diffusion Policy's
label semantics: `action[t]` is the measured joint configuration at `t+1`, and
the final frame repeats the final measured state. Set it to `"command"` to save
the taught joint target issued for the current frame instead. Camera and Beaver
readers retain recent timestamped frames, and teach collection selects the one
nearest to each measured robot-state timestamp. Closing the visualizer or
pressing Ctrl-C stops freedrive and closes the dataset safely.

**Undo episode** removes the most recently exported episode.

Validate a recorded episode without connecting to the robot:

```bash
python -m dataset_tool.replay_realman_lerobot \
    --dataset-dir ./datasets/realman_teach_lero \
    --episodes 0 \
    --dry-run
```

Replay it on RealMan:

```bash
python -m dataset_tool.replay_realman_lerobot \
    --dataset-dir ./datasets/realman_teach_lero \
    --robot-ip 192.168.1.18 \
    --episodes 0
```

The replay tool validates the selected joint or TCP schema, finite targets, and
recorded frame-to-frame motion before opening the robot connection. It uses
controller-planned motion to reach each episode's first pose, asks for a second
confirmation, and then replays the trajectory through low-follow CAN-FD at the
dataset FPS. `--yes` skips confirmations only after the motion has been checked
with `--dry-run`. At each prompt, press Enter to continue or type `c` to quit.

Joint replay defaults to `--source observation.state`, the measured trajectory.
This is the faithful and safe choice: recorded `action` targets can contain
command glitches (for example a single-frame jump while the arm settles into the
episode start pose), while the measured state stays smooth. Episodes may be
spread across several parquet files (chunked by frames); the tool locates rows
by `episode_index` automatically. `--initial-speed` (default 1.0 rad/s) sets the
speed of the controller-planned move to each episode's start pose. The
airo_robots wrapper scales the rad/s value to a percentage of the arm's maximum
joint speed (3.14 rad/s ≈ 100% on an RM75); values above the arm maximum are
rejected by its safety check.

Both recorded self-proprioception representations can be replayed. Joint mode
uses `observation.state` by default; pass `--source action` to command the
recorded action targets instead:

```bash
python -m dataset_tool.replay_realman_lerobot \
    --dataset-dir ./datasets/realman_teach_lero \
    --episodes 0 \
    --control-mode joint
```

TCP mode uses the recorded `[qx, qy, qz, qw, x, y, z]` values in
`extra.tcp_pose`:

```bash
python -m dataset_tool.replay_realman_lerobot \
    --dataset-dir ./datasets/realman_teach_lero \
    --episodes 0 \
    --control-mode tcp
```

TCP replay separately checks translation and rotation speed using
`--max-linear-speed` and `--max-angular-speed`.

### Recollect an existing RealMan dataset with 16-bit Beaver data

Use the recollection workflow when an existing LeRobot v3 trajectory should be
replayed while robot state, TCP pose, wrench, RealSense images, timestamps, and
Beaver distance maps are recorded again. Beaver's legacy 8-bit simulation is
forcibly disabled, so the received `uint16` millimetre values are preserved
without the old 10 mm quantization.

First inspect every selected episode length and replay speed without opening
the robot, cameras, Beaver serial port, or output dataset:

```bash
python -m dataset_tool.recollect_realman \
    --dataset-root ./datasets/realman_teach_lero \
    --dry-run
```

Start recollection and its local UI:

```bash
python -m dataset_tool.recollect_realman \
    --dataset-root ./datasets/realman_teach_lero \
    --robot-ip 192.168.1.18
```

The default output is the sibling dataset
`./datasets/realman_teach_lero_recollect`; use `--output-dataset` to override
it. The UI lists the source length of every episode. For each episode the robot
moves to frame 0 and pauses, while the UI shows the recorded first frame, the
current live camera, and a 50% alignment overlay. Adjust the scene/camera, then
press **Enter** or **Start replay**. Exactly one output frame is recorded per
source trajectory frame. After export, the next episode is loaded and paused at
its first frame automatically.

If the current source path is unsatisfactory, press **Teach replacement** (`T`)
while paused at frame 0. This is the same drag-teach then replay-collect path as
`realman_teachcollect.py`: freedrive records waypoints, **End teach** trims the
motion and moves the arm back to **that taught path's first pose** so the scene
can be set up, and **Collect replacement** then replays the new path into the
current output slot (length may differ from the source episode). After that
export, recollection still advances to the **original dataset's next episode**,
not a continuation of the taught path. **Cancel teach** returns to the current
source episode without consuming the slot. An existing matching `_recollect` dataset
resumes from its first unfinished episode; its source fingerprint and selected
episode list must match.

The same page also displays the nine live Beaver sensors as 4x4 (or 8x8)
distance heatmaps. Every valid cell shows its raw 16-bit millimetre value, and
each sensor reports online/stale state plus the minimum and mean valid distance.
The colours match the teachcollect visualizer: red at 0 mm, a smooth 1 mm-step
light-to-dark blue ramp from 1-400 mm, grey above 400 mm, and slate for invalid
cells.

Source episodes above `--max-joint-speed` preserve their original targets and
timing by default and show an amber warning in the UI before the operator
presses Enter. Use `--joint-jump-policy error` for strict rejection.
`--joint-jump-policy interpolate` remains available for confirmed sensor
glitches: it keeps the episode length and first/last pose unchanged while
smoothing the smallest local window. Do not use interpolation for a fast motion
that is part of the grasp contact sequence.

Useful options include `--episodes`, `--from-episode`, `--to-episode`,
`--source`, `--max-joint-speed`, `--joint-jump-policy`, `--initial-speed`,
`--beaver-port`, and `--beaver-grid-width`. The emergency-stop button requests
a RealMan trajectory slow stop and discards the partial episode.

### 3. Force/Tactile Visualizer
Run the standalone dashboard against a UR robot:
```bash
python test_tool/ForceVisualize.py --ip 10.42.0.162 --robot-type ur3e
```

Preview the shared visualizer UI without robot hardware:
```bash
python test_tool/ForceVisualize.py --mock
```

Show tactile data in the dashboard:
```bash
python test_tool/ForceVisualize.py --mock --mock-tactile
python test_tool/ForceVisualize.py --ip 10.42.0.162 --robot-type ur3e --tactile
```

Run a small TCP xyz experiment with fixed orientation using `servo_to_tcp_pose`:
```bash
python test_tool/ForceVisualize.py \
    --ip 10.42.0.162 \
    --robot-type ur3e \
    --payload-cog 0 0 0.058 \
    --tcp-xyz-experiment
```

### 4. Standalone VR Data Receiver
Test VR connection without robot hardware, running from the repository root:
```bash
python -m test_tool.vr_data
```
The receiver prints controller data and opens the 3D hand visualizer when the
headset switches from controller mode to hand tracking.

## VR Data Protocols

### Controller Data (via `DualControllerSender.cs`)
```
<timestamp_ms>,<left: 14 values>,<right: 14 values>
```
Per-hand fields: `px,py,pz, rx,ry,rz,rw, jx,jy, trigger, grip, AX, BY, joyPress`

### Hand Tracking Data (via `HandTrackingSender.cs`)
Text protocol:
```
H,<L|R>,<timestamp_ms>,<bone0_x>,<bone0_y>,<bone0_z>,...  (24 bones x 3 floats)
```
Binary protocol:
```
HB,<base64-encoded: [0x48, side, count_lo, count_hi, x0,y0,z0, ...]>
```

### HD Video Chunks — UDP Mode (via `UdpSocketMultiHD.cs`)
Each UDP packet: 12-byte big-endian header + JPEG payload
```
[frameId: u32] [chunkIndex: u16] [totalChunks: u16] [totalBytes: u32] [payload...]
```

### WebRTC Video — WebRTC Mode (via `WebRTCVideoReceiver.cs`)
- **Signaling**: WebSocket on `SIGNALING_PORT` (default 8765), JSON envelope: `{"type": "offer"|"answer"|"ice_candidate"|"hello", "session_id": "...", "payload": {...}}`
- **Video**: Single `RTCPeerConnection` with one `VideoStreamTrack` per camera (H.264/VP8 codec).
- **Control**: DataChannel `"control"` replaces UDP port 8005 for resolution/zoom commands.

## Project Structure

The tree below shows the upload scope. See the [module layout and migration guide](docs/module-layout.md) for directory responsibilities, old-to-new module paths, and packaged entry points.

```
doffy-teleop/
├── main.py                  # Classic teleop CLI launcher
├── realman_teleop.py         # RealMan teleop CLI launcher
├── realman_teachcollect.py   # Teach/replay/collect CLI launcher
├── realman_recollect.py      # Dataset recollection CLI launcher
├── teleop_body_visualizer.py # BODY viewer CLI launcher
├── doffy_teleop/
│   ├── config.py            # Central robot, sensor and recording configuration
│   ├── utils.py             # Shared filters, safety checks and helpers
│   ├── protocol/            # VR, BODY, control, JPEG and signaling formats
│   ├── media/               # Capture, UDP/WebRTC and socket services
│   ├── sensors/             # Beaver, MagTouch and wrench filtering
│   ├── control/             # Input mapping, IK, constraints and CAN-FD
│   ├── robots/              # Backends, grippers, BrainCo and Classic control
│   ├── recording/           # Datasets, schema, services, replay and recollection UI
│   ├── runtime/             # Teleop, teach/recollect and BODY lifecycles
│   ├── visualization/       # Teleop dashboard, BODY viewer and display config
│   └── body_visualization.py # BODY compatibility exports
├── dataset_tool/            # Dataset conversion, replay, recollection and annotation
├── test_tool/               # Standalone hardware/protocol tools
├── tests/                   # Teleop, BODY, dataset and protocol tests
├── hardware/beaver/         # Beaver firmware source
├── apk/                     # v0.9.7 ARM64 VR app and release manifest
├── docs/                    # Usage, architecture and validation records
└── scripts/teleop_refactor/  # Teleop validation harnesses and helper scripts
```

The local Seahorse launcher implementation and the research modules listed
above are excluded from the upload.
