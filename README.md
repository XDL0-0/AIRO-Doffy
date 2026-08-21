# VR Teleoperation for Robot Manipulators

> **Version**: This repository is compatible with the VR app **v0.6.0**.  
> **VR App**: The APK and Unity project files are maintained at **[AIRO-DOFFY-APP](https://github.com/XDL0-0/AIRO-DOFFY-APP)**.

A high-performance codebase for controlling robot manipulators (UR3e, UR5e, RealMan, or compatible backends) using VR controllers or hand tracking via UDP. It features camera streaming to the VR headset via **HD chunked UDP** or **WebRTC** (aiortc), low-latency robot control, tactile sensing integration, dataset recording (HDF5 & LeRobot formats), and policy inference evaluation.

## Key Features
- **Low-Latency Teleoperation**: Real-time VR controller tracking to robot end-effector mapping with backend-specific IK/control and safety limits.
- **Hand Tracking Support**: Receive and visualize 24-bone hand skeleton data from Meta Quest hand tracking (text and binary protocols).
- **HD Video Streaming**: Two transport options:
  - **UDP** (`udp.py`): Chunked JPEG transfer, compatible with `UdpSocketMultiHD.cs`.
  - **WebRTC** (`WebRTC_udp.py`): aiortc-based multi-track video (H.264/VP8) with WebSocket signaling + DataChannel for control. Lower bandwidth, NAT-friendly.
- **Fast Gripper Control**: Custom non-blocking TCP socket implementation for the Robotiq 2F-85 gripper.
- **Tactile & Force Integration**: Supports serial MagTouch and 4-taxel BLE MagTouch readers, UR force/torque readings, gravity compensation, baseline reset, and configurable wrench filtering.
- **Live Teleop Visualizer**: Optional multiprocessing matplotlib dashboard for force/torque, camera previews, TCP/joint status, tactile bubbles, dataset status, and last-episode rollback.
- **Dataset Recording**: Save robotic trajectories directly in ACT (HDF5) or Hugging Face `lerobot` formats.
- **Dataset Rollback**: Delete the latest recorded ACT/HDF5 or LeRobot episode from the VR record-control channel or the visualizer.
- **Policy Inference & Evaluation**: Load trained AI policies (e.g., ACT, Diffusion, Pi0) and evaluate them offline or run live robotics inference.
- **Hand Visualizer**: Real-time matplotlib 3D hand skeleton visualization with finger bone connections and dynamic axis scaling.

## Installation

### 1. Prerequisites
- **Python 3.10+** (Recommended: Conda environment `airo-mono`)
- **Robot**: Compatible robot backend, such as UR3e/UR5e with RTDE enabled or RealMan over its network API.
- **Cameras**: Intel RealSense Cameras.
- **VR Setup**: VR headset running the compatible Unity app (v0.6.0) with `DualControllerSender` / `HandTrackingSender` + `UdpSocketMultiHD` receiver. APK and project files: [AIRO-DOFFY-APP](https://github.com/XDL0-0/AIRO-DOFFY-APP).

### 2. Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
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
- `lerobot` (For Hugging Face dataset creation and policy inference)
- `sensor_comm_dds` (For tactile sensor communication, optional)

## Configuration
The system uses `config.py` as its central configuration. Key settings:

### Network & Robot
| Parameter | Description | Default |
|---|---|---|
| `ROBOT_TYPE` | Robot backend (`ur3e` / `ur5e` / `realman`) | `realman` |
| `ROBOT_IP` | Robot controller IP; UR types fall back to `UR_IP` when this is set to `None` | `192.168.1.18` |
| `UR_IP` | UR robot IP address fallback | `10.42.0.162` |
| `REALMAN_PORT` | RealMan API port | `8080` |
| `REALMAN_READ_RETRIES` | Attempts for transient RealMan state-read timeouts | `3` |
| `REALMAN_RETRY_DELAY` | Delay between RealMan state-read retries in seconds | `0.05` |
| `PC_IP` | Host PC address; also the destination for RealMan realtime UDP state push | `192.168.1.59` |
| `VR_IP` | VR headset IP address | `192.168.1.234` |
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

`visualizer_config.py` controls the live dashboard:

| Parameter | Description | Default |
|---|---|---|
| `ENABLED` | Start the shared dashboard from a teleoperation entry point | `True` |
| `HZ` | Visualizer refresh/publish rate | `30.0` |
| `WINDOW_S` | Plot history window in seconds | `8.0` |
| `FORCE_PANEL_RANGE` | Force plot and Fx/Fy panel +/- range in newtons | `30.0` |

## How to Run

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

### 2. RealMan CAN-FD Teleoperation

For RealMan high-rate teleoperation and dataset recording without gripper or
tactile hardware:

```bash
python realman_teleop.py
```

Set `ROBOT_TYPE="realman"`, choose `TELEOP_COMMAND_MODE="joint"` or `"tcp"`,
and keep `TRACKING_MODE="controller"`, `GRIPPER=False`, and
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

With `TCP_TOOL="Hand"`, controller tracking also enables the BrainCo Revo2
presets: push the right joystick forward to run the staged **grab** motion and
pull it backward to **release**. Each direction is edge-triggered; return the
stick to neutral before intentionally repeating the same motion. The threshold
is configured by `BRAINCO_HAND_JOYSTICK_THRESHOLD`.
The horizontal axis remains assigned to the robot wrist: move the right
joystick left or right while holding the grip trigger to rotate the final joint.

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

Set `PC_IP` to the address of the PC network interface that the RealMan
controller can reach. The controller sends UDP state packets to
`PC_IP:REALMAN_STATE_PUSH_PORT`; allow inbound UDP on that port in the PC
firewall, ensure both hosts have a valid route, and make sure another process is
not already using the port. Startup waits up to `REALMAN_STATE_PUSH_TIMEOUT` for
the first valid packet and fails with a connection diagnostic if none arrives.
The default 5 ms cycle provides joint, TCP, and integrated force state without
placing synchronous state reads in the CAN-FD command path.

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
Test VR connection without robot hardware:
```bash
# Controller mode — print controller data
python vr_data.py

# Hand tracking mode — 3D hand visualizer
python vr_data.py --visualize
```

### 4. Live Policy Inference
Execute a previously trained AI policy directly onto the robot:
```bash
python inference.py --policy username/my_act_policy
python inference.py --policy ./checkpoints/my_policy --device cuda --fps 10
```

### 5. Offline Policy Evaluation
Evaluate a trained policy against recorded dataset:
```bash
python eval_policy.py --policy username/my_policy
python eval_policy.py \
    --policy ./checkpoints/my_policy \
    --dataset ./datasets/my_dataset_lero \
    --episodes 0 1 2 \
    --no-show
```

`eval_policy.py` follows RDP's deployment-time latency matching: after each
replan it discards the first `EvalConfig.INFERENCE_LATENCY_STEPS` predictions
(default `4` at 24 Hz). Override it with `--latency-steps N`, or pass `0` to
disable matching. This does not shift dataset labels a second time.

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
```
airo-doffy/
├── config.py           # Central configuration
├── main.py             # Data collection entry point
├── realsense_camera.py # Shared local RealSense capture
├── udp.py              # UDP video/control transport using captured camera frames
├── WebRTC_udp.py       # Camera streaming (WebRTC) + VR data reception
├── parse_vr.py         # VR data parsing (controller + hand tracking)
├── robot_backend.py    # Robot backend adapters for UR, RealMan, and generic manipulators
├── robot_teleop.py     # Robot-agnostic teleoperation backend client
├── realman_teleop.py   # Lean RealMan camera/force teleop with 200 Hz CAN-FD
├── force_filter.py     # Shared 6D wrench filtering utilities
├── tactile_4point.py   # 4-taxel BLE MagTouch reader and tactile panel helpers
├── visualizer.py       # Shared live force/tactile/camera/dataset dashboard
├── visualizer_config.py # Visualizer settings
├── vr_data.py          # Standalone VR receiver + hand visualizer
├── data_schema.py      # Dataset and policy state/action schema helpers
├── dataset.py          # Dataset recording (HDF5 / LeRobot)
├── inference.py        # Policy inference
├── tactile.py          # Tactile sensor interface
├── udp_comms.py        # Two-way UDP communication
├── utils.py            # Filters, safety checks, helpers
└── example from VR/    # Unity C# source & Python examples
```
