# Airo-Doffy usage guide

For installation and Quest setup, see the [README](../README.md). Configure the runtime in [`doffy_teleop/config.py`](../doffy_teleop/config.py).

Replace `ROBOT_IP` in examples with your robot address. Run the following commands from the repository root in the environment for the selected runtime. The root launchers retain their existing arguments; equivalent `python -m doffy_teleop...` commands are listed in the [module layout guide](module-layout.md#running-the-entry-points).

## Data Collection & Teleoperation

Initiate the main teleoperation and dataset recording loop:
```bash
python main.py
```

- Open **Camera** in the Quest app and select the transport configured by `VIDEO_TRANSPORT`.
- Controller movements dictate the robot pose and, when `GRIPPER=True`, the gripper aperture.
- Press **Start Teleop**, then use **Start recording** and **Stop recording** in the record panel. **Undo episode** requires **Confirm undo** to remove the last episode.
- If `VisualizerConfig.ENABLED` is true, a dashboard opens with wrench plots, tactile bubbles, camera previews, robot status, dataset counters, and a rollback button.
- A VR record-control value of `Undo`, `Rollback`, or `DeleteLast` removes the latest saved episode and reuses its index.
- Pressing the reset trigger combination recalibrates force/tactile baselines when those sensors are enabled.

## Human pose visualization with teleop

Use the independent Meta body viewer without connecting to a robot:

```bash
python teleop_body_visualizer.py
# Opt in to RealMan teleop when needed:
python teleop_body_visualizer.py --teleop realman
# Synthetic preview without Quest or robot hardware:
python teleop_body_visualizer.py --demo
# Equivalent packaged entry point, run from this repository:
python -m doffy_teleop.runtime.body_visualizer
```

The viewer defaults to one enlarged skeleton, with view selection, four-view
and zoom buttons, joint angles, tracking validity and confidence on UDP 8015.
BODY sending defaults to OFF in the [v0.9.7 APK](../apk/AIRO_Doffy_v0.9.7_arm64_code18.apk).
Enable **Body data: ON** under **System Setting** to send
BODY telemetry; no robot session is required. The current
default is upper-body tracking; legs are displayed only when valid full-body
joints arrive. The root command remains a thin launcher for the modular
`doffy_teleop` implementation; `doffy_teleop.body_visualization` provides
compatibility exports for the BODY viewer API. See
[setup, usage and module ownership](body_visualization/README.md).

## RealMan CAN-FD Teleoperation

For RealMan high-rate teleoperation and dataset recording without gripper or
tactile hardware:

```bash
python realman_teleop.py
```

Run from the repository root in your teleop Python environment. See the
[teleop environment notes](teleop_refactor/teleop-environment.md) for dependency
constraints and setup.

Set `ROBOT_TYPE="realman"`, choose `TELEOP_COMMAND_MODE="joint"` or `"tcp"`,
and set `TRACKING_MODE="controller"` or `"hand"`. For operation without a
gripper or hand, set `TCP_TOOL="None"`; disable tactile streaming with
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

## RealMan Teach, Replay, and Collect

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
    --robot-ip ROBOT_IP \
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
   the first `Config.TEACH_INITIAL_DISCARD_FRAMES` samples (25 by default) are
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
    --robot-ip ROBOT_IP \
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

## Recollect an existing RealMan dataset with 16-bit Beaver data

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
    --robot-ip ROBOT_IP
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

## Force/Tactile Visualizer

Run the standalone dashboard against a UR robot:
```bash
python test_tool/ForceVisualize.py --ip ROBOT_IP --robot-type ur3e
```

Preview the shared visualizer UI without robot hardware:
```bash
python test_tool/ForceVisualize.py --mock
```

Show tactile data in the dashboard:
```bash
python test_tool/ForceVisualize.py --mock --mock-tactile
python test_tool/ForceVisualize.py --ip ROBOT_IP --robot-type ur3e --tactile
```

Run a small TCP xyz experiment with fixed orientation using `servo_to_tcp_pose`:
```bash
python test_tool/ForceVisualize.py \
    --ip ROBOT_IP \
    --robot-type ur3e \
    --payload-cog 0 0 0.058 \
    --tcp-xyz-experiment
```

## Standalone VR Data Receiver

Test VR connection without robot hardware, running from the repository root:
```bash
python -m test_tool.vr_data
```
The receiver prints controller data and opens the 3D hand visualizer when the
headset switches from controller mode to hand tracking.

## VR Data Protocols

These formats follow the [published Unity source](https://github.com/XDL0-0/AIRO-DOFFY-APP/tree/811d0b4f9e9d6368a7fb2402943b41eb324c7167/AIRO-Doffy/Assets/Teleop).

### Controller and hand tracking — UDP 8001

`DualControllerSender.cs` sends 31 comma-separated fields:

```text
C,<frameId>,<timestamp_ns>,<left: 14 values>,<right: 14 values>
```

Each side contains `px,py,pz,qx,qy,qz,qw,jx,jy,trigger,grip,AX,BY,joyPress`.
The PC also accepts the legacy 29-field controller format.

`HandTrackingSender.cs` uses 26 OpenXR joints. Its text format is:

```text
H,<L|R>,<frameId>,<timestamp_ns>,<wrist: px,py,pz,qx,qy,qz,qw>,<26 joint xyz triples>
```

The binary format is `HB,<base64>`. The decoded payload contains an 8-byte
header (`0x48`, ASCII side, uint16 joint count, uint32 frame ID), followed by
26 float32 XYZ triples. Numeric fields use little-endian byte order. The binary
packet has no separate timestamp or wrist quaternion.

### Recording, WRM and BODY

- UDP **8003**: `Start`, `Stop` and `Undo` recording commands.
- UDP **8005**: WRM upper-limb data when WRM is enabled; otherwise used for legacy video controls.
- UDP **8015**: optional BODY v1 JSON; see the [BODY guide](body_visualization/README.md).

### Video

UDP camera streams use ports `8000 + 2 × camera index`. Each JPEG packet has a
12-byte big-endian header followed by its image chunk:

```text
[frameId: u32] [chunkIndex: u16] [totalChunks: u16] [totalBytes: u32] [payload...]
```

WebRTC uses WebSocket signaling on `SIGNALING_PORT` (default **8765**) and a
video track per camera. Its `control` DataChannel carries resolution and zoom
commands; pose, recording, WRM and BODY retain their UDP channels.
