# Meta body pose visualization

Local paths and network addresses in this documentation are anonymized examples;
see [local setup and validation paths](../release-hygiene.md) before running checks.

`teleop_body_visualizer.py` opens only the body skeleton window by default. It **does not connect to RealMan, start robots or cameras, or wait for teleop controller data**. The window displays the body joints actually output by Meta to help inspect shoulder, elbow, wrist, finger, and torso poses, as well as leg poses in full-body mode. The default view is a single enlarged 3D skeleton, with front, side, top, and four-view options.

## Launch

The existing teleop environment is recommended. Run these commands from the repository root.

```bash
# Inspect Quest body data only; no RealMan, cameras, or robot session is required
./.venv/bin/python teleop_body_visualizer.py

# Add this option explicitly only when running RealMan teleop at the same time
./.venv/bin/python teleop_body_visualizer.py --teleop realman

# Synthetic pose demo; no Quest or robot is required
./.venv/bin/python teleop_body_visualizer.py --demo

# Use the original main.py teleop entry point
./.venv/bin/python teleop_body_visualizer.py --teleop classic
```

The root-level command remains unchanged. The package entry point can also be used from the repository directory with the same arguments:

```bash
cd /path/to/AIRO-Doffy
./.venv/bin/python -m doffy_teleop.runtime.body_visualizer
```

Optional arguments: `--bind-ip 0.0.0.0`, `--body-port 8015`, `--hz 30`, and `--stale-after 0.5`. When changing the port, also update Quest `BodyPoseTelemetrySender.destinationPort`. `--save-preview /absolute/path/preview.png` outputs a synthetic pose PNG clearly marked DEMO, without starting a robot or listening for UDP.

`--visualization-only` remains available and has the same behavior as the default. Only when `--teleop realman` or `--teleop classic` is specified does the visualizer start a teleop subprocess after visualization prechecks, using the same Python interpreter and repository directory. In this mode, closing the body window or pressing Ctrl-C sends SIGINT to teleop, giving the existing runtime time to clean up cameras, robot connections, and recording; if teleop exits early, the body window also closes and returns the corresponding exit code.

## Quest side

The Unity project is in [AIRO-DOFFY-APP / AIRO-Doffy](https://github.com/XDL0-0/AIRO-DOFFY-APP/tree/811d0b4f9e9d6368a7fb2402943b41eb324c7167/AIRO-Doffy). Open that subdirectory; the repository root contains the historical project. See the [release notes](../../apk/RELEASE.md) for version and build information.

- `Assets/Teleop/UpperLimb/UpperLimbAkmManager.cs`: provides a read-only copy of the currently sampled body state, reusing the existing tracking owner.
- `Assets/Teleop/UpperLimb/BodyPoseTelemetrySender.cs`: created automatically after the scene loads and disabled by default; when sending is enabled in System Setting, it sends diagnostic data at 25 Hz to UDP 8015 at the configured PC address.
- `Assets/Teleop/Protocol/BodyPoseWireFormat.cs`: BODY v1 JSON serialization and fixed SDK joint names/order.

Copies of the source files and their `.meta` files are stored in the [quest](quest/) directory. Existing projects should use the actual project above; the copies are for reference and porting, and the complete manager must not be used to overwrite custom changes in another version.

On the Quest, set this PC's IP in **Teleop Config → Connection & input** and Apply it, then switch **Body data: OFF** to **Body data: ON** in **System Setting**. Sending defaults to off on every application launch; turning it off immediately stops BODY data and closes the diagnostic socket. This switch is independent of Start/Stop Teleop, WRM, and controller and hand control data; **viewing body poses alone does not require pressing Start Teleop, enabling WRM, or completing robot calibration**. By default, the PC BODY receiver listens on all local network interfaces. Body tracking permission is required, and the headset must be worn with the application in focus.

Use the repository-provided [v0.9.7/code18 Android ARM64 APK](../../apk/AIRO_Doffy_v0.9.7_arm64_code18.apk), with package name `com.AIROLab.AIRODOFFY`, which includes BODY sending code. Install it from the repository root with `adb install -r apk/AIRO_Doffy_v0.9.7_arm64_code18.apk`. The previous automatic BODY sending behavior has been replaced by a manual switch; earlier versions have no BODY data. This APK has passed signature verification and Quest 3 installation/launch checks. The headset interface, interaction, and real body poses have not yet passed acceptance checks; see [validation.md](validation.md#097-release-apk) for the record.

The current default is Meta `UpperBody`, containing **70 body/hand joints** and no legs. The window is labeled `UPPER BODY · legs unavailable`; it does not infer legs from elbow progress, controller positions, or a template. To include legs, enable `requestFullBodyTracking` on Unity's `UpperLimbAkmManager` before tracking starts, configure full-body tracking as required by the platform, and rebuild the APK. When supported, it sends **84 joints** in `FullBody` mode; unsupported startup requests fall back to UpperBody. Changing the field at runtime does not restart tracking.

Body positions use Unity world coordinates, consistent with the original WRM shoulder/elbow positions: X right, Y up, Z forward, in meters. Rotations are XYZW quaternions. The hand stream used for existing controls may have undergone a teleop reference-frame transformation and cannot be directly overlaid with this world-coordinate skeleton. Fingers in the body window come from the same BODY frame and therefore share the torso's coordinate system.

## Viewing data

- The left side of the body is cyan, and the right side is orange. A single large skeleton is shown by default; the 3D plot can be dragged to rotate it.
- The `3D` / `Front` / `Side` / `Top` buttons select a single enlarged view; `Four views` / `Single skeleton` switches between four views and a single skeleton.
- `Zoom +` / `Zoom -` zooms in and out; `Fit pose` restores automatic fitting and following. The upper body is fitted to its actual extent instead of occupying a fixed full-body field of view.
- `Finger joints` controls finger visibility; `Joint names` labels major joints; `Joint axes (XYZ)` shows the red X, green Y, and blue Z axes corresponding to valid quaternions; `Follow hips` makes the viewport follow the body.
- The top displays the frame ID, receive rate, frame age since PC reception, valid joint count, data source, discarded invalid packet count, and SDK confidence.
- `LIVE` indicates a usable fresh pose. When tracking becomes invalid, confidence is at or below 0.5, no valid positions are available, or the stream stops, the last valid pose is retained, `HOLDING` or `STALE` is displayed, and the skeleton fades; updates resume automatically when valid tracking returns. The view stays blank until a valid pose has been received.
- The newer APP distinguishes SDK `valid` from `tracked`: valid coordinates can be inferred values after tracking has been lost. When previously actively tracked shoulder/arm/wrist joints become untracked, the entire arm on that side freezes; individual invalid or missing joints retain their historical positions. SDK-inferred joints with no active tracking history can still be displayed, such as legs in a full-body model. Changing sender or upper-body/full-body mode clears history to avoid inheriting poses from another source.
- The top diagnostics still show the currently received frame, confidence, and valid joint count; while frozen, they additionally show the number of held joints and the age of the valid pose. Angles are calculated from the displayed pose, and a frozen pose is not treated as `LIVE`. Coordinates at the origin are not mistaken for tracking failure.
- Arm elevation is measured relative to the world vertical downward direction: 0° when hanging down and 90° when horizontal. Elbows/knees show the interior angle, with 180° when straight; these are geometric angles calculated from joint positions. The bottom also shows the WRM alpha and calibration status associated with that BODY frame.

The separate diagnostic port does not compete for data with controller/hand UDP 8001 or WRM/control UDP 8005. Only when teleop is explicitly enabled do body rendering and robot teleop run in separate processes. The receive rate is determined by actual data; `--hz` controls only the window refresh rate.

## PC module responsibilities

The project's public name is **Airo-Doffy**, while the Python module name uses an underscore: `doffy_teleop`. The implementation resides in that package. `teleop_body_visualizer.py` only calls the package entry point; `doffy_teleop/body_visualization.py` retains compatibility exports for the BODY API from before modularization.

| Module | Responsibility |
| --- | --- |
| `protocol/body.py` | BODY v1 data structures, joint names/IDs, and strict parsing |
| `media/body.py` | Separate BODY UDP reception, latest frame, and receive statistics |
| `visualization/body_geometry.py` | Skeleton connections, coordinate projection, and joint angles |
| `visualization/body.py` | `BodyDashboard`, Matplotlib rendering, and window controls |
| `visualization/body_demo.py` | Synthetic BODY demo frames |
| `visualization/body_pose.py` | Displayed pose cache, freezing on tracking loss, and recovery |
| `runtime/body_viewer.py` | Window refresh loop, prechecks, and receiver/window resource cleanup |
| `runtime/body_visualizer.py` | CLI arguments and optional teleop subprocess lifecycle |

Paths in the table are relative to `doffy_teleop/`. Runtime and plotting modules depend directly on their respective implementation modules; compatibility exports do not participate in internal dependencies.

## Data protocol

Each frame is one UTF-8 JSON UDP datagram, with a maximum size of 65507 bytes. `joints` uses fixed SDK IDs: 0–69 for UpperBody and 0–83 for FullBody; `name` omits the `Body_`/`FullBody_` prefix. Examples include Head=7, LeftHandWrist=19, RightHandWrist=45, and LeftLowerLeg=71.

```json
{
  "type": "BODY", "version": 1, "frame_id": 42, "timestamp_ns": 1791200000000000000,
  "joint_set": "upper_body", "coordinate_space": "unity_world",
  "confidence": 0.95, "tracking_valid": true,
  "joints": [
    {"id": 7, "name": "Head", "position": [0.0, 1.7, 0.1],
     "rotation": [0.0, 0.0, 0.0, 1.0], "position_valid": true, "orientation_valid": true,
     "position_tracked": true, "orientation_tracked": true}
  ],
  "wrm": {"elbow_alpha": 0.4, "confidence": 0.95, "enabled": false, "calibrated": false}
}
```

Invalid positions or rotations are sent as `null` with the corresponding `false` flag. The PC rejects non-finite numbers, duplicate/out-of-range joint IDs, incorrect names, incorrect coordinate spaces, and incompatible protocol versions. SDK confidence represents the SDK output state and alone cannot prove that every joint matches the person's actual pose; the visualization is intended to inspect that output.

`position_tracked` / `orientation_tracked` are new optional boolean fields in BODY v1. APP version 0.9.6 and newer retain the actual SDK flags, and BODY display quality uses Meta `OVRBody`'s high-confidence threshold, `confidence > 0.5`; low-quality frames still send invalidity notifications and diagnostic confidence. This rule applies only to BODY visualization and does not change WRM or robot control. Older APP versions without tracked fields can still be parsed, but can freeze only according to the original validity and confidence checks; they cannot identify high-confidence untracked fallback poses.

## Validation

```bash
cd /path/to/AIRO-Doffy
./.venv/bin/python -m pytest -q \
  tests/test_body_visualization.py tests/test_teleop_body_visualizer.py \
  tests/test_body_module_imports.py tests/test_teleop_refactor_boundaries.py \
  tests/test_body_pose_hold.py tests/test_body_tracking_protocol.py
./.venv/bin/python teleop_body_visualizer.py \
  --save-preview output/body_visualization/preview.png
```

See [validation.md](validation.md) for software validation records and APK build results. Synthetic previews, C# compilation, and local UDP loopback do not constitute validation of actual Quest poses.
