# Airo-Doffy

Airo-Doffy is a VR teleoperation system for controlling UR and RealMan robots with a Meta Quest headset. It supports controller and hand tracking, live camera streaming, and demonstration recording.

This repository contains the Python runtime and Quest APK. The Unity app is maintained in [AIRO-DOFFY-APP](https://github.com/XDL0-0/AIRO-DOFFY-APP/tree/811d0b4f9e9d6368a7fb2402943b41eb324c7167/AIRO-Doffy).

## Features

- UR and RealMan robot teleoperation using controllers or hand tracking.
- RealSense camera streaming over WebRTC or UDP.
- HDF5 and LeRobot dataset recording, replay, and recollection.
- Optional gripper, BrainCo hand, force, and tactile sensor integration.
- Independent BODY pose visualization without a robot connection.

## Installation

Use Python 3.10+ and a Meta Quest headset. Robot teleoperation requires a supported robot; camera streaming requires Intel RealSense cameras.

```bash
git clone https://github.com/XDL0-0/AIRO-Doffy.git
cd AIRO-Doffy
python -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements-teleop-constraints.txt -r requirements.txt
```

Connect the Quest through ADB, authorize USB debugging, and install the [v0.9.7 / code 18 APK](apk/AIRO_Doffy_v0.9.7_arm64_code18.apk):

```bash
adb install -r apk/AIRO_Doffy_v0.9.7_arm64_code18.apk
```

## Configuration

Edit [`doffy_teleop/config.py`](doffy_teleop/config.py) for your setup:

- Set `ROBOT_TYPE`, `ROBOT_IP`, `PC_IP`, and `VR_IP` to match your hardware and network. For RealMan state feedback, also set `REALMAN_STATE_PUSH_IP` to a PC address reachable by the robot.
- Select `TRACKING_MODE` (`controller` or `hand`) and `VIDEO_TRANSPORT` (`webrtc` or `udp`).
- Configure the mounted tool, optional sensors, `DATASET_DIR`, and `DATASET_TYPE`.

Set `DOFFY_PC_IP`, `DOFFY_VR_IP`, `DOFFY_UR_IP`, and `DOFFY_REALMAN_STATE_PUSH_IP` in your shell to supply local network addresses. The PC/Quest/UR defaults are documentation examples; replace them before use. See [local setup](docs/release-hygiene.md#local-network-setup). The Quest and PC must be able to reach each other over the network.

## Usage

Run one teleoperation runtime from the repository root:

```bash
# RealMan teleoperation
python realman_teleop.py

# Classic teleoperation, including UR robots
python main.py
```

In the Quest app:

1. Open **Teleop Config → Connection & input**, enter the PC address with **Edit IP**, and press **Apply**. Choose **Controllers** or **Hand tracking** to match the PC configuration.
2. Use the **Alignment** tab to set the reference and robot base. In **Camera**, select **Use WebRTC** or **Use UDP** to match the PC.
3. Press **Start Teleop**. Use **Start recording** and **Stop recording** in the record panel to save episodes. **Undo episode → Confirm undo** removes the latest episode.

To view BODY tracking without connecting to a robot:

```bash
python teleop_body_visualizer.py
```

Set the PC address as above, then enable **System Setting → Body data: ON**. BODY telemetry is off on each app launch and does not require **Start Teleop**.

## Unity Project

Use the published source snapshot at [`811d0b4`](https://github.com/XDL0-0/AIRO-DOFFY-APP/tree/811d0b4f9e9d6368a7fb2402943b41eb324c7167/AIRO-Doffy):

```bash
git clone https://github.com/XDL0-0/AIRO-DOFFY-APP.git
cd AIRO-DOFFY-APP
git checkout 811d0b4f9e9d6368a7fb2402943b41eb324c7167
```

1. Open the **AIRO-Doffy** subfolder in Unity Hub; the repository root contains a historical project.
2. Use Unity **6000.5.6f1** with Android Build Support. Package Manager restores Meta XR All-in-One **205.0.0** and the locked dependencies.
3. Open `Assets/Scenes/Teleoperation.unity` and run **Tools → DOFFY → Validate scene**.
4. For the v0.9.7 / code 18 package, use **Tools → DOFFY → Build Meta ARM64-only update APK**.

This is a post-release source snapshot; the original APK build revision remains unverified. See the [release record](apk/RELEASE.md) and [APK manifest](apk/manifest.json) for source mapping, versions, and SHA256.

## Documentation

- [Detailed usage: teleoperation, teaching, replay, and protocols](docs/usage.md)
- [BODY visualization](docs/body_visualization/README.md)
- [Python module layout](docs/module-layout.md)
- [Validation records](docs/teleop_refactor/README.md#验证与复现)
