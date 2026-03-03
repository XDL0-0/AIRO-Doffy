# VR Teleoperation for UR Robots 🚀

A high-performance codebase for controlling UR robots (UR3e, UR5e) using VR controllers via UDP. It features real-time camera streaming to the VR headset, low-latency robot control, tactile sensing integration, dataset recording (HDF5 & LeRobot formats), and policy inference evaluation.

## 🌟 Key Features
- **Low-Latency Teleoperation**: Real-time VR controller tracking to robot end-effector mapping with analytical IK and safety limits.
- **Fast Gripper Control**: Custom non-blocking TCP socket implementation for the Robotiq 2F-85 gripper.
- **VR Streaming**: High-framerate UDP multi-camera streaming directly to your VR Headset.
- **Tactile & Force Integration**: Supports MagTouch tactile sensors and UR force/torque readings with calibration and gravity compensation.
- **Dataset Recording**: Save robotic trajectories directly in ACT (HDF5) or Hugging Face `lerobot` formats.
- **Policy Inference & Evaluation**: Load trained AI policies (e.g., ACT, Diffusion, Pi0) and evaluate them offline or run live robotics inference.

## 🛠️ Installation

### 1. Prerequisites
- **Python 3.10+** (Recommended: Conda or pyenv)
- **Robot**: Compatible UR Robot (e.g., UR3e, UR5e) with RTDE enabled.
- **Cameras**: Intel RealSense Cameras.
- **VR Setup**: VR headset connected and running the compatible Unity UDP receiver.

### 2. Dependencies
First, install the required standard Python packages:
```bash
pip install numpy opencv-python pyrealsense2 scipy h5py torch matplotlib loguru pyserial
```

Additionally, this project depends on custom robotic libraries. Ensure the following are securely installed in your environment:
- `airo-robots` (UR RTDE & Robotiq control)
- `airo-camera-toolkit` (RealSense wrappers)
- `airo-spatial-algebra` (SE3 containers)
- `ur_analytic_ik` (Analytic Inverse Kinematics for UR)
- `lerobot` (For Hugging Face dataset creation and policy inference)
- `sensor_comm_dds` (For tactile sensor communication)

## ⚙️ Configuration
The system uses `config.py` as its central brain. Adjust these primary settings before running:
- **Network Interfaces**: `UR_IP`, `PC_IP`, `VR_IP`.
- **Robot Configuration**: `ROBOT_TYPE` (e.g., "ur3e"), `INITIAL_JOINT`.
- **Dataset Options**: `DATASET_DIR` and `DATASET_TYPE` ('a' for ACT/HDF5, 'l' for LeRobot).
- **Sensor Toggles**: Toggle `TACTILE_TRANSFER` and `FORCE_COLLECT` to enable/disable data collection from auxiliary sensors.

## 🚀 How to Run

### 1. Data Collection & Teleoperation
Initiate the main teleoperation and dataset recording loop:
```bash
python main.py
```
- Real-time camera streams will pop up in the VR headset automatically.
- Controller movements dictate the Robot Pose / Gripper aperture.
- Squeeze trigger & buttons to map to dataset recording start / stop segments.

### 2. Live Policy Inference
Execute a previously trained Hugging Face or local AI policy directly onto the robot:
```bash
# Run with a HuggingFace Hub policy identifier
python inference.py --policy username/my_act_policy

# Run with a locally saved checkpoint
python inference.py --policy ./checkpoints/my_policy --device cuda --fps 10
```

### 3. Offline Policy Evaluation
Evaluate a trained open-loop policy against your recorded dataset, generating metric summaries and trajectory plots:
```bash
# Evaluate using the dataset assigned within the policy's train_config.json
python eval_policy.py --policy username/my_policy

# Evaluate exclusively on specific local episodes and skip plotting display
python eval_policy.py \
    --policy ./checkpoints/my_policy \
    --dataset ./datasets/my_dataset_lero \
    --episodes 0 1 2 \
    --no-show
```

## 🐛 Codebase Diagnostics
- **Bug Status**: Code architecture looks solid, concurrent multi-threading is cleanly implemented. 
- *Note on Concurrency*: Collection (`collect_loop`) and Export (`export_loop`) mechanisms both interact with the central data dictionary. Mutual-exclusion flags act correctly in standard operations. If rapidly toggling recording modes, ensure visual confirmation that no misaligned lengths result during frame packaging into HDF5/LeRobot formats. No critical structural bugs were identified.
