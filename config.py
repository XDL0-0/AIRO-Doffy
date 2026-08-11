from dataclasses import dataclass, field
import numpy as np
import utils


@dataclass
class Config:
    # PC_IP: str = "10.10.131.162"
    # VR_IP: str = "10.10.130.155"

    PC_IP: str = "192.168.43.198"
    VR_IP: str = "192.168.43.89"
    # ── Robot ──────────────────────────────────────────────────────────────
    # Supported: "ur3e", "ur5e", "realman".
    ROBOT_TYPE: str = "realman"
    ROBOT_IP: str | None = "192.168.1.18"
    UR_IP: str = "10.42.0.162"
    REALMAN_PORT: int = 8080
    REALMAN_READ_RETRIES: int = 3
    REALMAN_RETRY_DELAY: float = 0.05
    # Dedicated high-follow CAN-FD stream used by realman_teleop.py.
    # RealMan requires <=10 ms between high-follow setpoints, so keep this
    # strictly above 100 Hz. 200 Hz corresponds to a 5 ms period.
    REALMAN_CTRL_RATE: int = 180
    REALMAN_MIN_CANFD_RATE: float = 100
    REALMAN_RATE_CHECK_WINDOW: float = 1.0
    # Failed timing windows allowed while establishing the startup rate gate.
    # After verification, a single >10 ms gap/call stops the command stream.
    REALMAN_RATE_FAILURE_WINDOWS: int = 3
    REALMAN_CANFD_HEARTBEAT_TIMEOUT: float = 0.05
    REALMAN_SENSOR_RATE: float = 30.0
    REALMAN_VR_TIMEOUT: float = 0.25
    REALMAN_MAX_JOINT_SPEED = 0.5
    REALMAN_MAX_JOINT_ACCELERATION = 1.0

    REALMAN_MAX_LINEAR_SPEED = 0.25
    REALMAN_MAX_LINEAR_ACCELERATION = 0.8

    REALMAN_MAX_ANGULAR_SPEED = 0.5
    REALMAN_MAX_ANGULAR_ACCELERATION = 1.5
    REALMAN_CANFD_TRAJECTORY_MODE: int = 2
    REALMAN_CANFD_RADIO: int = 50
    # Dedicated continuous Cartesian solver recommended by RealMan for
    # teleoperation. The solver runs at REALMAN_CTRL_RATE and its output is
    # still passed through the host-side joint speed/safety gates.
    REALMAN_QP_IK_ENABLE: bool = True
    REALMAN_QP_DQ_WEIGHT: float = 0.5
    REALMAN_QP_LIMIT_HOLDON: bool = True
    # Keep the 7-DoF J4 (or 6-DoF J3) elbow on its initial side of zero.
    # RealMan warns that a fully straight 0-degree elbow can oscillate.
    REALMAN_QP_ELBOW_MARGIN_DEG: float = 3.0
    REALMAN_REALTIME_STATE_PUSH: bool = True
    REALMAN_STATE_PUSH_CYCLE_MS: int = 5
    REALMAN_STATE_PUSH_PORT: int = 8098
    REALMAN_STATE_PUSH_TIMEOUT: float = 2.0
    REALMAN_FORCE_COORDINATE: int = 0  # 0 sensor, 1 work, 2 tool
    # "joint" preserves the original VR -> IK -> joint-servo path.
    # "tcp" sends TCP targets through the selected backend when possible.
    TELEOP_COMMAND_MODE: str = "tcp"
    FREEZE_ROTATION: bool = False       # Keep TCP orientation fixed during teleoperation.



    # ── TCP tool ──────────────────────────────────────────────────────────
    # Supported: "Gripper", "None", "Hand". The BrainCo hand is available on
    # RealMan in both controller mode (joystick presets) and hand-tracking mode.
    TCP_TOOL: str = "Hand"
    # Backward-compatible mirror. __post_init__ derives this from TCP_TOOL.
    GRIPPER: bool = False
    GRIPPER_SPEED: float = 0.1       # m/s, full range (~0.085m) in ~0.57s
    GRIPPER_MAX: float = 0.085       # max opening width [m]
    BRAINCO_HAND_BAUDRATE: int = 460800
    BRAINCO_HAND_READ_RETRIES: int = 10
    BRAINCO_HAND_RETRY_DELAY: float = 0.25
    BRAINCO_HAND_MODE_SETTLE_DELAY: float = 2.0
    BRAINCO_HAND_MAX_SEND_HZ: float = 50.0
    # Right joystick Y: forward grabs, backward releases. The action is
    # edge-triggered and rearms after the stick returns inside this threshold.
    BRAINCO_HAND_JOYSTICK_THRESHOLD: float = 0.7
    # Ignore normalized finger changes smaller than this before smoothing.
    BRAINCO_HAND_DEAD_ZONE: float = 0.015
    # Set to 0 to disable smoothing while retaining the dead zone.
    BRAINCO_HAND_FILTER_CUTOFF_HZ: float = 8.0
    # OpenXR palm-width progress corresponding to the full thumb-rotation range.
    # The open endpoint is calibrated from the first valid VR hand frame.
    BRAINCO_THUMB_ROTATE_PROGRESS_RANGE: float = 1.2

    # ── Task / Dataset ────────────────────────────────────────────────────
    TASK_NAME: str = "pick_and_place"
    DATASET_DIR: str = "./datasets/pnp_long"
    DATASET_TYPE: str = "l"          # 'a' = ACT (hdf5), 'l' = lerobot
    PUSH_TO_HUB: bool = False
    SAVE_EEF: bool = False
    # Keep image compression and video encoding from starving high-rate robot
    # command threads during LeRobot recording.
    LEROBOT_IMAGE_WRITER_PROCESSES: int = 1
    LEROBOT_IMAGE_WRITER_THREADS: int = 1
    LEROBOT_VIDEO_CODEC: str = "h264"
    LEROBOT_ENCODER_THREADS: int = 1
    # Supported: "qpos"/"joint_configuration", "both", "tcp"/"tcp_quat"/"eef", "delta_tcp".
    # "both" keeps joint state/action and stores extra.tcp_pose for conversion or policies.
    DATA_TYPE: str = "both"
    DEPTH_INFO_ENABLE: bool = False
    # ── Tracking mode ─────────────────────────────────────────────────────
    TRACKING_MODE: str = "controller"   # "controller" or "hand"
    CONTROLLER_RESET_TRIGGER_THRESHOLD: float = 0.8  # trigger + joystick press → reset
    HAND_PALM_JUMP_THRESHOLD: float = 0.15   # m, discard frame if palm jumps more
    HAND_GRIPPER_OPEN_DIST: float = 0.06     # m, thumb–index > this → open
    HAND_GRIPPER_CLOSE_DIST: float = 0.03    # m, thumb–index < this → close

    HAND_MODE_TOGGLE_DIST: float = 0.02      # m, thumb–pinky touch → toggle mode
    HAND_RESET_DIST: float = 0.02            # m, thumb–ring touch → reset robot

    # ── Realsense camera ──────────────────────────────────────────────────
    REALSENSE_RESOLUTION: tuple = (640, 480)  # (width, height)
    # RESOLUTION_1080 = (1920, 1080)
    # RESOLUTION_720 = (1280, 720)
    # RESOLUTION_540 = (960, 540)

    # RESOLUTION_480 = (640, 480)
    REALSENSE_FPS: int = 30

    # ── Network ───────────────────────────────────────────────────────────
    IP_PORT: int = 8000                  # base port for UDP camera/VR sockets
    POSE_PORT: int = 8001                # Quest → PC: controller / hand pose data
    CONTROL_PORT: int = 8005             # Quest → PC: resolution / mode control
    SIGNALING_PORT: int = 8765           # WebRTC video signaling (WebSocket)

    # ── Video streaming ───────────────────────────────────────────────────
    VIDEO_TRANSPORT: str = "webrtc"      # "udp" = chunked JPEG (camera_udp), "webrtc" = WebRTC_udp
    JPEG_QUALITY: int = 50
    HD_CHUNK_SIZE: int = 60000          # max payload bytes per UDP chunk

    # ── Control rates (Hz) ────────────────────────────────────────────────
    UR_CTRL_RATE: int = 100
    KELO_CTRL_RATE: int = 10
    COLLECT_RATE: int = 10
    INFERENCE_FPS: int = 10



    # ── Joint configuration ───────────────────────────────────────────────
    # Defaults are selected in __post_init__ so Config(ROBOT_TYPE=...) gets
    # the matching axes and number of joints instead of the class default's.
    VR_TO_ROBOT_AXES: np.ndarray | None = None
    # Signs for controller-relative rotations in Unity local axes:
    # [pitch/X, yaw/Y, roll/Z]. RealMan pitch and roll are reversed to match
    # the physical tool motion.
    VR_ROTATION_AXIS_SIGNS: np.ndarray | None = None
    INITIAL_JOINT: np.ndarray | None = None
    # INITIAL_JOINT: np.ndarray = field(
    #     default_factory=lambda: np.array([1.57, -2.07, 1.25, -1.2, -1.62, 0])
    # )
    TCP_POSE: np.ndarray = field(
        default_factory=lambda: np.array([0, 0, 0, 0, 0, 0])
    )
    TCP_TRANSFORM: np.ndarray = field(default_factory=lambda: np.identity(4))
    MOVE_THRESHOLD: np.ndarray = field(
        default_factory=lambda: np.array([0.9, 0.9, 0.9, 0.9, 1.4, 1.4])
    )


    # ── Ruckig OTG ────────────────────────────────────────────────────────
    RUCKIG_ENABLE: bool = False
    RUCKIG_MAX_VEL: np.ndarray = field(
        default_factory=lambda: np.array([2.5, 2.5, 2.5, 3.0, 4.0, 4.0])
    )  # rad/s per joint; larger for wrist joints near the gripper
    RUCKIG_MAX_ACC: np.ndarray = field(
        default_factory=lambda: np.array([15.0, 15.0, 15.0, 18.0, 25.0, 25.0])
    )  # rad/s² per joint
    RUCKIG_MAX_JERK: np.ndarray = field(
        default_factory=lambda: np.array([150.0, 150.0, 150.0, 180.0, 250.0, 250.0])
    )  # rad/s³ per joint
    CARTESIAN_POS_FILTER_CUTOFF_HZ: float = 8.0
    CARTESIAN_ROT_FILTER_CUTOFF_HZ: float = 6.0
    HAND_JOINT_FILTER_CUTOFF_HZ: float = 10.0

    # ── Force / Torque ────────────────────────────────────────────────────
    TORQUE_MODE: bool = False
    # Record TCP force [Fx, Fy, Fz] when the selected backend exposes a wrench.
    FORCE_COLLECT: bool = False
    # Record TCP torque [Tx, Ty, Tz] when the selected backend exposes a wrench.
    TORQUE_COLLECT: bool = False
    GRAVITY_COMP: bool = False
    TOOL_MASS: float = 0.925         # kg — Robotiq 2F85 gripper
    TOOL_COM: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.058])
    )
    # Low-pass alpha used inside gravity compensation before the final wrench filter.
    GRAVITY_COMP_FILTER_ALPHA: float = 0.15
    FORCE_MOVING_AVERAGE_WINDOW: int = 8
    FORCE_LOW_PASS_ALPHA: float = 0.15
    GRAVITY_CALIB_SAMPLES: int = 200
    # Stream RealMan TCP position, [w,x,y,z] rotation, and [Fx,Fy,Fz] as JSON
    # to the Quest TCPPoseReceiver. This is the main combined state port.
    FORCE_ENABLE: bool = True
    FORCE_PORT: int = 8012
    FORCE_SEND_RATE: float = 30.0
    # RealMan 基座 → Unity 显示: (X前,Y左,Z上) → (X右,Y前,Z上)
    # 独立于 VR_TO_ROBOT_AXES, 使 RGB=XYZ 校准时 TCP 显示即对齐.
    TCP_DISPLAY_AXES: np.ndarray = field(
        default_factory=lambda: np.array([[0, -1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    )

    # ── Inference ─────────────────────────────────────────────────────────
    INFERENCE_FPS: int = 10
    INFERENCE_MAX_STEPS: int = 1000
    INFERENCE_EPISODES: int = 1

    # ── Tactile ───────────────────────────────────────────────────────────
    TACTILE_ENABLE: bool = False          # Start tactile hardware when VR transfer or visualizer needs it.
    TACTILE_TRANSFER: bool = False
    TACTILE_PORT: int = 8012             # PC → Quest: tactile sensor data
    TACTILE_READER: str = "ble4"         # "ble4" or "serial"
    TACTILE_SHAPE: tuple = (4, 3)        # one 4-taxel MagTouch, xyz per taxel
    TACTILE_SERIAL_COM: str = "/dev/ttyACM0"
    TACTILE_BLE_DEVICE_MAC: str = "ARDUINO7"
    TACTILE_BLE_HCI: str = "hci0"
    TACTILE_BLE_WINDOW_SIZE: int = 100
    TACTILE_FILTER_ALPHA: float = 0.75
    TACTILE_USE_KALMAN: bool = False
    TACTILE_KALMAN_Q: float = 2e-2
    TACTILE_KALMAN_R: float = 2e-2
    TACTILE_MAX_DELTA: float = 10000.0
    TACTILE_BASELINE_DRIFT_ALPHA: float = 0.0
    TACTILE_BASELINE_DRIFT_THRESHOLD: float = 80.0

    def __post_init__(self):
        from airo_spatial_algebra.se3 import SE3Container
        from data_schema import normalize_data_type

        self.ROBOT_TYPE = self.ROBOT_TYPE.lower()
        self.TRACKING_MODE = self.TRACKING_MODE.lower()
        tool_aliases = {
            "gripper": "Gripper",
            "none": "None",
            "hand": "Hand",
        }
        requested_tool = str(self.TCP_TOOL).strip().lower()
        if requested_tool not in tool_aliases:
            supported = ", ".join(tool_aliases.values())
            raise ValueError(
                f"Unsupported TCP_TOOL '{self.TCP_TOOL}'. Supported: {supported}"
            )
        self.TCP_TOOL = tool_aliases[requested_tool]
        if self.TCP_TOOL == "Hand" and self.ROBOT_TYPE != "realman":
            utils.logger.warning(
                "TCP_TOOL='Hand' requires ROBOT_TYPE='realman'; "
                "falling back to TCP_TOOL='None'."
            )
            self.TCP_TOOL = "None"
        self.GRIPPER = self.TCP_TOOL == "Gripper"
        self.DATA_TYPE = normalize_data_type(self.DATA_TYPE)
        if self.VR_TO_ROBOT_AXES is None:
            if self.ROBOT_TYPE == "realman":
                # Unity/Quest (+X right, +Y up, +Z forward) -> RealMan
                # (+X forward, +Y left, +Z up).
                self.VR_TO_ROBOT_AXES = np.array(
                    [
                        [0.0, 0.0, 1.0],
                        [-1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                    ]
                )
            else:
                # Unity/Quest axes -> UR base axes.
                self.VR_TO_ROBOT_AXES = np.array(
                    [
                        [-1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0],
                        [0.0, 1.0, 0.0],
                    ]
                )
        if self.VR_ROTATION_AXIS_SIGNS is None:
            self.VR_ROTATION_AXIS_SIGNS = (
                np.array([-1.0, 1.0, -1.0])
                if self.ROBOT_TYPE == "realman"
                else np.ones(3)
            )
        if self.INITIAL_JOINT is None:
            if self.ROBOT_TYPE == "realman":
                self.INITIAL_JOINT = np.array(
                    [
                        2.20,
                        -0.26,
                        0.10,
                        -1.17,
                        -0.11,
                        -0.89,
                        -0.50,
                    ]
                )
                        # 2.6,
                        # -0.725,
                        # 2.04,
                        # 1.55,
                        # 0.73,
                        # 1.29,
                        # -1.55
            else:
                self.INITIAL_JOINT = np.array(
                    [1.57, -1.57, 1.57, -1.57, -1.57, 0.0]
                )
        self.VR_TO_ROBOT_AXES = np.asarray(self.VR_TO_ROBOT_AXES, dtype=float)
        self.VR_ROTATION_AXIS_SIGNS = np.asarray(
            self.VR_ROTATION_AXIS_SIGNS,
            dtype=float,
        )
        self.INITIAL_JOINT = np.asarray(self.INITIAL_JOINT, dtype=float)
        if self.VR_TO_ROBOT_AXES.shape != (3, 3):
            raise ValueError("VR_TO_ROBOT_AXES must have shape (3, 3).")
        if not np.allclose(
            self.VR_TO_ROBOT_AXES.T @ self.VR_TO_ROBOT_AXES,
            np.identity(3),
        ):
            raise ValueError("VR_TO_ROBOT_AXES must be an orthogonal axis transform.")
        if self.VR_ROTATION_AXIS_SIGNS.shape != (3,) or not np.all(
            np.isin(self.VR_ROTATION_AXIS_SIGNS, (-1.0, 1.0))
        ):
            raise ValueError(
                "VR_ROTATION_AXIS_SIGNS must contain three values, each +1 or -1."
            )
        if self.TELEOP_COMMAND_MODE not in {"joint", "tcp"}:
            raise ValueError(f"Unsupported TELEOP_COMMAND_MODE: {self.TELEOP_COMMAND_MODE}")
        if self.TRACKING_MODE not in {"controller", "hand"}:
            raise ValueError(f"Unsupported TRACKING_MODE: {self.TRACKING_MODE}")
        if self.ROBOT_IP is None and self.ROBOT_TYPE in {"ur3e", "ur5e"}:
            self.ROBOT_IP = self.UR_IP
        if self.ROBOT_IP is None:
            raise ValueError(
                f"ROBOT_IP must be configured when ROBOT_TYPE='{self.ROBOT_TYPE}'."
            )
        if self.TORQUE_MODE and self.ROBOT_TYPE not in {"ur3e", "ur5e"}:
            raise ValueError("TORQUE_MODE is currently only supported for UR robots.")
        if self.REALMAN_READ_RETRIES < 1:
            raise ValueError("REALMAN_READ_RETRIES must be at least 1.")
        if self.REALMAN_RETRY_DELAY < 0.0:
            raise ValueError("REALMAN_RETRY_DELAY cannot be negative.")
        if self.BRAINCO_HAND_BAUDRATE not in {9600, 115200, 256000, 460800}:
            raise ValueError("Unsupported BRAINCO_HAND_BAUDRATE.")
        if self.BRAINCO_HAND_READ_RETRIES < 1:
            raise ValueError("BRAINCO_HAND_READ_RETRIES must be at least 1.")
        if self.BRAINCO_HAND_RETRY_DELAY < 0.0:
            raise ValueError("BRAINCO_HAND_RETRY_DELAY cannot be negative.")
        if self.BRAINCO_HAND_MODE_SETTLE_DELAY < 0.0:
            raise ValueError("BRAINCO_HAND_MODE_SETTLE_DELAY cannot be negative.")
        if self.BRAINCO_HAND_MAX_SEND_HZ <= 0.0:
            raise ValueError("BRAINCO_HAND_MAX_SEND_HZ must be positive.")
        if not 0.0 < self.BRAINCO_HAND_JOYSTICK_THRESHOLD <= 1.0:
            raise ValueError(
                "BRAINCO_HAND_JOYSTICK_THRESHOLD must be in (0, 1]."
            )
        if not 0.0 <= self.BRAINCO_HAND_DEAD_ZONE < 1.0:
            raise ValueError("BRAINCO_HAND_DEAD_ZONE must be in [0, 1).")
        if self.BRAINCO_HAND_FILTER_CUTOFF_HZ < 0.0:
            raise ValueError(
                "BRAINCO_HAND_FILTER_CUTOFF_HZ cannot be negative."
            )
        if self.BRAINCO_THUMB_ROTATE_PROGRESS_RANGE <= 0.0:
            raise ValueError(
                "BRAINCO_THUMB_ROTATE_PROGRESS_RANGE must be positive."
            )
        if self.REALMAN_MIN_CANFD_RATE < 100.0:
            raise ValueError(
                "REALMAN_MIN_CANFD_RATE must be at least 100 Hz."
            )
        if self.REALMAN_CTRL_RATE <= self.REALMAN_MIN_CANFD_RATE:
            raise ValueError(
                "REALMAN_CTRL_RATE must be strictly greater than "
                "REALMAN_MIN_CANFD_RATE for high-follow CAN-FD control."
            )
        if self.REALMAN_RATE_CHECK_WINDOW <= 0.0:
            raise ValueError("REALMAN_RATE_CHECK_WINDOW must be positive.")
        if self.REALMAN_RATE_FAILURE_WINDOWS < 1:
            raise ValueError("REALMAN_RATE_FAILURE_WINDOWS must be at least 1.")
        if self.REALMAN_CANFD_HEARTBEAT_TIMEOUT <= 0.01:
            raise ValueError(
                "REALMAN_CANFD_HEARTBEAT_TIMEOUT must be greater than 10 ms."
            )
        if self.REALMAN_SENSOR_RATE <= 0.0:
            raise ValueError("REALMAN_SENSOR_RATE must be positive.")
        if self.REALMAN_VR_TIMEOUT <= 0.0:
            raise ValueError("REALMAN_VR_TIMEOUT must be positive.")
        if self.REALMAN_MAX_JOINT_SPEED <= 0.0:
            raise ValueError("REALMAN_MAX_JOINT_SPEED must be positive.")
        if self.REALMAN_MAX_JOINT_ACCELERATION <= 0.0:
            raise ValueError(
                "REALMAN_MAX_JOINT_ACCELERATION must be positive."
            )
        if self.REALMAN_MAX_LINEAR_SPEED <= 0.0:
            raise ValueError("REALMAN_MAX_LINEAR_SPEED must be positive.")
        if self.REALMAN_MAX_LINEAR_ACCELERATION <= 0.0:
            raise ValueError(
                "REALMAN_MAX_LINEAR_ACCELERATION must be positive."
            )
        if self.REALMAN_MAX_ANGULAR_SPEED <= 0.0:
            raise ValueError("REALMAN_MAX_ANGULAR_SPEED must be positive.")
        if self.REALMAN_MAX_ANGULAR_ACCELERATION <= 0.0:
            raise ValueError(
                "REALMAN_MAX_ANGULAR_ACCELERATION must be positive."
            )
        if not 0.0 < self.REALMAN_QP_DQ_WEIGHT <= 1.0:
            raise ValueError("REALMAN_QP_DQ_WEIGHT must be in (0, 1].")
        if not 0.0 < self.REALMAN_QP_ELBOW_MARGIN_DEG < 90.0:
            raise ValueError(
                "REALMAN_QP_ELBOW_MARGIN_DEG must be between 0 and 90 degrees."
            )
        if self.REALMAN_CANFD_TRAJECTORY_MODE not in {0, 1, 2}:
            raise ValueError("REALMAN_CANFD_TRAJECTORY_MODE must be 0, 1, or 2.")
        if self.REALMAN_CANFD_RADIO < 0:
            raise ValueError("REALMAN_CANFD_RADIO cannot be negative.")
        if (
            self.REALMAN_CANFD_TRAJECTORY_MODE == 1
            and self.REALMAN_CANFD_RADIO > 100
        ):
            raise ValueError("Curve-fit REALMAN_CANFD_RADIO must be <= 100.")
        if (
            self.REALMAN_CANFD_TRAJECTORY_MODE == 2
            and self.REALMAN_CANFD_RADIO > 999
        ):
            raise ValueError("Filter REALMAN_CANFD_RADIO must be <= 999.")
        if (
            self.REALMAN_STATE_PUSH_CYCLE_MS < 5
            or self.REALMAN_STATE_PUSH_CYCLE_MS % 5
        ):
            raise ValueError(
                "REALMAN_STATE_PUSH_CYCLE_MS must be a positive multiple of 5 ms."
            )
        if not 1 <= self.REALMAN_STATE_PUSH_PORT <= 65535:
            raise ValueError("REALMAN_STATE_PUSH_PORT must be between 1 and 65535.")
        if self.REALMAN_STATE_PUSH_TIMEOUT <= 0.0:
            raise ValueError("REALMAN_STATE_PUSH_TIMEOUT must be positive.")
        if self.REALMAN_FORCE_COORDINATE not in {0, 1, 2}:
            raise ValueError("REALMAN_FORCE_COORDINATE must be 0, 1, or 2.")
        if self.DATA_TYPE == "delta_tcp" and self.SAVE_EEF:
            utils.logger.warning("SAVE_EEF is ignored when DATA_TYPE='delta_tcp'.")
        if self.DATA_TYPE == "tcp" and self.SAVE_EEF:
            utils.logger.warning("SAVE_EEF is redundant when DATA_TYPE='tcp'.")
        if self.DATA_TYPE not in {"qpos", "both", "tcp", "delta_tcp"}:
            raise ValueError(f"Unsupported DATA_TYPE: {self.DATA_TYPE}")
        if self.LEROBOT_IMAGE_WRITER_PROCESSES < 0:
            raise ValueError("LEROBOT_IMAGE_WRITER_PROCESSES cannot be negative.")
        if self.LEROBOT_IMAGE_WRITER_THREADS < 0:
            raise ValueError("LEROBOT_IMAGE_WRITER_THREADS cannot be negative.")
        if (
            self.LEROBOT_IMAGE_WRITER_PROCESSES == 0
            and self.LEROBOT_IMAGE_WRITER_THREADS == 0
        ):
            raise ValueError(
                "At least one LeRobot image-writer process or thread is required."
            )
        if self.LEROBOT_ENCODER_THREADS < 1:
            raise ValueError("LEROBOT_ENCODER_THREADS must be at least 1.")
        if self.TACTILE_READER not in {"ble4", "serial"}:
            raise ValueError(f"Unsupported TACTILE_READER: {self.TACTILE_READER}")
        if self.FORCE_MOVING_AVERAGE_WINDOW < 1:
            raise ValueError("FORCE_MOVING_AVERAGE_WINDOW must be at least 1.")
        if not 0.0 <= self.GRAVITY_COMP_FILTER_ALPHA <= 1.0:
            raise ValueError("GRAVITY_COMP_FILTER_ALPHA must be between 0 and 1.")
        if not 0.0 <= self.FORCE_LOW_PASS_ALPHA <= 1.0:
            raise ValueError("FORCE_LOW_PASS_ALPHA must be between 0 and 1.")
        if not 1 <= self.FORCE_PORT <= 65535:
            raise ValueError("FORCE_PORT must be between 1 and 65535.")
        if self.FORCE_SEND_RATE <= 0.0:
            raise ValueError("FORCE_SEND_RATE must be positive.")
        if np.any(self.TCP_POSE):
            self.TCP_TRANSFORM = SE3Container.from_rotation_vector_and_translation(
                self.TCP_POSE[3:6], self.TCP_POSE[0:3]
            ).homogeneous_matrix
            utils.logger.info(f"TCP transform initialized:\n{self.TCP_TRANSFORM}")
