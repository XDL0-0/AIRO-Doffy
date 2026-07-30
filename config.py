from dataclasses import dataclass, field
import numpy as np
import utils


@dataclass
class Config:
    # ── Robot ──────────────────────────────────────────────────────────────
    # Supported: "ur3e", "ur5e", "realman".
    ROBOT_TYPE: str = "ur3e"
    ROBOT_IP: str | None = None
    UR_IP: str = "10.42.0.162"
    REALMAN_PORT: int = 8080
    REALMAN_READ_RETRIES: int = 3
    REALMAN_RETRY_DELAY: float = 0.05
    # Dedicated high-follow CAN-FD stream used by realman_teleop.py.
    # RealMan requires <=10 ms between high-follow setpoints, so keep this
    # strictly above 100 Hz. 200 Hz corresponds to a 5 ms period.
    REALMAN_CTRL_RATE: int = 200
    REALMAN_MIN_CANFD_RATE: float = 100.0
    REALMAN_RATE_CHECK_WINDOW: float = 1.0
    # Failed timing windows allowed while establishing the startup rate gate.
    # After verification, a single >10 ms gap/call stops the command stream.
    REALMAN_RATE_FAILURE_WINDOWS: int = 3
    REALMAN_CANFD_HEARTBEAT_TIMEOUT: float = 0.05
    REALMAN_SENSOR_RATE: float = 30.0
    REALMAN_VR_TIMEOUT: float = 0.25
    REALMAN_MAX_JOINT_SPEED: float = 2.0  # rad/s, applied before CAN-FD
    REALMAN_MAX_LINEAR_SPEED: float = 0.25  # m/s
    REALMAN_MAX_ANGULAR_SPEED: float = 1.0  # rad/s
    REALMAN_CANFD_TRAJECTORY_MODE: int = 0
    REALMAN_CANFD_RADIO: int = 0
    REALMAN_REALTIME_STATE_PUSH: bool = True
    REALMAN_STATE_PUSH_CYCLE_MS: int = 5
    REALMAN_STATE_PUSH_PORT: int = 8098
    REALMAN_STATE_PUSH_TIMEOUT: float = 2.0
    REALMAN_FORCE_COORDINATE: int = 0  # 0 sensor, 1 work, 2 tool
    # "joint" preserves the original VR -> IK -> joint-servo path.
    # "tcp" sends TCP targets through the selected backend when possible.
    TELEOP_COMMAND_MODE: str = "joint"
    FREEZE_ROTATION: bool = True       # Keep TCP orientation fixed during teleoperation.

    PC_IP: str = "10.10.131.162"
    VR_IP: str = "10.10.130.155"

    # ── Task / Dataset ────────────────────────────────────────────────────
    TASK_NAME: str = "pick_and_place"
    DATASET_DIR: str = "./datasets/pnp_long"
    DATASET_TYPE: str = "l"          # 'a' = ACT (hdf5), 'l' = lerobot
    PUSH_TO_HUB: bool = False
    SAVE_EEF: bool = False
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
    REALSENSE_FPS: int = 60

    # ── Network ───────────────────────────────────────────────────────────
    IP_PORT: int = 8000                  # base port for UDP camera/VR sockets
    POSE_PORT: int = 8001                # Quest → PC: controller / hand pose data
    CONTROL_PORT: int = 8005             # Quest → PC: resolution / mode control
    SIGNALING_PORT: int = 8765           # WebRTC video signaling (WebSocket)

    # ── Video streaming ───────────────────────────────────────────────────
    VIDEO_TRANSPORT: str = "webrtc"      # "udp" = chunked JPEG (camera_udp), "webrtc" = WebRTC_udp
    JPEG_QUALITY: int = 100
    HD_CHUNK_SIZE: int = 60000          # max payload bytes per UDP chunk

    # ── Control rates (Hz) ────────────────────────────────────────────────
    UR_CTRL_RATE: int = 60
    KELO_CTRL_RATE: int = 10
    COLLECT_RATE: int = 10
    INFERENCE_FPS: int = 10

    # ── Gripper ───────────────────────────────────────────────────────────
    # When False, do not connect to/control a gripper.
    GRIPPER: bool = False
    GRIPPER_SPEED: float = 0.1       # m/s, full range (~0.085m) in ~0.57s
    GRIPPER_MAX: float = 0.085       # max opening width [m]

    # ── Joint configuration ───────────────────────────────────────────────
    # Defaults are selected in __post_init__ so Config(ROBOT_TYPE=...) gets
    # the matching axes and number of joints instead of the class default's.
    VR_TO_ROBOT_AXES: np.ndarray | None = None
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
    RUCKIG_ENABLE: bool = True
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

    # ── Inference ─────────────────────────────────────────────────────────
    INFERENCE_FPS: int = 10
    INFERENCE_MAX_STEPS: int = 1000
    INFERENCE_EPISODES: int = 1

    # ── Tactile ───────────────────────────────────────────────────────────
    TACTILE_ENABLE: bool = True          # Start tactile hardware when VR transfer or visualizer needs it.
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
        if self.INITIAL_JOINT is None:
            if self.ROBOT_TYPE == "realman":
                self.INITIAL_JOINT = np.array(
                    [
                        2.65586749,
                        -0.06628761,
                        -0.14056882,
                        -1.26216978,
                        0.11116002,
                        -1.11919238,
                        -0.45881216,
                    ]
                )
            else:
                self.INITIAL_JOINT = np.array(
                    [1.57, -1.57, 1.57, -1.57, -1.57, 0.0]
                )
        self.VR_TO_ROBOT_AXES = np.asarray(self.VR_TO_ROBOT_AXES, dtype=float)
        self.INITIAL_JOINT = np.asarray(self.INITIAL_JOINT, dtype=float)
        if self.VR_TO_ROBOT_AXES.shape != (3, 3):
            raise ValueError("VR_TO_ROBOT_AXES must have shape (3, 3).")
        if not np.allclose(
            self.VR_TO_ROBOT_AXES.T @ self.VR_TO_ROBOT_AXES,
            np.identity(3),
        ):
            raise ValueError("VR_TO_ROBOT_AXES must be an orthogonal axis transform.")
        if self.TELEOP_COMMAND_MODE not in {"joint", "tcp"}:
            raise ValueError(f"Unsupported TELEOP_COMMAND_MODE: {self.TELEOP_COMMAND_MODE}")
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
        if self.REALMAN_MAX_LINEAR_SPEED <= 0.0:
            raise ValueError("REALMAN_MAX_LINEAR_SPEED must be positive.")
        if self.REALMAN_MAX_ANGULAR_SPEED <= 0.0:
            raise ValueError("REALMAN_MAX_ANGULAR_SPEED must be positive.")
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
        if self.TACTILE_READER not in {"ble4", "serial"}:
            raise ValueError(f"Unsupported TACTILE_READER: {self.TACTILE_READER}")
        if self.FORCE_MOVING_AVERAGE_WINDOW < 1:
            raise ValueError("FORCE_MOVING_AVERAGE_WINDOW must be at least 1.")
        if not 0.0 <= self.GRAVITY_COMP_FILTER_ALPHA <= 1.0:
            raise ValueError("GRAVITY_COMP_FILTER_ALPHA must be between 0 and 1.")
        if not 0.0 <= self.FORCE_LOW_PASS_ALPHA <= 1.0:
            raise ValueError("FORCE_LOW_PASS_ALPHA must be between 0 and 1.")
        if np.any(self.TCP_POSE):
            self.TCP_TRANSFORM = SE3Container.from_rotation_vector_and_translation(
                self.TCP_POSE[3:6], self.TCP_POSE[0:3]
            ).homogeneous_matrix
            utils.logger.info(f"TCP transform initialized:\n{self.TCP_TRANSFORM}")
