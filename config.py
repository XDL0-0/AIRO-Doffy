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
    # "joint" preserves the original VR -> IK -> joint-servo path.
    # "tcp" sends TCP targets through the selected backend when possible.
    TELEOP_COMMAND_MODE: str = "joint"

    PC_IP: str = "10.10.129.200"
    VR_IP: str = "10.10.131.166"

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
    GRIPPER_SPEED: float = 0.1       # m/s, full range (~0.085m) in ~0.57s
    GRIPPER_MAX: float = 0.085       # max opening width [m]

    # ── Joint configuration ───────────────────────────────────────────────
    # INITIAL_JOINT: np.ndarray = field(
    #     default_factory=lambda: np.array([1.57, -1.57, 1.57, -1.57, -1.57, 0])
    # )
    INITIAL_JOINT: np.ndarray = field(
        default_factory=lambda: np.array([1.57, -2.07, 1.25, -1.2, -1.62, 0])
    )
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
    FORCE_FILTER_ALPHA: float = 0.15
    GRAVITY_CALIB_SAMPLES: int = 200

    # ── Inference ─────────────────────────────────────────────────────────
    INFERENCE_FPS: int = 10
    INFERENCE_MAX_STEPS: int = 1000
    INFERENCE_EPISODES: int = 1

    # ── Tactile ───────────────────────────────────────────────────────────
    TACTILE_TRANSFER: bool = False
    TACTILE_PORT: int = 8012             # PC → Quest: tactile sensor data

    def __post_init__(self):
        from airo_spatial_algebra.se3 import SE3Container
        from data_schema import normalize_data_type

        self.ROBOT_TYPE = self.ROBOT_TYPE.lower()
        self.DATA_TYPE = normalize_data_type(self.DATA_TYPE)
        if self.TELEOP_COMMAND_MODE not in {"joint", "tcp"}:
            raise ValueError(f"Unsupported TELEOP_COMMAND_MODE: {self.TELEOP_COMMAND_MODE}")
        if self.ROBOT_IP is None:
            self.ROBOT_IP = self.UR_IP
        if self.TORQUE_MODE and self.ROBOT_TYPE not in {"ur3e", "ur5e"}:
            raise ValueError("TORQUE_MODE is currently only supported for UR robots.")
        if self.DATA_TYPE == "delta_tcp" and self.SAVE_EEF:
            utils.logger.warning("SAVE_EEF is ignored when DATA_TYPE='delta_tcp'.")
        if self.DATA_TYPE == "tcp" and self.SAVE_EEF:
            utils.logger.warning("SAVE_EEF is redundant when DATA_TYPE='tcp'.")
        if self.DATA_TYPE not in {"qpos", "both", "tcp", "delta_tcp"}:
            raise ValueError(f"Unsupported DATA_TYPE: {self.DATA_TYPE}")
        if np.any(self.TCP_POSE):
            self.TCP_TRANSFORM = SE3Container.from_rotation_vector_and_translation(
                self.TCP_POSE[3:6], self.TCP_POSE[0:3]
            ).homogeneous_matrix
            utils.logger.info(f"TCP transform initialized:\n{self.TCP_TRANSFORM}")
