from dataclasses import dataclass, field
import numpy as np
import utils


@dataclass
class Config:
    # ── Robot ──────────────────────────────────────────────────────────────
    ROBOT_TYPE: str = "ur3e"
    UR_IP: str = "10.42.0.162"
    PC_IP: str = "10.10.131.72"
    VR_IP: str = "10.10.131.166"

    # ── Task / Dataset ────────────────────────────────────────────────────
    TASK_NAME: str = "pick_and_place"
    DATASET_DIR: str = "./datasets/sorting_cubes"
    DATASET_TYPE: str = "l"          # 'a' = ACT (hdf5), 'l' = lerobot
    PUSH_TO_HUB: bool = False
    SAVE_EEF: bool = False
    DATA_TYPE: str = "both"          # "qpos" or "both" (qpos + extra.tcp_pose)
    DEPTH_INFO_ENABLE: bool = False

    # ── Tracking mode ─────────────────────────────────────────────────────
    TRACKING_MODE: str = "controller"   # "controller" or "hand"
    HAND_PALM_JUMP_THRESHOLD: float = 0.15   # m, discard frame if palm jumps more
    HAND_GRIPPER_OPEN_DIST: float = 0.06     # m, thumb–index > this → open
    HAND_GRIPPER_CLOSE_DIST: float = 0.03    # m, thumb–index < this → close

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
    UR_CTRL_RATE: int = 100
    KELO_CTRL_RATE: int = 10
    COLLECT_RATE: int = 10

    # ── Gripper ───────────────────────────────────────────────────────────
    GRIPPER_SPEED: float = 0.1       # m/s, full range (~0.085m) in ~0.57s
    GRIPPER_MAX: float = 0.085       # max opening width [m]

    # ── Joint configuration ───────────────────────────────────────────────
    # INITIAL_JOINT: np.ndarray = field(
    #     default_factory=lambda: np.array([-1.57, -1.57, -1.57, 0, 1.57, 3.14])
    # )
    INITIAL_JOINT: np.ndarray = field(
        default_factory=lambda: np.array([1.57, -2.2, 1.57, -1.57, -1.57, 0])
    )
    TCP_POSE: np.ndarray = field(
        default_factory=lambda: np.array([0, 0, 0, 0, 0, 0])
    )
    TCP_TRANSFORM: np.ndarray = field(default_factory=lambda: np.identity(4))
    MOVE_THRESHOLD: np.ndarray = field(
        default_factory=lambda: np.array([0.6, 0.6, 0.6, 0.9, 0.9, 0.9])
    )


    # ── Ruckig OTG ────────────────────────────────────────────────────────
    RUCKIG_ENABLE: bool = False
    RUCKIG_MAX_VEL: float = 2.5      # rad/s per joint
    RUCKIG_MAX_ACC: float = 15.0      # rad/s² per joint
    RUCKIG_MAX_JERK: float = 150.0    # rad/s³ per joint

    # ── Force / Torque ────────────────────────────────────────────────────
    TORQUE_MODE: bool = False
    FORCE_COLLECT: bool = False
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
        if self.DATA_TYPE not in {"qpos", "both", "eef", "tcp_quat"}:
            raise ValueError(f"Unsupported DATA_TYPE: {self.DATA_TYPE}")
        if np.any(self.TCP_POSE):
            self.TCP_TRANSFORM = SE3Container.from_rotation_vector_and_translation(
                self.TCP_POSE[3:6], self.TCP_POSE[0:3]
            ).homogeneous_matrix
            utils.logger.info(f"TCP transform initialized:\n{self.TCP_TRANSFORM}")
