from dataclasses import dataclass, field
import numpy as np


@dataclass
class Config:
    # ── Robot ──────────────────────────────────────────────────────────────
    ROBOT_TYPE: str = "ur3e"
    UR_IP: str = "10.42.0.162"
    PC_IP: str = "10.10.131.72"
    VR_IP: str = "10.10.131.166"

    # ── Task / Dataset ────────────────────────────────────────────────────
    TASK_NAME: str = "wiperboard"
    DATASET_DIR: str = "./datasets/wiperboard_without_tactile_20260225"
    DATASET_TYPE: str = "l"          # 'a' = ACT (hdf5), 'l' = lerobot
    PUSH_TO_HUB: bool = False
    SAVE_EEF: bool = False
    DATA_TYPE: str = "qpos"

    # ── Network ───────────────────────────────────────────────────────────
    IP_PORT: int = 8000

    # ── Control rates (Hz) ────────────────────────────────────────────────
    UR_CTRL_RATE: int = 100
    KELO_CTRL_RATE: int = 10
    COLLECT_RATE: int = 10

    # ── Gripper ───────────────────────────────────────────────────────────
    GRIPPER_SPEED: float = 0.1       # m/s, full range (~0.085m) in ~0.57s
    GRIPPER_MAX: float = 0.085       # max opening width [m]

    # ── Joint configuration ───────────────────────────────────────────────
    INITIAL_JOINT: np.ndarray = field(
        default_factory=lambda: np.array([-1.596, -1.066, -0.973, -2.07, 1.686, 3.109])
    )
    TCP_TRANSFORM: np.ndarray = field(default_factory=lambda: np.identity(4))
    MOVE_THRESHOLD: float = 0.1
    TORQUE_MODE: bool = True

    # ── Force / Torque ────────────────────────────────────────────────────
    FORCE_COLLECT: bool = True
    GRAVITY_COMP: bool = True
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
    TACTILE_PORT: int = 8010
