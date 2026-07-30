import logging
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist

from airo_doffy.devices.wrench.compensation import (
    GravityCompensator as _PureGravityCompensator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("Teleop")

UR3E_JOINT_LIMITS = (-2 * np.pi, 2 * np.pi)


def quat_cal(matrix: np.ndarray, last_quat: np.ndarray | None = None) -> np.ndarray:
    """Convert rotation matrix to quaternion [x,y,z,w], ensuring continuity
    with *last_quat* by flipping sign when the dot-product is negative."""
    rot = R.from_matrix(matrix)
    quat = rot.as_quat()
    if last_quat is not None and np.dot(quat, last_quat) < 0:
        quat = -quat
    return quat


def is_joint_within_limits(
    joints: np.ndarray,
    limits: tuple[float, float] = UR3E_JOINT_LIMITS,
) -> bool:
    joints = np.asarray(joints)
    return bool(np.all(joints >= limits[0]) and np.all(joints <= limits[1]))


def is_joint_change_safe(
    previous_joints: np.ndarray | None,
    new_joints: np.ndarray,
    joint_threshold: float | np.ndarray,
) -> bool:
    if previous_joints is None:
        return True

    # Compute shortest angular distance to avoid wrap-around false positives
    # near the ±π boundary (e.g. -π+0.01 → π-0.01 is only 0.02 rad physically).
    raw_diff = np.array(new_joints) - np.array(previous_joints)
    joint_diff = np.abs(np.arctan2(np.sin(raw_diff), np.cos(raw_diff)))

    if np.any(joint_diff > joint_threshold):
        logger.warning(f"Joint change too large: {joint_diff}, keeping pose")
        return False
    return True

UR_DH_PARAMS = {
    "ur3e": [
        [0, 0.15185, np.pi/2],
        [-0.24355, 0, 0],
        [-0.2132, 0, 0],
        [0, 0.13105, np.pi/2],
        [0, 0.08535, -np.pi/2],
        [0, 0.0921, 0]
    ],
    "ur5e": [
        [0, 0.1625, np.pi/2],
        [-0.425, 0, 0],
        [-0.3922, 0, 0],
        [0, 0.1333, np.pi/2],
        [0, 0.0997, -np.pi/2],
        [0, 0.0996, 0]
    ]
}

def get_ur_fk_frames(joints: np.ndarray, robot_type: str = "ur3e") -> list[np.ndarray]:
    """Calculate forward kinematics frame origins using DH parameters."""
    dh = UR_DH_PARAMS.get(robot_type, UR_DH_PARAMS["ur3e"])
    frames = [np.eye(4)]
    T = np.eye(4)
    for i in range(6):
        a, d, alpha = dh[i]
        theta = joints[i]
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        
        T_i = np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [0,   sa,     ca,    d],
            [0,   0,      0,     1]
        ])
        T = T @ T_i
        frames.append(T)
    return [f[:3, 3] for f in frames]

def get_interpolated_points(a: np.ndarray, b: np.ndarray, num_points: int) -> list[np.ndarray]:
    if num_points <= 1:
        return [b]
    return [a + (b - a) * t for t in np.linspace(0, 1, num_points)]

def is_self_collision(
    joints: np.ndarray,
    robot_type: str = "ur3e",
    return_details: bool = False,
):
    """Check if the arm folds in onto itself using interpolated segment distances.

    Both proximal and distal link chains are densely interpolated, and all
    pair-wise distances are computed via scipy.spatial.distance.cdist for
    vectorised performance.
    """
    margin = 0.1 if robot_type == "ur3e" else 0.16
    pos = get_ur_fk_frames(joints, robot_type)

    # Generate points along the proximal structure (base → shoulder → elbow)
    base_link = get_interpolated_points(pos[0], pos[1], 3)
    upper_arm = get_interpolated_points(pos[1], pos[2], 6)
    proximal_points = base_link + upper_arm

    # Generate points along forearm (elbow → wrist1, skip last 2 to avoid false positives)
    forearm = get_interpolated_points(pos[2], pos[3], 6)

    # Dense interpolation along the distal chain (wrist1 → wrist2 → wrist3 → flange)
    distal_points = (
        get_interpolated_points(pos[3], pos[4], 4)
        + get_interpolated_points(pos[4], pos[5], 4)
        + get_interpolated_points(pos[5], pos[6], 4)
    )

    proximal_arr = np.array(proximal_points)
    distal_arr = np.array(distal_points)

    # 1. Check distal chain against base + upper arm (vectorised)
    dists = cdist(distal_arr, proximal_arr)
    min_prox_idx = np.unravel_index(np.argmin(dists), dists.shape)
    min_prox = float(dists[min_prox_idx])
    if min_prox < margin:
        if return_details:
            return True, {
                "pair": "distal-proximal",
                "distance": min_prox,
                "threshold": margin,
                "distal_point": distal_arr[min_prox_idx[0]],
                "proximal_point": proximal_arr[min_prox_idx[1]],
            }
        return True

    # 2. Check distal chain against forearm (skip last 2 forearm pts to avoid
    #    false positives at the wrist1 junction)
    if len(forearm) > 2:
        forearm_arr = np.array(forearm[:-2])
        dists_forearm = cdist(distal_arr, forearm_arr)
        forearm_margin = margin * 0.5
        min_forearm_idx = np.unravel_index(np.argmin(dists_forearm), dists_forearm.shape)
        min_forearm = float(dists_forearm[min_forearm_idx])
        if min_forearm < forearm_margin:
            if return_details:
                return True, {
                    "pair": "distal-forearm",
                    "distance": min_forearm,
                    "threshold": forearm_margin,
                    "distal_point": distal_arr[min_forearm_idx[0]],
                    "forearm_point": forearm_arr[min_forearm_idx[1]],
                }
            return True

    if return_details:
        return False, {
            "closest_distal_proximal": min_prox,
            "distal_proximal_threshold": margin,
        }
    return False

def is_pose_safe(
    joints: np.ndarray,
    tcp_position: np.ndarray | None = None,
    singularity_threshold: float = 0.08,
    shoulder_singularity_radius: float = 0.02,
    robot_type: str = "ur3e"
) -> bool:
    """
    Check if the robot pose avoids singularities (shoulder, elbow, wrist)
    and base self-collision.

    Shoulder singularity: the wrist center (Wrist-1 origin, pos[4]) lies on
    the base Z-axis (XY radius → 0), making joint-0 unable to change the
    wrist-center position.
    """
    joints = np.asarray(joints)

    # Wrist singularity (joint 4 near 0 or +-pi)
    j4_mod = abs(joints[4]) % np.pi
    if j4_mod < singularity_threshold or j4_mod > (np.pi - singularity_threshold):
        logger.warning(f"Pose rejected: Near wrist singularity (joint 4 = {joints[4]:.3f} rad)")
        return False

    # Elbow singularity (joint 2 near 0 or +-pi)
    j2_mod = abs(joints[2]) % np.pi
    if j2_mod < singularity_threshold or j2_mod > (np.pi - singularity_threshold):
        logger.warning(f"Pose rejected: Near elbow singularity (joint 2 = {joints[2]:.3f} rad)")
        return False

    # Shoulder singularity: wrist center (Wrist-1 origin = pos[4]) on base Z-axis
    fk_frames = get_ur_fk_frames(joints, robot_type)
    wrist_center = fk_frames[4]  # Wrist-1 origin in base frame
    wrist_xy_radius = np.linalg.norm(wrist_center[:2])
    if wrist_xy_radius < shoulder_singularity_radius:
        logger.warning(
            f"Pose rejected: Near shoulder singularity "
            f"(wrist center XY radius = {wrist_xy_radius:.4f}m)"
        )
        return False

    # Prevent base self-collision via forward kinematics distance checking
    collision, collision_details = is_self_collision(
        joints, robot_type, return_details=True
    )
    if collision:
        logger.warning(
            "Pose rejected: Analytic self-collision detected "
            f"({collision_details['pair']}, "
            f"distance={collision_details['distance']:.3f}m < "
            f"{collision_details['threshold']:.3f}m)"
        )
        return False

    # Simple preventative check for TCP entering a low radius area near origin
    if tcp_position is not None:
        r = np.linalg.norm(tcp_position[:2])
        z = tcp_position[2]
        if z < 0.05 and r < 0.15:
            logger.warning(f"Pose rejected: Too close to base origin (z={z:.3f}m, r={r:.3f}m)")
            return False

    return True


class ExponentialFilter:
    """Low-pass exponential moving average filter for noise reduction."""

    def __init__(self, alpha: float = 0.3, dim: int = 3):
        self.alpha = alpha
        self.dim = dim
        self.value = np.zeros(dim)
        self.initialized = False

    def update(self, measurement: np.ndarray) -> np.ndarray:
        if not self.initialized:
            self.value = np.asarray(measurement, dtype=float).copy()
            self.initialized = True
        else:
            self.value = self.alpha * measurement + (1 - self.alpha) * self.value
        return self.value.copy()

    def reset(self) -> None:
        self.value = np.zeros(self.dim)
        self.initialized = False


class TimeAwareLowPassFilter:
    """First-order low-pass filter whose smoothing follows the measured dt."""

    def __init__(self, cutoff_hz: float, dim: int):
        self.cutoff_hz = float(cutoff_hz)
        self.dim = dim
        self.value = np.zeros(dim)
        self.initialized = False

    def update(self, measurement: np.ndarray, dt: float) -> np.ndarray:
        measurement = np.asarray(measurement, dtype=float)
        if not self.initialized:
            self.value = measurement.copy()
            self.initialized = True
            return self.value.copy()

        dt = max(float(dt), 1e-6)
        alpha = 1.0 - np.exp(-2.0 * np.pi * self.cutoff_hz * dt)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        self.value = alpha * measurement + (1.0 - alpha) * self.value
        return self.value.copy()

    def reset(self) -> None:
        self.value = np.zeros(self.dim)
        self.initialized = False


class GravityCompensator:
    """NumPy compatibility adapter for the pure v2 gravity compensator."""

    GRAVITY_BASE = np.array([0.0, 0.0, -9.81])

    def __init__(self, mass: float, com: np.ndarray, filter_alpha: float = 0.15):
        self._impl = _PureGravityCompensator(
            mass,
            np.asarray(com, dtype=float).reshape(-1),
            filter_alpha=filter_alpha,
        )

    @property
    def mass(self) -> float:
        return self._impl.mass_kg

    @property
    def com(self) -> np.ndarray:
        return np.asarray(self._impl.center_of_mass_m, dtype=float)

    @property
    def force_bias(self) -> np.ndarray:
        return np.asarray(self._impl.force_bias, dtype=float)

    @property
    def calibrated(self) -> bool:
        return self._impl.calibrated

    def _gravity_wrench(self, R_tool_to_base: np.ndarray) -> np.ndarray:
        return np.asarray(
            self._impl.gravity_wrench(R_tool_to_base),
            dtype=float,
        )

    def compensate(
        self,
        raw_wrench: np.ndarray,
        R_tool_to_base: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(
            self._impl.compensate(raw_wrench, R_tool_to_base),
            dtype=float,
        )

    def add_calibration_sample(
        self,
        raw_wrench: np.ndarray,
        R_tool_to_base: np.ndarray,
    ) -> None:
        self._impl.add_calibration_sample(raw_wrench, R_tool_to_base)

    def finish_calibration(self) -> np.ndarray:
        bias = np.asarray(self._impl.finish_calibration(), dtype=float)
        if self.calibrated:
            logger.info(
                f"Force sensor calibrated - bias F=[{bias[0]:+.3f}, "
                f"{bias[1]:+.3f}, {bias[2]:+.3f}] N "
                f"T=[{bias[3]:+.4f}, {bias[4]:+.4f}, {bias[5]:+.4f}] Nm"
            )
        else:
            logger.warning("GravityCompensator: no calibration samples collected")
        return bias

    def reset_baseline(self, bias: np.ndarray | None = None) -> None:
        self._impl.reset_baseline(bias)
