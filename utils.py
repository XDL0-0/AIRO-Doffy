import logging
import numpy as np
from scipy.spatial.transform import Rotation as R

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
    joint_threshold: float,
) -> bool:
    if previous_joints is None:
        return True
    joint_diff = np.abs(np.array(new_joints) - np.array(previous_joints))
    if np.any(joint_diff > joint_threshold):
        logger.warning(f"Joint change too large: {joint_diff}, keeping pose")
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


class GravityCompensator:
    """Remove tool gravity and sensor bias from F/T readings.

    The UR RTDE ``getActualTCPForce`` returns the wrench at the TCP expressed
    in the **base** frame.  This class computes the gravitational wrench of the
    tool payload (gripper) at the TCP and subtracts it together with any
    constant sensor bias that was measured during an initial calibration phase.

    Coordinate convention (UR base frame):
        Z points **up**, so gravity = [0, 0, -9.81] m/s².
    """

    GRAVITY_BASE = np.array([0.0, 0.0, -9.81])

    def __init__(self, mass: float, com: np.ndarray, filter_alpha: float = 0.15):
        self.mass = mass
        self.com = np.asarray(com, dtype=float)
        self.force_bias = np.zeros(6)
        self.calibrated = False
        self._calib_buf: list[np.ndarray] = []
        self._filter = ExponentialFilter(alpha=filter_alpha, dim=6)

    def _gravity_wrench(self, R_tool_to_base: np.ndarray) -> np.ndarray:
        """Gravity wrench at TCP, expressed in the base frame."""
        f_grav = self.mass * self.GRAVITY_BASE
        com_base = R_tool_to_base @ self.com
        tau_grav = np.cross(com_base, f_grav)
        return np.concatenate([f_grav, tau_grav])

    def compensate(
        self, raw_wrench: np.ndarray, R_tool_to_base: np.ndarray
    ) -> np.ndarray:
        """Return the contact wrench with gravity and bias removed, then filtered."""
        contact = raw_wrench - self._gravity_wrench(R_tool_to_base) - self.force_bias
        return self._filter.update(contact)

    def add_calibration_sample(
        self, raw_wrench: np.ndarray, R_tool_to_base: np.ndarray
    ) -> None:
        residual = raw_wrench - self._gravity_wrench(R_tool_to_base)
        self._calib_buf.append(residual)

    def finish_calibration(self) -> np.ndarray:
        if not self._calib_buf:
            logger.warning("GravityCompensator: no calibration samples collected")
            return self.force_bias
        self.force_bias = np.mean(self._calib_buf, axis=0)
        self._calib_buf.clear()
        self.calibrated = True
        self._filter.reset()
        logger.info(
            f"Force sensor calibrated — bias F=[{self.force_bias[0]:+.3f}, "
            f"{self.force_bias[1]:+.3f}, {self.force_bias[2]:+.3f}] N  "
            f"T=[{self.force_bias[3]:+.4f}, {self.force_bias[4]:+.4f}, "
            f"{self.force_bias[5]:+.4f}] Nm"
        )
        return self.force_bias
