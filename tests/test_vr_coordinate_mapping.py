import numpy as np
from scipy.spatial.transform import Rotation

from config import Config
from robot_teleop import RobotTeleop


def _teleop_with_configured_axes(cfg: Config) -> RobotTeleop:
    teleop = object.__new__(RobotTeleop)
    teleop.vr_to_robot_axes = cfg.VR_TO_ROBOT_AXES
    teleop.vr_to_robot_handedness = float(np.linalg.det(cfg.VR_TO_ROBOT_AXES))
    teleop.vr_angular_axes = teleop.vr_to_robot_handedness * teleop.vr_to_robot_axes
    teleop.vr_rotation_axis_signs = cfg.VR_ROTATION_AXIS_SIGNS
    return teleop


def test_configured_vr_translation_axes() -> None:
    cfg = Config()
    teleop = _teleop_with_configured_axes(cfg)
    identity_quaternion = np.array([0.0, 0.0, 0.0, 1.0])

    right = teleop._vr_pose_to_robot_se3([1.0, 0.0, 0.0], identity_quaternion)
    up = teleop._vr_pose_to_robot_se3([0.0, 1.0, 0.0], identity_quaternion)
    forward = teleop._vr_pose_to_robot_se3([0.0, 0.0, 1.0], identity_quaternion)

    np.testing.assert_allclose(right.translation, cfg.VR_TO_ROBOT_AXES[:, 0])
    np.testing.assert_allclose(up.translation, cfg.VR_TO_ROBOT_AXES[:, 1])
    np.testing.assert_allclose(forward.translation, cfg.VR_TO_ROBOT_AXES[:, 2])


def test_vr_quaternion_uses_same_axis_transform() -> None:
    cfg = Config()
    teleop = _teleop_with_configured_axes(cfg)
    quaternion_vr = Rotation.from_euler("xyz", [0.2, -0.3, 0.4]).as_quat()

    pose_robot = teleop._vr_pose_to_robot_se3([0.0, 0.0, 0.0], quaternion_vr)
    rotation_vr = Rotation.from_quat(quaternion_vr).as_matrix()
    expected_rotation = (
        cfg.VR_TO_ROBOT_AXES
        @ rotation_vr
        @ cfg.VR_TO_ROBOT_AXES.T
    )

    np.testing.assert_allclose(
        pose_robot.rotation_matrix,
        expected_rotation,
        atol=1e-12,
    )


def test_realman_controller_rotation_axes_map_to_matching_eef_axes() -> None:
    cfg = Config(ROBOT_TYPE="realman", ROBOT_IP="192.0.2.1")
    teleop = _teleop_with_configured_axes(cfg)
    controller_axes = {
        "pitch": np.array([0.25, 0.0, 0.0]),
        "yaw": np.array([0.0, 0.25, 0.0]),
        "roll": np.array([0.0, 0.0, 0.25]),
    }
    expected_eef_axes = {
        "pitch": np.array([0.0, -0.25, 0.0]),
        "yaw": np.array([0.0, 0.0, 0.25]),
        "roll": np.array([-0.25, 0.0, 0.0]),
    }

    for name, rotation_vector_vr in controller_axes.items():
        rotation_vector_robot = teleop.vr_angular_axes @ rotation_vector_vr
        mapped = teleop._remap_controller_rotation_vector(
            rotation_vector_robot
        )
        np.testing.assert_allclose(
            mapped,
            expected_eef_axes[name],
            atol=1e-12,
            err_msg=name,
        )
