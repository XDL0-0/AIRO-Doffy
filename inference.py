"""LeRobot policy inference for UR robot control.

Usage:
    # From HuggingFace
    python inference.py --policy username/my_act_policy

    # From local checkpoint
    python inference.py --policy ./checkpoints/my_policy

    # With custom options
    python inference.py --policy username/my_policy --device cuda --fps 10 --episodes 3


    python inference.py   --policy ~/outputs/Baseline_ --action_type delta_tcp --horizon 16   --action_horizon 8   --n_obs_steps 2
    IXDLI/dp_pnp_long_filtered_h16_a8 
    baseline:
    IXDLI/fm_dit_pnp_long_100k
    IXDLI/fm_unet_pnp_long_100k
    IXDLI/ddpm_unet_pnp_long_100k
    IXDLI/ddim_unet_pnp_long_100k
    IXDLI/act_baseline_pnp_long_100k
    IXDLI/a2a_noise_pnp_long_100k
    outputs/train/fm_dit_pplong

Control mode (torque / position) is determined by Config.TORQUE_MODE.
"""

from __future__ import annotations

import time
import argparse
import sys
import threading
from collections import deque
import cv2
import numpy as np
import torch
import pyrealsense2 as rs

import utils
from config import Config
from data_schema import action_representation, build_data_schema, state_representation
from robot_backend import make_robot_backend
from airo_camera_toolkit.cameras.realsense.realsense import Realsense

cfg = Config()

POLICY_CLASS_MAP = {
    "act": "lerobot.policies.act.modeling_act.ACTPolicy",
    "diffusion": "lerobot.policies.diffusion.modeling_diffusion.DiffusionPolicy",
    "tdmpc": "lerobot.policies.tdmpc.modeling_tdmpc.TDMPCPolicy",
    "vqbet": "lerobot.policies.vqbet.modeling_vqbet.VQBeTPolicy",
    "pi0": "lerobot.policies.pi0.modeling_pi0.PI0Policy",
    "pi05": "lerobot.policies.pi05.modeling_pi05.PI05Policy",
    "smolvla": "lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy",
    "groot": "lerobot.policies.groot.modeling_groot.GrootPolicy",
    "act_baseline": "lerobot_policy_actiongen.act_baseline.modeling_act_baseline.ACTBaselinePolicy",
    "ddpm_unet": "lerobot_policy_actiongen.ddpm_unet.modeling_ddpm_unet.DDPMUnetPolicy",
    "ddim_unet": "lerobot_policy_actiongen.ddim_unet.modeling_ddim_unet.DDIMUnetPolicy",
    "score_unet": "lerobot_policy_actiongen.score_unet.modeling_score_unet.ScoreUnetPolicy",
    "a2a_noise": "lerobot_policy_actiongen.a2a_noise.modeling_a2a_noise.A2ANoisePolicy",
    "ddpm_dit": "lerobot_policy_actiongen.ddpm_dit.modeling_ddpm_dit.DDPMDiTPolicy",
    "fm_unet": "lerobot_policy_actiongen.fm_unet.modeling_fm_unet.FMUnetPolicy",
    "fm_dit": "lerobot_policy_actiongen.fm_dit.modeling_fm_dit.FMDiTPolicy",
    "forceflowpp": "policies.forceflowpp.modeling_forceflowpp.ForceFlowPPPolicy",
}


# ── Policy loading ────────────────────────────────────────────────────────

def _resolve_model_dir(pretrained_path: str):
    """Download (if needed) and locate the directory containing config.json."""
    from pathlib import Path

    local_path = Path(pretrained_path).resolve()

    if not local_path.exists():
        from huggingface_hub import snapshot_download
        utils.logger.info(f"Downloading from HuggingFace: {pretrained_path}")
        local_path = Path(snapshot_download(pretrained_path))

    if (local_path / "pretrained_model" / "config.json").exists():
        return local_path / "pretrained_model"
    if (local_path / "config.json").exists():
        return local_path

    raise FileNotFoundError(
        f"config.json not found in {local_path} or "
        f"{local_path / 'pretrained_model'}"
    )


def _get_policy_class(policy_type: str):
    """Resolve the concrete policy class from a type string (e.g. 'diffusion')."""
    from importlib import import_module

    qualified = POLICY_CLASS_MAP.get(policy_type)
    if qualified is None:
        raise ValueError(
            f"Unknown policy type '{policy_type}'. "
            f"Supported: {list(POLICY_CLASS_MAP.keys())}"
        )
    module_path, class_name = qualified.rsplit(".", 1)
    mod = import_module(module_path)
    return getattr(mod, class_name)


def load_pretrained_policy(pretrained_path: str, device: str):
    """Load a lerobot policy from HuggingFace hub or a local checkpoint directory.

    Handles two common checkpoint layouts:
      1. ``config.json`` at root  (standard HF / lerobot flat layout)
      2. ``pretrained_model/config.json``  (lerobot training output)

    If the saved config.json contains fields unknown to the installed lerobot
    version, those fields are automatically stripped before loading.
    """
    import json
    import os
    import shutil
    import tempfile
    from pathlib import Path

    utils.logger.info(f"Loading policy from: {pretrained_path}")
    model_dir = _resolve_model_dir(pretrained_path)
    utils.logger.info(f"Model directory: {model_dir}")

    with open(model_dir / "config.json") as f:
        raw_config = json.load(f)

    policy_type = raw_config.get("type", "").lower()
    utils.logger.info(f"Detected policy type: '{policy_type}'")
    policy_cls = _get_policy_class(policy_type)

    from dataclasses import fields as dc_fields
    valid_fields = {f.name for f in dc_fields(policy_cls.config_class)}
    unknown = set(raw_config.keys()) - valid_fields - {"type"}

    if unknown:
        utils.logger.warning(
            f"Config has unsupported fields: {unknown}. Patching..."
        )
        patched = {k: v for k, v in raw_config.items() if k not in unknown}

        tmp_dir = Path(tempfile.mkdtemp())
        try:
            with open(tmp_dir / "config.json", "w") as f:
                json.dump(patched, f, indent=4)
            for item in model_dir.iterdir():
                if item.name != "config.json":
                    os.symlink(item.resolve(), tmp_dir / item.name)
            policy = policy_cls.from_pretrained(str(tmp_dir))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        policy = policy_cls.from_pretrained(str(model_dir))

    policy.to(device)
    policy.eval()
    utils.logger.info(f"Loaded {type(policy).__name__} on {device}")
    return policy


def reset_policy(policy) -> None:
    """Reset policy state and keep per-camera image queues when needed."""
    policy.reset()

    image_features = getattr(getattr(policy, "config", None), "image_features", None)
    queues = getattr(policy, "_queues", None)
    n_obs_steps = getattr(getattr(policy, "config", None), "n_obs_steps", None)
    if image_features and isinstance(queues, dict) and n_obs_steps is not None:
        for key in image_features:
            queues.setdefault(key, deque(maxlen=n_obs_steps))


def load_policy_processor(pretrained_path: str, filename: str, device: str | None = None):
    """Load a LeRobot policy pre/post processor from a checkpoint, if present."""
    model_dir = _resolve_model_dir(pretrained_path)
    if not (model_dir / filename).exists():
        utils.logger.warning(f"{filename} not found in {model_dir}")
        return None

    try:
        from lerobot.processor import PolicyProcessorPipeline
    except ImportError:
        utils.logger.warning("Installed LeRobot does not expose PolicyProcessorPipeline")
        return None

    kwargs = {}
    if filename == "policy_postprocessor.json":
        try:
            from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action

            kwargs = {
                "to_transition": policy_action_to_transition,
                "to_output": transition_to_policy_action,
            }
        except ImportError:
            pass

    overrides = {}
    if filename == "policy_preprocessor.json" and device is not None:
        overrides["device_processor"] = {"device": device}
    elif filename == "policy_postprocessor.json":
        overrides["device_processor"] = {"device": "cpu"}

    processor = PolicyProcessorPipeline.from_pretrained(
        str(model_dir),
        config_filename=filename,
        overrides=overrides,
        **kwargs,
    )
    utils.logger.info(f"Loaded {filename} from {model_dir}")
    return processor


def apply_policy_timing_overrides(
    policy,
    horizon: int | None,
    action_horizon: int | None,
    n_obs_steps: int | None,
) -> None:
    overrides = {
        "horizon": horizon,
        "n_action_steps": action_horizon,
        "n_obs_steps": n_obs_steps,
    }
    for name, value in overrides.items():
        if value is None:
            continue
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        if not hasattr(policy.config, name):
            raise ValueError(f"Policy config does not support {name}")
        setattr(policy.config, name, value)

    if hasattr(policy, "diffusion") and hasattr(policy.diffusion, "config"):
        for name, value in overrides.items():
            if value is not None and hasattr(policy.diffusion.config, name):
                setattr(policy.diffusion.config, name, value)

    if hasattr(policy.config, "horizon") and hasattr(policy.config, "n_action_steps") and hasattr(policy.config, "n_obs_steps"):
        max_action_steps = policy.config.horizon - policy.config.n_obs_steps + 1
        if policy.config.n_action_steps > max_action_steps:
            raise ValueError(
                "action_horizon must be <= horizon - n_obs_steps + 1 "
                f"({max_action_steps}), got {policy.config.n_action_steps}"
            )

    reset_policy(policy)
    utils.logger.info(
        "Policy timing: "
        f"horizon={getattr(policy.config, 'horizon', 'n/a')}, "
        f"action_horizon={getattr(policy.config, 'n_action_steps', 'n/a')}, "
        f"n_obs_steps={getattr(policy.config, 'n_obs_steps', 'n/a')}"
    )


# ── Hardware wrappers ─────────────────────────────────────────────────────

class InferenceRobotController:
    """Robot + gripper controller for policy inference."""

    def __init__(self):
        self.torque_mode = cfg.TORQUE_MODE
        self.backend = make_robot_backend(cfg)
        self.ur = self.backend.robot
        self.gripper = self.backend.gripper
        self.dof = self.backend.dof
        self.initial_joint = self.backend.initial_joint_configuration(cfg.INITIAL_JOINT)
        self.schema = build_data_schema(cfg.DATA_TYPE, self.dof)
        self.state_representation = state_representation(cfg.DATA_TYPE)
        self.action_representation = action_representation(cfg.DATA_TYPE)
        self.last_tcp_quat: np.ndarray | None = None
        self.delta_tcp_target_pose: np.ndarray | None = None
        time.sleep(1.0)

        self._move_to_initial()

        self.wrench_mode = (cfg.FORCE_COLLECT or cfg.TORQUE_COLLECT) and self.backend.supports_force
        self.force_mode = cfg.FORCE_COLLECT and self.backend.supports_force
        self.torque_collect = cfg.TORQUE_COLLECT and self.backend.supports_force
        self.gravity_comp = cfg.GRAVITY_COMP and self.wrench_mode
        if self.gravity_comp:
            self._gravity_compensator = utils.GravityCompensator(
                mass=cfg.TOOL_MASS,
                com=cfg.TOOL_COM,
                filter_alpha=cfg.GRAVITY_COMP_FILTER_ALPHA,
            )
            time.sleep(1.0)
            self._calibrate_force()

        control_str = "torque" if self.torque_mode else "position"
        utils.logger.info(
            f"Robot ready | robot={self.backend.dataset_robot_type} | control={control_str} | dof={self.dof}"
        )

    def _move_to_initial(self) -> None:
        utils.logger.info("Moving to initial joint configuration...")
        self.delta_tcp_target_pose = None
        self.backend.reset(self.initial_joint)
        time.sleep(1.0)

    def _calibrate_force(self) -> None:
        n = cfg.GRAVITY_CALIB_SAMPLES
        utils.logger.info(f"Calibrating force sensor ({n} samples)...")
        for _ in range(n):
            raw = self._raw_force()
            R = self._tool_rotation()
            self._gravity_compensator.add_calibration_sample(raw, R)
            time.sleep(0.005)
        self._gravity_compensator.finish_calibration()

    def _raw_force(self) -> np.ndarray:
        force = self.backend.get_tcp_force()
        return np.zeros(6) if force is None else np.asarray(force, dtype=float)

    def _tool_rotation(self) -> np.ndarray:
        return self.backend.get_tcp_pose()[:3, :3]

    @staticmethod
    def _normalize_gripper_width(gripper_width_m: float) -> float:
        return float(np.clip(gripper_width_m / cfg.GRIPPER_MAX, 0.0, 1.0))

    @staticmethod
    def _denormalize_gripper_width(gripper_width_norm: float) -> float:
        return float(np.clip(gripper_width_norm, 0.0, 1.0) * cfg.GRIPPER_MAX)

    def get_state(self) -> np.ndarray:
        """Return observation.state according to Config.DATA_TYPE and robot DoF."""
        gripper = self._normalize_gripper_width(self.gripper.get_current_width())
        if self.state_representation == "tcp":
            return np.concatenate([self.get_tcp_pose_extra(), [gripper]]).astype(np.float32)
        joints = np.asarray(self.backend.get_joint_configuration(), dtype=float)
        return np.concatenate([joints, [gripper]]).astype(np.float32)

    def get_tcp_pose_extra(self) -> np.ndarray:
        """TCP pose as [qx, qy, qz, qw, x, y, z], matching DATA_TYPE='both'."""
        from airo_spatial_algebra.se3 import SE3Container

        tcp = self.backend.get_tcp_pose()
        se3 = SE3Container.from_homogeneous_matrix(tcp)
        self.last_tcp_quat = utils.quat_cal(se3.rotation_matrix, self.last_tcp_quat)
        return np.concatenate([self.last_tcp_quat, se3.translation]).astype(np.float32)

    def get_force(self) -> np.ndarray:
        """TCP force [Fx, Fy, Fz] shape (3,) float32."""
        return self.get_wrench()[:3]

    def get_torque(self) -> np.ndarray:
        """TCP torque [Tx, Ty, Tz] shape (3,) float32."""
        return self.get_wrench()[3:6]

    def get_wrench(self) -> np.ndarray:
        """Compensated TCP wrench [Fx, Fy, Fz, Tx, Ty, Tz] shape (6,) float32."""
        raw = self._raw_force()
        if self.gravity_comp:
            R = self._tool_rotation()
            return self._gravity_compensator.compensate(raw, R).astype(np.float32)
        return raw.astype(np.float32)

    def _current_tcp_pose(self) -> np.ndarray:
        return np.asarray(self.backend.get_tcp_pose(), dtype=float)

    def _servo_tcp_pose(self, tcp_pose: np.ndarray, dt: float) -> bool:
        result = self.backend.command_tcp_pose(tcp_pose, dt)
        return result.accepted

    def _tcp_pose_from_action(self, action: np.ndarray) -> tuple[np.ndarray, float | None]:
        from airo_spatial_algebra.se3 import SE3Container

        action = np.asarray(action, dtype=np.float64)
        if action.shape[0] >= 8:
            tcp = SE3Container.from_quaternion_and_translation(action[:4], action[4:7])
            gripper = float(action[7])
        elif action.shape[0] >= 7:
            tcp = SE3Container.from_rotation_vector_and_translation(action[3:6], action[:3])
            gripper = float(action[6])
        else:
            raise ValueError(
                "TCP action must be [qx,qy,qz,qw,x,y,z,gripper] or "
                "[x,y,z,rx,ry,rz,gripper]."
            )
        return tcp.homogeneous_matrix, gripper

    @staticmethod
    def _project_to_so3(rotation: np.ndarray) -> np.ndarray:
        """Project a near-rotation matrix back onto SO(3)."""
        rotation = np.asarray(rotation, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError(f"Expected a 3x3 rotation matrix, got shape {rotation.shape}")
        if not np.all(np.isfinite(rotation)):
            raise ValueError("Rotation matrix contains non-finite values.")

        u, _, vt = np.linalg.svd(rotation)
        projected = u @ vt
        if np.linalg.det(projected) < 0:
            u[:, -1] *= -1
            projected = u @ vt
        return projected

    def _delta_tcp_pose_from_action(self, action: np.ndarray) -> tuple[np.ndarray, float | None]:
        from airo_spatial_algebra.se3 import SE3Container

        action = np.asarray(action, dtype=np.float64)
        if action.shape[0] < 6:
            raise ValueError("Delta TCP action must have at least 6 pose dimensions.")

        if self.delta_tcp_target_pose is None:
            self.delta_tcp_target_pose = self._current_tcp_pose()

        current_target = SE3Container.from_homogeneous_matrix(self.delta_tcp_target_pose)
        delta_translation = action[:3]
        delta_rotation, _ = cv2.Rodrigues(action[3:6])
        target_rotation = self._project_to_so3(delta_rotation @ current_target.rotation_matrix)
        target_translation = current_target.translation + delta_translation
        target = SE3Container.from_rotation_matrix_and_translation(
            target_rotation, target_translation
        )
        self.delta_tcp_target_pose = target.homogeneous_matrix.copy()
        gripper = float(action[6]) if action.shape[0] > 6 else None
        return self.delta_tcp_target_pose, gripper

    def _set_gripper_from_normalized(self, gripper: float | None) -> None:
        if gripper is None:
            return
        self.gripper._set_target_width(self._denormalize_gripper_width(float(gripper)))

    def execute_action(
        self,
        action: np.ndarray,
        action_type: str = "joint",
        dt: float | None = None,
    ) -> bool:
        if dt is None:
            dt = 1 / cfg.UR_CTRL_RATE

        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action_type == "tcp":
            tcp_pose, gripper = self._tcp_pose_from_action(action)
            ok = self._servo_tcp_pose(tcp_pose, dt)
            if ok:
                self._set_gripper_from_normalized(gripper)
            return ok

        if action_type == "delta_tcp":
            previous_delta_target = (
                None if self.delta_tcp_target_pose is None else self.delta_tcp_target_pose.copy()
            )
            tcp_pose, gripper = self._delta_tcp_pose_from_action(action)
            ok = self._servo_tcp_pose(tcp_pose, dt)
            if not ok:
                self.delta_tcp_target_pose = previous_delta_target
            if ok:
                self._set_gripper_from_normalized(gripper)
            return ok

        # Joint action = [joint_targets(dof), gripper_target].
        if action.shape[0] < self.dof:
            raise ValueError(f"Joint action has {action.shape[0]} values, expected at least {self.dof}.")
        joint_target = np.asarray(action[: self.dof], dtype=np.float64)

        if self.backend.is_ur and not utils.is_joint_within_limits(joint_target):
            utils.logger.warning("Joint limits exceeded — skipping.")
            return False

        current_joints = np.asarray(self.backend.get_joint_configuration(), dtype=float)
        threshold = np.asarray(cfg.MOVE_THRESHOLD, dtype=float)
        if threshold.shape != (self.dof,):
            threshold = np.resize(threshold, self.dof)
        if not utils.is_joint_change_safe(current_joints, joint_target, threshold):
            return False

        self.backend.command_joint_configuration(joint_target, dt)

        if len(action) > self.dof:
            self._set_gripper_from_normalized(float(action[self.dof]))

        return True

    def start_freedrive(self) -> None:
        """Suspend robot control and enter freedrive/teach mode when supported."""
        self.backend.start_freedrive()
        utils.logger.info("Freedrive enabled. Move the robot, then press Enter to resume.")

    def stop_freedrive(self) -> None:
        """Exit freedrive/teach mode and make the current pose the next target."""
        self.backend.stop_freedrive()
        self.delta_tcp_target_pose = None
        utils.logger.info("Freedrive disabled. Inference will resume from current state.")

    def reset(self) -> None:
        self.delta_tcp_target_pose = None
        self.gripper.move(cfg.GRIPPER_MAX)
        self.backend.reset(self.initial_joint)
        time.sleep(1.0)
        utils.logger.info("Robot reset to initial pose.")

    def cleanup(self) -> None:
        try:
            self.backend.cleanup()
        except Exception as e:
            utils.logger.error(f"Cleanup error: {e}")


class InferenceCameraManager:
    """Direct Realsense camera access (no UDP streaming)."""

    def __init__(self):
        ctx = rs.context()
        devices = ctx.query_devices()
        self.cameras: dict[str, Realsense] = {}

        for i, dev in enumerate(devices):
            serial = dev.get_info(rs.camera_info.serial_number)
            cam = Realsense(
                fps=cfg.REALSENSE_FPS,
                resolution=cfg.REALSENSE_RESOLUTION,
                enable_depth=False,
                enable_pointcloud=False,
                enable_hole_filling=False,
                serial_number=serial,
            )
            self.cameras[f"camera_{i}"] = cam
            utils.logger.info(f"Camera {i}: serial={serial}")

        self.num_cameras = len(self.cameras)
        if not self.num_cameras:
            utils.logger.warning("No Realsense cameras detected!")

    def get_images(self) -> dict[str, np.ndarray]:
        return {n: c.get_rgb_image() for n, c in self.cameras.items()}


class TactileDataHolder:
    """Minimal container so the tactile reader thread can write data
    without depending on CameraUDPManager."""

    def __init__(self):
        self._lock = threading.Lock()
        self.tactile_data: np.ndarray | None = None
        self.tactile_byte: bytes | None = None
        self.data = (None, {"Joystick_Press": False})


# ── Pause / tuning control ────────────────────────────────────────────────

class EnterPauseController:
    """Background Enter listener used to toggle paused freedrive tuning."""

    def __init__(self):
        self._condition = threading.Condition()
        self._requests: list[str] = []
        self._closed = False
        self._thread = threading.Thread(target=self._listen, daemon=True)

    def start(self) -> None:
        self._thread.start()
        utils.logger.info("Press Enter to pause into freedrive tuning. While paused, press Enter to resume or type r to restart.")

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def consume_request(self) -> str | None:
        with self._condition:
            if not self._requests:
                return None
            return self._requests.pop(0)

    def wait_for_request(self) -> str | None:
        with self._condition:
            while not self._requests and not self._closed:
                self._condition.wait(timeout=0.2)
            if not self._requests:
                return None
            return self._requests.pop(0)

    def _listen(self) -> None:
        while True:
            line = sys.stdin.readline()
            if line == "":
                return
            command = line.strip().lower() or "pause"
            with self._condition:
                self._requests.append(command)
                self._condition.notify_all()


# ── Observation builder ───────────────────────────────────────────────────

def _feature_shape(feature) -> tuple[int, ...]:
    if hasattr(feature, "shape"):
        return tuple(feature.shape)
    return tuple(feature.get("shape", ()))


def _is_visual_feature(feature) -> bool:
    ft_type = feature.type if hasattr(feature, "type") else feature.get("type", "")
    ft_text = str(ft_type).upper()
    return "VISUAL" in ft_text or "IMAGE" in ft_text or "VIDEO" in ft_text


def _target_chw(shape: tuple[int, ...]) -> tuple[int, int, int] | None:
    if len(shape) != 3:
        return None
    if shape[0] in {1, 3}:
        return int(shape[0]), int(shape[1]), int(shape[2])
    if shape[2] in {1, 3}:
        return int(shape[2]), int(shape[0]), int(shape[1])
    return None


def image_to_tensor(
    image: np.ndarray,
    feature,
    device: str,
    add_batch: bool,
) -> torch.Tensor:
    """Convert RGB/HWC or CHW images to float CHW tensors in [0, 1]."""
    img = np.asarray(image)
    if img.ndim != 3:
        raise ValueError(f"Expected image with 3 dimensions, got shape {img.shape}")

    if img.shape[0] in {1, 3} and img.shape[-1] not in {1, 3}:
        chw = img
    else:
        chw = np.transpose(img, (2, 0, 1))

    target = _target_chw(_feature_shape(feature))
    if target is not None and tuple(chw.shape) != target:
        from PIL import Image

        channels, height, width = target
        hwc = np.transpose(chw, (1, 2, 0))
        if hwc.shape[-1] == 1:
            pil_img = Image.fromarray(hwc[..., 0].astype(np.uint8), mode="L")
            resized = np.asarray(pil_img.resize((width, height)))[..., np.newaxis]
        else:
            resized = np.asarray(Image.fromarray(hwc.astype(np.uint8)).resize((width, height)))
        chw = np.transpose(resized, (2, 0, 1))
        if channels == 1 and chw.shape[0] != 1:
            chw = chw[:1]

    tensor = torch.from_numpy(chw.copy()).float()
    if tensor.numel() and tensor.max() > 1.5:
        tensor = tensor / 255.0
    if add_batch:
        tensor = tensor.unsqueeze(0)
    return tensor.to(device)


def value_to_tensor(value: np.ndarray, device: str, add_batch: bool) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if add_batch:
        tensor = tensor.unsqueeze(0)
    return tensor.to(device)


def zeros_for_feature(feature) -> np.ndarray:
    shape = _feature_shape(feature)
    if not shape:
        shape = (1,)
    return np.zeros(shape, dtype=np.float32)


def build_observation(
    state: np.ndarray,
    images: dict[str, np.ndarray],
    device: str,
    policy_config=None,
    force: np.ndarray | None = None,
    torque: np.ndarray | None = None,
    tactile: np.ndarray | None = None,
    tcp_pose: np.ndarray | None = None,
    add_batch: bool = True,
) -> dict[str, torch.Tensor]:
    """Construct observation dict matching LeRobot dataset features."""
    values: dict[str, np.ndarray] = {
        "observation.state": np.asarray(state, dtype=np.float32),
    }
    if force is not None:
        values["observation.force"] = np.asarray(force, dtype=np.float32)
    if torque is not None:
        values["observation.torque"] = np.asarray(torque, dtype=np.float32)
    if tactile is not None:
        values["observation.tactile"] = np.asarray(tactile, dtype=np.float32)
    if tcp_pose is not None:
        values["extra.tcp_pose"] = np.asarray(tcp_pose, dtype=np.float32)

    if policy_config is None:
        obs = {key: value_to_tensor(value, device, add_batch) for key, value in values.items()}
        for cam_name, img in images.items():
            key = f"observation.images.{cam_name}"
            obs[key] = image_to_tensor(img, {"shape": ()}, device, add_batch)
        return obs

    obs: dict[str, torch.Tensor] = {}
    for policy_key, feature in policy_config.input_features.items():
        if _is_visual_feature(feature):
            if policy_key in images:
                img = images[policy_key]
            elif policy_key.startswith("observation.images."):
                img = images.get(policy_key.removeprefix("observation.images."))
            elif policy_key == "observation.image" and images:
                first_key = sorted(images)[0]
                img = images[first_key]
            else:
                utils.logger.warning(f"Missing image feature for policy key: {policy_key}")
                continue
            obs[policy_key] = image_to_tensor(img, feature, device, add_batch)
            continue

        if policy_key in values:
            obs[policy_key] = value_to_tensor(values[policy_key], device, add_batch)
        else:
            utils.logger.warning(f"Missing non-visual feature for policy key: {policy_key}")
            obs[policy_key] = value_to_tensor(zeros_for_feature(feature), device, add_batch)

    return obs


def normalize_action_type(action_type: str) -> str:
    aliases = {
        "joint": "joint",
        "joint_configuration": "joint",
        "qpos": "joint",
        "tcp": "tcp",
        "tcp_quat": "tcp",
        "delta_tcp": "delta_tcp",
    }
    try:
        return aliases[action_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported action_type: {action_type}") from exc


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LeRobot policy inference for UR robot"
    )
    parser.add_argument(
        "--policy", required=True,
        help="HuggingFace repo_id (e.g. user/model) or local checkpoint path",
    )
    parser.add_argument("--device", default=None, help="cpu / cuda / cuda:0")
    parser.add_argument(
        "--fps", type=int, default=cfg.COLLECT_RATE,
        help=f"Inference rate in Hz (default: {cfg.COLLECT_RATE})",
    )
    parser.add_argument(
        "--episodes", type=int, default=cfg.INFERENCE_EPISODES,
        help="Number of episodes to run (0 = loop until Ctrl-C)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=cfg.INFERENCE_MAX_STEPS,
        help="Max timesteps per episode",
    )
    parser.add_argument("--horizon", type=int, default=None, help="Override policy prediction horizon")
    parser.add_argument("--action_horizon", type=int, default=None, help="Override policy action horizon")
    parser.add_argument("--n_obs_steps", type=int, default=None, help="Override number of observation steps")
    parser.add_argument(
        "--action_type",
        "--action-type",
        default="joint",
        choices=["joint", "joint_configuration", "qpos", "tcp", "tcp_quat", "delta_tcp"],
        help=(
            "Policy action representation. joint/joint_configuration/qpos expects "
            "[joint0..joint5, gripper]; tcp expects [qx,qy,qz,qw,x,y,z,gripper] "
            "or [x,y,z,rx,ry,rz,gripper]; delta_tcp expects "
            "[dx,dy,dz,drx,dry,drz,gripper]."
        ),
    )
    args = parser.parse_args()
    action_type = normalize_action_type(args.action_type)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tactile_mode = cfg.TACTILE_TRANSFER

    utils.logger.info("=== Inference Configuration ===")
    utils.logger.info(f"Task:    {cfg.TASK_NAME}")
    utils.logger.info(f"Policy:  {args.policy}")
    utils.logger.info(f"Device:  {device}")
    utils.logger.info(f"Control: {'torque' if cfg.TORQUE_MODE else 'position'}")
    utils.logger.info(f"Action:  {action_type}")
    utils.logger.info(f"Tactile: {tactile_mode}")
    utils.logger.info(
        f"FPS: {args.fps}  |  Episodes: {args.episodes}  |  Max steps: {args.max_steps}"
    )

    cameras = InferenceCameraManager()
    robot = InferenceRobotController()
    wrench_mode = robot.wrench_mode
    force_mode = robot.force_mode
    torque_mode = robot.torque_collect
    utils.logger.info(f"Force:   {force_mode}")
    utils.logger.info(f"Torque:  {torque_mode}")
    pause_controller = EnterPauseController()
    pause_controller.start()
    freedrive_active = False

    tactile_holder = None
    if tactile_mode:
        from tactile import MagtouchIliasSerialReader, MagtouchIliasSerialReaderConfig

        tactile_holder = TactileDataHolder()
        t_reader = MagtouchIliasSerialReader(
            config=MagtouchIliasSerialReaderConfig(
                ENABLE_WS=False,
                COM="/dev/serial/by-id/usb-Arduino_IO_Coupling_C6E76762B4D1E02A-if00",
                START_BYTE=0xAA,
                END_BYTE=0xCC,
            )
        )
        threading.Thread(
            target=t_reader.run, args=(tactile_holder,), daemon=True
        ).start()
        time.sleep(2)

    policy = load_pretrained_policy(args.policy, device)
    apply_policy_timing_overrides(policy, args.horizon, args.action_horizon, args.n_obs_steps)
    preprocessor = load_policy_processor(args.policy, "policy_preprocessor.json", device=device)
    postprocessor = load_policy_processor(args.policy, "policy_postprocessor.json")
    policy_input_keys = set(policy.config.input_features)
    needs_tcp_pose = "extra.tcp_pose" in policy_input_keys
    needs_force = "observation.force" in policy_input_keys
    needs_torque = "observation.torque" in policy_input_keys
    utils.logger.info(
        "Inference path: "
        f"{'policy_preprocessor -> ' if preprocessor is not None else ''}"
        "select_action"
        f"{' -> policy_postprocessor' if postprocessor is not None else ''}"
    )

    time.sleep(2)
    utils.logger.info("All systems ready.")

    episode = 0
    target_episodes = args.episodes if args.episodes > 0 else float("inf")

    try:
        while episode < target_episodes:
            utils.logger.info(f"--- Episode {episode} ---")
            robot.reset()
            reset_policy(policy)
            time.sleep(1.0)
            restart_episode = False

            for step in range(args.max_steps):
                command = pause_controller.consume_request()
                if command is not None:
                    if command in {"r", "restart", "new"}:
                        utils.logger.info("Restart requested. Resetting robot, policy queues, and camera observation loop.")
                        robot.reset()
                        reset_policy(policy)
                        restart_episode = True
                        break

                    try:
                        robot.start_freedrive()
                        freedrive_active = True
                        resume_command = pause_controller.wait_for_request()
                    except Exception as e:
                        utils.logger.error(f"Could not enter freedrive tuning: {e}")
                        resume_command = None
                    finally:
                        if freedrive_active:
                            try:
                                robot.stop_freedrive()
                            finally:
                                freedrive_active = False
                    if resume_command in {"r", "restart", "new"}:
                        utils.logger.info("Restart requested. Recovering initial pose and clearing policy action queues.")
                        robot.reset()
                        reset_policy(policy)
                        restart_episode = True
                        break
                    reset_policy(policy)
                    time.sleep(0.5)
                    continue

                t0 = time.time()

                state = robot.get_state()
                images = cameras.get_images()
                wrench = robot.get_wrench() if wrench_mode else None
                force = wrench[:3] if (needs_force and force_mode and wrench is not None) else None
                torque = wrench[3:6] if (needs_torque and torque_mode and wrench is not None) else None
                tcp_pose = robot.get_tcp_pose_extra() if needs_tcp_pose else None
                tactile = (
                    tactile_holder.tactile_data
                    if tactile_mode and tactile_holder is not None
                       and tactile_holder.tactile_data is not None
                    else None
                )

                obs = build_observation(
                    state,
                    images,
                    device,
                    policy_config=policy.config,
                    force=force,
                    torque=torque,
                    tactile=tactile,
                    tcp_pose=tcp_pose,
                    add_batch=preprocessor is None,
                )
                if preprocessor is not None:
                    obs = preprocessor(obs)

                with torch.no_grad():
                    action = policy.select_action(obs)
                    if postprocessor is not None:
                        action = postprocessor(action)

                action_np = action.squeeze(0).detach().cpu().numpy()
                robot.execute_action(action_np, action_type=action_type, dt=1.0 / args.fps)

                elapsed = time.time() - t0
                remaining = 1.0 / args.fps - elapsed
                if remaining > 0:
                    time.sleep(remaining)
                elif step > 0 and step % 50 == 0:
                    utils.logger.warning(
                        f"Step {step}: loop {elapsed:.3f}s > target {1/args.fps:.3f}s"
                    )

            if restart_episode:
                utils.logger.info(f"Episode {episode} restarted from initial pose.")
                continue

            episode += 1
            utils.logger.info(f"Episode {episode - 1} done ({args.max_steps} steps).")

    except KeyboardInterrupt:
        utils.logger.info("Interrupted by user.")

    finally:
        pause_controller.close()
        if freedrive_active:
            try:
                robot.stop_freedrive()
            except Exception as e:
                utils.logger.error(f"Error leaving freedrive during cleanup: {e}")
        robot.cleanup()
        utils.logger.info("Inference finished.")


if __name__ == "__main__":
    main()
