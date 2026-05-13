"""LeRobot policy inference for UR robot control.

Usage:
    # From HuggingFace
    python inference.py --policy username/my_act_policy

    # From local checkpoint
    python inference.py --policy ./checkpoints/my_policy

    # With custom options
    python inference.py --policy username/my_policy --device cuda --fps 10 --episodes 3

Control mode (torque / position) is determined by Config.TORQUE_MODE.
"""

from __future__ import annotations

import time
import argparse
import threading
import numpy as np
import torch
import pyrealsense2 as rs

import utils
from config import Config
from ur_teleop import make_robot, FastRobotiq2F85
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


# ── Hardware wrappers ─────────────────────────────────────────────────────

class InferenceRobotController:
    """UR arm + gripper controller for policy inference."""

    def __init__(self):
        self.torque_mode = cfg.TORQUE_MODE
        self.initial_joint = cfg.INITIAL_JOINT

        ruckig_params = None
        if cfg.RUCKIG_ENABLE and self.torque_mode:
            ruckig_params = {
                "max_vel": cfg.RUCKIG_MAX_VEL,
                "max_acc": cfg.RUCKIG_MAX_ACC,
                "max_jerk": cfg.RUCKIG_MAX_JERK,
            }
        self.ur, self.ik = make_robot(
            cfg.UR_IP, cfg.ROBOT_TYPE, self.torque_mode, self.initial_joint, ruckig_params
        )
        self.gripper = FastRobotiq2F85(cfg.UR_IP)
        self.gripper.open()
        time.sleep(1.0)

        self._move_to_initial()

        self.force_mode = cfg.FORCE_COLLECT and self.torque_mode
        self.gravity_comp = cfg.GRAVITY_COMP and self.force_mode
        if self.gravity_comp:
            self._gravity_compensator = utils.GravityCompensator(
                mass=cfg.TOOL_MASS,
                com=cfg.TOOL_COM,
                filter_alpha=cfg.FORCE_FILTER_ALPHA,
            )
            time.sleep(1.0)
            self._calibrate_force()

        control_str = "torque" if self.torque_mode else "position"
        utils.logger.info(f"Robot ready | control={control_str}")

    def _move_to_initial(self) -> None:
        utils.logger.info("Moving to initial joint configuration...")
        if self.torque_mode:
            self.ur.target_pos = self.initial_joint.tolist()
        else:
            self.ur.move_to_joint_configuration(self.initial_joint, 0.5).wait()
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
        if self.torque_mode:
            return np.array(self.ur.get_cached_tcp_force())
        return np.array(self.ur.get_tcp_force())

    def _tool_rotation(self) -> np.ndarray:
        if self.torque_mode:
            return self.ur.get_cached_tcp_pose()[:3, :3]
        return self.ur.get_tcp_pose()[:3, :3]

    def get_state(self) -> np.ndarray:
        """[joint_0 ... joint_5, gripper_width]  shape (7,) float32."""
        if self.torque_mode:
            joints = np.array(self.ur.get_cached_joint_configuration())
        else:
            joints = np.array(self.ur.get_joint_configuration())
        gripper = np.array([self.gripper.get_current_width()])
        return np.concatenate([joints, gripper]).astype(np.float32)

    def get_force(self) -> np.ndarray:
        """Compensated TCP force/torque  shape (6,) float32."""
        raw = self._raw_force()
        if self.gravity_comp:
            R = self._tool_rotation()
            return self._gravity_compensator.compensate(raw, R).astype(np.float32)
        return raw.astype(np.float32)

    def execute_action(self, action: np.ndarray) -> bool:
        """Execute action = [joint_targets(6), gripper_target(1)]."""
        joint_target = np.asarray(action[:6], dtype=np.float64)

        if not utils.is_joint_within_limits(joint_target):
            utils.logger.warning("Joint limits exceeded — skipping.")
            return False

        current_joints = self.get_state()[:6]
        if not utils.is_joint_change_safe(current_joints, joint_target, cfg.MOVE_THRESHOLD):
            return False

        if self.torque_mode:
            self.ur.target_pos = joint_target.tolist()
        else:
            self.ur.servo_to_joint_configuration(joint_target, 1 / cfg.UR_CTRL_RATE)

        if len(action) > 6:
            g = float(np.clip(action[6], 0.0, cfg.GRIPPER_MAX))
            self.gripper._set_target_width(g)

        return True

    def reset(self) -> None:
        self.gripper.move(cfg.GRIPPER_MAX)
        if self.torque_mode:
            self.ur.tmp_move(self.initial_joint)
        else:
            self.ur.move_to_joint_configuration(self.initial_joint, 1.0).wait()
        time.sleep(1.0)
        utils.logger.info("Robot reset to initial pose.")

    def cleanup(self) -> None:
        try:
            if self.torque_mode:
                self.ur.disable_torque_control()
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


# ── Observation builder ───────────────────────────────────────────────────

def build_observation(
    state: np.ndarray,
    images: dict[str, np.ndarray],
    device: str,
    force: np.ndarray | None = None,
    tactile: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    """Construct observation dict matching lerobot dataset features (B=1)."""
    obs: dict[str, torch.Tensor] = {
        "observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
    }

    for cam_name, img in images.items():
        t = torch.from_numpy(img.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        obs[f"observation.images.{cam_name}"] = t.to(device)

    if force is not None:
        obs["observation.force"] = torch.from_numpy(force).unsqueeze(0).to(device)

    if tactile is not None:
        obs["observation.tactile"] = (
            torch.from_numpy(np.asarray(tactile, dtype=np.float32))
            .unsqueeze(0).to(device)
        )

    return obs


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
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    force_mode = cfg.FORCE_COLLECT and cfg.TORQUE_MODE
    tactile_mode = cfg.TACTILE_TRANSFER

    utils.logger.info("=== Inference Configuration ===")
    utils.logger.info(f"Task:    {cfg.TASK_NAME}")
    utils.logger.info(f"Policy:  {args.policy}")
    utils.logger.info(f"Device:  {device}")
    utils.logger.info(f"Control: {'torque' if cfg.TORQUE_MODE else 'position'}")
    utils.logger.info(f"Force:   {force_mode}")
    utils.logger.info(f"Tactile: {tactile_mode}")
    utils.logger.info(
        f"FPS: {args.fps}  |  Episodes: {args.episodes}  |  Max steps: {args.max_steps}"
    )

    cameras = InferenceCameraManager()
    robot = InferenceRobotController()

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

    time.sleep(2)
    utils.logger.info("All systems ready.")

    episode = 0
    target_episodes = args.episodes if args.episodes > 0 else float("inf")

    try:
        while episode < target_episodes:
            utils.logger.info(f"--- Episode {episode} ---")
            robot.reset()
            policy.reset()
            time.sleep(1.0)

            for step in range(args.max_steps):
                t0 = time.time()

                state = robot.get_state()
                images = cameras.get_images()
                force = robot.get_force() if force_mode else None
                tactile = (
                    tactile_holder.tactile_data
                    if tactile_mode and tactile_holder is not None
                       and tactile_holder.tactile_data is not None
                    else None
                )

                obs = build_observation(state, images, device, force, tactile)

                with torch.no_grad():
                    action = policy.select_action(obs)

                action_np = action.squeeze(0).cpu().numpy()
                robot.execute_action(action_np)

                elapsed = time.time() - t0
                remaining = 1.0 / args.fps - elapsed
                if remaining > 0:
                    time.sleep(remaining)
                elif step % 50 == 0:
                    utils.logger.warning(
                        f"Step {step}: loop {elapsed:.3f}s > target {1/args.fps:.3f}s"
                    )

            episode += 1
            utils.logger.info(f"Episode {episode - 1} done ({args.max_steps} steps).")

    except KeyboardInterrupt:
        utils.logger.info("Interrupted by user.")

    finally:
        robot.cleanup()
        utils.logger.info("Inference finished.")


if __name__ == "__main__":
    main()
