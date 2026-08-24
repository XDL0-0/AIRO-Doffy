"""Evaluate the trained RealMan-Beaver policies on the RM75 robot.

Loads a DP/FM checkpoint from ``policies/output`` and closes the
loop on hardware: Realsense camera + joint configuration (+ Beaver distance
grids for Beaver-aware variants) in, joint
targets out, commanded at the dataset rate (24 Hz default).

Beaver is initialized once at startup — before any policy is loaded — as soon
as at least one selected policy needs it, and is shared across policies.

A live monitor window (cv2) shows the camera feed, a rolling joint plot, and
the Beaver heatmaps. Each episode is recorded to
``policies/output/eval/<timestamp>/<policy>/episode_<i>/``:
  * ``video.mp4``  — camera feed at the control rate
  * ``log.jsonl``  — per-step joints, tcp pose, actions, and safety flags
  * ``summary.json`` — episode metrics (duration, dropped commands, verdict)

Usage:
    python eval_policy.py
    python eval_policy.py --policy dp_beaver
    python eval_policy.py --policy fm --policy rfm
    python eval_policy.py --checkpoint-root policies/output/WRM_grasp_cylinder_all
    python eval_policy.py --policy original_dp=path/to/step_050000.pt
    python eval_policy.py --episodes 3 --max-steps 400 --no-video

Interactive keys during a run:
    Enter  pause (robot holds its pose)
    while paused:  s = success (recorded, next episode)   f = failure (next)
                   r = restart episode    q = quit
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

import utils
from beaver import BeaverReader
from config import Config
from eval_config import EvalConfig
from inference import InferenceCameraManager
from robot_backend import make_robot_backend

SUPPORTED_POLICY_VARIANTS = frozenset(
    {
        "original_dp",
        "dp_beaver",
        "dp_beaver_enc",
        "dp_beaver_near",
        "dp_beaver_near_gate",
        "rdp_like",
        "fm",
        "fm_beaver",
        "rfm",
    }
)
BEAVER_POLICY_VARIANTS = frozenset(
    {
        "dp_beaver",
        "dp_beaver_enc",
        "dp_beaver_near",
        "dp_beaver_near_gate",
        "rdp_like",
        "fm_beaver",
        "rfm",
    }
)
EXPECTED_CHECKPOINT_KINDS = {
    "original_dp": "original_dp",
    "dp_beaver": "dp_beaver",
    "dp_beaver_enc": "dp_beaver_enc",
    "dp_beaver_near": "dp_beaver_near",
    "dp_beaver_near_gate": "dp_beaver_near_gate",
    "rdp_like": "latent_dp",
    "fm": "fm",
    "fm_beaver": "fm_beaver",
    "rfm": "latent_fm",
}


def policy_needs_beaver(variant: str) -> bool:
    """Return whether a deployment observation must contain Beaver fields."""
    if variant not in SUPPORTED_POLICY_VARIANTS:
        raise ValueError(f"Unsupported policy variant: {variant}")
    return variant in BEAVER_POLICY_VARIANTS


def validate_deployable_checkpoint(summary: dict[str, object]) -> str:
    """Validate checkpoint metadata and return its six-policy variant."""
    kind = str(summary.get("kind", "unknown"))
    if kind == "tokenizer":
        raise ValueError(
            "tokenizer-only checkpoint is not deployable; use the final "
            "reactive-policy last.pt"
        )
    variant = str(summary.get("variant", "unknown"))
    if variant not in SUPPORTED_POLICY_VARIANTS:
        raise ValueError(f"unsupported policy variant '{variant}'")
    expected_kind = EXPECTED_CHECKPOINT_KINDS[variant]
    if kind != expected_kind:
        raise ValueError(
            f"checkpoint kind '{kind}' does not match variant '{variant}' "
            f"(expected '{expected_kind}')"
        )
    return variant


def configure_policy_execution_window(policy, eval_cfg: EvalConfig) -> int:
    """Apply reactive replan overrides and return the executable chunk size."""
    variant = policy.config.model.variant
    if variant == "rdp_like":
        reactive = policy.config.rdp
        override = eval_cfg.RDP_SLOW_REPLAN_STEPS
    elif variant == "rfm":
        reactive = policy.config.rfm
        override = eval_cfg.RFM_SLOW_REPLAN_STEPS
    elif variant in SUPPORTED_POLICY_VARIANTS:
        return int(policy.config.model.n_action_steps)
    else:
        raise ValueError(f"Unsupported policy variant: {variant}")
    if override is not None:
        reactive.slow_replan_steps = override
    return int(reactive.slow_replan_steps)


def select_action_with_latency(
    policy,
    observation: dict[str, torch.Tensor],
    latency_steps: int,
) -> torch.Tensor:
    """Select the time-matched action from a newly predicted action chunk.

    RDP drops the first ``latency_steps`` predictions after a replan because
    those control instants elapsed while inference was running. Repeated
    ``select_action`` calls advance each policy's existing action queue or
    causal decoder without triggering another expensive replan.
    """
    if latency_steps < 0:
        raise ValueError("latency_steps cannot be negative")
    action = policy.select_action(observation)
    replanned = bool(getattr(policy, "last_replanned", False))
    applied_steps = 0
    if replanned:
        for _ in range(latency_steps):
            action = policy.select_action(observation)
            if bool(getattr(policy, "last_replanned", False)):
                raise RuntimeError(
                    "Inference latency exceeds the policy's executable "
                    "action chunk; reduce INFERENCE_LATENCY_STEPS."
                )
            applied_steps += 1
        # Keep logging semantics tied to the control tick: this tick did
        # generate a new chunk even though its stale prefix was discarded.
        policy.last_replanned = True
    policy.last_latency_steps = applied_steps
    return action


# ── Live monitor window ──────────────────────────────────────────────────

class MonitorWindow:
    """Live cv2 monitor: camera feed + rolling joint plot + Beaver heatmaps.

    Gracefully disables itself when no display is available.
    """

    JOINT_COLORS = [
        (66, 133, 244), (219, 68, 55), (244, 180, 0),
        (15, 157, 88), (171, 71, 188), (255, 112, 67), (38, 166, 154),
    ]

    def __init__(
        self,
        dof: int,
        distance_max_mm: float,
        fps: int,
        command_queue: queue.Queue[str] | None = None,
    ) -> None:
        self.dof = int(dof)
        self.distance_max_mm = float(distance_max_mm)
        self.fps = int(fps)
        self.history: deque[np.ndarray] = deque(maxlen=240)
        self._window_name = "Policy Eval Monitor"
        self._enabled: bool | None = None  # None = untested
        # Keys pressed while the monitor window has focus are forwarded here,
        # so Enter/s/f/r/q work without clicking the terminal first.
        self.command_queue = command_queue

    def _check_enabled(self) -> bool:
        if self._enabled is None:
            try:
                cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
                cv2.waitKey(1)
                self._enabled = True
            except cv2.error:
                self._enabled = False
                utils.logger.warning(
                    "No display available; live monitor disabled."
                )
        return self._enabled

    @staticmethod
    def _map_key(key: int) -> str | None:
        if key in (10, 13):  # Enter (LF / CR)
            return ""
        return {
            "s": "s",
            "f": "f",
            "r": "r",
            "q": "q",
            "p": "p",
            "y": "y",
            "n": "n",
        }.get(chr(key & 0xFF).lower())

    def pump_keys(self) -> None:
        """Poll the window's key buffer and forward commands to the queue."""
        if not self._check_enabled():
            return
        try:
            key = cv2.waitKey(1)
        except cv2.error:
            self._enabled = False
            return
        if key < 0 or self.command_queue is None:
            return
        command = self._map_key(key)
        if command is not None:
            self.command_queue.put(command)

    def show(
        self,
        *,
        camera: np.ndarray | None,
        joints: np.ndarray,
        beaver: tuple[np.ndarray, np.ndarray] | None,
        status: dict[str, object],
    ) -> None:
        if not self._check_enabled():
            return
        try:
            self.history.append(np.asarray(joints, dtype=float).copy())
            canvas = self._compose(camera, beaver, status)
            cv2.imshow(self._window_name, canvas)
            self.pump_keys()
        except cv2.error:
            self._enabled = False
            utils.logger.warning(
                "Monitor display failed at runtime; live monitor disabled."
            )

    def close(self) -> None:
        if self._enabled:
            try:
                cv2.destroyWindow(self._window_name)
            except cv2.error:
                pass

    # ── composition ──────────────────────────────────────────────────────

    def _compose(
        self,
        camera: np.ndarray | None,
        beaver: tuple[np.ndarray, np.ndarray] | None,
        status: dict[str, object],
    ) -> np.ndarray:
        # Layout: left camera (640x480), right joint plot (480x480) with
        # Beaver grid (480x240) beneath it.
        # Right column (joint plot + beaver grid) is 720 px tall; match it.
        left = np.zeros((720, 640, 3), np.uint8)
        if camera is not None:
            # Camera frames are RGB; cv2 displays and writes BGR.
            camera = cv2.cvtColor(camera, cv2.COLOR_RGB2BGR)
            left[: camera.shape[0], : camera.shape[1]] = camera
        joint_canvas = self._draw_joint_plot(np.zeros((480, 480, 3), np.uint8))
        beaver_canvas = self._draw_beaver_grid(
            np.zeros((240, 480, 3), np.uint8), beaver
        )
        right = np.vstack([joint_canvas, beaver_canvas])
        canvas = np.hstack([left, right])

        text = (
            f"policy={status.get('policy', '?')} "
            f"ep={status.get('episode', '?')} step={status.get('step', '?')} "
            f"cmd={status.get('commanded', 0)} drop={status.get('dropped', 0)} "
            f"fps={status.get('fps', 0):.1f} "
            f"beaver={'' if status.get('beaver_ok') else 'STALE'}"
        )
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(
            canvas, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (200, 200, 200), 1, cv2.LINE_AA,
        )
        return canvas

    def _draw_joint_plot(self, canvas: np.ndarray) -> np.ndarray:
        margin_l, margin_r, margin_b, margin_t = 42, 10, 26, 26
        plot_w = canvas.shape[1] - margin_l - margin_r
        plot_h = canvas.shape[0] - margin_t - margin_b
        # Frame.
        cv2.rectangle(
            canvas,
            (margin_l, margin_t),
            (margin_l + plot_w, margin_t + plot_h),
            (60, 60, 60), 1,
        )
        cv2.putText(
            canvas, "joints [rad]", (margin_l, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA,
        )
        if not self.history:
            return canvas
        data = np.stack(self.history)  # (T, dof)
        lo, hi = float(data.min()), float(data.max())
        pad = max(0.3, (hi - lo) * 0.15)
        lo, hi = lo - pad, hi + pad
        if hi - lo < 1e-6:
            hi = lo + 1.0
        n = data.shape[0]
        # Gridlines.
        for k in range(1, 5):
            y = margin_t + int(plot_h * k / 5)
            cv2.line(
                canvas, (margin_l, y),
                (margin_l + plot_w, y), (40, 40, 40), 1,
            )
        for j in range(self.dof):
            color = self.JOINT_COLORS[j % len(self.JOINT_COLORS)]
            points = []
            for i in range(n):
                x = margin_l + int(plot_w * i / max(1, n - 1))
                y = margin_t + int(
                    plot_h * (1.0 - (data[i, j] - lo) / (hi - lo))
                )
                points.append((x, y))
            if len(points) > 1:
                cv2.polylines(canvas, [np.array(points)], False, color, 1, cv2.LINE_AA)
        latest = data[-1]
        for j in range(self.dof):
            color = self.JOINT_COLORS[j % len(self.JOINT_COLORS)]
            lx = margin_l + int(plot_w * (j + 0.5) / self.dof)
            label = f"J{j + 1} {latest[j]:+.2f}"
            (tw, _), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1
            )
            cv2.putText(
                canvas, label,
                (max(margin_l, min(margin_l + plot_w - tw, lx - tw // 2)),
                 margin_t + plot_h + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA,
            )
        return canvas

    def _draw_beaver_grid(
        self,
        canvas: np.ndarray,
        beaver: tuple[np.ndarray, np.ndarray] | None,
    ) -> np.ndarray:
        cv2.putText(
            canvas, "beaver distance [mm]  (grey = sensor absent)",
            (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (160, 160, 160), 1, cv2.LINE_AA,
        )
        if beaver is None:
            return canvas
        distance, present = beaver
        n_sensors, grid, _ = distance.shape
        if n_sensors == 0:
            return canvas
        cell = 14
        tile = grid * cell
        gap = 6
        cols = 3
        rows = int(np.ceil(n_sensors / cols))
        x0 = (canvas.shape[1] - (cols * tile + (cols - 1) * gap)) // 2
        y0 = (canvas.shape[0] - (rows * tile + (rows - 1) * gap)) // 2 + 10
        norm = max(self.distance_max_mm, 1.0)
        for s in range(n_sensors):
            r, c = divmod(s, cols)
            ox = x0 + c * (tile + gap)
            oy = y0 + r * (tile + gap)
            present_s = bool(present[s]) if s < len(present) else False
            heat = np.zeros((grid, grid), np.uint8)
            if present_s:
                values = np.clip(distance[s] / norm * 255.0, 0, 255).astype(np.uint8)
                heat = cv2.applyColorMap(values, cv2.COLORMAP_VIRIDIS)
            else:
                heat = np.full((grid, grid, 3), 70, np.uint8)
            heat = cv2.resize(heat, (tile, tile), interpolation=cv2.INTER_NEAREST)
            canvas[oy:oy + tile, ox:ox + tile] = heat
            border = (80, 220, 80) if present_s else (70, 70, 200)
            cv2.rectangle(
                canvas, (ox, oy), (ox + tile - 1, oy + tile - 1), border, 1,
            )
            cv2.putText(
                canvas, str(s), (ox + 2, oy + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA,
            )
        return canvas


class _StdinListener:
    """Background stdin reader; each line becomes one command."""

    def __init__(self) -> None:
        self._commands: queue.Queue[str] = queue.Queue()

    def start(self) -> None:
        threading.Thread(target=self._listen, daemon=True).start()

    def _listen(self) -> None:
        for line in sys.stdin:
            self._commands.put(line.strip().lower())

    def poll(self) -> str | None:
        try:
            return self._commands.get_nowait()
        except queue.Empty:
            return None

    def wait(self) -> str:
        return self._commands.get()


class PolicyEvaluator:
    """Hardware-in-the-loop evaluation of one trained policy."""

    def __init__(
        self,
        name: str,
        checkpoint: str,
        eval_cfg: EvalConfig,
        hw_cfg: Config,
        session_dir: Path,
        beaver: BeaverReader | None = None,
        monitor: MonitorWindow | None = None,
        stdin: _StdinListener | None = None,
    ) -> None:
        self.name = name
        self.checkpoint = str(Path(checkpoint).resolve())
        self.cfg = eval_cfg
        self.hw = hw_cfg
        self.device = eval_cfg.DEVICE
        self.session_dir = session_dir

        self.backend = make_robot_backend(hw_cfg)
        self.dof = self.backend.dof
        if self.dof != 7:
            utils.logger.warning(
                f"Policy was trained for 7-DoF joints, robot reports {self.dof} DoF."
            )
        self.initial_joint = self.backend.initial_joint_configuration(
            hw_cfg.INITIAL_JOINT
        )

        # ── Policy ───────────────────────────────────────────────────────
        from policies.realman_beaver.checkpoint import load_policy

        utils.logger.info(f"Loading policy '{name}' from {self.checkpoint}")
        self.policy = load_policy(
            self.checkpoint, device=self.device, use_ema=eval_cfg.USE_EMA
        )
        self.variant = self.policy.config.model.variant
        reactive = None
        if self.variant == "rdp_like":
            reactive = self.policy.config.rdp
        elif self.variant == "rfm":
            reactive = self.policy.config.rfm
        old_replan_steps = reactive.slow_replan_steps if reactive else None
        execution_window = configure_policy_execution_window(self.policy, eval_cfg)
        if reactive is not None and old_replan_steps != reactive.slow_replan_steps:
            utils.logger.info(
                f"Policy '{name}': {self.variant} slow_replan_steps "
                f"{old_replan_steps} -> {reactive.slow_replan_steps}"
            )
        if eval_cfg.INFERENCE_LATENCY_STEPS >= execution_window:
            raise ValueError(
                "INFERENCE_LATENCY_STEPS must be smaller than the executable "
                f"action window ({execution_window}), got "
                f"{eval_cfg.INFERENCE_LATENCY_STEPS}."
            )
        if self.policy.config.model.action_dim != self.dof:
            raise ValueError(
                f"Policy action_dim={self.policy.config.model.action_dim} "
                f"does not match robot DoF={self.dof}."
            )
        self.needs_beaver = policy_needs_beaver(self.variant)
        utils.logger.info(
            f"Policy '{name}': variant={self.variant}, "
            f"needs_beaver={self.needs_beaver}, "
            f"inference_latency_steps={eval_cfg.INFERENCE_LATENCY_STEPS}"
        )

        # ── Sensors ──────────────────────────────────────────────────────
        self.cameras = InferenceCameraManager()
        if not self.cameras.num_cameras:
            raise RuntimeError("No Realsense cameras detected — cannot evaluate.")
        self.camera_name = (
            "camera_0"
            if "camera_0" in self.cameras.cameras
            else sorted(self.cameras.cameras)[0]
        )
        utils.logger.info(f"Using camera '{self.camera_name}' for observations.")

        # Beaver is normally created once in main() before any policy loads
        # and shared across evaluators; we only own it if none was passed in.
        self.beaver = beaver
        self._owns_beaver = False
        if self.needs_beaver and self.beaver is None:
            self.beaver = BeaverReader.from_config(hw_cfg)
            self.beaver.start()
            self._owns_beaver = True
            utils.logger.info(
                "Beaver reader created inside evaluator (not shared)."
            )
        elif self.beaver is not None:
            utils.logger.info("Using shared Beaver reader.")

        # ── Safety ───────────────────────────────────────────────────────
        self.move_threshold = np.resize(eval_cfg.MOVE_THRESHOLD, self.dof)
        self.max_joint_delta = eval_cfg.MAX_JOINT_DELTA
        self._hard_clamp = np.deg2rad(300.0)  # final per-joint guard
        utils.logger.info(
            f"Safety: move_threshold={self.move_threshold.round(2).tolist()} "
            f"max_delta={self.max_joint_delta}"
        )

        self.monitor = monitor
        # One shared stdin listener for the whole run: starting a second
        # thread that also reads sys.stdin would race the first one for
        # every line, and Enter could be swallowed by the previous
        # evaluator's (still alive) listener instead of ours.
        self._stdin = stdin
        if self._stdin is None:
            self._stdin = _StdinListener()
            self._stdin.start()
        if self.monitor is not None:
            # Window key presses (focus may be on the monitor, not the
            # terminal) are forwarded into the same command stream.
            self.monitor.command_queue = self._stdin._commands
        self._last_video_frame: dict[str, np.ndarray] = {}
        self._last_beaver_np: tuple[np.ndarray, np.ndarray] | None = None
        self._last_beaver_stale = False
        self._quit_requested = False
        self._last_tcp_quat: np.ndarray | None = None

    # ── Observation helpers ──────────────────────────────────────────────

    def _joints(self) -> np.ndarray:
        return np.asarray(
            self.backend.get_joint_configuration(), dtype=np.float64
        )

    def _tcp_pose(self) -> list[float] | None:
        """TCP pose as [qx, qy, qz, qw, x, y, z] for the log."""
        try:
            from airo_spatial_algebra.se3 import SE3Container

            tcp = self.backend.get_tcp_pose()
            se3 = SE3Container.from_homogeneous_matrix(tcp)
            self._last_tcp_quat = utils.quat_cal(
                se3.rotation_matrix, self._last_tcp_quat
            )
            return np.concatenate(
                [self._last_tcp_quat, se3.translation]
            ).round(4).tolist()
        except Exception as exc:
            utils.logger.warning(f"TCP pose read failed: {exc}")
            return None

    def _image_tensor(self) -> torch.Tensor:
        self._last_video_frame = self.cameras.get_images()
        image = self._last_video_frame[self.camera_name]
        # HWC uint8 RGB -> CHW float32 in [0, 1], matching the dataset.
        chw = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
        return (
            torch.from_numpy(chw).float().div_(255.0).unsqueeze(0).to(self.device)
        )

    def _beaver_obs(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """Return (distance_mm [1,9,4,4], present [1,9], status [1,9,4,4], stale)."""
        if self.beaver is None:
            return (
                torch.zeros(1, 9, 4, 4, dtype=torch.float32, device=self.device),
                torch.zeros(1, 9, dtype=torch.float32, device=self.device),
                torch.zeros(1, 9, 4, 4, dtype=torch.float32, device=self.device),
                True,
            )
        snapshot = self.beaver.snapshot()
        stale = (
            snapshot.timestamp_ns == 0
            or not snapshot.connected
            or (time.monotonic_ns() - snapshot.timestamp_ns) / 1e9
            > self.cfg.STALE_AFTER_S
        )
        # Shape/layout matches training: (n_sensors, grid, grid) + flags.
        distance = torch.from_numpy(
            np.asarray(snapshot.distance_mm, dtype=np.float32)
        ).unsqueeze(0)
        present = torch.from_numpy(
            np.asarray(snapshot.present, dtype=np.float32)
        ).unsqueeze(0)
        # VL53L7CX per-pixel target status; feeds the pixel-level mask that
        # zeroes no-target (255) pixels during normalization.
        status = torch.from_numpy(
            np.asarray(snapshot.target_status, dtype=np.float32)
        ).unsqueeze(0)
        return (
            distance.to(self.device),
            present.to(self.device),
            status.to(self.device),
            stale,
        )

    def _select_action(self, joints: np.ndarray) -> np.ndarray:
        obs = {
            "image": self._image_tensor(),
            "state": torch.as_tensor(
                joints, dtype=torch.float32, device=self.device
            ).unsqueeze(0),
        }
        if self.needs_beaver:
            distance, present, status, stale = self._beaver_obs()
            obs["beaver_distance"] = distance
            obs["beaver_present"] = present
            obs["beaver_status"] = status
            self._last_beaver_stale = stale
            self._last_beaver_np = (
                distance.detach().cpu().numpy()[0],
                present.detach().cpu().numpy()[0],
            )
        with torch.inference_mode():
            action = select_action_with_latency(
                self.policy,
                obs,
                self.cfg.INFERENCE_LATENCY_STEPS,
            )
        return action.squeeze(0).cpu().numpy()

    # ── Commanding ───────────────────────────────────────────────────────

    def _command(self, action: np.ndarray, dt: float) -> tuple[bool, str]:
        """Send a joint target with jump detection and a per-tick speed cap.

        Returns (commanded, reason) where reason describes a drop.
        """
        target = np.asarray(action, dtype=np.float64)[: self.dof]
        if not np.all(np.isfinite(target)):
            return False, "non-finite"
        current = self._joints()
        delta = target - current
        if np.any(np.abs(delta) > self.move_threshold):
            return False, "jump"
        if self.max_joint_delta is not None:
            delta = np.clip(delta, -self.max_joint_delta, self.max_joint_delta)
        delta = np.clip(delta, -self._hard_clamp, self._hard_clamp)
        self.backend.command_joint_configuration(current + delta, dt)
        return True, ""

    # ── Robot motion helpers ─────────────────────────────────────────────

    def _move_home(self) -> None:
        current = self._joints()
        if np.max(np.abs(current - self.initial_joint)) < 0.01:
            utils.logger.info("Already at initial joint configuration.")
            return
        utils.logger.info("Moving to initial joint configuration...")
        self.backend.reset(self.initial_joint)
        time.sleep(1.0)

    def _wait_settled(self, timeout_s: float = 15.0) -> None:
        deadline = time.monotonic() + timeout_s
        prev = self._joints()
        while time.monotonic() < deadline:
            time.sleep(0.2)
            current = self._joints()
            if np.max(np.abs(current - prev)) < 1e-3:
                utils.logger.info("Robot settled.")
                return
            prev = current
        utils.logger.warning("Robot did not settle in %.1f s; continuing.", timeout_s)

    # ── Episode ──────────────────────────────────────────────────────────

    def run_episode(self, out_dir: Path, index: int) -> dict[str, object]:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.policy.reset()
        # Home + settle now happen in evaluate() before the Enter prompt.

        log_file = open(out_dir / "log.jsonl", "w")
        video_writer: cv2.VideoWriter | None = None
        if self.cfg.SAVE_VIDEO:
            video_writer = self._open_video_writer(out_dir / "video.mp4")

        episode_start = time.monotonic()
        dt = 1.0 / self.cfg.FPS
        stats: dict[str, object] = {
            "policy": self.name,
            "variant": self.variant,
            "checkpoint": self.checkpoint,
            "episode": index,
            "steps": 0,
            "commanded": 0,
            "dropped": 0,
            "drops": {},
            "beaver_stale_steps": 0,
            "inference_latency_steps": self.cfg.INFERENCE_LATENCY_STEPS,
            "action_delta_max": 0.0,
            "success": None,
        }
        delta_magnitudes: list[float] = []
        restart = False

        utils.logger.info(f"--- Episode {index} ({self.name}) ---")
        try:
            for step in range(self.cfg.MAX_STEPS):
                paused, quit_requested, restart, verdict = self._handle_commands()
                if quit_requested:
                    raise KeyboardInterrupt
                if restart:
                    break
                if verdict is not None:
                    stats["success"] = verdict
                    break

                t0 = time.monotonic()
                joints = self._joints()
                self._last_beaver_stale = False
                action = self._select_action(joints)
                commanded, reason = self._command(action, dt)
                stats["steps"] = int(stats["steps"]) + 1
                if commanded:
                    stats["commanded"] = int(stats["commanded"]) + 1
                else:
                    stats["dropped"] = int(stats["dropped"]) + 1
                    drops = stats["drops"]
                    drops[reason] = drops.get(reason, 0) + 1
                if self._last_beaver_stale:
                    stats["beaver_stale_steps"] = int(
                        stats["beaver_stale_steps"]
                    ) + 1

                delta = np.abs(action[: self.dof] - joints)
                delta_magnitudes.append(float(np.max(delta)))
                stats["action_delta_max"] = max(
                    float(stats["action_delta_max"]), float(np.max(delta))
                )

                if self.cfg.SAVE_LOG:
                    record = {
                        "step": step,
                        "t": round(time.monotonic() - episode_start, 4),
                        "joints": joints.round(4).tolist(),
                        "tcp_pose": self._tcp_pose(),
                        "action": action[: self.dof].round(4).tolist(),
                        # Chunk bookkeeping: replan=True marks the tick where
                        # a new action chunk was generated; chunk_step is the
                        # position within that chunk (0 = first executed).
                        "replan": bool(
                            getattr(self.policy, "last_replanned", False)
                        ),
                        "chunk_step": int(
                            getattr(self.policy, "last_chunk_step", 0)
                        ),
                        "latency_steps": int(
                            getattr(self.policy, "last_latency_steps", 0)
                        ),
                        "commanded": commanded,
                        "drop_reason": reason or None,
                        "beaver_stale": bool(self._last_beaver_stale),
                    }
                    log_file.write(json.dumps(record) + "\n")
                    log_file.flush()

                if video_writer is not None and self._last_video_frame:
                    # Camera frames are RGB; VideoWriter expects BGR.
                    frame = self._last_video_frame[self.camera_name]
                    video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                if self.monitor is not None:
                    self.monitor.show(
                        camera=self._last_video_frame.get(self.camera_name),
                        joints=joints,
                        beaver=self._last_beaver_np,
                        status={
                            "policy": self.name,
                            "episode": index,
                            "step": step,
                            "commanded": int(stats["commanded"]),
                            "dropped": int(stats["dropped"]),
                            "fps": self.cfg.FPS,
                            "beaver_ok": not self._last_beaver_stale,
                        },
                    )

                elapsed = time.monotonic() - t0
                remaining = dt - elapsed
                if remaining > 0:
                    time.sleep(remaining)
                elif step > 0 and step % 50 == 0:
                    utils.logger.warning(
                        f"Step {step}: loop {elapsed:.3f}s > target {dt:.3f}s"
                    )
        finally:
            if video_writer is not None:
                video_writer.release()
            log_file.close()

        if not restart:
            stats["duration_s"] = round(time.monotonic() - episode_start, 2)
            if delta_magnitudes:
                stats["action_delta_mean"] = round(
                    float(np.mean(delta_magnitudes)), 4
                )
            stats["beaver_stale_fraction"] = round(
                int(stats["beaver_stale_steps"]) / max(1, int(stats["steps"])), 4
            )
            if stats["success"] is None and self.cfg.ASK_SUCCESS:
                # Read through the same stdin queue as the key listener —
                # input() would race the listener thread for the same line.
                utils.logger.info(
                    f"Episode {index} ({self.name}) finished. "
                    "Success? (y/n, Enter=skip, q=quit)"
                )
                verdict = self._wait_command()
                stats["success"] = {
                    "y": True,
                    "yes": True,
                    "n": False,
                    "no": False,
                }.get(verdict)
                if verdict == "q":
                    self._quit_requested = True
        else:
            stats["restarted"] = True

        self._write_summary(out_dir, stats)
        utils.logger.info(
            f"Episode {index} done: {stats['steps']} steps, "
            f"{stats['dropped']} dropped, success={stats['success']}."
        )
        return stats

    def _handle_commands(self) -> tuple[bool, bool, bool, bool | None]:
        """Return (paused, quit_requested, restart_requested, verdict).

        Verdict is True (success) or False (failure) when the user marked the
        episode from the paused state with `s`/`f`.
        """
        paused = False
        while True:
            command = self._stdin.poll()
            if command is None:
                break
            if command == "q":
                return False, True, False, None
            if command == "r":
                return False, False, True, None
            if command in {"", "p", "pause"}:
                paused = True
        if paused:
            utils.logger.info(
                "Paused (robot holds its pose). "
                "Enter=resume, s=success(next episode), "
                "f=fail(next episode), r=restart, q=quit."
            )
            while True:
                resume = self._wait_command()
                if resume in {"", "p", "pause"}:
                    utils.logger.info("Resuming.")
                    return False, False, False, None
                if resume == "q":
                    return False, True, False, None
                if resume == "r":
                    return False, False, True, None
                if resume in {"s", "success"}:
                    utils.logger.info("Marked SUCCESS.")
                    return False, False, False, True
                if resume in {"f", "fail", "failure"}:
                    utils.logger.info("Marked FAILURE.")
                    return False, False, False, False
        return False, False, False, None

    # ── Recording helpers ────────────────────────────────────────────────

    def _open_video_writer(self, path: Path) -> cv2.VideoWriter | None:
        width, height = self.hw.REALSENSE_RESOLUTION
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"X264"),
            self.cfg.FPS,
            (int(width), int(height)),
        )
        if not writer.isOpened():
            utils.logger.warning(
                "Could not open video writer (codec?); continuing without video."
            )
            return None
        utils.logger.info(f"Recording video to {path}")
        return writer

    def _write_summary(self, out_dir: Path, stats: dict[str, object]) -> None:
        summary = dict(stats)
        summary["max_joint_delta"] = self.max_joint_delta
        summary["move_threshold"] = self.move_threshold.round(2).tolist()
        summary_path = out_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str))
        utils.logger.info(f"Episode summary written to {summary_path}")

    # ── Session ──────────────────────────────────────────────────────────

    def _wait_command(self) -> str:
        """Block until a command arrives from stdin or the monitor window.

        Used wherever the program pauses for the user (before an episode,
        while paused mid-run) so that keys pressed while the monitor window
        has focus are handled as well as terminal keys.
        """
        while True:
            command = self._stdin.poll()
            if command is not None:
                return command
            if self.monitor is not None:
                self.monitor.pump_keys()
            time.sleep(0.05)

    def _announce_wait(self, message: str) -> None:
        """Make the wait-for-Enter pause unmistakable — it is not a hang."""
        utils.logger.warning(
            "============================================================\n"
            f"{message} to start.\n"
            "The program is paused for you now (not a hang).\n"
            "Keys while running:  ENTER=pause  s=success  f=failure\n"
            "                     r=restart episode  q=quit\n"
            "============================================================"
        )

    def evaluate(self, wait_first: bool) -> list[dict[str, object]]:
        episodes: list[dict[str, object]] = []
        target = self.cfg.EPISODES if self.cfg.EPISODES > 0 else float("inf")
        index = 0
        try:
            while index < target:
                # Go back to the initial pose first, then wait for the user
                # to confirm the scene before the episode starts.
                self._move_home()
                self._wait_settled()
                if index > 0:
                    self._announce_wait(
                        f"Robot is back at the initial pose. Reset the scene "
                        f"for episode {index}, then press ENTER"
                    )
                elif wait_first:
                    self._announce_wait(
                        "Robot is at the initial pose. Position the scene, "
                        "then press ENTER"
                    )
                if index > 0 or wait_first:
                    if self._wait_command() == "q":
                        break
                out_dir = self.session_dir / self.name / f"episode_{index:03d}"
                episodes.append(self.run_episode(out_dir, index))
                index += 1
                if self._quit_requested:
                    utils.logger.info("Quit requested after episode verdict.")
                    break
        except KeyboardInterrupt:
            # One Ctrl-C aborts the whole run: flag it so main() stops
            # after this policy instead of moving on to the next one
            # (which would just sit at its own wait prompt again).
            utils.logger.info("Evaluation interrupted by user.")
            self._quit_requested = True
        self._cleanup()
        return episodes

    def _cleanup(self) -> None:
        if self._owns_beaver and self.beaver is not None:
            try:
                self.beaver.close()
            except Exception as exc:  # pragma: no cover - hardware teardown
                utils.logger.error(f"Beaver close error: {exc}")
        try:
            self.backend.cleanup()
        except Exception as exc:  # pragma: no cover - hardware teardown
            utils.logger.error(f"Robot cleanup error: {exc}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained RealMan-Beaver policies on the RM75 robot."
    )
    parser.add_argument(
        "--policy",
        action="append",
        default=[],
        metavar="NAME[=CHECKPOINT]",
        help="Policy to evaluate; repeatable. NAME must be a key of "
        "EvalConfig.POLICIES; CHECKPOINT overrides its path "
        "(e.g. --policy dp_beaver=…/dp_beaver_step_100000.pt). "
        "Default: all policies in EvalConfig.",
    )
    parser.add_argument("--device", default=None, help="cuda:0 / cpu")
    parser.add_argument(
        "--checkpoint-root",
        default=None,
        help="Use ROOT/<policy>/last.pt for all default policy paths",
    )
    parser.add_argument("--fps", type=int, default=None, help="Control rate in Hz")
    parser.add_argument(
        "--latency-steps",
        type=int,
        default=None,
        help="Discard this many predicted actions after each replan (RDP: 4 at 24 Hz)",
    )
    parser.add_argument(
        "--episodes", type=int, default=None, help="Episodes per policy"
    )
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Max control steps per episode"
    )
    parser.add_argument("--output-dir", default=None, help="Run output root")
    parser.add_argument(
        "--no-video", action="store_true", help="Do not record camera video"
    )
    parser.add_argument(
        "--no-log", action="store_true", help="Do not write per-step JSONL logs"
    )
    parser.add_argument(
        "--no-monitor", action="store_true", help="Do not show the live monitor"
    )
    parser.add_argument(
        "--no-ask-success",
        action="store_true",
        help="Do not prompt for a success verdict after each episode",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Start episode 0 without waiting for Enter",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    eval_cfg = EvalConfig()
    hw_cfg = Config()

    if args.checkpoint_root:
        checkpoint_root = Path(args.checkpoint_root).expanduser()
        eval_cfg.POLICIES = {
            name: str(checkpoint_root / name / "last.pt")
            for name in eval_cfg.POLICIES
        }
    if args.device:
        eval_cfg.DEVICE = args.device
    if args.fps:
        eval_cfg.FPS = args.fps
        # Keep the per-tick speed cap consistent with the actual control rate.
        eval_cfg.MAX_JOINT_DELTA = hw_cfg.REALMAN_MAX_JOINT_SPEED / args.fps
    if args.latency_steps is not None:
        if args.latency_steps < 0:
            raise SystemExit("--latency-steps cannot be negative")
        eval_cfg.INFERENCE_LATENCY_STEPS = args.latency_steps
    if args.episodes is not None:
        eval_cfg.EPISODES = args.episodes
    if args.max_steps:
        eval_cfg.MAX_STEPS = args.max_steps
    if args.output_dir:
        eval_cfg.OUTPUT_DIR = args.output_dir
    if args.no_video:
        eval_cfg.SAVE_VIDEO = False
    if args.no_log:
        eval_cfg.SAVE_LOG = False
    if args.no_monitor:
        eval_cfg.MONITOR = False
    if args.no_ask_success:
        eval_cfg.ASK_SUCCESS = False
    if not torch.cuda.is_available() and eval_cfg.DEVICE.startswith("cuda"):
        utils.logger.warning("CUDA unavailable; falling back to CPU.")
        eval_cfg.DEVICE = "cpu"

    # Resolve the policies to evaluate.
    selections: dict[str, str] = {}
    if args.policy:
        for item in args.policy:
            name, sep, checkpoint = item.partition("=")
            if name not in eval_cfg.POLICIES:
                raise SystemExit(
                    f"Unknown policy '{name}'. "
                    f"Available: {list(eval_cfg.POLICIES)}"
                )
            selections[name] = checkpoint if sep else eval_cfg.POLICIES[name]
    else:
        selections = dict(eval_cfg.POLICIES)
    for name, checkpoint in selections.items():
        if not Path(checkpoint).exists():
            raise SystemExit(f"Checkpoint for '{name}' not found: {checkpoint}")

    # Pre-detect variants from the checkpoint configs (fast, no model build)
    # so Beaver can be started before anything else when any policy needs it.
    from policies.realman_beaver.checkpoint import checkpoint_summary

    variants: dict[str, str] = {}
    for name, checkpoint in selections.items():
        summary = checkpoint_summary(checkpoint)
        try:
            variant = validate_deployable_checkpoint(summary)
        except ValueError as exc:
            raise SystemExit(f"'{name}': {exc} in {checkpoint}.") from exc
        variants[name] = variant

    needs_beaver = any(policy_needs_beaver(variant) for variant in variants.values())
    shared_beaver: BeaverReader | None = None
    if needs_beaver:
        utils.logger.info(
            "Starting shared Beaver reader (used by: "
            f"{', '.join(n for n, v in variants.items() if policy_needs_beaver(v))})."
        )
        shared_beaver = BeaverReader.from_config(hw_cfg)
        shared_beaver.start()

    run_dir = Path(eval_cfg.OUTPUT_DIR) / time.strftime("%Y%m%dT%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    utils.logger.info("=== Policy Evaluation ===")
    utils.logger.info(f"Robot:   {hw_cfg.ROBOT_TYPE} ({hw_cfg.ROBOT_IP})")
    utils.logger.info(f"Device:  {eval_cfg.DEVICE}")
    utils.logger.info(
        f"FPS: {eval_cfg.FPS}  | Episodes: {eval_cfg.EPISODES}  | "
        f"Max steps: {eval_cfg.MAX_STEPS}  | "
        f"Latency steps: {eval_cfg.INFERENCE_LATENCY_STEPS}"
    )
    utils.logger.info(f"Output:  {run_dir}")

    monitor = (
        MonitorWindow(dof=7, distance_max_mm=2550.0, fps=eval_cfg.FPS)
        if eval_cfg.MONITOR
        else None
    )
    # Single stdin listener shared by every policy; see PolicyEvaluator.
    shared_stdin = _StdinListener()
    shared_stdin.start()
    results: list[dict[str, object]] = []
    try:
        for name, checkpoint in selections.items():
            evaluator = PolicyEvaluator(
                name,
                checkpoint,
                eval_cfg,
                hw_cfg,
                run_dir,
                beaver=shared_beaver,
                monitor=monitor,
                stdin=shared_stdin,
            )
            episodes = evaluator.evaluate(wait_first=not args.no_prompt)
            results.append(
                {
                    "policy": name,
                    "variant": evaluator.variant,
                    "checkpoint": str(checkpoint),
                    "episodes": episodes,
                }
            )
            if evaluator._quit_requested:
                utils.logger.info(
                    "Quit requested; skipping remaining policies."
                )
                break
    except KeyboardInterrupt:
        # Ctrl-C during policy/robot/camera setup: abort the whole run.
        utils.logger.info("Evaluation interrupted by user.")
    finally:
        if monitor is not None:
            monitor.close()
        if shared_beaver is not None:
            try:
                shared_beaver.close()
            except Exception as exc:  # pragma: no cover - hardware teardown
                utils.logger.error(f"Beaver close error: {exc}")

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2, default=str))
    utils.logger.info(f"Evaluation finished. Results: {summary_path}")


if __name__ == "__main__":
    main()
