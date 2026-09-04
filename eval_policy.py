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
    python eval_policy.py --policy path/to/step_050000.pt
    python eval_policy.py --policy WRM_wrap --wrap-near-mm 10 --wrap-lift-min 0.8 \
        --wrap-stop-close-j3 1.0 --wrap-stop-close-j4 1.0 \
        --wrap-contact-stop-mm 5
    python eval_policy.py --episodes 3 --max-steps 400 --no-video

Interactive keys during a run:
    Enter  pause (robot enters freedrive — move the arm by hand)
    while paused:  s = success (recorded, next episode)   f = failure (next)
                   r = restart episode    q = quit
                   Enter resumes from the current joint configuration
"""

from __future__ import annotations

import argparse
import json
import queue
import re
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

import utils
from beaver import VALID_STATUSES, BeaverReader
from config import Config
from eval_config import EvalConfig
from inference import InferenceCameraManager
from robot_backend import make_robot_backend

from policies.realman_beaver.eval_registry import (
    BEAVER_POLICY_VARIANTS,
    EXPECTED_CHECKPOINT_KINDS,
    SUPPORTED_POLICY_VARIANTS,
    policy_needs_beaver,
    validate_deployable_checkpoint,
)


def policy_step_window(policy) -> tuple[int, int]:
    """Return the configured (predicted steps, executed steps) pair."""
    variant = policy.config.model.variant
    if variant == "rdp_like":
        return (
            int(policy.config.rdp.action_horizon),
            int(policy.config.rdp.slow_replan_steps),
        )
    elif variant == "rfm":
        return (
            int(policy.config.rfm.action_horizon),
            int(policy.config.rfm.slow_replan_steps),
        )
    elif variant in SUPPORTED_POLICY_VARIANTS:
        return (
            int(policy.config.model.horizon),
            int(policy.config.model.n_action_steps),
        )
    raise ValueError(f"Unsupported policy variant: {variant}")


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
        (66, 133, 244),
        (219, 68, 55),
        (244, 180, 0),
        (15, 157, 88),
        (171, 71, 188),
        (255, 112, 67),
        (38, 166, 154),
    ]
    CANVAS_WIDTH = 1120
    CANVAS_HEIGHT = 720

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
                # WINDOW_GUI_NORMAL avoids Qt's zoom/pan toolbar and its
                # occasionally stale viewport. Explicitly size the image
                # area: WINDOW_NORMAL otherwise opens at a small backend-
                # dependent default size (typically 400x300).
                flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
                flags |= getattr(cv2, "WINDOW_GUI_NORMAL", 0)
                cv2.namedWindow(self._window_name, flags)
                placeholder = self._compose(
                    None,
                    None,
                    {"state": "STARTING", "beaver_ok": None},
                )
                cv2.imshow(self._window_name, placeholder)
                cv2.resizeWindow(
                    self._window_name, self.CANVAS_WIDTH, self.CANVAS_HEIGHT
                )
                cv2.waitKey(1)
                self._enabled = True
            except cv2.error:
                self._enabled = False
                utils.logger.warning("No display available; live monitor disabled.")
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
        beaver: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
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
        beaver: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
        status: dict[str, object],
    ) -> np.ndarray:
        # Layout: left camera (640x480), right joint plot (480x480) with
        # Beaver grid (480x240) beneath it.
        # Right column (joint plot + beaver grid) is 720 px tall; match it.
        left = np.zeros((720, 640, 3), np.uint8)
        if camera is not None:
            # Camera frames are RGB; cv2 displays and writes BGR.
            camera = cv2.cvtColor(camera, cv2.COLOR_RGB2BGR)
            # Fit arbitrary camera resolutions into the 640x480 panel. A
            # direct slice assignment fails (and disables the monitor) when
            # a camera is configured above 640x480.
            src_h, src_w = camera.shape[:2]
            scale = min(640 / src_w, 480 / src_h)
            dst_w = max(1, round(src_w * scale))
            dst_h = max(1, round(src_h * scale))
            camera = cv2.resize(camera, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
            x0 = (640 - dst_w) // 2
            y0 = (480 - dst_h) // 2
            left[y0 : y0 + dst_h, x0 : x0 + dst_w] = camera
        joint_canvas = self._draw_joint_plot(np.zeros((480, 480, 3), np.uint8))
        beaver_canvas = self._draw_beaver_grid(
            np.zeros((240, 480, 3), np.uint8), beaver
        )
        self._draw_gate_status(beaver_canvas, status)
        right = np.vstack([joint_canvas, beaver_canvas])
        canvas = np.hstack([left, right])

        beaver_ok = status.get("beaver_ok")
        beaver_text = "OFF" if beaver_ok is None else ("OK" if beaver_ok else "STALE")
        text = (
            f"{status.get('state', 'RUNNING')}  "
            f"policy={status.get('policy', '?')} "
            f"ep={status.get('episode', '?')} step={status.get('step', '?')} "
            f"cmd={status.get('commanded', 0)} drop={status.get('dropped', 0)} "
            f"fps={status.get('fps', 0):.1f} "
            f"beaver={beaver_text}"
        )
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(
            canvas,
            text,
            (6, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        return canvas

    @staticmethod
    def _gate_value(status: dict[str, object], name: str) -> bool | None:
        gate = status.get("gate")
        values = gate if isinstance(gate, dict) else status
        value = values.get(name)
        if value is None:
            return None
        return bool(value)

    def _draw_gate_status(
        self, canvas: np.ndarray, status: dict[str, object]
    ) -> np.ndarray:
        """Draw the current inference gate states below the Beaver tiles."""
        gate = status.get("gate")
        if not isinstance(gate, dict):
            return canvas
        if "j3_stop" in gate or "j4_stop" in gate:
            # Parameterized per-joint gate: show the two actual closure
            # decisions and the shared lift decision. There is no separate
            # overall contact-stop decision in this mode.
            labels = (
                ("J3_STOP", "j3_stop"),
                ("J4_STOP", "j4_stop"),
                ("LIFT_ENABLE", "lift_enabled"),
            )
        elif "contact_stop" in gate:
            # Ordinary/monitor gate: contact is one overall decision, so do
            # not display synthetic J3/J4 values.
            labels = (
                ("CONTACT_STOP", "contact_stop"),
                ("LIFT_ENABLE", "lift_enabled"),
            )
        else:
            return canvas
        x = 8
        y = canvas.shape[0] - 7
        for label, key in labels:
            value = self._gate_value(status, key)
            if value is None:
                text = f"{label}=--"
                color = (150, 150, 150)
            else:
                text = f"{label}={int(value)}"
                color = (80, 220, 80) if value else (80, 100, 255)
            cv2.putText(
                canvas,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA,
            )
            width = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1
            )[0][0]
            x += width + 12
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
            (60, 60, 60),
            1,
        )
        cv2.putText(
            canvas,
            "joints [rad]",
            (margin_l, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
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
                canvas,
                (margin_l, y),
                (margin_l + plot_w, y),
                (40, 40, 40),
                1,
            )
        for j in range(self.dof):
            color = self.JOINT_COLORS[j % len(self.JOINT_COLORS)]
            points = []
            for i in range(n):
                x = margin_l + int(plot_w * i / max(1, n - 1))
                y = margin_t + int(plot_h * (1.0 - (data[i, j] - lo) / (hi - lo)))
                points.append((x, y))
            if len(points) > 1:
                cv2.polylines(canvas, [np.array(points)], False, color, 1, cv2.LINE_AA)
        latest = data[-1]
        for j in range(self.dof):
            color = self.JOINT_COLORS[j % len(self.JOINT_COLORS)]
            lx = margin_l + int(plot_w * (j + 0.5) / self.dof)
            label = f"J{j + 1} {latest[j]:+.2f}"
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.putText(
                canvas,
                label,
                (
                    max(margin_l, min(margin_l + plot_w - tw, lx - tw // 2)),
                    margin_t + plot_h + 18,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )
        return canvas

    def _draw_beaver_grid(
        self,
        canvas: np.ndarray,
        beaver: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    ) -> np.ndarray:
        cv2.putText(
            canvas,
            "beaver distance [mm]  (min/avg: status 5/9; 0 mm valid)",
            (10, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )
        if beaver is None:
            return canvas
        distance, present, status = beaver
        n_sensors, grid, _ = distance.shape
        if n_sensors == 0 or status.shape != distance.shape:
            return canvas
        cols = 3
        rows = int(np.ceil(n_sensors / cols))
        tile = min(52, max(24, (195 - (rows - 1) * 6) // rows))
        gap_text = 6
        text_w = 64
        item_w = tile + gap_text + text_w
        gap_x = 16
        gap_y = 6
        total_w = cols * item_w + (cols - 1) * gap_x
        total_h = rows * tile + (rows - 1) * gap_y
        x0 = max(10, (canvas.shape[1] - total_w) // 2)
        y0 = max(24, 24 + (195 - total_h) // 2)
        norm = max(self.distance_max_mm, 1.0)
        for s in range(n_sensors):
            r, c = divmod(s, cols)
            ox = x0 + c * (item_w + gap_x)
            oy = y0 + r * (tile + gap_y)
            present_s = bool(present[s]) if s < len(present) else False
            valid = (
                present_s
                & np.isin(status[s], VALID_STATUSES)
                & np.isfinite(distance[s])
                & (distance[s] >= 0)
            )
            if np.any(valid):
                min_text = f"min {float(np.min(distance[s][valid])):.1f}"
                avg_text = f"avg {float(np.mean(distance[s][valid])):.1f}"
            else:
                min_text = "min --"
                avg_text = "avg --"
            heat = np.zeros((grid, grid), np.uint8)
            if present_s:
                values = np.clip(distance[s] / norm * 255.0, 0, 255).astype(np.uint8)
                heat = cv2.applyColorMap(values, cv2.COLORMAP_VIRIDIS)
                heat[~valid] = (70, 70, 70)
            else:
                heat = np.full((grid, grid, 3), 70, np.uint8)
            heat = cv2.resize(heat, (tile, tile), interpolation=cv2.INTER_NEAREST)
            canvas[oy : oy + tile, ox : ox + tile] = heat
            border = (80, 220, 80) if present_s else (70, 70, 200)
            cv2.rectangle(
                canvas,
                (ox, oy),
                (ox + tile - 1, oy + tile - 1),
                border,
                1,
            )
            tx = ox + tile + gap_text
            cv2.putText(
                canvas,
                f"S{s}",
                (tx, oy + 13),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 255) if present_s else (150, 150, 150),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                min_text,
                (tx, oy + 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                avg_text,
                (tx, oy + 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.30,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
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
        from policies.realman_beaver.checkpoint import checkpoint_summary, load_policy

        utils.logger.info(f"Loading policy '{name}' from {self.checkpoint}")
        checkpoint_variant = validate_deployable_checkpoint(
            checkpoint_summary(self.checkpoint)
        )
        try:
            requested_prediction_steps = eval_cfg.PREDICTION_STEPS[checkpoint_variant]
            requested_action_steps = eval_cfg.ACTION_STEPS[checkpoint_variant]
        except KeyError as exc:
            raise ValueError(
                f"Missing prediction/action step configuration for "
                f"'{checkpoint_variant}' in eval_config.py"
            ) from exc
        self.policy = load_policy(
            self.checkpoint,
            device=self.device,
            use_ema=eval_cfg.USE_EMA,
            prediction_steps=requested_prediction_steps,
            action_steps=requested_action_steps,
            noise_scheduler_type=eval_cfg.NOISE_SCHEDULER_TYPE,
            num_inference_steps=eval_cfg.NUM_INFERENCE_STEPS,
            wrap_near_threshold_mm=(
                eval_cfg.WRAP_NEAR_THRESHOLD_MM
                if checkpoint_variant in {"WRM_wrap", "WRM_wrap_delta"}
                else None
            ),
            wrap_range_scale_mm=(
                eval_cfg.WRAP_RANGE_SCALE_MM
                if checkpoint_variant in {"WRM_wrap", "WRM_wrap_delta"}
                else None
            ),
            wrap_lift_min_wrap=(
                eval_cfg.WRAP_LIFT_MIN_WRAP
                if checkpoint_variant in {"WRM_wrap", "WRM_wrap_delta"}
                else None
            ),
            wrap_stop_close_j3_wrap=(
                eval_cfg.WRAP_STOP_CLOSE_J3_WRAP
                if checkpoint_variant in {"WRM_wrap", "WRM_wrap_delta"}
                else None
            ),
            wrap_stop_close_j4_wrap=(
                eval_cfg.WRAP_STOP_CLOSE_J4_WRAP
                if checkpoint_variant in {"WRM_wrap", "WRM_wrap_delta"}
                else None
            ),
            wrap_stop_close_wrap=(
                eval_cfg.WRAP_STOP_CLOSE_WRAP
                if checkpoint_variant in {"WRM_wrap", "WRM_wrap_delta"}
                else None
            ),
            wrap_contact_stop_mm=(
                eval_cfg.WRAP_CONTACT_STOP_MM
                if checkpoint_variant in {
                    "WRM_wrap",
                    "WRM_wrap_delta",
                    "WRM_lobo_monitor",
                    "WRM_wrap_monitor",
                    "WRM_wrap_monitor_backup",
                }
                else None
            ),
            wrap_stop_hold_frames=(
                eval_cfg.WRAP_STOP_HOLD_FRAMES
                if checkpoint_variant in {"WRM_wrap", "WRM_wrap_delta"}
                else None
            ),
            wrap_lift_hold_frames=(
                eval_cfg.WRAP_LIFT_HOLD_FRAMES
                if checkpoint_variant in {"WRM_wrap", "WRM_wrap_delta"}
                else None
            ),
        )
        self.variant = self.policy.config.model.variant
        self.prediction_steps, self.action_steps = policy_step_window(self.policy)
        execution_window = self.action_steps
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
            f"prediction_steps={self.prediction_steps}, "
            f"action_steps={self.action_steps}, "
            f"inference_latency_steps={eval_cfg.INFERENCE_LATENCY_STEPS}"
        )
        if self.variant in {"WRM_wrap", "WRM_wrap_delta"}:
            model = self.policy.config.model
            utils.logger.info(
                "WRM_wrap gates: near_mm=%.3f closing_scale_mm=%.3f "
                "range_scale_mm=%.3f "
                "lift_min=%.3f stop_close_j3=%.3f stop_close_j4=%.3f "
                "contact_stop_mm=%.3f stop_hold=%d lift_hold=%d",
                model.beaver_wrap_near_threshold_mm,
                model.beaver_wrap_closing_scale_mm,
                model.beaver_wrap_range_scale_mm,
                model.beaver_wrap_lift_min_wrap,
                model.beaver_wrap_stop_close_j3_wrap
                if model.beaver_wrap_stop_close_j3_wrap is not None
                else model.beaver_wrap_stop_close_wrap,
                model.beaver_wrap_stop_close_j4_wrap
                if model.beaver_wrap_stop_close_j4_wrap is not None
                else model.beaver_wrap_stop_close_wrap,
                model.beaver_wrap_contact_stop_mm,
                model.beaver_wrap_stop_hold_frames,
                model.beaver_wrap_lift_hold_frames,
            )
        elif self.variant in {"WRM_wrap_monitor", "WRM_wrap_monitor_backup"}:
            utils.logger.info(
                "WRM_wrap learned gate: Beaver-only %s; fixed logit boundary=0",
                "temporal MLP"
                if self.variant == "WRM_wrap_monitor"
                else "Key4 backup MLP",
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
            utils.logger.info("Beaver reader created inside evaluator (not shared).")
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
        self._last_beaver_np: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._last_beaver_stale = False
        self._quit_requested = False
        self._last_tcp_quat: np.ndarray | None = None
        self._monitor_preview_warning_shown = False

    # ── Observation helpers ──────────────────────────────────────────────

    def _joints(self) -> np.ndarray:
        return np.asarray(self.backend.get_joint_configuration(), dtype=np.float64)

    def _tcp_pose(self) -> list[float] | None:
        """TCP pose as [qx, qy, qz, qw, x, y, z] for the log."""
        try:
            from airo_spatial_algebra.se3 import SE3Container

            tcp = self.backend.get_tcp_pose()
            se3 = SE3Container.from_homogeneous_matrix(tcp)
            self._last_tcp_quat = utils.quat_cal(
                se3.rotation_matrix, self._last_tcp_quat
            )
            return (
                np.concatenate([self._last_tcp_quat, se3.translation]).round(4).tolist()
            )
        except Exception as exc:
            utils.logger.warning(f"TCP pose read failed: {exc}")
            return None

    def _image_tensor(self) -> torch.Tensor:
        # airo-camera-toolkit returns RGB as float images in [0, 1], whereas
        # the training dataset, OpenCV monitor, and VideoWriter use uint8.
        # Convert once at acquisition so every consumer sees the same image
        # representation (and inference is not accidentally divided by 255
        # twice).
        self._last_video_frame = self._capture_images_uint8()
        image = self._last_video_frame[self.camera_name]
        # HWC uint8 RGB -> CHW float32 in [0, 1], matching the dataset.
        chw = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
        return torch.from_numpy(chw).float().div_(255.0).unsqueeze(0).to(self.device)

    def _capture_images_uint8(self) -> dict[str, np.ndarray]:
        return {
            name: self._rgb_frame_uint8(image)
            for name, image in self.cameras.get_images().items()
        }

    @staticmethod
    def _rgb_frame_uint8(image: np.ndarray) -> np.ndarray:
        """Return a contiguous HWC RGB frame suitable for OpenCV encoding."""
        frame = np.asarray(image)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected an HWC RGB frame, got shape {frame.shape}")
        if frame.dtype == np.uint8:
            return np.ascontiguousarray(frame)
        if np.issubdtype(frame.dtype, np.floating):
            frame = np.nan_to_num(frame, nan=0.0, posinf=1.0, neginf=0.0)
            if frame.size and float(frame.max()) <= 1.0:
                frame = frame * 255.0
        return np.ascontiguousarray(np.clip(frame, 0, 255).astype(np.uint8))

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
                status.detach().cpu().numpy()[0],
            )
        with torch.inference_mode():
            action = select_action_with_latency(
                self.policy,
                obs,
                self.cfg.INFERENCE_LATENCY_STEPS,
            )
        return action.squeeze(0).cpu().numpy()

    def _current_gate_status(self) -> dict[str, bool | None] | None:
        """Return the latest inference gate states for the live monitor."""
        variant = getattr(self, "variant", None)
        is_wrap_policy = variant in {"WRM_wrap", "WRM_wrap_delta"}
        is_monitor_gate_policy = variant in {
            "WRM_wrap_monitor",
            "WRM_wrap_monitor_backup",
            "WRM_lobo_monitor",
        }
        # The display follows the deployment parameters, not only the
        # checkpoint variant. The legacy shared flag means one overall
        # contact-stop state; explicit J3/J4 overrides mean independent
        # closure states. A WRM_wrap checkpoint can therefore use either
        # display mode depending on the command line.
        has_joint_stop_overrides = (
            getattr(self.cfg, "WRAP_STOP_CLOSE_J3_WRAP", None) is not None
            or getattr(self.cfg, "WRAP_STOP_CLOSE_J4_WRAP", None) is not None
        )
        use_per_joint_gate = is_wrap_policy and has_joint_stop_overrides
        use_overall_gate = is_monitor_gate_policy or (
            is_wrap_policy and not has_joint_stop_overrides
        )
        if not use_per_joint_gate and not use_overall_gate:
            return None
        lift_blocked = getattr(self.policy, "last_lift_blocked", None)
        lift_enabled = None if lift_blocked is None else not bool(lift_blocked)
        if use_per_joint_gate:
            return {
                "j3_stop": getattr(self.policy, "last_close_stopped_j3", None),
                "j4_stop": getattr(self.policy, "last_close_stopped_j4", None),
                "lift_enabled": lift_enabled,
            }
        contact_stopped = getattr(self.policy, "last_close_stopped", None)
        return {
            "contact_stop": (
                None if contact_stopped is None else bool(contact_stopped)
            ),
            "lift_enabled": lift_enabled,
        }

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
            "prediction_steps": self.prediction_steps,
            "action_steps": self.action_steps,
            "inference_latency_steps": self.cfg.INFERENCE_LATENCY_STEPS,
            "action_delta_max": 0.0,
            "success": None,
        }
        if self.variant in {"WRM_wrap", "WRM_wrap_delta"}:
            model = self.policy.config.model
            stats["wrap_gate"] = {
                "near_threshold_mm": model.beaver_wrap_near_threshold_mm,
                "closing_scale_mm": model.beaver_wrap_closing_scale_mm,
                "range_scale_mm": model.beaver_wrap_range_scale_mm,
                "lift_min_wrap": model.beaver_wrap_lift_min_wrap,
                "stop_close_j3_wrap": (
                    model.beaver_wrap_stop_close_j3_wrap
                    if model.beaver_wrap_stop_close_j3_wrap is not None
                    else model.beaver_wrap_stop_close_wrap
                ),
                "stop_close_j4_wrap": (
                    model.beaver_wrap_stop_close_j4_wrap
                    if model.beaver_wrap_stop_close_j4_wrap is not None
                    else model.beaver_wrap_stop_close_wrap
                ),
                "contact_stop_mm": model.beaver_wrap_contact_stop_mm,
                "stop_hold_frames": model.beaver_wrap_stop_hold_frames,
                "lift_hold_frames": model.beaver_wrap_lift_hold_frames,
            }
        elif self.variant in {"WRM_wrap_monitor", "WRM_wrap_monitor_backup"}:
            stats["monitor_gate"] = {
                "architecture": (
                    "temporal_mlp"
                    if self.variant == "WRM_wrap_monitor"
                    else "key4_backup_mlp"
                ),
                "decision_logit": 0.0,
                "parameter_overrides": False,
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
                    stats["beaver_stale_steps"] = int(stats["beaver_stale_steps"]) + 1

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
                        "replan": bool(getattr(self.policy, "last_replanned", False)),
                        "chunk_step": int(getattr(self.policy, "last_chunk_step", 0)),
                        "latency_steps": int(
                            getattr(self.policy, "last_latency_steps", 0)
                        ),
                        "commanded": commanded,
                        "drop_reason": reason or None,
                        "beaver_stale": bool(self._last_beaver_stale),
                    }
                    if self.variant == "WRM_temporal":
                        record["grasp_probability"] = float(
                            getattr(self.policy, "last_grasp_probability", 0.0)
                        )
                        record["beaver_feature_std"] = float(
                            getattr(self.policy, "last_beaver_feature_std", 0.0)
                        )
                        record["beaver_feature_mean"] = float(
                            getattr(self.policy, "last_beaver_feature_mean", 0.0)
                        )
                        record["beaver_sensor_token_std"] = dict(
                            getattr(self.policy, "last_sensor_token_std", {})
                        )
                    if self.variant in {
                        "WRM_wrap",
                        "WRM_wrap_delta",
                        "WRM_wrap_monitor",
                        "WRM_wrap_monitor_backup",
                        "WRM_lobo_monitor",
                    }:
                        record["wrap_progress"] = float(
                            getattr(self.policy, "last_wrap_progress", 0.0)
                        )
                        record["min_range_mm"] = float(
                            getattr(self.policy, "last_min_range_mm", 0.0)
                        )
                        record["lift_blocked"] = float(
                            getattr(self.policy, "last_lift_blocked", 0.0)
                        )
                        record["close_stopped"] = float(
                            getattr(self.policy, "last_close_stopped", 0.0)
                        )
                        record["close_stopped_j3"] = float(
                            getattr(self.policy, "last_close_stopped_j3", 0.0)
                        )
                        record["close_stopped_j4"] = float(
                            getattr(self.policy, "last_close_stopped_j4", 0.0)
                        )
                        record["jaw_wrap"] = float(
                            getattr(self.policy, "last_jaw_wrap", 0.0)
                        )
                        record["enclosed_hold"] = float(
                            getattr(self.policy, "last_enclosed_hold", 0.0)
                        )
                        record["beaver_feature_std"] = float(
                            getattr(self.policy, "last_beaver_feature_std", 0.0)
                        )
                    if self.variant in {
                        "WRM_wrap_monitor",
                        "WRM_wrap_monitor_backup",
                        "WRM_lobo_monitor",
                    }:
                        record["monitor_lift_probability"] = float(
                            getattr(
                                self.policy, "last_monitor_lift_probability", 0.0
                            )
                        )
                        record["monitor_contact_probability"] = float(
                            getattr(
                                self.policy,
                                "last_monitor_contact_probability",
                                0.0,
                            )
                        )
                        record["monitor_lift_state"] = float(
                            getattr(self.policy, "last_monitor_lift_state", 0.0)
                        )
                        record["monitor_contact_state"] = float(
                            getattr(self.policy, "last_monitor_contact_state", 0.0)
                        )
                    if self.variant == "WRM_adaptive":
                        record["grasp_probability"] = float(
                            getattr(self.policy, "last_grasp_probability", 0.0)
                        )
                        record["beaver_feature_std"] = float(
                            getattr(self.policy, "last_z_beaver_std", 0.0)
                        )
                        record["sensor_attention_entropy"] = float(
                            getattr(self.policy, "last_sensor_attention_entropy", 0.0)
                        )
                        record["near_field_fraction"] = float(
                            getattr(self.policy, "last_near_field_fraction", 0.0)
                        )
                        record["sensor_attention"] = dict(
                            getattr(self.policy, "last_sensor_attention", {})
                        )
                    if self.variant == "dp_beaver_closure":
                        record["grasp_probability"] = float(
                            getattr(self.policy, "last_grasp_probability", 0.0)
                        )
                        record["closure_gate_mean"] = float(
                            getattr(self.policy, "last_gate_mean", 0.0)
                        )
                        record["closure_gate_std"] = float(
                            getattr(self.policy, "last_gate_std", 0.0)
                        )
                        record["closure_residual_magnitude"] = float(
                            getattr(
                                self.policy,
                                "last_closure_residual_magnitude",
                                0.0,
                            )
                        )
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
                            "gate": self._current_gate_status(),
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
                stats["action_delta_mean"] = round(float(np.mean(delta_magnitudes)), 4)
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
                verdict = self._wait_command(monitor_state="VERDICT", episode=index)
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
            freedrive = self.cfg.PAUSE_FREEDRIVE and self.backend.supports_freedrive
            if freedrive:
                try:
                    if self.cfg.FREEDRIVE_SENSITIVITY is not None:
                        self.backend.set_freedrive_sensitivity(
                            self.cfg.FREEDRIVE_SENSITIVITY
                        )
                    self.backend.start_freedrive()
                except Exception as exc:
                    utils.logger.warning(
                        "Could not enter freedrive (%s); holding pose instead.",
                        exc,
                    )
                    freedrive = False
            utils.logger.info(
                "Paused (%s). Enter=resume, s=success(next episode), "
                "f=fail(next episode), r=restart, q=quit.",
                "freedrive active — you may move the arm by hand"
                if freedrive
                else "robot holds its pose",
            )
            try:
                while True:
                    resume = self._wait_command(monitor_state="PAUSED")
                    if resume in {"", "p", "pause"}:
                        # The operator may have moved the arm while paused
                        # (freedrive) or the pose stayed put; either way,
                        # clear the action queue and replan from the current
                        # joint configuration instead of executing stale
                        # targets.
                        self.policy.reset()
                        utils.logger.info(
                            "Resuming; replanning from the current configuration."
                        )
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
            finally:
                if freedrive:
                    try:
                        self.backend.stop_freedrive()
                    except Exception as exc:
                        utils.logger.warning("Error leaving freedrive: %s", exc)
        return False, False, False, None

    # ── Recording helpers ────────────────────────────────────────────────

    def _open_video_writer(self, path: Path) -> cv2.VideoWriter | None:
        width, height = self.hw.REALSENSE_RESOLUTION
        writer = cv2.VideoWriter(
            str(path),
            # The pip OpenCV wheels do not ship a software H.264 encoder.
            # MPEG-4 Part 2 is supported by OpenCV's bundled FFmpeg and keeps
            # the existing MP4 output format.
            cv2.VideoWriter_fourcc(*"mp4v"),
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

    def _show_monitor_preview(self, state: str, episode: int | None) -> None:
        """Refresh the monitor while evaluation is waiting for user input."""
        if self.monitor is None:
            return
        try:
            self._last_video_frame = self._capture_images_uint8()
            camera = self._last_video_frame.get(self.camera_name)
            joints = self._joints()
            beaver_np = self._last_beaver_np
            beaver_ok: bool | None = None
            if self.beaver is not None:
                snapshot = self.beaver.snapshot()
                beaver_np = (
                    np.asarray(snapshot.distance_mm),
                    np.asarray(snapshot.present),
                    np.asarray(snapshot.target_status),
                )
                stale = (
                    snapshot.timestamp_ns == 0
                    or not snapshot.connected
                    or (time.monotonic_ns() - snapshot.timestamp_ns) / 1e9
                    > self.cfg.STALE_AFTER_S
                )
                beaver_ok = not stale
            self.monitor.show(
                camera=camera,
                joints=joints,
                beaver=beaver_np,
                status={
                    "state": state,
                    "policy": self.name,
                    "episode": "?" if episode is None else episode,
                    "step": "-",
                    "fps": self.cfg.FPS,
                    "beaver_ok": beaver_ok,
                    "gate": self._current_gate_status(),
                },
            )
        except Exception as exc:  # noqa: BLE001 - hardware preview is best-effort
            # A preview failure must not prevent keyboard input or abort an
            # otherwise valid hardware evaluation.
            if not self._monitor_preview_warning_shown:
                utils.logger.warning(f"Monitor preview unavailable: {exc}")
                self._monitor_preview_warning_shown = True
            self.monitor.pump_keys()

    def _wait_command(
        self,
        *,
        monitor_state: str = "WAITING",
        episode: int | None = None,
    ) -> str:
        """Block until a command arrives from stdin or the monitor window.

        Used wherever the program pauses for the user (before an episode,
        while paused mid-run) so that keys pressed while the monitor window
        has focus are handled as well as terminal keys.
        """
        next_preview = 0.0
        while True:
            command = self._stdin.poll()
            if command is not None:
                return command
            if self.monitor is not None:
                now = time.monotonic()
                if now >= next_preview:
                    self._show_monitor_preview(monitor_state, episode)
                    next_preview = now + 0.2
                else:
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
                    if self._wait_command(monitor_state="READY", episode=index) == "q":
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


def _resolve_checkpoint_path(spec: str) -> str:
    """Resolve a checkpoint spec to an existing file.

    Accepts the exact path, or a partial filename (e.g. ``100000.pt``)
    when it uniquely matches one checkpoint in the same directory.
    """
    path = Path(spec)
    if path.exists():
        return str(path)
    if path.suffix != ".pt" or not path.parent.exists():
        return str(path)
    matches = sorted(path.parent.glob(f"*{path.name}"))
    if len(matches) == 1:
        utils.logger.info(
            "Checkpoint '%s' not found; using unique match '%s'.",
            spec,
            matches[0].name,
        )
        return str(matches[0])
    return str(path)


def _infer_policy_name(checkpoint: str, policies: dict[str, str]) -> str:
    """Best-effort: which registered policy does this checkpoint belong to?

    Resolution order:
      1. a registered name in the checkpoint's file name — the most
         specific signal (``.../dp_beaver_key4_PCA/.../dp_beaver_key4_pca_step_100000.pt``
         resolves to ``dp_beaver_key4_pca``, not its prefix ``dp_beaver_key4``);
      2. the variant recorded inside the checkpoint itself, when it is a
         registered name (covers renamed files, e.g. ``dp_beaver_temporal/
         checkpoints/last.pt`` → ``WRM_temporal``);
      3. a registered name anywhere in the path, longest match wins.
    """

    def _registered_in(text: str) -> list[str]:
        normalized = re.sub(r"[^a-z0-9]", "_", text.lower())
        return [
            name
            for name in policies
            if re.search(rf"(^|_){re.escape(name)}(_|$)", normalized)
        ]

    path = Path(checkpoint)
    matches = _registered_in(path.name)
    if matches:
        return max(matches, key=len)
    # The file name gave no answer; read the variant recorded in the
    # checkpoint config (mmap keeps this to a few seconds for 1 GB files).
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        variant = ck["config"]["model"]["variant"]
        if variant in policies:
            return variant
    except Exception:
        pass
    matches = _registered_in(str(path))
    if matches:
        return max(matches, key=len)
    raise SystemExit(
        f"Unknown policy '{checkpoint}'. Use NAME=PATH with one of "
        "the registered names, use a safe custom NAME=PATH label, or pass a "
        "checkpoint path whose saved variant is registered."
    )


def _resolve_policy_selections(
    items: list[str], policies: dict[str, str]
) -> dict[str, str]:
    """Resolve CLI policy specs while preserving distinct experiment labels."""
    selections: dict[str, str] = {}
    for item in items:
        name, sep, checkpoint = item.partition("=")
        if sep:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
                raise SystemExit(
                    f"Invalid policy label '{name}'. Use only letters, digits, "
                    "dot, underscore, and hyphen."
                )
            if not checkpoint:
                raise SystemExit(f"Missing checkpoint path after '{name}='.")
            selections[name] = _resolve_checkpoint_path(checkpoint)
        elif name in policies:
            selections[name] = policies[name]
        else:
            checkpoint = _resolve_checkpoint_path(item)
            inferred_name = _infer_policy_name(checkpoint, policies)
            selections[inferred_name] = checkpoint
    return selections


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained RealMan-Beaver policies on the RM75 robot."
    )
    parser.add_argument(
        "--policy",
        action="append",
        default=[],
        metavar="NAME[=CHECKPOINT]|CHECKPOINT",
        help="Policy to evaluate; repeatable. A registered NAME uses its "
        "configured path; NAME=CHECKPOINT may use a custom safe run label "
        "(e.g. --policy seed42=…/last.pt). "
        "A bare checkpoint path is also accepted; its policy name is "
        "inferred from the path (e.g. --policy …/dp_beaver_key4_PCA/…). "
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
        "--wrap-near-mm",
        type=float,
        default=None,
        help="Override WRM_wrap per-sensor near-field threshold in millimetres",
    )
    parser.add_argument(
        "--wrap-range-scale-mm",
        type=float,
        default=None,
        help="Override WRM_wrap enclosure distance normalization scale",
    )
    parser.add_argument(
        "--wrap-lift-min",
        type=float,
        default=None,
        help="Override WRM_wrap minimum wrap fraction required to release J1",
    )
    parser.add_argument(
        "--wrap-stop-close",
        type=float,
        default=None,
        help="Deprecated: set the stop-close fraction for both J3 and J4",
    )
    parser.add_argument(
        "--wrap-stop-close-j3",
        type=float,
        default=None,
        help="Override WRM_wrap near/contact fraction required to freeze J3",
    )
    parser.add_argument(
        "--wrap-stop-close-j4",
        type=float,
        default=None,
        help="Override WRM_wrap near/contact fraction required to freeze J4",
    )
    parser.add_argument(
        "--wrap-contact-stop-mm",
        type=float,
        default=None,
        help="Override WRM_wrap minimum-distance threshold for closure freeze",
    )
    parser.add_argument(
        "--wrap-stop-hold-frames",
        "--wrap-stop-hold",
        dest="wrap_stop_hold_frames",
        type=int,
        default=None,
        help="Frames both jaws must stay enclosed before J3/J4 freeze",
    )
    parser.add_argument(
        "--wrap-lift-hold-frames",
        "--wrap-lift-hold",
        dest="wrap_lift_hold_frames",
        type=int,
        default=None,
        help="Frames both jaws must stay enclosed before J1 is released",
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
    parser.add_argument(
        "--beaver-simulate-8bit",
        action="store_true",
        default=False,
        help="Quantize 16-bit Beaver distances to 10mm steps (matching legacy 8-bit dataset distribution)",
    )
    parser.add_argument(
        "--ddim",
        action="store_true",
        default=False,
        help="Use fast DDIM ODE scheduler for all evaluated Diffusion Policies",
    )
    parser.add_argument(
        "--scheduler",
        choices=["DDPM", "DDIM"],
        default=None,
        help="Explicitly choose diffusion noise scheduler (DDPM or DDIM)",
    )
    parser.add_argument(
        "--num-inference-steps",
        "--inference-steps",
        dest="num_inference_steps",
        type=int,
        default=None,
        help="Number of diffusion inference denoising steps (e.g. 15 for DDIM, 100 for DDPM)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    eval_cfg = EvalConfig()
    hw_cfg = Config()

    if args.beaver_simulate_8bit:
        hw_cfg.BEAVER_SIMULATE_8BIT = True

    if args.ddim:
        eval_cfg.NOISE_SCHEDULER_TYPE = "DDIM"
        if eval_cfg.NUM_INFERENCE_STEPS is None:
            eval_cfg.NUM_INFERENCE_STEPS = 15
    if args.scheduler is not None:
        eval_cfg.NOISE_SCHEDULER_TYPE = args.scheduler
    if args.num_inference_steps is not None:
        eval_cfg.NUM_INFERENCE_STEPS = args.num_inference_steps

    if args.checkpoint_root:
        checkpoint_root = Path(args.checkpoint_root).expanduser()
        eval_cfg.POLICIES = {
            name: str(checkpoint_root / name / "last.pt") for name in eval_cfg.POLICIES
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
    for argument, attribute in (
        (args.wrap_near_mm, "WRAP_NEAR_THRESHOLD_MM"),
        (args.wrap_range_scale_mm, "WRAP_RANGE_SCALE_MM"),
        (args.wrap_lift_min, "WRAP_LIFT_MIN_WRAP"),
        (args.wrap_stop_close_j3, "WRAP_STOP_CLOSE_J3_WRAP"),
        (args.wrap_stop_close_j4, "WRAP_STOP_CLOSE_J4_WRAP"),
        (args.wrap_stop_close, "WRAP_STOP_CLOSE_WRAP"),
        (args.wrap_contact_stop_mm, "WRAP_CONTACT_STOP_MM"),
        (args.wrap_stop_hold_frames, "WRAP_STOP_HOLD_FRAMES"),
        (args.wrap_lift_hold_frames, "WRAP_LIFT_HOLD_FRAMES"),
    ):
        if argument is not None:
            setattr(eval_cfg, attribute, argument)
    # Re-run deployment validation after applying command-line overrides.
    eval_cfg.__post_init__()
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
        selections = _resolve_policy_selections(args.policy, eval_cfg.POLICIES)
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
    for name, variant in variants.items():
        if (
            variant not in eval_cfg.PREDICTION_STEPS
            or variant not in eval_cfg.ACTION_STEPS
        ):
            raise SystemExit(
                f"Missing prediction/action step configuration for "
                f"'{variant}' in eval_config.py"
            )
        utils.logger.info(
            f"{name}: predict {eval_cfg.PREDICTION_STEPS[variant]} steps, "
            f"execute {eval_cfg.ACTION_STEPS[variant]} steps per replan"
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
                utils.logger.info("Quit requested; skipping remaining policies.")
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
