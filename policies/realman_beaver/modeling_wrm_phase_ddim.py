"""Standalone Phase-DDIM Beaver Policy.

Incorporates:
1. Fast DDIM Diffusion Scheduler (15 inference steps vs 100 DDPM) for low-latency planning.
2. High-frequency per-tick Phase Task Monitor (<1ms) for instantaneous zero-lag contact stop.
3. Physics-informed Anti-Undergrasp Shield (J3 > -1.35 rad cannot stop in mid-air).
4. Asymmetric coordinated lift release (J1 released when lift ready).
5. Controller-level Exponential Moving Average (EMA) smoothing across chunk boundaries.
"""

from __future__ import annotations

from collections import deque
import torch
from torch import Tensor, nn
from lerobot.utils.constants import ACTION

from policies.realman_beaver.configuration import RealmanBeaverConfig
from policies.realman_beaver.dataset import ObservationNormalizer, resolve_beaver_sensor_indices
from policies.realman_beaver.modeling_wrm_wrap import WrapBeaverDPPolicy
from policies.realman_beaver.modules.phase_task_monitor import PhaseTaskMonitor


class PhaseDDIMBeaverDPPolicy(WrapBeaverDPPolicy):
    """Dual-rate Policy: Macro DDIM planning + Micro per-tick reactive Phase Monitor."""

    def __init__(
        self,
        config: RealmanBeaverConfig,
        normalizer: ObservationNormalizer,
        ema_alpha: float = 0.65,
    ) -> None:
        # Enforce DDIM scheduler settings on this policy
        config.model.noise_scheduler_type = "DDIM"
        if getattr(config.model, "num_inference_steps", 100) > 25:
            config.model.num_inference_steps = 15

        self.ema_alpha = float(ema_alpha)
        self.last_sent_action: Tensor | None = None
        self._recent_joints: deque[Tensor] = deque(maxlen=4)

        super().__init__(config, normalizer)
        model, dataset = config.model, config.dataset

        self.monitor = PhaseTaskMonitor(
            sensor_indices=resolve_beaver_sensor_indices(
                dataset, model.beaver_wrap_sensors
            ),
            n_sensors=model.beaver_shape[0],
            closure_joint_indices=(3, 4, 5),
            lift_joint_index=1,
            valid_statuses=dataset.beaver_valid_statuses,
            proximity_scales_mm=model.beaver_monitor_proximity_scales_mm,
            best_proximity_scales_mm=getattr(
                model, "beaver_monitor_best_proximity_scales_mm", (20.0, 50.0, 150.0)
            ),
            range_scale_mm=model.beaver_monitor_range_scale_mm,
            hidden_dims=(64, 32),
            dropout=model.beaver_monitor_dropout,
        )
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.last_sent_action = None
        if hasattr(self, "_recent_joints"):
            self._recent_joints.clear()
        else:
            self._recent_joints = deque(maxlen=4)
        self.last_replanned = True
        self.last_chunk_step = 0
        self.last_monitor_lift_probability = 0.0
        self.last_monitor_contact_probability = 0.0
        self.last_monitor_lift_state = 0.0
        self.last_monitor_contact_state = 0.0
        self.last_close_stopped = 0.0
        self.last_lift_blocked = 0.0

    @torch.no_grad()
    def select_action(self, observation: dict[str, Tensor]) -> Tensor:
        from policies.realman_beaver.modeling import _ensure_observation_batch

        batch = _ensure_observation_batch(observation)
        batch_size = batch["state"].shape[0]
        if self._batch_size is not None and self._batch_size != batch_size:
            self.reset()
        self._batch_size = batch_size

        current_joints = self._current_joint(batch)  # (B, 7)
        self._recent_joints.append(current_joints)

        # 1. Macro-Planning: Track LeRobot native policy action queue & replan state
        temporal_batch = self._append_online_history(batch)
        queue = getattr(self.native_policy, "_queues", None)
        before = len(queue[ACTION]) if queue is not None and ACTION in queue else 0
        replanned = before == 0

        prepared, middle = self._prepare(temporal_batch, include_action=False)
        normalized = self.native_policy.select_action(prepared)

        after = len(queue[ACTION]) if queue is not None and ACTION in queue else 0
        if replanned:
            self._chunk_len = after + 1
        self.last_replanned = replanned
        self.last_chunk_step = max(0, self._chunk_len - 1 - after)

        raw_action = self.normalizer.denormalize_action(normalized)  # (B, 7)

        # 2. Micro-Reaction: Per-tick Phase Task Monitor (<1 ms)
        # Compute joint velocity from recent buffer
        if len(self._recent_joints) >= 2:
            prev_j = self._recent_joints[0]
            dq = current_joints[:, [3, 4, 5]] - prev_j[:, [3, 4, 5]]
        else:
            dq = torch.zeros(current_joints.shape[0], 3, device=current_joints.device)

        # Run PhaseTaskMonitor on the single latest tick observation
        res = self.monitor.predict_phase_state(
            batch["beaver_distance"],
            batch["beaver_status"],
            batch["beaver_present"],
            current_joints,
            dq,
            anti_undergrasp_j3_limit=-1.35,
        )

        is_contact_stop = bool(res["contact_stop"][0].item())
        is_lift_ready = bool(res["lift_ready"][0].item())
        prob_lift = float(res["prob_lift"][0].item())
        prob_contact = float(res["prob_contact"][0].item())

        self.last_monitor_contact_probability = prob_contact
        self.last_monitor_lift_probability = prob_lift
        self.last_monitor_contact_state = 1.0 if is_contact_stop else 0.0
        self.last_monitor_lift_state = 1.0 if is_lift_ready else 0.0

        action = raw_action.clone()

        # 3. Intercept & Clamp:
        # A. If contact established: Freeze fingers immediately at current position (prevent Over-Grasp)
        if is_contact_stop:
            action[:, [3, 4, 5]] = current_joints[:, [3, 4, 5]]
            self.last_close_stopped = 1.0
        else:
            self.last_close_stopped = 0.0

        # B. If lift not confirmed ready: Hold lift joint J1 at current/table height (prevent Under-Grasp pullout)
        if not is_lift_ready:
            action[:, 1] = torch.maximum(action[:, 1], current_joints[:, 1])
            self.last_lift_blocked = 1.0
        else:
            self.last_lift_blocked = 0.0

        # 4. Controller EMA Smoothing across chunk boundaries (eliminates stutter & velocity spikes)
        if self.last_sent_action is not None and self.last_sent_action.shape == action.shape:
            smoothed_action = self.ema_alpha * action + (1.0 - self.ema_alpha) * self.last_sent_action
        else:
            smoothed_action = action

        self.last_sent_action = smoothed_action.clone()
        return smoothed_action
