"""WRM_wrap policies whose execution gate is a trained Beaver-only monitor."""

from __future__ import annotations

import torch
from torch import Tensor

from policies.realman_beaver.configuration import (
    LOBO_MONITOR_BEAVER_VARIANT,
    WRAP_MONITOR_BACKUP_BEAVER_VARIANT,
    WRAP_MONITOR_BEAVER_VARIANT,
    WRAP_MONITOR_BEAVER_VARIANTS,
    RealmanBeaverConfig,
)
from policies.realman_beaver.dataset import ObservationNormalizer, resolve_beaver_sensor_indices
from policies.realman_beaver.modeling_wrm_wrap import WrapBeaverDPPolicy
from policies.realman_beaver.modules.beaver_monitor import (
    BackupBeaverMonitor,
    TemporalBeaverMonitor,
    monitor_states,
)
from policies.realman_beaver.modules.instant_contact import InstantContactMonitor


class MonitorWrapBeaverDPPolicy(WrapBeaverDPPolicy):
    """Unmodified WRM_wrap action generator plus a learned execution monitor."""

    def __init__(
        self, config: RealmanBeaverConfig, normalizer: ObservationNormalizer
    ) -> None:
        if config.model.variant not in WRAP_MONITOR_BEAVER_VARIANTS:
            raise ValueError(
                "MonitorWrapBeaverDPPolicy requires one of "
                f"{sorted(WRAP_MONITOR_BEAVER_VARIANTS)}"
            )
        super().__init__(config, normalizer)
        model, dataset = config.model, config.dataset
        if model.variant == WRAP_MONITOR_BEAVER_VARIANT:
            self.monitor = TemporalBeaverMonitor(
                n_sensors=model.beaver_shape[0],
                history_steps=model.beaver_wrap_history_steps,
                lag_steps=model.beaver_monitor_lag_steps,
                valid_statuses=dataset.beaver_valid_statuses,
                proximity_scales_mm=model.beaver_monitor_proximity_scales_mm,
                range_scale_mm=model.beaver_monitor_range_scale_mm,
                hidden_dims=model.beaver_monitor_hidden_dims,
                dropout=model.beaver_monitor_dropout,
            )
        elif model.variant == WRAP_MONITOR_BACKUP_BEAVER_VARIANT:
            self.monitor = BackupBeaverMonitor(
                sensor_indices=resolve_beaver_sensor_indices(
                    dataset, model.beaver_wrap_sensors
                ),
                valid_statuses=dataset.beaver_valid_statuses,
                hidden_dim=model.beaver_monitor_backup_hidden_dim,
            )
        elif model.variant == LOBO_MONITOR_BEAVER_VARIANT:
            self.monitor = InstantContactMonitor(
                sensor_indices=resolve_beaver_sensor_indices(
                    dataset, model.beaver_wrap_sensors
                ),
                n_sensors=model.beaver_shape[0],
                use_joints=True,
                joint_indices=(3, 4, 5),
                n_joints=3,
                valid_statuses=dataset.beaver_valid_statuses,
                proximity_scales_mm=model.beaver_monitor_proximity_scales_mm,
                range_scale_mm=model.beaver_monitor_range_scale_mm,
                hidden_dims=model.beaver_monitor_hidden_dims,
                dropout=model.beaver_monitor_dropout,
            )
        else:  # pragma: no cover - guarded above
            raise AssertionError(model.variant)
        self.reset()

    def _condition(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        conditioned_state, middle = super()._condition(batch)
        if isinstance(self.monitor, InstantContactMonitor):
            joints = self._current_joint(batch)
            logit = self.monitor(
                batch["beaver_history_distance"],
                batch["beaver_history_status"],
                batch["beaver_history_present"],
                joints,
            )
            lift_state = logit >= 0.0
            # Instant contact monitor unlocks the lift joint J1 upon contact.
            # Finger closure (J3, J4, J5) is commanded continuously by the Diffusion Policy
            # to maintain grasp force and avoid freezing fingers in mid-air.
            contact_state = torch.zeros_like(lift_state)
            prob = torch.sigmoid(logit)
            logits = torch.stack((logit, logit), dim=-1)
            states = torch.stack((lift_state, contact_state), dim=-1)
            probabilities = torch.stack((prob, prob), dim=-1)
        else:
            logits = self.monitor(
                batch["beaver_history_distance"],
                batch["beaver_history_status"],
                batch["beaver_history_present"],
            )
            states = monitor_states(logits)
            probabilities = torch.sigmoid(logits)
        middle["monitor_logits"] = logits
        middle["monitor_states"] = states
        middle["monitor_probabilities"] = probabilities
        self.last_monitor_lift_probability = float(
            probabilities[:, 0].detach().mean()
        )
        self.last_monitor_contact_probability = float(
            probabilities[:, 1].detach().mean()
        )
        self.last_monitor_lift_state = float(states[:, 0].float().detach().mean())
        self.last_monitor_contact_state = float(
            states[:, 1].float().detach().mean()
        )
        return conditioned_state, middle

    # NOTE: No _apply_wrap_lift_gate override.  The parent WrapBeaverDPPolicy
    # uses wrap_progress-based dual-threshold gating:
    #   lift unlock:  jaw_wrap >= lift_min_wrap  (0.25)
    #   close stop:   wrap >= stop_close_wrap (0.5) AND sensor <= contact_stop_mm
    # This creates a crucial ~1.8 s gap where J1 is free but fingers keep
    # closing, letting the diffusion backbone generate coordinated lift+close
    # actions.  Overriding this with a simultaneous unlock/freeze (gap = 0)
    # causes the backbone to never emit lift actions.

    def reset(self) -> None:
        super().reset()
        self.last_monitor_lift_probability = 0.0
        self.last_monitor_contact_probability = 0.0
        self.last_monitor_lift_state = 0.0
        self.last_monitor_contact_state = 0.0
