"""Beaver-aware closed-loop closure residual on an unchanged global DP."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from torch import Tensor, nn

from policies.realman_beaver.configuration import (
    BEAVER_CLOSURE_VARIANT,
    RealmanBeaverConfig,
)
from policies.realman_beaver.dataset import ObservationNormalizer
from policies.realman_beaver.modules.closure_beaver_encoder import (
    ClosureBeaverEncoder,
)


class DPBeaverClosurePolicy(nn.Module):
    """Global RGB/q DP plus a masked, Beaver-driven joint residual.

    The global Diffusion Policy never receives a Beaver feature. During
    training the demonstrated action is decomposed as
    ``a_demo = a_global_target + a_closure``. The native diffusion objective
    learns ``a_global_target`` and the closure branch receives gradients from
    that same demonstrated-action objective.
    """

    _ONLINE_KEYS = (
        "state",
        "beaver_distance",
        "beaver_status",
        "beaver_present",
    )

    def __init__(
        self, config: RealmanBeaverConfig, normalizer: ObservationNormalizer
    ) -> None:
        super().__init__()
        if config.model.variant != BEAVER_CLOSURE_VARIANT:
            raise ValueError(f"DPBeaverClosurePolicy requires {BEAVER_CLOSURE_VARIANT}")
        self.config = config
        self.normalizer = normalizer
        model, dataset = config.model, config.dataset

        self.beaver_encoder = ClosureBeaverEncoder(
            n_sensors=model.beaver_shape[0],
            sensor_shape=model.beaver_shape[1:],
            distance_max_mm=dataset.distance_max_mm,
            valid_statuses=dataset.beaver_valid_statuses,
            hidden_dim=model.closure_sensor_hidden_dim,
            output_dim=model.closure_beaver_encoder_dim,
        )
        delta_beaver_dim = (
            model.beaver_shape[0] * model.beaver_shape[1] * model.beaver_shape[2]
        )
        closure_input_dim = (
            model.closure_beaver_encoder_dim + 2 * model.state_dim + delta_beaver_dim
        )
        self.closure_encoder = nn.Sequential(
            nn.Linear(closure_input_dim, model.closure_hidden_dim),
            nn.SiLU(),
            nn.Linear(model.closure_hidden_dim, model.closure_hidden_dim),
            nn.SiLU(),
        )
        self.closure_residual_head = nn.Linear(
            model.closure_hidden_dim, model.action_dim
        )
        self.gate_head = nn.Linear(model.closure_hidden_dim, 1)
        grasp_hidden = max(model.closure_hidden_dim // 2, 1)
        self.grasp_state_head = nn.Sequential(
            nn.Linear(model.closure_hidden_dim, grasp_hidden),
            nn.SiLU(),
            nn.Linear(grasp_hidden, 1),
        )
        # Start from the exact global-DP behavior with a weak gate. The branch
        # becomes active only when demonstrated actions make it useful.
        nn.init.zeros_(self.closure_residual_head.weight)
        nn.init.zeros_(self.closure_residual_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -2.0)

        self.register_buffer(
            "closure_joint_mask",
            torch.tensor(model.closure_joint_mask, dtype=torch.float32),
        )
        from policies.realman_beaver.modeling import build_native_diffusion_config

        self.native_policy = DiffusionPolicy(build_native_diffusion_config(config))
        self.reset()

    def _global_prepare(
        self, batch: dict[str, Tensor], include_action: bool
    ) -> dict[str, Tensor]:
        prepared = {
            OBS_STATE: self.normalizer.normalize_state(batch["state"]),
            self.config.dataset.image_key: self.normalizer.normalize_image(
                batch["image"]
            ),
        }
        if include_action:
            prepared[ACTION] = self.normalizer.normalize_action(batch["action"])
            prepared["action_is_pad"] = batch["action_is_pad"]
        return prepared

    def _closure_outputs(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        required = {
            "state",
            "beaver_distance",
            "beaver_status",
            "beaver_present",
        }
        missing = required - batch.keys()
        if missing:
            raise KeyError(
                f"dp_beaver_closure batch is missing fields: {sorted(missing)}"
            )
        if batch["state"].ndim != 3 or batch["state"].shape[1] < 2:
            raise ValueError(
                "dp_beaver_closure requires at least the previous and current frame"
            )

        current_q = batch["state"][:, -1]
        previous_q = batch["state"][:, -2]
        normalized_q = self.normalizer.normalize_state(current_q)
        normalized_dq = (current_q - previous_q) / self.normalizer.state_scale
        beaver_embedding, middle = self.beaver_encoder(
            batch["beaver_distance"][:, -1],
            batch["beaver_distance"][:, -2],
            batch["beaver_status"][:, -1],
            batch["beaver_status"][:, -2],
            batch["beaver_present"][:, -1],
            batch["beaver_present"][:, -2],
            return_intermediates=True,
        )
        closure_input = torch.cat(
            (
                beaver_embedding,
                normalized_q,
                normalized_dq,
                middle["delta_flat"],
            ),
            dim=-1,
        )
        closure_latent = self.closure_encoder(closure_input)
        raw_residual = torch.tanh(self.closure_residual_head(closure_latent))
        raw_residual = raw_residual * self.config.model.closure_residual_scale
        gate = self.gate_head(closure_latent).sigmoid() * middle["any_available"]
        correction = raw_residual * gate * self.closure_joint_mask.view(1, -1)
        grasp_logit = self.grasp_state_head(closure_latent).squeeze(-1)

        self.last_gate_mean = float(gate.detach().mean())
        self.last_gate_std = float(gate.detach().std(unbiased=False))
        self.last_closure_residual_magnitude = float(
            self._residual_magnitude(correction).detach()
        )
        self.last_grasp_probability = float(grasp_logit.detach().sigmoid().mean())
        return correction, gate, grasp_logit, closure_latent

    def _residual_magnitude(
        self, correction: Tensor, valid_samples: Tensor | None = None
    ) -> Tensor:
        active_joints = self.closure_joint_mask.sum().clamp_min(1.0)
        per_sample = correction.square().sum(dim=-1).div(active_joints).sqrt()
        if valid_samples is None:
            return per_sample.mean()
        weights = valid_samples.to(per_sample.dtype)
        return (per_sample * weights).sum() / weights.sum().clamp_min(1.0)

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        if "grasp_state" not in batch:
            raise KeyError("dp_beaver_closure training batch is missing grasp_state")
        correction, gate, grasp_logit, _ = self._closure_outputs(batch)

        start = self.config.model.n_obs_steps - 1
        correction_window = torch.zeros_like(batch["action"])
        correction_window[:, start:] = correction.unsqueeze(1)
        global_batch = dict(batch)
        global_batch["action"] = batch["action"] - correction_window
        diffusion_loss, _ = self.native_policy(
            self._global_prepare(global_batch, include_action=True)
        )

        grasp_target = batch["grasp_state"][:, -1].to(
            device=grasp_logit.device, dtype=grasp_logit.dtype
        )
        if grasp_target.shape != grasp_logit.shape:
            raise ValueError(
                f"latest grasp_state shape {tuple(grasp_target.shape)} must match "
                f"grasp logits {tuple(grasp_logit.shape)}"
            )
        grasp_bce = F.binary_cross_entropy_with_logits(grasp_logit, grasp_target)

        active_joints = self.closure_joint_mask.sum().clamp_min(1.0)
        per_sample_abs = correction.abs().sum(dim=-1) / active_joints
        current_valid = (~batch["action_is_pad"][:, start]).to(per_sample_abs.dtype)
        tight_weights = current_valid * grasp_target
        residual_reg = (
            per_sample_abs * tight_weights
        ).sum() / tight_weights.sum().clamp_min(1.0)

        model = self.config.model
        total_loss = (
            diffusion_loss
            + model.closure_grasp_loss_weight * grasp_bce
            + model.closure_residual_loss_weight * residual_reg
        )
        residual_magnitude = self._residual_magnitude(correction, current_valid)
        grasp_probability = grasp_logit.sigmoid()
        return total_loss, {
            "loss": float(total_loss.detach()),
            "diffusion_loss": float(diffusion_loss.detach()),
            "grasp_bce": float(grasp_bce.detach()),
            "residual_reg": float(residual_reg.detach()),
            "gate_mean": float(gate.detach().mean()),
            "gate_std": float(gate.detach().std(unbiased=False)),
            "closure_residual_magnitude": float(residual_magnitude.detach()),
            "predicted_grasp_probability": float(grasp_probability.detach().mean()),
        }

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        prepared = self._global_prepare(batch, include_action=False)
        prepared[OBS_IMAGES] = torch.stack(
            [prepared[self.config.dataset.image_key]], dim=-4
        )
        normalized_global = self.native_policy.diffusion.generate_actions(prepared)
        global_action = self.normalizer.denormalize_action(normalized_global)
        correction, _, _, _ = self._closure_outputs(batch)
        return global_action + correction.unsqueeze(1)

    @torch.no_grad()
    def predict_actions(self, batch: dict[str, Tensor]) -> Tensor:
        return self.predict_action_chunk(batch)

    def _online_closure_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch_size = batch["state"].shape[0]
        if (
            self._online_batch_size is not None
            and self._online_batch_size != batch_size
        ):
            self.reset()
        self._online_batch_size = batch_size
        current = {key: batch[key] for key in self._ONLINE_KEYS}
        previous = self._previous_online_frame or current
        closure_batch = dict(batch)
        for key in self._ONLINE_KEYS:
            closure_batch[key] = torch.stack((previous[key], current[key]), dim=1)
        self._previous_online_frame = {
            key: value.detach().clone() for key, value in current.items()
        }
        return closure_batch

    @torch.no_grad()
    def select_action(self, observation: dict[str, Tensor]) -> Tensor:
        from policies.realman_beaver.modeling import _ensure_observation_batch

        batch = _ensure_observation_batch(observation)
        closure_batch = self._online_closure_batch(batch)
        correction, _, _, _ = self._closure_outputs(closure_batch)

        queue = getattr(self.native_policy, "_queues", None)
        before = len(queue[ACTION]) if queue is not None and ACTION in queue else 0
        replanned = before == 0
        normalized_global = self.native_policy.select_action(
            self._global_prepare(batch, include_action=False)
        )
        after = len(queue[ACTION]) if queue is not None and ACTION in queue else 0
        if replanned:
            self._chunk_len = after + 1
        self.last_replanned = replanned
        self.last_chunk_step = max(0, self._chunk_len - 1 - after)

        global_action = self.normalizer.denormalize_action(normalized_global)
        return global_action + correction

    def reset(self) -> None:
        self._previous_online_frame: dict[str, Tensor] | None = None
        self._online_batch_size: int | None = None
        self._chunk_len = 0
        self.last_replanned = True
        self.last_chunk_step = 0
        self.last_gate_mean = 0.0
        self.last_gate_std = 0.0
        self.last_closure_residual_magnitude = 0.0
        self.last_grasp_probability = 0.0
        self.native_policy.reset()


__all__ = ["DPBeaverClosurePolicy"]
