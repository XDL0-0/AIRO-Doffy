"""Replan-anchored relative-action ablation of WRM_wrap."""

from __future__ import annotations

import torch
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from torch import Tensor

from policies.realman_beaver.configuration import (
    WRAP_DELTA_BEAVER_VARIANT,
    RealmanBeaverConfig,
)
from policies.realman_beaver.dataset import ObservationNormalizer
from policies.realman_beaver.modeling_wrm_wrap import WrapBeaverDPPolicy


class WrapDeltaBeaverDPPolicy(WrapBeaverDPPolicy):
    """WRM_wrap with only the learned action representation changed.

    The diffusion target for every horizon step is ``q_target - q_anchor``,
    where ``q_anchor`` is the latest measured configuration at replan time.
    Generated deltas are converted back to absolute joint targets before the
    unchanged wrap/lift gate runs. No per-step cumulative integration is used.
    """

    def __init__(
        self, config: RealmanBeaverConfig, normalizer: ObservationNormalizer
    ) -> None:
        if config.model.variant != WRAP_DELTA_BEAVER_VARIANT:
            raise ValueError(
                "WrapDeltaBeaverDPPolicy requires "
                f"model.variant={WRAP_DELTA_BEAVER_VARIANT}"
            )
        if not normalizer.has_delta_action_statistics:
            raise ValueError(
                "WRM_wrap_delta requires train-split relative-action statistics"
            )
        super().__init__(config, normalizer)

    def _delta_target(self, batch: dict[str, Tensor]) -> Tensor:
        anchor = self._current_joint(batch)
        return self.normalizer.normalize_delta_action(
            batch["action"] - anchor.unsqueeze(1)
        )

    def _prepare(
        self, batch: dict[str, Tensor], include_action: bool
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        state, middle = self._condition(batch)
        prepared = {
            OBS_STATE: state,
            self.config.dataset.image_key: self.normalizer.normalize_image(
                batch["image"]
            ),
        }
        if include_action:
            prepared[ACTION] = self._delta_target(batch)
            prepared["action_is_pad"] = batch["action_is_pad"]
        return prepared, middle

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        loss, metrics = super().compute_loss(batch)
        with torch.no_grad():
            normalized_delta = self._delta_target(batch)
            valid = ~batch["action_is_pad"].bool()
            values = normalized_delta[valid]
            metrics["delta_action_std"] = float(
                values.std(unbiased=False) if values.numel() else 0.0
            )
        return loss, metrics

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        prepared, middle = self._prepare(batch, include_action=False)
        prepared[OBS_IMAGES] = torch.stack(
            [prepared[self.config.dataset.image_key]], dim=-4
        )
        normalized = self.native_policy.diffusion.generate_actions(prepared)
        anchor = self._current_joint(batch)
        actions = anchor.unsqueeze(1) + self.normalizer.denormalize_delta_action(
            normalized
        )
        gated_batch = dict(batch)
        gated_batch["state"] = anchor
        gated_steps = [
            self._apply_wrap_lift_gate(actions[:, step], gated_batch, middle)
            for step in range(actions.shape[1])
        ]
        return torch.stack(gated_steps, dim=1)

    @torch.no_grad()
    def select_action(self, observation: dict[str, Tensor]) -> Tensor:
        from policies.realman_beaver.modeling import _ensure_observation_batch

        batch = _ensure_observation_batch(observation)
        batch_size = batch["state"].shape[0]
        if self._batch_size is not None and self._batch_size != batch_size:
            self.reset()
        self._batch_size = batch_size
        temporal_batch = self._append_online_history(batch)
        queue = getattr(self.native_policy, "_queues", None)
        before = len(queue[ACTION]) if queue is not None and ACTION in queue else 0
        replanned = before == 0
        if replanned:
            self._action_anchor = batch["state"].detach().clone()
        prepared, middle = self._prepare(temporal_batch, include_action=False)
        normalized = self.native_policy.select_action(prepared)
        after = len(queue[ACTION]) if queue is not None and ACTION in queue else 0
        if replanned:
            self._chunk_len = after + 1
        if self._action_anchor is None:
            raise RuntimeError("relative-action chunk has no replan anchor")
        self.last_replanned = replanned
        self.last_chunk_step = max(0, self._chunk_len - 1 - after)
        action = self._action_anchor + self.normalizer.denormalize_delta_action(
            normalized
        )
        return self._apply_wrap_lift_gate(action, batch, middle)

    def reset(self) -> None:
        super().reset()
        self._action_anchor: Tensor | None = None

