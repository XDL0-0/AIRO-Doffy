"""Wrap-then-lift gated temporal Diffusion Policy."""

from __future__ import annotations

from collections import deque

import torch
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from torch import Tensor, nn

from policies.realman_beaver.configuration import (
    WRAP_BEAVER_VARIANTS,
    RealmanBeaverConfig,
)
from policies.realman_beaver.dataset import ObservationNormalizer, resolve_beaver_sensor_indices
from policies.realman_beaver.modules.wrap_beaver_encoder import WrapBeaverEncoder


class WrapBeaverDPPolicy(nn.Module):
    """Temporal Key4 DP that conditions on contact enclosure and gates lift.

    Binary grasp probability is not concatenated. The native DP sees joints,
    a 64-D temporal Beaver feature, and a 4-D enclosure vector. At execution,
    lift uses jaw enclosure (any near-field sensor on J3 and on J4) rather
    than 4/4 Key4 occupancy, so a geometrically blind sensor 11 cannot hold
    J1. J3/J4 freeze independently once their own group is near and in
    contact; a hold counter waits until that enclosure has lasted, so the
    first 10 mm reading during wrap does not freeze a small bottle. J5
    freezes after both closure groups have stopped.
    """

    def __init__(
        self, config: RealmanBeaverConfig, normalizer: ObservationNormalizer
    ) -> None:
        super().__init__()
        if config.model.variant not in WRAP_BEAVER_VARIANTS:
            raise ValueError(
                "WrapBeaverDPPolicy requires one of "
                f"{sorted(WRAP_BEAVER_VARIANTS)}"
            )
        self.config = config
        self.normalizer = normalizer
        model, dataset = config.model, config.dataset
        sensor_indices = resolve_beaver_sensor_indices(
            dataset, model.beaver_wrap_sensors
        )
        self.beaver_encoder = WrapBeaverEncoder(
            n_sensors=model.beaver_shape[0],
            sensor_indices=sensor_indices,
            history_steps=model.beaver_wrap_history_steps,
            valid_statuses=dataset.beaver_valid_statuses,
            proximity_scales_mm=model.beaver_wrap_proximity_scales_mm,
            near_threshold_mm=model.beaver_wrap_near_threshold_mm,
            closing_scale_mm=model.beaver_wrap_closing_scale_mm,
            range_scale_mm=model.beaver_wrap_range_scale_mm,
            frame_hidden_dim=model.beaver_wrap_frame_hidden_dim,
            frame_feature_dim=model.beaver_wrap_frame_feature_dim,
            temporal_hidden_dim=model.beaver_wrap_temporal_hidden_dim,
            output_dim=model.beaver_wrap_feature_dim,
            enclosure_dim=model.beaver_wrap_enclosure_dim,
        )
        from policies.realman_beaver.modeling import build_native_diffusion_config

        self.native_policy = DiffusionPolicy(build_native_diffusion_config(config))
        self.reset()

    def _condition(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        required = {
            "beaver_history_distance",
            "beaver_history_status",
            "beaver_history_present",
        }
        missing = required - batch.keys()
        if missing:
            raise KeyError(
                f"WRM_wrap batch is missing history fields: {sorted(missing)}"
            )
        beaver_feature, middle = self.beaver_encoder(
            batch["beaver_history_distance"],
            batch["beaver_history_status"],
            batch["beaver_history_present"],
            return_intermediates=True,
        )
        enclosure = middle["current_enclosure"]
        wrap_progress = middle["current_wrap_progress"]
        min_range_mm = middle["current_min_range_mm"]
        self.last_wrap_progress = float(wrap_progress.detach().mean())
        self.last_min_range_mm = float(min_range_mm.detach().mean())
        self.last_beaver_feature_mean = float(beaver_feature.detach().mean())
        self.last_beaver_feature_std = float(
            beaver_feature.detach().std(unbiased=False)
        )
        sensor_tokens = middle["sensor_tokens"]
        self.last_sensor_token_std = {
            sensor_name: float(
                sensor_tokens[..., sensor_position, :].detach().std(unbiased=False)
            )
            for sensor_position, sensor_name in enumerate(
                self.config.model.beaver_wrap_sensors
            )
        }
        conditioned_state = torch.cat(
            (
                self.normalizer.normalize_state(batch["state"]),
                beaver_feature,
                enclosure,
            ),
            dim=-1,
        )
        return conditioned_state, middle

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
            prepared[ACTION] = self.normalizer.normalize_action(batch["action"])
            prepared["action_is_pad"] = batch["action_is_pad"]
        return prepared, middle

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        prepared, middle = self._prepare(batch, include_action=True)
        diffusion_loss, _ = self.native_policy(prepared)
        wrap_progress = middle["current_wrap_progress"]
        min_range_mm = middle["current_min_range_mm"]
        contact_zero = middle["contact_zero"]
        genuine = middle["genuine"]
        return diffusion_loss, {
            "loss": float(diffusion_loss.detach()),
            "diffusion_loss": float(diffusion_loss.detach()),
            "wrap_progress": float(wrap_progress.detach().mean()),
            "min_range_mm": float(min_range_mm.detach().mean()),
            "contact_zero_fraction": float(contact_zero.float().mean()),
            "genuine_fraction": float(genuine.float().mean()),
            "beaver_feature_std": self.last_beaver_feature_std,
            **{
                f"beaver_sensor_{name}_token_std": std
                for name, std in self.last_sensor_token_std.items()
            },
        }

    def _current_joint(self, batch: dict[str, Tensor]) -> Tensor:
        state = batch["state"]
        if state.ndim == 3:
            return state[:, -1]
        return state

    def _jaw_gate_positions(self) -> tuple[list[int], list[int]]:
        model = self.config.model
        sensor_positions = {
            name: position
            for position, name in enumerate(model.beaver_wrap_sensors)
        }
        try:
            j3_positions = [
                sensor_positions[name] for name in model.beaver_wrap_j3_sensors
            ]
            j4_positions = [
                sensor_positions[name] for name in model.beaver_wrap_j4_sensors
            ]
        except KeyError as exc:  # pragma: no cover - config validation catches it
            raise ValueError(
                f"Wrap gate sensor group contains unknown sensor {exc.args[0]!r}"
            ) from exc
        return j3_positions, j4_positions

    def _update_enclosure_hold(
        self, enclosed: Tensor, sensor_min: Tensor
    ) -> Tensor:
        """Count consecutive both-jaw enclosure frames once per observation."""
        batch_size = enclosed.shape[0]
        if (
            self._enclosed_hold is None
            or self._enclosed_hold.shape[0] != batch_size
            or self._enclosed_hold.device != enclosed.device
        ):
            self._enclosed_hold = torch.zeros(
                batch_size, dtype=torch.long, device=enclosed.device
            )
        signature = (sensor_min.data_ptr(), tuple(sensor_min.shape))
        if signature != self._last_hold_signature:
            self._enclosed_hold = torch.where(
                enclosed,
                self._enclosed_hold + 1,
                torch.zeros_like(self._enclosed_hold),
            )
            self._last_hold_signature = signature
        return self._enclosed_hold

    def _apply_wrap_lift_gate(
        self, action: Tensor, batch: dict[str, Tensor], middle: dict[str, Tensor]
    ) -> Tensor:
        model = self.config.model
        sensor_min = middle["current_sensor_min_mm"]
        if sensor_min.ndim > 2:
            sensor_min = sensor_min.reshape(
                sensor_min.shape[0], -1, sensor_min.shape[-1]
            )[:, -1]
        joints = self._current_joint(batch)
        if sensor_min.shape[0] != action.shape[0]:
            raise ValueError("Wrap gate batch size does not match the action")
        gated = action.clone()
        j3_positions, j4_positions = self._jaw_gate_positions()
        near = sensor_min <= model.beaver_wrap_near_threshold_mm
        j3_wrap = near[:, j3_positions].to(dtype=sensor_min.dtype).mean(dim=-1)
        j4_wrap = near[:, j4_positions].to(dtype=sensor_min.dtype).mean(dim=-1)
        j3_contact = sensor_min[:, j3_positions].amin(dim=-1) <= (
            model.beaver_wrap_contact_stop_mm
        )
        j4_contact = sensor_min[:, j4_positions].amin(dim=-1) <= (
            model.beaver_wrap_contact_stop_mm
        )
        stop_close_j3_wrap = (
            model.beaver_wrap_stop_close_j3_wrap
            if model.beaver_wrap_stop_close_j3_wrap is not None
            else model.beaver_wrap_stop_close_wrap
        )
        stop_close_j4_wrap = (
            model.beaver_wrap_stop_close_j4_wrap
            if model.beaver_wrap_stop_close_j4_wrap is not None
            else model.beaver_wrap_stop_close_wrap
        )
        j3_enclosed = j3_wrap >= stop_close_j3_wrap
        j4_enclosed = j4_wrap >= stop_close_j4_wrap
        enclosed = j3_enclosed & j4_enclosed
        hold = self._update_enclosure_hold(enclosed, sensor_min)
        ready_j3 = j3_enclosed & j3_contact
        ready_j4 = j4_enclosed & j4_contact
        stop_hold = int(model.beaver_wrap_stop_hold_frames)
        lift_hold = int(model.beaver_wrap_lift_hold_frames)
        held_stop = hold >= stop_hold if stop_hold else torch.ones_like(enclosed)
        held_lift = hold >= lift_hold if lift_hold else torch.ones_like(enclosed)
        stop_close_j3 = ready_j3 & held_stop
        stop_close_j4 = ready_j4 & held_stop
        both_close_stopped = stop_close_j3 & stop_close_j4
        # Lift needs one near-field sensor on each jaw, not 4/4 Key4 cells.
        # Sensor 11 is frequently blind on large bottles and must not veto J1.
        jaw_wrap = (j3_enclosed.to(dtype=sensor_min.dtype) + j4_enclosed.to(
            dtype=sensor_min.dtype
        )) * 0.5
        block_lift = (jaw_wrap < model.beaver_wrap_lift_min_wrap) | (~held_lift)
        lift_joint = 1
        gated[:, lift_joint] = torch.where(
            block_lift, joints[:, lift_joint], gated[:, lift_joint]
        )
        for joint, stop_close in (
            (3, stop_close_j3),
            (4, stop_close_j4),
            (5, both_close_stopped),
        ):
            gated[:, joint] = torch.where(
                stop_close, joints[:, joint], gated[:, joint]
            )
        self.last_lift_blocked = float(block_lift.float().mean())
        self.last_close_stopped_j3 = float(stop_close_j3.float().mean())
        self.last_close_stopped_j4 = float(stop_close_j4.float().mean())
        self.last_close_stopped = float(both_close_stopped.float().mean())
        self.last_jaw_wrap = float(jaw_wrap.mean())
        self.last_enclosed_hold = float(hold.float().mean())
        return gated

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        prepared, middle = self._prepare(batch, include_action=False)
        prepared[OBS_IMAGES] = torch.stack(
            [prepared[self.config.dataset.image_key]], dim=-4
        )
        normalized = self.native_policy.diffusion.generate_actions(prepared)
        actions = self.normalizer.denormalize_action(normalized)
        current = self._current_joint(batch).unsqueeze(1).expand_as(actions)
        gated_batch = dict(batch)
        gated_batch["state"] = current[:, 0]
        gated_steps = [
            self._apply_wrap_lift_gate(actions[:, step], gated_batch, middle)
            for step in range(actions.shape[1])
        ]
        return torch.stack(gated_steps, dim=1)

    @torch.no_grad()
    def predict_actions(self, batch: dict[str, Tensor]) -> Tensor:
        return self.predict_action_chunk(batch)

    def _append_online_history(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        signature = tuple(
            batch[key].data_ptr()
            for key in ("beaver_distance", "beaver_status", "beaver_present")
        )
        frame = {
            "distance": batch["beaver_distance"],
            "status": batch["beaver_status"],
            "present": batch["beaver_present"],
        }
        if signature != self._last_online_frame_signature:
            self._beaver_history.append(frame)
            self._last_online_frame_signature = signature
        frames = list(self._beaver_history)
        padded_frames = [frames[0]] * (
            self.config.model.beaver_wrap_history_steps - len(frames)
        ) + frames
        temporal_batch = dict(batch)
        for output_key, frame_key in (
            ("beaver_history_distance", "distance"),
            ("beaver_history_status", "status"),
            ("beaver_history_present", "present"),
        ):
            temporal_batch[output_key] = torch.stack(
                [history_frame[frame_key] for history_frame in padded_frames],
                dim=1,
            )
        return temporal_batch

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
        prepared, middle = self._prepare(temporal_batch, include_action=False)
        normalized = self.native_policy.select_action(prepared)
        after = len(queue[ACTION]) if queue is not None and ACTION in queue else 0
        if replanned:
            self._chunk_len = after + 1
        self.last_replanned = replanned
        self.last_chunk_step = max(0, self._chunk_len - 1 - after)
        action = self.normalizer.denormalize_action(normalized)
        return self._apply_wrap_lift_gate(action, batch, middle)

    def reset(self) -> None:
        self._beaver_history: deque[dict[str, Tensor]] = deque(
            maxlen=self.config.model.beaver_wrap_history_steps
        )
        self._batch_size: int | None = None
        self._last_online_frame_signature: tuple[int, ...] | None = None
        self.last_replanned = True
        self.last_chunk_step = 0
        self.last_wrap_progress = 0.0
        self.last_min_range_mm = 0.0
        self.last_lift_blocked = 0.0
        self.last_close_stopped = 0.0
        self.last_close_stopped_j3 = 0.0
        self.last_close_stopped_j4 = 0.0
        self.last_jaw_wrap = 0.0
        self.last_enclosed_hold = 0.0
        self._enclosed_hold: Tensor | None = None
        self._last_hold_signature: tuple[int, tuple[int, ...]] | None = None
        self.last_beaver_feature_mean = 0.0
        self.last_beaver_feature_std = 0.0
        self.last_sensor_token_std = {
            sensor_name: 0.0 for sensor_name in self.config.model.beaver_wrap_sensors
        }
        self._chunk_len = 0
        self.native_policy.reset()
