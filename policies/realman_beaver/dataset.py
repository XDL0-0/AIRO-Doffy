"""LeRobot dataset adapters and normalization for Realman-Beaver policies."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch import Tensor, nn
from torch.utils.data import Dataset, Sampler

from policies.realman_beaver.configuration import (
    ADAPTIVE_BEAVER_VARIANT,
    ANTIGRAVITY_BEAVER_VARIANT,
    BEAVER_CLOSURE_VARIANT,
    CLAUDE_BEAVER_VARIANT,
    CODEX_BEAVER_VARIANT,
    DELTA_BEAVER_VARIANT,
    GRASP_STATE_VARIANTS,
    GROK_BEAVER_VARIANT,
    HISTORY_BEAVER_VARIANTS,
    MOTION_DELTA_VARIANTS,
    QWEN_BEAVER_VARIANT,
    RELATIVE_ACTION_VARIANTS,
    STRUCTURED_BEAVER_DP_VARIANTS,
    TEMPORAL_BEAVER_VARIANT,
    WRAP_BEAVER_VARIANT,
    WRAP_BEAVER_VARIANTS,
    WRAP_DELTA_BEAVER_VARIANT,
    LOBO_MONITOR_BEAVER_VARIANT,
    WRAP_MONITOR_BACKUP_BEAVER_VARIANT,
    WRAP_MONITOR_BEAVER_VARIANT,
    DatasetConfig,
    RealmanBeaverConfig,
)

_DIRECT_BEAVER_VARIANTS = {
    "dp_beaver",
    "fm_beaver",
    BEAVER_CLOSURE_VARIANT,
    *STRUCTURED_BEAVER_DP_VARIANTS,
    TEMPORAL_BEAVER_VARIANT,
    DELTA_BEAVER_VARIANT,
    ADAPTIVE_BEAVER_VARIANT,
    *HISTORY_BEAVER_VARIANTS,
}


def history_beaver_sensor_names(model) -> tuple[str, ...]:
    names = {
        ADAPTIVE_BEAVER_VARIANT: model.beaver_adaptive_sensors,
        ANTIGRAVITY_BEAVER_VARIANT: model.beaver_antigravity_sensors,
        GROK_BEAVER_VARIANT: model.beaver_grok_sensors,
        CODEX_BEAVER_VARIANT: model.codex_beaver_sensors,
        CLAUDE_BEAVER_VARIANT: model.claude_sensors,
        WRAP_BEAVER_VARIANT: model.beaver_wrap_sensors,
        WRAP_DELTA_BEAVER_VARIANT: model.beaver_wrap_sensors,
        WRAP_MONITOR_BEAVER_VARIANT: model.beaver_wrap_sensors,
        WRAP_MONITOR_BACKUP_BEAVER_VARIANT: model.beaver_wrap_sensors,
        LOBO_MONITOR_BEAVER_VARIANT: model.beaver_wrap_sensors,
    }
    return tuple(names.get(model.variant, model.beaver_temporal_sensors))


def history_beaver_steps(model) -> int:
    steps = {
        ADAPTIVE_BEAVER_VARIANT: model.beaver_adaptive_history_steps,
        ANTIGRAVITY_BEAVER_VARIANT: model.beaver_antigravity_history_steps,
        GROK_BEAVER_VARIANT: model.beaver_grok_history_steps,
        CODEX_BEAVER_VARIANT: model.codex_beaver_history_steps,
        CLAUDE_BEAVER_VARIANT: model.claude_history_steps,
        WRAP_BEAVER_VARIANT: model.beaver_wrap_history_steps,
        WRAP_DELTA_BEAVER_VARIANT: model.beaver_wrap_history_steps,
        WRAP_MONITOR_BEAVER_VARIANT: model.beaver_wrap_history_steps,
        WRAP_MONITOR_BACKUP_BEAVER_VARIANT: model.beaver_wrap_history_steps,
        LOBO_MONITOR_BEAVER_VARIANT: model.beaver_wrap_history_steps,
    }
    return int(steps.get(model.variant, model.beaver_history_steps))


def motion_lookback_steps(model) -> int | None:
    """Extra joint frames needed beyond ``n_obs_steps``, if any."""
    variant = model.variant
    if variant == ADAPTIVE_BEAVER_VARIANT:
        return int(model.beaver_adaptive_motion_delta_steps)
    if variant == GROK_BEAVER_VARIANT:
        return int(model.beaver_grok_motion_delta_steps)
    if variant == CLAUDE_BEAVER_VARIANT:
        return int(model.claude_motion_delta_steps)
    if variant == DELTA_BEAVER_VARIANT:
        return int(model.beaver_delta_steps)
    if variant == ANTIGRAVITY_BEAVER_VARIANT:
        return int(
            max(
                model.beaver_antigravity_motion_delta_steps,
                model.beaver_antigravity_motion_delta_long_steps,
            )
        )
    return None


class ObservationNormalizer(nn.Module):
    """Normalize raw LeRobot values before they enter a LeRobot policy."""

    _TEMPORAL_BUFFER_NAMES = (
        "beaver_temporal_p5",
        "beaver_temporal_p95",
        "beaver_temporal_median",
        "beaver_temporal_sensor_indices",
    )
    _DELTA_BUFFER_NAMES = (
        "beaver_delta_mean",
        "beaver_delta_std",
        "beaver_delta_sensor_indices",
    )
    _ACTION_DELTA_BUFFER_NAMES = ("action_delta_scale",)
    _QWEN_BUFFER_NAMES = (
        "delta_action_offset",
        "delta_action_scale",
    )

    def __init__(
        self,
        state_offset: Tensor,
        state_scale: Tensor,
        action_offset: Tensor,
        action_scale: Tensor,
        image_mean: Tensor,
        image_std: Tensor,
        distance_max_mm: float,
        valid_statuses: tuple[int, ...] = (5, 9),
        temporal_beaver_statistics: dict[str, Tensor] | None = None,
        delta_beaver_statistics: dict[str, Tensor] | None = None,
        action_delta_statistics: dict[str, Tensor] | None = None,
        delta_action_statistics: dict[str, Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("state_offset", state_offset.float())
        self.register_buffer("state_scale", state_scale.float())
        self.register_buffer("action_offset", action_offset.float())
        self.register_buffer("action_scale", action_scale.float())
        self.register_buffer("image_mean", image_mean.float().reshape(3, 1, 1))
        self.register_buffer("image_std", image_std.float().reshape(3, 1, 1))
        self.register_buffer("distance_max_mm", torch.tensor(float(distance_max_mm)))
        temporal_beaver_statistics = temporal_beaver_statistics or {}
        self.register_buffer(
            "beaver_temporal_p5",
            temporal_beaver_statistics.get("p5", torch.empty(0)).float(),
        )
        self.register_buffer(
            "beaver_temporal_p95",
            temporal_beaver_statistics.get("p95", torch.empty(0)).float(),
        )
        self.register_buffer(
            "beaver_temporal_median",
            temporal_beaver_statistics.get("median", torch.empty(0)).float(),
        )
        self.register_buffer(
            "beaver_temporal_sensor_indices",
            temporal_beaver_statistics.get(
                "sensor_indices", torch.empty(0, dtype=torch.long)
            ).long(),
        )
        delta_beaver_statistics = delta_beaver_statistics or {}
        self.register_buffer(
            "beaver_delta_mean",
            delta_beaver_statistics.get("mean", torch.empty(0)).float(),
        )
        self.register_buffer(
            "beaver_delta_std",
            delta_beaver_statistics.get("std", torch.empty(0)).float(),
        )
        self.register_buffer(
            "beaver_delta_sensor_indices",
            delta_beaver_statistics.get(
                "sensor_indices", torch.empty(0, dtype=torch.long)
            ).long(),
        )
        action_delta_statistics = action_delta_statistics or {}
        self.register_buffer(
            "action_delta_scale",
            action_delta_statistics.get("scale", torch.empty(0)).float(),
        )
        delta_action_statistics = delta_action_statistics or {}
        self.register_buffer(
            "delta_action_offset",
            delta_action_statistics.get("offset", torch.empty(0)).float(),
        )
        self.register_buffer(
            "delta_action_scale",
            delta_action_statistics.get("scale", torch.empty(0)).float(),
        )
        self._valid_statuses = tuple(valid_statuses)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Accept checkpoints created before optional temporal buffers existed."""
        for name in (
            *self._TEMPORAL_BUFFER_NAMES,
            *self._DELTA_BUFFER_NAMES,
            *self._ACTION_DELTA_BUFFER_NAMES,
            *self._QWEN_BUFFER_NAMES,
        ):
            key = f"{prefix}{name}"
            buffer = getattr(self, name)
            if key not in state_dict and buffer.numel() == 0:
                state_dict[key] = buffer
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @classmethod
    def from_lerobot_dataset(
        cls,
        config: RealmanBeaverConfig,
        episodes: Sequence[int] | None = None,
    ) -> ObservationNormalizer:
        """Build normalization statistics, optionally from selected episodes only."""
        dataset, model = config.dataset, config.model
        if dataset.normalization_source == "metadata":
            normalizer = cls._from_metadata(config)
        else:
            paths = sorted(
                (Path(dataset.root).expanduser() / "data").rglob("*.parquet")
            )
            if not paths:
                raise FileNotFoundError(
                    f"No LeRobot parquet shards found under {dataset.root}/data"
                )
            keys = (dataset.state_key, dataset.action_key)
            chunks: dict[str, list[np.ndarray]] = {key: [] for key in keys}
            included = set(episodes) if episodes is not None else None
            for path in paths:
                columns = (
                    [*keys, "episode_index"] if included is not None else list(keys)
                )
                table = pq.read_table(path, columns=columns)
                selected: np.ndarray | slice = slice(None)
                if included is not None:
                    episode_index = np.asarray(table["episode_index"]).reshape(-1)
                    selected = np.isin(episode_index, tuple(included))
                    if not selected.any():
                        continue
                for key in keys:
                    values = np.asarray(table[key].to_pylist(), dtype=np.float32)
                    chunks[key].append(values[selected])
            if any(not values for values in chunks.values()):
                raise ValueError(
                    "No state/action rows matched the normalization episodes"
                )
            state = np.concatenate(chunks[dataset.state_key], axis=0)
            action = np.concatenate(chunks[dataset.action_key], axis=0)
            if state.shape[1:] != (model.state_dim,) or action.shape[1:] != (
                model.action_dim,
            ):
                raise ValueError(
                    f"Unexpected state/action shapes: {state.shape}, {action.shape}"
                )
            normalizer = cls._from_arrays(state, action, dataset)

        if (
            model.variant in HISTORY_BEAVER_VARIANTS
            and model.variant not in WRAP_BEAVER_VARIANTS
        ):
            normalizer.set_temporal_beaver_statistics(
                fit_temporal_beaver_statistics(config, episodes)
            )
        if model.variant == DELTA_BEAVER_VARIANT:
            normalizer.set_delta_beaver_statistics(
                fit_delta_beaver_statistics(config, episodes)
            )
        if model.variant == CLAUDE_BEAVER_VARIANT:
            normalizer.set_action_delta_statistics(
                fit_action_delta_statistics(config, episodes)
            )
        if model.variant in RELATIVE_ACTION_VARIANTS:
            normalizer.set_delta_action_statistics(
                fit_delta_action_statistics(config, episodes)
            )
        return normalizer

    @classmethod
    def _from_metadata(cls, config: RealmanBeaverConfig) -> ObservationNormalizer:
        dataset, model = config.dataset, config.model
        path = Path(dataset.root).expanduser() / "meta" / "stats.json"
        with path.open("r", encoding="utf-8") as stream:
            stats = json.load(stream)

        def value(key: str, field: str, dimension: int) -> Tensor:
            result = torch.as_tensor(stats[key][field], dtype=torch.float32).flatten()
            if result.numel() != dimension:
                raise ValueError(
                    f"{key}.{field} has {result.numel()} values, expected {dimension}"
                )
            return result

        state_min = value(dataset.state_key, "min", model.state_dim)
        state_max = value(dataset.state_key, "max", model.state_dim)
        action_min = value(dataset.action_key, "min", model.action_dim)
        action_max = value(dataset.action_key, "max", model.action_dim)
        image_mean = value(dataset.image_key, "mean", 3)
        image_std = value(dataset.image_key, "std", 3).clamp_min(
            dataset.normalization_floor
        )
        return cls._from_tensors(
            state_min,
            state_max,
            action_min,
            action_max,
            image_mean,
            image_std,
            dataset,
        )

    @classmethod
    def _from_arrays(
        cls,
        state: np.ndarray,
        action: np.ndarray,
        dataset: DatasetConfig,
    ) -> ObservationNormalizer:
        path = Path(dataset.root).expanduser() / "meta" / "stats.json"
        with path.open("r", encoding="utf-8") as stream:
            stats = json.load(stream)
        image_mean = torch.as_tensor(stats[dataset.image_key]["mean"]).flatten()
        image_std = torch.as_tensor(stats[dataset.image_key]["std"]).flatten()
        return cls._from_tensors(
            torch.from_numpy(state.min(axis=0)),
            torch.from_numpy(state.max(axis=0)),
            torch.from_numpy(action.min(axis=0)),
            torch.from_numpy(action.max(axis=0)),
            image_mean,
            image_std.clamp_min(dataset.normalization_floor),
            dataset,
        )

    @classmethod
    def _from_tensors(
        cls,
        state_min: Tensor,
        state_max: Tensor,
        action_min: Tensor,
        action_max: Tensor,
        image_mean: Tensor,
        image_std: Tensor,
        dataset: DatasetConfig,
    ) -> ObservationNormalizer:
        state_offset = (state_min + state_max) / 2.0
        state_scale = ((state_max - state_min) / 2.0).clamp_min(
            dataset.normalization_floor
        )
        action_offset = (action_min + action_max) / 2.0
        action_scale = ((action_max - action_min) / 2.0).clamp_min(
            dataset.normalization_floor
        )
        return cls(
            state_offset,
            state_scale,
            action_offset,
            action_scale,
            image_mean,
            image_std,
            dataset.distance_max_mm,
            dataset.beaver_valid_statuses,
        )

    @classmethod
    def identity(
        cls,
        state_dim: int = 7,
        action_dim: int = 7,
        temporal_beaver_statistics: dict[str, Tensor] | None = None,
        delta_beaver_statistics: dict[str, Tensor] | None = None,
        action_delta_statistics: dict[str, Tensor] | None = None,
        delta_action_statistics: dict[str, Tensor] | None = None,
    ) -> ObservationNormalizer:
        return cls(
            torch.zeros(state_dim),
            torch.ones(state_dim),
            torch.zeros(action_dim),
            torch.ones(action_dim),
            torch.zeros(3),
            torch.ones(3),
            2550.0,
            temporal_beaver_statistics=temporal_beaver_statistics,
            delta_beaver_statistics=delta_beaver_statistics,
            action_delta_statistics=action_delta_statistics,
            delta_action_statistics=delta_action_statistics,
        )

    def set_temporal_beaver_statistics(self, statistics: dict[str, Tensor]) -> None:
        """Install train-split-only robust Beaver bounds as checkpoint buffers."""
        required = {"p5", "p95", "median", "sensor_indices"}
        missing = required - set(statistics)
        if missing:
            raise ValueError(
                f"Temporal Beaver statistics are missing: {sorted(missing)}"
            )
        device = self.beaver_temporal_p5.device
        p5 = torch.as_tensor(
            statistics["p5"], dtype=torch.float32, device=device
        ).flatten()
        p95 = torch.as_tensor(
            statistics["p95"], dtype=torch.float32, device=device
        ).flatten()
        median = torch.as_tensor(
            statistics["median"], dtype=torch.float32, device=device
        ).flatten()
        indices = torch.as_tensor(
            statistics["sensor_indices"], dtype=torch.long, device=device
        ).flatten()
        if not len(p5) or not (p5.shape == p95.shape == median.shape == indices.shape):
            raise ValueError(
                "Temporal Beaver statistics must have one value per sensor"
            )
        if torch.any(p95 <= p5):
            raise ValueError("Temporal Beaver P95 must be greater than P5")
        self.beaver_temporal_p5 = p5
        self.beaver_temporal_p95 = p95
        self.beaver_temporal_median = median
        self.beaver_temporal_sensor_indices = indices

    @property
    def has_temporal_beaver_statistics(self) -> bool:
        return bool(self.beaver_temporal_p5.numel())

    def set_delta_beaver_statistics(self, statistics: dict[str, Tensor]) -> None:
        """Install per-sensor train-split mean/std as checkpoint buffers."""
        required = {"mean", "std", "sensor_indices"}
        missing = required - set(statistics)
        if missing:
            raise ValueError(f"Delta Beaver statistics are missing: {sorted(missing)}")
        device = self.beaver_delta_mean.device
        mean = torch.as_tensor(
            statistics["mean"], dtype=torch.float32, device=device
        ).flatten()
        std = torch.as_tensor(
            statistics["std"], dtype=torch.float32, device=device
        ).flatten()
        indices = torch.as_tensor(
            statistics["sensor_indices"], dtype=torch.long, device=device
        ).flatten()
        if not len(mean) or not (mean.shape == std.shape == indices.shape):
            raise ValueError("Delta Beaver statistics must have one value per sensor")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("Delta Beaver statistics must be finite")
        if torch.any(std <= 0):
            raise ValueError("Delta Beaver standard deviations must be positive")
        self.beaver_delta_mean = mean
        self.beaver_delta_std = std
        self.beaver_delta_sensor_indices = indices

    @property
    def has_delta_beaver_statistics(self) -> bool:
        return bool(self.beaver_delta_mean.numel())

    def set_action_delta_statistics(self, statistics: dict[str, Tensor]) -> None:
        """Install per-joint train-split action-delta scales as checkpoint buffers."""
        required = {"scale"}
        missing = required - set(statistics)
        if missing:
            raise ValueError(f"Action delta statistics are missing: {sorted(missing)}")
        scale = torch.as_tensor(
            statistics["scale"],
            dtype=torch.float32,
            device=self.action_delta_scale.device,
        ).flatten()
        if not len(scale) or scale.numel() != self.action_scale.numel():
            raise ValueError("Action delta scale must have one value per joint")
        if not torch.isfinite(scale).all() or torch.any(scale <= 0):
            raise ValueError("Action delta scales must be finite and positive")
        self.action_delta_scale = scale

    @property
    def has_action_delta_statistics(self) -> bool:
        return bool(self.action_delta_scale.numel())

    def normalize_action_delta(self, value: Tensor) -> Tensor:
        if not self.has_action_delta_statistics:
            raise RuntimeError("Action delta normalization statistics are not set")
        return value / self.action_delta_scale

    def denormalize_action_delta(self, value: Tensor) -> Tensor:
        if not self.has_action_delta_statistics:
            raise RuntimeError("Action delta normalization statistics are not set")
        return value.clamp(-1.0, 1.0) * self.action_delta_scale

    def set_delta_action_statistics(
        self,
        statistics: dict[str, Tensor],
        floor: float = 1e-4,
    ) -> None:
        """Install per-joint train-split relative-action min/max as buffers."""
        required = {"min", "max"}
        missing = required - set(statistics)
        if missing:
            raise ValueError(f"Delta action statistics are missing: {sorted(missing)}")
        device = self.delta_action_offset.device
        minimum = torch.as_tensor(
            statistics["min"], dtype=torch.float32, device=device
        ).flatten()
        maximum = torch.as_tensor(
            statistics["max"], dtype=torch.float32, device=device
        ).flatten()
        if minimum.numel() != maximum.numel() or minimum.numel() == 0:
            raise ValueError(
                "Delta action statistics must contain one (min, max) pair "
                "per action dimension"
            )
        if not (torch.isfinite(minimum).all() and torch.isfinite(maximum).all()):
            raise ValueError("Delta action statistics must be finite")
        if torch.any(maximum <= minimum):
            raise ValueError("Delta action max must exceed min per joint")
        self.delta_action_offset = (minimum + maximum) / 2.0
        self.delta_action_scale = ((maximum - minimum) / 2.0).clamp_min(float(floor))

    @property
    def has_delta_action_statistics(self) -> bool:
        return bool(self.delta_action_offset.numel())

    def normalize_delta_action(self, value: Tensor) -> Tensor:
        if not self.has_delta_action_statistics:
            raise RuntimeError("Delta action normalization statistics are not fitted")
        return (value - self.delta_action_offset) / self.delta_action_scale

    def denormalize_delta_action(self, value: Tensor) -> Tensor:
        if not self.has_delta_action_statistics:
            raise RuntimeError("Delta action normalization statistics are not fitted")
        return (
            value.clamp(-1.0, 1.0) * self.delta_action_scale + self.delta_action_offset
        )

    def normalize_state(self, value: Tensor) -> Tensor:
        return (value - self.state_offset) / self.state_scale

    def normalize_image(self, value: Tensor) -> Tensor:
        return (value - self.image_mean) / self.image_std

    def normalize_action(self, value: Tensor) -> Tensor:
        return (value - self.action_offset) / self.action_scale

    def denormalize_action(self, value: Tensor) -> Tensor:
        return value.clamp(-1.0, 1.0) * self.action_scale + self.action_offset

    def normalize_beaver(
        self, value: Tensor, present: Tensor, status: Tensor | None = None
    ) -> Tensor:
        distance = torch.minimum(value.clamp_min(0.0), self.distance_max_mm)
        distance = (distance / self.distance_max_mm) * 2.0 - 1.0
        # Sensor-level mask: a disconnected sensor reads zero everywhere.
        distance = distance * present[..., :, None, None].clamp(0.0, 1.0)
        # Pixel-level mask: VL53L7CX status codes 5 (valid) and 9
        # (weak-signal) carry usable distances; no-target (255) pixels are
        # filtered out the same way (zeroed).
        if status is not None:
            valid = torch.zeros_like(distance)
            for code in self._valid_statuses:
                valid = torch.maximum(valid, (status == code).to(distance.dtype))
            distance = distance * valid
        return distance

    def normalize_temporal_beaver(
        self,
        value: Tensor,
        present: Tensor,
        status: Tensor,
    ) -> Tensor:
        """Build distance/delta/valid/zero features for temporal Beaver input.

        Inputs end in ``[history, sensor, 4, 4]`` and may contain either all
        physical sensors or the already-selected sensors. The returned shape
        appends a four-feature axis. Missing continuous readings are forward
        filled inside each history only, with the train-split sensor median as
        the initial neutral value. Delta is nonzero only for adjacent observed
        (valid, present, finite, non-zero) measurements.
        """
        if not self.has_temporal_beaver_statistics:
            raise RuntimeError("Temporal Beaver normalization statistics are not set")
        if value.ndim < 4 or value.shape[-2:] != (4, 4):
            raise ValueError(
                "temporal Beaver distance must end in [history, sensor, 4, 4]"
            )
        if status.shape != value.shape:
            raise ValueError("temporal Beaver distance/status shapes must match")
        if present.shape != value.shape[:-2]:
            raise ValueError(
                "temporal Beaver presence must match distance through sensor axis"
            )

        sensor_count = self.beaver_temporal_p5.numel()
        if value.shape[-3] != sensor_count:
            indices = self.beaver_temporal_sensor_indices.to(value.device)
            if not len(indices) or int(indices.max()) >= value.shape[-3]:
                raise ValueError(
                    "temporal Beaver inputs do not contain configured sensor slots"
                )
            value = value.index_select(-3, indices)
            status = status.index_select(-3, indices)
            present = present.index_select(-1, indices)

        statistic_shape = (1,) * (value.ndim - 3) + (sensor_count, 1, 1)
        p5 = self.beaver_temporal_p5.to(value.device).reshape(statistic_shape)
        p95 = self.beaver_temporal_p95.to(value.device).reshape(statistic_shape)
        median = self.beaver_temporal_median.to(value.device).reshape(statistic_shape)
        scale = p95 - p5
        normalized = ((value - p5) / scale).clamp(0.0, 1.0)
        median_normalized = ((median - p5) / scale).clamp(0.0, 1.0)

        status_valid = torch.zeros_like(status, dtype=torch.bool)
        for code in self._valid_statuses:
            status_valid |= status == code
        valid = status_valid & (present[..., None, None] > 0)
        zero = value == 0
        observed = valid & ~zero & torch.isfinite(value)

        history_axis = value.ndim - 4
        history_steps = value.shape[history_axis]
        normalized_steps = normalized.unbind(dim=history_axis)
        observed_steps = observed.unbind(dim=history_axis)
        valid_steps = valid.unbind(dim=history_axis)
        zero_steps = zero.unbind(dim=history_axis)

        # After removing the history axis, median_normalized still contains a
        # singleton at that position. Removing it makes the carry state match a
        # single history frame for arbitrary leading batch/observation axes.
        previous = median_normalized.squeeze(history_axis)
        previous_observed = torch.zeros_like(previous, dtype=torch.bool)
        feature_steps: list[Tensor] = []
        for index in range(history_steps):
            current_observed = observed_steps[index]
            current = torch.where(current_observed, normalized_steps[index], previous)
            delta_valid = current_observed & previous_observed
            delta = torch.where(
                delta_valid, current - previous, torch.zeros_like(current)
            )
            feature_steps.append(
                torch.stack(
                    (
                        current,
                        delta,
                        valid_steps[index].to(current.dtype),
                        zero_steps[index].to(current.dtype),
                    ),
                    dim=-1,
                )
            )
            previous = current
            previous_observed = current_observed
        return torch.stack(feature_steps, dim=history_axis)

    def augmented_state(
        self,
        state: Tensor,
        distance: Tensor,
        present: Tensor,
        status: Tensor | None = None,
    ) -> Tensor:
        normalized_state = self.normalize_state(state)
        normalized_distance = self.normalize_beaver(distance, present, status).flatten(
            start_dim=-3
        )
        return torch.cat(
            (normalized_state, normalized_distance, present.clamp(0.0, 1.0)), dim=-1
        )


class LatentNormalizer(nn.Module):
    def __init__(self, offset: Tensor, scale: Tensor) -> None:
        super().__init__()
        self.register_buffer("offset", offset.float())
        self.register_buffer("scale", scale.float())

    @classmethod
    def from_latents(cls, latent: Tensor, floor: float = 1e-4) -> LatentNormalizer:
        minimum = latent.amin(dim=(0, 1))
        maximum = latent.amax(dim=(0, 1))
        return cls(
            (minimum + maximum) / 2.0, ((maximum - minimum) / 2.0).clamp_min(floor)
        )

    @classmethod
    def identity(cls, latent_dim: int) -> LatentNormalizer:
        return cls(torch.zeros(latent_dim), torch.ones(latent_dim))

    def normalize(self, value: Tensor) -> Tensor:
        return (value - self.offset) / self.scale

    def denormalize(self, value: Tensor) -> Tensor:
        return value.clamp(-1.0, 1.0) * self.scale + self.offset


def episode_split(config: DatasetConfig) -> tuple[list[int], list[int]]:
    info_path = Path(config.root).expanduser() / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as stream:
        total = int(json.load(stream)["total_episodes"])
    episodes = list(range(total))
    if config.val_episodes is not None:
        validation = sorted(config.val_episodes)
        invalid = [episode for episode in validation if episode >= total]
        if invalid:
            raise ValueError(
                f"dataset.val_episodes contains indices outside [0, {total - 1}]: {invalid}"
            )
        validation_set = set(validation)
        training = [episode for episode in episodes if episode not in validation_set]
        if not training:
            raise ValueError(
                "dataset.val_episodes must leave at least one training episode"
            )
        return training, validation
    random.Random(config.split_seed).shuffle(episodes)
    validation_count = round(total * config.val_fraction)
    if config.val_fraction > 0 and total > 1:
        validation_count = max(1, min(total - 1, validation_count))
    return sorted(episodes[validation_count:]), sorted(episodes[:validation_count])


def bottle_id_from_episode(
    episode_index: np.ndarray, episodes_per_bottle: int = 25
) -> np.ndarray:
    """Map LeRobot episode indices onto consecutive bottle blocks."""
    if episodes_per_bottle <= 0:
        raise ValueError("episodes_per_bottle must be positive")
    return np.asarray(episode_index, dtype=np.int64) // int(episodes_per_bottle)


class BottleStratifiedBatchSampler(Sampler[list[int]]):
    """Yield batches that contain (almost) equal counts from every bottle."""

    def __init__(
        self,
        bottle_ids: Sequence[int],
        batch_size: int,
        *,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        ids = np.asarray(bottle_ids, dtype=np.int64)
        if ids.ndim != 1 or ids.size == 0:
            raise ValueError("bottle_ids must be a non-empty 1-D array")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        groups = {
            int(bottle): np.flatnonzero(ids == bottle).tolist()
            for bottle in sorted(np.unique(ids).tolist())
        }
        if any(not members for members in groups.values()):
            raise ValueError("every bottle group must contain at least one index")
        if batch_size < len(groups):
            raise ValueError("batch_size must be at least the number of bottles")
        self.groups = groups
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.n_groups = len(groups)
        self.base = self.batch_size // self.n_groups
        self.extra = self.batch_size % self.n_groups
        self._epoch = 0
        if drop_last:
            self._length = len(ids) // self.batch_size
        else:
            self._length = (len(ids) + self.batch_size - 1) // self.batch_size

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self._epoch)
        self._epoch += 1
        shuffled = {
            bottle: rng.permutation(members).tolist()
            for bottle, members in self.groups.items()
        }
        pointers = {bottle: 0 for bottle in self.groups}
        bottles = list(self.groups)
        for _ in range(self._length):
            rng.shuffle(bottles)
            batch: list[int] = []
            for slot, bottle in enumerate(bottles):
                take = self.base + (1 if slot < self.extra else 0)
                source = shuffled[bottle]
                for _ in range(take):
                    if pointers[bottle] >= len(source):
                        shuffled[bottle] = rng.permutation(self.groups[bottle]).tolist()
                        pointers[bottle] = 0
                        source = shuffled[bottle]
                    batch.append(source[pointers[bottle]])
                    pointers[bottle] += 1
            rng.shuffle(batch)
            yield batch


def resolve_beaver_sensor_indices(
    dataset_config: DatasetConfig,
    sensor_names: Sequence[str],
) -> tuple[int, ...]:
    """Resolve physical names through the configured tensor-slot layout."""
    layout = tuple(dataset_config.beaver_sensor_layout)
    requested = tuple(str(name) for name in sensor_names)
    if not requested:
        raise ValueError("At least one Beaver sensor name must be selected")
    if len(set(requested)) != len(requested):
        raise ValueError("Selected Beaver sensor names must be unique")
    lookup = {name: index for index, name in enumerate(layout)}
    unknown = [name for name in requested if name not in lookup]
    if unknown:
        raise ValueError(
            f"Unknown Beaver sensor names {unknown}; configured layout is {layout}"
        )
    return tuple(lookup[name] for name in requested)


def build_delta_pairs(
    combined_history: Tensor,
    *,
    n_obs_steps: int,
    delta_steps: int,
) -> tuple[Tensor, Tensor]:
    """Return current and t-k frames from an episode-clamped O+k sequence."""
    if n_obs_steps <= 0 or delta_steps <= 0:
        raise ValueError("n_obs_steps and delta_steps must be positive")
    expected = n_obs_steps + delta_steps
    if combined_history.shape[0] != expected:
        raise ValueError(
            f"combined delta history has {combined_history.shape[0]} steps, "
            f"expected {expected}"
        )
    return combined_history[delta_steps:], combined_history[:n_obs_steps]


def fit_delta_beaver_statistics(
    config: RealmanBeaverConfig,
    episodes: Sequence[int],
) -> dict[str, Tensor]:
    """Fit per-sensor mean/std on valid, present, finite, nonzero train cells."""
    dataset, model = config.dataset, config.model
    sensor_indices = resolve_beaver_sensor_indices(dataset, model.beaver_delta_sensors)
    if episodes is None:
        raise ValueError(
            "Delta Beaver statistics require explicit training episodes to "
            "prevent validation/test leakage"
        )
    included = {int(episode) for episode in episodes}
    if not included:
        raise ValueError("Delta Beaver fitting requires a non-empty training split")
    paths = sorted((Path(dataset.root).expanduser() / "data").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No LeRobot parquet shards found under {dataset.root}/data"
        )
    columns = [
        dataset.beaver_distance_key,
        dataset.beaver_present_key,
        dataset.beaver_status_key,
        "episode_index",
    ]
    sensor_values: list[list[np.ndarray]] = [[] for _ in sensor_indices]
    selected_indices = np.asarray(sensor_indices, dtype=np.int64)
    for path in paths:
        table = pq.read_table(path, columns=columns)
        episode_index = np.asarray(table["episode_index"]).reshape(-1)
        selected_rows = np.isin(episode_index, tuple(included))
        if not selected_rows.any():
            continue
        distance = np.asarray(
            table[dataset.beaver_distance_key].to_pylist(), dtype=np.float32
        )[selected_rows][:, selected_indices]
        present = np.asarray(
            table[dataset.beaver_present_key].to_pylist(), dtype=np.float32
        )[selected_rows][:, selected_indices]
        status = np.asarray(
            table[dataset.beaver_status_key].to_pylist(), dtype=np.int16
        )[selected_rows][:, selected_indices]
        valid = np.isin(status, dataset.beaver_valid_statuses)
        valid &= present.astype(bool)[..., None, None]
        valid &= np.isfinite(distance) & (distance != 0.0)
        for sensor_position in range(len(sensor_indices)):
            values = distance[:, sensor_position][valid[:, sensor_position]]
            if values.size:
                sensor_values[sensor_position].append(values)

    means, standard_deviations = [], []
    for sensor_name, chunks in zip(model.beaver_delta_sensors, sensor_values):
        if not chunks:
            raise ValueError(
                f"No valid non-zero training readings for Beaver sensor {sensor_name}"
            )
        values = np.concatenate(chunks).astype(np.float64, copy=False)
        means.append(float(values.mean()))
        standard_deviations.append(
            max(float(values.std()), float(dataset.normalization_floor))
        )
    return {
        "mean": torch.tensor(means, dtype=torch.float32),
        "std": torch.tensor(standard_deviations, dtype=torch.float32),
        "sensor_indices": torch.tensor(sensor_indices, dtype=torch.long),
    }


def fit_action_delta_statistics(
    config: RealmanBeaverConfig,
    episodes: Sequence[int],
) -> dict[str, Tensor]:
    """Fit per-joint scales for WRM_claude pose-anchored action deltas."""
    dataset, model = config.dataset, config.model
    if episodes is None:
        raise ValueError(
            "Action delta statistics require explicit training episodes to "
            "prevent validation/test leakage"
        )
    included = {int(episode) for episode in episodes}
    if not included:
        raise ValueError("Action delta fitting requires a non-empty training split")
    paths = sorted((Path(dataset.root).expanduser() / "data").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No LeRobot parquet shards found under {dataset.root}/data"
        )
    columns = [
        dataset.state_key,
        dataset.action_key,
        "episode_index",
        "frame_index",
    ]
    anchor_parts: list[np.ndarray] = []
    step_parts: list[np.ndarray] = []
    for path in paths:
        table = pq.read_table(path, columns=columns)
        episode_index = np.asarray(table["episode_index"]).reshape(-1)
        selected_rows = np.isin(episode_index, tuple(included))
        if not selected_rows.any():
            continue
        state = np.asarray(table[dataset.state_key].to_pylist(), dtype=np.float32)[
            selected_rows
        ]
        action = np.asarray(table[dataset.action_key].to_pylist(), dtype=np.float32)[
            selected_rows
        ]
        frame_index = np.asarray(table["frame_index"]).reshape(-1)[selected_rows]
        episode_index = episode_index[selected_rows]
        if state.shape[1:] != (model.state_dim,) or action.shape[1:] != (
            model.action_dim,
        ):
            raise ValueError(
                f"Unexpected state/action shapes: {state.shape}, {action.shape}"
            )
        anchor_parts.append(np.abs(action - state))
        step_chunks: list[np.ndarray] = []
        for episode in np.unique(episode_index):
            mask = episode_index == episode
            order = np.argsort(frame_index[mask], kind="stable")
            ordered = action[mask][order]
            if len(ordered) >= 2:
                step_chunks.append(np.abs(ordered[1:] - ordered[:-1]))
        if step_chunks:
            step_parts.append(np.concatenate(step_chunks, axis=0))
    if not anchor_parts or not step_parts:
        raise ValueError(
            "No action rows or within-episode transitions matched the training episodes"
        )
    anchors = np.concatenate(anchor_parts, axis=0)
    steps = np.concatenate(step_parts, axis=0)
    scales = np.maximum(
        np.percentile(anchors, 99.0, axis=0),
        np.percentile(steps, 99.0, axis=0),
    )
    scales = np.maximum(scales, float(dataset.normalization_floor))
    return {"scale": torch.tensor(scales, dtype=torch.float32)}


def fit_delta_action_statistics(
    config: RealmanBeaverConfig,
    episodes: Sequence[int],
) -> dict[str, Tensor]:
    """Fit per-joint min/max of a replan-anchored relative-action target."""
    dataset, model = config.dataset, config.model
    if episodes is None:
        raise ValueError(
            "Delta action statistics require explicit training episodes to "
            "prevent validation/test leakage"
        )
    included = {int(episode) for episode in episodes}
    if not included:
        raise ValueError("Delta action fitting requires a non-empty training split")
    paths = sorted((Path(dataset.root).expanduser() / "data").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No LeRobot parquet shards found under {dataset.root}/data"
        )
    columns = [dataset.state_key, dataset.action_key, "episode_index", "frame_index"]
    state_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    episode_parts: list[np.ndarray] = []
    frame_parts: list[np.ndarray] = []
    for path in paths:
        table = pq.read_table(path, columns=columns)
        episode_index = np.asarray(table["episode_index"]).reshape(-1)
        selected = np.isin(episode_index, tuple(included))
        if not selected.any():
            continue
        state_parts.append(
            np.asarray(table[dataset.state_key].to_pylist(), dtype=np.float32)[selected]
        )
        action_parts.append(
            np.asarray(table[dataset.action_key].to_pylist(), dtype=np.float32)[
                selected
            ]
        )
        episode_parts.append(episode_index[selected])
        frame_parts.append(np.asarray(table["frame_index"]).reshape(-1)[selected])
    if not state_parts:
        raise ValueError("No state/action rows matched the normalization episodes")

    state = np.concatenate(state_parts, axis=0)
    action = np.concatenate(action_parts, axis=0)
    episode_index = np.concatenate(episode_parts)
    frame_index = np.concatenate(frame_parts)
    if state.shape[1:] != (model.state_dim,) or action.shape[1:] != (model.action_dim,):
        raise ValueError(
            f"Unexpected state/action shapes: {state.shape}, {action.shape}"
        )

    horizon = model.horizon
    delta_minimum = np.zeros(model.action_dim, dtype=np.float64)
    delta_maximum = np.zeros(model.action_dim, dtype=np.float64)
    for episode in sorted(included):
        rows = np.flatnonzero(episode_index == episode)
        rows = rows[np.argsort(frame_index[rows], kind="stable")]
        if rows.size == 0:
            continue
        offsets = np.arange(rows.size)[:, None] - 1 + np.arange(horizon)[None, :]
        offsets = np.clip(offsets, 0, rows.size - 1)
        window = action[rows][offsets]
        deltas = window - state[rows][:, None, :]
        delta_minimum = np.minimum(delta_minimum, deltas.min(axis=(0, 1)))
        delta_maximum = np.maximum(delta_maximum, deltas.max(axis=(0, 1)))

    return {
        "min": torch.from_numpy(delta_minimum.astype(np.float32)),
        "max": torch.from_numpy(delta_maximum.astype(np.float32)),
    }


def build_temporal_history_windows(
    combined_history: Tensor,
    *,
    n_obs_steps: int,
    history_steps: int,
) -> Tensor:
    """Turn a clamped ``H+O-1`` sequence into ``[O,H,...]`` windows.

    LeRobot clamps every requested delta index to its current episode before
    this helper runs. Thus frames before an episode start already duplicate its
    earliest frame, and no window can cross into a preceding episode.
    """
    if n_obs_steps <= 0 or history_steps <= 0:
        raise ValueError("n_obs_steps and history_steps must be positive")
    expected = n_obs_steps + history_steps - 1
    if combined_history.shape[0] != expected:
        raise ValueError(
            f"combined temporal history has {combined_history.shape[0]} steps, "
            f"expected {expected}"
        )
    return torch.stack(
        [
            combined_history[start : start + history_steps]
            for start in range(n_obs_steps)
        ],
        dim=0,
    )


def fit_temporal_beaver_statistics(
    config: RealmanBeaverConfig,
    episodes: Sequence[int],
) -> dict[str, Tensor]:
    """Fit per-sensor P5/P95/median on valid non-zero training readings only."""
    dataset, model = config.dataset, config.model
    sensor_names = history_beaver_sensor_names(model)
    sensor_indices = resolve_beaver_sensor_indices(dataset, sensor_names)
    if episodes is None:
        raise ValueError(
            "Temporal Beaver statistics require explicit training episodes to "
            "prevent validation/test leakage"
        )
    included = {int(episode) for episode in episodes}
    if not included:
        raise ValueError("Temporal Beaver fitting requires a non-empty training split")

    paths = sorted((Path(dataset.root).expanduser() / "data").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No LeRobot parquet shards found under {dataset.root}/data"
        )
    columns = [
        dataset.beaver_distance_key,
        dataset.beaver_present_key,
        dataset.beaver_status_key,
        "episode_index",
    ]
    sensor_values: list[list[np.ndarray]] = [[] for _ in sensor_indices]
    selected_indices = np.asarray(sensor_indices, dtype=np.int64)
    for path in paths:
        table = pq.read_table(path, columns=columns)
        episode_index = np.asarray(table["episode_index"]).reshape(-1)
        selected_rows = np.isin(episode_index, tuple(included))
        if not selected_rows.any():
            continue

        distance = np.asarray(
            table[dataset.beaver_distance_key].to_pylist(), dtype=np.float32
        )[selected_rows][:, selected_indices]
        present = np.asarray(
            table[dataset.beaver_present_key].to_pylist(), dtype=np.float32
        )[selected_rows][:, selected_indices]
        status = np.asarray(
            table[dataset.beaver_status_key].to_pylist(), dtype=np.int16
        )[selected_rows][:, selected_indices]
        if distance.shape[1:] != (len(sensor_indices), 4, 4):
            raise ValueError(
                f"Unexpected temporal Beaver distance shape: {distance.shape}"
            )
        if present.shape[1:] != (len(sensor_indices),):
            raise ValueError(
                f"Unexpected temporal Beaver presence shape: {present.shape}"
            )
        if status.shape != distance.shape:
            raise ValueError(f"Unexpected temporal Beaver status shape: {status.shape}")

        valid = np.isin(status, dataset.beaver_valid_statuses)
        valid &= present.astype(bool)[..., None, None]
        valid &= np.isfinite(distance) & (distance != 0.0)
        for sensor_position in range(len(sensor_indices)):
            values = distance[:, sensor_position][valid[:, sensor_position]]
            if values.size:
                sensor_values[sensor_position].append(values)

    p5, p95, median = [], [], []
    for sensor_name, chunks in zip(sensor_names, sensor_values):
        if not chunks:
            raise ValueError(
                f"No valid non-zero training readings for Beaver sensor {sensor_name}"
            )
        values = np.concatenate(chunks).astype(np.float64, copy=False)
        lower, upper = np.percentile(values, (5.0, 95.0))
        lower = float(lower)
        upper = max(float(upper), lower + float(dataset.normalization_floor))
        p5.append(lower)
        p95.append(upper)
        median.append(float(np.median(values)))

    return {
        "p5": torch.tensor(p5, dtype=torch.float32),
        "p95": torch.tensor(p95, dtype=torch.float32),
        "median": torch.tensor(median, dtype=torch.float32),
        "sensor_indices": torch.tensor(sensor_indices, dtype=torch.long),
    }


def fit_key4_pca(
    config: RealmanBeaverConfig,
    episodes: Sequence[int],
) -> dict[str, Tensor]:
    """Fit independent Key4 PCA transforms from training episodes only.

    Each physical sensor contributes 32 inputs: 16 near-field proximity values
    and their 16 validity flags. Standardization and PCA are fitted separately
    for slots 01/02/10/11 so the transform cannot mix sensor identities.
    """
    dataset, model = config.dataset, config.model
    included = {int(episode) for episode in episodes}
    if not included:
        raise ValueError("PCA fitting requires at least one training episode")
    columns = [
        dataset.beaver_distance_key,
        dataset.beaver_present_key,
        dataset.beaver_status_key,
        "episode_index",
    ]
    distance_parts: list[np.ndarray] = []
    present_parts: list[np.ndarray] = []
    status_parts: list[np.ndarray] = []
    paths = sorted((Path(dataset.root).expanduser() / "data").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No LeRobot parquet shards found under {dataset.root}/data"
        )
    for path in paths:
        table = pq.read_table(path, columns=columns)
        episode_index = np.asarray(table["episode_index"]).reshape(-1)
        selected = np.isin(episode_index, tuple(included))
        if not selected.any():
            continue
        distance_parts.append(
            np.asarray(
                table[dataset.beaver_distance_key].to_pylist(), dtype=np.float32
            )[selected]
        )
        present_parts.append(
            np.asarray(table[dataset.beaver_present_key].to_pylist(), dtype=np.float32)[
                selected
            ]
        )
        status_parts.append(
            np.asarray(table[dataset.beaver_status_key].to_pylist(), dtype=np.int16)[
                selected
            ]
        )
    if not distance_parts:
        raise ValueError("No Beaver rows matched the PCA training episodes")

    key_indices = np.asarray(model.beaver_key_sensor_indices, dtype=np.int64)
    distance = np.concatenate(distance_parts)[:, key_indices]
    present = np.concatenate(present_parts)[:, key_indices]
    status = np.concatenate(status_parts)[:, key_indices]
    expected_distance_shape = (len(distance), 4, 4, 4)
    if distance.shape != expected_distance_shape or status.shape != distance.shape:
        raise ValueError(
            f"Unexpected Key4 distance/status shapes: {distance.shape}, {status.shape}"
        )
    if present.shape != (len(distance), 4):
        raise ValueError(f"Unexpected Key4 presence shape: {present.shape}")

    valid = np.isin(status, dataset.beaver_valid_statuses)
    valid &= present.astype(bool)[..., None, None]
    proximity = 1.0 - np.clip(
        distance / float(model.beaver_near_threshold_mm), 0.0, 1.0
    )
    proximity *= valid
    features = np.concatenate(
        (proximity.reshape(len(distance), 4, 16), valid.reshape(len(distance), 4, 16)),
        axis=-1,
    ).astype(np.float64)

    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.maximum(scale, float(dataset.normalization_floor))
    standardized = (features - mean[None]) / scale[None]
    component_count = model.beaver_pca_components
    bases: list[np.ndarray] = []
    ratios: list[np.ndarray] = []
    for sensor_id in range(4):
        values = standardized[:, sensor_id]
        covariance = values.T @ values / max(len(values) - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        basis = eigenvectors[:, order[:component_count]].T
        # Resolve the arbitrary PCA sign deterministically for reproducibility.
        maxima = np.argmax(np.abs(basis), axis=1)
        signs = np.sign(basis[np.arange(component_count), maxima])
        signs[signs == 0] = 1.0
        basis *= signs[:, None]
        total_variance = max(float(eigenvalues.sum()), np.finfo(np.float64).eps)
        bases.append(basis)
        ratios.append(eigenvalues[:component_count] / total_variance)

    return {
        "mean": torch.from_numpy(mean.astype(np.float32)),
        "scale": torch.from_numpy(scale.astype(np.float32)),
        "basis": torch.from_numpy(np.stack(bases).astype(np.float32)),
        "explained_variance_ratio": torch.from_numpy(
            np.stack(ratios).astype(np.float32)
        ),
    }


class RealmanPolicyDataset(Dataset[dict[str, Tensor]]):
    """Build direct-policy, tokenizer, or latent-policy sequences."""

    def __init__(
        self,
        config: RealmanBeaverConfig,
        episodes: Sequence[int] | None = None,
        stage: str = "policy",
    ) -> None:
        if stage not in {"policy", "tokenizer", "latent"}:
            raise ValueError("stage must be policy, tokenizer, or latent")
        self.config = config
        self.stage = stage
        self._episode_index: np.ndarray | None = None
        dataset, model = config.dataset, config.model
        reactive = config.rdp if model.variant == "rdp_like" else config.rfm
        root = Path(dataset.root).expanduser().resolve()
        history = list(range(1 - model.n_obs_steps, 1))

        if stage == "tokenizer":
            self.dataset = None
            self._load_tokenizer_parquet(root, episodes)
            return

        if stage == "policy":
            state_history = history
            lookback = motion_lookback_steps(model)
            if lookback is not None:
                state_history = list(range(1 - model.n_obs_steps - lookback, 1))
            if model.variant == QWEN_BEAVER_VARIANT:
                lag = model.qwen_joint_history_steps
                state_history = [-(lag + 1), -2, -1, 0]
            delta_timestamps = {
                dataset.image_key: [step / dataset.fps for step in history],
                dataset.state_key: [step / dataset.fps for step in state_history],
                # This is LeRobot's canonical DP alignment. With two observations,
                # action index 1 is the current action selected for execution.
                dataset.action_key: [
                    step / dataset.fps
                    for step in range(
                        1 - model.n_obs_steps, 1 - model.n_obs_steps + model.horizon
                    )
                ],
            }
            if model.variant in _DIRECT_BEAVER_VARIANTS:
                if model.variant in HISTORY_BEAVER_VARIANTS:
                    # Union of the H-frame windows ending at each of the O
                    # native DP observation times. LeRobot clamps these query
                    # indices within the current episode (earliest-frame pad).
                    history_steps = history_beaver_steps(model)
                    temporal_history = range(2 - model.n_obs_steps - history_steps, 1)
                    beaver_timestamps = [
                        step / dataset.fps for step in temporal_history
                    ]
                elif model.variant == DELTA_BEAVER_VARIANT:
                    delta_history = range(
                        1 - model.n_obs_steps - model.beaver_delta_steps, 1
                    )
                    beaver_timestamps = [step / dataset.fps for step in delta_history]
                else:
                    beaver_timestamps = [step / dataset.fps for step in history]
                delta_timestamps[dataset.beaver_distance_key] = beaver_timestamps
                delta_timestamps[dataset.beaver_present_key] = beaver_timestamps
                delta_timestamps[dataset.beaver_status_key] = beaver_timestamps
                if model.variant in GRASP_STATE_VARIANTS:
                    delta_timestamps[dataset.grasp_state_key] = [
                        step / dataset.fps for step in history
                    ]
        else:
            delta_timestamps = {
                dataset.image_key: [
                    step * reactive.slow_observation_stride / dataset.fps
                    for step in history
                ],
                dataset.state_key: [
                    step * reactive.slow_observation_stride / dataset.fps
                    for step in history
                ],
                dataset.action_key: [
                    step / dataset.fps for step in range(reactive.action_horizon)
                ],
            }

        self.dataset = LeRobotDataset(
            repo_id=dataset.repo_id,
            root=root,
            episodes=list(episodes) if episodes is not None else None,
            delta_timestamps=delta_timestamps,
            video_backend=dataset.video_backend,
        )
        self._validate_features()

    def _load_tokenizer_parquet(
        self, root: Path, episodes: Sequence[int] | None
    ) -> None:
        """Cache non-visual RDP fields and future-window indices in memory."""
        dataset = self.config.dataset
        columns = [
            dataset.action_key,
            dataset.beaver_distance_key,
            dataset.beaver_present_key,
            dataset.beaver_status_key,
            "episode_index",
            "frame_index",
        ]
        parts: dict[str, list[np.ndarray]] = {column: [] for column in columns}
        paths = sorted((root / "data").rglob("*.parquet"))
        if not paths:
            raise FileNotFoundError(
                f"No LeRobot parquet shards found under {root / 'data'}"
            )
        for path in paths:
            table = pq.read_table(path, columns=columns)
            for key in columns:
                if key in {
                    dataset.action_key,
                    dataset.beaver_distance_key,
                    dataset.beaver_present_key,
                    dataset.beaver_status_key,
                }:
                    value = np.asarray(table[key].to_pylist(), dtype=np.float32)
                else:
                    value = np.asarray(table[key]).reshape(-1)
                parts[key].append(value)

        action = np.concatenate(parts[dataset.action_key])
        distance = np.concatenate(parts[dataset.beaver_distance_key])
        present = np.concatenate(parts[dataset.beaver_present_key])
        status = np.concatenate(parts[dataset.beaver_status_key])
        episode_index = np.concatenate(parts["episode_index"])
        frame_index = np.concatenate(parts["frame_index"])
        if action.shape[1:] != (7,) or distance.shape[1:] != (9, 4, 4):
            raise ValueError(
                f"Unexpected tokenizer action/Beaver shapes: {action.shape}, {distance.shape}"
            )
        if present.shape[1:] != (9,):
            raise ValueError(f"Unexpected Beaver presence shape: {present.shape}")
        if status.shape[1:] != (9, 4, 4):
            raise ValueError(f"Unexpected Beaver status shape: {status.shape}")

        included = (
            set(episodes) if episodes is not None else set(np.unique(episode_index))
        )
        reactive = (
            self.config.rdp
            if self.config.model.variant == "rdp_like"
            else self.config.rfm
        )
        horizon = reactive.action_horizon
        queries, padding = [], []
        for episode in sorted(included):
            selected = np.flatnonzero(episode_index == episode)
            if not len(selected):
                raise ValueError(
                    f"Requested episode {episode} is absent from the parquet data"
                )
            selected = selected[np.argsort(frame_index[selected])]
            local_query = (
                np.arange(len(selected))[:, None] + np.arange(horizon)[None, :]
            )
            padding.append(torch.from_numpy(local_query >= len(selected)))
            queries.append(
                torch.from_numpy(selected[local_query.clip(max=len(selected) - 1)])
            )

        self._tokenizer_action = torch.from_numpy(action)
        self._tokenizer_distance = torch.from_numpy(distance)
        self._tokenizer_present = torch.from_numpy(present)
        self._tokenizer_status = torch.from_numpy(status)
        self._tokenizer_query = torch.cat(queries).long()
        self._tokenizer_padding = torch.cat(padding).bool()

    def _validate_features(self) -> None:
        if self.dataset is None:
            return
        features, config = self.dataset.features, self.config
        required: dict[str, tuple[int, ...] | None] = {config.dataset.action_key: (7,)}
        if self.stage in {"policy", "latent"}:
            required[config.dataset.image_key] = None
            required[config.dataset.state_key] = (7,)
        if self.stage == "tokenizer" or config.model.variant in _DIRECT_BEAVER_VARIANTS:
            required[config.dataset.beaver_distance_key] = (9, 4, 4)
            required[config.dataset.beaver_present_key] = (9,)
            required[config.dataset.beaver_status_key] = (9, 4, 4)
        if config.model.variant in GRASP_STATE_VARIANTS:
            required[config.dataset.grasp_state_key] = None
        for key, shape in required.items():
            if key not in features:
                raise KeyError(f"Required LeRobot feature is missing: {key}")
            if shape is not None and tuple(features[key]["shape"]) != shape:
                raise ValueError(
                    f"{key} has shape {features[key]['shape']}, expected {shape}"
                )

    @property
    def episode_index(self) -> np.ndarray:
        """Episode id for every policy sample, used by bottle-stratified batches."""
        if self.stage == "tokenizer":
            raise RuntimeError("tokenizer datasets do not expose episode_index")
        if self._episode_index is not None:
            return self._episode_index
        assert self.dataset is not None
        hf_dataset = getattr(self.dataset, "hf_dataset", None)
        if hf_dataset is not None and "episode_index" in getattr(
            hf_dataset, "column_names", ()
        ):
            values = np.asarray(hf_dataset["episode_index"]).reshape(-1)
        else:
            values = np.asarray(
                [
                    int(self.dataset[index]["episode_index"])
                    for index in range(len(self.dataset))
                ],
                dtype=np.int64,
            )
        if len(values) != len(self.dataset):
            raise ValueError("episode_index length does not match the policy dataset")
        self._episode_index = values.astype(np.int64, copy=False)
        return self._episode_index

    def __len__(self) -> int:
        if self.stage == "tokenizer":
            return len(self._tokenizer_query)
        assert self.dataset is not None
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if self.stage == "tokenizer":
            query = self._tokenizer_query[index]
            return {
                "action": self._tokenizer_action[query],
                "action_is_pad": self._tokenizer_padding[index],
                "beaver_distance": self._tokenizer_distance[query],
                "beaver_present": self._tokenizer_present[query],
                "beaver_status": self._tokenizer_status[query],
            }
        assert self.dataset is not None
        item, dataset = self.dataset[index], self.config.dataset
        action = item[dataset.action_key].float()
        sample = {
            "action": action,
            "action_is_pad": item.get(
                f"{dataset.action_key}_is_pad",
                torch.zeros(action.shape[0], dtype=torch.bool),
            ).bool(),
        }
        if self.stage in {"policy", "latent"}:
            sample["image"] = item[dataset.image_key].float()
            state = item[dataset.state_key].float()
            model = self.config.model
            if model.variant == ANTIGRAVITY_BEAVER_VARIANT:
                n_obs = model.n_obs_steps
                delta_short = model.beaver_antigravity_motion_delta_steps
                delta_long = model.beaver_antigravity_motion_delta_long_steps
                state_curr = state[-n_obs:]
                prev_state_short = state[
                    -n_obs - delta_short : len(state) - delta_short
                ]
                prev_state_long = state[-n_obs - delta_long : len(state) - delta_long]
                sample["delta_q"] = state_curr - prev_state_short
                sample["delta_q_long"] = state_curr - prev_state_long
                state = state_curr
            elif model.variant in MOTION_DELTA_VARIANTS:
                lookback = motion_lookback_steps(model)
                assert lookback is not None
                state, previous_state = build_delta_pairs(
                    state,
                    n_obs_steps=model.n_obs_steps,
                    delta_steps=lookback,
                )
                sample["delta_q"] = state - previous_state
            sample["state"] = state
            if model.variant == CLAUDE_BEAVER_VARIANT:
                action_delta = torch.empty_like(action)
                action_delta[0] = action[0] - state[0]
                action_delta[1] = action[1] - state[1]
                action_delta[2:] = action[2:] - action[1:-1]
                sample["action_delta"] = action_delta
        if (
            self.stage == "tokenizer"
            or self.config.model.variant in _DIRECT_BEAVER_VARIANTS
        ):
            distance = item[dataset.beaver_distance_key].float()
            present = item[dataset.beaver_present_key].float()
            status = item[dataset.beaver_status_key].float()
            if self.config.model.variant in HISTORY_BEAVER_VARIANTS:
                model = self.config.model
                history_steps = history_beaver_steps(model)
                distance = build_temporal_history_windows(
                    distance,
                    n_obs_steps=model.n_obs_steps,
                    history_steps=history_steps,
                )
                present = build_temporal_history_windows(
                    present,
                    n_obs_steps=model.n_obs_steps,
                    history_steps=history_steps,
                )
                status = build_temporal_history_windows(
                    status,
                    n_obs_steps=model.n_obs_steps,
                    history_steps=history_steps,
                )
                sample["beaver_history_distance"] = distance
                sample["beaver_history_present"] = present
                sample["beaver_history_status"] = status
            elif self.config.model.variant == DELTA_BEAVER_VARIANT:
                model = self.config.model
                distance, previous_distance = build_delta_pairs(
                    distance,
                    n_obs_steps=model.n_obs_steps,
                    delta_steps=model.beaver_delta_steps,
                )
                present, previous_present = build_delta_pairs(
                    present,
                    n_obs_steps=model.n_obs_steps,
                    delta_steps=model.beaver_delta_steps,
                )
                status, previous_status = build_delta_pairs(
                    status,
                    n_obs_steps=model.n_obs_steps,
                    delta_steps=model.beaver_delta_steps,
                )
                sample["beaver_distance"] = distance
                sample["beaver_present"] = present
                sample["beaver_status"] = status
                sample["beaver_previous_distance"] = previous_distance
                sample["beaver_previous_present"] = previous_present
                sample["beaver_previous_status"] = previous_status
            else:
                sample["beaver_distance"] = distance
                sample["beaver_present"] = present
                sample["beaver_status"] = status
        if self.config.model.variant in GRASP_STATE_VARIANTS:
            sample["grasp_state"] = item[dataset.grasp_state_key].float().reshape(-1)
        if self.config.model.variant == CODEX_BEAVER_VARIANT:
            for metadata_key in ("episode_index", "frame_index"):
                if metadata_key in item:
                    sample[metadata_key] = torch.as_tensor(item[metadata_key]).long()
        return sample
