"""LeRobot dataset adapters and normalization for all six baselines."""

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
from torch.utils.data import Dataset

from policies.realman_beaver.configuration import DatasetConfig, RealmanBeaverConfig


class ObservationNormalizer(nn.Module):
    """Normalize raw LeRobot values before they enter a LeRobot policy."""

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
    ) -> None:
        super().__init__()
        self.register_buffer("state_offset", state_offset.float())
        self.register_buffer("state_scale", state_scale.float())
        self.register_buffer("action_offset", action_offset.float())
        self.register_buffer("action_scale", action_scale.float())
        self.register_buffer("image_mean", image_mean.float().reshape(3, 1, 1))
        self.register_buffer("image_std", image_std.float().reshape(3, 1, 1))
        self.register_buffer("distance_max_mm", torch.tensor(float(distance_max_mm)))
        self._valid_statuses = tuple(valid_statuses)

    @classmethod
    def from_lerobot_dataset(cls, config: RealmanBeaverConfig) -> ObservationNormalizer:
        dataset, model = config.dataset, config.model
        if dataset.normalization_source == "metadata":
            return cls._from_metadata(config)

        paths = sorted((Path(dataset.root).expanduser() / "data").rglob("*.parquet"))
        if not paths:
            raise FileNotFoundError(
                f"No LeRobot parquet shards found under {dataset.root}/data"
            )
        keys = (dataset.state_key, dataset.action_key)
        chunks: dict[str, list[np.ndarray]] = {key: [] for key in keys}
        for path in paths:
            table = pq.read_table(path, columns=list(keys))
            for key in keys:
                chunks[key].append(np.asarray(table[key].to_pylist(), dtype=np.float32))
        state = np.concatenate(chunks[dataset.state_key], axis=0)
        action = np.concatenate(chunks[dataset.action_key], axis=0)
        if state.shape[1:] != (model.state_dim,) or action.shape[1:] != (
            model.action_dim,
        ):
            raise ValueError(
                f"Unexpected state/action shapes: {state.shape}, {action.shape}"
            )
        return cls._from_arrays(state, action, dataset)

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
    def identity(cls, state_dim: int = 7, action_dim: int = 7) -> ObservationNormalizer:
        return cls(
            torch.zeros(state_dim),
            torch.ones(state_dim),
            torch.zeros(action_dim),
            torch.ones(action_dim),
            torch.zeros(3),
            torch.ones(3),
            2550.0,
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
    random.Random(config.split_seed).shuffle(episodes)
    validation_count = round(total * config.val_fraction)
    if config.val_fraction > 0 and total > 1:
        validation_count = max(1, min(total - 1, validation_count))
    return sorted(episodes[validation_count:]), sorted(episodes[:validation_count])


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
        dataset, model = config.dataset, config.model
        reactive = config.rdp if model.variant == "rdp_like" else config.rfm
        root = Path(dataset.root).expanduser().resolve()
        history = list(range(1 - model.n_obs_steps, 1))

        if stage == "tokenizer":
            self.dataset = None
            self._load_tokenizer_parquet(root, episodes)
            return

        if stage == "policy":
            delta_timestamps = {
                dataset.image_key: [step / dataset.fps for step in history],
                dataset.state_key: [step / dataset.fps for step in history],
                # This is LeRobot's canonical DP alignment. With two observations,
                # action index 1 is the current action selected for execution.
                dataset.action_key: [
                    step / dataset.fps
                    for step in range(
                        1 - model.n_obs_steps, 1 - model.n_obs_steps + model.horizon
                    )
                ],
            }
            if model.variant in {"dp_beaver", "fm_beaver"}:
                delta_timestamps[dataset.beaver_distance_key] = [
                    step / dataset.fps for step in history
                ]
                delta_timestamps[dataset.beaver_present_key] = [
                    step / dataset.fps for step in history
                ]
                delta_timestamps[dataset.beaver_status_key] = [
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
        if self.stage == "tokenizer" or config.model.variant in {
            "dp_beaver",
            "fm_beaver",
        }:
            required[config.dataset.beaver_distance_key] = (9, 4, 4)
            required[config.dataset.beaver_present_key] = (9,)
            required[config.dataset.beaver_status_key] = (9, 4, 4)
        for key, shape in required.items():
            if key not in features:
                raise KeyError(f"Required LeRobot feature is missing: {key}")
            if shape is not None and tuple(features[key]["shape"]) != shape:
                raise ValueError(
                    f"{key} has shape {features[key]['shape']}, expected {shape}"
                )

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
            sample["state"] = item[dataset.state_key].float()
        if self.stage == "tokenizer" or self.config.model.variant in {
            "dp_beaver",
            "fm_beaver",
        }:
            sample["beaver_distance"] = item[dataset.beaver_distance_key].float()
            sample["beaver_present"] = item[dataset.beaver_present_key].float()
            sample["beaver_status"] = item[dataset.beaver_status_key].float()
        return sample
