from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from policies.realman_beaver.configuration import (
    DatasetConfig,
    ModelConfig,
    RealmanBeaverConfig,
)
from policies.realman_beaver.dataset import (
    ObservationNormalizer,
    build_temporal_history_windows,
    fit_temporal_beaver_statistics,
    resolve_beaver_sensor_indices,
)


class _FakeTemporalLeRobotDataset:
    def __init__(self, *, delta_timestamps: dict[str, list[float]], **_: object) -> None:
        self.delta_timestamps = delta_timestamps
        self.features = {
            "observation.images.camera_0": {"shape": (3, 64, 64)},
            "observation.state": {"shape": (7,)},
            "action": {"shape": (7,)},
            "observation.beaver.distance_mm": {"shape": (9, 4, 4)},
            "observation.beaver.present": {"shape": (9,)},
            "observation.beaver.target_status": {"shape": (9, 4, 4)},
            "tightness": {"shape": (1,)},
        }
        combined_steps = len(delta_timestamps["observation.beaver.distance_mm"])
        obs_steps = len(delta_timestamps["observation.state"])
        action_steps = len(delta_timestamps["action"])
        self.item = {
            "observation.images.camera_0": torch.rand(obs_steps, 3, 64, 64),
            "observation.state": torch.rand(obs_steps, 7),
            "action": torch.rand(action_steps, 7),
            "action_is_pad": torch.zeros(action_steps, dtype=torch.bool),
            "observation.beaver.distance_mm": torch.rand(
                combined_steps, 9, 4, 4
            ),
            "observation.beaver.present": torch.ones(combined_steps, 9),
            "observation.beaver.target_status": torch.full(
                (combined_steps, 9, 4, 4), 5.0
            ),
            "tightness": torch.tensor([0.0, 1.0]),
        }

    def __len__(self) -> int:
        return 1

    def __getitem__(self, _: int) -> dict[str, torch.Tensor]:
        return self.item


class TemporalHistoryTest(unittest.TestCase):
    def test_first_middle_last_and_episode_boundary_padding(self) -> None:
        # These are the combined H+O-1 values after LeRobot has clamped each
        # query index to this episode's [10, 11, 12, 13] frame range.
        cases = {
            "first": ([10, 10, 10, 10], [[10, 10, 10], [10, 10, 10]]),
            "middle": ([10, 10, 11, 12], [[10, 10, 11], [10, 11, 12]]),
            "last": ([10, 11, 12, 13], [[10, 11, 12], [11, 12, 13]]),
        }
        for name, (combined, expected) in cases.items():
            with self.subTest(name=name):
                windows = build_temporal_history_windows(
                    torch.tensor(combined), n_obs_steps=2, history_steps=3
                )
                torch.testing.assert_close(windows, torch.tensor(expected))

        # 99 is the preceding episode's last frame. It must never appear in an
        # episode-start window; clamping duplicates earliest frame 10 instead.
        boundary = build_temporal_history_windows(
            torch.tensor([10, 10, 10, 10]), n_obs_steps=2, history_steps=3
        )
        self.assertFalse(torch.any(boundary == 99))
        self.assertTrue(torch.all(boundary == 10))

    def test_policy_dataset_returns_nested_history_and_grasp_label(self) -> None:
        from policies.realman_beaver.dataset import RealmanPolicyDataset

        config = RealmanBeaverConfig(
            model=ModelConfig(
                variant="WRM_temporal",
                n_obs_steps=2,
                horizon=8,
                n_action_steps=2,
                down_dims=(32, 64),
                num_train_timesteps=8,
                num_inference_steps=2,
            )
        )
        config.dataset.image_shape = (3, 64, 64)
        config.validate()
        with patch(
            "policies.realman_beaver.dataset.LeRobotDataset",
            _FakeTemporalLeRobotDataset,
        ):
            dataset = RealmanPolicyDataset(config, [0], stage="policy")
        sample = dataset[0]
        self.assertEqual(sample["beaver_history_distance"].shape, (2, 12, 9, 4, 4))
        self.assertEqual(sample["beaver_history_status"].shape, (2, 12, 9, 4, 4))
        self.assertEqual(sample["beaver_history_present"].shape, (2, 12, 9))
        self.assertEqual(sample["grasp_state"].shape, (2,))
        self.assertEqual(
            dataset.dataset.delta_timestamps[
                config.dataset.beaver_distance_key
            ],
            [step / config.dataset.fps for step in range(-12, 1)],
        )


class TemporalNormalizationTest(unittest.TestCase):
    @staticmethod
    def _statistics() -> dict[str, torch.Tensor]:
        return {
            "p5": torch.full((4,), 100.0),
            "p95": torch.full((4,), 500.0),
            "median": torch.full((4,), 300.0),
            "sensor_indices": torch.tensor([1, 2, 5, 6]),
        }

    def test_sensor_names_resolve_through_layout(self) -> None:
        config = DatasetConfig()
        self.assertEqual(
            resolve_beaver_sensor_indices(config, ("01", "02", "10", "11")),
            (1, 2, 5, 6),
        )

    def test_zero_and_invalid_cells_are_neutrally_imputed_and_masked(self) -> None:
        normalizer = ObservationNormalizer.identity(
            temporal_beaver_statistics=self._statistics()
        )
        distance = torch.full((3, 9, 4, 4), 200.0)
        present = torch.ones(3, 9)
        status = torch.full((3, 9, 4, 4), 5.0)
        distance[0, 1, 0, 0] = 0.0
        distance[1, 1, 0, 1] = 500.0
        status[1, 1, 0, 1] = 255.0

        features = normalizer.normalize_temporal_beaver(distance, present, status)
        self.assertEqual(features.shape, (3, 4, 4, 4, 4))

        # A zero starts at the per-sensor median (0.5), never at an extreme;
        # project validity remains visible separately from the zero flag.
        self.assertEqual(features[0, 0, 0, 0, 0].item(), 0.5)
        self.assertEqual(features[0, 0, 0, 0, 1].item(), 0.0)
        self.assertEqual(features[0, 0, 0, 0, 2].item(), 1.0)
        self.assertEqual(features[0, 0, 0, 0, 3].item(), 1.0)

        # Invalid status cannot inject the 500 mm continuous value. The prior
        # 200 mm value is carried and both delta and valid mask are zero.
        self.assertEqual(features[1, 0, 0, 1, 0].item(), 0.25)
        self.assertEqual(features[1, 0, 0, 1, 1].item(), 0.0)
        self.assertEqual(features[1, 0, 0, 1, 2].item(), 0.0)

    def test_statistics_use_only_requested_training_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data" / "chunk-000").mkdir(parents=True)
            rows = 4
            distance = []
            for value in (100.0, 120.0, 1000.0, 1200.0):
                distance.append(
                    [[[value for _ in range(4)] for _ in range(4)] for _ in range(9)]
                )
            table = pa.table(
                {
                    "observation.beaver.distance_mm": distance,
                    "observation.beaver.present": [[1.0] * 9 for _ in range(rows)],
                    "observation.beaver.target_status": [
                        [[[5 for _ in range(4)] for _ in range(4)] for _ in range(9)]
                        for _ in range(rows)
                    ],
                    "episode_index": [0, 0, 1, 1],
                }
            )
            pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")
            config = RealmanBeaverConfig(
                dataset=DatasetConfig(root=str(root)),
                model=ModelConfig(variant="WRM_temporal"),
            )
            config.validate()

            training = fit_temporal_beaver_statistics(config, [0])
            leaked = fit_temporal_beaver_statistics(config, [0, 1])
            self.assertTrue(torch.all(training["p95"] < 200.0))
            self.assertTrue(torch.all(leaked["median"] > training["median"]))
            torch.testing.assert_close(
                training["sensor_indices"], torch.tensor([1, 2, 5, 6])
            )

    def test_statistics_require_an_explicit_training_split(self) -> None:
        config = RealmanBeaverConfig(model=ModelConfig(variant="WRM_temporal"))
        with self.assertRaisesRegex(ValueError, "explicit training episodes"):
            fit_temporal_beaver_statistics(config, None)


if __name__ == "__main__":
    unittest.main()
