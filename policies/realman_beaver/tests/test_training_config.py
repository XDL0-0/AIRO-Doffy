from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from policies.realman_beaver.configuration import DatasetConfig, RealmanBeaverConfig
from policies.realman_beaver.dataset import ObservationNormalizer, episode_split
from policies.realman_beaver.train import (
    _WANDB_METRIC_KINDS,
    _parse_episode_spec,
    _wandb_log,
)


class TrainingConfigTest(unittest.TestCase):
    def test_explicit_validation_episode_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "meta").mkdir()
            (root / "meta" / "info.json").write_text(
                json.dumps({"total_episodes": 125}), encoding="utf-8"
            )
            config = DatasetConfig(root=str(root), val_episodes=tuple(range(50, 75)))

            train, validation = episode_split(config)

        self.assertEqual(validation, list(range(50, 75)))
        self.assertEqual(len(train), 100)
        self.assertNotIn(50, train)
        self.assertNotIn(74, train)
        self.assertIn(49, train)
        self.assertIn(75, train)

    def test_episode_spec_uses_zero_based_inclusive_ranges(self) -> None:
        self.assertEqual(_parse_episode_spec("0,2-4,7"), (0, 2, 3, 4, 7))

    def test_parquet_normalization_uses_only_selected_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data" / "chunk-000").mkdir(parents=True)
            (root / "meta").mkdir()
            values = [[float(value)] * 7 for value in (0, 2, 100, 200)]
            pq.write_table(
                pa.table(
                    {
                        "observation.state": values,
                        "action": values,
                        "episode_index": [0, 0, 1, 1],
                    }
                ),
                root / "data" / "chunk-000" / "file-000.parquet",
            )
            (root / "meta" / "stats.json").write_text(
                json.dumps(
                    {
                        "observation.images.camera_0": {
                            "mean": [0.5, 0.5, 0.5],
                            "std": [0.25, 0.25, 0.25],
                        }
                    }
                ),
                encoding="utf-8",
            )
            config = RealmanBeaverConfig()
            config.dataset.root = str(root)

            normalizer = ObservationNormalizer.from_lerobot_dataset(config, [0])

        torch.testing.assert_close(normalizer.state_offset, torch.ones(7))
        torch.testing.assert_close(normalizer.state_scale, torch.ones(7))
        torch.testing.assert_close(normalizer.action_offset, torch.ones(7))
        torch.testing.assert_close(normalizer.action_scale, torch.ones(7))

    @patch("policies.realman_beaver.train.wandb")
    def test_wandb_uses_independent_stage_steps(self, wandb: Mock) -> None:
        wandb.run = object()
        _WANDB_METRIC_KINDS.clear()

        _wandb_log("tokenizer", 10, {"loss": 1.0})
        _wandb_log("latent_dp", 1, {"loss": 2.0})

        wandb.log.assert_any_call(
            {"tokenizer/loss": 1.0, "tokenizer/global_step": 10}
        )
        wandb.log.assert_any_call(
            {"latent_dp/loss": 2.0, "latent_dp/global_step": 1}
        )
        for call in wandb.log.call_args_list:
            self.assertNotIn("step", call.kwargs)


if __name__ == "__main__":
    unittest.main()
