from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from dataset_tool.visualize_lerobot_rerun import (
    FeatureSpec,
    _log_feature,
    build_blueprint,
    classify_feature,
    discover_features,
    feature_table,
    prepare_image,
    select_features,
)


class LeRobotRerunVisualizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.info = {
            "features": {
                "observation.images.camera_0": {
                    "dtype": "video",
                    "shape": [48, 64, 3],
                    "names": ["height", "width", "channel"],
                },
                "observation.depth.camera_0": {
                    "dtype": "image",
                    "shape": [48, 64, 1],
                    "names": ["height", "width", "channel"],
                },
                "action": {
                    "dtype": "float32",
                    "shape": [7],
                    "names": [f"joint_{index}" for index in range(7)],
                },
                "observation.tactile": {
                    "dtype": "float32",
                    "shape": [41, 3],
                    "names": ["sensor", "axis"],
                },
                "observation.beaver.distance_mm": {
                    "dtype": "float32",
                    "shape": [9, 4, 4],
                    "names": ["sensor", "row", "column"],
                },
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "caption": {"dtype": "string", "shape": [1], "names": None},
            }
        }

    def test_discovers_all_configured_types(self) -> None:
        specs = discover_features(self.info)
        self.assertEqual(
            [spec.kind for spec in specs],
            ["image", "depth", "vector", "tensor", "tensor", "scalar", "text"],
        )
        self.assertIn("observation.tactile", feature_table(specs))

    def test_depth_metadata_flag_is_recognized(self) -> None:
        spec = classify_feature(
            "camera",
            {
                "dtype": "video",
                "shape": [10, 20, 1],
                "info": {"is_depth_map": True},
            },
        )
        self.assertEqual(spec.kind, "depth")

    def test_feature_selection_accepts_globs_and_commas(self) -> None:
        specs = discover_features(self.info)
        selected = select_features(specs, ["action,observation.*"])
        self.assertEqual(
            [spec.key for spec in selected],
            [
                "observation.images.camera_0",
                "observation.depth.camera_0",
                "action",
                "observation.tactile",
                "observation.beaver.distance_mm",
            ],
        )
        with self.assertRaisesRegex(ValueError, "matched nothing"):
            select_features(specs, ["missing.*"])

    def test_prepare_image_converts_chw_float_to_hwc_uint8(self) -> None:
        spec = FeatureSpec(
            "camera", "video", (2, 4, 3), ("height", "width", "channel"), "image"
        )
        image = np.full((3, 2, 4), 0.5, dtype=np.float32)
        converted = prepare_image(image, spec)
        self.assertEqual(converted.shape, (2, 4, 3))
        self.assertEqual(converted.dtype, np.uint8)
        self.assertTrue(np.all(converted == 128))

    def test_blueprint_builds_for_every_supported_kind(self) -> None:
        blueprint = build_blueprint(discover_features(self.info), fps=30.0)
        self.assertIsNotNone(blueprint)

    @patch("rerun.log")
    def test_beaver_tensor_logs_nine_four_by_four_matrices(self, rerun_log) -> None:
        spec = classify_feature(
            "observation.beaver.distance_mm",
            {
                "dtype": "float32",
                "shape": [9, 4, 4],
                "names": ["sensor", "row", "column"],
            },
        )
        _log_feature(spec, np.arange(9 * 4 * 4, dtype=np.float32).reshape(9, 4, 4))

        self.assertEqual(rerun_log.call_count, 9)
        self.assertEqual(
            [call.args[0] for call in rerun_log.call_args_list],
            [
                f"/data/tensor/observation.beaver.distance_mm/sensor_{index:02d}"
                for index in range(9)
            ],
        )


if __name__ == "__main__":
    unittest.main()
