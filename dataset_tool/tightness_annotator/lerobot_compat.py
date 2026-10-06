"""Small compatibility surface for LeRobot releases that implement Dataset v3.0."""

from __future__ import annotations

from typing import Any

try:
    # LeRobot >= 0.5 keeps metadata in its own module.
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
except ImportError:
    # LeRobot 0.4.4 already implements Dataset v3.0, but defines this class here.
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata


def get_raw_item(dataset: Any, index: int) -> dict:
    """Read a row without video decoding across LeRobot 0.4.4 and 0.5.x."""
    method = getattr(dataset, "get_raw_item", None)
    if method is not None:
        return method(index)

    # 0.4.4 lazily initializes its public ``hf_dataset`` attribute.
    ensure_loaded = getattr(dataset, "_ensure_hf_dataset_loaded", None)
    if ensure_loaded is not None:
        ensure_loaded()
    if dataset.hf_dataset is None:
        raise RuntimeError("LeRobot did not load its frame table")
    return dataset.hf_dataset[index]


__all__ = ["LeRobotDatasetMetadata", "get_raw_item"]
