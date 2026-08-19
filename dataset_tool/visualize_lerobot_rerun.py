#!/usr/bin/env python3
"""Visualize any local LeRobot dataset episode with Rerun.

The visualizer treats ``meta/info.json`` as the source of truth.  Every
declared feature is classified before the dataset is loaded, then rendered
with a suitable Rerun archetype.  The generated blueprint groups modalities
into tabs, while Rerun's Blueprint panel can be used to show or hide individual
features.

Examples:

    python -m dataset_tool.visualize_lerobot_rerun /data/my_dataset --list-features
    python -m dataset_tool.visualize_lerobot_rerun /data/my_dataset --episode-index 3
    python -m dataset_tool.visualize_lerobot_rerun /data/my_dataset \
        --features action 'observation.*' --save episode_0.rrd
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger("lerobot_rerun")
NUMERIC_DTYPES = ("float", "int", "uint")
TEXT_DTYPES = {"string", "str", "language"}


@dataclass(frozen=True)
class FeatureSpec:
    """Visualization-relevant information from one info.json feature."""

    key: str
    dtype: str
    shape: tuple[int, ...]
    names: tuple[str, ...]
    kind: str


def load_info(dataset_root: Path) -> dict[str, Any]:
    """Load and minimally validate a LeRobot ``meta/info.json`` file."""
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(
            f"Missing LeRobot metadata: {info_path}. Pass the dataset root "
            "that contains meta/info.json."
        )
    with info_path.open(encoding="utf-8") as file:
        info = json.load(file)
    if not isinstance(info.get("features"), dict):
        raise TypeError(f"{info_path} does not contain a 'features' object.")
    return info


def _is_depth_feature(key: str, raw: dict[str, Any]) -> bool:
    normalized_key = key.lower().replace("/", ".")
    if "depth" in normalized_key.split("."):
        return True
    for section_name in ("info", "video_info"):
        section = raw.get(section_name)
        if isinstance(section, dict) and (
            section.get("is_depth_map") or section.get("video.is_depth_map")
        ):
            return True
    return False


def classify_feature(key: str, raw: dict[str, Any]) -> FeatureSpec:
    """Classify one LeRobot feature using only its config metadata."""
    dtype = str(raw.get("dtype", "unknown")).lower()
    try:
        shape = tuple(int(value) for value in (raw.get("shape") or ()))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Feature '{key}' has an invalid shape: {raw.get('shape')!r}") from exc

    raw_names = raw.get("names")
    names = (
        tuple(str(name) for name in raw_names)
        if isinstance(raw_names, (list, tuple))
        else ()
    )

    if dtype in {"image", "video"}:
        kind = "depth" if _is_depth_feature(key, raw) else "image"
    elif dtype in TEXT_DTYPES:
        kind = "text"
    elif dtype == "bool" or dtype.startswith(NUMERIC_DTYPES):
        element_count = int(np.prod(shape)) if shape else 1
        if element_count <= 1:
            kind = "scalar"
        elif len(shape) <= 1:
            kind = "vector"
        else:
            kind = "tensor"
    else:
        # Unknown structured types are still visible rather than silently lost.
        kind = "text"

    return FeatureSpec(key=key, dtype=dtype, shape=shape, names=names, kind=kind)


def discover_features(info: dict[str, Any]) -> list[FeatureSpec]:
    """Return all config-declared features in their metadata order."""
    return [classify_feature(key, raw) for key, raw in info["features"].items()]


def select_features(
    features: Iterable[FeatureSpec], patterns: list[str] | None
) -> list[FeatureSpec]:
    """Filter feature specs by exact names or shell-style wildcard patterns."""
    specs = list(features)
    if not patterns:
        return specs
    expanded = [part for value in patterns for part in value.split(",") if part]
    selected = [
        spec
        for spec in specs
        if any(spec.key == pattern or fnmatchcase(spec.key, pattern) for pattern in expanded)
    ]
    unmatched = [
        pattern
        for pattern in expanded
        if not any(spec.key == pattern or fnmatchcase(spec.key, pattern) for spec in specs)
    ]
    if unmatched:
        raise ValueError(f"Feature pattern(s) matched nothing: {', '.join(unmatched)}")
    return selected


def feature_table(features: Iterable[FeatureSpec]) -> str:
    """Format discovered features for both the terminal and metadata view."""
    rows = list(features)
    headers = ("feature", "dtype", "shape", "Rerun type")
    body = [
        (spec.key, spec.dtype, str(spec.shape or ()), spec.kind) for spec in rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in body))
        if body
        else len(headers[index])
        for index in range(len(headers))
    ]
    formatted = [
        "  ".join(headers[index].ljust(widths[index]) for index in range(len(headers))),
        "  ".join("-" * width for width in widths),
    ]
    formatted.extend(
        "  ".join(row[index].ljust(widths[index]) for index in range(len(headers)))
        for row in body
    )
    return "\n".join(formatted)


def _entity_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return slug or "unnamed"


def _entity_path(spec: FeatureSpec, group: str | None = None) -> str:
    section = group or spec.kind
    return f"/data/{section}/{_entity_slug(spec.key)}"


def _is_beaver_matrix_stack(spec: FeatureSpec) -> bool:
    """Return whether a feature is the configured nine-sensor Beaver grid."""
    return spec.key.startswith("observation.beaver.") and spec.shape == (9, 4, 4)


def _beaver_sensor_path(spec: FeatureSpec, sensor_index: int) -> str:
    return f"{_entity_path(spec)}/sensor_{sensor_index:02d}"


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _channel_last(array: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Convert LeRobot's CHW image tensors back to the configured HWC layout."""
    if array.ndim != 3:
        return array
    if len(shape) == 3 and tuple(array.shape) == (shape[2], shape[0], shape[1]):
        return np.transpose(array, (1, 2, 0))
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        return np.transpose(array, (1, 2, 0))
    return array


def prepare_image(value: Any, spec: FeatureSpec) -> np.ndarray:
    """Convert a LeRobot image/video value into a Rerun-compatible array."""
    image = _channel_last(_as_numpy(value), spec.shape)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    if image.ndim not in (2, 3):
        raise ValueError(f"expected a 2D/3D image, got shape {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        finite = image[np.isfinite(image)]
        if finite.size and finite.min() >= 0.0 and finite.max() <= 1.0:
            image = np.round(image * 255.0).astype(np.uint8)
    return image


def prepare_depth(value: Any, spec: FeatureSpec) -> np.ndarray:
    depth = _channel_last(_as_numpy(value), spec.shape)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"expected a 2D depth map, got shape {depth.shape}")
    return depth


def _scalar(value: Any, fallback: float) -> float:
    array = _as_numpy(value).reshape(-1)
    if array.size == 0:
        return fallback
    return float(array[0])


def _component_names(spec: FeatureSpec, count: int) -> list[str]:
    if len(spec.names) == count:
        return list(spec.names)
    return [f"dim_{index}" for index in range(count)]


def _text_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    array = _as_numpy(value)
    if array.ndim == 0:
        return str(array.item())
    return np.array2string(array, threshold=100)


def build_blueprint(features: Iterable[FeatureSpec], fps: float):
    """Build tabs that act as an in-viewer data-type selector."""
    import rerun.blueprint as rrb

    specs = list(features)
    tabs = []

    image_views = []
    for spec in specs:
        if spec.kind in {"image", "depth"}:
            image_views.append(
                rrb.Spatial2DView(origin=_entity_path(spec), name=spec.key)
            )
    if image_views:
        tabs.append(rrb.Grid(*image_views, name="Images and depth"))

    signal_specs = [spec for spec in specs if spec.kind in {"scalar", "vector"}]
    if signal_specs:
        tabs.append(
            rrb.TimeSeriesView(
                origin="/data/signals",
                contents="/data/signals/**",
                name="Scalar time series",
            )
        )

    vector_views = [
        rrb.BarChartView(origin=_entity_path(spec, "bars"), name=spec.key)
        for spec in specs
        if spec.kind == "vector"
    ]
    if vector_views:
        tabs.append(rrb.Grid(*vector_views, name="Vector values"))

    tensor_views = []
    for spec in specs:
        if spec.kind != "tensor":
            continue
        if _is_beaver_matrix_stack(spec):
            tensor_views.extend(
                rrb.TensorView(
                    origin=_beaver_sensor_path(spec, sensor_index),
                    name=f"{spec.key} · sensor {sensor_index + 1}",
                )
                for sensor_index in range(9)
            )
        else:
            tensor_views.append(rrb.TensorView(origin=_entity_path(spec), name=spec.key))
    if tensor_views:
        tabs.append(rrb.Grid(*tensor_views, name="Tensors"))

    text_views = [
        rrb.TextDocumentView(origin=_entity_path(spec), name=spec.key)
        for spec in specs
        if spec.kind == "text"
    ]
    if text_views:
        tabs.append(rrb.Grid(*text_views, name="Text"))

    tabs.append(rrb.TextDocumentView(origin="/metadata/schema", name="Dataset schema"))
    return rrb.Blueprint(
        rrb.Tabs(*tabs, active_tab=0, name="LeRobot data types"),
        rrb.BlueprintPanel(expanded=True),
        rrb.SelectionPanel(expanded=True),
        rrb.TimePanel(expanded=True, timeline="frame_index", fps=fps if fps > 0 else None),
        collapse_panels=False,
    )


def _log_feature(spec: FeatureSpec, value: Any) -> None:
    import rerun as rr

    if spec.kind == "image":
        rr.log(_entity_path(spec), rr.Image(prepare_image(value, spec)))
        return
    if spec.kind == "depth":
        # AIRO-Doffy records uint16 depth in millimetres.
        rr.log(_entity_path(spec), rr.DepthImage(prepare_depth(value, spec), meter=1000.0))
        return
    if spec.kind == "text":
        rr.log(_entity_path(spec), rr.TextDocument(_text_value(value)))
        return

    numeric = _as_numpy(value)
    if spec.kind == "scalar":
        rr.log(_entity_path(spec, "signals"), rr.Scalars([float(numeric.reshape(-1)[0])]))
    elif spec.kind == "vector":
        vector = numeric.reshape(-1)
        rr.log(_entity_path(spec, "bars"), rr.BarChart(vector))
        for index, (name, component) in enumerate(
            zip(_component_names(spec, len(vector)), vector, strict=True)
        ):
            component_slug = _entity_slug(f"{index:03d}_{name}")
            rr.log(
                f"{_entity_path(spec, 'signals')}/{component_slug}",
                rr.Scalars([float(component)]),
            )
    else:
        if _is_beaver_matrix_stack(spec):
            if numeric.shape != spec.shape:
                raise ValueError(
                    f"expected Beaver tensor shape {spec.shape}, got {numeric.shape}"
                )
            for sensor_index, matrix in enumerate(numeric):
                rr.log(
                    _beaver_sensor_path(spec, sensor_index),
                    rr.Tensor(matrix, dim_names=["row", "column"]),
                )
        else:
            dim_names = list(spec.names) if len(spec.names) == numeric.ndim else None
            rr.log(_entity_path(spec), rr.Tensor(numeric, dim_names=dim_names))


def _metadata_markdown(
    dataset_root: Path,
    info: dict[str, Any],
    all_features: list[FeatureSpec],
    selected: list[FeatureSpec],
) -> str:
    selected_keys = {spec.key for spec in selected}
    lines = [
        "# LeRobot dataset",
        "",
        f"- Root: `{dataset_root}`",
        f"- Robot type: `{info.get('robot_type', 'unknown')}`",
        f"- FPS: `{info.get('fps', 'unknown')}`",
        f"- Episodes: `{info.get('total_episodes', 'unknown')}`",
        "",
        (
            "Use the top tabs to select a data type. Use the Blueprint panel on the "
            "left to show or hide individual views/entities."
        ),
        "",
        "## Config-declared features",
        "",
        "| Feature | dtype | shape | Rerun type | Logged |",
        "|---|---|---|---|---|",
    ]
    lines.extend(
        f"| `{spec.key}` | `{spec.dtype}` | `{spec.shape or ()}` | "
        f"{spec.kind} | {'yes' if spec.key in selected_keys else 'no'} |"
        for spec in all_features
    )
    return "\n".join(lines)


def visualize(
    dataset_root: Path,
    info: dict[str, Any],
    all_features: list[FeatureSpec],
    selected: list[FeatureSpec],
    *,
    episode_index: int,
    video_backend: str,
    save: Path | None,
) -> None:
    """Load one episode and log selected features to Rerun."""
    import rerun as rr
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    total_episodes = int(info.get("total_episodes", 0))
    if episode_index < 0 or (total_episodes and episode_index >= total_episodes):
        raise IndexError(
            f"Episode {episode_index} is outside [0, {max(total_episodes - 1, 0)}]."
        )

    fps = float(info.get("fps", 0.0))
    dataset = LeRobotDataset(
        repo_id=dataset_root.name,
        root=dataset_root,
        episodes=[episode_index],
        video_backend=video_backend,
    )
    if len(dataset) == 0:
        raise ValueError(f"Episode {episode_index} contains no frames.")

    app_id = f"lerobot/{_entity_slug(dataset_root.name)}/episode_{episode_index}"
    blueprint = build_blueprint(selected, fps)
    rr.init(app_id, spawn=save is None)
    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        rr.save(save, default_blueprint=blueprint)
    else:
        rr.send_blueprint(blueprint)

    rr.log(
        "/metadata/schema",
        rr.TextDocument(
            _metadata_markdown(dataset_root, info, all_features, selected),
            media_type=rr.MediaType.MARKDOWN,
        ),
        static=True,
    )

    warned: set[str] = set()
    for offset in range(len(dataset)):
        item = dataset[offset]
        frame_index = int(_scalar(item.get("frame_index"), offset))
        timestamp = _scalar(item.get("timestamp"), offset / fps if fps > 0 else offset)
        rr.set_time("frame_index", sequence=frame_index)
        rr.set_time("timestamp", duration=timestamp)
        for spec in selected:
            if spec.key not in item:
                if spec.key not in warned:
                    LOGGER.warning("Feature '%s' is declared but absent from loaded frames.", spec.key)
                    warned.add(spec.key)
                continue
            try:
                _log_feature(spec, item[spec.key])
            except (TypeError, ValueError, IndexError) as exc:
                if spec.key not in warned:
                    LOGGER.warning("Could not visualize feature '%s': %s", spec.key, exc)
                    warned.add(spec.key)
        if (offset + 1) % 100 == 0 or offset + 1 == len(dataset):
            LOGGER.info("Logged %d/%d frames", offset + 1, len(dataset))

    recording = rr.get_global_data_recording()
    if recording is not None:
        recording.flush()
    if save is not None:
        LOGGER.info("Saved Rerun recording to %s", save)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a local LeRobot dataset episode in Rerun."
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Local LeRobot root containing meta/info.json.",
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--features",
        nargs="+",
        help=(
            "Optional exact feature names or quoted shell-style patterns. "
            "All config-declared features are logged by default."
        ),
    )
    parser.add_argument(
        "--list-features",
        action="store_true",
        help="Print config-declared feature types and exit without loading frames.",
    )
    parser.add_argument(
        "--video-backend",
        default="pyav",
        choices=("pyav", "torchcodec", "video_reader"),
        help="LeRobot video decoder backend (default: pyav).",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Save to an .rrd file instead of spawning the Rerun viewer.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    dataset_root = args.dataset_root.expanduser().resolve()
    info = load_info(dataset_root)
    all_features = discover_features(info)
    print(feature_table(all_features))
    if args.list_features:
        return 0
    selected = select_features(all_features, args.features)
    visualize(
        dataset_root,
        info,
        all_features,
        selected,
        episode_index=args.episode_index,
        video_backend=args.video_backend,
        save=args.save.expanduser().resolve() if args.save else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
