import argparse
import numpy as np
import torch
import rerun as rr
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import config
from pathlib import Path


#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Visualize data of **all** frames of any episode of a dataset of type LeRobotDataset.

Note: The last frame of the episode doesn't always correspond to a final state.
That's because our datasets are composed of transition from state to state up to
the antepenultimate state associated to the ultimate action to arrive in the final state.
However, there might not be a transition from a final state to another state.

Note: This script aims to visualize the data used to train the neural networks.
~What you see is what you get~. When visualizing image modality, it is often expected to observe
lossy compression artifacts since these images have been decoded from compressed mp4 videos to
save disk space. The compression factor applied has been tuned to not affect success rate.

Examples:

- Visualize data stored on a local machine:
```
local$ lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0
```

- Visualize data stored on a distant machine with a local viewer:
```
distant$ lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --save 1 \
    --output-dir path/to/directory

local$ scp distant:path/to/directory/lerobot_pusht_episode_0.rrd .
local$ rerun lerobot_pusht_episode_0.rrd
```

- Visualize data stored on a distant machine through streaming:
(You need to forward the websocket port to the distant machine, with
`ssh -L 9087:localhost:9087 username@remote-host`)
```
distant$ lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --mode distant \
    --ws-port 9087

local$ rerun ws://localhost:9087
```

"""

import argparse
import gc
import logging
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import rerun as rr
import torch
import torch.utils.data
import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, DONE, OBS_STATE, REWARD
TACTILE = "observation.tactile"


class TactileVisualizer:
    def __init__(self):
        # Define 9-row x 6-column grid (inferred from your ASCII drawing)
        # -1 indicates no sensor at that position
        # Your layout:
        # R8:    38 40 41
        # R7: 35 28 29 30 31 32
        # ...
        self.layout = np.array([
            [-1, -1, 38, 40, 41, -1],  # Row 8 (Tip) - Slightly shifted right to align
            [35, 28, 29, 30, 31, 32],  # Row 7
            [-1, 24, 25, 26, 27, -1],  # Row 6
            [36, 20, 21, 22, 23, 33],  # Row 5
            [-1, 16, 17, 18, 19, -1],  # Row 4
            [37, 12, 13, 14, 15, 34],  # Row 3
            [-1, 8, 9, 10, 11, -1],  # Row 2
            [-1, 4, 5, 6, 7, -1],  # Row 1
            [-1, 0, 1, 2, 3, -1],  # Row 0 (Base)
        ])

        # Map Layout ID to Data Index
        # Because data only has 41 rows (index 0-40), but Layout uses ID 41.
        # Assumption: ID 39 is skipped.
        # Mapping rule: For ID < 39, use ID directly; for ID >= 40, use ID-1
        self.id_map = np.full(42, -1, dtype=int)
        for i in range(42):
            if i < 39:
                self.id_map[i] = i
            elif i >= 40:
                self.id_map[i] = i - 1

        # Pre-compute valid (y, x) coordinates and corresponding data indices
        self.valid_mask = (self.layout != -1)
        self.valid_coords = np.where(self.valid_mask)  # (rows, cols)
        valid_ids = self.layout[self.valid_mask]
        self.data_indices = self.id_map[valid_ids]

    def create_heatmap(self, tactile_tensor):
        """
        Input: tactile_tensor (41, 3)
        Output: image (H, W) for Rerun display
        """
        # 1. Calculate force magnitude or take Z-axis (Normal force)
        # Here we take magnitude: sqrt(x^2 + y^2 + z^2)
        forces = np.linalg.norm(tactile_tensor, axis=1)

        # 2. Create image container (fill with NaN so Rerun treats it as background)
        heatmap = np.full(self.layout.shape, np.nan, dtype=np.float32)

        # 3. Fill data
        # Check if index is out of bounds (prevent data shape mismatch)
        max_idx = tactile_tensor.shape[0] - 1
        safe_indices = np.clip(self.data_indices, 0, max_idx)

        heatmap[self.valid_coords] = forces[safe_indices]

        return heatmap

class EpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
        to_idx = dataset.meta.episodes["dataset_to_index"][episode_index]
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self) -> Iterator:
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    assert chw_float32_torch.dtype == torch.float32
    assert chw_float32_torch.ndim == 3
    c, h, w = chw_float32_torch.shape
    assert c < h and c < w, f"expect channel first images, but instead {chw_float32_torch.shape}"
    hwc_uint8_numpy = (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
    return hwc_uint8_numpy


def visualize_dataset(
    dataset: LeRobotDataset,
    episode_index: int,
    batch_size: int = 32,
    num_workers: int = 0,
    mode: str = "local",
    web_port: int = 9090,
    ws_port: int = 9087,
    save: bool = False,
    output_dir: Path | None = None,
) -> Path | None:
    if save:
        assert output_dir is not None, (
            "Set an output directory where to write .rrd files with `--output-dir path/to/directory`."
        )

    repo_id = dataset.repo_id
    tactile_viz = TactileVisualizer()
    logging.info("Loading dataloader")
    episode_sampler = EpisodeSampler(dataset, episode_index)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        sampler=episode_sampler,
    )

    logging.info("Starting Rerun")

    if mode not in ["local", "distant"]:
        raise ValueError(mode)

    spawn_local_viewer = mode == "local" and not save
    rr.init(f"{repo_id}/episode_{episode_index}", spawn=spawn_local_viewer)

    # Manually call python garbage collector after `rr.init` to avoid hanging in a blocking flush
    # when iterating on a dataloader with `num_workers` > 0
    # TODO(rcadene): remove `gc.collect` when rerun version 0.16 is out, which includes a fix
    gc.collect()

    if mode == "distant":
        rr.serve_web_viewer(open_browser=False, web_port=web_port)

    logging.info("Logging to Rerun")

    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        # iterate over the batch
        for i in range(len(batch["index"])):
            rr.set_time("frame_index", sequence=batch["frame_index"][i].item())
            rr.set_time("timestamp", timestamp=batch["timestamp"][i].item())

            # display each camera image
            for key in dataset.meta.camera_keys:
                # TODO(rcadene): add `.compress()`? is it lossless?
                rr.log(key, rr.Image(to_hwc_uint8_numpy(batch[key][i])))

            # display each dimension of action space (e.g. actuators command)
            if ACTION in batch:
                for dim_idx, val in enumerate(batch[ACTION][i]):
                    rr.log(f"{ACTION}/{dim_idx}", rr.Scalars(val.item()))

            # display each dimension of observed state space (e.g. agent position in joint space)
            if OBS_STATE in batch:
                for dim_idx, val in enumerate(batch[OBS_STATE][i]):
                    rr.log(f"state/{dim_idx}", rr.Scalars(val.item()))

            if TACTILE in batch:
                # Get tactile data for current frame, shape should be (41, 3)
                tactile_data = batch[TACTILE][i]
                heatmap = tactile_viz.create_heatmap(tactile_data)

                # Log to Rerun
                # Using rr.Image automatically applies colormap based on value magnitude (default is grayscale, but adjustable in Viewer)
                # Can manually specify tensor and mark as depth view etc., but Image is most universal
                rr.log(
                    f"{TACTILE}/heatmap",
                    rr.Image(heatmap)
                )
                # Method A: Use Tensor visualization (recommended)
                # This displays as a grid heatmap in Rerun, intuitively showing which sensors have larger values
                # rr.log(TACTILE, rr.Tensor(tactile_data))

                # Method B (optional): If you want to see the magnitude of each sensor (total force size), uncomment below
                tactile_mag = torch.norm(tactile_data, dim=1) # Shape (41,)
                rr.log(f"{TACTILE}_magnitude", rr.BarChart(tactile_mag))
            if DONE in batch:
                rr.log(DONE, rr.Scalars(batch[DONE][i].item()))

            if REWARD in batch:
                rr.log(REWARD, rr.Scalars(batch[REWARD][i].item()))

            if "next.success" in batch:
                rr.log("next.success", rr.Scalars(batch["next.success"][i].item()))

    if mode == "local" and save:
        # save .rrd locally
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        repo_id_str = repo_id.replace("/", "_")
        rrd_path = output_dir / f"{repo_id_str}_episode_{episode_index}.rrd"
        rr.save(rrd_path)
        return rrd_path

    elif mode == "distant":
        # stop the process from exiting since it is serving the websocket connection
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("Ctrl-C received. Exiting.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Name of hugging face repository containing a LeRobotDataset dataset (e.g. `lerobot/pusht`).",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        required=True,
        help="Episode to visualize.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for the dataset stored locally (e.g. `--root data`). By default, the dataset will be loaded from hugging face cache folder, or downloaded from the hub if available.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory path to write a .rrd file when `--save 1` is set.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size loaded by DataLoader.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of processes of Dataloader for loading the data.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="local",
        help=(
            "Mode of viewing between 'local' or 'distant'. "
            "'local' requires data to be on a local machine. It spawns a viewer to visualize the data locally. "
            "'distant' creates a server on the distant machine where the data is stored. "
            "Visualize the data by connecting to the server with `rerun ws://localhost:PORT` on the local machine."
        ),
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=9090,
        help="Web port for rerun.io when `--mode distant` is set.",
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=9087,
        help="Web socket port for rerun.io when `--mode distant` is set.",
    )
    parser.add_argument(
        "--save",
        type=int,
        default=0,
        help=(
            "Save a .rrd file in the directory provided by `--output-dir`. "
            "It also deactivates the spawning of a viewer. "
            "Visualize the data by running `rerun path/to/file.rrd` on your local machine."
        ),
    )

    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help=(
            "Tolerance in seconds used to ensure data timestamps respect the dataset fps value"
            "This is argument passed to the constructor of LeRobotDataset and maps to its tolerance_s constructor argument"
            "If not given, defaults to 1e-4."
        ),
    )

    args = parser.parse_args()
    kwargs = vars(args)
    repo_id = kwargs.pop("repo_id")
    root = kwargs.pop("root")
    cfg = config.Config()

    # --- Key modification: Path parsing ---
    # Assume cfg.DATASET_DIR = "/home/user/datasets/data"
    # And your actual dataset folder is "/home/user/datasets/data_lero"
    dataset_dir = Path(cfg.DATASET_DIR + "_lero")

    # repo_id is usually the folder name (e.g. "WipeBoard_lero")
    repo_id = dataset_dir.name

    # root must be the parent directory containing the folder (e.g. "/home/user/datasets")
    root = dataset_dir
    tolerance_s = kwargs.pop("tolerance_s")

    logging.info("Loading dataset")
    dataset = LeRobotDataset(repo_id, episodes=[args.episode_index], root=root, tolerance_s=tolerance_s,video_backend="pyav")

    visualize_dataset(dataset, **vars(args))


if __name__ == "__main__":
    main()
