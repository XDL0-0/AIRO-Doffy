"""Dataset recording in HDF5 (ACT) or LeRobot format."""

from __future__ import annotations

import os
import time
import shutil
from pathlib import Path

import h5py
import numpy as np

import utils
from config import Config
from airo_camera_toolkit.cameras.realsense.realsense import Realsense


class DatasetRecorder:
    def __init__(self, camera_num: int):
        cfg = Config()
        self.save_eef = cfg.SAVE_EEF
        self.task_description = cfg.TASK_NAME
        self.camera_num = camera_num
        self.dataset_type = cfg.DATASET_TYPE
        self.data_type = cfg.DATA_TYPE
        self.push_to_hub = cfg.PUSH_TO_HUB if self.dataset_type == "l" else False
        self.tactile_mode = cfg.TACTILE_TRANSFER
        self.force_mode = cfg.FORCE_COLLECT and cfg.TORQUE_MODE
        self.fps = cfg.COLLECT_RATE

        suffix = "_lero" if self.dataset_type == "l" else "_hdf5"
        self.dataset_dir = Path(cfg.DATASET_DIR + suffix)

        self.feature_dim = 8 if self.save_eef else 7

        self.data_dict: dict[str, list] = {}
        self.collect_step = 0
        self.lerobot_dataset = None
        self.recorded_episodes = 0

        utils.logger.info(f"Dataset Dir: {self.dataset_dir}")
        utils.logger.info(f"Dataset Type: {self.dataset_type}")
        utils.logger.info(f"Data Type: {self.data_type}")
        utils.logger.info(f"Feature dim: {self.feature_dim}")

        self._reset_data_dict()
        self._init_dataset()

    # ── Data dict management ──────────────────────────────────────────────

    def _reset_data_dict(self) -> None:
        self.data_dict = {
            "/observations/qpos": [],
            "/action": [],
        }
        if self.force_mode:
            self.data_dict["/observations/force"] = []
        if self.tactile_mode:
            self.data_dict["/observations/tactile"] = []
        for i in range(self.camera_num):
            self.data_dict[f"/observations/images/camera_{i}"] = []
        self.collect_step = 0

    # ── Dataset initialisation ────────────────────────────────────────────

    def _init_dataset(self) -> None:
        if self.dataset_type == "a":
            self._init_hdf5()
        elif self.dataset_type == "l":
            self._init_lerobot()

    def _init_hdf5(self) -> None:
        os.makedirs(self.dataset_dir, exist_ok=True)
        existing = 0
        for f in os.listdir(self.dataset_dir):
            if f.startswith("episode_") and f.endswith(".hdf5"):
                try:
                    num = int(f.split("_")[1].split(".")[0])
                    existing = max(existing, num + 1)
                except ValueError:
                    continue
        if existing:
            utils.logger.warning(
                f"Dataset already exists. Recording from episode {existing}"
            )
        self.recorded_episodes = existing

    def _init_lerobot(self) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        repo_id = self.dataset_dir.name
        root = self.dataset_dir
        h, w = 480, 640

        features = {
            "action": {
                "dtype": "float32",
                "shape": (self.feature_dim,),
                "names": [f"motor_{i}" for i in range(self.feature_dim)],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (self.feature_dim,),
                "names": [f"joint_{i}" for i in range(self.feature_dim)],
            },
        }
        if self.force_mode:
            features["observation.force"] = {
                "dtype": "float32",
                "shape": (6,),
                "names": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
            }
        for i in range(self.camera_num):
            features[f"observation.images.camera_{i}"] = {
                "dtype": "video",
                "shape": (h, w, 3),
                "names": ["height", "width", "channel"],
            }
        if self.tactile_mode:
            features["observation.tactile"] = {
                "dtype": "float32",
                "shape": (41, 3),
                "names": ["sensor_idx", "axis"],
            }

        expected_keys = set(features.keys())

        try:
            self.lerobot_dataset = LeRobotDataset(repo_id=repo_id, root=root)

            loaded_keys = {
                k for k in self.lerobot_dataset.features
                if k.startswith(("action", "observation."))
            }
            if loaded_keys != expected_keys:
                missing = expected_keys - loaded_keys
                extra = loaded_keys - expected_keys
                utils.logger.warning("Feature schema mismatch!")
                if missing:
                    utils.logger.warning(f"  Missing: {missing}")
                if extra:
                    utils.logger.warning(f"  Extra:   {extra}")
                utils.logger.warning("Recreating dataset to match current config...")
                shutil.rmtree(root)
                raise ValueError("Feature schema mismatch")

            self.recorded_episodes = self.lerobot_dataset.num_episodes
            utils.logger.warning(
                f"LeRobot Dataset found. Continuing from episode {self.recorded_episodes}"
            )

        except (FileNotFoundError, ValueError, OSError):
            utils.logger.info("Creating new LeRobot Dataset.")
            if root.exists():
                shutil.rmtree(root)
            self.lerobot_dataset = LeRobotDataset.create(
                repo_id=repo_id,
                root=root,
                fps=self.fps,
                robot_type="ur_custom",
                features=features,
                use_videos=True,
                image_writer_processes=0,
            )
            self.recorded_episodes = 0

    # ── Data collection ───────────────────────────────────────────────────

    def data_collection(
        self,
        state: np.ndarray,
        action: np.ndarray,
        camera_images: dict[str, np.ndarray],
        tactile_data: np.ndarray | None = None,
        force_data: np.ndarray | None = None,
    ) -> None:
        self.collect_step += 1
        self.data_dict["/observations/qpos"].append(state)
        self.data_dict["/action"].append(action)

        if self.force_mode and force_data is not None:
            self.data_dict["/observations/force"].append(force_data)
        if self.tactile_mode and tactile_data is not None:
            self.data_dict["/observations/tactile"].append(tactile_data)
        for name, img in camera_images.items():
            self.data_dict[f"/observations/images/{name}"].append(img)

    # ── Data export ───────────────────────────────────────────────────────

    def data_export(self, cu_manager) -> None:
        t0 = time.time()
        max_timesteps = len(self.data_dict["/observations/qpos"])

        utils.logger.info(f"max_timesteps: {max_timesteps}")
        utils.logger.info(f"collect_step: {self.collect_step}")

        if self.dataset_type == "l":
            self._export_lerobot(max_timesteps)
        elif self.dataset_type == "a":
            self._export_hdf5(max_timesteps, cu_manager)

        self.recorded_episodes += 1
        utils.logger.info(f"Saving took {time.time() - t0:.1f}s")

    def _export_lerobot(self, max_timesteps: int) -> None:
        if self.lerobot_dataset is None:
            utils.logger.error("LeRobot Dataset not initialized!")
            return

        utils.logger.info(
            f"Exporting Episode {self.recorded_episodes} to LeRobot format..."
        )
        self.lerobot_dataset.create_episode_buffer(
            episode_index=self.recorded_episodes
        )

        for i in range(max_timesteps):
            if i % 50 == 0:
                utils.logger.info(f"Processing frame {i}/{max_timesteps}...")

            frame_data = {
                "observation.state": np.array(
                    self.data_dict["/observations/qpos"][i], dtype=np.float32
                ),
                "action": np.array(
                    self.data_dict["/action"][i], dtype=np.float32
                ),
                "task": self.task_description,
            }

            if self.force_mode:
                frame_data["observation.force"] = np.array(
                    self.data_dict["/observations/force"][i], dtype=np.float32
                )
            if self.tactile_mode:
                frame_data["observation.tactile"] = np.array(
                    self.data_dict["/observations/tactile"][i], dtype=np.float32
                )
            for key in self.data_dict:
                if "images" in key:
                    cam_name = key.split("/")[-1]
                    frame_data[f"observation.images.{cam_name}"] = np.array(
                        self.data_dict[key][i], dtype=np.uint8
                    )

            self.lerobot_dataset.add_frame(frame_data)

        self.lerobot_dataset.save_episode()
        utils.logger.info("LeRobot episode saved.")

    def _export_hdf5(self, max_timesteps: int, cu_manager) -> None:
        path = os.path.join(self.dataset_dir, f"episode_{self.recorded_episodes}.hdf5")

        with h5py.File(path, "w", rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs["sim"] = False
            obs = root.create_group("observations")
            image_grp = obs.create_group("images")

            obs.create_dataset("qpos", (max_timesteps, self.feature_dim))
            root.create_dataset("action", (max_timesteps, self.feature_dim))

            if self.force_mode:
                obs.create_dataset("force", (max_timesteps, 6))
            if self.tactile_mode:
                obs.create_dataset("tactile", (max_timesteps, 41, 3))

            for name in cu_manager.camera_images:
                image_grp.create_dataset(
                    name,
                    (max_timesteps, *Realsense.RESOLUTION_480[::-1], 3),
                    dtype="uint8",
                    chunks=(1, *Realsense.RESOLUTION_480[::-1], 3),
                )

            for name, array in self.data_dict.items():
                root[name][...] = array[:max_timesteps]

        desc_path = os.path.join(self.dataset_dir, "episode_descriptions.txt")
        with open(desc_path, "a") as f:
            f.write(
                f"Episode {self.recorded_episodes}: max_timesteps = {max_timesteps}\n"
            )

    # ── Finalize ──────────────────────────────────────────────────────────

    def close(self) -> None:
        if self.dataset_type == "l" and self.lerobot_dataset is not None:
            self.lerobot_dataset.finalize()
            if self.push_to_hub:
                self.lerobot_dataset.push_to_hub()
