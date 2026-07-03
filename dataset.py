"""Dataset recording in HDF5 (ACT) or LeRobot format."""

from __future__ import annotations

import os
import time
import shutil
import json
from pathlib import Path

import h5py
import numpy as np

import utils
from config import Config
from data_schema import build_data_schema, should_store_extra_tcp_pose


class DatasetRecorder:
    def __init__(
        self,
        camera_num: int,
        robot_dof: int | None = None,
        robot_type: str | None = None,
        force_collect: bool | None = None,
        torque_collect: bool | None = None,
    ):
        cfg = Config()
        self.save_eef = cfg.SAVE_EEF
        self.task_description = cfg.TASK_NAME
        self.camera_num = camera_num
        self.dataset_type = cfg.DATASET_TYPE
        self.data_type = cfg.DATA_TYPE
        self.robot_dof = int(robot_dof if robot_dof is not None else len(cfg.INITIAL_JOINT))
        self.robot_type = robot_type or cfg.ROBOT_TYPE
        self.schema = build_data_schema(self.data_type, self.robot_dof)
        self.push_to_hub = cfg.PUSH_TO_HUB if self.dataset_type == "l" else False
        self.tactile_mode = cfg.TACTILE_TRANSFER
        self.tactile_shape = tuple(cfg.TACTILE_SHAPE)
        self.force_collect = bool(force_collect) if force_collect is not None else cfg.FORCE_COLLECT
        self.torque_collect = bool(torque_collect) if torque_collect is not None else cfg.TORQUE_COLLECT
        self.depth_mode = cfg.DEPTH_INFO_ENABLE
        self.fps = cfg.COLLECT_RATE
        self.resolution = cfg.REALSENSE_RESOLUTION  # (width, height)

        suffix = "_lero" if self.dataset_type == "l" else "_hdf5"
        self.dataset_dir = Path(cfg.DATASET_DIR + suffix)

        self.collect_tcp_extra = should_store_extra_tcp_pose(self.data_type)
        self.state_dim = self.schema.state_dim
        self.action_dim = self.schema.action_dim
        self.tcp_pose_dim = 7
        self.timestamp_names = (
            ["collect", "robot_state", "robot_action", "vr_input", "tactile"]
            + [f"camera_{i}" for i in range(self.camera_num)]
        )
        self.timestamp_dim = len(self.timestamp_names)

        self.data_dict: dict[str, list] = {}
        self.collect_step = 0
        self.lerobot_dataset = None
        self.recorded_episodes = 0
        self._lerobot_episode_started = False
        self._last_episode_length_cache: int | None = None
        self._last_episode_length_cache_for: int | None = None

        utils.logger.info(f"Dataset Dir: {self.dataset_dir}")
        utils.logger.info(f"Dataset Type: {self.dataset_type}")
        utils.logger.info(f"Data Type: {self.data_type}")
        utils.logger.info(f"Robot DoF: {self.robot_dof}")
        utils.logger.info(f"State dim: {self.state_dim}")
        utils.logger.info(f"Action dim: {self.action_dim}")
        if self.tactile_mode:
            utils.logger.info(f"Tactile shape: {self.tactile_shape}")

        self._reset_data_dict()
        self._init_dataset()

    # ── Data dict management ──────────────────────────────────────────────

    def _reset_data_dict(self) -> None:
        self.data_dict = {
            "/observations/qpos": [],
            "/action": [],
            "/extra/timestamps_ns": [],
        }
        if self.collect_tcp_extra:
            self.data_dict["/extra/tcp_pose"] = []
        if self.force_collect:
            self.data_dict["/observations/force"] = []
        if self.torque_collect:
            self.data_dict["/observations/torque"] = []
        if self.tactile_mode:
            self.data_dict["/observations/tactile"] = []
        for i in range(self.camera_num):
            self.data_dict[f"/observations/images/camera_{i}"] = []
        if self.depth_mode:
            for i in range(self.camera_num):
                self.data_dict[f"/observations/depth/camera_{i}"] = []
        self.collect_step = 0
        self._lerobot_episode_started = False

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
        w, h = self.resolution  # (width, height) → unpack

        features = {
            "action": {
                "dtype": "float32",
                "shape": (self.action_dim,),
                "names": self.schema.action_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (self.state_dim,),
                "names": self.schema.state_names,
            },
            "extra.timestamps_ns": {
                "dtype": "int64",
                "shape": (self.timestamp_dim,),
                "names": self.timestamp_names,
            },
        }
        if self.force_collect:
            features["observation.force"] = {
                "dtype": "float32",
                "shape": (3,),
                "names": ["Fx", "Fy", "Fz"],
            }
        if self.torque_collect:
            features["observation.torque"] = {
                "dtype": "float32",
                "shape": (3,),
                "names": ["Tx", "Ty", "Tz"],
            }
        if self.collect_tcp_extra:
            features["extra.tcp_pose"] = {
                "dtype": "float32",
                "shape": (self.tcp_pose_dim,),
                "names": ["qx", "qy", "qz", "qw", "x", "y", "z"],
            }
        for i in range(self.camera_num):
            features[f"observation.images.camera_{i}"] = {
                "dtype": "video",
                "shape": (h, w, 3),
                "names": ["height", "width", "channel"],
            }
        if self.depth_mode:
            for i in range(self.camera_num):
                features[f"observation.depth.camera_{i}"] = {
                    "dtype": "image",
                    "shape": (h, w, 1),
                    "names": ["height", "width", "channel"],
                }
        if self.tactile_mode:
            features["observation.tactile"] = {
                "dtype": "float32",
                "shape": self.tactile_shape,
                "names": ["sensor_idx", "axis"],
            }

        expected_keys = set(features.keys())

        try:
            probe_dataset = LeRobotDataset(repo_id=repo_id, root=root)

            loaded_keys = {
                k for k in probe_dataset.features
                if k.startswith(("action", "observation.", "extra."))
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

            self.recorded_episodes = probe_dataset.num_episodes
            probe_dataset = None
            if hasattr(LeRobotDataset, "resume"):
                self.lerobot_dataset = LeRobotDataset.resume(repo_id=repo_id, root=root)
            else:
                self.lerobot_dataset = LeRobotDataset(repo_id=repo_id, root=root)
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
                robot_type=self.robot_type,
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
        wrench_data: np.ndarray | None = None,
        depth_images: dict[str, np.ndarray] | None = None,
        extra_data: dict[str, object] | None = None,
    ) -> None:
        state = self._coerce_vector(state, self.state_dim, "state")
        action = self._coerce_vector(action, self.action_dim, "action")
        # ── LeRobot: per-cycle add_frame (skip data_dict buffer) ──────────
        if self.dataset_type == "l":
            self._lerobot_add_frame(
                state,
                action,
                camera_images,
                tactile_data,
                wrench_data,
                depth_images,
                extra_data,
            )
            return

        # ── HDF5: buffer into data_dict as before ─────────────────────────
        self.collect_step += 1
        self.data_dict["/observations/qpos"].append(state)
        self.data_dict["/action"].append(action)
        if self.collect_tcp_extra:
            tcp_pose = self._get_tcp_pose_extra(extra_data)
            self.data_dict["/extra/tcp_pose"].append(tcp_pose)
        self.data_dict["/extra/timestamps_ns"].append(
            self._get_timestamps_extra(extra_data)
        )

        force, torque = self._split_wrench(wrench_data)
        if self.force_collect and force is not None:
            self.data_dict["/observations/force"].append(force)
        if self.torque_collect and torque is not None:
            self.data_dict["/observations/torque"].append(torque)
        if self.tactile_mode:
            self.data_dict["/observations/tactile"].append(
                self._format_tactile(tactile_data)
            )
        for name, img in camera_images.items():
            self.data_dict[f"/observations/images/{name}"].append(img)
        if self.depth_mode and depth_images is not None:
            for name, depth in depth_images.items():
                self.data_dict[f"/observations/depth/{name}"].append(depth)

    def _lerobot_add_frame(
        self,
        state: np.ndarray,
        action: np.ndarray,
        camera_images: dict[str, np.ndarray],
        tactile_data: np.ndarray | None,
        wrench_data: np.ndarray | None,
        depth_images: dict[str, np.ndarray] | None,
        extra_data: dict[str, object] | None,
    ) -> None:
        """Build one frame dict and call add_frame immediately (LeRobot only)."""
        if self.lerobot_dataset is None:
            self._init_lerobot()

        if not self._lerobot_episode_started:
            if hasattr(self.lerobot_dataset, "create_episode_buffer"):
                self.lerobot_dataset.episode_buffer = self.lerobot_dataset.create_episode_buffer(
                    episode_index=self.recorded_episodes
                )
            self._lerobot_episode_started = True

        frame_data: dict = {
            "observation.state": self._coerce_vector(state, self.state_dim, "state").astype(np.float32),
            "action": self._coerce_vector(action, self.action_dim, "action").astype(np.float32),
            "task": self.task_description,
        }

        force, torque = self._split_wrench(wrench_data)
        if self.force_collect and force is not None:
            frame_data["observation.force"] = np.array(force, dtype=np.float32)
        if self.torque_collect and torque is not None:
            frame_data["observation.torque"] = np.array(torque, dtype=np.float32)
        if self.collect_tcp_extra:
            frame_data["extra.tcp_pose"] = self._get_tcp_pose_extra(extra_data)
        frame_data["extra.timestamps_ns"] = self._get_timestamps_extra(extra_data)
        if self.tactile_mode:
            # Always write tactile: zeros fallback keeps episode schemas complete
            # while the sensor connects or calibrates.
            frame_data["observation.tactile"] = self._format_tactile(tactile_data)

        for name, img in camera_images.items():
            frame_data[f"observation.images.{name}"] = np.array(img, dtype=np.uint8)

        if self.depth_mode and depth_images is not None:
            for name, depth in depth_images.items():
                depth_m = np.array(depth, dtype=np.float32)
                depth_uint16 = (np.clip(depth_m, 0, 65.535) * 1000).astype(np.uint16)
                frame_data[f"observation.depth.{name}"] = depth_uint16[..., np.newaxis]

        self.lerobot_dataset.add_frame(frame_data)
        self.collect_step += 1

    def _get_tcp_pose_extra(
        self, extra_data: dict[str, object] | None
    ) -> np.ndarray:
        if extra_data is not None and "tcp_pose" in extra_data:
            return np.array(extra_data["tcp_pose"], dtype=np.float32)
        return np.zeros((self.tcp_pose_dim,), dtype=np.float32)

    def _format_tactile(self, tactile_data: np.ndarray | None) -> np.ndarray:
        if tactile_data is None:
            return np.zeros(self.tactile_shape, dtype=np.float32)

        tactile = np.asarray(tactile_data, dtype=np.float32)
        if tactile.shape == self.tactile_shape:
            return tactile
        if tactile.size == int(np.prod(self.tactile_shape)):
            return tactile.reshape(self.tactile_shape)
        raise ValueError(
            f"Expected tactile shape {self.tactile_shape}, got {tactile.shape}"
        )

    def _get_timestamps_extra(
        self, extra_data: dict[str, object] | None
    ) -> np.ndarray:
        values = np.zeros((self.timestamp_dim,), dtype=np.int64)
        if extra_data is None:
            return values

        scalar_keys = [
            "collect_timestamp_ns",
            "robot_state_timestamp_ns",
            "robot_action_timestamp_ns",
            "vr_input_timestamp_ns",
            "tactile_timestamp_ns",
        ]
        for idx, key in enumerate(scalar_keys):
            value = extra_data.get(key)
            if value is not None:
                values[idx] = int(np.asarray(value).item())

        camera_timestamps = extra_data.get("camera_timestamps_ns")
        if isinstance(camera_timestamps, dict):
            for cam_idx in range(self.camera_num):
                values[5 + cam_idx] = int(camera_timestamps.get(f"camera_{cam_idx}", 0))
        return values

    @staticmethod
    def _split_wrench(wrench_data: np.ndarray | None) -> tuple[np.ndarray | None, np.ndarray | None]:
        if wrench_data is None:
            return None, None
        wrench = np.asarray(wrench_data, dtype=np.float32).reshape(-1)
        if wrench.size < 6:
            raise ValueError(f"Wrench data has shape {wrench.shape}; expected at least 6 values.")
        return wrench[:3], wrench[3:6]

    @staticmethod
    def _coerce_vector(values: np.ndarray, expected_dim: int, name: str) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if arr.shape == (expected_dim,):
            return arr
        raise ValueError(f"Dataset {name} vector has shape {arr.shape}; expected ({expected_dim},).")

    # ── Data export ───────────────────────────────────────────────────────

    def data_export(self, cu_manager) -> None:
        t0 = time.time()

        utils.logger.info(f"collect_step: {self.collect_step}")

        exported = False
        if self.dataset_type == "l":
            exported = self._export_lerobot()
        elif self.dataset_type == "a":
            max_timesteps = len(self.data_dict["/observations/qpos"])
            utils.logger.info(f"max_timesteps: {max_timesteps}")
            self._export_hdf5(max_timesteps, cu_manager)
            exported = True

        if exported:
            self.recorded_episodes += 1
        utils.logger.info(f"Saving took {time.time() - t0:.1f}s")

    def recording_status(self, collecting: bool = False) -> dict[str, object]:
        """Small status snapshot for the live visualizer."""
        return {
            "dataset_type": self.dataset_type,
            "dataset_dir": str(self.dataset_dir),
            "recorded_episodes": int(self.recorded_episodes),
            "current_episode_frames": int(self.collect_step),
            "last_episode_length": self._cached_last_episode_length(),
            "collecting": bool(collecting),
        }

    def _cached_last_episode_length(self) -> int | None:
        episode_index = self.recorded_episodes - 1
        if self._last_episode_length_cache_for == episode_index:
            return self._last_episode_length_cache
        self._last_episode_length_cache_for = episode_index
        self._last_episode_length_cache = self._last_episode_length(episode_index)
        return self._last_episode_length_cache

    def _last_episode_length(self, episode_index: int) -> int | None:
        if episode_index < 0:
            return None
        if self.dataset_type == "a":
            path = self.dataset_dir / f"episode_{episode_index}.hdf5"
            if not path.exists():
                return None
            try:
                with h5py.File(path, "r") as root:
                    return int(root["/action"].shape[0])
            except Exception:
                return None
        if self.dataset_type == "l":
            try:
                for _path, df in self._load_lerobot_episode_metadata():
                    if "episode_index" not in df or "length" not in df:
                        continue
                    hit = df[df["episode_index"] == episode_index]
                    if not hit.empty:
                        return int(hit.iloc[-1]["length"])
            except Exception:
                return None
        return None

    def _export_lerobot(self) -> bool:
        """Finalize the current episode. Frames were already added per-cycle."""
        if self.lerobot_dataset is None:
            utils.logger.error("LeRobot Dataset not initialized!")
            return False
        if not self._lerobot_episode_started:
            utils.logger.error("No LeRobot episode in progress!")
            return False

        utils.logger.info(
            f"Saving Episode {self.recorded_episodes} "
            f"({self.collect_step} frames already added)..."
        )
        self.lerobot_dataset.save_episode()
        # LeRobot keeps parquet writers open for efficient batch recording.
        # Closing after each episode writes the parquet footer immediately, so
        # appended episodes remain replayable even if the recorder stops later.
        self.lerobot_dataset.finalize()
        self.lerobot_dataset = None
        utils.logger.info("LeRobot episode saved and finalized.")
        return True

    def _export_hdf5(self, max_timesteps: int, cu_manager) -> None:
        path = os.path.join(self.dataset_dir, f"episode_{self.recorded_episodes}.hdf5")

        with h5py.File(path, "w", rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs["sim"] = False
            obs = root.create_group("observations")
            image_grp = obs.create_group("images")

            obs.create_dataset("qpos", (max_timesteps, self.state_dim))
            root.create_dataset("action", (max_timesteps, self.action_dim))
            extra_grp = root.create_group("extra")
            ts_ds = extra_grp.create_dataset(
                "timestamps_ns", (max_timesteps, self.timestamp_dim), dtype="int64"
            )
            ts_ds.attrs["names"] = np.array(self.timestamp_names, dtype="S")
            if self.collect_tcp_extra:
                extra_grp.create_dataset("tcp_pose", (max_timesteps, self.tcp_pose_dim))

            if self.force_collect:
                obs.create_dataset("force", (max_timesteps, 3))
            if self.torque_collect:
                obs.create_dataset("torque", (max_timesteps, 3))
            if self.tactile_mode:
                obs.create_dataset("tactile", (max_timesteps, *self.tactile_shape))

            w, h = self.resolution
            img_shape = (h, w)  # (height, width) for numpy
            for name in cu_manager.camera_images:
                image_grp.create_dataset(
                    name,
                    (max_timesteps, *img_shape, 3),
                    dtype="uint8",
                    chunks=(1, *img_shape, 3),
                )

            if self.depth_mode:
                depth_grp = obs.create_group("depth")
                for name in cu_manager.camera_images:
                    depth_grp.create_dataset(
                        name,
                        (max_timesteps, *img_shape),
                        dtype="float32",
                        chunks=(1, *img_shape),
                    )

            for name, array in self.data_dict.items():
                root[name][...] = array[:max_timesteps]

        desc_path = os.path.join(self.dataset_dir, "episode_descriptions.txt")
        with open(desc_path, "a") as f:
            f.write(
                f"Episode {self.recorded_episodes}: max_timesteps = {max_timesteps}\n"
            )

    # ── Rollback ─────────────────────────────────────────────────────────

    def rollback_last_episode(self) -> bool:
        """Delete the most recently recorded episode and reuse its index."""
        if self.dataset_type == "l":
            return self._rollback_lerobot()
        if self.dataset_type == "a":
            return self._rollback_hdf5()
        utils.logger.error(f"Unsupported dataset type for rollback: {self.dataset_type}")
        return False

    def _rollback_hdf5(self) -> bool:
        if self.collect_step:
            self._reset_data_dict()
            utils.logger.info("Discarded unsaved HDF5 episode buffer.")
            return True

        if self.recorded_episodes <= 0:
            utils.logger.warning("No HDF5 episode to rollback.")
            return False

        episode_index = self.recorded_episodes - 1
        path = self.dataset_dir / f"episode_{episode_index}.hdf5"
        if not path.exists():
            utils.logger.error(f"Cannot rollback: missing {path}")
            return False

        path.unlink()
        self._trim_hdf5_description(episode_index)
        self.recorded_episodes = episode_index
        self._last_episode_length_cache_for = None
        utils.logger.info(f"Rolled back HDF5 episode {episode_index}.")
        return True

    def _trim_hdf5_description(self, episode_index: int) -> None:
        desc_path = self.dataset_dir / "episode_descriptions.txt"
        if not desc_path.exists():
            return
        lines = desc_path.read_text().splitlines()
        prefix = f"Episode {episode_index}:"
        kept = [line for line in lines if not line.startswith(prefix)]
        desc_path.write_text("\n".join(kept) + ("\n" if kept else ""))

    def _rollback_lerobot(self) -> bool:
        if self._has_lerobot_pending_frames():
            if self.lerobot_dataset is not None and hasattr(
                self.lerobot_dataset, "clear_episode_buffer"
            ):
                self.lerobot_dataset.clear_episode_buffer()
            self._reset_data_dict()
            utils.logger.info("Discarded unsaved LeRobot episode buffer.")
            return True

        if self.lerobot_dataset is not None:
            self.lerobot_dataset.finalize()
            self.lerobot_dataset = None

        info_path = self.dataset_dir / "meta" / "info.json"
        if not info_path.exists():
            utils.logger.warning("No LeRobot metadata found for rollback.")
            return False

        info = json.loads(info_path.read_text())
        total_episodes = int(info.get("total_episodes", self.recorded_episodes))
        if total_episodes <= 0:
            utils.logger.warning("No LeRobot episode to rollback.")
            self.recorded_episodes = 0
            return False

        episode_index = total_episodes - 1
        episodes_by_file = self._load_lerobot_episode_metadata()
        episode_row = None
        remaining_episodes = []

        for _meta_path, df in episodes_by_file:
            if "episode_index" not in df:
                continue
            hit = df[df["episode_index"] == episode_index]
            if not hit.empty and episode_row is None:
                episode_row = hit.iloc[-1]
            remaining_episodes.append(df[df["episode_index"] != episode_index])

        if episode_row is None:
            utils.logger.warning(
                f"Episode metadata for {episode_index} not found; deleting files by convention."
            )
            episode_length = self._rollback_lerobot_files_by_index(info, episode_index)
        else:
            episode_length = int(episode_row.get("length", 0))
            self._rollback_lerobot_data_file(info, episode_index, episode_row)
            self._rollback_lerobot_video_files(info, episode_index, episode_row, remaining_episodes)

        self._remove_lerobot_episode_metadata(episodes_by_file, episode_index)

        info["total_episodes"] = episode_index
        info["total_frames"] = max(0, int(info.get("total_frames", 0)) - episode_length)
        info["splits"] = {"train": f"0:{episode_index}"}
        if episode_index == 0:
            info["total_tasks"] = 0
            tasks_path = self.dataset_dir / "meta" / "tasks.parquet"
            if tasks_path.exists():
                tasks_path.unlink()
        info_path.write_text(json.dumps(info, indent=4) + "\n")

        stats_path = self.dataset_dir / "meta" / "stats.json"
        if stats_path.exists():
            stats_path.unlink()
            utils.logger.warning("Removed stale LeRobot stats.json; regenerate stats before training.")

        self.recorded_episodes = episode_index
        self._lerobot_episode_started = False
        self.collect_step = 0
        self._last_episode_length_cache_for = None
        utils.logger.info(f"Rolled back LeRobot episode {episode_index}.")
        return True

    def _has_lerobot_pending_frames(self) -> bool:
        if self.lerobot_dataset is not None and hasattr(
            self.lerobot_dataset, "has_pending_frames"
        ):
            return bool(self.lerobot_dataset.has_pending_frames())
        return bool(self.collect_step and self._lerobot_episode_started)

    def _load_lerobot_episode_metadata(self) -> list[tuple[Path, object]]:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("pandas is required to rollback LeRobot metadata") from exc

        episodes_root = self.dataset_dir / "meta" / "episodes"
        if not episodes_root.exists():
            return []

        result = []
        for path in sorted(episodes_root.rglob("*.parquet")):
            df = pd.read_parquet(path)
            result.append((path, df))
        return result

    def _remove_lerobot_episode_metadata(
        self, episodes_by_file: list[tuple[Path, object]], episode_index: int
    ) -> None:
        for path, df in episodes_by_file:
            if "episode_index" not in df:
                continue
            kept = df[df["episode_index"] != episode_index]
            if kept.empty:
                path.unlink()
                self._remove_empty_parents(path.parent, self.dataset_dir / "meta" / "episodes")
            elif len(kept) != len(df):
                self._delete_or_rewrite_episode_parquet(
                    path,
                    episode_index,
                    empty_parent_stop=self.dataset_dir / "meta" / "episodes",
                )

    def _rollback_lerobot_data_file(self, info: dict, episode_index: int, episode_row) -> None:
        data_path = self.dataset_dir / info["data_path"].format(
            chunk_index=int(episode_row["data/chunk_index"]),
            file_index=int(episode_row["data/file_index"]),
        )
        self._delete_or_rewrite_episode_parquet(data_path, episode_index)

    def _rollback_lerobot_video_files(
        self,
        info: dict,
        episode_index: int,
        episode_row,
        remaining_episodes: list,
    ) -> None:
        for key, feature in info.get("features", {}).items():
            if feature.get("dtype") != "video":
                continue
            chunk_key = f"videos/{key}/chunk_index"
            file_key = f"videos/{key}/file_index"
            if chunk_key not in episode_row or file_key not in episode_row:
                continue
            chunk_index = int(episode_row[chunk_key])
            file_index = int(episode_row[file_key])
            still_referenced = any(
                chunk_key in df
                and file_key in df
                and not df[(df[chunk_key] == chunk_index) & (df[file_key] == file_index)].empty
                for df in remaining_episodes
            )
            video_path = self.dataset_dir / info["video_path"].format(
                video_key=key,
                chunk_index=chunk_index,
                file_index=file_index,
            )
            if still_referenced:
                utils.logger.warning(
                    f"Keeping shared video file during rollback: {video_path}"
                )
                continue
            if video_path.exists():
                video_path.unlink()
                self._remove_empty_parents(video_path.parent, self.dataset_dir / "videos" / key)

    def _rollback_lerobot_files_by_index(self, info: dict, episode_index: int) -> int:
        chunks_size = int(info.get("chunks_size", 1000))
        chunk_index = episode_index // chunks_size
        file_index = episode_index % chunks_size
        data_path = self.dataset_dir / info["data_path"].format(
            chunk_index=chunk_index,
            file_index=file_index,
        )
        episode_length = 0
        if data_path.exists():
            episode_length = self._delete_or_rewrite_episode_parquet(data_path, episode_index)
        for key, feature in info.get("features", {}).items():
            if feature.get("dtype") != "video":
                continue
            video_path = self.dataset_dir / info["video_path"].format(
                video_key=key,
                chunk_index=chunk_index,
                file_index=file_index,
            )
            if video_path.exists():
                video_path.unlink()
        return episode_length

    def _delete_or_rewrite_episode_parquet(
        self,
        path: Path,
        episode_index: int,
        empty_parent_stop: Path | None = None,
    ) -> int:
        try:
            import pyarrow as pa
            import pyarrow.compute as pc
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("pyarrow is required to rollback LeRobot data") from exc

        if not path.exists():
            utils.logger.warning(f"LeRobot data file already missing: {path}")
            return 0

        parent_stop = empty_parent_stop or (self.dataset_dir / "data")
        table = pq.read_table(path)
        if "episode_index" not in table.column_names:
            path.unlink()
            self._remove_empty_parents(path.parent, parent_stop)
            return 0

        episode_column = table["episode_index"]
        remove_mask = pc.equal(
            episode_column,
            pa.scalar(episode_index, type=episode_column.type),
        )
        removed = int(pc.sum(pc.cast(remove_mask, pa.int64())).as_py() or 0)
        if removed == 0:
            utils.logger.warning(f"No rows for episode {episode_index} in {path}")
            return 0

        kept = table.filter(pc.invert(remove_mask))
        if kept.num_rows == 0:
            path.unlink()
            self._remove_empty_parents(path.parent, parent_stop)
        else:
            pq.write_table(kept, path)
        return removed

    @staticmethod
    def _remove_empty_parents(path: Path, stop: Path) -> None:
        stop = stop.resolve()
        path = path.resolve()
        while path != stop and stop in path.parents:
            try:
                path.rmdir()
            except OSError:
                break
            path = path.parent

    # ── Finalize ──────────────────────────────────────────────────────────

    def close(self) -> None:
        if self.dataset_type == "l" and self.lerobot_dataset is not None:
            self.lerobot_dataset.finalize()
            if self.push_to_hub:
                self.lerobot_dataset.push_to_hub()
