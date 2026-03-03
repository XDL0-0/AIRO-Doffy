"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil
import cv2
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import h5py
import tyro
import glob
import os
import re
from pathlib import Path


def natural_key(string):
    # Split the string into numeric and non-numeric parts for natural sorting
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string)]

def load_act_dataset(act_dataset_path):

    with h5py.File(act_dataset_path, 'r') as f:
        action = f['action'][:]
        images_0 = f['observations/images/camera_0'][:]
        images_1 = f['observations/images/camera_1'][:]
        resized_image0 = []
        resized_image1 = []

        for i in range(images_0.shape[0]):
            img_resized = cv2.resize(images_0[i,:,:,:], (256,256), interpolation=cv2.INTER_LINEAR)
            resized_image0.append(img_resized)
        for i in range(images_1.shape[0]):
            img_resized = cv2.resize(images_1[i,:,:,:], (256,256), interpolation=cv2.INTER_LINEAR)
            resized_image1.append(img_resized)

        qpos = f['observations/qpos'][:]

        n_steps = len(action)
        assert len(images_0) == n_steps and len(qpos) == n_steps

        return {
            'action': action,
            'images_0': resized_image0,
            'images_1': resized_image1,
            'qpos': qpos,
            'n_steps': n_steps,
        }

REPO_NAME = "/home/idlab504/VR_TELEOP/b2b_nofinecontrol"  # Name of the output dataset, also used for the Hugging Face Hub
# RAW_DATASET_NAMES = 'pick_cube'  # For simplicity we will combine multiple Libero datasets into one training dataset


def main(data_dir: str, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = Path(REPO_NAME)
    # if output_path.exists():
    #     shutil.rmtree(output_path)
    hdf5_files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))
    if not hdf5_files:
        raise ValueError(f"No HDF5 files found in {data_dir}")

    print(f"Found {len(hdf5_files)} HDF5 files")

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    if not output_path.exists():
        dataset = LeRobotDataset.create(
            repo_id=REPO_NAME,
            robot_type="ur3e",
            fps=10,
            features={
                "observation.image_0": {
                    "dtype": "image",
                    "shape": (256, 256, 3),
                    # "shape": (480, 640, 3),
                    "names": ["height", "width", "channel"],
                },
                "observation.image_1": {
                    "dtype": "image",
                    "shape": (256, 256, 3),
                    # "shape": (480, 640, 3),
                    "names": ["height", "width", "channel"],
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": (7,),
                    "names": ["state"],
                },
                "action": {
                    "dtype": "float32",
                    "shape": (7,),
                    "names": ["action"],
                },
            },
            image_writer_threads=10,
            image_writer_processes=5,
        )
    else:
        dataset = LeRobotDataset(repo_id=REPO_NAME)


    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    # for raw_dataset_name in RAW_DATASET_NAMES:
    #     raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
    #     for episode in raw_dataset:
    #         for step in episode["steps"].as_numpy_iterator():
    #             dataset.add_frame(
    #                 {
    #                     "image": step["observation"]["image"],
    #                     "wrist_image": step["observation"]["wrist_image"],
    #                     "state": step["observation"]["state"],
    #                     "actions": step["action"],
    #                 }
    #             )
    #         dataset.save_episode(task=step["language_instruction"].decode())

    hdf5_files.sort(key=lambda x: natural_key(os.path.basename(x)))

    # Then iterate and process
    for file_idx, hdf5_file in enumerate(hdf5_files):
        print(f"\\nProcessing file {file_idx + 1}/{len(hdf5_files)}: {os.path.basename(hdf5_file)}")
        data = load_act_dataset(hdf5_file)
        print(f"Including {data['n_steps']} steps")
        for step_idx in range(data['n_steps']):
            dataset.add_frame(
                {
                    "observation.image_0": data['images_0'][step_idx],
                    "observation.image_1": data['images_1'][step_idx],
                    "observation.state": data['qpos'][step_idx],
                    "action": data['action'][step_idx],
                },
            )
        dataset.save_episode()

    # # Consolidate the dataset, skip computing stats since we will do that later
    # dataset.consolidate(run_compute_stats=False)
    #
    # # Optionally push to the Hugging Face Hub
    # if push_to_hub:
    #     dataset.push_to_hub(
    #         tags=["pick", "hdf5"],
    #         private=False,
    #         push_videos=True,
    #         license="apache-2.0",
    #     )


if __name__ == "__main__":
    main('./test')
