import os
import numpy as np
import cv2
import h5py
import argparse
import matplotlib.pyplot as plt
import IPython
import logging


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Prevent errors, comment out if IPython is not needed
try:
    e = IPython.embed
except:
    pass

JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
STATE_NAMES = JOINT_NAMES + ["gripper"]


def load_hdf5(dataset_dir, dataset_name):
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        logger.error(f"Dataset does not exist at {dataset_path}")
        exit()

    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs['sim']
        qpos = root['/observations/qpos'][()]
        action = root['/action'][()]
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]

        # --- Read tactile data ---
        tactile = None
        if 'tactile' in root['/observations']:
            tactile = root['/observations/tactile'][()]  # (T, 41, 3)
        # ------------------

    return qpos, action, image_dict, tactile


def render_tactile_frame(tactile_frame, height=480, width=320):
    """
    Render (41, 3) tactile vector data into an arrow map
    Layout logic:
    - 0-31: 4x8 matrix (middle)
    - 32-34: 3 points on the right
    - 35-37: 3 points on the left
    - 38-40: 3 points at the front

    Visualization logic:
    - Arrow direction: determined by (fx, fy)
    - Arrow/dot color: determined by fz (pressure)
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # --- Parameter adjustment ---
    # arrow_scale: scale up force value to show in pixels. If force is small (around 0.1), set higher (e.g. 50)
    arrow_scale = 0.01
    # color_scale: used to normalize Z-axis pressure to display colors
    max_z_force = 1

    # Layout parameters
    cx, cy = width // 2, height // 2
    spacing = 35  # Increase spacing to prevent overlapping arrows

    # Helper function to define position
    def get_sensor_pos(idx):
        # 1. Middle 4x8 matrix (0-31)
        if idx < 32:
            r = idx // 8  # Row (0-3)
            c = idx % 8  # Column (0-7)
            # Center calculation
            start_x = cx - (4 * spacing) // 2 + spacing // 2
            start_y = cy - (8 * spacing) // 2 + spacing // 2
            return start_x + r * spacing, start_y + c * spacing

        # 2. 3 points on the right (32-34)
        elif 32 <= idx <= 34:
            i = idx - 32
            start_x = cx + (4 * spacing) // 2 + spacing  # Right of matrix
            start_y = cy - (3 * spacing) // 2 + spacing // 2
            return start_x, start_y + i * spacing

        # 3. 3 points on the left (35-37)
        elif 35 <= idx <= 37:
            i = idx - 35
            start_x = cx - (4 * spacing) // 2 - spacing  # Left of matrix
            start_y = cy - (3 * spacing) // 2 + spacing // 2
            return start_x, start_y + i * spacing

        # 4. 3 points at the front (38-40)
        elif 38 <= idx <= 40:
            i = idx - 38
            start_x = cx - (3 * spacing) // 2 + spacing // 2
            start_y = cy - (8 * spacing) // 2 - spacing  # Top of matrix
            return start_x + i * spacing, start_y
        return 0, 0

    # Generate color lookup table (JET: Blue=low pressure, Red=high pressure)
    lut = np.zeros((256, 1, 3), dtype=np.uint8)

    # 2. Set gradient logic
    # e.g.: Map to [Red] gradient (R channel in BGR)
    # B and G channels remain 0, R channel changes from 0 to 255
    lut[:, 0, 2] = np.arange(256)
    gradient_gray = np.arange(256, dtype=np.uint8).reshape(256, 1)

    # ==========================================
    # Core modification: First convert grayscale data to BGR (3 channels)
    # ==========================================
    gradient_bgr = cv2.cvtColor(gradient_gray, cv2.COLOR_GRAY2BGR)

    # 3. Now input is 3 channels, LUT is also 3 channels, matching successful
    colormap = cv2.LUT(gradient_bgr, lut)
    for i in range(41):
        # Get force components
        fx, fy, fz = tactile_frame[i]

        # Get base position of sensor on the map
        px, py = get_sensor_pos(i)
        px, py = int(px), int(py)

        # 1. Calculate color (based on Z-axis pressure magnitude)
        # Use abs(fz) because pressure can be negative, adjust according to specific sensor definition
        norm_z = np.clip(np.abs(fz) / max_z_force, 0, 1) * 255
        color = colormap[int(norm_z)][0].tolist()  # BGR

        # 2. Draw base point (sensor position)
        cv2.circle(canvas, (px, py), 5, (100, 100, 100), 2, cv2.LINE_AA)

        # 3. Draw arrow (based on XY components)
        # If force is too small (noise), don't draw arrow
        if abs(fx) > 0.1 or abs(fy) > 0.1:
            end_x = int(px + fx * arrow_scale)
            end_y = int(py + fy * arrow_scale)

            # Restrict arrow from drawing outside the screen
            end_x = np.clip(end_x, 0, width)
            end_y = np.clip(end_y, 0, height)

            # Draw arrow: tipLength controls the size of the arrowhead
            cv2.arrowedLine(canvas, (px, py), (end_x, end_y), (255, 255, 255), 2, tipLength=0.3)
        else:
            # If no lateral force, just draw an outer circle representing pressure
            if abs(fz) > 0.5:
                cv2.circle(canvas, (px, py), int(abs(fz) * 1), color, 1)

    # Mark orientation
    cv2.putText(canvas, "Front", (cx - 20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return canvas


def save_videos(video, tactile, dt, video_path=None):
    if isinstance(video, list):
        cam_names = list(video[0].keys())
        h, w, _ = video[0][cam_names[0]].shape

        # Tactile image width
        tactile_w = 320 if tactile is not None else 0
        total_w = w * len(cam_names) + tactile_w

        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (total_w, h))

        for ts, image_dict in enumerate(video):
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name]
                image = image[:, :, [2, 1, 0]]  # RGB to BGR
                images.append(image)

            frame_img = np.concatenate(images, axis=1)

            # Concatenate tactile
            if tactile is not None:
                curr_tactile = tactile[ts]
                tactile_img = render_tactile_frame(curr_tactile, height=h, width=tactile_w)
                frame_img = np.concatenate([frame_img, tactile_img], axis=1)

            out.write(frame_img)
        out.release()
        logger.info(f"Saved video to: {video_path}")

    elif isinstance(video, dict):
        cam_names = list(video.keys())
        all_cam_videos = []
        for cam_name in cam_names:
            all_cam_videos.append(video[cam_name])
        all_cam_videos = np.concatenate(all_cam_videos, axis=2)

        n_frames, h, w, _ = all_cam_videos.shape

        tactile_w = 320 if tactile is not None else 0
        total_w = w + tactile_w

        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (total_w, h))

        for t in range(n_frames):
            image = all_cam_videos[t]
            image = image[:, :, [2, 1, 0]]

            if tactile is not None:
                curr_tactile = tactile[t]
                tactile_img = render_tactile_frame(curr_tactile, height=h, width=tactile_w)
                image = np.concatenate([image, tactile_img], axis=1)

            out.write(image)
        out.release()
        logger.info(f"Saved video to: {video_path}")


def visualize_joints(qpos_list, command_list, plot_path=None, ylim=None, label_overwrite=None):
    if label_overwrite:
        label1, label2 = label_overwrite
    else:
        label1, label2 = 'State', 'Command'

    qpos = np.array(qpos_list)
    command = np.array(command_list)
    num_ts, num_dim = qpos.shape
    h, w = 2, num_dim

    # Dynamically adjust figsize to prevent it from being too large
    if num_dim > 8:
        w = 15  # Wider plot if there are many joints

    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))

    # Handle the case where axs is a single object
    if num_dim == 1:
        axs = [axs]

    all_names = [name + '' for name in STATE_NAMES] + [name + '_right' for name in STATE_NAMES]
    safe_names = all_names[:num_dim] if num_dim <= len(all_names) else [f"Joint {i}" for i in range(num_dim)]

    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(qpos[:, dim_idx], label=label1)
        ax.set_title(f'Joint {dim_idx}: {safe_names[dim_idx]}')
        ax.legend()

    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        if command.shape[1] > dim_idx:  # Ensure command dimension matches
            ax.plot(command[:, dim_idx], label=label2)
            ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            axs[dim_idx].set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    logger.info(f"Saved qpos plot to: {plot_path}")
    plt.close()


def main(episode_num, dataset_dir, record_rate, if_multi):
    if if_multi:
        for i in range(episode_num):
            dataset_name = f'episode_{i}'
            qpos, action, image_dict, tactile = load_hdf5(dataset_dir, dataset_name)
            save_videos(image_dict, tactile, 1 / record_rate,
                        video_path=os.path.join(dataset_dir, dataset_name + '_video.mp4'))
            visualize_joints(qpos, action, plot_path=os.path.join(dataset_dir, dataset_name + '_qpos.png'))
    else:
        dataset_name = f'episode_{episode_num}'
        qpos, action, image_dict, tactile = load_hdf5(dataset_dir, dataset_name)
        save_videos(image_dict, tactile, 1 / record_rate,
                    video_path=os.path.join(dataset_dir, dataset_name + '_video.mp4'))
        visualize_joints(qpos, action, plot_path=os.path.join(dataset_dir, dataset_name + '_qpos.png'))


if __name__ == '__main__':
    episode_num = 1
    dataset_dir = "/home/idlab504/PycharmProjects/airo-doffy/datasets/InsertPowerStrip_hdf5"

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)
        logger.warning(f"Created {dataset_dir}, but it is likely empty.")

    main(episode_num, dataset_dir, record_rate=10, if_multi=True)
