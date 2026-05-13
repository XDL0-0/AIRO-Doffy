import time
import numpy as np
import os
import h5py
import cv2
import logging
from airo_robots.manipulators.hardware.ur_rtde import URrtde
import pyrealsense2 as rs
from airo_robots.grippers import Robotiq2F85
from airo_camera_toolkit.cameras.realsense.realsense import Realsense
from airo_spatial_algebra.se3 import SE3Container
from dataset_tool.Visualize_hdf5_episodes import render_tactile_frame


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

data_type = "qpos"

def data_process( frame):
    frame = (frame * 255).astype(np.uint8)
    # resize image and lower image qualityy
    frame_resized = cv2.resize(frame, (640, 480))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb

ur = URrtde("10.42.0.162", URrtde.UR3E_CONFIG)
gripper = Robotiq2F85("10.42.0.162")
ur.gripper = gripper
gripper_delta_step_size = 0.01

dataset_dir = './test'
from_episode_idx = 3
to_episode_idx = 123
fps = 10

camera_test = input('Do u need check camera?(y/n)')
for i in range(from_episode_idx, to_episode_idx):
    dataset_path = os.path.join(dataset_dir, f'episode_{i}.hdf5')
    with h5py.File(dataset_path, 'r') as root:
        qpos = root['/observations/qpos'][()]
        # qvel = root['/observations/qvel'][()]
        action = root['/action'][()]
        # ── Load tactile data if present ──
        tactile = None
        if 'tactile' in root['/observations']:
            tactile = root['/observations/tactile'][()]  # (T, 41, 3)
            logger.info(f"Tactile data loaded: shape={tactile.shape}")
        if camera_test=='y':
            image_camera_0 = root[f'/observations/images/camera_0'][()]
            image_camera_1 = root[f'/observations/images/camera_1'][()]


    qpos_len = len(qpos)
    action_len = len(action)
    logger.info(f"qpos_len={qpos_len}, action_len={action_len}")
    episode_start_idx = 0

    # dataset_initial_image = dataset[episode_start_idx][dataset_image_key]
    dataset_initial_pose = qpos[episode_start_idx,:]


    if data_type == "qpos":
        ur.servo_to_joint_configuration(dataset_initial_pose[0:6],0.5)
        ur.gripper.move(dataset_initial_pose[6]).wait()
    elif data_type == "eef":
        tcp_target = SE3Container.from_euler_angles_and_translation(dataset_initial_pose[0:3],
                                                                    dataset_initial_pose[3:6])
        tcp_target_pose = tcp_target.homogeneous_matrix
        logger.info(f"Initial TCP target pose shape: {tcp_target_pose.shape}")
        # joint_solution = self.ik.inverse_kinematics_closest_with_tcp(tcp_target_pose, self.tcp_transform,
        #                                                              *self.ur.get_joint_configuration())
        ur.servo_to_tcp_pose(tcp_target_pose,0.5)
        ur.gripper.move(dataset_initial_pose[6]).wait()

    elif data_type == "tcp_quat":
        tcp_target = SE3Container.from_quaternion_and_translation(dataset_initial_pose[0:4],
                                                                    dataset_initial_pose[4:7])
        tcp_target_pose = tcp_target.homogeneous_matrix
        logger.info(f"Initial TCP target pose shape: {tcp_target_pose.shape}")
        # joint_solution = self.ik.inverse_kinematics_closest_with_tcp(tcp_target_pose, self.tcp_transform,
        #                                                              *self.ur.get_joint_configuration())
        ur.servo_to_tcp_pose(tcp_target_pose, 0.5)
        ur.gripper.move(dataset_initial_pose[7]).wait()

    # convert torch image to numpy image
    if camera_test=='y':
        context = rs.context()
        devices = context.query_devices()
        camera_num = len(devices)
        camera_series_num = []
        camera_list = {}
        if camera_num == 0:
            logger.warning("No RealSense connected.")
        else:
            for i, device in enumerate(devices):
                logger.info(f"camera {i}: {device.get_info(rs.camera_info.name)}")
                logger.info(f"serial number: {device.get_info(rs.camera_info.serial_number)}")
                camera_series_num.append(device.get_info(rs.camera_info.serial_number))
        for i in range(camera_num):
            camera_list[f'camera_{i}'] = Realsense(fps=30, resolution=Realsense.RESOLUTION_480, enable_depth=False,
                                                   enable_hole_filling=False, serial_number=camera_series_num[i])
        dataset_initial_image_0 = image_camera_0[0]
        dataset_initial_image_1 = image_camera_1[0]
        logger.debug(f"Initial dataset image 0 shape: {dataset_initial_image_0.shape}")
        # dataset_initial_image = dataset_initial_image.transpose(1, 2, 0).astype(np.uint8)

        while True:
            img_0 = data_process(camera_list[f'camera_0'].get_rgb_image())
            img_1 = data_process(camera_list[f'camera_1'].get_rgb_image())
            # blend the two images
            blended_image_0 = cv2.addWeighted(dataset_initial_image_0, 0.3, img_0, 0.5, 0)
            blended_image_1 = cv2.addWeighted(dataset_initial_image_1, 0.5, img_1, 0.5, 0)
            cv2.imshow("image-0", blended_image_0)
            cv2.imshow("image-1", blended_image_1)
            k = cv2.waitKey(1)
            logger.debug(f"OpenCV keypress: {k}")
            if k == ord('q'):
                break

    # input(f"Press Enter to start replay\n----episode{i}----")

    duration = 1.0 / fps
    for i in range(episode_start_idx, qpos_len):
        start_time = time.time()
        # dataset_pose = np.round(qpos[i, :],3)
        dataset_pose = qpos[i,:]

        # dataset_action = action[i,:]
        # ur.gripper.move(ur.gripper.get_current_width()+gripper_delta_step_size*dataset_action[6])
        if data_type == "qpos":
            ur.gripper.move(dataset_pose[6], 0.1)
            ur.servo_to_joint_configuration(dataset_pose[0:6], 1.0/fps)
        elif data_type == "eef":
            ur.gripper.move(dataset_pose[6], 0.1)
            tcp_target = SE3Container.from_euler_angles_and_translation(dataset_pose[0:3],
                                                                        dataset_pose[3:6])
            tcp_target_pose = tcp_target.homogeneous_matrix
            logger.debug(f"TCP target pose shape: {tcp_target_pose.shape}")
            ur.servo_to_tcp_pose(tcp_target_pose, 1.0/fps)
        elif data_type == "tcp_quat":
            logger.debug(f"Dataset pose: {dataset_pose}")
            ur.gripper.move(dataset_pose[7], 0.1)
            tcp_target = SE3Container.from_quaternion_and_translation(dataset_pose[0:4],
                                                                        dataset_pose[4:7])
            tcp_target_pose = tcp_target.homogeneous_matrix
            logger.debug(f"TCP target pose shape: {tcp_target_pose.shape}")
            ur.servo_to_tcp_pose(tcp_target_pose, 1.0/fps)

        # ── Tactile visualisation during replay ──
        if tactile is not None:
            tactile_img = render_tactile_frame(tactile[i])
            cv2.imshow("Tactile Replay", tactile_img)
            cv2.waitKey(1)

        interval = time.time() - start_time
        if interval<1.0/fps:
            time.sleep(1.0 / fps - interval)
        logger.info(f"Current interval: {1/(time.time() - start_time):5.2f}hz")

    # Close tactile window after episode
    if tactile is not None:
        cv2.destroyWindow("Tactile Replay")



logger.info(f"{i} episode replay finished")
