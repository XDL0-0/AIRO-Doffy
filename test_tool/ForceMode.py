from airo_robots.manipulators import URrtde
from airo_spatial_algebra.se3 import SE3Container
import numpy as np
import logging

from config import Config


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

cfg = Config()
robot = URrtde("10.42.0.162", URrtde.UR3E_CONFIG)
robot.move_to_joint_configuration(cfg.INITIAL_JOINT,0.3).wait()
logger.info("Initialization complete.")
# robot = URrtde("localhost", URrtde.UR5E_CONFIG)
task_frame = robot.get_tcp_pose()   # [x,y,z, rx,ry,rz] expressed in base coordinate system, meters & radians
SE = SE3Container.from_homogeneous_matrix(task_frame)
task_frame = np.concatenate([SE.translation,SE.orientation_as_euler_angles])
logger.info(f"Task frame: {task_frame}")
logger.info(f"TCP force: {robot.get_tcp_force()}")
selection_vector = [1, 1, 0, 0, 0, 0]    # [Fx,Fy,Fz, Tx,Ty,Tz] corresponds to [x,y,z, Rx,Ry,Rz]
wrench = [0, 0, 8.0, 0, 0, 0]           # About 8N downward force
type_ = 2
limits = [0.03, 0.03, 0.015,   0.15, 0.15, 0.15]
robot.rtde_control.forceMode(task_frame, selection_vector, robot.get_tcp_force(), type_, limits) # start freedrive
input("press enter to stop forcemode")
robot.rtde_control.forceModeStop()  # stop freedrive
# input("press enter to continue")
# robot.move_to_joint_configuration([-1.57, -1.57, -1.57, 0, 1.57, 0],0.3).wait()
# robot.rtde_control.teachMode()  # start freedrive
# input("press enter to continue")
# robot.rtde_control.endTeachMode()  # stop freedrive
