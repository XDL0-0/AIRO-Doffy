from airo_robots.manipulators import URrtde
import logging


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

robot = URrtde("10.42.0.162", URrtde.UR5E_CONFIG)
# robot = URrtde("localhost", URrtde.UR5E_CONFIG)
robot.rtde_control.teachMode()  # start freedrive
input("press enter to continue")
robot.rtde_control.endTeachMode()  # stop
# freedrive

# input("press enter to continue")

logger.info(f"Joint configuration: {robot.get_joint_configuration()}")
logger.info(f"TCP pose: {robot.get_tcp_pose()}")

# robot.move_to_joint_configuration([-1.57, -1.57, -1.57, 0, 1.57, 0],0.3).wait()
# robot.rtde_control.teachMode()  # start freedrive
# input("press enter to continue")
# robot.rtde_control.endTeachMode()  # stop freedrive
