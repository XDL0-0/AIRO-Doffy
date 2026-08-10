import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from airo_robots.manipulators.hardware.ur_rtde import URrtde
import logging
from airo_robots.manipulators.hardware.realman import RealmanControl

from config import Config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
cfg = Config()
if cfg.ROBOT_TYPE == "realman":
    robot = RealmanControl("192.168.1.18")
    robot.start_freedrive()
    input("press enter to continue")
    robot.stop_freedrive()
else:
    robot = URrtde("10.42.0.162", URrtde.UR3E_CONFIG)
# robot = URrtde("localhost", URrtde.UR5E_CONFIG)
    robot.rtde_control.servoStop()
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
