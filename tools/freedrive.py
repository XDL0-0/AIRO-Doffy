from airo_robots.manipulators import URrtde
robot = URrtde("10.42.0.162", URrtde.UR5E_CONFIG)
# robot = URrtde("localhost", URrtde.UR5E_CONFIG)
robot.rtde_control.teachMode()  # start freedrive
input("press enter to continue")
robot.rtde_control.endTeachMode()  # stop
# freedrive
# input("press enter to continue")

print(robot.get_joint_configuration())
print(robot.get_tcp_pose())
# robot.move_to_joint_configuration([-1.57, -1.57, -1.57, 0, 1.57, 0],0.3).wait()
# robot.rtde_control.teachMode()  # start freedrive
# input("press enter to continue")
# robot.rtde_control.endTeachMode()  # stop freedrive