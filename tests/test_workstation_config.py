"""Connection setup only; robot hardware is never opened by these tests."""
import os
import unittest
from unittest.mock import patch

from doffy_teleop.config import Config


class WorkstationConfigTests(unittest.TestCase):
    def test_environment_is_read_for_each_config_instance(self):
        with patch.dict(os.environ, {"DOFFY_PC_IP": "192.0.2.101", "DOFFY_VR_IP": "192.0.2.102",
                                     "DOFFY_UR_IP": "192.0.2.103", "DOFFY_REALMAN_STATE_PUSH_IP": "192.0.2.104"}):
            first = Config()
            self.assertEqual((first.PC_IP, first.VR_IP, first.UR_IP, first.REALMAN_STATE_PUSH_IP),
                             ("192.0.2.101", "192.0.2.102", "192.0.2.103", "192.0.2.104"))
            os.environ["DOFFY_PC_IP"] = "192.0.2.105"
            self.assertEqual(Config().PC_IP, "192.0.2.105")
            self.assertEqual(first.PC_IP, "192.0.2.101")

    def test_explicit_arguments_override_environment_and_preserve_none(self):
        with patch.dict(os.environ, {"DOFFY_PC_IP": "192.0.2.101", "DOFFY_VR_IP": "192.0.2.102",
                                     "DOFFY_UR_IP": "192.0.2.103", "DOFFY_REALMAN_STATE_PUSH_IP": "192.0.2.104"}):
            cfg = Config(PC_IP="192.0.2.201", VR_IP="192.0.2.202", UR_IP="192.0.2.203",
                         REALMAN_STATE_PUSH_IP=None)
            self.assertEqual((cfg.PC_IP, cfg.VR_IP, cfg.UR_IP), ("192.0.2.201", "192.0.2.202", "192.0.2.203"))
            self.assertIsNone(cfg.REALMAN_STATE_PUSH_IP)
            self.assertEqual(cfg.REALMAN_STATE_PUSH_IP or cfg.PC_IP, "192.0.2.201")

    def test_ur_fallback_uses_configured_address(self):
        with patch.dict(os.environ, {"DOFFY_UR_IP": "192.0.2.103"}):
            for robot in ("ur3e", "ur5e"):
                with self.subTest(robot=robot):
                    self.assertEqual(Config(ROBOT_TYPE=robot, ROBOT_IP=None).ROBOT_IP, "192.0.2.103")
                    self.assertEqual(Config(ROBOT_TYPE=robot, ROBOT_IP="192.0.2.200").ROBOT_IP, "192.0.2.200")


if __name__ == "__main__":
    unittest.main()
