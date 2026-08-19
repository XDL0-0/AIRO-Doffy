import unittest

import numpy as np

from visualizer import TeleopDashboard, TeleopSample


class CapturingText:
    def __init__(self):
        self.value = ""

    def set_text(self, value):
        self.value = str(value)


class CapturingAxis:
    def __init__(self):
        self.title = ""

    def set_title(self, value, **_kwargs):
        self.title = str(value)


class CapturingArtist:
    def __init__(self):
        self.data = None

    def set_data(self, value):
        self.data = value


class CapturingCanvas:
    def draw_idle(self):
        pass


class CapturingFigure:
    def __init__(self):
        self.canvas = CapturingCanvas()

    def suptitle(self, *_args, **_kwargs):
        pass


class WrmVisualizerTest(unittest.TestCase):
    def test_robot_state_panel_shows_alpha_confidence_and_target(self):
        dashboard = TeleopDashboard.__new__(TeleopDashboard)
        dashboard.pose_text = CapturingText()
        sample = TeleopSample(
            timestamp=0.0,
            wrench=np.zeros(6),
            joints=np.zeros(7),
            tcp_translation=np.array([0.4, 0.0, 0.3]),
            wrm={
                "elbow_alpha": 0.375,
                "confidence": 0.82,
                "arm_angle_target_deg": 14.25,
                "tcp_z_offset_m": -0.025,
                "tracking_frozen": False,
            },
        )

        dashboard._update_pose(sample)

        self.assertIn("WRM alpha 0.375", dashboard.pose_text.value)
        self.assertIn("conf 0.82  TRACK", dashboard.pose_text.value)
        self.assertIn("arm-angle target +14.2 deg", dashboard.pose_text.value)
        self.assertIn("WRM TCP z offset -2.5 cm", dashboard.pose_text.value)

    def test_robot_state_panel_marks_frozen_tracking(self):
        dashboard = TeleopDashboard.__new__(TeleopDashboard)
        dashboard.pose_text = CapturingText()
        sample = TeleopSample(
            timestamp=0.0,
            wrench=np.zeros(6),
            wrm={
                "elbow_alpha": 0.9,
                "confidence": 0.1,
                "arm_angle_target_deg": 20.0,
                "tracking_frozen": True,
            },
        )

        dashboard._update_pose(sample)

        self.assertIn("WRM alpha 0.900", dashboard.pose_text.value)
        self.assertIn("FROZEN", dashboard.pose_text.value)

    def test_beaver_titles_show_average_of_valid_distance_cells(self):
        dashboard = TeleopDashboard.__new__(TeleopDashboard)
        dashboard.beaver_enabled = True
        dashboard.beaver_layout = tuple((0, slot) for slot in range(9))
        dashboard.beaver_axes = [CapturingAxis() for _ in range(9)]
        dashboard.beaver_artists = [CapturingArtist() for _ in range(9)]
        dashboard.beaver_fig = CapturingFigure()

        distance = np.zeros((9, 4, 4), dtype=float)
        status = np.zeros((9, 4, 4), dtype=np.uint8)
        distance[0, 0, :4] = [100.0, 200.0, 300.0, np.nan]
        status[0, 0, :4] = [5, 9, 5, 5]
        distance[1, 0, 0] = 999.0
        status[1, 0, 0] = 1
        sample = TeleopSample(
            timestamp=0.0,
            wrench=np.zeros(6),
            beaver={
                "distance_mm": distance,
                "target_status": status,
                "present": np.ones(9, dtype=bool),
                "connected": True,
                "stale": False,
            },
        )

        dashboard._update_beaver(sample)

        self.assertIn("avg 200.0 mm", dashboard.beaver_axes[0].title)
        self.assertIn("avg --", dashboard.beaver_axes[1].title)
        self.assertEqual(
            int(np.ma.count(dashboard.beaver_artists[0].data)),
            3,
        )


if __name__ == "__main__":
    unittest.main()
