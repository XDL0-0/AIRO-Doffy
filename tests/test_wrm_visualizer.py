import unittest

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

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

    def test_beaver_zero_distance_reserves_magenta_under_colour(self):
        dashboard = TeleopDashboard.__new__(TeleopDashboard)
        dashboard.beaver_layout = tuple((0, slot) for slot in range(9))
        dashboard.beaver_max_mm = 2500.0
        dashboard.beaver_axes = []
        dashboard.beaver_artists = []
        dashboard._build_beaver_figure()
        try:
            for artist in dashboard.beaver_artists:
                cmap = artist.get_cmap()
                # Invalid pixels keep the original masked grey.
                np.testing.assert_allclose(cmap.get_bad(), to_rgba("#343c49"))
                # The wire encodes positive distances in 10 mm increments,
                # so every valid positive reading stays in the normal range
                # and only a zero can fall into the under-range slot.
                self.assertGreater(artist.norm.vmin, 0.0)
                self.assertLessEqual(artist.norm.vmin, 10.0)
                # The reserved under-range colour is conspicuous magenta.
                np.testing.assert_allclose(
                    cmap.get_under(), (1.0, 0.0, 1.0, 1.0)
                )
                np.testing.assert_allclose(
                    cmap(artist.norm(0.0)), (1.0, 0.0, 1.0, 1.0)
                )
                # A 10 mm reading stays on the turbo_r scale.
                self.assertGreaterEqual(artist.norm(10.0), 0.0)
                self.assertLessEqual(artist.norm(10.0), 1.0)
        finally:
            plt.close(dashboard.beaver_fig)

    def test_beaver_update_keeps_valid_zero_visible_and_invalid_zero_masked(self):
        dashboard = TeleopDashboard.__new__(TeleopDashboard)
        dashboard.beaver_enabled = True
        dashboard.beaver_layout = tuple((0, slot) for slot in range(9))
        dashboard.beaver_max_mm = 2500.0
        dashboard.beaver_axes = []
        dashboard.beaver_artists = []
        dashboard._build_beaver_figure()
        try:
            distance = np.zeros((9, 4, 4), dtype=float)
            status = np.zeros((9, 4, 4), dtype=np.uint8)
            distance[0, 0, 0] = 0.0
            status[0, 0, 0] = 5
            distance[0, 0, 1] = 0.0
            status[0, 0, 1] = 9
            distance[0, 0, 2] = 0.0
            status[0, 0, 2] = 1
            distance[0, 0, 3] = 0.0
            distance[1, 0, 0] = -10.0
            status[1, 0, 0] = 5
            distance[1, 0, 1] = np.nan
            status[1, 0, 1] = 9
            distance[1, 0, 2] = 10.0
            status[1, 0, 2] = 9
            distance[2, 0, 0] = 0.0
            status[2, 0, 0] = 5
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

            # Valid zeros (status 5/9) stay visible, invalid zeros stay masked.
            mask0 = np.ma.getmaskarray(dashboard.beaver_artists[0].get_array())
            self.assertEqual(mask0[0].tolist(), [False, False, True, True])
            self.assertEqual(float(dashboard.beaver_artists[0].get_array()[0, 0]), 0.0)
            # Negative and non-finite readings stay masked even with status 5/9.
            mask1 = np.ma.getmaskarray(dashboard.beaver_artists[1].get_array())
            self.assertEqual(mask1[0].tolist(), [True, True, False, True])
            mask2 = np.ma.getmaskarray(dashboard.beaver_artists[2].get_array())
            self.assertEqual(mask2[0].tolist(), [False, True, True, True])
            # A valid zero is a real reading, so a sensor whose only valid
            # pixel is zero averages 0.0 mm.
            self.assertIn("avg 0.0 mm", dashboard.beaver_axes[2].get_title())
            self.assertIn("avg 10.0 mm", dashboard.beaver_axes[1].get_title())
        finally:
            plt.close(dashboard.beaver_fig)


if __name__ == "__main__":
    unittest.main()
