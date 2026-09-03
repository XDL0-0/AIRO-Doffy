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

    def tick_params(self, **_kwargs):
        pass


class CapturingArtist:
    def __init__(self):
        self.data = None

    def set_data(self, value):
        self.data = value


class CapturingCanvas:
    def draw_idle(self):
        pass


class CapturingColorbar:
    def __init__(self):
        self.ax = CapturingAxis()

    def set_label(self, *_args, **_kwargs):
        pass

    def set_ticks(self, *_args, **_kwargs):
        pass


class CapturingFigure:
    def __init__(self):
        self.canvas = CapturingCanvas()

    def suptitle(self, *_args, **_kwargs):
        pass

    def colorbar(self, *_args, **_kwargs):
        return CapturingColorbar()

    def text(self, *_args, **_kwargs):
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
        distance[0, 1, 0] = 0.0
        status[0, 1, 0] = 5
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

        self.assertIn("min 0.0 mm", dashboard.beaver_axes[0].title)
        self.assertIn("avg 150.0 mm", dashboard.beaver_axes[0].title)
        self.assertIn("avg --", dashboard.beaver_axes[1].title)
        self.assertEqual(
            int(np.ma.count(dashboard.beaver_artists[0].data)),
            3,
        )

    def test_beaver_colormap_blue_ramp_within_400mm(self):
        dashboard = TeleopDashboard.__new__(TeleopDashboard)
        dashboard.beaver_layout = tuple((0, slot) for slot in range(9))
        dashboard.beaver_max_mm = 400.0
        dashboard.beaver_axes = []
        dashboard.beaver_artists = []
        dashboard._build_beaver_figure()
        try:
            for artist in dashboard.beaver_artists:
                cmap = artist.get_cmap()
                # Invalid pixels keep the masked slate grey.
                np.testing.assert_allclose(cmap.get_bad(), to_rgba("#343c49"))
                # Beyond the 400 mm focus zone everything is uniform grey.
                np.testing.assert_allclose(cmap.get_over(), to_rgba("#6e6e6e"))
                np.testing.assert_allclose(
                    cmap(artist.norm(401.0)), to_rgba("#6e6e6e")
                )
                # A valid zero reading (contact) is flagged red via the
                # under-range slot; the wire encodes positive distances in
                # 10 mm increments, so nothing valid falls between 0 and 5.
                self.assertEqual(artist.norm.vmin, 5.0)
                self.assertEqual(artist.norm.vmax, 400.0)
                np.testing.assert_allclose(cmap.get_under(), to_rgba("#ff1a1a"))
                np.testing.assert_allclose(
                    cmap(artist.norm(0.0)), to_rgba("#ff1a1a")
                )
                # 10-400 mm: 40 ten-mm bins, every one blue (blue channel
                # dominates red).
                self.assertEqual(len(cmap.colors), 40)
                for color in cmap.colors:
                    self.assertGreater(color[2], color[0])
                # Near is light, far is dark.
                self.assertGreater(
                    np.sum(cmap(artist.norm(10.0))),
                    np.sum(cmap(artist.norm(400.0))),
                )
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
