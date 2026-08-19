from __future__ import annotations

import os
import queue
import unittest

os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent

from visualizer import TeleopDashboard, TeleopSample


class VisualizerLayoutTests(unittest.TestCase):
    def make_dashboard(self, *, show_tactile_panel: bool) -> TeleopDashboard:
        return TeleopDashboard(
            queue.Queue(),
            queue.Queue(),
            camera_num=2,
            show_tactile_panel=show_tactile_panel,
        )

    def test_tactile_disabled_removes_panel_and_update_is_noop(self) -> None:
        dashboard = self.make_dashboard(show_tactile_panel=False)
        try:
            self.assertIsNone(dashboard.tactile_ax)
            self.assertIsNone(dashboard.tactile_panel)
            self.assertIsNone(dashboard.tactile_text)
            self.assertNotIn(
                "Tactile",
                {axis.get_title(loc="left") for axis in dashboard.fig.axes},
            )
            dashboard._update_tactile(
                TeleopSample(timestamp=0.0, wrench=[0.0] * 6)
            )
        finally:
            plt.close(dashboard.fig)

    def test_tactile_remains_enabled_by_default_option(self) -> None:
        dashboard = self.make_dashboard(show_tactile_panel=True)
        try:
            self.assertIsNotNone(dashboard.tactile_ax)
            self.assertIsNotNone(dashboard.tactile_panel)
            self.assertIn(
                "Tactile",
                {axis.get_title(loc="left") for axis in dashboard.fig.axes},
            )
        finally:
            plt.close(dashboard.fig)

    def test_teach_controls_follow_workflow_state_and_queue_commands(self) -> None:
        command_queue = queue.Queue()
        dashboard = TeleopDashboard(
            queue.Queue(),
            command_queue,
            show_record_button=False,
            show_rollback_button=True,
            show_teach_controls=True,
        )
        try:
            dashboard.latest = TeleopSample(
                timestamp=0.0,
                wrench=[0.0] * 6,
                dataset={
                    "dataset_type": "l",
                    "recorded_episodes": 1,
                    "current_episode_frames": 0,
                    "collect_rate_hz": 24.0,
                    "collecting": False,
                },
                teach={
                    "state": "idle",
                    "trajectory_frames": 0,
                    "message": (
                        "Trajectory is cleared, please press Teach to create a new one."
                    ),
                    "teach_enabled": True,
                    "reteach_enabled": False,
                    "replay_enabled": False,
                    "initial_pose_enabled": True,
                },
            )
            dashboard._update_dataset_status(dashboard.latest)
            dashboard._update_teach_status(dashboard.latest)
            self.assertIn("24 Hz", dashboard.dataset_text.get_text())
            self.assertEqual(dashboard.teach_button.label.get_text(), "Teach")
            self.assertLess(dashboard.replay_button_ax.get_alpha(), 1.0)
            self.assertTrue(dashboard.rollback_button_ax.get_visible())
            self.assertEqual(
                dashboard.workflow_message_text.get_text(),
                "Trajectory is cleared, please press Teach to create a new one.",
            )

            dashboard._request_teach_toggle(None)
            dashboard._request_replay_collect(None)
            self.assertEqual(command_queue.get_nowait()["command"], "toggle_teach")
            self.assertTrue(command_queue.empty())

            dashboard.latest.teach.update(
                state="ready",
                trajectory_frames=12,
                teach_enabled=False,
                reteach_enabled=True,
                replay_enabled=True,
            )
            dashboard._update_teach_status(dashboard.latest)
            dashboard._request_replay_collect(None)
            self.assertEqual(command_queue.get_nowait()["command"], "replay_collect")
            dashboard._request_rollback(None)
            self.assertEqual(
                command_queue.get_nowait()["command"], "rollback_last_episode"
            )
        finally:
            plt.close(dashboard.fig)

    def test_hidden_legacy_button_cannot_grab_teach_click(self) -> None:
        command_queue = queue.Queue()
        dashboard = TeleopDashboard(
            queue.Queue(),
            command_queue,
            show_record_button=False,
            show_rollback_button=False,
            show_teach_controls=True,
        )
        try:
            dashboard.latest = TeleopSample(
                timestamp=0.0,
                wrench=[0.0] * 6,
                teach={
                    "state": "idle",
                    "trajectory_frames": 0,
                    "teach_enabled": True,
                    "reteach_enabled": False,
                    "replay_enabled": False,
                    "initial_pose_enabled": True,
                },
            )
            dashboard.fig.canvas.draw()
            legacy = dashboard.record_button_ax.bbox
            teach = dashboard.teach_button_ax.bbox
            x0, x1 = max(legacy.x0, teach.x0), min(legacy.x1, teach.x1)
            y0, y1 = max(legacy.y0, teach.y0), min(legacy.y1, teach.y1)
            self.assertLess(x0, x1)
            self.assertLess(y0, y1)
            x, y = (x0 + x1) / 2.0, (y0 + y1) / 2.0

            dashboard.fig.canvas.callbacks.process(
                "button_press_event",
                MouseEvent(
                    "button_press_event", dashboard.fig.canvas, x, y, button=1
                ),
            )
            dashboard.fig.canvas.callbacks.process(
                "button_release_event",
                MouseEvent(
                    "button_release_event", dashboard.fig.canvas, x, y, button=1
                ),
            )

            self.assertEqual(command_queue.get_nowait()["command"], "toggle_teach")
            self.assertIsNone(dashboard.fig.canvas.mouse_grabber)
        finally:
            plt.close(dashboard.fig)

    def test_teach_controls_fit_compact_window(self) -> None:
        dashboard = TeleopDashboard(
            queue.Queue(),
            queue.Queue(),
            show_record_button=False,
            show_rollback_button=True,
            show_teach_controls=True,
        )
        try:
            dashboard.fig.set_size_inches(9, 6)
            dashboard.latest = TeleopSample(
                timestamp=0.0,
                wrench=[0.0] * 6,
                dataset={
                    "dataset_type": "l",
                    "recorded_episodes": 1,
                    "current_episode_frames": 0,
                    "collect_rate_hz": 24.0,
                    "collecting": False,
                },
                teach={
                    "state": "ready",
                    "trajectory_frames": 12,
                    "message": "Ready",
                    "teach_enabled": False,
                    "reteach_enabled": True,
                    "replay_enabled": True,
                    "initial_pose_enabled": True,
                },
            )
            dashboard._update_dataset_status(dashboard.latest)
            dashboard._update_teach_status(dashboard.latest)
            dashboard.fig.canvas.draw()
            renderer = dashboard.fig.canvas.get_renderer()

            self.assertGreater(
                dashboard.status_ax.bbox.width,
                dashboard.axes[0].bbox.width,
            )
            for button in (
                dashboard.teach_button,
                dashboard.reteach_button,
                dashboard.replay_button,
                dashboard.initial_pose_button,
                dashboard.rollback_button,
            ):
                button_bounds = button.ax.get_window_extent(renderer)
                label_bounds = button.label.get_window_extent(renderer)
                self.assertGreaterEqual(label_bounds.x0, button_bounds.x0)
                self.assertLessEqual(label_bounds.x1, button_bounds.x1)
                self.assertGreaterEqual(label_bounds.y0, button_bounds.y0)
                self.assertLessEqual(label_bounds.y1, button_bounds.y1)
        finally:
            plt.close(dashboard.fig)


if __name__ == "__main__":
    unittest.main()
