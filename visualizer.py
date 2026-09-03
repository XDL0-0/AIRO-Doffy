"""Teleop dashboard fed by snapshots from main.py.

This module does not open RTDE, cameras, or tactile devices.  It only displays
the latest data pushed through a multiprocessing queue by the teleop process.
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
import queue
import signal
import time
from collections import deque
from dataclasses import dataclass

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-airo-doffy"

import matplotlib


def _configure_matplotlib_backend() -> None:
    requested = os.environ.get("MPLBACKEND", "")
    if requested and not requested.lower().startswith("qt"):
        return

    candidates = ["TkAgg"] if os.environ.get("DISPLAY") else []
    candidates.append("Agg")
    for backend in candidates:
        try:
            matplotlib.use(backend, force=True)
            return
        except Exception:
            continue


_configure_matplotlib_backend()

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.widgets import Button

from visualizer_config import VisualizerConfig

logger = logging.getLogger(__name__)

CHANNELS = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
UNITS = ("N", "N", "N", "Nm", "Nm", "Nm")
COLORS = ("#24d9ff", "#47f091", "#ff9b66", "#ff4db8", "#b063ff", "#ffd94d")
QUEUE_SIZE = 2


@dataclass
class TeleopSample:
    timestamp: float
    wrench: np.ndarray
    joints: np.ndarray | None = None
    tcp_translation: np.ndarray | None = None
    image: np.ndarray | None = None
    images: dict[str, np.ndarray] | None = None
    camera_count: int = 0
    tactile: np.ndarray | None = None
    tactile_timestamp_ns: int = 0
    dataset: dict | None = None
    teach: dict | None = None
    wrm: dict | None = None
    beaver: dict | None = None
    source_label: str = "teleop data"
    status_extra: str = ""
    connected: bool = True
    error: str = ""


class VisualizerHandle:
    def __init__(self, process: mp.Process, data_queue: mp.Queue, command_queue: mp.Queue):
        self.process = process
        self.data_queue = data_queue
        self.command_queue = command_queue

    def publish(self, sample: dict) -> None:
        while True:
            try:
                self.data_queue.put_nowait(sample)
                return
            except queue.Full:
                try:
                    self.data_queue.get_nowait()
                except queue.Empty:
                    return

    def drain_commands(self) -> list[dict]:
        commands = []
        while True:
            try:
                commands.append(self.command_queue.get_nowait())
            except queue.Empty:
                return commands

    def close(self) -> None:
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1.0)
        for q in (self.data_queue, self.command_queue):
            try:
                q.cancel_join_thread()
                q.close()
            except Exception:
                pass


def _beaver_blue_ramp(n_steps: int = 40) -> list[tuple[float, float, float]]:
    """Blue ramp for the Beaver distance focus zone.

    Each step covers 10 mm of the focus zone (10-400 mm by default: the
    nearest bin is light blue, the farthest dark blue). A zero-distance
    contact reading is drawn red via the under-range slot, not as part of
    this ramp.
    """
    near = np.array(matplotlib.colors.to_rgb("#d6eaff"))
    far = np.array(matplotlib.colors.to_rgb("#081f66"))
    return [
        tuple(near + (far - near) * idx / max(1, n_steps - 1))
        for idx in range(n_steps)
    ]


def _color_fader_rgb255(c1: str, c2: str, mix: float = 0.0) -> list[int]:
    mix = float(np.clip(mix, 0.0, 1.0))
    rgb1 = np.array(matplotlib.colors.to_rgb(c1))
    rgb2 = np.array(matplotlib.colors.to_rgb(c2))
    return [int(v * 255) for v in (1.0 - mix) * rgb1 + mix * rgb2]


class _FallbackMagTouchRawViewModel:
    """Same fallback formula used by test_tool/ForceVisualize.py."""

    def __init__(self, view, c1: str = "#434A52", c2: str = "#7DB5A8"):
        self.view = view
        self.grid_size = self.view.grid_size
        self.c1 = c1
        self.c2 = c2
        self.max_xy = 4000
        self.min_norm = 100
        self.max_norm = self.max_xy * np.sqrt(3)

    def data_to_formatted_rgb(self, data: np.ndarray) -> list[list[list[int]]]:
        return [
            [
                _color_fader_rgb255(
                    self.c1,
                    self.c2,
                    mix=min(1, np.linalg.norm(data[row][column]) / self.max_norm),
                )
                for column in range(self.grid_size[1])
            ]
            for row in range(self.grid_size[0])
        ]

    def update_view(self, data: np.ndarray) -> None:
        _data = data.copy()
        for row in range(self.view.grid_size[0]):
            for column in range(self.view.grid_size[1]):
                self.view.circle_radii[row][column] = (
                    np.sqrt(
                        max(abs(_data[row][column][2]) - self.min_norm, 0)
                        / self.max_norm
                    )
                    * self.view.radius_max
                    + self.view.radius_min
                )
                self.view.circle_offsets[row][column] = (
                    _data[row][column][0] / self.max_xy * self.view.offset_max,
                    _data[row][column][1] / self.max_xy * self.view.offset_max,
                )
        self.view.circle_colors = self.data_to_formatted_rgb(_data)
        self.view.update_view()


class MatplotlibBubbleView:
    """Matplotlib adapter matching ForceVisualize's MagTouch raw panel."""

    def __init__(self, ax, name: str = "MagTouchRaw0", grid_size: tuple[int, int] = (2, 2)):
        self.ax = ax
        self.grid_size = grid_size
        self.radius_min = 0.045
        self.radius_max = 0.13
        self.offset_max = 0.13
        self.circle_radii = [
            [self.radius_min for _ in range(self.grid_size[1])]
            for _ in range(self.grid_size[0])
        ]
        self.circle_offsets = [
            [(0.0, 0.0) for _ in range(self.grid_size[1])]
            for _ in range(self.grid_size[0])
        ]
        self.circle_colors = [
            [[0, 0, 0] for _ in range(self.grid_size[1])]
            for _ in range(self.grid_size[0])
        ]
        self.nominal_x_positions = [
            [0.34 + column * 0.32 for column in range(self.grid_size[1])]
            for _ in range(self.grid_size[0])
        ]
        self.nominal_y_positions = [
            [0.66 - row * 0.32 for _ in range(self.grid_size[1])]
            for row in range(self.grid_size[0])
        ]

        self.ax.set_xlim(0.08, 0.92)
        self.ax.set_ylim(0.08, 0.92)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_facecolor("#ffffff")
        self.label = self.ax.text(
            0.5,
            0.08,
            name,
            transform=self.ax.transAxes,
            ha="center",
            va="center",
            color="#000000",
            fontsize=10,
            fontweight="bold",
        )

        self.lines = []
        self.circles = []
        for row in range(self.grid_size[0]):
            line_row = []
            circle_row = []
            for column in range(self.grid_size[1]):
                line = self.ax.plot([], [], color="#D6AE72", linewidth=2.4)[0]
                circle = Circle(
                    (0.0, 0.0),
                    self.radius_min,
                    facecolor="#000000",
                    edgecolor="#000000",
                    linewidth=0.8,
                )
                self.ax.add_patch(circle)
                line_row.append(line)
                circle_row.append(circle)
            self.lines.append(line_row)
            self.circles.append(circle_row)
        self.redraw()

    def redraw(self) -> None:
        for row in range(self.grid_size[0]):
            for column in range(self.grid_size[1]):
                base_x = self.nominal_x_positions[row][column]
                base_y = self.nominal_y_positions[row][column]
                offset_x, offset_y = self.circle_offsets[row][column]
                x = base_x + offset_x
                y = base_y - offset_y
                radius = max(min(self.circle_radii[row][column], self.radius_max), self.radius_min)
                rgb = np.asarray(self.circle_colors[row][column], dtype=float) / 255.0
                color = tuple(np.clip(rgb, 0.0, 1.0))

                self.lines[row][column].set_data([base_x, x], [base_y, y])
                self.circles[row][column].center = (x, y)
                self.circles[row][column].radius = radius
                self.circles[row][column].set_facecolor(color)
                self.circles[row][column].set_edgecolor(color)

    def update_view(self) -> None:
        self.redraw()


class MagTouchRawDashboardPanel:
    def __init__(self, ax, topic_name: str = "MagTouchRaw0"):
        self.view = MatplotlibBubbleView(ax=ax, name=topic_name, grid_size=(2, 2))
        self.viewmodel = self._create_viewmodel()

    def _create_viewmodel(self):
        try:
            from sensor_comm_dds.visualisation.viewmodel.magtouch_raw_viewmodel import (
                MagTouchRawViewModel,
            )

            return MagTouchRawViewModel(view=self.view)
        except Exception:
            return _FallbackMagTouchRawViewModel(view=self.view)

    @staticmethod
    def _format_raw_visualiser_data(tactile_data: np.ndarray | None) -> np.ndarray | None:
        if tactile_data is None:
            return None

        tactile = np.asarray(tactile_data, dtype=np.float64)
        if tactile.shape == (2, 2, 3):
            return tactile
        if tactile.ndim != 2 or tactile.shape[0] < 4 or tactile.shape[1] < 3:
            return None

        data = np.zeros((2, 2, 3), dtype=np.float64)
        for i, taxel in enumerate(tactile[:4, :3]):
            mapped = np.array([-taxel[0], taxel[1], taxel[2]], dtype=np.float64)
            if i == 0:
                data[1, 0] = mapped
            elif i == 1:
                data[1, 1] = mapped
            elif i == 2:
                data[0, 1] = mapped
            elif i == 3:
                data[0, 0] = mapped
        return data

    def update(self, tactile_data: np.ndarray | None) -> None:
        data = self._format_raw_visualiser_data(tactile_data)
        if data is None:
            data = np.zeros((2, 2, 3), dtype=np.float64)
        self.viewmodel.update_view(data)


class TeleopDashboard:
    def __init__(
        self,
        data_queue: mp.Queue,
        command_queue: mp.Queue,
        window_s: float = 8.0,
        hz: float = 30.0,
        title: str = "Teleop Visualizer",
        force_panel_range: float = 5.0,
        camera_num: int = 1,
        show_rollback_button: bool = True,
        show_record_button: bool = False,
        show_teach_controls: bool = False,
        show_tactile_panel: bool = True,
        beaver_enabled: bool = False,
        beaver_layout: tuple[tuple[int, int], ...] | None = None,
        beaver_max_mm: float = 400.0,
    ) -> None:
        self.data_queue = data_queue
        self.command_queue = command_queue
        self.window_s = window_s
        self.force_panel_range = max(1e-6, float(force_panel_range))
        self.camera_count = max(0, int(camera_num))
        self.camera_slots = max(1, self.camera_count)
        self.show_rollback_button = show_rollback_button
        self.show_record_button = show_record_button
        self.show_teach_controls = show_teach_controls
        self.show_tactile_panel = bool(show_tactile_panel)
        self.beaver_enabled = bool(beaver_enabled)
        self.beaver_layout = tuple(beaver_layout or ())
        self.beaver_max_mm = max(1.0, float(beaver_max_mm))
        self.interval_ms = max(10, int(1000 / hz))
        self.latest = TeleopSample(timestamp=time.monotonic(), wrench=np.zeros(6))
        self.times: deque[float] = deque(maxlen=max(10, int(window_s * hz * 2)))
        self.values: deque[np.ndarray] = deque(maxlen=max(10, int(window_s * hz * 2)))
        self.t0: float | None = None

        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(15.5, 8.6), facecolor="#080b10")
        self.fig.canvas.manager.set_window_title(title)
        outer = self.fig.add_gridspec(
            1,
            2,
            width_ratios=[1.55, 1.0],
            wspace=0.08,
            left=0.035,
            right=0.985,
            top=0.9,
            bottom=0.07,
        )
        plot_grid = outer[0, 0].subgridspec(3, 2, hspace=0.28, wspace=0.18)
        camera_rows, camera_cols = self._camera_grid_shape(self.camera_slots)
        camera_height = max(0.8, 0.62 * camera_rows)
        if self.show_tactile_panel:
            side_grid = outer[0, 1].subgridspec(
                5,
                1,
                height_ratios=[1.35, 0.8, 0.95, camera_height, 0.62],
                hspace=0.16,
            )
            tactile_index = 2
            camera_index = 3
            pose_index = 4
        else:
            side_grid = outer[0, 1].subgridspec(
                4,
                1,
                height_ratios=[1.35, 0.8, camera_height, 0.62],
                hspace=0.16,
            )
            tactile_index = None
            camera_index = 2
            pose_index = 3

        self.axes = []
        self.lines = []
        self.value_texts = []
        for idx, name in enumerate(CHANNELS):
            ax = self.fig.add_subplot(plot_grid[idx // 2, idx % 2])
            self._style_plot_axis(ax, name, idx)
            (line,) = ax.plot([], [], color=COLORS[idx], linewidth=1.8)
            value_text = ax.text(
                0.02,
                0.88,
                "--",
                transform=ax.transAxes,
                color=COLORS[idx],
                fontsize=12,
                fontweight="bold",
            )
            self.axes.append(ax)
            self.lines.append(line)
            self.value_texts.append(value_text)

        self.status_ax = self.fig.add_subplot(side_grid[0])
        self.vector_ax = self.fig.add_subplot(side_grid[1])
        self.tactile_ax = (
            None
            if tactile_index is None
            else self.fig.add_subplot(side_grid[tactile_index])
        )
        self.pose_ax = self.fig.add_subplot(side_grid[pose_index])
        panel_axes = [
            (self.status_ax, "Status"),
            (self.vector_ax, "TCP Force Vector"),
            (self.pose_ax, "Robot State"),
        ]
        if self.tactile_ax is not None:
            panel_axes.insert(2, (self.tactile_ax, "Tactile"))
        for ax, name in panel_axes:
            self._style_panel_axis(ax, name)
        camera_grid = side_grid[camera_index].subgridspec(
            camera_rows,
            camera_cols,
            hspace=0.18,
            wspace=0.08,
        )
        self.camera_axes = []
        self.camera_artists = []
        self.camera_placeholders = []
        for idx in range(camera_rows * camera_cols):
            ax = self.fig.add_subplot(camera_grid[idx // camera_cols, idx % camera_cols])
            if idx < self.camera_slots:
                title_text = f"Camera {idx}" if self.camera_count else "Camera"
                self._style_panel_axis(ax, title_text)
                placeholder = ax.text(
                    0.5,
                    0.5,
                    "No camera",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="#7e8ca0",
                    fontsize=11,
                )
                self.camera_axes.append(ax)
                self.camera_artists.append(None)
                self.camera_placeholders.append(placeholder)
            else:
                ax.set_visible(False)

        self.status_text = self.status_ax.text(0.04, 0.84, "", transform=self.status_ax.transAxes, fontsize=11)
        self.dataset_text = self.status_ax.text(
            0.04,
            0.12,
            "",
            transform=self.status_ax.transAxes,
            va="bottom",
            family="monospace",
            fontsize=8.8,
            color="#9fb0c5",
        )
        self.force_mag_text = self.status_ax.text(
            0.04,
            0.58,
            "",
            transform=self.status_ax.transAxes,
            fontsize=15,
            fontweight="bold",
            color="#24d9ff",
        )
        self.torque_mag_text = self.status_ax.text(
            0.38,
            0.58,
            "",
            transform=self.status_ax.transAxes,
            fontsize=15,
            fontweight="bold",
            color="#ffd94d",
        )
        self.record_button_ax = self.status_ax.inset_axes([0.38, 0.12, 0.27, 0.24])
        self.record_button = Button(
            self.record_button_ax,
            "Start\nrecord",
            color="#234936",
            hovercolor="#326b4d",
        )
        self.record_button.label.set_color("#e8f1ff")
        self.record_button.label.set_fontsize(8)
        self.record_button.on_clicked(self._request_record_toggle)
        if not self.show_record_button:
            self.record_button_ax.set_visible(False)
            # Hidden Button axes still participate in Matplotlib hit-testing.
            # Disconnect them so they cannot grab a click intended for an
            # overlapping teach control.
            self.record_button.disconnect_events()

        rollback_bounds = (
            [0.67, 0.07, 0.31, 0.18]
            if self.show_teach_controls
            else [0.68, 0.12, 0.28, 0.24]
        )
        self.rollback_button_ax = self.status_ax.inset_axes(rollback_bounds)
        self.rollback_button = Button(
            self.rollback_button_ax,
            "Undo\nepisode",
            color="#263446",
            hovercolor="#3b4b62",
        )
        self.rollback_button.label.set_color("#e8f1ff")
        self.rollback_button.label.set_fontsize(8)
        self.rollback_button.on_clicked(self._request_rollback)
        if not self.show_rollback_button:
            self.rollback_button_ax.set_visible(False)
            self.rollback_button.disconnect_events()

        self.teach_button_ax = self.status_ax.inset_axes([0.38, 0.30, 0.18, 0.18])
        self.teach_button = Button(
            self.teach_button_ax,
            "Teach",
            color="#234936",
            hovercolor="#326b4d",
        )
        self.teach_button.on_clicked(self._request_teach_toggle)

        self.reteach_button_ax = self.status_ax.inset_axes([0.58, 0.30, 0.18, 0.18])
        self.reteach_button = Button(
            self.reteach_button_ax,
            "Reteach",
            color="#5a4321",
            hovercolor="#7a5d2e",
        )
        self.reteach_button.on_clicked(self._request_reteach)

        self.replay_button_ax = self.status_ax.inset_axes([0.78, 0.30, 0.20, 0.18])
        self.replay_button = Button(
            self.replay_button_ax,
            "Replay\ncollect",
            color="#24486a",
            hovercolor="#32648f",
        )
        self.replay_button.on_clicked(self._request_replay_collect)

        self.initial_pose_button_ax = self.status_ax.inset_axes(
            [0.38, 0.07, 0.27, 0.18]
        )
        self.initial_pose_button = Button(
            self.initial_pose_button_ax,
            "Initial\npose",
            color="#263446",
            hovercolor="#3b4b62",
        )
        self.initial_pose_button.on_clicked(self._request_initial_pose)
        self.teach_control_axes = (
            self.teach_button_ax,
            self.reteach_button_ax,
            self.replay_button_ax,
            self.initial_pose_button_ax,
        )
        self.teach_control_buttons = (
            self.teach_button,
            self.reteach_button,
            self.replay_button,
            self.initial_pose_button,
        )
        for button in self.teach_control_buttons:
            button.label.set_color("#e8f1ff")
            button.label.set_fontsize(8)
        for axis in self.teach_control_axes:
            axis.set_visible(self.show_teach_controls)
        if not self.show_teach_controls:
            for button in self.teach_control_buttons:
                button.disconnect_events()

        self.vector_ax.set_xlim(-self.force_panel_range, self.force_panel_range)
        self.vector_ax.set_ylim(-self.force_panel_range, self.force_panel_range)
        self.vector_ax.set_aspect("equal", adjustable="box")
        self.vector_ax.axhline(0, color="#263446", linewidth=1)
        self.vector_ax.axvline(0, color="#263446", linewidth=1)
        self.vector_arrow = self.vector_ax.arrow(0, 0, 0, 0, color="#24d9ff", width=0.015)
        self.vector_label = self.vector_ax.text(0.04, 0.9, "", transform=self.vector_ax.transAxes, fontsize=10)

        self.tactile_panel = None
        self.tactile_text = None
        if self.tactile_ax is not None:
            self.tactile_panel = MagTouchRawDashboardPanel(self.tactile_ax)
            self.tactile_text = self.tactile_ax.text(
                0.04,
                0.88,
                "waiting",
                transform=self.tactile_ax.transAxes,
                color="#9fb0c5",
                fontsize=10,
            )

        self.pose_text = self.pose_ax.text(
            0.04,
            0.8,
            "",
            transform=self.pose_ax.transAxes,
            va="top",
            family="monospace",
            fontsize=10,
        )

        self.fig.suptitle(
            title,
            x=0.035,
            ha="left",
            color="#e8f1ff",
            fontsize=16,
            fontweight="bold",
        )
        self.workflow_message_text = self.fig.text(
            0.035,
            0.025,
            "",
            color="#ffd94d",
            fontsize=10,
        )
        self.beaver_fig = None
        self.beaver_axes = []
        self.beaver_artists = []
        if self.beaver_enabled:
            self._build_beaver_figure()

    def _build_beaver_figure(self) -> None:
        self.beaver_fig, axes = plt.subplots(
            3,
            3,
            figsize=(10.5, 9.0),
            facecolor="#080b10",
            constrained_layout=True,
        )
        self.beaver_fig.canvas.manager.set_window_title("Beaver distance sensors")
        # Focus zone: 0-400 mm binned in 10 mm steps, light blue near to dark
        # blue far, so distance within reach is readable at a glance. A valid
        # zero reading (contact) is flagged red via the under-range slot; the
        # wire encodes positive distances in 10 mm increments, so no valid
        # positive reading can fall below the 5 mm vmin. Everything beyond
        # the focus zone is uniformly grey (out of range); masked invalid
        # cells keep the dark slate colour.
        cmap = matplotlib.colors.ListedColormap(_beaver_blue_ramp())
        cmap.set_bad("#343c49")
        cmap.set_under("#ff1a1a")
        cmap.set_over("#6e6e6e")
        layout = self.beaver_layout or tuple((0, idx) for idx in range(9))
        for slot, ax in enumerate(axes.flat):
            bus, sensor = layout[slot]
            ax.set_facecolor("#101722")
            ax.set_title(
                f"B{bus}S{sensor} · waiting",
                color="#ffd94d",
                fontsize=10,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            artist = ax.imshow(
                np.ma.masked_all((1, 1)),
                cmap=cmap,
                vmin=5.0,
                vmax=self.beaver_max_mm,
                interpolation="nearest",
            )
            self.beaver_axes.append(ax)
            self.beaver_artists.append(artist)
        # Reserve the bottom strip for the scheme legend text below.
        self.beaver_fig.get_layout_engine().set(rect=(0.02, 0.04, 0.98, 0.98))
        cbar = self.beaver_fig.colorbar(
            self.beaver_artists[0],
            ax=axes.ravel().tolist(),
            extend="both",
            pad=0.02,
        )
        cbar.set_label("distance (mm)", color="#e8f1ff", fontsize=10)
        cbar.ax.tick_params(colors="#9fb0c5", labelsize=8)
        cbar.set_ticks([10, 50, 100, 200, 300, 400])
        self.beaver_fig.text(
            0.5,
            0.015,
            "0 mm = red contact · 10-400 mm blue light→dark · >400 mm grey",
            ha="center",
            va="bottom",
            color="#9fb0c5",
            fontsize=9,
        )

    @staticmethod
    def _camera_grid_shape(camera_slots: int) -> tuple[int, int]:
        if camera_slots <= 2:
            return camera_slots, 1
        cols = 2
        rows = int(math.ceil(camera_slots / cols))
        return rows, cols

    @staticmethod
    def _style_plot_axis(ax, name: str, idx: int) -> None:
        ax.set_facecolor("#101722")
        for spine in ax.spines.values():
            spine.set_color("#425066")
        ax.grid(True, color="#263446", alpha=0.7, linewidth=0.8)
        ax.tick_params(colors="#9fb0c5", labelsize=8)
        ax.set_title(f"{name} ({UNITS[idx]})", loc="left", color="#e8f1ff", fontsize=10, pad=7)
        ax.set_xlabel("seconds", color="#9fb0c5", fontsize=8)
        ax.set_ylabel(UNITS[idx], color="#9fb0c5", fontsize=8)

    @staticmethod
    def _style_panel_axis(ax, title: str) -> None:
        ax.set_facecolor("#101722")
        for spine in ax.spines.values():
            spine.set_color("#425066")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, loc="left", color="#e8f1ff", fontsize=10, pad=7)

    def start(self) -> None:
        previous_sigint = signal.getsignal(signal.SIGINT)

        def close_on_sigint(_signum, _frame):
            plt.close(self.fig)

        signal.signal(signal.SIGINT, close_on_sigint)
        ani = animation.FuncAnimation(
            self.fig,
            self._update,
            interval=self.interval_ms,
            blit=False,
            cache_frame_data=False,
        )
        self._animation = ani
        try:
            plt.show()
        finally:
            signal.signal(signal.SIGINT, previous_sigint)

    def _read_latest(self) -> None:
        while True:
            try:
                raw = self.data_queue.get_nowait()
            except queue.Empty:
                return

            raw_images = raw.get("images")
            images = raw_images if isinstance(raw_images, dict) else None
            if images is None and raw.get("image") is not None:
                images = {"camera_0": raw.get("image")}

            self.latest = TeleopSample(
                timestamp=float(raw.get("timestamp", time.monotonic())),
                wrench=np.asarray(raw.get("wrench", np.zeros(6)), dtype=float),
                joints=None if raw.get("joints") is None else np.asarray(raw["joints"], dtype=float),
                tcp_translation=None
                if raw.get("tcp_translation") is None
                else np.asarray(raw["tcp_translation"], dtype=float),
                image=raw.get("image"),
                images=images,
                camera_count=int(raw.get("camera_count", self.camera_count)),
                tactile=raw.get("tactile"),
                tactile_timestamp_ns=int(raw.get("tactile_timestamp_ns", 0)),
                dataset=raw.get("dataset"),
                teach=(
                    raw.get("teach")
                    if isinstance(raw.get("teach"), dict)
                    else None
                ),
                wrm=raw.get("wrm") if isinstance(raw.get("wrm"), dict) else None,
                beaver=(
                    raw.get("beaver")
                    if isinstance(raw.get("beaver"), dict)
                    else None
                ),
                source_label=str(raw.get("source_label", "teleop data")),
                status_extra=str(raw.get("status_extra", "")),
                connected=bool(raw.get("connected", True)),
                error=str(raw.get("error", "")),
            )

    def _request_rollback(self, _event) -> None:
        try:
            self.command_queue.put_nowait(
                {"command": "rollback_last_episode", "timestamp": time.monotonic()}
            )
            self.rollback_button.label.set_text("Undo queued")
        except queue.Full:
            self.rollback_button.label.set_text("Queue full")

    def _request_record_toggle(self, _event) -> None:
        try:
            self.command_queue.put_nowait(
                {"command": "toggle_recording", "timestamp": time.monotonic()}
            )
            self.record_button.label.set_text("Queued")
        except queue.Full:
            self.record_button.label.set_text("Queue full")

    def _queue_teach_command(self, command: str, label) -> None:
        try:
            self.command_queue.put_nowait(
                {"command": command, "timestamp": time.monotonic()}
            )
            label.set_text("Queued")
        except queue.Full:
            label.set_text("Queue full")

    def _request_teach_toggle(self, _event) -> None:
        status = self.latest.teach or {}
        if not status.get("teach_enabled", False):
            return
        self._queue_teach_command("toggle_teach", self.teach_button.label)

    def _request_reteach(self, _event) -> None:
        status = self.latest.teach or {}
        if not status.get("reteach_enabled", False):
            return
        self._queue_teach_command("reteach", self.reteach_button.label)

    def _request_replay_collect(self, _event) -> None:
        status = self.latest.teach or {}
        if not status.get("replay_enabled", False):
            return
        self._queue_teach_command("replay_collect", self.replay_button.label)

    def _request_initial_pose(self, _event) -> None:
        status = self.latest.teach or {}
        if not status.get("initial_pose_enabled", False):
            return
        self._queue_teach_command("initial_pose", self.initial_pose_button.label)

    def _update(self, _frame):
        self._read_latest()
        sample = self.latest
        wrench = np.asarray(sample.wrench, dtype=float).reshape(-1)[:6]
        if wrench.size < 6 or not np.all(np.isfinite(wrench)):
            wrench = np.zeros(6, dtype=float)

        if self.t0 is None:
            self.t0 = sample.timestamp
        self.times.append(sample.timestamp - self.t0)
        self.values.append(wrench.copy())

        self._update_plots(sample, wrench)
        self._update_status(sample, wrench)
        self._update_vector(wrench)
        self._update_tactile(sample)
        self._update_beaver(sample)
        self._update_camera(sample)
        self._update_pose(sample)
        return []

    def _update_plots(self, sample: TeleopSample, wrench: np.ndarray) -> None:
        if not self.times:
            return
        xs = np.asarray(self.times)
        ys = np.vstack(self.values)
        xmin = max(0.0, xs[-1] - self.window_s)
        xmax = max(self.window_s, xs[-1])

        for idx, ax in enumerate(self.axes):
            self.lines[idx].set_data(xs, ys[:, idx])
            ax.set_xlim(xmin, xmax)
            if idx < 3:
                ax.set_ylim(-self.force_panel_range, self.force_panel_range)
                self.value_texts[idx].set_text(f"{wrench[idx]:+.3f} {UNITS[idx]}")
                continue
            recent = ys[xs >= xmin, idx]
            center = 0.0
            spread = 1.0
            if recent.size:
                lo = float(np.nanmin(recent))
                hi = float(np.nanmax(recent))
                center = 0.5 * (lo + hi)
                spread = max(1e-3, hi - lo)
            margin = max(0.4 if idx < 3 else 0.04, spread * 0.65)
            ax.set_ylim(center - margin, center + margin)
            self.value_texts[idx].set_text(f"{wrench[idx]:+.3f} {UNITS[idx]}")

    def _update_status(self, sample: TeleopSample, wrench: np.ndarray) -> None:
        age = time.monotonic() - sample.timestamp
        state = "CONNECTED" if sample.connected else "ERROR"
        self.status_text.set_color("#47f091" if sample.connected else "#ff6b6b")
        extra = f"  |  {sample.status_extra}" if sample.status_extra else ""
        self.status_text.set_text(
            f"{state}  |  {sample.source_label}  |  latency {age * 1000:.0f} ms{extra}"
        )
        self.force_mag_text.set_text(f"|F| {float(np.linalg.norm(wrench[:3])):5.2f} N")
        self.torque_mag_text.set_text(f"|T| {float(np.linalg.norm(wrench[3:])):5.3f} Nm")
        self._update_dataset_status(sample)
        self._update_teach_status(sample)
        if sample.error:
            self.status_text.set_text(f"{state}  |  {sample.error[:58]}")

    def _update_dataset_status(self, sample: TeleopSample) -> None:
        if sample.dataset is None:
            self.dataset_text.set_text("")
            self.rollback_button_ax.set_visible(False)
            self.record_button_ax.set_visible(False)
            return
        if self.show_rollback_button:
            self.rollback_button_ax.set_visible(True)
        if self.show_record_button:
            self.record_button_ax.set_visible(True)
        status = sample.dataset or {}
        recorded = int(status.get("recorded_episodes", 0))
        current_frames = int(status.get("current_episode_frames", 0))
        last_length = status.get("last_episode_length")
        last_text = "--" if last_length is None else str(int(last_length))
        collecting = "REC" if status.get("collecting") else "idle"
        dataset_type = str(status.get("dataset_type", "?"))
        collect_rate = status.get("collect_rate_hz")
        rate_text = (
            ""
            if collect_rate is None
            else f"  {float(collect_rate):g} Hz"
        )
        self.dataset_text.set_text(
            f"{dataset_type} {collecting}{rate_text}\n"
            f"eps {recorded:04d}  cur {current_frames:04d}\n"
            f"last {last_text}"
        )
        self.record_button.label.set_text(
            "Save\nepisode" if status.get("collecting") else "Start\nrecord"
        )
        self.rollback_button.label.set_text("Undo\nepisode")
        if not self.show_record_button:
            self.record_button_ax.set_visible(False)
        if not self.show_rollback_button:
            self.rollback_button_ax.set_visible(False)
        elif recorded <= 0 and current_frames <= 0:
            self.rollback_button_ax.set_alpha(0.45)
        else:
            self.rollback_button_ax.set_alpha(1.0)

    @staticmethod
    def _set_control_enabled(axis, enabled: bool) -> None:
        axis.set_alpha(1.0 if enabled else 0.35)

    def _update_teach_status(self, sample: TeleopSample) -> None:
        if not self.show_teach_controls:
            for axis in self.teach_control_axes:
                axis.set_visible(False)
            return
        for axis in self.teach_control_axes:
            axis.set_visible(sample.teach is not None)
        if sample.teach is None:
            return

        status = sample.teach
        state = str(status.get("state", "idle"))
        frames = int(status.get("trajectory_frames", 0))
        state_text = {
            "teaching": "TEACH",
            "ready": "ready",
            "replaying": "REPLAY",
            "moving_initial": "INIT",
        }.get(state, "idle")
        dataset_text = self.dataset_text.get_text()
        self.dataset_text.set_text(
            f"{dataset_text}\npath {state_text} {frames:04d}".strip()
        )
        self.workflow_message_text.set_text(str(status.get("message", "")))
        self.teach_button.label.set_text(
            "End Teach" if state == "teaching" else "Teach"
        )
        self.reteach_button.label.set_text("Reteach")
        self.replay_button.label.set_text(
            "Collecting" if state == "replaying" else "Replay\ncollect"
        )
        self.initial_pose_button.label.set_text(
            "Moving..." if state == "moving_initial" else "Initial\npose"
        )
        self._set_control_enabled(
            self.teach_button_ax, bool(status.get("teach_enabled", False))
        )
        self._set_control_enabled(
            self.reteach_button_ax, bool(status.get("reteach_enabled", False))
        )
        self._set_control_enabled(
            self.replay_button_ax, bool(status.get("replay_enabled", False))
        )
        self._set_control_enabled(
            self.initial_pose_button_ax,
            bool(status.get("initial_pose_enabled", False)),
        )

    def _update_vector(self, wrench: np.ndarray) -> None:
        self.vector_arrow.remove()
        fxy = wrench[:2]
        fz = float(wrench[2])
        fxy_clipped = np.clip(fxy, -self.force_panel_range, self.force_panel_range)
        self.vector_arrow = self.vector_ax.arrow(
            0,
            0,
            fxy_clipped[0],
            fxy_clipped[1],
            color="#24d9ff",
            width=0.018,
            length_includes_head=True,
            head_width=max(0.09, self.force_panel_range * 0.045),
        )
        self.vector_label.set_text(
            f"Fx {fxy[0]:+.2f} N   Fy {fxy[1]:+.2f} N   Fz {fz:+.2f} N"
        )

    def _update_tactile(self, sample: TeleopSample) -> None:
        if self.tactile_panel is None or self.tactile_text is None:
            return
        self.tactile_panel.update(sample.tactile)
        if sample.tactile is None:
            self.tactile_text.set_text("waiting")
            self.tactile_text.set_color("#ffd94d")
            return
        age_ms = (
            (time.monotonic_ns() - sample.tactile_timestamp_ns) / 1e6
            if sample.tactile_timestamp_ns
            else 0.0
        )
        self.tactile_text.set_text(f"sensor {tuple(sample.tactile.shape)}  {age_ms:.0f} ms")
        self.tactile_text.set_color("#47f091")

    def _update_beaver(self, sample: TeleopSample) -> None:
        if not self.beaver_enabled or sample.beaver is None:
            return
        data = sample.beaver
        distance = np.asarray(data.get("distance_mm"))
        status = np.asarray(data.get("target_status"))
        present = np.asarray(data.get("present"), dtype=bool).reshape(-1)
        if (
            distance.ndim != 3
            or distance.shape != status.shape
            or distance.shape[0] != len(self.beaver_artists)
            or distance.shape[1] != distance.shape[2]
            or distance.shape[1] not in (4, 8)
        ):
            return
        stale = bool(data.get("stale", False))
        for slot, (ax, artist) in enumerate(
            zip(self.beaver_axes, self.beaver_artists)
        ):
            slot_distance = distance[slot]
            online = slot < present.size and bool(present[slot])
            # A zero distance with a valid status is a real reading: keep it
            # unmasked so it takes the reserved red contact colour, while
            # negative readings stay masked.
            valid = (
                online
                & np.isin(status[slot], (5, 9))
                & np.isfinite(slot_distance)
                & (slot_distance >= 0)
            )
            image = np.ma.masked_where(~valid, slot_distance)
            artist.set_data(image)
            bus, sensor = (
                self.beaver_layout[slot]
                if slot < len(self.beaver_layout)
                else (0, slot)
            )
            state = "stale" if stale and online else ("online" if online else "missing")
            min_text = (
                f"min {float(np.min(slot_distance[valid])):.1f} mm"
                if np.any(valid)
                else "min --"
            )
            average_text = (
                f"avg {float(np.mean(slot_distance[valid])):.1f} mm"
                if np.any(valid)
                else "avg --"
            )
            ax.set_title(
                f"B{bus}S{sensor} · {state} · {min_text} · {average_text}",
                color="#47f091" if online and not stale else "#ff6b6b",
                fontsize=10,
            )
        if self.beaver_fig is not None:
            connection = "CONNECTED" if data.get("connected") and not stale else "WAITING"
            error = str(data.get("error", ""))
            suffix = f" · {error[:80]}" if error else ""
            self.beaver_fig.suptitle(
                f"Beaver 9-sensor {distance.shape[1]}x{distance.shape[2]} view · "
                f"{connection} · frame "
                f"{int(data.get('frame_count', 0))} · lost "
                f"{int(data.get('lost_frames', 0))}{suffix}",
                color="#e8f1ff",
                fontsize=13,
            )
            self.beaver_fig.canvas.draw_idle()

    def _update_camera(self, sample: TeleopSample) -> None:
        images = sample.images or {}
        if not images and sample.image is not None:
            images = {"camera_0": sample.image}
        if not images:
            return
        for idx, ax in enumerate(self.camera_axes):
            key = f"camera_{idx}"
            image = images.get(key)
            if image is None and idx < len(images):
                image = images.get(sorted(images)[idx])
            if image is None:
                continue
            self.camera_placeholders[idx].set_visible(False)
            if self.camera_artists[idx] is None:
                self.camera_artists[idx] = ax.imshow(image)
            else:
                self.camera_artists[idx].set_data(image)

    def _update_pose(self, sample: TeleopSample) -> None:
        lines = []
        if sample.tcp_translation is not None and sample.tcp_translation.size >= 3:
            x, y, z = sample.tcp_translation[:3]
            lines.append(f"tcp xyz  {x:+.3f} {y:+.3f} {z:+.3f} m")
        else:
            lines.append("tcp xyz  unavailable")

        if sample.joints is not None and sample.joints.size:
            deg = np.degrees(sample.joints)
            lines.append("joint deg")
            columns = 4 if deg.size > 6 else 3
            for start in range(0, deg.size, columns):
                lines.append(" ".join(f"{v:+5.1f}" for v in deg[start : start + columns]))
        else:
            lines.append("joint deg unavailable")

        if sample.wrm is not None:
            alpha = sample.wrm.get("elbow_alpha")
            confidence = sample.wrm.get("confidence")
            target = sample.wrm.get("arm_angle_target_deg")
            tcp_z_offset = sample.wrm.get("tcp_z_offset_m")
            frozen = bool(sample.wrm.get("tracking_frozen", True))
            alpha_text = "--" if alpha is None else f"{float(alpha):.3f}"
            confidence_text = (
                "--" if confidence is None else f"{float(confidence):.2f}"
            )
            target_text = "--" if target is None else f"{float(target):+.1f} deg"
            lines.append(
                f"WRM alpha {alpha_text}  conf {confidence_text}  "
                f"{'FROZEN' if frozen else 'TRACK'}"
            )
            lines.append(f"arm-angle target {target_text}")
            if tcp_z_offset is not None:
                lines.append(
                    f"WRM TCP z offset {float(tcp_z_offset) * 100.0:+.1f} cm"
                )
        self.pose_text.set_text("\n".join(lines))


def _run_visualizer(
    data_queue: mp.Queue,
    command_queue: mp.Queue,
    _moving_average_window: int,
    _low_pass_alpha: float,
    hz: float,
    window_s: float,
    title: str,
    force_panel_range: float,
    camera_num: int,
    show_rollback_button: bool,
    show_record_button: bool,
    show_teach_controls: bool,
    show_tactile_panel: bool,
    beaver_enabled: bool,
    beaver_layout: tuple[tuple[int, int], ...],
    beaver_max_mm: float,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if matplotlib.get_backend().lower() == "agg":
        logger.warning("No interactive Matplotlib backend available; teleop visualizer will not open a window.")
        return
    dashboard = TeleopDashboard(
        data_queue,
        command_queue,
        window_s=window_s,
        hz=hz,
        title=title,
        force_panel_range=force_panel_range,
        camera_num=camera_num,
        show_rollback_button=show_rollback_button,
        show_record_button=show_record_button,
        show_teach_controls=show_teach_controls,
        show_tactile_panel=show_tactile_panel,
        beaver_enabled=beaver_enabled,
        beaver_layout=beaver_layout,
        beaver_max_mm=beaver_max_mm,
    )
    dashboard.start()


def start_visualizer(
    moving_average_window: int | None = None,
    low_pass_alpha: float | None = None,
    hz: float | None = None,
    window_s: float | None = None,
    title: str = "Teleop Visualizer",
    force_panel_range: float | None = None,
    camera_num: int = 1,
    show_rollback_button: bool = True,
    show_record_button: bool = False,
    show_teach_controls: bool = False,
    show_tactile_panel: bool = True,
    beaver_enabled: bool = False,
    beaver_layout: tuple[tuple[int, int], ...] = (),
    beaver_max_mm: float = 2500.0,
) -> VisualizerHandle:
    cfg = VisualizerConfig()
    # Kept for compatibility with older callers; real wrench filtering happens
    # before samples are published to this display process.
    moving_average_window = 1 if moving_average_window is None else moving_average_window
    low_pass_alpha = 0.0 if low_pass_alpha is None else low_pass_alpha
    hz = cfg.HZ if hz is None else hz
    window_s = cfg.WINDOW_S if window_s is None else window_s
    force_panel_range = (
        cfg.FORCE_PANEL_RANGE if force_panel_range is None else force_panel_range
    )
    ctx = mp.get_context("spawn")
    data_queue = ctx.Queue(maxsize=QUEUE_SIZE)
    command_queue = ctx.Queue(maxsize=8)
    process = ctx.Process(
        target=_run_visualizer,
        args=(
            data_queue,
            command_queue,
            moving_average_window,
            low_pass_alpha,
            hz,
            window_s,
            title,
            force_panel_range,
            camera_num,
            show_rollback_button,
            show_record_button,
            show_teach_controls,
            show_tactile_panel,
            beaver_enabled,
            beaver_layout,
            beaver_max_mm,
        ),
        daemon=True,
    )
    process.start()
    return VisualizerHandle(process, data_queue, command_queue)
