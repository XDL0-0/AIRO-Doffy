"""Read-only Matplotlib dashboard for received BODY telemetry."""

from __future__ import annotations

import time

import numpy as np

from doffy_teleop.protocol.body import BodyFrame
from .body_geometry import BODY_EDGES, KEY_JOINTS, _color, display_coordinates, pose_angles
from .body_pose import BodyPoseHold


class BodyDashboard:
    def __init__(self, plt, *, stale_after: float = 0.5, demo: bool = False, endpoint: str = ""):
        from matplotlib.collections import LineCollection
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        from matplotlib.widgets import Button, CheckButtons

        self.plt, self.stale_after, self.demo, self.endpoint = plt, stale_after, demo, endpoint
        self.show_hands, self.show_labels, self.show_axes, self.follow = True, False, False, True
        self._span = 2.2
        self._zoom = 1.0
        self._center = np.array([0.0, 0.9, 0.0])
        self._labels = []
        self.single_view = True
        self.selected_view = 0
        self._last_frame = None
        self._last_details = {}
        self._pose_hold = BodyPoseHold(stale_after)
        self.fig = plt.figure(figsize=(14, 9), facecolor="#0b1220")
        try:
            self.fig.canvas.manager.set_window_title("Meta human pose · AIRO-Doffy")
        except AttributeError:
            pass
        grid = self.fig.add_gridspec(2, 2, left=0.065, right=0.98, bottom=0.26,
                                    top=0.75, wspace=0.22, hspace=0.32)
        self.axes = [self.fig.add_subplot(grid[0, 0], projection="3d")]
        self.axes.extend(self.fig.add_subplot(spec) for spec in (grid[0, 1], grid[1, 0], grid[1, 1]))
        titles = ("3D · drag to rotate", "Front · X / Y", "Side · Z / Y", "Top · X / Z")
        self.lines, self.points = [], []
        for index, (axis, title) in enumerate(zip(self.axes, titles)):
            axis.set_facecolor("#111c2e")
            axis.set_title(title, color="#e2e8f0", fontsize=11, loc="left", pad=10)
            axis.tick_params(colors="#94a3b8", labelsize=8)
            if index == 0:
                axis.set_xlabel("X (m)", color="#94a3b8")
                axis.set_ylabel("Z (m)", color="#94a3b8")
                axis.set_zlabel("Y (m)", color="#94a3b8")
                axis.set_box_aspect((1, 1, 1))
                axis.view_init(elev=15, azim=-65)
                for dim in (axis.xaxis, axis.yaxis, axis.zaxis):
                    dim.set_pane_color((0.067, 0.11, 0.18, 1))
                lines = Line3DCollection([], linewidths=2)
                axis.add_collection(lines, autolim=False)
                points = axis.scatter([], [], [], s=14, depthshade=False)
            else:
                axis.set_aspect("equal", adjustable="box")
                axis.grid(color="#23334a", linewidth=0.6)
                for spine in axis.spines.values():
                    spine.set_color("#334155")
                axis.set_xlabel(("X (m)", "Z (m)", "X (m)")[index - 1], color="#94a3b8")
                axis.set_ylabel(("Y (m)", "Y (m)", "Z (m)")[index - 1], color="#94a3b8")
                lines = LineCollection([], linewidths=2)
                axis.add_collection(lines)
                points = axis.scatter([], [], s=14)
            self.lines.append(lines)
            self.points.append(points)
        self.orientation_lines = Line3DCollection([], linewidths=1, alpha=0.75)
        self.axes[0].add_collection(self.orientation_lines, autolim=False)
        self.orientation_projections = []
        for axis in self.axes[1:]:
            projected = LineCollection([], linewidths=1, alpha=0.75)
            axis.add_collection(projected)
            self.orientation_projections.append(projected)
        self.fig.text(0.04, 0.957, "META HUMAN POSE", color="#f8fafc", fontsize=19, weight="bold")
        self.fig.text(0.04, 0.922, "Left  ● cyan    Right  ● orange    Unity world coordinates · metres · Y up",
                      color="#94a3b8", fontsize=10)
        self.status = self.fig.text(0.04, 0.88, "Waiting for BODY telemetry", color="#fbbf24", fontsize=11)
        self.details = self.fig.text(0.04, 0.853, "", color="#94a3b8", fontsize=9)
        self.metrics = self.fig.text(0.04, 0.138, "", color="#cbd5e1", fontsize=10, linespacing=1.7)
        self.note = self.fig.text(0.04, 0.055, "", color="#94a3b8", fontsize=9)
        controls = self.fig.add_axes([0.73, 0.035, 0.245, 0.13], facecolor="#111c2e")
        self.checks = CheckButtons(controls, ["Finger joints", "Joint names", "Joint axes (XYZ)", "Follow hips"],
                                  [True, False, False, True])
        for label in self.checks.labels:
            label.set_color("#cbd5e1")
            label.set_fontsize(9)
        if hasattr(self.checks, "set_check_props"):
            self.checks.set_check_props({"facecolor": "#38bdf8"})
            self.checks.set_frame_props({"edgecolor": "#64748b"})
        self.checks.on_clicked(self._toggle)
        self._multi_positions = [axis.get_position(original=True).frozen() for axis in self.axes]

        def button(rect, label, callback):
            widget = Button(self.fig.add_axes(rect), label, color="#1e293b", hovercolor="#334155")
            widget.label.set_color("#e2e8f0")
            widget.label.set_fontsize(9)
            widget.on_clicked(callback)
            return widget

        self.layout_button = button([0.04, 0.79, 0.13, 0.037], "Four views", self._switch_layout)
        self.view_buttons = [button([0.20 + index * 0.095, 0.79, 0.085, 0.037], name,
                                    lambda _event, selected=index: self._select_view(selected))
                             for index, name in enumerate(("3D", "Front", "Side", "Top"))]
        self.zoom_in_button = button([0.61, 0.79, 0.075, 0.037], "Zoom +", lambda _: self._change_zoom(1.25))
        self.zoom_out_button = button([0.695, 0.79, 0.075, 0.037], "Zoom -", lambda _: self._change_zoom(0.8))
        self.fit_button = button([0.78, 0.79, 0.10, 0.037], "Fit pose", self._fit_pose)
        self._apply_view_layout()
        self.update(None)

    def _toggle(self, _label):
        self.show_hands, self.show_labels, self.show_axes, self.follow = self.checks.get_status()
        self._redisplay()

    def _redisplay(self):
        self.update(self._last_frame, **self._last_details)
        self.fig.canvas.draw_idle()

    def _apply_view_layout(self):
        for index, axis in enumerate(self.axes):
            axis.set_visible(not self.single_view or index == self.selected_view)
            axis.set_position([0.075, 0.23, 0.89, 0.52] if self.single_view else self._multi_positions[index])
            self.view_buttons[index].color = "#0369a1" if self.single_view and index == self.selected_view else "#1e293b"
            self.view_buttons[index].ax.set_facecolor(self.view_buttons[index].color)
        self.layout_button.label.set_text("Four views" if self.single_view else "Single skeleton")

    def _switch_layout(self, _event=None):
        self.single_view = not self.single_view
        self._apply_view_layout()
        self._redisplay()

    def _select_view(self, index: int):
        self.selected_view = index
        self.single_view = True
        self._apply_view_layout()
        self._redisplay()

    def _change_zoom(self, factor: float):
        self._zoom = float(np.clip(self._zoom * factor, 0.4, 8.0))
        self._redisplay()

    def _fit_pose(self, _event=None):
        self._zoom = 1.0
        self.follow = True
        if not self.checks.get_status()[3]:
            self.checks.set_active(3)
        self._redisplay()

    def _set_bounds(self, points: dict[str, np.ndarray]):
        if self.follow and points:
            # Root is an origin rather than an anatomical endpoint.
            meaningful = np.array([point for name, point in points.items() if name != "Root"])
            if len(meaningful):
                hips = points.get("Hips", meaningful.mean(axis=0))
                self._center = np.array([hips[0], (meaningful[:, 1].min() + meaningful[:, 1].max()) / 2, hips[2]])
                self._span = max(0.6, 2 * float(np.abs(meaningful - self._center).max()) + 0.18)
        visible_span = self._span / self._zoom
        lo, hi = self._center - visible_span / 2, self._center + visible_span / 2
        self.axes[0].set_xlim(lo[0], hi[0]); self.axes[0].set_ylim(lo[2], hi[2]); self.axes[0].set_zlim(lo[1], hi[1])
        for axis, dims in zip(self.axes[1:], ((0, 1), (2, 1), (0, 2))):
            axis.set_xlim(lo[dims[0]], hi[dims[0]]); axis.set_ylim(lo[dims[1]], hi[dims[1]])

    def _orientations(self, frame: BodyFrame | None, points: dict[str, np.ndarray], held=frozenset()):
        segments, colors = [], []
        if self.show_axes and frame is not None:
            for joint in frame.joints:
                if joint.name not in KEY_JOINTS or joint.name not in points or joint.rotation is None or joint.name in held:
                    continue
                x, y, z, w = np.array(joint.rotation) / np.linalg.norm(joint.rotation)
                matrix = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                                   [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                                   [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
                for direction, color in zip(matrix.T, ("#ef4444", "#22c55e", "#3b82f6")):
                    segments.append([joint.position, np.array(joint.position) + direction * 0.09])
                    colors.append(color)
        self.orientation_lines.set_segments(display_coordinates(segments) if segments else [])
        self.orientation_lines.set_color(colors)
        for collection, dims in zip(self.orientation_projections, ((0, 1), (2, 1), (0, 2))):
            collection.set_segments(np.array(segments)[:, :, dims] if segments else [])
            collection.set_color(colors)

    def update(self, frame: BodyFrame | None, *, now_ns: int | None = None,
               receive_hz: float = 0.0, source: str = "", rejected: int = 0):
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        self._last_frame = frame
        self._last_details = {"now_ns": now_ns, "receive_hz": receive_hz, "source": source, "rejected": rejected}
        age = (now_ns - frame.received_ns) / 1e9 if frame is not None else None
        stale = age is not None and age > self.stale_after
        displayed = self._pose_hold.update(frame, now_ns=now_ns, source=source)
        pose = displayed.frame
        holding = bool(displayed.reason)
        points = pose.positions(show_hands=self.show_hands) if pose is not None else {}
        segments, colors = [], []
        for first, second in BODY_EDGES:
            if first in points and second in points:
                segments.append([points[first], points[second]])
                colors.append(_color(second))
        xyz = np.array(list(points.values())).reshape(-1, 3)
        point_colors = [_color(name) for name in points]
        for index, (lines, scatter) in enumerate(zip(self.lines, self.points)):
            if index == 0:
                lines.set_segments(display_coordinates(segments) if segments else [])
                reordered = display_coordinates(xyz)
                scatter._offsets3d = (reordered[:, 0], reordered[:, 1], reordered[:, 2])
            else:
                dims = ((0, 1), (2, 1), (0, 2))[index - 1]
                lines.set_segments(np.array(segments)[:, :, dims] if segments else [])
                scatter.set_offsets(xyz[:, dims])
            lines.set_color(colors)
            scatter.set_color(point_colors)
            lines.set_alpha(0.23 if stale else 0.45 if holding else 0.9)
            scatter.set_alpha(0.23 if stale else 0.45 if holding else 0.9)
        for label in self._labels:
            label.remove()
        self._labels.clear()
        if self.show_labels:
            for name in KEY_JOINTS:
                if name in points:
                    short = name.replace("Left", "L ").replace("Right", "R ").replace("ArmLower", "elbow").replace("HandWrist", "wrist").replace("LowerLeg", "knee").replace("FootAnkle", "ankle")
                    for index, axis in enumerate(self.axes):
                        if not axis.get_visible():
                            continue
                        if index == 0:
                            x, z, y = display_coordinates(points[name])
                            label = axis.text(x, z, y + 0.025, short, color=_color(name), fontsize=7)
                        else:
                            dims = ((0, 1), (2, 1), (0, 2))[index - 1]
                            label = axis.annotate(short, points[name][list(dims)], xytext=(4, 4),
                                                  textcoords="offset points", color=_color(name), fontsize=7)
                        self._labels.append(label)
        self._set_bounds(points)
        self._orientations(pose if not stale else None, points,
                           displayed.held_positions | displayed.held_rotations)
        if frame is None and pose is None:
            state, color = "WAITING · no BODY packets received", "#fbbf24"
            self.details.set_text(f"Listening on {self.endpoint} · enable Body data in Quest Session")
            self.metrics.set_text("Arm raise: down = 0°, horizontal = 90°\nElbow / knee inner angle: straight = 180°")
            self.note.set_text("Enable body tracking permission on Quest. Upper-body packets contain no legs.")
        else:
            if stale or displayed.reason == "stale":
                state = "STALE · holding last valid pose"
            elif not points:
                state = "TRACKING LOST · waiting for a valid pose"
            elif holding:
                reason = "tracking lost" if displayed.reason == "invalid" else displayed.reason
                state = f"HOLDING · {reason}"
            else:
                state = "LIVE"
            color = "#f87171" if not points else "#fbbf24" if holding else "#4ade80"
            raw = frame if frame is not None else pose
            kind = "FULL BODY" if raw.joint_set == "full_body" else "UPPER BODY · legs unavailable"
            state += f"  |  {kind}  |  confidence {raw.confidence:.2f}"
            pose_age = (now_ns - displayed.last_valid_received_ns) / 1e9 if displayed.last_valid_received_ns is not None else None
            hold_text = f" · held joints {len(displayed.held_positions | displayed.held_rotations)} · valid pose age {pose_age * 1000:.0f} ms" if holding and pose_age is not None else ""
            self.details.set_text(f"Frame {raw.frame_id} · {receive_hz:.1f} Hz received · age {(age or 0) * 1000:.0f} ms · valid positions {len(raw.positions())}/{len(raw.joints)} · {source} · rejected {rejected}" + hold_text)
            angles = pose_angles(pose) if pose is not None else {}
            fmt = lambda value: "—" if value is None else f"{value:.1f}°"
            self.metrics.set_text("    ".join(f"{side[0]} arm raise {fmt(angles.get(side + ' arm raise'))}    elbow {fmt(angles.get(side + ' elbow'))}    knee {fmt(angles.get(side + ' knee'))}" for side in ("Left", "Right")) +
                                  "\nArm raise: down = 0°, horizontal = 90° · Elbow / knee: straight = 180°" +
                                  (f" · WRM alpha {pose.wrm.get('elbow_alpha', 0):.3f} / {'calibrated' if pose.wrm.get('calibrated') else 'uncalibrated'}" if pose is not None and pose.wrm else ""))
            self.note.set_text("Pose frozen where tracking is lost; valid tracking resumes updates automatically." if holding else
                               "Full-body joints are the Meta SDK output; inspect their validity and confidence." if raw.joint_set == "full_body" else
                               "Only received upper-body joints are shown. Joints without a valid history stay empty.")
        self.status.set_text(("DEMO · synthetic data  |  " if self.demo else "") + state)
        self.status.set_color(color)
