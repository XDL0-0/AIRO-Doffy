"""BODY telemetry checks at the JSON, socket, and visible dashboard boundaries."""

from __future__ import annotations

import builtins
from dataclasses import replace
import json
import select
import socket
import struct

import numpy as np
import pytest

from doffy_teleop import body_visualization as body
from doffy_teleop.runtime import body_viewer as viewer


def _joint(joint_name, position=(0.0, 1.0, 0.0), **changes):
    record = {
        "id": body.JOINT_IDS[joint_name], "name": joint_name,
        "position": None if position is None else list(position),
        "rotation": [0.0, 0.0, 0.0, 1.0],
        "position_valid": True, "orientation_valid": True,
    }
    record.update(changes)
    return record


def _packet(*joints, joint_set="full_body", frame_id=1, timestamp_ns=10):
    return {
        "type": "BODY", "version": 1, "coordinate_space": "unity_world",
        "joint_set": joint_set, "frame_id": frame_id,
        "timestamp_ns": timestamp_ns, "confidence": 0.9,
        "tracking_valid": True, "joints": list(joints or [_joint("Hips")]),
        "wrm": {"enabled": True, "calibrated": False, "elbow_alpha": 0.25},
    }


def _frame(payload, *, received_ns=10_000_000_000):
    frame = body.parse_body_packet(json.dumps(payload), received_ns=received_ns)
    assert frame is not None
    return frame


@pytest.fixture
def plt():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as pyplot
    yield pyplot
    pyplot.close("all")


def test_canonical_ids_agree_with_meta_upper_and_full_body_boundary():
    # These SDK landmarks are the protocol boundary, rather than a copy of the
    # implementation's name-generation loop.
    landmarks = {
        0: "Root", 1: "Hips", 7: "Head", 8: "LeftShoulder",
        11: "LeftArmLower", 13: "RightShoulder", 16: "RightArmLower",
        18: "LeftHandPalm", 19: "LeftHandWrist", 43: "LeftHandLittleTip",
        44: "RightHandPalm", 45: "RightHandWrist", 69: "RightHandLittleTip",
        70: "LeftUpperLeg", 73: "LeftFootAnkle", 77: "RightUpperLeg",
        80: "RightFootAnkle", 83: "RightFootBall",
    }
    assert len(body.JOINT_NAMES) == len(set(body.JOINT_NAMES)) == 84
    for joint_id, name in landmarks.items():
        assert body.JOINT_NAMES[joint_id] == name
        assert body.JOINT_IDS[name] == joint_id


@pytest.mark.parametrize("joint_set, count", [("upper_body", 70), ("full_body", 84)])
def test_complete_body_packet_preserves_all_received_joints(joint_set, count):
    records = [_joint(name, (index / 100, 1.0, -0.3))
               for index, name in enumerate(body.JOINT_NAMES[:count])]
    # Receive-order does not change anatomical identity.
    frame = _frame(_packet(*reversed(records), joint_set=joint_set))
    assert len(frame.positions()) == count
    np.testing.assert_allclose(frame.positions()["RightHandLittleTip"], [0.69, 1, -0.3])
    assert frame.received_ns == 10_000_000_000
    if joint_set == "upper_body":
        assert "LeftUpperLeg" not in frame.positions()


@pytest.mark.parametrize("field, invalid", [
    ("type", "WRM"), ("version", 2), ("version", True),
    ("coordinate_space", "tracking_space"), ("joint_set", "estimated_body"),
    ("joint_set", []), ("frame_id", True), ("frame_id", -1),
    ("timestamp_ns", "10"), ("timestamp_ns", -1),
    ("tracking_valid", 1), ("confidence", True), ("confidence", -0.1),
    ("confidence", 1.1), ("confidence", float("nan")),
    ("confidence", float("inf")), ("joints", {}), ("joints", [None]),
    ("joints", [[]]), ("joints", ["Head"]), ("wrm", []),
    ("wrm", {"elbow_alpha": float("nan")}),
    ("wrm", {"confidence": 1.1}), ("wrm", {"enabled": 1}),
    ("wrm", {"calibrated": "yes"}),
])
def test_malformed_metadata_is_ignored_without_raising(field, invalid):
    payload = _packet()
    payload[field] = invalid
    assert body.parse_body_packet(json.dumps(payload)) is None


@pytest.mark.parametrize("missing", [
    "coordinate_space", "joint_set", "frame_id", "timestamp_ns",
    "confidence", "tracking_valid", "joints",
])
def test_required_body_metadata_cannot_be_omitted(missing):
    payload = _packet()
    del payload[missing]
    assert body.parse_body_packet(json.dumps(payload)) is None


@pytest.mark.parametrize("raw", [
    b"\xff\xfe", b"{", b"null", b"[]", b"42", b"C,1,20,0,0",
    b" " * 65508,
])
def test_non_body_and_oversize_datagrams_are_ignored(raw):
    assert body.parse_body_packet(raw) is None


@pytest.mark.parametrize("changes", [
    {"id": True}, {"id": -1}, {"id": 84}, {"id": "1"},
    {"name": "Head"}, {"position": [1, 2]}, {"position": [1, 2, 3, 4]},
    {"position": {"x": 1, "y": 2, "z": 3}}, {"position": [1, True, 3]},
    {"position": [1, float("nan"), 3]}, {"position": [float("inf"), 2, 3]},
    {"position": None}, {"position_valid": 1},
    {"rotation": [0, 0, 1]}, {"rotation": [0, 0, 0, 0]},
    {"rotation": [0, 0, 0, float("inf")]}, {"rotation": [0, 0, 0, True]},
    {"rotation": None}, {"orientation_valid": "true"},
])
def test_malformed_joint_rejects_the_whole_frame(changes):
    # Never display a partial frame silently after a bad joint record.
    payload = _packet(_joint("Head"), _joint("Hips", **changes))
    assert body.parse_body_packet(json.dumps(payload)) is None


@pytest.mark.parametrize("missing", [
    "id", "name", "position", "rotation", "position_valid", "orientation_valid",
])
def test_required_joint_fields_cannot_be_omitted(missing):
    payload = _packet()
    del payload["joints"][0][missing]
    assert body.parse_body_packet(json.dumps(payload)) is None


def test_duplicate_ids_and_upper_body_leg_claims_are_rejected():
    assert body.parse_body_packet(json.dumps(_packet(_joint("Hips"), _joint("Hips")))) is None
    assert body.parse_body_packet(json.dumps(_packet(_joint("LeftUpperLeg"), joint_set="upper_body"))) is None
    too_many = _packet(*[_joint("Hips") for _ in range(71)], joint_set="upper_body")
    assert body.parse_body_packet(json.dumps(too_many)) is None


def test_joint_and_whole_frame_validity_leave_unavailable_geometry_empty():
    payload = _packet(
        _joint("Head"),
        _joint("Hips", position=None, position_valid=False),
        _joint("RightShoulder", rotation=None, orientation_valid=False),
    )
    frame = _frame(payload)
    assert set(frame.positions()) == {"Head", "RightShoulder"}
    assert {joint.name: joint.rotation for joint in frame.joints}["RightShoulder"] is None
    payload["tracking_valid"] = False
    invalid = _frame(payload)
    assert invalid.positions() == {}
    assert all(value is None for value in body.pose_angles(invalid).values())


def _deliver(receiver, sender, payload):
    raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    sender.sendto(raw, receiver.address)
    readable, _, _ = select.select([receiver.socket], [], [], 1.0)
    assert readable, "localhost body datagram did not reach the socket"
    return receiver.poll()


def test_real_udp_keeps_newest_pose_rejects_bad_data_and_clears_on_tracking_loss():
    receiver = body.BodyReceiver("127.0.0.1", 0)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sender.bind(("127.0.0.1", 0))
    try:
        assert receiver.poll() is None
        first = _deliver(receiver, sender, _packet(_joint("Head"), timestamp_ns=100))
        assert first.frame_id == 1
        assert receiver.source == sender.getsockname()
        assert _deliver(receiver, sender, b"not JSON") is first
        assert receiver.rejected == 1
        malformed = _packet(_joint("Head", position=[float("nan"), 0, 0]), timestamp_ns=999)
        assert _deliver(receiver, sender, malformed) is first
        assert receiver.rejected == 2

        lost = _packet(_joint("Head"), frame_id=2, timestamp_ns=200)
        lost["tracking_valid"] = False
        latest = _deliver(receiver, sender, lost)
        assert latest.frame_id == 2
        assert latest.positions() == {}
        # A delayed valid packet must not restore the previous skeleton.
        assert _deliver(receiver, sender, _packet(_joint("Head"), timestamp_ns=150)) is latest
        assert receiver.accepted == 2
        recovered = _deliver(receiver, sender, _packet(_joint("Head"), frame_id=3, timestamp_ns=300))
        assert recovered.frame_id == 3
        assert "Head" in recovered.positions()
        assert receiver.accepted == 3
        assert receiver.receive_hz(recovered.received_ns + 2_000_000_000) == 0.0
    finally:
        sender.close()
        receiver.close()
    assert receiver.socket.fileno() == -1


def test_real_udp_drains_bursts_and_accepts_a_new_source_clock():
    receiver = body.BodyReceiver("127.0.0.1", 0)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    restarted = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sender.bind(("127.0.0.1", 0))
    restarted.bind(("127.0.0.1", 0))
    try:
        for frame_id in range(1, 4):
            sender.sendto(json.dumps(_packet(frame_id=frame_id, timestamp_ns=frame_id * 100)).encode(), receiver.address)
        assert select.select([receiver.socket], [], [], 1.0)[0]
        receiver.poll(limit=1)
        assert receiver.latest.frame_id == 1
        receiver.poll()
        assert receiver.latest.frame_id == 3
        assert receiver.accepted == 3
        new_session = _deliver(receiver, restarted, _packet(frame_id=0, timestamp_ns=1))
        assert new_session.frame_id == 0
        assert receiver.source == restarted.getsockname()
    finally:
        restarted.close()
        sender.close()
        receiver.close()


def test_anatomical_angles_use_inner_joint_angle_and_downward_arm_zero():
    frame = _frame(_packet(
        _joint("LeftShoulder", (-1, 2, 0)), _joint("LeftArmLower", (-1, 1, 0)),
        _joint("LeftHandWrist", (-1, 0, 0)),
        _joint("LeftUpperLeg", (-1, 0, 0)), _joint("LeftLowerLeg", (-1, -1, 0)),
        _joint("LeftFootAnkle", (-1, -2, 0)),
        _joint("RightShoulder", (1, 2, 0)), _joint("RightArmLower", (2, 2, 0)),
        _joint("RightHandWrist", (2, 1, 0)),
        _joint("RightUpperLeg", (1, 0, 0)), _joint("RightLowerLeg", (1, -1, 0)),
        _joint("RightFootAnkle", (1, -1, 1)),
    ))
    assert body.pose_angles(frame) == pytest.approx({
        "Left elbow": 180, "Left knee": 180, "Left arm raise": 0,
        "Right elbow": 90, "Right knee": 90, "Right arm raise": 90,
    })
    degenerate = _frame(_packet(_joint("LeftShoulder"), _joint("LeftArmLower")))
    assert body.pose_angles(degenerate)["Left arm raise"] is None
    assert body.pose_angles(degenerate)["Left knee"] is None


def test_display_axes_reorder_unity_coordinates_without_mirroring():
    np.testing.assert_array_equal(body.display_coordinates(np.array([
        [-2, 3, -4], [5, -6, 7],
    ])), [[-2, -4, 3], [5, 7, -6]])


def _display_frame(*, tracking_valid=True, joint_set="upper_body"):
    payload = _packet(
        _joint("Hips", (0, 1, 0)), _joint("Chest", (0, 1.5, 0)),
        _joint("Head", (0, 1.8, 0)),
        _joint("LeftShoulder", (-0.2, 1.5, 0.1)),
        _joint("LeftArmLower", (-0.2, 1.1, 0.2)),
        _joint("LeftHandWrist", (-0.2, 0.8, 0.3), orientation_valid=False),
        _joint("LeftHandIndexTip", (-0.3, 0.8, 0.4)),
        joint_set=joint_set,
    )
    payload["tracking_valid"] = tracking_valid
    return _frame(payload)


def test_dashboard_waiting_live_invalid_and_stale_states_render_on_agg(plt):
    dashboard = body.BodyDashboard(plt, endpoint="UDP 127.0.0.1:8015", stale_after=0.5)
    dashboard.fig.canvas.draw()
    assert "WAITING" in dashboard.status.get_text()
    assert "no BODY" in dashboard.status.get_text()
    assert all(len(scatter.get_offsets()) == 0 for scatter in dashboard.points[1:])

    frame = _display_frame()
    dashboard.update(frame, now_ns=frame.received_ns, receive_hz=30, rejected=2)
    dashboard.fig.canvas.draw()
    assert "LIVE" in dashboard.status.get_text()
    assert "legs unavailable" in dashboard.status.get_text()
    assert "30.0 Hz" in dashboard.details.get_text()
    assert "rejected 2" in dashboard.details.get_text()
    assert "knee —" in dashboard.metrics.get_text()
    # Check all anatomical projections with differing XYZ values.
    np.testing.assert_allclose(dashboard.points[1].get_offsets()[3], [-0.2, 1.5])
    np.testing.assert_allclose(dashboard.points[2].get_offsets()[3], [0.1, 1.5])
    np.testing.assert_allclose(dashboard.points[3].get_offsets()[3], [-0.2, 0.1])
    xyz = np.column_stack(dashboard.points[0]._offsets3d)
    np.testing.assert_allclose(xyz[3], [-0.2, 0.1, 1.5])

    dashboard.update(frame, now_ns=frame.received_ns + 1_000_000_000)
    dashboard.fig.canvas.draw()
    assert "STALE" in dashboard.status.get_text()
    assert len(dashboard.points[1].get_offsets()) == len(frame.positions())
    assert dashboard.points[1].get_alpha() < 0.5
    assert "age 1000 ms" in dashboard.details.get_text()

    invalid = _display_frame(tracking_valid=False)
    dashboard.update(invalid, now_ns=invalid.received_ns)
    dashboard.fig.canvas.draw()
    assert "HOLDING" in dashboard.status.get_text()
    assert "tracking lost" in dashboard.status.get_text()
    assert "valid positions 0/" in dashboard.details.get_text()
    assert "held joints 7" in dashboard.details.get_text()
    assert len(dashboard.points[1].get_offsets()) == len(frame.positions())
    np.testing.assert_allclose(dashboard.points[1].get_offsets()[3], [-0.2, 1.5])
    assert dashboard.lines[1].get_segments()


def test_dashboard_controls_labels_fingers_and_valid_rotation_axes(plt):
    dashboard = body.BodyDashboard(plt)
    frame = _display_frame()
    dashboard.update(frame, now_ns=frame.received_ns)
    assert not dashboard.axes[0].texts
    assert not dashboard.orientation_lines._segments3d
    dashboard.checks.set_active(0)  # hide finger joints
    dashboard.checks.set_active(1)  # show key joint names
    dashboard.checks.set_active(2)  # show local XYZ rotation axes
    dashboard.update(frame, now_ns=frame.received_ns)
    dashboard.fig.canvas.draw()
    assert len(dashboard.points[1].get_offsets()) == len(frame.positions()) - 1
    assert {text.get_text() for text in dashboard.axes[0].texts} == {
        "Hips", "Head", "L Shoulder", "L elbow", "L wrist",
    }
    # Hips, Head, shoulder and elbow have valid orientation; wrist doesn't.
    axes_segments = dashboard.orientation_lines._segments3d
    assert len(axes_segments) == 4 * 3
    np.testing.assert_allclose(axes_segments[0], [[0, 0, 1], [0.09, 0, 1]])
    np.testing.assert_allclose(axes_segments[1], [[0, 0, 1], [0, 0, 1.09]])
    np.testing.assert_allclose(axes_segments[2], [[0, 0, 1], [0, 0.09, 1]])

    dashboard.update(frame, now_ns=frame.received_ns + 1_000_000_000)
    assert not dashboard.orientation_lines._segments3d  # old rotations cannot look live
    dashboard.update(_display_frame(tracking_valid=False), now_ns=frame.received_ns)
    assert dashboard.axes[0].texts  # frozen joints keep their labels
    assert not dashboard.orientation_lines._segments3d

    dashboard.checks.set_active(1)
    dashboard.checks.set_active(2)
    dashboard.update(frame, now_ns=frame.received_ns)
    assert not dashboard.axes[0].texts
    assert not dashboard.orientation_lines._segments3d


def test_run_visualizer_does_not_signal_ready_on_import_failure(monkeypatch):
    ready = []
    original_import = builtins.__import__

    def fail_pyplot(name, *args, **kwargs):
        if name == "matplotlib.pyplot":
            raise ImportError("test plotting dependency unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_pyplot)
    with pytest.raises(ImportError, match="plotting dependency"):
        body.run_visualizer(demo=True, on_ready=lambda: ready.append(True))
    assert ready == []


def test_run_visualizer_does_not_signal_ready_on_real_bind_failure(plt):
    occupied = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    occupied.bind(("127.0.0.1", 0))
    ready = []
    try:
        with pytest.raises(OSError):
            body.run_visualizer("127.0.0.1", occupied.getsockname()[1],
                                on_ready=lambda: ready.append(True))
        assert ready == []
        assert plt.get_fignums() == []
    finally:
        occupied.close()


def test_run_visualizer_render_failure_never_signals_ready_and_releases_socket(monkeypatch, plt):
    ready, receivers, dashboards = [], [], []
    original_receiver, original_dashboard = body.BodyReceiver, body.BodyDashboard

    def receiver(*args, **kwargs):
        value = original_receiver(*args, **kwargs)
        receivers.append(value)
        return value

    def dashboard(*args, **kwargs):
        value = original_dashboard(*args, **kwargs)
        dashboards.append(value)

        def fail_draw(*_args, **_kwargs):
            raise RuntimeError("test initial rendering failed")

        monkeypatch.setattr(value.fig.canvas, "draw", fail_draw)
        return value

    monkeypatch.setattr(viewer, "BodyReceiver", receiver)
    monkeypatch.setattr(viewer, "BodyDashboard", dashboard)
    with pytest.raises(RuntimeError, match="initial rendering"):
        body.run_visualizer("127.0.0.1", 0, on_ready=lambda: ready.append(True))
    assert ready == []
    assert receivers[0].socket.fileno() == -1
    assert not plt.fignum_exists(dashboards[0].fig.number)
    rebound = original_receiver(*receivers[0].address)
    rebound.close()


def test_run_visualizer_noninteractive_backend_fails_before_ready(plt):
    ready = []
    with pytest.raises(RuntimeError, match="interactive Matplotlib backend"):
        body.run_visualizer("127.0.0.1", 0, on_ready=lambda: ready.append(True))
    assert ready == []
    assert plt.get_fignums() == []


def test_save_preview_uses_no_network_and_never_signals_teleop_ready(monkeypatch, plt, tmp_path):
    ready, dashboards = [], []
    original_dashboard = body.BodyDashboard

    def forbidden_receiver(*_args, **_kwargs):
        pytest.fail("save_preview must never open a telemetry socket")

    def dashboard(*args, **kwargs):
        value = original_dashboard(*args, **kwargs)
        dashboards.append(value)
        return value

    monkeypatch.setattr(viewer, "BodyReceiver", forbidden_receiver)
    monkeypatch.setattr(viewer, "BodyDashboard", dashboard)
    destination = tmp_path / "preview" / "body.png"
    body.run_visualizer(save_preview=destination, on_ready=lambda: ready.append(True))
    raw = destination.read_bytes()
    assert raw.startswith(b"\x89PNG\r\n\x1a\n")
    assert struct.unpack(">II", raw[16:24]) == (2100, 1350)
    assert "DEMO" in dashboards[0].status.get_text()
    assert "synthetic UI preview" in dashboards[0].details.get_text()
    assert ready == []
    assert plt.get_fignums() == []


def _click_dashboard_button(dashboard, widget):
    from matplotlib.backend_bases import MouseEvent

    dashboard.fig.canvas.draw()
    x, y = widget.ax.transAxes.transform((0.5, 0.5))
    for name in ("button_press_event", "button_release_event"):
        event = MouseEvent(name, dashboard.fig.canvas, x, y, button=1)
        dashboard.fig.canvas.callbacks.process(name, event)


def test_single_skeleton_buttons_expand_chosen_view_and_restore_four_views(plt):
    dashboard = body.BodyDashboard(plt)
    frame = body.demo_frame()
    dashboard.update(frame, now_ns=frame.received_ns)
    assert [axis.get_visible() for axis in dashboard.axes] == [True, False, False, False]
    large_height = dashboard.axes[0].get_position().height
    dashboard.axes[0].view_init(elev=22, azim=-35)
    _click_dashboard_button(dashboard, dashboard.layout_button)
    assert all(axis.get_visible() for axis in dashboard.axes)
    assert dashboard.axes[0].get_position().height < large_height / 1.5
    assert dashboard.layout_button.label.get_text() == "Single skeleton"
    _click_dashboard_button(dashboard, dashboard.view_buttons[1])
    assert [axis.get_visible() for axis in dashboard.axes] == [False, True, False, False]
    assert dashboard.axes[1].get_position().height > large_height * 0.9
    assert len(dashboard.points[1].get_offsets()) == 84
    _click_dashboard_button(dashboard, dashboard.view_buttons[0])
    assert dashboard.axes[0].get_visible()
    assert dashboard.axes[0].azim == -35
    assert dashboard.axes[0].elev == 22


def test_zoom_and_fit_buttons_work_without_new_packets_and_fit_upper_body(plt):
    dashboard = body.BodyDashboard(plt)
    full = body.demo_frame()
    dashboard.update(full, now_ns=full.received_ns)
    full_span = np.ptp(dashboard.axes[0].get_zlim())
    upper = _display_frame()
    dashboard.update(upper, now_ns=upper.received_ns)
    fitted = np.ptp(dashboard.axes[0].get_zlim())
    assert fitted < full_span
    _click_dashboard_button(dashboard, dashboard.zoom_in_button)
    assert np.ptp(dashboard.axes[0].get_zlim()) == pytest.approx(fitted / 1.25)
    assert "LIVE" in dashboard.status.get_text()
    _click_dashboard_button(dashboard, dashboard.zoom_out_button)
    assert np.ptp(dashboard.axes[0].get_zlim()) == pytest.approx(fitted)
    dashboard.checks.set_active(3)  # Manual view; Fit restores following.
    _click_dashboard_button(dashboard, dashboard.zoom_in_button)
    _click_dashboard_button(dashboard, dashboard.fit_button)
    assert dashboard.follow and dashboard.checks.get_status()[3]
    assert np.ptp(dashboard.axes[0].get_zlim()) == pytest.approx(fitted)


def test_projection_views_show_names_and_orientation_axes_when_selected(plt):
    dashboard = body.BodyDashboard(plt)
    frame = _display_frame()
    dashboard.update(frame, now_ns=frame.received_ns)
    dashboard.checks.set_active(1)
    dashboard.checks.set_active(2)
    for index in (1, 2, 3):
        _click_dashboard_button(dashboard, dashboard.view_buttons[index])
        assert dashboard.axes[index].texts
        assert not dashboard.axes[0].texts
        assert len(dashboard.orientation_projections[index - 1].get_segments()) == 12
    invalid = _display_frame(tracking_valid=False)
    dashboard.update(invalid, now_ns=invalid.received_ns)
    assert dashboard.axes[3].texts  # keep the last pose in the selected view
    assert not any(collection.get_segments() for collection in dashboard.orientation_projections)


def test_low_confidence_fallback_is_frozen_across_view_controls_and_then_recovers(plt):
    dashboard = body.BodyDashboard(plt)
    good = _display_frame()
    dashboard.update(good, now_ns=good.received_ns)
    original_points = np.array(dashboard.points[1].get_offsets())
    original_metrics = dashboard.metrics.get_text()
    original_bounds = dashboard.axes[0].get_zlim()
    fallback = replace(good, frame_id=2, confidence=0.4, received_ns=good.received_ns + 100_000_000,
                       joints=tuple(replace(joint, position=(0, 0, 0)) for joint in good.joints))
    dashboard.update(fallback, now_ns=fallback.received_ns)
    assert "HOLDING" in dashboard.status.get_text()
    assert "confidence 0.40" in dashboard.status.get_text()
    assert "Frame 2" in dashboard.details.get_text()
    np.testing.assert_allclose(dashboard.points[1].get_offsets(), original_points)
    assert dashboard.metrics.get_text() == original_metrics
    assert dashboard.axes[0].get_zlim() == original_bounds
    _click_dashboard_button(dashboard, dashboard.view_buttons[1])
    _click_dashboard_button(dashboard, dashboard.zoom_in_button)
    np.testing.assert_allclose(dashboard.points[1].get_offsets(), original_points)
    assert "HOLDING" in dashboard.status.get_text()
    restored = replace(good, frame_id=3, received_ns=fallback.received_ns + 100_000_000,
                       joints=tuple(replace(joint, position=tuple(np.array(joint.position) + [0.3, 0, 0]))
                                    for joint in good.joints))
    dashboard.update(restored, now_ns=restored.received_ns)
    assert "LIVE" in dashboard.status.get_text()
    np.testing.assert_allclose(dashboard.points[1].get_offsets(), original_points + [0.3, 0])


def test_initial_tracking_loss_has_no_invented_skeleton(plt):
    dashboard = body.BodyDashboard(plt)
    lost = _display_frame(tracking_valid=False)
    dashboard.update(lost, now_ns=lost.received_ns)
    assert "TRACKING LOST" in dashboard.status.get_text()
    assert "waiting for a valid pose" in dashboard.status.get_text()
    assert all(len(scatter.get_offsets()) == 0 for scatter in dashboard.points[1:])
    assert all(len(lines.get_segments()) == 0 for lines in dashboard.lines[1:])
