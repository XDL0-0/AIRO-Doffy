"""BODY tracking quality is retained separately from geometry validity."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import pytest

from doffy_teleop.protocol.body import BodyJoint, parse_body_packet


def _joint(name="Head", joint_id=7, **changes):
    record = {
        "id": joint_id,
        "name": name,
        "position": [0, 1.7, 0],
        "rotation": [0, 0, 0, 1],
        "position_valid": True,
        "orientation_valid": True,
    }
    record.update(changes)
    return record


def _packet(*joints, **changes):
    payload = {
        "type": "BODY",
        "version": 1,
        "coordinate_space": "unity_world",
        "joint_set": "upper_body",
        "frame_id": 1,
        "timestamp_ns": 10,
        "confidence": 0.9,
        "tracking_valid": True,
        "joints": list(joints or [_joint()]),
    }
    payload.update(changes)
    return json.dumps(payload)


def _frame(*joints, **changes):
    frame = parse_body_packet(_packet(*joints, **changes), received_ns=123)
    assert frame is not None
    return frame


@pytest.mark.parametrize("position_tracked, orientation_tracked", [
    (True, True), (True, False), (False, True), (False, False),
])
def test_joint_tracking_flags_are_retained_independently(position_tracked, orientation_tracked):
    frame = _frame(_joint(position_tracked=position_tracked,
                          orientation_tracked=orientation_tracked))
    joint = frame.joints[0]
    assert joint.position_tracked is position_tracked
    assert joint.orientation_tracked is orientation_tracked
    assert joint.position == (0.0, 1.7, 0.0)
    assert joint.rotation == (0.0, 0.0, 0.0, 1.0)


def test_untracked_estimates_remain_distinct_from_invalid_geometry():
    frame = _frame(
        _joint(position_tracked=False, orientation_tracked=False),
        _joint("Hips", 1, position_valid=False, orientation_valid=False,
               position_tracked=True, orientation_tracked=True),
    )
    estimate, invalid = frame.joints
    assert estimate.position is not None
    assert estimate.rotation is not None
    assert estimate.position_tracked is estimate.orientation_tracked is False
    assert invalid.position is invalid.rotation is None
    assert invalid.position_tracked is invalid.orientation_tracked is True
    # Raw positions follow validity and do not apply a display tracking policy.
    assert set(frame.positions()) == {"Head"}
    assert tuple(frame.positions()["Head"]) == (0.0, 1.7, 0.0)


@pytest.mark.parametrize("changes, expected", [
    ({}, (None, None)),
    ({"position_tracked": False}, (False, None)),
    ({"orientation_tracked": True}, (None, True)),
])
def test_legacy_and_partially_extended_packets_leave_absent_flags_unknown(changes, expected):
    joint = _frame(_joint(**changes)).joints[0]
    assert (joint.position_tracked, joint.orientation_tracked) == expected


@pytest.mark.parametrize("field", ["position_tracked", "orientation_tracked"])
@pytest.mark.parametrize("invalid", [None, 0, 1, 0.0, 1.0, "true", "false", [], {}])
def test_present_tracking_flags_require_actual_booleans(field, invalid):
    assert parse_body_packet(_packet(_joint(**{field: invalid}))) is None


@pytest.mark.parametrize("confidence", [0.0, 0.2, 0.5])
def test_low_confidence_is_preserved_for_the_display_policy(confidence):
    frame = _frame(_joint(position_tracked=True, orientation_tracked=True),
                   confidence=confidence)
    assert frame.confidence == confidence
    assert frame.tracking_valid is True
    assert frame.joints[0].position_tracked is True
    assert set(frame.positions()) == {"Head"}


def test_frame_tracking_validity_keeps_its_raw_position_semantics():
    frame = _frame(_joint(position_tracked=True, orientation_tracked=True),
                   tracking_valid=False)
    assert frame.tracking_valid is False
    assert frame.confidence == 0.9
    assert frame.joints[0].position == (0.0, 1.7, 0.0)
    assert frame.joints[0].position_tracked is True
    assert frame.positions() == {}


def test_existing_positional_joint_construction_is_compatible_and_frozen():
    joint = BodyJoint(7, "Head", (0, 1.7, 0), (0, 0, 0, 1))
    assert joint.position_tracked is joint.orientation_tracked is None
    with pytest.raises(FrozenInstanceError):
        joint.position_tracked = False
