"""Loss and reacquisition sequences for the display-only BODY pose cache."""

from dataclasses import replace

import pytest

from doffy_teleop.protocol.body import BodyFrame, BodyJoint, JOINT_IDS
from doffy_teleop.visualization.body_pose import BodyPoseHold


def joint(name, position=(0, 1, 0), tracked=None, rotation=(0, 0, 0, 1)):
    return BodyJoint(JOINT_IDS[name], name, position, rotation, tracked, tracked)


def frame(*joints, received=1_000_000_000, confidence=0.9, valid=True, kind="upper_body"):
    return BodyFrame(1, received, received, kind, confidence, valid,
                     joints or (joint("Head"),), {})


def positions(result):
    return {item.name: item.position for item in result.frame.joints} if result.frame else {}


@pytest.mark.parametrize("changes, reason", [
    ({"tracking_valid": False}, "invalid"),
    ({"confidence": 0.0}, "low confidence"),
    ({"confidence": 0.5}, "low confidence"),
    ({"joints": (joint("Head", None),)}, "no positions"),
])
def test_invalid_fallback_never_overwrites_good_pose(changes, reason):
    hold = BodyPoseHold()
    good = frame(joint("Head", (0.3, 1.8, 0.4)))
    hold.update(good, now_ns=good.received_ns)
    fallback = replace(frame(joint("Head", (0, 0, 0)), received=1_100_000_000), **changes)
    for now in (fallback.received_ns, fallback.received_ns + 100_000_000):
        result = hold.update(fallback, now_ns=now)
        assert result.reason == reason
        assert positions(result)["Head"] == (0.3, 1.8, 0.4)
        assert result.held_positions == {"Head"}
        assert result.last_valid_received_ns == good.received_ns
    assert fallback.tracking_valid == changes.get("tracking_valid", True)
    assert fallback.joints[0].position == changes.get("joints", fallback.joints)[0].position


def test_stale_packet_with_new_coordinates_cannot_replace_pose():
    hold = BodyPoseHold(stale_after=0.5)
    good = frame(joint("Head", (0, 1.8, 0)))
    hold.update(good, now_ns=good.received_ns)
    fallback = frame(joint("Head", (0, 0, 0)), received=1_100_000_000)
    result = hold.update(fallback, now_ns=2_000_000_000)
    assert result.reason == "stale"
    assert positions(result)["Head"] == (0, 1.8, 0)
    assert hold.update(None, now_ns=2_100_000_000).frame is result.frame


def test_no_good_baseline_stays_blank():
    hold = BodyPoseHold()
    assert hold.update(None, now_ns=0).frame is None
    invalid = frame(valid=False)
    assert hold.update(invalid, now_ns=invalid.received_ns).frame is None
    low = frame(confidence=0.5)
    assert hold.update(low, now_ns=low.received_ns).frame is None


def test_missing_joint_keeps_cached_position_while_other_joints_update():
    hold = BodyPoseHold()
    good = frame(joint("Hips", (0, 1, 0)), joint("Head", (0, 1.8, 0)))
    hold.update(good, now_ns=good.received_ns)
    partial = frame(joint("Hips", (0.2, 1, 0)), received=1_100_000_000)
    result = hold.update(partial, now_ns=partial.received_ns)
    assert positions(result) == {"Hips": (0.2, 1, 0), "Head": (0, 1.8, 0)}
    assert result.held_positions == {"Head"}
    assert result.reason == "partial"
    assert result.last_valid_received_ns == good.received_ns


@pytest.mark.parametrize("lost_anchor", ["LeftArmUpper", "LeftArmLower", "LeftHandWrist"])
def test_tracked_arm_loss_holds_whole_arm_and_recovery_resumes(lost_anchor):
    hold = BodyPoseHold()
    names = ("LeftShoulder", "LeftArmUpper", "LeftArmLower", "LeftHandWrist", "LeftHandPalm")
    good = frame(*(joint(name, (-0.2, 1.5 - index * 0.1, 0.3), True)
                   for index, name in enumerate(names)), joint("Head", (0, 1.8, 0), True))
    hold.update(good, now_ns=good.received_ns)
    fallback = frame(*(joint(name, (-0.2, 0.1, 0), name != lost_anchor) for name in names),
                     joint("Head", (0.1, 1.8, 0), True), received=1_100_000_000)
    result = hold.update(fallback, now_ns=fallback.received_ns)
    for name in names:
        assert positions(result)[name] == next(item.position for item in good.joints if item.name == name)
    assert result.held_positions == set(names)
    assert positions(result)["Head"] == (0.1, 1.8, 0)
    # The high-confidence fallback still contains the received valid coordinates.
    assert fallback.confidence == 0.9 and fallback.tracking_valid
    assert fallback.positions()[lost_anchor][1] == 0.1
    restored = frame(*(joint(name, (-0.4, 1.6, 0.2), True) for name in names),
                     joint("Head", (0.1, 1.8, 0), True), received=1_200_000_000)
    result = hold.update(restored, now_ns=restored.received_ns)
    assert not result.reason and not result.held_positions
    assert positions(result)[lost_anchor] == (-0.4, 1.6, 0.2)


def test_invalid_wrist_holds_arm_even_for_legacy_packets():
    hold = BodyPoseHold()
    good = frame(joint("LeftArmLower", (0, 1.5, 0)), joint("LeftHandWrist", (0, 1.2, 0)))
    hold.update(good, now_ns=good.received_ns)
    lost = frame(joint("LeftArmLower", (0, 0.4, 0)), joint("LeftHandWrist", None),
                 received=1_100_000_000)
    result = hold.update(lost, now_ns=lost.received_ns)
    assert positions(result)["LeftArmLower"] == (0, 1.5, 0)
    assert result.held_positions == {"LeftArmLower", "LeftHandWrist"}


def test_inferred_legs_without_a_tracked_history_stay_live():
    hold = BodyPoseHold()
    first = frame(joint("LeftLowerLeg", (0, 0.5, 0), False), kind="full_body")
    hold.update(first, now_ns=first.received_ns)
    second = frame(joint("LeftLowerLeg", (0.1, 0.6, 0), False), kind="full_body",
                   received=1_100_000_000)
    result = hold.update(second, now_ns=second.received_ns)
    assert positions(result)["LeftLowerLeg"] == (0.1, 0.6, 0)
    assert not result.held_positions


def test_orientation_loss_keeps_rotation_without_stopping_valid_position():
    hold = BodyPoseHold()
    good = frame(joint("Head", tracked=True))
    hold.update(good, now_ns=good.received_ns)
    changed = replace(joint("Head", (0.2, 1.8, 0), True, (0, 0, 1, 0)), orientation_tracked=False)
    received = frame(changed, received=1_100_000_000)
    result = hold.update(received, now_ns=received.received_ns)
    assert result.frame.joints[0].position == (0.2, 1.8, 0)
    assert result.frame.joints[0].rotation == (0, 0, 0, 1)
    assert result.held_rotations == {"Head"}
    assert not result.held_positions


@pytest.mark.parametrize("switch", ["source", "joint_set"])
def test_new_sender_or_joint_set_does_not_inherit_old_pose(switch):
    hold = BodyPoseHold()
    good = frame(joint("Head"), joint("LeftLowerLeg", (0, 0.5, 0)), kind="full_body")
    hold.update(good, now_ns=good.received_ns, source="quest-one:1000")
    lost = frame(valid=False, kind="upper_body" if switch == "joint_set" else "full_body",
                 received=1_100_000_000)
    result = hold.update(lost, now_ns=lost.received_ns,
                         source="quest-two:1000" if switch == "source" else "quest-one:1000")
    assert result.frame is None
    restored = replace(lost, tracking_valid=True)
    result = hold.update(restored, now_ns=restored.received_ns)
    assert positions(result) == {"Head": (0, 1, 0)}
