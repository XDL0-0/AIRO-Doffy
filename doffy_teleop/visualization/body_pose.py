"""Keep display poses steady while preserving the received tracking diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, replace

from doffy_teleop.protocol.body import BodyFrame, BodyJoint


@dataclass(frozen=True)
class HeldBodyPose:
    frame: BodyFrame | None
    held_positions: frozenset[str] = frozenset()
    held_rotations: frozenset[str] = frozenset()
    reason: str = ""
    last_valid_received_ns: int | None = None


class BodyPoseHold:
    """Display-only cache; never changes or forwards the incoming BODY frame.

    Confidence follows Meta's high-confidence cutoff. Tracked flags are optional
    for older senders. Inferred joints without a tracked history stay visible;
    loss of a previously tracked arm anchor holds that arm as a unit.
    """

    def __init__(self, stale_after: float = 0.5):
        self.stale_after = stale_after
        self._source = ""
        self._joint_set: str | None = None
        self._clear()

    def _clear(self):
        self._joints: dict[str, BodyJoint] = {}
        self._pose: BodyFrame | None = None
        self._position_tracked: set[str] = set()
        self._orientation_tracked: set[str] = set()
        self._last_valid_received_ns: int | None = None

    @staticmethod
    def _arm_joint(name: str, side: str) -> bool:
        return name.startswith(side) and name[len(side):].startswith(
            ("Shoulder", "Scapula", "Arm", "Hand")
        )

    def _hold_all(self, reason: str) -> HeldBodyPose:
        return HeldBodyPose(
            self._pose,
            frozenset(name for name, joint in self._joints.items() if joint.position is not None),
            frozenset(name for name, joint in self._joints.items() if joint.rotation is not None),
            reason, self._last_valid_received_ns,
        )

    def update(self, frame: BodyFrame | None, *, now_ns: int, source: str = "") -> HeldBodyPose:
        if source and self._source and source != self._source:
            self._clear()
            self._joint_set = None
        if source:
            self._source = source
        if frame is not None and frame.joint_set != self._joint_set:
            self._clear()
            self._joint_set = frame.joint_set
        if frame is None:
            return self._hold_all("stale" if self._pose else "")
        if (now_ns - frame.received_ns) / 1e9 > self.stale_after:
            return self._hold_all("stale")
        if not frame.tracking_valid:
            return self._hold_all("invalid")
        if frame.confidence <= 0.5:
            return self._hold_all("low confidence")
        if not any(joint.position is not None for joint in frame.joints):
            return self._hold_all("no positions")

        current = {joint.name: joint for joint in frame.joints}
        held_arms = set()
        for side in ("Left", "Right"):
            for suffix in ("ArmUpper", "ArmLower", "HandWrist"):
                name = side + suffix
                previous = self._joints.get(name)
                joint = current.get(name)
                if previous is not None and previous.position is not None and (
                    joint is None or joint.position is None or
                    (name in self._position_tracked and joint.position_tracked is False)
                ):
                    held_arms.add(side)
                    break

        held_positions, held_rotations = set(), set()
        updated_positions = False
        for name in self._joints.keys() | current.keys():
            previous, joint = self._joints.get(name), current.get(name)
            hold_arm = any(self._arm_joint(name, side) for side in held_arms)
            use_position = joint is not None and joint.position is not None and not hold_arm and not (
                joint.position_tracked is False and name in self._position_tracked
            )
            use_rotation = joint is not None and joint.rotation is not None and use_position and not (
                joint.orientation_tracked is False and name in self._orientation_tracked
            )
            position = joint.position if use_position else previous.position if previous else None
            rotation = joint.rotation if use_rotation else previous.rotation if previous else None
            if position is None:
                continue
            if not use_position:
                held_positions.add(name)
            else:
                updated_positions = True
                if joint.position_tracked is True:
                    self._position_tracked.add(name)
            if not use_rotation and rotation is not None:
                held_rotations.add(name)
            elif use_rotation and joint.orientation_tracked is True:
                self._orientation_tracked.add(name)
            basis = joint if joint is not None else previous
            self._joints[name] = replace(basis, position=position, rotation=rotation)

        # Fresh torso packets must not make a frozen arm's pose age look fresh.
        # Record the last display pose with no held positions or rotations.
        if updated_positions and not held_positions and not held_rotations:
            self._last_valid_received_ns = frame.received_ns
        # This frame belongs only to the renderer. Raw validity and confidence
        # remain available on the original received frame in the dashboard.
        self._pose = replace(frame, tracking_valid=True,
                             joints=tuple(sorted(self._joints.values(), key=lambda joint: joint.id)))
        return HeldBodyPose(self._pose, frozenset(held_positions), frozenset(held_rotations),
                            "partial" if held_positions or held_rotations else "",
                            self._last_valid_received_ns)
