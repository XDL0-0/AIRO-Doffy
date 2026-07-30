# Phase 9 Teleoperation and Safety Baseline

日期：2026-07-30

## 1. Current implementation boundary

Root `robot_teleop.py::RobotTeleop` currently combines:

- controller and hand parsing assumptions;
- coordinate conversion and relative pose mapping;
- mutable controller/robot references;
- fine mode and tracking-mode gestures;
- gripper integration and direct gripper calls;
- inverse kinematics and backend-specific safety queries;
- joint smoothing/Ruckig;
- robot commands and state reads;
- wrench capture, reset and dataset snapshots.

Root `realman_teleop.py` duplicates controller pose conversion, reference handling,
fine scaling, rotation mapping, IK validation, smoothing and stale-input handling
inside its RealMan-specific runtime.

The v2 package currently has only `TeleopMapping` and `ActionFilter` Protocol
skeletons. No production package mapping or safety implementation is selected by
the root runtime.

## 2. Controller pose mapping

The general UR path converts the right controller with hard-coded rules:

```text
position [x,y,z] -> [-x,-z,y]
quaternion [x,y,z,w] -> [x,z,-y,w]
```

These equal the configured UR `VR_TO_ROBOT_AXES` position mapping plus the
determinant sign required when transforming the quaternion vector through an
improper axis transform.

The RealMan path uses configured axes:

```text
[[ 0, 0, 1],
 [-1, 0, 0],
 [ 0, 1, 0]]
```

and likewise multiplies the quaternion vector by the transform determinant.

Both paths compute controller-relative translation and rotation, low-pass the
deltas, scale fine mode translation by `0.3` and rotation by `0.4`, and add the
delta to a measured robot reference.

There is one material rotation-composition difference:

- general `RobotTeleop`: `controller_delta @ robot_reference`;
- RealMan-specific teleop: `robot_reference @ controller_delta`.

The new mapping must make this convention explicit rather than silently choosing
one while migrating both old paths.

## 3. Engagement and references

General controller mode:

- right `GripTrigger` false means standby;
- standby refreshes controller and measured TCP references every step;
- a fine-mode edge refreshes both references;
- reset completion also refreshes robot reference;
- while grip is active, each loop derives and sends a target.

RealMan controller mode:

- startup, reconnect/stale recovery and fine-mode edges require a new reference;
- releasing grip requests hold and refreshes the reference;
- recovery consumes one valid frame to set a reference before movement resumes;
- active grip submits latest TCP or joint targets to the CAN-FD owner loop.

## 4. Hand mapping and gestures

The general path expects a right hand with exactly 26 OpenXR joints:

- wrist pose is preferred; otherwise joint 0 with identity rotation is used;
- a wrist/palm jump greater than `0.15 m` discards the frame and invalidates the
  reference;
- first valid hand frame captures hand and robot references without motion;
- thumb tip index 5 and index tip index 10 control the gripper;
- distance greater than `0.06 m` opens;
- distance less than `0.03 m` closes;
- the interval between thresholds holds.

Thumb-to-pinky below `0.02 m` toggles controller/hand mode with a 1 second debounce.
Thumb-to-ring below `0.02 m` resets the robot/gripper with a 2 second debounce.
Those gestures cause runtime commands and are not part of pure pose mapping.

## 5. Gripper mapping

Controller mode uses negative right-joystick Y:

```text
> +0.7 -> open
< -0.7 -> close
otherwise hold
```

The target changes at `0.1 m/s`, is clamped to `[0, 0.085] m`, and is reconciled
with measured width. The old object directly calls private gripper target methods.

The v2 mapping must only produce `RobotAction.gripper_width_m`; hardware calls
remain backend/executor responsibility.

## 6. Command mode and IK

Configured command mode is `joint` or `tcp`.

- Joint mode maps controller delta to TCP, runs backend IK, validates the result,
  smooths/rate-limits it, then sends a joint target.
- TCP mode sends the TCP target through the backend.
- Missing IK solution keeps the previous action.
- The generic UR path has an additional joystick-X joint-5 bias when rotation is
  not frozen.

The v2 package must express command selection through an injected IK boundary.
Mapping must not import a robot backend or SDK.

## 7. Current safety behavior

General UR safety is distributed across `utils.py`, backend facades and teleop:

- every joint is limited to `[-2π, 2π]`;
- shortest angular step is compared with `MOVE_THRESHOLD`;
- wrist, elbow and shoulder singularity heuristics;
- analytic self-collision approximation;
- low TCP near-base rejection (`z < 0.05 m` and XY radius `< 0.15 m`);
- Ruckig velocity/acceleration/jerk when installed;
- fallback EMA plus maximum joint step `0.02 rad`;
- invalid IK or unsafe result keeps the previous joint/TCP target.

RealMan adds:

- backend/controller joint limits when exposed;
- filtered joint target revalidation;
- configured joint/linear/angular slew limits in the CAN-FD executor;
- strict CAN-FD timing and heartbeat supervision.

There is no composable package filter for workspace, joint limits, velocity,
acceleration, freshness, IK or action rate.

## 8. Stale input and hold

RealMan has a `0.25 s` VR timeout and robot-state freshness check. A stale event:

1. requests CAN-FD hold;
2. marks input stale;
3. requires a fresh controller reference;
4. clears active grip;
5. prevents in-flight IK from publishing a new target.

Recovery requires a fresh robot state and a fresh VR frame; the first recovery
frame captures references without moving.

The general UR path reuses the most recently received controller state at 60 Hz
and has no equivalent stale VR watchdog. Velocity/twist commands therefore do not
have a shared package rule that explicitly emits zero or hold on stale input.

## 9. Compatibility boundary

Phase 9 package components are opt-in. They must not silently change:

- root `RobotTeleop` and `RealManTeleop` selection;
- controller axis transforms;
- fine scales `0.3/0.4`;
- grip release reference behavior;
- hand joint indices and thresholds;
- legacy gesture debounce/reset semantics;
- UR joint-5 bias;
- RealMan CAN-FD cadence, slew limiting or timeout;
- dataset state/action representations.

Root replacement belongs to the later composition phase after hardware validation.

## 10. Required new policies and tests

1. Dependency-free quaternion, axis, pose-delta and composition golden tests.
2. Explicit left/right rotation composition.
3. Controller reference, grip, fine scale and gripper tests.
4. Hand wrist/fallback, jump rejection and gripper threshold tests.
5. TCP/joint command selection with injected IK and failed IK rejection.
6. Workspace and joint-limit rejection.
7. Joint/TCP velocity and acceleration limiting.
8. Monotonic action freshness and incomparable-clock rejection.
9. Deterministic rate limiting.
10. Filter-chain ordering and rejection metrics.
11. VR and robot stale watchdog transitions.
12. Velocity/twist zeroing or hold on watchdog trip.
13. Explicit recovery requiring fresh samples plus reference acknowledgement.
14. Package imports without NumPy, SciPy, OpenCV, Ruckig or robot SDKs.

## 11. Hardware validation

- Unity controller/hand axes and quaternion conventions for both UR and RealMan;
- UR left-composed versus RealMan right-composed rotation behavior;
- controller grip/fine transition without target jumps;
- hand wrist fallback and palm-jump recovery;
- IK solver results near wrap, singularity and joint limits;
- real workspace limits for each cell;
- gripper speed and measured-width reconciliation;
- stale VR, stale robot state, packet pause and reconnect;
- velocity/twist stop and position hold latency;
- Ruckig/package-filter interaction without double limiting;
- RealMan CAN-FD timing after package composition.
