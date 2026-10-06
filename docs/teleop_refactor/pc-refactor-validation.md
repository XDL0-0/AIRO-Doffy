# PC teleoperation refactor validation

Date: 2026-09-16\
Scope: PC teleoperation entry points and robot/control runtime only.

This record describes the control/runtime source split in the current worktree.
Training, evaluation, dataset, Beaver and configuration implementations remain
outside this refactor. The parallel UDP/WebRTC implementation and camera tests
are documented in [media validation](media-validation.md).

## Result

The four historical root entry points are now compatibility facades:

| Root path | Compatibility surface | Implementation |
| --- | --- | --- |
| `realman_teleop.py` | RealMan classes, CAN-FD contracts, Quest packet sender, visualizer helpers, `main()` | `doffy_teleop/control/` and `doffy_teleop/runtime/` |
| `robot_backend.py` | `RobotBackend`, backend classes, gripper classes, `make_robot_backend`, `make_robot` | `doffy_teleop/robots/` |
| `robot_teleop.py` | `RobotTeleop`, `URTeleop`, `FastRobotiq2F85`, replay factory helpers | `doffy_teleop/robots/legacy/` |
| `main.py` | Classic data collection helpers, config aliases, `main()` CLI | `doffy_teleop/runtime/classic*.py` |

The facades keep the existing module names used by scripts, replay tools, and
tests. The root RealMan and classic entry points still run with
`python realman_teleop.py` and `python main.py` respectively. The legacy
`RobotTeleop` facade also keeps its backend factory patch point; the package
runtime accepts an explicit factory for that compatibility path.

## Module responsibilities

### RealMan control and runtime

| Module | Responsibility |
| --- | --- |
| `doffy_teleop/control/contracts.py` | Quest TCP packet encoding, immutable CAN-FD diagnostics, cached-state snapshots, and the fixed-rate Quest state sender. |
| `doffy_teleop/control/realman_qp.py` | Optional RealMan remote IK/arm-angle solver and its availability/configuration checks. |
| `doffy_teleop/control/canfd_setpoints.py` | Safe joint/TCP target storage, resolver callbacks, maintenance callbacks, velocity/acceleration limiting, hold behavior, and reached checks. |
| `doffy_teleop/control/canfd_loop.py` | The CAN-FD owner thread, resend cadence, SDK call timing, deadline diagnostics, heartbeat/error reporting, and health wait. |
| `doffy_teleop/control/input_mapping.py` | VR/hand pose conversion, configured axis and handedness mapping, reference handling, gripper/hand failure fallback, and target filtering. |
| `doffy_teleop/control/target_policy.py` | Controller and hand target policy, WRM tracking state, IK target resolution, and stale-input handling. |
| `doffy_teleop/runtime/state.py` | Sensor cache, state-push callback lifecycle, SDK worker ownership, stale-state checks, reset/close/quarantine lifecycle. |
| `doffy_teleop/runtime/realman.py` | Small RealMan composition class: configuration validation, backend startup, initial sensor validation, filters, target state, and composition of the control/state helpers. |
| `doffy_teleop/runtime/publisher.py` | Camera manager construction and the RealMan visualizer publication loop. |
| `doffy_teleop/runtime/realman_entrypoint.py` | RealMan CLI orchestration: camera, teleop, CAN-FD, state push, recorder, visualizer, shutdown, and cleanup. |

The RealMan composition retains the existing safety limits and timeout paths.
The CAN-FD loop remains the owner of its command SDK handle; sensor/state
workers and Quest/visualizer publication continue to use their existing
separate ownership paths.

### Robot adapters and classic teleop

| Module | Responsibility |
| --- | --- |
| `doffy_teleop/robots/contracts.py` | `RobotBackend` interface and `CommandResult` contract. |
| `doffy_teleop/robots/gripper.py` | Null gripper and the bounded Robotiq adapter. |
| `doffy_teleop/robots/backends.py` | UR position/torque and RealMan adapters. The current RealMan freedrive stop behavior reads and holds the measured joints before issuing the post-drag command. |
| `doffy_teleop/robots/factory.py` | Lazy vendor imports and both current backend and historical raw-UR factories. |
| `doffy_teleop/robots/legacy/state.py` | Classic robot reads, dataset vectors/snapshots, Ruckig state, references, and sensor/force capture. |
| `doffy_teleop/robots/legacy/control.py` | Gripper/reset handling, safe command sending, delta targets, standby, and controller mode. |
| `doffy_teleop/robots/legacy/hand.py` | OpenXR hand pose, hand reference, jump filtering, and hand target step. |
| `doffy_teleop/robots/legacy/step.py` | Top-level controller/hand mode selection and one-step orchestration. |
| `doffy_teleop/robots/legacy/runtime.py` | Small `RobotTeleop` constructor and mixin composition. |
| `doffy_teleop/runtime/classic_config.py` | Classic `Config`/`VisualizerConfig` instances and loop-rate constants. |
| `doffy_teleop/runtime/classic_helpers.py` | Tactile holder/reader bridge, recording frame assembly, image previews, and visualizer publication. It continues to call the existing root `udp.py` and `WebRTC_udp.py` managers. |
| `doffy_teleop/runtime/classic.py` | Classic data-collection lifecycle and shutdown orchestration. |

No module uses runtime `exec`, AST forwarding, or a copied second copy of a
3000-line entry point. The extracted methods remain ordinary Python methods;
the legacy mixins are combined through normal class inheritance.

## Public names and dependency boundaries

The following historical imports remain available:

* `realman_teleop`: `RealManTeleop`, `CanfdCommandLoop`,
  `CanfdLoopSnapshot`, `RealManStateSnapshot`, `RealManRemoteIkSolver`,
  `QuestTcpStateSender`, `pack_quest_tcp_state_packet`, recorder and
  visualizer helpers, `main`, plus the existing `Config`, `RealManBackend`,
  `VisualizerConfig`, `WrenchFilter`, and `SE3Container` names.
* `robot_backend`: all prior backend/gripper classes and both factory helpers.
* `robot_teleop`: `RobotTeleop`, `URTeleop`, `FastRobotiq2F85`, `make_robot`,
  `make_robot_backend`, and the previously imported configuration/helper
  classes.
* `main`: `TactileDataHolder`, tactile/recording/visualizer helpers,
  `CameraManager`, the rate constants, configuration aliases, manager classes,
  and `main`.

Package robot modules now import their package contracts/factories directly.
Root facades inject patchable historical factories at the compatibility boundary.
Classic recording still uses the existing `dataset` and `data_recording` modules.
The root `udp.py`/`WebRTC_udp.py` media names delegate to `doffy_teleop/media`.

## Validation evidence

Environment: `/tmp/airo-teleop-qa/bin/python`.

The no-hardware baseline and the new refactor checks were run together:

```text
/tmp/airo-teleop-qa/bin/python -m pytest \
  tests/test_realman_teleop_loop.py \
  tests/test_udp.py \
  tests/test_wrm_akm.py \
  tests/test_realsense_camera.py \
  tests/test_vr_coordinate_mapping.py \
  tests/test_teleop_refactor_imports.py \
  tests/test_teleop_refactor_boundaries.py -q

75 passed in 1.72s
```

The pre-refactor-compatible portion is 68 passing tests; the seven added
checks verify root/package imports, classic helper behavior, module size, and
explicit composition boundaries. All implementation modules listed above are
at most 500 lines; the largest is `control/canfd_loop.py` at 478 lines. The
four root facades are 29--106 lines.

All touched Python modules were also compiled with:

```text
/tmp/airo-teleop-qa/bin/python -m py_compile \
  realman_teleop.py robot_backend.py robot_teleop.py main.py \
  doffy_teleop/control/*.py doffy_teleop/runtime/*.py \
  doffy_teleop/robots/*.py doffy_teleop/robots/legacy/*.py
```

No robot, RealSense device, CAN-FD bus, BLE reader, or external media endpoint
was opened by the validation commands. The manual hardware scripts
`tests/test_realman_canfd.py` and `tests/test_vr_brainco_hand_teleop.py` were
therefore not run.

## Known limits

The new package classes are the implementation boundary, while the root
facades are retained for migration. Code that imports private implementation
globals from the old monolith should migrate to the corresponding package
module; supported classes/functions and CLI paths remain available at their
old paths. Later acceptance fixed live `main.cfg` replacement and injected the
historical RealMan CLI dependencies explicitly; behavioral tests cover both.
See [independent acceptance](acceptance-review.md) for the full suite, the
pre-existing policy configuration failure, and the entry-point regression results.
