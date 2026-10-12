# PC teleoperation refactor acceptance review

## Scope and evidence

This audit performed a read-only inspection of
`doffy_teleop/control`, `doffy_teleop/runtime`, and `doffy_teleop/robots` in the current
`/path/to/AIRO-Doffy`, using the snapshot extracted from
`/path/to/unity/.backups/pc-teleop-baseline-20260916.tar.gz`
as the sole implementation baseline. The comparison files were the snapshot's
`realman_teleop.py`, `robot_teleop.py`, `main.py`, `robot_backend.py`, and
`eval_config.py`; datasets and weights were not scanned, and no robot was connected or operated.

## Results

- The RealMan runtime combines `doffy_teleop/runtime/realman.py` with
  `runtime/state.py`, `control/input_mapping.py`, and `control/target_policy.py`.
  Its AST method set exactly matches the snapshot's `RealManTeleop` (41 methods).
- `control/canfd_loop.py` + `control/canfd_setpoints.py` exactly match the snapshot's
  `CanfdCommandLoop` (24 methods); the methods of
  `RealManRemoteIkSolver` in `control/realman_qp.py` also match exactly.
- `RobotTeleop` assembled from `robots/legacy/*` exactly matches the snapshot (51 methods).
  The method implementations of `RobotBackend`, `PositionManipulatorBackend`, `URPositionBackend`,
  `RealManBackend`, `URTorqueBackend`, `FastRobotiq2F85`, and `NullGripper`
  match the snapshot.
- The top-level `make_robot_backend` and `make_robot` function bodies match the snapshot. The two
  current runtime constructors add only optional `backend`/`backend_factory` injection arguments
  for hardware-free testing and root entry-point patching; the original default factories
  and configuration defaults remain unchanged.
- Root entry points `realman_teleop.py`, `robot_teleop.py`, `robot_backend.py`,
  and `main.py` all retain public exports of historical classes, factories, and recording/visualization
  helpers; module imports, `tests/test_teleop_refactor_imports.py`,
  and boundary checks passed.
- Two root entry-point compatibility issues were fixed after review: the reset helper in `main.py`
  reads the current root module configuration through an explicit `cfg` argument;
  `realman_teleop.main()` explicitly injects the snapshot's replaceable
  `Config`, `VisualizerConfig`, `_create_camera_manager`, `RealManTeleop`,
  `RealManEpisodeRecorder`, `QuestTcpStateSender`, and
  `visualizer_publish_loop` into `realman_entrypoint.main()`.

## Lifecycle, threads, and safety gates

- `StateLifecycleMixin` uses `_accept_state_callbacks`, a condition variable, and
  `_state_callbacks_in_flight` to reject late callbacks during shutdown, and waits for
  callbacks that have already entered to exit according to `REALMAN_STATE_PUSH_TIMEOUT`.
- `close()` explicitly refuses to close while the SDK worker is still alive; the normal path first stops realtime push,
  then calls backend cleanup. When the worker cannot be safely stopped, the entry point calls
  `quarantine_without_sdk_cleanup()` to retain objects and callback memory, avoiding invalidating the SDK
  during in-flight calls. A backend created by the factory during construction is cleaned up if subsequent
  initialization fails; an explicitly injected backend is owned by the caller, consistent with injection semantics.
- `CanfdCommandLoop`'s `send_once()`, continuous IK resolver, and sensor maintenance callbacks
  for the polling fallback all run on the same CAN-FD owner thread; the VR processing thread only publishes
  setpoints/requests and does not call the SDK directly. Before cleanup, the entry point gathers and waits for
  the teleop SDK worker and other background workers.
- `process_controller()` and `process_hand()` call
  `mark_input_stale()` when sensor data is stale; this function retains the current setpoint,
  clears grip activity, and requires a new reference.
  `_resolve_joint_target()` rechecks stale/grip state under the control lock,
  so pending IK results cannot drive the arm again after input becomes stale or the trigger is released. The reset,
  Start/Stop/Undo state-machine methods and default thresholds all match the snapshot.

## Tests

This round used `/tmp/airo-teleop-qa/bin/python` to run:

```text
PYTHONWARNINGS=ignore /tmp/airo-teleop-qa/bin/python -m pytest -q
698 passed, 95 subtests passed, 1 failed in 75.56s (final integration rerun)

PYTHONWARNINGS=ignore /tmp/airo-teleop-qa/bin/python -m pytest -q \
  tests/test_teleop_refactor_imports.py tests/test_teleop_refactor_boundaries.py
10 passed in 1.72s
```

The sole failure was
`tests/test_eval_policy_latency.py::PolicyCompatibilityTest.test_dp_contact_no_vision_registration_and_resolution`: the current assertion requires `ACTION_STEPS` to be 8 for all four aliases, while `ICRA_DP_contact_no_vision` is actually 4.
This value was not introduced by this refactor: the snapshot's `eval_config.py:439` explicitly contains
`"ICRA_DP_contact_no_vision": 4`, and the other three aliases in the snapshot at lines 440--442 are all
8; the current `eval_config.py` retains exactly the same four values. This failure should therefore be recorded
as an existing eval configuration/test mismatch and cannot be attributed to the PC refactor.
The other 698 tests and 95 subtests passed.

## Compatibility fix evidence

In the snapshot's `main.py`, `controller_reset_requested()` reads the module's `cfg` directly.
The current root entry point calls the shared helper with an explicit argument, so directly replacing `main.cfg` remains effective:

```python
from types import SimpleNamespace
import main

main.cfg = SimpleNamespace(CONTROLLER_RESET_TRIGGER_THRESHOLD=1.0)
data = [None, {"Joystick_Press": True, "IndexTrigger": 0.9}]
assert main.controller_reset_requested(data) is False
```

`realman_entrypoint.main()` now accepts named factory arguments; on each call, `realman_teleop.main()`
reads the historical replacement points from the root module and passes them in,
so old dependency replacement does not require dynamic execution or global synchronization.
New hardware-free regression tests cover both behaviors.

Apart from the entry-point semantics fixed above, this comparison found no missing methods, default-value drift,
stale gate failures, SDK worker concurrency ownership errors, or new cleanup omissions relative to the snapshot.
