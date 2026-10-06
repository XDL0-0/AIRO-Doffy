# PC teleoperation refactor acceptance review

## Scope and evidence

本审计只读检查了当前 `/home/yuyuan/AIRO-Doffy` 中的
`doffy_teleop/control`、`doffy_teleop/runtime`、`doffy_teleop/robots`，并以
`/home/yuyuan/UNITY_Project/.backups/pc-teleop-baseline-20260916.tar.gz`
解出的快照作为唯一实现基线。对照文件为快照的
`realman_teleop.py`、`robot_teleop.py`、`main.py`、`robot_backend.py` 和
`eval_config.py`；没有扫描数据集、权重，也没有连接或操作机器人。

## 结果

- RealMan 运行时由 `doffy_teleop/runtime/realman.py` 与
  `runtime/state.py`、`control/input_mapping.py`、`control/target_policy.py`
  组合。AST 方法集合与快照的 `RealManTeleop` 完全一致（41 个方法）。
- `control/canfd_loop.py` + `control/canfd_setpoints.py` 与快照
  `CanfdCommandLoop` 完全一致（24 个方法）；`control/realman_qp.py` 的
  `RealManRemoteIkSolver` 方法也完全一致。
- `robots/legacy/*` 组合后的 `RobotTeleop` 与快照完全一致（51 个方法）。
  `RobotBackend`、`PositionManipulatorBackend`、`URPositionBackend`、
  `RealManBackend`、`URTorqueBackend`、`FastRobotiq2F85` 和 `NullGripper`
  的方法实现与快照一致。
- 顶层 `make_robot_backend`、`make_robot` 的函数体与快照一致。当前两个
  运行时构造器只增加可选的 `backend`/`backend_factory` 注入参数，以便无硬件
  测试和根入口补丁；原有默认工厂和配置默认值保持不变。
- 根入口 `realman_teleop.py`、`robot_teleop.py`、`robot_backend.py`、
  `main.py` 均保留历史类、工厂、记录/可视化辅助函数的公开导出；模块导入和
  `tests/test_teleop_refactor_imports.py`、边界检查通过。
- 复核后已修复两处根入口兼容性：`main.py` 的 reset helper 通过显式 `cfg`
  参数读取当前根模块配置；`realman_teleop.main()` 将快照中的可替换
  `Config`、`VisualizerConfig`、`_create_camera_manager`、`RealManTeleop`、
  `RealManEpisodeRecorder`、`QuestTcpStateSender` 和
  `visualizer_publish_loop` 显式注入 `realman_entrypoint.main()`。

## 生命周期、线程和安全门

- `StateLifecycleMixin` 用 `_accept_state_callbacks`、条件变量和
  `_state_callbacks_in_flight` 拒绝关闭期间的迟到回调，并按
  `REALMAN_STATE_PUSH_TIMEOUT` 等待已进入的回调退出。
- `close()` 在 SDK worker 仍存活时明确拒绝关闭；正常路径先停 realtime push，
  再调用 backend cleanup。无法安全停掉 worker 时，入口调用
  `quarantine_without_sdk_cleanup()` 保留对象和回调内存，避免使 SDK 在飞行
  调用中失效。构造期间由工厂创建的 backend 在后续初始化失败时清理；显式注入
  的 backend 由调用者拥有，这与注入语义一致。
- `CanfdCommandLoop` 的 `send_once()`、连续 IK resolver 和 polling fallback 的
  传感器维护回调都在同一个 CAN-FD owner thread 上运行；VR 处理线程只发布
  setpoint/request，不直接调用 SDK。入口在清理前合并并等待 teleop SDK worker
  和其他后台 worker。
- `process_controller()`、`process_hand()` 在传感器过期时调用
  `mark_input_stale()`；该函数保持当前 setpoint、清除 grip 活动并要求新的
  reference。`_resolve_joint_target()` 在 control lock 下再次检查 stale/grip
  状态，因此过期或释放触发器后的挂起 IK 结果不会重新驱动机械臂。reset、
  Start/Stop/Undo 相关状态机方法和默认阈值均与快照一致。

## 测试

本轮使用 `/tmp/airo-teleop-qa/bin/python` 执行：

```text
PYTHONWARNINGS=ignore /tmp/airo-teleop-qa/bin/python -m pytest -q
698 passed, 95 subtests passed, 1 failed in 75.56s（最终整合复跑）

PYTHONWARNINGS=ignore /tmp/airo-teleop-qa/bin/python -m pytest -q \
  tests/test_teleop_refactor_imports.py tests/test_teleop_refactor_boundaries.py
10 passed in 1.72s
```

唯一失败为
`tests/test_eval_policy_latency.py::PolicyCompatibilityTest.test_dp_contact_no_vision_registration_and_resolution`，当前断言要求四个别名的 `ACTION_STEPS` 都为 8，而实际 `ICRA_DP_contact_no_vision` 为 4。
该值不是本轮重构引入：快照 `eval_config.py:439` 已明确是
`"ICRA_DP_contact_no_vision": 4`，快照中的其他三个别名在 440--442 行均为
8；当前 `eval_config.py` 保持完全相同的四项值。因此该失败应作为既有
eval 配置/测试不一致记录，不能归因于 PC 重构。其余 698 项及 95 个子测试通过。

## 兼容性修复证据

快照 `main.py` 中 `controller_reset_requested()` 直接读取该模块的 `cfg`。
当前根入口通过显式参数调用共享 helper，因此直接替换 `main.cfg` 会继续生效：

```python
from types import SimpleNamespace
import main

main.cfg = SimpleNamespace(CONTROLLER_RESET_TRIGGER_THRESHOLD=1.0)
data = [None, {"Joystick_Press": True, "IndexTrigger": 0.9}]
assert main.controller_reset_requested(data) is False
```

`realman_entrypoint.main()` 现在接受命名工厂参数；`realman_teleop.main()` 每次
调用时从根模块读取并传入历史替换点，所以旧的依赖替换不需要动态执行或全局
同步。新增的无硬件回归测试覆盖了这两条行为。

除上述已修复入口语义外，本次对照没有发现缺失方法、默认值漂移、stale gate
失效、SDK worker 并发所有权错误或相对快照新增的 cleanup 遗漏。
