# Phase 3 重构报告：机器人后端、执行器与夹爪

日期：2026-07-30

分支：`codex/v2-refactor`

前置阶段：配置重构（`462663b`）

## 1. 阶段结论

Phase 3 的机器人基础层已经完成。UR、RealMan、mock、通用命令调度、RealMan
high-follow 调度和 Robotiq 夹爪现在各有独立模块，并共同使用 Phase 2 的
`RobotBackend`、`RobotState` 和 `RobotAction` 契约。新机器人包可以在没有 NumPy、
OpenCV、SciPy、`airo_robots`、`ur_analytic_ik` 或 Ruckig 的环境中安全导入。

连接设备、创建 socket 和加载 SDK 都延迟到显式 lifecycle。组合函数只选择未启动的
adapter，不会连接、移动机器人或打开夹爪。真实设备验证没有在本阶段执行。

根 `robot_backend.py` 继续服务旧 `main.py`、`robot_teleop.py`、`realman_teleop.py`
和 `inference.py`，但已缩减为延迟兼容 facade。宽旧实现集中在
`airo_doffy.robots.legacy`，不会被新机器人包导入。旧运行入口尚未切换到新 backend，
因此本阶段没有暗中改变现有实机启动和运动行为。

## 2. 后端拆分

| 模块 | 责任 | 运行依赖 |
|---|---|---|
| `robots/base.py` | 共同 `RobotBackend` 原子端口 | 标准库与 core types |
| `robots/mock.py` | 可配置状态、命令捕获、延迟和故障注入 | 标准库 |
| `robots/ur.py` | UR3e/UR5e position/torque RTDE adapter | 仅 `start()`/动作时延迟加载 UR extras |
| `robots/realman.py` | RM75 状态读取和单包 CAN-FD 翻译 | 仅 `start()` 时延迟加载 RealMan extras |
| `robots/executor.py` | 通用 latest-action cadence | 标准库 |
| `robots/realman_executor.py` | RealMan high-follow、限速、时序门与 heartbeat | 标准库 |
| `robots/grippers/base.py` | 独立 `Gripper` 端口 | 标准库 |
| `robots/grippers/mock.py` | disabled/null gripper | 标准库 |
| `robots/grippers/robotiq_2f85.py` | Robotiq URCap 持久 TCP adapter | 标准库 socket |
| `robots/factory.py` | 未启动 backend/gripper 的组合路由 | 上述模块 |

UR 与 RealMan 都返回同一个不可变 `RobotState`：

- UR 固定 6 个 radians joint；RealMan 固定 7 个。
- TCP pose 固定为 `4 x 4` matrix。
- wrench 有能力时固定为 `Fx,Fy,Fz,Tx,Ty,Tz`，否则为 `None`。
- 每次状态读取带单调时钟 source timestamp 和递增 sequence。
- gripper width 不由 arm backend 伪装提供；gripper 是独立组件。

## 3. UR adapter

`URRobotBackend` 支持：

- position RTDE 的 joint position、TCP pose；
- vendor 实现存在时的 joint velocity、TCP twist；
- hold、controlled stop；
- torque RTDE 的共享 joint target；
- torque TCP action 经延迟 analytic IK 或注入 IK 转成 joint target；
- cached torque state/wrench；
- 关闭 torque control 和幂等资源释放。

构造 backend 不解析 SDK，也不连接 IP。未配置 `robot.ip` 时，只有 `start()` 会失败。
新 adapter 不自动移动到初始关节，也不自动创建/打开 Robotiq；这些动作必须由显式 session
和 gripper lifecycle 决定。

UR 实机仍需 `H-UR` 验证：当前安装版本的 RTDE velocity 方法、analytic IK API、RTDE
close/disconnect、tool TCP frame、force frame、UR3e/UR5e config 常量和低速安全运动。

## 4. RealMan adapter 与 high-follow executor

`RealManRobotBackend` 保留了旧实现的关键边界：

- joint CAN-FD 从 radians 转 degrees；
- `follow=True`、`expand=0`、trajectory mode 和 radio 原样传入；
- TCP `4 x 4` pose 用纯标准库转换为 RealMan `x,y,z,rx,ry,rz`；
- SDK `error code -2` 按配置做有界读取重试；
- wrench 优先选择 `zero_force_data`，再依次回退其他 6D key；
- 非零 SDK code 转为明确失败；
- velocity mode 未配置时显式拒绝，而不是猜测 vendor 调用。

`RealManCanfdExecutor` 独立承担 cadence 和安全时序：

- 单槽 latest target，拒绝 stale/duplicate sequence；
- 以配置频率持续重发安全 setpoint；
- joint 每包按最大 rad/s 限步，不跨 `±π` 偷走捷径；
- TCP translation 和 rotation 分别按线速度/角速度限步；
- hold 冻结当前已提交 setpoint，同时继续发送 high-follow；
- stop 只发送一次并成为 terminal state；
- 启动窗口必须实测严格 `>100 Hz`；
- ready 后单次 command gap 或 SDK call 超过 10 ms 即失败；
- 记录 achieved Hz、gap、SDK overrun、last success 和 heartbeat；
- `run()` 阻塞，由调用方显式拥有 SDK thread；活跃/in-flight 时拒绝 close。

Windows 的 `Event.wait(5 ms)` 在本环境会周期性产生约 15 ms 唤醒延迟。执行器因此用
高分辨率 `time.sleep()` 做 cadence，而 stop 的最坏响应仍小于一个目标周期。200 Hz
mock timing gate 连续执行 10 次均通过；12 ms 人工 adapter 延迟会按设计失败。

RealMan realtime UDP state-push callback 仍在旧 `realman_teleop.py`。把 callback/state
buffer 接入新 backend 是后续 session/state acquisition 迁移，不在本次原子 adapter
提交中伪装为已经完成。真实 RM75 仍必须执行 `H-RM`。

## 5. 通用 executor 与 mock

`LatestActionExecutor` 支持 joint position、TCP pose、可选 velocity、hold 和 stop：

- 使用 `LatestValueBuffer`，只有最新且 sequence 更新的动作被接受；
- active position/velocity target 可按 cadence 重复；
- hold/stop 不被无意义重复；
- stop 后拒绝恢复运动；
- backend 异常进入 health snapshot 并停止调度；
- 不创建隐藏 thread，`run()` 的 owner 由 runtime 管理；
- 运行中或 SDK call in-flight 时拒绝关闭 backend。

`MockRobotBackend` 支持 6/7 DoF、初始/动态状态、位置和带 duration 的 joint velocity
模拟、gripper width、捕获全部动作、统一人工延迟、逐操作排队故障、hold/stop 和幂等
close，可用于后续全链路 mock session。

## 6. 独立夹爪

`Gripper` 端口只暴露 lifecycle、width read 和 width command。

- `NullGripper` 处理禁用夹爪和无硬件测试。
- `Robotiq2F85Gripper` 构造时不联网；`start()` 才连接 `63352/TCP`。
- 保留旧实现的 `0..230` register 与 `0..max_width_m` 反向映射。
- 保留持久 socket、`SPE=255` 和一次断线重连。
- socket/read response、clamp、重复 close 均由 fake socket 测试保护。

## 7. 兼容与依赖边界

- `robot_backend.py` 公开导出名称保持不变，并发出 `DeprecationWarning`。
- 只有访问旧导出时才加载 `airo_doffy.robots.legacy` 和旧 SDK 依赖。
- 新 `create_robot_backend()` / `create_gripper()` 只返回未启动对象。
- typed `RobotFactory` 可指向
  `airo_doffy.robots.factory:create_robot_backend`。
- 旧 `RobotTeleop` 和 `RealManTeleop` 仍走兼容实现；未声称新旧完整 teleop 数值已经
  接线等价。
- tool TCP transform、mapping、workspace/joint safety 和 VR stale watchdog 属于后续
  teleop mapping/safety/session 阶段，不复制进 vendor adapter。

## 8. 验证结果

| 验证 | 结果 |
|---|---|
| Phase 3 新增机器人/夹爪测试 | 36/36 通过 |
| 当前全部 v2 unit tests | 75/75 通过 |
| 包布局与兼容 smoke tests | 6/6 通过 |
| RealMan 200 Hz mock timing gate | 连续 10 次通过 |
| RealMan 12 ms 人工 SDK latency | 正确触发 timing failure |
| `import airo_doffy.config, airo_doffy.robots` | 未加载列出的数值/视觉/机器人可选依赖 |
| `import robot_backend` | 不加载 legacy、NumPy、SciPy 或厂商 SDK |
| 可编辑安装 | `airo-doffy==2.0.0.dev0` 构建安装通过 |
| 全仓 Python AST | 139 个文件通过 |
| `git diff --check` / 新包 100 字符行检查 | 通过 |

旧 `tests/test_realman_teleop_loop.py` 在当前最小环境仍于收集阶段缺少 `cv2`，所以其
28 个旧测试未计入通过数；实际错误为 `ModuleNotFoundError: No module named 'cv2'`。
没有通过改写测试或伪造模块来掩盖这一点。

## 9. 硬件验证与下一阶段

本阶段没有连接 UR3e、UR5e、RM75 或 Robotiq。进入真实运行 composition 前必须分别完成
`H-UR` 和 `H-RM`，包括低速动作、急停、断线、力坐标、SDK shutdown、CAN-FD cadence
和 state push。

按已评审迁移顺序，下一阶段进入触觉与 wrench：

1. 把 4-taxel BLE reader 拆成通信、校准、滤波和私有 latest state；
2. 增加 mock tactile；
3. 把 `force_filter.py` 拆为纯 wrench processing；
4. 保持根 tactile/wrench 兼容入口，并用数值 golden 与无硬件断连测试保护。
