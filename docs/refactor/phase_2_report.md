# Phase 2 重构报告：核心类型、接口、Buffer 与事件

日期：2026-07-30

分支：`codex/v2-refactor`

前置阶段：Phase 1 (`962464d`)

## 1. 阶段结论

Phase 2 已完成。仓库现在具有不依赖 NumPy、机器人 SDK、相机 SDK、网络库或
ML 框架的核心契约层。高频数据使用冻结对象、显式序列号和源时间戳；设备、
视频、映射、安全、机器人和记录接口由各自领域拥有；实时最新值使用单槽
buffer，不再要求通过宽 manager 或无界队列共享。

本阶段没有实现设备 adapter、协议 serializer、机器人控制 cadence 或运行时
session。接口定义不等于现有硬件模块已经迁移，后续阶段仍需在旧实现外加测试
后逐个接入。

## 2. 核心组件

| 组件 | 责任 | 主要约束 |
|---|---|---|
| `core/types.py` | 定义相机、处理后帧、编码帧、触觉、力矩、控制器、手、VR、机器人、动作和 Observation。 | frozen、slots、复制可变 buffer、有限数值、固定形状、序列号和时间戳。 |
| `core/events.py` | 定义可靠命令和可观察运行事件。 | enum 命令类型、命令 ID、序列、时间戳、不可变 details；命令值按类型校验。 |
| `core/buffers.py` | 保存一个最新值并协调生产者/消费者。 | 常量内存、严格新序列、可选模数回绕、超时等待、关闭唤醒、关闭后拒绝发布。 |
| `core/clocks.py` | 提供可注入纳秒时钟。 | 区分 monotonic 与 Unix wall clock。 |
| `core/errors.py` | 提供共享错误层次。 | 校验、buffer 关闭、生命周期和可选依赖错误互相区分。 |
| `core/interfaces.py` | 仅保存跨域 `Startable`、`Closable`、`Lifecycle`。 | 不重复定义设备或机器人 ports。 |

## 3. 数据契约

- `CameraFrame` / `ProcessedFrame` 使用不可变 `bytes` 和显式 packed
  pixel format；shape 与 payload 字节数必须一致。
- `EncodedFrame` 包含 codec、宽高、keyframe、源时间和编码时间。
- 支持的 `TactileSample` 固定为 `(4, 3)`，不会把 deprecated 41-taxel
  形状重新引入支持 API。
- `WrenchSample` 固定顺序为 `Fx,Fy,Fz,Tx,Ty,Tz`。
- `ControllerState` 对应当前每手 14 字段中的 pose、joystick、trigger 和
  button；`HandState` 固定 26 个 OpenXR xyz joints。
- `VRInputState` 明确区分 controller 和 hand mode，并拒绝重复 side。
- `RobotState` 接受 UR 6 DOF 或 RealMan 7 DOF，TCP pose 固定 `(4, 4)`。
- `RobotAction` 按 command type 校验 6/7 关节、16 值 homogeneous TCP
  pose、6 值 twist，`HOLD` / `STOP` 不允许携带运动值。
- `Observation` 组合独立可选来源，并拒绝重复 camera stream ID。

为了保持基础安装零第三方依赖，核心数值用不可变 tuple，packed 图像用 bytes。
adapter 可以接收 NumPy，但必须在领域边界转换；从 `bytearray` 或可写
memoryview 创建帧时会复制一次，以保证发布后数据不被生产者修改。

## 4. 领域接口与依赖方向

| Port | 所有者 |
|---|---|
| `CameraSource` | `devices/cameras/base.py` |
| `TactileSensor` | `devices/tactile/base.py` |
| `VRInputSource` | `devices/vr/base.py` |
| `FrameProcessor`、`VideoEncoder`、`VideoTransport` | `streaming/video/base.py` |
| `TeleopMapping` | `teleop/mappings/base.py` |
| `ActionFilter` | `teleop/safety/base.py` |
| `RobotBackend` | `robots/base.py` |
| `EpisodeRecorder` | `recording/base.py` |

这是 Phase 0 迁移图记录的结构修订：领域拥有自己的 port，`core/interfaces.py`
不再成为另一套重复真值。所有 port 只导入核心类型和标准库 `Protocol`，不会
导入厂商 SDK。

## 5. Latest-value 语义

- 内部只保存一个 value 和一个 sequence；发布 10,000 次后仍为单槽状态。
- 默认使用普通递增整数。提供 `sequence_modulus` 时采用半序列空间规则处理
  uint32 等回绕；重复、反向和恰好半范围的歧义值被拒绝。
- `publish()` 对 stale/duplicate 返回 `False` 并累计拒绝计数。
- `wait_for_new()` 可立即读取现值、等待指定 sequence 之后的新值、超时返回
  `None`，或在关闭时被唤醒。
- `close()` 幂等；关闭后消费者仍可读取最后值，但生产者发布会得到
  `BufferClosedError`。

buffer 假定发布的是不可变对象；核心 samples 和 runtime messages 已满足该
条件。它不承担 serializer、transport 或业务队列职责。

## 6. Typed command 落地

Visualizer 的 rollback multiprocessing queue 已从
`{"command": "rollback_last_episode"}` 改为可 pickle 的
`RuntimeCommand(RuntimeCommandType.ROLLBACK_LAST_EPISODE)`；主数据采集循环
按 enum 分支，不再比较任意内部字符串。

UDP/WebRTC 当前接收的 `Start`、`Stop`、`Undo` 等仍属于旧线协议文本。它们
将在协议/可靠命令阶段由 parser 转成 `RuntimeCommand`；本阶段没有擅自修改
现有 Unity 兼容面。

## 7. 验证结果

| 验证 | 结果 |
|---|---|
| 可编辑安装 | 通过，重新构建并安装 `airo-doffy==2.0.0.dev0`。 |
| Phase 1 结构/兼容测试 | 5/5 通过。 |
| Phase 2 纯单元测试 | 24/24 通过。 |
| 全仓 Python AST | 116 个文件通过。 |
| 核心及所有 domain port 安全导入 | 通过；未加载 NumPy、Torch、OpenCV、SciPy、RealSense、aiortc、serial 或 airo_robots。 |
| shape、非有限值、冻结/复制 | 通过单元测试。 |
| buffer stale、duplicate、回绕、timeout、并发等待和 close | 通过单元测试。 |
| RuntimeCommand multiprocessing pickle round-trip | 通过。 |
| `git diff --check` | 通过。 |

原 RealMan 测试模块在当前最小环境仍因缺少可选 `cv2` 而无法收集；这不是本
阶段测试伪通过。Phase 2 测试不需要或模拟任何物理硬件。

## 8. 硬件验证

未连接 UR3e、UR5e、RealMan RM75、RealSense、Quest/Unity、BLE4 或旧串口
触觉。Phase 2 为 `H0` 契约工作；实际 adapter 接入时仍需执行各自硬件门禁。

## 9. 下一步顺序

按照 Phase 0 评审后的修订顺序，下一步应先把原 Phase 13 的 typed config、
loader 和 factories 前移到机器人/相机 adapter 拆分之前，并用当前
`Config` / `VisualizerConfig` snapshot 证明默认行为不变。随后再进入
机器人 backend/executor 迁移，避免新 adapter 继续依赖旧的宽全局配置。
