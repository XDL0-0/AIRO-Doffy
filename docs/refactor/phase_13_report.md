# Phase 13 配置重构报告

日期：2026-07-30

分支：`codex/v2-refactor`

主要提交：

- `33e44a3`：类型化配置模型
- `a155263`：分层 YAML 配置
- `462663b`：早期配置阶段报告
- `f389b9a`：与 Phase 10–12 的录制、可视化和运行时能力对齐

## 结论

Phase 13 已完成。配置边界由 13 个不可变、分区的类型化模型组成，
加载顺序固定为默认 YAML、机器人 YAML、实验 YAML、环境变量和 CLI
覆盖。8 个硬件或可选组件工厂只接收其所需配置分区，并延迟导入具体实现。

配置基础设施依照 Phase 0 的迁移顺序在设备适配器之前提前建立。本阶段回到
Phase 13 时，重点不是重写已经工作的加载器，而是将 Phase 10–12 后来新增的
有界队列和会话生命周期参数纳入同一个配置边界，并重新检查工厂协议。

## 需求对应

### 13.1 类型化配置模型

`AiroDoffyConfig` 聚合以下分区：

- `NetworkConfig`
- `RobotConfig`
- `CameraConfig`
- `VRConfig`
- `TeleopConfig`
- `TactileConfig`
- `RecordingConfig`
- `VisualizationConfig`
- `VideoStreamingConfig`
- `StateTransportConfig`
- `CommandTransportConfig`
- `WrenchConfig`
- `RuntimeConfig`

所有分区均为 frozen/slotted dataclass，并在构造时验证枚举、端口、尺寸、
速率、容量及跨字段约束。工厂和应用 composition factory 接收窄配置分区，
设备组件不需要依赖完整配置对象。

本轮新增并验证了以下后期配置：

| 分区 | 字段 | 默认值 | 约束 |
|---|---|---:|---|
| `RecordingConfig` | `export_queue_capacity` | `2` | 正整数 |
| `RecordingConfig` | `sample_capacity` | `null` | `null` 或正整数 |
| `VisualizationConfig` | `command_queue_capacity` | `8` | 正整数 |
| `RuntimeConfig` | `worker_join_timeout_s` | `5.0` | 正有限数 |
| `RuntimeConfig` | `max_control_dt_s` | `0.05` | 正有限数 |

`TeleopSession` 现在接受并校验 `worker_join_timeout_s`，再把它传给受管理的
机器人执行器工作线程。`max_control_dt_s` 对应会话已有的 `max_dt_s`
构造参数，具体 composition factory 负责从窄配置中注入。

### 13.2 YAML 与覆盖优先级

`load_config()` 的覆盖顺序由测试固定为：

1. `configs/default.yaml`
2. `configs/robots/*.yaml`
3. `configs/experiments/*.yaml`
4. `AIRO_DOFFY__SECTION__FIELD` 环境变量
5. `section.field` CLI 覆盖

`configs/default.yaml` 已包含新增字段，读取后仍与
`AiroDoffyConfig()` 完全相等。仓库配置未包含密码、令牌、API key、
用户目录或 `G:\Projects` 等机器路径。PC、VR、机器人地址和相机序列号
保持为 `null`，由部署配置显式提供。

### 13.3 聚焦工厂

以下延迟工厂保持可用：

| 工厂 | 输入 | 返回协议 |
|---|---|---|
| `RobotFactory` | `RobotConfig` | `RobotBackend` |
| `CameraFactory` | `CameraConfig` | `CameraSource` |
| `EncoderFactory` | `VideoStreamingConfig` | `VideoEncoder` |
| `VideoTransportFactory` | video + network | `VideoTransport` |
| `VRSourceFactory` | VR + network | `VRInputSource` |
| `TactileFactory` | `TactileConfig` | `TactileSensor` |
| `RecorderFactory` | `RecordingConfig` | `EpisodeRecorder` |
| `VisualizerFactory` | `VisualizationConfig` | `SnapshotConsumer` |

`VisualizerFactory` 原先只验证通用 `Lifecycle`，无法保证实例具有
`publish(snapshot)`。本轮将返回协议收窄为 `SnapshotConsumer`，并添加
“只有 start/close、没有 publish”的反例测试。

工厂构造阶段仍不会加载硬件 SDK。只有调用 `create()` 时才解析
`module:symbol`，缺失可选依赖、无效目标或协议不匹配均转换为明确的 v2
错误。

## 兼容边界

- 根目录旧 `Config` 与 `VisualizerConfig` 尚未删除，旧运行路径没有被暗中
  切换。
- 新 CLI 继续要求显式 session factory 或对应环境变量；仓库尚未假设唯一的
  硬件组合。
- `RecorderFactory` 保留 Phase 2 的兼容 `EpisodeRecorder` 边界。新的
  `RecordingCycleExtension`、writer 和 sample factory 在运行时组合层装配，
  避免把整个录制编排塞回单个工厂。
- 本阶段没有连接机器人、VR、相机、触觉设备或网络端点。

## 验证

| 检查 | 结果 |
|---|---|
| 全部 v2 单元测试 | 276 通过，3 跳过 |
| 包布局/兼容测试 | 11 通过 |
| 定向配置、工厂、会话、CLI 测试 | 25 通过 |
| 仓库配置 secret/机器路径扫描 | 未发现 |
| tracked Python AST 解析 | 237/237 通过 |
| `compileall` | 通过 |
| `git diff --check` | 通过 |
| Python 100 字符行限制 | 通过 |

3 个跳过测试为真实 HDF5 写入集成测试；当前虚拟环境未安装可选 `h5py`。
Ruff 和 Pyright 在当前环境不可用，因此没有伪造对应结果。

## 下一阶段

进入 Phase 14，按计划逐项盘点测试矩阵。优先补齐跨模块、故障注入和
可重复基准测试，不用更多文档或知识图谱代替行为验证。
