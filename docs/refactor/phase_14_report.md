# Phase 14 测试与质量报告

日期：2026-07-30

分支：`codex/v2-refactor`

主要提交：

- `4e24a03`：完整 mock 会话集成测试
- `a8c2d92`：默认跳过的硬件 smoke tests
- `cd1cd03`：可移植质量工具配置

## 结论

Phase 14 已完成。计划要求的单元测试主题均有硬件无关覆盖；新增的集成测试
通过真实 v2 端口和组件运行一个短数据采集会话；硬件检查被放在独立目录，
只有设备专用环境变量存在时才会连接设备。

质量配置统一到 `pyproject.toml` 和 `.pre-commit-config.yaml`。旧
`pyrightconfig.json` 中硬编码的 Linux Conda 路径已删除，避免覆盖可移植的
Pyright 设置。

## 14.1 单元测试覆盖

| 计划主题 | 主要测试 |
|---|---|
| VR 文本、二进制协议与接收 | `test_vr_protocol.py`、`test_vr_binary_v2.py`、`test_vr_receiver.py` |
| 状态、命令序列化与序列拒绝 | `test_state_protocol.py`、`test_command_protocol.py`、对应 channel tests |
| latest-value buffer | `test_core_buffers.py`、state/video channel tests |
| 遥操作映射与坐标变换 | `test_teleop_mappings.py`、`test_teleop_transforms.py` |
| 安全过滤与 watchdog | `test_teleop_safety.py`、`test_teleop_watchdog.py` |
| 触觉与 wrench 处理 | `test_tactile_filters.py`、`test_wrench_filters.py`、`test_wrench_processing.py` |
| 帧处理、编码和 UDP 分包 | `test_frame_processor.py`、video encoder/pipeline/UDP/RTP tests |
| 数据集 schema、writer 与 rollback | recording schema/writer/state/rollback/export tests |
| 运行时生命周期 | runtime lifecycle/session/data-collection tests |

单元测试通过依赖注入、内存 transport、mock SDK 和虚拟时钟运行，不连接物理
硬件。包边界测试还会在子进程中验证基础包导入不会加载相机、机器人、网络、
视频编解码或策略可选依赖。

## 14.2 完整 mock 集成

`tests/integration/test_mock_session.py` 组合并运行：

1. `MockRobotBackend` 与 `LatestActionExecutor`
2. `MockVRInputSource`
3. `MockCameraSource` 与 `PackedFrameProcessor`
4. `LatestVideoEncodingPipeline`、内存 encoder 与 transport
5. `MockTactileSensor`
6. `RecordingCycleExtension` 与内存 episode writer
7. `TypedSnapshotConsumer` 与 `MemorySnapshotRenderer`
8. `TeleopSession` 与 `DataCollectionSession`

测试运行 3 个控制周期并验证：

- 安全动作到达 mock robot；
- 相机帧被处理、编码并按序发送；
- 触觉和图像进入 3 条不可变录制样本；
- episode 在异步 export worker 中完成；
- 可视化收到最终序列快照；
- 关闭会话后 encoder、transport 和 writer 均被释放。

测试中的组合扩展只存在于 `tests/integration/`。它不会成为新的生产
all-in-one manager，也不会改变正式循环的职责边界。

## 14.3 硬件测试

`tests/hardware/test_devices.py` 提供以下显式启用检查：

| 环境变量 | 检查 |
|---|---|
| `AIRO_DOFFY_TEST_UR_IP` | 启动 UR backend 并读取一次 6-DOF 状态 |
| `AIRO_DOFFY_TEST_REALMAN_IP` | 启动 RealMan backend 并读取一次 7-DOF 状态 |
| `AIRO_DOFFY_TEST_REALSENSE=1` | 获取一帧 RealSense RGB 图像 |
| `AIRO_DOFFY_TEST_BLE4=1` | 等待一条 `(4, 3)` BLE4 触觉样本 |

默认 discovery 中四项全部跳过。机器人测试不提交运动动作，但 backend
关闭时仍可能执行正常 stop，因此运行说明要求受监督、安全的工作单元。

## 14.4 质量工具

`pyproject.toml` 已配置：

- pytest test path 与 hardware marker；
- Ruff 目标版本、100 字符行宽和 lint 规则；
- Pyright include、Python 版本和 basic 类型检查；
- `dev` extra 中的 build、pytest、Ruff、Pyright 和 pre-commit。

本地 pre-commit hooks 分别执行：

- Ruff lint；
- Ruff format check；
- Pyright；
- unit tests；
- mock integration tests；
- package boundary tests；
- hardware tests 的默认跳过检查。

旧 `tests/test_realman_teleop_loop.py` 是兼容行为刻画套件。它现在会在缺少
OpenCV、NumPy、SciPy 或空间代数可选依赖时整模块跳过，使最小安装可以完成
统一测试收集，而不会把缺失依赖误报成产品失败。

## 验证

| 检查 | 结果 |
|---|---|
| `unittest discover -s tests` | 293 通过，8 跳过 |
| v2 unit tests | 276 通过，3 跳过 |
| 完整 mock integration | 1 通过 |
| 包布局/兼容边界 | 11 通过 |
| 硬件默认门禁 | 4 跳过，0 个设备连接 |
| 旧 RealMan 可选依赖套件 | 1 个模块跳过 |
| tracked Python AST | 241/241 通过 |
| `compileall` | 通过 |
| `git diff --check` | 通过 |
| Python 100 字符行限制 | 通过 |
| 离线 editable install | `airo-doffy==2.0.0.dev0` 通过 |
| teleop/collect CLI `--help` | 通过 |

3 个 unit skip 是缺少 `h5py` 的真实 HDF5 集成测试。当前虚拟环境也未安装
pytest、Ruff、Pyright 和 pre-commit，因此本阶段完成了这些工具的配置，
但不声明其命令已在本机执行。

## 下一阶段

进入 Phase 15，重写面向 v2 的 README、架构和通信文档，补充扩展指南与
迁移/发布说明。文档必须以当前代码和测试证据为准，并明确尚未完成的真实
硬件、HDF5 可选依赖和 release candidate 验证。
