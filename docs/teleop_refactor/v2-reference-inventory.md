# AIRO-DOFFY-v2 参考仓库盘点

审计日期：2026-09-16\
目标：为 `AIRO-Doffy` 的 Unity + PC 全功能重构提供可复用代码、协议、测试和缺口清单。\
范围：只读检查 `/home/yuyuan/AIRO-DOFFY-v2` 的当前工作树、文档和测试；未检查大数据、权重及 `outputs/` 内容，也未修改 v2。

## 结论先行

- v2 最有价值的可复用部分是 `src/airo_doffy/core` 的不可变数据模型和端口、`devices/vr` 的协议解码与 latest-only 接收器、`teleop` 的变换/映射/安全链、`streaming` 的视频/状态/命令协议、`recording` 的 schema/回滚/异步导出，以及 `runtime` 的生命周期和单控制循环。这些都有硬件无关的 fake/mock 测试。
- 当前工作树不是可直接发布的稳定参考：分支为 `AIRO-DOFFY-v2.0`，HEAD 为 `9274594`，`git status` 显示约 110 个文件有删除、修改或未跟踪变化。旧根模块、`deprecated/`、`dataset_tool/`、`test_tool/`、旧兼容入口被当前工作树删除，同时 Beaver、policy evaluation、UDP VR、Matplotlib 等新文件仍未跟踪。引用实现前应先固定一个提交并逐项审查这些差异。
- 当前内置 `airo-doffy-teleop` 只覆盖基本 Quest UDP pose → UR/RealMan 机器人路径；相机、视频、触觉、wrench、状态通道、可靠命令、录制和 gripper 没有接入这个 factory。`airo-doffy-collect` 仍必须由外部 `--session-factory` 组合。
- v2 没有 Unity/C# 客户端、Unity 工程、场景、协议 fixture 或跨语言 golden bytes；`unity/README.md` 明确只是占位。Unity+PC 重构必须自行实现客户端和互操作验收。
- 代码层面的协议、mock runtime 和多数适配器已较完整；真实 UR/RealMan/RealSense/Quest/BLE4、PyAV/aiortc、真实 HDF5/LeRobot、Unity RTP/WebRTC 和完整 workcell composition 均未完成硬件或网络端到端验证。

## 1. 参考树基线与入口

`README.md`（当前工作树）称版本为 `2.0.0.dev0`，并列出 UR3e/UR5e、RealMan RM75、RealSense、Quest、四点 BLE4 MagTouch、视频传输、状态/命令通道和 HDF5/LeRobot 等能力。`docs/refactor/phase_1_report.md` 至 `phase_15_report.md`、`docs/release_checklist.md` 保存了各阶段证据，但部分是旧工作树快照；例如历史报告仍假定根兼容 wrapper 存在，而当前 `README.md`、`tests/test_package_layout.py` 和文件系统已把它们删除。

| 入口/位置 | 当前行为 | 重构价值与限制 |
|---|---|---|
| `src/airo_doffy/apps/common.py` | 解析配置、`module:symbol` factory、生命周期和 Ctrl-C 清理 | 可作为 PC 进程入口边界；factory 只负责组合，不能代替完整部署编排 |
| `src/airo_doffy/apps/teleop.py` | `airo-doffy-teleop`，默认 `airo_doffy.apps.deployment:build_teleop_session` | 可启动基本 UDP Quest + 机器人；默认不含相机/录制/命令/状态通道/gripper |
| `src/airo_doffy/apps/collect.py` | `airo-doffy-collect`，要求 `--session-factory` 或环境变量 | 接口清楚，但没有可直接运行的全功能 collection composition |
| `src/airo_doffy/apps/deployment.py`（未跟踪） | 基本 UR/RealMan deployment；可选 Matplotlib；另有 RM75 policy-evaluation composition | 适合作为实验起点；当前实现仍是基本路径，RealMan joint 的 IK 责任推给 CAN-FD executor，且未接入所有 v2 组件 |
| `src/airo_doffy/apps/evaluate.py`（未跟踪） | 先校验 checkpoint，再加载外部 `beaver_policies` adapter，运行 RM75 评估 | 是独立评估工具，不是通用 Unity teleop；外部 policy/Beaver 依赖和硬件仍需部署验证 |
| `configs/default.yaml`、`configs/robots/*.yaml` | 当前是 JSON-compatible YAML；支持分层、环境变量和 `--set` | 配置机制可复用，但当前 default 写入了 `10.135.223.48`、`10.135.223.229`，RealMan profile 写入 `192.168.1.18`；这与文档所说的“默认无地址”冲突，应在新 PC 配置中清除运行现场地址 |
| `scripts/` | 诊断、dataset 转换/回放、视频和轨迹 benchmark | 可按需迁移 CLI；当前树已删除多项旧脚本，不能假设 v1 脚本兼容 |
| `unity/` | 只有 `unity/README.md` | 无客户端实现，只有协议边界说明 |

配置模型集中在 `src/airo_doffy/config/models.py`，当前聚合了 network、robot、camera、VR、teleop、tactile、Beaver、policy evaluation、recording、visualization、video、state transport、command transport、wrench、runtime 等 15 类设置。`src/airo_doffy/config/loader.py` 做深度合并、类型转换、环境变量 `AIRO_DOFFY__SECTION__FIELD` 和 `--set` 覆盖，并拒绝未知字段；`config/factories.py` 提供 robot/camera/encoder/video/VR/tactile/recorder/visualizer 八个窄 factory。它们是好的依赖注入边界，但不会自动构建一个多设备 session。

## 2. 模块化实现盘点

这里的“软件完整”表示接口、错误处理和硬件无关测试基本齐全，不表示已通过真实设备验收。

| 模块 | 关键文件位置 | 判断 | 可复用项/主要缺口 |
|---|---|---|---|
| 核心模型与基础设施 | `src/airo_doffy/core/types.py`, `interfaces.py`, `buffers.py`, `clocks.py`, `errors.py`, `events.py` | **完整，可直接复用** | frozen/slots 样本、shape/finite 检查、时钟/序列号、latest buffer、错误和事件。依赖少，适合作为 Unity PC 进程的领域模型；Unity 仍需独立 C# 表示 |
| 相机 | `devices/cameras/base.py`, `mock.py`, `realsense.py` | **软件完整，实机未验** | RealSense 延迟导入 SDK、采集线程、最新帧和健康状态；mock 支持失败/延迟/过期。当前 deployment 未创建 camera/video pipeline |
| VR 输入 | `devices/vr/protocol.py`, `binary_v2.py`, `receiver.py`, `mock.py`, `udp.py` | **协议与接收器完整，可直接复用** | 支持历史文本、`AVR2` binary v2、聚合双手、重复/乱序/uint32 wrap、stale 和 latest-only；UDP transport 是当前未跟踪新增。兼容解析器仍有宽松点（HB 尾随字节/base64、旧负时间戳识别），新 Unity 应使用严格 binary v2 |
| 触觉 | `devices/tactile/base.py`, `source.py`, `magtouch_ble4.py`, `filters.py`, `mock.py` | **软件完整，设备未验** | BLE4 `(4,3)` 样本、校准、deadband/EMA/Kalman/漂移和断线语义；旧 serial/legacy 文件在当前工作树被删除，README 说不再支持。deployment 未接入 |
| wrench | `devices/wrench/base.py`, `robot_source.py`, `compensation.py`, `filters.py`, `pipeline.py` | **处理链完整，来源/接线不足** | 重力/偏置补偿、均值/低通/deadband/clamp 都是纯逻辑；当前没有独立物理 wrench adapter，也未进入基本 teleop composition |
| 机器人与执行器 | `robots/base.py`, `mock.py`, `ur.py`, `realman.py`, `executor.py`, `realman_executor.py`, `grippers/*` | **软件适配器基本完整，实机未验** | UR/RealMan 延迟 SDK、Mock、通用 latest executor、RealMan CAN-FD high-follow executor、Robotiq/Null gripper。必须遵守 `docs/architecture.md` 的 state-source 线程归属约束：不能让 session 与 executor 并发读写 thread-affine SDK；当前基本 RealMan composition 仍需硬件确认 |
| 变换/映射/安全 | `teleop/transforms/*`, `teleop/mappings/*`, `teleop/safety/*` | **完整，可直接复用** | 轴映射、四元数、局部/世界旋转、reference/rebase、joint/TCP/IK/gripper 映射、workspace/joint/velocity/acceleration/rate、freshness/watchdog；测试是确定性的。需用 v1 录制轨迹和真实工位重新核对坐标标定 |
| 视频 | `streaming/video/frame_processor.py`, `h264_encoder.py`, `encoding_pipeline.py`, `legacy_jpeg_udp.py`, `rtp_h264_udp.py`, `webrtc_transport.py` | **组件完整，生产链未组合/实传未验** | 处理、drop-oldest/latest、PyAV/NVENC→x264 选择、兼容 JPEG chunk、RFC6184 RTP/FU-A、aiortc transport。真实 codec、SDP/ICE、MTU、Quest/Unity receiver、重连未验；旧 JPEG 12B `!IHHI` header 保留兼容但 60,000B chunk 可能导致 IP fragmentation |
| 状态/命令通道 | `streaming/state/{protocol,channels}.py`, `streaming/commands/{protocol,channels,router}.py` | **协议/假传输完整，runtime 未接线** | latest-only binary state、WebRTC unordered/unreliable adapter、ordered reliable command、ACK/超时/去重/router 均有测试；基本 deployment 没有创建 channel/handler，Unity 端不存在 |
| 录制与导出 | `recording/schema.py`, `samples.py`, `state.py`, `writers/{hdf5,lerobot,rollback}.py`, `export_worker.py` | **边界与状态机完整，真实库未验** | 不可变 sample、episode 编号、rollback/reuse、bounded worker、失败可见/retry/discard；`DataCollectionSession` 是同一 teleop loop 的 extension。当前 collect 无内置 composition；h5py/LeRobot 真实存储比较未完成 |
| runtime | `runtime/lifecycle.py`, `session.py`, `data_collection.py`, `ports.py` | **控制循环与生命周期完整** | 启动回滚、反向 close、worker health、HOLD/STOP、单循环 TeleopCycle、录制 extension。它不会自动发现或启动 camera/video/tactile/state/command，需上层 composition |
| 可视化 | `visualization/models.py`, `consumer.py`, `mock.py`, `commands.py`; 未跟踪 `extension.py`, `matplotlib.py`, `opencv_eval.py` | **headless 边界完整，GUI/接线未验** | typed latest-only snapshot、错误隔离、内存 renderer 可用于 CI；Matplotlib/评估 renderer 和多传感器 snapshot 在真实桌面/完整 session 未验，Unity UI 为空 |
| policy/Beaver | 未跟踪 `policies/base.py`, `policies/forceflowpp/*`, `devices/beaver/*`, `runtime/policy_evaluation.py` | **新增实验路径，不能当稳定基线** | Beaver serial 解码/reconnect/stale、adapter contract、RM75 dry-run/execute safety 有测试；依赖外部 `beaver_policies`、checkpoint 和硬件，未形成通用 teleop |

设计规则和扩展约束可读 `docs/architecture.md`、`docs/extension_guide.md`：构造函数不做 I/O，`start()` 获取资源，数据跨边界必须不可变，后台队列有上限和 drop policy，SDK thread-affinity 由部署 factory 显式处理。

## 3. 协议与 Unity 端需要实现的内容

| 通道 | v2 实现/规范 | 适合复用的测试 | 当前互操作状态 |
|---|---|---|---|
| VR 输入 | `docs/protocols/vr_binary_v2.md`, `devices/vr/binary_v2.py`；little-endian `AVR2` v2，24B header，controller/hand entity，float32、quaternion XYZW、uint32 sequence | `tests/unit/test_vr_binary_v2.py`, `test_vr_protocol.py`, `test_vr_receiver.py` | Python 内部 round-trip/坏包/乱序已验；Unity/C# packing 只有文档 outline，无 fixture |
| realtime state | `docs/protocols/state_channel_v1.md`, `streaming/state/protocol.py`；20B envelope，magic `0xAD20`，VR/robot payload，latest-only | `tests/unit/test_state_protocol.py`, `test_state_channels.py` | fake channel 已验；WebRTC data channel 和 Unity decoder 未验；没有接入基本 deployment |
| runtime commands | `docs/protocols/reliable_commands_v1.md`, `streaming/commands/*`；严格 canonical JSON v1，ordered reliable，ACK、command id 去重 | `tests/unit/test_command_protocol.py`, `test_command_channels.py`, `test_command_router.py` | Python fake/retry/duplicate 已验；没有 Unity serializer/ACK fixture/真实 WebRTC 验收 |
| video | `docs/communication.md`；WebRTC H.264、RTP H.264、旧 JPEG chunk；默认 RTP MTU 1200、JPEG compatibility header `!IHHI` | `test_frame_processor.py`, `test_h264_encoder.py`, `test_video_encoding_pipeline.py`, `test_legacy_jpeg_udp.py`, `test_rtp_h264_udp.py`, `test_webrtc_transport.py` | packetizer/假 encoder 已验；实际 aiortc/PyAV/Unity receiver/ICE/重连未验 |

端口和标签在 `docs/communication.md`：旧基础/pose/control 为 `8000/8001/8005`，signaling `8765`，RTP `5004`，诊断 state `5005`，RealMan realtime state push 示例为 `8098`。新 PC/Unity 设计应把这些作为配置，不要硬编码到客户端；状态与命令必须使用不同 DataChannel（状态 unordered/unreliable，命令 ordered/reliable）。

`docs/communication.md` 和 `docs/protocols/vr_binary_v2.md` 已写 C# 字段顺序、little-endian、float32、row-major 4×4、quaternion XYZW、uint64 timestamp、sequence wrap 等指导，但没有可供 Unity 自动回归的跨语言字节样本。Unity 最小交付应包括：独立 pack/unpack 类、与 Python 共享的 golden bytes（正常/截断/错误版本/非法 entity/sequence wrap）、WebRTC channel label/可靠性配置、command ACK/去重、RTP/H.264 接收日志。

## 4. 可直接验证的无硬件路径

在 v2 当前工作树中，未安装 editable package 时需要显式设置 `PYTHONPATH=src`。已执行并通过：

```bash
cd /home/yuyuan/AIRO-DOFFY-v2
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /home/yuyuan/miniconda3/bin/python -m unittest discover -s tests -q
# Ran 316 tests in 1.462s
# OK (skipped=7)

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /home/yuyuan/miniconda3/bin/python -m compileall -q src tests
# exit 0
```

还可直接执行并返回 0 的入口 smoke check：

```bash
PYTHONPATH=src python -m airo_doffy.apps.teleop --help
PYTHONPATH=src python -m airo_doffy.apps.collect --help
PYTHONPATH=src python -m airo_doffy.apps.evaluate --help
```

重点的硬件无关覆盖如下：

- 核心/模型/缓冲：`tests/unit/test_core_types.py`, `test_core_buffers.py`, `test_domain_interfaces.py`。
- config/入口/factory：`test_config_loader.py`, `test_config_models.py`, `test_config_factories.py`, `test_app_entrypoints.py`, `tests/test_package_layout.py`。
- Mock session/生命周期/录制：`tests/integration/test_mock_session.py`, `test_runtime_lifecycle.py`, `test_runtime_session.py`, `test_runtime_data_collection.py`，以及 `test_recording_*`。
- 机器人/teleop：`test_mock_robot.py`, `test_robot_executor.py`, `test_realman_executor.py`, `test_ur_robot.py`, `test_realman_robot.py`, `test_teleop_{transforms,mappings,safety,watchdog}.py`。
- 相机/触觉/VR/wrench：`test_mock_camera.py`, `test_realsense_camera.py`, `test_mock_tactile.py`, `test_magtouch_ble4.py`, `test_tactile_filters.py`, `test_mock_vr.py`, `test_vr_*.py`, `test_wrench_*.py`。
- 网络/编码/状态/命令：`test_{h264_encoder,video_encoding_pipeline,legacy_jpeg_udp,rtp_h264_udp,webrtc_transport,state_protocol,state_channels,command_protocol,command_channels,command_router}.py`。
- 当前新增评估路径：`test_beaver_source.py`, `test_policy_evaluation.py`, `test_policy_eval_cli.py`；因其未跟踪，需先确认是否纳入正式提交。

`tests/hardware/test_devices.py` 默认跳过，启用前必须阅读 `tests/hardware/README.md`。环境变量为 `AIRO_DOFFY_TEST_UR_IP`、`AIRO_DOFFY_TEST_REALMAN_IP`、`AIRO_DOFFY_TEST_REALSENSE=1`（可选 `AIRO_DOFFY_TEST_REALSENSE_SERIAL`）和 `AIRO_DOFFY_TEST_BLE4=1`。机器人 smoke 只读一次 state 并 close，不主动提交运动 action，但 close 可能发送适配器正常 stop；必须在受监督安全工位执行。

当前环境有 Python 3.13 和 `ruff`，未发现 `pytest` 或 `pyright` 命令。`docs/release_checklist.md` 也把 pytest/ruff/pyright/pre-commit、HDF5、Unity、硬件、网络 E2E 和性能列为未勾选项。因此 316 个 unittest 只能证明纯 Python/Mock 路径，不能证明生产部署。

## 5. 面向 v1 用途的明确缺口

1. **全功能 PC composition 尚未存在。** v1 的实际用途是把 VR、机器人、相机/视频、触觉/wrench、可视化、录制和控制放在一次 teleop 工作流；v2 的模块已拆开，但 `build_teleop_session()` 只组装基本 VR + robot + mapping/safety/executor。应沿 `runtime` ports 组合一个单控制循环和多个 acquisition/stream/export worker，再把 `RecordingCycleExtension`、visualization、state/command router 显式挂入。
2. **Unity 输入/显示/控制端为空。** v2 仅有协议文档；跨语言字节、时钟/坐标、DataChannel 建立顺序、ACK/去重、H.264 接收、断线/重连都没有 Unity 验收。Unity 是本轮重构的第一项新增工程，而不是可从 v2 复制的文件夹。
3. **v2 通道实现尚未进入应用路径。** state/commands 的协议和 fake adapters 存在，但当前 deployment 没有构造它们，也没有把 reset/recording/recalibrate/mode/safe-hold 等命令路由到 runtime。
4. **线程归属存在生产风险。** `docs/architecture.md` 明确指出 thread-affine vendor SDK 不能由 `TeleopSession` 与 executor 并发读取/写入；RealMan 应由 executor owner thread 或 realtime state-push/cache 提供 `RobotStateSource`。当前 deployment 直接把 backend 作为 state source，必须在真实 RM75 上确认 SDK 约束后再定型。
5. **旧兼容面被当前工作树删除。** 根层 `main.py`、`realman_teleop.py`、`robot_backend.py`、`parse_vr.py`、`WebRTC_udp.py`、`dataset.py`、`tactile*.py` 等，以及 `deprecated/`、`dataset_tool/`、`test_tool/` 当前都是 deletion。`docs/migration_v2.md`/README 选择了“removed roots unsupported”，但这会破坏 v1 外部 import 和脚本调用；若本轮要保留 v1 workflow，应单独加薄 wrapper/迁移命令并为其写测试。
6. **真实媒体/存储/设备仍未验。** RealSense、BLE4、UR/RealMan SDK、aiortc/PyAV/NVENC、Unity RTP、h5py/LeRobot 共享存储、真实网络丢包和 stop/reconnect 仍是 release checklist 的未完成项。
7. **配置存在现场漂移。** 当前 `configs/default.yaml` 和 RealMan profile 含实际局域网地址，与文档声称的安全默认值不一致；不能把这个工作树的配置直接复制到新 PC 或提交到公共仓库。
8. **新增 policy/Beaver 不是通用 teleop。** 它们是未跟踪、依赖外部包和特定 RM75/数据的评估路径；可作为 v1 policy/eval 参考，但不要让 Unity teleop 的基础依赖强制加载它们。

## 6. 给主架构/集成的建议

- 把 v2 当作“端口、数据模型、协议和安全算法参考”，先固定一个干净提交；逐项决定哪些当前未跟踪实现（尤其 `deployment.py`、Beaver、policy evaluation、Matplotlib）进入基线。
- PC 端按 `core → devices/robots → teleop → runtime` 依赖方向组合：VR/相机/触觉/wrench 各自 latest-only；视频编码/传输、状态发送、可靠命令、录制导出和可视化作为有界 worker/extension；控制循环只消费快照并向唯一 robot owner executor 提交安全 action。
- 先冻结协议 fixture，再写 Unity C#：VR binary v2、state v1、command JSON/ACK、RTP packetizer 各建立 Python↔C# golden tests；日志必须记录 sequence gap、stale、ACK timeout、codec/ICE 和 stop reason。
- 用 MockRobot/MockVR/MockCamera/MockTactile/MemoryRenderer 做无硬件 full-session；再做 localhost UDP/WebRTC loopback；最后按 `tests/hardware/README.md` 的 supervised matrix 做设备验收。每一步都覆盖 watchdog HOLD、STOP、断线、重连、重复命令和有界队列丢帧。
- 若产品仍需 v1 根脚本/数据工具，保留明确的兼容层和迁移说明；若决定破坏兼容，则把当前 deletion 与新 CLI 的行为差异写成验收清单，避免把未提交工作树误认为已完成迁移。
- 将真实地址、串口、数据集路径和运行时开关移出默认配置；默认配置应是可导入、可 mock、无硬件副作用的模板。完成 Unity + network E2E、真实存储和硬件 stop 验证前，不应把 v2 标为 release candidate。
