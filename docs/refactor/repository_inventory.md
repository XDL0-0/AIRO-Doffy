# AIRO-Doffy v2.0 仓库清单

## 1. 审计边界

- 源仓库：`G:\Projects\AIRO-Doffy`
- 审计分支：`realman-arm-only`
- 审计提交：`56dcaa23b0ef518338875b8a7fceb5f70effbe53`
- Git tracked 文件：62 个，其中 Python/配置/文档 55 个，PNG 评估图 7 个。
- 审计方法：`git ls-files`、逐文件静态阅读、AST/结构提取、内部 import 反向索引及入口副作用检查。
- 本阶段没有修改任何旧仓库实现文件。

`AIRO_Doffy_v2.0_PLAN_TASK.md` 与 `.understand-anything/` 在源仓库中均为 untracked：前者是本次审计依据，后者是用户明确授权生成的分析产物，不计入下述 62 个 tracked 文件。

风险含义：

- 低：纯移动或文档/资源调整，行为面窄。
- 中：存在公开导入、CLI、文件格式或可选依赖兼容面。
- 高：涉及机器人运动、网络协议、实时线程、数据集写入或多个职责的拆分。

“包装”表示迁移期间是否应保留旧 import/CLI 路径的薄兼容层；包装层只转发和发出弃用告警，不应继续承载实现。

## 2. 根目录运行时与公共模块

| 路径 | 目的与主要类/函数 | 已知仓内 importer；运行参与 | 分类 | 建议去向 | 风险；包装 |
|---|---|---|---|---|---|
| `__init__.py` | 空根包标记，无符号。 | 无；仅影响把仓库根当作 Python 包的外部调用。 | legacy/module | `src/airo_doffy/__init__.py`；根文件最终移除。 | 低；短期是 |
| `camera_udp.py` | `CameraUDPManager`；RealSense 枚举/采集、裁剪缩放、JPEG 编码分片、UDP VR/录制/分辨率/触觉通道及共享状态。 | `main.py`、`test_tool/camera_test.py`；`VIDEO_TRANSPORT=udp` 时核心运行路径。 | module | 拆到 `devices/cameras/realsense.py`、`streaming/video/frame_processor.py`、`streaming/video/legacy_jpeg_udp.py`、`streaming/state/udp_transport.py`、`devices/vr/receiver.py`。 | 高；是 |
| `config.py` | `Config`；集中保存并校验机器人、网络、相机、数据集、力觉、触觉、推理参数，还选择坐标变换和初始关节。 | `camera_udp.py`、`dataset.py`、`inference.py`、`main.py`、`realman_teleop.py`、`robot_teleop.py`、多个工具/测试、`WebRTC_udp.py`；几乎所有运行路径。 | module | `config/models.py`、`config/loader.py`、`config/factories.py` 与 `configs/*.yaml`。 | 高；是 |
| `data_schema.py` | `DataSchema`、`normalize_data_type`、`build_data_schema` 等；定义 qpos/TCP/delta-TCP 状态动作维度和别名。 | `dataset.py`、`inference.py`、`robot_teleop.py`；记录与策略契约。 | module | `recording/schema.py`；与通用 sample 类型共享的枚举可进 `core/types.py`。 | 高；是 |
| `dataset.py` | `DatasetRecorder`；内存采样、HDF5/LeRobot 写入、episode 编号、Hub 推送、回滚及元数据/视频修复。 | `main.py`；数据采集主路径。 | module | 拆到 `recording/episode.py`、`recorder.py`、`hdf5_writer.py`、`lerobot_writer.py`、`rollback.py`。 | 高；是 |
| `force_filter.py` | `WrenchFilter`；力/力矩 deadband、滑动平均、EMA 低通和 reset。 | `realman_teleop.py`、`robot_teleop.py`、`test_tool/ForceVisualize.py`；遥操作与诊断。 | module | `devices/wrench/filters.py`。 | 中；是 |
| `inference.py` | 策略/processor 加载、观测张量构建、相机/机器人/触觉控制、暂停恢复及实时入口；主要含 `load_pretrained_policy`、`image_to_tensor`、`main` 等。 | `test_tool/eval_with_datasets.py` 复用部分函数；直接 CLI 运行。 | app/module | 拆到 `policies/inference.py`、`runtime/inference_session.py`（目标树需新增）与 `apps/inference.py`。 | 高；是 |
| `main.py` | `main`、采集/触觉桥/可视化/导出循环及 `TactileDataHolder`；组装所有设备并管理 episode。 | 无仓内 importer；主要采集 CLI。 | app | `apps/collect.py`、`runtime/data_collection_session.py`、`runtime/lifecycle.py`；共享 holder 改 typed buffer。 | 高；是 |
| `parse_vr.py` | `detect_packet_type`、`parse_data`、`parse_hand_data`；解析控制器旧/新文本、手部文本和 `HB` Base64 二进制。 | `camera_udp.py`、`WebRTC_udp.py`、`test_tool/vr_data.py`；所有 VR 输入。 | module | `devices/vr/protocol.py`、`controller_types.py`、`hand_types.py`。 | 高；是 |
| `realman_teleop.py` | `CanfdCommandLoop`、`RealManTeleop`、快照 dataclass、相机/可视化循环及 `main`；含 CAN-FD 高频发送、映射、安全、状态推送和生命周期。 | `tests/test_realman_teleop_loop.py`；RealMan CLI。 | app/module | 拆到 `robots/realman.py`、`robots/executors.py`、`teleop/mappings/*`、`teleop/safety/*`、`runtime/teleop_session.py`、`apps/teleop.py`。 | 高；是 |
| `robot_backend.py` | `RobotBackend`、UR/RealMan/torque adapter、`FastRobotiq2F85`、工厂及命令结果。 | `inference.py`、`realman_teleop.py`、`robot_teleop.py`；机器人构造与命令核心。 | module | 拆到 `robots/base.py`、`ur.py`、`realman.py`、`executors.py`、`grippers/{base,robotiq_2f85}.py`。 | 高；是 |
| `robot_teleop.py` | `RobotTeleop`；控制器/手追踪映射、参考姿态、Ruckig、边界/碰撞安全、力补偿、夹爪、状态快照。 | `main.py`、`ur_teleop.py`；UR 采集主路径。 | module | 拆到 `teleop/actions.py`、`mappings/*`、`transforms/*`、`safety/*`，硬件执行留在 `robots/*`。 | 高；是 |
| `tactile.py` | `MagtouchIliasSerialReaderConfig`、`MagtouchIliasSerialReader`；旧 41-taxel 串口采集、基线、Kalman、DDS 发布。构造即打开串口并阻塞校准。 | 无静态仓内 importer，但可能有外部用户；独立硬件 CLI。 | legacy | `deprecated/tactile/magtouch_ilias_41taxel.py`。 | 中；是，至少一个发布周期 |
| `tactile_4point.py` | `FourPointTactileBleReader`、`_PanelVisualizerHandle`；4-taxel BLE、重连、校准、解绕、deadband/EMA/Kalman、DDS、共享对象和 GUI 子进程。 | 由 `main.py`/诊断代码延迟导入；采集默认触觉路径。 | module | 采集到 `devices/tactile/magtouch_ble4.py`，接口到 `devices/tactile/base.py`，面板到 `visualization/*`。 | 高；是 |
| `udp_comms.py` | `UdpComms`；双向 UDP、后台接收、有界 `deque(maxlen=128)` 及旧 API 别名。 | `camera_udp.py`、`WebRTC_udp.py`、`test_tool/vr_data.py`。 | module | `streaming/state/udp_transport.py` 或更通用的 `streaming/transports/udp.py`；协议不得放入该层。 | 高；是 |
| `ur_teleop.py` | 兼容重导出 `robot_teleop`，但还请求不存在的 `URTeleop`，当前可能导入失败。 | 无直接 importer；旧外部导入面。 | legacy | 保留为根兼容 shim，转发到 `apps/teleop.py`/新 session 后弃用。 | 中；是 |
| `utils.py` | FK、关节/位姿/自碰撞安全、插值、四元数连续化、低通与 `GravityCompensator`；导入时还配置全局日志。 | `config.py`、两种相机 manager、数据集、推理、遥操作、后端和多个工具，共 13 个 importer。 | module | 分到 `teleop/transforms/*`、`teleop/safety/*`、`devices/wrench/compensation.py`、`robots/ur_kinematics.py`、`core/math.py`；移除 import-time logging。 | 高；是 |
| `visualizer.py` | `TeleopSample`、`VisualizerHandle`、`TeleopDashboard`、触觉 panel；独立进程消费快照，但导入时设置 `MPLCONFIGDIR`/backend。 | 运行端多为延迟导入；由 `main.py`、`realman_teleop.py`、诊断脚本启动。 | module | `visualization/models.py`、`publisher.py`、`dashboard.py`。 | 高；是 |
| `visualizer_config.py` | `VisualizerConfig`；enabled、刷新率、窗口和力图范围校验。 | `main.py`、`realman_teleop.py`、`robot_teleop.py`、测试、`visualizer.py`。 | module/config | 合入 `config/models.py` 的独立 immutable `VisualizationConfig`。 | 中；是 |
| `WebRTC_udp.py` | `RealsenseCameraTrack`、`WebRTCUDPManager`；RealSense、视频轨、aiohttp signaling、peer/DataChannel，同时重复 UDP VR/录制/触觉状态。 | `main.py`、`test_tool/camera_test.py`；当前默认视频路径。 | module | 拆到 camera source、frame processor、`video/webrtc_transport.py`、`state/*`、`commands/*`、VR receiver。 | 高；是 |

## 3. 策略实现

| 路径 | 目的与主要类/函数 | 已知仓内 importer；运行参与 | 分类 | 建议去向 | 风险；包装 |
|---|---|---|---|---|---|
| `policies/__init__.py` | 本地策略包标记。 | 无；包导入。 | module | `src/airo_doffy/policies/__init__.py`。 | 低；是 |
| `policies/forceflowpp/__init__.py` | 重导出 `ForceFlowPPConfig`、`ForceFlowPPPolicy`。 | 外部 LeRobot/训练代码可能导入。 | module | `src/airo_doffy/policies/forceflowpp/__init__.py`。 | 中；是 |
| `policies/forceflowpp/configuration_forceflowpp.py` | `ForceFlowPPConfig`；时序、DiT、接触先验、训练超参及 feature 校验。 | ForceFlow++ model、processor、包入口。 | module | 同名移动到新 package。 | 中；是 |
| `policies/forceflowpp/contact_labeling.py` | `force_sequence_to_phase`；从力序列生成接触阶段。 | `modeling_forceflowpp.py`。 | module | 同名移动，或改为 `contact.py` 并保留导入 alias。 | 中；是 |
| `policies/forceflowpp/modeling_forceflowpp.py` | `ForceFlowPPPolicy`；多模态编码、接触先验、DiT、flow-matching 训练和推理。 | 包入口；推理/训练由外部加载。 | module | 同名移动到 `src/airo_doffy/policies/forceflowpp/`。 | 高；是 |
| `policies/forceflowpp/modules/__init__.py` | ForceFlow++ 内部模块命名空间。 | 包内部。 | module | 同名移动。 | 低；否 |
| `policies/forceflowpp/modules/adaln_dit.py` | `SinusoidalPosEmb`、`ForceAdaLNDiTBlock`、`ForceFlowPPDiT`。 | `modeling_forceflowpp.py`。 | module | 同名移动。 | 中；否 |
| `policies/forceflowpp/modules/encoders.py` | `SmallImageEncoder`、`ForceFlowPPObservationEncoder`。 | `modeling_forceflowpp.py`。 | module | 同名移动。 | 中；否 |
| `policies/forceflowpp/modules/flow_matching.py` | `ForceFlowMatchingObjective`；训练目标与 Euler 采样。 | `modeling_forceflowpp.py`。 | module | 同名移动。 | 中；否 |
| `policies/forceflowpp/prior_library.py` | `ContactPriorLibrary`、`uniform_prior_weights`；接触模式高斯先验与 KL。 | `modeling_forceflowpp.py`。 | module | 同名移动。 | 中；否 |
| `policies/forceflowpp/processor_forceflowpp.py` | `make_forceflowpp_pre_post_processors`；LeRobot 前后处理链。 | 外部训练/推理与配置模块。 | module | 同名移动。 | 中；是 |

目标计划原树只列 `policies/inference.py` 和 `evaluation.py`，会遗漏已经存在且独立的 ForceFlow++ 实现；新树必须保留 `policies/forceflowpp/`，并增加单独的策略迁移阶段。

## 4. 数据集工具

| 路径 | 目的与主要类/函数 | 已知仓内 importer；运行参与 | 分类 | 建议去向 | 风险；包装 |
|---|---|---|---|---|---|
| `dataset_tool/convert_delta_tcp_dataset.py` | `convert_dataset`、`compute_episode_actions`；把 LeRobot 数据转换为 delta-TCP action，重建 features/episodes并可推 Hub。 | `SG_filter_for_action.py` 复用；离线 CLI。 | script | `scripts/dataset/convert_delta_tcp.py`，可复用算法下沉 `recording/converters.py`。 | 高；是 |
| `dataset_tool/convert_openpi_way.py` | `load_act_dataset`、`main`；ACT HDF5 图像/状态转 LeRobot/OpenPI 风格并可推 Hub。 | 无 importer；离线 CLI。 | script | `scripts/dataset/convert_openpi.py`。 | 高；是 |
| `dataset_tool/replay_hdf5_episodes.py` | `data_process` 及顶层回放；读取 HDF5 并直接连接/驱动机器人。 | 无 importer；硬件 CLI，且 import 有副作用。 | script | `scripts/dataset/replay_hdf5.py`，入口加 `main` guard，执行依赖注入。 | 高；是 |
| `dataset_tool/replay_lerobot_episodes.py` | `EpisodeVideoReader`、`VisualizationSink` 及 1100+ 行加载、视频、标注、可视化和机器人回放。 | 无 importer；离线/硬件 CLI，顶层执行。 | script | 拆为 `scripts/dataset/replay_lerobot.py`、`recording/readers.py`、`visualization/replay.py`。 | 高；是 |
| `dataset_tool/SG_filter_for_action.py` | `smooth_dataset`、`smooth_episode_actions`；Savitzky–Golay 平滑并重写 LeRobot 数据集。 | 无 importer；会导入转换工具；离线 CLI。 | script | `scripts/dataset/smooth_actions.py`，共享转换函数下沉。 | 高；是 |
| `dataset_tool/tag_HF.py` | 顶层调用 Hugging Face API 给远程 repo 打 tag。 | 无 importer；导入即可能修改远端。 | script | `scripts/dataset/tag_hub.py`，必须 `main` guard、dry-run 和显式确认。 | 高；否 |
| `dataset_tool/Visualize_hdf5_episodes.py` | `load_hdf5`、`save_videos`、`visualize_joints`、`main`；HDF5 检查和视频/曲线输出。 | `replay_hdf5_episodes.py` 导入。 | script | `scripts/dataset/visualize_hdf5.py`，纯读取函数可进 `recording/readers.py`。 | 中；是 |
| `dataset_tool/Visualize_tactile_lero.py` | `TactileVisualizer`、`EpisodeSampler`、`visualize_dataset`；Rerun 显示 LeRobot 图像、动作、状态和 41 点触觉。 | 无 importer；离线 CLI。 | script | `scripts/dataset/visualize_tactile.py`。 | 中；是 |
| `dataset_tool/Visualize_torque_data.py` | `plot_torque_log`；读取并绘制 torque 数据。 | 无 importer；离线 CLI。 | script | `scripts/dataset/visualize_torque.py` 或 `scripts/diagnostics/`。 | 低；是 |

## 5. 测试与诊断工具

| 路径 | 目的与主要类/函数 | 已知仓内 importer；运行参与 | 分类 | 建议去向 | 风险；包装 |
|---|---|---|---|---|---|
| `test_tool/__init__.py` | 空工具包标记。 | 无。 | legacy/module | 随工具迁移后移除。 | 低；否 |
| `test_tool/camera_test.py` | `main`；按配置创建 UDP/WebRTC manager，启动真实相机/网络并检查 RGB 帧。 | 无 importer；人工硬件 smoke test。 | script | `tests/hardware/test_camera_stream.py` 与 `scripts/diagnostics/camera_stream.py` 分离。 | 高；是 |
| `test_tool/eval_with_datasets.py` | checkpoint/数据集解析、逐帧推理、统计和对比图。 | 无 importer；离线评估 CLI，复用 `inference.py`。 | script | `policies/evaluation.py` + `apps/evaluate.py`。 | 高；是 |
| `test_tool/ForceMode.py` | 顶层连接 UR、移动初始位、进入 `forceMode` 并等待输入。 | 无 importer；导入即操纵实机。 | script | `scripts/diagnostics/ur_force_mode.py`，必须 main guard、确认和安全退出。 | 高；否 |
| `test_tool/ForceVisualize.py` | `URForceSource`、`ReceiveOnlyRobot`、mock/tactile/相机/自由拖动/TCP 实验和仪表板。 | 无 importer；人工诊断 CLI。 | script/example | 数据源接口下沉 `devices/wrench/`，CLI 到 `scripts/diagnostics/force_visualizer.py`。 | 高；是 |
| `test_tool/freedrive.py` | 顶层连接固定 IP UR、进入 teach mode、等待回车并退出。 | 无 importer；导入即操纵实机。 | script | `scripts/diagnostics/ur_freedrive.py`，main guard 和 finally 强制退出 teach mode。 | 高；否 |
| `test_tool/TEST_csv.py` | `plot_all_joints`；读取 `torque_log.csv` 并画 3×6 曲线。 | 无 importer；离线 GUI。 | script | `scripts/diagnostics/plot_torque_csv.py`。 | 低；是 |
| `test_tool/test_pt2.py` | `sim_interp_and_ff`；导入时直接运行插值/二阶前馈仿真，可能被 pytest 误收集。 | 无 importer；数值实验。 | example | 纯函数转 `tests/unit/test_trajectory_filter.py`，演示转 `scripts/diagnostics/trajectory_sim.py`。 | 中；否 |
| `test_tool/vr_data.py` | `VRDataReceiver`、`HandVisualizer`；无相机/机器人接收 VR 数据并终端或 3D 显示。 | 无 importer；人工协议诊断 CLI。 | script | `scripts/diagnostics/vr_input.py`，复用新 VR receiver/protocol。 | 中；是 |
| `tests/test_realman_teleop_loop.py` | 3 个 `TestCase`、28 个测试；fake SDK/backend 验证 CAN-FD 频率、所有权、映射、状态、生命周期和配置。 | 测试发现入口；无实机。 | test | 拆为 `tests/unit/test_canfd_executor.py`、`tests/integration/test_realman_session.py`、`tests/unit/test_config_defaults.py`。 | 中；否 |

`test_tool/` 中的 8 个可执行脚本不是自动化测试；仓库级 `pytest` 还会误收集并执行 `test_tool/test_pt2.py` 的顶层仿真。当前可靠发现范围应显式限制为 `tests/`。

## 6. 配置、文档与评估资产

| 路径 | 目的与主要内容 | 已知仓内 importer；运行参与 | 分类 | 建议去向 | 风险；包装 |
|---|---|---|---|---|---|
| `.gitignore` | 忽略 Python、数据集、checkpoint、日志及本地产物。 | Git 行为。 | config | 保留根目录并补充构建/知识图谱策略。 | 低；否 |
| `pyrightconfig.json` | 指向硬编码 Linux Conda 环境和根导入路径。 | Pyright；不参与运行。 | config | 根目录重写为 `src` layout、无机器路径配置。 | 中；否 |
| `README.md` | 安装、配置、运行、协议与目录说明；部分命令/默认值已漂移。 | 人员入口。 | documentation | 根目录重写，旧行为保存在 migration notes。 | 中；否 |
| `reminder.md` | 一条 `sensor_comm_dds` MagTouchRaw0 可视化命令。 | 人工诊断。 | documentation | `docs/diagnostics/tactile.md`，合并后删除原文件。 | 低；否 |
| `requirements.txt` | 未锁定运行依赖；漏列 `aiohttp` 及多个工具依赖，可选项边界不清。 | 安装入口。 | config | `pyproject.toml` 核心依赖 + `video`、`ur`、`realman`、`tactile`、`policy`、`dev` extras；保留短期 shim。 | 高；是 |
| `eval_dataset_plots/episode_000000_policy_vs_dataset.png` | episode 0 策略与数据对比图。 | 不参与运行。 | asset | `docs/assets/evaluation/` 或外部 artifact，不装入 wheel。 | 低；否 |
| `eval_dataset_plots/episode_000001_policy_vs_dataset.png` | episode 1 对比图。 | 不参与运行。 | asset | 同上。 | 低；否 |
| `eval_dataset_plots/episode_000002_policy_vs_dataset.png` | episode 2 对比图。 | 不参与运行。 | asset | 同上。 | 低；否 |
| `eval_dataset_plots/episode_000003_policy_vs_dataset.png` | episode 3 对比图。 | 不参与运行。 | asset | 同上。 | 低；否 |
| `eval_dataset_plots/episode_000004_policy_vs_dataset.png` | episode 4 对比图。 | 不参与运行。 | asset | 同上。 | 低；否 |
| `eval_dataset_plots/episode_000005_policy_vs_dataset.png` | episode 5 对比图。 | 不参与运行。 | asset | 同上。 | 低；否 |
| `eval_dataset_plots/episode_000006_policy_vs_dataset.png` | episode 6 对比图。 | 不参与运行。 | asset | 同上。 | 低；否 |

## 7. 运行入口与副作用

### 正式/准正式入口

- `main.py`：数据采集与遥操作；默认会创建 WebRTC/RealSense、机器人、触觉和 visualizer 相关对象。
- `realman_teleop.py`：RealMan CAN-FD 遥操作；README 说“无数据集/夹爪/触觉”，但默认 `Config.ROBOT_TYPE` 实际是 `ur3e`，直接运行前必须显式校验配置。
- `inference.py`：实机策略推理；当前先创建并移动机器人，后加载 policy，错误 checkpoint 也可能在加载失败前引发运动。
- `tactile.py`、`tactile_4point.py`：触觉硬件诊断入口。
- `dataset_tool/*.py`、`test_tool/*.py`：离线或硬件脚本；不得把它们整体当成测试套件。

### 导入即有危险副作用

- `test_tool/ForceMode.py`：连接和移动 UR、进入 force mode。
- `test_tool/freedrive.py`：连接 UR 并进入 teach mode。
- `dataset_tool/replay_hdf5_episodes.py`：连接并回放实机轨迹。
- `dataset_tool/replay_lerobot_episodes.py`：解析/下载数据并可能连接回放。
- `dataset_tool/tag_HF.py`：修改远程 Hugging Face 仓库 tag。
- `test_tool/test_pt2.py`：执行顶层仿真并修改日志配置，虽不触碰硬件但会污染测试发现。
- `utils.py` 修改全局 logging；`visualizer.py` 修改 Matplotlib 环境/backend。

### 构造即有硬件或全局副作用

- `RobotTeleop`、`RealManTeleop` 与 backend 工厂会连接/初始化机器人；部分路径移动到初始位。
- `MagtouchIliasSerialReader` 构造即打开串口并同步采集 320 个校准包。
- `FourPointTactileBleReader` 构造会调用外部基类、临时替换 `atexit.register` 并重配全局 loguru；运行时才连接 BLE/启动 GUI。
- 两个 camera manager 构造/启动负责相机枚举、socket 和线程，不是纯值对象。

## 8. 重复职责和热点

- `camera_udp.py` 与 `WebRTC_udp.py` 重复：相机发现/创建、裁剪缩放、socket 分配、VR 接收解析、录制指令、分辨率/zoom 控制、触觉发送、共享可变字段、线程 start/close。二者真正应不同的只有视频 transport/encoding。
- `robot_teleop.py` 与 `realman_teleop.py` 重复：VR 参考姿态、控制映射、安全限幅、wrench/visualizer payload、生命周期；RealMan 的 CAN-FD executor 与状态 push 是后端特有部分。
- `tactile.py` 与 `tactile_4point.py` 共享 acquisition/calibration/filter/publish 概念，但硬件和形状不同；应共享接口与 sample，不共享一个大 reader。
- `inference.py` 与 `test_tool/eval_with_datasets.py` 重复 checkpoint/processor、feature/image tensor 和 timing 逻辑。
- 数据集转换/回放/可视化脚本重复 episode 枚举、feature 提取、视频读取和 LeRobot 版本兼容代码。

## 9. 清单验收

- [x] 62/62 tracked 文件均列入。
- [x] 所有根目录 Python 文件均列入。
- [x] `tactile.py`、`tactile_4point.py` 及触觉可视化/配置均识别。
- [x] 正式 CLI、硬件诊断入口和 import-time 副作用均识别。
- [x] UDP/WebRTC、遥操作、触觉、策略与数据脚本的重复职责已记录。
- [x] 每个文件都有分类、目标、风险和兼容包装建议。
- [x] 源仓库实现文件未修改。
