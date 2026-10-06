# DOFFY 遥操作重构

Unity 新工程：`/home/yuyuan/UNITY_Project/Codex`。PC 工程：`/home/yuyuan/AIRO-Doffy`。

旧 Codex 已先归档再删除，新工程以 Classic 的 XR rig、有效场景和协议为基础重建。Classic 与 AIRO-DOFFY-v2 参考工程没有修改。Meta XR All-in-One 固定为官方 registry 核实的最新稳定版 **205.0.0**；Audio 85.0.0、Voice 85.0.1 是该版本指定的依赖，并非遗漏升级。

## 功能与界面

[Classic 完整功能盘点](classic-feature-inventory.md) 区分了 21 项功能和 7 项运行时自动创建机制，以及当前场景与历史可选功能。[架构与通信契约](architecture.md) 说明两端的模块边界、端口与坐标约定。[逐项迁移验收](feature-parity.md) 标明已验证和仍待真机验收的范围。

新空间工作台包含 Session、Cameras、Alignment、Upper limb、Display、Help 六页，底部固定会话、录制与重校准操作。数字键盘支持 PC 地址和相机端口；相机窗口可以移动、缩放、排列及关闭。新工作台支持 Meta ray/poke，保留原 rig 的手部和手柄能力。机器人反馈过期时会标出 STALE，并隐藏陈旧力箭头；该提示不等于机器人急停。

布局预览：[连接页](workspace-session.svg)、[相机页](workspace-cameras.svg)。这些 SVG 是设计预览，**不是 Unity/Quest 运行截图**。真正的工作台由 `WorkspaceShell` 在进入 Play/运行场景时构建。

## PC 结构

项目名称为 `doffy-teleop`，Python 模块名称使用下划线 `doffy_teleop`。

- `doffy_teleop/config.py`、`doffy_teleop/utils.py`：中央配置、滤波、安全检查和通用辅助函数。
- `doffy_teleop/protocol/`：Classic C/H/HB、录制、视频控制、JPEG 分片、信令信封。
- `doffy_teleop/media/`：相机帧、UDP 视频、WebRTC peer、WebSocket 信令及媒体编排。
- `doffy_teleop/sensors/`：Beaver、串口/BLE MagTouch 与六维力滤波。
- `doffy_teleop/control/`：输入映射、目标约束、CAN-FD 循环与远程 IK。
- `doffy_teleop/robots/`：机器人后端、夹爪、BrainCo 手与 Classic 控制实现。
- `doffy_teleop/recording/`：数据 schema、HDF5/LeRobot 录制服务、轨迹回放及重采集界面。
- `doffy_teleop/runtime/`：Classic、RealMan、教学采集、重采集和 BODY 查看器的启动、线程生命周期和发布；本地 `seahorse.py` 不属于上传范围。
- `doffy_teleop/visualization/`：力/触觉/相机/数据仪表板、BODY 绘图及显示配置。
- `doffy_teleop/body_visualization.py`：BODY 查看器 API 的兼容导出。

公开根目录保留五个 CLI 启动器：`main.py`、`realman_teleop.py`、`realman_teachcollect.py`、`realman_recollect.py`、`teleop_body_visualizer.py`。这些命令和参数仍可使用；库实现使用 `doffy_teleop` 内的路径，旧根库 import 需要更新。`dataset_tool/` 保留数据集工具 CLI，回放与重采集 UI 实现在 `doffy_teleop/recording/`。目录职责、迁移映射和 `python -m` 命令见[模块布局说明](../module-layout.md)。遥操作运行默认参数、录制 schema、机器人约束及已有数据格式保持。

上传范围是 UR/RealMan 遥操作、WRM、BrainCo、Beaver、VR/摄像头、BODY 可视化、dataset tools 及相关测试、协议源码和文档。Seahorse、策略训练/推理/评估、Jev 与独立 experiments 留在本地研究工作树中，不属于此次上传。

## 验证与复现

**公开复现范围限制（2026-10-07 审计）**：下方 Unity/C# 命令依赖工作站上的完整工程和缓存。公开仓库仅包含部分 C# 副本、测试与文档，未包含当前 v0.9.7 的完整 Unity 工程或 source SHA。`AIRO-DOFFY-APP` 的历史 `v0.6.0` 标签不能填补该缺口；不要将本机路径或历史验证结果作为公开 APK 可复现的证明。详见[发布关联审计](../../apk/RELEASE.md)。

本次上传子集的 Python 验收结果：**433 项测试、12 项子测试通过，3 项跳过**。测试范围按保留的 teleop、BODY、dataset tools 及相关测试筛选，未向导入路径添加外部 simulation 目录。此结果与下方历史完整工作树回归记录分开，也不替代 Unity/Quest 或机器人实机验收。

以下是历史重构专项验证的复现命令。原始验证使用工作站已有依赖环境；当时 pytest 单独安装在 `/tmp/airo-teleop-qa`，未改变现有 conda 环境。

```bash
# 生产 C# 协议、滤波及事件生命周期；SDK 事件用测试适配器触发。
python3 scripts/teleop_refactor/run_csharp_protocol_checks.py --export-dir /tmp/doffy-csharp-checks

# Core 状态、偏好持久化和 25 项异步会话时序检查。
python3 scripts/teleop_refactor/run_csharp_session_checks.py

# 双向实际 UDP：控制器数据、TCP 坐标和力。
/tmp/airo-teleop-qa/bin/python scripts/teleop_refactor/csharp_interop.py

# JPEG 乱序/丢包回环、生产 C# 接收器；显式指定才打开 RealSense。
/tmp/airo-teleop-qa/bin/python scripts/teleop_refactor/media_loopback.py \
  --csharp --harness /tmp/doffy-csharp-checks/ProtocolHarness.exe --realsense --frames 3

# 针对已有 Unity/Meta 缓存程序集的源代码/API 编译及静态引用检查。
python3 scripts/teleop_refactor/compile_unity_sources.py --include-editor
python3 scripts/teleop_refactor/audit_unity_project.py
```

历史完整工作树回归记录：698 项测试、95 项子测试通过，1 项既有策略配置/测试不一致失败（`ACTION_STEPS` 4 与预期 8）；该值在重构前快照中已存在。这份历史记录包含当前上传范围之外的本地研究测试，不能代替上方本次上传子集的验收结果。C# 专项包含 47 项协议/数学/生命周期、10 项追踪事件和 25 项异步会话检查，另通过 Core 设置/状态测试。

证据分别见 [PC](pc-refactor-validation.md)、[媒体](media-validation.md)、[独立验收复核](acceptance-review.md) 和 [Unity 集成](unity-integration-validation.md)。

## Unity / Quest 验收入口

使用 Unity **6000.5.6f1** 打开新工程，打开 `Assets/Scenes/Teleoperation.unity`，进入 Play 查看工作台。菜单 `Tools → DOFFY → Validate scene` 检查场景，`Build Quest APK` 构建 Android ARM64 / IL2CPP。

早期重构验证时，本机 Unity Editor 返回许可证缺失，且 ADB 未发现 Quest。当前发布的 [v0.9.7/code18 ARM64 APK](../../apk/AIRO_Doffy_v0.9.7_arm64_code18.apk) 已通过签名验证及 Quest 3 安装/启动检查，包含默认关闭、需在 Session 中启用的 BODY 发送代码；记录见 [BODY 验证](../body_visualization/validation.md#097-发布-apk)。Quest 头显实际界面、无线视频、射线/手部交互、真实 BODY 姿态及机器人实机验收仍未完成。源码/API 编译、本机传输与安装/启动检查不替代这些验收，未完成项不标记通过。

## 备份与已有工作

- 旧 Codex：`/home/yuyuan/UNITY_Project/.backups/codex-before-refactor-20260916.tar.gz`
- PC 工作树源码/测试基线：`/home/yuyuan/UNITY_Project/.backups/pc-teleop-baseline-20260916.tar.gz`
- Classic 继承的失效资产清理：[明细](inherited-reference-cleanup.json)，原内容另行归档。
- 重构前 git 状态保留在本机的 `pre-refactor-git-status.txt`，用于区别已有训练、评估、数据与论文改动，不随源码仓库上传。没有将这些既有修改回退或并入本次重构。
