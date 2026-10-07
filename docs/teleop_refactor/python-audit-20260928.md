# Python teleoperation 审计（2026-09-28）

> Public copy: local paths, device identifiers and site addresses are anonymized; recorded test results and version information are retained.

> 本文保留修复前的发现与复现记录。后续代码修复、网络恢复和验收结果见
> [修复验收记录](python-fixes-20260928.md)；下文的旧地址、失败结果和测试数量不代表修复后的状态。

本次检查当前工作区代码，范围为 Python teleoperation、共享录制服务、传感器接口，以及与当前 `CodexBracelet` Unity 协议的对应关系。没有启动真实机器人控制，没有删除现有数据集，也没有修改生产逻辑。

**结论：核心拆分与本地通信基本有效，但当前不能验收为“全部功能有效”。** 现有相关测试分三组执行，去重后 129 项通过；额外集成验证发现 UDP 录制死锁、LeRobot Undo 后统计错误等原测试未覆盖的问题，且当前真机网络尚未打通。

## 验证范围与环境

- 运行环境：`/path/to/source-env/bin/python`，Python 3.10.0。
- 测试环境：`/tmp/airo-teleop-qa/bin/python`，继承上述环境的 site-packages。
- Unity 协议来源：`/path/to/unity/CodexBracelet/Assets/Teleop`。
- 机器人控制测试使用 fake backend；录制/删除测试仅使用临时目录。
- 真硬件验证包括 Quest 网络诊断和 RealSense 取帧；没有执行机器人复位、运动、夹爪或灵巧手动作。

## 已完成验证

| 范围 | 结果 | 实际证明的内容 |
|---|---|---|
| 核心控制、映射、启动依赖注入、模块边界、BrainCo 算法、相机接口、可视化 | 100 项测试通过 | 使用模拟硬件验证逻辑；不等于真实机器人闭环通过 |
| 协议、UDP 图像分包、WebRTC 组件、UDP 基础接口、C# 源码契约 | 12 项测试通过 | 现有通信测试通过；生命周期仍有未被原测试覆盖的问题 |
| UDP 控制/手柄/手部 | 本地回环通过 | 实际 UDP socket；C、H、HB、Start/Stop/Undo、三个 zoom 端口 |
| Python → 当前 C# 状态包 | 本地回环通过 | `rightTCP` 位置、wxyz 四元数、力数据被当前 C# 解析器接受 |
| WebSocket + WebRTC 视频 | 本地回环通过 | 生产 Python 信令服务完成 hello/ack、offer/answer、真实 aiortc 连接与视频解码、stop_video 清理 |
| 当前 Unity C# 图像组帧 | 本地回环通过 | Python 发出的 37 个 JPEG 分片由当前 C# assembler 重建，图像与发送结果一致 |
| HDF5 / LeRobot Undo 正常路径 | 部分通过 | 文件/episode 元数据/编号与继续录制通过；LeRobot 聚合统计仍有明确错误，见下文 |
| 真实 RealSense D435 | 通过 | 640×480 RGB uint8，10 次采样得到 10 个不同采集时间戳，关闭后取帧线程全部退出 |
| 主要 Python 模块导入 | 通过 | 基础环境可导入 main、realman_teleop、udp、WebRTC_udp、dataset、brainco_hand、beaver |
| Quest ↔ PC 当前网络 | 未建立 | 双向 ping 无响应；Quest 到 PC 两个实际网卡地址的 TCP 探测超时，UDP 标记包均未收到 |
| RealMan API 当前网络 | 未建立 | 连接配置地址 `192.168.1.18:8080` 超时；没有发送机器人命令 |
| Beaver / BrainCo / BLE 触觉真硬件 | 未完成 | 未识别到 Beaver USB 串口，机器人 API 未连通，BLE 未实测；算法/解析测试不能代替实际硬件验证 |

本地 WebRTC 测试协商了 DataChannel，但未通过它发送 zoom payload。手部 H/HB 验证使用当前源码定义的 26 关节布局，没有执行依赖 OVR SDK 的真实手部采集器。当前 Unity 可生成不足 26 关节的包，而 Python 严格要求 26；实际设备是否出现这种帧还需实测。

## 当前环境阻塞

`config.py:11-18` 配置 PC 为 `192.0.2.16`、VR 为 `192.0.2.17`、机器人为 `192.168.1.18:8080`。实际 PC 有 `192.0.2.12/22` 和 `192.0.2.10/22`；Quest 是 `192.0.2.14/21`。

直接尝试绑定 `Config.PC_IP` 返回 **`OSError: [Errno 99] Cannot assign requested address`**。这能确定当前默认配置无法在这台电脑正常启动接收 socket。不同网段本身不代表必然不通；实际双向 ping 和 Quest 发往两个 PC 网卡的临时 TCP/UDP listener 探测也没有成功。USB ADB 已授权只证明 USB 调试链路可用。

应先恢复设备之间可路由的网络，再配置 Python 的 PC/VR 地址和 Quest 的目标 PC 地址。机器人控制器需要能访问配置的 PC 地址，才能发送实时状态推送。目前不能宣布端到端通信全部建立。

## 已复现的代码问题

### P1：默认 LeRobot 格式在 Undo 后继续录制会生成不完整统计

`dataset.py:867-870` 删除 `meta/stats.json`，却没有从保留的 episode 重建统计。当前安装的 LeRobot 在统计不存在时，会把下一次保存的 episode 统计直接作为整个数据集的统计。

用真实 LeRobot API、临时目录和 7 维 state/action 复现：每个 episode 一帧，依次保存值 0、值 10，Undo 删除值 10，再保存值 20。最终保留的是值 0 与值 20 两帧：

| 字段 | 期望 | 实际 |
|---|---|---|
| `info.total_frames` | 2 | 2 |
| state/action `stats.count` | 2 | 1 |
| state/action `stats.mean`（每维） | 10 | 20 |
| state/action `stats.min`（每维） | 0 | 20 |
| state/action `stats.std`（每维） | 10 | 0 |

所以“撤销后还能保存”和“stats.json 又出现了”不足以证明数据集正确。这会影响依赖该统计的训练归一化。应从剩余 episode 的统计或数据重建聚合统计，并加入“撤销 → 继续录制 → 比较全部保留数据统计”的回归测试。

### P1：UDP 传输模式的录制控制存在重复加锁死锁

- `realsense_camera.py:31` 创建不可重入的 `threading.Lock()`。
- `doffy_teleop/media/udp.py:124` 将它作为 manager 的锁；`:150-158` 的状态属性 setter 再次获取这把锁。
- `data_recording.py:112-145` 的 `ManagerRecordingControl` 先持有同一把锁，再写这些属性。

使用真实 `UDPManagerCore` 与不接硬件的 camera/socket stub，先通过 `_apply_record_control('Undo')` 模拟已收到 Quest 命令，再执行真实 HDF5 rollback。结果是最后一个文件已删除、计数已减少，但 `process_pending_once()` 卡在 `clear_rollback()`，线程在有界等待后仍存活，rollback flag 和 `pause_event` 都没有清除。`start_recording()` 也能复现相同死锁。

默认 WebRTC 独立创建相机时使用 RLock，未复现这项死锁；注入使用普通 Lock 的相机对象仍有同类风险。修复应明确锁的归属，通过控制状态的原子方法操作，或一致使用可重入锁，避免依赖 stub 恰好使用 RLock。

### P2：录制命令只有布尔标志，I/O 期间的新命令可能丢失

`data_recording.py:272-305` 先读请求标志，执行文件操作，再无条件清标志。通过事件屏障阻塞第一次 rollback、在期间提交第二次 Undo，最后只执行了一次删除；第二个请求被清掉。`doffy_teleop/protocol/control.py:91-106` 中 Start/Stop/Undo 也会互相覆盖标志。

应使用有顺序的命令队列或带编号的状态转换，明确执行中、完成和失败，并使 Quest 能收到处理结果。不能仅把按钮本地状态当作服务端状态。

### Python 与当前 Quest 录制状态存在源码层面的同步缺口

当前 `CodexBracelet/Assets/Teleop/Core/RecordingController.cs:32-35` 的 Undo 只发命令，不清除 `IsRecording`；会话面板在录制中仍允许 Undo（`UI/TeleopRecordPanel.cs:170-175`）。Python 收到 Undo 会停止采集，因此录制中撤销后可能仍显示 Stop recording。这个结论来自当前两端源码，未在头显交互中执行该操作。需要与 Python 命令状态/回执一起修复。

### WebRTC 路径忽略入口配置

- `doffy_teleop/runtime/publisher.py:15-24`
- `doffy_teleop/runtime/classic.py:50-54`
- `WebRTC_udp.py:90-99`

两个入口都调用 `WebRTCUDPManager()`，虽然其构造器接受 `config`。Mock 验证捕获的参数均为 `call()`，而不是传入的配置实例。因此入口临时更改 IP、分辨率等设置时，WebRTC 会重新创建默认 `Config()`，可能与控制/录制模块使用不同设置。UDP 分支已有 `config=cfg`。

Classic 的 `RobotTeleop` 也自行创建 `Config()`（`doffy_teleop/robots/legacy/runtime.py:37-46`）。建议所有组件共享入口配置实例。

### UDP 接收线程不能正常退出

`udp_comms.py:49-53` 仅关闭 socket；`:93-98` 的接收循环没有退出条件，`:84-90` 又吞掉 EBADF。隔离进程验证 `close()` 后 `_rx_thread.is_alive()` 仍为 True。应添加停止状态、使阻塞接收可退出，并等待线程结束。

### Classic 启动失败绕过清理

`doffy_teleop/runtime/classic.py:44-143` 的资源初始化位于主 `try/finally` 之前。模拟 `cu_manager.test_connection()` 抛出超时异常，验证 `cu_manager.close()` 调用次数为 **0**。相机、网络或后续设备初始化失败时，已创建资源得不到统一释放。RealMan 新入口的初始化在清理保护范围内，可作为统一生命周期的参考。

## Undo 的实际语义与正常路径验证

当前 Quest `RecordingController.Undo()` 发出字面量 `Undo` 到 UDP 8003；Python 的 socket_1 接收后设置 rollback 请求，由 Classic/RealMan 共用的 `DataRecordingService` 调用 `DatasetRecorder.rollback_last_episode()`。

- **没有 episode：** 不删除文件，计数保持 0。
- **正在录制且已有未保存帧：** 丢弃当前未保存 buffer，保留之前已经保存的 episode。
- **HDF5 已保存 episode：** 删除最后一个 HDF5 文件及对应描述行，减少计数，保留前一个文件；下次可复用编号。
- **LeRobot 已保存 episode：** 临时目录中实际写入两集带 64×48 RGB/H264 视频的数据，Undo 后 data parquet 与 episode metadata 只保留第 0 集，info 计数同步；随后成功在编号 1 继续录制。
- **共享视频文件：** 两集共享 `file-000.mp4` 时，撤销后一集会保留整段视频文件，因为第 0 集仍引用它。episode 元数据和数据行会删除，但不物理裁切共享视频中的被撤销片段。继续录制创建 `file-001.mp4`。
- **统计正确性：** 上面的正常保存/续录验证通过，但聚合统计验证失败，见 P1 问题。

这些结果来自真实文件与 LeRobot 库；锁、并发状态问题分别使用隔离模拟环境复现。真实 Quest 点击 → Python 执行 → UI 确认的完整链路因当前网络未连通而没有验收。

## 模块化与代码精简

拆分有实质效果：四个兼容入口共 277 行；审计范围内的 `doffy_teleop` 五个主要子包共 39 个 Python 文件、7,336 行，最大文件 `media/webrtc.py` 为 487 行；扫描未发现直接 import 环。职责划分已经清楚：

| 子包 | 职责 |
|---|---|
| protocol | 数据包解析、控制消息、信令格式 |
| media | 相机帧、UDP 图像传输、WebRTC |
| control | VR 映射、目标策略、CAN-FD 控制循环 |
| robots | 机器人后端、Classic 控制实现 |
| runtime | Classic / RealMan 启动、运行与退出 |

进一步精简应优先改善状态和依赖边界：

1. **统一配置和生命周期。** 当前存在上面的配置传递和启动清理缺口。
2. **明确 mixin 所需状态。** Classic 的四个 mixin、RealMan 的映射/策略/生命周期 mixin 依赖其他文件初始化大量 `self` 属性，拆文件后仍共享隐式状态。应逐步定义显式状态对象或 Protocol 接口。
3. **减少包对根目录兼容入口的反向依赖。** `runtime` 仍导入 `robot_teleop`、`udp`、`WebRTC_udp` 等 facade，以及根目录配置、录制、解析工具。可通过组合入口注入实现，保留外部兼容导入。
4. **合并有重复的组件。** `realsense_camera.py` 与 `media/camera.py` 的取帧循环重叠，但前者另有预热和时间戳历史，应保留这些行为；两处图像缩小函数、两处 dummy controller 可共享。`beaver_reader.py` 未找到引用，移除前需确认外部使用者。
5. **整理运行依赖。** 当前没有独立安装包配置，requirements 同时包括 teleop 和训练用途，且没有版本约束。应给 teleop 建立经过验证的约束/锁文件及可选硬件依赖组。

当前适合继续维护，但还不能认为模块化和精简已经完全完成。文件行数限制与兼容导入测试不能证明线程生命周期、并发录制或真实硬件行为正确。

## 依赖检查

基础 `airo-doffy` 环境的 `python -m pip check` 返回 4 项冲突：

| 使用方 | 声明范围 | 当前安装 |
|---|---|---|
| lerobot 0.4.4 → av | >=15, <16 | 17.1.0 |
| lerobot 0.4.4 → huggingface-hub | >=0.34.2, <0.36 | 1.29.0 |
| lerobot 0.4.4 → packaging | >=24.2, <26 | 26.3 |
| datasets 4.8.5 → fsspec | >=2023.1.0, <=2026.2.0 | 2026.7.0 |

主要模块能导入，不代表这些版本组合获得兼容性保证。应先保存当前可工作环境，再验证并固定兼容组合。QA overlay 还额外报告 setuptools 57.4.0 低于 LeRobot 声明要求；这项不出现在上述基础环境的检查结果中。

## 可重复的现有测试命令

```bash
MPLBACKEND=Agg /tmp/airo-teleop-qa/bin/python -m pytest -q \
  tests/test_realman_teleop_loop.py tests/test_wrm_akm.py \
  tests/test_realsense_camera.py tests/test_vr_coordinate_mapping.py \
  tests/test_teleop_refactor_imports.py tests/test_teleop_refactor_boundaries.py \
  tests/test_brainco_hand.py tests/test_wrm_visualizer.py tests/test_visualizer_layout.py
# 100 passed in 2.96s

PYTHONWARNINGS=ignore /tmp/airo-teleop-qa/bin/python -m pytest -q \
  tests/test_teleop_media_protocol.py tests/test_teleop_media_udp_loopback.py \
  tests/test_teleop_media_webrtc.py tests/test_udp.py tests/test_webrtc_csharp_source.py
# 12 passed in 13.27s

/tmp/airo-teleop-qa/bin/python -m pytest -q \
  tests/test_teleop_media_protocol.py::LegacyProtocolTests::test_control_state_preserves_record_and_zoom_semantics \
  tests/test_beaver_recording.py tests/test_dataset_recording_status.py \
  tests/test_realman_teleop_loop.py::RealManTeleopTest::test_episode_recorder_collects_cached_robot_and_camera_data \
  tests/test_realman_teleop_loop.py::RealManTeleopTest::test_episode_recorder_vr_export_and_visualizer_rollback \
  tests/test_realman_teachcollect.py::RealManTeachCollectorTests::test_force_dropout_during_replay_rolls_back_without_stopping_collector \
  tests/test_realman_teachcollect.py::RealManTeachCollectorTests::test_visualizer_commands_teach_then_replay_collect
# 20 passed；其中 3 项与前两组重复，去重后合计 129 项。
```

当前手环版源码另以 `DOFFY_UNITY_PROJECT=/path/to/unity/CodexBracelet` 运行 `tests/test_webrtc_csharp_source.py`，1 项通过，覆盖 C# 信令客户端并发发送、UTF-8 分片、重连、超大消息拒绝。

额外临时验证代码保存在 `/tmp/doffy-python-audit/comms` 与 `/tmp/doffy-python-audit/undo`，不是生产实现。C# 协议 harness 的 47 项检查通过；组合检查脚本随后在独立的 TrackingGuard 编译阶段因缺少 `OVRHand` 引用退出 1，所以没有把整条 C# 组合检查命令计为通过。当前源码的后续协议互通检查已单独通过。

人工硬件脚本 `tests/test_realman_canfd.py` 等未运行；它们会控制真实机器人，不能当作普通单元测试直接执行。

## 建议修复顺序

1. 修复 LeRobot Undo 聚合统计和 UDP 录制死锁，并增加针对真实持久化数据和普通 Lock 的回归测试。
2. 统一录制命令顺序与回执，更新 Quest 本地录制状态。
3. 修复 UDP 接收线程退出、Classic 启动失败清理及统一配置传递。
4. 恢复设备网络并做真实头显、机器人、灵巧手和传感器联调。
5. 再整理显式状态接口、共享相机实现和依赖锁定。
