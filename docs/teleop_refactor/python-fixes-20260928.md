# Teleoperation 修复与验收（2026-09-28）

> Public copy: local paths, device identifiers and site addresses are anonymized; recorded test results and version information are retained.

本轮修复了 [先前审计](python-audit-20260928.md) 复现的录制死锁、并发命令丢失、LeRobot 撤销后统计错误、配置未传递、启动失败资源泄漏及 UDP 接收线程不退出问题，并通过真机验证发现和修正了状态推送周期单位错误。Python 相关测试 **232 项通过，另有 3 个参数子测试通过**。手环版已构建并安装为 **0.9.4 / versionCode 13**，包名仍为独立的 `org.airolab.doffy.bracelet`。

## 修复内容

| 问题 | 实现与验收 |
|---|---|
| UDP 录制控制重复获取相机普通 Lock，导致 Start / Undo 卡住 | 独立的 `doffy_teleop/recording_control.py` 持有录制状态锁；两个生产 manager 直接共享它。使用真实 manager、普通 Lock 和临时 HDF5 验证导出与 Undo 完成，暂停状态恢复。 |
| 导出/撤销过程中到来的请求被旧操作清除 | Stop / Undo 使用有编号的 FIFO 请求，每次只完成所认领的请求；后到的 Start 在待处理文件操作全部结束后恢复采集。验证连续 Undo、Stop→Undo→Start、导出异常和退出排空队列。 |
| 接收线程启动时丢弃第一条录制消息 | 两种 transport 都保留启动前已收到的命令，并逐条处理接收队列。 |
| LeRobot Undo 后继续录制的归一化统计只包含新一集 | 在删除前从保留 episode 的 feature statistics 重建聚合统计；先生成可序列化的替换文件，再执行回滚。缺少统计、缺少最新集元数据、无效 data 引用时拒绝删除。移除按 episode 编号猜测文件名的回退逻辑。 |
| LeRobot 共享视频文件误删风险 | 只删除不再被保留 episode 引用的视频文件。验证共享视频保留、Undo 后续录，以及全部撤销后重录。 |
| Quest Undo 后仍显示录制中 | `RecordingController.Undo()` 清除本地 `IsRecording`，更新按钮，只发送 Undo；不会额外发送 Stop 将待丢弃内容先保存。 |
| Quest 不完整手部骨骼包与 Python 26 关节协议不一致 | H / HB 发送前一次性验证并读取完整的 26 个关节；不完整帧不发送。 |
| WebRTC 与 Classic robot 使用各自的默认配置 | 从入口传递同一个配置实例；兼容原来的简单 factory 接口，并修正 Classic 循环使用实际配置的控制频率。 |
| Classic 或 media 初始化中途失败时已创建资源泄漏 | 初始化纳入清理范围；相机创建或 socket 分配失败时关闭此前成功创建的资源。UDP / WebRTC 共用 socket allocator。 |
| 退出时排队操作失败会跳过剩余请求和数据集关闭 | 记录每次操作错误并继续排空已接受的请求；最终保存即使失败也执行数据集关闭，清理后传播首个错误。后台退出失败会写入错误状态。 |
| UDP close 后接收线程继续运行 | stop event、有限接收超时、socket 关闭及有界 join；重复 close 可用，非法 UTF-8 不杀死接收线程。 |
| PC 对机器人和 Quest 使用不同网卡 | 新增 `REALMAN_STATE_PUSH_IP`，从 Quest 的 `PC_IP` 中分离机器人遥测接收地址；设为 None 时兼容回退。 |
| 实时状态推送周期单位错误 | 配置保持毫秒，SDK 边界按 `cycle_ms // 5` 转换；5 / 10 / 25 ms 分别发送 SDK 值 1 / 2 / 5，启用和关闭路径一致。真机复测见下文。 |
| LeRobot / WebRTC 依赖冲突 | 新建经过验证的 teleop venv，固定相关兼容版本，`pip check` 通过；原共享 Conda 环境保持不变。 |

LeRobot 统计回归测试实际保存值 0、10，撤销 10，再保存 20；结果保留两帧，state/action 的 count=2、mean=10、min=0、max=20、std=10。测试同时覆盖 Beaver feature 和视频统计。HDF5 / LeRobot 删除验证全部使用临时目录。

推送周期问题来自本轮真机验收：传入 SDK `cycle=5` 时实测约 40 Hz。[RealMan 官方协议](https://develop.realman-robotics.com/robot4th/json/udpConfig/) 明确 `cycle` 以 5 ms 为单位，因此原配置的 5 ms 应传 `cycle=1`，直接传 5 实际为 25 ms。

## 设备与网络

| 用途 | 当前地址 |
|---|---|
| PC 与 Quest 通信，`PC_IP` | `192.0.2.10` |
| Quest，`VR_IP` | `192.0.2.20` |
| PC 机器人网卡，`REALMAN_STATE_PUSH_IP` | `192.168.1.100` |
| RealMan API | `192.168.1.18:8080` |

- Quest 到 PC 的 ping、实际 TCP 连接和 UDP 标记包均成功。
- 从 Quest 经真实网络向生产 `UDPManagerCore` 发送 Start / Stop / Undo，连接真实 `DataRecordingService` 和临时 HDF5：两帧保存成功，Undo 删除刚保存的 episode，计数归零，状态和暂停标志恢复，接收线程退出。
- RealMan SDK 建立连接并读取当前 7 关节与 TCP 状态成功，返回码为 0，随后正常关闭句柄。
- RealMan → PC `192.168.1.100:8098` 实时推送：修正前 `cycle=5` 得到 63/63 有效回调，39.999 Hz；修正后 `cycle=1` 得到 321/321 有效回调，200.005 Hz。回调来自机器人地址，`errCode=0`，7 关节及 TCP xyz/RPY 均有效。两次均在测试前备份推送配置，测试后恢复并逐字段读回核对一致；句柄删除和 SDK 销毁返回 0，端口无遗留监听。
- 先前审计已验证真实 RealSense D435 的 640×480 RGB 取帧及线程退出；本轮未重复相机取帧验证。

这里的 Quest 录制验证通过 ADB 在头显上发送命令，使用模拟帧数据；没有把它称为佩戴头显点击 UI 的完整验收。没有执行机器人复位、运动、夹爪或灵巧手动作。Beaver / BrainCo / BLE 的真实硬件联动仍待现场验证。

遥测前后实测结果分别保存在本机 `/tmp/doffy-python-audit/realman-push-before-result.json` 与 `/tmp/doffy-python-audit/realman-push-result.json`。

## Python 验证

环境为 `.venv`；具体依赖、重建方法及继承范围见 [环境说明](teleop-environment.md)。该环境继承本机已安装的 SDK / Torch 等包，约束文件是本机兼容性配置，并非跨平台完整 lockfile。

```bash
HF_HUB_OFFLINE=1 MPLBACKEND=Agg \
DOFFY_UNITY_PROJECT=/path/to/unity/CodexBracelet \
./.venv/bin/python -m pytest -q \
  tests/test_realman_teleop_loop.py tests/test_wrm_akm.py \
  tests/test_realsense_camera.py tests/test_vr_coordinate_mapping.py \
  tests/test_teleop_refactor_imports.py tests/test_teleop_refactor_boundaries.py \
  tests/test_brainco_hand.py tests/test_wrm_visualizer.py tests/test_visualizer_layout.py \
  tests/test_teleop_media_protocol.py tests/test_teleop_media_udp_loopback.py \
  tests/test_teleop_media_webrtc.py tests/test_udp.py tests/test_webrtc_csharp_source.py \
  tests/test_beaver_recording.py tests/test_dataset_recording_status.py \
  tests/test_realman_teachcollect.py tests/test_realman_recollect.py \
  tests/test_replay_realman_lerobot.py tests/test_dataset_rollback.py \
  tests/test_recording_commands.py tests/test_udp_lifecycle.py \
  tests/test_runtime_lifecycle.py tests/test_media_startup_cleanup.py
# 232 passed, 3 subtests passed in 17.22s
```

结果 XML 保存在本机 `/tmp/doffy-python-audit/fixed-teleop-tests.xml`。这组测试包含真实 localhost UDP / WebRTC、真实临时文件和 LeRobot API；机器人运动使用 fake backend。会操纵硬件的人工测试脚本没有混入自动测试。

## Quest 构建与安装

- 项目：`/path/to/unity/CodexBracelet`。
- APK：`/path/to/unity/App_output/AIRO_Doffy_bracelet.apk`。
- 旧版保存至：`/path/to/unity/App_output/bracelet-v0.9.3/AIRO_Doffy_bracelet.apk`。
- C# 验证：18 项录制/手部发送检查、10 项追踪保护检查、47 项协议/生命周期检查；97 个 runtime 与 3 个 editor 源文件编译通过，Unity Android build 成功。
- `adb install -r` 成功，设备 package 信息确认版本 0.9.4 / 13。

应用启动后 OpenXR 初始化，但很快进入暂停，未到达 `AppManager.Start()` 日志。因此没有确认当前应用保存的服务器 IP，也没有完成佩戴时的 Unity UI→Python→机器人闭环。使用时请在头显设置中确认目标 PC IP 为 `192.0.2.10`。

## 使用与边界

从仓库目录启动：

```bash
cd /path/to/AIRO-Doffy
./.venv/bin/python realman_teleop.py
```

Undo 语义保持：当前有未保存帧时丢弃本次录制；否则删除最后一个已保存 episode。若 LeRobot episode 共享一个视频容器，保留集引用的容器不会删除，也不会物理裁切其中已撤销片段。

录制队列保证 Python 已接收请求的处理顺序；现有 UDP 线协议仍没有请求 ID、服务端确认及重试，不能将本地按钮状态视为磁盘保存成功的确认。实际导出/撤销结果仍以 Python 日志与数据集状态为准。

模块化方面，本轮把录制控制状态和 socket 分配集中到独立模块，统一配置传递并补齐资源生命周期。原审计建议的所有 mixin 状态显式化、相机取帧实现合并、完整可安装包与可选依赖分组仍属于后续结构改进，本轮没有进行与故障无关的大范围重写。
