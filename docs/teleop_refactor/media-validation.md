# PC media/通信拆分验证

日期：2026-09-16\
范围：仅修改本任务拥有的 `udp.py`、`WebRTC_udp.py`、`doffy_teleop/media/`、`doffy_teleop/protocol/`、媒体测试、回环脚本和本报告。`config.py`、`main.py`、`realman_teleop.py` 及其他根模块未改动。

## 实现

两个根文件现在是兼容薄入口，核心实现拆到了独立职责模块：

- `doffy_teleop/protocol/vr.py`：经典 `C,`、旧 controller CSV、`H,`、`HB,` 分派；`LegacyVRPacketDecoder` 可注入旧根函数，因此既共享处理逻辑，也保留 `patch("udp.parse_data")` 等测试语义。
- `doffy_teleop/protocol/control.py`：线程安全的 VR 状态、`Start`/`Stop`/`Undo`/`Rollback`/`DeleteLast` 状态转移、旧 port-key 和 WebRTC camera-key 两种 zoom 控制。属性名继续暴露为 `data`、`hand_data`、`data_collecting_state`、`data_export_state`、`data_rollback_state`、`tactile_byte` 等。
- `doffy_teleop/protocol/jpeg.py`：冻结的 `!IHHI`（12 字节）JPEG header、`JpegChunkSender`、`JpegChunkAssembler` 和真实 socket receiver。assembler 支持乱序、重复、坏包、声明大小/索引校验、in-flight 上限、缺片过期、完成 frame-id 去重及显式 `reset()`。
- `doffy_teleop/protocol/signaling.py`：严格 JSON signaling envelope，保留 `hello`/`hello_ack`/`offer`/`answer`/`ice_candidate`/`stop_video` 形状。
- `doffy_teleop/media/frames.py`：RGB→BGR、uint8 归一化和 center zoom 纯函数；UDP 与 WebRTC 共用。
- `doffy_teleop/media/camera.py`：`FrameStore`、有界职责清楚的 `CameraCaptureService`、`CameraFrameProvider` 和 `CameraVideoTrack`。采集独立于传输，帧始终从共享 latest store 读取。
- `doffy_teleop/media/udp.py`：`UDPManagerCore` 和 socket/发送/接收 worker；经典端口布局、最多五路 JPEG camera TX、VR pose、录制控制、zoom、触觉 TX 仍兼容。
- `doffy_teleop/media/webrtc_peer.py`：`WebRTCSession` 只拥有 aiortc peer、tracks、`control` DataChannel 和 trickle ICE 回调。
- `doffy_teleop/media/webrtc_signaling.py`：`WebRTCSignalingServer` 只拥有 aiohttp WebSocket signaling、session 生命周期和 `hello`/offer/answer/ICE/stop 消息。
- `doffy_teleop/media/webrtc.py`：`WebRTCUDPManagerCore` 组合 camera capture 与旧 UDP pose/record/tactile 兼容面；根入口仍从这里导出历史名称。
- 根 `udp.py`/`WebRTC_udp.py` 绑定旧模块的 `Config`、`U.UdpComms`、VR parser、RealSense factory 和 aiortc symbols，避免现有调用方被迫迁移。

## 自动化验证

使用题目指定环境 `/tmp/airo-teleop-qa/bin/python`：

```text
tests/test_udp.py                                          4 passed
tests/test_teleop_media_protocol.py                       4 passed
tests/test_teleop_media_udp_loopback.py                   2 passed
tests/test_teleop_media_webrtc.py                         1 passed
---------------------------------------------------------------
相关集合                                                  11 passed
```

实际命令：

```bash
PYTHONWARNINGS=ignore /tmp/airo-teleop-qa/bin/python -m pytest -q \
  tests/test_teleop_media_*.py tests/test_udp.py
# 11 passed in 11.57s
```

`test_teleop_media_webrtc.py` 没有 mock aiortc：创建真实的 server/client `RTCPeerConnection`，完成 offer/answer/ICE、接收真实 `VideoStreamTrack`、解码 `VideoFrame` 并检查尺寸/颜色分区及连接状态。

`test_teleop_media_udp_loopback.py` 使用真实 `AF_INET/SOCK_DGRAM` socket，发送坏包、乱序 chunk、提前重复 chunk、缺最后 chunk 后过期，再 reset 接受重启后的 frame id；JPEG 解码尺寸保持不变。

全仓库旧基线命令也执行过：

```bash
/tmp/airo-teleop-qa/bin/python -m pytest -q
# 686 passed, 95 subtests passed, 1 failed
```

唯一失败为已有的 `tests/test_eval_policy_latency.py::PolicyCompatibilityTest.test_dp_contact_no_vision_registration_and_resolution`，`eval_config.py` 的 `ACTION_STEPS` 为 4 而测试要求 8；该文件不在本任务 ownership 内，与媒体改动无关。

## Python→Unity C# 互操作证据

主代理提供的 `/tmp/teleop-protocol-tests.exe` 使用 Unity 同一份 `JpegFrameAssembler`/DatagramReceiver。运行命令：

```bash
/tmp/airo-teleop-qa/bin/python scripts/teleop_refactor/media_loopback.py --csharp
```

实际输出中的关键结果：

```json
{
  "status": "passed",
  "transport": "python-to-unity-csharp-udp",
  "shape": [72, 96],
  "jpeg_sha256": "32811e18b69c9e1d45b35c84a9de31ab4aab9ce739d3ebd174b9b15fda2938fc",
  "chunks": 37,
  "ready": "READY 34599"
}
```

脚本会比较 C# 输出 JPEG 与 Python sender 编码的完整 SHA256，不只是比较能否解码；发送端以乱序顺序发出 37 个 datagram，C# harness 返回码为 0。Python 自身真实 UDP 结果为 61 chunks、`malformed=1`、`duplicates=1`、`expired=1`、`restarts=1`，解码尺寸 `[96, 128]`。

## RealSense 现场检查

使用：

```bash
/tmp/airo-teleop-qa/bin/python scripts/teleop_refactor/media_loopback.py --realsense --frames 3
```

本机发现 RealSense D435 serial `231122072220`。脚本只打开相机，抓取三帧并通过并发排空的本地 Python UDP；实际输出为 `status=passed`、`captured_frames=3`，三帧解码尺寸均为 `[480, 640]`。发送和接收均为 `127.0.0.1`，没有导入或操作机器人。脚本在相机初始化、抓帧、JPEG 编码、UDP bind、UDP loopback 各阶段记录失败位置；若设备被占用或现场采集超时，会报告 `status=unavailable` 与具体 `stage`，不会把相机失败伪装成网络通过。

## 剩余集成边界

- `WebRTCSession` 的双端本地 aiortc 协商和视频解码已验证；真实 Unity WebRTC signaling/ICE、H.264 codec 选择、跨设备网络与重连仍需主代理的 Unity 工程验收。
- 经典 JPEG receiver 已有 Python↔C# 字节级证据；C# 侧完整 UI/摄像头订阅生命周期仍由主代理负责。
- WebRTC manager 的内置组合仍保留 v1 的基本相机/VR/触觉路径；`main.py`/`realman_teleop.py` 的机器人控制和录制流程尚未在本任务中重新编排，以避免越过 ownership。
- 旧 `UdpComms` 仍位于根 `udp_comms.py`，本拆分通过同一模块对象保留 `udp.U.UdpComms` 和 `WebRTC_udp.U.UdpComms` patch 入口；它未被本任务移动。
# 最终跨语言摄像头验收补充

最终集成已使用 `scripts/teleop_refactor/media_loopback.py --csharp --harness /tmp/doffy-csharp-checks/ProtocolHarness.exe --realsense --frames 3` 采集真实 D435 三帧 640×480 图像，并把最后一帧经过生产 Python JPEG 分片发送到生产 C# 重组模块。38 个分片重组后的 JPEG SHA256 与发送前一致，解码尺寸为 640×480。机器可读结果见 [camera-interop-result.json](camera-interop-result.json)。该测试运行于本机 Mono，不等于 Quest 纹理渲染或无线链路验收。
