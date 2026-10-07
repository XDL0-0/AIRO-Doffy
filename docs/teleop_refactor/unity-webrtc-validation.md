# Unity WebRTC refactor validation

本次提交只涉及 Unity 视频链路的信令、PeerConnection receiver 和会话编排。Unity 项目位于 `/path/to/unity/Codex`，目标缓存 SDK 为 `com.unity.webrtc 3.0.0`；没有修改 AppManager、核心 UI、场景或其他代理负责的媒体协议文件。

## 代码边界

- `Assets/Teleop/Media/WebRTC/VideoSignalingClient.cs` 保留 `Envelope`、`OnEnvelope`、`OnError`、`OnConnected`、`OnDisconnected` 和 `ConnectAsync`/`DisconnectAsync`/`SendAsync`/`Dispose` API。每个连接保存自己的 socket、CTS 和 generation；`SemaphoreSlim` 串行化发送；接收按完整 WebSocket message 累积，限制为 1 MiB，完成后一次性严格 UTF-8 解码。
- `Assets/Teleop/Media/WebRTC/WebRTCVideoReceiver.cs` 保留既有序列化字段 `expectedTrackCount` 与既有事件/API，增加 `OnDataChannelOpened` 和返回发送结果的 `TrySendControlMessage`。用缓存 SDK 的强类型 `AddTransceiver(TrackKind.Video, RecvOnly)` 取代反射；peer、轨道、DataChannel 和待完成 SDP 操作在 `ClosePeer` 中解绑/释放，旧 callback 和 coroutine 由 generation predicate 失效。
- `Assets/Teleop/Media/WebRTCSessionOperations.cs` 只负责 offer/answer coroutine 的 native async operation 顺序，receiver 负责 peer 所有权、代际判断和 TCS 完成。
- `Assets/Teleop/Media/WebRTC/VideoStreamManager.cs` 保留 `videoReceiver`、`singlePanels`、`dualPanels`、`triPanels`、`resolutionLabels` 序列化字段及原有会话/控制入口。信令 callback 先入主线程队列；每次启动、停止和销毁都会推进 generation；旧 `OnConnected` 只能在当前 `Connecting` 会话设置状态。`ExpectedTrackCount` 在 `InitializePeer` 前设置。分辨率消息只有在 DataChannel 真正 Open 且 Send 成功后才记为已发送，并由 `OnDataChannelOpened` flush 初始/重连值；信令断开和 peer failed/disconnected 经过统一视频故障路径，不改变 teleop 生命周期。

实现文件的物理行数为：signaling 413、receiver 459、manager 467；职责 helper 分别为 envelope codec 36、WebSocket reader 44、session bindings 91、session protocol 120、DataChannel bridge 78、peer factory 29、receiver media state 126、SDP operations 164。三个核心入口均控制在约 500 行内，native 资源清理和代际检查保留在其拥有者或对应 helper 中。

## 可重复验证

使用已提供的缓存程序集进行源码/API 编译（不启动 Unity Editor）：

```bash
python3 scripts/teleop_refactor/compile_unity_sources.py \
  --project /path/to/unity/Codex \
  --reference /path/to/unity/classic \
  --editor-data /path/to/Unity/Hub/Editor/6000.5.6f1/Editor/Data \
  --output /tmp/airo-teleop-unity-webrtc
```

最终集成结果：`C# source/API check: 82 source files, exit 0.` 旧键盘已更新到 Unity 6 当前 API；runtime 与 Editor 工具源码检查均为 0 错误、0 警告。

真实 C# 信令测试：

```bash
/tmp/airo-teleop-qa/bin/python -m pytest -q \
  tests/test_webrtc_csharp_source.py
```

结果：`1 passed`。pytest 会编译实际 `VideoSignalingClient.cs`、`SignalingEnvelopeCodec.cs` 和 `WebSocketMessageReader.cs`，启动本机 aiohttp WebSocket server，再运行 embedded Mono C# client。该真实交互覆盖 64 个重叠 `SendAsync`、服务端把多字节中文/emoji 从 UTF-8 codepoint 中间拆成两帧、服务端 close 后 `OnDisconnected`、同一 client 重连，以及超过 1 MiB 的 inbound message 被拒绝并断开；C# runner 输出 `REAL_WEBRTC_SIGNALING_CHECKS_PASS`。

## 尚未验证的部分

Unity Editor 当前没有可用 license，未绕过 license，也未声称完成场景播放、Android 打包、真实 WebRTC 协商、远端多 track、纹理渲染或硬件控制验证。上述 compile 是 SDK/API 源码检查；真实 C# 信令测试只覆盖 ClientWebSocket 与本机 aiohttp server。连接到真实 PC/Unity runtime 后，应补做一次 1/2/3 track、首次 DataChannel 控制消息、stop→start 快速重启和 signaling server 断开测试。
