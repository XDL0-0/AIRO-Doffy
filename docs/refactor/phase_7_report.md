# Phase 7 重构报告：Video Streaming

日期：2026-07-30

分支：`codex/v2-refactor`

前置阶段：VR 输入与协议（`3de4ccf`）

## 1. 阶段结论

Phase 7 的软件重构已完成。frame processing、同步 encoding、有界异步 encoding、
legacy JPEG/UDP、WebRTC H.264、实验性 RTP/H.264 和 benchmark 现在是互相独立的
组件，不构造 camera，也不读取 VR、robot、tactile、recording、dataset 或
visualization 状态。

旧 `CameraUDPManager` 仍是兼容 façade，但其 JPEG wire packetization 已改为调用
同一个经过 golden tests 的纯 packetizer，删除了第二份 `struct.pack` 实现。
旧 `WebRTCUDPManager` 暂未切换；因此当前默认运行路径、signaling control channel
和 Unity 兼容行为没有被静默替换。

阶段提交：

- `7b1b367 docs: characterize legacy video streaming`
- `1b0f3b5 refactor: define bounded video pipeline contracts`
- `3505271 refactor: add low latency h264 encoding pipeline`
- `4472191 refactor: isolate legacy jpeg udp transport`
- `b1de32d refactor: add experimental rtp h264 transport`
- `3a24477 refactor: extract camera-free webrtc video transport`
- `529bd92 perf: add comparable video benchmark harness`
- `565a448 refactor: reuse frozen legacy jpeg packetizer`
- `83d9333 build: isolate optional video dependencies`

## 2. Components Changed

| 组件 | 变更 | 单一责任 |
|---|---|---|
| `docs/refactor/video_behavior_baseline.md` | 新建 | 固定旧 UDP/WebRTC、buffer、signaling 和 shutdown 行为 |
| `streaming/video/base.py` | 扩展 | 定义 processor、sync encoder、bounded pipeline 与 transport ports |
| `streaming/video/h264_encoder.py` | 新建 | 延迟加载并编码一帧低延迟 H.264 |
| `streaming/video/encoding_pipeline.py` | 新建 | 单 worker、有界 drop-oldest encoding |
| `streaming/video/legacy_jpeg_udp.py` | 新建 | legacy JPEG encode、packetize 与 UDP send |
| `streaming/video/rtp_h264_udp.py` | 新建 | RFC 6184 packetization、UDP send 与 bounded jitter reassembly |
| `streaming/video/webrtc_transport.py` | 新建 | signaling、peer、encoded tracks、H.264 negotiation 与 shutdown |
| `streaming/video/benchmark.py` | 新建 | 统一 timing、drop、bitrate、CPU/GPU 与 delivery receipt metrics |
| `scripts/benchmarks/video_transport.py` | 新建 | 从 generated frames 运行所选 video path |
| `config/models.py::VideoStreamingConfig` | 扩展 | encoder、queue、RTP、age 和 target FPS 参数 |
| `config/models.py::NetworkConfig` | 扩展 | 增加显式 RTP video port |
| `configs/default.yaml` | 扩展 | 记录全部新增安全默认值 |
| `pyproject.toml` | 扩展 | 分离 `video-h264`、`video-jpeg` 与 `video-webrtc` extras |
| `camera_udp.py` | 修改 | 复用 frozen legacy packetizer，保留 manager façade |
| video package exports | 修改 | 公开新 ports、adapters、metrics 和 benchmark API |

## 3. Responsibilities and Dependencies

### 3.1 Interfaces

- `FrameProcessor.process(CameraFrame) -> ProcessedFrame`
- `VideoEncoder.encode(ProcessedFrame) -> EncodedFrame`
- `VideoEncodingPipeline.submit/read_latest`
- `VideoTransport.start/send/close`

接口模块只依赖 typed core models 和 lifecycle Protocol。普通导入不会加载 NumPy、
OpenCV、PyAV、aiortc、aiohttp、pyrealsense2 或 camera toolkit。

### 3.2 H.264 encoder

`LowLatencyH264Encoder`：

- 只接收 `ProcessedFrame`；
- 显式 `start()`，第一次 `encode()` 才导入 PyAV/NumPy 和创建 codec；
- `auto` 按 `h264_nvenc -> libx264` 尝试；
- `nvenc` 不做未授权 software fallback；
- `software` 只选择 `libx264`；
- 固定 `max_b_frames=0`；
- 两个 backend 均设置 `bf=0`、`rc-lookahead=0`；
- NVENC 使用 `p1/ull/zerolatency`；
- libx264 使用 `ultrafast/zerolatency` 与固定 keyint/min-keyint；
- 配置 bitrate、GOP 与 target FPS；
- 拒绝 depth、奇数 yuv420p 尺寸及运行中 geometry/pixel-format 改变；
- 保留 stream、sequence、source/receive timestamp，并增加 encode timestamp。

`video-h264` extra 只声明 `av` 与 `numpy`，不要求 camera 或 WebRTC。

### 3.3 Bounded encoding pipeline

`LatestVideoEncodingPipeline`：

- 一个 encoding worker；
- 默认 input/output capacity 均为 1；
- input overload 丢最旧 frame；
- output overload 丢最旧 encoded frame；
- duplicate/stale sequence 在进入 queue 前拒绝；
- 暴露 submit、encode、drop、error、bytes 和 encode-time metrics；
- encoder error 保存为 health error 并停止 worker；
- close 清空 pending input、join worker，再关闭 encoder；
- 无法停止 worker 时抛出 lifecycle failure。

### 3.4 Legacy JPEG UDP

`LegacyJpegEncoder` 延迟加载 OpenCV/NumPy，保持 BGR JPEG 与配置 quality。
`packetize_legacy_jpeg` 固定：

```text
!IHHI
uint32 frame_id
uint16 chunk_index
uint16 total_chunks
uint32 total_jpeg_bytes
JPEG payload
```

默认 payload 仍为 60,000 bytes；frame id 仍取低 32 bits。新 packetizer 对空 payload、
无法表示的 chunk count/length 和超过 UDP payload 上限的 chunk size 显式失败。

`LegacyJpegUdpTransport` 只拥有一个 outgoing socket 和一个 target，不绑定 VR pose、
record 或 control RX。类名、文档与 `DeprecationWarning` 均明确其 legacy 状态。

### 3.5 RTP/H.264

`RtpH264UdpTransport`：

- 接受 Annex B、4-byte AVCC 或 raw single NAL；
- RFC 6184 single NAL 和 FU-A；
- RTP v2 header、dynamic payload type、uint16 sequence、uint32 SSRC；
- 使用 90 kHz video timestamp；
- MTU 默认 1200 bytes；
- 一个 transport 只允许一个 stream id；
- 拒绝 stale frame；
- 按 encode timestamp 丢弃超过 `max_frame_age_s` 的 late frame；
- 无 frame queue，发送失败显式上抛并计数。

`RtpH264JitterBuffer` 用固定 frame capacity 接收乱序 RTP packet，拒绝 late/duplicate/
malformed packet，并把 single NAL/FU-A 重组为 Annex B access unit。RTP 路径保持
experimental，不是默认 production transport。

### 3.6 WebRTC

`WebRTCVideoTransport`：

- construction 和 package import 不加载 WebRTC dependencies；
- 显式 start 后在自己的 async-loop thread 启动 aiohttp signaling；
- 保留 `/` 与 `type/session_id/payload` envelope；
- 保留 hello/hello_ack、offer/answer、ICE、start/stop message types；
- 每个配置 stream 一个 latest-only encoded H.264 track；
- track 返回带 90 kHz PTS/time base 的 PyAV `Packet`；
- H.264 codec preference 显式设置，packetization-mode 1 优先；
- 新 offer 关闭旧 peer；
- failed/closed connection 清理 peer；
- 暴露 connection state、peer、signaling、submit、deliver 与 drop metrics；
- close 停止 server、关闭 peer/runner、join loop thread 并唤醒 track waiters。

该实现依据 aiortc 官方 API 中 `MediaStreamTrack.recv()` 可返回 `Packet` 的契约。
新 video transport 不创建 control DataChannel；resolution、state 和 reliable
command 进入 Phase 8 的独立通道。旧 manager 继续保留旧 `"control"` channel，
所以现有部署尚未发生行为变化。

### 3.7 Benchmark

`VideoBenchmarkRunner` 对任意 encoder/transport pair 统一记录：

- frames attempted/encoded/submitted/errors；
- component drop counters；
- encoded payload bytes 与 transport/receipt wire bytes；
- encoded/wire bitrate；
- queue、encode、send、receive 与 end-to-end latency；
- process CPU time/percentage；
- 可注入 GPU utilization sampler；
- 可注入 receiver/display `DeliveryProbe`。

没有远端 receipt 时 `delivery_loss_rate=None`，不会把“没有观测”伪装成零丢包。
CLI 支持重复 `--path` 对 legacy JPEG UDP、WebRTC H.264 和 RTP H.264 使用相同
generated packed frames。

## 4. Coupling Removed

- camera acquisition 不再负责 encoding 或 transport。
- frame processing 不再导入 codec/network。
- H.264 encoder 不知道 camera、socket、peer 或 signaling。
- encoding queue 不知道具体 codec 或 transport。
- UDP transports 不读取 VR、record、control 或 tactile。
- WebRTC transport 不持有 RealSense、NumPy image dict、dataset snapshot 或 VR state。
- WebRTC signaling/peer 生命周期不再混在 camera manager 中。
- legacy JPEG packet bytes 只有一份实现，root manager 通过兼容调用复用。
- benchmark 不构造硬件，接受任意 typed processed frames 和 probes。
- video optional dependencies 按 adapter 分组，不再通过 camera extra 间接获得。

## 5. Remaining Coupling

- root `CameraUDPManager` 仍同时拥有 camera、VR、record、control、tactile 与 threads；
  本阶段只替换其 packetizer。
- root `WebRTCUDPManager` 仍是旧 all-in-one implementation，尚未组合新 source、
  processor、encoder 与 transport。
- `main.py` 和 RealMan compatibility factory 仍直接选择两个旧 manager。
- 新 WebRTC transport 的 external H.264 packet path 未在安装了 aiortc/PyAV 的环境中
  执行 integration test。
- 仓库没有 tracked Unity/C# source，只有 `unity/README.md` placeholder，因此无法在
  repo 内新增或验证 Quest RTP receiver。
- WebRTC `"control"` DataChannel 仍留在旧 manager；Phase 8 需要把 state/command
  channel 独立实现后，Phase 12 才能安全替换 manager。
- NVENC availability、profile/level、driver 与实际 output Annex B/AVCC 需部署验证。
- benchmark 的 GPU 与 E2E 指标必须由部署-specific sampler/receipt 提供。

## 6. Behavior Preservation

当前 root runtime 尚未切换到新 video composition。以下均未改变：

- legacy JPEG `!IHHI` 字节布局；
- JPEG quality 100 默认值；
- 60,000-byte payload 默认值；
- 每 camera 独立 uint32 frame id；
- camera UDP port 分配；
- root OpenCV RGB→BGR 与 center zoom；
- root WebRTC signaling envelope；
- root WebRTC `"control"` DataChannel；
- VR、record、rollback 与 tactile UDP；
- dataset RGB/depth schema；
- camera cadence、robot safety 与 control frequency。

root JPEG packetizer 对所有可表示的旧输入产生相同字节。新增的 size/overflow rejection
只作用于过去会超过 UDP 或 header 表示范围的无效配置。

新的 H.264、RTP 与 standalone WebRTC APIs 是 opt-in。typed config 仍默认
`transport="webrtc"`，但现有 root app 尚未使用该 typed factory 自动切换路径。

## 7. Thread and Loop Ownership

### Encoding loop

| 属性 | 定义 |
|---|---|
| Input | bounded `ProcessedFrame` queue |
| Output | bounded latest `EncodedFrame` queue |
| Target frequency | producer/config driven，默认 target 30 FPS |
| Owned resources | one worker、condition、two deques、encoder lifecycle |
| Queue capacity | 默认 input 1、output 1 |
| Drop policy | overload drop oldest；stale reject |
| Shutdown | stop flag、clear pending、join 5 s、encoder close |
| Error | 保存 health error、计数并停止 worker |

### WebRTC loop

| 属性 | 定义 |
|---|---|
| Input | per-stream latest `EncodedFrame` |
| Output | WebRTC H.264 tracks |
| Target frequency | peer pull / encoded producer driven |
| Owned resources | event loop thread、aiohttp runner、peer、tracks |
| Buffer | 每 stream constant-memory latest buffer |
| Queue capacity | 1 |
| Drop policy | overwrite unread latest；stale reject |
| Shutdown | async stop、peer/runner cleanup、thread join 10 s |
| Error | startup/runtime error 可观察并转为 lifecycle failure |

### UDP send

Legacy 与 RTP transport 均为 caller-owned synchronous send，没有隐藏 worker。
它们只拥有 outgoing UDP socket；backpressure 表现为 send 调用耗时或显式异常。

## 8. Tests

Phase 7 专项：

- `test_h264_encoder.py`：5 个 backend/options/metadata/lifecycle tests；
- `test_video_encoding_pipeline.py`：3 个 queue/drop/error/close tests；
- `test_legacy_jpeg_udp.py`：6 个 golden/packet/encoder/transport/factory tests；
- `test_rtp_h264_udp.py`：6 个 NAL/FU-A/RTP/jitter/late/factory tests；
- `test_webrtc_transport.py`：5 个 signaling/latest/lifecycle/factory tests；
- `test_video_benchmark.py`：3 个 percentile/metrics/comparison tests；
- package test 增加 root manager 复用 packetizer 的静态约束。

阶段验收：

| 验证 | 结果 |
|---|---|
| 全部 v2 unit tests | 163/163 通过 |
| package/compatibility tests | 8/8 通过 |
| Phase 7 专项 tests | 28/28 通过 |
| video package optional-import check | 通过 |
| generated benchmark CLI `--help` | 通过 |
| tracked Python AST | 181 个文件通过 |
| `src/` 100 字符行检查 | 通过 |
| `git diff --check` | 通过 |

当前环境没有安装 `ruff`、`pyright`、PyAV、aiortc、OpenCV 或 GPU SDK，因此没有
伪造这些工具或真实 codec 的结果。所有 codec/WebRTC unit tests 使用窄 fake backend，
真实 adapter 被明确列入 hardware/integration validation。

## 9. Hardware and Integration Validation

1. PyAV `h264_nvenc` discovery、driver、open 与持续 encode。
2. auto 模式 NVENC failure 后 libx264 fallback。
3. 真实输出的 SPS/PPS、Annex B/AVCC、keyframe、profile、level 与 GOP。
4. no B-frame/no lookahead 的 ffprobe 验证。
5. 640×480、720p、多相机的 encode P50/P95/P99、CPU、GPU 与 bitrate。
6. aiortc track 返回预编码 `Packet` 的 Chrome/Unity interoperability。
7. H.264-only SDP negotiation、packetization-mode 1、ICE、reconnect 与 headset sleep。
8. WebRTC close 后 peer、runner、event loop、waiter 和 thread leak。
9. Unity legacy JPEG golden reassembly 与 frame-id wrap。
10. 60,012-byte legacy datagram 在目标 LAN 的 fragmentation/loss baseline。
11. Quest RTP receiver 的 FU-A、sequence wrap、jitter capacity 与 late drop。
12. 三条路径同场 benchmark，并用 receiver/display receipt 计算真实 loss/E2E delay。

## 10. 下一阶段

按计划进入 Phase 8 State and Command Channels：

1. 实现 unordered/no-retransmit/latest-only realtime state channel。
2. 增加独立 serializer、sequence rejection 与 metrics。
3. 实现 ordered/reliable command channel、ack、timeout 与 duplicate handling。
4. 增加 command router。
5. 保留 UDP diagnostic adapter，但不把 control channel 放回 video transport。
