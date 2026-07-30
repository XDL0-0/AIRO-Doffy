# Phase 7 Video Streaming Behavior Baseline

日期：2026-07-30

范围：`camera_udp.py`、`WebRTC_udp.py`、`udp_comms.py`、Unity 兼容边界，以及
Phase 5 已提取的 camera/frame components。

## 1. 当前组件边界

`CameraUDPManager` 和 `WebRTCUDPManager` 都不是单一 video transport。两者均拥有：

1. RealSense discovery、construction 与 acquisition threads；
2. RGB/depth latest dict 与 dataset snapshot dict；
3. RGB→BGR、center zoom 和 OpenCV processing；
4. VR controller/hand UDP receive 与解析；
5. record/rollback control；
6. resolution/zoom/fine-mode control；
7. tactile UDP forwarding；
8. camera、socket、receive、send 与 async loop shutdown。

两条路径真正不同的只有：

- `CameraUDPManager`：OpenCV JPEG encode 后执行 legacy UDP chunking；
- `WebRTCUDPManager`：把 BGR NumPy frame 交给 aiortc `VideoStreamTrack`。

新 video modules 不得读取 VR、record、tactile、robot、dataset 或 visualization
状态，也不得构造 camera。

## 2. Legacy JPEG UDP

### 2.1 Wire format

每个 UDP datagram：

```text
network byte order / big-endian

offset  size  type     field
0       4     uint32   frame_id
4       2     uint16   chunk_index
6       2     uint16   total_chunks
8       4     uint32   total_jpeg_bytes
12      N     bytes    JPEG slice
```

Python `struct` format 固定为 `!IHHI`，header 固定为 12 bytes。Unity
`UdpSocketMultiHD.cs` 依赖同一格式。

### 2.2 Encode and packetization behavior

- 输入按 BGR 交给 `cv2.imencode(".jpg", ...)`。
- 默认 JPEG quality 为 100。
- 默认 chunk payload 为 60,000 bytes。
- 最大默认 UDP datagram 为 60,012 bytes，不含 UDP/IP headers。
- `total_chunks = ceil(total_bytes / chunk_size)`。
- 每个 camera index 有独立 counter，初值 0。
- wire `frame_id` 是 counter 的低 32 bits。
- chunk index 从 0 单调到 `total_chunks - 1`。
- 所有 chunk 重复相同 `frame_id`、`total_chunks` 与 `total_bytes`。
- encode 返回失败时发送 0 个 chunk。
- 当前没有 checksum、重传、ACK、FEC、超时或拥塞控制。
- 当前没有显式检查 uint16 chunk count 或 UDP payload 上限。

这些是迁移期兼容行为。新 module 可以显式拒绝无法表示的输入，但不得静默改变
可表示 packet 的字节布局。

### 2.3 Socket behavior

旧实现通过 `UdpComms._sock` 直接调用 `sendto()`，目标为
`(sock.send_ip, sock.udp_send_port)`。因此 packetizer 与 socket adapter 必须分离；
新 transport 不应访问另一个对象的 private socket。

每台相机旧端口为：

```text
VR TX port = IP_PORT + 2 * camera_index
```

默认从 8000 开始，最多 5 台相机。相机 0/1/2 的同一个 socket 同时绑定 pose、
record、control RX，但新 video transport 不得拥有这些 RX 职责。

## 3. WebRTC

### 3.1 Signaling

- aiohttp WebSocket 只监听 `/`。
- 默认绑定 `PC_IP:8765`。
- envelope 为 `type/session_id/payload` JSON object。
- `hello` 回复 `hello_ack`。
- `start_video` 只写日志。
- `offer` 关闭旧 peer，创建新 peer，并回复 `answer`。
- `ice_candidate` 双向传输 candidate、sdpMid 与 sdpMLineIndex。
- `stop_video` 关闭当前 peer。
- 当前只支持一个 manager-owned peer；新 offer 替换旧 peer。
- 当前没有认证、schema version 或严格 session ownership。

### 3.2 Tracks and codec behavior

- 每台 camera 创建一个 `RealsenseCameraTrack`。
- track 从 manager 的共享 NumPy dict 取最新 frame。
- 没有 frame 时每 5 ms async poll。
- track 内部再次执行 manager image processing 并更新 dataset snapshot。
- 返回 `av.VideoFrame(format="bgr24")`。
- PTS/time base 来自 `VideoStreamTrack.next_timestamp()`。
- 代码没有设置 codec preference。
- 代码没有强制 H.264，也没有禁止 VP8。
- 代码没有设置 bitrate、GOP、B-frame、lookahead、NVENC 或 software fallback。
- 最终 codec 和 encoder 由 SDP 与 aiortc/PyAV 环境决定。

因此 README 的“H.264/VP8”只能视为可能的协商结果，不能作为低延迟 H.264
实现证明。

### 3.3 Control DataChannel

- Python 创建 label 为 `control` 的 DataChannel。
- `open` 只写日志。
- `message` 直接调用 resolution/zoom/fine-mode parser。
- 数字 key 在 WebRTC parser 中直接作为 camera index。
- 非数字 key 设置 `fine_mode`。

该 control channel 不是计划中的 realtime state 或 reliable command channel；
这些职责属于 Phase 8，不能放入新 video transport。

## 4. Cadence, Buffering, and Metrics

### Legacy UDP

- capture thread 成功读取后额外等待 `1/30 s`。
- 每 camera 一个 send thread，也在每轮尾部等待 `1/30 s`。
- camera dict 只存最新 raw frame，因此没有无界 frame queue。
- send thread 可以重复发送同一 raw frame，没有 sequence-based duplicate suppression。
- 每 5 s 记录 frames/chunks 和计算 FPS。
- `stats_bytes` 已声明但没有更新或报告。

### WebRTC

- capture thread 使用同样的 30 Hz 额外等待。
- aiortc 按 track pull model 消费最新共享 frame。
- 没有显式 queue depth、drop counter、encode time、bitrate、keyframe 或端到端延迟指标。

新 pipeline 必须：

- input queue 有界，默认容量 1；
- overload 时 drop oldest；
- 不在 camera acquisition loop 内编码或发送；
- 分别记录 submit、drop、encode、send、bytes、error 和 timing；
- timestamp/sequence 从 typed frame 贯穿到 encoded frame；
- 不在没有测量的情况下承诺固定延迟数值。

## 5. Shutdown

### Legacy UDP

1. `running=False`。
2. daemon threads 各 join 最多 1 s。
3. camera `pipeline.stop()` 或 `close()`。
4. 所有 `UdpComms.close()`。
5. join 后是否仍存活不会导致失败。

### WebRTC

1. `running=False`。
2. async shutdown event 被设置。
3. event-loop thread join 最多 10 s。
4. 未退出时强制 `loop.stop()`，再 join 2 s。
5. 其他 daemon threads 各 join 最多 1 s。
6. camera 与 UDP sockets 关闭。
7. peer、tracks 与 signaling runner 由 async loop 清理。

新 components 必须幂等关闭，只释放自己拥有的资源；如果 worker/loop 仍存活，
必须暴露 lifecycle failure，不能宣称成功。

## 6. Optional Dependency Boundary

当前 `WebRTC_udp.py` 顶层导入：

- `aiortc`
- `aiohttp`
- `av`
- `cv2`
- `numpy`
- `pyrealsense2`
- `airo_camera_toolkit`

因此导入旧 manager 同时要求 WebRTC、image 和 camera SDK。新 package 的普通
import 必须不加载任何这些模块：

```text
airo_doffy.streaming.video
```

具体 adapter 只有在显式 construction/start 时才可加载其 optional dependency。

## 7. Compatibility Tests Required Before Wiring

### Legacy JPEG UDP

1. 单 chunk golden bytes。
2. 多 chunk header 与 payload reconstruction。
3. uint32 frame id wrap。
4. 每 stream 独立 sequence。
5. 空 payload、非法 chunk size 和 uint16 overflow rejection。
6. socket target 与 send order。
7. JPEG codec identity rejection。

### H.264 encoder

1. NVENC selection 与 unavailable fallback。
2. software encoder fallback。
3. no B-frame、no lookahead、bitrate 与 GOP options。
4. RGB/BGR input mapping。
5. even dimensions。
6. keyframe metadata。
7. input queue capacity 与 drop-oldest。
8. no optional codec import at package import time。

### WebRTC

1. hello/offer/answer/candidate/stop envelope compatibility。
2. one track per configured stream。
3. reconnect closes previous peer。
4. failed/disconnected peer cleanup。
5. negotiated codec inspection。
6. close releases peer, runner, sockets and event-loop thread。

### RTP/UDP H.264

1. RTP header、sequence、90 kHz timestamp 与 marker。
2. single NAL 和 FU-A fragmentation。
3. Annex B access-unit parsing。
4. stale frame rejection。
5. configured MTU boundary。
6. receiver jitter/late-packet policy。

## 8. Hardware and Cross-Language Validation

必须在替换旧路径前完成：

1. Unity legacy JPEG receiver 对 golden packets 的 reassembly。
2. 0xffffffff→0 frame id wrap。
3. 真实 LAN 下 60,012-byte datagram 的 fragmentation/loss baseline。
4. 多相机端口与独立 frame id。
5. Unity WebRTC H.264 capability、profile、level 与 hardware decode。
6. PC NVENC availability 与 driver/PyAV codec visibility。
7. software fallback CPU、latency 与 sustained multi-camera FPS。
8. WebRTC reconnect、ICE failure、headset sleep/wake 和 shutdown。
9. RTP experimental receiver 的 FU-A、jitter 与 late-drop 互操作。
10. JPEG UDP、WebRTC H.264、RTP H.264 的同场 benchmark。

## 9. Migration Boundary

Phase 7 新 modules 先作为独立可测试组件加入。root `camera_udp.py` 和
`WebRTC_udp.py` 在 camera、VR、record、tactile 与 runtime composition 尚未完成前
保留为兼容 façade。任何默认路径切换都必须等待：

- packet/signaling compatibility tests；
- Unity/Quest validation；
- latency/loss benchmark；
- session composition；
- 用户明确批准移除旧 wrapper。
