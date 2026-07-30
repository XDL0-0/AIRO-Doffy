# Phase 8 重构报告：State and Command Channels

日期：2026-07-30

分支：`codex/v2-refactor`

前置阶段：Video Streaming（`1ded24a`）

## 1. 阶段结论

Phase 8 的软件重构已经完成。高频状态和低频可靠命令现在是两个互不混用的
package 边界：

- 状态采用二进制、无序、零重传、latest-only 通道；
- 命令采用独立的有序可靠通道；
- serializer 不依赖 WebRTC、UDP、Unity 或硬件 SDK；
- 状态接收按消息类型拒绝重复/过期序列；
- 命令提供 ACK、超时、错误状态和有界幂等去重；
- `CommandRouter` 只持有注入 handler，不导入组件实现。

旧 root runtime 尚未切换到这些新通道，所以旧 Quest UDP、录制命令、相机控制、
可视化回滚和旧 WebRTC `"control"` DataChannel 行为未被静默替换。

阶段提交：

- `1960c3a docs: characterize state and command channels`
- `493466f refactor: add latest-only realtime state channels`
- `1dc6fcc refactor: define reliable command protocol`
- `38ac196 refactor: add reliable command channel`
- `938c59e refactor: add runtime command router`

## 2. 组件变更

| 组件 | 变更 | 单一职责 |
|---|---|---|
| `docs/refactor/channel_behavior_baseline.md` | 新建 | 固定旧状态/命令路径和兼容边界 |
| `streaming/state/protocol.py` | 新建 | 状态 envelope 与 VR/robot payload codec |
| `streaming/state/channels.py` | 新建 | latest sender/receiver、WebRTC 和诊断 UDP adapter |
| `streaming/commands/protocol.py` | 新建 | 严格 command/ACK JSON codec |
| `streaming/commands/channels.py` | 新建 | reliable sender/receiver、ACK、timeout、dedupe |
| `streaming/commands/router.py` | 新建 | enum 到注入 handler 的路由与事件 |
| `core/events.py` | 扩展 | 补齐计划要求的显式命令枚举 |
| `core/errors.py` | 扩展 | 命令超时与业务拒绝 domain errors |
| `config/models.py` | 扩展 | channel label、诊断端口、ACK timeout、dedupe capacity |
| `configs/default.yaml` | 扩展 | 显式记录 Phase 8 安全默认值 |
| `docs/protocols/state_channel_v1.md` | 新建 | 状态字节布局与 Unity 约束 |
| `docs/protocols/reliable_commands_v1.md` | 新建 | 命令/ACK、重试和幂等约束 |

## 3. 实时状态路径

### 3.1 二进制 envelope

公共 header 为 little-endian `<HBBIQHH>`，固定 20 bytes：

```text
magic 0xAD20       uint16
version 1          uint8
message_type       uint8
sequence           uint32
timestamp_ns       uint64
payload_size       uint16
flags              uint16
```

flags 低两位记录 monotonic、Unix、device 或 unspecified clock domain；其余位必须
为零。decoder 对 magic、version、type、flags、length、payload shape 和 metadata
全部严格校验。

VR payload 复用 Phase 6 已经过 golden tests 的 binary v2，并要求内外 sequence/
timestamp 一致。Robot payload 使用 6/7 个 joint `float32`、row-major 4×4 TCP、
可选 gripper width 和可选 6 维 wrench。

### 3.2 Latest-only sender

`LatestStateSender` 只接受：

```text
ordered = false
max_retransmits = 0
```

它只有一个 worker，pending storage 对每种消息类型最多一个 packet；当前协议只有
VR 和 robot 两类，所以内存上限恒定为两个 packet。相同类型的新 packet 覆盖尚未
发送的旧 packet。序列比较使用 uint32 modular ordering，并分别记录 submitted、
sent、overwrite drop、stale reject、bytes 和 send error。

### 3.3 Receiver 与 adapters

`LatestStateReceiver` 分别保存最新 VR 和 robot state；一个类型的序列不影响另一个
类型。malformed、duplicate、stale 和 wrap-around 都有确定行为和 metrics。

`create_aiortc_realtime_state_channel()` 只调用注入 peer 的
`createDataChannel(label, ordered=False, maxRetransmits=0)`，模块本身不导入 aiortc。

`UdpDiagnosticStateChannel` 是 opt-in、单 datagram、无重传的诊断 adapter，默认配置
端口为 5005；它不是生产可靠路径，也没有在 payload 层增加 fragmentation。

## 4. 可靠命令路径

### 4.1 Typed command 与协议

计划要求的命令均为显式 `RuntimeCommandType`：录制 start/stop、rollback、触觉重标定、
wrench baseline reset、teleop mode、camera zoom/resolution、safe hold 和 controlled
stop。既有 reset reference、pause/resume、video profile 和 shutdown 继续保留。

命令和 ACK 使用 version 1 canonical UTF-8 JSON。该路径是低频控制路径，不是高频
state path。parser 拒绝：

- 非 UTF-8 或非法 JSON；
- 超过 65,536 bytes；
- root 非 object；
- 未知/缺少/重复字段；
- 未知 version 或 message type；
- 非法 command enum、value、timestamp、sequence、clock domain 或 ACK status。

### 4.2 Sender ACK 与 timeout

`ReliableCommandSender` 在 send 前先登记 `command_id`，因此同步到达的 ACK 不会丢失。
ACK 必须同时匹配 command ID 和 sequence。默认等待 1 秒，超时抛出
`CommandTimeoutError`；late/unexpected/malformed ACK 分别计数。关闭 sender 会唤醒
所有正在等待的调用，不留下等待线程。

### 4.3 Receiver 幂等与错误

接收端按 canonical command bytes 保存 LRU record：

1. 新 ID：dispatch 一次并在发送 ACK 前缓存结果；
2. 同 ID、同 payload：不再 dispatch，返回缓存状态且 `duplicate=true`；
3. 同 ID、不同 payload：不 dispatch，明确返回 rejected；
4. 超过容量：淘汰 least-recently-used ID。

“先缓存、后发 ACK”保证 ACK 丢失后的同 payload retry 不会重复执行副作用。默认容量
1024；超过 retention window 的重试不能再被识别，因此部署端仍需约束重试窗口和
command ID 唯一性。

### 4.4 WebRTC policy

可靠 adapter 要求：

```text
label = "commands"
ordered = true
max_retransmits = None
max_packet_lifetime = None
```

factory 不传 `maxRetransmits` 或 `maxPacketLifeTime`，保留 aiortc 的 fully reliable
默认值。任何显式有限重传或 lifetime channel 都在构造 sender/receiver 时被拒绝。

## 5. Command Router

`CommandRouter` 构造参数是 `RuntimeCommandType -> Callable` mapping。handler 可以：

- 正常返回 `str | None`，产生 `COMMAND_ACCEPTED`；
- 抛出 `CommandRejectedError`，产生 warning `COMMAND_REJECTED`；
- 抛出其他异常，产生 error `COMMAND_REJECTED`；
- 未注册时产生 warning `COMMAND_REJECTED`。

router 返回 immutable `RuntimeEvent`，带 command ID、command kind、origin、event
sequence、monotonic timestamp 和 severity。它不知道 recording、dataset、camera、
tactile、wrench、teleop、robot 或 runtime 的成员结构。

## 6. 解耦结果

- 高频状态 serializer 不知道 DataChannel 或 socket；
- WebRTC adapter 不导入 aiortc，只包装 composition 注入的对象；
- UDP adapter 不解析 VR 或 robot payload；
- VR 和 robot state 不共享 freshness counter；
- command parser 不执行 command；
- sender 不知道 handler；
- receiver 只依赖 `CommandDispatcher` Protocol；
- router 不知道 transport 和 ACK；
- router handler 由 composition root 注入，不直接修改 manager flags；
- state/command package 普通导入不加载 camera、codec、robot 或 WebRTC SDK。

## 7. 兼容性

本阶段所有新路径均为 opt-in，以下旧行为保持：

- Quest state 的 legacy controller text 与 `HB,` Base64；
- root `UdpComms` 的 128-entry receive deque；
- 旧 manager 100 Hz pose polling；
- record socket 的 `Start`、`Stop`、`Undo`、`Rollback`、`DeleteLast`；
- camera zoom/resolution/fine-mode legacy text；
- visualizer 的 in-process typed rollback queue；
- 旧 WebRTC `"control"` DataChannel；
- dataset export/rollback 时机；
- robot safety 与 control loop frequency。

Phase 12 composition migration 前，不能删除这些兼容路径。

## 8. 生命周期与有界性

| 组件 | Background loop | Buffer/pending 上限 | Shutdown |
|---|---|---|---|
| `LatestStateSender` | 1 worker | 每种 state type 1 | stop、clear、join、channel close |
| `LatestStateReceiver` | 无 | 每种 state type 1 | caller-owned |
| `UdpDiagnosticStateChannel` | 无 | 无 queue | socket close |
| `ReliableCommandSender` | 无内部 worker | concurrent in-flight IDs | close 唤醒 waiters |
| `ReliableCommandReceiver` | 无 | config dedupe capacity | channel close |
| `CommandRouter` | 无 | handler mapping | caller-owned |

reliable sender 的并发 pending 数由调用方并发数决定，不是内部生产队列；每个调用都在
ACK/timeout/close 后删除自己的 pending entry。

## 9. 测试与验收

Phase 8 专项覆盖：

- 20-byte header golden bytes；
- VR controller 与 6/7-DOF robot state round trip；
- header/type/version/flags/length/inner metadata rejection；
- uint32 wrap、duplicate/stale rejection；
- blocked channel 下 latest-only overwrite；
- state WebRTC 参数与 UDP target/lifecycle/metrics；
- command/ACK canonical round trip；
- unknown/missing/duplicate field rejection；
- planned command enum coverage；
- ordered fully-reliable WebRTC 参数；
- ACK matching、timeout、late/malformed ACK；
- close 唤醒 ACK waiter；
- same/conflicting duplicate 与 LRU eviction；
- accepted/rejected/error ACK；
- router selection、unhandled、business rejection 和 exception；
- state/command package optional-import isolation。

阶段验收结果：

| 验证 | 结果 |
|---|---|
| 全部 v2 unit tests | 195/195 通过 |
| package/compatibility tests | 10/10 通过 |
| tracked Python AST | 191 个文件通过 |
| `src/` 100 字符行检查 | 通过 |
| `python -m compileall -q src` | 通过 |
| offline editable install | 通过 |
| `git diff --check` | 通过 |

当前环境没有安装 `ruff`、`pyright` 或 aiortc。测试使用 fake peer/channel/socket，
没有伪造真实网络、Unity 或 Quest 验证结果。`tests/test_realman_teleop_loop.py` 仍因
当前环境缺少 OpenCV 而不在 package-layout 验收命令中。

## 10. 硬件与集成验证

1. Unity/C# 对 state v1 header、float32、robot row-major TCP 的 golden decode。
2. Unity controller 与 26-joint OpenXR hand payload 的端到端 decode。
3. Python-created 与 Unity-created DataChannel negotiation。
4. Wi-Fi loss/reordering 下 unordered/no-retransmit state freshness。
5. state sequence wrap、sender restart 与 reconnect policy。
6. robot state downsample 频率、datagram/association message size 和 MTU。
7. ordered reliable command delivery与 ACK round-trip latency。
8. 主动丢弃 ACK 后 retry，确认 recording/rollback 等副作用只执行一次。
9. peer reconnect 后 command ID retention 与 retry window。
10. malformed/oversized command 的拒绝和日志/metrics。
11. safe hold 与 controlled stop 的真实 handler 和 robot safety integration。
12. camera/tactile/wrench/recording handlers 的状态前置条件与业务拒绝。
13. close 后 peer、DataChannel、waiting caller 和 thread leak。

## 11. 剩余耦合

- root `CameraUDPManager` 和 `WebRTCUDPManager` 仍持有多域状态与 threads；
- `main.py` 尚未组装新 state/command channel；
- legacy command strings 尚未通过 adapter 转为 `RuntimeCommand`；
- 新 router 只有通用机制，真实 component handlers 尚未在 composition root 注入；
- Unity source 仍只有 placeholder，无法在 repo 内实现/验证对端；
- WebRTC signaling 尚未把两个 DataChannel 与 Phase 7 peer composition 组装；
- reconnect、sequence epoch 和 dedupe persistence 是部署策略，尚未由 root runtime 决定。

这些属于后续 composition/runtime 阶段，不应在 Phase 8 协议组件中重新引入 manager
耦合。

## 12. 下一阶段

按计划进入 Phase 9 Tactile Refactor：

1. 固定旧 4-taxel BLE 数据、滤波、标定和线程行为；
2. 只保留 4-taxel BLE 为 supported backend；
3. 把 source、calibration 和 processing pipeline 分离；
4. 保持 synthetic/mock source 与无硬件测试；
5. 将旧 41-taxel/serial 路径继续留在 `deprecated/`，不恢复到主运行路径。
