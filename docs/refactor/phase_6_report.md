# Phase 6 重构报告：VR 输入与协议

日期：2026-07-30

分支：`codex/v2-refactor`

前置阶段：相机采集与 Frame Processing（`80e8309`）

## 1. 阶段结论

Phase 6 的软件重构已经完成。旧 controller CSV、hand text 和 HB binary 解析已从
root manager 中提取为纯函数；新的 binary v2、传输无关接收器和可脚本化 Mock VR
均可在不导入 camera、robot、tactile、recording 或 visualization 的情况下独立运行。

旧 Unity wire format 没有被替换。root `parse_vr.py` 仍作为兼容入口转发至新解析模块，
binary v2 只是一条显式选择的新路径，尚未在 Unity 部署端启用。

阶段提交：

- `cabf788 refactor: extract pure legacy vr protocol`
- `575bce7 feat: add versioned vr binary protocol v2`
- `1557317 refactor: add independent vr receiver`
- `3ff7f51 test: add scriptable mock vr source`

## 2. Components Changed

| 组件 | 变更 | 单一责任 |
|---|---|---|
| `src/airo_doffy/devices/vr/base.py` | 已沿用 | 定义 latest-only `VRInputSource` port |
| `src/airo_doffy/devices/vr/protocol.py` | 新建 | 检测、解析和类型化迁移期 VR 消息 |
| `src/airo_doffy/devices/vr/binary_v2.py` | 新建 | 严格编码/解码 binary v2 状态包 |
| `src/airo_doffy/devices/vr/receiver.py` | 新建 | 接收、打时间戳、解码、拒绝陈旧包并发布最新状态 |
| `src/airo_doffy/devices/vr/mock.py` | 新建 | 生成或回放 VR 状态并注入确定性异常 |
| `src/airo_doffy/devices/vr/__init__.py` | 修改 | 公开 VR ports、协议、receiver 和 mock |
| `parse_vr.py` | 改为 wrapper | 保持旧 import 与字典返回契约 |
| `docs/protocols/vr_binary_v2.md` | 新建 | 记录 Unity 可实现的字节布局与序列规则 |
| `tests/unit/test_vr_protocol.py` | 新建 | 固定全部旧格式及 typed conversion |
| `tests/unit/test_vr_binary_v2.py` | 新建 | 验证 v2 布局、round trip 和严格拒绝 |
| `tests/unit/test_vr_receiver.py` | 新建 | 验证接收、聚合、陈旧拒绝、错误与生命周期 |
| `tests/unit/test_mock_vr.py` | 新建 | 验证轨迹、按钮、超时、延迟、丢包与乱序 |

## 3. Responsibilities and Dependencies

### Pure protocol

`protocol.py` 只依赖：

- 标准库 `base64`、`struct` 和类型工具；
- `core.types` 中不可变 VR models；
- 同域 `binary_v2.py`。

它不创建 socket，不读取相机，不启动 thread，也不访问其他设备或 runtime 状态。

### Binary v2

`binary_v2.py` 只依赖标准库和 `core.types/core.errors`。固定 header 为 24 bytes：

```text
<4sBBBBIQI
magic, version, mode, entity_count, flags,
uint32 sequence, uint64 source_timestamp_ns, uint32 payload_length
```

controller entity 固定为 48 bytes；hand entity 包含 side/flags/joint count、可选 wrist
和固定 26 个 `float32 xyz` joints。解码要求完整长度、已知 flags、合法 entity count、
零 reserved 字段及无 trailing bytes。

### Receiver

`VRReceiver` 接收一个窄 `RawVRTransport`，也可用于测试时直接 push 消息。它拥有：

- 一个 VR receive worker；
- 一个 constant-memory `LatestValueBuffer`；
- 每个 controller stream 或 hand side 的最后 wire sequence；
- malformed、stale、accepted 和 transport error counters；
- 显式 start、幂等 close 和可观察 health error。

controller stream 作为一个整体排序；左右 hand 分别排序，因此相同 frame id 的左右手包
可以聚合。uint32 sequence 使用模数顺序，允许 `0xffffffff -> 0` 正常回绕。

### Mock source

`MockVRInputSource` 是 on-demand source，不创建隐藏 thread。它支持：

- controller 或 hand 默认有效状态；
- 任意 `VRInputState` 轨迹；
- 保留脚本中的 position、button 和 sequence；
- 相邻包成对交换；
- 每 N 次读取丢包；
- 人工延迟；
- 启动后超时进入 stale/no-fresh-input；
- 运行中强制 stale；
- 有限回放或循环回放。

`create_mock_vr(config, network)` 与现有 `VRSourceFactory` 契约一致，但不使用 network，
证明 mock 不依赖真实地址。

## 4. Coupling Removed

- VR packet parsing 不再要求构造 UDP/WebRTC、camera 或 robot manager。
- 旧 controller/hand parser 不再分散在多个 manager 调用点中。
- wire decode 与接收循环、latest-state storage 和 stale policy 分离。
- receiver 只依赖 raw-message transport port，不知道 socket、DataChannel 或 signaling。
- 左右手聚合状态由 receiver 私有持有，不再写入外部 holder。
- malformed/stale/transport error 不再隐含在共享字典或裸异常中。
- binary serialization 不依赖 Unity、网络或 runtime。
- VR 故障场景不再要求真实 Quest，可供后续 mapping、watchdog 和 session 测试复用。

## 5. Remaining Coupling

- root `camera_udp.py` 与 `WebRTC_udp.py` 仍拥有旧 socket/DataChannel receive 路径，
  尚未组合 `VRReceiver`。
- root manager 仍通过兼容 `parse_vr.py` 使用旧字典模型；typed state 尚未接入 mapping。
- 生产 UDP/DataChannel adapter 尚未从旧 manager 中抽出为 `RawVRTransport` 实现。
- binary v2 尚未在 Unity sender 中实现，当前运行时仍默认使用旧格式。
- hand text 与 HB binary 的单位和 joint order 来自旧实现，仍需 Quest 实机确认。
- stale input 当前只在 receiver 层报告；Phase 9 watchdog 才负责触发安全 hold。

这些剩余项不会反向引入 camera、robot、tactile、recording 或 visualization 到新 VR
package。

## 6. Behavior Preservation

以下旧行为未改变：

- controller 旧格式和新 CSV 格式；
- hand text 格式；
- `HB,` base64 little-endian 格式；
- root `parse_data()` 和 `parse_hand_data()` 的字典形状；
- 旧 manager 的 UDP/WebRTC receive 行为；
- controller 坐标、button、trigger 和 gripper mapping；
- 数据集 schema、robot safety、control frequency 与视频协议。

旧无 frame id controller 包继续使用 receive time 形成 sequence；其毫秒 timestamp
继续转换为纳秒。新 controller CSV 继续使用发送端 frame id 和 timestamp。

新 API 的受控行为：

- malformed 消息返回未接受并增加计数；
- duplicate、倒序及恰好半个 uint32 空间的歧义序列被视为 stale；
- binary v2 使用明确的 `float32`、little-endian、version、sequence 和 timestamp；
- receiver 重新分配单调的本地 output sequence，同时保留 entity wire sequence。

binary v2 是 opt-in，未静默改变任何现有 Unity 消息。

## 7. Thread and Loop Ownership

VR receive loop 的边界：

| 属性 | 定义 |
|---|---|
| Input | `RawVRTransport.receive(timeout_s)` 返回的完整消息 |
| Output | 最新一个不可变 `VRInputState` |
| Target frequency | transport 驱动；poll timeout 默认 50 ms |
| Owned resources | receiver thread、stop event、latest buffer、sequence/counters |
| Buffer | constant-memory latest-value buffer |
| Queue capacity | 1 |
| Drop policy | 新状态覆盖旧状态；malformed/stale 不发布 |
| Shutdown | stop event、transport close、bounded join |
| Error behavior | 保存 health error，增加 transport error，并停止 receive loop |
| Watchdog | 本阶段只暴露 stale/no-update；安全 hold 留给 Phase 9 |

Mock source 没有 background loop；每次 `read_latest()` 由调用方驱动一次状态产生或故障注入。

## 8. Tests

Phase 6 专项：

- `test_vr_protocol.py`：6 个 legacy/golden/typed tests；
- `test_vr_binary_v2.py`：4 个 layout/round-trip/validation tests；
- `test_vr_receiver.py`：4 个 lifecycle/stale/wrap/aggregation/transport tests；
- `test_mock_vr.py`：5 个 trajectory/button/stale/delay/loss/reorder/factory tests。

阶段验收：

| 验证 | 结果 |
|---|---|
| 全部 v2 unit tests | 135/135 通过 |
| package/compatibility tests | 7/7 通过 |
| Phase 6 专项 tests | 19/19 通过 |
| VR package import | 通过，未加载可选硬件 SDK |
| editable install | `--no-deps --no-build-isolation` 通过 |
| tracked Python AST | 168 个文件通过 |
| `src/` 100 字符行检查 | 通过 |
| `git diff --check` | 通过 |

当前最小环境没有安装 `ruff` 和 `pyright`，因此没有伪造其结果。首次默认 editable
install 因隔离构建尝试联网获取 setuptools 失败，改用已安装 build backend 的离线模式
后成功。旧 `tests/test_realman_teleop_loop.py` 仍因缺少 `cv2` 不属于默认回归集合，
与前阶段记录一致。

## 9. Hardware Validation

仍需 Quest/Unity 与真实网络验证：

1. 旧 controller CSV 两个版本的字段顺序、单位、button 和 frame cadence。
2. hand text 与 HB binary 的 26-joint 顺序、左右手 side code 和 wrist semantics。
3. Unity/C# 对 binary v2 header、little-endian、`float32` 和 flags 的交叉语言 golden。
4. controller 两 entity 与一/两 hand entity 的实际 v2 round trip。
5. sender restart、duplicate、reorder、packet loss 和 uint32 wrap。
6. device timestamp 与 PC receive monotonic timestamp 的延迟分析。
7. UDP/DataChannel adapter close 能否解除 blocking receive 并在时限内 join。
8. 30 分钟运行中的 thread、socket、内存和 malformed/stale counters。
9. Unity 迁移期间 legacy 与 v2 双协议互操作。

## 10. 下一阶段

按计划进入 Phase 7 Video Streaming Refactor：

1. 定义独立 processor、encoder 与 transport ports。
2. 建立旧 JPEG UDP 与 WebRTC 行为 baseline。
3. 实现 bounded/drop-oldest 的低延迟 H.264 encoder。
4. 从 root manager 中提取 WebRTC video transport。
5. 隔离并标记 legacy JPEG UDP，再增加 RTP/UDP benchmark path。
