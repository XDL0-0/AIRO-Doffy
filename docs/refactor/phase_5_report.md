# Phase 5 重构报告：相机采集与 Frame Processing

日期：2026-07-30

分支：`codex/v2-refactor`

前置阶段：触觉与 wrench 重构（`cbf1fdf`）

## 1. 阶段结论

Phase 5 的软件重构已完成。RealSense discovery/acquisition、mock camera 和
packed frame processing 现在是三个独立组件。它们通过既有 `CameraFrame`、
`ProcessedFrame`、`CameraSource` 与 `FrameProcessor` 契约组合，不创建
UDP/WebRTC transport，不解析 VR，不访问 tactile/robot/recorder/visualizer。

新的 RealSense source 构造时不导入 SDK、不枚举设备、不创建 pipeline；
只有显式 `start()` 才选择 serial 和创建设备。相机 worker、latest buffers、
health error 与 SDK handle 都由 source 私有拥有。

真实 RealSense 尚未连接。旧 `camera_udp.py`、`WebRTC_udp.py` 和
`InferenceCameraManager` 暂时保持原路径，因此没有在本阶段暗中改变现有视频、
VR、record 或 dataset 行为。

## 2. Components Changed

| 组件 | 变更 | 单一责任 |
|---|---|---|
| `docs/refactor/camera_behavior_baseline.md` | 新建 | 固定 discovery、capture、processing、timestamp 和 shutdown 旧行为 |
| `config/models.py::CameraConfig` | 扩展 | 保存 resolution、设备 FPS、采集 cadence、stream/serial、retry 参数 |
| `configs/default.yaml` | 扩展 | 显式记录默认 30 Hz acquisition cadence |
| `devices/cameras/base.py` | 扩展 | 定义 color-only 与 color+depth typed source ports |
| `devices/cameras/realsense.py` | 新建 | discovery、serial selection、SDK initialization、capture、timestamp、shutdown |
| `devices/cameras/mock.py` | 新建 | static、generated、in-memory video、drop、delay 与 disconnect |
| `streaming/video/frame_processor.py` | 新建 | packed crop、zoom、resize、rotation、color 与 even-dimension preparation |
| camera/video package exports | 修改 | 公开新 ports、sources 和 processor |
| `pyproject.toml` | 修改 | 声明 RealSense wrapper 的直接 optional dependency |

阶段提交：

- `d28520c docs: characterize camera pipeline`
- `cf8bfff refactor: add camera acquisition configuration`
- `36b0dbc refactor: add isolated realsense camera source`
- `5cde9a0 refactor: add mock camera source`
- `75d570e refactor: add packed frame processor`
- `fdac0de fix: clear recovered camera health error`
- `5672b72 build: declare realsense wrapper dependency`

## 3. Dependencies

- `cameras/base.py`：`core.interfaces`、`core.types`。
- `cameras/realsense.py`：typed config、latest buffer、clock、core frame/error。
  `pyrealsense2` 只用于未指定 serial 时的启动期 discovery；
  `airo_camera_toolkit` 只用于启动期设备创建。
- `cameras/mock.py`：标准库、typed config/core。
- `frame_processor.py`：标准库、typed frame、clock/error。

普通导入：

```text
airo_doffy.devices.cameras
airo_doffy.streaming.video
```

不会加载 NumPy、OpenCV、`pyrealsense2`、`airo_camera_toolkit`、aiortc 或 av。

`camera-realsense` extra 现在直接声明：

- `airo-camera-toolkit`
- `opencv-python`
- `pyrealsense2`

## 4. Coupling Removed

- RealSense discovery 不再需要构造 UDP/WebRTC manager。
- serial selection 和 camera initialization 不再绑定 socket 分配。
- acquisition worker 不读取 VR command、record flag、tactile 或 robot state。
- RGB/depth 通过不可变 frame 和 constant-memory latest buffer 发布。
- capture sequence、monotonic timestamp 和 health error 不再藏在共享 manager dict。
- color/depth source 的关闭不负责 socket、signaling 或 DataChannel。
- crop、resize、zoom、rotation 和颜色转换不再要求运行 camera 或 transport。
- mock camera 可独立驱动后续 encoder、transport、recording 和 runtime 测试。

## 5. RealSense Source Behavior

`RealSenseCameraSource`：

- 一个实例拥有一个 serial/stream；多相机通过多个实例组合。
- 显式 serial 优先；`serial_number=None` 保留 first-discovered 兼容选择。
- toolkit 参数保留 device FPS、resolution、depth、no pointcloud 和 hole filling。
- 默认 device request 仍为 `640×480@60`，成功读取后额外按 30 Hz cadence 等待。
- RGB 使用 `get_rgb_image()` 并标记 `RGB8`。
- depth 开启时保留旧 `_retrieve_depth_map()` 边界并标记 `DEPTH_U16`。
- 同一 acquisition 的 color/depth 使用同一 sequence 和 timestamp。
- 每种流只保存最新不可变 frame。
- depth 单次 `RuntimeError`/`AttributeError` 不阻止本轮 RGB。
- RGB 连续错误默认 10 次后结束 worker；短暂错误恢复后 health error 清零。
- `wait_for_first_frame()` 可区分成功、timeout 和 worker 终止。
- `close()` 先请求停止 SDK，再 join worker；无法停止或 SDK cleanup 失败会显式失败。
- 未启动对象的 `close()` 和重复 `close()` 是幂等的。

## 6. Mock Camera

`MockCameraSource` 支持：

- 固定 packed frame。
- 默认 sequence pattern 或注入 generator。
- 内存 frame 序列循环/单次播放。
- 每 N 次 acquisition 丢一帧。
- 每次读取的人工 latency。
- 运行中断连/恢复。
- RGB、BGR、gray 和 depth packed shape validation。

它采用与 Phase 4 mock tactile 相同的 on-demand 语义，不创建隐藏 thread。

## 7. Frame Processor

`PackedFrameProcessor` 的固定次序：

```text
crop
  -> center zoom
  -> nearest-neighbor resize
  -> clockwise rotation
  -> even-dimension crop for encoder preparation
  -> color conversion
```

支持：

- packed RGB8、BGR8、GRAY8、DEPTH_U16。
- RGB/BGR channel swap。
- gray↔three-channel conversion。
- RGB/BGR→gray 的固定整数 luminance。
- crop bounds validation。
- nearest-neighbor resize。
- center zoom。
- 0/90/180/270 度 rotation。
- H.264 等 encoder 常用的偶数 width/height preparation。
- 完整保留 sequence、source/receive timestamp、clock domain 和 stream id。

纯 Python processor 是 dependency-free reference implementation。高分辨率几何
处理在接入低延迟生产路径前仍需 OpenCV/硬件加速 adapter 与 benchmark。

## 8. Remaining Coupling

- 根 `camera_udp.py` 和 `WebRTC_udp.py` 仍各自重复 RealSense discovery、
  capture、OpenCV zoom 和 shared dict；必须等 Phase 6/7 拆出 VR 与 transport
  后再缩减为组合 facade。
- `InferenceCameraManager` 仍使用第三套直接相机路径且没有 close。
- 旧 manager 仍在构造阶段枚举/创建相机，新 source 的安全 lifecycle 尚未接线。
- multi-camera serial→logical stream mapping 尚未进入 session composition。
- depth 仍调用 toolkit 私有 `_retrieve_depth_map()`；dtype、单位、alignment
  和版本稳定性需要实机确认。
- source 假定 toolkit RGB 为 packed uint8、depth 为 packed uint16。
- pure nearest-neighbor processor 不等同于旧 OpenCV `INTER_LINEAR` zoom；
  新 processor 尚未替换旧路径，所以当前兼容输出没有变化。

## 9. Behavior Preservation

现有 root runtime 未切换，以下均未改变：

- JPEG chunk header/payload。
- WebRTC signaling、track 与 DataChannel。
- UDP port allocation。
- VR controller/hand parsing。
- record/rollback flags。
- dataset RGB/depth schema。
- legacy OpenCV RGB→BGR 和 `INTER_LINEAR` center zoom。

新 API 保留主要 acquisition 默认值：

- resolution `(640, 480)`。
- device FPS `60`。
- application acquisition cadence `30 Hz`。
- depth 默认关闭。
- retry 10 次、retry delay 1 秒。

新增显式 serial、sequence、typed health 和 strict shutdown 是 v2 API 的受控改进，
没有反向修改 legacy manager。

## 10. Tests

Phase 5 专项：

- `test_realsense_camera.py`：5 个 discovery、serial、RGB/depth、retry recovery、
  bounded failure、lazy factory、close 测试。
- `test_mock_camera.py`：5 个 static、generated、video、drop、delay、
  disconnect、factory、validation 测试。
- `test_frame_processor.py`：6 个 color、crop、zoom、resize、rotation、
  depth、metadata、encoding-preparation golden。
- config tests 扩展 camera safe defaults 与参数 validation。
- domain import test 覆盖 camera package 的 optional dependency 隔离。

阶段验收：

| 验证 | 结果 |
|---|---|
| 全部 v2 unit tests | 116/116 通过 |
| package/compatibility tests | 7/7 通过 |
| Phase 5 专项 unit tests | 16/16 通过 |
| camera/video import optional dependency check | 通过 |
| editable install | `airo-doffy==2.0.0.dev0` 构建安装通过 |
| 全仓 Python AST | 160 个文件通过 |
| `src/` 100 字符行检查 | 通过 |
| `git diff --check` | 通过 |

`ruff`、`pyright` 和真实 OpenCV/RealSense tests 在当前最小环境不可用，未伪装为
已执行。旧 `tests/test_realman_teleop_loop.py` 仍因缺 `cv2` 无法收集，
与 Phase 4 报告一致。

## 11. Hardware Validation

未连接任何 RealSense。`H-CAM` 仍必须覆盖：

1. D435/D455 等部署型号的 640×480@60 支持。
2. 显式 serial、first-device fallback 和多相机 logical stream 映射。
3. RGB shape/dtype/channel order。
4. depth shape/dtype/单位、hole filling 和 RGB-depth alignment。
5. 30 Hz application cadence 与 P50/P95/P99 interval。
6. color/depth sequence 与 monotonic timestamp。
7. 临时读取错误恢复、连续 10 次错误、拔插。
8. `pipeline.stop()`、worker join、重复 close 和 thread leak。
9. 多相机 USB bandwidth 与 30 分钟稳定运行。
10. 新 processor 与旧 OpenCV zoom 的视觉/性能对比。

## 12. 下一阶段

按计划进入 Phase 6 VR Input and Protocol Refactor：

1. 用现有 controller/hand/HB 行为 baseline 建立纯 parser golden。
2. 把 controller、hand text 与 hand binary decode 移到
   `devices/vr/protocol.py`。
3. 把 UDP/WebRTC receive 与 protocol semantics 分开。
4. 增加 mock VR source。
5. 设计 binary state protocol v2，但不静默改变 Unity 兼容 wire format。
