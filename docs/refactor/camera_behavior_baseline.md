# Phase 5 Camera Behavior Baseline

日期：2026-07-30

范围：`camera_udp.py`、`WebRTC_udp.py`、`inference.py` 中的 RealSense
discovery、capture、processing 与 shutdown。

## 1. 重复实现

`CameraUDPManager` 和 `WebRTCUDPManager` 各自实现：

1. `pyrealsense2.context().query_devices()` 枚举。
2. 按 SDK 枚举顺序读取 name 与 serial。
3. 用 `airo_camera_toolkit.Realsense` 创建相机。
4. 每台相机一个 acquisition thread。
5. 在共享 dict 中保存 RGB、depth 与 timestamp。
6. RGB→BGR、center zoom 与 dataset RGB snapshot。
7. manager shutdown 时停止 thread 与 SDK pipeline。

两者真正不同的是下游 JPEG/UDP 与 WebRTC transport；采集行为不应重复。
`InferenceCameraManager` 第三次实现同样的枚举和创建，但没有显式 close。

## 2. Discovery 与 Selection

- 构造 manager 时立即导入 SDK、枚举并创建硬件对象。
- 枚举结果没有稳定排序或配置映射；`camera_0` 等名称取决于 SDK 顺序。
- UDP/WebRTC 最多创建 5 台相机。
- inference 创建枚举到的全部相机，没有 5 台限制。
- 某个 index 缺 serial 时跳过，但后续循环仍可能按原 count 索引 dict。
- 没有部署级 serial allowlist，也没有“未配置 serial 时禁止连接”的安全边界。

新 source 必须允许显式 serial；`None` 只可作为明确的 first-device
兼容选择，不能在构造阶段连接。

## 3. Initialization

manager 路径传给 toolkit：

| 参数 | 默认值 |
|---|---|
| `fps` | `60` |
| `resolution` | `(640, 480)` |
| `enable_depth` | config，默认 `False` |
| `enable_pointcloud` | `False` |
| `enable_hole_filling` | 与 depth 相同 |
| `serial_number` | 枚举得到的 serial |

inference 强制关闭 depth 与 hole filling。设备对象构造时是否已经启动 pipeline
由当前 `airo_camera_toolkit` 版本决定，旧代码没有独立的 initialize/start 边界。

## 4. Acquisition、Timestamp 与 Latest State

- RGB 使用 `camera.get_rgb_image()`，按调用方约定视为 RGB。
- depth 开启时调用 toolkit 私有方法 `_retrieve_depth_map()`。
- depth 的 `RuntimeError` 或 `AttributeError` 被忽略，本轮仍可发布 RGB。
- `time.monotonic_ns()` 在 RGB/depth 调用之后生成。
- 同一轮 RGB 与 depth 共用 timestamp。
- 共享 dict 只保留每个 stream 的最新值，没有无界 frame queue。
- 每轮成功采集后额外 `sleep(1 / 30)`，即使相机请求为 60 FPS。
- UDP 路径连续 `RuntimeError` 10 次后结束该相机 thread；每次错误等待 1 秒。
- WebRTC 路径首次 `RuntimeError` 就结束该相机 thread。
- thread 结束没有 typed health 状态，消费者可能继续读最后一帧。

新 source 默认保留 30 Hz 额外 cadence 和 10 次 retry，但通过显式 health
error 暴露终止。sequence 必须单调，latest state 必须是不可变 `CameraFrame`。

## 5. Frame Processing

两个 manager 的 `data_process()` 相同：

1. 非 `uint8` RGB 执行 `frame * 255` 后转 `uint8`。
2. OpenCV `COLOR_RGB2BGR`。
3. 读取每 camera zoom。
4. zoom 大于 1 时先 `INTER_LINEAR` resize，再做 center crop 回原尺寸。
5. zoom 小于等于 1 时 resize 回原尺寸，实质不缩小视野。
6. transport 使用 BGR zoom frame。
7. dataset 保存未 zoom 的原始 RGB frame。

crop、resize、rotation、gray/depth conversion 没有可复用组件。新
`FrameProcessor` 的几何次序必须显式固定，输出仍为 packed immutable bytes；
在 legacy manager 接线前不改变旧 OpenCV 路径。

## 6. Shutdown

- manager 先将 `running=False`。
- 每个 daemon thread 最多 join 1 秒。
- camera 优先调用 `pipeline.stop()`，否则调用 `close()`。
- SDK stop/close 异常只写 warning，manager 继续关闭 socket。
- 没有检查 1 秒后仍存活的 thread。
- inference camera manager 没有 close。

新 source 必须拥有自己的 worker 与 SDK handle，`close()` 幂等，并在无法终止
worker 时抛出 lifecycle failure，不能静默宣称资源已释放。

## 7. Compatibility Boundary

Phase 5 新组件不会立即替换 UDP/WebRTC manager 的 transport/VR/record
行为。根 manager 继续保持旧 API，直到后续 transport、VR protocol 与 runtime
阶段具备组合条件。

本阶段不得改变：

- JPEG chunk wire format。
- WebRTC signaling/DataChannel 行为。
- dataset RGB/depth schema。
- VR command/record flags。
- camera socket port 分配。

## 8. Hardware Checks

`H-CAM` 必须覆盖：

- 每个部署型号的 640×480@60 支持情况。
- SDK 枚举顺序与显式 serial 的稳定映射。
- RGB/BGR 色彩、depth dtype/单位、RGB-depth 时序。
- 实际 capture FPS、P50/P95/P99 frame interval 与 timestamp monotonicity。
- 拔插、10 次读取错误、pipeline stop、close 与 thread leak。
- 多相机 30 分钟运行、USB bandwidth 和 shutdown。
