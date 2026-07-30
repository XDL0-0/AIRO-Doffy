# Phase 4 重构报告：触觉与 Wrench

日期：2026-07-30

分支：`codex/v2-refactor`

前置阶段：机器人后端重构（`dd83a92`）

## 1. 阶段结论

Phase 4 的软件重构已完成。4-taxel BLE MagTouch、mock tactile、原始 wrench
读取边界、重力与偏置补偿、滤波和兼容入口现在具有独立模块与无硬件测试。

新的支持路径只产生不可变 `TactileSample` 或 `WrenchSample`，不再写入外部 holder，
也不直接导入 VR、相机、记录器或可视化代码。构造组件不会扫描或连接硬件；
BLE SDK、NumPy 和设备连接只在 `SensorCommBle4Source.start()` 后进入。

真实 BLE、UR/RealMan 力传感器和 SDK shutdown 尚未验证，因此本阶段是
“软件完成、硬件验证待执行”，不是实机验收完成。

## 2. Components Changed

| 组件 | 变更 | 单一责任 |
|---|---|---|
| `devices/tactile/base.py` | 使用既有 typed port | 定义 `TactileSensor` lifecycle、latest read 与 recalibration |
| `devices/tactile/source.py` | 新建 | 解码 24 字节 BLE4 包，拥有 BLE event loop/thread，并处理重连和关闭 |
| `devices/tactile/filters.py` | 新建 | 纯标准库 baseline、deadband、drift、slew、EMA 与可选 Kalman |
| `devices/tactile/magtouch_ble4.py` | 新建 | 私有 thread-safe calibration/latest state，并产生 `TactileSample` |
| `devices/tactile/mock.py` | 新建 | fixed、random、periodic、disconnected 与 delayed 无硬件触觉源 |
| `devices/tactile/legacy.py` | 移入 | 保留旧 holder、DDS 和 panel 行为，仅由兼容入口惰性加载 |
| 根 `tactile_4point.py` | 缩减为 facade | 保留旧导出名和 CLI 转发，不在普通导入时加载 SDK |
| `devices/wrench/base.py` | 新建 | 定义不拥有机器人 lifecycle 的 typed latest wrench port |
| `devices/wrench/robot_source.py` | 新建 | 将已有 `RobotBackend` 状态投影为原始 `WrenchSample` |
| `devices/wrench/compensation.py` | 新建 | 纯标准库 payload gravity、sensor bias calibration 与 baseline reset |
| `devices/wrench/filters.py` | 新建 | 纯标准库 deadband、moving average 与 low-pass |
| `devices/wrench/pipeline.py` | 新建 | 组合 compensation 和 filtering，并保留样本元数据 |
| 根 `force_filter.py` | 修改 | 保留旧 NumPy 输入/返回契约，委托给纯过滤器 |
| 根 `utils.py` | 修改 | 保留旧 NumPy `GravityCompensator` API，委托给纯补偿实现 |
| typed config/default profile | 修改 | 固化 BLE filter、noise、clip、drift 与 wrench 参数 |
| `pyproject.toml` | 修改 | 为 BLE4 和 legacy tactile extras 声明直接运行依赖 |

阶段提交：

- `a2ea200 refactor: extract ble4 tactile filtering`
- `ed237d2 refactor: add mock tactile sensor`
- `ad15ad6 refactor: extract pure wrench filtering`
- `d0448a3 refactor: add private-state ble4 sensor`
- `d14b850 refactor: isolate legacy tactile facade`
- `0dfaf2d refactor: separate wrench acquisition and compensation`
- `0dbd57a test: isolate optional dependency import check`
- `8dfce1b build: declare tactile runtime dependencies`

## 3. Dependencies

支持路径的依赖边界：

- `tactile/base.py`：`core.interfaces`、`core.types`。
- `tactile/filters.py`：标准库、`config.models`、`core.errors`。
- `tactile/magtouch_ble4.py`：typed config/core、tactile port/filter/raw source。
- `tactile/source.py`：标准库；仅在启动 worker 后延迟加载 NumPy 和
  `sensor_comm_dds`。
- `tactile/mock.py`：标准库、typed config/core。
- `wrench/base.py`：`core.types`。
- `wrench/robot_source.py`：`RobotBackend` 与 `core.types`。
- `wrench/compensation.py`、`filters.py`：标准库与 `core.errors`。
- `wrench/pipeline.py`：typed sample、compensator 与 filter。

新 `airo_doffy.devices.tactile` 和 `airo_doffy.devices.wrench` 包的普通导入
不会加载 NumPy、SciPy、OpenCV、BLE SDK 或机器人 SDK。

## 4. Coupling Removed

- BLE callback 不再修改任意 `cu.tactile_data`、bytes 或 timestamp 属性。
- BLE acquisition 不再启动 DDS publisher、GUI process 或读取 VR reset 状态。
- BLE packet decode、16-bit unwrap、baseline calibration 和 signal filtering
  可各自在无 SDK 环境测试。
- 最新触觉状态归 `MagtouchBle4Sensor` 私有锁保护；消费者只能读取不可变 snapshot。
- BLE event loop 和 thread 由 raw source 显式拥有，`close()` 可重复调用并等待退出。
- Wrench acquisition 不再与 gravity、deadband、moving average 和 low-pass
  写在同一 teleop method 中。
- 新 wrench pipeline 不依赖 NumPy、SciPy、UR RTDE 或 RealMan SDK。
- 旧 root API 只负责兼容类型转换，数值算法不再复制两套。

## 5. Remaining Coupling

- `main.py` 和旧诊断入口仍通过根 `tactile_4point.py` 使用
  `FourPointTactileBleReader` 与 holder mutation；这条路径为兼容保留。
  新 runtime/session 阶段需要改为组合 `MagtouchBle4Sensor`。
- legacy tactile 内仍包含 DDS publish、panel process、CLI 和全局 loguru
  配置；它已隔离，但尚未删除。
- UR/RealMan backend 仍负责 vendor raw wrench read。
  `RobotStateWrenchSource` 只建立窄 typed 边界，后续 session 需要完成 wiring。
- 旧 `robot_teleop.py`、`realman_teleop.py` 和 `inference.py` 尚未统一改用
  `WrenchProcessor`；兼容 adapter 保持当前数值契约。
- 当前 BLE SDK adapter 按仓库旧版 `sensor_comm_dds` API 实现，真实安装版本、
  MAC enum、HCI、unsubscribe/disconnect 和断线事件仍需实机确认。

## 6. Behavior Preservation

BLE 协议与数值边界：

- 每个样本继续读取 24 字节、4 taxels × 3 axes。
- 每轴继续按 big-endian 16-bit word 解码后执行 bitwise complement。
- wrap 继续从 `value - 65536`、`value`、`value + 65536` 中选择最接近前值者。
- calibration 继续使用 per-axis median baseline 和 MAD deadband。
- processing 次序保持 deadband、absolute clip、delta limit、EMA、可选 Kalman。
- unloaded baseline drift 和断线后的 dynamic-filter reset 行为保留。
- 首次连接与运行中断线都会重试；只有显式 `close()` 才结束 worker。

支持的 v2 API 不再发布 DDS、启动 panel 或写外部 holder，这是计划明确要求的
解耦行为。旧 API 仍通过 lazy legacy facade 保留原行为。

Wrench 数值边界：

- 保留 `[Fx,Fy,Fz,Tx,Ty,Tz]` 顺序。
- 根 `WrenchFilter` 继续返回 NumPy array。
- deadband 仍从超过阈值的值中减去阈值；moving average 先于 low-pass。
- gravity convention 保持 base-frame `Z` 向上、重力为 `[0,0,-9.81] m/s²`。
- gravity torque 继续由 base-frame COM 与 gravity force 的叉积计算。
- sensor bias 继续取 calibration residual 的算术平均。
- baseline reset 现在是显式 API，并同时清除 compensation/filter dynamic state。

没有改变 dataset schema、网络协议、机器人 safety limit 或 control frequency。

## 7. Tests

阶段新增或扩展的独立测试：

- `test_tactile_filters.py`：5 个 calibration/filter 数值 golden。
- `test_mock_tactile.py`：5 个 mode、lifecycle 和 fault behavior。
- `test_magtouch_ble4.py`：5 个 packet、unwrap、private state、initial retry、
  disconnect/recalibration/close/factory 测试。
- `test_wrench_filters.py`：5 个纯过滤和根 NumPy 兼容测试。
- `test_wrench_processing.py`：5 个 raw source、gravity、bias、pipeline、
  invalid input 和 `utils` adapter 测试。
- `test_package_layout.py`：lazy tactile facade 和 optional import 隔离。
- `test_domain_interfaces.py`：在新进程中验证 domain port 不加载可选 SDK。

阶段验收结果：

| 验证 | 结果 |
|---|---|
| 全部 v2 unit tests | 100/100 通过 |
| package/compatibility tests | 7/7 通过 |
| Phase 4 专项 unit tests | 25/25 通过 |
| `import tactile/wrench` optional dependency check | 通过 |
| editable install | `airo-doffy==2.0.0.dev0` 构建安装通过 |
| 全仓 Python AST | 154 个文件通过 |
| `src/` 100 字符行检查 | 通过 |
| `git diff --check` | 通过 |

`ruff` 与 `pyright` 在当前最小环境未安装，因此没有伪装为已执行；Phase 14
仍需在 dev extra 环境执行。旧 `tests/test_realman_teleop_loop.py` 的 28 个测试
仍在收集阶段因缺少 `cv2` 失败，错误为
`ModuleNotFoundError: No module named 'cv2'`，未计入通过数。

## 8. Hardware Validation

尚未连接 BLE tactile、UR3e、UR5e 或 RM75。

进入生产组合前必须完成：

1. `H-TAC`：确认 24 字节 packet、axis order、scale、MAC enum 与 HCI。
2. `H-TAC`：确认首次连接重试、运行中断线、重新订阅和 baseline 保留。
3. `H-TAC`：确认显式 close、unsubscribe、disconnect 和 thread join。
4. `H-TAC`：对比旧实现与新实现的 raw、baseline、deadband 和 filtered golden。
5. `H-UR`：确认 RTDE wrench frame、tool rotation、payload mass/COM 和 bias reset。
6. `H-RM`：确认 `zero_force_data` 的 frame/unit 和 controller-side compensation。
7. 两种机器人都需验证静止漂移、接触方向、过载/断线和 shutdown。

## 9. 下一阶段

按计划进入 Phase 5 Camera Acquisition Refactor：

1. 先刻画 `camera_udp.py` 和 `realsense_multi.py` 的 serial、resolution、
   timestamp、depth/color 对齐和 shutdown 行为。
2. 提取只负责 discovery/acquisition 的 `RealSenseCameraSource`。
3. 增加 fixed/recorded/delayed/disconnected mock camera。
4. 把 resize、crop、rotation、color conversion 和 depth conversion 放入独立
   `FrameProcessor`，不启动网络或访问 runtime state。
