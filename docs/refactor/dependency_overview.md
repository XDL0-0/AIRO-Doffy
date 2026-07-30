# AIRO-Doffy v2.0 依赖概览

## 1. 结论

当前仓库没有发现由静态 Python import 构成的直接循环，但存在更难治理的运行时双向耦合：producer、consumer、orchestrator 通过 manager 的公开可变字段互相读写，并把设备采集、协议解析、传输、录制控制和生命周期揉在同一对象中。最紧密的耦合中心是 `config.py`、`utils.py`、`camera_udp.py` / `WebRTC_udp.py`、`robot_teleop.py` / `realman_teleop.py`、`main.py` 与 `dataset.py`。

静态图还会低估耦合：`main.py` 和多个工具使用函数内 import 来加载触觉、可视化和硬件依赖；这些边不会全部出现在普通顶层 import 扫描中。

## 2. 当前文本依赖图

箭头表示“左侧 import、构造、调用或直接读取/写入右侧”。方括号表示外部依赖或硬件。

```text
main.py
├─> config.py ─> utils.py
├─> camera_udp.py ─┬─> parse_vr.py ─> utils.py
│                  ├─> udp_comms.py ─> [UDP sockets]
│                  ├─> config.py
│                  ├─> utils.py
│                  └─> [pyrealsense2, OpenCV]
├─> WebRTC_udp.py ─┬─> parse_vr.py / udp_comms.py / config.py / utils.py
│                  ├─> [pyrealsense2, OpenCV]
│                  └─> [aiortc, aiohttp, av, WebSocket]
├─> robot_teleop.py ─┬─> robot_backend.py ─> [UR/RealMan SDK, Robotiq]
│                    ├─> data_schema.py
│                    ├─> force_filter.py
│                    ├─> utils.py
│                    └─> visualizer_config.py
├─> dataset.py ─┬─> data_schema.py
│               ├─> config.py / utils.py
│               └─> [h5py, LeRobot, Hugging Face Hub, video codecs]
├─(delayed)> tactile_4point.py ─> [BLE MagTouch, sensor_comm_dds]
├─(delayed)> tactile.py ─> [serial, sensor_comm_dds]
├─(delayed)> visualizer.py ─> [multiprocessing, Matplotlib, OpenCV]
└─> visualizer_config.py

realman_teleop.py
├─> config.py / utils.py / force_filter.py / visualizer_config.py
├─> robot_backend.py ─> [RealMan SDK]
├─(factory)> camera_udp.py OR WebRTC_udp.py
└─(delayed)> visualizer.py

inference.py
├─> config.py / data_schema.py / utils.py
├─> robot_backend.py
├─(factory)> camera_udp.py OR WebRTC_udp.py
├─(optional)> tactile_4point.py / visualizer.py
└─> [torch, LeRobot policy/processor, checkpoint]

policies/forceflowpp/modeling_forceflowpp.py
├─> configuration_forceflowpp.py
├─> contact_labeling.py
├─> modules/adaln_dit.py
├─> modules/encoders.py
├─> modules/flow_matching.py
├─> prior_library.py
└─> [torch, LeRobot]

dataset_tool/*
├─> [HDF5/LeRobot/Parquet/video/Hub]
├─> dataset_tool/convert_delta_tcp_dataset.py
└─> dataset_tool/Visualize_hdf5_episodes.py

test_tool/*
├─> production modules for manual diagnostics
└─> [real camera, UDP, UR hardware, Matplotlib]
```

## 3. 扇入、扇出与变化放大

| 模块 | 依赖特征 | 风险 |
|---|---|---|
| `config.py` | 至少 12 个静态仓内 importer；默认值、校验、机器人专用派生值集中在一个可变 dataclass。 | 改一个字段可能破坏采集、推理、相机、触觉或测试；无法只安装某个硬件 extra。 |
| `utils.py` | 13 个静态 importer；同时含日志配置、机器人运动学、安全、滤波、姿态和重力补偿。 | 任何轻量模块 import `utils` 都被迫加载数值/机器人栈并接受全局日志副作用。 |
| `camera_udp.py` / `WebRTC_udp.py` | 对外呈现近似 manager API，但内部各自扇出到相机、VR、UDP、触觉、录制状态和线程。 | 两份实现会持续漂移；修复协议或 shutdown 必须重复。 |
| `robot_teleop.py` | 同时依赖 schema、backend、filter、safety/utils、visualizer config。 | 映射改动可间接改变硬件执行或记录格式，难以纯单测。 |
| `realman_teleop.py` | 1823 行，既是后端特化、实时 executor、mapping/safety、状态采集，又是 app。 | 线程所有权与 SDK 安全非常敏感，移动代码可能改变 >100 Hz 门槛。 |
| `main.py` | 顶层组合大量具体类，并直接访问它们的内部共享字段。 | 入口不是薄 orchestration；任何设备替换都要求修改主循环。 |
| `dataset.py` | 单类同时管理 episode 状态、两种 serializer、Hub 与破坏性 rollback。 | schema 与 I/O、恢复策略无法独立测试；LeRobot 私有布局升级风险高。 |
| `inference.py` | 同时做 CLI、policy loading、feature conversion、设备生命周期与控制循环。 | policy 错误与实机初始化顺序耦合；当前可能先移动机器人再发现 checkpoint 无效。 |

静态高价值公共契约是 `data_schema.py`、`parse_vr.py`、`udp_comms.py` 与 `force_filter.py`。这些文件规模不一定最大，但协议/数据语义的扇出高，必须先用 characterization tests 固定行为。

## 4. 循环与双向依赖

### 4.1 静态 import

对扫描到的仓内 import map 未发现强连通分量大于 1，即没有确定的 `A imports B imports A`。这不等于架构无循环。

### 4.2 运行时双向耦合

- `main.py` 启动 camera manager；manager 的 RX 线程写 `controller_data`、hand data、record/rollback flags 和 camera frames；`main.py` 轮询这些字段，又反向写 manager 的 `tactile_data`、清除命令 flag。控制流与数据流双向穿过同一对象。
- visualizer 本应是 consumer，但 visualizer command 又通过共享 handle/字段请求 reset、rollback、tactile recalibration，最后由采集/导出线程消费。
- tactile reader 接收整个外部 holder/control object，既读控制器按钮又写 tactile array、bytes、timestamp 和重校准状态。
- `DatasetRecorder` 被采集线程追加样本，被导出线程 serialize/close/rollback；协调依赖外部 `Event`、flag 与调用约定，而不是窄接口。
- RealMan 的 command loop、实时状态 callback、teleop object 和 app shutdown 通过锁、事件、线程列表和 SDK handle 互相约束。

这些是“共享状态环”，应改为单向 typed sample、command/event 和明确 owner，而不是仅消除 import 环。

## 5. 显式重点模块审计

### 5.1 `main.py`

- 模块导入时创建全局 `cfg = Config()`，配置不是由入口参数显式注入。
- `main` 直接选择具体 UDP/WebRTC 类、构造 robot teleop、等待 VR/robot、启动多个线程，并读写 manager 内部字段。
- `collect_loop` 同时读取机器人、相机、触觉与时间戳并构造 recorder payload。
- `export_loop` 根据 camera manager 上的命令 flag 写盘或 rollback。
- visualizer payload 在 app 中逐字段拼装，导致 runtime 知道 UI 模型。
- shutdown 有 `finally`，但大量线程只做 timeout join；没有统一 owner 能证明所有后台线程已退出。

建议：入口仅加载配置并调用 `DataCollectionSession.run()`；session 依赖窄 ports、typed latest-value buffers 与 `Lifecycle`，不依赖具体 SDK。

### 5.2 `camera_udp.py`

- `CameraUDPManager` 的主职责至少有八个：设备发现、采集、图像处理、JPEG 编码、分片协议、UDP transport、VR 协议/命令、触觉转发。
- 相机线程、VR RX、录制 RX、控制 RX、触觉 TX 的所有权都在同一个 manager。
- `camera_data`、`depth_data`、controller/hand data、record flags 等为跨线程可变状态；读者需要知道锁与字段约定。
- `close()` 用 timeout join 后继续释放相机/socket，不能证明仍存活线程不会再访问资源。

建议：`RealSenseCameraSource -> FrameProcessor -> Encoder -> VideoTransport` 单向流水线；VR state 与 reliable command 使用独立 receiver/router。

### 5.3 `WebRTC_udp.py`

- WebRTC 视频只占职责的一部分；其余 VR/record/resolution/tactile UDP 行为从 `camera_udp.py` 复制。
- `RealsenseCameraTrack` 同时从共享相机状态取帧并做视频适配；manager 又自行管理相机 read thread。
- 同时拥有 asyncio event loop thread、aiohttp runner、peer connections、DataChannel 和普通线程，shutdown 状态机复杂。
- signaling 与业务命令边界不清：`control` DataChannel 目前仅承担 zoom，pose 与录制命令仍走 UDP。

建议：复用统一 camera source 和 protocol service；`WebRtcVideoTransport` 只接受已定义的 frame/encoded frame 契约，signaling server 与 command channel 独立测试。

### 5.4 `robot_teleop.py`

- `RobotTeleop` 既计算 controller/hand mapping，又执行 Ruckig、workspace/joint/self-collision 安全、wrench filtering/compensation、夹爪和 backend 命令。
- 构造函数会创建 backend、连接设备并可能移动机器人；对象无法无硬件实例化。
- 直接使用宽 `Config`，mapping/safety 无法只接收所需不可变参数。
- 与 `data_schema.py` 和 visualizer payload 耦合，使控制域知道记录/UI 需求。

建议：纯 `Mapper`、`Transform`、`SafetyFilter` 按 typed input/output 串联；executor 是唯一硬件 command owner。

### 5.5 tactile readers

- `tactile.py` 是旧 41-taxel 串口实现；构造即打开串口、创建 publisher、阻塞收集 320 个基线包并重配 loguru。
- `tactile_4point.py` 是 4-taxel BLE 实现，但也包含 GUI 子进程、DDS 发布、滤波、重连、对外 holder mutation 和全局 logging/`atexit` 操作。
- 两者返回形状和生命周期不同，没有共同 `TactileSensor` contract；`Config.TACTILE_READER` 用字符串选择实现。

建议：共同接口只暴露 `start/stop/latest_sample`（或 iterator）；BLE acquisition、calibration/filter pipeline、publisher 与 UI panel 分开；旧串口实现进入 `deprecated/`。

### 5.6 recording 与 visualization

- 录制层直接认识相机 key、深度、force/torque/tactile 和具体 schema；`main` 负责把设备内部结构拼成 recorder 输入。
- rollback 不只是更新应用状态，还直接改 LeRobot metadata、Parquet、视频和 episode index。
- visualizer 的模型/发布逻辑散在 `main.py`、`realman_teleop.py`、`robot_teleop.py` 和 `test_tool/ForceVisualize.py`。
- visualization consumer 还承载 command 回流，因此不是纯只读 consumer。

建议：runtime 产生统一 immutable `TeleopSnapshot`；recorder 和 visualizer 各自订阅。UI command 走独立 `RuntimeCommand` queue，不通过 sample/manager 字段反写。

## 6. 硬件层对运行时/界面的反向依赖

- `realman_teleop.py` 把 RealMan SDK adapter、executor 和整个 app 放在一起，并直接创建 camera/visualizer。
- `robot_backend.py` 同时包含 domain interface、具体 UR/RealMan adapter、Robotiq socket client 和 factories；基础 abstraction 依赖具体实现所在文件。
- `tactile_4point.py` 内置 Matplotlib panel 和外部 holder mutation。
- `camera_udp.py` / `WebRTC_udp.py` 的设备 acquisition 直接知道 VR、record control 和 tactile。
- `robot_teleop.py` 的控制域直接知道 visualizer config 和 data schema。

目标应反转为：runtime/app 依赖 domain ports，外层 factory 注入硬件 adapter；硬件模块绝不 import runtime session、visualization 或 recording。

## 7. 外部共享状态的修改者

| 对象/函数 | 修改的外部或无关状态 | 后果 |
|---|---|---|
| `CameraUDPManager` / `WebRTCUDPManager` RX/TX threads | manager 上的 camera/depth、controller/hand、record、rollback、zoom、tactile 字段。 | 多个 consumer 依赖锁和字段时序；无 schema/version。 |
| `main.py` loops | 反向清除 manager flags、更新 tactile payload、控制 pause/stop、操作同一 recorder。 | orchestrator 变为共享状态协调器。 |
| `MagtouchIliasSerialReader.run` | 读 `cu.controller_data`，写 `cu.tactile_data` 等。 | 设备 reader 依赖整个上层对象。 |
| `FourPointTactileBleReader.data_callback` | 写入传入 control/holder 的 array、bytes、timestamp；还发布 DDS/GUI。 | acquisition 与多个 consumer 强耦合。 |
| visualizer handle / command path | 从 UI 触发 reset、rollback、recalibrate。 | consumer 与生产/控制路径形成反向边。 |
| `DatasetRecorder.rollback` | 修改文件、LeRobot metadata/Parquet/video 与内部 episode 索引。 | 操作非原子，崩溃可能留下跨文件不一致。 |
| `utils.py`、`tactile*.py`、`visualizer.py` | 全局 logging、loguru sinks、`MPLCONFIGDIR`、Matplotlib backend、短暂替换 `atexit.register`。 | 单纯 import/构造可污染宿主进程和测试。 |
| `test_tool/ForceMode.py`、`freedrive.py`、dataset replay/tag scripts | import-time 机器人/网络/远端状态。 | 测试发现或文档工具可能意外产生实体副作用。 |

## 8. 全局配置依赖

- `Config` 是宽、可变 dataclass；所有模块拿到整个对象，而非自己的窄配置。
- `Config.__post_init__` 做类型归一化、机器人选择、数组派生和延迟 import，不是纯 values-only 模型。
- `main.py` 在 import time 创建全局实例；测试 monkeypatch/实例间变更容易泄漏。
- IP、端口、硬件名和 Linux 串口均硬编码默认值；诊断脚本还有自己的固定 UR IP。
- `pyrightconfig.json` 硬编码 `/home/airo/...` Conda 环境。
- README 默认值与源码已漂移，例如 JPEG quality、重力补偿。

建议先于 runtime composition 引入 typed immutable 配置：模型和 loader 不 import hardware；outer factory 根据 profile 创建 adapter。原计划 Phase 13 应提前到 package/core contract 之后、设备拆分之前。

## 9. 过早导入的可选依赖

| 依赖 | 当前触发路径 | 问题 |
|---|---|---|
| `aiortc`、`aiohttp`、`av` | `main.py` 顶层导入 `WebRTC_udp.py`，即使选择 UDP。 | `aiohttp` 未列入 `requirements.txt`；无 WebRTC 环境不能导入 app。 |
| `pyrealsense2`、OpenCV | 两个 camera manager 顶层。 | 无相机的协议/控制测试也可能导入失败。 |
| UR/RealMan SDK、`rtde_*`、`airo_robots`、`ur_analytic_ik`、`ruckig` | backend/teleop 模块顶层或构造路径。 | mock/unit test 无法只安装核心包；`GRIPPER=False` 时 UR factory 仍可能创建 Robotiq。 |
| `sensor_comm_dds` 与 BLE MagTouch 包 | tactile modules 顶层；requirements 中却仅注释。 | 默认 `TACTILE_ENABLE=True`，正常入口可能在未安装 internal dependency 时失败。 |
| `h5py`、LeRobot、Hugging Face、PyArrow/video | `dataset.py` 与工具。 | 不记录数据的 teleop 仍可能承受重依赖；工具依赖也未完整声明。 |
| `torch`、LeRobot policy APIs | `inference.py`、ForceFlow++。 | 控制/配置模块与 ML 环境不能独立安装；LeRobot 新旧 API 路径混用。 |
| Matplotlib、Rerun、Pandas、Pillow、IPython、TQDM | visualizer/工具。 | 多个未列依赖；headless 与最小运行环境不稳定。 |

要求：按 `video`、`camera-realsense`、`robot-ur`、`robot-realman`、`tactile-ble4`、`tactile-legacy`、`recording`、`policy`、`visualization`、`dev` extras 分组；adapter 模块可延迟 import 并在 factory 给出可操作错误。

## 10. 推荐依赖方向

```text
apps / CLI
    ↓
runtime sessions + lifecycle + command router
    ↓
domain services (teleop mappings, safety, recording state, policies)
    ↓
domain-owned ports + immutable core types/events/buffers/clocks/errors
    ↑
hardware adapters / transports / serializers / UI consumers
    ↑
third-party SDKs and OS/network/filesystem
```

约束：

1. `core` 只含跨域值类型、错误、clock、buffer、event；不得 import NumPy 之外的重硬件/ML SDK。
2. 端口归属其领域：`robots/base.py`、`devices/cameras/base.py`、`devices/tactile/base.py`、`streaming/video/base.py`。不要同时在 `core/interfaces.py` 再定义一套重复协议。
3. protocol 只做 bytes/typed message 转换；transport 只收发，不解析业务字段。
4. producer 只发布 immutable sample；recorder、visualizer、streamer 是独立 consumer。
5. executor 独占 SDK command handle 和控制 cadence；backend 提供原子能力，runtime 不从多个线程直接调用 SDK。
6. config loader 产生 values；factory 位于 composition boundary，按需 import adapter。
7. 所有 background thread/process/task 注册到同一 lifecycle owner；测试必须断言退出后没有残留 worker。

## 11. 拆分前必备依赖测试

- import smoke matrix：core-only、UDP-only、WebRTC、UR、RealMan、BLE tactile、policy、visualization extras 分别安装/缺失时的行为。
- 旧 import/CLI compatibility：根模块 shim 与新 package 返回同一公开符号/退出码。
- UDP/WebRTC 共用 VR protocol golden vectors，确保拆分 transport 后字节/字段不变。
- fake camera/VR/robot/tactile/recorder/visualizer integration，验证数据只单向流动。
- lifecycle 测试：正常退出、初始化半途失败、超时、peer disconnect、BLE reconnect、robot callback in-flight。
- recorder 与 visualizer 同时订阅时互不修改 sample；command 只能通过 command router 回流。
- 配置快照测试：旧 `Config` 默认值/校验与新 YAML profile 一一对应。

## 12. 未决风险

- H.264 预编码帧能否以目标时延稳定接入 aiortc，需要独立 feasibility spike，不能在 transport 重写时顺便假设。
- RealMan >100 Hz / 10 ms safety gate、UR servo cadence、watchdog hold/recovery 需要明确数值和实机验收。
- `HB` 二进制 v2 尚缺完整的 endian、version、flags、sequence wrap、timestamp clock 和 payload-length 规范。
- LeRobot 依赖的具体版本及私有 rollback 文件布局未锁定；升级前要保存真实 fixture。
