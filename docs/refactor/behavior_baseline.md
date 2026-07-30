# 当前行为基线（Phase 0 / Task 0.3）

## 0. 范围、证据与判定口径

本文件是只读审计结果，不是设计说明，也不表示现有行为正确。审计对象为
`G:\Projects\AIRO-Doffy`，源码提交为
`56dcaa23b0ef518338875b8a7fceb5f70effbe53`；计划文件
`AIRO_Doffy_v2.0_PLAN_TASK.md` 在该工作树中未跟踪，但已完整阅读其中
Phase 0 Task 0.3、§11 兼容性约束、§12 性能目标、§15 Codex 执行规则。
取证日期为 2026-07-30。

本文使用三种明确标签：

- **源码事实**：可由当前 Python 源码或文件布局直接证明；行号以审计快照为准。
- **README 差异**：README 的命令、默认值或协议描述与当前源码不一致。
- **尚需实机验证**：静态审计无法证明，需要机器人、相机、VR、触觉设备或可运行的依赖环境。

Phase 0 的边界是“不修改源行为”。计划还规定 HDF5/LeRobot schema、Unity
消息、UDP 包、机器人安全限制、控制频率、初始关节、坐标映射、夹爪映射、
触觉形状、力/力矩约定、episode 编号与 rollback 语义均不得未经批准修改。
本任务没有修改实现或测试，也没有启动任何硬件、网络服务或数据写入流程。

## 1. 当前运行命令与错误路径

### 1.1 路径和参数层面存在的入口

| 命令 | 源码事实 | 首个主要副作用或失败边界 |
|---|---|---|
| `python main.py` | `main.py:373-542` 存在入口；默认选择 WebRTC。 | 构造 manager 时枚举 RealSense、创建相机对象，并绑定本机 UDP 8001/8003/8005；随后最多等待 VR UDP 数据 60 s，再连接并移动机器人到初始位、打开夹爪。默认 visualizer 开启，因此即使 `TACTILE_TRANSFER=False` 也会启动 BLE 触觉线程。 |
| `python realman_teleop.py` | `realman_teleop.py:1626-1823` 存在入口。 | README 已提示必须先把 `ROBOT_TYPE` 改为 `realman`，但源码默认是 `ur3e`；脚本先创建相机 manager 并等待 VR，之后 `RealManTeleop` 才在 `:654-661` 校验配置，所以默认配置可能先在 IP 绑定/相机/VR 处失败，而不是立即报机器人类型错误。`ROBOT_TYPE="realman"` 时还必须显式设置非空 `ROBOT_IP`。 |
| `python test_tool/ForceVisualize.py --mock` | 文件和 `--mock` 参数存在（`:820-905`）；这是仓库中最接近纯无硬件 UI 冒烟测试的命令。 | 仍需 Python、NumPy、OpenCV、matplotlib 等软件依赖和可用 GUI/后端。 |
| README 中其余 `ForceVisualize.py` 命令 | 路径与参数均存在。 | 非 `--mock` 模式连接 UR；`--tactile` 启动真实触觉；`--tcp-xyz-experiment` 会发送机器人运动指令，不能作为无硬件测试。 |
| `python test_tool/vr_data.py` | 实际 VR 接收器在此路径；没有 CLI 参数，自动识别 controller/hand 并显示或打印（`:499-513`）。 | 构造时绑定 UDP，等待 VR 数据；需要网络接口与 Unity 发送端。 |
| `python inference.py --policy … [--device … --fps 10]` | 入口及这些参数存在（`inference.py:741-777`）。 | 在加载策略前即枚举相机、连接机器人、移动到初始位并启动 stdin 监听线程；不是离线命令。策略可能触发 Hugging Face 网络访问。 |
| `python test_tool/eval_with_datasets.py --policy … --dataset … --episodes … --no-show` | 这是当前实际离线评估入口，README 示例参数均可映射到该脚本（`:475-507`）。 | 仍需 LeRobot、Torch、AV、数据集与策略；省略本地资源时可能访问 Hugging Face。 |
| `python -m unittest discover -s tests -v` | 当前唯一自动测试入口；详见 §8。 | 本次审计环境找不到 `python`、`py` 或 `pytest` 可执行文件，因此未执行。 |

“存在”只表示文件和参数可由源码证明，不表示已在本机或硬件上运行通过。

### 1.2 README 声称但当前不能按原文运行的命令

| README 命令/说法 | README 差异与确定错误路径 |
|---|---|
| `python vr_data.py` | 根目录没有 `vr_data.py`，Python 会报“can't open file”；实际文件为 `test_tool/vr_data.py`。 |
| `python vr_data.py --visualize` | 除根路径不存在外，实际 `test_tool/vr_data.py` 没有 argparse，也不接受 `--visualize`；它自动切换 controller 打印与 hand 可视化。 |
| `python eval_policy.py …` | 根目录没有 `eval_policy.py`，Python 会报“can't open file”；实际相近入口为 `test_tool/eval_with_datasets.py`。 |
| “安装 `requirements.txt` 即可运行” | `requirements.txt` 未列直接导入的 `aiohttp`，并注释掉默认 BLE 路径会导入的 `sensor-comm-dds`；README 的分步安装文字补充了 `aiohttp`，但默认 `main.py` 仍会因 visualizer 开启而尝试触觉。 |
| WebRTC 为“可选依赖” | `main.py` 顶层同时导入 `camera_udp` 和 `WebRTC_udp`，因此即使配置选择 UDP，导入 `main.py` 仍需要 aiortc/aiohttp/av；反向也需要 UDP 模块的依赖。 |

### 1.3 本次执行证据

本次尝试以 `PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v`
运行无硬件测试，PowerShell 在启动测试前即返回
`CommandNotFoundException: python`。随后检查 `py`、`python`、`python3`、
`pytest` 均无结果。因此：

- 没有测试被执行，也没有“通过”结论；
- 没有通过导入源码来读取默认值，以避免硬件/套接字副作用；
- 以下结论来自逐行静态取证，实机项目见 §9。

## 2. `Config` 与 `VisualizerConfig` 当前默认值

### 2.1 机器人、网络和控制

| 字段 | 源码默认值 |
|---|---|
| `ROBOT_TYPE` | `"ur3e"` |
| `ROBOT_IP` | `None`；`__post_init__` 对 UR 自动改为 `UR_IP`，对 RealMan 仍为 `None` 并抛错 |
| `UR_IP` | `"10.42.0.162"` |
| `REALMAN_PORT` | `8080` |
| `REALMAN_READ_RETRIES` / `REALMAN_RETRY_DELAY` | `3` / `0.05 s` |
| `REALMAN_CTRL_RATE` / `REALMAN_MIN_CANFD_RATE` | `200 Hz` / `100.0 Hz`（目标必须严格大于最小值） |
| `REALMAN_RATE_CHECK_WINDOW` / `REALMAN_RATE_FAILURE_WINDOWS` | `1.0 s` / `3` |
| `REALMAN_CANFD_HEARTBEAT_TIMEOUT` | `0.05 s` |
| `REALMAN_SENSOR_RATE` / `REALMAN_VR_TIMEOUT` | `30.0 Hz` / `0.25 s` |
| `REALMAN_MAX_JOINT_SPEED` | `2.0 rad/s` |
| `REALMAN_MAX_LINEAR_SPEED` / `REALMAN_MAX_ANGULAR_SPEED` | `0.25 m/s` / `1.0 rad/s` |
| `REALMAN_CANFD_TRAJECTORY_MODE` / `REALMAN_CANFD_RADIO` | `0` / `0` |
| `REALMAN_REALTIME_STATE_PUSH` | `True` |
| `REALMAN_STATE_PUSH_CYCLE_MS` / `REALMAN_STATE_PUSH_PORT` | `5 ms` / `8098` |
| `REALMAN_STATE_PUSH_TIMEOUT` / `REALMAN_FORCE_COORDINATE` | `2.0 s` / `0`（sensor frame） |
| `TELEOP_COMMAND_MODE` / `FREEZE_ROTATION` | `"joint"` / `True` |
| `PC_IP` / `VR_IP` | `"10.10.131.162"` / `"10.10.130.155"` |
| `IP_PORT` / `POSE_PORT` / `CONTROL_PORT` | `8000` / `8001` / `8005` |
| `SIGNALING_PORT` | `8765` |
| `UR_CTRL_RATE` / `KELO_CTRL_RATE` | `60 Hz` / `10 Hz`（KELO 当前入口未见使用） |

### 2.2 任务、跟踪、相机、视频与采集

| 字段 | 源码默认值 |
|---|---|
| `TASK_NAME` | `"pick_and_place"` |
| `DATASET_DIR` | `"./datasets/pnp_long"`；recorder 追加 `_lero` 或 `_hdf5` |
| `DATASET_TYPE` / `PUSH_TO_HUB` | `"l"` / `False` |
| `SAVE_EEF` / `DATA_TYPE` | `False` / `"both"` |
| `DEPTH_INFO_ENABLE` | `False` |
| `TRACKING_MODE` | `"controller"` |
| `CONTROLLER_RESET_TRIGGER_THRESHOLD` | `0.8` |
| `HAND_PALM_JUMP_THRESHOLD` | `0.15 m` |
| `HAND_GRIPPER_OPEN_DIST` / `HAND_GRIPPER_CLOSE_DIST` | `0.06 m` / `0.03 m` |
| `HAND_MODE_TOGGLE_DIST` / `HAND_RESET_DIST` | `0.02 m` / `0.02 m` |
| `REALSENSE_RESOLUTION` / `REALSENSE_FPS` | `(640, 480)` / `60` |
| `VIDEO_TRANSPORT` | `"webrtc"` |
| `JPEG_QUALITY` / `HD_CHUNK_SIZE` | `100` / `60000 B` |
| `COLLECT_RATE` | `10 Hz` |
| `INFERENCE_FPS` | `10 Hz`；字段在 dataclass 中重复声明两次，值相同 |
| `INFERENCE_MAX_STEPS` / `INFERENCE_EPISODES` | `1000` / `1` |

### 2.3 夹爪、初始姿态、安全与滤波

| 字段 | 源码默认值 |
|---|---|
| `GRIPPER` / `GRIPPER_SPEED` / `GRIPPER_MAX` | `False` / `0.1 m/s` / `0.085 m` |
| `VR_TO_ROBOT_AXES` | 声明为 `None`；UR 派生为 `[[-1,0,0],[0,0,-1],[0,1,0]]`，RealMan 派生为 `[[0,0,1],[-1,0,0],[0,1,0]]` |
| `INITIAL_JOINT` | 声明为 `None`；UR 派生为 `[1.57,-1.57,1.57,-1.57,-1.57,0]`；RealMan 派生为 `[2.65586749,-0.06628761,-0.14056882,-1.26216978,0.11116002,-1.11919238,-0.45881216]` |
| `TCP_POSE` / `TCP_TRANSFORM` | 六个 0 / `4×4` 单位阵 |
| `MOVE_THRESHOLD` | `[0.9,0.9,0.9,0.9,1.4,1.4]` |
| `RUCKIG_ENABLE` | `True` |
| `RUCKIG_MAX_VEL` | `[2.5,2.5,2.5,3.0,4.0,4.0] rad/s` |
| `RUCKIG_MAX_ACC` | `[15,15,15,18,25,25] rad/s²` |
| `RUCKIG_MAX_JERK` | `[150,150,150,180,250,250] rad/s³` |
| `CARTESIAN_POS_FILTER_CUTOFF_HZ` / `CARTESIAN_ROT_FILTER_CUTOFF_HZ` | `8.0` / `6.0` |
| `HAND_JOINT_FILTER_CUTOFF_HZ` | `10.0` |

### 2.4 力/力矩与触觉

| 字段 | 源码默认值 |
|---|---|
| `TORQUE_MODE` / `FORCE_COLLECT` / `TORQUE_COLLECT` | `False` / `False` / `False` |
| `GRAVITY_COMP` | `False` |
| `TOOL_MASS` / `TOOL_COM` | `0.925 kg` / `[0,0,0.058] m` |
| `GRAVITY_COMP_FILTER_ALPHA` / `GRAVITY_CALIB_SAMPLES` | `0.15` / `200` |
| `FORCE_MOVING_AVERAGE_WINDOW` / `FORCE_LOW_PASS_ALPHA` | `8` / `0.15` |
| `TACTILE_ENABLE` / `TACTILE_TRANSFER` | `True` / `False` |
| `TACTILE_PORT` | `8012` |
| `TACTILE_READER` / `TACTILE_SHAPE` | `"ble4"` / `(4,3)` |
| `TACTILE_SERIAL_COM` | `"/dev/ttyACM0"` |
| `TACTILE_BLE_DEVICE_MAC` / `TACTILE_BLE_HCI` | `"ARDUINO7"` / `"hci0"` |
| `TACTILE_BLE_WINDOW_SIZE` | `100` calibration samples |
| `TACTILE_FILTER_ALPHA` / `TACTILE_USE_KALMAN` | `0.75` / `False` |
| `TACTILE_KALMAN_Q` / `TACTILE_KALMAN_R` | `0.02` / `0.02` |
| `TACTILE_MAX_DELTA` | `10000.0` |
| `TACTILE_BASELINE_DRIFT_ALPHA` / `TACTILE_BASELINE_DRIFT_THRESHOLD` | `0.0` / `80.0` |

### 2.5 `VisualizerConfig`

| 字段 | 源码默认值 |
|---|---|
| `ENABLED` | `True` |
| `HZ` | `30.0` |
| `WINDOW_S` | `8.0` |
| `FORCE_PANEL_RANGE` | `30.0 N`（±范围） |

### 2.6 README 默认值漂移

**README 差异：**

- README 写 `PC_IP=10.10.131.72`，源码是 `10.10.131.162`。
- README 写 `VR_IP=10.10.131.166`，源码是 `10.10.130.155`（连子网段也不同）。
- README 写 `JPEG_QUALITY=50`，源码是 `100`。
- README 写 `GRAVITY_COMP=True`，源码是 `False`。
- README 没有完整列出 `VIDEO_TRANSPORT="webrtc"`、UDP 基础端口、控制频率、
  初始关节、安全/插值参数以及大量触觉可靠性参数。

## 3. UDP、VR 文本/二进制与 WebRTC

### 3.1 UDP 端口分配

**源码事实（`camera_udp.py:96-168`、`WebRTC_udp.py:181-264`）：**

- 每个 `UdpComms` 都用同一个 UDP socket：发送到 `VR_IP:port_tx`，并且无论
  `enable_rx` 是否为真都绑定 `PC_IP:port_rx`；只有 `enable_rx=True` 才启动
  后台接收线程。
- legacy JPEG manager 至少创建三组 socket；每台相机最多五台，各占一组：

| 名称 | PC 绑定 RX | VR TX | 含义 |
|---|---:|---:|---|
| `socket_0` | 8001 | 8000 | VR pose RX；UDP 模式也发送 camera 0 JPEG |
| `socket_1` | 8003 | 8002 | 录制控制 `Start`/`Stop`/rollback；UDP 模式也发送 camera 1 |
| `socket_2` | 8005 | 8004 | legacy resolution/zoom/fine-mode；UDP 模式也发送 camera 2 |
| `socket_3` / `socket_4`（有相机时） | 8007 / 8009 | 8006 / 8008 | camera 3/4 JPEG；不开接收线程但仍绑定 RX |
| `socket_tactile`（仅 transfer 开） | 8013 | 8012 | PC 向 VR 发送触觉原始数组字节 |

- `POSE_PORT` 和 `CONTROL_PORT` 只是配置字段，manager 不直接引用它们；当前
  自增算法恰好得到 8001 和 8005。改动 `IP_PORT` 而不联动这两个字段时，后者
  不会影响实际绑定。
- WebRTC manager 不使用这些 socket 发送视频，但仍保留 0/1/2 三组 UDP socket：
  pose、record、legacy control；触觉仍是 UDP。
- 接收 `recvfrom(4096)` 后严格按 UTF-8 解码；队列为 `deque(maxlen=128)`，
  满时自动丢最旧数据。pose 每轮 `read_all()`，record/control 每轮只 `read()`
  一条；轮询目标 100 Hz。
- record 文本精确匹配：`Start` 开始、`Stop` 停止并请求导出、
  `Undo|Rollback|DeleteLast` 请求删除/丢弃最后 episode。

### 3.2 JPEG UDP 分片格式

**源码事实（`camera_udp.py:29-31,172-221`）：**

```text
网络字节序 / big-endian，struct "!IHHI"，固定 12 字节

offset  size  field
0       4     frame_id        uint32
4       2     chunk_index     uint16，0 起
6       2     total_chunks    uint16
8       4     total_bytes     uint32，整张 JPEG 字节数
12      N     JPEG payload
```

- 每个相机有独立计数器，从 0 开始；发送字段取计数器低 32 位。
- 默认 OpenCV JPEG quality 为 **100**；默认 payload 为 60000 B，所以最大
  UDP datagram 为 60012 B（另有 UDP/IP header），会依赖 IP 分片。它虽低于
  UDP payload 上限，却直接违反计划 §12 “Avoid packet fragmentation” 的目标。
- `total_chunks=ceil(total_bytes/chunk_size)`；没有对 uint16 溢出、空 JPEG、
  丢包、乱序、重复、重传、checksum 或超时进行发送侧协议处理。
- 这是兼容基线，不应在 characterization test 前“顺手修正”。

### 3.3 controller 文本格式（旧/新）

**源码事实（`parse_vr.py:3-7,47-123`）：**

```text
旧：<timestamp_ms>,<left 14 fields>,<right 14 fields>              # 29 fields
新：C,<frame_id>,<timestamp_ns>,<left 14 fields>,<right 14 fields> # 31 fields
```

每手 14 项顺序：

```text
px, py, pz,
rx, ry, rz, rw,
joystick_x, joystick_y,
index_trigger, grip_trigger,
button_AX, button_BY, joystick_press
```

解析结果固定为 `[LTouch, RTouch]` 两个 dict；新格式保留 `FrameId`，旧格式
令它为 0。字段数必须精确相等；header 和数值不可转换时返回 `None`。按钮先
按 float 读取再转 int。包类型只把 `C,` 或首字符为十进制数字的文本识别为
controller，因此负号开头的旧 timestamp 不会被识别。

### 3.4 hand 文本格式

**源码事实（`parse_vr.py:9-11,145-182`）：**

```text
H,<side>,<frame_id>,<timestamp_ns>,
<wrist_px>,<wrist_py>,<wrist_pz>,
<wrist_qx>,<wrist_qy>,<wrist_qz>,<wrist_qw>,
<joint_0_xyz> ... <joint_25_xyz>
```

固定 89 个 CSV 字段、26 个 OpenXR joint。返回 `side`、`frame_id`、
`timestamp`、含 position/rotation 的 `wrist_pose` 和 26×3 `bones`。源码不
校验 `side` 必须为 `L`/`R`。

### 3.5 `HB,` base64 二进制格式

**源码事实（`parse_vr.py:185-231`）：**

外层仍是 UTF-8 文本 `HB,<base64>`。base64 解码后的二进制是：

```text
offset  size   type/endian       field
0       1      byte              0x48 ('H')
1       1      byte              side 的 ASCII
2       2      uint16 little     joint_count，必须为 26
4       4      uint32 little     frame_id
8       312    26 × 3 × float32 little，依次 x/y/z
```

最小 raw 长度 320 B；允许尾随字节。二进制中没有 wrist quaternion，也没有
发送端 timestamp；解析器令 `wrist_pose=None`，timestamp 用接收时本机
`time.time()*1000` 生成。下游在 wrist 缺失时把 `bones[0]` 当 palm/wrist
位置并使用单位旋转。

**README 差异：** README 描述的是 24 bones，且给出的 header 只有
`[0x48, side, count_lo, count_hi]`；当前源码实际是 **26 joints、8 字节
header、额外 little-endian frame_id**。README 文本 hand 格式同样遗漏
frame_id、wrist 7 项并写成 24 bones。这是 Unity 兼容迁移前必须先测试的高风险
差异。

### 3.6 触觉 UDP bytes

- BLE4 路径发送 `(4,3)` 的 native-endian contiguous `float32.tobytes()`，
  共 48 B，无 header、版本、shape、timestamp 或显式 endian。
- serial 路径当前输出 41 taxels（32+3+3+3）×3 的 native-endian
  `int32.tobytes()`，共 492 B；这与配置 `TACTILE_SHAPE=(4,3)` 不一致。
- manager 以 100 Hz 重发最新 `tactile_byte`。因此接收方只能依赖预先知道的
  reader 类型解释 dtype/shape；该 wire format 也必须纳入兼容测试。

### 3.7 WebRTC signaling、视频轨与 DataChannel

**源码事实（`WebRTC_udp.py:337-533`）：**

- aiohttp WebSocket 只注册路径 `/`，绑定 `PC_IP:SIGNALING_PORT`，默认
  `ws://10.10.131.162:8765/`。
- 收发 envelope 为 `{"type": ..., "session_id": ..., "payload": {...}}`。
- `hello` 保存 session 到内存 map 并回复 `hello_ack`；`start_video` 只记录
  日志；`offer` 的 `payload.sdp` 创建新 peer；`ice_candidate` 解析并加入；
  `stop_video` 关闭 peer。没有鉴权、schema 校验或 session 隔离验证。
- 每个 offer 会先关闭旧 peer，再为当时每个相机加入一个
  `RealsenseCameraTrack`，并由 Python 端创建标签为 `"control"` 的
  DataChannel，随后 set remote offer、create/set local answer，回：
  `{"type":"answer","session_id":sid,"payload":{"sdp":local_sdp}}`。
- 本地 ICE candidate 以 `candidate/sdpMid/sdpMLineIndex` 回传；远端 candidate
  接受带或不带 `candidate:` 前缀。
- 没有 `setCodecPreferences`、bitrate、H.264 profile、NVENC、B-frame、
  lookahead 或 GOP 设置。实际 codec 由 SDP/aiortc 协商；README 所说
  “H.264/VP8”不能被读成“代码强制 H.264”。
- `"control"` message 直接传 `_parse_resolution_control`。协议为分号分隔的
  `key,value`：数字 key 在 WebRTC 路径直接作为 camera index，`value` 会无条件
  丢掉首字符后转 float（预期类似 `x1.5`）；非数字 key 把 `fine_mode` 设为
  value。UDP legacy manager 的数字 key 则按端口号
  `(int(key) % IP_PORT) // 2` 映射。WebRTC 保留的 UDP control 却调用
  WebRTC 版“直接 index”解析器，兼容性需要 Unity 抓包确认。
- DataChannel 只承载 resolution/zoom/fine-mode；高频 pose、record 和 tactile
  仍走 UDP，当前并没有计划目标所述的二进制 state DataChannel。
- `main.py` 在 `test_connection()` 收到 UDP pose 后才
  `start_comms_threads()`，而 signaling server 也在后者中启动；因此启动握手
  顺序要求 Unity 先送 pose，WebRTC signaling 才出现。

## 4. 数据集 schema 与 episode 语义

### 4.1 state/action 表示

`data_schema.py:25-101` 规定：

| `DATA_TYPE`（规范化后） | state | action | `extra.tcp_pose` |
|---|---|---|---|
| `qpos`（含 `joint`/`joint_configuration` alias） | `dof+1`：joint + normalized gripper | 同 state | 无 |
| `both`（默认） | `dof+1`：joint + normalized gripper | 同 state | 7：`qx,qy,qz,qw,x,y,z` |
| `tcp`（含 `tcp_quat`/`eef` alias） | 8：quat + xyz + gripper | 同 state | 无 |
| `delta_tcp` | `dof+1`：joint + gripper | 7：`dx,dy,dz,drotvec_x/y/z,gripper` | 7：当前 TCP quat + xyz |

默认 UR3e 为 6 DoF，因此默认 `both` 的 state/action 均为 7 维。

### 4.2 HDF5（ACT）schema

输出目录为 `DATASET_DIR + "_hdf5"`，文件名严格为
`episode_<zero-based-index>.hdf5`。根属性 `sim=False`。

| path | shape | dtype/条件 |
|---|---|---|
| `/observations/qpos` | `(T,state_dim)` | 创建时未显式 dtype；当前 h5py 默认浮点类型需测试固化 |
| `/action` | `(T,action_dim)` | 同上 |
| `/extra/timestamps_ns` | `(T,5+camera_num)` | `int64`；attr `names` 为字节串数组：`collect,robot_state,robot_action,vr_input,tactile,camera_0...` |
| `/extra/tcp_pose` | `(T,7)` | 仅 `both`/`delta_tcp` |
| `/observations/force` | `(T,3)` | 仅 `FORCE_COLLECT` 且 backend 支持 |
| `/observations/torque` | `(T,3)` | 仅 `TORQUE_COLLECT` 且 backend 支持 |
| `/observations/tactile` | `(T,*TACTILE_SHAPE)` | 仅 `TACTILE_TRANSFER`；默认 `(T,4,3)` |
| `/observations/images/camera_i` | `(T,480,640,3)`（默认） | `uint8`，chunk `(1,480,640,3)`；只为导出时 `cu_manager.camera_images` 里的 key 建立 |
| `/observations/depth/camera_i` | `(T,480,640)`（默认） | `float32`，仅 depth 开启 |

同目录追加 `episode_descriptions.txt`，行为为
`Episode N: max_timesteps = T`。HDF5 在内存 buffer 收集，`Stop` 后写整集。

### 4.3 LeRobot 显式 feature schema

输出目录为 `DATASET_DIR + "_lero"`；默认 `fps=10`、
`robot_type="ur3e"`、`use_videos=True`、`image_writer_processes=0`。

| feature | dtype / shape / names | 条件 |
|---|---|---|
| `action` | `float32 (action_dim,)`，names 由 §4.1 决定 | 总是 |
| `observation.state` | `float32 (state_dim,)` | 总是 |
| `extra.timestamps_ns` | `int64 (5+camera_num,)`，names 同 HDF5 | 总是 |
| `extra.tcp_pose` | `float32 (7,)`，quat + xyz | `both`/`delta_tcp` |
| `observation.force` | `float32 (3,)`，`Fx,Fy,Fz` | 可选 |
| `observation.torque` | `float32 (3,)`，`Tx,Ty,Tz` | 可选 |
| `observation.images.camera_i` | `video (480,640,3)`，H/W/C | 每台相机 |
| `observation.depth.camera_i` | `image (480,640,1)` | depth 可选；frame 写入为毫米 `uint16` |
| `observation.tactile` | `float32 (4,3)`，`sensor_idx,axis` | transfer 可选 |

每帧还传入 `"task": TASK_NAME`。`episode_index/frame_index/timestamp/next.done/
task_index` 等 LeRobot 标准列并非本仓库显式 feature，而由安装的 LeRobot
版本在 `add_frame/save_episode` 中注入；其精确 schema 必须用受支持版本生成
golden dataset 后固定，不能仅凭当前代码猜测。

### 4.4 数据集启动、续录与 rollback

- HDF5 通过扫描现有 `episode_*.hdf5` 的最大编号加一续录；编号从 0 开始。
- LeRobot 先加载现有 dataset 并比较以 `action`、`observation.`、`extra.` 开头
  的 key。**若不完全相等，源码会 `shutil.rmtree(root)` 删除整个现有数据集并
  重建**（`dataset.py:191-224`）。即使其他异常进入创建分支，只要 root 存在也
  会递归删除。这是高风险启动副作用。
- LeRobot 每采样周期立即 `add_frame`，Stop 时 `save_episode()` 后
  `finalize()`；HDF5 到 Stop 才落盘。
- rollback 优先丢弃当前未保存 buffer；否则删除最后一个 zero-based episode
  并复用编号。HDF5 删除对应文件和 description 行。LeRobot 会重写/删除
  parquet、视频、episode metadata，更新 `info.json` 的 totals/split，最后一集
  删除后删 `tasks.parquet`，并总是删 `stats.json` 以避免陈旧统计。
- `DatasetRecorder.close()` 会 finalize LeRobot；`PUSH_TO_HUB=True` 时还会在
  shutdown 推送远端。

## 5. 频率、相机、力与触觉

### 5.1 实际调度频率

| 路径 | 源码事实 |
|---|---|
| UR 通用 teleop | `main.py` 目标 60 Hz；循环持有最近一次 controller/hand 状态并重复 `teleop.step`。GripTrigger active 时通常每轮发一次 joint/TCP 命令；没有对普通 UR VR 输入的 0.25 s stale watchdog。 |
| RealMan 专用 | 独立 CAN-FD owner thread 目标 200 Hz，连续重发最新安全 setpoint；启动需一个 1 s 窗口测得严格 `>100 Hz`。启动验证后任意 command gap 或 SDK call `>10 ms` 立即停止；验证前最多 3 个失败窗口。heartbeat 50 ms。 |
| RealMan state | 默认 controller UDP push 每 5 ms（200 Hz 配置请求）；首包超时 2 s。关闭 push 时在同一个 SDK owner thread 以 30 Hz 同步维护读取，可能占用 CAN-FD 时序预算。 |
| VR receive | UDP queue 轮询目标 100 Hz；queue 深度 128，不是 latest-only。 |
| JPEG/WebRTC camera acquisition | RealSense 对象请求 640×480@60，但 Python acquisition loop 额外 sleep `1/30`，故应用层最多约 30 Hz，再叠加阻塞读取时间。 |
| JPEG send | 每相机目标 30 Hz；每轮重新 JPEG 编码并同步发送全部 chunks。 |
| WebRTC track | aiortc `next_timestamp()`；源码定义 30 Hz interval 但该成员未参与 sleep。track 每次读取 shared dict 中最新帧，无显式有界 queue。 |
| tactile bridge / UDP TX | 各 100 Hz；BLE callback 的真实采样率由设备决定。 |
| dataset collection | 10 Hz，独立线程，过期 tick 不追赶。 |
| visualizer publish | 30 Hz。 |
| inference | CLI `--fps` 默认取 `COLLECT_RATE=10`，不是重复字段 `INFERENCE_FPS`；每步 `execute_action(...,dt=1/fps)` 后按 wall clock 补 sleep。 |
| export loop | 2 Hz（`wait(0.5)`）。 |

这些都是目标/调度行为，不是已测得的稳定吞吐。RealMan 测速已有运行时
instrumentation；其余路径大多没有端到端 latency、jitter、CPU/GPU 或丢包测量。

### 5.2 相机参数与数据流

- 最多 5 台 RealSense，按枚举顺序和 serial 创建。
- 请求参数：RGB `640×480@60`；depth 默认关；pointcloud 关；manager depth
  打开时 hole filling 开。
- manager 把 SDK RGB 视为 RGB；UDP 发送前转 BGR、中心 zoom，再 JPEG。
  dataset 保留原 RGB。Inference 直接取 SDK RGB，不经 UDP/WebRTC。
- zoom 初值每相机 1.0；大于 1 时 resize 后中心裁切，小于等于 1 时 resize 回
  原尺寸（不是 letterbox）。
- WebRTC `jpeg_quality` 仅被读入对象，track 实际不做 JPEG quality conversion；
  docstring 的该说法与实现不符。
- shared `camera_data`/`camera_images` 是按 key 覆盖的 latest value，避免相机帧
  queue 累积；但同一帧可能被 WebRTC track 重复取用，未按 capture sequence 去重。

### 5.3 wrench 形状与滤波

- backend wrench 统一按前 6 项 `[Fx,Fy,Fz,Tx,Ty,Tz]`。
- `WrenchFilter` 当前配置：先窗口 8 的移动平均，再
  `filtered = 0.15*value + 0.85*previous`；deadband 构造参数未由通用 teleop
  传入，故为 0。首帧低通直接采用当前移动平均。
- `GRAVITY_COMP=False`，所以默认不做软件重力补偿。若开启，工具质量/质心见
  §2；采 200 点，每点 sleep 5 ms，并另有 1 s 等待。
- RealMan 专用入口使用 controller 的 `zero_force_data`，明确忽略本仓库额外
  gravity compensation，但仍应用相同 wrench moving-average/LPF。

### 5.4 tactile 形状与滤波

**默认 BLE4 路径：**

- 固定一个 2×2 sensor，输出 `(4,3)` `float32`。
- 启动用 100 点 raw calibration 的中位数作 baseline；MAD×1.4826，再乘
  `deadband_sigma=3`，与 `noise_floor=2` 取较大者作为逐通道 deadband。
- 顺序：减 baseline → 小于 deadband 置 0 → clip 到 ±20000 → 相对 last-good
  单步 clip ±10000 → EMA（alpha 0.75）→ 可选 scalar Kalman
  (`Q=R=0.02`，默认关) → `nan_to_num`。
- baseline drift 默认 alpha 0，故关闭；若开，仅当四 taxel 最大向量范数
  `<80` 时跟踪。
- reset 手势/显式请求重新采 baseline；断线时清空输出并重连。

**serial legacy 路径：**

- 构造函数立即打开 serial 且阻塞采 baseline；实际输出 `(41,3)` int32。
- reader 自身默认 Kalman 开 (`Q=0.001,R=0.01`)，但 `create_tactile_reader`
  没有把 Config 的 BLE Kalman 参数传给它。
- `DatasetRecorder` 在 transfer 开启时只接受 `TACTILE_SHAPE`，默认 `(4,3)`；
  因而 `TACTILE_READER="serial"` 与默认 dataset schema 不能直接兼容，采集时会
  `ValueError`。这是源码事实，具体历史意图尚需确认。

## 6. 启动副作用与 shutdown 行为

### 6.1 启动顺序和副作用

`main.py` 顶层就实例化 `Config`/`VisualizerConfig`，并导入 robot、dataset、
UDP、WebRTC 及其重依赖。进入 `main()` 后当前顺序是：

1. 构造 camera manager：枚举设备、创建 RealSense、创建并绑定 UDP sockets；
2. `test_connection()`：UDP 模式还会同步抓图/发 JPEG；等待 VR 30 s（UDP）
   或 60 s（WebRTC）；
3. 构造 `RobotTeleop`：连接 backend，阻塞移动到初始关节，打开/读取夹爪；
4. 默认 visualizer 开使 BLE tactile reader 启动（即使不传输/记录 tactile）；
5. 固定 sleep 5 s；
6. 初始化 dataset；LeRobot 可能续录，亦可能递归删除 schema 不匹配目录；
7. 才启动 camera/VR/signaling、visualizer、collect/export 和控制循环。

关键点：`main.py` 的 `try/finally` 只包住第 7 步之后的主循环。第 1–7 步中
任何构造、VR timeout、机器人连接、触觉或 dataset 异常都不会进入统一 cleanup，
因此已打开的 camera/socket/robot/thread 可能泄漏。

`inference.py` 在自己的 `try/finally` 之前完成相机枚举、机器人连接与初始运动、
stdin thread、可选 serial tactile、策略/processor 加载；这些步骤失败同样不会
执行最终 cleanup。

### 6.2 正常 `Ctrl+C` / close 路径

- `main.py`：设置 stop event；请求 tactile reader stop；关闭 camera manager；
  bounded join tactile/bridge/visualizer/collect/export；关闭 visualizer；
  `RobotTeleop.close()` 调 backend cleanup 和 GC。
- camera managers：`running=False`；join 自己登记的线程（1 s）；停 camera
  pipeline；关闭 sockets。WebRTC 还通知 async shutdown，最多等 10 s，必要时
  强制 stop 再等 2 s，然后关闭 peer 与 aiohttp runner。
- export thread 的 `finally` 调 dataset close；因此 LeRobot finalize，配置开启
  时可能 push hub。
- RealMan 专用入口的 setup 在 try 内，cleanup 覆盖面更完整。它先 stop、
  close camera、join workers；若 SDK worker 仍活着，故意不关闭 handle，而是
  quarantine 并最终抛 RuntimeError，避免销毁 in-flight SDK call。
- inference finally 只关闭 stdin controller 的逻辑状态、退出 freedrive并
  `backend.cleanup()`；`InferenceCameraManager` 没有 close，RealSense 不会在此
  显式停止；tactile reader/thread 不会 stop/join；stdin daemon thread也不 join。

### 6.3 已知 shutdown 风险

- `UdpComms._read_udp_loop()` 是 `while True`，没有 stop event；`close()` 只关
  socket，不 join `_rx_thread`。关闭后的阻塞 `recvfrom` 是否退出依赖平台；
  manager 输出“All threads stopped”并不包含这些内部 RX daemon threads。
- 多数 close 吞掉或只记录异常；没有普遍的 repeated-close characterization。
- 所有 manager/worker 多为 daemon thread，进程退出可掩盖未正确停止的问题。
- UR position backend 的通用 cleanup 只显式关闭 gripper，未见显式关闭
  manipulator/RTDE 对象；是否由第三方析构释放需要实机/SDK 验证。
- normal shutdown 不自动导出 HDF5 当前未 Stop 的 buffer；LeRobot close 会
  finalize dataset 对象，但未保存 episode buffer 的精确行为依赖 LeRobot 版本。

## 7. 对计划性能目标的当前基线差距

这不是 Phase 1 的修复清单，只记录迁移前事实：

- 视频 latest dict 有利于丢旧帧，但 JPEG 默认 60 kB chunks 会 IP fragmentation。
- WebRTC 没有显式 codec/encoder/queue/B-frame/lookahead/bitrate 配置和端到端
  latency instrumentation。
- pose 仍是 UTF-8 CSV 或 base64 文本外壳；不是 primary binary DataChannel。
- UDP RX 是 128 深度 FIFO，而非 latest-only；没有 sequence stale rejection，
  pose 也不重传但可在一轮中依次处理积压包。
- record/control 没有 ack、失败回复、duplicate id 或 ordered reliable channel。
- 通用 UR path 没有 stale-input watchdog；RealMan 专用 path 有 0.25 s stale
  hold 和严格 CAN-FD watchdog。
- 只有 RealMan CAN-FD 有较完整 timing metrics；在测量完善前不得声称满足更严
  数值性能目标，也不得先优化后补测量。

## 8. 现有测试、无法执行原因与无硬件 characterization matrix

### 8.1 现有测试

仓库只有 `tests/test_realman_teleop_loop.py`，使用 `unittest`，共 28 个 test
method。覆盖：

- Config 的 UR/RealMan 派生 axes、初始关节和 UR IP fallback；
- CAN-FD joint/TCP SDK 参数、degree conversion、trajectory high-follow；
- joint/translation/rotation slew limit；
- latest pending IK、hold 与 in-flight race；
- heartbeat、>100 Hz startup gate、10 ms violation、低速与 SDK error stop；
- realtime push callback enable/disable、单位/坐标/wrench 解析、坏数据拒绝；
- repeated start、thread start failure、live worker 不关闭 SDK；
- 30 Hz polling fallback 的 SDK thread ownership；
- controller rotation composition、visualizer 只读 cached state。

没有现有自动测试覆盖 README 命令、UDP JPEG header/ports、controller/hand/HB
解析、WebRTC signaling/DataChannel、HDF5/LeRobot schema、rollback、通用 UR
控制频率、camera settings、触觉滤波或全局 shutdown。

### 8.2 本环境为什么不能执行

- `python`、`py`、`python3`、`pytest` 均不在 PATH；测试命令在 collection 前
  即失败。
- 即使有解释器，test module 顶层仍要求 `cv2`、NumPy、SciPy、
  `airo_spatial_algebra`，并经 `realman_teleop`/`robot_backend` 导入 airo
  robotics stack；这些依赖是否与目标 SDK 版本兼容尚未验证。
- 当前没有锁文件或声明的 Python/LeRobot/aiortc 精确版本，schema 和 signaling
  结果可能随依赖版本变化。

### 8.3 无硬件 characterization test matrix 与验收 oracle

以下测试均应在临时目录、loopback/fake socket、fake clock、mock SDK 上运行，
且测试结束后不得留下 thread/process/socket。Phase 0 本任务按要求不新增测试，
只定义应在行为变化前固化的矩阵。

| ID | 无硬件场景 | 输入/夹具 | 验收 oracle（必须精确） |
|---|---|---|---|
| C-01 | Config 快照 | 实例化默认、UR5e、RealMan（给 dummy IP） | 所有 §2 标量/数组逐项相等；UR/RealMan 派生 axes、joint、IP 规则一致；非法值抛当前异常类型/消息。 |
| P-01 | controller old/new parser | golden 29/31-field strings及边界坏包 | 输出两个 dict、字段顺序/类型/FrameId/timestamp 精确；错误字段数与非数值返回 `None`。 |
| P-02 | hand text parser | 89-field L/R golden | wrist 7 项与 26×3 joints bitwise/近似相等；缺/多字段返回 `None`。 |
| P-03 | HB binary parser | 手工构造 320 B little-endian raw 再 base64 | magic/side/count/frame/joint 坐标精确；`wrist_pose is None`；坏 magic/count/短包返回 `None`；timestamp 只断言接收时 ms 范围。 |
| P-04 | JPEG chunks | 固定小图、固定 JPEG quality、fake socket | 每包前 12 B 与 `!IHHI` 解包精确；index 0..N-1、total/bytes 一致；拼接 payload 等于 OpenCV golden JPEG；每 camera frame id 独立递增并 wrap。 |
| P-05 | 端口分配 | fake `UdpComms` 捕获构造参数，0/1/3/5 cameras，tactile on/off | 端口严格等于 §3.1；至少三 socket；WebRTC 无 camera TX socket；不得真实 bind。 |
| P-06 | control/record 文本 | golden `Start/Stop/Undo` 与 zoom/fine 字符串 | 三个状态 flag 和 camera zoom/fine_mode 与当前 parser 精确；分别固化 UDP port-key 与 WebRTC index-key 差异。 |
| W-01 | WebSocket signaling | aiohttp test client + mocked peer/track | hello→hello_ack；offer→按相机数 addTrack、创建 `"control"`、answer envelope；ICE 字段映射；stop/新 offer 恰好关闭旧 peer。 |
| W-02 | DataChannel | fake channel 注入消息 | `"control"` 唯一标签；message 调当前 WebRTC control parser；legacy UDP control 仍可达。 |
| D-01 | HDF5 golden | temp dir、fake manager、T=2，配置组合参数化 | 完整 path tree、attrs、shape、dtype、timestamp names、RGB/depth/tactile值与 §4.2 一致；文件名/description/index 精确。 |
| D-02 | LeRobot feature contract | mock `LeRobotDataset.create/resume/add_frame` | 传入 feature dict、fps、robot_type、video flag、每帧 task/zeros fallback/uint16 depth 与 §4.3 精确。另在支持版本生成可提交的 meta/parquet golden。 |
| D-03 | schema mismatch | temp dataset sentinel + mocked loaded features | 当前基线 oracle 是 root 被删除并 recreate；测试必须在 temp dir，显式标为“危险兼容行为”，防止误触真实数据。 |
| D-04 | rollback | temp HDF5/合成 LeRobot parquet+video+meta | unsaved buffer 优先丢弃；否则只删最后 episode、编号复用、totals/splits/description/stats 变化与 §4.4 精确。 |
| R-01 | UR 60 Hz 调度 | fake monotonic/sleep、fake backend、active/standby/stale pose | active 每目标 tick 的 call 次数与 dt=1/60；standby 不发运动；记录当前“stale pose 仍重复”的行为。 |
| R-02 | RealMan 200 Hz | 现有 fake arm/fake clock测试扩展 | 一个窗口 achieved `>100` 才 ready；验证后 gap/call `>10 ms` 单次 stop；heartbeat 50 ms；overrun 不 burst catch-up。 |
| R-03 | collection/inference | fake clock，阻塞/超时两类 workload | collect 默认 10 Hz 且不追赶；inference 默认 CLI 10 Hz、execute dt=0.1；超时只报警不反向追赶。 |
| A-01 | camera latest behavior | fake camera 输出递增 sequence，慢 consumer | shared dict 只保留最新 frame，无无界 queue；dataset image timestamp 与对应 captured frame 一致；重复 WebRTC frame行为被固定。 |
| T-01 | BLE tactile filter | 构造已校准 reader 的纯数组状态 | baseline/deadband/clip/max-delta/EMA/Kalman 的逐步数值与 NumPy oracle 相等；shape `(4,3)`、dtype float32、wire 48 B。 |
| T-02 | serial tactile contract | fake serial packets | 当前输出 `(41,3)` int32/wire 492 B；喂默认 recorder 时断言 shape `ValueError`，从而显式暴露而非掩盖不兼容。 |
| F-01 | wrench filter | 6D impulse/ramp/短向量 | MA8→LPF0.15 顺序精确；短于 6 返回零；reset 清窗口/initialized；force/torque split 符号不变。 |
| L-01 | manager shutdown | fake camera/socket/thread/async peer，多次 close | close 不挂死；所有非预先存在 thread/process 均停止；camera/peer/runner/socket 各关闭一次或幂等；当前 UdpComms 无限 RX 问题应先由失败 characterization 暴露。 |
| L-02 | setup failure cleanup | 在每个 startup stage 注入异常 | 记录当前泄漏点作为 failing/expected characterization；后续修复时 oracle 改为所有已创建资源逆序关闭，且原异常不被吞。 |
| I-01 | import/CLI smoke | 隔离环境，`--help`/module import，mock hardware modules | import/`--help` 不打开 socket、相机、serial、robot、不创建/删除 dataset；README 每条命令路径要么 0 退出，要么有明确迁移 wrapper。 |

通用 suite 的额外 oracle：

1. 测试前后枚举非 daemon/daemon threads、child processes、打开的 fake sockets；
   增量必须为 0。
2. 所有磁盘测试只可写临时目录；真实 `DATASET_DIR` 必须有 sentinel 防护。
3. protocol golden 同时保存原始 bytes 与语义字段；不能只断言“解析成功”。
4. rate 测试用 fake monotonic/perf-counter，避免 CI wall-clock flaky。
5. 可选依赖缺失应使对应 adapter 有清晰错误，不得使无关纯 parser/schema 测试
   collection 失败。

## 9. 尚需实机验证

| 设备/链路 | 必须验证的项目 | 实机验收 oracle |
|---|---|---|
| Unity/Quest controller | old/new CSV 实际字段、时间单位、坐标/按钮、包率、乱序/丢包 | Python 与 Unity 抓包逐字段一致；frame 单调；60/90/120 Hz headset 模式下无错误映射；断流后安全行为已明确。 |
| Unity hand | 当前 Unity 究竟发 24 还是 26 joints、HB 是否含 frame id、side 与 joint order | 抓取 text/HB golden，Unity/Python joint index 可视对应；README 与实现选择一个受版本标记的真值。 |
| Unity JPEG UDP | 大 datagram 在真实 NIC/Wi-Fi/Quest 的分片、丢包、重组超时 | 多相机目标 FPS 下测 loss、重组失败、P50/P95/P99 延迟；header 与 `UdpSocketMultiHD.cs` byte-for-byte。 |
| Unity WebRTC | offer/answer/ICE 兼容、Python-created DataChannel 是否被 Unity 接受、codec | 实际 SDP 与 stats 证明 codec/profile/分辨率/FPS/bitrate；control 双向可用；断线重连无旧 peer 泄漏。 |
| RealSense | 640×480@60 支持、实际 acquisition 为约 30 Hz、RGB/BGR/depth单位 | 每型号/多相机连续 30 min；frame/timestamp 单调，dataset 色彩和 depth 米制正确，无 SDK thread 泄漏。 |
| UR3e/UR5e | 初始关节、axis map、IK、安全阈值、60 Hz servo、stale input | e-stop/保护停机准备下低速测试；命令 rate/jitter、workspace、关节/夹爪方向正确；VR 断流产生批准的 safe hold。 |
| RealMan | 200 Hz CAN-FD、UDP state push、force frame、watchdog/quarantine | 启动 gate 与 >10 ms/50 ms 故障注入按预期停；5 ms push 持续；joint/TCP 单位、frame 与 integrated wrench正确。 |
| BLE4 tactile | taxel order、符号、量程、baseline、断线重连、48 B Unity解码 | 已知加载逐 taxel响应；空载漂移/噪声/饱和统计；重校准与重连后 shape/dtype不变。 |
| serial tactile | 41-taxel legacy 意图及与 `(4,3)` schema 的关系 | 由负责人决定它是受支持 legacy schema 还是错误配置；决定前不得静默 reshape/truncate。 |
| HDF5/LeRobot | 真实训练/回放工具读取、LeRobot 版本生成的标准列与视频编码 | 写一集→重启续录→训练/回放→rollback→再录；episode index、帧数、视频、meta、时间戳全链路一致。 |
| shutdown | Ctrl+C、VR timeout、相机断线、SDK hang、dataset error 每个阶段 | 进程在约定时间退出；无残留端口/thread/process；robot safe hold/stop；数据文件可重开且不被意外删除。 |

## 10. 基线结论

当前最需要在迁移前锁定的兼容面是：

1. README 与源码已经在 IP、JPEG quality、重力补偿、hand joint/header 以及两个
   根级命令上发生漂移；
2. 高频 state 当前仍是 UDP 文本/FIFO，WebRTC DataChannel 只做 control；
3. JPEG header 明确但默认 datagram 依赖 IP 分片；
4. HDF5 和显式 LeRobot features 可由源码定义，但 LeRobot 标准列受版本影响，
   且 schema mismatch 会删除原目录；
5. 通用 UR、RealMan、采集、相机、触觉、推理各有不同实际调度频率，不能用单一
   “实时”描述替代测量；
6. 正常 main/RealMan shutdown 有清理路径，但 startup exception、UdpComms RX、
   inference camera/tactile/stdin 存在未闭合风险；
7. 现有测试集中在 RealMan CAN-FD，协议、dataset、触觉、命令入口和生命周期仍
   缺 characterization。

在 §8 的 golden/oracle 和 §9 的实机矩阵完成前，不应修改上述协议、schema、
安全参数、初始位、坐标/夹爪映射、控制频率或 episode/rollback 语义。
