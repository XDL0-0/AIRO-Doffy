# classic Quest 遥操作功能清单

> Public copy: local paths, device identifiers and site addresses are anonymized; recorded test results and version information are retained.

## 范围与判定口径

这是对 `/path/to/unity/classic` 的只读盘点，快照日期为 2026-09-16。检查了：

- `Assets/Scripts`（包括 `New`、`MultipleVideoStream`、`TCP`、`Force`、`Tactile Visualization`、`UpperLimb`、`VirtualRobot`、`Unused`、`Viz`）；
- `Assets/Editor/TactARSceneSetup.cs`；
- `Assets/Scenes/V0.6.0 Realtime_Force.unity` 与 `Assets/Scenes/V0.7.0 Upper Limb.unity` 的序列化对象、父子关系和 `UnityEvent`；
- `Assets/Prefabs/Canvas.prefab` 及相关箭头/触觉预制体；
- `Assets/Settings/TeleopConfigUpperLimb.asset`。

本文中的三种标记含义如下：

- **Active scene wiring**：V0.6/V0.7 场景已经序列化的活动组件、引用、对象状态或按钮事件。面板初始 inactive、但由已接线管理器按需启用的对象也列在这里。
- **Runtime auto-created**：场景没有序列化组件，代码在 `Awake`、`Start` 或 `[RuntimeInitializeOnLoadMethod]` 中创建/补线；重构时不能只按场景 YAML 迁移。
- **Unused legacy**：当前两个目标场景没有活动引用，或对象明确 inactive，或仅存在历史场景/注释代码。仍记录其可见效果和协议，避免误删仍被用户或外部机器人依赖的路径。

除本文件外没有写入 classic；没有运行会保存场景的 Editor 菜单命令。场景实际差异只有：V0.7 增加 Upper Limb AKM 组件和三项 UI 子对象，另有 Panel alpha 改动；V0.6 的 `UpperLimbAkmBootstrap` 仍会在运行时补上 AKM 管理器/UI，但场景没有对应按钮。

## 两个目标场景的接线基线

### Active scene wiring

下表是从两个场景的 `m_EditorClassIdentifier`、序列化字段和父子 `fileID` 读取的事实。除注明 V0.7 的行外，V0.6/V0.7 相同。

| 对象/路径 | 组件和实际接线 | 初始状态与意义 | 代码路径 |
|---|---|---|---|
| `AppManager` | `AppManager`；IP 输入 `485358093`、WebRTC 数量输入 `2134343646`、状态文字 `1985840377`、Start `192050695`、控制模式文字 `972898708`、重定位按钮 `566497589`、WebRTC 按钮 `1078572146`、1/2/3 路面板、Add UDP 按钮和 `VideoWindowManager` 已接入。`teleopConfig=0`、`sessionStateText=0`、`recalibrateButton=0`、`debugInfoToggle=0`、`menuPanel=0`、`controlModeButton=0`、`versionText=0`。 | 组件活动，但关键配置资产和部分可选 UI 没有序列化引用；控制模式按钮仍由场景的持久 `UnityEvent` 直接调用，所以不能仅依据 `controlModeButton=0` 判断按钮无效。 | `Assets/Scripts/New/AppManager.cs:1-577`；V0.6 场景约 `22853-22878`，V0.7 同位置 |
| `UdpSocket` | 8001/8003/8005 三个发送端点；IP 输入和提示文字已接入，后台接收线程启动。 | 活动；默认 hard-code IP 为 `192.0.2.11`，启动时再读 `PlayerPrefs`。 | `Assets/Scripts/UdpSocket.cs:1-213` |
| `DualControllerSender` | `udpSocket=1442753682`、`stateSending=1`；状态文字为空。 | 活动；每 10 ms 取左右 Touch，先过 `CanSendTeleopData` 安全门。 | `Assets/Scripts/DualControllerSender.cs:1-126` |
| `HandTrackingManager` | `HandTrackingSender`；左右 `OVRHand`/`OVRSkeleton` 已接入，`stateSending=1`、`sendRateHz=60`、`useBinaryProtocol=0`、`onlySendWhenTracked=1`、`skipWhenControllerActive=1`。 | 活动；本场景实际发送文本 H 协议且为 60 Hz，不是脚本默认的 binary/30 Hz。 | `Assets/Scripts/HandTrackingSender.cs:1-206` |
| `TrackingModeManager` | `dualControllerSender=1108104674`、`handTrackingSender=140660024`、`modeText=2059397721`，`currentMode=0`；`Hand_controller_change_button` 持久调用 `ToggleTrackingMode`。 | 活动；启动读 `cfg_trackingMode`，0=Controllers、1=Hands，并只启用一个 sender。 | `Assets/Scripts/TrackingModeManager.cs:1-67`；场景约 `56319-56331`（V0.6）、`57071-57083`（V0.7） |
| `Calibration` | `CalibrationTool` 引用 `Coord=1933274868`、`autoCreateCoord=1`、`showGizmoWhileCalibrating=1`、`enableControllerShortcut=1`；同一对象挂 `TCPPoseReceiver`，监听 8012，右 TCP=`1631814663`，右力=`1727653315`，比例 `.01`，自动补 TCP。 | `Calibration` 活动；`Coord` 初始 inactive，`RightTCP` 是 `Calibration` 直接子物体，右力箭头是 TCP 子物体。8012 是当前主 TCP+力 JSON 通道。 | `Assets/Scripts/Calibration/CalibrationTool.cs:1-280`；`Assets/Scripts/TCP/TCPPoseReceiver.cs:1-260`；场景约 `18291-18331` |
| `RightForce` | `ForceArrow`，`tcpMarkerRadius=.015`、`headRadius=.02`、`shaftRadius=.004`，白色 marker、绿→红力颜色。`Arrow` 子物体初始 inactive。 | 活动；`ForceArrow.Awake` 会生成/重建 marker、head、shaft，收到非零力才显示箭头；随 `RightTCP`/Calibration 坐标移动。 | `Assets/Scripts/Force/ForceArrow.cs:1-130`；场景约 `54799-54930`（V0.6）、`55551-55682`（V0.7） |
| `ForceSensorReceiver` | `listenPort=8013`、target=Right，左右旧引用为空，右力/右 TCP 已指向活动对象。 | 对象明确 inactive；是 8013 备份/兼容接收器，主路径由 8012 `TCPPoseReceiver` 占用。 | `Assets/Scripts/Force/ForceSensorReceiver.cs:1-300`；场景约 `26484-26530` |
| `WebRTCReceiver` | `WebRTCVideoReceiver` 活动；无场景自定义字段。 | 活动但不会自己开始连接；由 `VideoStreamManager` 创建 receive-only peer/transceiver。 | `Assets/Scripts/New/WebRTCVideoReceiver.cs:1-240` |
| `WebRTCStreamManager` | `VideoStreamManager`；receiver 已接入，single/dual/tri panel 分别接入 1/2/3 个 `RawImage` 面板。 | 活动；六个面板初始 inactive，WebRTC 开关和 track count 决定启用组。 | `Assets/Scripts/New/VideoStreamManager.cs:1-340`；场景约 `webRTCSingle=292111265`、dual `{2040674143,2096875186}`、tri `{1371036347,293473024,790547081}` |
| `VideoWindowManager` | `UdpWindowManager`；预制体 GUID `14d06f2d739b37549abe5c5d3e780fe8`，容器 `2136216292`，IP 输入 `485358093`，`sendInterval=.05`，`receiveTactile=0`，`FocusModeButton=0`、`FocusModeButtonLabel=0`、tactile generator=0，reset threshold `.8`。 | 活动；Add UDP 按钮能动态创建窗口；场景关闭触觉接收以避开 8012 冲突。因为 Focus 两个引用为空，旧的焦点/分辨率 UDP8005 循环在此场景被跳过。 | `Assets/Scripts/MultipleVideoStream/UDPManager.cs:1-520`；场景约 `42892-42920`（V0.6）、`43644-43672`（V0.7） |
| `RecordingController` | `recordButtonText=1957610462`、`udpSocket=1442753682`；`Record_button` 持久调用 `Recording`。 | 活动；录制状态由按钮文字 `Start Record`/`Stop Record` 推断，协议发到 8003。 | `Assets/Scripts/RecordingController.cs:1-36`；场景约 `23777-23789` |
| `Folding Manager` | `CollapsibleCanvas`；折叠按钮/header/body 已接入，`showWhenCollapsed` 白名单包含主要按钮、输入、窗口面板；`startCollapsed=0`、`hideMode=CanvasGroup`。 | 活动；折叠用 alpha/interactable/raycast 关闭，不破坏动态子树；启动展开。 | `Assets/Scripts/FoldingCanvas.cs:1-380` |
| `Instruction_display_button` | `WindowToggle`；`UserManual` CanvasGroup 和 toggleButton 已接入。 | 活动；手册启动隐藏，按钮切换 alpha 与 blocksRaycasts。 | `Assets/Scripts/User_Manual_control.cs:1-49` |
| `AppController` | `Quit Button` 持久调用 `QuitApp`。 | 活动；调用 `Application.Quit()`。 | `Assets/Scripts/AppController.cs:1-30`；场景约 `23026-23038` |
| `Monitor_L`/`Monitor_R` | `Controller`/`MetaControllerData`，左右 `controllerType=1/2`，分别接 `Python Test_L/R`；监视对象和文字 inactive。 | 组件存在但监视 UI inactive；Controller 仍会采样并把 CSV 写入关联 `PythonTest`，不是网络发送器。 | `Assets/Scripts/Controller.cs:1-180`；`Assets/Scripts/PythonTest.cs:1-45` |
| V0.7 `UpperLimbAKM` | `UpperLimbAkmManager` + `UpperLimbAkmUi` 同一根对象，参数见下方；按钮 `WRM_enable Button` 持久调用 `ToggleWrm`，`AKM Calibration Button` 持久调用 `OnCalibrationClicked`。 | 活动；V0.6 没有这三个对象，依赖 bootstrap 运行时创建管理器/UI，因此 V0.6 没有可操作按钮。 | `Assets/Scripts/UpperLimb/UpperLimbAkmManager.cs:1-590`；`UpperLimbAkmUi.cs:1-180`；V0.7 diff 约 `31501-32250` |

### 持久化按钮/输入事件

两个目标场景都存在如下 YAML `UnityEvent`，重构 UI 后必须保留等价行为和调用目标：

| 场景对象 | 用户操作 | 持久目标/方法 | 代码/场景证据 |
|---|---|---|---|
| `Start_Streaming_button` | 点击开始/停止 | `AppManager.ToggleStreaming` | 场景约 `5807-5819`；`AppManager.ToggleStreaming` |
| `CONTROL MODE Button` | 点击 Mirror/View 切换 | `AppManager.ToggleTeleopControlMode` | 场景约 `12768-12780`；`AppManager.ToggleTeleopControlMode` |
| `ADD UDP CAMERA Button` | 创建一扇 UDP 视频窗 | `UdpWindowManager.CreateUdpWindow` | 场景约 `13088-13100`；`UDPManager.CreateUdpWindow` |
| `Reposition Button` | 进入机器人底座摆放 | `AppManager.BeginRobotBaseReposition` | 场景约 `20711-20723` |
| `Quit Button` | 退出应用 | `AppController.QuitApp` | 场景约 `23026-23038` |
| `Record_button` | 开始/停止录制 | `RecordingController.Recording` | 场景约 `23777-23789` |
| `Hand_controller_change_button` | 手柄/手部模式切换 | `TrackingModeManager.ToggleTrackingMode` | V0.6 约 `56319-56331`，V0.7 约 `57071-57083` |
| `IP Input` | 输入框结束编辑 | `UdpSocket.UpdateIP`（持久 `OnEndEdit`；运行时 `Start` 也追加监听） | V0.6 约 `17227-17239`、`67610-67622`；V0.7 约 `17230-17242`、`68366-68378` |
| V0.7 `WRM_enable Button` | 开关上肢 WRM | `UpperLimbAkmManager.ToggleWrm` | V0.7 约 `31696-31708` |
| V0.7 `AKM Calibration Button` | 开始/重新校准上肢 | `UpperLimbAkmManager.OnCalibrationClicked` | V0.7 约 `31966-31978` |

`num_WebRTC Input` 没有持久自定义调用；`AppManager.Start` 追加 `onEndEdit`，把数值限制到 1..3 并保存。V0.6/V0.7 的 `AppManager.controlModeButton` 字段虽然为 0，按钮的持久调用仍直接打到 `AppManager`，而 `AppManager` 只在有引用时追加 listener；这是迁移时容易丢掉的两种接线来源。

## 功能矩阵

每一项都给出用户动作、输入/输出与依赖、应保留行为和可执行验收。网络验收可用同网 PC 的 UDP/WebSocket 抓包或最小假接收器；不要求在 classic 内添加测试代码。

| ID / 状态 | 用户操作与效果 | 输入、输出、依赖 | 应保留的行为 | 测试/验收方法 | 代码路径 |
|---|---|---|---|---|---|
| **F-001** Active scene wiring：会话启动/停止 | 启动应用，填写 IP/路数，按 `Start Streaming`；再次按下停止；左手柄 Start 可停止，追踪丢失后右手柄 Start 可触发重校准。 | `AppManager`、`UdpSocket`、`TeleopReferenceFrame`、WebRTC manager；状态 `Idle → Streaming`，开启视频时经过 `VideoConnecting`，停止经过 `Stopping`。 | 开始时清错、保存 PlayerPrefs、校准 teleop frame、按钮改为 `Stop Streaming`；停止时关闭视频、清 frame、恢复菜单/`System Ready`；视频失败只降级视频状态，仍保留 teleop。 | 无视频开启/停止各一次；检查状态文字和 Start 文案；抓包确认停止后姿态包停止；开启 WebRTC 后断信令，确认状态提示视频不可用而控制会话未被误停。 | `New/AppManager.cs:80-227, 247-365`；`TeleopSessionState.cs:1-20` |
| **F-002** Active scene wiring：IP、配置与 UDP 基础通道 | 在 IP 输入框写合法/非法地址并结束编辑；重启后检查保存值。 | 文本 UTF-8 发往同一目标的 UDP 8001（姿态）、8003（录制）、8005（控制）；接收线程异步收包并主线程显示错误/状态。`PlayerPrefs` keys：`cfg_ip`、`cfg_trackingMode`、`cfg_sendRateHz`、`cfg_useBinary`、`cfg_numWebRTC`。 | 合法 IP 才更新 endpoint；非法输入显示 invalid/不应崩溃；关闭时停止线程并释放 socket；持久化值优先于默认值。 | 输入 `192.0.2.11` 与非法字符串；检查提示、目标地址和重启恢复；发送空消息/断网后确认不会卡住主线程。 | `UdpSocket.cs:46-213`；`AppManager.cs:151-192` |
| **F-003** Active scene wiring：双手柄姿态 | 手握左右 Touch，移动/转动、摇杆、扳机、握把、A/X、B/Y、摇杆按下。 | OVR `LTouch`/`RTouch`；每 10 ms（100 Hz）采样，经过 Mirror/View 变换；UDP8001 文本：`C,<frame_id>,<timestamp_ns>,<leftData>,<rightData>`，每侧含 type、position xyz、rotation xyzw、joystick xy、trigger、grip、A/X、B/Y、joystickPress。 | 只在 `AppManager.CanSendTeleopData` 为真时发；单帧同时采左右，使用 monotonic frame/timestamp；落后超过 30 ms 对齐当前时间且最多 3 次追赶，避免爆发；保留按钮边沿/轴值语义和坐标符号。 | 假接收器统计约 100 Hz；静止时 frame/timestamp 单调；分别按每个按钮确认对应字段；停止 streaming 或 tracking lost 后确认无 C 包；Mirror 与 View 各做固定姿态对照。 | `DualControllerSender.cs:1-126`；`TeleopControlModeManager.cs:117-150` |
| **F-004** Active scene wiring：手部骨骼发送 | 切换到 Hands，手掌/手指移动、摘掉控制器；切回 Controllers。 | 左右 `OVRHand`/`OVRSkeleton`，要求 tracked 且至少 26 bones；场景实际文本 H：`H,<L/R>,<frame_id>,<timestamp_ns>,wrist pos+quat,26×bone xyz`，UDP8001，60 Hz；binary `HB,<base64>` 仍是脚本支持协议。 | `onlySendWhenTracked`、`skipWhenControllerActive` 过滤必须保留；手部位置经过 `TeleopReferenceFrame.TransformWorldPoint`；TrackingModeManager 只允许当前模式 sender 发。 | 佩戴/摘下手部追踪，统计 L/R H 包；不足 26 bones 或控制器连接时应跳过；切 mode 后旧 sender 无包、新 sender 有包；检查 60 Hz 和 frame/timestamp。 | `HandTrackingSender.cs:1-206`；`TrackingModeManager.cs:22-61` |
| **F-005** Active scene wiring：手柄/手部模式 | 点击 `Hand_controller_change_button` 多次；重启应用。 | `TrackingModeManager`、两个 sender、`cfg_trackingMode`、`Hand_controller_change_TEXT`。 | 按钮在 Controllers/Hands 间循环，持久化；ApplyMode 开启一个 sender、关闭另一个，文字同步；上肢 WRM UI 读取同一 key 时，Hands 模式应禁用 WRM。 | 点击后检查文字、两 sender `stateSending`；重启确认模式保持；V0.7 进入 Hands 时 WRM 按钮不可用且已开启 WRM 自动关闭。 | `TrackingModeManager.cs:1-67`；`UpperLimbAkmUi.cs:81-160` |
| **F-006** Active scene wiring + runtime：坐标/手动校准 | 开 streaming 自动校准；按右手 A+左手 X chord 进入/退出校准；校准中右扳机转 yaw、左扳机平移，A/X 缩小 gizmo、B/Y 放大。 | `CalibrationTool`、`Coord/CalibrationGizmo`、`TeleopReferenceFrame`、`Camera.main`；自动模式用 head origin + inverse yaw，手动模式保存 inverse rotation/origin。 | 进入时显示 Coord gizmo，退出时设置手动 calibration，并同步/创建 `TeleopWorld` robot-base anchor；清理时恢复；世界点/旋转和控制器包要使用同一符号约定。 | 固定头部朝向/控制器位置记录 packet，改变头部后自动校准结果应保持；手动摆放后移动头部，验证 packet 不漂移；按 chord 及各轴控制检查 gizmo 与状态。 | `Calibration/CalibrationTool.cs:1-280`；`Calibration/CalibrationGizmo.cs:1-120`；`TeleopReferenceFrame.cs:1-123` |
| **F-007** Runtime auto-created：Mirror/View 与机器人底座 | 点击控制模式按钮；在 View 中点击 `Reposition`，用右手柄移动定位，右摇杆按下确认位置，再转动手柄并按下确认方向。 | `TeleopControlModeManager`、`TeleopWorld`、`TeleopViewModeMapper`、`RobotBasePlacementTool`；View packet 映射为 robot-base 坐标（含 position/rotation 轴交换和符号）。 | Mirror 使用 reference frame；View 只在 anchor 存在时转换，否则安全回退 Mirror 并记录日志；View 显示 robot-base axes；重定位自动切 View，两个确认阶段和锁定/预览行为保持；右摇杆 reset 与 UDP 窗口 reset 的冲突抑制保持。 | Mirror/View 对同一个手柄位姿抓包比较；完成位置/方向两阶段后检查 axes、anchor 和 packet；未创建 anchor 时确认不发送错误变换；在放置时按右摇杆不应同时 reset 窗口。 | `TeleopControlModeManager.cs:1-210`；`TeleopWorld.cs:1-230`；`TeleopViewModeMapper.cs:1-110`；`RobotBasePlacementTool.cs:1-350` |
| **F-008** V0.7 Active scene / V0.6 Runtime auto-created：上肢 WRM/AKM | V0.7 点击 `WRM_enable Button` 开关；点击 `AKM Calibration Button`，或右手 A 按住默认 2 s；先采 hanging/down，再采 horizontal；按右手握把 clutch；按右摇杆 recenter。 | Meta XR body tracking/OpenXR 权限；右肩、左肩、右下臂、右腕有效；`UpperLimbAkmManager` 每 30 Hz 通过 UDP8005 发 JSON：`{"type":"WRM","frame_id":n,"timestamp_ns":t,"elbow_alpha":a,"confidence":c,"grip_trigger":0/1,"joystick_press":true/false}`。该 sender 的 `UpdateSend` 直接以 `WrmEnabled` 为门，不读取 AppManager 的 `CanSendTeleopData`。 | 校准状态机 `Idle/WaitingDown/CalibratingDown/WaitingHorizontal/CalibratingHorizontal/Calibrated`、2 s hold、至少 10 样本、角度/高度跨度拒绝条件、EMA `alphaFilterTau=.12`、dead/full zone、tracking invalid 时 confidence=0 且保持 alpha、recenter 脉冲只发一次都要保持；Hands 模式禁用 WRM；V0.6 无 UI 但 bootstrap 不能因无按钮而重复创建；若新安全策略统一 gate，须显式决定是否改变“仅开启 WRM 即发包”的兼容行为。 | 在 V0.7 无权限/权限成功各测一次；用模拟有效关节完成两段校准，检查 alpha 0→1 平滑及 confidence；按/松 A 验证 hold/cancel；在 streaming 前后分别抓 UDP8005 检查实际 30 Hz JSON、clutch/recenter；Hands 模式确认自动关 WRM。 | `UpperLimb/UpperLimbAkmManager.cs:145-590`；`UpperLimb/UpperLimbAkmUi.cs:28-180`；`UpperLimb/UpperLimbAkmBootstrap.cs:1-35` |
| **F-009** Active scene wiring：8012 TCP+力显示 | 机器人端向 8012 发组合 JSON；观察右 TCP 和箭头；发送零力/非零力/扭矩。 | `TCPPoseReceiver` 只解析 UTF-8 JSON：`rightTCP/leftTCP` 各含 `position`、`rotation`（w,x,y,z）、`force`、`torque`；当前只接右侧；position/rotation 赋给 `RightTCP` local transform，force×`.01` 调 `ForceArrow`。 | 8012 单一主接收器；RightTCP 在 Calibration 下，校准/手动坐标改变时整个显示跟随；w-first quaternion 转 Unity xyzw；零力隐藏 head/shaft，marker 仍可见；线程收包、主线程应用。 | 发送已知位置/四元数并与 Unity local transform 比对；发送 [0,0,0] 力应隐藏箭头，发送 +X/+Y/+Z 检查方向和颜色；校准前后确认 TCP/arrow 同步移动；重复绑定端口时应给出可诊断错误。 | `TCP/TCPPoseReceiver.cs:1-260`；`Force/ForceArrow.cs:1-130`；场景 `Calibration`/`RightTCP`/`RightForce` 接线 |
| **F-010** Unused legacy（备份）：ForceSensorReceiver | 手动把 inactive receiver 打开或部署兼容二进制/JSON 传感器时显示右力。 | 8013；二进制至少 24 bytes 的 6 个 little-endian int32（force/torque）按 `.0001` 缩放，或 JSON `device_id`/start/end/scale；可 target Left/Right/Both。 | 作为备用输入保留但默认关闭；不要与 8012 `TCPPoseReceiver` 或旧 tactile listener 同时绑定；自动补 `ForceArrow`/TCP mount 的行为需可用。 | 开启独立测试端口发送二进制和 JSON，检查 target/比例；重新关闭后 8012 主路径仍能独占；端口占用时错误不能静默替换主 receiver。 | `Force/ForceSensorReceiver.cs:1-300`；`Editor/TactARSceneSetup.cs:34-65, 225-290` |
| **F-011** Active scene wiring：WebRTC 多窗口视频 | 设置数量 1/2/3，点击 `Enable WebRTC CAMERA`；查看 single/dual/tri 面板；再次点击禁用。 | `VideoStreamManager` + `WebRTCVideoReceiver` + `VideoSignalingClient`；默认 signal `ws://<ServerIP>:8765`，连接超时 4 s；receive-only transceiver，texture 按 track index 投到 RawImage；data channel 发送分辨率/控制字符串。 | 开启 WebRTC 先启用对应面板、关闭并清空所有 UDP 窗口、禁用 Add UDP；按数量将 receiver `ExpectedTrackCount` 限制 1..3；停止发送 `stop_video`、关闭 peer/signaling；信令/视频失败不能停止 teleop，仅显示 Video issue；切回 UDP 恢复 Add。 | 1/2/3 路分别连接假信令/视频 server；检查面板数量、track index、texture；开启 WebRTC 时确认 UDP 窗口被关闭/Add 置灰；断信令后抓控制仍在、状态为黄色；重复同一 resolution control 不应反复发送。 | `New/VideoStreamManager.cs:1-340`；`New/WebRTCVideoReceiver.cs:1-240`；`New/VideoSignalingClient.cs:1-220` |
| **F-012** Active scene wiring + prefab：UDP 视频多窗口 | 点击 `ADD UDP CAMERA Button`；输入窗口端口；点每窗 `ZOOM Button`；拖动窗口标题/根；按右 B/Y 循环前/左/右布局；右摇杆按下且扳机≥.8 将所有窗口回到 x1.0；点关闭。 | `UdpWindowManager` 动态实例化 `Canvas.prefab`；端口依次 8000、8002、8004、8006、8008（最多 5）；prefab 根含 `VideoWindowController`，RawImage 含 `DraggableWindowWorld`，`Plane` 含 `UdpSocketMultiHD`；UDP frame header 为大端 `frameId:uint32, chunkIndex:uint16,totalChunks:uint16,totalBytes:uint32`，支持分片 JPEG/PNG 和单包兼容。 | 每窗独立 IP/端口、关闭释放端口、端口输入非法不重启接收；分辨率标签循环 x1.0→x1.5→x2.0→x1.0；拖动只改变父级 local XY、子控件点击不误拖；B/Y 三位置及旋转保持；reset 遇 placement suppress 时不执行；完成帧在主线程 `LoadImage`，不在接收线程触 Unity。 | 创建 5 窗后第 6 次应提示/不越界；每窗发单包和多片帧，检查图像、重复片和端口；拖动按钮/输入框确认不带窗；验证 zoom 标签；按 B/Y、reset 与 View placement 组合检查布局和抑制。 | `MultipleVideoStream/UDPManager.cs:1-520`；`VideoCreateManager.cs:1-140`；`UdpSocketMultiHD.cs:1-240`；`Prefabs/Canvas.prefab`（GUID `14d06f2d739b37549abe5c5d3e780fe8`） |
| **F-013** Active scene wiring（局部）+ legacy 控制：焦点/缩放消息 | 用户可以点每窗 zoom；若重新接入 Focus 按钮，右 A 可切 Fine Control Mode；面板可拖动。 | `FocusModeButton`/`FocusModeButtonLabel` 在两个目标场景是 null，所以 `FocusControl` 和 `Resolution_loop` 的 UDP8005 发送条件不成立；每窗 `ChangeResolution` 仍更新本地文字；旧实现曾向 8005 发 `<port>,<label>;...;Fine Control Mode,ON/OFF;`。 | 保留“本地 zoom 可用、当前场景不自动发旧分辨率控制”这个实际行为，或在新 UI 明确决定重新接入时同时解决 8005 WRM/控制消息仲裁；不要让恢复 Focus A 与 AKM calibrate A 意外抢按键。 | 目标场景点 zoom 后抓包应无旧 resolution message；若显式接入 Focus，再验证 label、每 50 ms 控制循环和 WRM 共存；按右 A 检查单一拥有者。 | `MultipleVideoStream/UDPManager.cs:111-239, 344-390`；`UpperLimbAkmManager.cs:484-532` |
| **F-014** Unused legacy（历史场景）：41 点触觉可视化 | 在带 tactile UI 的历史 V0.3/V0.5 场景，接收传感器后点击 `Tactile UI Button` 显示/隐藏 41 个箭头；观察颜色、长度。 | `TactileUIManager` 首次数据后按钮变绿可用；`TactileSensorGenerator` 4×8 主阵列 + front/top/bottom 各 3 点=41，固定每点 3 个 little-endian int32，乘 `.0001`；`TactileArrow` 按 magnitude 蓝→红、长度随力变化；`Tactile Visualization/GrabbleWindows.cs` 支持世界窗口拖动。 | V0.6/V0.7 中该路径被 Editor 工具删除且 `VideoWindowManager.receiveTactile=0`，不应误标成当前活动功能；如果重构承诺兼容历史演示，保留 byte table、41 点布局、首次数据按钮状态、拖动和可见性。8012 不可与当前 TCP/力主通道共用。 | 在历史场景或隔离测试端口发送 41×3×int32，检查布局、比例、箭头颜色/长度和 Show/Hide；目标场景确认没有第二个 8012 listener。 | `Tactile Visualization/TactileUIManager.cs:1-180`；`TactileSensorGenerator.cs:1-230`；`TactileArrow.cs:1-130`；`GrabbleWindows.cs:1-120`；`Editor/TactARSceneSetup.cs:30-65, 75-115` |
| **F-015** Active scene wiring：录制按钮 | 点击 `Record_button`，观察 `Start Record`/`Stop Record`；再次点击。 | `RecordingController` 检查 `recordButtonText.text == "Stop Record"`；当前为 Stop 时发 UDP8003 `Stop` 并改 Start，否则发 `Start` 并改 Stop。 | 保留文字精确值、按钮直接事件、UDP8003 字符串和幂等的 Start/Stop 交替；外部录制器可能只认这两个命令。 | 假 UDP 接收器点击 10 次，包序列必须 Start/Stop 交替；重建 UI 后文字初始值与第一次发送必须一致；不应把视频/姿态 Stop 混用。 | `RecordingController.cs:1-36` |
| **F-016** Active scene wiring：折叠菜单、手册与运行时 UI 样式 | 点击折叠按钮；展开/隐藏 User Manual；动态创建窗口后观察样式。 | `CollapsibleCanvas` 使用 CanvasGroup，白名单保留窗口/按钮/input 等；`WindowToggle` 控制手册 CanvasGroup；`RuntimeUITheme.ApplySceneTheme` 在 AppManager 中被注释，`RuntimeUITheme.ApplyTo` 在新 UDP 窗实例化时调用。 | 折叠应隐藏 descendants 但保留 whitelist、交互和 snapshot；手册默认隐藏且不可 raycast；动态窗口继承圆角/阴影/按钮文字样式；不要假设全场景主题一定被应用。 | 折叠前后打开窗口/输入一次再展开，检查状态恢复；手册点击检查 raycast；新建 UDP 窗比较 prefab 与运行时样式；确认未调用注释掉的全局主题时 UI 仍可用。 | `FoldingCanvas.cs:1-380`；`User_Manual_control.cs:1-49`；`RuntimeUITheme.cs:1-620`；`UDPManager.cs:241-300` |
| **F-017** Runtime auto-created：Quest VR 数字键盘 | 用 Quest ray 选中 `IP Input` 或 `num_WebRTC Input`；按数字、点号、Back、Enter；点键盘外提交。 | 场景没有键盘对象；`UpperLimbVrKeyboardBootstrap` 每个场景加载后创建；只绑定精确命名的 TMP fields，关闭系统软键盘；键盘是 world-space Canvas + 有限 BoxCollider/RayInteractable，布局 3×5，前方约 1 m、眼睛下 `.15 m`，worldScale `.0012`。 | Enter/外部 Select 都显式 `onEndEdit`；数字键即时改 text，Back 删除末字符；合法 IP 才写 `cfg_ip`；键盘显示时场景其它可交互表面会提交并隐藏；失去焦点不应因 EventSystem 空选中而误隐藏。 | 在 Quest 真机逐字输入 IP/1..3；Enter 后重启确认 IP；非法 IP 不写 PlayerPrefs；点 WRM/Start/场景表面检查提交和隐藏；低头/转头检查键盘水平放置和不挡全场。 | `UpperLimb/UpperLimbVrKeyboard.cs:1-490`（bootstrap 约 `473-486`） |
| **F-018** Active + Runtime auto-created：追踪安全门 | 戴上/摘下 HMD、失焦、应用暂停、OVR 追踪丢失/恢复；观察状态和网络。 | `TeleopTrackingGuard` 订阅 `OVRManager` HMD mounted/unmounted、VR/input focus、TrackingLost/Acquired、OnApplicationPause；调用 AppManager。发送器统一读取 `CanSendTeleopData`。 | streaming 中追踪丢失时暂停控制，按配置要求 `NeedsRecalibration`；恢复后若需重校准只提示，成功校准才恢复；Stopping/TrackingLost 始终禁发；断开/停止清理 frame。 | streaming 后逐项触发 tracking/focus/pause，抓 UDP8001/8005 应立即无新控制包；恢复但未校准仍无包；重校准后恢复；Idle 时事件不应误显示 TrackingLost。 | `TeleopTrackingGuard.cs:1-120`；`AppManager.cs:262-310`；`DualControllerSender.cs:56-114`；`HandTrackingSender.cs:108-124` |
| **F-019** Active scene/XR building blocks：手部/手柄交互与移动 | 用 controller ray/poke/grab、hand pinch/poke、teleport/slide/turn 与场景 UI 交互；观察控制器/手部模型和射线。 | 两场景都含 Meta XR Building Blocks：左右 Controller/Hand Tracking、`Main Canvas` `PointableCanvas`、`TrackedDeviceRaycaster`、Ray/Poke/Grab/Teleport/Locomotion 组件；一侧 `ControllerTurnerInteractor`/`TeleportControllerInteractor` active、另一侧同名对象有 inactive 变体；`ControllerStepInteractor`、部分 glow/rail inactive。 | 保留 Quest 射线命中 Main Canvas、UI 事件路由、手柄/手部互斥可见性和现有 active/inactive 侧配置；键盘、窗口拖动依赖这些交互组件。 | 真机分别以 controller ray、hand pinch、poke 命中 Start/IP/窗口，确认事件一次且无重复；验证 teleport/slide/turn 不改变 teleop frame；切 tracking mode 检查模型/交互切换。 | 场景序列化 Building Block 对象；`UpperLimbVrKeyboard.cs`、`Canvas.prefab`；具体 Meta/XRI 脚本来自 Packages，不在 `Assets/Scripts` |
| **F-020** Active/legacy 边界：退出、Passthrough、调试 | 点击 Quit；若产品暴露 Passthrough 则切换真实透视层/跟随屏幕；观察 monitor debug。 | `AppController` 活动并只调用 `Application.Quit`；场景有活动 `[BuildingBlock] Passthrough`，但自定义 `PassthroughManager` (`PassthroughToggle`) 根对象 inactive 且 layer/screen/button 全部 null；`Controller`→`PythonTest` 仅是本地 CSV 调试，Monitor/Python UI inactive。 | Quit 行为保持；不要因为看到活动 Building Block 就自动启用有 null 引用的自定义 toggle；调试文字可选但不可成为网络依赖。 | 点击 Quit 用 development build 观察退出；不激活自定义 Passthrough 时无 NullReference；若启用需先补齐三引用并测试 toggle/屏幕跟随；监视 UI 开关前后不影响姿态发送。 | `AppController.cs`；`PassthroughManager.cs:1-55`；`Controller.cs`；`PythonTest.cs`；场景 `[BuildingBlock] Passthrough` 与 `PassthroughManager active=0` |
| **F-021** Unused legacy：虚拟机器人关节状态 | 外部发送 VRJS 关节状态时显示 URDF/Robotiq 虚拟机器人（若产品仍需要此观察模式）。 | `VirtualRobotJointStateReceiver` 监听 UDP8011，解析 `VRJS,dof,actual joints,command joints,gripper`，选择实际/命令关节；`VirtualUrdfJointDriver` 驱动 articulation xDrive，`Robotiq2F85MimicJointDriver` 做夹爪 mimic。两个目标场景无这些自定义组件引用。 | 作为可选观察/回归能力保留协议、关节顺序、actual/command 选择和夹爪关系；不应在默认场景隐式占用 8011 或创建 robot。 | 隔离场景发送已知 VRJS，比较各关节/夹爪；默认 V0.6/V0.7 检查无 listener；与姿态发送并行确认无端口/主线程冲突。 | `VirtualRobot/VirtualRobotJointStateReceiver.cs`；`VirtualUrdfJointDriver.cs`；`Robotiq2F85MimicJointDriver.cs` |

## 运行时自动创建清单

这些项目在目标场景 YAML 中找不到对应自定义组件，但实际运行会出现：

| ID | 创建触发与对象 | 当前场景下的结果 | 迁移要求/验收 |
|---|---|---|---|
| **R-001** | `AppManager.Start → EnsureControlModeManager`；找不到即新建 `TeleopControlModeManager`。 | V0.6/V0.7 都没有序列化 `TeleopControlModeManager`；`Awake` 进一步确保 `TeleopWorld`、`RobotBasePlacementTool`、`TeleopViewModeMapper` 和 tracking space 引用。 | 新场景即使只保留 AppManager 也要得到这些 singleton/工具；检查层级中对象只出现一次。 |
| **R-002** | `AppManager.Start → EnsureTrackingGuard`；找不到即把 `TeleopTrackingGuard` 加到 AppManager GO。 | 两场景无序列化 tracking guard，但安全暂停仍实际生效。 | 不能只迁移场景组件；启动后检查 guard 事件订阅和销毁退订。 |
| **R-003** | `UpperLimbAkmBootstrap.AfterSceneLoad`。 | V0.7 已有 `UpperLimbAKM`/UI，bootstrap 幂等跳过；V0.6 会新建 manager 和 UI。V0.6 UI 找不到 `WRM_enable Button`/calibration/status 子对象，只记录 missing/部分 UI 不工作，但 manager 仍会请求 body tracking、校准和发送（默认 Wrm off）。 | V0.6 兼容模式须允许“无 UI 的后台 manager”，不能强行创建第二个 manager；V0.7 必须把 scene UI 接到同一 manager。 |
| **R-004** | `UpperLimbVrKeyboardBootstrap.AfterSceneLoad`。 | 两场景都没有键盘根；只要找到精确命名 TMP field 就创建；找不到则记录 disabled。 | 保留命名发现和键盘外提交语义；场景重载/域重载后不能留下重复键盘。 |
| **R-005** | `TCPPoseReceiver.OnEnable → EnsureTcpObjects`。 | 当前 `RightTCP` 已序列化；若缺失会在 `Calibration` 下自动创建 `LeftTCP`/`RightTCP`，默认可继续接收。 | 新场景删除 TCP 子物体仍应可启动；验收自动创建、引用和 parent。 |
| **R-006** | `ForceArrow.Awake`、`ForceSensorReceiver.EnsureArrow`。 | `RightForce` 组件活动，运行时构建 marker/head/shaft；inactive fallback 启用时可补箭头。 | 不要把运行时生成的 child 当成场景缺失而删掉；验收零力隐藏、非零力建可视几何。 |
| **R-007** | View/reposition 时由 `TeleopWorld`、`RobotBasePlacementTool` 创建 anchor/axis/preview。 | `TeleopWorld` 默认 `createAnchorOnStart=false`，anchor 直到 View/校准/摆放路径才出现；axes/preview 是 runtime primitives，collider disabled。 | 迁移时保留 anchor 的创建时机、坐标轴方向和 placement reset 抑制；不要在启动自动改变 Mirror 结果。 |

## Unused legacy 与历史资源索引

“Unused”表示当前 V0.6/V0.7 没有活动路径，不表示可以直接删除；外部设备、旧录制或历史场景可能仍依赖它们。

| 资源/脚本 | 证据与历史效果 | 处置建议 |
|---|---|---|
| `Assets/Scripts/Position_transfer.cs` | 当前 `SendPosition` 场景对象 inactive；脚本是空壳，注释说明行为已拆到 `DualControllerSender`、`RecordingController`、`AppController`，保留它是为了旧 scene reference。 | 新架构可不执行，但保留 class/兼容 GUID 或迁移旧引用，避免场景反序列化 missing script。 |
| `Assets/Scripts/Unused/A_button_laser_control.cs` / `X_button_laser_control.cs` | 旧的 A/X 按键按下/松开分别禁用/恢复 ray interactor；目标场景无组件引用。 | 若产品要保留旧“按键临时收 ray”手势，应作为显式可选模式回归；不要默认与 AKM A/X 校准复用。 |
| `Assets/Scripts/Unused/WindowJoystickHoldMove.cs` | 旧窗口悬停后 RT index trigger 抓取，右摇杆移动，A 1.8×加速，parent 边界 clamp；当前 prefab 使用的是 `DraggableWindowWorld`，无此脚本引用。 | 保留为历史交互规格；默认新窗口继续使用 pointer drag，除非用户要求 controller joystick 兼容。 |
| `Assets/Scripts/Unused/CanvasMovement.cs` | 旧 `IMovementProvider` 仅输出 XY、固定 Z、identity rotation；无当前引用。 | 不作为当前控制协议，但如迁移旧 Canvas 交互需保留接口。 |
| `Assets/Scripts/Unused/Position_transfer_Multi.cs` | 全部注释的旧多窗口/文字拼包发送器，含退出、camera 切换、位置发送；无当前引用。 | 不误接回 8005；把当前 UDP/WebRTC/record API 作为唯一活动路径。 |
| `Assets/Scripts/MultipleVideoStream/UDPSocket_MULTI.cs` | 活动 class `UdpSocketMulti` 是旧单 UDP JPEG receiver；当前 `Canvas.prefab` 明确使用 `UdpSocketMultiHD`，scene manager 预制体 GUID 也指向 Canvas。 | 保留源码供旧场景，但不要与 HD receiver 同挂；可隔离到 legacy assembly。 |
| `Assets/Scripts/Tactile Visualization/*.cs`、`TensorPrefab.prefab`、`TactileMesh.prefab`、`Arrow1/Arrow2/Dot.prefab` | V0.3/V0.4/V0.5 与 backup scene 有 TactileMesh/触觉对象；V0.6 Editor 配置会删除触觉根、按钮/IP，关闭 `receiveTactile`。`TensorPrefab` 挂 `TactileArrow`，`maxForce=.08`。 | 若重构承诺历史 tactile 演示，按 F-014 隔离恢复；默认场景继续让 8012 归 TCP+力，禁止第二 listener。 |
| `Assets/Scripts/PassthroughManager.cs` | `PassthroughToggle` 要求 `OVRPassthroughLayer`、screen、button 三引用；目标自定义对象 inactive 且 refs=0，直接激活会在 Start NullReference。 | 仅在补齐接线并专门验收后启用；活动 Meta Building Block 与自定义 toggle 是两条不同路径。 |
| `Assets/Scripts/VirtualRobot/*.cs` | 文件存在，但 `rg` 未发现 V0.6/V0.7 自定义组件引用；协议/driver 可供独立 URDF 场景。 | 作为可选观察模式，不放入默认运行时。 |
| `Assets/Scripts/Viz/RobotTcpAxes.cs` | Editor 工具注释称会在 `RightTCP` 添加该组件；但检查的 V0.6/V0.7 scene 没有其 GUID `acedd6afebe89583ba984cc0dcf8ac05`，因此没有证据表明轴组件已保存到目标场景。 | 迁移时以实际场景为准：若要保留 TCP 三轴，显式重新接入并验收，不能把 Editor 注释当成 active wiring。 |
| `Assets/Editor/TactARSceneSetup.cs` | 仅 Editor 菜单/批处理方法；会删除 tactile 对象、将 fallback Force receiver 设 8013、创建 Calibration/Coord/RightTCP/RightForce/TCPPose 接线。 | 不在运行时调用；把它当作一次性场景转换脚本和事实来源，重构场景需手动复核其结果。 |
| `RuntimeUITheme.ApplySceneTheme` | 方法完整存在，但 `AppManager.Start` 调用被注释；只有 `UdpWindowManager.CreateUdpWindow` 对新窗调用 `RuntimeUITheme.ApplyTo`。 | 不要预期启动后整个旧 Canvas 自动重皮肤；若新 UI依赖该效果，要显式调用并回归布局。 |

## 配置、预制体与协议要点

### `TeleopConfigUpperLimb.asset` 与实际运行默认值

`Assets/Settings/TeleopConfigUpperLimb.asset` 序列化为：`defaultServerIP=192.0.2.16`、pose 8001、control 8005、virtual robot 8011、tactile 8012、signaling 8765、UDP video base 8000、默认 WebRTC track 1、send rate 30、binary=1、三个 safety bool 均为 1。

两个目标场景的 `AppManager.teleopConfig` 都是 `{fileID: 0}`。因此 `AppManager.ApplyConfigDefaults()` 不会读取上述 asset，实际初始 IP 是代码字段的 `192.0.2.11`，其余端口仍按代码字段 8001/8005/8012/8765/8000；`PreventSleep`、tracking pause、recalibrate 也因为 null config 走 `true`。这是行为事实，不是 asset 的推断。

`PlayerPrefs` 加载发生在 `AppManager.Start` 的 `LoadConfig()`，所以旧设备上已有 `cfg_ip`、mode、rate、binary、track count 会覆盖这些初始值。迁移必须决定是接入 asset 并保持旧 PlayerPrefs 优先，还是显式兼容现有 hard-code fallback。

### `Canvas.prefab`（GUID `14d06f2d739b37549abe5c5d3e780fe8`）

这是 `UdpWindowManager.udpWindowPrefab` 实际指向的预制体，不是单纯 RawImage：

- 根 `Canvas` 上挂 `VideoWindowController`（脚本 GUID `e7d67a6de85fee54690a2da32347c7a9`），接 close、port input、`ZOOM Button`、resolution label、status；
- 根另有 disabled 的旧 `DraggableWindow`，但活动 `RawImage` 挂 `DraggableWindowWorld`，target 是根 Canvas；
- `Plane` 子物体挂 `UdpSocketMultiHD`，并有 mesh/renderer；接收 IP/port 在运行时 `Initialize`；
- 有 world-space Canvas、GraphicRaycaster、TrackedDeviceRaycaster、BoxCollider/Rigidbody 交互基础；输入默认文字 `PORT`；resolution label 默认 `x1.0`。

HD receiver 需要保留 12-byte 大端分片头、最多 8 个 frame buffer、重复 chunk 忽略、完成帧回主线程。它没有场景化端口，完全依赖 `VideoWindowController.Initialize`。

### 通道分工（当前目标场景）

| 通道 | 当前发送/接收 | 关键格式/行为 |
|---|---|---|
| UDP 8001 | Controller `C` 或 Hand `H/HB` 姿态发送 | 发送受 streaming/tracking safety gate；手柄 100 Hz，手部场景 60 Hz。 |
| UDP 8003 | `RecordingController` | ASCII `Start`/`Stop`。 |
| UDP 8005 | AppManager control message、V0.7 WRM JSON；旧 UDP window resolution/focus 只有 Focus label 非空才发送 | 必须做消息所有权/去重；当前 scene Focus refs null，所以旧 resolution loop 不运行。 |
| UDP 8012 | `TCPPoseReceiver` 主 TCP+force JSON | 右侧接线，比例 `.01`；不可同时启用旧 tactile listener。 |
| UDP 8013 | inactive `ForceSensorReceiver` fallback | binary 6D/JSON；仅备份。 |
| UDP 8000 + 2i | 动态 `UdpSocketMultiHD` 窗口 | i=0..4，JPEG/PNG 单包或分片。 |
| WebSocket 8765 | `VideoSignalingClient` | `ws://IP:8765`，hello/start_video/offer/answer/ICE/stop_video。 |
| UDP 8011 | 仅 VirtualRobot legacy | `VRJS` 关节状态；目标场景没有 listener。 |

## 迁移优先级与主要风险

1. **配置资产未接入是首要风险。** 场景 `teleopConfig=0`，运行 IP 是 `192.0.2.11`，而同目录 asset 写的是 `192.0.2.16`。如果新 AppManager 直接绑定 asset，首次运行目标会变；如果只迁移 scene wiring，asset 的安全/端口字段永远不起作用。需定义明确优先级：旧 PlayerPrefs、asset、还是代码 fallback，并对旧设备做一次升级验收。
2. **8012/8005 的端口所有权必须显式设计。** 当前 8012 由活动 `TCPPoseReceiver` 使用，`ForceSensorReceiver` 8013 inactive，VideoWindowManager tactile 接收关闭；8005 同时是 AppManager 控制、V0.7 WRM、旧窗口 resolution 的潜在通道。恢复 Focus UI 或 WRM 后若无仲裁，会产生解析冲突/丢包/重复状态。
3. **安全 gate 是发送器的共同前置条件。** `DualControllerSender` 与 `HandTrackingSender` 都直接依赖 `AppManager.CanSendTeleopData`；tracking lost、recalibration、Stopping 任一状态都应禁发。新 UI 的 Start/状态实现不能只改文字而绕过 `NeedsRecalibration`。
   `UpperLimbAkmManager` 是已验证的例外：其 WRM loop 只检查 `WrmEnabled`，所以“统一安全 gate”若要改变此行为必须在迁移决策和回归中明确记录。
4. **场景接线来源不止字段引用。** 控制模式、录制、退出、IP、Add UDP、重定位等有持久 `UnityEvent`；同时 AppManager 只对非空字段追加 listener。尤其 `controlModeButton=0` 但场景事件仍有效，迁移 UI 时容易“看起来有按钮、实际没有调用”或重复调用。
5. **V0.6/V0.7 的 WRM 生命周期不同。** V0.7 有序列化 manager/UI/按钮；V0.6 依赖 bootstrap，只能后台运行且 UI 找不到对象。新场景要避免 bootstrap 与场景对象各创建一份，Hands 模式下还要同步关闭 WRM。
6. **坐标变换不能用直觉重写。** Mirror 的自动 yaw/manual 6DoF、View 的 robot-base 转换分别有平移/四元数轴交换和符号约定；TCP/force 位于 Calibration 子树，校准时会随父节点移动。必须以固定姿态 packet golden values 验证，不能只做视觉对齐。
7. **时序与线程语义是协议的一部分。** Controller 100 Hz 的 catch-up 限制、Hand 场景 60 Hz 文本协议、WRM 30 Hz JSON、UDP frame 分片在后台线程接收且主线程贴图，都会影响机器人端和 Quest 性能。新 UI/会话层不应把它们统一成一个未经验证的 Update 频率。
8. **WebRTC 开关会销毁 UDP 窗口。** 开启 WebRTC 会 `CloseAllWindows()` 并禁用 Add UDP；禁用才恢复创建。保留视频失败“不影响 teleop”与窗口互斥这两条副作用，否则用户可能看见画面但姿态已经意外停止。
9. **UDP 窗口的实际交互来自 prefab。** `VideoWindowController`、活动 `DraggableWindowWorld`、HD receiver 和 `ZOOM Button` 都不在目标 scene 的自定义组件表中，单迁 scene 会漏掉它们；同时 prefab 还含 disabled 的旧 `DraggableWindow`，不能错误启用两种拖动器。
10. **焦点/缩放功能有“本地有效、网络未接线”的状态。** 当前每窗 zoom 标签会循环，但 FocusModeButton/Label 为 null，旧 resolution message 不发。若新 UI 把 Focus 接回来，需要单独回归右 A 与 WRM calibrate A 冲突以及 8005 消息协议。
11. **TCP 力显示与历史触觉显示不是同一输入。** 当前主路径是 8012 组合 TCP+force JSON；历史 41 点 tactile 也曾使用 8012，Editor 工具明确将其关闭。新实现若为了“保留触觉”重新启 listener，会直接破坏当前 RightTCP/RightForce。
12. **`RobotTcpAxes` 只有 Editor 意图，没有场景证据。** `TactARSceneSetup` 注释说会添加三轴，但两份目标 scene 没有该脚本 GUID。迁移验收应把“是否有 TCP 红/绿/蓝轴”作为显式产品选择，而不是假设 active。
13. **静态/运行时状态不随场景天然持久。** `TeleopReferenceFrame`、`TeleopControlModeManager.CurrentMode`、robot-base anchor 和 PlayerPrefs 组合会受 scene reload/domain reload 影响；停止 streaming 会清 frame，重启仍会读配置。需要在新生命周期中验证重载、停止、tracking lost 三种清理顺序。
14. **VR 键盘是输入提交协议，不只是视觉组件。** 它故意绕开 Quest 系统键盘和 TMP 编辑态，Enter/点击外部才显式 `onEndEdit`；若用普通 `ActivateInputField` 替换，IP 保存和 Quest 交互会回归失败。
15. **录制状态从文字推断。** 外部录制器只收到 `Start`/`Stop`，按钮文字是唯一状态源；任何新 UI 文案、翻译或初始值变化都会改变下一次命令。重构可在内部增加布尔状态，但要为旧 scene event 保留 `Recording()` 和精确命令。
16. **inactive 自定义 Passthrough 不能被误激活。** 场景的 Meta Passthrough Building Block 活动，并不意味着 `PassthroughToggle` 已接好；后者 refs 全空，直接启用会异常。需把它作为显式 opt-in。

## 最小端到端回归顺序

在新 app 宣布替代 classic 前，建议按以下顺序跑一次 V0.6 和 V0.7 等价场景：启动并确认无重复 runtime objects；输入 IP/1..3 路并重启；Controllers 模式抓 C 包；Hands 模式抓 H 包；开始/停止 streaming；触发 tracking lost→恢复→重校准；Mirror/View 与两阶段底座摆放；V0.7 WRM 两段校准和 8005 JSON；8012 TCP/force JSON；创建、拖动、zoom、B/Y 布局、关闭 UDP 窗；WebRTC 1/2/3 路开关及失败降级；录制 Start/Stop；折叠菜单、手册、VR 键盘；最后再检查 Quit 和可选 legacy fallback。每一步都应同时看 UI 状态、端口抓包和 Quest 主线程是否有异常。
