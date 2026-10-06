# Classic 功能迁移与验收矩阵

基线是 [Classic 只读盘点](classic-feature-inventory.md)，不是旧 Codex 的功能推测。新工程位于 `/home/yuyuan/UNITY_Project/Codex`；主场景 `Assets/Scenes/Teleoperation.unity`。所有下表代码目录相对 `Assets/Teleop`。`保留` 表示源码/场景路径存在，**不等于已经通过 Quest 真机验收**。

| 编号 | 迁移结果 / 新入口 | 证据与尚待验收 |
|---|---|---|
| F-001 会话 | `Core` 状态机与协调器；常驻 Start/Stop session | 状态/取消/视频失败隔离回归；真机停止后抓包待验 |
| F-002 配置/IP/UDP | `Core` settings/store/validation；`Networking/UdpSocket`；Session 页 | IPv4 验证、旧 PlayerPrefs keys、端口兼容；跨语言真实 UDP 已测 |
| F-003 双手柄 | `Input/DualControllerSender`；100 Hz、31 字段 C 协议保留 | 法语区域设置 C# 格式化→Python 实际解析通过；OVR 采样频率、实体按钮待真机 |
| F-004 手部 | `Input/HandTrackingSender`；H/HB、26 bones、追踪过滤保留 | 字节序/浮点及 PC 手部协议测试；实际骨骼采样待真机 |
| F-005 输入模式 | `Input/TrackingModeManager`；Session 页两种模式 | 持久化、sender 互斥；Hands 关闭 WRM；切换后重校准 |
| F-006 参考系校准 | `Calibration`；Alignment 页与原 A+X/扳机快捷键 | 原坐标实现保留、PC 坐标回归通过；Quest gizmo/姿态对照待验 |
| F-007 Mirror/View/底座 | `Calibration`；Session/Alignment 页 | 原双阶段摆放、轴向与 reset 抑制保留；真机拖放/坐标对照待验 |
| F-008 WRM/AKM | `UpperLimb`；Upper limb 页、原按住 A 校准 | 两段校准、30 Hz、confidence/握把/recenter 协议保留；高度映射端点/滤波/失踪保持已测；Meta body tracking 待真机 |
| F-009 TCP/力 | `Feedback/TCPPoseReceiver` + `ForceArrow`；Display 页 | Python 实际 TCP 打包→C# UDP/解析通过；wxyz、有限值/长度检查；主线程显示及箭头方向待真机 |
| F-010 旧力接收器 | `Feedback/ForceSensorReceiver`，默认 inactive、8013 | 源码与旧场景引用保留；独立传感器二进制/JSON 与视觉待硬件 |
| F-011 WebRTC | `Media/WebRTC`；Cameras 页 1/2/3 路 | Python aiortc 实际协商和视频解码；C# SDK API 编译；Quest native peer/纹理待验，见 WebRTC 报告 |
| F-012 UDP 视频窗 | `Media/UDP`；Cameras 页 Add/Close/Arrange、每窗端口/缩放/拖动 | 五窗及端口分配保留；乱序/重复/超时/重启解析已测；真实 D435→生产 C# 重组后 JPEG 一致；Quest 纹理和交互待验 |
| F-013 焦点/缩放 | 每窗本地 UV 缩放；WebRTC DataChannel 分辨率控制 | 旧 8005 resolution/focus 仅在显式绑定旧 UI 且 WRM 关闭时发送，避免共用通道冲突 |
| F-014 41 点触觉 | `Visualization/Tactile` + `Networking/TactileArrayStream` | 原 41 点布局、箭头、TensorPrefab 保留；V0.7 基线本来未启用。需旧场景接线及独立端口，默认 UI 显示不可用；不占用当前 TCP 的 8012 |
| F-015 录制 | `Core/RecordingController`；常驻录制按钮、Help 页 Undo | 显式状态替代读取文案；仍发 8003 Start/Stop/Undo；Stop/换 host 先向旧目标停录；无 ACK 协议，不显示“PC 已确认” |
| F-016 折叠/手册/样式 | `UI/WorkspaceShell`；Compact view、Help 页 | 常驻会话操作；动态摄像头窗共用主题；旧折叠/手册类在 `Legacy/Presentation` 保留 |
| F-017 数字键盘 | `UI/WorkspaceKeypad`；IP 与摄像头端口输入 | 大按钮、退格、提交/关闭；旧键盘类保留；Quest ray/poke 输入待验 |
| F-018 追踪安全 | `Input/TeleopTrackingGuard` + Core send gate | 10 项真实 guard 事件/生命周期测试（SDK 事件适配器）；多个丢失信号独立锁存、初始化读取当前 SDK 状态；恢复不自动解除重校准 |
| F-019 XR 交互/移动 | 原 Meta rig/building blocks；新 Canvas 的 Pointable/Ray/Poke | 原左右 active/inactive 配置保留；新 UI SDK205 API 编译；手柄、pinch、poke、移动手势待真机 |
| F-020 退出/透视/调试 | Help/Display 页；`Diagnostics` | 透视操作接现有真实 layer；退出 API 保留；旧自定义透视空引用组件保持 inactive |
| F-021 虚拟机器人 | `Visualization/VirtualRobot` | 原 VRJS、关节顺序、actual/command、Robotiq mimic 保留；基线无活动实例，默认不占 8011；URDF 场景待验 |

| 自动创建项 | 新工程处理 |
|---|---|
| R-001 控制模式/World/工具 | 原幂等创建链保留，Core 组合根调用 |
| R-002 TrackingGuard | Core 启动时确保单实例并初始化 |
| R-003 Upper limb | 原 bootstrap 保留，场景已有组件则复用 |
| R-004 VR 键盘 | 新 workspace 按需创建；旧命名发现键盘保留给兼容 UI |
| R-005 TCP 对象 | Receiver 自动补 TCP 子物体保留 |
| R-006 力箭头 | ForceArrow 自动建 marker/head/shaft 保留 |
| R-007 anchor/axis/preview | 按 View/校准/摆放路径创建，启动时不强制改变 Mirror 参考系 |

## 有意调整的行为

- 旧 Codex 已先备份再删除。新工程保留 Classic 的 XR rig、有效预制体和脚本 GUID，重建应用层与 UI。
- WRM 与其它控制统一受会话/追踪安全门限制；仅开启 WRM、尚未 Start session 时不发控制包。视频恢复、host/输入模式/参考系调整不会绕过重校准要求。
- UDP 缩放现在实际改变显示 UV；旧场景仅改变标签的行为得到补齐。重启 PC 后视频帧计数归零可恢复接收；陈旧 TCP 反馈隐藏力箭头并显示 STALE。
- 录制按钮依赖显式状态，结束会话自动停止录制；Undo 增加第二次点击确认。协议字符串不变。
- 手动调整后的 Confirm & resume 和随后 Start session 均保留手动参考系；Recalibrate 明确切回自动参考系。底座编辑改为显式进入，可用 Cancel base placement 退出。
- 默认配置显式绑定到场景，保持 Classic 实际默认 IP `10.10.131.72`、手部 60 Hz 文本协议、旧偏好优先。
- 历史失效 URP/XR/触觉资源已归档到备份目录，清理前后记录在 [引用清理清单](inherited-reference-cleanup.json)。当前 Classic 活动功能与有效触觉资源保留。

UI SVG 是布局预览，不能作为 Unity 实际渲染或 Quest 交互验收证据。最终硬件验收步骤见 [Unity 集成验收](unity-integration-validation.md)。
