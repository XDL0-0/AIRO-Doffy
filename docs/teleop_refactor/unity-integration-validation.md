# Unity 集成验收记录

日期：2026-09-16。新工程 `/home/yuyuan/UNITY_Project/Codex`，参考 `/home/yuyuan/UNITY_Project/classic`。源代码/API 编译使用参考工程缓存的 Unity 6000.5.6f1、Meta XR 205.0.0、WebRTC 3.0.0 程序集；没有绕过 Editor 授权。

## 已取得的证据

| 验证层 | 结果 | 边界 |
|---|---|---|
| 运行时 C# 源码/API | 82 个源文件编译通过，0 错误、0 警告 | Roslyn + 已有 SDK 程序集，不运行 Unity native API |
| Editor 工具源码/API | 1 个构建/场景检查文件编译通过 | 不等于实际 Editor 导入或 Android 打包 |
| 静态资产引用 | 846 个脚本引用、362 个本地 GUID；0 缺失/重复/孤立引用错误 | 场景/预制体序列化审计，不等于启动时无异常 |
| 纯 C# 会话协调器 | Core 测试 + 25 项异步行为检查通过 | 实际生产协调器，底层服务测试适配器 |
| 纯 C# 协议/数学/生命周期 | 47 项通过 | 生产 JPEG/TCP/wire/height mapping/socket 源码；不包含 Meta 采样 |
| 追踪 guard | 10 项事件/生命周期检查通过 | 实际 guard 源码，SDK 事件由测试适配器触发 |
| 控制器跨语言 UDP | C# fr-FR 格式化→Python 生产解析通过 | 31 字段、轴值、按钮、frame；没有发送到机器人 |
| TCP/力跨语言 UDP | Python 生产封装→C# 生产解析通过 | 已知坐标/力变换与 wxyz 顺序一致；不含 Unity Transform 渲染 |
| 真实 D435 图像→C# | 三帧 640×480 采集通过；最后一帧经生产 UDP 分片→生产 C# 重组后 JPEG SHA256 一致 | 本机真实相机、真实 UDP、Mono 执行 C#；非 Quest 无线链路 |
| Python WebRTC | aiortc 实际协商、视频解码与消息往返通过 | localhost；非 Unity native peer |
| C# WebSocket | 实际 ClientWebSocket↔aiohttp 通过 | 64 并发发送、分片中文/emoji、重连、1 MiB 上限；非 native WebRTC |

摄像头原始结果见 [camera-interop-result.json](camera-interop-result.json)，信令实现与复现见 [WebRTC 验收](unity-webrtc-validation.md)。纯 Core 与异步会话验证见 [Core 验收](unity-core-validation.md)。

## Editor/头显实际检查

执行了真正的 Editor 场景验收命令：

```bash
/home/yuyuan/Unity/Hub/Editor/6000.5.6f1/Editor/Unity \
  -batchmode -nographics -quit \
  -projectPath /home/yuyuan/UNITY_Project/Codex \
  -executeMethod Doffy.Editor.TeleopBuild.ValidateScene \
  -logFile /tmp/doffy-unity-final-validation.log
```

结果：退出码 **198**。日志 `No valid Unity Editor license found. Please activate your license.`；Editor 未执行场景验收。ADB `devices -l` 只返回表头，没有连接设备。由此 **Editor 导入、APK 构建、Quest 运行和端到端机器人验收仍未完成**。

这与安全审批无关，不需要再次批准代码修改。继续这部分需要本机有效 Unity Editor 授权及可连接的 Quest。

## 真机验收步骤

1. 激活有效 Editor 授权，打开项目等待包导入，运行 `Tools/DOFFY/Validate scene`；Play 中检查新工作台和唯一的 runtime managers，没有 missing script/NullReference。
2. `Build Quest APK`，安装新 APK；确认 hand/body tracking 权限、透视、controller ray、hand pinch/poke、键盘、拖动与 Compact view。SVG 预览不能替代该步骤。
3. PC 与 Quest 同网：UDP 1→5 窗，WebRTC 1/2/3 路，连续切换至少 20 次；停止/重启 PC，确认视频恢复、旧端口释放且无旧帧覆盖新会话。
4. 校验 IP/路数持久化；非法地址不改变目标；活动会话换 host 先停录并要求重新校准。分别验证开始/停止、录制 Start/Stop/Undo 的 PC 收包。
5. 以测试接收端抓 C/H/HB/WRM 数据，验证采样频率、字段、按钮和坐标；先不连接会发运动指令的机器人循环。
6. 发送已知 TCP/力数据确认位置、方向、零力隐藏、数据过期提示；执行 Mirror/View 和手动校准，确认 Confirm & resume 保留手动参考系。
7. 逐项模拟摘头显、失焦、暂停、追踪丢失，确认姿态与 WRM 停发；恢复及视频连接成功不能自行继续控制，只有明确校准/确认后恢复。
8. 使用准备好的真实机器人进行最终控制、停止、上肢映射、相机延迟与录制回放验收；记录设备、网络、帧率/延迟及结果。

当前源码完整性与本机传输证据已具备，发布状态仍是 **待 Editor/Quest 验收的重构版本**。
