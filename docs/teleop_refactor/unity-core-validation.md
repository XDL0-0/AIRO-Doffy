# Unity 会话核心验收

新工程：`/home/yuyuan/UNITY_Project/Codex`。Classic 保持只读。原脚本移至 `Assets/Teleop` 时保留 `.meta` GUID；迁移清单见 [unity-source-migration.json](unity-source-migration.json)。

## 实现

`Core/AppManager` 保留旧场景序列化字段及公开调用，承担场景装配。独立模块分别负责运行配置、IPv4/路数验证、旧 PlayerPrefs 读写、状态机、会话生命周期、视频生命周期和旧 UI 兼容绑定。新工作台读取公开状态，录制和会话逻辑不再读取按钮文案。

新增工作台接口包括 `ConfigureConnection(host, tracks)`、`SetTrackingMode`、`SetWebRTCTrackCount`、`SetDebugVisible`、`StatusText`、`LastError`、`Changed` 和 `ConfirmAlignment`。

- 8001/8003/8005、旧 PlayerPrefs keys 和现有文本/二进制协议保持。活动会话更换 host 先向旧目标停录，再更新所有网络消费者并关闭控制门，等待重新校准。
- 会话与视频分开管理。过期的异步启动、视频连接成功/失败、信令恢复均不能清除追踪暂停。停止幂等，清理参考系、录制、视频资源。
- `CanSendTeleopData` 同时检查会话状态、独立 HMD/focus/tracking/pause flags 和坐标编辑状态。开始/重校准也检查实际 guard 可用性，既有 pause=false 配置不能绕过追踪暂停。
- Controllers/Hands 互斥。切换输入或控制参考系会暂停；Hands 会关闭 WRM；WRM 与 C/H/HB 统一受发送门限制。
- 手动坐标编辑与两阶段底座摆放互斥。Alignment 页的 **Confirm & resume** 接受当前手动参考系，不用自动头部 yaw 覆盖。**Recalibrate** 仍执行明确的自动校准。会话启动保留已设置的手动参考系；自动校准仍可明确覆盖。底座工具默认 Locked，仅显式 Place robot base 后进入编辑，提供 Cancel base placement，避免保存为 View 后被隐式编辑锁住启动。
- `RecordingController` 显式保存请求状态，发送原 Start/Stop/Undo 字符串；没有 PC ACK，因此 UI 状态不表示远端确认。

## 实际 C# 测试

```bash
python3 scripts/teleop_refactor/run_csharp_session_checks.py
python3 scripts/teleop_refactor/run_csharp_protocol_checks.py
```

结果：

```text
PASS TeleopCoreTests.RunAll + 25 session coordinator behavior checks
PASS 10 tracking event/lifecycle checks (SDK event adapters)
PASS 47 C# protocol/lifecycle checks
```

`TeleopCoreTests.RunAll` 覆盖非法输入、旧偏好持久化、视频与追踪状态互不覆盖。25 项协调器检查编译并执行生产 `TeleopSessionCoordinator` / `TeleopVideoSessionCoordinator`，使用 TaskCompletionSource 控制真实异步时序：启动中停止、立即替换启动、迟到视频结果、追踪暂停、校准失败、停录/参考系清理、Stop 幂等。测试适配器只代替底层 video/record/frame 服务。

10 项 guard 检查执行生产 `TeleopTrackingGuard`，SDK 事件以适配器触发，验证独立丢失原因、事件订阅/退订，以及组件订阅前已存在的失焦状态。

最终源码/API 检查为 **82 个 runtime 源文件 + 1 个 Editor 工具文件，0 错误、0 警告**；静态 **846 个脚本引用，0 错误**。这些是源码与本机行为证据，不能替代 Unity Editor native 运行。UI、Meta 采样、手动参考系实际姿态、Quest 网络和 Android 构建仍需按 [集成验收](unity-integration-validation.md) 完成。
