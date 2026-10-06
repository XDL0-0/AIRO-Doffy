# Teleoperation rebuild — 2026-09-16

## 已完成的源码工作

- Classic/v1/v2 盘点完成：21 项功能、7 项运行时自动创建机制；当前场景与历史可选能力分别记录。
- 旧 Codex 先归档、验证归档后删除；从 Classic 的有效场景/rig/资产重建。参考工程保持只读。
- Meta XR All-in-One 固定最新稳定 205.0.0，保留其规定的 Audio/Voice 依赖；Android ARM64/IL2CPP 构建入口已提供。
- 重建六页空间工作台、数字键盘、常驻会话/录制操作；保留视频窗口、力/TCP、校准、上肢和原 XR rig。提供明确标注的 SVG 设计预览。
- 52 个原自定义脚本保留 GUID 迁移到功能目录；新增协议、网络、Core、媒体和 UI 模块；清理继承的失效资产并单独归档。
- PC 按 protocol/media/control/robots/runtime 拆分，根 CLI/import 与替换点保留。机器人约束、状态/超时、CAN-FD 所有权及录制 schema 保持。
- 最后 UI 审查修复：启动保留显式手动参考系，底座编辑默认锁定且提供取消入口；再次通过源码/API 编译和会话回归。
- 修复 locale 数字格式、UDP 分片/PC 重启恢复、陈旧反馈、异步视频取消/回调、DataChannel 初始消息，以及追踪与参考系编辑的发送门。

## 验证结果

- 全仓库：698 passed、95 subtests passed、1 failed。唯一失败是重构前已存在的策略 ACTION_STEPS 配置 4 / 测试期望 8；未改动无关策略配置。
- C#：82 runtime + 1 Editor 文件 API 编译，0 错误/警告；846 脚本引用静态审计，0 错误。
- 生产纯 C#：47 协议/数学/生命周期、10 追踪事件、25 异步会话行为检查通过；Core 设置/状态测试通过。
- 实际 C# ClientWebSocket↔Python aiohttp、Python aiortc WebRTC 回环、双向控制器/TCP UDP 均通过。
- 真实 D435 三帧采集及最后一帧传到生产 C# JPEG 重组器通过，640×480、38 分片、JPEG SHA256 一致。

## 未完成的硬件验收

真正 Unity Editor 验收返回 exit 198：没有有效 Editor license；ADB 无连接 Quest。尚未完成 Editor 导入/Play、APK 构建、Quest 纹理/交互/无线延迟及机器人端到端验收。源码编译和本机传输证据不替代这些验收，也未标记发布通过。

待本机有效 Unity 授权和 Quest 可用后，按 `unity-integration-validation.md` 执行。

## 备份

- `/home/yuyuan/UNITY_Project/.backups/codex-before-refactor-20260916.tar.gz`
- `/home/yuyuan/UNITY_Project/.backups/pc-teleop-baseline-20260916.tar.gz`
- `/home/yuyuan/UNITY_Project/.backups/codex-inherited-stale-assets-20260916`

PC 基线来自任务开始时的工作树，而非 git HEAD；保留用户已有实验、评估、数据和论文修改。未启动机器人控制或策略训练任务。
