# Phase 15 文档与发布准备报告

日期：2026-07-30

分支：`codex/v2-refactor`

主要提交：

- `7fad5c0`：README 改为 v2-first
- `221ed14`：架构与通信文档
- `64675c2`：扩展与迁移指南
- `4a06979`：CHANGELOG 与 release candidate 清单

## 结论

Phase 15 的文档与发布准备已完成。仓库现在从 v2 包、显式 lifecycle、分层
配置和 composition factory 出发说明使用方式，不再把旧 `main.py`、
`config.py` 或耦合相机管理器当成新架构。

没有创建 release candidate 或最终 v2.0 标签。这是有意的安全结论，而不是
遗漏：当前环境没有运行 Ruff/Pyright/pytest/pre-commit，缺少 `h5py`，
也没有 UR、RealMan、RealSense、Quest、BLE4、Unity 和网络端到端证据。
`docs/release_checklist.md` 将这些项保留为明确阻塞条件。

## 15.1 README

README 现在包含：

- 当前 `2.0.0.dev0` 状态和真机验证边界；
- v2 能力与依赖方向；
- base 安装和可选 extras；
- 五层配置优先级与无默认设备地址政策；
- teleop/collect CLI 和显式 session factory；
- UR、RealMan、VR、相机/视频、触觉/wrench、recording 支持；
- state 与 command 通道分离；
- policy 兼容状态、诊断、测试和仓库布局；
- compatibility wrapper 弃用政策。

文档明确根入口仍为迁移兼容路径，不能作为新 composition 示例。

## 15.2 架构文档

`docs/architecture.md` 记录：

- 模块职责与禁止职责；
- dependency direction 和运行时数据流；
- immutable sample、clock domain 与 sequence 约定；
- 每个连续 loop 的输入、输出、频率、资源、buffer 和 drop policy；
- lifecycle 启动、部分失败回滚、逆序关闭和 worker join；
- error model、watchdog/HOLD/STOP 行为；
- composition boundary 和 extension points；
- executor-owned/cache state source 的生产约束；
- 当前 release 限制。

特别没有掩盖一个关键组合风险：thread-affine SDK 不能由 session 和 executor
并发读取/写入，部署必须注入 owner-thread cache 或 state-push source。

## 15.3 通信文档

`docs/communication.md` 统一说明：

- 默认端口和地址注入；
- WebRTC signaling envelope 与每流 latest-only video track；
- NVENC/libx264 低延迟 H.264 设置；
- RTP/H.264 FU-A、MTU 与 jitter 行为；
- legacy JPEG/UDP 12-byte frozen header；
- binary realtime state v1；
- reliable command JSON v1、ACK 与 dedupe；
- VR binary v2；
- timestamp/clock domain、uint32 sequence wrap；
- Unity/C# 字段打包和迁移注意事项；
- compatibility matrix 与 failure handling。

文档基于实际协议实现，没有声称当前 wire format 包含不存在的 CRC。

## 15.4 扩展指南

`docs/extension_guide.md` 分别给出新增以下组件的接口、lifecycle、线程、
可选依赖和测试要求：

- robot/backend/executor；
- gripper；
- tactile；
- camera 和 frame processor；
- video encoder/transport；
- VR protocol/transport；
- teleoperation mapping；
- safety filter；
- dataset writer/rollback；
- visualizer；
- session extension；
- deployment composition factory。

指南要求新实现面向最窄协议、构造无 I/O、队列有界、单元测试无硬件、
compatibility bytes/schema 有 golden test。

## 15.5 迁移与发布

`docs/migration_v2.md` 提供 root/legacy 模块到 v2 模块的映射、分阶段迁移
流程、breaking changes、兼容保证和切换检查表。

`CHANGELOG.md` 记录 development release 的 added/changed/compatibility/
breaking changes，并把未验证项列为 release blockers。

`docs/release_checklist.md` 分为：

- 自动化；
- compatibility；
- hardware；
- performance；
- documentation/deployment；
- tagging rule。

没有删除 compatibility wrapper，也没有执行 Git tag。标签只允许在适用于
目标部署的未完成检查全部取得证据后创建。

## 最终验证

| 检查 | 结果 |
|---|---|
| `unittest discover -s tests` | 293 通过，8 跳过 |
| v2 unit tests | 276 通过，3 跳过 |
| complete mock integration | 1 通过 |
| package boundary/compatibility | 11 通过 |
| hardware default gate | 4 跳过，0 个设备连接 |
| legacy RealMan optional suite | 1 个模块跳过 |
| tracked Python AST | 241/241 通过 |
| `compileall` | 通过 |
| `git diff --check` | 通过 |
| 离线 editable install | `airo-doffy==2.0.0.dev0` 通过 |
| 新 Markdown 本地链接 | 全部可解析 |

未运行并明确保留：

- real HDF5 integration：缺少可选 `h5py`；
- Ruff、Pyright、pytest、pre-commit：当前虚拟环境未安装；
- 所有 supervised hardware、Unity、网络和性能 release checks。

## 后续动作

代码重构计划的仓库内实现和文档阶段到此收口。下一步不是继续增加抽象，而是
为一个明确 workcell 实现并评审 production composition factory，然后按
release checklist 安装 dev/recording extras、运行静态工具与 HDF5 测试，
最后执行受监督真机和 Unity 验证。通过后才能创建 `v2.0.0-rcN`。
