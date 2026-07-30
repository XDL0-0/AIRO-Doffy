# Phase 12 重构报告：运行时编排

日期：2026-07-30  
分支：`codex/v2-refactor`  
前置阶段：Phase 11（`2881211`）

## 1. 结论

Phase 12 已建立确定性的资源生命周期、单一 teleop loop、组合式 data collection 和可安装薄入口。session 只协调 typed ports，不实现 robot、VR、mapping、safety、watchdog、recording 或 visualization 的内部逻辑。

对应提交：

- `5dc22e8`：记录根运行时启动、线程、控制循环和关闭行为基线。
- `9f2ffa0`：新增 `LifecycleManager` 和 `ManagedWorker`。
- `55c106b`：新增组合式 `TeleopSession`。
- `705df0f`：新增 recording extension 和 `DataCollectionSession`。
- `a7db3e9`：新增可安装薄应用入口。

## 2. LifecycleManager

`LifecycleManager`：

- 按构造时声明的顺序调用 `start()`。
- 只记录已成功启动的组件。
- 中途启动失败时立即逆序关闭已成功组件。
- 正常关闭时严格逆序调用 `close()`。
- 某个 close 失败后仍继续关闭其他组件。
- 重复 `close()` 安全。
- snapshot 暴露 configured、started、state 和 error。

`ManagedWorker`：

- 为一个 thread target 提供显式 stop event。
- 保存 thread handle，关闭时必须 join。
- 不把 worker thread 默认为 daemon。
- 捕获 worker 异常并由 `check_health()` 报告。
- join timeout 会成为明确的 `LifecycleError`。

## 3. TeleopSession

每轮唯一控制路径为：

1. 排空 typed commands 并交给注入 dispatcher。
2. 读取 immutable robot state 和 latest VR input。
3. 评估 watchdog。
4. 执行 injected mapping。
5. 执行 injected action filter。
6. safety rejection 转为 HOLD，避免 executor 重复上一条活动动作。
7. watchdog 处理 stale、zero velocity、hold 和 reference recovery。
8. 向 caller-owned executor 提交单调序列动作。
9. 生成 immutable `TeleopCycle`。
10. 通知可选 session extensions。

VR 缺失时，即使 watchdog 未配置也会生成 HOLD。watchdog 初始恢复时，mapping 的第一次调用捕获参考，随后显式 ACK，再允许 ACTIVE 动作。

## 4. 资源 ownership

TeleopSession 的默认启动顺序：

1. action executor（其 `start` 拥有 robot backend）。
2. executor `ManagedWorker`。
3. VR source。
4. 可选 extensions。

关闭严格逆序，因此 extensions 和 VR 先停止，executor worker join 后才关闭 executor/backend。

state source 与 executor 分离注入，允许后续使用 executor-owned cached state，避免强制从 session 线程调用有线程亲和性的 SDK。

## 5. DataCollectionSession

`RecordingCycleExtension`：

- 作为 `SessionExtension` 消费同一个 `TeleopCycle`。
- sample factory 负责从 cycle 构造冻结 `RecordingSample`。
- start/stop/rollback 复用 Phase 10 状态机和有界 export worker。
- stop 立即 seal episode 并非阻塞入队。
- 活动 episode 回滚只丢弃内存缓冲。
- 已完成 episode 回滚通过同一串行存储队列执行并复用编号。
- export 失败保留 episode，可显式 retry 或 discard。

`DataCollectionSession` 只委托 `TeleopSession.start/run/request_stop/close`，没有第二套控制循环。

## 6. 薄应用入口

新增安装脚本：

- `airo-doffy-teleop`
- `airo-doffy-collect`

入口只负责：

- 解析 base、robot 和 experiment 配置路径。
- 解析重复的 `--set SECTION.FIELD=VALUE`。
- 加载 typed `AiroDoffyConfig`。
- 加载显式 `MODULE:SYMBOL` composition factory。
- 创建、启动、运行并关闭 session。
- 将 Ctrl+C 转为 cooperative `request_stop`。

session factory 可由 CLI 或
`AIRO_DOFFY_TELEOP_SESSION_FACTORY` /
`AIRO_DOFFY_COLLECT_SESSION_FACTORY` 提供。入口模块不导入硬件 SDK。

入口位于可安装的 `src/airo_doffy/apps`，这是相对计划中仓库根 `apps/` 的有意调整。

## 7. 验证

- 单元测试：276 项通过，3 项跳过。
- 3 项跳过仍是因当前环境缺少可选 `h5py` 的真实 HDF5 集成测试。
- 包结构测试：11 项通过。
- mock session 覆盖正常动作、safety rejection、VR 缺失、watchdog 恢复/失联、可选扩展关闭和重复 close。
- data collection 覆盖同 loop 取样、后台导出、活动回滚和已完成回滚。
- 两个安装后 CLI 的 `--help` 均成功执行。
- 受 Git 跟踪的 Python 文件：237 个，全部通过 AST 解析。
- `src` 通过 `compileall`。
- `src` Python 文件最大行宽检查通过（不超过 100）。
- 离线 editable install 通过。
- 当前环境未安装 Ruff 与 Pyright。

## 8. 尚未消除的风险

- 根 `main.py` 仍是旧兼容运行时，而不是薄 wrapper。具体 UR/RealMan composition factory 尚未完成前保留它，避免切断现有硬件入口。
- 新 CLI 当前要求显式 session factory；内置 UR/RealMan factory 将在后续配置/运行时集成阶段提供。
- camera/video/tactile/visualization 已有 typed component，但尚未全部实现为具体 `SessionExtension`。
- 尚未在真实 UR/RealMan SDK 上验证 state source 与 executor owner thread 的组合。
- session 的硬件 stop/HOLD、Ctrl+C 和部分启动失败仍需 integration/hardware test。

## 9. 下一阶段

按照计划，下一阶段将继续配置与 composition：

- 补齐 recording/export queue 等 typed 配置。
- 提供 UR 与 RealMan session factory。
- 将 CLI factory target 收敛为配置选择。
- 保留环境变量与 CLI 的最高优先级。
- 逐步把根 `main.py` 降为兼容 wrapper。
