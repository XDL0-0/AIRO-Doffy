# Phase 11 重构报告：可视化

日期：2026-07-30  
分支：`codex/v2-refactor`  
前置阶段：Phase 10（`e097545`）

## 1. 结论

Phase 11 已建立不依赖 GUI、硬件或 recorder 对象的 typed visualization 边界。可视化 consumer 只接收冻结快照，以 latest-only 方式调用注入 renderer；renderer 关闭或失败不会把异常传播到运行时发布者。所有传感器字段均可关闭，mock renderer 可在无硬件环境完整运行。

对应提交：

- `22aad8f`：记录现有 dashboard 数据、mock 和关闭行为基线。
- `189c064`：新增 typed snapshot、latest-only consumer 和 mock renderer。
- `2ac90d0`：新增有界 typed UI command outbox。

## 2. Typed snapshot

`VisualizationSnapshot` 只包含显示所需的不可变值：

- 可选 `RobotState`。
- 零个或多个 `ProcessedFrame`。
- 可选 `TactileSample`。
- 可选 `WrenchSample`。
- 可选 typed `RecordingView`。
- 连接状态、错误、source label 和额外状态文本。
- sequence、source timestamp 和 clock domain。

consumer 不接收 `RobotTeleop`、camera source、tactile source、recorder 或任意硬件 handle。

## 3. RecordingView

可视化不再需要理解 recorder 实现，只读取复制出的：

- dataset type 和目录；
- 已完成 episode 数；
- 当前 episode 帧数；
- 上一 episode 长度；
- collecting 状态；
- 待导出数量；
- 最后错误。

该对象冻结且校验非负计数，因此 dashboard 无法修改录制状态。

## 4. Latest-only consumer

`TypedSnapshotConsumer`：

- 使用已有 `LatestValueBuffer` 保持常量内存。
- 拒绝重复或过期 sequence。
- 在独立线程调用注入的 `SnapshotRenderer`。
- `publish` 不等待 renderer。
- renderer 返回 `False` 时视为窗口正常关闭。
- renderer 抛出异常时记录 `last_error` 并停止自身。
- renderer 停止后，后续 `publish` 返回 `False`，不向主运行时抛错。
- `close()` 幂等，并唤醒等待线程后协作式退出。

metrics 暴露 running、accepting、published、rejected、rendered、renderer_closed 和
last_error。

## 5. Mock 与可选传感器

- `VisualizationSnapshot` 允许 robot、camera、tactile、wrench 和 recording 全部为空。
- `MemorySnapshotRenderer` 不加载 GUI 或硬件依赖，可保留快照供 mock session 和测试使用。
- 可配置在指定帧数后模拟用户关闭窗口。
- 测试覆盖全传感器禁用、完整传感器快照、重复关闭、窗口关闭和 renderer 异常。
- camera stream id 必须唯一，避免同一面板被两份数据覆盖。

## 6. UI 命令

`VisualizationCommandOutbox` 是注入式、有界的命令 sink：

- 只接受 `RuntimeCommand`。
- UI 不持有命令 handler，也不直接修改录制或遥操作状态。
- 队列满或 outbox 已关闭时返回 `False`。
- Phase 12 runtime 负责 drain、路由、确认和错误处理。

## 7. 验证

- 单元测试：258 项通过，3 项跳过。
- 3 项跳过仍是 Phase 10 中因缺少可选 `h5py` 的真实 HDF5 集成测试。
- 包结构测试：11 项通过。
- visualization 包导入不加载 NumPy、Matplotlib 或硬件 SDK。
- 受 Git 跟踪的 Python 文件：225 个，全部通过 AST 解析。
- `src` 通过 `compileall`。
- `src` Python 文件最大行宽检查通过（不超过 100）。
- 离线 editable install 通过。
- 当前环境未安装 Ruff 与 Pyright。

## 8. 尚未消除的风险

- 根目录 Matplotlib `visualizer.py` 和 `main.py::visualizer_publish_loop` 仍使用兼容字典路径；
  Phase 12 runtime 将改用 typed producer/consumer 组合。
- 当前轻量环境未安装可选 Matplotlib，因此没有执行真实窗口渲染测试。
- multiprocessing dashboard 的协作式关闭适配仍需在具体 renderer adapter 中完成；当前包级 consumer
  已提供正确的停止语义。
- `ProcessedFrame` 到 GUI 像素数组的颜色格式转换属于 renderer adapter，仍需针对 RGB/BGR 和 depth
  显示验证。

## 9. 下一阶段

Phase 12 将建立运行时编排：

- 依赖顺序启动和逆序关闭。
- 部分初始化失败时只关闭已成功启动的资源。
- `TeleopSession` 组合 VR、robot、mapping、safety、executor 和 watchdog。
- `DataCollectionSession` 在不复制 teleop loop 的前提下组合 recording。
- 根目录 `main.py` 降为兼容入口。
