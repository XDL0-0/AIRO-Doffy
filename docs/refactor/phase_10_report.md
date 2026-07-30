# Phase 10 重构报告：录制

日期：2026-07-30  
分支：`codex/v2-refactor`  
前置阶段：Phase 9（`a999160`）

## 1. 结论

Phase 10 已将录制状态、样本缓冲、HDF5 序列化、LeRobot 序列化、回滚和后台导出拆分为独立组件。新组件不读取硬件、不依赖可视化，并保持现有路径、形状、数据类型、表示方式、episode 编号和回滚语义。

对应提交：

- `3a43112`：记录现有录制 schema、生命周期和回滚行为基线。
- `8f9649f`：拆分纯 recording schema、冻结样本缓冲和 episode 状态机。
- `29cb46e`：拆分 HDF5/LeRobot writer 和存储回滚。
- `ea19cbb`：新增有界、非阻塞的后台导出 worker。

## 2. 新增组件

| 组件 | 职责 |
| --- | --- |
| `recording.schema` | 推导 state/action 维度、字段名、HDF5 路径和 LeRobot features |
| `recording.samples` | 以冻结字节数组和不可变 episode 隔离采集线程与 serializer |
| `recording.state` | 管理录制、待导出、导出失败、待回滚和关闭状态 |
| `recording.writers.hdf5` | 原子写入 ACT/HDF5 episode 和描述文件 |
| `recording.writers.lerobot` | 通过注入的数据集 provider 写 LeRobot episode |
| `recording.writers.rollback` | 删除 LeRobot data、metadata、视频和派生统计 |
| `recording.export_worker` | 在容量受限的单独线程中串行执行写入和回滚 |

## 3. 生命周期与编号

- episode 只有在包含至少一个样本时才能进入待导出状态。
- `next_episode_index` 只有在 writer 成功返回后才推进。
- 导出失败保留原编号，必须显式重试或丢弃失败 episode。
- 回滚优先丢弃当前未保存 episode，不访问磁盘。
- 已保存 episode 只允许从最后一集回滚，并复用删除后的编号。
- 待导出或待回滚时禁止关闭状态机，避免静默丢失存储任务。

## 4. Schema 兼容性

详细基线见 `docs/refactor/recording_behavior_baseline.md`。

已通过测试固定：

- `qpos`、`both`、`tcp` 和 `delta_tcp` 的 state/action 表示及别名。
- TCP 四元数与位置顺序、delta TCP 分量顺序。
- 公共时间戳顺序和相机时间戳扩展。
- HDF5 的路径、形状以及现有 float64/float32/int64/uint8 差异。
- LeRobot feature keys、形状、名称以及 float32/int64/video/image 类型。
- HDF5 以最大现有编号加一，而不是按文件数量编号。
- HDF5 描述文件追加和回滚修剪。
- LeRobot 深度从米转换为裁剪后的 uint16 毫米。

## 5. 存储安全

- HDF5 先写同目录临时文件，再原子替换为最终 episode 文件。
- 已存在的 episode 文件不会被覆盖。
- HDF5 写入或描述更新失败时清理未完成文件。
- LeRobot feature 不匹配时抛出明确错误，不再自动删除现有数据集。
- LeRobot metadata 中的 data/video 路径必须解析在数据集目录内部。
- 回滚只接受显式非负编号，且 LeRobot 只删除 metadata 声明的最后一集。

## 6. 非阻塞导出

`ExportWorker` 使用固定容量队列：

- `submit_export` 和 `submit_rollback` 使用非阻塞入队。
- 队列满时立即抛出 `ExportQueueFullError`，控制线程不会等待磁盘。
- 每个任务返回 `ExportTicket`，可读取成功路径、存储是否改变或原始异常。
- 单 worker 串行执行写入与回滚，避免两类磁盘操作互相竞争。
- metrics 暴露 queued、busy、submitted、completed、failed 和 rejected。
- 关闭时默认 drain；也可显式取消尚未执行的排队任务。

## 7. 验证

- 单元测试：248 项通过，3 项跳过。
- 跳过项均为真实 HDF5 集成测试，因为当前轻量虚拟环境未安装可选 `h5py`。
- LeRobot writer 的 feature、dtype、深度转换、finalize 和 schema mismatch 通过注入式假数据集验证。
- 包结构测试：11 项通过。
- 受 Git 跟踪的 Python 文件：219 个，全部通过 AST 解析。
- `src` 通过 `compileall`。
- `src` Python 文件最大行宽检查通过（不超过 100）。
- 离线 editable install 通过。
- 当前环境未安装 Ruff 与 Pyright，因此未声称完成这两项检查。

## 8. 尚未消除的风险

- 根目录 `DatasetRecorder`、`collect_loop` 和 `export_loop` 仍使用旧实现；计划在 Phase 12
  `DataCollectionSession` 中切换到新组合。
- 当前环境无法执行真实 HDF5 writer 集成测试。
- 尚未针对实际安装的 LeRobot/pandas/pyarrow 版本执行完整数据集写入和共享 parquet/video
  回滚验证。
- 冻结图像会产生一次受控内存复制；它避免生产者继续修改数据，但仍需在实际分辨率和 episode
  长度下测量内存占用。
- 有界队列提供背压语义，但应用层仍需决定队列满时提示操作者、重试还是保留当前 episode。

## 9. 下一阶段

Phase 11 将把可视化器改为 typed snapshot consumer：

- 不直接读取硬件。
- 不直接读取或修改录制状态。
- 保留 mock 模式和传感器可选关闭行为。
- 确保关闭可视化不会终止运行时。
