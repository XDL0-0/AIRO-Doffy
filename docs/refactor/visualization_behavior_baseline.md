# Phase 11 Visualization Behavior Baseline

日期：2026-07-30

## 1. 当前实现边界

根目录 `visualizer.py` 是独立 Matplotlib 子进程。它本身不连接机器人、相机、VR 或触觉硬件，但存在以下耦合：

- `main.py::visualizer_publish_loop` 直接读取 `RobotTeleop`、`CameraManager`、
  `TactileDataHolder` 和 `DatasetRecorder`。
- 主进程以无类型 `dict` 拼装快照。
- dashboard 再从字典中猜测字段、默认值和数据类型。
- dashboard 直接理解 recorder 返回的 `dataset` 字典结构。
- dashboard 的 rollback 按钮直接创建运行时命令。

因此显示进程虽然不是硬件 owner，快照生产、录制状态和 UI 表示仍未形成稳定接口。

## 2. 当前快照字段

`main.py` 当前发布：

- `timestamp`：主进程 monotonic 秒。
- `wrench`：6 维力/力矩数组。
- `joints`：可选关节弧度数组。
- `tcp_translation`：可选 TCP xyz。
- `images`：相机名到 RGB/BGR 数组的字典。
- `camera_count`：配置相机数量。
- `tactile`：可选触觉数组。
- `tactile_timestamp_ns`：触觉 monotonic 时间戳。
- `dataset`：可选录制状态字典。
- `connected`、`error`：运行状态。

`TeleopSample` 还支持兼容字段 `image`、`source_label` 和 `status_extra`。

## 3. 数据队列与命令

- 主进程到 dashboard 的 multiprocessing queue 容量为 2。
- 发布时队列满会丢弃一个旧快照后重试，目标是 latest-only。
- dashboard 每次刷新会排空队列并使用最后一个快照。
- dashboard 到运行时的命令队列容量为 8。
- 当前唯一 UI 命令是 `rollback_last_episode`。
- 非 `RuntimeCommand` 命令会被忽略并记录警告。

## 4. 可选数据和 mock 行为

- 默认快照使用零 wrench，其他传感器数据为空。
- 无相机时仍显示一个 `No camera` 占位面板。
- 某个相机缺帧时保留该面板的上一帧或占位符。
- 无 tactile 时显示零触觉气泡和 `waiting`。
- 无 joints/TCP 时显示 `unavailable`。
- 无 dataset 状态时隐藏 rollback 按钮。
- 非法或非有限 wrench 会替换为全零。
- 未安装 `sensor_comm_dds` 时使用内置 MagTouch 映射。
- 没有交互式 Matplotlib backend 时，子进程记录警告后退出。

## 5. 关闭行为

- 主运行时 `finally` 调用 `VisualizerHandle.close()`。
- handle 先 `terminate()` 仍存活的子进程，再最多等待一秒。
- 随后取消 queue join 并关闭两个 multiprocessing queue。
- 异常会被吞掉，使关闭路径不向主运行时传播。
- 用户直接关闭 Matplotlib 窗口时 dashboard 的 `plt.show()` 返回，子进程结束。

当前关闭不是协作式 shutdown；没有显式 close 消息、完成确认、错误状态或重复关闭测试。

## 6. Phase 11 兼容要求

- 用冻结 typed snapshot 替换字典猜测。
- snapshot producer 才能读取硬件或录制状态；consumer 只解释快照。
- recording status 必须是 typed value，而非 recorder 对象或可变字典。
- 所有传感器字段保持可选，零相机和全传感器禁用必须合法。
- 保留 latest-only 有界更新，不让 UI 反压控制循环。
- renderer 关闭或失败只停止可视化，不得终止主运行时。
- `close()` 必须幂等并优先协作式停止。
- UI 命令通过注入的命令 sink 发送，不直接修改录制或遥操作状态。
