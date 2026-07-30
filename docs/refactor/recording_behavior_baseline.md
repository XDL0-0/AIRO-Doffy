# Phase 10 Recording Behavior Baseline

日期：2026-07-30

## 1. 当前实现边界

根目录 `dataset.py::DatasetRecorder` 当前同时负责：

- 读取全局 `Config` 并推导录制 schema；
- 查找或创建数据集并决定下一个 episode 编号；
- 保存当前 episode 的可变状态；
- HDF5 全 episode 内存缓冲；
- LeRobot 逐帧 `add_frame`；
- HDF5 与 LeRobot 序列化；
- HDF5 与 LeRobot 回滚；
- 向可视化器提供录制状态；
- 关闭 LeRobot writer，并可选推送到 Hub。

根目录 `main.py` 另外持有录制控制状态：

- VR 的 `Start`、`Stop` 和 `Undo` 文本命令写入 `CameraManager` 布尔标志。
- `collect_loop` 读取硬件快照并直接调用 `DatasetRecorder.data_collection`。
- `export_loop` 轮询导出/回滚标志，暂停采集线程并同步执行磁盘工作。
- `visualizer_publish_loop` 直接读取 `DatasetRecorder.recording_status`。

这使录制状态、硬件采样、序列化和可视化互相耦合。

## 2. Episode 控制语义

- `Start`：开始采样，同时清除待导出和待回滚标志。
- `Stop`：停止采样并设置待导出标志。
- `Undo`、`Rollback`、`DeleteLast`：停止采样、清除待导出并设置待回滚标志。
- 仅在 `data_collecting_state` 为真且检测到运动时追加样本。
- HDF5 与 LeRobot 都以从零开始的稳定 episode index 编号。
- 成功导出后 `recorded_episodes` 加一；失败或空 episode 不应增加编号。
- 回滚优先丢弃尚未保存的当前 episode；否则删除最后一个已保存 episode。
- 已保存 episode 回滚后复用其编号。

## 3. 数据表示

`data_schema.py` 的当前映射：

| `DATA_TYPE` | state | action | extra TCP |
| --- | --- | --- | --- |
| `qpos` / `joint` / `joint_configuration` | `dof + 1` joint + gripper | `dof + 1` joint + gripper | 否 |
| `both` | `dof + 1` joint + gripper | `dof + 1` joint + gripper | 7 维 TCP pose |
| `tcp` / `tcp_quat` / `eef` | 8 维 TCP + gripper | 8 维 TCP + gripper | 否 |
| `delta_tcp` | `dof + 1` joint + gripper | 7 维 delta TCP + gripper | 7 维 TCP pose |

TCP pose 顺序为 `qx, qy, qz, qw, x, y, z`。  
TCP 状态/动作顺序为 `qx, qy, qz, qw, x, y, z, gripper`。  
Delta TCP 动作顺序为
`delta_x, delta_y, delta_z, delta_rotvec_x, delta_rotvec_y, delta_rotvec_z, gripper`。

## 4. 公共时间戳 schema

`/extra/timestamps_ns` 或 `extra.timestamps_ns` 是固定宽度 `int64` 向量，顺序为：

1. `collect`
2. `robot_state`
3. `robot_action`
4. `vr_input`
5. `tactile`
6. `camera_0 ... camera_n`

缺失时间戳填零。

## 5. HDF5 schema

文件名为 `episode_<index>.hdf5`，根属性 `sim = False`。

| 路径 | 形状 | 当前 dtype |
| --- | --- | --- |
| `/observations/qpos` | `(T, state_dim)` | `float64` |
| `/action` | `(T, action_dim)` | `float64` |
| `/extra/timestamps_ns` | `(T, 5 + camera_num)` | `int64` |
| `/extra/tcp_pose` | `(T, 7)` | `float64` |
| `/observations/force` | `(T, 3)` | `float64` |
| `/observations/torque` | `(T, 3)` | `float64` |
| `/observations/tactile` | `(T, *tactile_shape)` | `float64` |
| `/observations/images/camera_i` | `(T, height, width, 3)` | `uint8` |
| `/observations/depth/camera_i` | `(T, height, width)` | `float32` |

`timestamps_ns` 的 `names` 属性保存上述名称的字节字符串数组。可选路径只在对应配置启用时存在。每次写入后向
`episode_descriptions.txt` 追加
`Episode <index>: max_timesteps = <T>`。

HDF5 启动编号是目录中所有合法 `episode_<n>.hdf5` 的最大编号加一，而不是文件数量。

## 6. LeRobot schema

- `action`：`float32`，形状 `(action_dim,)`。
- `observation.state`：`float32`，形状 `(state_dim,)`。
- `extra.timestamps_ns`：`int64`，形状 `(5 + camera_num,)`。
- `extra.tcp_pose`：可选 `float32`，形状 `(7,)`。
- `observation.force` / `observation.torque`：可选 `float32`，形状 `(3,)`。
- `observation.tactile`：可选 `float32`，形状为 `TACTILE_SHAPE`。
- `observation.images.camera_i`：`video`，形状 `(height, width, 3)`。
- `observation.depth.camera_i`：`image`，形状 `(height, width, 1)`。
- 深度输入从 float 米转换为裁剪后的 uint16 毫米。
- 缺失 tactile 填 float32 零数组，缺失 extra TCP pose 和时间戳也填零。
- `task` 使用 `TASK_NAME`。

现有数据集的可录制 feature keys 与当前配置不一致时，旧实现会删除数据集目录并重建。这是高风险兼容行为，重构实现不得在未显式授权的情况下自动删除已有数据。

## 7. 回滚语义

### HDF5

- 当前缓冲非空时只清空缓冲，不删除磁盘文件。
- 否则删除 `episode_<recorded_episodes - 1>.hdf5`。
- 同时移除描述文件中以对应 `Episode <index>:` 开头的行。
- 缺少目标文件时回滚失败且编号不变。

### LeRobot

- 有未保存帧时优先清空 episode buffer。
- 已保存 episode 回滚依据 `meta/info.json` 的 `total_episodes` 定位最后一集。
- 从 data parquet、episode metadata 和未被其他 episode 共用的视频文件中移除该集。
- 更新 `total_episodes`、`total_frames` 和 `splits`。
- 回滚到零集时移除 tasks metadata，并始终移除已过期的 stats。
- 回滚成功后复用被删除 episode 的编号。

## 8. 性能与一致性基线

- HDF5：采集线程只追加内存列表，但导出时需要暂停采集并同步写完整 episode。
- LeRobot：采集线程每帧直接执行 `add_frame`，可能被图像编码或磁盘路径阻塞。
- 导出请求是两个可覆盖的布尔标志，不是有界队列。
- 没有明确的队列满策略、任务完成结果或关闭时 drain 语义。
- recorder 与采集线程之间没有内部锁，依赖外部 `pause_event` 协调。
- 相机、force、torque 或 depth 每帧缺失时可能形成不等长 HDF5 列。

Phase 10 的实现必须以测试固定上述格式和编号/回滚行为，再把硬件快照、内存状态和磁盘工作分离。
