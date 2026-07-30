# Phase 12 Runtime Orchestration Baseline

日期：2026-07-30

## 1. 当前入口

根目录 `main.py` 当前同时负责：

- 读取模块级全局配置。
- 选择 Camera UDP 或 WebRTC manager。
- 构造 `RobotTeleop` 和 `DatasetRecorder`。
- 决定 tactile 与 visualizer 是否启用。
- 创建、启动和 join 所有后台线程。
- 执行 teleop 控制循环。
- 解释 visualizer 命令。
- 决定录制、导出和回滚状态。
- 处理 Ctrl+C 和资源关闭。

没有包级 `LifecycleManager`、`TeleopSession` 或 `DataCollectionSession`。

## 2. 当前启动顺序

1. 构造 camera/VR manager。
2. 通过 manager connection 结果构造 `RobotTeleop`。
3. 可选启动 tactile reader daemon thread。
4. 可选启动 tactile bridge daemon thread。
5. 固定等待 5 秒。
6. 构造 dataset recorder。
7. 启动 manager 通信线程。
8. 可选启动 visualizer 子进程和 publisher daemon thread。
9. 启动 collection daemon thread。
10. 启动 export daemon thread。
11. 在主线程执行 teleop loop。

组件构造、资源获取和线程启动混合在一起。任何中间异常都可能发生在主 `try/finally` 之前，因此没有统一的部分启动回滚。

## 3. 当前控制循环

- 主线程读取 controller 或 hand 的共享可变状态。
- 直接调用 `RobotTeleop.step`，其中包含映射、安全、IK、平滑、机器人命令和部分状态读取。
- 主线程检测 reset gesture 并请求 tactile recalibration。
- 主线程排空 visualizer 命令并直接修改 CameraManager 的 recording flags。
- 通过 `MIN_DT` 实现 deadline 调度，超时时跳过 catch-up。

控制 loop 无 typed cycle 结果、健康快照或可注入 observer。

## 4. 当前线程 ownership

- tactile reader、tactile bridge、visualizer publisher、collector 和 exporter 均为 daemon thread。
- camera/VR manager 和 WebRTC adapter 还创建自己的内部线程。
- visualizer 使用 multiprocessing 子进程。
- 根入口只保存部分 thread handle。
- thread target 的异常通常由 target 自行记录，主循环没有统一健康检查。

## 5. 当前关闭顺序

1. 设置全局 stop event。
2. 请求 tactile reader 停止。
3. 关闭 camera/VR manager。
4. join tactile reader、bridge 和 visualizer publisher。
5. 关闭 visualizer 子进程。
6. join collector 和 exporter。
7. 关闭 teleop/robot。

问题：

- 关闭顺序由入口手工维护，不是启动顺序的严格逆序。
- join timeout 后没有统一检查 thread 是否仍存活。
- close 异常有的记录后继续，有的在线程内部吞掉。
- 重复调用整个 shutdown 路径没有测试。
- 部分启动失败时可能没有执行该路径。

## 6. Phase 12 兼容要求

- 资源按显式依赖顺序启动、严格逆序关闭。
- 启动失败时只回滚已成功启动的资源。
- thread 必须有 owner、stop signal、join 和健康检查。
- session loop 只协调 typed VR、robot state、mapping、safety、executor、watchdog 和扩展。
- 可选 camera、video、tactile、visualization、commands 和 recording 通过 composition 加入。
- 所有可选组件关闭时 session 仍可运行。
- Ctrl+C 只触发 session stop/close，不实现另一份关闭逻辑。
- `DataCollectionSession` 复用 teleop cycle，不复制 teleop loop。
- 根 `main.py` 暂时保留兼容，最终只负责配置、构造和运行 session。
