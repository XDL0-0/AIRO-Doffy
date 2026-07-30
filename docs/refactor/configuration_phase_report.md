# 配置重构阶段报告：类型模型、分层加载与延迟工厂

日期：2026-07-30

分支：`codex/v2-refactor`

前置阶段：Phase 2（`f6b2522`）

对应原计划：Phase 13（按 Phase 0 迁移图提前到设备适配器之前）

## 1. 阶段结论

配置基础重构已经完成。新包提供 13 个不可变、分区的配置模型，按固定优先级合并 YAML、
环境变量和 CLI 覆盖，并通过 8 个聚焦工厂延迟加载具体实现。配置模块只依赖 Python
标准库和 v2 端口定义；导入或读取默认配置不会加载 NumPy、OpenCV、RealSense、aiortc、
Ruckig、BLE/DDS 或机器人 SDK。

本阶段没有把旧运行入口一次性切换到新配置。根目录 `config.py` 和
`visualizer_config.py` 继续作为旧运行路径的兼容实现，避免在设备 adapter 尚未迁移时同时
改变配置与硬件行为。后续每迁移一个 adapter，就在 composition root 为其注入相应的新配置
section 和工厂。

## 2. 类型化配置

`src/airo_doffy/config/models.py` 提供：

- `NetworkConfig`
- `RobotConfig`
- `CameraConfig`
- `VRConfig`
- `TeleopConfig`
- `TactileConfig`
- `RecordingConfig`
- `VisualizationConfig`
- `VideoStreamingConfig`
- `StateTransportConfig`
- `CommandTransportConfig`
- `WrenchConfig`
- `RuntimeConfig`
- 聚合边界对象 `AiroDoffyConfig`

所有 section 都使用 frozen/slotted dataclass，并在构造时校验枚举、端口、速率、尺寸、数组
形状和互斥条件。坐标轴映射必须是正交 `3 x 3` 矩阵；TCP axis-angle 到齐次变换的推导使用
纯标准库实现，不会为了配置计算引入 NumPy/SciPy。

旧默认值中的相机分辨率与帧率、遥操作频率、4-taxel BLE 形状、视频端口、记录模式、
触觉滤波和可视化参数均由快照测试保护。固定的 PC、VR 和机器人局域网地址有意不进入
新默认值；它们必须由环境或部署侧覆盖。

## 3. 分层 YAML

`src/airo_doffy/config/loader.py` 的合并顺序为：

1. `configs/default.yaml`
2. `configs/robots/*.yaml`
3. `configs/experiments/*.yaml`
4. `AIRO_DOFFY__SECTION__FIELD` 环境变量
5. `section.field` CLI 覆盖

递归合并不会修改输入 mapping。未知 section、未知字段、非 mapping section 和非法覆盖名
都会在组件构造前失败。环境变量和 CLI 值优先按 JSON 标量/数组解析，再回退为普通字符串。

仓库内 YAML 使用 JSON 兼容语法，因此最小安装只用标准库即可读取；需要普通 YAML 语法时，
可安装 `airo-doffy[config]`，加载器才会延迟导入 PyYAML。

已增加以下无秘密 profile：

| 类型 | 文件 |
|---|---|
| 默认 | `configs/default.yaml` |
| 机器人 | `ur3e.yaml`、`ur5e.yaml`、`realman_rm75.yaml` |
| 实验 | `collect_ur3e.yaml`、`collect_rm75.yaml`、`vr_hand_tracking.yaml` |

默认 YAML round-trip 后与 `AiroDoffyConfig()` 完全相等。RealMan profile 保留 7 关节初始
位置和独立 VR 坐标轴映射，但不保存设备 IP。

## 4. 聚焦工厂

`src/airo_doffy/config/factories.py` 提供：

| 工厂 | 输入配置 | 输出端口 |
|---|---|---|
| `RobotFactory` | `RobotConfig` | `RobotBackend` |
| `CameraFactory` | `CameraConfig` | `CameraSource` |
| `EncoderFactory` | `VideoStreamingConfig` | `VideoEncoder` |
| `VideoTransportFactory` | `VideoStreamingConfig`、`NetworkConfig` | `VideoTransport` |
| `VRSourceFactory` | `VRConfig`、`NetworkConfig` | `VRInputSource` |
| `TactileFactory` | `TactileConfig` | `TactileSensor` |
| `RecorderFactory` | `RecordingConfig` | `EpisodeRecorder` |
| `VisualizerFactory` | `VisualizationConfig` | `Lifecycle` |

每个工厂持有一个 `module:symbol` 注入目标。创建工厂不会导入目标模块；只有显式调用
`create()` 才加载 adapter，并在返回后执行结构化端口检查。缺少可选依赖、目标不存在、
目标不可调用或返回对象不满足端口，都会转换成明确的 v2 错误。

本阶段没有为尚未迁移的旧实现注册虚假的默认目标。绑定将在对应 adapter 有测试保护并满足
端口后逐个加入。

## 5. 兼容性边界

- 旧入口继续使用根 `Config` / `VisualizerConfig`，当前运行默认行为没有被暗中切换。
- 新配置不打开 socket、不连接机器人、不发现相机、不启动线程，也不实例化运行组件。
- 旧 `Config` 中的硬编码 IP 仍只存在于兼容路径；新 profile 不复制这些地址。
- RealMan/UR 的真实连接、运动和安全行为没有在本阶段执行。
- 后续 adapter 不得接收完整 `AiroDoffyConfig`，而应只接收表中所需的窄 section。

## 6. 验证结果

| 验证 | 结果 |
|---|---|
| 配置模型快照与校验测试 | 6/6 通过 |
| YAML/覆盖/round-trip 测试 | 6/6 通过 |
| 延迟工厂与端口检查测试 | 3/3 通过 |
| 当前全部 v2 unit tests | 39/39 通过 |
| Phase 1 包布局/兼容测试 | 5/5 通过 |
| 可编辑安装 | `airo-doffy==2.0.0.dev0` 构建安装通过 |
| 全仓 Python AST | 122 个文件通过 |
| 配置包安全导入 | 未加载任何已列硬件、编解码或数值可选依赖 |
| `git diff --check` / 100 字符行检查 | 通过 |

旧 RealMan 大型测试在当前最小环境中仍因缺少可选 `cv2` 无法收集；没有伪造这项结果。
新增测试不连接任何物理硬件。

## 7. 下一阶段

进入机器人适配层前先刻画旧 `Robot` facade、UR 和 RealMan 的状态/动作行为。随后按
mock backend → UR backend → RealMan backend/executor 的顺序迁移；每一步保留根模块兼容
入口，并将 SDK import、连接、初始移动和关闭动作限制在显式 lifecycle 内。
