# Phase 1 重构报告：包骨架与安全目录整理

日期：2026-07-30

分支：`codex/v2-refactor`

基线：`56dcaa2` (`realman-arm-only`)

## 1. 阶段结论

Phase 1 已完成。仓库现在具有可编辑安装的 `src/` 包骨架、零必需硬件
依赖的基础包、明确的弃用规则，以及按用途划分的脚本、文档资产和 Unity
边界。旧命令路径暂时保留兼容入口，危险的旧入口不再使用硬编码参数直接
连接机器人或修改远程仓库。

本阶段只建立结构和安全边界，没有提前搬运遥操作、相机、协议、记录或策略
实现。后续仍应按照计划逐层抽取，避免在一次提交中同时改变目录、协议和
运行行为。

## 2. 新增组件及责任

| 组件 | 单一责任 | 运行依赖 |
|---|---|---|
| `pyproject.toml` | 定义构建、`src` 包发现、可选依赖组和质量工具入口。 | 基础安装无第三方运行依赖。 |
| `src/airo_doffy/` | 提供 v2 命名空间和目标模块边界；当前 `import airo_doffy` 无硬件副作用。 | 仅 Python 标准库。 |
| `deprecated/` | 保存不受支持的历史实现、原因、替代方案及移除规则。 | 不进入 wheel，不得由支持代码导入。 |
| `scripts/dataset/` | 数据转换、回放、评估、可视化和 Hub 操作入口。 | 按命令需要 LeRobot、HDF5、Torch、OpenCV、Hub 等可选依赖。 |
| `scripts/diagnostics/` | 人工硬件检查和可视化；明确不属于自动化测试。 | 按命令需要机器人、相机、VR、GUI 等可选依赖。 |
| `scripts/benchmarks/` | 数值与性能特征测量。 | 按基准需要 NumPy/绘图库。 |
| `scripts/calibration/` | 预留显式校准入口，并记录确认和清理要求。 | 当前无校准实现。 |
| `tests/test_package_layout.py` | 检查安全导入、弃用边界、BLE4 路径和危险旧入口的无副作用导入。 | 仅标准库和已安装的基础包。 |
| `unity/` | 定义 Unity/C# 客户端边界。 | 基线没有可迁移的已跟踪 Unity 源码。 |

## 3. 依赖方向和耦合变化

- 基础包没有必需第三方依赖。相机、视频、机器人、触觉、记录、策略与可视化
  依赖均进入 `project.optional-dependencies`。
- `src/airo_doffy/` 的包初始化不会导入 Torch、OpenCV、RealSense、
  WebRTC、串口或硬件 SDK。
- 主数据采集入口只保留 4-taxel BLE 触觉构造路径。`main.py` 和
  `inference.py` 不再导入旧 41-taxel 串口实现。
- 41-taxel 实现通过 `git mv` 移至
  `deprecated/tactile/magtouch_ilias_41taxel.py`，`git log --follow`
  可以继续追溯原历史。
- 数据集脚本内部引用已改到 `scripts.dataset`，不再依赖旧
  `dataset_tool` 路径。
- `dataset_tool/` 和 `test_tool/` 现在只承担临时命令兼容；真实实现位于
  `scripts/`。它们不属于安装包，也不参与默认测试发现。

仍存在的高耦合没有在本阶段伪装成已解决：`replay_lerobot.py`、
`force_visualizer.py` 和根目录运行模块仍然较大。它们需要在对应的
协议、设备、运行时和记录阶段按测试保护逐步拆分。

## 4. 行为与安全兼容

- `TACTILE_READER="ble4"` 的构造和运行参数保持不变。
- `python tactile.py` 仍是旧 41-taxel 诊断兼容入口，但会发出弃用警告；
  支持运行时不会选择它。
- 普通已迁移命令的旧文件路径会转发到新模块并发出弃用警告。
- 原 `tag_HF.py` 的硬编码远程写入被安全停止；新命令默认 dry-run，只有
  `--execute` 才修改 Hub。
- 原 `ForceMode.py` 和 `freedrive.py` 的硬编码机器人入口被安全停止；
  新命令要求机器人 IP、型号和 `--confirm-hardware`，并在 `finally`
  中退出 force/teach mode。
- HDF5 和 LeRobot 实机回放要求显式硬件确认；LeRobot 的 `--no_robot`
  dry-run 不需要确认。
- 历史评估 PNG 已移至 `docs/assets/evaluation/`，不会进入运行包。

这些安全停止是迁移图明确要求的例外：危险旧命令不自动复现原硬编码副作用。

## 5. 验证结果

| 验证 | 结果 |
|---|---|
| `pip install -e . --no-deps --no-build-isolation` | 通过，构建并安装 `airo-doffy==2.0.0.dev0`。 |
| `python -c "import airo_doffy"` | 通过；没有加载 Torch、OpenCV、RealSense、WebRTC 或串口模块。 |
| 全仓 Python AST 解析 | 通过，98 个 Python 文件。 |
| `tests.test_package_layout` | 4/4 通过。 |
| 新 Hub 标记命令 dry-run | 通过；未发生网络写入。 |
| UR force-mode 无确认调用 | 按设计拒绝；未连接硬件。 |
| HDF5/LeRobot 回放 `--help` | 通过；不会加载硬件 SDK 或连接设备。 |
| `git diff --check` / 提交检查 | 通过。 |
| 原 RealMan 单元测试模块 | 当前最小环境缺少 `cv2`，在测试收集阶段停止；未改写或伪造结果。 |

## 6. 未执行的硬件验证

本阶段没有连接 UR3e、UR5e、RealMan RM75、RealSense、Quest/Unity、
BLE4 或旧串口触觉设备。目录移动没有授权真实运动或网络设备操作。相应验证
继续按迁移图中的 `H-UR`、`H-RM`、`H-CAM`、`H-VR`、`H-TAC` 和
`H-MULTI` 门禁执行。

## 7. 下一阶段入口

Phase 2 应只实现：

1. 不可变、带形状验证的核心领域模型；
2. 不导入硬件 SDK 的窄接口；
3. 常量内存、序列感知、可关闭的 latest-value buffer；
4. 运行时命令和事件模型。

进入设备或协议迁移前，先为上述组件补齐纯单元测试，并保持现有根模块作为
兼容调用方。
