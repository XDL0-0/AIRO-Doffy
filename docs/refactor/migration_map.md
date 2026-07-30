# AIRO-Doffy v2.0 迁移映射

## 1. 迁移前置条件

本文件是 Phase 0 审计产物，不授权移动或修改实现。进入 Phase 1 前必须先评审本映射。

当前 `G:\Projects\AIRO-DOFFY-2.0` 是新建目录，不包含旧仓库 Git 历史。后续计划要求尽量使用 `git mv`，因此最安全的 Phase 1 起点不是把 62 个文件散拷到空目录，而是：

1. 从 `G:\Projects\AIRO-Doffy` 的 `realman-arm-only` 分支克隆或创建保留历史的 worktree 到新目录。
2. 确认基线提交 `56dcaa23b0ef518338875b8a7fceb5f70effbe53` 和 working tree 状态。
3. 将本计划及四份 Phase 0 文档加入新仓库。
4. 在 `codex/v2-refactor` 分支按小提交使用 `git mv`；每次移动先保留旧路径 shim，再迁移调用方。

若直接覆盖当前新目录，会失去可审查的 rename lineage；执行前应先备份/确认该目录内只有计划和 Phase 0 文档。

## 2. 验证代码

为使逐文件表保持可读，下面代码代表迁移前必须通过的测试。一个文件可以要求多组。

| 代码 | 必须具备的验证 |
|---|---|
| `T0` | 全仓 AST/compile、格式、静态类型和“核心包无可选依赖”import smoke。 |
| `T1` | 旧 import 路径、公开符号、旧 CLI 参数/退出码的兼容测试；shim 发出一次弃用告警。 |
| `T2` | 当前 `Config` / `VisualizerConfig` 默认值、派生数组、非法组合与 YAML round-trip 快照。 |
| `T3` | VR controller 29/31 字段、hand text 89 字段、`HB` payload 的 golden vectors、畸形/超时/sequence tests。 |
| `T4` | UDP 端口映射、JPEG `!IHHI` 包头、分片/重组、乱序/丢包/超 MTU 和旧 Unity 互通 fixture。 |
| `T5` | WebRTC signaling JSON、offer/answer/ICE/stop、`control` DataChannel、peer disconnect 和 shutdown。 |
| `T6` | fake robot/gripper 的 mapping、安全、命令所有权、异常和 lifecycle integration。 |
| `T7` | RealMan CAN-FD >100 Hz、5 ms 目标、10 ms gap、heartbeat、callback in-flight、quarantine/close。 |
| `T8` | 4×3 tactile sample、BLE payload/unwrap、校准/deadband/EMA/Kalman/reconnect；41-taxel legacy fixture。 |
| `T9` | HDF5/LeRobot golden datasets、feature shape/dtype/name、episode index、rollback、跨版本读取。 |
| `T10` | policy config/serialization、processor、feature conversion、训练 loss、推理 action shape/timing。 |
| `T11` | script import 不产生硬件/网络/远端写；`--help`、dry-run、显式确认和失败清理。 |
| `T12` | visualizer/model snapshot、headless smoke、资产链接与生成脚本输出一致性。 |

协议代码：

| 代码 | 必须保持的兼容面 |
|---|---|
| `P0` | 无线上/磁盘协议；只要求 Python/CLI 兼容。 |
| `P-VR` | Unity VR controller/hand text 与 `HB` 二进制。 |
| `P-UDP` | UDP 地址、端口、JPEG 分片与消息队列行为。 |
| `P-WEBRTC` | WebSocket signaling、SDP/ICE 和 DataChannel 命令。 |
| `P-CTRL` | 机器人命令 cadence、安全 hold/stop、夹爪语义。 |
| `P-TACT` | tactile shape、轴/顺序、校准与序列化。 |
| `P-DATA` | HDF5 与 LeRobot schema、命名、dtype、时间戳与 episode/rollback。 |

硬件代码：

| 代码 | Phase 内硬件验证 |
|---|---|
| `H0` | 不需要；CI/mocks 即可。 |
| `H-CAM` | RealSense 枚举、RGB/depth、目标 FPS、断开重连。 |
| `H-VR` | Quest/Unity 旧客户端互通、超时与控制命令。 |
| `H-UR` | UR3e/UR5e dry-run 后低速实机、急停/断线/夹爪。 |
| `H-RM` | RealMan RM75 实机、CAN-FD cadence、state push 与安全停机。 |
| `H-TAC` | BLE4 与旧串口传感器校准、断连、数值/轴方向。 |
| `H-MULTI` | 相机+VR+机器人+触觉+记录的完整台架。 |

## 3. 根模块迁移

| 当前文件 | 目标与策略 | 迁移前测试 | 协议 | 硬件 |
|---|---|---|---|---|
| `__init__.py` | 新建 `src/airo_doffy/__init__.py`；根文件先改为兼容 shim，所有外部导入迁完再移除。 | `T0,T1` | `P0` | `H0` |
| `camera_udp.py` | **split + wrapper**：camera source、frame processor、legacy JPEG UDP、VR receiver、state transport；旧 `CameraUDPManager` 用 façade 组合新组件。 | `T0,T1,T3,T4,T6` | `P-VR,P-UDP,P-TACT` | `H-CAM,H-VR,H-MULTI` |
| `config.py` | **split + wrapper**：`config/models.py`、`loader.py`、`factories.py` 与 YAML profiles；旧 `Config` adapter 保持字段。配置阶段提前。 | `T0,T1,T2` | `P0` | `H0`，profile 最终随各设备验证 |
| `data_schema.py` | **move/split + wrapper** 到 `recording/schema.py`；跨域枚举/typed shape 可在 `core/types.py`，不得复制两套真值。 | `T0,T1,T9,T10` | `P-DATA` | `H0` |
| `dataset.py` | **split + wrapper** 为 episode state、recorder、HDF5 writer、LeRobot writer、rollback；serializer 不依赖 runtime manager。 | `T0,T1,T9` | `P-DATA` | `H0`，最后 `H-MULTI` |
| `force_filter.py` | **move + wrapper** 到 `devices/wrench/filters.py`；保持数值顺序和 reset。 | `T0,T1,T6` + 数值 golden | `P-CTRL` | `H0`，后续 `H-UR,H-RM` |
| `inference.py` | **split + CLI wrapper** 到 `policies/inference.py`、新增 `runtime/inference_session.py`、`apps/inference.py`；policy 必须在 robot connect/move 前验证。 | `T0,T1,T6,T9,T10` | `P-CTRL,P-DATA` | `H-UR,H-RM,H-CAM` |
| `main.py` | **replace with thin wrapper**；逻辑迁到 `runtime/data_collection_session.py`、`lifecycle.py`、`apps/collect.py`。 | `T0,T1,T3,T6,T8,T9` + mock E2E | 全部现有运行协议 | `H-MULTI` |
| `parse_vr.py` | **move/split + wrapper** 到 `devices/vr/protocol.py`、controller/hand types；必须保持 parser 的拒绝/None 行为。 | `T0,T1,T3` | `P-VR` | `H-VR` |
| `realman_teleop.py` | **large split + CLI wrapper**：RealMan adapter、executor、mapping/safety、session/app；一次只抽一层，先保留旧 loop characterization。 | `T0,T1,T3,T6,T7` | `P-VR,P-CTRL` | `H-RM,H-VR` |
| `robot_backend.py` | **split + wrapper**：domain port、UR adapter、RealMan adapter、gripper、executor/factory；backend 原子能力与 cadence executor 明确分界。 | `T0,T1,T6,T7` | `P-CTRL` | `H-UR,H-RM` |
| `robot_teleop.py` | **large split + wrapper**：actions、controller/hand/gripper mappings、coordinate/pose transforms、workspace/joint/velocity safety；不直接构造硬件。 | `T0,T1,T3,T6` | `P-VR,P-CTRL` | `H-UR`，映射亦在 `H-RM` 验证 |
| `tactile.py` | **deprecate + move** 到 `deprecated/tactile/magtouch_ilias_41taxel.py`；根 shim 发警告，至少一个 release 后再考虑删除。 | `T0,T1,T8` | `P-TACT` | `H-TAC` |
| `tactile_4point.py` | **split + wrapper**：`devices/tactile/base.py`、`magtouch_ble4.py`、filter/calibration helper；GUI 移到 visualization。 | `T0,T1,T8,T12` | `P-TACT` | `H-TAC` |
| `udp_comms.py` | **move/refine + wrapper** 到 `streaming/transports/udp.py`（目标树新增），state adapter 再依赖它；保留 deque/旧 aliases。 | `T0,T1,T3,T4` | `P-UDP,P-VR` | `H-VR` |
| `ur_teleop.py` | **repair shim then deprecate**；只转发新 teleop app/session，不再导入不存在的 `URTeleop`。 | `T0,T1,T6` | `P-CTRL` | `H-UR` |
| `utils.py` | **split + wrapper**：math、UR kinematics、transforms、安全、wrench compensation；移除 import-time logging。逐函数迁移并固定数值。 | `T0,T1,T6` + FK/quat/collision/filter golden | `P-CTRL` | `H0`，后续 `H-UR,H-RM` |
| `visualizer.py` | **split + wrapper** 到 models/publisher/dashboard；只消费 immutable snapshot，command 用独立 router。 | `T0,T1,T12` | `P0` | `H0`，最后 `H-MULTI` |
| `visualizer_config.py` | **merge + wrapper** 到 `config/models.py::VisualizationConfig`。 | `T0,T1,T2,T12` | `P0` | `H0` |
| `WebRTC_udp.py` | **split + wrapper**：统一 camera source/VR receiver，视频仅进 `webrtc_transport.py`，state/commands 分离；先做编码 feasibility spike。 | `T0,T1,T3,T5` + latency benchmark | `P-VR,P-WEBRTC,P-UDP` | `H-CAM,H-VR,H-MULTI` |

## 4. ForceFlow++ 与策略迁移

目标计划缺少现有 ForceFlow++ 子包；必须新增 `src/airo_doffy/policies/forceflowpp/`，并把策略迁移作为独立阶段，不可顺带塞进 runtime。

| 当前文件 | 目标与策略 | 迁移前测试 | 协议 | 硬件 |
|---|---|---|---|---|
| `policies/__init__.py` | **move + top-level package alias** 到 `src/airo_doffy/policies/__init__.py`。 | `T0,T1,T10` | `P0` | `H0` |
| `policies/forceflowpp/__init__.py` | **move + wrapper**，保持公开 `Config`/`Policy` 导出。 | `T0,T1,T10` | `P0` | `H0` |
| `policies/forceflowpp/configuration_forceflowpp.py` | **move**；先锁定序列化字段、默认值和 validation。 | `T0,T10` | `P-DATA` | `H0` |
| `policies/forceflowpp/contact_labeling.py` | **move**；如改名为 `contact.py`，旧路径保留 alias。 | `T0,T1,T10` | `P-DATA` | `H0` |
| `policies/forceflowpp/modeling_forceflowpp.py` | **move, then internal refactor**；先保存 loss/action golden，后再拆 model components。 | `T0,T10` | `P-DATA` | `H0`；集成后 `H-UR/H-RM` |
| `policies/forceflowpp/modules/__init__.py` | **move**。 | `T0,T10` | `P0` | `H0` |
| `policies/forceflowpp/modules/adaln_dit.py` | **move**，不在结构阶段改数学。 | `T0,T10` | `P0` | `H0` |
| `policies/forceflowpp/modules/encoders.py` | **move**，固定 modality/token shape。 | `T0,T9,T10` | `P-DATA` | `H0` |
| `policies/forceflowpp/modules/flow_matching.py` | **move**，固定随机种子下 loss/sample。 | `T0,T10` | `P0` | `H0` |
| `policies/forceflowpp/prior_library.py` | **move**，固定 phase mixture/KL。 | `T0,T10` | `P0` | `H0` |
| `policies/forceflowpp/processor_forceflowpp.py` | **move + wrapper**；锁定 LeRobot processor API 版本与 feature rename/normalization。 | `T0,T1,T9,T10` | `P-DATA` | `H0` |

## 5. 数据集脚本迁移

| 当前文件 | 目标与策略 | 迁移前测试 | 协议 | 硬件 |
|---|---|---|---|---|
| `dataset_tool/convert_delta_tcp_dataset.py` | **move/split + CLI wrapper** 到 `scripts/dataset/convert_delta_tcp.py`；四元数/delta 算法下沉可测 library。 | `T1,T9,T11` | `P-DATA` | `H0` |
| `dataset_tool/convert_openpi_way.py` | **move + wrapper** 到 `scripts/dataset/convert_openpi.py`；先保存小 HDF5 fixture 的输出。 | `T1,T9,T11` | `P-DATA` | `H0` |
| `dataset_tool/replay_hdf5_episodes.py` | **move + safety rewrite** 到 `scripts/dataset/replay_hdf5.py`；先加 main guard/dry-run，再注入 robot。 | `T6,T9,T11` | `P-DATA,P-CTRL` | `H-UR/H-RM`（低速、空场地） |
| `dataset_tool/replay_lerobot_episodes.py` | **split + wrapper**：reader/video/visualization/robot replay；先消除 1100+ 行单脚本和 import-time 执行。 | `T6,T9,T11,T12` | `P-DATA,P-CTRL` | `H-UR/H-RM`（低速） |
| `dataset_tool/SG_filter_for_action.py` | **move + wrapper** 到 `scripts/dataset/smooth_actions.py`；算法与 I/O 分离。 | `T9,T11` | `P-DATA` | `H0` |
| `dataset_tool/tag_HF.py` | **replace, no compatibility auto-run** 为 `scripts/dataset/tag_hub.py`；main guard、repo/tag 参数、dry-run/确认。 | `T11` + fake Hub | 远端 Hub 操作语义 | `H0` |
| `dataset_tool/Visualize_hdf5_episodes.py` | **move/split + wrapper** 到 `scripts/dataset/visualize_hdf5.py` 与共享 reader。 | `T9,T11,T12` | `P-DATA` | `H0` |
| `dataset_tool/Visualize_tactile_lero.py` | **move + wrapper** 到 `scripts/dataset/visualize_tactile.py`；41-taxel fixture 保留。 | `T8,T9,T11,T12` | `P-DATA,P-TACT` | `H0` |
| `dataset_tool/Visualize_torque_data.py` | **move + wrapper** 到 `scripts/diagnostics/visualize_torque.py`。 | `T11,T12` | 输入日志列名 | `H0` |

## 6. 测试与诊断迁移

| 当前文件 | 目标与策略 | 迁移前测试 | 协议 | 硬件 |
|---|---|---|---|---|
| `test_tool/__init__.py` | 工具迁空后 **later removal**。 | `T0` | `P0` | `H0` |
| `test_tool/camera_test.py` | **split** 为可自动化 hardware test 与人工 diagnostics CLI；旧路径 wrapper。 | `T1,T4,T5,T11` | `P-UDP,P-WEBRTC` | `H-CAM,H-VR` |
| `test_tool/eval_with_datasets.py` | **split + wrapper** 到 `policies/evaluation.py`、`apps/evaluate.py`；与 inference 共用 feature conversion。 | `T1,T9,T10,T11` | `P-DATA` | `H0` |
| `test_tool/ForceMode.py` | **move + safety rewrite, no auto-run wrapper** 到 `scripts/diagnostics/ur_force_mode.py`；显式 `--confirm-hardware`。 | `T6,T11` | `P-CTRL` | `H-UR` |
| `test_tool/ForceVisualize.py` | **split + wrapper**：wrench source、mock source、UI consumer、diagnostics app。 | `T6,T8,T11,T12` | `P-CTRL,P-TACT` | `H-UR,H-TAC,H-CAM` |
| `test_tool/freedrive.py` | **move + safety rewrite, no auto-run wrapper** 到 `scripts/diagnostics/ur_freedrive.py`；finally 退出 teach mode。 | `T6,T11` | `P-CTRL` | `H-UR` |
| `test_tool/TEST_csv.py` | **move + wrapper** 到 `scripts/diagnostics/plot_torque_csv.py`。 | `T11,T12` | CSV 列名 | `H0` |
| `test_tool/test_pt2.py` | **split**：纯函数/断言到 `tests/unit/test_trajectory_filter.py`，示例到 diagnostics；移除顶层执行。 | `T0,T6,T11` | `P-CTRL` | `H0` |
| `test_tool/vr_data.py` | **move + wrapper** 到 `scripts/diagnostics/vr_input.py`，只依赖 VR port。 | `T1,T3,T11,T12` | `P-VR,P-UDP` | `H-VR` |
| `tests/test_realman_teleop_loop.py` | **split** 为 CAN-FD executor unit、RealMan session integration、config defaults；保留原用例映射。 | 原 28 tests + `T2,T6,T7` | `P-CTRL` | `H0`；另加独立 `tests/hardware` |

## 7. 配置、文档与资产迁移

| 当前文件 | 目标与策略 | 迁移前测试 | 协议 | 硬件 |
|---|---|---|---|---|
| `.gitignore` | **keep/update**；增加 build/cache/knowledge-graph 约定，不忽略需发布的 fixtures。 | `T0` + ignored-file audit | `P0` | `H0` |
| `pyrightconfig.json` | **rewrite** 为 `src` layout，移除机器绝对路径；配置转 `pyproject.toml` 优先。 | `T0` | `P0` | `H0` |
| `README.md` | **rewrite incrementally**；每个入口/默认值由测试或配置生成，修正 `vr_data.py`、`eval_policy.py` 路径。 | 文档命令 smoke + `T1,T2` | 文档必须与所有兼容面一致 | 按所述命令分级标注 |
| `reminder.md` | **merge then remove** 到 `docs/diagnostics/tactile.md`。 | 链接/命令检查 | `P-TACT` | `H-TAC` |
| `requirements.txt` | **replace with temporary wrapper**：核心/optional extras 迁到 `pyproject.toml`，固定支持版本；旧文件可 `-e .[recommended]` 或说明。 | 安装矩阵 `T0` | `P0` | `H0` |
| `eval_dataset_plots/episode_000000_policy_vs_dataset.png` | **move** 到 `docs/assets/evaluation/episode_000000...png` 或发布 artifact。 | `T12` | `P0` | `H0` |
| `eval_dataset_plots/episode_000001_policy_vs_dataset.png` | **move** 到同一文档资产目录。 | `T12` | `P0` | `H0` |
| `eval_dataset_plots/episode_000002_policy_vs_dataset.png` | **move** 到同一文档资产目录。 | `T12` | `P0` | `H0` |
| `eval_dataset_plots/episode_000003_policy_vs_dataset.png` | **move** 到同一文档资产目录。 | `T12` | `P0` | `H0` |
| `eval_dataset_plots/episode_000004_policy_vs_dataset.png` | **move** 到同一文档资产目录。 | `T12` | `P0` | `H0` |
| `eval_dataset_plots/episode_000005_policy_vs_dataset.png` | **move** 到同一文档资产目录。 | `T12` | `P0` | `H0` |
| `eval_dataset_plots/episode_000006_policy_vs_dataset.png` | **move** 到同一文档资产目录。 | `T12` | `P0` | `H0` |

## 8. 实仓检查后对目标结构的修订

以下偏差应在实现前写入目标结构，而不是做到一半再决定：

1. **领域拥有 ports。** `robots/base.py`、`devices/*/base.py`、`streaming/video/base.py` 是各自接口真值；`core/interfaces.py` 只保留真正跨域接口或删除，避免重复协议。
2. **配置提前。** 原 Phase 13 调整到 core contract 之后、具体设备拆分之前；否则新 adapter 仍会依赖旧宽 `Config`。
3. **保留 ForceFlow++。** 增加 `policies/forceflowpp/` 和独立 policy migration phase；目标树当前遗漏它。
4. **增加 inference runtime。** 新增 `runtime/inference_session.py`，避免 `policies/inference.py` 拥有相机/机器人生命周期。
5. **补齐 wrench contract。** `devices/wrench/base.py`、`source.py`（或 backend capability）与 `filters.py`、`compensation.py` 分开。
6. **补齐 mocks。** 除 camera/robot/tactile/VR 外，还要有 video transport、recorder writer、visualizer publisher 的 mock/in-memory adapter。
7. **澄清 backend/executor。** backend 提供线程安全的原子 capability；executor 独占 command SDK 与 cadence。若 SDK 不线程安全，所有调用都经同一 executor/mailbox。
8. **增加通用 transport 层。** `udp_comms.py` 不完全等同 state protocol；建议 `streaming/transports/udp.py`，上层 state/video legacy adapter 组合它。
9. **资产不进运行 package。** 评估 PNG 移入 docs assets 或 CI artifact。
10. **脚本必须是安全入口。** 所有实机/远端脚本都有 main guard、dry-run/确认、明确 cleanup；不得保留“导入即执行”的兼容行为。

## 9. 兼容规范必须先补全的空白

### Binary state v2

在实现 `Binary State Protocol v2` 前写正式 spec 和 golden bytes，至少明确：

- magic、version、message type、flags；
- endian；
- header/payload length；
- sequence 位宽、wrap 和丢包判断；
- timestamp 的 clock domain、单位、同步/未知值；
- controller 与 26-joint hand 的字段顺序、float precision；
- checksum/完整性、最大 payload；
- v1/v2 negotiation 与未知版本行为。

### Watchdog

先定义各输入的 stale threshold、连续失败计数、hold pose/zero velocity/stop 的区别、恢复是否自动、需要多少新鲜帧、重连期间谁拥有命令权，以及 UR/RealMan 各自的安全动作。

### 性能

先在旧实现测量并记录 P50/P95/P99：camera capture→encode→send、VR RX→robot command、control-loop jitter、state age、recorder queue depth/drop、shutdown time。计划中的“low latency”不能作为可判定验收标准。

### H.264/WebRTC

先做独立 spike：比较 aiortc raw `VideoFrame`、外部硬件/软件 H.264 编码与可行的 packetization 接口，测量 copy 次数、keyframe/bitrate 控制、Quest compatibility。验证前不承诺“把预编码 H.264 帧直接喂给 aiortc”。

## 10. 最安全的修订实施顺序

1. **Gate 0：评审本 Phase 0。** 不改实现。
2. **保留历史并建立基线分支。** 克隆/worktree 到目标目录，确认旧命令、fixtures、Git 状态。
3. **测试先行。** 增加 protocol/data golden fixtures、Config snapshots、safe import tests；每一后续 phase 都同步测试，不能等到原 Phase 14。
4. **package skeleton + deprecation policy。** 建 `src`、`pyproject.toml`、optional extras、旧路径 shims；只做结构。
5. **core types/buffers/events/clocks/errors + domain ports。** 不引入 SDK。
6. **typed config/loader/factories。** 提前执行原 Phase 13；用 snapshots 证明默认行为不变。
7. **先抽纯契约。** VR parser、UDP packet codec、recording schema、teleop transforms/safety、wrench math；仍让旧入口调用它们。
8. **建立 mocks 与 lifecycle harness。** camera/VR/robot/tactile/video/recording/visualizer 全链路无硬件 integration。
9. **机器人逐后端迁移。** 先 abstraction 和 mock，再 UR，再 RealMan executor；RealMan 每一步都过 `T7`，安全行为不顺手修改。
10. **tactile 与 wrench。** 先 BLE4，旧 41-taxel 入 deprecated；移出 UI/shared holder。
11. **camera pipeline + legacy UDP。** acquisition/process/encode/transport 分离，旧 Unity JPEG golden 先过。
12. **H.264 feasibility spike 后迁 WebRTC。** 再拆 state 与 reliable commands；不在同一提交改变协议。
13. **mapping/safety/watchdog。** 用 fake executor 对照旧输出；watchdog 新行为单独评审。
14. **recording。** 分离 state/serializer/rollback，逐 schema fixture 对比且保留恢复策略。
15. **visualization。** 转为纯 consumer，command router 单独。
16. **ForceFlow++、evaluation、inference。** 先锁 LeRobot 版本与纯 policy 输出，再接 runtime；robot connect/move 必须后于 policy validation。
17. **薄 apps 与删除重复实现。** 只有所有兼容 tests、台架 tests 通过后，旧 manager 变纯 façade；任何 legacy 删除另行批准。
18. **持续文档/测量/硬件矩阵。** architecture、communication、extension guide 每阶段更新；最终才做 release gate。

每个阶段只允许一个主要责任变化，并按计划报告组件、依赖、去耦、兼容、测试与硬件状态。

## 11. 覆盖与门禁

- [x] 62/62 tracked 文件均有独立迁移行。
- [x] 每行均指定目标、策略、迁移前测试、协议与硬件验证。
- [x] 识别所有 root modules、tactile、ForceFlow++、数据脚本、诊断入口和 PNG。
- [x] 记录目标结构的必要修订和安全实施顺序。
- [x] 未执行 move/split/delete，也未修改旧仓库实现。
- [ ] 用户评审并明确批准后，才进入 Phase 1。
