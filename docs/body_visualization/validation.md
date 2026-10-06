# 验证记录

> Public copy: local paths, device identifiers and site addresses are anonymized; recorded results and version/hash data are unchanged.

历史实现记录日期：2026-10-05；发布 APK 信息更新于 2026-10-06。下方保留各次实现和安装记录；最新发布 APK 见文末 **0.9.7 发布 APK**，BODY 追踪失效冻结的实现记录见 **0.9.6 追踪失效冻结**。未运行机器人 teleop。

## 软件检查

- Python：`./.venv/bin/python -m pytest -q tests/test_body_visualization.py tests/test_teleop_body_visualizer.py`：**103 passed**。覆盖严格协议解析、本机真实 UDP、无效/缺失关节、断流、角度与投影、控件、可视化启动前失败、teleop 子进程监视与退出清理。
- GUI：TkAgg 合成动画可打开，并通过退出回调关闭；生成的本地 `output/body_visualization/preview.png` 使用明确标注的合成数据，不随源码上传。已检查四个视图、底部角度和控件布局。
- C# runtime：`scripts/teleop_refactor/compile_unity_sources.py --project /path/to/unity/CodexBracelet --reference /path/to/unity/CodexBracelet --output /tmp/codex-bracelet-body-compile`：**99 个 runtime 源文件编译通过**。
- 协议：`docs/body_visualization/quest/check_body_wire_format.py`：**7 个严格 JSON 场景通过**，核对已安装 Meta SDK 的 70/84 个 joint 名称/ID、fr-FR 小数序列化、追踪失效、非有限数、有效零坐标，以及 8610–18601 字节的数据报大小。
- 互操作：上述生产 C# 编码器输出的 **7 个数据报** 经真实 localhost UDP 传到生产 Python `BodyReceiver`，全部接受，拒绝数 0。

## Android 构建

使用已有 `Doffy.Editor.TeleopBuild.BuildQuest` 完成 Unity Android ARM64 构建和场景检查，进程退出码 0。

- APK：`/path/to/unity/App_output/body-visualization/AIRO_Doffy_bracelet_body_visualization.apk`
- 包名：`org.airolab.doffy.bracelet`，版本 `0.9.4` / versionCode `13`，沿用现有项目版本设置。
- 大小：93,415,049 字节。
- SHA256：`8a937327f924fc6632c333b18d808f359d53ba6dd9b1f0bf7ac2d7d0f0255d79`。
- 日志：`/tmp/doffy-bracelet-body-visualization-build.log`，包含 `DOFFY Quest build succeeded`。
- APK manifest 已确认 `com.oculus.permission.BODY_TRACKING` 和 `arm64-v8a`。
- 构建前 APK 备份：`/path/to/unity/App_output/before-body-visualization/1277555568d5f58fe885cd2c1e28116cbc60fca7da1be5b480660b335bd6d184/AIRO_Doffy_bracelet.apk`。

安装到已连接的 Quest 后即可运行新发送器：

```bash
adb install -r /path/to/unity/App_output/body-visualization/AIRO_Doffy_bracelet_body_visualization.apk
```

## 尚需实际设备检查

佩戴 Quest、授予身体追踪权限、在连接界面设置本机 IP，启动 PC 的 `--visualization-only`。确认状态为 LIVE、接收频率接近 25 Hz，分别检查抬臂、屈肘、手指动作、转身和走动；失焦/摘下头显应显示 INVALID 或随后 STALE。只有实际收到腿部有效坐标后，才能验收全身姿态。机器人联动和实时 CAN-FD 时序尚未进行硬件验收。

## 0.9.5 修订：独立查看、大骨架与 Session 发送开关

- 默认命令只运行身体 UDP 接收和 GUI，机器人 teleop 改为显式 `--teleop realman|classic`。实际 TkAgg 窗口检查确认无机器人模块导入、无 teleop 子进程启动。
- 默认单个大骨架；实际鼠标事件验证四视图/单骨架、3D/Front/Side/Top、Zoom +/-、Fit pose。上半身按实际关节范围适配；二维投影也能显示关节名称和方向轴。
- Python：上述两组测试 **113 passed**，包括新增显示按钮、投影标签/方向轴，以及默认禁止机器人启动的检查。
- C#：**99 runtime + 4 Editor 文件**编译通过，7 个 BODY wire 检查通过。
- Unity native Play：`WristUIValidation.RunPlaySmoke` 退出码 0；日志 `/tmp/doffy-bracelet-v095-body-play-smoke.log`。真实 Meta ray/poke 事件经 Canvas 路由到 Session 新按钮，检查默认 OFF 无 socket/无数据、ON 真实本机 UDP、OFF 立即关 socket 且不发 heartbeat、重新开启、页面关闭重开、组件停用和重建。机器人会话、WRM、标定和追踪模式保持原状态。此 headless 检查仍有原有缺失 Meta native 库日志，不能替代头显中的物理交互验收。
- Android ARM64：Unity 构建退出码 0，日志 `/tmp/doffy-bracelet-v095-body-build.log`。
- 当次 APK：`/path/to/unity/App_output/body-visualization-v0.9.5/AIRO_Doffy_bracelet_body_visualization.apk`，版本 **0.9.5 / code 14**，93,413,329 字节。
- SHA256：`555650d4b51f48fd5dfbbe05d21cead287b1b261c2d4c357bc70b9876ba3e49c`。
- 已通过 `adb install -r` 安装到 Quest 3 `QUEST_SERIAL_REDACTED`，返回 Success；设备查询确认 0.9.5/code14，BODY_TRACKING 权限 granted=true。
- 应用冷启动 `am start -W` 返回 Status ok，进程仍运行；所检查的启动日志没有 BODY sender 的 NullReference/MissingReference 或 fatal 异常。

使用时先在头显 Session 中设置并 Apply 本机 IP，再点击 **Body data: OFF → ON**。只查看姿态不需要 Start session。关闭 BODY 开关后，PC 会在默认 0.5 s 内显示 STALE。真实人体姿态和佩戴头显时的按钮触感仍需实际检查。

## PC 模块化整理

- 实现按职责移入当前 `doffy_teleop` 的 `protocol`、`media`、`visualization` 和 `runtime`，详见 [模块职责](README.md#pc-模块职责)。根目录 `teleop_body_visualizer.py` 为 8 行入口；原 `doffy_teleop.body_visualization` 为 23 行兼容导出。原根目录命令及默认独立查看行为保持一致。
- BODY、启动生命周期、架构边界及原媒体/协议/兼容导入测试：**140 passed, 1 skipped**。跳过的是现有 Unity JPEG C# 回环测试，缺少 `/tmp/teleop-protocol-tests.exe` 测试程序。
- 新增独立导入测试：**2 passed**。`python -S` 下 BODY 解析/接收模块不打开 socket，不加载 NumPy、Matplotlib、相机、WebRTC 或机器人依赖；原协议/媒体公开导出保持可用。另在完整 teleop 环境中确认 **27 个实际公开导出**均指向原定义。
- 根脚本从其他目录运行和包入口的 `--help` 均在 `python -S` 下通过。启动测试直接验证迁移后的实现模块及原 teleop 脚本路径。
- 实际 TkAgg 独立窗口启动/关闭检查通过：没有 teleop 子进程和机器人/相机/WebRTC 模块导入，窗口和 BODY socket 正常清理。根入口成功生成 DEMO PNG 预览。

## 0.9.6 追踪失效冻结

- SDK 坐标有效不代表仍在主动追踪。APP 的 BODY 报文新增 `position_tracked` / `orientation_tracked`，保留实际 SDK 标记；BODY 显示质量采用 Meta `OVRBody` 的门槛 `confidence > 0.5`。失效时继续发送失效通知和诊断 confidence；WRM 采样、标定和机器人控制逻辑未修改。
- PC 的 `visualization/body_pose.py` 缓存显示姿态。追踪失效、低置信度、空位置和断流时保留历史；此前已追踪的手臂关键关节转为 untracked 时冻结整条手臂。恢复后自动更新，界面分别显示新报文诊断与 `HOLDING` / `STALE`。最初无有效姿态时保持空白；不同发送端或 joint_set 不继承历史。
- Python 六组测试 **170 passed**：BODY、tracked 字段、冻结/恢复、启动生命周期、独立导入及模块边界。包含低置信度下垂坐标、连续失效、已追踪手臂回退、缺失关节、投影视图/缩放、恢复，以及全身推测腿部继续更新的场景。
- C# **99 runtime + 4 Editor** 编译通过；wire 检查 **8 场景**通过，最大数据报 22,885 字节；生产 sender 的纯 C# 适配测试 **13 场景**通过，覆盖置信度边界、失焦/暂停、恢复、异常数值、有效但 untracked 的 SDK 数据及 WRM 诊断。
- 上述生产 C# sender 输出的 **13 个数据报**经真实 localhost UDP 传给 Python `BodyReceiver`，全部接受、拒绝 0；再交给生产 `BodyPoseHold` 验证冻结与恢复。
- 实际 TkAgg 窗口和 localhost UDP 联合检查通过：先接收有效姿态，再接收低置信度下垂回退，窗口坐标保持一致且显示 `HOLDING`；恢复高质量数据后回到 `LIVE` 并更新坐标，退出正常释放资源。
- 本地冻结预览 `output/body_visualization/tracking_hold_preview.png` 明确标记 DEMO，不随源码上传；已检查冻结骨架、状态、原始 confidence 和帧龄显示。
- Android ARM64 构建退出码 0；日志 `/tmp/doffy-bracelet-v096-body-hold-build.log` 包含 `DOFFY Quest build succeeded`。
- APK：`/path/to/unity/App_output/body-visualization-v0.9.6/AIRO_Doffy_bracelet_body_visualization.apk`，包名 `org.airolab.doffy.bracelet`，**0.9.6 / code 15**，93,417,833 字节，BODY_TRACKING 权限声明存在。
- SHA256：`0c046644dde56e557ed69857565001139ab5d6e9a37c4e989b74d71dbc296f54`。
- 安装尚未执行：构建后 `adb devices` 未识别到连接的 Quest。上一条已确认的设备安装记录为 0.9.5/code14；0.9.6 APK 已就绪，需连接设备后安装并实际观察 SDK 的手臂追踪标记。

## 0.9.7 发布 APK

- 发布文件：[AIRO_Doffy_v0.9.7_arm64_code18.apk](../../apk/AIRO_Doffy_v0.9.7_arm64_code18.apk)，Android ARM64，版本 **0.9.7 / code 18**，包名 `com.AIROLab.AIRODOFFY`。
- 大小：92,549,501 字节；SHA256：`3a1e95322c83c865ba6729a5317bcc06b60875241c78c03d8e28752db6d24a4e`。机器可读信息见 [APK manifest](../../apk/manifest.json)。
- 包含 BODY 发送代码，签名验证通过，Quest 3 安装及应用启动成功。
- 每次启动 BODY 发送默认 **OFF**。在 Session 页面设置并 Apply PC 地址，再切换 **Body data: OFF → ON**；独立查看人体姿态无需 Start session。
- 头显实际界面、交互和真实 BODY 姿态仍未验收。签名、安装和启动成功不能替代这些检查，也不能替代机器人联动及 CAN-FD 时序验收。

从仓库根目录安装：

```bash
adb install -r apk/AIRO_Doffy_v0.9.7_arm64_code18.apk
```
