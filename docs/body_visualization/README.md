# Meta 人体姿态可视化

`teleop_body_visualizer.py` 默认只打开人体骨架窗口，**不连接 RealMan、不启动机器人或相机，也不等待 teleop 手柄数据**。窗口显示 Meta 实际输出的身体关节，帮助检查肩、肘、腕、手指、躯干，以及全身模式下的腿部姿态。默认显示单个放大的 3D 骨架，可切换正面、侧面、俯视和四视图。

## 启动

推荐使用现有 teleop 环境。以下命令可以从任意目录执行：

```bash
# 只检查 Quest 人体数据，不需要 RealMan、相机或机器人会话
/home/yuyuan/.venvs/airo-teleop/bin/python /home/yuyuan/AIRO-Doffy/teleop_body_visualizer.py

# 需要同时运行 RealMan teleop 时才显式添加此参数
/home/yuyuan/.venvs/airo-teleop/bin/python /home/yuyuan/AIRO-Doffy/teleop_body_visualizer.py --teleop realman

# 合成姿态演示，不需要 Quest 或机器人
/home/yuyuan/.venvs/airo-teleop/bin/python /home/yuyuan/AIRO-Doffy/teleop_body_visualizer.py --demo

# 使用原 main.py teleop 入口
/home/yuyuan/.venvs/airo-teleop/bin/python /home/yuyuan/AIRO-Doffy/teleop_body_visualizer.py --teleop classic
```

根目录命令保持不变，也可从仓库目录使用包入口（参数相同）：

```bash
cd /home/yuyuan/AIRO-Doffy
/home/yuyuan/.venvs/airo-teleop/bin/python -m doffy_teleop.runtime.body_visualizer
```

可选参数：`--bind-ip 0.0.0.0`、`--body-port 8015`、`--hz 30`、`--stale-after 0.5`。修改端口时也要修改 Quest `BodyPoseTelemetrySender.destinationPort`。`--save-preview /absolute/path/preview.png` 输出明确标记 DEMO 的合成姿态 PNG，不启动机器人、不监听 UDP。

`--visualization-only` 仍可使用，与默认行为相同。只有指定 `--teleop realman` 或 `--teleop classic` 时，才会在可视化预检查完成后启动 teleop 子进程，沿用同一 Python 解释器和仓库目录。此模式下，关闭人体窗口或按 Ctrl-C 会向 teleop 发送 SIGINT，给现有运行时清理相机、机器人连接和录制的时间；teleop 提前退出也会关闭人体窗口并返回对应退出码。

## Quest 端

本次修改对应 `/home/yuyuan/UNITY_Project/CodexBracelet`：

- `Assets/Teleop/UpperLimb/UpperLimbAkmManager.cs`：提供当前已采样身体状态的只读副本，复用原来的追踪所有者。
- `Assets/Teleop/UpperLimb/BodyPoseTelemetrySender.cs`：场景加载后自动创建，默认关闭；在 Session 中开启发送后，以 25 Hz 向已配置 PC 地址的 UDP 8015 发送诊断数据。
- `Assets/Teleop/Protocol/BodyPoseWireFormat.cs`：BODY v1 JSON 序列化与固定 SDK 关节名称/顺序。

源文件及 `.meta` 副本保存在 [quest](quest/) 目录。已有项目应使用上面的实际工程；副本用于查阅和移植，不能直接用完整 manager 覆盖另一版本的自定义修改。

在 Quest 的 **Session** 页面设置并 Apply 这台 PC 的 IP，然后将 **Body data: OFF** 切换为 **Body data: ON**。每次启动应用默认关闭，关闭按钮立即停止 BODY 数据并关闭诊断 socket。该开关独立于 Start/Stop session、WRM、手柄和手部控制数据；**只查看人体姿态时无需按 Start session、开启 WRM 或完成机器人标定**。PC 的 BODY 默认监听所有本机网卡。需要身体追踪权限、佩戴头显并保持应用获得焦点。

使用仓库提供的 [v0.9.7/code18 Android ARM64 APK](../../apk/AIRO_Doffy_v0.9.7_arm64_code18.apk)，包名为 `com.AIROLab.AIRODOFFY`，包含 BODY 发送代码。可从仓库根目录执行 `adb install -r apk/AIRO_Doffy_v0.9.7_arm64_code18.apk` 安装。旧版自动发送 BODY 的行为已改为手动开关；更早版本没有 BODY 数据。该 APK 已通过签名验证及 Quest 3 安装/启动检查，头显界面、交互和真实人体姿态仍未验收；具体记录见 [validation.md](validation.md#097-发布-apk)。

当前默认是 Meta `UpperBody`，包含 **70 个身体/手部关节**，不含腿部。窗口标记 `UPPER BODY · legs unavailable`，不会根据肘部进度、手柄位置或模板补出腿部。需要腿部时，在 Unity 的 `UpperLimbAkmManager` 上于追踪启动前启用 `requestFullBodyTracking`，并按平台要求配置全身追踪、重建 APK。支持时发送 `FullBody` 的 **84 个关节**；启动请求不支持时回退 UpperBody。运行中改变该字段不会重启追踪。

身体位置使用 Unity world 坐标，与原 WRM 肩肘位置一致：X 向右、Y 向上、Z 向前，单位米。旋转是 XYZW 四元数。现有控制用的手部流可能经过 teleop 参考系变换，不能直接与这个 world 坐标骨架叠加。人体窗口的手指来自同一 BODY 帧，因此与躯干使用相同坐标。

## 查看数据

- 左侧身体为青色，右侧为橙色。默认显示一个大骨架，3D 图可拖动旋转。
- `3D` / `Front` / `Side` / `Top` 按钮选择单个放大视图；`Four views` / `Single skeleton` 在四视图和单骨架之间切换。
- `Zoom +` / `Zoom -` 放大缩小；`Fit pose` 恢复自动适配和跟随。上半身会按实际范围适配，不再固定占用完整身体的视野。
- `Finger joints` 控制手指显示；`Joint names` 标记主要关节；`Joint axes (XYZ)` 显示有效四元数对应的红 X、绿 Y、蓝 Z 轴；`Follow hips` 让视窗跟随人体。
- 顶部显示 frame ID、接收频率、PC 接收后的帧龄、有效关节数、数据源、丢弃的错误包数和 SDK confidence。
- `LIVE` 为可用的新姿态。追踪失效、置信度不高于 0.5、无有效位置或断流时，保持最后一次有效姿态，显示 `HOLDING` 或 `STALE` 并将骨架变淡；恢复有效追踪后自动更新。尚未收到过有效姿态时保持空白。
- 新版 APP 区分 SDK 的 `valid` 与 `tracked`：坐标有效也可能是失去追踪后的推测值。此前主动追踪过的肩臂/腕关节变为 untracked 时，冻结该侧整条手臂；个别无效或缺失关节保留历史位置。没有主动追踪历史的 SDK 推测关节仍可显示，例如全身模型的腿部。切换发送端或上半身/全身模式会清除历史，避免继承其他来源的姿态。
- 顶部诊断仍显示当前收到的 frame、confidence、有效关节数；冻结时另外显示保持的关节数和有效姿态帧龄。角度从正在显示的姿态计算，冻结的姿态不会被当成 `LIVE`。原点坐标不会被误当成追踪失败。
- 抬臂角相对于世界竖直向下方向，下垂 0°、水平 90°。肘/膝显示内角，伸直 180°，这些是从关节位置计算的几何角度。底部同时显示该 BODY 帧对应的 WRM alpha 和标定状态。

独立诊断端口不会与控制器/手部 UDP 8001 或 WRM/control UDP 8005 争抢数据。仅显式启用 teleop 时，人体渲染与 robot teleop 分属两个进程。接收频率由实际数据决定，`--hz` 仅控制窗口刷新频率。

## PC 模块职责

项目对外名称为 `doffy-teleop`，Python 模块名使用下划线 `doffy_teleop`。实现位于该包内。`teleop_body_visualizer.py` 只调用包入口，`doffy_teleop/body_visualization.py` 保留模块化前 BODY API 的兼容导出。

| 模块 | 职责 |
| --- | --- |
| `protocol/body.py` | BODY v1 数据结构、关节名称/ID 和严格解析 |
| `media/body.py` | 独立 BODY UDP 接收、最新帧和接收统计 |
| `visualization/body_geometry.py` | 骨架连线、坐标投影和关节角度 |
| `visualization/body.py` | `BodyDashboard`、Matplotlib 绘制和窗口控件 |
| `visualization/body_demo.py` | 合成 BODY 演示帧 |
| `visualization/body_pose.py` | 显示姿态缓存、追踪失效冻结与恢复 |
| `runtime/body_viewer.py` | 窗口刷新循环、预检查和接收/窗口资源清理 |
| `runtime/body_visualizer.py` | CLI 参数及可选 teleop 子进程生命周期 |

表中路径相对于 `doffy_teleop/`。运行时和绘图模块直接依赖各自的实现模块，兼容导出不参与内部依赖。

## 数据协议

每帧一个 UTF-8 JSON UDP 数据报，最大 65507 字节。`joints` 使用 SDK 的固定 ID，UpperBody 为 0–69，FullBody 为 0–83；`name` 去掉 `Body_`/`FullBody_` 前缀。例如 Head=7、LeftHandWrist=19、RightHandWrist=45、LeftLowerLeg=71。

```json
{
  "type": "BODY", "version": 1, "frame_id": 42, "timestamp_ns": 1791200000000000000,
  "joint_set": "upper_body", "coordinate_space": "unity_world",
  "confidence": 0.95, "tracking_valid": true,
  "joints": [
    {"id": 7, "name": "Head", "position": [0.0, 1.7, 0.1],
     "rotation": [0.0, 0.0, 0.0, 1.0], "position_valid": true, "orientation_valid": true,
     "position_tracked": true, "orientation_tracked": true}
  ],
  "wrm": {"elbow_alpha": 0.4, "confidence": 0.95, "enabled": false, "calibrated": false}
}
```

无效位置或旋转发送 `null` 和相应 `false`。PC 拒绝非有限数、重复/越界 joint ID、错误名称、错误坐标空间和不兼容协议版本。SDK confidence 表示 SDK 输出状态，不能单独证明每个关节与人的真实姿态一致；可视化的目的就是检查该输出。

`position_tracked` / `orientation_tracked` 为 BODY v1 的新增可选布尔字段。0.9.6 及更新版 APP 保留 SDK 的实际标记，BODY 显示质量采用 Meta `OVRBody` 的高置信度门槛 `confidence > 0.5`；低质量帧仍发送失效通知和诊断置信度。此规则仅用于 BODY 可视化，不改变 WRM 或机器人控制。旧 APP 缺少 tracked 字段时仍可解析，但只能按原有效性和置信度冻结，无法识别高置信度的 untracked 回退姿态。

## 验证

```bash
cd /home/yuyuan/AIRO-Doffy
/home/yuyuan/.venvs/airo-teleop/bin/python -m pytest -q \
  tests/test_body_visualization.py tests/test_teleop_body_visualizer.py \
  tests/test_body_module_imports.py tests/test_teleop_refactor_boundaries.py \
  tests/test_body_pose_hold.py tests/test_body_tracking_protocol.py
/home/yuyuan/.venvs/airo-teleop/bin/python teleop_body_visualizer.py \
  --save-preview output/body_visualization/preview.png
```

软件验证记录和 APK 构建结果见 [validation.md](validation.md)。合成预览、C# 编译和本机 UDP 回环不等同于实际 Quest 姿态验证。
