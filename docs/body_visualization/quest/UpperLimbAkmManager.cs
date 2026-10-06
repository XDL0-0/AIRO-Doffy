using System;
using System.Globalization;
using System.Text;
using Doffy.Protocol;
using UnityEngine;

/// <summary>
/// Meta upper-body sampling and two-pose calibration for RM75 WRM.
/// Alpha uses fixed shoulder–elbow vertical height: down=0, shoulder height=1.
/// Forward and side raises both reach 1. Robot IK remains on the workstation.
/// </summary>
public class UpperLimbAkmManager : MonoBehaviour
{
    /// <summary>标定状态机。</summary>
    public enum CalibrationState
    {
        Idle,                    // 未开始标定
        WaitingDown,             // 等待：摁右手A键进行 elbow down calibration
        CalibratingDown,         // 按住A键采样中（elbow_down）
        WaitingHorizontal,       // 等待：摁右手A键进行侧平举（side raise）标定
        CalibratingHorizontal,   // 按住A键采样中（side raise）
        Calibrated               // 标定完成
    }

    // ===== 可调参数（公开字段，可在 Inspector 覆盖；自举创建时用默认值）=====
    [Header("Calibration")]
    [Tooltip("Hold right-hand A to sample for this duration (seconds); completes the current pose")]
    public float calibrationHoldSeconds = 2f;
    [Tooltip("Angle below this is a degenerate calibration; requires recalibration")]
    public float calibrationMinAngleDeg = 5f;

    [Header("elbow_alpha Filtering")]
    [Tooltip("Dead zone: alpha below this is forced to zero (anti-jitter near rest pose)")]
    public float alphaDeadZone = 0.03f;
    [Tooltip("Full zone: alpha above (1 - this) is forced to one (anti-jitter near shoulder-level pose)")]
    public float alphaFullZone = 0.03f;
    [Tooltip("Low-pass time constant tau (s); smaller = faster response")]
    public float alphaFilterTau = 0.12f;

    [Header("UDP Send")]
    [Tooltip("Send rate (Hz)")]
    public float sendRateHz = 30f;

    [Header("Body Tracking")]
    [Tooltip("Opt in before tracking starts: request FullBody and fall back to UpperBody if unavailable. Changing this during tracking does not restart it.")]
    public bool requestFullBodyTracking;

    [Header("Plane Reference")]
    [Tooltip("DEPRECATED: 平面投影方案已移除(实测真实前平举在三种平面变体下都读出 0.7-0.9,无法区分)。保留字段仅为序列化兼容,不再参与计算。")]
    public int planeMode = 1;

    [Header("Vertical Position")]
    [Tooltip("DEPRECATED: 运行时不再跟随下垂基准，保留字段仅为序列化兼容")]
    public float hangTrackTau = 0.5f;
    [Tooltip("标定退化阈值(m):两次标定(下垂/水平)的肩-肘高度差必须相差 ≥ 该值,否则判为退化标定")]
    public float minHeightSpan = 0.1f;
    [Tooltip("DEPRECATED: 运行时不再跟随下垂基准，保留字段仅为序列化兼容")]
    public float hangTrackMargin = 0.05f;

    [Header("Button Mapping")]
    [Tooltip("Calibration confirm button (hold to sample); shares right-hand A with legacy UdpWindowManager.FocusControl, use per spec")]
    public OVRInput.Button calibrateButton = OVRInput.Button.One;
    [Tooltip("Clutch button (held: keeps sending 1)")]
    public OVRInput.Button clutchButton = OVRInput.Button.PrimaryHandTrigger;
    [Tooltip("Recenter button (press edge sends 1 pulse)")]
    public OVRInput.Button recenterButton = OVRInput.Button.PrimaryThumbstick;
    public OVRInput.Controller controller = OVRInput.Controller.RTouch;

    [Header("Debug")]
    [Tooltip("调试：每 0.5s 打印关节世界位置与 alpha 中间量，用于核对 Meta 关节数据构成")]
    public bool logJointDebug;

    // ===== 对外状态 =====
    public CalibrationState State { get; private set; } = CalibrationState.Idle;
    public bool IsCalibrated => State == CalibrationState.Calibrated;
    public bool WrmEnabled { get; private set; }

    /// <summary>身体追踪数据是否有效（权限已授予、追踪已启动、关节位置有效）。</summary>
    public bool IsTrackingValid { get; private set; }

    /// <summary>身体追踪置信度（0~1，无效时为 0）。</summary>
    public float Confidence { get; private set; }

    /// <summary>右肩/左肩/右肘/右腕世界系位置（无效时为 zero）。</summary>
    public Vector3 ShoulderPos { get; private set; }
    public Vector3 LeftShoulderPos { get; private set; }
    public Vector3 ElbowPos { get; private set; }
    public Vector3 WristPos { get; private set; }

    /// <summary>最终 elbow_alpha（已 clip + 死区 + 低通 + smoothstep），未标定时为 0。</summary>
    public float ElbowAlpha => _heightMapping.Value;

    /// <summary>clutch（握把按住）。</summary>
    public bool IsClutched { get; private set; }

    // ===== 内部状态 =====
    private OVRPlugin.BodyState _bodyState;
    private bool _trackingStarted;
    private bool _bodyPoseSampleValid;
    private bool _fullBodyUnavailable;
    private OVRPlugin.BodyJointSet _activeBodyJointSet = OVRPlugin.BodyJointSet.UpperBody;
    private bool _permissionRequested;
    private OVRCameraRig _cameraRig;
    private Transform _trackingSpace;
    private Matrix4x4 _sampleTrackingToWorld = Matrix4x4.identity;
    private Quaternion _sampleTrackingRotation = Quaternion.identity;
    private UdpSocket _udpSocket;

    // 标定采样
    private Vector3 _downDir;    // 标定：自然下垂时 shoulder→elbow 单位方向
    private Vector3 _horizDir;   // 标定：上臂水平时 shoulder→elbow 单位方向
    private Vector3 _sampleSum;  // 采样窗口内方向向量累加和
    private int _sampleCount;
    private float _holdTimer;

    // 垂直位置方案：alpha = (下垂高度差 - 当前高度差) / 下垂高度差，
    // 高度差 = 肩.y - 肘.y（下垂为正，水平≈0）。无参考系、转身免疫。
    private float _hangHeightDiff;  // 标定下垂：肩-肘垂直差（肘低于肩为正，标定后固定）
    private float _sideHeightDiff;  // 标定水平（侧平举）：肩-肘垂直差（≈0）
    private float _heightSum;       // 标定采样窗口内 (肩.y-肘.y) 累加

    // 滤波
    private readonly UpperLimbHeightMapping _heightMapping = new UpperLimbHeightMapping();

    // UDP
    private float _sendTimer;
    private int _frameCounter;
    private bool _recenterPulse;
    private readonly StringBuilder _sb = new StringBuilder(128);

    // 调试
    private float _debugTimer;

    /// <summary>Unix 纪元 Ticks（DateTime.UtcNow.Ticks 为 100ns 步长）。</summary>
    private static readonly long UnixEpochTicks =
        new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc).Ticks;

    /// <summary>
    /// A read-only copy of the most recent sample used by WRM. No SDK polling or
    /// tracking startup occurs here. Joint values are copied only when requested;
    /// a failed poll never exposes locations left over from a previous frame.
    /// </summary>
    public readonly struct BodyPoseSnapshot
    {
        private readonly OVRPlugin.BodyJointLocation[] _jointLocations;
        public bool TrackingValid { get; }
        public float Confidence { get; }
        public double SampleTime { get; }
        public OVRPlugin.BodyJointSet JointSet { get; }
        public Transform TrackingSpace { get; }
        public Matrix4x4 TrackingToWorld { get; }
        public Quaternion TrackingRotation { get; }
        public int JointCount => _jointLocations == null ? 0 : _jointLocations.Length;

        internal BodyPoseSnapshot(OVRPlugin.BodyState state, bool valid,
            OVRPlugin.BodyJointSet jointSet, Transform trackingSpace,
            Matrix4x4 trackingToWorld, Quaternion trackingRotation)
        {
            TrackingValid = valid && trackingSpace != null;
            Confidence = TrackingValid ? state.Confidence : 0f;
            SampleTime = state.Time;
            JointSet = jointSet;
            TrackingSpace = trackingSpace;
            TrackingToWorld = trackingToWorld;
            TrackingRotation = trackingRotation;
            _jointLocations = TrackingValid && state.JointLocations != null
                ? (OVRPlugin.BodyJointLocation[])state.JointLocations.Clone() : null;
        }

        public OVRPlugin.BodyJointLocation GetJointLocation(int index)
        {
            return _jointLocations != null && index >= 0 && index < _jointLocations.Length
                ? _jointLocations[index] : OVRPlugin.BodyJointLocation.invalid;
        }
    }

    public BodyPoseSnapshot CaptureBodyPoseSnapshot()
    {
        // The body's SDK validity is independent of WRM's four-joint/calibration
        // requirements; the visualization remains available while WRM is idle.
        OVRPlugin.BodyJointSet sampledSet = _bodyState.JointLocations != null &&
            _bodyState.JointLocations.Length == (int)OVRPlugin.BoneId.FullBody_End
            ? OVRPlugin.BodyJointSet.FullBody : _activeBodyJointSet;
        return new BodyPoseSnapshot(_bodyState, _bodyPoseSampleValid && isActiveAndEnabled,
            sampledSet, _trackingSpace, _sampleTrackingToWorld, _sampleTrackingRotation);
    }

    private void Awake()
    {
        State = CalibrationState.Idle;
        if (!OVRPlugin.bodyTrackingSupported)
        {
            LogManager.Log("UpperLimbAKM", "Body tracking not supported, AKM unavailable");
            return;
        }

        // 申请身体追踪权限（系统弹窗）；授予后在 Update 中启动追踪
        if (!_permissionRequested)
        {
            OVRPermissionsRequester.Request(new[] { OVRPermissionsRequester.Permission.BodyTracking });
            _permissionRequested = true;
        }
    }

    private void Update()
    {
        float dt = Time.deltaTime;

        EnsureTrackingStarted();
        PollBodyState();

        UpdateCalibration(dt);
        UpdateAlpha(dt);
        UpdateInputs();
        UpdateSend();
        UpdateJointDebug(dt);
    }

    /// <summary>调试：降频打印关节世界位置与标定方向，核对 Meta 关节数据构成（肩点漂移排查）。</summary>
    private void UpdateJointDebug(float dt)
    {
        if (!logJointDebug) return;

        _debugTimer += dt;
        if (_debugTimer < 0.5f) return;
        _debugTimer = 0f;

        if (_bodyState.JointLocations == null) return;

        Vector3 Chest = JointWorld(OVRPlugin.BoneId.Body_Chest);
        Vector3 Scapula = JointWorld(OVRPlugin.BoneId.Body_RightScapula);
        Vector3 ArmUp = JointWorld(OVRPlugin.BoneId.Body_RightArmUpper);

        LogManager.Log("UpperLimbAKM",
            $"RS={ShoulderPos} LS={LeftShoulderPos} Scap={Scapula} ArmUp={ArmUp} Chest={Chest} RE={ElbowPos} RW={WristPos}");
        if (IsCalibrated && ShoulderPos.sqrMagnitude > 1e-6f && ElbowPos.sqrMagnitude > 1e-6f)
        {
            // 垂直位置方案中间量：diff(肩-肘垂直差)、hangDiff(标定下垂)、sideDiff(标定水平)、span、alphaRaw
            float dbgDiff = ShoulderPos.y - ElbowPos.y;
            float dbgSpan = _hangHeightDiff - _sideHeightDiff;
            float dbgRaw = _hangHeightDiff <= 1e-4f ? -1f
                : Mathf.Clamp01((_hangHeightDiff - dbgDiff) / _hangHeightDiff);
            LogManager.Log("UpperLimbAKM",
                $"down={_downDir} horiz={_horizDir} diff={dbgDiff:F3} hangDiff={_hangHeightDiff:F3} sideDiff={_sideHeightDiff:F3} span={dbgSpan:F3} raw={dbgRaw:F3} alpha={ElbowAlpha:F3} conf={Confidence:F2}");
        }
    }

    /// <summary>调试辅助：取任意关节的世界系朝向（无效时 identity）。</summary>
    private Quaternion JointOrientationWorld(OVRPlugin.BoneId boneId)
    {
        int index = (int)boneId;
        if (_bodyState.JointLocations == null || index < 0 || index >= _bodyState.JointLocations.Length)
            return Quaternion.identity;
        OVRPlugin.BodyJointLocation loc = _bodyState.JointLocations[index];
        if (!loc.OrientationValid || _trackingSpace == null) return Quaternion.identity;
        return _trackingSpace.rotation * loc.Pose.Orientation.FromFlippedZQuatf();
    }

    /// <summary>调试辅助：取任意关节的世界系位置（无效时 zero）。</summary>
    private Vector3 JointWorld(OVRPlugin.BoneId boneId)
    {
        int index = (int)boneId;
        if (_bodyState.JointLocations == null || index < 0 || index >= _bodyState.JointLocations.Length)
            return Vector3.zero;
        OVRPlugin.BodyJointLocation loc = _bodyState.JointLocations[index];
        if (!loc.PositionValid || _trackingSpace == null) return Vector3.zero;
        return _trackingSpace.localToWorldMatrix.MultiplyPoint3x4(loc.Pose.Position.FromFlippedZVector3f());
    }

    /// <summary>权限授予后启动身体追踪（仅一次）。</summary>
    private void EnsureTrackingStarted()
    {
        if (_trackingStarted || !OVRPlugin.bodyTrackingSupported) return;

        if (!OVRPermissionsRequester.IsPermissionGranted(OVRPermissionsRequester.Permission.BodyTracking))
            return;

        if (OVRPlugin.nativeXrApi != OVRPlugin.XrApi.OpenXR)
        {
            LogManager.Log("UpperLimbAKM", "Body tracking requires OpenXR, current XrApi unavailable");
            _trackingStarted = true; // 阻止反复重试
            return;
        }

        // Full-body tracking is opt-in and selected only before startup. Never
        // stop/restart tracking to satisfy a diagnostic telemetry consumer.
        if (requestFullBodyTracking && !_fullBodyUnavailable)
        {
            if (OVRPlugin.StartBodyTracking2(OVRPlugin.BodyJointSet.FullBody))
            {
                _activeBodyJointSet = OVRPlugin.BodyJointSet.FullBody;
                _trackingStarted = true;
                LogManager.Log("UpperLimbAKM", "Full Body tracking started");
                return;
            }
            _fullBodyUnavailable = true;
            LogManager.Log("UpperLimbAKM", "Full Body unavailable, falling back to Upper Body");
        }

        if (OVRPlugin.StartBodyTracking2(OVRPlugin.BodyJointSet.UpperBody))
        {
            _activeBodyJointSet = OVRPlugin.BodyJointSet.UpperBody;
            _trackingStarted = true;
            LogManager.Log("UpperLimbAKM", "Upper Body tracking started");
        }
        else
        {
            LogManager.Log("UpperLimbAKM", "StartBodyTracking2 failed, body tracking not started");
        }
    }

    /// <summary>每帧读取身体骨骼位姿（tracking space），并转换到世界系。</summary>
    private void PollBodyState()
    {
        ShoulderPos = LeftShoulderPos = ElbowPos = WristPos = Vector3.zero;
        IsTrackingValid = false;
        Confidence = 0f;
        _bodyPoseSampleValid = false;

        if (!_trackingStarted) return;

        if (!OVRPlugin.GetBodyState4(OVRPlugin.Step.Render, _activeBodyJointSet, ref _bodyState))
        {
            IsTrackingValid = false;
            return;
        }

        Confidence = _bodyState.Confidence;

        if (_cameraRig == null)
            _cameraRig = FindAnyObjectByType<OVRCameraRig>();
        if (_cameraRig == null || _cameraRig.trackingSpace == null) return;
        _trackingSpace = _cameraRig.trackingSpace;
        // Capture the same reference transform used for WRM below. LateUpdate
        // telemetry stays aligned even if another component moves the rig later.
        _sampleTrackingToWorld = _trackingSpace.localToWorldMatrix;
        _sampleTrackingRotation = _trackingSpace.rotation;
        _bodyPoseSampleValid = _bodyState.JointLocations != null &&
            !float.IsNaN(Confidence) && !float.IsInfinity(Confidence) && Confidence > 0f;
        if (_bodyState.JointLocations == null) return;

        // 骨骼数据在 tracking space，转世界系（与 OVRSkeleton 内部转换一致，无镜像）
        Vector3 GetJointPos(OVRPlugin.BoneId boneId)
        {
            int index = (int)boneId;
            if (index < 0 || index >= _bodyState.JointLocations.Length) return Vector3.zero;
            OVRPlugin.BodyJointLocation loc = _bodyState.JointLocations[index];
            if (!loc.PositionValid) return Vector3.zero;
            return _trackingSpace.localToWorldMatrix.MultiplyPoint3x4(loc.Pose.Position.FromFlippedZVector3f());
        }

        Vector3 shoulder = GetJointPos(OVRPlugin.BoneId.Body_RightShoulder);
        Vector3 leftShoulder = GetJointPos(OVRPlugin.BoneId.Body_LeftShoulder);  // 外展平面解剖参考
        Vector3 elbow = GetJointPos(OVRPlugin.BoneId.Body_RightArmLower);   // ArmUpper/ArmLower 交界 = 肘
        Vector3 wrist = GetJointPos(OVRPlugin.BoneId.Body_RightHandWrist);

        bool valid = shoulder != Vector3.zero && leftShoulder != Vector3.zero
            && elbow != Vector3.zero && wrist != Vector3.zero;
        if (valid)
        {
            ShoulderPos = shoulder;
            LeftShoulderPos = leftShoulder;
            ElbowPos = elbow;
            WristPos = wrist;
            IsTrackingValid = true;
        }

    }

    /// <summary>标定状态机：Waiting →（右手A键按下）→ Calibrating（按住满时长自动完成，提前松手取消）。</summary>
    private void UpdateCalibration(float dt)
    {
        // 等待状态：右手A键按下沿进入采样
        if (State == CalibrationState.WaitingDown || State == CalibrationState.WaitingHorizontal)
        {
            if (OVRInput.GetDown(calibrateButton, controller))
            {
                State = State == CalibrationState.WaitingDown
                    ? CalibrationState.CalibratingDown
                    : CalibrationState.CalibratingHorizontal;
                _sampleCount = 0;
                _sampleSum = Vector3.zero;
                _heightSum = 0f;
                _holdTimer = 0f;
                LogManager.Log("UpperLimbAKM", "Sampling started: hold your pose, keep right-hand A pressed");
            }
            return;
        }

        if (State != CalibrationState.CalibratingDown &&
            State != CalibrationState.CalibratingHorizontal)
        {
            return;
        }

        if (!OVRInput.Get(calibrateButton, controller))
        {
            // 提前松手：取消本次采样，回到等待提示
            CancelCalibration();
            return;
        }

        _holdTimer += dt;
        if (IsTrackingValid)
        {
            Vector3 dir = (ElbowPos - ShoulderPos).normalized;
            if (dir.sqrMagnitude > 0.5f)
            {
                _sampleSum += dir;
                _sampleCount++;
                _heightSum += ShoulderPos.y - ElbowPos.y; // 肩-肘垂直差（下垂为正）
            }
        }

        if (_holdTimer >= calibrationHoldSeconds)
        {
            CompleteCalibration();
        }
    }

    private void CancelCalibration()
    {
        if (State == CalibrationState.CalibratingDown)
            State = CalibrationState.WaitingDown;
        else if (State == CalibrationState.CalibratingHorizontal)
            State = CalibrationState.WaitingHorizontal;

        _sampleCount = 0;
        _sampleSum = Vector3.zero;
        _heightSum = 0f;
        _holdTimer = 0f;
        LogManager.Log("UpperLimbAKM", "Calibration sampling cancelled (released too early)");
    }

    private void CompleteCalibration()
    {
        if (_sampleCount < 10)
        {
            LogManager.Log("UpperLimbAKM", "Calibration failed: no valid body tracking data in sampling window");
            State = State == CalibrationState.CalibratingDown
                ? CalibrationState.WaitingDown
                : CalibrationState.WaitingHorizontal;
            _sampleCount = 0;
            _sampleSum = Vector3.zero;
            _heightSum = 0f;
            return;
        }

        Vector3 dir = (_sampleSum / _sampleCount).normalized;
        float heightDiff = _heightSum / _sampleCount; // 肩-肘垂直差（下垂为正）

        if (State == CalibrationState.CalibratingDown)
        {
            _downDir = dir;
            _hangHeightDiff = heightDiff; // 下垂高度基准；标定后固定，避免追随抬臂动作导致量程坍缩
            LogManager.Log("UpperLimbAKM", $"elbow_down calibration done: {_downDir} hangDiff={_hangHeightDiff:F3}");

            State = CalibrationState.WaitingHorizontal;
        }
        else
        {
            _horizDir = dir;
            _sideHeightDiff = heightDiff;
            LogManager.Log("UpperLimbAKM", $"side raise calibration done: {_horizDir} sideDiff={_sideHeightDiff:F3}");

            // 退化检查1：两次标定方向夹角过小（如两次姿态一样）
            float angleDeg = Vector3.Angle(_downDir, _horizDir);
            // 退化检查2：两次标定高度差过小（如水平姿态没抬够）
            float heightSpan = _hangHeightDiff - _sideHeightDiff;
            if (angleDeg < calibrationMinAngleDeg || heightSpan < minHeightSpan)
            {
                LogManager.Log("UpperLimbAKM",
                    $"Degenerate calibration: angle {angleDeg:F1}° span {heightSpan:F3}m (< {minHeightSpan:F2}m), please recalibrate");
                State = CalibrationState.WaitingDown;
            }
            else
            {
                State = CalibrationState.Calibrated;
                _heightMapping.Reset();
                LogManager.Log("UpperLimbAKM", $"Elbow calibration done, angle {angleDeg:F1}° heightSpan {heightSpan:F3}m");
            }
        }

        _sampleCount = 0;
        _sampleSum = Vector3.zero;
        _heightSum = 0f;
        _holdTimer = 0f;
    }

    private void UpdateAlpha(float dt)
    {
        _heightMapping.Step(IsCalibrated, IsTrackingValid, _hangHeightDiff,
            ShoulderPos.y - ElbowPos.y, dt, alphaDeadZone, alphaFullZone, alphaFilterTau);
    }

    private void UpdateInputs()
    {
        IsClutched = OVRInput.Get(clutchButton, controller);
        if (OVRInput.GetDown(recenterButton, controller))
            _recenterPulse = true;
    }

    /// <summary>
    /// UDP 8005 发送（仅 WRM 启用时），JSON 格式（与 Python 端 parse_wrm_unity_packet 对齐）：
    /// {"type":"WRM","frame_id":n,"timestamp_ns":t,"elbow_alpha":a,"confidence":c,
    ///  "grip_trigger":0|1,"joystick_press":true|false}
    /// clutch → grip_trigger，recenter → joystick_press。
    /// 注意：Python 端 WRM 模式下会关闭原 socket_2 并由 WrmUdpReceiver 接管 8005，
    /// UdpWindowManager 遗留的 {port},{res};...;Fine Control Mode,ON; 消息到达后被解析丢弃，互不干扰。
    /// </summary>
    private void UpdateSend()
    {
        // WRM shares the hand tracking mode and UDP 8005 safety gate. A mode
        // switch or tracking loss must discard pending pulses immediately.
        if (AppManager.Instance != null && AppManager.Instance.TrackingMode == 1)
        {
            if (WrmEnabled) SetWrmEnabled(false);
            _recenterPulse = false;
            return;
        }

        if (AppManager.Instance != null && !AppManager.Instance.CanSendTeleopData)
        {
            _recenterPulse = false;
            return;
        }

        if (!WrmEnabled)
        {
            _recenterPulse = false; // 未启用时丢弃脉冲，避免延迟发送
            return;
        }

        _sendTimer += Time.deltaTime;
        if (_sendTimer < 1f / Mathf.Max(sendRateHz, 1f)) return;
        _sendTimer = 0f;

        if (_udpSocket == null)
            _udpSocket = FindAnyObjectByType<UdpSocket>();
        if (_udpSocket == null) return;

        _sb.Clear();
        _frameCounter++;
        long timestampNs = (DateTime.UtcNow.Ticks - UnixEpochTicks) * 100L;

        _sb.Append("{\"type\":\"WRM\",\"frame_id\":").Append(_frameCounter).Append(',');
        _sb.Append("\"timestamp_ns\":").Append(timestampNs).Append(',');
        _sb.Append("\"elbow_alpha\":").Append(ElbowAlpha.ToString("F4", CultureInfo.InvariantCulture)).Append(',');
        _sb.Append("\"confidence\":").Append(Confidence.ToString("F4", CultureInfo.InvariantCulture)).Append(',');
        _sb.Append("\"grip_trigger\":").Append(IsClutched ? "1.0" : "0.0").Append(',');
        _sb.Append("\"joystick_press\":").Append(_recenterPulse ? "true" : "false").Append('}');

        _recenterPulse = false;
        _udpSocket.SendData8005(_sb.ToString());
    }

    // ===== 对外操作（供 UI 调用）=====

    /// <summary>进入标定流程（已标定时等价于重新标定）。</summary>
    public void StartCalibration()
    {
        State = CalibrationState.WaitingDown;
        _sampleCount = 0;
        _sampleSum = Vector3.zero;
        _heightSum = 0f;
        _holdTimer = 0f;
        LogManager.Log("UpperLimbAKM", "Start elbow calibration: keep right arm hanging down");
    }

    /// <summary>重新标定（保留当前 alpha 输出，标定完成后重新计算）。</summary>
    public void Recalibrate()
    {
        _downDir = _horizDir = Vector3.zero;
        _heightMapping.Reset();
        StartCalibration();
    }

    /// <summary>WRM 主开关（由 WRM_enable 按钮切换）。</summary>
    public void SetWrmEnabled(bool enabled)
    {
        if (enabled && AppManager.Instance != null && AppManager.Instance.TrackingMode == 1)
            enabled = false;
        if (WrmEnabled == enabled) return;
        WrmEnabled = enabled;
        LogManager.Log("UpperLimbAKM", enabled ? "WRM enabled" : "WRM disabled");
    }

    /// <summary>供场景 WRM_enable 按钮 onClick 调用（无参）：切换 WRM 开关。</summary>
    public void ToggleWrm()
    {
        SetWrmEnabled(!WrmEnabled);
    }

    /// <summary>供场景 Calibration 按钮 onClick 调用（无参）：未标定则开始标定，已标定则重新标定。</summary>
    public void OnCalibrationClicked()
    {
        if (IsCalibrated)
            Recalibrate();
        else
            StartCalibration();
    }

    private void OnDestroy()
    {
        if (_trackingStarted)
            OVRPlugin.StopBodyTracking();
    }
}
