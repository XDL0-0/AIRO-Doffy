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
    /// <summary>Calibration state machine.</summary>
    public enum CalibrationState
    {
        Idle,                    // Calibration has not started
        WaitingDown,             // Waiting: press right-hand A for elbow-down calibration
        CalibratingDown,         // Sampling elbow_down while A is held
        WaitingHorizontal,       // Waiting: press right-hand A for side-raise calibration
        CalibratingHorizontal,   // Sampling the side raise while A is held
        Calibrated               // Calibration complete
    }

    // ===== Tunable parameters (public Inspector fields; bootstrap creation uses defaults)=====
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
    [Tooltip("DEPRECATED: Plane projection was removed: real forward raises read 0.7-0.9 in all three plane variants and could not be distinguished. Retained only for serialization compatibility; unused in calculations.")]
    public int planeMode = 1;

    [Header("Vertical Position")]
    [Tooltip("DEPRECATED: The hanging baseline is no longer updated at runtime. Retained only for serialization compatibility.")]
    public float hangTrackTau = 0.5f;
    [Tooltip("Degeneracy threshold (m): the shoulder-elbow height differences from the hanging and horizontal calibrations must differ by at least this value.")]
    public float minHeightSpan = 0.1f;
    [Tooltip("DEPRECATED: The hanging baseline is no longer updated at runtime. Retained only for serialization compatibility.")]
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
    [Tooltip("Debug: log world-space joint positions and intermediate alpha values every 0.5 s to inspect Meta joint data.")]
    public bool logJointDebug;

    // ===== Public state =====
    public CalibrationState State { get; private set; } = CalibrationState.Idle;
    public bool IsCalibrated => State == CalibrationState.Calibrated;
    public bool WrmEnabled { get; private set; }

    /// <summary>Whether body tracking is valid: permission granted, tracking started and joint positions valid.</summary>
    public bool IsTrackingValid { get; private set; }

    /// <summary>Body tracking confidence (0-1; zero when invalid).</summary>
    public float Confidence { get; private set; }

    /// <summary>World-space right/left shoulder, right elbow and right wrist positions (zero when invalid).</summary>
    public Vector3 ShoulderPos { get; private set; }
    public Vector3 LeftShoulderPos { get; private set; }
    public Vector3 ElbowPos { get; private set; }
    public Vector3 WristPos { get; private set; }

    /// <summary>Final elbow_alpha after clipping, dead zones, low-pass filtering and smoothstep; zero until calibrated.</summary>
    public float ElbowAlpha => _heightMapping.Value;

    /// <summary>Clutch state (grip held).</summary>
    public bool IsClutched { get; private set; }

    // ===== Internal state =====
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

    // Calibration sampling
    private Vector3 _downDir;    // Calibration: unit shoulder-to-elbow direction with the arm hanging naturally
    private Vector3 _horizDir;   // Calibration: unit shoulder-to-elbow direction with the upper arm horizontal
    private Vector3 _sampleSum;  // Sum of direction vectors within the sampling window
    private int _sampleCount;
    private float _holdTimer;

    // Vertical-height mapping: alpha = (hanging difference - current difference) / hanging difference.
    // Height difference = shoulder.y - elbow.y (positive when hanging, near zero when horizontal). No horizontal reference plane; unaffected by turning.
    private float _hangHeightDiff;  // Hanging calibration: shoulder-elbow vertical difference (positive when the elbow is lower; fixed after calibration)
    private float _sideHeightDiff;  // Horizontal side-raise calibration: shoulder-elbow vertical difference (near zero)
    private float _heightSum;       // Sum of shoulder.y - elbow.y within the calibration sampling window

    // Filtering
    private readonly UpperLimbHeightMapping _heightMapping = new UpperLimbHeightMapping();

    // UDP
    private float _sendTimer;
    private int _frameCounter;
    private bool _recenterPulse;
    private readonly StringBuilder _sb = new StringBuilder(128);

    // Debugging
    private float _debugTimer;

    /// <summary>Unix epoch ticks (DateTime.UtcNow.Ticks uses 100 ns units).</summary>
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

        // Request body tracking permission through the system dialog; start tracking in Update after it is granted.
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

    /// <summary>Debug: log world-space joints and calibration directions at a reduced rate to inspect Meta data and shoulder drift.</summary>
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
            // Vertical-height intermediates: diff (shoulder-elbow), hangDiff (hanging calibration), sideDiff (horizontal calibration), span and alphaRaw.
            float dbgDiff = ShoulderPos.y - ElbowPos.y;
            float dbgSpan = _hangHeightDiff - _sideHeightDiff;
            float dbgRaw = _hangHeightDiff <= 1e-4f ? -1f
                : Mathf.Clamp01((_hangHeightDiff - dbgDiff) / _hangHeightDiff);
            LogManager.Log("UpperLimbAKM",
                $"down={_downDir} horiz={_horizDir} diff={dbgDiff:F3} hangDiff={_hangHeightDiff:F3} sideDiff={_sideHeightDiff:F3} span={dbgSpan:F3} raw={dbgRaw:F3} alpha={ElbowAlpha:F3} conf={Confidence:F2}");
        }
    }

    /// <summary>Debug helper: get a joint orientation in world space (identity when invalid).</summary>
    private Quaternion JointOrientationWorld(OVRPlugin.BoneId boneId)
    {
        int index = (int)boneId;
        if (_bodyState.JointLocations == null || index < 0 || index >= _bodyState.JointLocations.Length)
            return Quaternion.identity;
        OVRPlugin.BodyJointLocation loc = _bodyState.JointLocations[index];
        if (!loc.OrientationValid || _trackingSpace == null) return Quaternion.identity;
        return _trackingSpace.rotation * loc.Pose.Orientation.FromFlippedZQuatf();
    }

    /// <summary>Debug helper: get a joint position in world space (zero when invalid).</summary>
    private Vector3 JointWorld(OVRPlugin.BoneId boneId)
    {
        int index = (int)boneId;
        if (_bodyState.JointLocations == null || index < 0 || index >= _bodyState.JointLocations.Length)
            return Vector3.zero;
        OVRPlugin.BodyJointLocation loc = _bodyState.JointLocations[index];
        if (!loc.PositionValid || _trackingSpace == null) return Vector3.zero;
        return _trackingSpace.localToWorldMatrix.MultiplyPoint3x4(loc.Pose.Position.FromFlippedZVector3f());
    }

    /// <summary>Start body tracking once permission has been granted.</summary>
    private void EnsureTrackingStarted()
    {
        if (_trackingStarted || !OVRPlugin.bodyTrackingSupported) return;

        if (!OVRPermissionsRequester.IsPermissionGranted(OVRPermissionsRequester.Permission.BodyTracking))
            return;

        if (OVRPlugin.nativeXrApi != OVRPlugin.XrApi.OpenXR)
        {
            LogManager.Log("UpperLimbAKM", "Body tracking requires OpenXR, current XrApi unavailable");
            _trackingStarted = true; // Prevent repeated retries
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

    /// <summary>Read the body skeleton pose in tracking space each frame and convert it to world space.</summary>
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

        // Convert tracking-space skeleton data to world space, matching OVRSkeleton without mirroring.
        Vector3 GetJointPos(OVRPlugin.BoneId boneId)
        {
            int index = (int)boneId;
            if (index < 0 || index >= _bodyState.JointLocations.Length) return Vector3.zero;
            OVRPlugin.BodyJointLocation loc = _bodyState.JointLocations[index];
            if (!loc.PositionValid) return Vector3.zero;
            return _trackingSpace.localToWorldMatrix.MultiplyPoint3x4(loc.Pose.Position.FromFlippedZVector3f());
        }

        Vector3 shoulder = GetJointPos(OVRPlugin.BoneId.Body_RightShoulder);
        Vector3 leftShoulder = GetJointPos(OVRPlugin.BoneId.Body_LeftShoulder);  // Anatomical reference for the abduction plane
        Vector3 elbow = GetJointPos(OVRPlugin.BoneId.Body_RightArmLower);   // The ArmUpper/ArmLower boundary is the elbow
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

    /// <summary>Calibration state machine: Waiting -> right-hand A pressed -> Calibrating; holding for the full duration completes sampling, early release cancels it.</summary>
    private void UpdateCalibration(float dt)
    {
        // In a waiting state, the right-hand A press edge starts sampling.
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
            // Early release cancels sampling and returns to the waiting prompt.
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
                _heightSum += ShoulderPos.y - ElbowPos.y; // Shoulder-elbow vertical difference (positive when hanging)
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
        float heightDiff = _heightSum / _sampleCount; // Shoulder-elbow vertical difference (positive when hanging)

        if (State == CalibrationState.CalibratingDown)
        {
            _downDir = dir;
            _hangHeightDiff = heightDiff; // Hanging baseline stays fixed after calibration so following an arm raise cannot collapse the range.
            LogManager.Log("UpperLimbAKM", $"elbow_down calibration done: {_downDir} hangDiff={_hangHeightDiff:F3}");

            State = CalibrationState.WaitingHorizontal;
        }
        else
        {
            _horizDir = dir;
            _sideHeightDiff = heightDiff;
            LogManager.Log("UpperLimbAKM", $"side raise calibration done: {_horizDir} sideDiff={_sideHeightDiff:F3}");

            // Degeneracy check 1: calibration directions are too similar, for example when both poses are identical.
            float angleDeg = Vector3.Angle(_downDir, _horizDir);
            // Degeneracy check 2: calibration heights are too similar, for example when the horizontal pose is too low.
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
    /// Send JSON on UDP 8005 only when WRM is enabled, matching Python parse_wrm_unity_packet:
    /// {"type":"WRM","frame_id":n,"timestamp_ns":t,"elbow_alpha":a,"confidence":c,
    ///  "grip_trigger":0|1,"joystick_press":true|false}
    /// clutch -> grip_trigger; recenter -> joystick_press.
    /// In Python WRM mode, the original socket_2 is closed and WrmUdpReceiver takes over port 8005.
    /// Legacy UdpWindowManager messages ({port},{res};...;Fine Control Mode,ON;) are parsed and discarded without interfering.
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
            _recenterPulse = false; // Discard pulses while disabled to avoid delayed delivery
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

    // ===== Public operations called by the UI=====

    /// <summary>Start calibration; if already calibrated, this begins recalibration.</summary>
    public void StartCalibration()
    {
        State = CalibrationState.WaitingDown;
        _sampleCount = 0;
        _sampleSum = Vector3.zero;
        _heightSum = 0f;
        _holdTimer = 0f;
        LogManager.Log("UpperLimbAKM", "Start elbow calibration: keep right arm hanging down");
    }

    /// <summary>Recalibrate, retaining the current alpha output until it is recomputed after calibration.</summary>
    public void Recalibrate()
    {
        _downDir = _horizDir = Vector3.zero;
        _heightMapping.Reset();
        StartCalibration();
    }

    /// <summary>Main WRM switch, toggled by the WRM_enable button.</summary>
    public void SetWrmEnabled(bool enabled)
    {
        if (enabled && AppManager.Instance != null && AppManager.Instance.TrackingMode == 1)
            enabled = false;
        if (WrmEnabled == enabled) return;
        WrmEnabled = enabled;
        LogManager.Log("UpperLimbAKM", enabled ? "WRM enabled" : "WRM disabled");
    }

    /// <summary>Parameterless onClick handler for the scene WRM_enable button; toggle WRM.</summary>
    public void ToggleWrm()
    {
        SetWrmEnabled(!WrmEnabled);
    }

    /// <summary>Parameterless onClick handler for the scene Calibration button; start calibration or recalibrate if already calibrated.</summary>
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
