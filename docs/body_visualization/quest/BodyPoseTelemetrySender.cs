using System;
using System.Net.Sockets;
using System.Text;
using Doffy.Protocol;
using UnityEngine;

/// <summary>
/// Diagnostic body telemetry on its own UDP port. Consumes WRM's sampled state;
/// never starts tracking, starts a teleop session, calibrates, or sends controls.
/// </summary>
public sealed class BodyPoseTelemetrySender : MonoBehaviour
{
    [Tooltip("Independent body visualization destination port (PC receiver).")]
    public int destinationPort = 8015;
    [Tooltip("Diagnostic send rate, independent of robot/WRM transmission.")]
    public float sendRateHz = 25f;

    /// <summary>Explicit, per-launch diagnostic transmission opt-in; never stored in preferences.</summary>
    public bool SendingEnabled { get; private set; }

    /// <summary>Resolve lazily so Session UI creation does not depend on bootstrap order.</summary>
    public static BodyPoseTelemetrySender Instance
    {
        get
        {
            if (_instance == null)
                _instance = FindAnyObjectByType<BodyPoseTelemetrySender>(FindObjectsInactive.Include);
            if (_instance == null)
                _instance = new GameObject("Body Pose Telemetry").AddComponent<BodyPoseTelemetrySender>();
            return _instance;
        }
    }

    private static BodyPoseTelemetrySender _instance;
    private static readonly long UnixEpochTicks =
        new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc).Ticks;
    private readonly BodyPoseWireFormat.Joint[] _joints =
        new BodyPoseWireFormat.Joint[BodyPoseWireFormat.FullBodyJointCount];
    private readonly StringBuilder _buffer = new StringBuilder(18000);
    private UpperLimbAkmManager _manager;
    private UdpClient _client;
    private string _destinationHost;
    private int _connectedPort;
    private float _nextConnectionAttempt;
    private float _sendTimer;
    private long _frameCounter;
    private bool _paused;
    private bool _focused = true;
    private bool _lastFullBody;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
    private static void ResetInstance() => _instance = null;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void AutoCreate()
    {
        _ = Instance;
    }

    private void Awake()
    {
        if (_instance != null && _instance != this)
        {
            Destroy(gameObject);
            return;
        }
        _instance = this;
        SendingEnabled = false;
        DontDestroyOnLoad(gameObject);
    }

    public void SetSendingEnabled(bool enabled)
    {
        bool next = enabled && isActiveAndEnabled;
        bool changed = SendingEnabled != next;
        SendingEnabled = next;
        _sendTimer = 0f;
        _nextConnectionAttempt = 0f;
        if (!next) CloseSocket();
        if (changed) LogManager.Log("BodyTelemetry", next ? "BODY data sending enabled" : "BODY data sending disabled");
    }

    private void LateUpdate()
    {
        if (!SendingEnabled) return;
        _sendTimer += Time.unscaledDeltaTime;
        float rate = float.IsNaN(sendRateHz) || float.IsInfinity(sendRateHz) ? 25f : sendRateHz;
        if (_sendTimer < 1f / Mathf.Clamp(rate, 1f, 30f)) return;
        _sendTimer = 0f;
        if (!EnsureDestination()) return;

        if (_manager == null) _manager = FindAnyObjectByType<UpperLimbAkmManager>();
        bool trackingValid = false;
        float confidence = 0f;
        Array.Clear(_joints, 0, _joints.Length);
        if (_manager != null)
        {
            UpperLimbAkmManager.BodyPoseSnapshot snapshot = _manager.CaptureBodyPoseSnapshot();
            _lastFullBody = snapshot.JointSet == OVRPlugin.BodyJointSet.FullBody;
            // Match OVRBody's high-confidence criterion for diagnostic display.
            // Keep the measured confidence when display quality drops so the PC
            // can explain why it is holding the previous pose.
            confidence = snapshot.Confidence;
            trackingValid = snapshot.TrackingValid && !_paused && _focused &&
                !float.IsNaN(confidence) && !float.IsInfinity(confidence) &&
                confidence > 0.5f && confidence <= 1f;
            int count = _lastFullBody ? BodyPoseWireFormat.FullBodyJointCount : BodyPoseWireFormat.UpperBodyJointCount;
            for (int i = 0; trackingValid && i < count; i++)
            {
                OVRPlugin.BodyJointLocation location = snapshot.GetJointLocation(i);
                Vector3 position = location.PositionValid
                    ? snapshot.TrackingToWorld.MultiplyPoint3x4(location.Pose.Position.FromFlippedZVector3f())
                    : Vector3.zero;
                Quaternion rotation = location.OrientationValid
                    ? snapshot.TrackingRotation * location.Pose.Orientation.FromFlippedZQuatf()
                    : Quaternion.identity;
                _joints[i] = new BodyPoseWireFormat.Joint(position.x, position.y, position.z,
                    rotation.x, rotation.y, rotation.z, rotation.w,
                    location.PositionValid, location.OrientationValid,
                    location.PositionTracked, location.OrientationTracked);
            }
        }

        string json = BodyPoseWireFormat.Serialize(++_frameCounter,
            (DateTime.UtcNow.Ticks - UnixEpochTicks) * 100L, _lastFullBody, confidence,
            trackingValid, _joints, _manager != null ? _manager.ElbowAlpha : 0f,
            _manager != null ? _manager.Confidence : 0f,
            _manager != null && _manager.WrmEnabled, _manager != null && _manager.IsCalibrated, _buffer);
        byte[] packet = Encoding.UTF8.GetBytes(json);
        if (packet.Length > BodyPoseWireFormat.MaxUdpPayloadBytes)
        {
            LogManager.Log("BodyTelemetry", "BODY packet exceeds UDP payload limit; skipped");
            return;
        }
        try
        {
            _client.Send(packet, packet.Length);
        }
        catch (Exception error) when (error is SocketException || error is ObjectDisposedException)
        {
            CloseSocket();
            _nextConnectionAttempt = Time.unscaledTime + 5f;
            LogManager.Log("BodyTelemetry", "BODY send failed: " + error.Message);
        }
    }

    private bool EnsureDestination()
    {
        if (!SendingEnabled || !isActiveAndEnabled) return false;
        // AppManager loads the same cfg_ip preference before this bootstrap. Its
        // host property also follows connection-form edits before session Start.
        string host = AppManager.Instance != null ? AppManager.Instance.ServerIP :
            PlayerPrefs.GetString(TeleopSettingsStore.HostKey, TeleopRuntimeSettings.Defaults().ServerHost);
        host = (host ?? string.Empty).Trim();
        if (!string.Equals(host, _destinationHost, StringComparison.OrdinalIgnoreCase) ||
            destinationPort != _connectedPort)
        {
            CloseSocket();
            _destinationHost = host;
            _connectedPort = destinationPort;
            _nextConnectionAttempt = 0f;
        }
        if (_client != null) return true;
        if (string.IsNullOrEmpty(host) || destinationPort < 1 || destinationPort > 65535 ||
            Time.unscaledTime < _nextConnectionAttempt) return false;
        UdpClient candidate = null;
        try
        {
            candidate = new UdpClient();
            candidate.Connect(host, destinationPort);
            _client = candidate;
            LogManager.Log("BodyTelemetry", $"BODY v1 telemetry destination {host}:{destinationPort}, {sendRateHz} Hz");
            return true;
        }
        catch (Exception error) when (error is SocketException || error is ArgumentException)
        {
            candidate?.Close();
            _nextConnectionAttempt = Time.unscaledTime + 5f;
            LogManager.Log("BodyTelemetry", "BODY destination unavailable: " + error.Message);
            return false;
        }
    }

    private void CloseSocket()
    {
        _client?.Close();
        _client = null;
    }

    private void OnApplicationPause(bool paused) => _paused = paused;
    private void OnApplicationFocus(bool focused) => _focused = focused;
    private void OnDisable() => SetSendingEnabled(false);
    private void OnDestroy()
    {
        CloseSocket();
        if (_instance == this) _instance = null;
    }
}
