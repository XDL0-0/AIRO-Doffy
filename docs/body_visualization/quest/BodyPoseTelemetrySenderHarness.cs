using System;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
using System.Text;

// SDK/Unity adapters only. Compile the deployed sender unchanged and exercise
// its real LateUpdate path with a loopback receiver; no Unity or hardware runs.
namespace UnityEngine
{
    public enum FindObjectsInactive { Exclude, Include }
    public enum RuntimeInitializeLoadType { SubsystemRegistration, AfterSceneLoad }
    [AttributeUsage(AttributeTargets.Field)]
    public sealed class TooltipAttribute : Attribute { public TooltipAttribute(string text) { } }
    [AttributeUsage(AttributeTargets.Method)]
    public sealed class RuntimeInitializeOnLoadMethodAttribute : Attribute
    { public RuntimeInitializeOnLoadMethodAttribute(RuntimeInitializeLoadType kind) { } }
    public class Object
    {
        public static T FindAnyObjectByType<T>() { return default(T); }
        public static T FindAnyObjectByType<T>(FindObjectsInactive inactive) { return default(T); }
        public static void Destroy(Object item) { }
        public static void DontDestroyOnLoad(Object item) { }
    }
    public class MonoBehaviour : Object
    {
        public bool isActiveAndEnabled = true;
        public GameObject gameObject;
    }
    public class GameObject : Object
    {
        public GameObject(string name) { }
        public T AddComponent<T>() where T : new() { return new T(); }
    }
    public static class Time { public static float unscaledDeltaTime = 1f, unscaledTime; }
    public static class Mathf
    { public static float Clamp(float value, float low, float high) { return Math.Max(low, Math.Min(high, value)); } }
    public static class PlayerPrefs
    { public static string GetString(string key, string fallback) { return fallback; } }
    public struct Vector3
    {
        public float x, y, z;
        public Vector3(float x, float y, float z) { this.x = x; this.y = y; this.z = z; }
        public static Vector3 zero { get { return new Vector3(); } }
    }
    public struct Quaternion
    {
        public float x, y, z, w;
        public Quaternion(float x, float y, float z, float w) { this.x = x; this.y = y; this.z = z; this.w = w; }
        public static Quaternion identity { get { return new Quaternion(0, 0, 0, 1); } }
        public static Quaternion operator *(Quaternion lhs, Quaternion rhs)
        {
            return new Quaternion(lhs.w * rhs.x + lhs.x * rhs.w + lhs.y * rhs.z - lhs.z * rhs.y,
                lhs.w * rhs.y - lhs.x * rhs.z + lhs.y * rhs.w + lhs.z * rhs.x,
                lhs.w * rhs.z + lhs.x * rhs.y - lhs.y * rhs.x + lhs.z * rhs.w,
                lhs.w * rhs.w - lhs.x * rhs.x - lhs.y * rhs.y - lhs.z * rhs.z);
        }
    }
    public struct Matrix4x4
    { public Vector3 MultiplyPoint3x4(Vector3 value) { return value; } }
}

public static class OVRPlugin
{
    public enum BodyJointSet { UpperBody, FullBody }
    public struct Vector3f
    {
        public float x, y, z;
        public UnityEngine.Vector3 FromFlippedZVector3f() { return new UnityEngine.Vector3(x, y, -z); }
    }
    public struct Quatf
    {
        public float x, y, z, w;
        public UnityEngine.Quaternion FromFlippedZQuatf() { return new UnityEngine.Quaternion(-x, -y, z, w); }
    }
    public struct Posef { public Vector3f Position; public Quatf Orientation; }
    public struct BodyJointLocation
    { public Posef Pose; public bool PositionValid, OrientationValid, PositionTracked, OrientationTracked; }
}

public sealed class UpperLimbAkmManager
{
    public float ElbowAlpha = 0.25f, Confidence = 0.75f;
    public bool WrmEnabled = true, IsCalibrated = true;
    public int SnapshotReads;
    public BodyPoseSnapshot Snapshot;
    public BodyPoseSnapshot CaptureBodyPoseSnapshot() { SnapshotReads++; return Snapshot; }
    public struct BodyPoseSnapshot
    {
        public OVRPlugin.BodyJointSet JointSet;
        public bool TrackingValid;
        public float Confidence;
        public UnityEngine.Matrix4x4 TrackingToWorld;
        public UnityEngine.Quaternion TrackingRotation;
        public OVRPlugin.BodyJointLocation[] Joints;
        public OVRPlugin.BodyJointLocation GetJointLocation(int index)
        { return Joints != null && index < Joints.Length ? Joints[index] : default(OVRPlugin.BodyJointLocation); }
    }
}
public sealed class AppManager
{
    public static AppManager Instance = new AppManager();
    public string ServerIP = "127.0.0.1";
}
public static class TeleopSettingsStore { public const string HostKey = "cfg_ip"; }
public sealed class TeleopRuntimeSettings
{
    public string ServerHost = "127.0.0.1";
    public static TeleopRuntimeSettings Defaults() { return new TeleopRuntimeSettings(); }
}
public static class LogManager { public static void Log(string category, string message) { } }

internal static class BodyPoseTelemetrySenderHarness
{
    private static void Invoke(BodyPoseTelemetrySender sender, string method, params object[] args)
    {
        typeof(BodyPoseTelemetrySender).GetMethod(method, BindingFlags.Instance | BindingFlags.NonPublic)
            .Invoke(sender, args);
    }

    private static void Emit(BodyPoseTelemetrySender sender, UpperLimbAkmManager manager,
        UdpClient receiver, string label, float confidence, bool snapshotValid = true,
        bool paused = false, bool focused = true, bool tracked = true)
    {
        var joints = new OVRPlugin.BodyJointLocation[70];
        for (int i = 0; i < joints.Length; i++)
        {
            joints[i] = new OVRPlugin.BodyJointLocation
            {
                Pose = new OVRPlugin.Posef
                {
                    Position = new OVRPlugin.Vector3f { x = i + 0.25f, y = 1.5f, z = 2.5f },
                    Orientation = new OVRPlugin.Quatf { w = 1f }
                },
                PositionValid = true, OrientationValid = true,
                PositionTracked = tracked, OrientationTracked = tracked
            };
        }
        manager.Snapshot = new UpperLimbAkmManager.BodyPoseSnapshot
        {
            JointSet = OVRPlugin.BodyJointSet.UpperBody,
            TrackingValid = snapshotValid, Confidence = confidence,
            TrackingRotation = UnityEngine.Quaternion.identity, Joints = joints
        };
        Invoke(sender, "OnApplicationPause", paused);
        Invoke(sender, "OnApplicationFocus", focused);
        Invoke(sender, "LateUpdate");
        IPEndPoint remote = new IPEndPoint(IPAddress.Loopback, 0);
        Console.WriteLine(label + "\t" + Encoding.UTF8.GetString(receiver.Receive(ref remote)));
    }

    private static int Main()
    {
        using (var receiver = new UdpClient(new IPEndPoint(IPAddress.Loopback, 0)))
        {
            receiver.Client.ReceiveTimeout = 3000;
            var manager = new UpperLimbAkmManager();
            var sender = new BodyPoseTelemetrySender
            { destinationPort = ((IPEndPoint)receiver.Client.LocalEndPoint).Port };
            typeof(BodyPoseTelemetrySender).GetField("_manager", BindingFlags.Instance | BindingFlags.NonPublic)
                .SetValue(sender, manager);
            Invoke(sender, "LateUpdate");
            if (manager.SnapshotReads != 0) throw new Exception("Disabled sender must not sample BODY or WRM");
            sender.SetSendingEnabled(true);
            try
            {
                Emit(sender, manager, receiver, "high", 0.75f);
                Emit(sender, manager, receiver, "boundary", 0.5f);
                Emit(sender, manager, receiver, "low", 0.25f);
                Emit(sender, manager, receiver, "zero", 0f);
                Emit(sender, manager, receiver, "recovered", 0.5001f);
                Emit(sender, manager, receiver, "paused", 0.875f, paused: true);
                Emit(sender, manager, receiver, "unfocused", 0.875f, focused: false);
                Emit(sender, manager, receiver, "resumed", 0.875f);
                Emit(sender, manager, receiver, "sdk_invalid", 0.75f, snapshotValid: false);
                Emit(sender, manager, receiver, "nan", float.NaN);
                Emit(sender, manager, receiver, "infinite", float.PositiveInfinity);
                Emit(sender, manager, receiver, "out_of_range", 1.25f);
                Emit(sender, manager, receiver, "valid_untracked", 0.875f, tracked: false);
                if (manager.SnapshotReads != 13 || manager.Confidence != 0.75f ||
                    manager.ElbowAlpha != 0.25f || !manager.WrmEnabled || !manager.IsCalibrated)
                    throw new Exception("BODY sends must leave WRM state unchanged");
            }
            finally { sender.SetSendingEnabled(false); }
        }
        return 0;
    }
}
