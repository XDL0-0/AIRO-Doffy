using System;
using System.Collections.Generic;
using System.Reflection;

namespace UnityEngine
{
    [AttributeUsage(AttributeTargets.Field)]
    public sealed class SerializeField : Attribute { }

    [AttributeUsage(AttributeTargets.Field)]
    public sealed class HeaderAttribute : Attribute { public HeaderAttribute(string text) { } }

    [AttributeUsage(AttributeTargets.Field)]
    public sealed class TooltipAttribute : Attribute { public TooltipAttribute(string text) { } }

    public class Object
    {
        public static T FindAnyObjectByType<T>() where T : class { return null; }
    }

    public class MonoBehaviour : Object { }
    public static class Debug { public static void LogWarning(string message) { } }
    public static class Time { public static float deltaTime; }

    public struct Vector3
    {
        public float x, y, z;
        public Vector3(float x, float y, float z) { this.x = x; this.y = y; this.z = z; }
    }

    public struct Quaternion
    {
        public float x, y, z, w;
        public static Quaternion identity { get { return new Quaternion { w = 1f }; } }
    }

    public struct Pose
    {
        public Vector3 position;
        public Quaternion rotation;
        public Pose(Vector3 position, Quaternion rotation) { this.position = position; this.rotation = rotation; }
    }
}

namespace TMPro
{
    public class TextMeshProUGUI { public string text; }
}

public class UdpSocket
{
    public readonly List<string> RecordingCommands = new List<string>();
    public readonly List<string> HandPackets = new List<string>();
    public void SendData8003(string message) { RecordingCommands.Add(message); }
    public void SendData8001(string message) { HandPackets.Add(message); }
}

public class OVRHand
{
    public bool IsTracked = true;
}

public class OVRSkeleton
{
    public IList<OVRBone> Bones;
}

public class OVRBone
{
    public UnityEngine.Transform Transform;
}

namespace UnityEngine
{
    public class Transform
    {
        public Vector3 position;
        public Quaternion rotation = Quaternion.identity;
    }
}

public static class OVRInput
{
    public enum Controller { LTouch, RTouch }
    public static bool IsControllerConnected(Controller controller) { return false; }
}

public class AppManager
{
    public static AppManager Instance;
    public bool CanSendTeleopData = true;
}

public static class TeleopReferenceFrame
{
    public static UnityEngine.Vector3 TransformWorldPoint(UnityEngine.Vector3 value) { return value; }
    public static UnityEngine.Quaternion TransformWorldRotation(UnityEngine.Quaternion value) { return value; }
}

public static class RecordingAndHandSenderHarness
{
    private static int checks;

    private static void Check(bool condition, string name)
    {
        checks++;
        if (!condition) throw new Exception("RecordingAndHandSenderHarness: " + name);
    }

    private static void SetField(object target, string name, object value)
    {
        FieldInfo field = target.GetType().GetField(name, BindingFlags.Instance | BindingFlags.NonPublic);
        if (field == null) throw new MissingFieldException(target.GetType().Name, name);
        field.SetValue(target, value);
    }

    private static void Invoke(object target, string name, params object[] args)
    {
        MethodInfo method = target.GetType().GetMethod(name, BindingFlags.Instance | BindingFlags.NonPublic);
        if (method == null) throw new MissingMethodException(target.GetType().Name, name);
        method.Invoke(target, args);
    }

    public static int Main()
    {
        CheckRecordingUndoState();
        CheckCompleteHandFrames();
        Console.WriteLine("PASS " + checks + " recording and complete hand-frame checks (production sources)");
        return 0;
    }

    private static void CheckRecordingUndoState()
    {
        var socket = new UdpSocket();
        var label = new TMPro.TextMeshProUGUI();
        var recorder = new RecordingController();
        SetField(recorder, "udpSocket", socket);
        SetField(recorder, "recordButtonText", label);
        Invoke(recorder, "Start");
        Check(!recorder.IsRecording && label.text == "Start Record", "initial state and label are idle");

        recorder.Undo();
        Check(!recorder.IsRecording && label.text == "Start Record", "idle undo keeps local state and label idle");
        Check(HasCommands(socket, "Undo"), "idle undo preserves the legacy Undo packet");

        recorder.Recording();
        Check(recorder.IsRecording && label.text == "Stop Record", "Start enters recording state");
        recorder.Undo();
        Check(!recorder.IsRecording && label.text == "Start Record", "undo exits recording state and refreshes label");
        Check(HasCommands(socket, "Undo", "Start", "Undo"), "active undo sends no extra Stop packet");

        recorder.Recording();
        Check(recorder.IsRecording && Last(socket.RecordingCommands) == "Start", "Start works immediately after undo");
        recorder.StopRecording();
        int stoppedCount = socket.RecordingCommands.Count;
        recorder.StopRecording();
        Check(!recorder.IsRecording && label.text == "Start Record", "session stop clears recording state");
        Check(stoppedCount == socket.RecordingCommands.Count && Last(socket.RecordingCommands) == "Stop",
            "session stop sends exactly one Stop packet");
    }

    private static bool HasCommands(UdpSocket socket, params string[] expected)
    {
        if (socket.RecordingCommands.Count != expected.Length) return false;
        for (int i = 0; i < expected.Length; i++)
            if (socket.RecordingCommands[i] != expected[i]) return false;
        return true;
    }

    private static string Last(List<string> values) { return values[values.Count - 1]; }

    private static void CheckCompleteHandFrames()
    {
        var socket = new UdpSocket();
        var sender = new HandTrackingSender();
        SetField(sender, "udpSocket", socket);
        var hand = new OVRHand();
        var skeleton = new OVRSkeleton { Bones = MakeBones(26) };

        SetField(sender, "useBinaryProtocol", false);
        Send(sender, hand, skeleton, 'L', OVRInput.Controller.LTouch);
        Check(socket.HandPackets.Count == 1, "valid text hand frame is sent");
        string[] textFields = socket.HandPackets[0].TrimEnd('\n').Split(',');
        Check(textFields.Length == 89 && textFields[0] == "H" && textFields[1] == "L",
            "text H packet contains wrist data and exactly 26 complete bones");

        socket.HandPackets.Clear();
        skeleton.Bones = MakeBones(25);
        Send(sender, hand, skeleton, 'L', OVRInput.Controller.LTouch);
        Check(socket.HandPackets.Count == 0, "short skeleton sends no partial text packet");

        skeleton.Bones = MakeBones(26);
        skeleton.Bones[25].Transform = null;
        Send(sender, hand, skeleton, 'L', OVRInput.Controller.LTouch);
        Check(socket.HandPackets.Count == 0, "missing final bone transform sends no partial text packet");

        skeleton.Bones = MakeBones(26);
        skeleton.Bones[25].Transform.position.x = float.NaN;
        Send(sender, hand, skeleton, 'L', OVRInput.Controller.LTouch);
        Check(socket.HandPackets.Count == 0, "non-finite bone position sends no text packet");

        skeleton.Bones = MakeBones(26);
        skeleton.Bones[0].Transform.rotation = new UnityEngine.Quaternion();
        Send(sender, hand, skeleton, 'L', OVRInput.Controller.LTouch);
        Check(socket.HandPackets.Count == 0, "invalid wrist rotation sends no text packet");

        SetField(sender, "useBinaryProtocol", true);
        skeleton.Bones = MakeBones(26);
        Send(sender, hand, skeleton, 'R', OVRInput.Controller.RTouch);
        Check(socket.HandPackets.Count == 1 && socket.HandPackets[0].StartsWith("HB,"),
            "valid binary hand frame is sent with the legacy HB prefix");
        byte[] payload = Convert.FromBase64String(socket.HandPackets[0].Substring(3));
        Check(payload.Length == 320 && payload[0] == (byte)'H' && payload[1] == (byte)'R' &&
            payload[2] == 26 && payload[3] == 0,
            "binary HB packet carries one complete 26-bone frame");

        socket.HandPackets.Clear();
        skeleton.Bones = MakeBones(26);
        skeleton.Bones[12].Transform = null;
        Send(sender, hand, skeleton, 'R', OVRInput.Controller.RTouch);
        Check(socket.HandPackets.Count == 0, "missing binary bone transform sends no partial packet");
    }

    private static void Send(HandTrackingSender sender, OVRHand hand, OVRSkeleton skeleton,
        char side, OVRInput.Controller controller)
    {
        Invoke(sender, "SendHandData", hand, skeleton, side, controller);
    }

    private static List<OVRBone> MakeBones(int count)
    {
        var bones = new List<OVRBone>();
        for (int i = 0; i < count; i++)
            bones.Add(new OVRBone
            {
                Transform = new UnityEngine.Transform
                {
                    position = new UnityEngine.Vector3(i, i + 1, i + 2),
                    rotation = UnityEngine.Quaternion.identity
                }
            });
        return bones;
    }
}
