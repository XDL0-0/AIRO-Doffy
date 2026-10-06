using System;
using System.Reflection;

// Test adapters only: exercise the production guard's real event subscriptions.
namespace UnityEngine
{
    public enum FindObjectsInactive { Exclude, Include }

    [AttributeUsage(AttributeTargets.Class)]
    public sealed class DefaultExecutionOrderAttribute : Attribute
    {
        public DefaultExecutionOrderAttribute(int order) { }
    }

    public class Object
    {
        public static T[] FindObjectsByType<T>(FindObjectsInactive inactive) { return new T[0]; }
    }

    public class MonoBehaviour : Object { }
}
public class OVRHand : UnityEngine.MonoBehaviour { public bool IsSystemGestureInProgress; }
public class AppManager
{
    public int Lost, Available;
    public bool IsStreaming;
    public int TrackingMode;
    public void HandleTrackingLost(string reason) { Lost++; }
    public void HandleTrackingAvailable(string reason) { Available++; }
}
public class OVRManager
{
    public static OVRManager instance = new OVRManager();
    public bool isUserPresent = true;
    public static bool isHmdPresent = true, hasVrFocus = true, hasInputFocus = true;
    public static OVRTracker tracker = new OVRTracker();
    public static event Action HMDUnmounted, HMDMounted, VrFocusLost, VrFocusAcquired,
        InputFocusLost, InputFocusAcquired, TrackingLost, TrackingAcquired;
    public static void Unmount() { HMDUnmounted?.Invoke(); }
    public static void Mount() { HMDMounted?.Invoke(); }
    public static void LoseTracking() { TrackingLost?.Invoke(); }
    public static void AcquireTracking() { TrackingAcquired?.Invoke(); }
    public static void LoseVrFocus() { VrFocusLost?.Invoke(); }
    public static void AcquireVrFocus() { VrFocusAcquired?.Invoke(); }
    public static void LoseInputFocus() { InputFocusLost?.Invoke(); }
    public static void AcquireInputFocus() { InputFocusAcquired?.Invoke(); }
}
public class OVRTracker { public bool isPositionTracked = true; }
public static class TrackingGuardHarness
{
    private static void Invoke(TeleopTrackingGuard guard, string name, params object[] args)
    { typeof(TeleopTrackingGuard).GetMethod(name, BindingFlags.Instance | BindingFlags.NonPublic).Invoke(guard, args); }
    private static void Check(bool condition, string name)
    { if (!condition) throw new Exception(name); }
    public static int Main()
    {
        var app = new AppManager();
        var guard = new TeleopTrackingGuard();
        guard.Initialize(app);
        Invoke(guard, "OnEnable");
        OVRManager.Unmount(); OVRManager.LoseTracking();
        Check(!guard.IsAvailable && app.Lost == 2, "independent losses must close availability");
        OVRManager.AcquireTracking();
        Check(!guard.IsAvailable && app.Available == 0, "tracking recovery cannot override an unmounted headset");
        OVRManager.Mount();
        Check(guard.IsAvailable && app.Available == 1, "all recovered signals report availability once");
        OVRManager.LoseVrFocus(); OVRManager.LoseInputFocus();
        OVRManager.AcquireVrFocus();
        Check(!guard.IsAvailable, "VR focus cannot override lost input focus");
        OVRManager.AcquireInputFocus();
        Check(guard.IsAvailable, "input focus recovery completes the pair");
        Invoke(guard, "OnApplicationPause", true);
        OVRManager.AcquireTracking();
        Check(!guard.IsAvailable, "tracking callback cannot override application pause");
        Invoke(guard, "OnApplicationPause", false);
        Check(guard.IsAvailable, "resume permits explicit recalibration");
        Invoke(guard, "OnDisable");
        int lost = app.Lost;
        OVRManager.Unmount();
        Check(app.Lost == lost, "disabled guard unsubscribes");
        OVRManager.hasInputFocus = false;
        Invoke(guard, "OnEnable");
        Check(!guard.IsAvailable && app.Lost == lost + 1, "pre-existing focus loss is seeded on subscription");
        OVRManager.hasInputFocus = true;
        OVRManager.AcquireInputFocus();
        Check(guard.IsAvailable, "seeded loss recovers through the matching SDK event");
        Invoke(guard, "OnDisable");
        Console.WriteLine("PASS 10 tracking event/lifecycle checks (SDK event adapters)");
        return 0;
    }
}
