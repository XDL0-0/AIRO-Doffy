using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

// Minimal adapters for the production coordinators. Unity scene components are
// deliberately not referenced: every asynchronous boundary is controlled by a
// TaskCompletionSource so the race and cleanup behavior is observable.
public sealed class VideoStreamManager
{
    public sealed class StartCall
    {
        public string Host;
        public int SignalingPort;
        public string Preset;
        public readonly TaskCompletionSource<bool> Completion =
            new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
    }

    public readonly List<StartCall> Starts = new List<StartCall>();
    public readonly List<string> Stops = new List<string>();

    public Task<bool> StartVideoSession(string host, int signalingPort, string preset = "720p30")
    {
        var call = new StartCall
        {
            Host = host,
            SignalingPort = signalingPort,
            Preset = preset
        };
        Starts.Add(call);
        return call.Completion.Task;
    }

    public Task StopVideoSession(string reason = "user_stop")
    {
        Stops.Add(reason ?? string.Empty);
        return Task.CompletedTask;
    }

    public void CompleteStart(int index, bool success)
    {
        Starts[index].Completion.TrySetResult(success);
    }
}

public sealed class RecordingController
{
    public bool IsRecording { get; private set; }
    public int StopCalls { get; private set; }

    public void StartRecording()
    {
        IsRecording = true;
    }

    public void StopRecording()
    {
        if (!IsRecording) return;
        IsRecording = false;
        StopCalls++;
    }
}

public static class TeleopReferenceFrame
{
    public static int ClearCalls { get; private set; }

    public static void Clear()
    {
        ClearCalls++;
    }

    public static void Reset()
    {
        ClearCalls = 0;
    }
}

public static class SessionHarness
{
    private const int TimeoutMs = 4000;
    private static int checks;

    private sealed class Fixture
    {
        public readonly VideoStreamManager Video = new VideoStreamManager();
        public readonly RecordingController Recorder = new RecordingController();
        public readonly TeleopSessionCoordinator Session;

        public Fixture()
        {
            Session = new TeleopSessionCoordinator(
                () => Video,
                () => "127.0.0.1",
                () => 8765,
                () => true,
                () => false,
                () => Recorder,
                _ => { });
        }
    }

    private static void Check(bool condition, string message)
    {
        if (!condition) throw new InvalidOperationException(message);
        checks++;
    }

    private static async Task WaitFor(Func<bool> condition, string name)
    {
        DateTime deadline = DateTime.UtcNow.AddMilliseconds(TimeoutMs);
        while (!condition())
        {
            if (DateTime.UtcNow >= deadline)
                throw new TimeoutException("Timed out waiting for " + name);
            await Task.Delay(1);
        }
    }

    private static async Task WaitForTask(Task task, string name)
    {
        Task completed = await Task.WhenAny(task, Task.Delay(TimeoutMs));
        if (!ReferenceEquals(completed, task))
            throw new TimeoutException("Timed out waiting for " + name);
        await task;
    }

    private static async Task StopDuringStart()
    {
        TeleopReferenceFrame.Reset();
        var fixture = new Fixture();
        fixture.Recorder.StartRecording();

        Task start = fixture.Session.StartAsync(() => true);
        Check(fixture.Session.State == TeleopSessionState.VideoConnecting,
            "stop-during-start entered video connecting");
        Check(fixture.Video.Starts.Count == 1, "video start was issued once");

        fixture.Session.Stop("stop_during_start");
        Check(fixture.Session.State == TeleopSessionState.Idle,
            "stop-during-start returned to idle immediately");
        Check(!fixture.Session.IsStreaming && !fixture.Session.CanSend,
            "stop-during-start closed the send gate");
        Check(fixture.Recorder.StopCalls == 1 && !fixture.Recorder.IsRecording,
            "stop-during-start stopped recording");
        Check(TeleopReferenceFrame.ClearCalls == 1,
            "stop-during-start cleared the reference frame");

        // The old video operation completes after Stop. Its success must be
        // discarded and its manager must still be stopped.
        fixture.Video.CompleteStart(0, true);
        await WaitForTask(start, "cancelled session start");
        await WaitFor(() => fixture.Video.Stops.Count >= 2,
            "cancelled video manager stop");
        Check(fixture.Session.State == TeleopSessionState.Idle && !fixture.Session.IsStreaming,
            "late start completion did not revive the stopped session");
    }

    private static async Task ImmediateReplacementStart()
    {
        TeleopReferenceFrame.Reset();
        var fixture = new Fixture();

        Task first = fixture.Session.StartAsync(() => true);
        await WaitFor(() => fixture.Video.Starts.Count == 1, "first video start");
        fixture.Session.Stop("replace");
        Task replacement = fixture.Session.StartAsync(() => true);
        Check(fixture.Session.IsStarting && fixture.Session.IsStreaming,
            "replacement start became active immediately");

        // The replacement waits for the cancelled operation to unwind before
        // touching the manager. Releasing the first start unlocks that queue.
        fixture.Video.CompleteStart(0, true);
        await WaitFor(() => fixture.Video.Starts.Count == 2,
            "replacement video start");
        fixture.Video.CompleteStart(1, true);
        await WaitForTask(first, "stale first start");
        await WaitForTask(replacement, "replacement session start");

        Check(fixture.Session.State == TeleopSessionState.Streaming,
            "replacement start reached streaming");
        Check(fixture.Session.CanSend && !fixture.Session.NeedsRecalibration,
            "replacement start reopened the send gate");

        fixture.Session.Stop("replacement_cleanup");
        await WaitFor(() => fixture.Video.Stops.Count >= 3,
            "replacement cleanup stop");
    }

    private static async Task LateVideoSuccessCannotClearTrackingPause()
    {
        TeleopReferenceFrame.Reset();
        var fixture = new Fixture();
        Task start = fixture.Session.StartAsync(() => true);
        await WaitFor(() => fixture.Video.Starts.Count == 1, "tracking test video start");

        fixture.Session.HandleTrackingLost("focus lost", true);
        Check(fixture.Session.State == TeleopSessionState.TrackingLost,
            "tracking loss entered paused state");
        Check(fixture.Session.NeedsRecalibration && !fixture.Session.CanSend,
            "tracking loss closed control before video completion");

        fixture.Video.CompleteStart(0, true);
        await WaitForTask(start, "late video completion");
        Check(fixture.Session.State == TeleopSessionState.TrackingLost,
            "late video success did not clear tracking pause");
        Check(fixture.Session.NeedsRecalibration && !fixture.Session.CanSend,
            "late video success did not reopen control");

        fixture.Session.Stop("tracking_cleanup");
        await WaitFor(() => fixture.Video.Stops.Count >= 2,
            "tracking cleanup stop");
    }

    private static async Task FailedCalibrationStopsRecordingAndControl()
    {
        TeleopReferenceFrame.Reset();
        var fixture = new Fixture();
        fixture.Recorder.StartRecording();

        Task start = fixture.Session.StartAsync(() => false);
        await WaitForTask(start, "failed calibration");
        Check(fixture.Session.State == TeleopSessionState.Idle,
            "failed calibration returned to idle");
        Check(!fixture.Session.IsStreaming && !fixture.Session.CanSend,
            "failed calibration disallowed control");
        Check(fixture.Session.LastError == "Error: Reference calibration failed",
            "failed calibration exposed the expected error");
        Check(fixture.Recorder.StopCalls == 1 && !fixture.Recorder.IsRecording,
            "failed calibration stopped recording");
        Check(TeleopReferenceFrame.ClearCalls == 1,
            "failed calibration cleared the reference frame");
        Check(fixture.Video.Starts.Count == 0,
            "failed calibration did not start video");
        await WaitFor(() => fixture.Video.Stops.Count >= 1,
            "failed calibration video cleanup");
    }

    private static async Task StopIsIdempotent()
    {
        TeleopReferenceFrame.Reset();
        var fixture = new Fixture();
        Task start = fixture.Session.StartAsync(() => true);
        await WaitFor(() => fixture.Video.Starts.Count == 1, "idempotence video start");
        fixture.Video.CompleteStart(0, true);
        await WaitForTask(start, "idempotence session start");
        Check(fixture.Session.State == TeleopSessionState.Streaming,
            "idempotence setup reached streaming");

        fixture.Recorder.StartRecording();
        int stopsBefore = fixture.Video.Stops.Count;
        fixture.Session.Stop("first_stop");
        await WaitFor(() => fixture.Video.Stops.Count > stopsBefore,
            "first idempotent stop");
        fixture.Session.Stop("second_stop");
        await Task.Delay(20);

        Check(fixture.Session.State == TeleopSessionState.Idle && !fixture.Session.CanSend,
            "idempotent stop remains idle and gated");
        Check(fixture.Recorder.StopCalls == 1,
            "second stop did not stop recording again");
        Check(TeleopReferenceFrame.ClearCalls == 1,
            "second stop did not clear the reference twice");
        Check(fixture.Video.Stops.Count == stopsBefore + 1,
            "second stop did not issue another manager stop");
    }

    public static int Main()
    {
        try
        {
            TeleopCoreTests.RunAll();
            StopDuringStart().GetAwaiter().GetResult();
            ImmediateReplacementStart().GetAwaiter().GetResult();
            LateVideoSuccessCannotClearTrackingPause().GetAwaiter().GetResult();
            FailedCalibrationStopsRecordingAndControl().GetAwaiter().GetResult();
            StopIsIdempotent().GetAwaiter().GetResult();
            Console.WriteLine("PASS TeleopCoreTests.RunAll + " + checks +
                " session coordinator behavior checks");
            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine("FAIL session coordinator harness: " + error);
            return 1;
        }
    }
}
