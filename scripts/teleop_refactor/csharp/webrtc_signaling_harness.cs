using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading.Tasks;

internal static class WebRtcSignalingHarness
{
    private const int TimeoutMs = 8000;

    private static async Task<T> Wait<T>(Task<T> task, string name)
    {
        Task completed = await Task.WhenAny(task, Task.Delay(TimeoutMs));
        if (!ReferenceEquals(completed, task))
            throw new Exception("Timed out waiting for " + name);
        return await task;
    }

    private static async Task Wait(Task task, string name)
    {
        Task completed = await Task.WhenAny(task, Task.Delay(TimeoutMs));
        if (!ReferenceEquals(completed, task))
            throw new Exception("Timed out waiting for " + name);
        await task;
    }

    private static void Check(bool condition, string message)
    {
        if (!condition) throw new Exception(message);
    }

    private static async Task RunAsync(string host, int port)
    {
        VideoSignalingClient client = new VideoSignalingClient();
        ConcurrentQueue<string> errors = new ConcurrentQueue<string>();
        TaskCompletionSource<bool> connected = NewSignal();
        TaskCompletionSource<bool> disconnected = NewSignal();
        TaskCompletionSource<VideoSignalingClient.Envelope> unicode =
            new TaskCompletionSource<VideoSignalingClient.Envelope>();

        client.OnConnected += () => connected.TrySetResult(true);
        client.OnDisconnected += () => disconnected.TrySetResult(true);
        client.OnError += error => errors.Enqueue(error ?? "");
        client.OnEnvelope += envelope =>
        {
            if (envelope != null && envelope.Type == "unicode")
                unicode.TrySetResult(envelope);
        };

        Check(await client.ConnectAsync(host, port, TimeoutMs), "initial connect failed");
        await Wait(connected.Task, "initial connected event");

        // All calls overlap deliberately; ClientWebSocket itself allows one send at a
        // time, so this verifies the production send gate instead of relying on ordering.
        Task[] sends = Enumerable.Range(0, 64).Select(index =>
            client.SendAsync("concurrent", "session-a",
                "{\"index\":" + index + ",\"text\":\"并发🙂\"}"))
            .ToArray();
        await Task.WhenAll(sends);
        VideoSignalingClient.Envelope received = await Wait(unicode.Task, "fragmented UTF-8 envelope");
        Check(received.SessionId == "session-a", "fragmented envelope session id mismatch");
        Check(received.PayloadJson.Contains("分片🙂"), "fragmented UTF-8 payload was corrupted");

        await client.SendAsync("close", "session-a", "{}");
        await Wait(disconnected.Task, "close disconnect event");
        Check(!client.IsConnected, "client remained connected after server close");

        connected = NewSignal();
        disconnected = NewSignal();
        Check(await client.ConnectAsync(host, port, TimeoutMs), "reconnect failed");
        await Wait(connected.Task, "reconnect event");
        Check(client.IsConnected, "reconnected client is not open");

        await client.SendAsync("oversize", "session-b", "{}");
        await Wait(disconnected.Task, "oversize disconnect event");
        Check(errors.Any(error => error.Contains("exceeds")),
            "oversize message did not report bounded receive error");
        Check(!client.IsConnected, "client remained connected after oversize message");

        client.Dispose();
        Console.WriteLine("REAL_WEBRTC_SIGNALING_CHECKS_PASS");
    }

    private static TaskCompletionSource<bool> NewSignal()
    {
        return new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
    }

    public static int Main(string[] args)
    {
        if (args.Length != 2)
        {
            Console.Error.WriteLine("usage: harness HOST PORT");
            return 2;
        }
        try
        {
            RunAsync(args[0], int.Parse(args[1])).GetAwaiter().GetResult();
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine("REAL_WEBRTC_SIGNALING_CHECKS_FAIL: " + ex);
            return 1;
        }
    }
}
