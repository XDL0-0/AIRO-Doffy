using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Text;
using System.Globalization;
using Doffy.Protocol;
using Doffy.Networking;

public static class ProtocolHarness
{
    private static int checks;
    private static void Check(bool condition, string description)
    {
        if (!condition) throw new Exception(description);
        checks++;
    }

    private static byte[] Packet(uint frame, int index, int count, int size, byte[] payload)
    {
        byte[] packet = new byte[12 + payload.Length];
        Write32(packet, 0, frame); Write32(packet, 8, (uint)size);
        packet[4] = (byte)(index >> 8); packet[5] = (byte)index;
        packet[6] = (byte)(count >> 8); packet[7] = (byte)count;
        Buffer.BlockCopy(payload, 0, packet, 12, payload.Length);
        return packet;
    }

    private static void Write32(byte[] p, int i, uint x)
    { p[i] = (byte)(x >> 24); p[i + 1] = (byte)(x >> 16); p[i + 2] = (byte)(x >> 8); p[i + 3] = (byte)x; }

    private static void AssemblyChecks()
    {
        var a = new JpegFrameAssembler(1024, 2, .5);
        byte[] result;
        byte[] first = Packet(12, 0, 2, 5, new byte[] { 1, 2, 3 });
        byte[] last = Packet(12, 1, 2, 5, new byte[] { 4, 5 });
        Check(!a.TryAdd(last, 0, out result), "last chunk first remains incomplete");
        Check(!a.TryAdd(last, .01, out result), "duplicate does not complete frame");
        Check(a.TryAdd(first, .02, out result) && result.SequenceEqual(new byte[] { 1, 2, 3, 4, 5 }), "out-of-order variable-size chunks preserve bytes");
        Check(!a.TryAdd(first, .03, out result), "delivered frame does not reappear");
        Check(!a.TryAdd(Packet(13, 2, 2, 4, new byte[] { 1 }), .1, out result), "invalid chunk index rejected");
        Check(!a.TryAdd(Packet(13, 0, 1, 5, new byte[] { 1 }), .1, out result), "incorrect complete length rejected");
        Check(!a.TryAdd(Packet(13, 0, 1, 2000, new byte[] { 1 }), .1, out result), "oversized frame rejected");
        a.TryAdd(Packet(14, 0, 2, 2, new byte[] { 1 }), .2, out result);
        a.Expire(.8);
        Check(a.IncompleteFrames == 0, "missing chunks expire");
        Check(a.TryAdd(Packet(0, 0, 1, 1, new byte[] { 7 }), 3, out result), "restarted sender recovers after a quiet gap");
        a.Clear();
        Check(a.TryAdd(Packet(uint.MaxValue, 0, 1, 1, new byte[] { 1 }), 4, out result), "uint maximum accepted");
        Check(a.TryAdd(Packet(0, 0, 1, 1, new byte[] { 2 }), 4.1, out result), "sequence wrap accepted");
        a.Clear();
        for (uint i = 0; i < 8; i++) a.TryAdd(Packet(i, 0, 2, 2, new byte[] { 1 }), 5, out result);
        Check(a.IncompleteFrames <= 2, "inflight frame count stays bounded");
        Check(a.TryAdd(new byte[] { 255, 216, 255, 224, 0 }, 5.1, out result), "legacy raw JPEG accepted");
        Check(!a.TryAdd(new byte[] { 1, 2 }, 5.2, out result), "short malformed datagram rejected");
    }

    private static void SocketChecks()
    {
        for (int cycle = 0; cycle < 5; cycle++)
        {
            int port;
            using (var probe = new UdpClient(0)) port = ((IPEndPoint)probe.Client.LocalEndPoint).Port;
            using (var received = new ManualResetEventSlim())
            using (var receiver = new DatagramReceiver(port, packet => { if (packet.SequenceEqual(new byte[] { 8, 9 })) received.Set(); }, "127.0.0.1"))
            using (var sender = new UdpClient())
            {
                sender.Send(new byte[] { 8, 9 }, 2, new IPEndPoint(IPAddress.Loopback, port));
                Check(received.Wait(2000), "real UDP callback cycle " + cycle);
                receiver.Dispose();
                Check(!receiver.IsRunning, "worker stopped cycle " + cycle);
                using (var rebound = new UdpClient(port)) Check(rebound.Client.IsBound, "port reusable cycle " + cycle);
            }
        }
    }

    private static void TcpChecks()
    {
        TcpStatePacket packet;
        Check(TcpStatePacket.TryParse(Encoding.UTF8.GetBytes("{\"rightTCP\":{\"position\":[1,2,3],\"rotation\":[2,0,0,0],\"force\":[4,5,6]}}"), out packet), "classic TCP/force JSON accepted");
        Check(packet.rightTCP.position[2] == 3 && packet.rightTCP.rotation[0] == 1 && packet.rightTCP.force[0] == 4, "TCP units and wxyz order preserved, quaternion normalized");
        Check(TcpStatePacket.TryParse(Encoding.UTF8.GetBytes("{\"leftTCP\":{\"force\":[1,2,3]}}"), out packet), "optional legacy partial left-arm feedback accepted");
        Check(!TcpStatePacket.TryParse(Encoding.UTF8.GetBytes("{\"rightTCP\":{\"position\":[NaN,0,0]}}"), out packet), "nonfinite feedback rejected");
        Check(!TcpStatePacket.TryParse(Encoding.UTF8.GetBytes("{\"rightTCP\":{\"rotation\":[0,0,0,0]}}"), out packet), "zero quaternion rejected");
        Check(!TcpStatePacket.TryParse(Encoding.UTF8.GetBytes("{\"rightTCP\":{\"force\":[1,2]}}"), out packet), "wrong vector shape rejected");
        Check(!TcpStatePacket.TryParse(Encoding.UTF8.GetBytes("{}"), out packet), "unrelated JSON cannot mark feedback fresh");
        Check(!TcpStatePacket.TryParse(new byte[70000], out packet), "oversized JSON rejected before parsing");
    }

    private static void HeightMappingChecks()
    {
        var map = new UpperLimbHeightMapping();
        Check(map.Step(true, true, .3f, .3f, 1, .03f, .03f, .12f) == 0, "down calibration anchors zero");
        float raised = map.Step(true, true, .3f, 0, 1, .03f, .03f, .12f);
        Check(raised > .99f && raised <= 1, "shoulder-height elbow anchors one");
        Check(map.Step(true, false, .3f, .3f, 1, .03f, .03f, .12f) == raised, "tracking loss holds alpha instead of jumping to zero");
        Check(map.Step(false, true, .3f, 0, 1, .03f, .03f, .12f) == 0, "new calibration resets filter");
        Check(map.Step(true, true, 0, 0, 1, .03f, .03f, .12f) == 0, "degenerate hang reference cannot divide by zero");
        float last = 0;
        bool bounded = true;
        for (int i = 0; i < 180; i++)
        {
            float value = map.Step(true, true, .3f, .15f, 1f / 60, .03f, .03f, .12f);
            bounded &= value >= last && value <= .5f + 1e-6f;
            last = value;
        }
        Check(bounded, "half-height convergence stays monotonic and bounded across 180 frames");
        Check(Math.Abs(last - .5f) < .0001f, "half height converges to half alpha");
    }

    private static void WireFormatChecks()
    {
        var culture = CultureInfo.CurrentCulture;
        try
        {
            Thread.CurrentThread.CurrentCulture = CultureInfo.GetCultureInfo("fr-FR");
            string side = TeleopWireFormat.ControllerSide(1.25f, -2.5f, 3, 0, 0, 0, 1, .25f, -.5f, .75f, 1, 0, 1, 0);
            Check(side.Split(',').Length == 14 && side.StartsWith("1.250000,-2.500000"), "controller CSV is independent of headset locale");
            var text = new StringBuilder();
            TeleopWireFormat.AppendFloat(text, .125f, "F4");
            Check(text.ToString() == ",0.1250", "hand CSV keeps decimal point under French locale");
            var bytes = new byte[4];
            TeleopWireFormat.WriteFloatLittleEndian(bytes, 0, -1.25f);
            Check(bytes.SequenceEqual(new byte[] { 0, 0, 160, 191 }), "hand float bytes preserve little-endian IEEE754");
        }
        finally { Thread.CurrentThread.CurrentCulture = culture; }
    }

    // Optional interoperability receiver: Python can send actual classic JPEG chunks.
    private static void ReceiveImage(int port, string output)
    {
        var a = new JpegFrameAssembler();
        using (var ready = new ManualResetEventSlim())
        using (var receiver = new DatagramReceiver(port, packet =>
        {
            byte[] image;
            if (a.TryAdd(packet, DateTime.UtcNow.Ticks / 10000000.0, out image))
            { File.WriteAllBytes(output, image); ready.Set(); }
        }, "127.0.0.1"))
        {
            Console.WriteLine("READY " + port);
            if (!ready.Wait(10000)) throw new Exception("No complete image received");
        }
    }

    private static void ReceiveTcp(int port, string output)
    {
        using (var ready = new ManualResetEventSlim())
        using (var receiver = new DatagramReceiver(port, bytes =>
        {
            TcpStatePacket sample;
            if (TcpStatePacket.TryParse(bytes, out sample))
            { File.WriteAllText(output, Newtonsoft.Json.JsonConvert.SerializeObject(sample)); ready.Set(); }
        }, "127.0.0.1"))
        {
            Console.WriteLine("READY " + port);
            if (!ready.Wait(5000)) throw new Exception("No valid TCP sample received");
        }
    }

    private static void SendController(int port)
    {
        Thread.CurrentThread.CurrentCulture = CultureInfo.GetCultureInfo("fr-FR");
        string left = TeleopWireFormat.ControllerSide(1.25f, -2.5f, 3, 0, 0, 0, 1, .25f, -.5f, .75f, 1, 0, 1, 0);
        string right = TeleopWireFormat.ControllerSide(-1.25f, 2.5f, -3, 0, 0, 0, 1, -.25f, .5f, .25f, 0, 1, 0, 1);
        byte[] packet = Encoding.UTF8.GetBytes("C,42,1234567890," + left + "," + right);
        using (var sender = new UdpClient()) sender.Send(packet, packet.Length, new IPEndPoint(IPAddress.Loopback, port));
    }

    public static int Main(string[] args)
    {
        try
        {
            if (args.Length == 3 && args[0] == "receive")
                ReceiveImage(int.Parse(args[1]), args[2]);
            else if (args.Length == 3 && args[0] == "tcp")
                ReceiveTcp(int.Parse(args[1]), args[2]);
            else if (args.Length == 2 && args[0] == "controller")
                SendController(int.Parse(args[1]));
            else { AssemblyChecks(); TcpChecks(); WireFormatChecks(); HeightMappingChecks(); SocketChecks(); Console.WriteLine("PASS " + checks + " C# protocol/lifecycle checks"); }
            return 0;
        }
        catch (Exception error) { Console.Error.WriteLine(error); return 1; }
    }
}
