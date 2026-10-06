using System;
using Doffy.UI;

public static class WristVisibilityHarness
{
    private static int checks;
    private static void Require(bool condition, string name)
    {
        checks++;
        if (!condition) throw new Exception(name);
    }
    public static void Main()
    {
        var gate = new WristVisibilityGate();
        Require(!gate.Step(false, true, false, .1f, .18f, .25f), "raise requires dwell");
        Require(gate.Step(false, true, false, .1f, .18f, .25f), "sustained view opens");
        Require(gate.Step(false, false, false, 5, .18f, .25f), "hysteresis band retains visible");
        Require(gate.Step(false, false, true, .1f, .18f, .25f), "brief resting excursion does not flicker");
        Require(gate.Step(false, false, false, .1f, .18f, .25f), "return cancels hide timer");
        Require(gate.Step(false, false, true, .2f, .18f, .25f), "hide timer restarted");
        Require(!gate.Step(false, false, true, .1f, .18f, .25f), "sustained resting pose hides");
        Require(!gate.Step(false, false, false, 5, .18f, .25f), "hysteresis band retains hidden");
        Require(gate.Step(false, true, false, .2f, .18f, .25f), "raise restores UI");
        Require(!gate.Step(true, true, false, 0, .18f, .25f), "teleop/tracking block immediate even in viewing pose");
        Require(!gate.Step(false, true, false, .1f, .18f, .25f), "block resets reveal dwell");
        Require(!gate.Step(false, false, true, 1, .18f, .25f), "teleop end at rest cannot reveal");
        Require(!gate.Step(false, true, false, -.1f, .18f, .25f), "negative time ignored");
        Require(gate.Step(false, true, false, .2f, .18f, .25f), "new raise reveals");
        gate.Reset();
        Require(!gate.Visible, "disable resets state");
        Console.WriteLine("PASS: " + checks + " wrist visibility checks");
    }
}
