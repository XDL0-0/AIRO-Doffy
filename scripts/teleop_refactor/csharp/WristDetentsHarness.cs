using System;

public static class WristDetentsHarness
{
    private static int _checks;

    private static void Check(bool condition, string message)
    {
        if (!condition)
            throw new InvalidOperationException(message);
        _checks++;
    }

    private static WristDetentTracker NewTracker()
    {
        return new WristDetentTracker(10f, 0.75f);
    }

    private static void ResetAndNoMotion()
    {
        WristDetentTracker tracker = NewTracker();
        tracker.Reset();
        tracker.BeginSample(37f);
        Check(tracker.UpdateSample(37f) == 0, "reset and same-angle sample must not pulse");
        Check(tracker.AngleDegrees == 0f, "initial controller angle must not offset the logical dial");
        Check(tracker.AdvanceBy(0f) == 0, "no motion must not pulse");

        tracker.AdvanceBy(35f);
        tracker.Reset();
        tracker.BeginSample(-130f);
        Check(tracker.UpdateSample(-130f) == 0, "reset after movement must not emit a starting pulse");
        Check(tracker.AngleDegrees == 0f, "reset after movement must return the logical dial to zero");

        tracker.AdvanceBy(35f);
        tracker.BeginSample(20f);
        tracker.ClearSample();
        Check(tracker.UpdateSample(-170f) == 0, "a new drag baseline must not use the prior drag's sample");
        Check(tracker.AngleDegrees == 35f, "clearing a sample must preserve the accumulated dial angle");
    }

    private static void CrossingsWorkInBothDirections()
    {
        WristDetentTracker clockwise = NewTracker();
        Check(clockwise.AdvanceBy(10.5f) == 0, "positive jitter below the hysteresis threshold must not pulse");
        Check(clockwise.AdvanceBy(0.3f) == 1, "positive movement past one detent must pulse once");
        Check(clockwise.AdvanceBy(10f) == 1, "positive movement across the next detent must pulse once");
        Check(clockwise.DetentIndex == 2, "positive crossings must advance the detent index");

        WristDetentTracker counterClockwise = NewTracker();
        Check(counterClockwise.AdvanceBy(-10.5f) == 0, "negative jitter below the hysteresis threshold must not pulse");
        Check(counterClockwise.AdvanceBy(-0.3f) == -1, "negative movement past one detent must pulse once");
        Check(counterClockwise.AdvanceBy(-10f) == -1, "negative movement across the next detent must pulse once");
        Check(counterClockwise.DetentIndex == -2, "negative crossings must decrease the detent index");
    }

    private static void WrapIsUnwrapped()
    {
        Check(Math.Abs(WristDetentTracker.UnwrapDelta(179f, -179f) - 2f) < 0.0001f,
            "+180 wrap must take the short positive path");
        Check(Math.Abs(WristDetentTracker.UnwrapDelta(-179f, 179f) + 2f) < 0.0001f,
            "-180 wrap must take the short negative path");

        WristDetentTracker tracker = NewTracker();
        tracker.BeginSample(179f);
        Check(tracker.UpdateSample(-179f) == 0, "two-degree wrap must not create a detent");
        Check(Math.Abs(tracker.AngleDegrees - 2f) < 0.0001f, "wrapped samples must accumulate continuously");
    }

    private static void HysteresisSuppressesJitter()
    {
        WristDetentTracker tracker = NewTracker();
        Check(tracker.AdvanceBy(10.8f) == 1, "crossing outward beyond the positive threshold must pulse");
        Check(tracker.AdvanceBy(-1.4f) == 0, "jitter inside the hysteresis band must not reverse the tick");
        Check(tracker.AdvanceBy(-0.2f) == -1, "crossing the return threshold must reverse the tick once");
        Check(tracker.AdvanceBy(1f) == 0, "re-entering the hysteresis band must stay quiet");
        Check(tracker.AdvanceBy(1f) == 1, "crossing the outward threshold again must pulse once");
    }

    private static void LargeMovesCountMultipleSteps()
    {
        WristDetentTracker tracker = NewTracker();
        Check(tracker.AdvanceBy(35f) == 3, "a multi-detent positive move must report every crossed detent");
        Check(tracker.AdvanceBy(-70f) == -6, "a multi-detent negative move must report every crossed detent");
        Check(tracker.DetentIndex == -3, "multi-step movement must end at the matching negative index");
    }

    private static void HugeMovesDoNotDrainOnStationaryFrames()
    {
        WristDetentTracker tracker = NewTracker();
        int reported = tracker.AdvanceBy(2000f);
        Check(reported == 64, "a pathological jump must have bounded per-update work");
        Check(tracker.AdvanceBy(0f) == 0, "a stationary frame must not drain crossings left by a large jump");
        Check(tracker.AdvanceBy(0f) == 0, "resynchronization must remain quiet on later stationary frames");
    }

    public static int Main()
    {
        ResetAndNoMotion();
        CrossingsWorkInBothDirections();
        WrapIsUnwrapped();
        HysteresisSuppressesJitter();
        LargeMovesCountMultipleSteps();
        HugeMovesDoNotDrainOnStationaryFrames();
        Console.WriteLine("PASS " + _checks + " wrist detent checks");
        return 0;
    }
}
