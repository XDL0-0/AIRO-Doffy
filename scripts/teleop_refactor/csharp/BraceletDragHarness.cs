using System;

internal static class BraceletDragMathChecks
{
    private static int _checks;

    private static void Check(bool condition, string name)
    {
        _checks++;
        if (!condition)
            throw new Exception("FAILED: " + name);
    }

    private static float Wrap(float degrees)
    {
        float wrapped = degrees % 360f;
        if (wrapped > 180f) wrapped -= 360f;
        else if (wrapped < -180f) wrapped += 360f;
        return wrapped;
    }

    private static bool CanAcquire(
        float radialGap, float axialOffset, float previousGap,
        bool hasPrevious, bool selected)
    {
        return BraceletDragMath.IsWithinPokeAcquireBand(
                   radialGap, axialOffset, 0.010f, 0.005f, 0.020f, 0.002f) &&
               BraceletDragMath.ShouldAcquirePoke(
                   radialGap, previousGap, hasPrevious, selected,
                   0.005f, 0.003f, 0.020f, 0.0005f);
    }

    public static int Main()
    {
        foreach (int direction in new[] { 1, -1 })
        {
            var tracker = new WristDetentTracker();
            tracker.BeginSample(0f);
            for (int step = 1; step <= 24; step++)
                tracker.UpdateSample(Wrap(direction * step * 45f));
            Check(Math.Abs(tracker.AngleDegrees - direction * 1080f) < 0.001f,
                (direction > 0 ? "clockwise" : "counterclockwise") + " three full turns");
        }

        Check(Math.Abs(BraceletDragMath.UnwrapDeltaDegrees(179f, -179f) - 2f) < 0.001,
            "+180 boundary crossing");
        Check(Math.Abs(BraceletDragMath.UnwrapDeltaDegrees(-179f, 179f) + 2f) < 0.001,
            "-180 boundary crossing");
        Check(BraceletDragMath.UnwrapDeltaDegrees(0f, 180f) == 180f,
            "positive exact half-turn");
        Check(BraceletDragMath.UnwrapDeltaDegrees(0f, -180f) == -180f,
            "negative exact half-turn");

        var rebased = new WristDetentTracker();
        rebased.BeginSample(0f);
        rebased.UpdateSample(30f);
        rebased.ClearSample();
        rebased.BeginSample(179f);
        Check(Math.Abs(rebased.AngleDegrees - 30f) < 0.001f,
            "tracking recovery baseline does not jump angle");
        rebased.UpdateSample(-179f);
        Check(Math.Abs(rebased.AngleDegrees - 32f) < 0.001f,
            "motion resumes smoothly across 180 after rebase");

        Check(BraceletDragMath.TryGetCylinderAngle(
                2, 0, -1, 0, 1, false, 0, 0, .035, out float nearest) && Math.Abs(nearest) < .001f,
            "first ray sample selects nearest forward root");
        Check(BraceletDragMath.TryGetCylinderAngle(
                2, 0, -1, 0, 1, true, 179, 1, .035, out float continuedPositive) &&
              Math.Abs(continuedPositive - 180f) < .001f,
            "antipodal root follows previous angle across 180");
        Check(BraceletDragMath.TryGetCylinderAngle(
                2, 0, -1, 0, 1, true, 1, 1, .035, out float continuedNearZero) &&
              Math.Abs(continuedNearZero) < .001f,
            "root selection stays on near-zero branch");
        Check(BraceletDragMath.TryGetCylinderAngle(
                2, 0, -1, 0, 1, true, 90, 1, .035, out float positiveTie) &&
              Math.Abs(positiveTie - 180f) < .001f,
            "positive travel direction resolves exact antipodal tie");
        Check(BraceletDragMath.TryGetCylinderAngle(
                2, 0, -1, 0, 1, true, 90, -1, .035, out float negativeTie) &&
              Math.Abs(negativeTie) < .001f,
            "negative travel direction resolves exact antipodal tie");

        Check(!BraceletDragMath.TryGetPointAngle(.0001, 0, out _),
            "near-axis bearing rejected");
        Check(!BraceletDragMath.TryGetCylinderAngle(
                .0001, 0, 0, 0, 1, false, 0, 0, .035, out _),
            "axis-parallel ray near axis rejected");
        Check(BraceletDragMath.TryGetCylinderAngle(
                -2, 1, 1, 0, 1, false, 0, 0, .035, out float tangent) &&
              Math.Abs(tangent - 90f) < .001f,
            "exact grazing root remains stable");
        Check(BraceletDragMath.TryGetCylinderAngle(
                -2, 1.01, 1, 0, 1, false, 0, 0, .035, out float nearGrazing) &&
              Math.Abs(nearGrazing - 90f) < .001f,
            "near-grazing miss uses closest forward point");
        Check(!BraceletDragMath.TryGetCylinderAngle(
                -2, 1.036, 1, 0, 1, false, 0, 0, .035, out _),
            "grazing miss beyond tolerance rejected");

        Check(CanAcquire(.020f, .0115f, .021f, true, false),
            "radial approach acquires inside strip");
        Check(!CanAcquire(.020f, .013f, .021f, true, true),
            "adjacent navigation tile selection cannot acquire cuff");
        Check(CanAcquire(.007f, 0, .007f, true, false),
            "fingertip contact acquires without deep poke");
        Check(!CanAcquire(.018f, 0, .018f, true, false),
            "static non-contact fingertip does not acquire");

        bool requireLift = true;
        Check(!BraceletDragMath.ObservePokeLift(ref requireLift, true) && requireLift,
            "tracking-loss rearm stays blocked while finger remains in release band");
        Check(BraceletDragMath.ObservePokeLift(ref requireLift, false) && !requireLift,
            "rearm opens on observed lift");
        Check(!CanAcquire(.018f, 0, .018f, true, false) &&
              CanAcquire(.007f, 0, .018f, true, false),
            "reacquisition requires a new approach after lift");

        Console.WriteLine("PASS " + _checks + " checks");
        return 0;
    }
}
