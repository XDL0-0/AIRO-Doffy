using System;
using System.Globalization;
using System.Threading;
using Doffy.Protocol;

internal static class BodyPoseWireFormatHarness
{
    private static int Main()
    {
        CultureInfo french = CultureInfo.GetCultureInfo("fr-FR");
        Thread.CurrentThread.CurrentCulture = french;
        Thread.CurrentThread.CurrentUICulture = french;

        Emit(100, false, 0.75f, true, MakeJoints(BodyPoseWireFormat.UpperBodyJointCount), 0.25f, 0.625f, true, false);
        Emit(101, true, 0.875f, true, MakeJoints(BodyPoseWireFormat.FullBodyJointCount), 0.5f, 0.75f, true, true);

        Emit(102, true, 0.9f, false, MakeJoints(BodyPoseWireFormat.FullBodyJointCount), 0.4f, 0.8f, true, true);

        BodyPoseWireFormat.Joint[] edgeCases = MakeJoints(BodyPoseWireFormat.UpperBodyJointCount);
        edgeCases[0] = new BodyPoseWireFormat.Joint(0f, 0f, 0f, 0f, 0f, 0f, 1f, true, true);
        edgeCases[1] = new BodyPoseWireFormat.Joint(1.25f, -2.5f, 3.75f, 0f, 0f, 0.5f, 0.5f, false, true);
        edgeCases[2] = new BodyPoseWireFormat.Joint(float.NaN, 2f, 3f, 0f, 0f, 0.25f, 0.75f, true, true);
        edgeCases[3] = new BodyPoseWireFormat.Joint(1f, float.PositiveInfinity, 3f, 0f, 0f, 0.25f, 0.75f, true, true);
        edgeCases[4] = new BodyPoseWireFormat.Joint(1f, 2f, 3f, float.NaN, 0f, 0f, 1f, true, true);
        edgeCases[5] = new BodyPoseWireFormat.Joint(1f, 2f, 3f, 0f, 0f, 0f, 0f, true, true);
        edgeCases[6] = new BodyPoseWireFormat.Joint(4.5f, 5.5f, 6.5f, 0f, 0f, 1f, 0f, true, false);
        edgeCases[7] = new BodyPoseWireFormat.Joint(1f, 2f, float.NegativeInfinity, 0f, float.PositiveInfinity, 0f, 1f, true, true);
        edgeCases[8] = new BodyPoseWireFormat.Joint(1f, 2f, 3f, 0f, 0f, 0.25f, 0.75f, false, false);
        Emit(103, false, 0.5f, true, edgeCases, 0.25f, 0.5f, false, false);

        BodyPoseWireFormat.Joint[] stress = new BodyPoseWireFormat.Joint[BodyPoseWireFormat.FullBodyJointCount];
        for (int i = 0; i < stress.Length; i++)
        {
            stress[i] = new BodyPoseWireFormat.Joint(float.MaxValue, float.MinValue, float.Epsilon,
                float.MaxValue, float.MinValue, float.MaxValue, float.MinValue, true, true);
        }
        Emit(104, true, float.MaxValue, true, stress, float.MinValue, float.NaN, true, false);

        Emit(105, false, 0.2f, true, null, 0.1f, 0.2f, false, false);

        BodyPoseWireFormat.Joint[] partial = MakeJoints(2);
        Emit(106, false, 0.3f, true, partial, 0.1f, 0.2f, false, false);

        // Valid estimates may survive without either current tracking bit. Explicit
        // tracking bits also must not override invalid position/orientation data.
        BodyPoseWireFormat.Joint[] trackingBits = MakeJoints(BodyPoseWireFormat.UpperBodyJointCount);
        trackingBits[0] = new BodyPoseWireFormat.Joint(0.25f, -0.5f, 0f, 0f, 0f, 0.25f, 0.75f,
            true, true, false, false);
        trackingBits[1] = new BodyPoseWireFormat.Joint(1.25f, -1.5f, 0.125f, 0f, 0f, 0.25f, 0.75f,
            true, true, true, false);
        trackingBits[2] = new BodyPoseWireFormat.Joint(2.25f, -2.5f, 0.25f, 0f, 0f, 0.25f, 0.75f,
            true, true, false, true);
        trackingBits[3] = new BodyPoseWireFormat.Joint(3.25f, -3.5f, 0.375f, 0f, 0f, 0.25f, 0.75f,
            false, true, true, true);
        trackingBits[4] = new BodyPoseWireFormat.Joint(4.25f, -4.5f, 0.5f, 0f, 0f, 0.25f, 0.75f,
            true, false, true, true);
        Emit(107, false, 0.6f, true, trackingBits, 0.1f, 0.2f, false, false);
        return 0;
    }

    private static BodyPoseWireFormat.Joint[] MakeJoints(int count)
    {
        BodyPoseWireFormat.Joint[] joints = new BodyPoseWireFormat.Joint[count];
        for (int i = 0; i < joints.Length; i++)
        {
            joints[i] = new BodyPoseWireFormat.Joint(i + 0.25f, -0.5f - i, i * 0.125f,
                0f, 0f, 0.25f, 0.75f, true, true);
        }
        return joints;
    }

    private static void Emit(long frameId, bool fullBody, float confidence, bool trackingValid,
        BodyPoseWireFormat.Joint[] joints, float elbowAlpha, float wrmConfidence, bool wrmEnabled,
        bool wrmCalibrated)
    {
        string packet = BodyPoseWireFormat.Serialize(frameId, 1728123456789012345L, fullBody,
            confidence, trackingValid, joints, elbowAlpha, wrmConfidence, wrmEnabled, wrmCalibrated);
        Console.WriteLine(packet);
    }
}
