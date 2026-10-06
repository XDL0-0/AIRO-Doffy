using System;
using System.Globalization;
using System.Text;

namespace Doffy.Protocol
{
    /// <summary>
    /// Pure, locale-independent BODY v1 JSON encoding for diagnostic UDP.
    /// Tracked flags are additive v1 fields; validity alone can include inferred poses.
    /// </summary>
    public static class BodyPoseWireFormat
    {
        public const int UpperBodyJointCount = 70;
        public const int FullBodyJointCount = 84;
        public const int MaxUdpPayloadBytes = 65507;

        // Explicit SDK ordering avoids BoneId's duplicate hand/body enum aliases.
        // FullBody 0..69 shares the UpperBody ordering; only 70..83 are legs.
        private static readonly string[] JointNames =
        {
            "Root", "Hips", "SpineLower", "SpineMiddle", "SpineUpper", "Chest", "Neck", "Head",
            "LeftShoulder", "LeftScapula", "LeftArmUpper", "LeftArmLower", "LeftHandWristTwist",
            "RightShoulder", "RightScapula", "RightArmUpper", "RightArmLower", "RightHandWristTwist",
            "LeftHandPalm", "LeftHandWrist", "LeftHandThumbMetacarpal", "LeftHandThumbProximal",
            "LeftHandThumbDistal", "LeftHandThumbTip", "LeftHandIndexMetacarpal", "LeftHandIndexProximal",
            "LeftHandIndexIntermediate", "LeftHandIndexDistal", "LeftHandIndexTip", "LeftHandMiddleMetacarpal",
            "LeftHandMiddleProximal", "LeftHandMiddleIntermediate", "LeftHandMiddleDistal", "LeftHandMiddleTip",
            "LeftHandRingMetacarpal", "LeftHandRingProximal", "LeftHandRingIntermediate", "LeftHandRingDistal",
            "LeftHandRingTip", "LeftHandLittleMetacarpal", "LeftHandLittleProximal", "LeftHandLittleIntermediate",
            "LeftHandLittleDistal", "LeftHandLittleTip", "RightHandPalm", "RightHandWrist",
            "RightHandThumbMetacarpal", "RightHandThumbProximal", "RightHandThumbDistal", "RightHandThumbTip",
            "RightHandIndexMetacarpal", "RightHandIndexProximal", "RightHandIndexIntermediate",
            "RightHandIndexDistal", "RightHandIndexTip", "RightHandMiddleMetacarpal", "RightHandMiddleProximal",
            "RightHandMiddleIntermediate", "RightHandMiddleDistal", "RightHandMiddleTip", "RightHandRingMetacarpal",
            "RightHandRingProximal", "RightHandRingIntermediate", "RightHandRingDistal", "RightHandRingTip",
            "RightHandLittleMetacarpal", "RightHandLittleProximal", "RightHandLittleIntermediate",
            "RightHandLittleDistal", "RightHandLittleTip", "LeftUpperLeg", "LeftLowerLeg", "LeftFootAnkleTwist",
            "LeftFootAnkle", "LeftFootSubtalar", "LeftFootTransverse", "LeftFootBall", "RightUpperLeg",
            "RightLowerLeg", "RightFootAnkleTwist", "RightFootAnkle", "RightFootSubtalar",
            "RightFootTransverse", "RightFootBall"
        };

        public readonly struct Joint
        {
            public readonly float Px, Py, Pz, Qx, Qy, Qz, Qw;
            public readonly bool PositionValid, OrientationValid;
            public readonly bool PositionTracked, OrientationTracked;

            // Compatibility for producers written before tracked flags were exposed.
            public Joint(float px, float py, float pz, float qx, float qy, float qz, float qw,
                bool positionValid, bool orientationValid)
                : this(px, py, pz, qx, qy, qz, qw, positionValid, orientationValid,
                    positionValid, orientationValid)
            {
            }

            public Joint(float px, float py, float pz, float qx, float qy, float qz, float qw,
                bool positionValid, bool orientationValid, bool positionTracked, bool orientationTracked)
            {
                Px = px; Py = py; Pz = pz;
                Qx = qx; Qy = qy; Qz = qz; Qw = qw;
                PositionValid = positionValid;
                OrientationValid = orientationValid;
                PositionTracked = positionTracked;
                OrientationTracked = orientationTracked;
            }
        }

        public static string CanonicalJointName(int index)
        {
            if (index < 0 || index >= JointNames.Length) throw new ArgumentOutOfRangeException(nameof(index));
            return JointNames[index];
        }

        public static string Serialize(long frameId, long timestampNs, bool fullBody, float confidence,
            bool trackingValid, Joint[] joints, float elbowAlpha, float wrmConfidence,
            bool wrmEnabled, bool wrmCalibrated, StringBuilder buffer = null)
        {
            StringBuilder sb = buffer ?? new StringBuilder(18000);
            sb.Clear();
            sb.Append("{\"type\":\"BODY\",\"version\":1,\"frame_id\":")
                .Append(frameId.ToString(CultureInfo.InvariantCulture))
                .Append(",\"timestamp_ns\":").Append(timestampNs.ToString(CultureInfo.InvariantCulture))
                .Append(",\"joint_set\":\"").Append(fullBody ? "full_body" : "upper_body")
                .Append("\",\"coordinate_space\":\"unity_world\",\"confidence\":");
            AppendNumber(sb, UnitInterval(confidence));
            sb.Append(",\"tracking_valid\":").Append(trackingValid ? "true" : "false")
                .Append(",\"joints\":[");

            int count = fullBody ? FullBodyJointCount : UpperBodyJointCount;
            for (int i = 0; i < count; i++)
            {
                Joint joint = joints != null && i < joints.Length ? joints[i] : default;
                bool positionValid = trackingValid && joint.PositionValid &&
                    IsFinite(joint.Px) && IsFinite(joint.Py) && IsFinite(joint.Pz);
                bool orientationValid = trackingValid && joint.OrientationValid &&
                    IsFinite(joint.Qx) && IsFinite(joint.Qy) && IsFinite(joint.Qz) && IsFinite(joint.Qw) &&
                    (joint.Qx != 0f || joint.Qy != 0f || joint.Qz != 0f || joint.Qw != 0f);
                if (i > 0) sb.Append(',');
                sb.Append("{\"id\":").Append(i.ToString(CultureInfo.InvariantCulture))
                    .Append(",\"name\":\"").Append(JointNames[i]).Append("\",\"position\":");
                if (positionValid)
                {
                    sb.Append('['); AppendNumber(sb, joint.Px); sb.Append(',');
                    AppendNumber(sb, joint.Py); sb.Append(','); AppendNumber(sb, joint.Pz); sb.Append(']');
                }
                else sb.Append("null");
                sb.Append(",\"rotation\":");
                if (orientationValid)
                {
                    sb.Append('['); AppendNumber(sb, joint.Qx); sb.Append(',');
                    AppendNumber(sb, joint.Qy); sb.Append(','); AppendNumber(sb, joint.Qz); sb.Append(',');
                    AppendNumber(sb, joint.Qw); sb.Append(']');
                }
                else sb.Append("null");
                sb.Append(",\"position_valid\":").Append(positionValid ? "true" : "false")
                    .Append(",\"orientation_valid\":").Append(orientationValid ? "true" : "false")
                    .Append(",\"position_tracked\":").Append(positionValid && joint.PositionTracked ? "true" : "false")
                    .Append(",\"orientation_tracked\":").Append(orientationValid && joint.OrientationTracked ? "true" : "false")
                    .Append('}');
            }
            sb.Append("],\"wrm\":{\"elbow_alpha\":"); AppendNumber(sb, UnitInterval(elbowAlpha));
            sb.Append(",\"confidence\":"); AppendNumber(sb, UnitInterval(wrmConfidence));
            sb.Append(",\"enabled\":").Append(wrmEnabled ? "true" : "false")
                .Append(",\"calibrated\":").Append(wrmCalibrated ? "true" : "false").Append("}}");
            return sb.ToString();
        }

        private static bool IsFinite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);
        private static float UnitInterval(float value) => !IsFinite(value) ? 0f : Math.Max(0f, Math.Min(1f, value));
        private static void AppendNumber(StringBuilder sb, float value) =>
            sb.Append(value.ToString("G9", CultureInfo.InvariantCulture));
    }
}
