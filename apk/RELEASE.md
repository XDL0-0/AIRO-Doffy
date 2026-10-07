# Airo-Doffy v0.9.7

Quest app release for Android ARM64, published on 2026-10-06.

## Download and install

Download [AIRO_Doffy_v0.9.7_arm64_code18.apk](AIRO_Doffy_v0.9.7_arm64_code18.apk). With the Quest connected through ADB and USB debugging authorized, run from the repository root:

```bash
adb install -r apk/AIRO_Doffy_v0.9.7_arm64_code18.apk
```

## Release information

| Item | Version |
| --- | --- |
| App | 0.9.7 / code 18 |
| Android package | `com.AIROLab.AIRODOFFY` |
| Architecture | ARM64 |
| Unity | 6000.5.6f1 |
| Meta XR All-in-One | 205.0.0 |

BODY telemetry is off on each launch. Set and Apply the PC address in **Teleop Config → Connection & input**, then enable **System Setting → Body data: ON**. The [BODY guide](../docs/body_visualization/README.md) explains the PC viewer.

## Unity source and build

Open the **AIRO-Doffy** subfolder of [AIRO-DOFFY-APP at `811d0b4`](https://github.com/XDL0-0/AIRO-DOFFY-APP/tree/811d0b4f9e9d6368a7fb2402943b41eb324c7167/AIRO-Doffy) with Unity 6000.5.6f1. Use **Tools → DOFFY → Build Meta ARM64-only update APK** for the code 18 package; the other build menus use code 16.

This source snapshot was published after the APK. The original build revision has not been confirmed.

## Release maintenance

For each new APK, update [manifest.json](manifest.json) with the version, build settings and full Unity source revision, and update these release notes. The repository's automated release checks must pass before publication.
