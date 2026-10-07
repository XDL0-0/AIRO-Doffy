# Public release hygiene

This patch separates local setup from reproducible robot settings. Historical
validation records contain anonymized paths, device placeholders and example
network addresses; their test outcomes, pending acceptance items, SDK versions,
APK versions, sizes and experiment parameters are retained. Placeholder paths do
not assert that a local Unity project is available in this repository.

## Reviewed classification

| Information | Public treatment | Reason |
|---|---|---|
| Workstation home directories, interpreter paths, Unity projects and caches | Explicit CLI arguments/environment variables; `/path/to/...` or repository-relative examples in documents | Machine layout is not a build or experiment parameter |
| Quest and RealSense serial numbers | `QUEST_SERIAL_REDACTED` / `REALSENSE_SERIAL_REDACTED` | Models, versions and validation results remain sufficient to interpret the records |
| Campus/Wi-Fi, hotspot, Quest and workstation addresses; deployed UR address | Documentation-only `192.0.2.x` examples; local environment overrides for executable tools | These identify a deployment, not a portable robot default |
| RM75 controller `192.168.1.18`, API port `8080` | Retained | Existing robot connection default used by the teleop/replay tools |
| Wired host `192.168.1.100` | Retained as an example with `DOFFY_REALMAN_STATE_PUSH_IP` override | Preserves the separate robot-facing interface; not a manufacturer default or a claim about the operator's PC |
| Loopback and wildcard bind addresses | Retained | They define local test/bind semantics |
| Control rates, watchdogs, limits, joint/TCP calibration, protocol ports, data schema/FPS and package versions | Retained | Required to reproduce behavior and interpret results |

## Local network setup

`Config` reads these variables when an instance is created. Explicit constructor
arguments still take precedence. Replace the documentation-only addresses below
with your own interface/endpoint addresses before running teleop:

```bash
export DOFFY_PC_IP=192.0.2.10
export DOFFY_VR_IP=192.0.2.20
export DOFFY_UR_IP=192.0.2.30
export DOFFY_REALMAN_STATE_PUSH_IP=192.0.2.40
python realman_teleop.py
```

The PC and state-push addresses may refer to different interfaces. Passing
`REALMAN_STATE_PUSH_IP=None` to `Config` retains the existing fallback to `PC_IP`.
For UR, the existing `Config(ROBOT_TYPE="ur3e", ROBOT_IP=None)` fallback uses
`UR_IP`; the RealMan controller default is unchanged. The standalone UR replay,
force and freedrive tools also use `DOFFY_UR_IP`, while existing `--robot_ip` and
`--ip` arguments remain available and take priority. Protocols, motion mapping,
control loops and safety gates are unchanged; local connection setup is required.

If you keep exports in the already-ignored `.env`, load it explicitly in your
shell (`set -a; . ./.env; set +a`). The application does not auto-load dotenv files.
Keep raw device/network audit output local, e.g. under the ignored `evidence/`.

## Unity validation paths

There is no default personal Unity installation. Existing CLI flags take
precedence over the matching environment variables; missing required inputs
produce a usage error before compilation or migration. Relative CLI paths are
resolved from the invoking directory. Select the intended project variant
(`Codex`, `CodexBracelet`, or another checkout) explicitly.

| CLI argument | Environment variable |
|---|---|
| `--project` | `DOFFY_UNITY_PROJECT` |
| `--reference` | `DOFFY_UNITY_REFERENCE_PROJECT` |
| `--editor-data` | `DOFFY_UNITY_EDITOR_DATA` |
| `--package-cache` | `DOFFY_UNITY_PACKAGE_CACHE` |
| `--mono-bin` | `DOFFY_MONO_BIN` |
| `--mono` | `DOFFY_MONO` |
| BODY wire checker `--source`, `--ovr-plugin`, `--mcs` | `DOFFY_BODY_WIRE_SOURCE`, `DOFFY_OVR_PLUGIN`, `DOFFY_MCS` |

For example, set your local paths before running the validation commands in the
historical records:

```bash
export DOFFY_UNITY_PROJECT=/path/to/unity/CodexBracelet
export DOFFY_UNITY_REFERENCE_PROJECT=/path/to/unity/classic
export DOFFY_UNITY_EDITOR_DATA=/path/to/Unity/Hub/Editor/6000.5.6f1/Editor/Data
export DOFFY_UNITY_PACKAGE_CACHE="$DOFFY_UNITY_REFERENCE_PROJECT/Library/PackageCache"
export DOFFY_MONO_BIN="$DOFFY_UNITY_EDITOR_DATA/MonoBleedingEdge/bin"
export DOFFY_MONO="$DOFFY_MONO_BIN/mono"
export DOFFY_MCS="$DOFFY_MONO_BIN/mcs"
export DOFFY_BODY_WIRE_SOURCE="$DOFFY_UNITY_PROJECT/Assets/Teleop/Protocol/BodyPoseWireFormat.cs"
# Historical Meta cache revision; use the installed SDK source matching your project lock.
export DOFFY_OVR_PLUGIN="$DOFFY_UNITY_PROJECT/Library/PackageCache/com.meta.xr.sdk.core@c0efcbf2ba70/Scripts/OVRPlugin.cs"
python scripts/teleop_refactor/compile_unity_sources.py --include-editor
```

The WebRTC runner uses `DOFFY_UNITY_PROJECT` / `DOFFY_UNITY_EDITOR_DATA`; its
reference project is optional and defaults to the selected project. Its existing
`bin-linux64` Mono layout is retained. The optional pytest integrations skip if
the required project/tools are absent. The media loopback can also find Mono on
`PATH`; `--mono` or `DOFFY_MONO` selects a specific runtime.

Dataset helpers now accept `Visualize_hdf5_episodes.py --dataset-dir ...` and
`convert_openpi_way.py --output-dir ... --repo-id ...`. The converter retains the
`b2b_nofinecontrol` dataset name, uses a local `datasets/` output by default, and
passes the output root explicitly to LeRobot for both creation and reopening.
Its image processing, schema, FPS and local-only upload behavior are unchanged.

## Regression check

Stage new files, then run:

```bash
python scripts/check_release_hygiene.py
python -m unittest discover -s tests -p test_release_hygiene.py -v
```

The dependency-free check scans tracked UTF-8 text, including dotfiles, scripts,
documents, JSON and SVG. It rejects personal home paths, recognizable Quest IDs,
labeled device serials and RFC1918 IPv4 literals unless an exact file/address pair
has a documented exception. New exceptions need review and a reason; a private
subnet is never globally allowed. Loopback, wildcard binds, documentation-only
addresses remain valid.

GitHub Actions runs on pull requests and pushes to `main`. Requiring the
`source-hygiene` job in branch protection is a separate repository setting; this
patch does not change protection rules. This guard does not inspect signed APK
contents, git history or arbitrary credential/hostname formats, and makes no
claim to erase already-published history.
