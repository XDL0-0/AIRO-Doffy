# Python module layout

The project is named **Airo-Doffy**; Python imports and `python -m` commands use the underscore module name `doffy_teleop`. The teleoperation implementation is owned by that package. The published repository root contains five Python CLI launchers, so the teleop, teach/recollect and BODY launch commands remain usable. Shared library code uses the canonical package paths below.

The upload covers UR/RealMan teleop, WRM, BrainCo, Beaver, VR/camera services, BODY visualization, dataset tools and related tests. This migration preserves their runtime defaults, robot constraints, recording schema and existing datasets; it does not require regenerating data. Local Seahorse, policy training/inference/evaluation, Jev and standalone experiments are outside the upload scope.

## Directory responsibilities

| Path | Responsibility |
|---|---|
| [`doffy_teleop/config.py`](../doffy_teleop/config.py) | Central robot, network, camera, sensor and recording configuration. |
| [`doffy_teleop/utils.py`](../doffy_teleop/utils.py) | Shared math, filters, gravity compensation and safety helpers. |
| [`doffy_teleop/protocol/`](../doffy_teleop/protocol/) | Controller/hand and BODY packets, recording controls, JPEG chunks and signaling messages. |
| [`doffy_teleop/media/`](../doffy_teleop/media/) | RealSense capture, frame stores, sockets, UDP/WebRTC transport and media orchestration. |
| [`doffy_teleop/sensors/`](../doffy_teleop/sensors/) | Beaver readers, serial/BLE MagTouch readers and wrench filtering. |
| [`doffy_teleop/control/`](../doffy_teleop/control/) | Input mapping, target limits, RealMan IK/QP, WRM geometry and CAN-FD command loops. |
| [`doffy_teleop/robots/`](../doffy_teleop/robots/) | Backend interfaces and factories, grippers, BrainCo hand control and Classic robot control. |
| [`doffy_teleop/recording/`](../doffy_teleop/recording/) | HDF5/LeRobot datasets, schema and recording services, trajectory replay and the recollection UI. |
| [`doffy_teleop/runtime/`](../doffy_teleop/runtime/) | CLI composition, Classic/RealMan lifecycles, teach/replay/collect, recollection and BODY viewer entry points; the local `seahorse.py` implementation is excluded. |
| [`doffy_teleop/visualization/`](../doffy_teleop/visualization/) | Teleop dashboard, BODY rendering and visualizer configuration. |
| [`doffy_teleop/body_visualization.py`](../doffy_teleop/body_visualization.py) | Compatibility exports for the modular BODY viewer. |
| [`dataset_tool/`](../dataset_tool/) | Dataset conversion, visualization, replay, recollection and annotation CLIs. |
| [`test_tool/`](../test_tool/) | Standalone teleop hardware, camera and VR/protocol tools. |
| [`tests/`](../tests/) | Automated tests for the retained teleop, BODY, media and dataset workflows. |

The dashboard settings live in [`doffy_teleop/visualization/config.py`](../doffy_teleop/visualization/config.py).

`doffy_teleop/recording_control.py` contains ordered recording requests. `doffy_teleop/body_visualization.py` provides package compatibility exports for the modular BODY viewer API.

## Old modules and canonical paths

The old root library files have moved; they are no longer import shims at the repository root. The rows for CLI files identify their packaged implementation while the root launcher remains available.

| Previous root file | Canonical implementation |
|---|---|
| `config.py` | `doffy_teleop/config.py` |
| `utils.py` | `doffy_teleop/utils.py` |
| `visualizer_config.py` | `doffy_teleop/visualization/config.py` |
| `visualizer.py` | `doffy_teleop/visualization/dashboard.py` |
| `beaver.py` | `doffy_teleop/sensors/beaver.py` |
| `beaver_reader.py` | `doffy_teleop/sensors/beaver_reader.py` |
| `tactile.py` | `doffy_teleop/sensors/tactile.py` |
| `tactile_4point.py` | `doffy_teleop/sensors/tactile_4point.py` |
| `force_filter.py` | `doffy_teleop/sensors/force_filter.py` |
| `brainco_hand.py` | `doffy_teleop/robots/brainco_hand.py` |
| `wrm_akm.py` | `doffy_teleop/control/wrm_akm.py` |
| `realsense_camera.py` | `doffy_teleop/media/realsense_camera.py` |
| `parse_vr.py` | `doffy_teleop/protocol/parse_vr.py` |
| `udp_comms.py` | `doffy_teleop/media/udp_comms.py` |
| `data_schema.py` | `doffy_teleop/recording/schema.py` |
| `dataset.py` | `doffy_teleop/recording/dataset.py` |
| `data_recording.py` | `doffy_teleop/recording/service.py` |
| `realman_teachcollect.py` | `doffy_teleop/runtime/teachcollect.py` |
| `realman_recollect.py` | `doffy_teleop/runtime/recollect.py` |
| `main.py` | `doffy_teleop/runtime/classic_entrypoint.py` |
| `realman_teleop.py` | `doffy_teleop/runtime/realman_cli.py` |
| `robot_backend.py` | `doffy_teleop/robots/backend_api.py` |
| `robot_teleop.py` | `doffy_teleop/robots/teleop.py` |
| `ur_teleop.py` | `doffy_teleop/robots/ur_teleop.py` |
| `udp.py` | `doffy_teleop/media/udp_manager.py` |
| `WebRTC_udp.py` | `doffy_teleop/media/webrtc_manager.py` |
| `teleop_body_visualizer.py` | `doffy_teleop/runtime/body_visualizer.py` |

The recording runtime also owns the implementations previously reached through dataset tools:

| Previous implementation | Canonical implementation | Retained tool entry |
|---|---|---|
| `dataset_tool/replay_realman_lerobot.py` | [`doffy_teleop/recording/replay.py`](../doffy_teleop/recording/replay.py) | `dataset_tool/replay_realman_lerobot.py` |
| `dataset_tool/recollect_ui/` | [`doffy_teleop/recording/recollect_ui/`](../doffy_teleop/recording/recollect_ui/) | `dataset_tool/recollect_ui/` |

The recollection UI's static assets live alongside its packaged implementation in `doffy_teleop/recording/recollect_ui/static/`.

## Updating library imports

Update imports in external scripts and integrations to use `doffy_teleop`. For example, replace `from config import Config`, `from visualizer_config import VisualizerConfig`, `from dataset import DatasetRecorder` and `from udp import UDPManager` with:

```python
from doffy_teleop.config import Config
from doffy_teleop.visualization.config import VisualizerConfig
from doffy_teleop.recording.dataset import DatasetRecorder
from doffy_teleop.media.udp_manager import UDPManager
```

Use dotted package names derived from the table for dynamic imports and test patch targets as well, for example `doffy_teleop.sensors.beaver.open_port`. Library consumers should import the packaged implementation even when a root CLI launcher with the old name still exists.

## Running the entry points

Run commands from the repository root using the environment for that runtime. The package must be available on Python's import path; running here makes it available without changing existing environment setup. Existing relative dataset paths retain their repository-root interpretation.

| Existing command | Equivalent packaged command |
|---|---|
| `python main.py` | `python -m doffy_teleop.runtime.classic_entrypoint` |
| `python realman_teleop.py` | `python -m doffy_teleop.runtime.realman_cli` |
| `python realman_teachcollect.py` | `python -m doffy_teleop.runtime.teachcollect` |
| `python realman_recollect.py` | `python -m doffy_teleop.runtime.recollect` |
| `python teleop_body_visualizer.py` | `python -m doffy_teleop.runtime.body_visualizer` |

Pass the same arguments to either form. For example:

```bash
python teleop_body_visualizer.py --demo
python -m doffy_teleop.runtime.body_visualizer --demo
```

The root files contain launcher logic; runtime implementation and configuration edits belong in the package. Standalone tools remain in their directories, for example `python -m test_tool.vr_data` and `python dataset_tool/replay_realman_lerobot.py ...`. The standalone tactile readers can be run as `python -m doffy_teleop.sensors.tactile` or `python -m doffy_teleop.sensors.tactile_4point` with their existing options.

## Data and validation records

Datasets and generated outputs stay local and retain their existing locations. The migration changes Python ownership and imports while preserving teleop control behavior, defaults and stored formats. The [main README](../README.md) documents runtime options, configuration values and the upload boundary.

The current upload subset passed **433 Python tests and 12 subtests**, with **3 skipped**, without adding an external simulation directory to the import path. The [teleoperation refactor records](teleop_refactor/README.md) also retain historical validation results and outstanding Unity/Quest acceptance items. Historical full-worktree results may include local research tests outside the upload scope; they are separate from this subset result. This layout guide does not claim new hardware or Unity validation.
