# Deprecated code policy

This directory preserves unsupported implementations that are still useful for
reference, migration, or temporary command compatibility. It is not part of the
installable `airo_doffy` package.

## Rules

- Supported modules under `src/airo_doffy/` must never import this directory.
- Deprecated modules must not be selected by the main runtime or by default
  configuration.
- Do not add new features here. Limit changes to safety, security, data-loss, or
  short-lived compatibility fixes.
- Keep hardware dependencies optional and document any physical setup required
  before running a deprecated module.
- Preserve Git history with `git mv`; do not copy-and-delete legacy modules.
- A compatibility wrapper may remain at an old command path, but it must emit a
  deprecation warning and point to the supported replacement.
- Removal requires an explicit review after at least one published migration
  cycle. Do not remove code merely because it has been moved here.

## Inventory

| Legacy implementation | Status | Reason | Supported replacement | Deprecated on | Migration |
|---|---|---|---|---|---|
| `tactile/magtouch_ilias_41taxel.py` | Unsupported | The 41-taxel serial device is no longer an actively supported backend and couples acquisition to legacy DDS publishing. | `airo_doffy.devices.tactile` with the 4-taxel BLE backend (currently available through the legacy `tactile_4point.py` adapter until its package migration). | 2026-07-30 | Select `TACTILE_READER="ble4"`. The old `python tactile.py` diagnostic command remains temporarily available; it requires the legacy serial and `sensor_comm_dds` dependencies. |

## Ownership and safety

Deprecated hardware code has no active compatibility guarantee and is excluded
from normal CI. Before using it, inspect the device address, serial port, baud
rate, calibration procedure, and stop behavior. Never assume that a deprecated
module has the same safety or lifecycle guarantees as the supported runtime.
