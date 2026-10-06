# BEAVER architecture reference

This file is the project-level source of truth for future paper writing about
BEAVER. The author-confirmed deployed hardware has **nine sensors arranged in
two rings**. Earlier 18-sensor/three-ring wording is historical and superseded.

## Name

**BEAVER**: **B**racelet **E**nabling **A**ccurate **V**isual **E**xternal
**R**anging.

## Paper-facing system architecture

- BEAVER is a modular, robot-mounted exteroceptive ranging system built from
  VL53L7CX time-of-flight (ToF) sensor modules.
- Each sensor produces a spatial distance map with either 8 x 8 zones or
  4 x 4 zones. Its nominal maximum ranging distance is up to 3.5 m; the paper
  should cite the VL53L7CX datasheet for the measurement conditions and detailed
  performance limits.
- The final system is mounted on a RealMan RM75 manipulator as two separate
  bracelet-like rings containing nine sensor boards total. The exact physical
  placement of the five and four sensors is not newly author-confirmed.
- One BEAVER sensor board measures 17 mm x 24 mm.
- Every board contains a VL53L7CX and an LTC4316 I2C address translator. The
  translator gives each module a distinct translated address even though all
  VL53L7CX devices use the same fixed native I2C address.
- The two rings are supplied from 5 V and communicate with the acquisition
  controller over the configured I2C buses. Current host/software configuration
  groups the nine slots as 5 + 4 across two bus IDs; this is code evidence and
  does not establish the exact physical placement.
- The ESP32-S3 collects the sensor maps into system frames over USB. Aggregate
  frame rates for the installed nine-sensor system remain pending measurement.
- The ESP32-S3 is connected to the host PC over USB. It streams the ranging
  frames to the host, where they are synchronized with robot observations and
  consumed by the robot policy.

## Derived system dimensions

| Mode | Zones per sensor | Zones across installed 9-sensor system (derived) | Frame rate |
|---|---:|---:|---:|
| 8 x 8 | 64 | 576 | pending measurement |
| 4 x 4 | 16 | 144 | pending measurement |

These are spatial ranging zones, not RGB image pixels. Prefer "zones",
"ranging cells", or "distance-map elements" in technical prose.

## Data flow

```text
9 x (VL53L7CX + LTC4316)
        |
        |  two 5 V rings; current software grouping is 5 + 4 over two bus IDs
        v
central ESP32-S3 DevKitC-1
        |
        |  assembled BEAVER frames over USB
        v
host PC acquisition and time synchronization
        |
        v
dataset recorder / robot policy
```

## Host-side protocol confirmed from `beaver.py`

The current host reader supports a binary stream with:

- frame magic `0x5A5A`;
- a 16-bit sequence counter, sensor count, and grid-width flag;
- grid widths of 4 and 8 (a zero legacy flag is interpreted as 8 x 8);
- up to 18 sensor records per wire frame (protocol capacity, not installed count);
- per-sensor bus ID, sensor index, valid-zone count, average range,
  temperature, stream counter, distance map, and target-status map;
- raw 16-bit millimetre distances in the current high-precision format, while
  retaining compatibility with a legacy 8-bit format quantized in 10 mm units;
- host selection of 4 x 4 or 8 x 8 mode by periodically sending the requested
  grid width to the ESP32 over the serial connection;
- USB serial auto-detection for the ESP32-S3 and a configured baud rate of
  115200 bit/s;
- non-blocking acquisition, reconnection, frame-loss accounting, and a short
  timestamped history used to select the BEAVER frame nearest to a robot or
  camera observation.

The paper does not need to expose all wire-format details unless discussing the
acquisition stack or temporal synchronization.

## Firmware source audit (2026-09-08)

- A saved source snapshot is archived at
  [`firmware_archive/2026-09-03-source/`](firmware_archive/2026-09-03-source/),
  with provenance in [`SOURCE_PROVENANCE.md`](firmware_archive/2026-09-03-source/SOURCE_PROVENANCE.md).
  It was audited from the two identical saved copies under `Downloads`.
- The snapshot implements two ESP32-S3 hardware I2C buses (GPIO 8/9 and
  12/13), dynamically scans up to six VL53L7CX devices per bus, and has a
  twelve-sensor controller capacity. It contains no third bus, no exact
  translated-address table, and no LTC4316-specific setup. This is historical
  source evidence; it does not define the installed nine-sensor count.
- `DESIGN.md` proposes a second DevKit for a third ring. That proposal is
  historical and superseded by the author-confirmed two-ring hardware.
- The firmware configures each sensor at 30 Hz in 4x4 mode and 15 Hz in 8x8
  mode, then gates a complete frame on fresh records from all active sensors.
  These are per-sensor settings and a frame-assembly policy; the architecture's
  aggregate frame-rate values remain pending measurement for the installed
  nine-sensor system.
- This source archive is not deployment evidence: build, flash, hardware, and
  serial capture status remain unknown. Detailed byte-level findings are in
  the local manuscript notes `paper/icra2027_beaver/FIRMWARE_EVIDENCE.md`,
  which are excluded from this source repository.

## Important repository status (2026-09-08)

- `hardware/beaver/firmware_archive/2026-09-04/` exists but is empty in the
  current workspace. The dated source audit and archive above provide firmware
  implementation evidence; deployment remains unverified.
- The active host configuration in `config.py` still declares nine sensors
  split 5+4 across two bus IDs. Many recorded datasets and policy models also
  use tensors shaped `(9, 4, 4)`. This is consistent with the current installed
  count, while the exact physical placement remains unconfirmed.
- Before a methods section is finalized, verify deployment against the source
  snapshot: exact I2C translated addresses, sensor initialization order,
  timing budget, frequency definition, and USB packet serialization.

## Paper-ready concise description

BEAVER (Bracelet Enabling Accurate Visual External Ranging) is a modular
exteroceptive sensing system for the RealMan RM75 manipulator. It comprises
nine VL53L7CX time-of-flight modules arranged in two bracelet-like rings; the
exact five/four physical placement is not yet author-confirmed. Each 17 mm x
24 mm sensor board integrates an LTC4316 I2C address translator. The rings
connect to the acquisition controller, which aggregates distance maps and
streams them over USB to the host PC for synchronization with robot
observations and use by the control policy. Each module provides either an
8 x 8 or 4 x 4 distance map; aggregate frame rates remain pending measurement.
