# BEAVER Firmware (ESP32-S3 + VL53L7CX)

Streams distance measurements from up to 12 VL53L7CX ToF sensors (6 per bus,
2 I2C buses @ 400 kHz) over the ESP32-S3 built-in USB Serial/JTAG CDC port.
Configured for a 4x4 zone grid at 30 Hz (`GRID_WIDTH` in `main/main.c` selects
4x4 or 8x8 @ 15 Hz). Targets ESP-IDF v5.0 (see `eim_config.toml`).

## Build and flash

```sh
eim run "idf.py build"          # or: idf.py set-target esp32s3 && idf.py build
eim run "idf.py -p /dev/ttyACM0 flash"
```

The native USB port is used for both flashing and data. Opening the CDC port
resets the board; with all sensors connected, init takes ~30 s. The firmware
recovers an I2C bus left mid-transaction by that reset.

## Host tools

```sh
python3 -m pip install -r client/requirements.txt
python3 client/beaver_client.py            # console reader, auto-detects 303a:1001
python3 client/beaver_visualizer.py        # tkinter heatmap dashboard
python3 -m unittest discover -s client -v  # protocol + visualizer unit tests
```

The client re-synchronizes on the frame magic if bytes are lost. Add
`--diagnostics` for firmware poll/error counters. While streaming, the
commands `help`, `status`, `stream 0`, and `stream 1` can be entered on stdin.

## Binary wire format

Big-endian. Header (6 B): `0x5A5A` magic, uint16 sequence, uint8 sensor count,
uint8 grid width (4 or 8; legacy 0 means 8). Then one record per sensor
(`N = grid^2` zones, so 42 B for 4x4, 138 B for 8x8):

| Offset | Size | Field |
| ---: | ---: | --- |
| 0 | 1 | I2C bus (0/1) |
| 1 | 1 | Sensor index on bus (0-5) |
| 2 | 1 | Valid zone count |
| 3 | 2 | Average valid distance (mm) |
| 5 | 1 | Silicon temperature (int8, deg C) |
| 6 | 4 | Stream count |
| 10 | N | Zone distances (10 mm units) |
| 10+N | N | Raw target status per zone |

Target statuses 5 and 9 count as valid.

## Layout

- `main/main.c` - single-file firmware: init (scan, firmware download, start
  ranging), two per-bus ranging tasks, one streaming task
- `managed_components/vl53l7cx` - Twistx77/V53L7CX-Library (managed component)
- `client/` - Python host reader, visualizer, tests
