# Multi-bus design for 2 or 3 rings

## Goal
Same firmware works for:
- DevKit A: 2 rings (12 sensors) on I2C bus 0 + bus 1
- DevKit B: 1 ring (6 sensors) on I2C bus 0 only
Both use the same binary, auto-detecting which buses have sensors.

## Hardware
ESP32-S3 has 2 hardware I2C peripherals (port 0 and 1).

| Ring | Bus | Port | SDA | SCL | Notes |
|------|-----|------|-----|-----|-------|
| Ring 1 | Bus 0 | I2C_NUM_0 | GPIO 8 | GPIO 9 | Primary bus, always present |
| Ring 2 | Bus 1 | I2C_NUM_1 | GPIO 12 | GPIO 13 | Optional second ring |

Ring 3 goes on a second DevKit (same firmware, only bus 0 used).

## Design

### Sensor struct with bus info
```c
typedef struct {
    VL53L7CX_Configuration config;  // the ST library device struct
    i2c_port_t bus;                 // which I2C port this sensor lives on
    uint8_t addr_7;                 // 7-bit I2C address
    bool active;
} sensor_t;

#define MAX_BUSES    2
#define MAX_PER_BUS  6
#define MAX_SENSORS  (MAX_BUSES * MAX_PER_BUS)  // 12
```

### Bus scan array
```c
typedef struct {
    i2c_port_t port;
    int gpio_sda, gpio_scl;
    uint8_t sensor_addrs[MAX_PER_BUS];
    int sensor_addr_count;
    int init_count;    // passed firmware download
    int active_count;  // ranging started
} bus_info_t;
```

### Init phases (same as now, but per-bus loop)
```
Phase 0: init both I2C buses
Phase 1: scan bus 0, then bus 1 (skip bus 1 if no ACKs found)
Phase 2: firmware download: bus 0 sensors first, then bus 1 sensors
  (bus 0 finishes all before bus 1 starts, so I2C state is clean)
Phase 3: start ranging on all init'd sensors
```

### Auto-detect: which buses are present?
After Phase 0 (both buses initialized), Phase 1 scans each.
If bus 1 gets zero ACKs across the whole address range, mark it inactive.
No sensors on that bus = bus not used (single-ring DevKit).

### I2C transaction context switching
Between buses, need to switch I2C context. The Twistx77 library stores
`i2c_port_t` in `VL53L7CX_Platform`, so the library knows which bus to use.
No extra work needed beyond setting `dev->platform.port` correctly.

### Read loop
Round-robin across all sensors regardless of bus:
```
for each active sensor:
    check_data_ready -> get_ranging_data -> print
```

### Logging format
```
I B0:  I2C up  SDA=8  SCL=9  @400kHz
I B1:  I2C up  SDA=12 SCL=13 @400kHz
I Phase 1: scanning bus B0 ...
I B0:  0x2E VL53L7CX ...
I B0:  Found 6 VL53L7CX
I Phase 1: scanning bus B1 ...
I B1:  (no devices found - bus inactive)
I Found 6/6 sensors on 1 bus(es)
I Phase 2: firmware download ...
...
I Active: 6 sensor(s) on 1 bus(es)
```

### Edge cases
- Bus 1 has devices but they're not VL53L7CX: still counted as "active" bus
- Bus 1 cable unplugged: all probes timeout, bus marked inactive (graceful)
- Only 3 sensors on bus 1: works fine, init_count reflects actual count
- If bus 0 fails entirely: retry whole sequence every 10s

### Watchdog consideration
Firmware download for 12 sensors at 100kHz ~ 90s total. Current watchdog is
60s. Either extend it to 120s or feed the watchdog between sensors.
At 400kHz it should be ~25s total, well within 60s.

## Implementation notes
- Keep `sensor_addrs[]` per-bus, not global
- `sensor_count` is total across all buses
- `sensors[]` array is flat (all buses), indexed 0..N-1
- Bus init uses the same `i2c_bus_init()` pattern with different GPIO pairs
- If bus 1 init fails (e.g. I2C_NUM_1 not available), treat as single-bus