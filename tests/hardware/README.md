# Hardware tests

Hardware checks are skipped unless their device-specific environment variable is
set. They are never part of the default unit or integration suite.

Each robot check starts the v2 backend, reads one state, and closes it. It does
not submit a motion action. Closing a robot backend may send the adapter's normal
stop command.

Examples:

```powershell
$env:AIRO_DOFFY_TEST_UR_IP = "192.0.2.10"
python -m unittest tests.hardware.test_devices.HardwareDeviceTest.test_ur_state_read

$env:AIRO_DOFFY_TEST_REALMAN_IP = "192.0.2.20"
python -m unittest tests.hardware.test_devices.HardwareDeviceTest.test_realman_state_read

$env:AIRO_DOFFY_TEST_REALSENSE = "1"
$env:AIRO_DOFFY_TEST_REALSENSE_SERIAL = "optional-serial"
python -m unittest tests.hardware.test_devices.HardwareDeviceTest.test_realsense_frame

$env:AIRO_DOFFY_TEST_BLE4 = "1"
python -m unittest tests.hardware.test_devices.HardwareDeviceTest.test_ble4_sample
```

Install the corresponding optional dependency group before enabling a check.
Run robot tests only when the workcell is supervised and safe.
