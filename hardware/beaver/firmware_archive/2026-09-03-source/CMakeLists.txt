cmake_minimum_required(VERSION 3.16)
set(CMAKE_C_STANDARD 11)

# Only distance, target count and target status are used by Beaver.  Apply
# these definitions project-wide so both the VL53L7CX component and main agree
# on the result structure, without editing the generated managed component.
add_compile_definitions(
    VL53L7CX_DISABLE_AMBIENT_PER_SPAD
    VL53L7CX_DISABLE_NB_SPADS_ENABLED
    VL53L7CX_DISABLE_SIGNAL_PER_SPAD
    VL53L7CX_DISABLE_RANGE_SIGMA_MM
    VL53L7CX_DISABLE_REFLECTANCE_PERCENT
    VL53L7CX_DISABLE_MOTION_INDICATOR
)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(vl53l7cx_twistx)
