/**
 * VL53L7CX multi-ring readout on ESP32-S3 (ESP-IDF v5.0)
 *
 * Multi-bus I2C (SDA=8/SCL=9 + SDA=12/SCL=13, 400 kHz)
 * Console: USB-CDC (usb_serial_jtag)
 * Sensor lib: Twistx77/V53L7CX-Library (managed component)
 *
 * Binary streaming protocol:
 *   Header (6B):  0x5A 0x5A | seq(16 BE) | n_sensors(8) | grid_width(8)
 *   Per sensor: bus(8) | idx(8) | n_valid(8) | avg_mm(16 BE) |
 *               temp(i8) | sc(32 BE) | dist[zones](8) | status[zones](8)
 *   zones = grid_width * grid_width. A legacy grid_width of 0 means 8x8.
 *
 * ASCII commands (stdin, newline-terminated):
 *   help, status, stream 0/1
 */

#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "freertos/event_groups.h"
#include "esp_log.h"
#include "esp_rom_sys.h"
#include "esp_vfs_dev.h"
#include "driver/gpio.h"
#include "driver/i2c.h"
#include "driver/usb_serial_jtag.h"
#include "vl53l7cx_api.h"
#include "esp_system.h"

#define TAG         "VL53L7CX"
#define MAX_BUSES   2
#define I2C_CLK_HZ  400000
#define MAX_PER_BUS 6
#define MAX_SENSORS (MAX_BUSES * MAX_PER_BUS)
#define MAX_GRID_WIDTH 8
#define MAX_ZONE_COUNT (MAX_GRID_WIDTH * MAX_GRID_WIDTH)

/* Protocol constants */
#define FRAME_MAGIC 0x5A5A
#define MAX_SENSOR_SZ   (10 + 3 * MAX_ZONE_COUNT)
#define HEADER_SZ   6
#define MAX_MAX_FRAME (HEADER_SZ + MAX_SENSOR_SZ * MAX_SENSORS)

/* ------------------------------------------------------------------ */
/*  Global state                                                       */
/* ------------------------------------------------------------------ */

typedef struct {
    i2c_port_t port;
    int gpio_sda, gpio_scl;
    uint8_t  sensor_addrs[MAX_PER_BUS];
    int ack_count;
    int sensor_addr_count;
    int init_count;
    int active_count;
} bus_info_t;

static bus_info_t buses[MAX_BUSES];
static VL53L7CX_Configuration sensors[MAX_SENSORS];
static bool sensor_initialized[MAX_SENSORS];
static bool sensor_active[MAX_SENSORS];
static uint8_t sensor_bus[MAX_SENSORS];
static uint8_t sensor_index[MAX_SENSORS];
static volatile uint32_t sensor_polls[MAX_SENSORS];
static volatile uint32_t sensor_ready_events[MAX_SENSORS];
static volatile uint32_t sensor_check_errors[MAX_SENSORS];
static volatile uint32_t sensor_read_errors[MAX_SENSORS];
static volatile uint8_t sensor_last_status[MAX_SENSORS];
static volatile int sensor_count = 0;

/* USB streaming state */
static SemaphoreHandle_t usb_send_mux;
static SemaphoreHandle_t sensor_records_mux;
static EventGroupHandle_t fresh_sensor_bits;
static volatile uint8_t stream_enabled = 1;
static volatile uint8_t stream_running = 0;
static uint16_t frame_seq = 0;
static volatile uint32_t stream_loops = 0;
static uint8_t sensor_records[MAX_SENSORS][MAX_SENSOR_SZ];
static uint8_t grid_width;
static uint8_t cfg_res = 0;
static uint8_t cfg_freq_hz = 0;
static uint16_t sensor_sz = 0;
static uint16_t max_frame = 0;

/* ------------------------------------------------------------------ */
/*  USB helpers                                                        */
/* ------------------------------------------------------------------ */

static int usb_send(const void *buf, int len)
{
    if (!usb_send_mux) return -1;
    if (xSemaphoreTake(usb_send_mux, pdMS_TO_TICKS(100)) != pdTRUE)
        return -1;

    const uint8_t *src = buf;
    int sent = 0;
    while (sent < len) {
        int n = usb_serial_jtag_write_bytes(src + sent, len - sent,
                                            pdMS_TO_TICKS(100));
        if (n <= 0) break;
        sent += n;
    }

    xSemaphoreGive(usb_send_mux);
    return sent == len ? sent : -1;
}

static int usb_read_last(void)
{
    uint8_t byte;
    int last = -1;

    while (usb_serial_jtag_read_bytes(&byte, 1, 0) > 0) {
        last = byte;
    }

    return last;
}

static int usb_puts(const char *s)
{
    return usb_send(s, strlen(s));
}

/* ------------------------------------------------------------------ */
/*  Command parser                                                     */
/* ------------------------------------------------------------------ */

static void cmd_status(void)
{
    static char buf[1024];
    int n = 0;
    n += snprintf(buf + n, sizeof(buf) - n, "Buses:\n");
    for (int b = 0; b < MAX_BUSES && (size_t)n < sizeof(buf) - 80; b++) {
        n += snprintf(buf + n, sizeof(buf) - n,
                      "  B%d: SDA=%d SCL=%d acks=%d found=%d active=%d\n",
                      b, buses[b].gpio_sda, buses[b].gpio_scl,
                      buses[b].ack_count, buses[b].sensor_addr_count,
                      buses[b].active_count);
    }
    n += snprintf(buf + n, sizeof(buf) - n,
                  "Total: %d grid=%dx%d freq=%dHz stream=%s loops=%lu seq=%u\n",
                  sensor_count, grid_width, grid_width, cfg_freq_hz,
                  stream_enabled ? "on" : "off",
                  (unsigned long)stream_loops, frame_seq);
    for (int s = 0; s < MAX_SENSORS && (size_t)n < sizeof(buf) - 80; s++) {
        if (!sensor_active[s]) continue;
        n += snprintf(buf + n, sizeof(buf) - n,
                      "  B%dS%d: polls=%lu ready=%lu check_err=%lu "
                      "read_err=%lu last=%u\n",
                      sensor_bus[s], sensor_index[s],
                      (unsigned long)sensor_polls[s],
                      (unsigned long)sensor_ready_events[s],
                      (unsigned long)sensor_check_errors[s],
                      (unsigned long)sensor_read_errors[s],
                      sensor_last_status[s]);
    }
    usb_send(buf, n);
}

static void parse_cmd(char *line)
{
    char *e = line + strlen(line);
    while (e > line && (*e == '\n' || *e == '\r' || *e == ' ')) *--e = 0;
    if (!*line) return;

    if (strcmp(line, "help") == 0) {
        usb_puts("Commands: help status stream 0/1\n");
    } else if (strcmp(line, "status") == 0) {
        cmd_status();
    } else if (strcmp(line, "stream 0") == 0 ||
               strcmp(line, "stream 1") == 0) {
        stream_enabled = line[7] == '1';
        char buf[32];
        int n = snprintf(buf, sizeof(buf), "stream %s\n",
                         stream_enabled ? "on" : "off");
        usb_send(buf, n);
    } else {
        char buf[128];
        int n = snprintf(buf, sizeof(buf), "unknown: %s\n", line);
        usb_send(buf, n);
    }
}

static void usb_cmd_task(void *arg)
{
    char line[64];
    int pos = 0;
    while (1) {
        if (usb_serial_jtag_read_bytes(line + pos, 1, pdMS_TO_TICKS(100)) != 1)
            continue;
        if (line[pos] == '\n' || line[pos] == '\r') {
            line[pos] = 0;
            parse_cmd(line);
            pos = 0;
        } else {
            pos++;
            if (pos >= (int)(sizeof(line) - 1)) {
                line[pos] = 0;
                parse_cmd(line);
                pos = 0;
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/*  I2C helpers (per-bus)                                              */
/* ------------------------------------------------------------------ */

static void bus_recover(bus_info_t *b)
{
    gpio_config_t cfg = {
        .pin_bit_mask = (1ULL << b->gpio_sda) | (1ULL << b->gpio_scl),
        .mode = GPIO_MODE_OUTPUT_OD,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&cfg));

    /* Release both lines. If a slave was interrupted while transmitting,
     * clock it to the end of its byte and then generate a STOP condition. */
    gpio_set_level(b->gpio_sda, 1);
    gpio_set_level(b->gpio_scl, 1);
    esp_rom_delay_us(10);

    if (!gpio_get_level(b->gpio_sda)) {
        ESP_LOGW(TAG, "B%d: SDA held low; recovering I2C bus", b->port);
        for (int pulse = 0; pulse < 18 && !gpio_get_level(b->gpio_sda); pulse++) {
            gpio_set_level(b->gpio_scl, 0);
            esp_rom_delay_us(10);
            gpio_set_level(b->gpio_scl, 1);
            esp_rom_delay_us(10);
        }
    }

    gpio_set_level(b->gpio_sda, 0);
    esp_rom_delay_us(10);
    gpio_set_level(b->gpio_scl, 1);
    esp_rom_delay_us(10);
    gpio_set_level(b->gpio_sda, 1);
    esp_rom_delay_us(10);
}

static esp_err_t bus_init(bus_info_t *b)
{
    i2c_config_t cfg = {
        .mode              = I2C_MODE_MASTER,
        .sda_io_num        = b->gpio_sda,
        .scl_io_num        = b->gpio_scl,
        .sda_pullup_en     = GPIO_PULLUP_ENABLE,
        .scl_pullup_en     = GPIO_PULLUP_ENABLE,
        .master.clk_speed  = I2C_CLK_HZ,
    };
    esp_err_t err = i2c_param_config(b->port, &cfg);
    if (err != ESP_OK) return err;
    err = i2c_driver_install(b->port, I2C_MODE_MASTER, 0, 0, 0);
    if (err != ESP_OK) return err;
    return i2c_set_timeout(b->port, 0x1F);
}

static esp_err_t i2c_probe(i2c_port_t p, uint8_t a7)
{
    i2c_cmd_handle_t c = i2c_cmd_link_create();
    i2c_master_start(c);
    i2c_master_write_byte(c, (a7 << 1) | I2C_MASTER_WRITE, true);
    i2c_master_stop(c);
    /* An address probe is one byte; keep disconnected buses responsive. */
    esp_err_t r = i2c_master_cmd_begin(p, c, pdMS_TO_TICKS(10));
    i2c_cmd_link_delete(c);
    return r;
}

static esp_err_t i2c_wreg(i2c_port_t p, uint8_t a7, uint16_t reg, uint8_t v)
{
    i2c_cmd_handle_t c = i2c_cmd_link_create();
    i2c_master_start(c);
    i2c_master_write_byte(c, (a7 << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(c, (reg >> 8) & 0xFF, true);
    i2c_master_write_byte(c, reg & 0xFF, true);
    i2c_master_write_byte(c, v, true);
    i2c_master_stop(c);
    esp_err_t r = i2c_master_cmd_begin(p, c, pdMS_TO_TICKS(100));
    i2c_cmd_link_delete(c);
    return r;
}

static esp_err_t i2c_rreg(i2c_port_t p, uint8_t a7, uint16_t reg, uint8_t *v)
{
    i2c_cmd_handle_t c = i2c_cmd_link_create();
    i2c_master_start(c);
    i2c_master_write_byte(c, (a7 << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(c, (reg >> 8) & 0xFF, true);
    i2c_master_write_byte(c, reg & 0xFF, true);
    i2c_master_start(c);
    i2c_master_write_byte(c, (a7 << 1) | I2C_MASTER_READ, true);
    i2c_master_read_byte(c, v, 1);
    i2c_master_stop(c);
    esp_err_t r = i2c_master_cmd_begin(p, c, pdMS_TO_TICKS(100));
    i2c_cmd_link_delete(c);
    return r;
}

static bool i2c_identify(i2c_port_t p, uint8_t a7)
{
    uint8_t di, ri;
    if (i2c_wreg(p, a7, 0x7FFF, 0x00) != ESP_OK) return false;
    if (i2c_rreg(p, a7, 0x0000, &di)  != ESP_OK) return false;
    if (i2c_rreg(p, a7, 0x0001, &ri)  != ESP_OK) return false;
    return (di == 0xF0 && ri == 0x02);
}

/* ------------------------------------------------------------------ */
/*  Phase 0-3: init                                                    */
/* ------------------------------------------------------------------ */

static void do_scan(void)
{
    for (int b = 0; b < MAX_BUSES; b++) {
        buses[b].ack_count = 0;
        buses[b].sensor_addr_count = 0;
        if (!gpio_get_level(buses[b].gpio_sda) ||
            !gpio_get_level(buses[b].gpio_scl)) {
            ESP_LOGW(TAG, "B%d: I2C bus held low (SDA=%d SCL=%d); skipping",
                     b, gpio_get_level(buses[b].gpio_sda),
                     gpio_get_level(buses[b].gpio_scl));
            continue;
        }
        for (int a = 1; a <= 0x77; a++) {
            if (i2c_probe(buses[b].port, a) != ESP_OK) continue;
            buses[b].ack_count++;
            ESP_LOGI(TAG, "B%d: I2C ACK at 0x%02X", b, a);
            if (!i2c_identify(buses[b].port, a)) continue;
            if (buses[b].sensor_addr_count >= MAX_PER_BUS) break;
            buses[b].sensor_addrs[buses[b].sensor_addr_count++] = (uint8_t)a;
        }
        if (buses[b].sensor_addr_count > 0)
            ESP_LOGI(TAG, "B%d: Found %d VL53L7CX", b, buses[b].sensor_addr_count);
    }
}

static void do_firmware(void)
{
    memset(sensor_initialized, 0, sizeof(sensor_initialized));
    memset(sensor_active, 0, sizeof(sensor_active));
    memset((void *)sensor_polls, 0, sizeof(sensor_polls));
    memset((void *)sensor_ready_events, 0, sizeof(sensor_ready_events));
    memset((void *)sensor_check_errors, 0, sizeof(sensor_check_errors));
    memset((void *)sensor_read_errors, 0, sizeof(sensor_read_errors));
    memset((void *)sensor_last_status, 0, sizeof(sensor_last_status));

    int base = 0;
    for (int b = 0; b < MAX_BUSES; b++) {
        buses[b].init_count = 0;
        if (buses[b].sensor_addr_count == 0) continue;
        ESP_LOGI(TAG, "B%d: firmware download ...", b);
        for (int i = 0; i < buses[b].sensor_addr_count; i++) {
            int s = base + i;
            uint8_t a7 = buses[b].sensor_addrs[i];
            sensor_bus[s] = b;
            sensor_index[s] = i;
            ESP_LOGI(TAG, "B%d S%d (0x%02X) ...", b, s, a7);

            VL53L7CX_Configuration *d = &sensors[s];
            memset(d, 0, sizeof(*d));
            d->platform.address = (uint16_t)(a7 << 1);
            d->platform.port    = buses[b].port;

            uint8_t st = vl53l7cx_init(d);
            if (st) {
                ESP_LOGE(TAG, "B%d S%d init failed: %d", b, s, st);
                continue;
            }
            sensor_initialized[s] = true;
            buses[b].init_count++;
            ESP_LOGI(TAG, "B%d S%d firmware OK", b, s);
        }
        base += buses[b].sensor_addr_count;
    }
}

static void do_start(void)
{
    sensor_count = 0;
    memset(sensor_active, 0, sizeof(sensor_active));

    int base = 0;
    for (int b = 0; b < MAX_BUSES; b++) {
        buses[b].active_count = 0;
        if (buses[b].init_count == 0) {
            base += buses[b].sensor_addr_count;
            continue;
        }
        ESP_LOGI(TAG, "B%d: start ranging ...", b);
        for (int i = 0; i < buses[b].sensor_addr_count; i++) {
            int s = base + i;
            if (!sensor_initialized[s]) continue;
            VL53L7CX_Configuration *d = &sensors[s];
            uint8_t st = vl53l7cx_set_resolution(d, cfg_res);
            if (!st) st = vl53l7cx_set_ranging_mode(
                d, VL53L7CX_RANGING_MODE_CONTINUOUS);
            if (!st) st = vl53l7cx_set_ranging_frequency_hz(d, cfg_freq_hz);
            if (!st) st = vl53l7cx_start_ranging(d);
            if (st) {
                ESP_LOGE(TAG, "B%d S%d start_ranging failed: %d", b, s, st);
                continue;
            }
            sensor_active[s] = true;
            sensor_count++;
            buses[b].active_count++;
            ESP_LOGI(TAG, "B%d S%d OK", b, s);
        }
        base += buses[b].sensor_addr_count;
    }
    int bc = (buses[0].active_count > 0) + (buses[1].active_count > 0);
    ESP_LOGI(TAG, "Active: %d sensor(s) on %d bus(es)", sensor_count, bc);
}

static void init_buses(void)
{
    buses[0].port = I2C_NUM_0; buses[0].gpio_sda = 8;  buses[0].gpio_scl = 9;
    buses[1].port = I2C_NUM_1; buses[1].gpio_sda = 12; buses[1].gpio_scl = 13;
    for (int b = 0; b < MAX_BUSES; b++) {
        bus_recover(&buses[b]);
        ESP_ERROR_CHECK(bus_init(&buses[b]));
        buses[b].ack_count = buses[b].sensor_addr_count = 0;
        buses[b].init_count = buses[b].active_count = 0;
        ESP_LOGI(TAG, "B%d: I2C up SDA=%d SCL=%d @%dHz",
                 b, buses[b].gpio_sda, buses[b].gpio_scl, I2C_CLK_HZ);
    }
}

static void init_sensors(void)
{
    sensor_count = 0;
    memset(sensor_initialized, 0, sizeof(sensor_initialized));
    memset(sensor_active, 0, sizeof(sensor_active));
    for (int b = 0; b < MAX_BUSES; b++) {
        buses[b].init_count = 0;
        buses[b].active_count = 0;
    }
    do_scan();
    int tf = buses[0].sensor_addr_count + buses[1].sensor_addr_count;
    if (tf > 0) do_firmware();
    int ti = buses[0].init_count + buses[1].init_count;
    if (ti > 0) do_start();
}

/* ------------------------------------------------------------------ */
/*  Parallel ranging and streaming tasks                               */
/* ------------------------------------------------------------------ */

static void pack_sensor_record(int s, const VL53L7CX_ResultsData *res,
                               uint8_t *sp)
{
    sp[0] = sensor_bus[s];
    sp[1] = sensor_index[s];
    int valid = 0, avg = 0;
    sp[2] = 0;
    sp[5] = (uint8_t)res->silicon_temp_degc;

    uint32_t sc = sensors[s].streamcount;
    sp[6] = (sc >> 24) & 0xFF;
    sp[7] = (sc >> 16) & 0xFF;
    sp[8] = (sc >> 8) & 0xFF;
    sp[9] = sc & 0xFF;

    /* Distances use 1 mm units; statuses are unmodified. */
    for (int z = 0; z < grid_width*grid_width; z++) {
        int zi = z * VL53L7CX_NB_TARGET_PER_ZONE;
        uint8_t ts = res->target_status[zi];
        int raw = res->distance_mm[zi];
        sp[10 + grid_width*grid_width*2 + z] = ts;
        sp[10 + z*2] = raw <= 0 ? 0 : (raw & 0xFF);
        sp[10 + z*2 + 1] = raw <= 0 ? 0 : ((raw >> 8) & 0xFF);
        if (ts == 5 || ts == 9) { valid++; avg += raw; }
    }
    sp[2] = valid;
    int avg_mm = (valid > 0) ? (avg / valid) : 0;
    sp[3] = (avg_mm >> 8) & 0xFF;
    sp[4] = avg_mm & 0xFF;
}

static EventBits_t active_sensor_mask(void)
{
    EventBits_t mask = 0;
    for (int s = 0; s < MAX_SENSORS; s++) {
        if (sensor_active[s]) mask |= (EventBits_t)1U << s;
    }
    return mask;
}

static void bus_reader_task(void *arg)
{
    int bus = (int)(intptr_t)arg;
    uint8_t record[sensor_sz];

    while (1) {
        if (!stream_running || !stream_enabled || !sensor_count) {
            vTaskDelay(pdMS_TO_TICKS(20));
            continue;
        }

        for (int s = 0; s < MAX_SENSORS; s++) {
            if (!sensor_active[s] || sensor_bus[s] != bus) continue;

            uint8_t ready = 0;
            sensor_polls[s]++;
            uint8_t status = vl53l7cx_check_data_ready(&sensors[s], &ready);
            sensor_last_status[s] = status;
            if (status != 0) {
                sensor_check_errors[s]++;
                continue;
            }
            if (!ready) continue;
            sensor_ready_events[s]++;

            VL53L7CX_ResultsData res;
            status = vl53l7cx_get_ranging_data(&sensors[s], &res);
            sensor_last_status[s] = status;
            if (status != 0) {
                sensor_read_errors[s]++;
                continue;
            }

            pack_sensor_record(s, &res, record);
            xSemaphoreTake(sensor_records_mux, portMAX_DELAY);
            memcpy(sensor_records[s], record, sensor_sz);
            xSemaphoreGive(sensor_records_mux);
            xEventGroupSetBits(fresh_sensor_bits, (EventBits_t)1U << s);
        }

        /* Poll fast enough for 60 Hz while yielding to the other bus task. */
        vTaskDelay(pdMS_TO_TICKS(1));
    }
}

static void streaming_task(void *arg)
{
    static uint8_t buf[MAX_MAX_FRAME];
    TickType_t started_at = xTaskGetTickCount();
    bool no_data_reported = false;

    while (1) {
        stream_loops++;
        if (!no_data_reported && frame_seq == 0 &&
            xTaskGetTickCount() - started_at >= pdMS_TO_TICKS(5000)) {
            usb_puts("NO DATA after 5s; ranging diagnostics follow\n");
            cmd_status();
            no_data_reported = true;
        }
        if (!stream_running || !stream_enabled || !sensor_count) {
            xEventGroupClearBits(fresh_sensor_bits, 0x00FFFFFFU);
            vTaskDelay(pdMS_TO_TICKS(20));
            continue;
        }

        EventBits_t required = active_sensor_mask();
        EventBits_t ready = xEventGroupWaitBits(
            fresh_sensor_bits, required, pdTRUE, pdTRUE, pdMS_TO_TICKS(100));
        if ((ready & required) != required) continue;

        /* Build a complete, deterministic nine-sensor frame. */
        uint16_t seq = frame_seq;
        buf[0] = (FRAME_MAGIC >> 8) & 0xFF;
        buf[1] = FRAME_MAGIC & 0xFF;
        buf[2] = (seq >> 8) & 0xFF;
        buf[3] = seq & 0xFF;
        buf[4] = sensor_count;
        buf[5] = grid_width;

        int idx = 0;
        xSemaphoreTake(sensor_records_mux, portMAX_DELAY);
        for (int s = 0; s < MAX_SENSORS; s++) {
            if (!sensor_active[s]) continue;
            memcpy(buf + HEADER_SZ + idx * sensor_sz,
                   sensor_records[s], sensor_sz);
            idx++;
        }
        xSemaphoreGive(sensor_records_mux);

        if (usb_send(buf, HEADER_SZ + idx * sensor_sz) >= 0)
            frame_seq++;
    }
}

/* ------------------------------------------------------------------ */
/*  app_main                                                           */
/* ------------------------------------------------------------------ */

void app_main(void)
{
    ESP_LOGI(TAG, "VL53L7CX multi-ring readout  lib=%s", VL53L7CX_API_REVISION);

    usb_serial_jtag_driver_config_t uc = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    /* The default 256-byte TX ring silently rejects a complete multi-sensor
     * frame. Keep room for two maximum-sized frames so writes are atomic. */
    uc.tx_buffer_size = MAX_MAX_FRAME * 2;
    ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&uc));
    esp_vfs_usb_serial_jtag_use_driver();

    while(true) {
        uint8_t data = usb_read_last();
        if(data != 4 && data != 8) continue;
        grid_width = data;
        break;
    }

    switch(grid_width) {
        case 4:
            cfg_freq_hz = 30;
            cfg_res = VL53L7CX_RESOLUTION_4X4;
            break;
        case 8:
            cfg_freq_hz = 15;
            cfg_res = VL53L7CX_RESOLUTION_8X8;
            break;
        default:
            break;
    }

    sensor_sz = (10 + 3 * grid_width * grid_width);
    max_frame = (HEADER_SZ + sensor_sz * MAX_SENSORS);

    usb_send_mux = xSemaphoreCreateMutex();
    sensor_records_mux = xSemaphoreCreateMutex();
    fresh_sensor_bits = xEventGroupCreate();
    configASSERT(usb_send_mux != NULL);
    configASSERT(sensor_records_mux != NULL);
    configASSERT(fresh_sensor_bits != NULL);

    init_buses();
    init_sensors();

    char boot[64];
    int n = snprintf(boot, sizeof(boot),
                     "READY: %d sensor(s), %dx%d @ %d Hz\n",
                     sensor_count, grid_width, grid_width, cfg_freq_hz);
    usb_send(boot, n);

    /* Keep the binary stream clean after the human-readable boot log. */
    esp_log_level_set("*", ESP_LOG_NONE);

    BaseType_t cmd_task_ok = xTaskCreate(
        usb_cmd_task, "usb_cmd", 2048, NULL, 3, NULL);
    BaseType_t bus0_task_ok = xTaskCreate(
        bus_reader_task, "range_b0", 4096, (void *)(intptr_t)0, 6, NULL);
    BaseType_t bus1_task_ok = xTaskCreate(
        bus_reader_task, "range_b1", 4096, (void *)(intptr_t)1, 6, NULL);
    BaseType_t stream_task_ok = xTaskCreate(
        streaming_task, "stream", 4096, NULL, 5, NULL);
    configASSERT(cmd_task_ok == pdPASS);
    configASSERT(bus0_task_ok == pdPASS);
    configASSERT(bus1_task_ok == pdPASS);
    configASSERT(stream_task_ok == pdPASS);
    stream_running = 1;

    TickType_t t0 = xTaskGetTickCount();

    while (1) {
        if (!sensor_count) {
            if (xTaskGetTickCount() - t0 >= pdMS_TO_TICKS(10000)) {
                init_sensors();
                t0 = xTaskGetTickCount();
            }
        }
        int usb_data = usb_read_last();
        if(usb_data != -1 && usb_data != grid_width) {
            esp_restart();
        }
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}
