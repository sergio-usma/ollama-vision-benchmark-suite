# Hardware Monitoring

## Overview

The `HardwareMonitor` class runs in a background daemon thread, sampling hardware telemetry at 1 Hz. It provides two modes: real (jtop/Jetson) and simulation.

---

## Real Mode (jtop)

Requires `jetson-stats` package and a Jetson device.

```bash
sudo pip install -U jetson-stats --break-system-packages
```

### Metrics Collected

| Metric | Source | Notes |
|--------|--------|-------|
| `temp_gpu_c` | jtop `stats["Temp GPU"]` | Primary thermal indicator |
| `temp_cpu_c` | jtop `stats["Temp CPU"]` | CPU cluster temperature |
| `temp_soc_c` | jtop `stats["Temp SOC0"]` | SoC temperature |
| `temp_tj_c` | jtop `stats["Temp tj"]` | Junction temperature (max thermal) |
| `power_gpu_soc_mw` | jtop `power["VDD_GPU_SOC"]` | GPU+SOC power rail (mW) |
| `power_cpu_cv_mw` | jtop `power["VDD_CPU_CV"]` | CPU+CV power rail (mW) |
| `power_sys5v0_mw` | jtop `power["VIN_SYS_5V0"]` | System 5V rail (mW) |
| `power_total_mw` | Sum of all rails | Total system power |
| `ram_used_mb` | jtop `stats["RAM"]` | Used RAM in MB |
| `ram_percent` | Computed | Used / Total × 100 |
| `gpu_load_pct` | jtop `stats["GPU"]` | GPU utilization (%) |
| `gpu_freq_mhz` | jtop `GPU.frq` | GPU clock frequency (MHz) |
| `fan_speed_rpm` | jtop `fan` | Fan speed (RPM) |
| `disk_read_mbs` | jtop `disk` | Disk read bandwidth (MB/s) |
| `disk_write_mbs` | jtop `disk` | Disk write bandwidth (MB/s) |
| `cpu_avg_load_pct` | jtop `cpu` | Average load across all CPU cores |

### Jetson AGX Orin Power Architecture

The Orin uses three main power rails measured by the INA3221 sensors:

```
VDD_GPU_SOC  → GPU + SoC fabric + memory controller
VDD_CPU_CV   → CPU clusters + CV accelerators (DLA, NVENC, NVDEC)
VIN_SYS_5V0  → System 5V bus (storage, PCIe, USB, etc.)
```

Total power ≈ VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0

For a 7B model inference:
- VDD_GPU_SOC typically dominates (GPU doing the matrix multiplications)
- VDD_CPU_CV is lower but non-zero (CPU handles tokenization, streaming)
- VIN_SYS_5V0 is the baseline system power

---

## Simulation Mode

Activated automatically when jtop is unavailable, or forced with `--simulate`.

Uses synthetic telemetry based on Jetson AGX Orin baseline values with Gaussian noise:

```python
# Baseline values (approximate for a 7B model inference)
BASE_TEMP_GPU  = 45.0    # °C
BASE_TEMP_CPU  = 40.0    # °C
BASE_RAM_MB    = 8_000   # MB
BASE_GPU_LOAD  = 65.0    # %
BASE_POWER_GPU = 12_000  # mW
BASE_POWER_CPU = 3_000   # mW
BASE_POWER_SYS = 4_000   # mW
```

Each sample adds Gaussian noise (σ ≈ 5% of the value) to simulate realistic variation. This allows the full benchmark pipeline to run on any Linux machine.

---

## Hardware Window Averaging

Each inference run records `hw_start_ts` and `hw_end_ts` timestamps. After the run completes, `get_window(start_ts, end_ts)` averages all hardware samples that fall within this window.

This is important because:
- The GPU temperature at run start may be different from the end
- Power spikes during initial token generation differ from steady-state
- Simple post-run snapshots would miss the full inference profile

```python
# In _single_inference():
hw_start_ts = time.time()
resp, elapsed, err = self.client.run_generate(model, category)
hw_end_ts = time.time()

# After run:
hw_avg = self.hw_mon.get_window(hw_start_ts, hw_end_ts)
```

---

## Adaptive Cooldown

Before benchmarking each model, the controller waits for the GPU to cool down to `COOLDOWN_TEMP` (default 50°C).

```python
def wait_for_cooldown(self, target_temp: float = COOLDOWN_TEMP) -> None:
    deadline = time.time() + COOLDOWN_TIMEOUT
    while time.time() < deadline:
        # Check last 30 history entries (≈30 seconds)
        recent = list(self.history)[-30:]
        if recent and all(s.temp_gpu_c <= target_temp for s in recent):
            return  # Stable and cool
        time.sleep(2.0)
```

The cooldown is **skipped entirely** if the GPU is already below `COOLDOWN_TEMP` when the check runs. This avoids unnecessary waits at the start of a benchmark on a cold machine.

---

## History Buffer

The `history` deque stores the last `HW_HISTORY_SIZE` (default 300) samples — 5 minutes of data. This is used by:

1. **Cooldown detection** — checks if the last 30 seconds are consistently below threshold
2. **Window averaging** — retrieves samples between two timestamps

The buffer is a `collections.deque(maxlen=HW_HISTORY_SIZE)`, so old samples are automatically discarded.

---

## Thread Safety

The `HardwareMonitor` uses a `threading.Lock` to protect concurrent access to the `history` deque:
- The background thread writes samples at 1 Hz
- The main thread reads samples for window averaging during inference

The lock is held only for the brief duration of a deque append or deque copy — not during the slow jtop read itself.
