# Architecture Reference

## Overview

OllamaVision Benchmark Suite follows the **Model-View-Controller (MVC)** pattern. All configuration lives in a single file (`config.py`) read by every module. This avoids circular imports and makes the system easy to tune without touching application logic.

```
config.py
    │
    ├── controllers/benchmark_controller.py   ← orchestrator (single entry point)
    │         │
    │    ┌────┴─────────────────────────────────┐
    │    ▼                                      ▼
    ├── models/                             views/
    │   ├── data_model.py                   ├── terminal_view.py
    │   ├── ollama_model.py                 └── dashboard_view.py
    │   └── hardware_model.py
    │
    └── utils/
        ├── checkpoint.py
        ├── logger.py
        └── system_utils.py
```

---

## Module Descriptions

### `config.py`

Single source of truth for all tunable parameters. Exports:
- API endpoints, timeouts, retry counts
- Benchmark parameters (`NUM_RUNS`, `WARMUP_RUN`, `INTER_RUN_SLEEP`)
- Scoring weights and thresholds
- Prompt sets (`PROMPTS`, `PROMPTS_MULTI`)
- Model category overrides (`MODEL_CATEGORY_OVERRIDE`)
- File paths (CSV, dashboard, logs, checkpoint)

**Design rule:** No module should define its own magic constants. All configuration is imported from `config.py`.

---

### `controllers/benchmark_controller.py`

`BenchmarkController` — the central orchestrator.

**Responsibilities:**
1. Pre-flight checks (Ollama alive, model list, hardware monitor)
2. Model discovery via `/api/tags` with fallback to `ollama list`
3. Filter application (`--filter`, `--models`)
4. Checkpoint load/create/resume logic
5. Benchmark loop:
   - Adaptive cooldown (GPU ≤ `COOLDOWN_TEMP`)
   - Memory flush (`drop_caches` + `unload_all`)
   - Warmup run
   - N measured runs with prompt rotation
   - Per-run hardware window averaging
6. Score recalculation after all models complete (final percentile computation)
7. CSV rewrite with final scores
8. JSON summary export
9. HTML dashboard generation

**Key design choices:**
- Hardware telemetry is captured as a window (`hw_start_ts` → `hw_end_ts`) during actual inference, not just a post-run snapshot.
- Scores are calculated twice: once per-model (preliminary) and once at the end (final, with correct category percentiles).
- Interrupt handler (`SIGINT`) saves checkpoint and regenerates dashboard before exiting.

---

### `models/data_model.py`

Core data structures.

#### `HardwareSnapshot`
A dataclass capturing a single 1 Hz hardware sample:
- Temperatures: GPU, CPU, SOC, TJ (°C)
- Power rails: GPU+SOC, CPU+CV, SYS5V0 (mW), total
- Memory: RAM used (MB), RAM percent
- GPU: load (%), frequency (MHz)
- Fan speed (RPM), disk I/O (MB/s)

#### `RunStat`
Result of a single inference run:
- TPS, total duration, load duration, API latency
- Prompt/eval token counts
- Aggregated hardware snapshot for this run
- Error string (if failed)

#### `BenchmarkResult`
Aggregated result for one model across all runs:
- All `RunStat` values averaged
- Multi-run statistics: median, stdev, CV, min, max
- Stability label (`STABLE`/`GOOD`/`VARIABLE`/`UNSTABLE`)
- Derived metrics: `tokens_per_W`, `category_percentile`, `efficiency_score`
- Recommendation and decision reasons
- TTFT (time-to-first-token) — in-memory only, exported to JSON

#### `CSVManager`
Atomic CSV append with header creation on first write. Converts `BenchmarkResult` to a flat dict of CSV-compatible values.

#### `StatsAggregator`
Computes multi-run statistics from a list of `RunStat` objects. Returns aggregated values and stability label.

---

### `models/ollama_model.py`

`OllamaClient` — REST client wrapping the Ollama API.

**Endpoints used:**
- `GET /api/tags` — list installed models
- `POST /api/show` — model metadata
- `GET /api/ps` — running models
- `POST /api/generate` — text/vision generation (streaming or blocking)
- `POST /api/embed` — embeddings (Ollama ≥0.4)
- `POST /api/embeddings` — embeddings fallback (Ollama <0.4)
- `POST /api/generate` with `keep_alive=0` — unload model

**Streaming mode:**
Default (`OLLAMA_STREAM=True`). Uses `stream=True` + `iter_lines()` to measure TTFT (time to first token). Falls back to blocking on `Timeout` or `ConnectionError`.

**Retry logic:**
`_post_with_retry()` retries `OLLAMA_MAX_RETRIES` times with exponential backoff (1s, 2s) on `ConnectionError`. Timeouts and HTTP 4xx are not retried (deterministic failures).

**Timeout:**
Per-instance via `OllamaClient(timeout=N)`. The `--timeout` CLI arg passes through without mutating the global config constant.

---

### `models/hardware_model.py`

`HardwareMonitor` — background 1 Hz sampling thread.

**Two modes:**
1. **Real mode** (`USE_SIMULATION=False`): Uses `jtop` (jetson-stats). Reads all Jetson power rails, GPU state, temperatures, fan, and disk I/O.
2. **Simulation mode** (`USE_SIMULATION=True`): Generates realistic synthetic telemetry based on Jetson AGX Orin baselines, with Gaussian noise.

**Key features:**
- `history`: deque of last N `HardwareSnapshot` objects (sliding window)
- `get_window(start_ts, end_ts)`: returns average snapshot for a time range
- `wait_for_cooldown(target_temp)`: blocks until GPU is below target temperature, checking the last 30 history entries (≈30s)
- Daemon thread: auto-terminates when the main process exits

---

### `views/terminal_view.py`

`TerminalView` — Rich Live 6-panel terminal dashboard, refreshed at 2 Hz.

**Layout:**
```
┌─────────────────────────────────┬──────────────────────┐
│  ⚡ PROGRESS                    │ 🔧 LIVE HARDWARE     │
│  Model progress bar, current    │ Temps, power, GPU,   │
│  model, run, ETA                │ RAM, fan             │
├─────────────────────────────────┤                      │
│  📈 LIVE STATISTICS             │                      │
│  Tested/success/failed counts,  │                      │
│  avg TPS, best model            ├──────────────────────┤
├─────────────────────────────────┤ 📋 LIVE LOG          │
│  RESULTS [N completed]          │ Last 20 log lines    │
│  Scrollable results table       │                      │
└─────────────────────────────────┴──────────────────────┘
```

**Update flow:** A thread calls `view.update(result, hw_snapshot, stats)` which refreshes all panels atomically inside the Rich `Live` context.

---

### `views/dashboard_view.py`

`DashboardView` — Generates a single self-contained HTML file (~1 MB) with all charts embedded.

**Charts (16 total):**
1. TPS bar chart with error bars (±1σ)
2. Efficiency score horizontal bar
3. Tokens per watt bar
4. TPS vs RAM scatter (size = power)
5. Temperatures per sensor
6. Power by rail (stacked bar)
7. GPU load vs GPU frequency (dual-axis)
8. RAM usage (color-coded)
9. Latency breakdown (model load + inference)
10. Category comparison (6 sub-plots)
11. Radar chart by category
12. Recommendation distribution (donut)
13. Inter-run stability (CV%)
14. Decision matrix (TPS vs Score)
15. Multi-run box-plots
16. Load time vs inference time scatter

**Theme:** IBM Plex Mono font, neon green/cyan/pink color palette on dark background.

**Template:** A single `_HTML` format string with `{PLACEHOLDER}` slots for all charts and tables. All CSS/JS is inlined (no external dependencies except Plotly CDN and Google Fonts).

---

### `utils/checkpoint.py`

`CheckpointManager` — Atomic JSON checkpoint for benchmark resume.

**Schema v3.0:**
```json
{
  "schema_version": "3.0",
  "run_id": 1234567890,
  "created_at": "2026-03-14 10:00:00",
  "updated_at": "2026-03-14 10:30:00",
  "total_models": 15,
  "completed": ["model-a:latest", "model-b:latest"],
  "failed": {"model-c:latest": "Timeout after 300s"},
  "stats": {}
}
```

**Atomicity:** Writes to `.tmp` file then renames atomically to prevent corruption on interruption.

**Backward compatibility:** Migrates schema v1/v2 checkpoints (different key names) on load.

---

### `utils/logger.py`

`BenchmarkLogger` — Dual-channel logger.

**Channels:**
1. **Rich console:** Colored, timestamped, with semantic prefixes/icons
2. **Log file:** Plain text with ISO timestamps, auto-rotating (10 MB × 5 backups)

**Semantic levels** (more informative than Python's standard levels):

| Method | Icon | Purpose |
|--------|------|---------|
| `perf()` | ⚡ PERF | Performance metrics after a model run |
| `thermal()` | 🌡 THERM | Temperature report |
| `power()` | 🔋 POWER | Power consumption report |
| `status()` | ✅ STATUS | Positive status update |
| `warn()` | ⚠️ WARN | Recoverable warning |
| `model_start()` | 🤖 MODEL | Model benchmark beginning |
| `model_skip()` | ⏭ SKIP | Model skipped (checkpoint) |
| `model_error()` | ❌ ERROR | Model failure |
| `cooldown()` | ❄️ COOL | Waiting for GPU cooldown |
| `bench_start()` | 📊 BENCH | Benchmark session start |
| `bench_end()` | 📊 BENCH | Benchmark session end |

---

### `utils/system_utils.py`

`SystemUtils` — Static OS utility methods.

| Method | Purpose |
|--------|---------|
| `is_root()` | Check if running as root |
| `sync_filesystem()` | Flush write buffers via `sync(1)` |
| `drop_caches()` | Release Linux kernel page/dentry/inode caches |
| `clean_memory()` | Full memory cleanup sequence |
| `schedule_shutdown()` | Schedule system shutdown via `shutdown(8)` |
| `cancel_shutdown()` | Cancel a pending shutdown |
| `get_disk_info()` | Disk usage stats for a path |
| `get_memory_info()` | RAM stats from `/proc/meminfo` |
| `check_ollama_running()` | Check if `ollama` process is running |
| `get_gpu_driver_version()` | NVIDIA driver version from `/proc/driver/nvidia/version` |
| `environment_check()` | Full environment status dict |

---

## Key Design Decisions

### Why category-scoped scoring?

A reasoning model running at 4 tok/s is not "worse" than a general model at 30 tok/s — they serve different purposes and have different computational requirements. Comparing them on the same scale would be misleading. The benchmark compares each model only against models in the same category.

### Why two scoring passes?

Category percentiles can only be computed correctly after all models in a category have been benchmarked. The first pass produces preliminary scores (useful for live display). The second pass (`_recalculate_scores_final`) recomputes percentiles with the complete dataset.

### Why streaming inference by default?

`OLLAMA_STREAM=True` measures TTFT (time-to-first-token) precisely by detecting the first non-empty token in the stream. This metric is important for interactive applications where perceived responsiveness matters as much as raw throughput.

### Why atomic checkpoints?

If the process is killed (power loss, OOM) between writes, a non-atomic write would leave the checkpoint file in an inconsistent state. Writing to `.tmp` then renaming is an atomic operation on Linux filesystems, ensuring the checkpoint is always either the previous valid state or the new complete state.

### Why `keep_alive=0` after each model?

Ollama keeps models loaded in VRAM between requests by default. Unloading after each benchmark ensures the next model starts from a cold state, making measurements comparable regardless of benchmark order.
