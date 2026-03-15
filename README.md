# 🚀 OllamaVision Benchmark Suite

<div align="center">

**Production-grade LLM benchmarking for NVIDIA Jetson AGX Orin edge hardware**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-0.1.30%2B-black?logo=ollama)](https://ollama.ai/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Jetson%20AGX%20Orin-76b900?logo=nvidia)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)

*Multi-run · Category-scoped scoring · Cyberpunk interactive HTML dashboard*

</div>

---

## 📖 What is this?

A complete benchmarking suite that evaluates **every model installed in Ollama**, purpose-built for **NVIDIA Jetson AGX Orin** (64 GB unified memory) edge hardware. It combines inference performance metrics with real-time hardware telemetry to produce data-driven decisions about which models to keep, review, or remove.

**Problem it solves:** Single-run, single-prompt benchmarks are misleading. A model can be slow on a generic prompt yet excellent in its specific domain. This suite runs multiple inference passes with rotating prompts, compares models only within their own category, and produces a multi-dimensional efficiency score reflecting real-world performance.

---

## ✨ Key Features

| Feature | Details |
|---------|---------|
| 🔁 **Multi-run** | 3 measured runs + 1 warmup per model |
| 🔀 **Prompt rotation** | 3 distinct prompts per run — avoids KV-cache bias |
| 📊 **Category-scoped scoring** | Reasoning models aren't unfairly compared to GENERAL models |
| 🎯 **6-factor efficiency score** | Speed · Energy · Thermal · RAM · Stability · Load speed |
| 🌐 **Interactive dashboard** | 16 Plotly charts across 7 tabs, cyberpunk aesthetic |
| 📈 **Category comparison** | Dedicated tab with 6 sub-charts per category |
| 🎯 **Decision Matrix** | TPS vs Score scatter, colored by recommendation |
| 📦 **Multi-run box-plots** | TPS distribution across runs per model |
| 🔍 **Dynamic model detection** | `ollama list` + API fallback — no hardcoded models |
| 💾 **Checkpoint / resume** | Resumes from where it stopped if interrupted |
| 🖥️ **Hardware monitoring** | jtop + simulation mode for development without Jetson |
| ⚡ **Streaming inference** | TTFT (time-to-first-token) measurement via SSE streaming |

---

## 🛠️ Requirements

### Hardware
- **Recommended:** NVIDIA Jetson AGX Orin (any memory config)
- **Minimum:** Any Linux machine with Ollama running; `--simulate` works without Jetson

### Software

| Dependency | Version | Notes |
|-----------|---------|-------|
| Python | 3.10+ | |
| Ollama | 0.1.30+ | Running via `ollama serve` |
| jetson-stats (jtop) | 4.2+ | Jetson only, optional |

### Python Dependencies

```bash
# Standard install
pip install -r requirements.txt

# On Jetson with system Python
pip install -r requirements.txt --break-system-packages
```

**`requirements.txt`:**
```
requests>=2.31.0
rich>=13.7.0
plotly>=5.18.0
pandas>=2.0.0
numpy>=1.24.0
```

### jetson-stats (real hardware on Jetson)

```bash
sudo pip install -U jetson-stats --break-system-packages
# Requires restart after install
```

Without `jtop`, the benchmark automatically activates simulation mode (synthetic hardware data).

---

## 🚀 Quick Start

```bash
# 1. Clone or extract the project
git clone <repo-url>
cd ollama_benchmark

# 2. Install dependencies
pip install -r requirements.txt --break-system-packages

# 3. (Optional) Add an image for VISION models
cp your_image.jpg assets/test_image.jpg

# 4. Verify environment
python3 main.py --check

# 5. Run full benchmark
sudo python3 main.py
```

---

## 📋 Command Reference

### `main.py` — Main entry point

```
python3 main.py [options]
```

| Option | Description |
|--------|-------------|
| *(no options)* | Full benchmark with interactive confirmation |
| `--simulate` | Simulate hardware (useful without Jetson or for development) |
| `--no-resume` | Ignore previous checkpoint, start from scratch |
| `--shutdown` | Shut down system 5 minutes after completion |
| `--no-dashboard` | Skip HTML dashboard generation during benchmark |
| `--open-browser` | Open dashboard in browser when finished |
| `--dashboard-only` | Only regenerate HTML from existing CSV |
| `--analyze` | Show analysis table in console (no benchmark) |
| `--check` | Verify environment: Ollama, jtop, directories, RAM |
| `--yes` / `-y` | Skip all interactive confirmations |
| `--filter CATS` | Only benchmark specific categories (e.g. `CODE,MATH`) |
| `--models TERMS` | Only models whose names contain these terms (e.g. `qwen,phi4`) |
| `--runs N` | Override `NUM_RUNS` (default: 3) |
| `--timeout N` | Override per-inference timeout in seconds |
| `--csv PATH` | Results CSV path (default: `orin_performance_report.csv`) |
| `--output-html PATH` | Dashboard HTML path (default: `reports/dashboard.html`) |

### `regen_dashboard.py` — Quick regeneration

```bash
# From current CSV
python3 regen_dashboard.py

# From a specific CSV
python3 regen_dashboard.py --csv previous_results.csv

# Open in browser when done
python3 regen_dashboard.py --open
```

Useful for testing changes to `dashboard_view.py` or updating the visualization without re-running the benchmark.

---

## 🔄 Execution Flow

```
main.py
  └── BenchmarkController.run()
        ├── _pre_flight_check()     ← verify Ollama, jtop, root
        ├── _discover_models()      ← API /api/tags + fallback 'ollama list'
        ├── _manage_checkpoint()    ← load/create checkpoint
        ├── HardwareMonitor.start() ← start jtop reading at 1Hz
        ├── TerminalView.start()    ← start Rich Live (2Hz refresh)
        └── _benchmark_loop()
              ├── [per model]
              │     ├── _do_cooldown()          ← wait GPU ≤ 50°C
              │     ├── _do_memory_clean()      ← drop_caches + unload_all
              │     └── _run_model_multi()
              │           ├── warmup run        ← (not counted)
              │           ├── run 1 — prompt A
              │           ├── run 2 — prompt B
              │           └── run 3 — prompt C
              │                 └── RunStat {tps, latency, hw_snapshot}
              ├── aggregate_runs()    ← mean, median, stdev, CV
              ├── calculate_derived() ← multi-dimensional score
              ├── CSVManager.append() ← write to CSV
              └── CheckpointManager.mark_completed()
        └── _recalculate_scores_final()  ← rescore with full percentiles
              └── DashboardView.generate() ← final HTML
```

---

## 🎯 Scoring System

### Efficiency Score Formula (0–100)

```
Score = (speed_percentile      × 0.35)
      + (energy_efficiency     × 0.25)
      − (thermal_penalty       × 0.15)
      − (ram_penalty           × 0.10)
      + (inter_run_stability   × 0.10)
      + (load_speed            × 0.05)
```

| Component | Weight | Description |
|-----------|--------|-------------|
| **Speed** | 35% | TPS percentile within the model's own category |
| **Energy efficiency** | 25% | `tokens_per_second / watts`, normalized to 4 tok/W reference |
| **Thermal penalty** | 15% | GPU temp during inference. No penalty below 62°C; linear penalty to 100°C |
| **RAM penalty** | 10% | RAM usage during inference. No penalty below 12 GB; linear to 24 GB |
| **Stability** | 10% | Inversely proportional to CV (stdev/mean). CV ≤ 3% → max score |
| **Load speed** | 5% | Model load time from disk. Models loading under 30s are not penalized |

### Minimum TPS per Category

| Category | Min TPS | Rationale |
|----------|---------|-----------|
| `GENERAL` | 10 tok/s | Basic fluent conversation |
| `CODE` | 8 tok/s | Code generation, tolerates more latency |
| `MATH` | 5 tok/s | Step-by-step reasoning, long responses |
| `REASONING` | 4 tok/s | R1-style models are slow but useful |
| `VISION` | 3 tok/s | Multimodal is inherently slower |
| `EMBEDDING` | N/A | TPS doesn't apply |

### Recommendation Table

| Recommendation | Condition | Suggested Action |
|----------------|-----------|-----------------|
| `KEEP★` | Score ≥ 68 | Keep — excellent performance/efficiency ratio |
| `KEEP` | Score ≥ 42 | Keep — good performance in its category |
| `OPTIONAL` | Score ≥ 22 | Keep if usage justifies it |
| `REVIEW` | TPS < category minimum | Evaluate whether quality justifies the slowness |
| `REMOVE` | Low TPS AND high RAM | Remove — uses disproportionate resources |
| `EMBEDDING` | EMBEDDING category | TPS ranking doesn't apply |
| `ERROR` | Benchmark failed | Retry or check Ollama compatibility |

> **Note:** `REVIEW` does not mean the model is bad — it means it was slower than expected for its category under benchmark conditions. It may be excellent for specific use cases that prioritize quality over speed.

---

## 📊 Inter-Run Stability

The coefficient of variation (CV = stdev/mean) measures how consistent results are across the 3 runs. A model with highly variable results might be swapping, competing for resources, or exhibiting non-deterministic behavior.

| Label | CV | Interpretation |
|-------|----|----------------|
| `STABLE` | ≤ 3% | Highly reproducible results |
| `GOOD` | ≤ 8% | Normal and acceptable variation |
| `VARIABLE` | ≤ 15% | Review: possible resource contention |
| `UNSTABLE` | > 15% | Issue: swap, thermal throttling, or model bug |

---

## 🌐 Dashboard — Tab Guide

### `[ OVERVIEW ]`
Global view: KPI cards with top models, TPS chart with error bars (±1σ), recommendation donut, TPS vs RAM scatter, and horizontal efficiency score.

### `[ PERFORMANCE ]`
Pure speed analysis: TPS per model, efficiency score, tokens per watt, latency with load/inference breakdown, and load vs inference scatter (useful for detecting models that don't fit in VRAM).

### `[ BY CATEGORY ]`
6 sub-charts comparing average TPS, max TPS, efficiency score, tok/W, GPU temperature and power by category. Table with the best model per category. Multidimensional radar.

### `[ THERMAL & POWER ]`
Temperatures per sensor (GPU, CPU, SOC, TJ) during inference, power by power rail (GPU+SOC, CPU+CV, SYS5V0), GPU load and frequency, and RAM usage.

### `[ STABILITY ]`
Coefficient of variation (CV%) per model with visual thresholds, and box-plot of TPS distribution across individual runs.

### `[ DECISIONS ]`
Decision matrix (TPS vs Score, colored by recommendation), table of removal/review candidates with detailed reasons, models with errors, and embedding models.

### `[ COMPLETE DATA ]`
HTML table with all models and all CSV columns. Sortable and searchable from the browser.

---

## 🏗️ Project Structure

```
ollama_benchmark/
├── main.py                          # Main CLI entry point
├── config.py                        # Centralized configuration ← edit here
├── regen_dashboard.py               # Regenerate dashboard without re-running
├── requirements.txt
│
├── models/
│   ├── data_model.py                # BenchmarkResult, RunStat, CSVManager, StatsAggregator
│   ├── ollama_model.py              # OllamaClient — HTTP API to Ollama
│   └── hardware_model.py            # HardwareMonitor — jtop + simulation
│
├── views/
│   ├── terminal_view.py             # TerminalView — Rich Live 6-panel, 2Hz
│   └── dashboard_view.py            # DashboardView — 16 Plotly charts, cyberpunk HTML
│
├── controllers/
│   └── benchmark_controller.py      # BenchmarkController — MVC orchestrator
│
├── utils/
│   ├── checkpoint.py                # CheckpointManager — atomic JSON + resume
│   ├── logger.py                    # BenchmarkLogger — semantic levels
│   └── system_utils.py              # SystemUtils — drop_caches, shutdown, env_check
│
├── assets/                          # test_image.jpg for VISION models
├── logs/                            # benchmark.log (10MB × 5 rotation)
├── checkpoints/                     # benchmark_state.json
├── reports/                         # dashboard.html
└── docs/                            # Full project documentation
```

---

## ⚙️ Configuration — `config.py`

All tunable parameters are documented in `config.py`. The most important:

```python
# ─── Benchmark rigor ──────────────────────────────────────────
NUM_RUNS         = 3      # Measured runs per model
WARMUP_RUN       = True   # Warmup run (not counted)
INTER_RUN_SLEEP  = 2.0    # Seconds between runs
USE_MULTI_PROMPT = True   # Rotate prompts between runs

# ─── Cooldown temperature ─────────────────────────────────────
COOLDOWN_TEMP    = 50.0   # °C — wait before next model

# ─── Recommendation thresholds ───────────────────────────────
SCORE_KEEP_STAR  = 68     # ≥68 → KEEP★
SCORE_KEEP       = 42     # ≥42 → KEEP
SCORE_OPTIONAL   = 22     # ≥22 → OPTIONAL

# ─── Minimum TPS per category ────────────────────────────────
MIN_TPS_BY_CATEGORY = {
    "GENERAL":   10.0,
    "CODE":       8.0,
    "MATH":       5.0,
    "REASONING":  4.0,
    "VISION":     3.0,
}
```

### Adding a new model or changing its category

In `MODEL_CATEGORY_OVERRIDE` in `config.py`:

```python
MODEL_CATEGORY_OVERRIDE = {
    # ... existing models ...
    "my-new-model:latest": "CODE",   # force category
}
```

If a model is not in the list, the system uses name-based heuristics (looks for `coder`, `vision`, `math`, etc.).

### Adding custom prompts

```python
PROMPTS_MULTI = {
    "CODE": [
        "Your prompt 1 for code",
        "Your prompt 2 for code",
        "Your prompt 3 for code",
    ],
    # ...
}
```

---

## 📁 CSV Results Columns

The CSV `orin_performance_report.csv` contains one row per benchmarked model.

### Identification
| Column | Description |
|--------|-------------|
| `Timestamp` | Benchmark date and time |
| `Model` | Model name in Ollama |
| `Category` | Assigned category |
| `Run_ID` | Sequential model index in this benchmark |

### Inference metrics (average across runs)
| Column | Description |
|--------|-------------|
| `Tokens_per_second` | Average TPS across successful runs |
| `Total_duration_s` | Total inference duration (s) |
| `Load_duration_s` | Model load time (s) |
| `API_latency_s` | API round-trip latency |
| `Prompt_eval_count` | Prompt tokens processed |
| `Eval_count` | Tokens generated |

### Multi-run statistics
| Column | Description |
|--------|-------------|
| `TPS_median` | TPS median across runs |
| `TPS_stdev` | TPS standard deviation |
| `TPS_min` / `TPS_max` | TPS range |
| `TPS_cv` | Coefficient of variation (0–1) |
| `TPS_runs` | TPS per individual run (`23.1\|24.5\|22.8`) |
| `Num_runs_ok` | Successfully completed runs |
| `Stability` | `STABLE / GOOD / VARIABLE / UNSTABLE` |

### Hardware (average of snapshots during inference)
| Column | Description |
|--------|-------------|
| `RAM_used_MB` | RAM used during inference (MB) |
| `GPU_load_%` | GPU load (%) |
| `GPU_freq_MHz` | GPU frequency (MHz) |
| `Temp_GPU_C` | GPU temperature (°C) |
| `Temp_CPU_C` | CPU temperature (°C) |
| `Power_Total_mW` | Total system power (mW) |
| `VDD_GPU_SOC_mW` | GPU+SOC power rail (mW) |
| `VDD_CPU_CV_mW` | CPU+CV power rail (mW) |
| `CPU_avg_load_%` | Average CPU load (%) |

### Decision
| Column | Description |
|--------|-------------|
| `Tokens_per_W` | Tokens generated per watt |
| `Category_pct` | TPS percentile within category (0–100) |
| `Efficiency_score` | Final score (0–100) |
| `Recommendation` | `KEEP★ / KEEP / OPTIONAL / REVIEW / REMOVE / EMBEDDING / ERROR` |
| `Decision_reasons` | Explanatory text of the recommendation |
| `Score_breakdown` | Numeric breakdown of each score component |

---

## 🔧 Troubleshooting

### Ollama not responding
```bash
# Check Ollama is running
systemctl status ollama
# or
ollama serve &
# Verify
python3 main.py --check
```

### No models installed
```bash
ollama list              # see installed models
ollama pull mistral      # install a model
```

### jtop / hardware error
```bash
# Without jtop, benchmark automatically activates simulation mode
# To install on Jetson:
sudo pip install -U jetson-stats --break-system-packages
# May require system restart
```

### Benchmark interrupted mid-way
```bash
# Checkpoint was saved automatically — just run again
python3 main.py

# To ignore checkpoint and start fresh:
python3 main.py --no-resume
```

### Dashboard with empty sections
```bash
# Verify the CSV has data
python3 main.py --analyze

# Regenerate the dashboard
python3 regen_dashboard.py
```

### Model marked REVIEW but actually works well
The benchmark measures generation speed with predefined prompts. If the model is slow on generic prompts but excellent in real usage, adjust the threshold in `config.py`:

```python
MIN_TPS_BY_CATEGORY["REASONING"] = 2.0   # more lenient
```

Or add the model to `MODEL_CATEGORY_OVERRIDE` with a category whose threshold is more appropriate.

---

## 📊 Interpreting Results

### What is a good TPS on Jetson AGX Orin?

| TPS | Interpretation |
|-----|----------------|
| > 50 | Excellent — small models (1–3B) |
| 25–50 | Very good — medium models (7B) |
| 15–25 | Good — large models (13–14B) |
| 8–15 | Acceptable — very large models (27B+) or under high load |
| < 8 | Review — may be acceptable for REASONING/VISION |

### When to remove a model?
A model gets `REMOVE` when two conditions are met simultaneously: TPS below category minimum **and** RAM usage above 28 GB. This indicates the model consumes a disproportionate amount of resources without providing acceptable speed.

### When to keep a `REVIEW` model?
`REVIEW` only indicates low TPS in this benchmark. Consider keeping it if:
- The model produces higher quality responses than faster alternatives
- The use case tolerates higher latency (batch generation, overnight use)
- The model is unique in its category (no installed alternative)

---

## 🏛️ Architecture

The project follows the **MVC pattern**:

```
config.py          ← single source of truth (read by all modules)
    │
    ├── controllers/benchmark_controller.py   ← orchestrator
    │         │
    │    ┌────┴─────────────────────────────┐
    │    ▼                                  ▼
    ├── models/                         views/
    │   ├── data_model.py               ├── terminal_view.py   (Rich Live)
    │   ├── ollama_model.py             └── dashboard_view.py  (Plotly HTML)
    │   └── hardware_model.py
    │
    └── utils/
        ├── checkpoint.py
        ├── logger.py
        └── system_utils.py
```

---

## 📄 License

MIT License — free for personal, academic and commercial use.

---

<div align="center">

*OllamaVision Benchmark Suite v4.0 — Edge AI · NVIDIA Jetson AGX Orin*

</div>
