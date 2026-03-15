# Configuration Reference

All configuration is in `config.py`. Edit this file to tune the benchmark without touching application logic.

---

## Ollama API

```python
OLLAMA_API        = "http://localhost:11434/api/"  # Ollama REST API base URL
OLLAMA_TIMEOUT    = 300   # Per-inference timeout (seconds)
OLLAMA_MAX_RETRIES = 2    # Retries on ConnectionError (exponential backoff)
OLLAMA_STREAM     = True  # Streaming inference (measures TTFT precisely)
OLLAMA_CTX        = 2048  # Context window (num_ctx)
OLLAMA_TEMP       = 0.1   # Temperature (low = deterministic)
OLLAMA_KEEP_ALIVE = 0     # Unload model after each run (0 = immediate)
```

**Notes:**
- `OLLAMA_TIMEOUT` can be overridden at runtime with `--timeout N`
- `OLLAMA_KEEP_ALIVE = 0` ensures each model starts cold (no VRAM caching between models)
- `OLLAMA_TEMP = 0.1` with `seed=42` makes results more reproducible

---

## Benchmark Parameters

```python
NUM_RUNS         = 3      # Measured runs per model
WARMUP_RUN       = True   # Warmup run before measured runs (not counted)
INTER_RUN_SLEEP  = 2.0    # Seconds to sleep between runs
USE_MULTI_PROMPT = True   # Rotate prompts between runs (avoids KV-cache)
```

**Increasing `NUM_RUNS`:** More runs = better statistics but longer benchmark. 3 is the recommended minimum for meaningful CV calculation.

**`USE_MULTI_PROMPT = False`:** Uses the same prompt for all runs. Faster but susceptible to KV-cache bias (later runs are artificially faster).

---

## Hardware Monitoring

```python
HW_SAMPLE_RATE   = 1.0    # Hardware sampling rate (Hz)
HW_HISTORY_SIZE  = 300    # History buffer size (300 entries = 5 minutes)
USE_SIMULATION   = False  # Force simulation mode (auto-detected if jtop unavailable)
```

**Simulation mode** is activated automatically when jtop is not importable. You can also force it with `--simulate`.

---

## Cooldown

```python
COOLDOWN_TEMP      = 50.0  # GPU temperature threshold to wait before next model (°C)
COOLDOWN_TIMEOUT   = 300   # Maximum cooldown wait time (seconds)
COOLDOWN_CHECK_WINDOW = 30 # Seconds of history to check for stable temperature
```

**Lower `COOLDOWN_TEMP`:** More conservative, longer waits between models, more consistent thermal conditions.

**Higher `COOLDOWN_TEMP`:** Faster benchmark, but thermal conditions vary more between models.

---

## Scoring Weights

```python
SCORE_WEIGHTS = {
    "speed":    0.35,   # TPS percentile within category
    "energy":   0.25,   # tokens per watt
    "thermal":  0.15,   # GPU temperature penalty
    "ram":      0.10,   # RAM usage penalty
    "stability":0.10,   # inter-run CV (inverse)
    "load":     0.05,   # model load speed
}
```

**Recommendation thresholds:**

```python
SCORE_KEEP_STAR  = 68   # ≥68 → KEEP★
SCORE_KEEP       = 42   # ≥42 → KEEP
SCORE_OPTIONAL   = 22   # ≥22 → OPTIONAL
# <22 and TPS ok → REVIEW
# <22 and TPS low → REMOVE
```

---

## Minimum TPS per Category

```python
MIN_TPS_BY_CATEGORY = {
    "GENERAL":   10.0,
    "CODE":       8.0,
    "MATH":       5.0,
    "REASONING":  4.0,
    "VISION":     3.0,
}
```

A model below its category minimum is flagged `REVIEW` (or `REMOVE` if also high RAM). Adjust these thresholds to match your hardware and use case expectations.

---

## Model Category Override

```python
MODEL_CATEGORY_OVERRIDE = {
    # Examples:
    "qwen2.5-coder:7b":      "CODE",
    "llama3.2-vision:11b":   "VISION",
    "nomic-embed-text:latest": "EMBEDDING",
    # Add your models here:
    "my-model:latest":        "REASONING",
}
```

If a model is not listed here, category is auto-detected from the model name using keyword heuristics:

| Keywords | Category |
|----------|----------|
| `embed`, `embedding` | EMBEDDING |
| `vision`, `vl`, `llava`, `moondream` | VISION |
| `coder`, `code`, `codestral`, `deepseek-coder` | CODE |
| `math`, `deepseek-math`, `qwen-math` | MATH |
| `r1`, `reasoning`, `think`, `thinker` | REASONING |
| *(anything else)* | GENERAL |

---

## Prompts

### Single-prompt mode (`PROMPTS`)

Used when `USE_MULTI_PROMPT=False` or for EMBEDDING/VISION:

```python
PROMPTS = {
    "GENERAL":   "Explain the importance of AI in embedded edge systems.",
    "CODE":      "Write a Python function to search a binary tree...",
    "MATH":      "Solve the system of equations: 3x + 2y = 12, x - y = 1.",
    "REASONING": "John has 3 boxes. Each box contains 4 bags...",
    "VISION":    "Describe this image in detail...",
    "EMBEDDING": "The quick brown fox jumps over the lazy dog.",
}
```

### Multi-prompt rotation (`PROMPTS_MULTI`)

Used when `USE_MULTI_PROMPT=True`. 3 prompts per category, rotated across runs:

```python
PROMPTS_MULTI = {
    "GENERAL": [
        "Explain in 3 paragraphs the importance of AI in embedded edge systems.",
        "Describe the advantages and disadvantages of machine learning on IoT devices...",
        "How does edge computing work? Give 3 examples of real-time applications.",
    ],
    "CODE": [...],
    "MATH": [...],
    "REASONING": [...],
    "VISION": [...],
    "EMBEDDING": [...],
}
```

---

## File Paths

```python
BASE_DIR         = Path(__file__).parent
ASSETS_DIR       = BASE_DIR / "assets"        # test_image.jpg for VISION
CSV_FILE         = BASE_DIR / "orin_performance_report.csv"
DASHBOARD_FILE   = BASE_DIR / "reports" / "dashboard.html"
CHECKPOINT_FILE  = BASE_DIR / "checkpoints" / "benchmark_state.json"
LOG_FILE         = BASE_DIR / "logs" / "benchmark.log"
JSON_SUMMARY     = BASE_DIR / "orin_performance_report.json"
```

---

## Logging

```python
LOG_LEVEL        = "DEBUG"       # Minimum log level for file
LOG_MAX_BYTES    = 10_485_760    # 10 MB per log file
LOG_BACKUP_COUNT = 5             # Keep 5 rotated log files
LOG_ENCODING     = "utf-8"
```

---

## Dashboard Colors

```python
CATEGORY_COLORS = {
    "GENERAL":   "#4cc9f0",   # blue
    "CODE":      "#39ff14",   # neon green
    "MATH":      "#ffb700",   # amber
    "REASONING": "#f72585",   # pink
    "VISION":    "#7209b7",   # purple
    "EMBEDDING": "#4a6070",   # dim
}

RECOMMENDATION_COLORS = {
    "KEEP★":    "#39ff14",   # neon green
    "KEEP":     "#00f5d4",   # cyan
    "OPTIONAL": "#ffb700",   # amber
    "REVIEW":   "#f8961e",   # orange
    "REMOVE":   "#f72585",   # pink
    "EMBEDDING":"#4a6070",   # dim
    "ERROR":    "#ff1010",   # red
}
```

---

## Application Metadata

```python
APP_NAME    = "OllamaVision Benchmark Suite"
APP_VERSION = "4.0"
APP_AUTHOR  = "Edge AI Lab"
```
