#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py — OllamaVision Benchmark Suite v3.0
Centralized configuration. Adjust all parameters here.
"""
from pathlib import Path

# ─── PATHS ────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
ASSETS_DIR      = BASE_DIR / "assets"
LOGS_DIR        = BASE_DIR / "logs"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
REPORTS_DIR     = BASE_DIR / "reports"
CSV_OUTPUT      = BASE_DIR / "orin_performance_report.csv"
CHECKPOINT_FILE = CHECKPOINTS_DIR / "benchmark_state.json"
LOG_FILE        = LOGS_DIR / "benchmark.log"
DASHBOARD_FILE  = REPORTS_DIR / "dashboard.html"

for _d in (ASSETS_DIR, LOGS_DIR, CHECKPOINTS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ─── OLLAMA API ────────────────────────────────────────────────────────────────
OLLAMA_API        = "http://localhost:11434/api/"
OLLAMA_TIMEOUT    = 600
OLLAMA_CTX        = 4096
OLLAMA_TEMP       = 0.0
OLLAMA_KEEP_ALIVE = 0
OLLAMA_MAX_RETRIES = 2    # Retries on transient network errors
OLLAMA_STREAM      = True  # Use streaming for more precise TTFT measurement

# ─── BENCHMARK — MULTI-RUN ────────────────────────────────────────────────────
NUM_RUNS         = 3      # Measurement runs per model (recommended ≥3)
WARMUP_RUN       = True   # 1 warmup run before measuring
INTER_RUN_SLEEP  = 2.0    # Seconds between runs of the same model
USE_MULTI_PROMPT = True   # Rotate prompts between runs to avoid KV-cache bias

# ─── HARDWARE / THERMAL ───────────────────────────────────────────────────────
COOLDOWN_TEMP     = 50.0
MAX_COOLDOWN_WAIT = 300
MONITOR_INTERVAL  = 1.0
MONITOR_HISTORY   = 300
DROP_CACHES       = True

# ─── BENCHMARK GENERAL ────────────────────────────────────────────────────────
SHUTDOWN_DELAY_MIN     = 5
DASHBOARD_UPDATE_EVERY = 3
MEMORY_CLEAN_SLEEP     = 2

# ─── DECISION SYSTEM ──────────────────────────────────────────────────────────
# Minimum TPS per category (conservative and realistic thresholds)
MIN_TPS_BY_CATEGORY = {
    "GENERAL":   10.0,
    "CODE":       8.0,
    "MATH":       5.0,
    "REASONING":  4.0,
    "VISION":     3.0,
    "EMBEDDING":  0.0,
}
MIN_TOKENS_DECISION = 8.0
MAX_RAM_DECISION_MB = 28_000  # Jetson AGX Orin 64GB unified memory

# Efficiency Score weights (multi-dimensional, sum = 1.0)
SCORE_WEIGHT_SPEED_PCT = 0.35  # TPS percentile within its category
SCORE_WEIGHT_ENERGY    = 0.25  # tok/W
SCORE_WEIGHT_THERMAL   = 0.15  # thermal penalty
SCORE_WEIGHT_RAM       = 0.10  # RAM penalty
SCORE_WEIGHT_STABILITY = 0.10  # inter-run stability (low CV = better)
SCORE_WEIGHT_LOAD      = 0.05  # load speed

# Absolute references (fallback when only 1 model)
SCORE_REF_TPS      = 50.0
SCORE_REF_TPW      = 4.0
SCORE_THERMAL_WARN = 62.0
SCORE_RAM_WARN_MB  = 12_000

# Recommendation thresholds
SCORE_KEEP_STAR = 68
SCORE_KEEP      = 42
SCORE_OPTIONAL  = 22

# ─── PROMPTS MULTI (rotation between runs) ────────────────────────────────────
PROMPTS_MULTI = {
    "GENERAL": [
        "Explain in 3 paragraphs the importance of artificial intelligence in embedded edge systems.",
        "Describe the advantages and disadvantages of machine learning on IoT devices with limited resources.",
        "How does edge computing work? Give 3 examples of real-time applications.",
    ],
    "MATH": [
        "Solve step by step: find the derivative of f(x) = x^3 + 5x^2 - 7 at x=2 and verify the result.",
        "Calculate the definite integral of f(x) = 2x^2 + 3x from 0 to 4. Show all steps.",
        "Solve the system: 3x + 2y = 12, x - y = 1. Explain the method used.",
    ],
    "CODE": [
        "Write a Python decorator that measures the execution time of any function and displays it in ms.",
        "Implement a LRU cache in Python without using functools.lru_cache.",
        "Write a Python context manager to handle SQLite connections with error handling.",
    ],
    "REASONING": [
        "If 8 people take 8 hours to dig 8 holes, how long do 4 people take to dig 4 holes? Reason step by step.",
        "You have 3 boxes: apples, oranges, mixed. All labels are wrong. How do you identify all by taking out one fruit?",
        "A train leaves A at 60km/h, another from B at 90km/h, distance 300km. When do they meet and how far from A?",
    ],
    "VISION": [
        "Describe this image in detail: objects, colors, composition and usage contexts.",
        "Analyze the visual composition: main elements, planes, lighting and narrative.",
        "What technical information can you extract? Describe materials, estimated dimensions and function.",
    ],
    "EMBEDDING": [
        "Artificial intelligence at the edge for real-time embedded systems.",
        "Machine learning deployment on embedded systems with resource constraints.",
        "Embedded systems: energy efficiency and data processing pipelines.",
    ],
}

PROMPTS = {k: v[0] for k, v in PROMPTS_MULTI.items()}

# ─── MODEL CLASSIFICATION ─────────────────────────────────────────────────────
MODEL_CATEGORY_OVERRIDE = {
    "cogito:latest": "GENERAL", "openthinker:latest": "GENERAL",
    "qwen2.5:latest": "GENERAL", "qwen3:14b": "GENERAL",
    "qwen3.5:latest": "GENERAL", "qwen3-next:latest": "GENERAL",
    "olmo-3.1:latest": "GENERAL", "mistral:latest": "GENERAL",
    "mistral-nemo:latest": "GENERAL", "mistrallite:latest": "GENERAL",
    "mistral-small3.2:latest": "GENERAL", "ministral-3:latest": "GENERAL",
    "magistral:latest": "GENERAL", "phi4:latest": "GENERAL",
    "nemotron:latest": "GENERAL", "nemotron-3-nano:latest": "GENERAL",
    "glm-4.7-flash:latest": "GENERAL", "gpt-oss:20b": "GENERAL",
    "gemma3:27b": "GENERAL",
    "codeqwen:latest": "CODE", "qwen2.5-coder:latest": "CODE",
    "qwen3-coder:latest": "CODE", "qwen3-coder-next:latest": "CODE",
    "opencoder:latest": "CODE", "deepcoder:latest": "CODE",
    "granite3.3:latest": "CODE", "devstral:latest": "CODE",
    "devstral-small-2:latest": "CODE",
    "qwen2-math:latest": "MATH", "mathstral:latest": "MATH",
    "phi4-mini-reasoning:latest": "MATH", "phi4-reasoning:latest": "MATH",
    "llama3.2-vision:latest": "VISION", "qwen3-vl:latest": "VISION",
    "qwen2.5vl:latest": "VISION", "deepseek-ocr:latest": "VISION",
    "glm-ocr:latest": "VISION",
    "nomic-embed-text-v2-moe:latest": "EMBEDDING",
    "embeddinggemma:latest": "EMBEDDING", "qwen3-embedding:latest": "EMBEDDING",
    "lfm2.5-thinking:latest": "REASONING", "lfm2:latest": "REASONING",
    "r1-1776:latest": "REASONING", "deepseek-r1:14b": "REASONING",
    "deepseek-r1:8b": "REASONING",
}

# ─── COLORS ───────────────────────────────────────────────────────────────────
CATEGORY_COLORS = {
    "GENERAL": "#00f5d4", "CODE": "#7209b7", "MATH": "#f72585",
    "REASONING": "#4cc9f0", "VISION": "#f8961e", "EMBEDDING": "#43aa8b",
}
RECOMMENDATION_COLORS = {
    "KEEP★": "#39ff14", "KEEP": "#00f5d4", "REVIEW": "#ffb700",
    "REMOVE": "#f72585", "OPTIONAL": "#7209b7",
    "EMBEDDING": "#43aa8b", "ERROR": "#ff4444",
}

# ─── LOGGING ──────────────────────────────────────────────────────────────────
LOG_LEVEL        = "DEBUG"
LOG_MAX_BYTES    = 10 * 1024 * 1024
LOG_BACKUP_COUNT = 5
LOG_ENCODING     = "utf-8"

# ─── APP META ─────────────────────────────────────────────────────────────────
APP_NAME    = "OllamaVision Benchmark Suite"
APP_VERSION = "4.0.0"
APP_AUTHOR  = "Jetson AGX Orin — Edge AI Lab"

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def classify_model(model_name: str) -> str:
    if model_name in MODEL_CATEGORY_OVERRIDE:
        return MODEL_CATEGORY_OVERRIDE[model_name]
    n = model_name.lower()
    if any(x in n for x in ["vision","vl:","ocr","llava","cogvlm","moondream","clip"]):
        return "VISION"
    if any(x in n for x in ["math","mathstral","deepseek-math","numina"]):
        return "MATH"
    if any(x in n for x in ["coder","code","opencoder","devstral","codellama",
                              "deepcoder","starcoder"]):
        return "CODE"
    if any(x in n for x in ["reasoning","think",":r1","deepseek-r1","r1-","lfm",
                              "skywork","qwq","marco-o"]):
        return "REASONING"
    if any(x in n for x in ["embed","nomic-embed","snowflake","bge-","e5-",
                              "minilm","sentence"]):
        return "EMBEDDING"
    return "GENERAL"

def get_min_tps(category: str) -> float:
    return MIN_TPS_BY_CATEGORY.get(category, MIN_TOKENS_DECISION)

def get_prompts_for_category(category: str, num: int = 3):
    prompts = PROMPTS_MULTI.get(category, PROMPTS_MULTI["GENERAL"])
    return (prompts * ((num // len(prompts)) + 1))[:num]
