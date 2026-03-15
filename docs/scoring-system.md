# Scoring System

## Overview

The efficiency score (0–100) is a weighted combination of 6 factors, all normalized to 0–100 before weighting.

```
Score = speed_score     × 0.35
      + energy_score    × 0.25
      − thermal_penalty × 0.15
      − ram_penalty     × 0.10
      + stability_score × 0.10
      + load_score      × 0.05
```

---

## Factor Details

### Speed (35%)

```python
speed_score = category_percentile
```

`category_percentile` is the TPS percentile of this model among all models in the same category (0–100). Computed after all models are benchmarked.

**Why percentile?** A reasoning model at 4 tok/s might be the 90th percentile in REASONING. Comparing raw TPS across categories would unfairly penalize specialized models.

**Recalculation:** Scores are calculated twice — once during the benchmark (with partial percentiles) and once at the end (`_recalculate_scores_final`) when all models in each category have been measured.

---

### Energy Efficiency (25%)

```python
energy_score = min(tokens_per_W / 4.0, 1.0) × 100
```

`tokens_per_W = tokens_per_second / (power_total_W)`.

The reference point is **4 tok/W** — a reasonable target for a 7B model on Jetson. A model achieving 4 tok/W gets 100% on this factor. The score is capped at 100 (no bonus for exceeding 4 tok/W).

---

### Thermal Penalty (15%)

```python
GPU_TEMP_OK  = 62.0   # No penalty below this
GPU_TEMP_MAX = 100.0  # Full penalty at this

if temp <= GPU_TEMP_OK:
    thermal_penalty = 0
else:
    thermal_penalty = (temp - GPU_TEMP_OK) / (GPU_TEMP_MAX - GPU_TEMP_OK) × 100
```

The penalty is 0 up to 62°C, then increases linearly to 100 at 100°C. A model that causes the GPU to run at 80°C receives a penalty of ~(80−62)/(100−62) × 100 ≈ 47 points.

**Important:** This is a penalty on the final score, not a raw component. It is subtracted (weighted at 15%), not added.

---

### RAM Penalty (10%)

```python
RAM_OK_MB  = 12_288   # 12 GB — no penalty below this
RAM_MAX_MB = 28_672   # 28 GB — full penalty at this

if ram_used <= RAM_OK_MB:
    ram_penalty = 0
else:
    ram_penalty = (ram_used - RAM_OK_MB) / (RAM_MAX_MB - RAM_OK_MB) × 100
```

Similarly a penalty subtracted from the score. Models using less than 12 GB have no RAM penalty. Models using more than 28 GB receive the full penalty.

---

### Stability (10%)

```python
CV_STABLE   = 0.03   # ≤3% → full score
CV_UNSTABLE = 0.20   # ≥20% → zero score

stability_score = max(0, (CV_UNSTABLE − cv) / (CV_UNSTABLE − CV_STABLE)) × 100
```

Coefficient of variation = `stdev / mean`. Inversely proportional to the score. Very stable models (CV ≤ 3%) receive a full 100-point bonus on this factor; very unstable models (CV ≥ 20%) receive 0.

If `NUM_RUNS = 1`, CV = 0 (only one data point), which gives the maximum stability score by default.

---

### Load Speed (5%)

```python
LOAD_FAST_S = 30.0    # ≤30s → full score
LOAD_SLOW_S = 300.0   # ≥300s → zero score

load_score = max(0, (LOAD_SLOW_S − load_s) / (LOAD_SLOW_S − LOAD_FAST_S)) × 100
```

Models that load in under 30 seconds receive full marks. Models that take 5 minutes to load receive 0. This factor is low-weighted (5%) because load time is a startup cost, not a per-inference cost.

---

## Recommendations

Derived from the final score and per-category TPS thresholds:

```
if category == "EMBEDDING":
    → EMBEDDING  (no TPS ranking applies)

elif benchmark failed:
    → ERROR

elif score >= SCORE_KEEP_STAR (68):
    → KEEP★

elif score >= SCORE_KEEP (42):
    → KEEP

elif score >= SCORE_OPTIONAL (22):
    → OPTIONAL

elif tps < min_tps_for_category AND ram > 28 GB:
    → REMOVE  (slow AND memory-hungry)

else:
    → REVIEW  (slow but not necessarily bad)
```

### Recommendation Labels

| Label | Meaning |
|-------|---------|
| `KEEP★` | Excellent model for its category. Deploy with confidence. |
| `KEEP` | Good model. Worth keeping in your active Ollama installation. |
| `OPTIONAL` | Acceptable. Keep if you have a specific use case for it. |
| `REVIEW` | Below speed threshold for category. Evaluate quality manually. |
| `REMOVE` | Below threshold AND high RAM. Strong candidate for `ollama rm`. |
| `EMBEDDING` | Embedding model — evaluated on latency, not TPS. |
| `ERROR` | Failed to benchmark. Try `ollama run <model>` manually to diagnose. |

---

## Decision Reasons

The `Decision_reasons` field in the CSV provides a human-readable explanation. Examples:

```
Excellent performance: 94th percentile in CODE
Efficient: 6.1 tok/W
Note: CV=0.04 — slight variability across runs

Slow for category GENERAL (7.3 < 10.0 tok/s recommended)
High RAM usage: 31.2 GB (> 28 GB limit)
Note: CV=0.21 — unstable results across runs
```

---

## Tuning the Scoring

All thresholds are in `config.py`. Common adjustments:

**More lenient for large models:**
```python
MIN_TPS_BY_CATEGORY["GENERAL"] = 5.0   # was 10.0
MIN_TPS_BY_CATEGORY["CODE"]    = 4.0   # was 8.0
```

**Prioritize energy efficiency:**
```python
SCORE_WEIGHTS["energy"]  = 0.35   # increase from 0.25
SCORE_WEIGHTS["speed"]   = 0.25   # decrease from 0.35
```

**Raise KEEP threshold:**
```python
SCORE_KEEP_STAR = 75   # was 68 — stricter KEEP★
SCORE_KEEP      = 50   # was 42 — stricter KEEP
```
