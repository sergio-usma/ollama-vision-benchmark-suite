# Dashboard Guide

The HTML dashboard (`reports/dashboard.html`) is a single self-contained file with 16 interactive Plotly charts across 7 navigation tabs.

## Generating the Dashboard

```bash
# Generated automatically at end of benchmark
python3 main.py

# Regenerate from existing CSV without re-running benchmark
python3 regen_dashboard.py

# Regenerate from a specific CSV
python3 regen_dashboard.py --csv path/to/results.csv --open
```

---

## Navigation Tabs

### `[ OVERVIEW ]`

The landing tab. Gives a full-picture view at a glance.

**KPI Cards (top):**
| Card | Description |
|------|-------------|
| ⚡ FASTEST | Model with highest average TPS |
| 🎯 BEST SCORE | Model with highest efficiency score |
| 🔋 MOST EFFICIENT | Model with highest tokens/watt |
| ⚡ AVG POWER | Average power across all models (W) |
| 🌡 AVG GPU TEMP | Average GPU temperature during inference |
| ✅ KEEP / REMOVE | Count breakdown: keep / review / remove |

**Charts:**
- **TPS bar chart with ±1σ error bars** — horizontal error bars show inter-run variability. A large error bar indicates inconsistent results across runs.
- **Recommendation donut** — proportion of each recommendation label in the full model set.
- **TPS vs RAM scatter** — each dot is a model. Dot size = total power. Ideal position: top-left (high TPS, low RAM).
- **Efficiency score** — horizontal bar sorted by score, color gradient green → red, with KEEP★/KEEP/OPTIONAL threshold lines.

---

### `[ PERFORMANCE ]`

Detailed performance analysis.

**Charts:**
- **TPS bar** (full width) — all models sorted by TPS with category color coding
- **Efficiency score** — same as overview but larger
- **Tokens per watt** — energy efficiency ranking
- **Latency breakdown** — stacked bar: purple = model load time, green = inference time. A very large purple segment means the model doesn't fit comfortably in VRAM (slow to load from disk each time).
- **Load vs Inference scatter** — X axis = load time, Y axis = inference time, size = RAM used. Models in the top-left are fast to load AND fast to infer. Models with high load time relative to inference time are VRAM-starved.

---

### `[ BY CATEGORY ]`

Designed for cross-category strategic decisions.

**6 sub-charts:**
| Sub-chart | What to look for |
|-----------|-----------------|
| Average TPS by category | Which categories are fastest overall |
| Max TPS achieved | The ceiling in each category |
| Average efficiency score | Which categories have better efficiency |
| Tokens per watt | Energy efficiency by category |
| Average GPU temperature | Which categories run hotter |
| Average power | Which categories are most power-hungry |

**Best model per category table** — ranked by efficiency score within each category.

**Radar chart** — normalized multidimensional comparison. Each axis is normalized 0–100. Thermal score is inverted (higher = cooler). A larger polygon area = better overall profile.

---

### `[ THERMAL & POWER ]`

Hardware conditions during inference.

**Charts:**
- **Temperatures** — grouped bars per model, one group per sensor (GPU, CPU, SOC, TJ). Red dashed line at 75°C and 85°C. Consistent GPU temps above 75°C suggest thermal throttling may have affected results.
- **Power by rail** — stacked bars showing GPU+SOC, CPU+CV, and SYS5V0 power rails. Total stack height = total system power.
- **GPU load vs frequency** — dual Y-axis. Bars = GPU load (%), line = GPU frequency (MHz). If GPU frequency drops while load is high, the GPU is throttling.
- **RAM usage** — color-coded from dark (low) to red (high). Dashed line at 28 GB recommended limit. Models above this line are using more than the recommended memory budget.

---

### `[ STABILITY ]`

Multi-run consistency analysis. Requires `NUM_RUNS ≥ 2`.

**Charts:**
- **Inter-run stability (CV%)** — coefficient of variation per model. Lower = more stable.
  - ≤ 3% STABLE (green) — highly reproducible
  - ≤ 8% GOOD (cyan) — acceptable
  - ≤ 15% VARIABLE (amber) — worth investigating
  - > 15% UNSTABLE (pink) — check for swap or thermal throttling

- **Multi-run box-plot** — shows the full distribution of individual run TPS values. The box spans Q1–Q3, the line is the median, whiskers show min/max. A narrow box = consistent model. A box with one outlier run often indicates a cold-cache effect on run 1.

---

### `[ DECISIONS ]`

Decision support for model fleet management.

**Decision Matrix (main chart):**
- X axis: TPS
- Y axis: Efficiency Score
- Color: recommendation label
- Ideal quadrant: **top-right** (high TPS + high score)
- Threshold lines: horizontal at score=42 (KEEP threshold), vertical at TPS=10 (min for GENERAL)

**Tables:**
- **Models for review/removal** — filtered to REVIEW and REMOVE models, with decision reasons
- **Benchmark errors** — models that failed, with error messages
- **Embedding models** — latency and RAM stats for embedding-only models

---

### `[ COMPLETE DATA ]`

Full scrollable HTML table with all models and all CSV columns. Use browser Ctrl+F to search for a specific model.

---

## Reading Decision Reasons

The `Decision_reasons` column and the decision matrix annotations explain each recommendation:

| Phrase | Meaning |
|--------|---------|
| `Excellent performance: 95th percentile in CODE` | Top 5% of CODE models by TPS |
| `Slow for category REASONING (3.1 < 4.0 tok/s recommended)` | Below minimum TPS threshold |
| `High RAM usage: 29.5 GB (> 28 GB limit)` | Uses more than the memory budget |
| `Note: CV=0.18 — variable results across runs` | High inter-run variability flagged |
| `Efficient: 5.2 tok/W` | Above the 4 tok/W reference threshold |

---

## Exporting and Sharing

The dashboard is a single `.html` file — no server required. Share it by:

```bash
# Copy to a web server
scp reports/dashboard.html user@server:/var/www/html/benchmark/

# Or just open in any browser
xdg-open reports/dashboard.html    # Linux
open reports/dashboard.html         # macOS
```

All Plotly charts are interactive: hover for exact values, click legend items to toggle categories, use the toolbar to zoom/pan.
