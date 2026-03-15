#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/data_model.py — Data models v3.0
Multi-run support, inter-run statistics, category percentile-based scoring,
multi-dimensional decision system.
"""
from __future__ import annotations

import csv
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# HARDWARE SNAPSHOT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HardwareSnapshot:
    ts: float = 0.0
    cpu_loads:   List[float] = field(default_factory=list)
    cpu_freqs:   List[int]   = field(default_factory=list)
    gpu_load:    float = 0.0
    gpu_freq:    int   = 0
    emc_freq:    int   = 0
    vdd_gpu_soc: int   = 0
    vdd_cpu_cv:  int   = 0
    vin_sys_5v0: int   = 0
    power_total: int   = 0
    temp_gpu:    float = 0.0
    temp_cpu:    float = 0.0
    temp_soc:    float = 0.0
    temp_tj:     float = 0.0
    ram_used:    float = 0.0
    ram_total:   float = 1.0
    swap_used:   float = 0.0
    swap_total:  float = 1.0
    fan_speeds:  str   = ""
    fan_rpms:    str   = ""
    fan_profile: str   = ""
    disk_total:     float = 0.0
    disk_used:      float = 0.0
    disk_available: float = 0.0
    uptime_seconds: float = 0.0

    @property
    def ram_percent(self) -> float:
        return (self.ram_used / self.ram_total * 100.0) if self.ram_total > 0 else 0.0

    @property
    def swap_percent(self) -> float:
        return (self.swap_used / self.swap_total * 100.0) if self.swap_total > 0 else 0.0

    @property
    def cpu_avg_load(self) -> float:
        return statistics.mean(self.cpu_loads) if self.cpu_loads else 0.0

    @property
    def cpu_max_load(self) -> float:
        return max(self.cpu_loads) if self.cpu_loads else 0.0

    @property
    def power_watts(self) -> float:
        return self.power_total / 1000.0

    @property
    def temp_max(self) -> float:
        return max(self.temp_gpu, self.temp_cpu, self.temp_soc, self.temp_tj)

    def cpu_loads_str(self) -> str:
        return "|".join(f"{l:.1f}" for l in self.cpu_loads)

    def cpu_freqs_str(self) -> str:
        return "|".join(str(f) for f in self.cpu_freqs)

    @classmethod
    def average(cls, snaps: List["HardwareSnapshot"]) -> "HardwareSnapshot":
        """Average a list of snapshots — useful to represent multi-run hardware data."""
        if not snaps:
            return cls()
        avg = cls()
        avg.ts = snaps[-1].ts
        n = len(snaps)

        def _avg(attr):
            vals = [getattr(s, attr) for s in snaps if isinstance(getattr(s, attr), (int, float))]
            return sum(vals) / len(vals) if vals else 0.0

        avg.gpu_load    = _avg("gpu_load")
        avg.gpu_freq    = int(_avg("gpu_freq"))
        avg.emc_freq    = int(_avg("emc_freq"))
        avg.vdd_gpu_soc = int(_avg("vdd_gpu_soc"))
        avg.vdd_cpu_cv  = int(_avg("vdd_cpu_cv"))
        avg.vin_sys_5v0 = int(_avg("vin_sys_5v0"))
        avg.power_total = int(_avg("power_total"))
        avg.temp_gpu    = _avg("temp_gpu")
        avg.temp_cpu    = _avg("temp_cpu")
        avg.temp_soc    = _avg("temp_soc")
        avg.temp_tj     = _avg("temp_tj")
        avg.ram_used    = _avg("ram_used")
        avg.ram_total   = snaps[0].ram_total
        avg.swap_used   = _avg("swap_used")
        avg.swap_total  = snaps[0].swap_total
        avg.disk_total  = snaps[0].disk_total
        avg.disk_used   = _avg("disk_used")
        avg.disk_available = _avg("disk_available")
        avg.uptime_seconds = snaps[-1].uptime_seconds
        avg.fan_speeds  = snaps[-1].fan_speeds
        avg.fan_rpms    = snaps[-1].fan_rpms
        avg.fan_profile = snaps[-1].fan_profile

        # CPU: average per core
        n_cores = max(len(s.cpu_loads) for s in snaps)
        if n_cores > 0:
            avg.cpu_loads = []
            avg.cpu_freqs = []
            for i in range(n_cores):
                loads = [s.cpu_loads[i] for s in snaps if i < len(s.cpu_loads)]
                freqs = [s.cpu_freqs[i] for s in snaps if i < len(s.cpu_freqs)]
                avg.cpu_loads.append(sum(loads) / len(loads) if loads else 0.0)
                avg.cpu_freqs.append(int(sum(freqs) / len(freqs)) if freqs else 0)
        return avg


# ═══════════════════════════════════════════════════════════════════════════════
# RUN STAT — estadísticas de un solo run
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RunStat:
    """Resultado de un único run de inferencia."""
    run_num:           int   = 0
    tokens_per_second: float = 0.0
    total_duration_s:  float = 0.0
    load_duration_s:   float = 0.0
    prompt_eval_count: int   = 0
    eval_count:        int   = 0
    api_latency_s:     float = 0.0
    response_preview:  str   = ""
    success:           bool  = True
    error_msg:         str   = ""
    ttft_s:            float = 0.0  # Time-to-first-token (streaming only)
    hw: HardwareSnapshot = field(default_factory=HardwareSnapshot)


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK RESULT — agregado multi-run
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    """
    Aggregated result from N runs of a model.
    Includes multi-run statistics and multi-dimensional scoring.
    """
    timestamp:   str  = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    model:       str  = ""
    category:    str  = ""
    run_id:      int  = 0
    success:     bool = True
    error_msg:   str  = ""

    # ── Aggregated metrics (average of successful runs) ──
    tokens_per_second:   float = 0.0
    total_duration_s:    float = 0.0
    load_duration_s:     float = 0.0
    prompt_eval_count:   int   = 0
    eval_count:          int   = 0
    api_latency_s:       float = 0.0
    response_text:       str   = ""

    # ── Inter-run statistics (multi-run) ──
    tps_runs:     List[float] = field(default_factory=list)  # TPS por run
    tps_median:   float = 0.0
    tps_stdev:    float = 0.0
    tps_min:      float = 0.0
    tps_max:      float = 0.0
    tps_cv:       float = 0.0   # Coefficient of variation (stdev/mean) — stability
    num_runs_ok:  int   = 0     # Runs exitosos
    num_runs_total: int = 0     # Runs intentados

    # ── Hardware (average of all runs) ──
    hw: HardwareSnapshot = field(default_factory=HardwareSnapshot)

    # ── TTFT (Time-To-First-Token) — via streaming, not stored in CSV ──
    ttft_s:     float = 0.0   # promedio de runs
    ttft_min:   float = 0.0
    ttft_max:   float = 0.0

    # ── Derived metrics / decision ──
    tokens_per_watt:  float = 0.0
    efficiency_score: float = 0.0
    category_percentile: float = 0.0  # percentil TPS dentro de su categoría
    recommendation:   str   = ""
    score_breakdown:  str   = ""
    decision_reasons: str   = ""      # texto explicativo de la decisión

    @property
    def is_embedding(self) -> bool:
        return self.category == "EMBEDDING"

    @property
    def tps_display(self) -> str:
        if self.is_embedding:
            return "embed"
        return f"{self.tokens_per_second:.1f}"

    @property
    def ttft_display(self) -> str:
        """TTFT in ms, or N/A if not available."""
        if self.is_embedding or self.ttft_s <= 0:
            return "N/A"
        ms = self.ttft_s * 1000
        return f"{ms:.0f}ms" if ms < 10_000 else f"{ms/1000:.1f}s"

    @property
    def stability_label(self) -> str:
        if self.tps_cv <= 0.03:  return "STABLE"
        if self.tps_cv <= 0.08:  return "GOOD"
        if self.tps_cv <= 0.15:  return "VARIABLE"
        return "UNSTABLE"

    def aggregate_runs(self, runs: List[RunStat]) -> None:
        """Aggregate a list of RunStat into this result's metrics."""
        ok_runs = [r for r in runs if r.success and r.tokens_per_second > 0]
        self.num_runs_total = len(runs)
        self.num_runs_ok    = len(ok_runs)

        if not ok_runs:
            self.success = False
            if runs:
                self.error_msg = runs[-1].error_msg
            return

        self.success = True
        self.tps_runs = [r.tokens_per_second for r in ok_runs]

        # Average
        self.tokens_per_second = statistics.mean(self.tps_runs)
        self.tps_median        = statistics.median(self.tps_runs)
        self.tps_min           = min(self.tps_runs)
        self.tps_max           = max(self.tps_runs)
        self.tps_stdev         = statistics.stdev(self.tps_runs) if len(self.tps_runs) > 1 else 0.0
        self.tps_cv            = (self.tps_stdev / self.tokens_per_second) if self.tokens_per_second > 0 else 0.0

        # Other metrics (average of runs)
        self.total_duration_s  = statistics.mean([r.total_duration_s for r in ok_runs])
        self.load_duration_s   = statistics.mean([r.load_duration_s  for r in ok_runs])
        self.api_latency_s     = statistics.mean([r.api_latency_s    for r in ok_runs])
        self.prompt_eval_count = int(statistics.mean([r.prompt_eval_count for r in ok_runs]))
        self.eval_count        = int(statistics.mean([r.eval_count        for r in ok_runs]))
        self.response_text     = ok_runs[-1].response_preview[:200]

        # Averaged hardware
        hw_snaps = [r.hw for r in ok_runs]
        self.hw  = HardwareSnapshot.average(hw_snaps)

        # TTFT (only if available via streaming)
        ttft_vals = [r.ttft_s for r in ok_runs if r.ttft_s > 0]
        if ttft_vals:
            self.ttft_s   = statistics.mean(ttft_vals)
            self.ttft_min = min(ttft_vals)
            self.ttft_max = max(ttft_vals)

    def calculate_derived(self, category_tps_list: Optional[List[float]] = None) -> None:
        """
        Calculate tok/W, efficiency_score and recommendation.
        If category_tps_list is provided (all TPS in the category),
        uses relative percentile → fairer for large/specialized models.
        """
        from config import (
            SCORE_WEIGHT_SPEED_PCT, SCORE_WEIGHT_ENERGY, SCORE_WEIGHT_THERMAL,
            SCORE_WEIGHT_RAM, SCORE_WEIGHT_STABILITY, SCORE_WEIGHT_LOAD,
            SCORE_REF_TPS, SCORE_REF_TPW,
            SCORE_THERMAL_WARN, SCORE_RAM_WARN_MB,
            SCORE_KEEP_STAR, SCORE_KEEP, SCORE_OPTIONAL,
            get_min_tps,
        )

        pwr_w = max(self.hw.power_watts, 0.1)
        self.tokens_per_watt = self.tokens_per_second / pwr_w if self.tokens_per_second > 0 else 0.0

        if self.is_embedding or self.tokens_per_second <= 0:
            self.efficiency_score = 0.0
            self.recommendation   = "EMBEDDING" if self.is_embedding else "ERROR"
            self.score_breakdown  = "N/A"
            self.decision_reasons = "Embedding model — TPS not applicable" if self.is_embedding else "No tokens generated"
            return

        # ── 1. Speed — percentile within category ──
        if category_tps_list and len(category_tps_list) > 1:
            n_below = sum(1 for t in category_tps_list if t < self.tokens_per_second)
            self.category_percentile = (n_below / (len(category_tps_list) - 1)) * 100.0
            speed_score = self.category_percentile
        else:
            # Fallback: absolute normalization
            speed_score = min(self.tokens_per_second / SCORE_REF_TPS, 1.0) * 100.0
            self.category_percentile = speed_score

        # ── 2. Energy efficiency ──
        energy_score = min(self.tokens_per_watt / SCORE_REF_TPW, 1.0) * 100.0

        # ── 3. Thermal penalty (linear from SCORE_THERMAL_WARN to 100°C) ──
        thermal_range = 100.0 - SCORE_THERMAL_WARN
        thermal_pen   = max(0.0, (self.hw.temp_gpu - SCORE_THERMAL_WARN) / thermal_range) * 100.0

        # ── 4. RAM penalty ──
        ram_range = SCORE_RAM_WARN_MB
        ram_pen   = max(0.0, (self.hw.ram_used - SCORE_RAM_WARN_MB) / ram_range) * 100.0

        # ── 5. Inter-run stability (low CV = stable = better) ──
        stability_score = max(0.0, (1.0 - self.tps_cv * 5.0)) * 100.0
        stability_score = min(stability_score, 100.0)

        # ── 6. Slow-load penalty (load > 30s is penalized) ──
        load_pen = min(max(0.0, (self.load_duration_s - 30.0) / 60.0), 1.0) * 100.0
        load_score = 100.0 - load_pen

        # ── Final score ──
        raw = (
            speed_score     * SCORE_WEIGHT_SPEED_PCT
          + energy_score    * SCORE_WEIGHT_ENERGY
          - thermal_pen     * SCORE_WEIGHT_THERMAL
          - ram_pen         * SCORE_WEIGHT_RAM
          + stability_score * SCORE_WEIGHT_STABILITY
          + load_score      * SCORE_WEIGHT_LOAD
        )
        self.efficiency_score = round(max(0.0, min(100.0, raw)), 2)

        self.score_breakdown = (
            f"spd_pct={speed_score:.0f}×{SCORE_WEIGHT_SPEED_PCT} "
            f"nrg={energy_score:.0f}×{SCORE_WEIGHT_ENERGY} "
            f"therm_pen={thermal_pen:.0f}×{SCORE_WEIGHT_THERMAL} "
            f"ram_pen={ram_pen:.0f}×{SCORE_WEIGHT_RAM} "
            f"stab={stability_score:.0f}×{SCORE_WEIGHT_STABILITY} "
            f"load={load_score:.0f}×{SCORE_WEIGHT_LOAD}"
        )

        # ── Recommendation ──
        min_tps    = get_min_tps(self.category)
        slow       = self.tokens_per_second < min_tps
        heavy      = self.hw.ram_used > 28_000
        unstable   = self.tps_cv > 0.20

        reasons = []
        if not self.success:
            self.recommendation = "ERROR"
            reasons.append(f"Error: {self.error_msg[:80]}")
        elif slow and heavy:
            self.recommendation = "REMOVE"
            reasons.append(f"Slow ({self.tokens_per_second:.1f} < {min_tps} tok/s) AND heavy ({self.hw.ram_used/1024:.0f}GB RAM)")
        elif slow:
            self.recommendation = "REVIEW"
            reasons.append(f"Slow for category {self.category} ({self.tokens_per_second:.1f} < {min_tps} tok/s recommended)")
            reasons.append(f"Consider if quality justifies the speed trade-off")
        elif self.efficiency_score >= SCORE_KEEP_STAR:
            self.recommendation = "KEEP★"
            reasons.append(f"Excellent performance: {self.category_percentile:.0f}th percentile in {self.category}")
            reasons.append(f"{self.tokens_per_watt:.2f} tok/W — very efficient")
        elif self.efficiency_score >= SCORE_KEEP:
            self.recommendation = "KEEP"
            reasons.append(f"Good performance: {self.tokens_per_second:.1f} tok/s")
            if unstable:
                reasons.append(f"Note: CV={self.tps_cv:.2f} — variable results across runs")
        elif self.efficiency_score >= SCORE_OPTIONAL:
            self.recommendation = "OPTIONAL"
            reasons.append(f"Acceptable performance but not the best in {self.category}")
            reasons.append(f"Consider using a better-ranked model in this category")
        else:
            self.recommendation = "REVIEW"
            reasons.append(f"Low score ({self.efficiency_score:.0f}/100)")
            if thermal_pen > 30:
                reasons.append(f"High temperature: {self.hw.temp_gpu:.0f}°C penalizes score")
            if ram_pen > 30:
                reasons.append(f"High RAM usage: {self.hw.ram_used/1024:.0f}GB")

        self.decision_reasons = " | ".join(reasons)

    def to_csv_row(self) -> Dict[str, Any]:
        hw = self.hw
        return {
            "Timestamp":         self.timestamp,
            "Model":             self.model,
            "Category":          self.category,
            "Run_ID":            self.run_id,
            "Success":           self.success,
            "Error":             self.error_msg,
            # Averaged metrics
            "Tokens_per_second": round(self.tokens_per_second, 3),
            "Total_duration_s":  round(self.total_duration_s, 3),
            "Load_duration_s":   round(self.load_duration_s, 3),
            "Prompt_eval_count": self.prompt_eval_count,
            "Eval_count":        self.eval_count,
            "API_latency_s":     round(self.api_latency_s, 3),
            "Response_preview":  self.response_text[:120].replace("\n", " "),
            # Multi-run statistics
            "TPS_median":        round(self.tps_median, 3),
            "TPS_stdev":         round(self.tps_stdev, 3),
            "TPS_min":           round(self.tps_min, 3),
            "TPS_max":           round(self.tps_max, 3),
            "TPS_cv":            round(self.tps_cv, 4),
            "TPS_runs":          "|".join(f"{t:.2f}" for t in self.tps_runs),
            "Num_runs_ok":       self.num_runs_ok,
            "Num_runs_total":    self.num_runs_total,
            "Stability":         self.stability_label,
            # Memory
            "RAM_used_MB":       round(hw.ram_used, 1),
            "RAM_total_MB":      round(hw.ram_total, 1),
            "RAM_percent":       round(hw.ram_percent, 1),
            "SWAP_used_MB":      round(hw.swap_used, 1),
            "SWAP_percent":      round(hw.swap_percent, 1),
            # CPU
            "CPU_avg_load_%":    round(hw.cpu_avg_load, 1),
            "CPU_max_load_%":    round(hw.cpu_max_load, 1),
            "CPU_loads_%":       hw.cpu_loads_str(),
            "CPU_freqs_MHz":     hw.cpu_freqs_str(),
            # GPU
            "GPU_load_%":        round(hw.gpu_load, 1),
            "GPU_freq_MHz":      hw.gpu_freq,
            "EMC_freq_MHz":      hw.emc_freq,
            # Power
            "VDD_GPU_SOC_mW":    hw.vdd_gpu_soc,
            "VDD_CPU_CV_mW":     hw.vdd_cpu_cv,
            "VIN_SYS_5V0_mW":    hw.vin_sys_5v0,
            "Power_Total_mW":    hw.power_total,
            # Temperatures
            "Temp_GPU_C":        round(hw.temp_gpu, 1),
            "Temp_CPU_C":        round(hw.temp_cpu, 1),
            "Temp_SOC_C":        round(hw.temp_soc, 1),
            "Temp_TJ_C":         round(hw.temp_tj, 1),
            # Fan & Disk
            "Fan_speeds":        hw.fan_speeds,
            "Fan_RPMs":          hw.fan_rpms,
            "Fan_profile":       hw.fan_profile,
            "Disk_total_GB":     round(hw.disk_total, 1),
            "Disk_used_GB":      round(hw.disk_used, 1),
            "Disk_available_GB": round(hw.disk_available, 1),
            "Uptime_seconds":    round(hw.uptime_seconds, 0),
            # Derived
            "Tokens_per_W":      round(self.tokens_per_watt, 3),
            "Category_pct":      round(self.category_percentile, 1),
            "Efficiency_score":  self.efficiency_score,
            "Recommendation":    self.recommendation,
            "Decision_reasons":  self.decision_reasons,
            "Score_breakdown":   self.score_breakdown,
        }

    @classmethod
    def get_csv_fieldnames(cls) -> List[str]:
        return list(cls().to_csv_row().keys())


# ═══════════════════════════════════════════════════════════════════════════════
# CSV MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class CSVManager:
    FIELDNAMES = BenchmarkResult.get_csv_fieldnames()

    def __init__(self, path: Path):
        self.path = Path(path)
        self._ensure_header()

    def _ensure_header(self) -> None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.FIELDNAMES,
                               extrasaction="ignore").writeheader()

    def append(self, result: BenchmarkResult) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.FIELDNAMES,
                           extrasaction="ignore").writerow(result.to_csv_row())

    def load_dataframe(self) -> pd.DataFrame:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return pd.DataFrame(columns=self.FIELDNAMES)
        df = pd.read_csv(self.path, encoding="utf-8")
        df.columns = df.columns.str.strip()
        num_cols = [
            "Tokens_per_second","TPS_median","TPS_stdev","TPS_min","TPS_max","TPS_cv",
            "Power_Total_mW","RAM_used_MB","RAM_percent","Temp_GPU_C","Temp_CPU_C",
            "Temp_SOC_C","Temp_TJ_C","GPU_load_%","GPU_freq_MHz","Tokens_per_W",
            "Efficiency_score","Category_pct","API_latency_s","VDD_GPU_SOC_mW",
            "VDD_CPU_CV_mW","VIN_SYS_5V0_mW","Load_duration_s","Total_duration_s",
            "Num_runs_ok","Num_runs_total",
        ]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return df

    def get_completed_models(self) -> List[str]:
        df = self.load_dataframe()
        if "Model" not in df.columns or df.empty:
            return []
        return df["Model"].dropna().unique().tolist()


# ═══════════════════════════════════════════════════════════════════════════════
# STATS AGGREGATOR
# ═══════════════════════════════════════════════════════════════════════════════

class StatsAggregator:
    def __init__(self):
        self.results: List[BenchmarkResult] = []

    def add(self, r: BenchmarkResult) -> None:
        self.results.append(r)

    def clear(self) -> None:
        self.results.clear()

    def get_ok_results(self) -> List[BenchmarkResult]:
        return [r for r in self.results if r.success and r.tokens_per_second > 0]

    def get_category_tps_map(self) -> Dict[str, List[float]]:
        """Returns {category: [tps1, tps2, ...]} for percentile-based scoring."""
        cat_map: Dict[str, List[float]] = {}
        for r in self.get_ok_results():
            cat_map.setdefault(r.category, []).append(r.tokens_per_second)
        return cat_map

    def get_live_stats(self) -> Dict[str, Any]:
        all_r = self.results
        ok    = self.get_ok_results()

        if not all_r:
            return {"total_tested": 0}

        stats: Dict[str, Any] = {
            "total_tested":     len(all_r),
            "total_success":    len(ok),
            "total_failed":     len([r for r in all_r if not r.success]),
            "total_embeddings": len([r for r in all_r if r.is_embedding]),
        }

        if ok:
            toks  = [r.tokens_per_second for r in ok]
            pwrs  = [r.hw.power_total    for r in ok]
            temps = [r.hw.temp_gpu       for r in ok]
            effs  = [r.efficiency_score  for r in ok]
            tpws  = [r.tokens_per_watt   for r in ok]
            cvs   = [r.tps_cv            for r in ok]

            best_tps = max(ok, key=lambda r: r.tokens_per_second)
            best_eff = max(ok, key=lambda r: r.efficiency_score)
            best_tpw = max(ok, key=lambda r: r.tokens_per_watt)

            by_cat: Dict[str, List[float]] = {}
            for r in ok:
                by_cat.setdefault(r.category, []).append(r.tokens_per_second)

            recs: Dict[str, List[str]] = {}
            for r in self.results:
                recs.setdefault(r.recommendation, []).append(r.model)

            stats.update({
                "avg_tokens_s":    round(statistics.mean(toks), 2),
                "max_tokens_s":    round(max(toks), 2),
                "min_tokens_s":    round(min(toks), 2),
                "median_tokens_s": round(statistics.median(toks), 2),
                "stdev_tokens_s":  round(statistics.stdev(toks), 2) if len(toks) > 1 else 0.0,
                "avg_power_mw":    round(statistics.mean(pwrs), 0),
                "avg_temp_gpu":    round(statistics.mean(temps), 1),
                "max_temp_gpu":    round(max(temps), 1),
                "avg_efficiency":  round(statistics.mean(effs), 1),
                "avg_tok_per_w":   round(statistics.mean(tpws), 3),
                "avg_cv":          round(statistics.mean(cvs), 3),
                "best_model":      best_tps.model,
                "best_tokens_s":   best_tps.tokens_per_second,
                "most_efficient":  best_eff.model,
                "best_eff_score":  best_eff.efficiency_score,
                "most_eco":        best_tpw.model,
                "best_tok_per_w":  best_tpw.tokens_per_watt,
                "by_category":     {k: round(statistics.mean(v), 2) for k, v in by_cat.items()},
                "by_recommendation": {k: len(v) for k, v in recs.items()},
                "remove_candidates": recs.get("REMOVE", []),
                "keep_stars":        recs.get("KEEP★", []),
                "keep_list":         recs.get("KEEP", []),
                "review_list":       recs.get("REVIEW", []),
            })

        return stats

    def summary_table(self) -> List[Dict[str, Any]]:
        ok = self.get_ok_results()
        rows = []
        for r in sorted(ok, key=lambda x: x.efficiency_score, reverse=True):
            rows.append({
                "Model":       r.model,
                "Cat":         r.category,
                "TPS":         f"{r.tokens_per_second:.1f}",
                "TPS±":        f"±{r.tps_stdev:.1f}",
                "Stab":        r.stability_label,
                "Lat(s)":      f"{r.api_latency_s:.1f}",
                "RAM(MB)":     f"{r.hw.ram_used:.0f}",
                "GPU°C":       f"{r.hw.temp_gpu:.0f}",
                "Pwr(W)":      f"{r.hw.power_watts:.1f}",
                "tok/W":       f"{r.tokens_per_watt:.2f}",
                "Pct":         f"{r.category_percentile:.0f}%",
                "Score":       f"{r.efficiency_score:.0f}",
                "Rec":         r.recommendation,
            })
        return rows
