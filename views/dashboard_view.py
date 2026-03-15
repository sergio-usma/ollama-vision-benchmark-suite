#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
views/dashboard_view.py — HTML Dashboard v3.0
Cyberpunk theme — all charts guaranteed, category comparison,
multi-run stability box-plots, decision matrix, no empty sections.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from config import (
    APP_AUTHOR, APP_NAME, APP_VERSION,
    CATEGORY_COLORS, DASHBOARD_FILE,
    RECOMMENDATION_COLORS,
)

# ─── CYBERPUNK THEME ──────────────────────────────────────────────────────────
_BG   = "#080c14"
_BG2  = "#0a0e1a"
_BG3  = "#0d1220"
_NEON = "#39ff14"
_CYAN = "#00f5d4"
_PINK = "#f72585"
_AMBR = "#ffb700"
_BLUE = "#4cc9f0"
_PURP = "#7209b7"
_ORG  = "#f8961e"
_TEXT = "#c0ccd0"
_DIM  = "#4a6070"
_GRID = "#1a2233"


def _apply_theme(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(family="IBM Plex Mono", color=_NEON, size=12),
                   pad=dict(t=8)),
        height=height,
        plot_bgcolor=_BG, paper_bgcolor=_BG2,
        font=dict(family="IBM Plex Mono, monospace", color=_TEXT, size=10),
        legend=dict(bgcolor=_BG, bordercolor=_GRID, borderwidth=1,
                    font=dict(color=_TEXT, size=9)),
        margin=dict(l=60, r=24, t=52, b=70),
    )
    fig.update_xaxes(gridcolor=_GRID, linecolor=_DIM,
                     tickfont=dict(color=_DIM, size=9), showgrid=True)
    fig.update_yaxes(gridcolor=_GRID, linecolor=_DIM,
                     tickfont=dict(color=_DIM, size=9), showgrid=True)
    return fig


def _fig_to_div(fig: go.Figure, div_id: str) -> str:
    return pio.to_html(
        fig, include_plotlyjs=False, full_html=False, div_id=div_id,
        config={"responsive": True, "displayModeBar": True, "displaylogo": False,
                "modeBarButtonsToRemove": ["toImage", "select2d", "lasso2d"]},
    )


def _empty_div(msg: str = "Insufficient data") -> str:
    return (f'<div style="display:flex;align-items:center;justify-content:center;'
            f'height:300px;color:{_DIM};font-family:IBM Plex Mono;'
            f'font-size:.75rem;border:1px solid {_GRID}">{msg}</div>')


# ─── SHORT MODEL NAME ──────────────────────────────────────────────────────────
def _short(name: str, n: int = 22) -> str:
    return name.split(":")[0][-n:] if len(name.split(":")[0]) > n else name.split(":")[0]


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD VIEW
# ═══════════════════════════════════════════════════════════════════════════════

class DashboardView:
    def __init__(self, output_path: Path = DASHBOARD_FILE):
        self.output_path = Path(output_path)
        self.df_all: pd.DataFrame = pd.DataFrame()
        self.df_gen: pd.DataFrame = pd.DataFrame()
        self.df_emb: pd.DataFrame = pd.DataFrame()
        self._generated_at: str = ""

    # ─── DATA LOADING ──────────────────────────────────────────────────────────

    def load(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        self.df_all = df.copy()

        # Normalize numeric columns
        num_cols = [
            "Tokens_per_second", "TPS_median", "TPS_stdev", "TPS_min", "TPS_max", "TPS_cv",
            "Power_Total_mW", "RAM_used_MB", "RAM_percent", "Temp_GPU_C", "Temp_CPU_C",
            "Temp_SOC_C", "Temp_TJ_C", "GPU_load_%", "GPU_freq_MHz", "Tokens_per_W",
            "Efficiency_score", "Category_pct", "API_latency_s", "Load_duration_s",
            "VDD_GPU_SOC_mW", "VDD_CPU_CV_mW", "VIN_SYS_5V0_mW", "Total_duration_s",
            "Num_runs_ok", "Num_runs_total",
        ]
        for col in num_cols:
            if col in self.df_all.columns:
                self.df_all[col] = pd.to_numeric(self.df_all[col], errors="coerce").fillna(0)

        # Ensure essential columns exist
        if "Category" not in self.df_all.columns:
            self.df_all["Category"] = "GENERAL"
        if "Recommendation" not in self.df_all.columns:
            self.df_all["Recommendation"] = "REVIEW"

        # Split generation vs embeddings
        tps_col = "Tokens_per_second"
        if tps_col in self.df_all.columns:
            self.df_gen = self.df_all[self.df_all[tps_col] > 0].copy()
            self.df_emb = self.df_all[self.df_all[tps_col] == 0].copy()
        else:
            self.df_gen = self.df_all.copy()
            self.df_emb = pd.DataFrame()

        # Add short name column
        if "Model" in self.df_gen.columns:
            self.df_gen["Model_short"] = self.df_gen["Model"].apply(_short)

        # Calculate Tokens_per_W if missing or zero
        if "Tokens_per_W" not in self.df_gen.columns or (self.df_gen.get("Tokens_per_W", pd.Series([0])) == 0).all():
            if "Power_Total_mW" in self.df_gen.columns:
                self.df_gen["Tokens_per_W"] = (
                    self.df_gen["Tokens_per_second"] /
                    (self.df_gen["Power_Total_mW"].clip(lower=100) / 1000.0)
                )

    # ─── GENERATE ──────────────────────────────────────────────────────────────

    def generate(self) -> Path:
        self._generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        charts  = self._build_all_charts()
        html    = self._build_full_html(charts)
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(html)
        return self.output_path

    # ═══ CHARTS ═══════════════════════════════════════════════════════════════

    def _build_all_charts(self) -> Dict[str, str]:
        c: Dict[str, str] = {}
        g = self.df_gen

        if g.empty:
            empty = _empty_div()
            for k in ["tps", "escore", "tpw", "scatter", "temp", "pwr", "gpu", "ram",
                       "latency", "radar", "rec_pie", "cat_compare", "stability",
                       "decision_matrix", "multi_run_box", "load_vs_infer"]:
                c[k] = empty
            return c

        c["tps"]            = self._chart_tps_bar()
        c["escore"]         = self._chart_efficiency_score()
        c["tpw"]            = self._chart_tokens_per_watt()
        c["scatter"]        = self._chart_tps_vs_ram_scatter()
        c["temp"]           = self._chart_temperatures()
        c["pwr"]            = self._chart_power_rails()
        c["gpu"]            = self._chart_gpu_load_vs_freq()
        c["ram"]            = self._chart_ram_usage()
        c["latency"]        = self._chart_latency_breakdown()
        c["radar"]          = self._chart_radar_by_category()
        c["rec_pie"]        = self._chart_recommendation_pie()
        c["cat_compare"]    = self._chart_category_comparison()
        c["stability"]      = self._chart_stability_cv()
        c["decision_matrix"]= self._chart_decision_matrix()
        c["multi_run_box"]  = self._chart_multi_run_boxplot()
        c["load_vs_infer"]  = self._chart_load_vs_inference()
        return c

    # ── Velocidad ───────────────────────────────────────────────────────────

    def _chart_tps_bar(self) -> str:
        g = self.df_gen.sort_values("Tokens_per_second", ascending=False)
        col_name = "Model_short" if "Model_short" in g.columns else "Model"

        fig = go.Figure()
        for cat in g["Category"].unique():
            sub = g[g["Category"] == cat]
            fig.add_trace(go.Bar(
                x=sub[col_name],
                y=sub["Tokens_per_second"],
                name=cat,
                marker_color=CATEGORY_COLORS.get(cat, _CYAN),
                text=[f"{v:.1f}" for v in sub["Tokens_per_second"]],
                textposition="outside",
                textfont=dict(color=_TEXT, size=9),
                # Error bars if multi-run stdev available
                error_y=dict(
                    type="data",
                    array=sub["TPS_stdev"].tolist() if "TPS_stdev" in sub.columns else [],
                    visible="TPS_stdev" in sub.columns,
                    color=_DIM, thickness=1.5, width=3,
                ),
            ))

        fig.update_layout(barmode="group", xaxis_tickangle=-50, showlegend=True)
        fig.add_hline(y=10, line_dash="dot", line_color=_PINK, opacity=0.6,
                      annotation_text="Recommended min (10 tok/s)",
                      annotation_font=dict(color=_PINK, size=9))
        return _fig_to_div(_apply_theme(fig, "⚡ TOKENS / SECOND — error bars = ±1σ inter-run"), "c_tps")

    def _chart_efficiency_score(self) -> str:
        g = self.df_gen.sort_values("Efficiency_score", ascending=True).tail(35)
        col_name = "Model_short" if "Model_short" in g.columns else "Model"

        colors = g["Efficiency_score"].apply(
            lambda s: _NEON if s >= 68 else (_CYAN if s >= 42 else (_AMBR if s >= 22 else _PINK))
        )
        fig = go.Figure(go.Bar(
            x=g["Efficiency_score"], y=g[col_name], orientation="h",
            marker=dict(
                color=g["Efficiency_score"].tolist(),
                colorscale=[[0, "#1a0a14"], [0.25, _PINK], [0.5, _AMBR], [0.75, _CYAN], [1.0, _NEON]],
                cmin=0, cmax=100, showscale=True,
                colorbar=dict(title=dict(text="Score", font=dict(color=_DIM, size=9)),
                              tickfont=dict(color=_DIM, size=8), bgcolor=_BG2, outlinecolor=_GRID,
                              x=1.02),
            ),
            text=[f"{s:.0f}  {r}" for s, r in zip(g["Efficiency_score"],
                  g.get("Recommendation", [""] * len(g)))],
            textposition="outside", textfont=dict(color=_TEXT, size=9),
        ))
        for thresh, color, label in [(68, _NEON, "KEEP★"), (42, _CYAN, "KEEP"), (22, _AMBR, "OPTIONAL")]:
            fig.add_vline(x=thresh, line_dash="dot", line_color=color, opacity=0.55,
                          annotation_text=label, annotation_font=dict(color=color, size=9))
        fig.update_layout(xaxis_range=[0, 115])
        return _fig_to_div(_apply_theme(fig, "🎯 EFFICIENCY SCORE — speed×percentile + energy − thermal/RAM − instability", 520), "c_escore")

    def _chart_tokens_per_watt(self) -> str:
        g = self.df_gen.sort_values("Tokens_per_W", ascending=False)
        col_name = "Model_short" if "Model_short" in g.columns else "Model"
        fig = px.bar(g, x=col_name, y="Tokens_per_W", color="Category",
                     color_discrete_map=CATEGORY_COLORS, text_auto=".2f")
        fig.update_traces(textfont=dict(color=_BG, size=9), marker_line_width=0)
        fig.update_layout(xaxis_tickangle=-50)
        fig.add_hline(y=2.0, line_dash="dot", line_color=_CYAN, opacity=0.5,
                      annotation_text="Reference 2 tok/W",
                      annotation_font=dict(color=_CYAN, size=9))
        return _fig_to_div(_apply_theme(fig, "🔋 ENERGY EFFICIENCY — tokens per watt (higher = better)"), "c_tpw")

    def _chart_tps_vs_ram_scatter(self) -> str:
        g = self.df_gen.copy()
        col_name = "Model_short" if "Model_short" in g.columns else "Model"
        pwr_col = "Power_Total_mW"
        g["_pwr_norm"] = (g[pwr_col].clip(lower=1000) / 1000.0).clip(upper=35) if pwr_col in g.columns else 10

        fig = px.scatter(
            g, x="RAM_used_MB", y="Tokens_per_second",
            color="Category", size="_pwr_norm",
            hover_name="Model",
            hover_data={
                "Efficiency_score": True, "Recommendation": True,
                "Tokens_per_W": ":.2f", "_pwr_norm": False,
                "RAM_used_MB": ":.0f", "Tokens_per_second": ":.1f",
            },
            color_discrete_map=CATEGORY_COLORS,
            text=col_name if len(g) <= 20 else None,
        )
        fig.update_traces(
            marker=dict(opacity=0.85, line=dict(width=1, color=_BG)),
            textfont=dict(color=_TEXT, size=8),
            textposition="top center",
        )
        fig.update_layout(xaxis_title="RAM used (MB)", yaxis_title="TPS (tok/s)")
        return _fig_to_div(_apply_theme(fig, "📊 TPS vs RAM — point size = total power"), "c_scatter")

    # ── Térmica / Potencia ──────────────────────────────────────────────────

    def _chart_temperatures(self) -> str:
        g = self.df_gen
        col_name = "Model_short" if "Model_short" in g.columns else "Model"
        temp_cols = {"Temp_GPU_C": "GPU", "Temp_CPU_C": "CPU", "Temp_SOC_C": "SOC", "Temp_TJ_C": "TJ"}
        avail     = {v: k for k, v in temp_cols.items() if k in g.columns}

        if not avail:
            return _empty_div("No temperature data")

        fig = go.Figure()
        colors_t = {"GPU": _PINK, "CPU": _AMBR, "SOC": _BLUE, "TJ": _NEON}
        g_sorted = g.sort_values("Temp_GPU_C" if "Temp_GPU_C" in g.columns else list(avail.values())[0], ascending=False)

        for label, col in avail.items():
            fig.add_trace(go.Bar(
                name=label, x=g_sorted[col_name], y=g_sorted[col],
                marker_color=colors_t.get(label, _CYAN),
                text=[f"{v:.0f}" for v in g_sorted[col]],
                textposition="outside", textfont=dict(size=8, color=_TEXT),
            ))

        for thresh, color, lbl in [(75, _PINK, "⚠ 75°C"), (85, "#ff2020", "🔴 85°C")]:
            fig.add_hline(y=thresh, line_dash="dot", line_color=color,
                          annotation_text=lbl, annotation_font=dict(color=color, size=9))
        fig.update_layout(barmode="group", xaxis_tickangle=-50)
        return _fig_to_div(_apply_theme(fig, "🌡 TEMPERATURES — per sensor during inference (°C)"), "c_temp")

    def _chart_power_rails(self) -> str:
        g = self.df_gen
        col_name = "Model_short" if "Model_short" in g.columns else "Model"
        rails    = {"VDD_GPU_SOC_mW": "GPU+SOC", "VDD_CPU_CV_mW": "CPU+CV", "VIN_SYS_5V0_mW": "SYS5V0"}
        avail    = {v: k for k, v in rails.items() if k in g.columns}

        if not avail:
            # Fallback: only Power_Total_mW
            if "Power_Total_mW" not in g.columns:
                return _empty_div("No power data")
            g_s = g.sort_values("Power_Total_mW", ascending=False)
            fig = px.bar(g_s, x=col_name, y="Power_Total_mW", color="Category",
                         color_discrete_map=CATEGORY_COLORS, text_auto=".0f")
            fig.update_layout(xaxis_tickangle=-50)
            return _fig_to_div(_apply_theme(fig, "⚡ TOTAL POWER (mW)"), "c_pwr")

        g_s = g.sort_values("Power_Total_mW", ascending=False)
        fig = go.Figure()
        rail_colors = {"GPU+SOC": _PINK, "CPU+CV": _AMBR, "SYS5V0": _BLUE}
        for label, col in avail.items():
            fig.add_trace(go.Bar(
                name=label, x=g_s[col_name], y=g_s[col],
                marker_color=rail_colors.get(label, _CYAN),
            ))
        fig.update_layout(barmode="stack", xaxis_tickangle=-50)
        return _fig_to_div(_apply_theme(fig, "⚡ POWER BY RAIL — distribution (mW)"), "c_pwr")

    def _chart_gpu_load_vs_freq(self) -> str:
        g = self.df_gen.sort_values("Tokens_per_second", ascending=False)
        col_name = "Model_short" if "Model_short" in g.columns else "Model"

        if "GPU_load_%" not in g.columns:
            return _empty_div("No GPU data")

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=g[col_name], y=g["GPU_load_%"], name="GPU Load (%)",
                   marker_color=_PURP, opacity=0.85,
                   text=[f"{v:.0f}%" for v in g["GPU_load_%"]],
                   textposition="outside", textfont=dict(size=8, color=_TEXT)),
            secondary_y=False,
        )
        if "GPU_freq_MHz" in g.columns:
            fig.add_trace(
                go.Scatter(x=g[col_name], y=g["GPU_freq_MHz"], mode="lines+markers",
                           name="GPU Freq (MHz)", line=dict(color=_NEON, width=2),
                           marker=dict(size=5, color=_NEON)),
                secondary_y=True,
            )
        fig.update_yaxes(title_text="GPU Load (%)", gridcolor=_GRID,
                         tickfont=dict(color=_DIM, size=9), secondary_y=False)
        fig.update_yaxes(title_text="Frequency (MHz)", tickfont=dict(color=_DIM, size=9),
                         secondary_y=True)
        fig.update_xaxes(tickangle=-50)
        _apply_theme(fig, "🎮 GPU LOAD vs GPU FREQUENCY")
        return _fig_to_div(fig, "c_gpu")

    def _chart_ram_usage(self) -> str:
        g = self.df_gen.sort_values("RAM_used_MB", ascending=False)
        col_name = "Model_short" if "Model_short" in g.columns else "Model"
        gb = g["RAM_used_MB"] / 1024.0

        fig = go.Figure(go.Bar(
            x=g[col_name], y=gb,
            marker=dict(
                color=g["RAM_used_MB"].tolist(),
                colorscale=[[0, _BG3], [0.4, _PURP], [0.7, _PINK], [1.0, "#ff1010"]],
                cmin=0, cmax=32768, showscale=True,
                colorbar=dict(title=dict(text="MB", font=dict(color=_DIM, size=9)),
                              tickfont=dict(color=_DIM, size=8), bgcolor=_BG2, outlinecolor=_GRID),
            ),
            text=[f"{v:.1f}GB" for v in gb],
            textposition="outside", textfont=dict(color=_TEXT, size=9),
        ))
        fig.add_hline(y=28, line_dash="dot", line_color=_PINK, opacity=0.6,
                      annotation_text="Recommended limit (28GB)",
                      annotation_font=dict(color=_PINK, size=9))
        fig.update_layout(xaxis_tickangle=-50, yaxis_title="GB RAM")
        return _fig_to_div(_apply_theme(fig, "💾 RAM USAGE — during inference"), "c_ram")

    def _chart_latency_breakdown(self) -> str:
        g = self.df_gen.sort_values("API_latency_s", ascending=False)
        col_name = "Model_short" if "Model_short" in g.columns else "Model"
        fig = go.Figure()

        if "Load_duration_s" in g.columns and "Total_duration_s" in g.columns:
            infer_s = (g["Total_duration_s"] - g["Load_duration_s"]).clip(lower=0)
            fig.add_trace(go.Bar(name="Model load (s)", x=g[col_name], y=g["Load_duration_s"],
                                 marker_color=_PURP, text=[f"{v:.1f}" for v in g["Load_duration_s"]],
                                 textposition="inside", textfont=dict(size=8, color=_TEXT)))
            fig.add_trace(go.Bar(name="Inference (s)", x=g[col_name], y=infer_s,
                                 marker_color=_NEON, text=[f"{v:.1f}" for v in infer_s],
                                 textposition="inside", textfont=dict(size=8, color=_BG)))
            fig.update_layout(barmode="stack", xaxis_tickangle=-50)
        else:
            fig.add_trace(go.Bar(x=g[col_name], y=g["API_latency_s"], marker_color=_CYAN,
                                 text=[f"{v:.1f}s" for v in g["API_latency_s"]],
                                 textposition="outside", textfont=dict(color=_TEXT, size=9)))
            fig.update_layout(xaxis_tickangle=-50)

        return _fig_to_div(_apply_theme(fig, "⏱ LATENCY — breakdown: model load vs inference (seconds)"), "c_latency")

    # ── Comparativa por categoría (NUEVO) ──────────────────────────────────

    def _chart_category_comparison(self) -> str:
        """
        Grouped bar chart comparing key metrics by CATEGORY.
        Category-level decision chart.
        """
        g = self.df_gen
        if "Category" not in g.columns or g.empty:
            return _empty_div("No category data")

        cat_stats = g.groupby("Category").agg(
            n=("Model", "count"),
            avg_tps=("Tokens_per_second", "mean"),
            max_tps=("Tokens_per_second", "max"),
            avg_score=("Efficiency_score", "mean"),
            avg_tpw=("Tokens_per_W", "mean"),
            avg_temp=("Temp_GPU_C", "mean"),
            avg_pwr=("Power_Total_mW", "mean"),
        ).reset_index().sort_values("avg_tps", ascending=False)

        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                "Average TPS by category", "Max TPS achieved",
                "Average Efficiency Score", "Tokens per Watt (tok/W)",
                "Average GPU Temperature (°C)", "Average Power (W)",
            ],
            vertical_spacing=0.18, horizontal_spacing=0.10,
        )

        def _bar(row, col, y_col, label, color_list, fmt=".1f"):
            fig.add_trace(go.Bar(
                x=cat_stats["Category"], y=cat_stats[y_col],
                name=label, showlegend=False,
                marker=dict(color=color_list, line=dict(width=0)),
                text=[f"{v:{fmt}}" for v in cat_stats[y_col]],
                textposition="outside", textfont=dict(size=9, color=_TEXT),
            ), row=row, col=col)

        cat_colors = [CATEGORY_COLORS.get(c, _CYAN) for c in cat_stats["Category"]]

        _bar(1, 1, "avg_tps",   "Avg TPS",     cat_colors, ".1f")
        _bar(1, 2, "max_tps",   "Max TPS",     cat_colors, ".1f")
        _bar(1, 3, "avg_score", "Score",        cat_colors, ".0f")
        _bar(2, 1, "avg_tpw",   "tok/W",        cat_colors, ".2f")

        # Temperature — inverted (lower is better): red if high
        temp_colors = [_NEON if t < 60 else (_AMBR if t < 75 else _PINK)
                       for t in cat_stats["avg_temp"]]
        fig.add_trace(go.Bar(
            x=cat_stats["Category"], y=cat_stats["avg_temp"],
            name="Temp", showlegend=False,
            marker=dict(color=temp_colors),
            text=[f"{v:.0f}°C" for v in cat_stats["avg_temp"]],
            textposition="outside", textfont=dict(size=9, color=_TEXT),
        ), row=2, col=2)

        # Power in W
        pwr_w = cat_stats["avg_pwr"] / 1000.0
        pwr_colors = [_NEON if p < 12 else (_AMBR if p < 20 else _PINK) for p in pwr_w]
        fig.add_trace(go.Bar(
            x=cat_stats["Category"], y=pwr_w,
            name="Pwr", showlegend=False,
            marker=dict(color=pwr_colors),
            text=[f"{v:.1f}W" for v in pwr_w],
            textposition="outside", textfont=dict(size=9, color=_TEXT),
        ), row=2, col=3)

        fig.update_layout(
            height=560, plot_bgcolor=_BG, paper_bgcolor=_BG2,
            font=dict(family="IBM Plex Mono", color=_TEXT, size=9),
            title=dict(text="📊 CATEGORY COMPARISON — global view for decisions",
                       font=dict(color=_NEON, size=12)),
            margin=dict(l=40, r=40, t=70, b=40),
        )
        for ax in fig.layout:
            if ax.startswith("xaxis") or ax.startswith("yaxis"):
                fig.layout[ax].update(gridcolor=_GRID, linecolor=_DIM,
                                      tickfont=dict(color=_DIM, size=8))
        for ann in fig.layout.annotations:
            ann.font.update(color=_CYAN, size=10)

        return _fig_to_div(fig, "c_cat_compare")

    def _chart_radar_by_category(self) -> str:
        g = self.df_gen
        if g.empty or "Category" not in g.columns:
            return _empty_div()

        cats = g.groupby("Category").agg({
            "Tokens_per_second": "mean",
            "Efficiency_score":  "mean",
            "Tokens_per_W":      "mean",
            "GPU_load_%":        "mean",
            "Temp_GPU_C":        "mean",
        }).reset_index()

        # Normalizar 0–100
        for col in ["Tokens_per_second", "Efficiency_score", "Tokens_per_W", "GPU_load_%"]:
            mx = cats[col].max()
            if mx > 0:
                cats[col] = (cats[col] / mx * 100).clip(0, 100)

        # Temperatura invertida
        mx_t = cats["Temp_GPU_C"].max()
        cats["Thermal_score"] = ((1 - cats["Temp_GPU_C"] / max(mx_t, 1)) * 100).clip(0, 100)

        dims   = ["Tokens_per_second", "Efficiency_score", "Tokens_per_W", "Thermal_score", "GPU_load_%"]
        labels = ["Speed", "Efficiency", "Eco (tok/W)", "Thermal", "GPU%"]

        fig = go.Figure()
        for _, row in cats.iterrows():
            cat   = row["Category"]
            color = CATEGORY_COLORS.get(cat, _CYAN)
            vals  = [float(row.get(d, 0)) for d in dims]
            vals  += [vals[0]]
            lbls  = labels + [labels[0]]
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=lbls, name=cat, fill="toself",
                opacity=0.55, line=dict(width=2, color=color), fillcolor=color,
            ))

        fig.update_layout(
            polar=dict(
                bgcolor=_BG,
                radialaxis=dict(visible=True, range=[0, 100], gridcolor=_GRID,
                                tickfont=dict(color=_DIM, size=8)),
                angularaxis=dict(gridcolor=_GRID, tickfont=dict(color=_TEXT, size=10)),
            ),
            legend=dict(bgcolor=_BG, bordercolor=_GRID, font=dict(color=_TEXT)),
        )
        return _fig_to_div(_apply_theme(fig, "🕸 RADAR — multidimensional comparison by category", 450), "c_radar")

    def _chart_recommendation_pie(self) -> str:
        if "Recommendation" not in self.df_all.columns or self.df_all.empty:
            return _empty_div("No recommendation data")

        counts = self.df_all["Recommendation"].value_counts().reset_index()
        counts.columns = ["Recommendation", "Count"]

        fig = go.Figure(go.Pie(
            labels=counts["Recommendation"],
            values=counts["Count"],
            marker=dict(
                colors=[RECOMMENDATION_COLORS.get(r, "#888") for r in counts["Recommendation"]],
                line=dict(color=_BG, width=2),
            ),
            hole=0.55,
            textfont=dict(family="IBM Plex Mono", size=10),
            hovertemplate="<b>%{label}</b><br>%{value} models<br>%{percent}<extra></extra>",
        ))
        fig.update_layout(showlegend=True,
                          legend=dict(bgcolor=_BG, font=dict(color=_TEXT, size=9)))
        return _fig_to_div(_apply_theme(fig, "📋 RECOMMENDATION DISTRIBUTION", 380), "c_rec_pie")

    def _chart_stability_cv(self) -> str:
        """Inter-run coefficient of variation — measures result reliability."""
        if "TPS_cv" not in self.df_gen.columns or (self.df_gen["TPS_cv"] == 0).all():
            return _empty_div("No multi-run data (TPS_cv=0 — run with NUM_RUNS≥2)")

        g = self.df_gen[self.df_gen["TPS_cv"] > 0].sort_values("TPS_cv", ascending=True)
        col_name = "Model_short" if "Model_short" in g.columns else "Model"

        cv_colors = [_NEON if v <= 0.03 else (_CYAN if v <= 0.08 else (_AMBR if v <= 0.15 else _PINK))
                     for v in g["TPS_cv"]]

        fig = go.Figure(go.Bar(
            x=g[col_name], y=g["TPS_cv"] * 100,
            marker=dict(color=cv_colors, line=dict(width=0)),
            text=[f"{v*100:.1f}%" for v in g["TPS_cv"]],
            textposition="outside", textfont=dict(color=_TEXT, size=9),
        ))

        for thresh, color, lbl in [(3, _NEON, "STABLE ≤3%"), (8, _CYAN, "GOOD ≤8%"),
                                    (15, _AMBR, "VARIABLE ≤15%")]:
            fig.add_hline(y=thresh, line_dash="dot", line_color=color, opacity=0.55,
                          annotation_text=lbl, annotation_font=dict(color=color, size=9))

        fig.update_layout(xaxis_tickangle=-50, yaxis_title="CV (%)")
        return _fig_to_div(_apply_theme(fig, "📈 INTER-RUN STABILITY — Coefficient of Variation (lower = more stable)"), "c_stability")

    def _chart_decision_matrix(self) -> str:
        """
        Scatter TPS vs Efficiency Score, colored by recommendation.
        Ideal quadrant: high TPS + high score.
        """
        g = self.df_gen
        col_name = "Model_short" if "Model_short" in g.columns else "Model"

        fig = go.Figure()
        for rec in ["KEEP★", "KEEP", "OPTIONAL", "REVIEW", "REMOVE", "ERROR"]:
            sub = g[g.get("Recommendation", pd.Series([""] * len(g))) == rec] if "Recommendation" in g.columns else pd.DataFrame()
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["Tokens_per_second"],
                y=sub["Efficiency_score"],
                mode="markers+text",
                name=rec,
                text=sub[col_name],
                textposition="top center",
                textfont=dict(size=8, color=_TEXT),
                marker=dict(
                    color=RECOMMENDATION_COLORS.get(rec, "#888"),
                    size=12, opacity=0.85,
                    line=dict(width=1, color=_BG),
                    symbol="circle",
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "TPS: %{x:.1f}<br>"
                    "Score: %{y:.0f}<br>"
                    f"Rec: {rec}<extra></extra>"
                ),
            ))

        # Threshold lines
        if not g.empty:
            fig.add_hline(y=42, line_dash="dot", line_color=_CYAN, opacity=0.5,
                          annotation_text="Score KEEP", annotation_font=dict(color=_CYAN, size=9))
            fig.add_vline(x=10, line_dash="dot", line_color=_PINK, opacity=0.5,
                          annotation_text="TPS min", annotation_font=dict(color=_PINK, size=9))

        fig.update_layout(xaxis_title="TPS (tok/s)", yaxis_title="Efficiency Score")
        return _fig_to_div(_apply_theme(fig, "🎯 DECISION MATRIX — TPS vs Score (top-right = ideal)", 500), "c_decision")

    def _chart_multi_run_boxplot(self) -> str:
        """Box plot of TPS per model using individual run data."""
        g = self.df_gen
        if "TPS_runs" not in g.columns:
            return _empty_div("No multi-run data (requires benchmark with NUM_RUNS≥2)")

        # Unpack TPS_runs
        rows_expanded = []
        for _, row in g.iterrows():
            tps_runs_str = str(row.get("TPS_runs", ""))
            if not tps_runs_str or tps_runs_str == "0":
                continue
            try:
                tps_list = [float(x) for x in tps_runs_str.split("|") if x]
                for tps in tps_list:
                    rows_expanded.append({
                        "Model": _short(str(row.get("Model", "")), 20),
                        "Category": row.get("Category", "?"),
                        "TPS": tps,
                    })
            except Exception:
                continue

        if not rows_expanded:
            return _empty_div("No individual run data")

        df_exp = pd.DataFrame(rows_expanded)
        df_exp = df_exp.sort_values("Model")

        fig = go.Figure()
        for cat in df_exp["Category"].unique():
            sub = df_exp[df_exp["Category"] == cat]
            cat_color = CATEGORY_COLORS.get(cat, _CYAN)
            fig.add_trace(go.Box(
                y=sub["TPS"], x=sub["Model"],
                name=cat, boxpoints="all", jitter=0.4, pointpos=0,
                marker=dict(size=6, color=cat_color, opacity=0.7),
                line=dict(color=cat_color, width=2),
            ))

        fig.update_layout(boxmode="group", xaxis_tickangle=-50)
        return _fig_to_div(_apply_theme(fig, "📦 MULTI-RUN BOX-PLOT — TPS distribution per run (higher consistency = smaller box)", 450), "c_boxplot")

    def _chart_load_vs_inference(self) -> str:
        """
        Scatter: load time vs inference time.
        Models with high relative load time have insufficient VRAM available.
        """
        g = self.df_gen
        if "Load_duration_s" not in g.columns or "Total_duration_s" not in g.columns:
            return _empty_div("No detailed duration data")

        col_name = "Model_short" if "Model_short" in g.columns else "Model"
        g = g.copy()
        g["infer_s"] = (g["Total_duration_s"] - g["Load_duration_s"]).clip(lower=0)
        g["load_ratio"] = g["Load_duration_s"] / g["Total_duration_s"].clip(lower=0.1)

        fig = px.scatter(
            g, x="Load_duration_s", y="infer_s",
            color="Category", size="RAM_used_MB",
            hover_name="Model",
            hover_data={"load_ratio": ":.0%", "Tokens_per_second": ":.1f"},
            color_discrete_map=CATEGORY_COLORS,
            text=col_name if len(g) <= 15 else None,
        )
        fig.update_traces(
            marker=dict(opacity=0.8, line=dict(width=1, color=_BG)),
            textfont=dict(color=_TEXT, size=8),
        )
        fig.update_layout(xaxis_title="Load time (s)", yaxis_title="Inference time (s)")
        return _fig_to_div(_apply_theme(fig, "⏳ LOAD vs INFERENCE — size = RAM used (high load = model doesn't fit in VRAM)"), "c_load_infer")

    # ═══ TABLAS ════════════════════════════════════════════════════════════════

    def _kpi_cards_html(self) -> str:
        g = self.df_gen
        if g.empty:
            return "<p style='color:#4a6070;padding:20px'>No generation data</p>"

        def _safe_idxmax(col):
            if col not in g.columns or g[col].isna().all():
                return None
            return g.loc[g[col].idxmax()]

        best_tps  = _safe_idxmax("Tokens_per_second")
        best_eff  = _safe_idxmax("Efficiency_score")
        best_tpw  = _safe_idxmax("Tokens_per_W")

        n_keep    = int(g["Recommendation"].str.startswith("KEEP").sum()) if "Recommendation" in g.columns else 0
        n_remove  = int((g["Recommendation"] == "REMOVE").sum()) if "Recommendation" in g.columns else 0
        n_review  = int((g["Recommendation"] == "REVIEW").sum()) if "Recommendation" in g.columns else 0
        avg_pwr   = g["Power_Total_mW"].mean() / 1000.0 if "Power_Total_mW" in g.columns else 0
        avg_temp  = g["Temp_GPU_C"].mean() if "Temp_GPU_C" in g.columns else 0

        def _card(icon, title, val, sub, color):
            return (
                f'<div class="kpi-card" style="border-left-color:{color}">'
                f'<div class="kpi-icon">{icon}</div>'
                f'<div class="kpi-title">{title}</div>'
                f'<div class="kpi-value" style="color:{color}">{val}</div>'
                f'<div class="kpi-sub">{sub}</div>'
                f'</div>'
            )

        cards = [
            _card("⚡", "FASTEST",
                  _short(best_tps["Model"]) if best_tps is not None else "?",
                  f"{best_tps['Tokens_per_second']:.1f} tok/s" if best_tps is not None else "?",
                  _NEON),
            _card("🎯", "BEST SCORE",
                  _short(best_eff["Model"]) if best_eff is not None else "?",
                  f"Score {best_eff['Efficiency_score']:.0f}/100" if best_eff is not None else "?",
                  _CYAN),
            _card("🔋", "MOST EFFICIENT",
                  _short(best_tpw["Model"]) if best_tpw is not None else "?",
                  f"{best_tpw['Tokens_per_W']:.2f} tok/W" if best_tpw is not None else "?",
                  _AMBR),
            _card("⚡", "AVG POWER",
                  f"{avg_pwr:.1f} W",
                  f"{len(g)} models benchmarked", _ORG),
            _card("🌡", "AVG GPU TEMP",
                  f"{avg_temp:.1f}°C",
                  "during inference", _PINK),
            _card("✅", "KEEP / REMOVE",
                  f"{n_keep} keep",
                  f"{n_review} review | {n_remove} remove", _PURP),
        ]

        return f'<div class="kpi-row">{"".join(cards)}</div>'

    def _table_html(self, df: pd.DataFrame, cols: List[str], table_id: str, warn_col: str = "") -> str:
        if df.empty:
            return "<p class='no-data'>No data</p>"
        avail = [c for c in cols if c in df.columns]
        if not avail:
            return "<p class='no-data'>No columns available</p>"
        sub = df[avail].copy()
        # Formatear floats
        for col in sub.select_dtypes(include="float").columns:
            sub[col] = sub[col].apply(lambda x: f"{x:.2f}" if abs(x) < 10000 else f"{x:.0f}")
        return sub.to_html(index=False, classes="data-table", border=0,
                           table_id=table_id, na_rep="—")

    def _table_top_per_category(self) -> str:
        if self.df_gen.empty:
            return "<p class='no-data'>No data</p>"
        cols = ["Category", "Model", "Tokens_per_second", "TPS_median", "TPS_stdev",
                "Tokens_per_W", "Efficiency_score", "Category_pct",
                "RAM_used_MB", "Power_Total_mW", "Recommendation", "Decision_reasons"]
        top = self.df_gen.loc[
            self.df_gen.groupby("Category")["Efficiency_score"].idxmax()
        ].sort_values("Efficiency_score", ascending=False)
        return self._table_html(top, cols, "tbl_top_cat")

    def _table_full_results(self) -> str:
        cols = ["Model", "Category", "Tokens_per_second", "TPS_median", "TPS_stdev",
                "TPS_cv", "Stability", "Tokens_per_W", "API_latency_s", "Load_duration_s",
                "RAM_used_MB", "GPU_load_%", "Temp_GPU_C", "Power_Total_mW",
                "Efficiency_score", "Category_pct", "Recommendation", "Decision_reasons"]
        df_s = self.df_gen.sort_values("Efficiency_score", ascending=False)
        return self._table_html(df_s, cols, "tbl_all")

    def _table_remove_candidates(self) -> str:
        if "Recommendation" not in self.df_gen.columns:
            return "<p class='no-data'>No data</p>"
        cand = self.df_gen[self.df_gen["Recommendation"].isin(["REMOVE", "REVIEW"])].sort_values("Tokens_per_second")
        if cand.empty:
            return "<p class='ok-msg'>✅ No models candidates for removal or review</p>"
        cols = ["Model", "Category", "Tokens_per_second", "RAM_used_MB",
                "Efficiency_score", "Recommendation", "Decision_reasons"]
        return self._table_html(cand, cols, "tbl_remove")

    def _table_embeddings(self) -> str:
        if self.df_emb.empty:
            return "<p class='no-data'>No embedding models</p>"
        cols = ["Model", "Category", "API_latency_s", "RAM_used_MB", "Recommendation"]
        return self._table_html(self.df_emb, cols, "tbl_emb")

    def _table_errors(self) -> str:
        if "Success" not in self.df_all.columns:
            return "<p class='no-data'>No error info</p>"
        errs = self.df_all[self.df_all["Success"].isin([False, "False", 0, "0"])]
        if errs.empty:
            return "<p class='ok-msg'>✅ No errors during benchmark</p>"
        cols = ["Model", "Category", "Error", "Recommendation"]
        return self._table_html(errs, cols, "tbl_errors")

    # ═══ HTML FINAL ════════════════════════════════════════════════════════════

    def _build_full_html(self, charts: Dict[str, str]) -> str:
        n_total = len(self.df_all)
        n_gen   = len(self.df_gen)
        n_emb   = len(self.df_emb)
        num_runs_val = int(self.df_gen["Num_runs_ok"].max()) if "Num_runs_ok" in self.df_gen.columns and not self.df_gen.empty else 1

        return _HTML.format(
            APP_NAME    = APP_NAME,
            APP_VERSION = APP_VERSION,
            APP_AUTHOR  = APP_AUTHOR,
            GEN_AT      = self._generated_at,
            N_TOTAL     = n_total,
            N_GEN       = n_gen,
            N_EMB       = n_emb,
            NUM_RUNS    = num_runs_val,
            KPI_CARDS   = self._kpi_cards_html(),
            # Charts
            C_TPS        = charts.get("tps",             _empty_div()),
            C_ESCORE     = charts.get("escore",          _empty_div()),
            C_TPW        = charts.get("tpw",             _empty_div()),
            C_SCATTER    = charts.get("scatter",         _empty_div()),
            C_TEMP       = charts.get("temp",            _empty_div()),
            C_PWR        = charts.get("pwr",             _empty_div()),
            C_GPU        = charts.get("gpu",             _empty_div()),
            C_RAM        = charts.get("ram",             _empty_div()),
            C_LATENCY    = charts.get("latency",         _empty_div()),
            C_RADAR      = charts.get("radar",           _empty_div()),
            C_REC_PIE    = charts.get("rec_pie",         _empty_div()),
            C_CAT_COMPARE= charts.get("cat_compare",     _empty_div()),
            C_STABILITY  = charts.get("stability",       _empty_div()),
            C_DECISION   = charts.get("decision_matrix", _empty_div()),
            C_BOXPLOT    = charts.get("multi_run_box",   _empty_div()),
            C_LOAD_INFER = charts.get("load_vs_infer",   _empty_div()),
            # Tables
            T_TOP_CAT    = self._table_top_per_category(),
            T_ALL        = self._table_full_results(),
            T_REMOVE     = self._table_remove_candidates(),
            T_EMB        = self._table_embeddings(),
            T_ERRORS     = self._table_errors(),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# HTML TEMPLATE  (usa {{ }} para llaves literales en el CSS/JS)
# ═══════════════════════════════════════════════════════════════════════════════

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OllamaVision — Benchmark Dashboard v3</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;700&family=Orbitron:wght@400;700;900&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
:root{{
  --bg:#080c14;--bg2:#0a0e1a;--bg3:#0d1220;--b1:#1a2233;--b2:#243040;
  --neon:#39ff14;--cyan:#00f5d4;--pink:#f72585;--ambr:#ffb700;
  --blue:#4cc9f0;--purp:#7209b7;--org:#f8961e;
  --text:#c0ccd0;--dim:#4a6070;--grid:#1a2233;
}}
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
html,body{{background:var(--bg);color:var(--text);font-family:'IBM Plex Mono',monospace;min-height:100vh;overflow-x:hidden}}
body::before{{content:'';position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,.03) 2px,rgba(0,0,0,.03) 4px);pointer-events:none;z-index:9998}}

/* ─── HEADER ─── */
.site-header{{position:sticky;top:0;z-index:200;background:linear-gradient(180deg,#030609 0%,var(--bg) 100%);border-bottom:1px solid var(--neon);box-shadow:0 0 12px rgba(57,255,20,.25);padding:10px 28px;display:flex;justify-content:space-between;align-items:center;gap:12px}}
.brand{{font-family:'Orbitron',monospace;font-size:1.1rem;font-weight:900;color:var(--neon);text-shadow:0 0 10px rgba(57,255,20,.5);letter-spacing:.12em}}
.brand span{{color:var(--cyan)}}
.header-meta{{font-size:.64rem;color:var(--dim);text-align:right;line-height:1.8}}
.header-meta b{{color:var(--cyan)}}
.badge{{display:inline-block;padding:2px 7px;background:var(--ambr);color:#080c14;font-size:.58rem;font-weight:700;letter-spacing:.08em;margin-left:6px;vertical-align:middle}}

/* ─── NAV ─── */
.nav{{display:flex;background:var(--bg2);border-bottom:1px solid var(--b1);padding:0 28px;overflow-x:auto;scrollbar-width:thin;gap:0}}
.nav-btn{{padding:10px 16px;font-family:'IBM Plex Mono';font-size:.68rem;color:var(--dim);border:none;background:none;cursor:pointer;border-bottom:2px solid transparent;transition:all .2s;white-space:nowrap;letter-spacing:.05em}}
.nav-btn:hover{{color:var(--neon)}}
.nav-btn.active{{color:var(--neon);border-bottom-color:var(--neon);text-shadow:0 0 8px rgba(57,255,20,.4)}}

/* ─── SECTIONS ─── */
.section{{display:none;padding:22px 28px 60px}}
.section.active{{display:block}}
.sec-title{{font-family:'Orbitron';font-size:.75rem;color:var(--cyan);text-transform:uppercase;letter-spacing:.15em;margin-bottom:18px;padding-bottom:8px;border-bottom:1px solid var(--b1);text-shadow:0 0 8px rgba(0,245,212,.3)}}

/* ─── KPI CARDS ─── */
.kpi-row{{display:grid;grid-template-columns:repeat(auto-fill,minmax(185px,1fr));gap:12px;margin-bottom:22px}}
.kpi-card{{background:var(--bg3);border:1px solid var(--b1);border-left:3px solid var(--neon);padding:14px 16px;transition:transform .2s,box-shadow .2s;cursor:default}}
.kpi-card:hover{{transform:translateY(-2px);box-shadow:0 4px 20px rgba(0,0,0,.5)}}
.kpi-icon{{font-size:1.1rem;margin-bottom:4px}}
.kpi-title{{font-size:.58rem;color:var(--dim);text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px}}
.kpi-value{{font-size:.98rem;font-weight:700;margin-bottom:3px}}
.kpi-sub{{font-size:.6rem;color:var(--dim)}}

/* ─── CHART GRID ─── */
.g1{{display:grid;grid-template-columns:1fr;gap:12px;margin-bottom:12px}}
.g2{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-bottom:12px}}
.g3{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:12px}}
.chart-box{{background:var(--bg2);border:1px solid var(--b1);padding:6px;transition:border-color .2s;min-height:180px}}
.chart-box:hover{{border-color:var(--neon);box-shadow:0 0 10px rgba(57,255,20,.15)}}
.full-chart{{background:var(--bg2);border:1px solid var(--b1);padding:6px;margin-bottom:12px}}

/* ─── TABLES ─── */
.tbl-wrap{{overflow-x:auto;margin-bottom:22px}}
.tbl-label{{font-size:.65rem;color:var(--cyan);text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px;margin-top:18px}}
.data-table{{width:100%;border-collapse:collapse;font-size:.68rem}}
.data-table th{{background:#0c1424;color:var(--cyan);padding:7px 10px;text-align:left;border-bottom:1px solid var(--neon);font-weight:500;text-transform:uppercase;letter-spacing:.06em;white-space:nowrap}}
.data-table td{{padding:5px 10px;border-bottom:1px solid var(--b1);color:var(--text);white-space:nowrap}}
.data-table tr:hover td{{background:var(--bg3)}}
.no-data{{color:var(--dim);font-size:.75rem;padding:12px 0}}
.ok-msg{{color:var(--neon);font-size:.75rem;padding:12px 0}}

/* ─── TICKER ─── */
.ticker{{position:fixed;bottom:0;left:0;right:0;background:#030609;border-top:1px solid var(--neon);padding:5px 16px;font-size:.6rem;color:var(--dim);display:flex;align-items:center;gap:18px;z-index:200}}
.tick-dot{{width:6px;height:6px;background:var(--neon);border-radius:50%;animation:pulse 1.5s infinite;flex-shrink:0}}
@keyframes pulse{{0%,100%{{opacity:1;box-shadow:0 0 5px var(--neon)}}50%{{opacity:.2;box-shadow:none}}}}
.ticker-spacer{{margin-left:auto;color:var(--neon)}}

/* ─── SCROLLBAR ─── */
::-webkit-scrollbar{{width:4px;height:4px}}
::-webkit-scrollbar-track{{background:var(--bg)}}
::-webkit-scrollbar-thumb{{background:var(--b2)}}
::-webkit-scrollbar-thumb:hover{{background:var(--neon)}}

/* ─── ANIM ─── */
.section.active .chart-box,.section.active .full-chart{{animation:fadeUp .3s ease-out both}}
@keyframes fadeUp{{from{{opacity:0;transform:translateY(6px)}}to{{opacity:1;transform:none}}}}

@media(max-width:900px){{.g2,.g3{{grid-template-columns:1fr}}.kpi-row{{grid-template-columns:repeat(2,1fr)}}.site-header,.nav,.section{{padding-left:14px;padding-right:14px}}}}
</style>
</head>
<body>

<!-- HEADER -->
<header class="site-header">
  <div class="brand">OLLAMA<span>VISION</span> // BENCHMARK</div>
  <div class="header-meta">
    <b>{APP_NAME}</b> v{APP_VERSION}&nbsp;|&nbsp;{APP_AUTHOR}<br>
    Generated: <b>{GEN_AT}</b>&nbsp;|&nbsp;
    <b>{N_GEN}</b> gen models + <b>{N_EMB}</b> embed&nbsp;|&nbsp;
    {NUM_RUNS} runs/model <span class="badge">MULTI-RUN</span>
  </div>
</header>

<!-- NAV -->
<nav class="nav">
  <button class="nav-btn active" onclick="showTab('overview',this)">[ OVERVIEW ]</button>
  <button class="nav-btn" onclick="showTab('performance',this)">[ PERFORMANCE ]</button>
  <button class="nav-btn" onclick="showTab('category',this)">[ BY CATEGORY ]</button>
  <button class="nav-btn" onclick="showTab('thermal',this)">[ THERMAL &amp; POWER ]</button>
  <button class="nav-btn" onclick="showTab('reliability',this)">[ STABILITY ]</button>
  <button class="nav-btn" onclick="showTab('decisions',this)">[ DECISIONS ]</button>
  <button class="nav-btn" onclick="showTab('data',this)">[ COMPLETE DATA ]</button>
</nav>

<!-- ══ OVERVIEW ═══════════════════════════════════════════════════════════════ -->
<section class="section active" id="tab-overview">
  <div class="sec-title">▸ SYSTEM OVERVIEW</div>
  {KPI_CARDS}
  <div class="g2">
    <div class="chart-box">{C_TPS}</div>
    <div class="chart-box">{C_REC_PIE}</div>
  </div>
  <div class="g2">
    <div class="chart-box">{C_SCATTER}</div>
    <div class="chart-box">{C_ESCORE}</div>
  </div>
</section>

<!-- ══ PERFORMANCE ═══════════════════════════════════════════════════════════ -->
<section class="section" id="tab-performance">
  <div class="sec-title">▸ PERFORMANCE METRICS</div>
  <div class="full-chart">{C_TPS}</div>
  <div class="g2">
    <div class="chart-box">{C_ESCORE}</div>
    <div class="chart-box">{C_TPW}</div>
  </div>
  <div class="g2">
    <div class="chart-box">{C_LATENCY}</div>
    <div class="chart-box">{C_LOAD_INFER}</div>
  </div>
</section>

<!-- ══ BY CATEGORY ════════════════════════════════════════════════════════════ -->
<section class="section" id="tab-category">
  <div class="sec-title">▸ CATEGORY COMPARISON</div>
  <div class="full-chart">{C_CAT_COMPARE}</div>
  <div class="g2">
    <div class="chart-box">{C_RADAR}</div>
    <div class="chart-box">{C_REC_PIE}</div>
  </div>
  <div class="tbl-label">🏆 Best model per category (highest Efficiency Score)</div>
  <div class="tbl-wrap">{T_TOP_CAT}</div>
</section>

<!-- ══ THERMAL & POWER ══════════════════════════════════════════════════════ -->
<section class="section" id="tab-thermal">
  <div class="sec-title">▸ THERMAL &amp; POWER ANALYSIS</div>
  <div class="g2">
    <div class="chart-box">{C_TEMP}</div>
    <div class="chart-box">{C_PWR}</div>
  </div>
  <div class="g2">
    <div class="chart-box">{C_GPU}</div>
    <div class="chart-box">{C_RAM}</div>
  </div>
</section>

<!-- ══ STABILITY ═════════════════════════════════════════════════════════════ -->
<section class="section" id="tab-reliability">
  <div class="sec-title">▸ INTER-RUN STABILITY ({NUM_RUNS} runs/model)</div>
  <div class="full-chart">{C_STABILITY}</div>
  <div class="full-chart">{C_BOXPLOT}</div>
</section>

<!-- ══ DECISIONS ═════════════════════════════════════════════════════════════ -->
<section class="section" id="tab-decisions">
  <div class="sec-title">▸ DECISION INTELLIGENCE</div>
  <div class="g2">
    <div class="chart-box">{C_DECISION}</div>
    <div class="chart-box">{C_ESCORE}</div>
  </div>
  <div class="tbl-label">⚠ Models candidates for review or removal</div>
  <div class="tbl-wrap">{T_REMOVE}</div>
  <div class="tbl-label">❌ Models with errors during benchmark</div>
  <div class="tbl-wrap">{T_ERRORS}</div>
  <div class="tbl-label">🔲 Embedding models</div>
  <div class="tbl-wrap">{T_EMB}</div>
</section>

<!-- ══ COMPLETE DATA ══════════════════════════════════════════════════════════ -->
<section class="section" id="tab-data">
  <div class="sec-title">▸ COMPLETE RESULTS</div>
  <div class="tbl-wrap">{T_ALL}</div>
</section>

<!-- TICKER -->
<div class="ticker">
  <div class="tick-dot"></div>
  <span>{APP_NAME} v{APP_VERSION}</span>
  <span>Generated: {GEN_AT}</span>
  <span>Total models: {N_TOTAL}</span>
  <span>{NUM_RUNS} runs/model</span>
  <span class="ticker-spacer">{APP_AUTHOR}</span>
</div>

<script>
function showTab(id, btn) {{
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + id).classList.add('active');
  btn.classList.add('active');
  setTimeout(() => {{
    document.querySelectorAll('[id^="c_"]').forEach(el => {{
      try {{ Plotly.Plots.resize(el); }} catch(e) {{}}
    }});
  }}, 60);
}}
</script>
</body>
</html>
"""
