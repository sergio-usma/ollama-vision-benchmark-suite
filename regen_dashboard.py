#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regen_dashboard.py — Regenerate the HTML dashboard from an existing CSV.
Usage: python3 regen_dashboard.py [--csv path.csv] [--open]

Useful for:
  - Viewing current data without re-running the full benchmark
  - Testing visualization changes
  - Regenerating after adjusting thresholds in config.py
"""
import argparse
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from config import CSV_OUTPUT, DASHBOARD_FILE
from models.data_model import CSVManager
from views.dashboard_view import DashboardView


def regen(csv_path: Path, open_browser: bool = False) -> None:
    csv_mgr = CSVManager(csv_path)
    df = csv_mgr.load_dataframe()

    if df.empty:
        print(f"❌ CSV empty or not found: {csv_path}")
        sys.exit(1)

    print(f"📂 CSV: {csv_path} ({len(df)} rows)")

    # Compatibility with CSV v2 (without multi-run columns)
    multi_run_cols = {
        'TPS_median': df['Tokens_per_second'].astype(float),
        'TPS_stdev':  0.0,
        'TPS_min':    df['Tokens_per_second'].astype(float),
        'TPS_max':    df['Tokens_per_second'].astype(float),
        'TPS_cv':     0.0,
        'TPS_runs':   df['Tokens_per_second'].apply(lambda x: f"{x:.2f}" if x > 0 else ""),
        'Num_runs_ok': 1,
        'Num_runs_total': 1,
        'Stability':  'SINGLE_RUN',
        'Category_pct': 0.0,
        'Decision_reasons': '',
    }
    for col, default in multi_run_cols.items():
        if col not in df.columns:
            df[col] = default
        # Force numeric types where applicable
        if col in ['TPS_median','TPS_stdev','TPS_min','TPS_max','TPS_cv','Category_pct']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # Calculate category percentile if not present
    ok_mask = df['Tokens_per_second'] > 0
    ok = df[ok_mask]
    for cat in ok['Category'].unique():
        cat_mask = ok['Category'] == cat
        tps_list = ok.loc[cat_mask, 'Tokens_per_second'].tolist()
        for row_idx, tps in zip(ok.loc[cat_mask].index, tps_list):
            n_below = sum(1 for t in tps_list if t < tps)
            pct = (n_below / (len(tps_list)-1)*100.0) if len(tps_list) > 1 else 50.0
            df.at[row_idx, 'Category_pct'] = round(pct, 1)

    # Generate
    print(f"🔄 Generating dashboard...")
    dash = DashboardView()
    dash.load(df)
    path = dash.generate()
    size_kb = path.stat().st_size / 1024
    print(f"✅ Dashboard generated: {path} ({size_kb:.0f} KB)")

    n_gen = (df['Tokens_per_second'] > 0).sum()
    n_emb = (df['Tokens_per_second'] == 0).sum()
    print(f"   Models: {n_gen} generation + {n_emb} embedding = {len(df)} total")

    if 'Recommendation' in df.columns:
        recs = df['Recommendation'].value_counts().to_dict()
        print(f"   Recommendations: {recs}")

    if open_browser:
        webbrowser.open(f"file://{path.resolve()}")
        print("🌐 Opened in browser")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Regenerate HTML dashboard")
    p.add_argument("--csv", type=Path, default=CSV_OUTPUT, help="Path to results CSV")
    p.add_argument("--open", action="store_true", help="Open in browser when done")
    args = p.parse_args()
    regen(args.csv, args.open)
