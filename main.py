#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — OllamaVision Benchmark Suite v4.0
Main entry point with interactive CLI.

Usage:
    python3 main.py                         # full benchmark (interactive)
    python3 main.py --simulate              # without Jetson/jtop (development)
    python3 main.py --no-resume             # ignore previous checkpoint
    python3 main.py --shutdown              # auto shutdown when done
    python3 main.py --no-dashboard          # skip HTML dashboard
    python3 main.py --open-browser          # open dashboard in browser on finish
    python3 main.py --dashboard-only        # only generate HTML from CSV
    python3 main.py --analyze               # statistical analysis to console
    python3 main.py --check                 # verify environment
    python3 main.py --filter CODE,MATH      # only specified categories
    python3 main.py --models qwen,phi4      # only matching models
    python3 main.py --runs 5                # override NUM_RUNS
    python3 main.py --timeout 300           # override per-inference timeout (s)
    python3 main.py --help
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from config import (
    APP_AUTHOR, APP_NAME, APP_VERSION,
    CSV_OUTPUT, DASHBOARD_FILE, SHUTDOWN_DELAY_MIN,
    NUM_RUNS, OLLAMA_TIMEOUT,
)

console = Console()


# ═══════════════════════════════════════════════════════════════════════════════
# BANNER
# ═══════════════════════════════════════════════════════════════════════════════

BANNER = r"""
   ___  _ _                    __   ___         _
  / _ \| | | __ _ _ __ ___    / /  | _ )___ _ _| |_  |  v{version}
 | (_) | | |/ _` | '_ ` _ \  / /   | _ / -_) ' \  _| |  {author}
  \___/|_|_|\__,_|_| |_| |_|/_/    |___\___|_||_\__| |
"""


def _print_banner() -> None:
    txt = BANNER.format(version=APP_VERSION, author=APP_AUTHOR)
    console.print(Panel(
        Text(txt, style="bold bright_cyan", justify="center"),
        border_style="bright_cyan",
        padding=(0, 6),
    ))


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ollama_benchmark",
        description=f"{APP_NAME} v{APP_VERSION}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--simulate",       action="store_true",
                   help="Simulate hardware (no jtop/Jetson required)")
    p.add_argument("--no-resume",      action="store_true",
                   help="Ignore previous checkpoint")
    p.add_argument("--shutdown",       action="store_true",
                   help=f"Shutdown system {SHUTDOWN_DELAY_MIN} min after completion")
    p.add_argument("--no-dashboard",   action="store_true",
                   help="Skip HTML dashboard generation")
    p.add_argument("--open-browser",   action="store_true",
                   help="Open dashboard in browser on completion")
    p.add_argument("--dashboard-only", action="store_true",
                   help="Only generate dashboard from existing CSV")
    p.add_argument("--analyze",        action="store_true",
                   help="Statistical analysis to console from CSV")
    p.add_argument("--check",          action="store_true",
                   help="Verify environment and show component status")
    p.add_argument("--yes", "-y",      action="store_true",
                   help="Answer yes to all confirmations")
    p.add_argument("--csv",            type=Path, default=CSV_OUTPUT,
                   help=f"Results CSV path (default: {CSV_OUTPUT})")
    p.add_argument("--output-html",    type=Path, default=DASHBOARD_FILE,
                   help=f"HTML dashboard path (default: {DASHBOARD_FILE})")

    # ── Execution filters ─────────────────────────────────────────────────────
    p.add_argument("--filter",         metavar="CATEGORY[,...]", default=None,
                   help="Only benchmark specified categories (e.g. CODE,MATH,REASONING)")
    p.add_argument("--models",         metavar="TERM[,...]", default=None,
                   help="Only models whose name contains any of the terms (e.g. qwen,phi4)")
    p.add_argument("--runs",           type=int, default=None,
                   help=f"Number of measurement runs per model (default: {NUM_RUNS})")
    p.add_argument("--timeout",        type=int, default=None,
                   help=f"Per-inference timeout in seconds (default: {OLLAMA_TIMEOUT})")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# MODOS ESPECIALES
# ═══════════════════════════════════════════════════════════════════════════════

def mode_environment_check() -> None:
    """Verify environment and display status report."""
    from utils.system_utils import SystemUtils
    console.print("\n[bold cyan]⚙  ENVIRONMENT CHECK[/bold cyan]\n")
    checks = SystemUtils.environment_check()

    tbl = Table(show_header=True, header_style="bold bright_cyan", box=None)
    tbl.add_column("Component",   style="dim white", width=24)
    tbl.add_column("Status",      width=10)
    tbl.add_column("Detail",      style="dim")

    status_map = {
        True:  ("[bold green]✅ OK[/bold green]",   "green"),
        False: ("[bold red]❌ FAIL[/bold red]",     "red"),
    }

    rows = [
        ("Root (sudo)",       checks.get("is_root", False),         "Recommended for drop_caches"),
        ("Ollama API",        checks.get("ollama_alive", False),     checks.get("ollama_url","")),
        ("jtop (jetson-stats)", checks.get("jtop_available", False),"sudo pip install -U jetson-stats"),
        ("Assets dir",        checks.get("assets_dir", False),      "Required for VISION models"),
        ("Test image",        checks.get("test_image", False),       "assets/test_image.jpg"),
    ]

    for label, ok, detail in rows:
        icon, _ = status_map[ok]
        tbl.add_row(label, icon, detail)

    tbl.add_row("Hostname",    "", str(checks.get("hostname","")))
    tbl.add_row("Kernel",      "", str(checks.get("kernel","")))
    tbl.add_row("Total RAM",   "", f"{checks.get('ram_total_mb',0):.0f} MB")
    tbl.add_row("Free RAM",    "", f"{checks.get('ram_available_mb',0):.0f} MB")
    tbl.add_row("Free Disk",   "", f"{checks.get('disk_free_gb',0):.1f} GB")

    console.print(tbl)


def mode_dashboard_only(csv_path: Path, output_path: Path, open_browser: bool) -> None:
    """Generate HTML dashboard from existing CSV."""
    import pandas as pd
    from views.dashboard_view import DashboardView
    from utils.logger import BenchmarkLogger

    log = BenchmarkLogger()

    if not csv_path.exists():
        log.model_error("DASHBOARD", f"File not found: {csv_path}")
        sys.exit(1)

    log.info(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    dash = DashboardView(output_path=output_path)
    dash.load(df)
    out = dash.generate()
    log.dashboard_saved(out)
    log.status(f"Dashboard generated: {out}  ({out.stat().st_size // 1024}KB)")

    if open_browser:
        webbrowser.open(f"file://{out.resolve()}")


def mode_analyze(csv_path: Path) -> None:
    """Statistical analysis to console from CSV."""
    import pandas as pd
    from rich import box as rbox

    if not csv_path.exists():
        console.print(f"[red]File not found: {csv_path}[/red]")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().mean() > 0.5:
            df[col] = converted

    df_gen = df[df["Tokens_per_second"] > 0].copy() if "Tokens_per_second" in df.columns else df

    console.rule("[bold cyan]📊 RESULTS ANALYSIS[/bold cyan]")
    console.print(f"\n[dim]CSV: {csv_path}  |  Rows: {len(df)}  |  Gen models: {len(df_gen)}[/dim]\n")

    if df_gen.empty:
        console.print("[yellow]No generation results found[/yellow]")
        return

    # Main table
    tbl = Table(
        title="Results by Efficiency Score (desc)",
        box=rbox.SIMPLE_HEAVY,
        header_style="bold bright_cyan",
        show_lines=False,
    )
    for col, w in [("Model",20),("Category",10),("TPS",7),("RAM(MB)",8),
                   ("GPU°C",6),("Pwr(W)",7),("tok/W",6),("Score",6),("Rec",10)]:
        tbl.add_column(col, width=w, justify="right" if col not in ("Model","Category","Rec") else "left")

    df_sorted = df_gen.sort_values("Efficiency_score", ascending=False) if "Efficiency_score" in df_gen.columns else df_gen
    for _, row in df_sorted.iterrows():
        tps   = float(row.get("Tokens_per_second", 0))
        score = float(row.get("Efficiency_score", 0))
        rec   = str(row.get("Recommendation",""))
        tcolor = "bright_green" if tps >= 20 else ("yellow" if tps >= 10 else "red")
        rcolor = "bold green" if rec.startswith("KEEP") else ("red" if rec == "REMOVE" else "yellow")
        tbl.add_row(
            str(row.get("Model",""))[:20],
            str(row.get("Category",""))[:9],
            Text(f"{tps:.1f}", style=tcolor),
            f"{float(row.get('RAM_used_MB',0)):.0f}",
            f"{float(row.get('Temp_GPU_C',0)):.0f}",
            f"{float(row.get('Power_Total_mW',0))/1000:.1f}",
            f"{float(row.get('Tokens_per_W',0)):.2f}",
            Text(f"{score:.0f}", style="yellow"),
            Text(rec, style=rcolor),
        )
    console.print(tbl)

    # Global stats
    console.rule("[dim]GLOBAL STATISTICS[/dim]")
    stats_tbl = Table(box=None, show_header=False, padding=(0,2))
    stats_tbl.add_column(style="dim", width=22)
    stats_tbl.add_column(style="bright_cyan")
    for k, v in [
        ("Avg TPS",         f"{df_gen['Tokens_per_second'].mean():.2f}"),
        ("Max TPS",         f"{df_gen['Tokens_per_second'].max():.2f}"),
        ("Min TPS",         f"{df_gen['Tokens_per_second'].min():.2f}"),
        ("Avg power",       f"{df_gen['Power_Total_mW'].mean()/1000:.1f} W" if "Power_Total_mW" in df_gen.columns else "N/A"),
        ("Avg GPU temp",    f"{df_gen['Temp_GPU_C'].mean():.1f}°C" if "Temp_GPU_C" in df_gen.columns else "N/A"),
        ("Avg score",       f"{df_gen['Efficiency_score'].mean():.1f}/100" if "Efficiency_score" in df_gen.columns else "N/A"),
    ]:
        stats_tbl.add_row(k, v)
    console.print(stats_tbl)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIRMACIÓN INTERACTIVA
# ═══════════════════════════════════════════════════════════════════════════════

def _interactive_confirm(args: argparse.Namespace) -> None:
    """Display configuration and ask for user confirmation."""
    from utils.checkpoint import CheckpointManager
    ck = CheckpointManager()
    has_ck = ck.load()

    # Configuration table
    cfg = Table(box=None, show_header=False, padding=(0, 2))
    cfg.add_column(style="dim white", width=22)
    cfg.add_column(style="bright_cyan")

    hw_mode  = "[bold yellow]⚙ SIMULATION[/bold yellow]" if args.simulate else "[green]REAL (jtop)[/green]"
    resume_v = "[red]NO[/red]" if args.no_resume else (
        f"[green]YES[/green] — {ck.count_done()} completed" if has_ck else "[dim]YES (no checkpoint)[/dim]"
    )
    shut_v   = f"[bold red]YES — in {SHUTDOWN_DELAY_MIN} min[/bold red]" if args.shutdown else "[dim]NO[/dim]"
    dash_v   = "[dim]NO[/dim]" if args.no_dashboard else f"[green]YES (every {5} models)[/green]"
    runs_v   = f"[yellow]{args.runs}[/yellow]" if args.runs else f"[dim]{NUM_RUNS} (default)[/dim]"
    to_v     = f"[yellow]{args.timeout}s[/yellow]" if args.timeout else f"[dim]{OLLAMA_TIMEOUT}s (default)[/dim]"
    filt_v   = f"[yellow]{args.filter}[/yellow]" if args.filter else "[dim]all categories[/dim]"
    mods_v   = f"[yellow]{args.models}[/yellow]" if args.models else "[dim]all models[/dim]"

    cfg.add_row("Hardware",          hw_mode)
    cfg.add_row("Resume checkpoint", resume_v)
    cfg.add_row("Auto-shutdown",     shut_v)
    cfg.add_row("Dashboard live",    dash_v)
    cfg.add_row("Runs per model",    runs_v)
    cfg.add_row("Inference timeout", to_v)
    cfg.add_row("Category filter",   filt_v)
    cfg.add_row("Model filter",      mods_v)
    cfg.add_row("CSV output",        str(args.csv))
    cfg.add_row("HTML output",       str(args.output_html))

    console.print(Panel(cfg, title="[bold cyan]⚙ CONFIGURATION[/bold cyan]",
                        border_style="cyan"))

    if args.shutdown:
        console.print(
            f"\n[bold red]⚠  WARNING: The system will SHUTDOWN automatically "
            f"{SHUTDOWN_DELAY_MIN} minutes after completing the benchmark.[/bold red]"
        )
        if not args.yes:
            if not Confirm.ask("[bold red]Confirm auto-shutdown?[/bold red]", default=False):
                args.shutdown = False
                console.print("[dim]Shutdown cancelled.[/dim]")

    if not args.yes:
        if not Confirm.ask("\n[bold cyan]Start benchmark?[/bold cyan]", default=True):
            console.print("[dim]Cancelled.[/dim]")
            sys.exit(0)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def _print_usage_guide() -> None:
    """Display a quick guide of CLI options."""
    guide = Table(box=None, show_header=False, padding=(0, 2))
    guide.add_column(style="dim white", width=22)
    guide.add_column(style="bright_cyan")
    guide.add_row("Option", "Description")
    guide.add_row("--simulate",          "Simulate hardware (no jtop)")
    guide.add_row("--no-resume",         "Ignore previous checkpoint")
    guide.add_row("--shutdown",          f"Shutdown system on completion (in {SHUTDOWN_DELAY_MIN} min)")
    guide.add_row("--no-dashboard",      "Skip HTML dashboard")
    guide.add_row("--open-browser",      "Open dashboard on completion")
    guide.add_row("--dashboard-only",    "Only generate HTML from CSV")
    guide.add_row("--analyze",           "Statistical analysis from CSV")
    guide.add_row("--check",             "Verify environment")
    guide.add_row("--filter CAT[,...]",  "Only benchmark categories (CODE,MATH…)")
    guide.add_row("--models TERM[,...]", "Only matching models (qwen,phi4…)")
    guide.add_row("--runs N",            f"Measurement runs per model (default: {NUM_RUNS})")
    guide.add_row("--timeout N",         f"Per-inference timeout in sec (default: {OLLAMA_TIMEOUT})")
    guide.add_row("--yes, -y",           "Answer yes to all confirmations")
    guide.add_row("--csv PATH",          f"CSV path (default: {CSV_OUTPUT})")
    guide.add_row("--output-html PATH",  f"HTML path (default: {DASHBOARD_FILE})")
    console.print(Panel(guide, title="[bold cyan]📋 QUICK GUIDE[/bold cyan]",
                        border_style="cyan", padding=(1, 2)))
    console.print("[dim]Run with --help for more details.[/dim]\n")

def main() -> None:
    _print_banner()
    args = _parse_args()

    # Modos especiales
    if args.check:
        mode_environment_check()
        return
    if args.dashboard_only:
        mode_dashboard_only(args.csv, args.output_html, args.open_browser)
        return
    if args.analyze:
        mode_analyze(args.csv)
        return

    # Mostrar guía rápida antes de la confirmación interactiva
    _print_usage_guide()
    _interactive_confirm(args)

    from controllers.benchmark_controller import BenchmarkController

    # Parsear filtros de CLI en listas
    filter_cats  = [c.strip() for c in args.filter.split(",") if c.strip()] \
                   if args.filter else None
    filter_mods  = [m.strip() for m in args.models.split(",") if m.strip()] \
                   if args.models else None

    ctrl = BenchmarkController(
        csv_path           = args.csv,
        simulate_hw        = args.simulate,
        auto_shutdown      = args.shutdown,
        resume             = not args.no_resume,
        live_dashboard     = not args.no_dashboard,
        open_browser       = args.open_browser,
        num_runs           = args.runs if args.runs else NUM_RUNS,
        filter_categories  = filter_cats,
        filter_models      = filter_mods,
        timeout            = args.timeout,
    )
    ctrl.dashboard.output_path = args.output_html
    exit_code = ctrl.run()
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
