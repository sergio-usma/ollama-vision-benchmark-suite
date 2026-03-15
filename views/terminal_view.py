#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
views/terminal_view.py — Real-time terminal dashboard using Rich.

Layout (6 panels updated at 2Hz):
┌─────────────────────────────────────────────────────────────────────────┐
│  HEADER: app name, current model, ETA, timestamp                        │
├─────────────────────────────────────────────────────────────────────────┤
│  PROGRESS: global bar + current model bar                               │
├───────────────────────────────────────┬─────────────────────────────────┤
│  RESULTS TABLE (last 18 models)       │  LIVE STATS                    │
├───────────────────────────────────────┤  (metrics, top, candidates)    │
│  HARDWARE (temps, power, GPU, RAM)    │                                 │
│                                       ├─────────────────────────────────┤
│                                       │  LIVE LOG (last lines)         │
├─────────────────────────────────────────────────────────────────────────┤
│  FOOTER: elapsed, hint                                                  │
└─────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from config import APP_NAME, APP_VERSION, CATEGORY_COLORS, RECOMMENDATION_COLORS
from models.data_model import BenchmarkResult, HardwareSnapshot


# ═══════════════════════════════════════════════════════════════════════════════
# PALETA DE COLORES
# ═══════════════════════════════════════════════════════════════════════════════

def _temp_color(t: float) -> str:
    if t > 80: return "red1"
    if t > 70: return "orange3"
    if t > 60: return "yellow"
    return "bright_green"

def _tps_color(t: float) -> str:
    if t >= 30: return "bright_green"
    if t >= 15: return "yellow"
    if t > 0:   return "orange3"
    return "dim"

def _pct_color(p: float) -> str:
    if p > 85: return "red1"
    if p > 70: return "orange3"
    return "bright_green"

def _rec_style(r: str) -> str:
    return {
        "KEEP★":    "bold bright_green",
        "KEEP":     "green",
        "REVIEW":   "yellow",
        "REMOVE":   "bold red1",
        "OPTIONAL": "dim",
        "EMBEDDING":"cyan",
        "ERROR":    "red",
    }.get(r, "white")


# ═══════════════════════════════════════════════════════════════════════════════
# PROGRESS BARS
# ═══════════════════════════════════════════════════════════════════════════════

class BenchmarkProgress:
    """Barras de progreso dual: global + modelo actual."""

    def __init__(self):
        self.global_bar = Progress(
            SpinnerColumn(spinner_name="dots12", style="bright_cyan"),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=38, style="cyan", complete_style="bright_green",
                      finished_style="bold bright_green"),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            expand=True,
        )
        self.model_bar = Progress(
            SpinnerColumn(spinner_name="arc", style="magenta"),
            TextColumn("[magenta]{task.description}"),
            BarColumn(bar_width=28, style="magenta", complete_style="bright_magenta"),
            TaskProgressColumn(),
            expand=True,
        )
        self._gt = self.global_bar.add_task("Global benchmark", total=100)
        self._mt = self.model_bar.add_task("Preparing...",      total=100)

    def set_total(self, n: int) -> None:
        self.global_bar.reset(self._gt, total=n, completed=0)

    def advance_global(self, n: int = 1) -> None:
        self.global_bar.advance(self._gt, n)

    def set_global_desc(self, desc: str) -> None:
        self.global_bar.update(self._gt, description=desc)

    def set_model(self, model: str, pct: float) -> None:
        self.model_bar.update(
            self._mt,
            description=f"[bold]{model[:30]}",
            completed=pct,
            total=100,
        )

    def finish_model(self) -> None:
        self.model_bar.update(self._mt, completed=100)

    def render(self) -> Panel:
        return Panel(
            Columns([self.global_bar, self.model_bar], equal=True),
            title="[bold cyan]⚡ PROGRESS",
            border_style="cyan",
            padding=(0, 1),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL: HARDWARE
# ═══════════════════════════════════════════════════════════════════════════════

def _panel_hardware(snap: HardwareSnapshot) -> Panel:
    root = Table.grid(expand=True, padding=(0, 1))
    root.add_column(ratio=1)
    root.add_column(ratio=1)

    # — Temperaturas —
    tc = _temp_color
    t_tbl = Table(box=box.MINIMAL, show_header=False, expand=True, padding=(0, 1))
    t_tbl.add_column("sensor", style="dim white", width=7)
    t_tbl.add_column("value",  justify="right", width=9)
    t_tbl.add_row("GPU",  Text(f"{snap.temp_gpu:.1f}°C", style=tc(snap.temp_gpu)))
    t_tbl.add_row("CPU",  Text(f"{snap.temp_cpu:.1f}°C", style=tc(snap.temp_cpu)))
    t_tbl.add_row("SOC",  Text(f"{snap.temp_soc:.1f}°C", style=tc(snap.temp_soc)))
    t_tbl.add_row("TJ",   Text(f"{snap.temp_tj:.1f}°C",  style=tc(snap.temp_tj)))

    # — Potencia —
    p_color = "bright_green" if snap.power_total < 15_000 else \
              ("yellow" if snap.power_total < 25_000 else "red1")
    p_tbl = Table(box=box.MINIMAL, show_header=False, expand=True, padding=(0, 1))
    p_tbl.add_column("rail", style="dim white", width=9)
    p_tbl.add_column("val",  justify="right", width=9)
    p_tbl.add_row("GPU+SOC", f"{snap.vdd_gpu_soc:,}mW")
    p_tbl.add_row("CPU+CV",  f"{snap.vdd_cpu_cv:,}mW")
    p_tbl.add_row("SYS5V0",  f"{snap.vin_sys_5v0:,}mW")
    p_tbl.add_row("TOTAL",   Text(f"{snap.power_total:,}mW", style=f"bold {p_color}"))

    root.add_row(
        Panel(t_tbl, title="🌡 TEMPS",   border_style="red",    padding=(0, 0)),
        Panel(p_tbl, title="⚡ POWER",   border_style="yellow", padding=(0, 0)),
    )

    # — GPU y RAM —
    def _bar(pct: float, width: int = 16) -> str:
        filled = int(pct / 100 * width)
        return "█" * filled + "░" * (width - filled)

    gr_tbl = Table(box=box.MINIMAL, show_header=False, expand=True, padding=(0, 1))
    gr_tbl.add_column("k", style="dim white", width=10)
    gr_tbl.add_column("v", justify="left")

    gpu_c = _pct_color(snap.gpu_load)
    ram_c = _pct_color(snap.ram_percent)

    gr_tbl.add_row("GPU load",  Text(f"{_bar(snap.gpu_load)}  {snap.gpu_load:.0f}%", style=gpu_c))
    gr_tbl.add_row("GPU freq",  f"{snap.gpu_freq:,} MHz")
    gr_tbl.add_row("EMC",       f"{snap.emc_freq:,} MHz")
    gr_tbl.add_row("RAM",       Text(f"{_bar(snap.ram_percent)} {snap.ram_percent:.0f}% ({snap.ram_used:.0f}MB)", style=ram_c))
    gr_tbl.add_row("SWAP",      f"{snap.swap_percent:.0f}% ({snap.swap_used:.0f}MB)")
    gr_tbl.add_row("CPU avg",   Text(f"{_bar(snap.cpu_avg_load)} {snap.cpu_avg_load:.0f}%",
                                     style=_pct_color(snap.cpu_avg_load)))

    root.add_row(
        Panel(gr_tbl, title="🎮 GPU / MEM", border_style="cyan", padding=(0, 0)),
        Panel(_panel_fan_disk(snap), title="💨 FAN / DISK", border_style="dim", padding=(0, 0)),
    )

    return Panel(
        root,
        title="[bold red]🔧 LIVE HARDWARE[/bold red]",
        border_style="red",
        padding=(0, 1),
    )


def _panel_fan_disk(snap: HardwareSnapshot) -> Table:
    tbl = Table(box=box.MINIMAL, show_header=False, expand=True, padding=(0, 1))
    tbl.add_column("k", style="dim white", width=10)
    tbl.add_column("v", justify="left")
    # Fan
    fan_str = snap.fan_speeds[:20] if snap.fan_speeds else "N/A"
    tbl.add_row("Fan speed", fan_str)
    tbl.add_row("Profile",   snap.fan_profile[:18] or "N/A")
    # Disk
    disk_pct = (snap.disk_used / snap.disk_total * 100) if snap.disk_total > 0 else 0
    tbl.add_row("Disk used",  f"{snap.disk_used:.0f}/{snap.disk_total:.0f}GB ({disk_pct:.0f}%)")
    tbl.add_row("Disk free",  f"{snap.disk_available:.0f}GB")
    # Uptime
    up = timedelta(seconds=int(snap.uptime_seconds))
    tbl.add_row("Uptime", str(up))
    return tbl


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL: RESULTADOS
# ═══════════════════════════════════════════════════════════════════════════════

def _panel_results(results: List[BenchmarkResult]) -> Panel:
    t = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold bright_cyan",
        expand=True,
        row_styles=["", "dim"],
        padding=(0, 0),
    )
    t.add_column("#",       width=3,  justify="right")
    t.add_column("Model",   width=22)
    t.add_column("Cat",     width=8)
    t.add_column("TPS",     width=7,  justify="right")
    t.add_column("TTFT",    width=7,  justify="right")
    t.add_column("RAM(MB)", width=8,  justify="right")
    t.add_column("GPU°C",   width=6,  justify="right")
    t.add_column("Pwr(W)",  width=7,  justify="right")
    t.add_column("tok/W",   width=6,  justify="right")
    t.add_column("Score",   width=6,  justify="right")
    t.add_column("Rec",     width=10)

    for i, r in enumerate(results[-18:], 1):
        hw = r.hw
        ttft_str = r.ttft_display if hasattr(r, "ttft_display") else "N/A"
        t.add_row(
            Text(str(i), style="dim"),
            Text(r.model[:21], style="bright_cyan"),
            Text(r.category[:7], style="yellow"),
            Text(r.tps_display, style=_tps_color(r.tokens_per_second)),
            Text(ttft_str, style="cyan"),
            f"{hw.ram_used:.0f}",
            Text(f"{hw.temp_gpu:.0f}", style=_temp_color(hw.temp_gpu)),
            f"{hw.power_watts:.1f}",
            f"{r.tokens_per_watt:.2f}",
            Text(f"{r.efficiency_score:.0f}", style="yellow"),
            Text(r.recommendation, style=_rec_style(r.recommendation)),
        )

    return Panel(
        t,
        title=f"[bold green]📊 RESULTS  [{len(results)} completed][/bold green]",
        border_style="green",
        padding=(0, 0),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL: ESTADÍSTICAS EN VIVO
# ═══════════════════════════════════════════════════════════════════════════════

def _panel_stats(stats: Dict[str, Any]) -> Panel:
    if not stats or stats.get("total_tested", 0) == 0:
        return Panel(
            Text("No data yet...", style="dim"),
            title="📈 STATISTICS",
            border_style="bright_blue",
        )

    tbl = Table(box=box.MINIMAL, show_header=False, expand=True, padding=(0, 1))
    tbl.add_column("k", style="dim white", width=16)
    tbl.add_column("v", justify="right")

    def row(k: str, v: Any, style: str = "white") -> None:
        tbl.add_row(k, Text(str(v), style=style))

    row("Tested",         stats.get("total_tested", 0))
    row("Successful",     stats.get("total_success", 0),   "bright_green")
    row("Failed",         stats.get("total_failed", 0),    "red" if stats.get("total_failed",0) else "dim")
    row("Embeddings",     stats.get("total_embeddings", 0),"cyan")
    tbl.add_row("", "")
    row("Avg TPS",        f"{stats.get('avg_tokens_s',0):.1f}",     "bright_cyan")
    row("Max TPS",        f"{stats.get('max_tokens_s',0):.1f}",     "bright_green")
    row("Min TPS",        f"{stats.get('min_tokens_s',0):.1f}",     "dim")
    row("Median TPS",     f"{stats.get('median_tokens_s',0):.1f}",  "cyan")
    tbl.add_row("", "")
    row("Avg tok/W",      f"{stats.get('avg_tok_per_w',0):.2f}",    "yellow")
    row("Avg power",      f"{stats.get('avg_power_mw',0)/1000:.1f}W","yellow")
    row("Avg GPU temp",   f"{stats.get('avg_temp_gpu',0):.1f}°C",
        _temp_color(stats.get("avg_temp_gpu", 0)))
    row("Avg score",      f"{stats.get('avg_efficiency',0):.0f}/100","bright_cyan")
    tbl.add_row("", "")

    best = str(stats.get("best_model", ""))[:16]
    eff  = str(stats.get("most_efficient", ""))[:16]
    eco  = str(stats.get("most_eco", ""))[:16]
    row("🏆 Fastest",     best, "bold bright_green")
    row("🎯 Best eff",    eff,  "bold cyan")
    row("🔋 Best eco",    eco,  "bold yellow")

    tbl.add_row("", "")
    cats = stats.get("by_category", {})
    if cats:
        tbl.add_row(Text("── By category ──", style="dim"), "")
        for cat, avg in sorted(cats.items(), key=lambda x: x[1], reverse=True):
            tbl.add_row(
                Text(f"  {cat}", style="dim yellow"),
                Text(f"{avg:.0f} TPS", style="bright_cyan"),
            )

    remove = stats.get("remove_candidates", [])
    if remove:
        tbl.add_row("", "")
        tbl.add_row(
            Text("⚠ REMOVE:", style="bold red"),
            Text(f"{len(remove)} modelos", style="red"),
        )

    return Panel(
        tbl,
        title="[bold bright_blue]📈 LIVE STATISTICS[/bold bright_blue]",
        border_style="bright_blue",
        padding=(0, 1),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL: LOG EN VIVO
# ═══════════════════════════════════════════════════════════════════════════════

def _panel_log(lines: List[str]) -> Panel:
    text = Text()
    for line in lines[-10:]:
        text.append(line[:70] + "\n", style="dim white")
    return Panel(
        text,
        title="[dim]📋 LIVE LOG[/dim]",
        border_style="dim",
        padding=(0, 1),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER & FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

def _panel_header(
    current_model: str,
    eta:           str,
    simulated:     bool,
) -> Panel:
    now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    txt = Text()
    txt.append(f" {APP_NAME} ", style="bold bright_cyan on dark_blue")
    txt.append(f" v{APP_VERSION} ", style="dim white")
    if simulated:
        txt.append(" [SIM] ", style="bold yellow on dark_red")
    txt.append(f"  {now} ", style="dim cyan")
    if current_model:
        txt.append(f"  🤖 {current_model[:28]} ", style="bold magenta")
    if eta and eta != "--:--":
        txt.append(f"  ETA {eta} ", style="dim yellow")
    return Panel(txt, border_style="bright_cyan", padding=(0, 2))


def _panel_footer(elapsed: float, msg: str = "") -> Panel:
    e_str = str(timedelta(seconds=int(elapsed)))
    hint  = msg or "Ctrl+C saves checkpoint and allows resuming later"
    txt   = Text(f"  ⏱  Elapsed: {e_str}  │  {hint}", style="dim white")
    return Panel(txt, border_style="dim", padding=(0, 0))


# ═══════════════════════════════════════════════════════════════════════════════
# TERMINAL VIEW (MVC — Vista principal)
# ═══════════════════════════════════════════════════════════════════════════════

class TerminalView:
    """
    Vista de terminal en tiempo real.
    Usa Rich Live para refrescar el layout completo a 2Hz.
    Expone API pública para que el Controller actualice los datos.
    """

    def __init__(self, simulated: bool = False):
        self._console        = Console()
        self._live: Optional[Live] = None
        self._layout         = self._build_layout()
        self.progress        = BenchmarkProgress()
        self._results:       List[BenchmarkResult] = []
        self._hw_snap        = HardwareSnapshot()
        self._stats:         Dict[str, Any] = {}
        self._log_lines:     List[str] = []
        self._current_model  = ""
        self._start_time     = time.time()
        self._simulated      = simulated

    # ── Layout ───────────────────────────────────────────────────────────────

    def _build_layout(self) -> Layout:
        root = Layout(name="root")
        root.split_column(
            Layout(name="header",   size=3),
            Layout(name="progress", size=5),
            Layout(name="main"),
            Layout(name="footer",   size=3),
        )
        root["main"].split_row(
            Layout(name="left",  ratio=3),
            Layout(name="right", ratio=1),
        )
        root["left"].split_column(
            Layout(name="results",  ratio=3),
            Layout(name="hardware", ratio=2),
        )
        root["right"].split_column(
            Layout(name="stats",   ratio=5),
            Layout(name="logpane", ratio=2),
        )
        return root

    # ── Ciclo de vida ────────────────────────────────────────────────────────

    def start(self) -> None:
        self._live = Live(
            self._layout,
            console=self._console,
            refresh_per_second=2,
            screen=True,
            transient=False,
        )
        self._live.start()
        self._refresh()

    def stop(self) -> None:
        if self._live:
            try:
                self._live.stop()
            except Exception:
                pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    # ── Refresh interno ──────────────────────────────────────────────────────

    def _refresh(self) -> None:
        elapsed = time.time() - self._start_time
        eta     = self._calc_eta()
        try:
            self._layout["header"].update(
                _panel_header(self._current_model, eta, self._simulated))
            self._layout["progress"].update(self.progress.render())
            self._layout["results"].update(_panel_results(self._results))
            self._layout["hardware"].update(_panel_hardware(self._hw_snap))
            self._layout["stats"].update(_panel_stats(self._stats))
            self._layout["logpane"].update(_panel_log(self._log_lines))
            self._layout["footer"].update(_panel_footer(elapsed))
        except Exception:
            pass

    def _calc_eta(self) -> str:
        done  = self._stats.get("total_tested", 0)
        total = self._stats.get("total_models", 0)
        if done <= 0 or total <= 0:
            return "--:--"
        elapsed = max(time.time() - self._start_time, 1.0)
        # Use recent models (last 5) for moving average to adapt to model size changes
        if len(self._results) >= 2:
            recent_n = min(5, len(self._results))
            recent_elapsed = elapsed * recent_n / max(done, 1)
            avg_per_model  = recent_elapsed / recent_n
        else:
            avg_per_model = elapsed / done
        remain = (total - done) * avg_per_model
        return str(timedelta(seconds=int(remain)))

    # ══ API pública para el Controller ══════════════════════════════════════

    def set_total_models(self, n: int) -> None:
        self._stats["total_models"] = n
        self.progress.set_total(n)
        self._refresh()

    def model_starting(self, model: str) -> None:
        self._current_model = model
        self.progress.set_model(model, 5)
        self._refresh()

    def model_progress(self, pct: float) -> None:
        self.progress.set_model(self._current_model, pct)
        self._refresh()

    def model_done(self, result: BenchmarkResult) -> None:
        self._results.append(result)
        self.progress.finish_model()
        self.progress.advance_global(1)
        self._refresh()

    def update_hardware(self, snap: HardwareSnapshot) -> None:
        self._hw_snap = snap
        self._refresh()

    def update_stats(self, stats: Dict[str, Any]) -> None:
        self._stats.update(stats)
        self._refresh()

    def log(self, line: str) -> None:
        self._log_lines.append(line)
        if len(self._log_lines) > 500:
            self._log_lines = self._log_lines[-500:]
        self._refresh()

    def print_outside(self, *args, **kwargs) -> None:
        """Print outside Live context (for important post-benchmark messages)."""
        if self._live:
            self._live.console.print(*args, **kwargs)

    def set_footer_msg(self, msg: str) -> None:
        elapsed = time.time() - self._start_time
        self._layout["footer"].update(_panel_footer(elapsed, msg))
        