#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/logger.py — Specialized logger with semantic levels.

Dual output:
  - Console: Rich with colors, icons and contextual styles.
  - File:    Plain text with ISO timestamps (automatic rotation).

Semantic levels (more informative than DEBUG/INFO/WARNING):
  PERF    → performance metrics
  THERMAL → temperatures
  POWER   → power consumption
  STATUS  → general positive status
  WARN    → recoverable warning
  ERROR   → error that stops or degrades
  SKIP    → model skipped by checkpoint
  RESUME  → resuming from checkpoint
  BENCH   → benchmark events (start, end, separators)
  DASH    → dashboard generated
  SHUTDOWN→ scheduled shutdown
  NET     → connectivity
  DEBUG   → debug information
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.text import Text

from config import (
    APP_NAME, LOG_FILE, LOG_LEVEL,
    LOG_MAX_BYTES, LOG_BACKUP_COUNT, LOG_ENCODING,
)


# ═══════════════════════════════════════════════════════════════════════════════
# RICH CONSOLE
# ═══════════════════════════════════════════════════════════════════════════════

_CONSOLE = Console(highlight=False, markup=True)
_CONSOLE_ERR = Console(stderr=True, highlight=False, markup=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STYLE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

class _S:
    """Centralized Rich styles."""
    PERF     = "bold bright_cyan"
    THERMAL  = "bold red"
    POWER    = "bold yellow"
    STATUS   = "bold bright_green"
    WARN     = "bold orange3"
    ERROR    = "bold red1"
    SKIP     = "dim cyan"
    RESUME   = "bold magenta"
    BENCH    = "bold bright_cyan"
    DASH     = "bold green"
    SHUTDOWN = "bold red"
    NET      = "bright_blue"
    DEBUG    = "dim white"
    MODEL    = "bold bright_cyan"
    CAT      = "italic yellow"
    DIM      = "dim white"
    GOOD     = "green"
    WARN_V   = "yellow"
    BAD      = "red"


class _P:
    """Semantic prefixes with icons."""
    PERF     = "⚡ PERF    "
    THERMAL  = "🌡  THERM  "
    POWER    = "🔋 POWER   "
    STATUS   = "✅ STATUS  "
    WARN     = "⚠️  WARN    "
    ERROR    = "❌ ERROR   "
    SKIP     = "⏭  SKIP    "
    RESUME   = "🔄 RESUME  "
    BENCH    = "📊 BENCH   "
    DASH     = "📈 DASH    "
    SHUTDOWN = "🔌 SHUTDOWN"
    NET      = "🌐 NET     "
    DEBUG    = "🔍 DEBUG   "
    MODEL    = "🤖 MODEL   "
    COOL     = "❄️  COOL    "


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK LOGGER
# ═══════════════════════════════════════════════════════════════════════════════

class BenchmarkLogger:
    """
    Dual-channel logger with semantic levels.
    Safe to instantiate multiple times (shares the same FileHandler).
    """

    def __init__(
        self,
        log_file: Path   = LOG_FILE,
        level:    str    = LOG_LEVEL,
        console:  Console = _CONSOLE,
    ):
        self._console  = console
        self._log_path = Path(log_file)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        # ── File logger with automatic rotation ──
        self._flog = logging.getLogger(f"{APP_NAME}.{id(self)}")
        self._flog.setLevel(getattr(logging, level.upper(), logging.DEBUG))
        self._flog.propagate = False

        if not self._flog.handlers:
            handler = logging.handlers.RotatingFileHandler(
                filename    = self._log_path,
                maxBytes    = LOG_MAX_BYTES,
                backupCount = LOG_BACKUP_COUNT,
                encoding    = LOG_ENCODING,
            )
            handler.setFormatter(logging.Formatter(
                "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            self._flog.addHandler(handler)

    # ── Internal emit method ─────────────────────────────────────────────────

    def _emit(
        self,
        prefix:  str,
        message: str,
        style:   str = "",
        level:   int = logging.INFO,
    ) -> None:
        """Emits to Rich console + log file."""
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # Rich console
        rich_line = f"[dim]{ts}[/dim] [{style}]{prefix}[/{style}] {message}" if style else \
                    f"[dim]{ts}[/dim] {prefix} {message}"
        try:
            self._console.print(rich_line)
        except Exception:
            pass

        # Plain file (no Rich markup)
        plain = f"{ts} {prefix} {_strip_rich(message)}"
        try:
            self._flog.log(level, plain)
        except Exception:
            pass

    # ── Public semantic methods ───────────────────────────────────────────────

    def perf(
        self,
        model: str,
        tps:   float,
        ram_mb: float,
        power_mw: int,
        tok_per_w: float,
        score: float,
        recommendation: str,
    ) -> None:
        """Performance metrics for a model."""
        tps_color = _S.GOOD if tps >= 20 else (_S.WARN_V if tps >= 10 else _S.BAD)
        rec_color = _S.GOOD if recommendation.startswith("KEEP") else \
                    (_S.BAD if recommendation == "REMOVE" else _S.WARN_V)
        msg = (
            f"[{_S.MODEL}]{model}[/{_S.MODEL}]  "
            f"TPS=[{tps_color}]{tps:.2f}[/{tps_color}]  "
            f"RAM={ram_mb:.0f}MB  "
            f"PWR={power_mw:,}mW  "
            f"tok/W=[bold]{tok_per_w:.2f}[/bold]  "
            f"Score=[bold]{score:.1f}[/bold]  "
            f"→ [{rec_color}]{recommendation}[/{rec_color}]"
        )
        self._emit(_P.PERF, msg, _S.PERF)

    def thermal(
        self,
        model: str,
        gpu: float,
        cpu: float,
        soc: float,
        tj:  float,
    ) -> None:
        """Thermal report for a model."""
        def tc(t: float) -> str:
            return _S.BAD if t > 80 else (_S.WARN_V if t > 65 else _S.GOOD)
        msg = (
            f"[{_S.MODEL}]{model}[/{_S.MODEL}]  "
            f"GPU=[{tc(gpu)}]{gpu:.1f}°C[/{tc(gpu)}]  "
            f"CPU=[{tc(cpu)}]{cpu:.1f}°C[/{tc(cpu)}]  "
            f"SOC=[{tc(soc)}]{soc:.1f}°C[/{tc(soc)}]  "
            f"TJ=[{tc(tj)}]{tj:.1f}°C[/{tc(tj)}]"
        )
        self._emit(_P.THERMAL, msg, _S.THERMAL)

    def power(
        self,
        model: str,
        gpu_soc: int,
        cpu_cv: int,
        sys5v0: int,
        total: int,
    ) -> None:
        """Power consumption report."""
        color = _S.GOOD if total < 15_000 else (_S.WARN_V if total < 25_000 else _S.BAD)
        msg = (
            f"[{_S.MODEL}]{model}[/{_S.MODEL}]  "
            f"GPU+SOC={gpu_soc:,}mW  CPU+CV={cpu_cv:,}mW  "
            f"SYS5V0={sys5v0:,}mW  "
            f"Total=[{color}]{total:,}mW ({total/1000:.1f}W)[/{color}]"
        )
        self._emit(_P.POWER, msg, _S.POWER)

    def model_start(
        self,
        model:    str,
        category: str,
        idx:      int,
        total:    int,
    ) -> None:
        """Start of benchmark for a model."""
        msg = (
            f"[dim][{idx:3}/{total}][/dim]  "
            f"[{_S.MODEL}]{model}[/{_S.MODEL}]  "
            f"cat=[{_S.CAT}]{category}[/{_S.CAT}]"
        )
        self._emit(_P.MODEL, msg)

    def model_skip(self, model: str, reason: str = "checkpoint") -> None:
        """Model skipped (already processed in a previous run)."""
        self._emit(_P.SKIP, f"[{_S.MODEL}]{model}[/{_S.MODEL}] — {reason}", _S.SKIP)

    def model_error(self, model: str, error: str) -> None:
        """Error in a model."""
        self._emit(_P.ERROR, f"[{_S.MODEL}]{model}[/{_S.MODEL}] — {error[:150]}", _S.ERROR,
                   level=logging.ERROR)

    def status(self, msg: str) -> None:
        self._emit(_P.STATUS, msg, _S.STATUS)

    def warn(self, msg: str) -> None:
        self._emit(_P.WARN, msg, _S.WARN, level=logging.WARNING)

    def info(self, msg: str) -> None:
        self._emit("ℹ  INFO    ", msg, _S.DIM)

    def debug(self, msg: str) -> None:
        self._emit(_P.DEBUG, msg, _S.DEBUG, level=logging.DEBUG)

    def net(self, msg: str) -> None:
        self._emit(_P.NET, msg, _S.NET)

    def resume(self, msg: str) -> None:
        self._emit(_P.RESUME, msg, _S.RESUME)

    def cooldown(self, current: float, target: float) -> None:
        """Waiting for cooldown."""
        self._emit(
            _P.COOL,
            f"GPU={current:.1f}°C → waiting for {target:.0f}°C...",
            _S.THERMAL,
        )

    def dashboard_saved(self, path: Path) -> None:
        self._emit(_P.DASH, f"[underline]{path}[/underline]", _S.DASH)

    def shutdown_scheduled(self, minutes: int) -> None:
        self._emit(
            _P.SHUTDOWN,
            f"System will shut down in {minutes} minutes",
            _S.SHUTDOWN,
        )

    def bench_start(self, total: int, resumed: bool = False) -> None:
        suffix = " [bold magenta](RESUMING)[/bold magenta]" if resumed else ""
        self._console.rule(
            f"[bold bright_cyan]🚀 BENCHMARK STARTED — {total} models{suffix}[/bold bright_cyan]"
        )
        self._flog.info(f"=== BENCHMARK START — {total} models ===")

    def bench_end(
        self,
        tested:   int,
        success:  int,
        elapsed_min: float,
    ) -> None:
        self._console.rule(
            f"[bold green]🏁 COMPLETED — {success}/{tested} OK — "
            f"{elapsed_min:.1f} min[/bold green]"
        )
        self._flog.info(f"=== BENCHMARK END — {success}/{tested} OK — {elapsed_min:.1f}min ===")

    def separator(self, label: str = "") -> None:
        self._console.rule(f"[dim]{label}[/dim]" if label else "")

    def print_table(self, rows: list, title: str = "") -> None:
        """Prints a list of dicts as a Rich Table."""
        if not rows:
            return
        from rich.table import Table
        from rich import box as rbox
        t = Table(
            title=title,
            box=rbox.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold bright_cyan",
        )
        for col in rows[0].keys():
            t.add_column(col, style="white")
        for row in rows:
            t.add_row(*[str(v) for v in row.values()])
        self._console.print(t)

    @property
    def console(self) -> Console:
        return self._console

    @property
    def log_path(self) -> Path:
        return self._log_path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _strip_rich(text: str) -> str:
    """Removes Rich markup tags for plain text output."""
    import re
    return re.sub(r"\[/?[^\[\]]*\]", "", text)
