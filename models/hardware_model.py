#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/hardware_model.py — Hardware monitor for NVIDIA Jetson AGX Orin.
Uses jtop on real Jetson; simulation mode for development without hardware.
Background thread with sliding history buffer and update callbacks.
"""

from __future__ import annotations

import math
import random
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, List, Optional

from models.data_model import HardwareSnapshot
from config import MONITOR_INTERVAL, MONITOR_HISTORY

# ── Conditional jtop import ──────────────────────────────────────────────────
try:
    from jtop import jtop, JtopException
    JTOP_AVAILABLE = True
except ImportError:
    JTOP_AVAILABLE = False
    JtopException = Exception


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: safe access to nested structures
# ═══════════════════════════════════════════════════════════════════════════════

def _safe(obj: Any, keys: list, default: Any = 0) -> Any:
    """
    Safely navigate dicts, lists or object attributes.
    Returns default if any step fails.
    """
    cur = obj
    for k in keys:
        try:
            if isinstance(cur, dict):
                cur = cur[k]
            elif isinstance(cur, list) and isinstance(k, int):
                cur = cur[k]
            elif hasattr(cur, str(k)):
                cur = getattr(cur, str(k))
            else:
                return default
        except (KeyError, IndexError, TypeError, AttributeError):
            return default
    return cur


# ═══════════════════════════════════════════════════════════════════════════════
# HARDWARE MONITOR
# ═══════════════════════════════════════════════════════════════════════════════

class HardwareMonitor:
    """
    Daemon thread that samples hardware every `interval` seconds.

    Public API:
        monitor.start()                  → start thread
        monitor.stop()                   → stop thread
        monitor.current                  → most recent HardwareSnapshot
        monitor.history                  → list of HardwareSnapshots
        monitor.get_temp_series()        → dict of temperature time series
        monitor.get_power_series()       → dict of power time series
        monitor.on_update(callback)      → register callback(snap) on each sample
        monitor.wait_for_cooldown(t, w)  → block until GPU < t°C (or timeout)

    On non-Jetson hardware or if jtop is unavailable, activates simulation
    mode automatically.
    """

    def __init__(
        self,
        interval:   float = MONITOR_INTERVAL,
        history_sz: int   = MONITOR_HISTORY,
        simulate:   bool  = False,
    ):
        self.interval    = interval
        self._simulate   = simulate or not JTOP_AVAILABLE
        self._lock       = threading.RLock()
        self._stop_evt   = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._current    = HardwareSnapshot()
        self._history: Deque[HardwareSnapshot] = deque(maxlen=history_sz)
        self._callbacks: List[Callable[[HardwareSnapshot], None]] = []
        self._error:     Optional[str] = None
        self._sample_count = 0

    # ── Ciclo de vida ────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        target = self._loop_simulate if self._simulate else self._loop_jtop
        self._thread = threading.Thread(
            target=target,
            daemon=True,
            name="HardwareMonitor",
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    # ── Registro de callbacks ────────────────────────────────────────────────

    def on_update(self, callback: Callable[[HardwareSnapshot], None]) -> None:
        """Register a function called on each new snapshot."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable) -> None:
        try:
            self._callbacks.remove(callback)
        except ValueError:
            pass

    # ── Acceso a datos ───────────────────────────────────────────────────────

    @property
    def current(self) -> HardwareSnapshot:
        with self._lock:
            return self._current

    @property
    def history(self) -> List[HardwareSnapshot]:
        with self._lock:
            return list(self._history)

    @property
    def is_simulated(self) -> bool:
        return self._simulate

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def last_error(self) -> Optional[str]:
        return self._error

    # ── Series temporales ────────────────────────────────────────────────────

    def get_temp_series(self) -> dict:
        h = self.history
        if not h:
            return {"ts": [], "gpu": [], "cpu": [], "soc": [], "tj": []}
        return {
            "ts":  [s.ts  for s in h],
            "gpu": [s.temp_gpu for s in h],
            "cpu": [s.temp_cpu for s in h],
            "soc": [s.temp_soc for s in h],
            "tj":  [s.temp_tj  for s in h],
        }

    def get_power_series(self) -> dict:
        h = self.history
        if not h:
            return {"ts": [], "gpu_soc": [], "cpu_cv": [], "sys_5v0": [], "total": []}
        return {
            "ts":      [s.ts            for s in h],
            "gpu_soc": [s.vdd_gpu_soc   for s in h],
            "cpu_cv":  [s.vdd_cpu_cv    for s in h],
            "sys_5v0": [s.vin_sys_5v0   for s in h],
            "total":   [s.power_total   for s in h],
        }

    def get_gpu_series(self) -> dict:
        h = self.history
        return {
            "ts":   [s.ts          for s in h],
            "load": [s.gpu_load    for s in h],
            "freq": [s.gpu_freq    for s in h],
            "emc":  [s.emc_freq    for s in h],
        }

    def get_ram_series(self) -> dict:
        h = self.history
        return {
            "ts":         [s.ts            for s in h],
            "ram_used":   [s.ram_used      for s in h],
            "ram_pct":    [s.ram_percent   for s in h],
            "swap_used":  [s.swap_used     for s in h],
        }

    # ── Cooldown ─────────────────────────────────────────────────────────────

    def wait_for_cooldown(
        self,
        target_temp: float,
        max_wait:    float = 300.0,
        check_interval: float = 5.0,
    ) -> bool:
        """
        Block until temp_gpu <= target_temp or max_wait expires.
        Returns True if target temperature was reached.
        """
        t0 = time.time()
        while time.time() - t0 < max_wait:
            if self._current.temp_gpu <= target_temp:
                return True
            time.sleep(check_interval)
        return False

    # ── Loop jtop ─────────────────────────────────────────────────────────────

    def _loop_jtop(self) -> None:
        """Main loop using jtop on real hardware."""
        try:
            with jtop() as jetson:
                if not jetson.ok():
                    raise JtopException("jtop.ok() == False")
                while not self._stop_evt.is_set():
                    snap = self._extract_jtop(jetson)
                    self._publish(snap)
                    time.sleep(self.interval)
        except Exception as e:
            self._error = str(e)
            # Automatic fallback to simulation
            self._simulate = True
            self._loop_simulate()

    def _extract_jtop(self, j) -> HardwareSnapshot:
        """Extract all data from an active jtop object."""
        snap = HardwareSnapshot()
        snap.ts = time.time()

        # ── CPU ─────────────────────────────────────────────────────────────
        cpu_list = _safe(j.cpu, ["cpu"], [])
        for i in range(12):
            if i < len(cpu_list):
                snap.cpu_loads.append(float(_safe(cpu_list, [i, "load"], 0.0)))
                snap.cpu_freqs.append(int  (_safe(cpu_list, [i, "freq"], 0)))
            else:
                snap.cpu_loads.append(0.0)
                snap.cpu_freqs.append(0)

        # ── GPU ──
        snap.gpu_load = float(_safe(j.gpu, ["gpu", "status", "load"], 0.0))
        snap.gpu_freq = int  (_safe(j.gpu, ["gpu", "freq", "cur"],    0))
        snap.emc_freq = int  (j.stats.get("EMC", 0) if hasattr(j, "stats") else 0)

        # ── Power ──
        rails = j.power.get("rail", {}) if hasattr(j, "power") else {}
        snap.vdd_gpu_soc = int(_safe(rails, ["VDD_GPU_SOC", "power"], 0))
        snap.vdd_cpu_cv  = int(_safe(rails, ["VDD_CPU_CV",  "power"], 0))
        snap.vin_sys_5v0 = int(_safe(rails, ["VIN_SYS_5V0", "power"], 0))
        snap.power_total = int(_safe(j.power, ["tot", "power"], 0)) if hasattr(j, "power") else (
            snap.vdd_gpu_soc + snap.vdd_cpu_cv + snap.vin_sys_5v0
        )

        # ── Temperatures ──
        temps = j.temperature if hasattr(j, "temperature") else {}
        snap.temp_gpu = float(_safe(temps, ["gpu", "temp"], 0))
        snap.temp_cpu = float(_safe(temps, ["cpu", "temp"], 0))
        snap.temp_soc = float(_safe(temps, ["soc", "temp"], 0))
        snap.temp_tj  = float(_safe(temps, ["tj",  "temp"], 0))

        # ── Memory ──
        mem = j.memory if hasattr(j, "memory") else {}
        snap.ram_used   = _safe(mem, ["RAM", "used"], 0) / 1024.0
        snap.ram_total  = _safe(mem, ["RAM", "tot"],  1) / 1024.0
        snap.swap_used  = _safe(mem, ["SWAP","used"], 0) / 1024.0
        snap.swap_total = _safe(mem, ["SWAP","tot"],  1) / 1024.0

        # ── Fan ──
        fans = _safe(j, ["fan"], {})
        spd, rpm, prof = [], [], []
        if isinstance(fans, dict):
            for fn, fd in fans.items():
                if isinstance(fd, dict):
                    for i, s in enumerate(fd.get("speed", [])):
                        spd.append(f"{fn}_{i}={s}")
                    for i, r in enumerate(fd.get("rpm", [])):
                        rpm.append(f"{fn}_{i}={r}")
                    prof.append(f"{fn}={fd.get('profile','')}")
        snap.fan_speeds  = "|".join(spd)
        snap.fan_rpms    = "|".join(rpm)
        snap.fan_profile = "|".join(prof)

        # ── Disk ──
        disk = j.disk if hasattr(j, "disk") else {}
        snap.disk_total    = float(_safe(disk, ["total"],    0))
        snap.disk_used     = float(_safe(disk, ["used"],     0))
        snap.disk_available = float(_safe(disk, ["available"], 0))

        # ── Uptime ─────────────────────────────────────────────────────────
        uptime = _safe(j, ["uptime"], None)
        if uptime is not None and hasattr(uptime, "total_seconds"):
            snap.uptime_seconds = uptime.total_seconds()
        elif isinstance(uptime, (int, float)):
            snap.uptime_seconds = float(uptime)

        return snap

    # ── Loop simulación ───────────────────────────────────────────────────────

    def _loop_simulate(self) -> None:
        """
        Generate synthetic but realistic hardware data.
        Simulates gradual heating, load variations and power cycles.
        """
        t0        = time.time()
        rng       = random.Random(42)      # fixed seed for partial reproducibility
        phase_gpu = rng.uniform(0, math.pi * 2)
        phase_cpu = rng.uniform(0, math.pi * 2)

        # Baseline Jetson AGX Orin
        BASE_TEMP_GPU   = 44.0
        BASE_TEMP_CPU   = 40.0
        BASE_RAM_MB     = 6_000.0
        BASE_POWER      = 11_000
        GPU_FREQ_MAX    = 1300
        CPU_FREQ_MAX    = 2201
        RAMP_TIME       = 120.0       # segundos para calentamiento completo
        LOAD_CYCLE      = 60.0        # período de ciclo de carga

        while not self._stop_evt.is_set():
            t    = time.time()
            age  = t - t0
            ramp = min(age / RAMP_TIME, 1.0)                # 0→1 en 2 minutos

            # Factor de ciclo de carga (simula que el modelo trabaja)
            load_wave = 0.5 + 0.5 * math.sin(2 * math.pi * age / LOAD_CYCLE + phase_gpu)

            snap = HardwareSnapshot()
            snap.ts = t

            # CPU
            for i in range(12):
                cpu_ramp = ramp * 0.7 + load_wave * 0.3
                load = 10.0 + cpu_ramp * 70.0 + rng.gauss(0, 3)
                freq = int(400 + cpu_ramp * (CPU_FREQ_MAX - 400) + rng.gauss(0, 50))
                snap.cpu_loads.append(round(max(5.0, min(99.9, load)), 1))
                snap.cpu_freqs.append(max(400, min(CPU_FREQ_MAX, freq)))

            # GPU
            gpu_load = 30.0 + ramp * 55.0 + load_wave * 10.0 + rng.gauss(0, 2)
            snap.gpu_load = round(max(20.0, min(99.0, gpu_load)), 1)
            gpu_freq_ramp = ramp * 0.8 + load_wave * 0.2
            snap.gpu_freq = int(450 + gpu_freq_ramp * (GPU_FREQ_MAX - 450) + rng.gauss(0, 30))
            snap.emc_freq = int(1600 + ramp * 1600 + rng.gauss(0, 100))

            # Power (correlates with GPU load)
            pwr_factor      = 0.4 + ramp * 0.5 + load_wave * 0.1
            snap.vdd_gpu_soc = int(BASE_POWER * 0.58 * pwr_factor + rng.gauss(0, 300))
            snap.vdd_cpu_cv  = int(BASE_POWER * 0.26 * pwr_factor + rng.gauss(0, 150))
            snap.vin_sys_5v0 = int(BASE_POWER * 0.16 * pwr_factor + rng.gauss(0, 80))
            snap.power_total = snap.vdd_gpu_soc + snap.vdd_cpu_cv + snap.vin_sys_5v0

            # Temperatures (follow thermal inertia)
            temp_inertia     = 1.0 - math.exp(-age / 40.0)   # escala en ~40s
            snap.temp_gpu    = BASE_TEMP_GPU + temp_inertia * 30.0 * ramp + rng.gauss(0, 0.5)
            snap.temp_cpu    = BASE_TEMP_CPU + temp_inertia * 22.0 * ramp + rng.gauss(0, 0.4)
            snap.temp_soc    = snap.temp_cpu - 3.0 + rng.gauss(0, 0.3)
            snap.temp_tj     = max(snap.temp_gpu, snap.temp_cpu, snap.temp_soc)

            # RAM memory (grows while model is loaded)
            ram_model        = BASE_RAM_MB + ramp * 10_000.0
            snap.ram_used    = ram_model + rng.gauss(0, 200)
            snap.ram_total   = 32_768.0
            snap.swap_used   = 512.0 + ramp * 3_000.0
            snap.swap_total  = 16_384.0

            # Fan (correlates with temperature)
            fan_speed = int(30 + (snap.temp_gpu - 40) * 2.5)
            fan_speed = max(0, min(100, fan_speed))
            snap.fan_speeds  = f"fan0_0={fan_speed}"
            snap.fan_rpms    = f"fan0_0={fan_speed * 20}"
            snap.fan_profile = "fan0=performance"

            # Disco
            snap.disk_total     = 500.0
            snap.disk_used      = 220.0 + ramp * 5.0
            snap.disk_available = snap.disk_total - snap.disk_used

            snap.uptime_seconds = age

            self._publish(snap)
            time.sleep(self.interval)

    # ── Publicar snapshot ────────────────────────────────────────────────────

    def _publish(self, snap: HardwareSnapshot) -> None:
        with self._lock:
            self._current = snap
            self._history.append(snap)
            self._sample_count += 1
        for cb in self._callbacks:
            try:
                cb(snap)
            except Exception:
                pass    # never let a callback break the monitor
                