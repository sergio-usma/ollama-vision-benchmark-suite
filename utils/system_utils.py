#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/system_utils.py — Operating system utilities.
Memory cleanup, scheduled shutdown, environment checks.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple


class SystemUtils:
    """OS operations encapsulated safely."""

    # ── Memory ────────────────────────────────────────────────────────────────

    @staticmethod
    def is_root() -> bool:
        """Checks if the process is running as root."""
        try:
            return os.getuid() == 0
        except AttributeError:
            return False  # Windows

    @staticmethod
    def sync_filesystem() -> bool:
        """Calls sync(1) to flush write buffers."""
        try:
            subprocess.run(
                ["sync"],
                check=True,
                capture_output=True,
                timeout=10,
            )
            return True
        except Exception:
            return False

    @staticmethod
    def drop_caches() -> Tuple[bool, str]:
        """
        Releases Linux kernel page cache, dentries and inodes.
        Requires root.
        Returns (success, message).
        """
        if not SystemUtils.is_root():
            return False, "Requires root for drop_caches"
        try:
            SystemUtils.sync_filesystem()
            with open("/proc/sys/vm/drop_caches", "w") as f:
                f.write("3\n")
            return True, "Caches cleared (drop_caches=3)"
        except PermissionError:
            return False, "Permission denied — /proc/sys/vm/drop_caches"
        except FileNotFoundError:
            return False, "Not available — /proc/sys/vm/drop_caches does not exist"
        except Exception as e:
            return False, f"Error: {e}"

    @staticmethod
    def clean_memory(sleep_after: float = 2.0) -> str:
        """
        Full memory cleanup sequence.
        Returns a summary of what was performed.
        """
        actions = []

        ok, msg = SystemUtils.drop_caches()
        actions.append(f"drop_caches: {'✓' if ok else '✗'} {msg}")

        if sleep_after > 0:
            time.sleep(sleep_after)

        return " | ".join(actions)

    # ── Shutdown ──────────────────────────────────────────────────────────────

    @staticmethod
    def schedule_shutdown(minutes: int = 5) -> Tuple[bool, str]:
        """
        Schedules system shutdown in `minutes` minutes.
        Uses Linux shutdown(8).
        Returns (success, message).
        """
        try:
            result = subprocess.run(
                ["shutdown", "-h", f"+{minutes}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return True, f"Shutdown scheduled in {minutes} minutes"
            return False, result.stderr.strip() or "Unknown error"
        except FileNotFoundError:
            return False, "shutdown not available on this system"
        except Exception as e:
            return False, str(e)

    @staticmethod
    def cancel_shutdown() -> Tuple[bool, str]:
        """Cancels a scheduled shutdown."""
        try:
            result = subprocess.run(
                ["shutdown", "-c"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0, "Shutdown cancelled"
        except Exception as e:
            return False, str(e)

    # ── System information ────────────────────────────────────────────────────

    @staticmethod
    def get_hostname() -> str:
        try:
            import socket
            return socket.gethostname()
        except Exception:
            return "unknown"

    @staticmethod
    def get_kernel_version() -> str:
        try:
            return subprocess.check_output(["uname", "-r"], text=True).strip()
        except Exception:
            return "unknown"

    @staticmethod
    def get_disk_info(path: str = "/") -> dict:
        """Returns disk info for the specified path."""
        try:
            import shutil
            usage = shutil.disk_usage(path)
            return {
                "total_gb":    round(usage.total / 1e9, 1),
                "used_gb":     round(usage.used  / 1e9, 1),
                "free_gb":     round(usage.free  / 1e9, 1),
                "percent_used": round(usage.used / usage.total * 100, 1),
            }
        except Exception:
            return {}

    @staticmethod
    def get_memory_info() -> dict:
        """Returns RAM info from /proc/meminfo."""
        try:
            info: dict = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(":")
                        val = int(parts[1]) / 1024.0  # kB → MB
                        info[key] = round(val, 1)
            return {
                "total_mb":     info.get("MemTotal", 0),
                "available_mb": info.get("MemAvailable", 0),
                "free_mb":      info.get("MemFree", 0),
                "swap_total_mb": info.get("SwapTotal", 0),
                "swap_free_mb":  info.get("SwapFree", 0),
            }
        except Exception:
            return {}

    @staticmethod
    def check_ollama_running() -> bool:
        """Checks if the ollama process is running."""
        try:
            result = subprocess.run(
                ["pgrep", "-x", "ollama"],
                capture_output=True,
                timeout=3,
            )
            return result.returncode == 0
        except Exception:
            return False

    @staticmethod
    def get_gpu_driver_version() -> str:
        """Returns NVIDIA driver version if available."""
        try:
            out = subprocess.check_output(
                ["cat", "/proc/driver/nvidia/version"],
                text=True,
                timeout=3,
            )
            for line in out.splitlines():
                if "Kernel Module" in line:
                    parts = line.split()
                    if len(parts) >= 3:
                        return parts[2]
        except Exception:
            pass
        return "unknown"

    @staticmethod
    def environment_check() -> dict:
        """
        Checks the full environment.
        Returns a dict with the status of each component.
        """
        from config import OLLAMA_API, ASSETS_DIR
        import requests

        checks = {}

        # Root
        checks["is_root"]          = SystemUtils.is_root()
        checks["hostname"]         = SystemUtils.get_hostname()
        checks["kernel"]           = SystemUtils.get_kernel_version()

        # Ollama
        try:
            r = requests.get(OLLAMA_API + "tags", timeout=3)
            checks["ollama_alive"] = r.status_code == 200
            checks["ollama_url"]   = OLLAMA_API
        except Exception:
            checks["ollama_alive"] = False
            checks["ollama_url"]   = OLLAMA_API

        # jtop
        try:
            from jtop import jtop
            checks["jtop_available"] = True
        except ImportError:
            checks["jtop_available"] = False

        # Directories
        checks["assets_dir"]  = Path(ASSETS_DIR).exists()
        checks["test_image"]  = any(
            Path(ASSETS_DIR).glob(p) for p in ("*.jpg", "*.jpeg", "*.png", "*.webp")
        )

        # Memory
        mem = SystemUtils.get_memory_info()
        checks["ram_total_mb"] = mem.get("total_mb", 0)
        checks["ram_available_mb"] = mem.get("available_mb", 0)

        # Disk
        disk = SystemUtils.get_disk_info()
        checks["disk_free_gb"] = disk.get("free_gb", 0)

        return checks
    