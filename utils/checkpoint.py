#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils/checkpoint.py — Atomic checkpoint management v3.0
Multi-run support: saves completed runs per model.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import CHECKPOINT_FILE


class CheckpointManager:
    SCHEMA_VERSION = "3.0"

    def __init__(self, path: Path = CHECKPOINT_FILE):
        self.path    = Path(path)
        self.run_id  = int(time.time())
        self._data: Dict[str, Any] = self._empty()

    def _empty(self) -> Dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "run_id":         self.run_id,
            "created_at":     time.strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at":     time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_models":   0,
            "completed":      [],   # list of model names
            "failed":         {},   # {model: error_msg}
            "stats":          {},
        }

    def load(self) -> bool:
        if not self.path.exists():
            return False
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("schema_version", "").startswith("3"):
                self._data = data
                self.run_id = data.get("run_id", self.run_id)
                return True
            # Migrate old schema
            self._data = self._empty()
            self._data["completed"] = data.get("completed_models", data.get("completed", []))
            return bool(self._data["completed"])
        except Exception:
            return False

    def save(self) -> None:
        self._data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)
        tmp.replace(self.path)

    def clear(self) -> None:
        self._data = self._empty()
        if self.path.exists():
            self.path.unlink()

    def set_total_models(self, n: int) -> None:
        self._data["total_models"] = n
        self.save()

    def mark_completed(self, model: str) -> None:
        if model not in self._data["completed"]:
            self._data["completed"].append(model)
        self._data["failed"].pop(model, None)
        self.save()

    def mark_failed(self, model: str, error: str) -> None:
        self._data["failed"][model] = error
        self.save()

    def get_completed_models(self) -> List[str]:
        return list(self._data.get("completed", []))

    def update_stats(self, stats: Dict[str, Any]) -> None:
        self._data["stats"] = stats
        self.save()

    def count_done(self) -> int:
        """Returns the number of completed models."""
        return len(self._data.get("completed", []))

    def summary(self) -> str:
        done  = len(self._data.get("completed", []))
        total = self._data.get("total_models", "?")
        failed= len(self._data.get("failed", {}))
        return f"{done}/{total} completed, {failed} failed"

    def detailed_report(self) -> str:
        lines = [f"Run ID: {self.run_id}", self.summary()]
        failed = self._data.get("failed", {})
        if failed:
            lines.append(f"Failed: {', '.join(failed.keys())}")
        return " | ".join(lines)
