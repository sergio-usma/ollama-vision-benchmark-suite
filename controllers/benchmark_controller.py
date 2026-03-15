#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
controllers/benchmark_controller.py — Orchestrator v4.0
- Hardware captured DURING inference (historical window average)
- Adaptive cooldown (skips if GPU is already cool)
- Filtering by category and model name (--filter / --models)
- Per-instance configurable timeout (--timeout)
- Exports summary.json on completion
"""
from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import (
    APP_NAME, APP_VERSION,
    COOLDOWN_TEMP, CSV_OUTPUT,
    DASHBOARD_UPDATE_EVERY, MEMORY_CLEAN_SLEEP,
    NUM_RUNS, WARMUP_RUN, INTER_RUN_SLEEP, USE_MULTI_PROMPT,
    SHUTDOWN_DELAY_MIN, OLLAMA_TEMP, OLLAMA_CTX, OLLAMA_TIMEOUT,
    classify_model, get_min_tps, get_prompts_for_category,
)
from models.data_model import BenchmarkResult, CSVManager, HardwareSnapshot, RunStat, StatsAggregator
from models.hardware_model import HardwareMonitor
from models.ollama_model import OllamaClient
from utils.checkpoint import CheckpointManager
from utils.logger import BenchmarkLogger
from utils.system_utils import SystemUtils
from views.dashboard_view import DashboardView
from views.terminal_view import TerminalView


class BenchmarkController:
    """
    Central benchmark orchestrator v4.0.
    - Dynamic model discovery (Ollama API + fallback 'ollama list')
    - Warmup run + N measurement runs per model
    - Prompt rotation between runs to avoid KV-cache bias
    - Category percentile-based scoring
    - Hardware sampled during inference (historical window, not post-run snapshot)
    - Adaptive cooldown: skips if GPU already below COOLDOWN_TEMP
    - Filtering by category and/or model name
    - Exports machine-readable JSON on completion
    """

    def __init__(
        self,
        csv_path:           Path           = CSV_OUTPUT,
        simulate_hw:        bool           = False,
        auto_shutdown:      bool           = False,
        resume:             bool           = True,
        live_dashboard:     bool           = True,
        open_browser:       bool           = False,
        num_runs:           int            = NUM_RUNS,
        filter_categories:  Optional[List[str]] = None,
        filter_models:      Optional[List[str]] = None,
        timeout:            Optional[int]  = None,
    ):
        self.simulate_hw       = simulate_hw
        self.auto_shutdown     = auto_shutdown
        self.resume            = resume
        self.live_dashboard    = live_dashboard
        self.open_browser      = open_browser
        self.num_runs          = max(1, num_runs)
        self.filter_categories = [c.upper() for c in filter_categories] if filter_categories else []
        self.filter_models     = [m.lower() for m in filter_models] if filter_models else []
        self._timeout          = timeout or OLLAMA_TIMEOUT

        self.ollama     = OllamaClient(timeout=self._timeout)
        self.hw_mon     = HardwareMonitor(simulate=simulate_hw)
        self.csv_mgr    = CSVManager(Path(csv_path))
        self.aggregator = StatsAggregator()

        self.checkpoint = CheckpointManager()
        self.logger     = BenchmarkLogger()
        self.sysutils   = SystemUtils()

        self.terminal   = TerminalView(simulated=simulate_hw)
        self.dashboard  = DashboardView()

        self._start_time:  float = 0.0
        self._interrupted: bool  = False
        self._run_id:      int   = int(time.time())

    # ─── PUNTO DE ENTRADA ──────────────────────────────────────────────────────

    def run(self) -> int:
        self._start_time = time.time()
        self._setup_signals()

        if not self._pre_flight_check():
            return 1

        models_raw = self._discover_models()
        if not models_raw:
            self.logger.warn("No models found in Ollama. Run 'ollama pull <model>'")
            return 1

        # Apply filters before counting total
        models_raw = self._apply_filters(models_raw)
        if not models_raw:
            self.logger.warn(
                "No models match the applied filters. "
                "Check --filter / --models."
            )
            return 1

        total = len(models_raw)
        resumed, skip_list = self._manage_checkpoint(models_raw, total)

        self.hw_mon.start()
        self.hw_mon.on_update(self.terminal.update_hardware)
        self.terminal.set_total_models(total)
        self.terminal.start()
        self.logger.bench_start(total, resumed)

        try:
            self._benchmark_loop(models_raw, total, skip_list)
        except KeyboardInterrupt:
            self._handle_interrupt()
        except Exception as e:
            import traceback
            self.logger.model_error("CONTROLLER", f"Error fatal: {e}\n{traceback.format_exc()}")
        finally:
            self.terminal.stop()
            self.hw_mon.stop()

        elapsed = time.time() - self._start_time
        self._finalize(elapsed)

        if self.auto_shutdown and not self._interrupted:
            ok, msg = self.sysutils.schedule_shutdown(SHUTDOWN_DELAY_MIN)
            self.logger.shutdown_scheduled(SHUTDOWN_DELAY_MIN)
            if not ok:
                self.logger.warn(f"No se pudo programar apagado: {msg}")

        return 0

    # ─── FILTRADO DE MODELOS ────────────────────────────────────────────────────

    def _apply_filters(self, models_raw: List[Dict]) -> List[Dict]:
        """
        Apply category and name filters.
        --filter CODE,MATH     → only models in those categories
        --models devstral,phi4 → only those models (partial match)
        """
        if not self.filter_categories and not self.filter_models:
            return models_raw  # no filters

        filtered = []
        for m in models_raw:
            name = m["name"]
            cat  = classify_model(name)

            # Category filter
            if self.filter_categories and cat not in self.filter_categories:
                continue

            # Name filter (partial match, case-insensitive)
            if self.filter_models:
                name_lc = name.lower()
                if not any(f in name_lc for f in self.filter_models):
                    continue

            filtered.append(m)

        if self.filter_categories:
            self.logger.info(f"Filtro activo: categorías={self.filter_categories} → {len(filtered)} modelos")
        if self.filter_models:
            self.logger.info(f"Filtro activo: modelos={self.filter_models[:5]} → {len(filtered)} modelos")

        return filtered

    # ─── DESCUBRIMIENTO DINÁMICO DE MODELOS ────────────────────────────────────

    def _discover_models(self) -> List[Dict]:
        """
        Get installed models with their size in GB.
        1. Try via Ollama API (/api/tags)
        2. Fallback: run 'ollama list' in shell
        """
        try:
            models = self.ollama.get_models()
            if models:
                # Enrich with human-readable size in GB
                for m in models:
                    size_b = m.get("size", 0)
                    m["size_gb"] = round(size_b / 1e9, 1) if size_b else 0.0
                self.logger.net(f"Discovered {len(models)} models via Ollama API")
                return models
        except Exception as e:
            self.logger.warn(f"Ollama API did not respond: {e} — trying 'ollama list'...")

        # Fallback: ollama list
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                models = []
                for line in result.stdout.strip().splitlines()[1:]:  # Skip header
                    parts = line.split()
                    if parts:
                        name = parts[0]
                        size_str = parts[2] if len(parts) > 2 else "0"
                        # Parsear "4.7GB" → 4.7
                        try:
                            size_gb = float(size_str.upper().replace("GB", "").replace("MB", "")) / (
                                1 if "MB" not in size_str.upper() else 1024
                            )
                        except ValueError:
                            size_gb = 0.0
                        models.append({"name": name, "size": 0, "size_gb": size_gb})
                self.logger.net(f"Discovered {len(models)} models via 'ollama list'")
                return models
            else:
                self.logger.warn(f"'ollama list' failed: {result.stderr[:200]}")
        except FileNotFoundError:
            self.logger.warn("'ollama' not in PATH. Make sure Ollama is installed.")
        except Exception as e:
            self.logger.warn(f"Error running 'ollama list': {e}")

        return []

    # ─── PRE-FLIGHT ────────────────────────────────────────────────────────────

    def _pre_flight_check(self) -> bool:
        self.logger.info(f"Starting {APP_NAME} v{APP_VERSION}")
        self.logger.info(
            f"Config: {self.num_runs} runs/model, warmup={WARMUP_RUN}, "
            f"multi-prompt={USE_MULTI_PROMPT}, timeout={self._timeout}s"
        )

        if self.simulate_hw:
            self.logger.warn("SIMULATION mode active — synthetic hardware data")

        self.logger.net("Checking Ollama...")
        if not self.ollama.is_alive():
            self.logger.model_error("SYSTEM", "Ollama not responding. Run: 'ollama serve'")
            return False

        version = self.ollama.get_version()
        self.logger.net(f"Ollama v{version} — OK")

        if not self.simulate_hw:
            try:
                from jtop import jtop
                self.logger.status("jtop available — real hardware")
            except ImportError:
                self.logger.warn("jtop not available → simulation mode. Install: sudo pip install -U jetson-stats")
                self.simulate_hw = True
                self.hw_mon      = HardwareMonitor(simulate=True)

        if not self.sysutils.is_root():
            self.logger.warn("No root → drop_caches disabled. For better precision: sudo python3 main.py")

        return True

    # ─── CHECKPOINT ────────────────────────────────────────────────────────────

    def _manage_checkpoint(self, models_raw: list, total: int) -> Tuple[bool, List[str]]:
        skip_list: List[str] = []
        resumed = False

        if self.resume and self.checkpoint.load():
            completed = self.checkpoint.get_completed_models()
            skip_list = [m["name"] for m in models_raw if m["name"] in completed]
            if skip_list:
                resumed = True
                self.logger.resume(
                    f"Checkpoint: {len(skip_list)}/{total} already completed | {self.checkpoint.summary()}"
                )
            else:
                self.checkpoint.clear()
        else:
            self.checkpoint.clear()

        self.checkpoint.set_total_models(total)
        self._run_id = self.checkpoint.run_id
        return resumed, skip_list

    # ─── LOOP PRINCIPAL ────────────────────────────────────────────────────────

    def _benchmark_loop(self, models_raw: list, total: int, skip_list: List[str]) -> None:
        for idx, model_info in enumerate(models_raw, 1):
            if self._interrupted:
                break

            name     = model_info["name"]
            category = classify_model(name)
            size_gb  = model_info.get("size_gb", 0.0)

            if name in skip_list:
                self.logger.model_skip(name, "previous checkpoint")
                self.terminal.log(f"⏭  SKIP  {name}")
                self.terminal.progress.advance_global(1)
                continue

            size_label = f" [{size_gb:.1f}GB]" if size_gb > 0 else ""
            self.logger.model_start(name, category, idx, total)
            self.terminal.model_starting(name)
            self.terminal.log(f"▶ [{idx:2}/{total}] {name}{size_label} ({category})")

            # Cooldown térmico adaptativo
            self._do_cooldown(name)
            self.terminal.model_progress(15)

            # Limpieza de memoria
            self._do_memory_clean()
            self.terminal.model_progress(25)

            # Ejecutar multi-run
            result = self._run_model_multi(name, category, idx)
            self.terminal.model_progress(90)

            # Recalcular scoring con percentiles de categoría
            cat_tps = self.aggregator.get_category_tps_map().get(category, [])
            if result.tokens_per_second > 0:
                cat_tps.append(result.tokens_per_second)
            result.calculate_derived(category_tps_list=cat_tps)

            # Persistir
            self.csv_mgr.append(result)
            self.aggregator.add(result)

            # Checkpoint
            if result.success:
                self.checkpoint.mark_completed(name)
            else:
                self.checkpoint.mark_failed(name, result.error_msg)

            # Actualizar vistas
            stats = self.aggregator.get_live_stats()
            stats["total_models"] = total
            self.terminal.update_stats(stats)
            self.terminal.model_done(result)
            self.checkpoint.update_stats(stats)

            # Log resultado
            self._log_result(result)

            # Dashboard parcial
            if self.live_dashboard and idx % DASHBOARD_UPDATE_EVERY == 0:
                self._update_dashboard()

            self.logger.separator()

        # Recalcular scores finales con percentiles completos
        self._recalculate_scores_final()

    # ─── EJECUCIÓN MULTI-RUN ──────────────────────────────────────────────────

    def _run_model_multi(self, model: str, category: str, run_idx: int) -> BenchmarkResult:
        """
        Execute warmup + NUM_RUNS measurements. Aggregates results statistically.
        Rotates prompts between runs if USE_MULTI_PROMPT=True.
        """
        result = BenchmarkResult(model=model, category=category, run_id=run_idx)
        prompts = get_prompts_for_category(category, self.num_runs + (1 if WARMUP_RUN else 0))
        prompt_idx = 0
        all_run_stats: List[RunStat] = []

        # ── Warmup (not counted) ──
        if WARMUP_RUN:
            self.logger.debug(f"  [warmup] {model}")
            self.terminal.log(f"  🔥 Warmup {model[:28]}...")
            warmup_prompt = prompts[prompt_idx] if prompts else None
            prompt_idx += 1
            self._single_inference(model, category, warmup_prompt, run_num=0, is_warmup=True)
            time.sleep(INTER_RUN_SLEEP)

        # ── Measurement runs ──
        for i in range(1, self.num_runs + 1):
            if self._interrupted:
                break

            prompt = prompts[prompt_idx % len(prompts)] if prompts else None
            prompt_idx += 1

            pct_done = 25 + int((i / self.num_runs) * 65)
            self.terminal.model_progress(pct_done)
            self.terminal.log(f"  📊 Run {i}/{self.num_runs}: {model[:26]}...")

            run_stat = self._single_inference(model, category, prompt, run_num=i)
            all_run_stats.append(run_stat)

            if run_stat.success and run_stat.tokens_per_second > 0:
                ttft_str = f" TTFT={run_stat.ttft_s*1000:.0f}ms" if run_stat.ttft_s > 0 else ""
                self.terminal.log(
                    f"    ✓ {run_stat.tokens_per_second:.1f} tok/s | "
                    f"{run_stat.api_latency_s:.1f}s{ttft_str} | "
                    f"RAM {run_stat.hw.ram_used:.0f}MB | "
                    f"GPU {run_stat.hw.temp_gpu:.0f}°C"
                )
            else:
                self.terminal.log(f"    ✗ Error: {run_stat.error_msg[:60]}")

            if i < self.num_runs:
                time.sleep(INTER_RUN_SLEEP)

        # Agregar estadísticas
        result.aggregate_runs(all_run_stats)

        # Descargar modelo al finalizar
        self.ollama.unload_model(model)
        return result

    def _single_inference(
        self,
        model:     str,
        category:  str,
        prompt:    Optional[str],
        run_num:   int,
        is_warmup: bool = False,
    ) -> RunStat:
        """
        Execute ONE inference and return RunStat.
        Hardware sampled over the inference WINDOW (not post-run snapshot).
        """
        stat = RunStat(run_num=run_num)
        hw_start_ts = time.time()

        try:
            if category == "EMBEDDING":
                resp, elapsed, err = self.ollama.run_embedding(model, custom_prompt=prompt)
            else:
                resp, elapsed, err = self.ollama.run_generate(model, category, custom_prompt=prompt)

            if err:
                stat.success   = False
                stat.error_msg = err
            else:
                if category == "EMBEDDING":
                    parsed = OllamaClient.parse_embedding_response(resp, elapsed)
                else:
                    parsed = OllamaClient.parse_generate_response(resp, elapsed)

                stat.tokens_per_second = parsed["tokens_per_second"]
                stat.total_duration_s  = parsed["total_duration_s"]
                stat.load_duration_s   = parsed["load_duration_s"]
                stat.prompt_eval_count = parsed["prompt_eval_count"]
                stat.eval_count        = parsed["eval_count"]
                stat.api_latency_s     = parsed["api_latency_s"]
                stat.response_preview  = parsed.get("response_text", "")[:200]
                stat.ttft_s            = parsed.get("ttft_s", 0.0)

        except Exception as e:
            stat.success   = False
            stat.error_msg = str(e)

        # ── Hardware: average of snapshots DURING inference ──────────────────
        hw_end_ts = time.time()
        history   = self.hw_mon.history  # copia thread-safe
        window    = [s for s in history if hw_start_ts <= s.ts <= hw_end_ts]
        if len(window) >= 2:
            stat.hw = HardwareSnapshot.average(window)
        elif len(window) == 1:
            stat.hw = window[0]
        else:
            stat.hw = self.hw_mon.current   # fallback: último snapshot disponible

        return stat

    # ─── RECÁLCULO FINAL DE SCORES ─────────────────────────────────────────────

    def _recalculate_scores_final(self) -> None:
        """
        Once ALL results are collected, recalculate scores
        with complete category percentiles and update the CSV.
        """
        cat_map = self.aggregator.get_category_tps_map()
        updated = []

        for r in self.aggregator.results:
            if r.success and r.tokens_per_second > 0:
                cat_tps = cat_map.get(r.category, [r.tokens_per_second])
                r.calculate_derived(category_tps_list=cat_tps)
                updated.append(r)

        if updated:
            self.logger.info(f"Recalculated final scores for {len(updated)} models")
            try:
                self._rewrite_csv()
            except Exception as e:
                self.logger.debug(f"Error rewriting CSV: {e}")

    def _rewrite_csv(self) -> None:
        """Rewrite the full CSV with recalculated scores."""
        import csv as _csv
        from models.data_model import BenchmarkResult as BR
        tmp = self.csv_mgr.path.with_suffix(".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(f, fieldnames=BR.get_csv_fieldnames(), extrasaction="ignore")
            writer.writeheader()
            for r in self.aggregator.results:
                writer.writerow(r.to_csv_row())
        tmp.replace(self.csv_mgr.path)

    # ─── COOLDOWN ADAPTATIVO ───────────────────────────────────────────────────

    def _do_cooldown(self, model: str) -> None:
        """
        Adaptive cooldown:
        - If the last ~30 seconds of history show GPU < COOLDOWN_TEMP, skip wait.
        - Otherwise, wait until GPU drops or time runs out.
        """
        history = self.hw_mon.history
        # Check last ~30 samples (1 Hz → 30s)
        recent = list(history)[-30:] if len(history) >= 5 else []
        if recent and max(s.temp_gpu for s in recent) <= COOLDOWN_TEMP:
            return  # GPU already cool, skip cooldown

        t0, max_wait, check = time.time(), 300, 5.0
        while time.time() - t0 < max_wait:
            temp = self.hw_mon.current.temp_gpu
            if temp <= COOLDOWN_TEMP:
                return
            self.logger.cooldown(temp, COOLDOWN_TEMP)
            self.terminal.log(f"❄  Cooldown {temp:.1f}°C → {COOLDOWN_TEMP}°C")
            time.sleep(check)

    def _do_memory_clean(self) -> None:
        unloaded = self.ollama.unload_all()
        if unloaded:
            self.terminal.log(f"🔄 Unloaded {unloaded} models")
        msg = self.sysutils.clean_memory(sleep_after=MEMORY_CLEAN_SLEEP)
        self.logger.debug(f"Memory clean: {msg}")

    # ─── LOG DE RESULTADO ──────────────────────────────────────────────────────

    def _log_result(self, r: BenchmarkResult) -> None:
        hw = r.hw
        if r.success and r.tokens_per_second > 0:
            self.logger.perf(
                r.model, r.tokens_per_second, hw.ram_used,
                hw.power_total, r.tokens_per_watt,
                r.efficiency_score, r.recommendation,
            )
            if r.tps_stdev > 0:
                self.logger.info(
                    f"  Multi-run: median={r.tps_median:.1f} stdev=±{r.tps_stdev:.1f} "
                    f"min={r.tps_min:.1f} max={r.tps_max:.1f} CV={r.tps_cv:.2f} ({r.stability_label})"
                )
            if r.ttft_s > 0:
                self.logger.info(
                    f"  TTFT: {r.ttft_display} "
                    f"(min={r.ttft_min*1000:.0f}ms max={r.ttft_max*1000:.0f}ms)"
                )
            self.logger.thermal(r.model, hw.temp_gpu, hw.temp_cpu, hw.temp_soc, hw.temp_tj)
            self.logger.power(r.model, hw.vdd_gpu_soc, hw.vdd_cpu_cv, hw.vin_sys_5v0, hw.power_total)
            if r.decision_reasons:
                self.logger.info(f"  Decisión: {r.decision_reasons}")
        elif r.is_embedding:
            self.logger.info(f"{r.model} [EMBEDDING] lat={r.api_latency_s:.2f}s")
        else:
            self.logger.model_error(r.model, r.error_msg or "No tokens generated")

        icon = "✅" if r.recommendation.startswith("KEEP") else \
               "❌" if r.recommendation in ("REMOVE", "ERROR") else \
               "🔲" if r.is_embedding else "⚠"
        self.terminal.log(
            f"{icon} {r.model[:22]}  {r.tps_display:>8} tok/s  "
            f"score={r.efficiency_score:.0f}  pct={r.category_percentile:.0f}%  {r.recommendation}"
        )

    # ─── DASHBOARD ─────────────────────────────────────────────────────────────

    def _update_dashboard(self) -> None:
        try:
            df = self.csv_mgr.load_dataframe()
            if not df.empty:
                self.dashboard.load(df)
                path = self.dashboard.generate()
                self.logger.dashboard_saved(path)
                self.terminal.log(f"📈 Dashboard → {path.name}")
        except Exception as e:
            self.logger.debug(f"Dashboard update failed: {e}")

    # ─── FINALIZACIÓN ──────────────────────────────────────────────────────────

    def _finalize(self, elapsed: float) -> None:
        stats = self.aggregator.get_live_stats()
        self.logger.bench_end(stats.get("total_tested", 0), stats.get("total_success", 0), elapsed / 60.0)
        self._update_dashboard()
        self._export_json_summary(elapsed)

        if not self._interrupted:
            self.checkpoint.clear()

        rows = self.aggregator.summary_table()
        if rows:
            self.logger.separator("FINAL SUMMARY — sorted by Efficiency Score")
            self.logger.print_table(rows[:30], title="📊 Top models")

        if stats:
            self.logger.status(
                f"Best: {stats.get('best_model','?')} ({stats.get('best_tokens_s',0):.1f} tok/s) | "
                f"Dashboard: {self.dashboard.output_path}"
            )
            if stats.get("remove_candidates"):
                self.logger.warn(
                    f"Removal candidates: " + ", ".join(stats["remove_candidates"][:5])
                )

        if self.open_browser and self.dashboard.output_path.exists():
            import webbrowser
            webbrowser.open(f"file://{self.dashboard.output_path.resolve()}")

    def _export_json_summary(self, elapsed: float) -> None:
        """
        Generate a machine-readable JSON summary alongside the CSV.
        Includes TTFT, hw_window, and all decision fields.
        """
        try:
            summary = {
                "schema_version":    "4.0",
                "run_id":            self._run_id,
                "generated_at":      datetime.now().isoformat(),
                "app_version":       APP_VERSION,
                "elapsed_minutes":   round(elapsed / 60, 2),
                "hardware_simulated": self.simulate_hw,
                "num_runs_per_model": self.num_runs,
                "timeout_s":         self._timeout,
                "filter_categories": self.filter_categories or None,
                "filter_models":     self.filter_models or None,
                "stats":             self.aggregator.get_live_stats(),
                "models": [],
            }

            for r in sorted(
                self.aggregator.results,
                key=lambda x: x.efficiency_score,
                reverse=True,
            ):
                entry: Dict = {
                    "model":               r.model,
                    "category":            r.category,
                    "success":             r.success,
                    "tps":                 round(r.tokens_per_second, 2),
                    "tps_median":          round(r.tps_median, 2),
                    "tps_stdev":           round(r.tps_stdev, 3),
                    "tps_cv":              round(r.tps_cv, 4),
                    "tps_runs":            r.tps_runs,
                    "stability":           r.stability_label,
                    "ttft_ms":             round(r.ttft_s * 1000, 1) if r.ttft_s > 0 else None,
                    "ttft_min_ms":         round(r.ttft_min * 1000, 1) if r.ttft_min > 0 else None,
                    "ttft_max_ms":         round(r.ttft_max * 1000, 1) if r.ttft_max > 0 else None,
                    "load_duration_s":     round(r.load_duration_s, 2),
                    "total_duration_s":    round(r.total_duration_s, 2),
                    "efficiency_score":    r.efficiency_score,
                    "category_percentile": round(r.category_percentile, 1),
                    "tokens_per_w":        round(r.tokens_per_watt, 3),
                    "recommendation":      r.recommendation,
                    "decision_reasons":    r.decision_reasons,
                    "score_breakdown":     r.score_breakdown,
                    "hw": {
                        "ram_mb":       round(r.hw.ram_used, 0),
                        "ram_pct":      round(r.hw.ram_percent, 1),
                        "power_w":      round(r.hw.power_watts, 2),
                        "temp_gpu_c":   round(r.hw.temp_gpu, 1),
                        "temp_cpu_c":   round(r.hw.temp_cpu, 1),
                        "gpu_load_pct": round(r.hw.gpu_load, 1),
                        "gpu_freq_mhz": r.hw.gpu_freq,
                    },
                }
                if r.error_msg:
                    entry["error"] = r.error_msg
                summary["models"].append(entry)

            json_path = self.csv_mgr.path.with_suffix(".json")
            tmp       = json_path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            tmp.replace(json_path)
            self.logger.status(f"JSON summary exported: {json_path}")

        except Exception as e:
            self.logger.debug(f"Error exporting JSON: {e}")

    # ─── SEÑALES ───────────────────────────────────────────────────────────────

    def _setup_signals(self) -> None:
        def _handler(sig, frame):
            self._handle_interrupt()
            sys.exit(0)
        signal.signal(signal.SIGINT,  _handler)
        signal.signal(signal.SIGTERM, _handler)

    def _handle_interrupt(self) -> None:
        self._interrupted = True
        self.checkpoint.save()
        self.logger.warn(f"Interrupted — checkpoint saved: {self.checkpoint.summary()}")
        time.sleep(2)
