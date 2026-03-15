#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/ollama_model.py — Ollama API Client v4.0
- Streaming by default (more precise TTFT + TPS)
- Retry with exponential backoff on transient network errors
- Per-instance timeout (overridable via CLI --timeout)
- Embedding: tries /api/embed (Ollama ≥0.4) then falls back to /api/embeddings on 404
- Unloads model after each run (keep_alive=0)
"""
from __future__ import annotations

import base64
import json as _json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from config import (
    OLLAMA_API, OLLAMA_CTX, OLLAMA_KEEP_ALIVE,
    OLLAMA_TEMP, OLLAMA_TIMEOUT, OLLAMA_MAX_RETRIES, OLLAMA_STREAM,
    PROMPTS, ASSETS_DIR,
)


class OllamaClient:
    def __init__(
        self,
        api_base: str  = OLLAMA_API,
        timeout:  int  = OLLAMA_TIMEOUT,
        retries:  int  = OLLAMA_MAX_RETRIES,
    ):
        self.api_base = api_base.rstrip("/") + "/"
        self.timeout  = timeout
        self.retries  = retries
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "User-Agent":   "OllamaVision-Benchmark/4.0",
        })

    # ── Healthcheck & discovery ───────────────────────────────────────────────

    def is_alive(self, timeout: float = 5.0) -> bool:
        try:
            r = self._session.get(self.api_base + "tags", timeout=timeout)
            return r.status_code == 200
        except Exception:
            return False

    def get_version(self) -> str:
        try:
            r = self._session.get(
                self.api_base.replace("/api/", "/") + "version", timeout=5
            )
            return r.json().get("version", "unknown")
        except Exception:
            return "unknown"

    def get_models(self) -> List[Dict[str, Any]]:
        """Returns list of installed models via API /api/tags."""
        r = self._session.get(self.api_base + "tags", timeout=10)
        r.raise_for_status()
        return r.json().get("models", [])

    def get_model_info(self, model: str) -> Dict[str, Any]:
        """
        Returns model metadata (size, family, parameter_size, etc.)
        On failure, returns empty dict (does not block the benchmark).
        """
        try:
            r = self._session.post(
                self.api_base + "show",
                json={"name": model},
                timeout=8,
            )
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return {}

    def get_running_models(self) -> List[str]:
        try:
            r = self._session.get(self.api_base + "ps", timeout=5)
            if r.status_code == 200:
                return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            pass
        return []

    # ── Image loading ────────────────────────────────────────────────────────

    def load_image_b64(self, filename: str = "test_image.jpg") -> Optional[str]:
        path = Path(ASSETS_DIR) / filename
        if not path.exists():
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
                matches = list(Path(ASSETS_DIR).glob(ext))
                if matches:
                    path = matches[0]
                    break
            else:
                return None
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # ── Generate ─────────────────────────────────────────────────────────────

    def run_generate(
        self,
        model:         str,
        category:      str,
        custom_prompt: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], float, Optional[str]]:
        """
        Execute text generation inference.
        Uses streaming if OLLAMA_STREAM=True (measures TTFT precisely).
        Automatic fallback to blocking mode if streaming fails.
        """
        if OLLAMA_STREAM:
            return self._run_generate_streaming(model, category, custom_prompt)
        return self._run_generate_blocking(model, category, custom_prompt)

    def _build_generate_payload(
        self,
        model:    str,
        category: str,
        prompt:   str,
        stream:   bool,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model":      model,
            "prompt":     prompt,
            "stream":     stream,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {
                "num_ctx":     OLLAMA_CTX,
                "temperature": OLLAMA_TEMP,
                "seed":        42,
            },
        }
        if category == "VISION":
            img = self.load_image_b64()
            if img:
                payload["images"] = [img]
        return payload

    def _run_generate_streaming(
        self,
        model:         str,
        category:      str,
        custom_prompt: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], float, Optional[str]]:
        """
        Streaming inference:
        - Measures TTFT (time to first token)
        - TPS calculated from eval_count / eval_duration of the final chunk
        - Timeout: (connect=10s, read=timeout) to avoid blocking on slow model load
        """
        prompt  = custom_prompt or PROMPTS.get(category, PROMPTS["GENERAL"])
        payload = self._build_generate_payload(model, category, prompt, stream=True)
        url     = self.api_base + "generate"
        t0      = time.perf_counter()
        ttft    = 0.0
        full_response = ""

        for attempt in range(self.retries + 1):
            try:
                r = self._session.post(
                    url, json=payload,
                    timeout=(10, self.timeout),
                    stream=True,
                )
                if r.status_code != 200:
                    elapsed = time.perf_counter() - t0
                    return {}, elapsed, f"HTTP {r.status_code}: {r.text[:300]}"

                final: Dict[str, Any] = {}
                ttft = 0.0
                full_response = ""

                for raw_line in r.iter_lines():
                    if not raw_line:
                        continue
                    chunk = _json.loads(raw_line)
                    tok   = chunk.get("response", "")
                    if ttft == 0.0 and tok:
                        ttft = time.perf_counter() - t0
                    full_response += tok
                    if chunk.get("done"):
                        final = chunk
                        break

                elapsed          = time.perf_counter() - t0
                final["response"] = full_response
                final["ttft_s"]   = round(ttft, 4)
                return final, elapsed, None

            except requests.exceptions.Timeout:
                elapsed = time.perf_counter() - t0
                return {}, elapsed, f"Timeout tras {self.timeout}s"

            except requests.exceptions.ConnectionError as e:
                elapsed = time.perf_counter() - t0
                if attempt < self.retries:
                    time.sleep(2 ** attempt)   # 1s, 2s backoff
                    continue
                return {}, elapsed, f"ConnectionError: {e}"

            except Exception as e:
                elapsed = time.perf_counter() - t0
                return {}, elapsed, f"Error: {e}"

        return {}, time.perf_counter() - t0, "Max retries exceeded"

    def _run_generate_blocking(
        self,
        model:         str,
        category:      str,
        custom_prompt: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], float, Optional[str]]:
        """Blocking inference (fallback or when OLLAMA_STREAM=False)."""
        prompt  = custom_prompt or PROMPTS.get(category, PROMPTS["GENERAL"])
        payload = self._build_generate_payload(model, category, prompt, stream=False)
        resp, elapsed, err = self._post_with_retry("generate", payload)
        if resp:
            resp["ttft_s"] = 0.0   # not available without streaming
        return resp, elapsed, err

    # ── Embeddings ───────────────────────────────────────────────────────────

    def run_embedding(
        self,
        model:         str,
        custom_prompt: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], float, Optional[str]]:
        """
        Embeddings: tries /api/embed first (Ollama ≥0.4),
        then falls back to /api/embeddings (older versions).
        """
        prompt = custom_prompt or PROMPTS["EMBEDDING"]

        # ── Nuevo endpoint (Ollama ≥0.4): /api/embed ─────────────────────
        payload_new = {
            "model":      model,
            "input":      prompt,
            "keep_alive": OLLAMA_KEEP_ALIVE,
        }
        resp, elapsed, err = self._post_with_retry("embed", payload_new)
        if err is None:
            # Normalizar: el nuevo endpoint retorna {"embeddings": [[...]]}
            if "embeddings" in resp and "embedding" not in resp:
                first = resp["embeddings"]
                if first and isinstance(first[0], list):
                    resp["embedding"] = first[0]
                elif first:
                    resp["embedding"] = first
            return resp, elapsed, None

        # ── Fallback: /api/embeddings (Ollama <0.4) ───────────────────────
        if "404" in str(err) or "HTTP 4" in str(err):
            payload_old = {
                "model":      model,
                "prompt":     prompt,
                "keep_alive": OLLAMA_KEEP_ALIVE,
            }
            return self._post_with_retry("embeddings", payload_old)

        return resp, elapsed, err

    # ── Model lifecycle ───────────────────────────────────────────────────────

    def unload_model(self, model: str, timeout: float = 3.0) -> bool:
        try:
            r = self._session.post(
                self.api_base + "generate",
                json={"model": model, "keep_alive": 0},
                timeout=timeout,
            )
            return r.status_code == 200
        except Exception:
            return False

    def unload_all(self) -> int:
        running = self.get_running_models()
        count = 0
        for m in running:
            if self.unload_model(m):
                count += 1
        return count

    # ── Internal HTTP helpers ─────────────────────────────────────────────────

    def _post(
        self,
        endpoint: str,
        payload:  Dict[str, Any],
    ) -> Tuple[Dict[str, Any], float, Optional[str]]:
        """Blocking POST without retry."""
        url = self.api_base + endpoint
        t0  = time.perf_counter()
        try:
            r       = self._session.post(url, json=payload, timeout=self.timeout)
            elapsed = time.perf_counter() - t0
            if r.status_code != 200:
                return {}, elapsed, f"HTTP {r.status_code}: {r.text[:300]}"
            return r.json(), elapsed, None
        except requests.exceptions.Timeout:
            elapsed = time.perf_counter() - t0
            return {}, elapsed, f"Timeout tras {self.timeout}s"
        except requests.exceptions.ConnectionError as e:
            elapsed = time.perf_counter() - t0
            return {}, elapsed, f"ConnectionError: {e}"
        except Exception as e:
            elapsed = time.perf_counter() - t0
            return {}, elapsed, f"Error: {e}"

    def _post_with_retry(
        self,
        endpoint: str,
        payload:  Dict[str, Any],
    ) -> Tuple[Dict[str, Any], float, Optional[str]]:
        """POST with automatic retry on transient network errors."""
        last_err: Optional[str] = None
        for attempt in range(self.retries + 1):
            resp, elapsed, err = self._post(endpoint, payload)
            if err is None:
                return resp, elapsed, None
            # Do not retry deterministic errors (total timeout, HTTP 4xx)
            if err.startswith("Timeout") or ("HTTP 4" in err):
                return resp, elapsed, err
            last_err = err
            if attempt < self.retries:
                time.sleep(2 ** attempt)   # backoff: 1s, 2s
        return {}, 0.0, last_err or "Max retries exceeded"

    # ── Response parsers ──────────────────────────────────────────────────────

    @staticmethod
    def parse_generate_response(resp: Dict[str, Any], elapsed: float) -> Dict[str, Any]:
        NS = 1_000_000_000.0
        eval_count   = resp.get("eval_count",        0)
        eval_dur_ns  = resp.get("eval_duration",     1)
        total_dur_ns = resp.get("total_duration",    1)
        load_dur_ns  = resp.get("load_duration",     1)
        prompt_count = resp.get("prompt_eval_count", 0)
        response_txt = resp.get("response", "")
        eval_dur_s   = max(eval_dur_ns / NS, 1e-9)
        tps          = eval_count / eval_dur_s if eval_count > 0 else 0.0
        return {
            "tokens_per_second": round(tps, 3),
            "total_duration_s":  round(total_dur_ns / NS, 4),
            "load_duration_s":   round(load_dur_ns  / NS, 4),
            "prompt_eval_count": prompt_count,
            "eval_count":        eval_count,
            "api_latency_s":     round(elapsed, 4),
            "response_text":     response_txt[:200],
            "ttft_s":            resp.get("ttft_s", 0.0),
        }

    @staticmethod
    def parse_embedding_response(resp: Dict[str, Any], elapsed: float) -> Dict[str, Any]:
        embedding = resp.get("embedding", [])
        dim = len(embedding) if isinstance(embedding, list) else 0
        return {
            "tokens_per_second": 0.0,
            "total_duration_s":  0.0,
            "load_duration_s":   0.0,
            "prompt_eval_count": 0,
            "eval_count":        dim,
            "api_latency_s":     round(elapsed, 4),
            "response_text":     f"[embedding dim={dim}]",
            "ttft_s":            0.0,
        }

    # ── Context manager ───────────────────────────────────────────────────────

    def close(self) -> None:
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
