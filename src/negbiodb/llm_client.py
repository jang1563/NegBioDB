"""Unified LLM client for vLLM (local), OpenAI, and Gemini (API) inference.

Supports OpenAI-compatible API (vLLM server, OpenAI) and Google Gemini API.
Includes rate limiter for Gemini free tier (250 RPD Flash, 1000 RPD Flash-Lite).
"""

import fcntl
import json
import os
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

# Rate limit state file (survives crashes)
_RATE_LIMIT_DIR = Path.home() / ".config" / "negbiodb"


class LLMClient:
    """Unified client for local vLLM and Gemini API models."""

    def __init__(
        self,
        provider: str,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        """Initialize LLM client.

        Args:
            provider: 'vllm', 'openai', or 'gemini'
            model: Model name/path (e.g., 'gpt-4o-mini', 'gemini-2.5-flash')
            api_base: API base URL (vLLM: 'http://localhost:8000/v1')
            api_key: API key (read from env/file if not provided)
            temperature: Generation temperature (default 0 for determinism)
            max_tokens: Max output tokens
        """
        self.provider = provider
        self.model = model
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens

        if provider == "gemini":
            self.api_key = api_key or self._load_gemini_key()
            self.rate_limiter = GeminiRateLimiter(model)
        elif provider == "openai":
            self.api_base = api_base or "https://api.openai.com/v1"
            self.api_key = api_key or self._load_openai_key()
            self.rate_limiter = None
        elif provider == "vllm":
            self.api_base = api_base or "http://localhost:8000/v1"
            self.api_key = api_key or "EMPTY"
            self.rate_limiter = None
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _load_openai_key(self) -> str:
        """Load OpenAI API key from env or file."""
        key = os.environ.get("OPENAI_API_KEY")
        if key:
            return key
        key_file = _RATE_LIMIT_DIR / "openai_api_key.txt"
        if key_file.exists():
            return key_file.read_text().strip()
        raise ValueError(
            "OpenAI API key not found. Set OPENAI_API_KEY env var or "
            f"create {key_file}"
        )

    def _load_gemini_key(self) -> str:
        """Load Gemini API key from env or file."""
        key = os.environ.get("GEMINI_API_KEY")
        if key:
            return key
        key_file = _RATE_LIMIT_DIR / "gemini_api_key.txt"
        if key_file.exists():
            return key_file.read_text().strip()
        raise ValueError(
            "Gemini API key not found. Set GEMINI_API_KEY env var or "
            f"create {key_file}"
        )

    def generate(self, prompt: str, system: str = "") -> str:
        """Generate a single completion."""
        if self.provider in ("vllm", "openai"):
            return self._generate_openai_compat(prompt, system)
        elif self.provider == "gemini":
            return self._generate_gemini(prompt, system)
        raise ValueError(f"Unknown provider: {self.provider}")

    def generate_batch(
        self, prompts: list[tuple[str, str]], progress: bool = True
    ) -> list[str]:
        """Generate completions for a batch of (system, user) prompts."""
        results = []
        for i, (system, user) in enumerate(prompts):
            if progress and (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(prompts)}")
            try:
                result = self.generate(user, system)
            except Exception as e:
                print(f"  Error on prompt {i}: {e}")
                result = f"ERROR: {e}"
            results.append(result)
        return results

    # ── OpenAI-compatible API (vLLM, OpenAI) ─────────────────────────────────

    def _generate_openai_compat(self, prompt: str, system: str) -> str:
        """Generate via OpenAI-compatible chat completions API with retry."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        url = f"{self.api_base}/chat/completions"
        data = json.dumps(payload).encode("utf-8")

        max_retries = 8
        for attempt in range(max_retries):
            req = urllib.request.Request(
                url,
                data=data,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    result = json.loads(resp.read())
                return result["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < max_retries - 1:
                    wait = min(2 ** (attempt + 1), 128)
                    print(f"  Rate limited (429), retry {attempt + 1}/{max_retries} in {wait}s")
                    time.sleep(wait)
                elif e.code >= 500 and attempt < max_retries - 1:
                    wait = min(2 ** (attempt + 1), 64)
                    print(f"  Server error ({e.code}), retry in {wait}s")
                    time.sleep(wait)
                else:
                    raise
        else:
            raise RuntimeError(f"OpenAI-compat API failed after {max_retries} retries")

    # ── Gemini API ────────────────────────────────────────────────────────────

    def _generate_gemini(self, prompt: str, system: str) -> str:
        """Generate via Gemini API REST endpoint."""
        if self.rate_limiter:
            self.rate_limiter.wait()

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )

        contents = [{"role": "user", "parts": [{"text": prompt}]}]

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        data = json.dumps(payload).encode("utf-8")

        max_retries = 8
        for attempt in range(max_retries):
            req = urllib.request.Request(
                url,
                data=data,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    result = json.loads(resp.read())
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < max_retries - 1:
                    wait = min(2 ** (attempt + 1), 128)
                    print(f"  Rate limited (429), retry {attempt + 1}/{max_retries} in {wait}s")
                    time.sleep(wait)
                elif e.code >= 500 and attempt < max_retries - 1:
                    wait = min(2 ** (attempt + 1), 64)
                    print(f"  Server error ({e.code}), retry in {wait}s")
                    time.sleep(wait)
                else:
                    raise
        else:
            raise RuntimeError(f"Gemini API failed after {max_retries} retries")

        candidates = result.get("candidates", [])
        if not candidates:
            return "ERROR: No candidates in response"

        parts = candidates[0].get("content", {}).get("parts", [])
        return parts[0].get("text", "") if parts else ""


class GeminiRateLimiter:
    """Rate limiter for Gemini free tier.

    Persists daily call count to disk for crash recovery.
    Daily reset at midnight Pacific Time.
    """

    # Free tier limits (March 2026)
    LIMITS = {
        "gemini-2.5-flash": {"rpd": 250, "rpm": 10},
        "gemini-2.5-flash-lite": {"rpd": 1000, "rpm": 15},
    }

    def __init__(self, model: str):
        self.model = model
        limits = self.LIMITS.get(model, {"rpd": 250, "rpm": 10})
        self.max_rpd = limits["rpd"]
        self.max_rpm = limits["rpm"]
        self.state_file = _RATE_LIMIT_DIR / f"rate_state_{model.replace('/', '_')}.json"
        _RATE_LIMIT_DIR.mkdir(parents=True, exist_ok=True)

    def wait(self):
        """Wait if necessary to respect rate limits.

        Uses file locking to coordinate across concurrent SLURM jobs.
        The entire read-check-increment-write cycle is protected by a
        single LOCK_EX to prevent TOCTOU races between concurrent jobs.
        """
        while True:
            # Acquire exclusive lock for the entire check-increment-save cycle
            self.state_file.touch(exist_ok=True)
            with open(self.state_file, "r+") as f:
                fcntl.flock(f, fcntl.LOCK_EX)

                # Read state
                content = f.read().strip()
                if content:
                    try:
                        state = json.loads(content)
                    except (json.JSONDecodeError, ValueError):
                        print("  Warning: corrupted rate state file, resetting")
                        state = {}
                else:
                    state = {}

                daily_count = state.get("daily_count", 0)
                daily_date = state.get("daily_date", "")
                minute_timestamps = state.get("minute_timestamps", [])

                # Reset if new day
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                if daily_date != today:
                    daily_count = 0
                    daily_date = today
                    minute_timestamps = []

                # Check daily limit
                if daily_count >= self.max_rpd:
                    # Release lock before sleeping
                    fcntl.flock(f, fcntl.LOCK_UN)
                    print(
                        f"  Daily limit reached ({daily_count}/{self.max_rpd}). "
                        f"Waiting 1h."
                    )
                    time.sleep(3600)
                    continue  # retry from top

                # Check per-minute limit
                now = time.time()
                minute_timestamps = [
                    t for t in minute_timestamps if now - t < 60
                ]
                if len(minute_timestamps) >= self.max_rpm:
                    wait_time = 60 - (now - minute_timestamps[0]) + 1
                    # Release lock before sleeping
                    fcntl.flock(f, fcntl.LOCK_UN)
                    if wait_time > 0:
                        time.sleep(wait_time)
                    continue  # retry from top

                # Record this call and write back atomically
                daily_count += 1
                minute_timestamps.append(time.time())
                new_state = {
                    "daily_count": daily_count,
                    "daily_date": daily_date,
                    "minute_timestamps": minute_timestamps[-self.max_rpm :],
                }
                f.seek(0)
                f.truncate()
                f.write(json.dumps(new_state))
                f.flush()
                os.fsync(f.fileno())
                fcntl.flock(f, fcntl.LOCK_UN)
                return  # slot acquired
