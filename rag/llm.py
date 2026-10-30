"""Minimal client for any OpenAI-compatible chat API (Groq, OpenAI, OpenRouter, local Ollama).

Uses only the standard library, so there is no SDK to install.
Configure with environment variables:
    LLM_PROVIDER  groq | openai | openrouter | ollama   (sets base URL and a default model)
    LLM_API_KEY   your key (not needed for ollama)
    LLM_MODEL     optional, overrides the provider's default model
    LLM_BASE_URL  optional, overrides the provider's base URL
"""
import json
import os
import time
import urllib.error
import urllib.request

PROVIDERS = {
    "groq":       ("https://api.groq.com/openai/v1", "openai/gpt-oss-20b"),
    "openai":     ("https://api.openai.com/v1", "gpt-4o-mini"),
    "openrouter": ("https://openrouter.ai/api/v1", "meta-llama/llama-3.1-8b-instruct"),
    "ollama":     ("http://localhost:11434/v1", "llama3.1:8b"),
}


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "groq")).lower()
        if self.provider not in PROVIDERS:
            raise ValueError(f"Unknown provider '{self.provider}'. Pick one of {list(PROVIDERS)}")
        base, default_model = PROVIDERS[self.provider]
        self.base_url = os.getenv("LLM_BASE_URL", base).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL", default_model)
        self.api_key = os.getenv("LLM_API_KEY", "")
        if not self.api_key and self.provider != "ollama":
            raise RuntimeError("Set LLM_API_KEY before using an LLM (or use --mode extractive).")

    def chat(self, system: str, user: str, max_tokens: int = 1000, retries: int = 3) -> str:
        body = json.dumps({
            "model": self.model,
            "temperature": 0,
            # reasoning models (like gpt-oss) spend part of this budget thinking before answering
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }).encode()
        # Groq's firewall blocks Python's default "Python-urllib" user agent (error 1010)
        headers = {"Content-Type": "application/json", "User-Agent": "pdf-rag-qa/1.0"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", body, headers)

        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.load(resp)
                content = data["choices"][0]["message"].get("content") or ""
                return content.strip()
            except urllib.error.HTTPError as e:
                # free tiers rate-limit quickly; back off and try again
                if e.code == 429 and attempt < retries - 1:
                    time.sleep(2 ** (attempt + 2))
                    continue
                raise RuntimeError(f"LLM API error {e.code}: {e.read().decode()[:300]}") from e
        raise RuntimeError("LLM API kept rate-limiting; try again later")