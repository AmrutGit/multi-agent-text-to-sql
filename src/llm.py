"""LLM access: one function, `complete(system, user) -> str`, behind a provider switch.

Agents depend on that function only, so providers are swappable and tests patch one
name. Providers:
  gemini         google-generativeai SDK
  anthropic      Messages API over plain HTTPS
  openai_compat  any OpenAI-style /chat/completions server: OpenAI itself, vLLM,
                 Ollama (/v1), llama.cpp server... i.e. how a locally served QLoRA
                 model plugs in. See README.

Every call also records latency and token usage into the active `track()` scope; the
benchmark uses that to report latency and cost per question.
"""
import json
import os
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from src.config import settings


@dataclass(frozen=True)
class ModelConfig:
    name: str  # label used in reports
    provider: str  # gemini | anthropic | openai_compat
    model: str
    base_url: str = ""
    api_key_env: str = ""  # NAME of the env var holding the key (never the key itself)
    price_in: float | None = None  # USD per 1M input tokens; None = unknown
    price_out: float | None = None  # USD per 1M output tokens
    max_tokens: int = 1024
    # None = do not send the parameter (provider default). Claude Haiku 5.5 rejects any
    # non-default sampling value with HTTP 400, so it must be None there.
    temperature: float | None = 0.0
    # Reasoning depth for models that think by default. Sent as `output_config.effort`
    # (Anthropic) or `thinking_level` (Gemini). None = provider default. Set explicitly in
    # models.yaml so a provider changing its default cannot silently change results.
    effort: str | None = None
    timeout_s: float = 120.0


def default_model() -> ModelConfig:
    return ModelConfig(
        name=settings.llm_model, provider=settings.llm_provider, model=settings.llm_model,
        base_url=settings.llm_base_url, api_key_env=settings.llm_api_key_env,
    )


_active: ContextVar[ModelConfig | None] = ContextVar("llm_active_model", default=None)
_calls: ContextVar[list | None] = ContextVar("llm_calls", default=None)


@contextmanager
def use_model(cfg: ModelConfig):
    """Route `complete()` calls inside this block to `cfg` (thread/async safe)."""
    tok = _active.set(cfg)
    try:
        yield
    finally:
        _active.reset(tok)


@contextmanager
def track():
    """Collect per-call usage: yields a list of {in_tokens, out_tokens, seconds}."""
    calls: list = []
    tok = _calls.set(calls)
    try:
        yield calls
    finally:
        _calls.reset(tok)


def _api_key(cfg: ModelConfig) -> str:
    key = os.environ.get(cfg.api_key_env, "") if cfg.api_key_env else ""
    if not key and cfg.api_key_env == "GOOGLE_API_KEY":
        key = settings.google_api_key  # value loaded from .env
    return key


def _post_json(url: str, headers: dict, body: dict, timeout: float, retries: int = 3) -> dict:
    """POST JSON with exponential backoff on rate limits / transient server errors."""
    data = json.dumps(body).encode()
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            if e.code in (429, 500, 502, 503, 529) and attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"LLM HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"LLM connection error: {e.reason}") from e
    raise RuntimeError("unreachable")


def _anthropic(cfg: ModelConfig, system: str, user: str) -> tuple[str, int, int]:
    key = _api_key(cfg)
    if not key:
        raise RuntimeError(f"{cfg.api_key_env or 'API key'} is empty — set it in the environment.")
    body = {"model": cfg.model, "max_tokens": cfg.max_tokens, "system": system,
            "messages": [{"role": "user", "content": user}]}
    if cfg.temperature is not None:
        body["temperature"] = cfg.temperature
    if cfg.effort:
        body["output_config"] = {"effort": cfg.effort}
    out = _post_json(cfg.base_url.rstrip("/") + "/v1/messages" if cfg.base_url
                     else "https://api.anthropic.com/v1/messages",
                     {"x-api-key": key, "anthropic-version": "2023-06-01"}, body, cfg.timeout_s)
    if out.get("stop_reason") == "refusal":  # HTTP 200, but no answer: a provider failure, not SQL
        raise RuntimeError(f"LLM refused the request: {out.get('stop_details')}")
    # Thinking blocks are skipped; their tokens are already inside usage.output_tokens.
    text = "".join(b.get("text", "") for b in out.get("content", []) if b.get("type") == "text")
    u = out.get("usage", {})
    return text, u.get("input_tokens", 0), u.get("output_tokens", 0)


def _openai_compat(cfg: ModelConfig, system: str, user: str) -> tuple[str, int, int]:
    if not cfg.base_url:
        raise RuntimeError("openai_compat provider needs base_url (e.g. http://localhost:8001/v1).")
    key = _api_key(cfg)  # optional: local servers usually need none
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    body = {"model": cfg.model, "max_tokens": cfg.max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if cfg.temperature is not None:
        body["temperature"] = cfg.temperature
    out = _post_json(cfg.base_url.rstrip("/") + "/chat/completions", headers, body, cfg.timeout_s)
    try:
        text = out["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected response shape: {str(out)[:200]}") from e
    u = out.get("usage", {})
    return text, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


_gemini_configured = False


def _gemini(cfg: ModelConfig, system: str, user: str) -> tuple[str, int, int]:
    global _gemini_configured
    import google.generativeai as genai  # lazy: only needed for this provider

    if not _gemini_configured:
        key = _api_key(cfg)
        if not key:
            raise RuntimeError("GOOGLE_API_KEY is empty — copy .env.example to .env and set it.")
        genai.configure(api_key=key)
        _gemini_configured = True
    model = genai.GenerativeModel(cfg.model, system_instruction=system,
                                  generation_config={"temperature": cfg.temperature or 0})
    resp = model.generate_content(user)
    try:
        text = resp.text
    except ValueError as e:  # blocked / empty candidate: .text raises
        raise RuntimeError(f"LLM returned no text (blocked or empty response): {e}") from e
    um = getattr(resp, "usage_metadata", None)
    return text, getattr(um, "prompt_token_count", 0) or 0, getattr(um, "candidates_token_count", 0) or 0


_PROVIDERS = {"gemini": _gemini, "anthropic": _anthropic, "openai_compat": _openai_compat}


def complete(system: str, user: str) -> str:
    """Single-turn completion. Returns raw model text.

    temperature=0 by default: SQL generation should be as deterministic as possible,
    otherwise eval scores are noisy run to run.
    """
    cfg = _active.get() or default_model()
    fn = _PROVIDERS.get(cfg.provider)
    if fn is None:
        raise RuntimeError(f"Unknown provider '{cfg.provider}'. Use one of {sorted(_PROVIDERS)}.")
    start = time.perf_counter()
    text, tin, tout = fn(cfg, system, user)
    calls = _calls.get()
    if calls is not None:
        calls.append({"in_tokens": tin, "out_tokens": tout, "seconds": time.perf_counter() - start})
    return text
