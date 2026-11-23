"""LLM access: one function, `complete(system, user) -> str`, behind a provider switch.

Agents depend on that function only, so providers are swappable and tests patch one
name. Providers:
  gemini         google-genai SDK
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
from functools import lru_cache

from dotenv import dotenv_values

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
    # Prompt-cache prices (USD per 1M). None = standard multipliers of price_in:
    # 1.25x for a 5-minute cache write, 0.1x for a cache read.
    price_cache_write: float | None = None
    price_cache_read: float | None = None
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


# google-genai reads GEMINI_API_KEY; older setups of this repo used GOOGLE_API_KEY.
_KEY_ALIASES = {"GEMINI_API_KEY": ("GOOGLE_API_KEY",), "GOOGLE_API_KEY": ("GEMINI_API_KEY",)}


@lru_cache(maxsize=1)
def _env_file_values() -> dict:
    """Keys from ./.env, so ANTHROPIC_API_KEY etc. work without exporting them."""
    return {k: v for k, v in dotenv_values(".env").items() if v}


def _api_key(cfg: ModelConfig) -> str:
    """Environment first, then ./.env. Returns "" when the key is not set anywhere."""
    if not cfg.api_key_env:
        return ""
    for name in (cfg.api_key_env, *_KEY_ALIASES.get(cfg.api_key_env, ())):
        key = os.environ.get(name) or _env_file_values().get(name)
        if key:
            return key
    if cfg.api_key_env in _KEY_ALIASES:
        return settings.google_api_key  # legacy pydantic setting, kept for backwards compatibility
    return ""


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


def _anthropic(cfg: ModelConfig, system: str, user: str) -> tuple[str, int, int, dict]:
    key = _api_key(cfg)
    if not key:
        raise RuntimeError(f"{cfg.api_key_env or 'API key'} is empty — set it in the environment.")
    # Prompt caching: the system prompt (rules + schema) is identical for every question on
    # the same database, and only the user turn changes. Marking the end of the system
    # prompt caches it for 5 minutes; later calls read it at 0.1x the input price. Prefixes
    # under the model's minimum (512 tokens on Haiku 5.5) are simply not cached, no error.
    body = {"model": cfg.model, "max_tokens": cfg.max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
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
    # Anthropic's input_tokens excludes cached tokens. Report the full prompt size as
    # in_tokens and the cached parts separately, so cost can price each part correctly.
    cw, cr = u.get("cache_creation_input_tokens") or 0, u.get("cache_read_input_tokens") or 0
    tin = u.get("input_tokens", 0) + cw + cr
    return text, tin, u.get("output_tokens", 0), {"cache_write_tokens": cw, "cache_read_tokens": cr}


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


_gemini_clients: dict[str, object] = {}  # one client per API key; created lazily


def _gemini_client(key: str):
    from google import genai  # lazy: only needed for this provider

    if key not in _gemini_clients:
        _gemini_clients[key] = genai.Client(api_key=key)
    return _gemini_clients[key]


def _gemini(cfg: ModelConfig, system: str, user: str, retries: int = 3) -> tuple[str, int, int]:
    from google.genai import errors, types

    key = _api_key(cfg)
    if not key:
        raise RuntimeError(f"{cfg.api_key_env or 'GEMINI_API_KEY'} is empty — set it in .env or the environment.")
    config = types.GenerateContentConfig(
        system_instruction=system, max_output_tokens=cfg.max_tokens, temperature=cfg.temperature,
        thinking_config=types.ThinkingConfig(thinking_level=cfg.effort) if cfg.effort else None,
    )
    for attempt in range(retries + 1):  # same backoff policy as the HTTP providers
        try:
            resp = _gemini_client(key).models.generate_content(model=cfg.model, contents=user, config=config)
            break
        except errors.APIError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"LLM HTTP {e.code}: {str(e)[:300]}") from e
    text = resp.text  # None when blocked or empty
    if not text:
        reason = resp.candidates[0].finish_reason if resp.candidates else "no candidates"
        raise RuntimeError(f"LLM returned no text (blocked or empty response): {reason}")
    um = resp.usage_metadata
    tin = (um.prompt_token_count or 0) if um else 0
    # Thinking tokens are billed as output, so they count toward cost.
    tout = ((um.candidates_token_count or 0) + (um.thoughts_token_count or 0)) if um else 0
    return text, tin, tout


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
    text, tin, tout, *extra = fn(cfg, system, user)  # extra: optional cache-usage dict
    calls = _calls.get()
    if calls is not None:
        cache = extra[0] if extra else {}
        calls.append({"in_tokens": tin, "out_tokens": tout, "seconds": time.perf_counter() - start,
                      "cache_write_tokens": cache.get("cache_write_tokens", 0),
                      "cache_read_tokens": cache.get("cache_read_tokens", 0)})
    return text
