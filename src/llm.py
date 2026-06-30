"""Thin LLM client. Provider plumbing only — your *prompts* live in the agents.

Swap providers by changing only this file (e.g. OpenAI, Anthropic, a local model).
That isolation is itself a design decision worth mentioning in your README.
"""
import google.generativeai as genai
from src.config import settings

_configured = False


def _ensure():
    global _configured
    if not _configured:
        if not settings.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is empty — copy .env.example to .env and set it.")
        genai.configure(api_key=settings.google_api_key)
        _configured = True


def complete(system: str, user: str) -> str:
    """Single-turn completion. Returns raw model text."""
    _ensure()
    model = genai.GenerativeModel(settings.llm_model, system_instruction=system)
    return model.generate_content(user).text
