"""Provider tests against a fake local HTTP server: no network, no keys."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from src import llm


class _Server:
    def __init__(self, replies):
        self.requests, self.replies = [], list(replies)
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, "body": body,
                                       "headers": {k.lower(): v for k, v in self.headers.items()}})
                status, payload = outer.replies.pop(0)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def server():
    made = []

    def make(replies):
        s = _Server(replies)
        made.append(s)
        return s
    yield make
    for s in made:
        s.close()


def test_openai_compat_request_response_and_usage(server):
    s = server([(200, {"choices": [{"message": {"content": "SELECT 1"}}],
                       "usage": {"prompt_tokens": 11, "completion_tokens": 3}})])
    cfg = llm.ModelConfig("local", "openai_compat", "sql", base_url=s.url + "/v1")
    with llm.use_model(cfg), llm.track() as calls:
        assert llm.complete("sys", "usr") == "SELECT 1"
    req = s.requests[0]
    assert req["path"] == "/v1/chat/completions"
    assert req["body"]["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]
    assert req["body"]["temperature"] == 0.0 and "authorization" not in req["headers"]
    assert calls[0]["in_tokens"] == 11 and calls[0]["out_tokens"] == 3 and calls[0]["seconds"] >= 0


def test_openai_compat_sends_bearer_when_key_set(server, monkeypatch):
    monkeypatch.setenv("MY_KEY", "sekret")
    s = server([(200, {"choices": [{"message": {"content": "x"}}]})])
    cfg = llm.ModelConfig("m", "openai_compat", "m", base_url=s.url, api_key_env="MY_KEY")
    with llm.use_model(cfg):
        llm.complete("a", "b")
    assert s.requests[0]["headers"]["authorization"] == "Bearer sekret"


def test_anthropic_request_shape(server, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    s = server([(200, {"content": [{"type": "text", "text": "SELECT 2"}],
                       "usage": {"input_tokens": 5, "output_tokens": 2}})])
    cfg = llm.ModelConfig("c", "anthropic", "some-model", base_url=s.url, api_key_env="ANTHROPIC_API_KEY")
    with llm.use_model(cfg), llm.track() as calls:
        assert llm.complete("sys", "usr") == "SELECT 2"
    r = s.requests[0]
    assert r["path"] == "/v1/messages" and r["headers"]["x-api-key"] == "k"
    assert r["body"]["system"] == [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}]
    assert r["body"]["messages"][0]["content"] == "usr"
    assert (calls[0]["in_tokens"], calls[0]["out_tokens"]) == (5, 2)


def test_anthropic_cache_usage_is_tracked_and_priced(server, monkeypatch):
    """input_tokens excludes cached tokens; in_tokens must be the full prompt size."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    s = server([(200, {"content": [{"type": "text", "text": "SELECT 1"}],
                       "usage": {"input_tokens": 10, "output_tokens": 20,
                                 "cache_creation_input_tokens": 0, "cache_read_input_tokens": 1000}})])
    cfg = llm.ModelConfig("h", "anthropic", "m", base_url=s.url, api_key_env="ANTHROPIC_API_KEY",
                          price_in=0.10, price_out=0.50, price_cache_read=0.01)
    with llm.use_model(cfg), llm.track() as calls:
        llm.complete("sys", "usr")
    c = calls[0]
    assert (c["in_tokens"], c["cache_read_tokens"], c["cache_write_tokens"]) == (1010, 1000, 0)
    from evals.benchmark import cost_usd
    # 10 uncached * 0.10 + 1000 cached * 0.01 + 20 out * 0.50, per 1M tokens
    assert cost_usd(cfg, 1010, 20, 0, 1000) == pytest.approx((1.0 + 10.0 + 10.0) / 1_000_000)


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = llm.ModelConfig("c", "anthropic", "m", api_key_env="ANTHROPIC_API_KEY")
    with llm.use_model(cfg), pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        llm.complete("a", "b")


def test_retries_transient_errors_then_succeeds(server, monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    s = server([(429, {"error": "slow down"}), (503, {"error": "x"}),
                (200, {"choices": [{"message": {"content": "ok"}}]})])
    cfg = llm.ModelConfig("m", "openai_compat", "m", base_url=s.url)
    with llm.use_model(cfg):
        assert llm.complete("a", "b") == "ok"
    assert len(s.requests) == 3


def test_client_error_is_not_retried(server):
    s = server([(400, {"error": "bad"})])
    cfg = llm.ModelConfig("m", "openai_compat", "m", base_url=s.url)
    with llm.use_model(cfg), pytest.raises(RuntimeError, match="HTTP 400"):
        llm.complete("a", "b")
    assert len(s.requests) == 1


def test_anthropic_effort_and_no_temperature(server, monkeypatch):
    """Haiku 5.5 rejects sampling params: temperature=None must leave the field out."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    s = server([(200, {"content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": "SELECT 3"}],
                       "stop_reason": "end_turn", "usage": {"input_tokens": 9, "output_tokens": 40}})])
    cfg = llm.ModelConfig("h", "anthropic", "claude-haiku-5-5", base_url=s.url, api_key_env="ANTHROPIC_API_KEY",
                          temperature=None, effort="medium")
    with llm.use_model(cfg), llm.track() as calls:
        assert llm.complete("sys", "usr") == "SELECT 3"  # thinking block skipped
    body = s.requests[0]["body"]
    assert "temperature" not in body and body["output_config"] == {"effort": "medium"}
    assert calls[0]["out_tokens"] == 40


def test_anthropic_refusal_is_an_error(server, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    s = server([(200, {"content": [], "stop_reason": "refusal", "stop_details": {"category": None}})])
    cfg = llm.ModelConfig("h", "anthropic", "m", base_url=s.url, api_key_env="ANTHROPIC_API_KEY")
    with llm.use_model(cfg), pytest.raises(RuntimeError, match="refused"):
        llm.complete("a", "b")


def test_key_read_from_env_file_and_alias(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(llm, "_env_file_values", lambda: {"GOOGLE_API_KEY": "from-file"})
    assert llm._api_key(llm.ModelConfig("g", "gemini", "m", api_key_env="GEMINI_API_KEY")) == "from-file"
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    assert llm._api_key(llm.ModelConfig("g", "gemini", "m", api_key_env="GEMINI_API_KEY")) == "from-env"


class _FakeGenai:
    """Stands in for google.genai.Client: records the call, returns a canned response."""

    def __init__(self, resp=None, exc=None):
        self.calls, self.resp, self.exc = [], resp, list(exc or [])
        self.models = self

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.exc:
            raise self.exc.pop(0)
        return self.resp


def _gemini_resp(text, prompt=7, cand=4, thoughts=30):
    from types import SimpleNamespace as NS
    return NS(text=text, candidates=[NS(finish_reason="STOP")],
              usage_metadata=NS(prompt_token_count=prompt, candidates_token_count=cand, thoughts_token_count=thoughts))


def test_gemini_request_and_thinking_tokens_count_as_output(monkeypatch):
    fake = _FakeGenai(_gemini_resp("SELECT 4"))
    monkeypatch.setattr(llm, "_gemini_client", lambda key: fake)
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    cfg = llm.ModelConfig("gf", "gemini", "gemini-3.8-flash", api_key_env="GEMINI_API_KEY",
                          temperature=None, effort="medium", max_tokens=8192)
    with llm.use_model(cfg), llm.track() as calls:
        assert llm.complete("sys", "usr") == "SELECT 4"
    c = fake.calls[0]
    assert c["model"] == "gemini-3.8-flash" and c["contents"] == "usr"
    assert c["config"].system_instruction == "sys" and c["config"].temperature is None
    assert c["config"].thinking_config.thinking_level.lower() == "medium"
    assert c["config"].max_output_tokens == 8192
    assert (calls[0]["in_tokens"], calls[0]["out_tokens"]) == (7, 34)  # 4 answer + 30 thinking


def test_gemini_empty_response_is_an_error(monkeypatch):
    monkeypatch.setattr(llm, "_gemini_client", lambda key: _FakeGenai(_gemini_resp(None)))
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    with llm.use_model(llm.ModelConfig("gf", "gemini", "m", api_key_env="GEMINI_API_KEY")), \
            pytest.raises(RuntimeError, match="no text"):
        llm.complete("a", "b")


def test_gemini_retries_429_then_succeeds(monkeypatch):
    from google.genai import errors
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    fake = _FakeGenai(_gemini_resp("ok"), exc=[errors.APIError(429, {"error": {"message": "slow"}})])
    monkeypatch.setattr(llm, "_gemini_client", lambda key: fake)
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    with llm.use_model(llm.ModelConfig("gf", "gemini", "m", api_key_env="GEMINI_API_KEY")):
        assert llm.complete("a", "b") == "ok"
    assert len(fake.calls) == 2


def test_unknown_provider():
    with llm.use_model(llm.ModelConfig("m", "nope", "m")), pytest.raises(RuntimeError, match="Unknown provider"):
        llm.complete("a", "b")
