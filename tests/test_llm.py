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
    assert r["body"]["system"] == "sys" and r["body"]["messages"][0]["content"] == "usr"
    assert (calls[0]["in_tokens"], calls[0]["out_tokens"]) == (5, 2)


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


def test_unknown_provider():
    with llm.use_model(llm.ModelConfig("m", "nope", "m")), pytest.raises(RuntimeError, match="Unknown provider"):
        llm.complete("a", "b")
