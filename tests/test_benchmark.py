"""End-to-end benchmark logic with a scripted fake model on a tiny DB."""
import json

import pytest

from evals import benchmark, report
from evals.datasets import Example
from evals.metrics import execute_sql
from src import llm

CFG = llm.ModelConfig("fake", "openai_compat", "m", price_in=1.0, price_out=2.0)


def _ex(db, gold="SELECT name FROM schools WHERE county='Marin'"):
    return Example("t-1", "bird", "tiny", db, "Which Marin schools?", "", gold, "easy")


def _score(db, fake_llm, replies, retries=2, gold=None):
    ex = _ex(db) if gold is None else _ex(db, gold)
    fake_llm(replies)
    return benchmark.evaluate_one(ex, CFG, execute_sql(db, ex.gold_sql), retries, "full", True, 5)


def test_pass(tiny_db, fake_llm):
    r = _score(tiny_db, fake_llm, ["SELECT name FROM schools WHERE county = 'Marin'"])
    assert r["outcome"] == "pass" and r["correct"] is True and r["retries"] == 0


def test_wrong_values_vs_missing_rows(tiny_db, fake_llm):
    r = _score(tiny_db, fake_llm, ["SELECT name FROM schools WHERE county = 'Napa'"])
    assert r["outcome"] == "wrong_values" and r["correct"] is False
    r = _score(tiny_db, fake_llm, ["SELECT name FROM schools WHERE id = 1"])
    assert r["outcome"] == "missing_rows"


def test_hallucinated_column_is_classified_after_retries(tiny_db, fake_llm):
    r = _score(tiny_db, fake_llm, ["SELECT title FROM schools"] * 3)
    assert r["outcome"] == "hallucinated_column" and r["retries"] == 2 and "no such column" in r["error"]


def test_self_correction_counts_as_retry_and_passes(tiny_db, fake_llm):
    r = _score(tiny_db, fake_llm, ["SELECT title FROM schools", "SELECT name FROM schools WHERE county='Marin'"])
    assert r["correct"] is True and r["retries"] == 1


def test_llm_exception_is_recorded_not_raised(tiny_db, monkeypatch):
    def boom(system, user):
        raise RuntimeError("provider down")
    monkeypatch.setattr("src.agents.sql_writer.complete", boom)
    ex = _ex(tiny_db)
    r = benchmark.evaluate_one(ex, CFG, execute_sql(tiny_db, ex.gold_sql), 2, "full", True, 5)
    assert r["outcome"] == "llm_error" and "provider down" in r["error"]


def test_broken_gold_is_excluded_not_scored(tiny_db, fake_llm):
    ex = _ex(tiny_db, "SELECT nope FROM schools")
    r = benchmark.evaluate_one(ex, CFG, execute_sql(tiny_db, ex.gold_sql), 2, "full", True, 5)
    assert r["outcome"] == "gold_error" and r["correct"] is None


def test_cost_and_token_accounting(tiny_db, monkeypatch):
    # Patch at the provider level so llm.complete's real tracking path is exercised.
    monkeypatch.setitem(llm._PROVIDERS, "openai_compat",
                        lambda cfg, s, u: ("SELECT name FROM schools WHERE county='Marin'", 1000, 100))
    ex = _ex(tiny_db)
    r = benchmark.evaluate_one(ex, CFG, execute_sql(tiny_db, ex.gold_sql), 2, "full", True, 5)
    assert (r["in_tokens"], r["out_tokens"], r["llm_calls"]) == (1000, 100, 1)
    assert r["cost_usd"] == pytest.approx((1000 * 1.0 + 100 * 2.0) / 1e6)
    assert r["latency_s"] >= 0


def test_unknown_price_gives_none_not_zero():
    assert benchmark.cost_usd(llm.ModelConfig("m", "x", "m"), 1000, 1000) is None


def test_models_yaml_is_valid():
    models = benchmark.load_models()
    assert {"qlora-llama", "claude-haiku", "gemini-flash"} <= set(models)
    assert models["qlora-llama"].provider == "openai_compat"


def test_api_models_are_priced_and_send_no_temperature():
    models = benchmark.load_models()
    for name in ("gemini-flash", "claude-haiku"):
        m = models[name]
        assert m.price_in and m.price_out and m.price_out > m.price_in
        assert m.temperature is None and m.effort == "medium"


def _fake_record(ex, cost):
    return {"id": ex, "correct": True, "cost_usd": cost}


def test_budget_stops_before_next_batch():
    recs, stopped = benchmark.run_with_budget(list(range(10)), lambda e: _fake_record(e, 0.5), 2, max_usd=1.0)
    assert stopped and len(recs) == 2  # batch 1 spends $1.00 -> batch 2 never starts


def test_budget_none_or_unpriced_runs_everything():
    recs, stopped = benchmark.run_with_budget(list(range(5)), lambda e: _fake_record(e, None), 2, max_usd=0.01)
    assert not stopped and [r["id"] for r in recs] == list(range(5))  # order preserved
    recs, stopped = benchmark.run_with_budget(list(range(5)), lambda e: _fake_record(e, 9.0), 3, max_usd=None)
    assert not stopped and len(recs) == 5


def test_report_flags_budget_stopped_run(tmp_path):
    p = tmp_path / "a.jsonl"
    _write_run(p, "modelA", {i: True for i in range(3)})
    lines = p.read_text().splitlines()
    meta = json.loads(lines[0])
    meta["_meta"].update(stopped_by_budget=True, n=10)
    p.write_text("\n".join([json.dumps(meta), *lines[1:]]) + "\n")
    assert "stopped by --max-usd" in report.render([report.load(str(p))])


def _write_run(path, model, correct_by_id, retries=2):
    meta = {"_meta": {"model": model, "dataset": "bird", "n": len(correct_by_id), "seed": 0,
                      "retries": retries, "schema_mode": "retrieve", "evidence": True}}
    with open(path, "w") as f:
        f.write(json.dumps(meta) + "\n")
        for i, ok in correct_by_id.items():
            f.write(json.dumps({"id": i, "difficulty": "easy", "correct": ok,
                                "outcome": "pass" if ok else "wrong_values", "latency_s": 1.0 + i,
                                "in_tokens": 100, "out_tokens": 10, "cost_usd": 0.001,
                                "retries": 0}) + "\n")


def test_report_renders_ci_failures_and_paired_comparison(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write_run(a, "modelA", {i: i < 8 for i in range(10)})
    _write_run(b, "modelB", {i: i < 5 for i in range(10)})
    text = report.render([report.load(str(a)), report.load(str(b))])
    assert "8/10 (80%)" in text and "5/10 (50%)" in text and "wrong_values" in text
    assert "| 3 | 0 |" in text  # 3 questions only A got right, 0 only B
    assert "sign-test" in text


def test_sign_test_known_values():
    assert report.sign_test(0, 0) == 1.0
    assert report.sign_test(10, 0) == pytest.approx(2 * (1 / 1024))
    assert report.sign_test(5, 5) == 1.0


def test_percentile():
    assert report.percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert report.percentile([1, 2], 0.5) == 1.5


def test_failures_view_lists_gold_vs_pred(tmp_path):
    f = tmp_path / "f.jsonl"
    f.write_text(json.dumps({"_meta": {"model": "m", "dataset": "bird", "n": 2, "seed": 0, "retries": 2,
                                       "schema_mode": "auto"}}) + "\n" +
                 json.dumps({"id": "b-1", "difficulty": "hard", "correct": False, "outcome": "wrong_values",
                             "question": "Q1?", "gold_sql": "SELECT 1", "pred_sql": "SELECT 2", "error": ""}) + "\n" +
                 json.dumps({"id": "b-2", "difficulty": "easy", "correct": True, "outcome": "pass"}) + "\n")
    text = report.render_failures(report.load(str(f)))
    assert "b-1" in text and "GOLD: SELECT 1" in text and "PRED: SELECT 2" in text and "b-2" not in text
    assert "b-1" not in report.render_failures(report.load(str(f)), outcome="timeout")
