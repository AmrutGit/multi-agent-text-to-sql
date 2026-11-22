"""Execution accuracy: does the agent's SQL return the same rows as gold_sql?
Scores whatever the orchestrator produces. Run: `make eval` (needs GEMINI_API_KEY).

Comparison is order-insensitive and rounds floats, so semantically equal results that
differ only in row order or float noise (sum order) still match.
"""
import sqlite3
from collections import defaultdict

import yaml

from src.config import settings
from src.orchestrator import run


def _norm(v):
    return round(v, 4) if isinstance(v, float) else v


def _rows(sql: str):
    con = sqlite3.connect(f"file:{settings.db_path}?mode=ro", uri=True)
    try:
        return sorted(str(tuple(_norm(v) for v in row)) for row in con.execute(sql).fetchall())
    finally:
        con.close()


def main():
    with open("evals/cases.yaml") as f:
        cases = yaml.safe_load(f)
    by_diff = defaultdict(lambda: [0, 0])  # difficulty -> [passed, total]
    total_retries = 0
    print(f"{'Q':<55} {'result':>8}")
    print("-" * 66)
    for c in cases:
        diff = c.get("difficulty", "?")
        by_diff[diff][1] += 1
        try:
            gold = _rows(c["gold_sql"])
            result = run(c["question"])
            total_retries += result["retries"]
            ok = result["valid"] and _rows(result["sql"]) == gold
        except Exception as e:
            ok = False
            print(f"  (error: {type(e).__name__}: {str(e)[:60]})")
        by_diff[diff][0] += ok
        print(f"{c['question'][:54]:<55} {'PASS' if ok else 'FAIL':>8}")
    print("-" * 66)
    for diff in ("easy", "medium", "hard"):
        if diff in by_diff:
            p, t = by_diff[diff]
            print(f"{diff:<8} {p}/{t} ({p / t * 100:.0f}%)")
    passed = sum(p for p, _ in by_diff.values())
    print(f"Execution accuracy: {passed}/{len(cases)} ({passed / len(cases) * 100:.0f}%)"
          f" | self-correction passes used: {total_retries}")


if __name__ == "__main__":
    main()
