"""Execution accuracy: does the agent's SQL return the same rows as gold_sql?
Plumbing is done — it scores whatever your orchestrator produces. Run: `make eval`.
"""
import sqlite3
import yaml
from src.config import settings
from src.orchestrator import run


def _rows(sql: str):
    con = sqlite3.connect(f"file:{settings.db_path}?mode=ro", uri=True)
    try:
        return sorted(map(str, con.execute(sql).fetchall()))
    finally:
        con.close()


def main():
    cases = yaml.safe_load(open("evals/cases.yaml"))
    passed = 0
    print(f"{'Q':<55} {'result':>8}")
    print("-" * 66)
    for c in cases:
        try:
            gold = _rows(c["gold_sql"])
            got = _rows(run(c["question"])["sql"])
            ok = gold == got
        except NotImplementedError:
            print("Agent not implemented yet — build the orchestrator first.")
            return
        except Exception as e:
            ok = False
            print(f"  (error: {type(e).__name__}: {str(e)[:40]})")
        passed += ok
        print(f"{c['question'][:54]:<55} {'PASS' if ok else 'FAIL':>8}")
    print("-" * 66)
    print(f"Execution accuracy: {passed}/{len(cases)} ({passed/len(cases)*100:.0f}%)")


if __name__ == "__main__":
    main()
