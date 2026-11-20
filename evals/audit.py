"""No-LLM audit of the pipeline's deterministic parts, on real benchmark data.

  python -m evals.audit --dataset bird --data-dir data/bird_raw

Measures, over EVERY question in the dev split (not a sample, since no model is called):
  1. Retrieval recall: are all tables the gold SQL needs in the retrieved schema slice?
     A miss here is a guaranteed downstream failure, so it is an upper bound on accuracy.
  2. Prompt size: retrieved slice vs. full schema (the benefit retrieval buys).
  3. Validator false rejections: valid gold queries the guard wrongly rejects.
These are exactly the failure sources you can fix without touching the model.
"""
import argparse
import re
from collections import defaultdict

from evals.datasets import LOADERS
from src.agents.schema_retriever import retrieve_schema_context, select_tables
from src.agents.validator import validate
from src.db.schema import get_full_schema, get_table_schemas


def gold_tables(sql: str, tables: list[str]) -> set[str]:
    """Tables referenced by `sql`: identifier match, quote/backtick/bracket tolerant."""
    s = sql.lower()
    found = set()
    for t in tables:
        pat = r"(?<![\w])[`\"\[]?" + re.escape(t.lower()) + r"[`\"\]]?(?![\w])"
        if re.search(pat, s):
            found.add(t)
    return found


def audit(examples) -> dict:
    n = hit = 0
    sel_sizes, total_sizes, ctx_chars, full_chars = [], [], [], []
    misses, rejects = [], []
    per_db = defaultdict(lambda: [0, 0])
    for ex in examples:
        tables = list(get_table_schemas(ex.db_path))
        need = gold_tables(ex.gold_sql, tables)
        if not need:
            continue
        sel = set(select_tables(ex.question + " " + ex.evidence, ex.db_path)) or set(tables)
        n += 1
        ok = need <= sel
        hit += ok
        per_db[ex.db_id][0] += ok
        per_db[ex.db_id][1] += 1
        if not ok:
            misses.append((ex.id, ex.question, sorted(need - sel)))
        sel_sizes.append(len(sel))
        total_sizes.append(len(tables))
        ctx_chars.append(len(retrieve_schema_context(ex.question + " " + ex.evidence, ex.db_path, "retrieve")))
        full_chars.append(len(get_full_schema(ex.db_path)))
        valid, err = validate(ex.gold_sql, ex.db_path)
        if not valid:
            rejects.append((ex.id, err, ex.gold_sql[:120]))
    def avg(xs):
        return sum(xs) / max(len(xs), 1)

    return {"n": n, "recall": hit / max(n, 1), "avg_selected": avg(sel_sizes), "avg_total": avg(total_sizes),
            "avg_ctx_chars": avg(ctx_chars), "avg_full_chars": avg(full_chars),
            "misses": misses, "rejects": rejects, "per_db": dict(per_db)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(LOADERS), required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--show", type=int, default=8, help="how many example misses to print")
    a = ap.parse_args(argv)
    r = audit(LOADERS[a.dataset](a.data_dir))
    print(f"questions audited: {r['n']}")
    print(f"retrieval recall (all gold tables kept): {r['recall']:.1%}  <- accuracy ceiling from retrieval")
    print(f"tables kept: {r['avg_selected']:.1f} of {r['avg_total']:.1f} on average")
    print(f"prompt schema size: {r['avg_ctx_chars']:.0f} chars vs {r['avg_full_chars']:.0f} full "
          f"({r['avg_ctx_chars'] / max(r['avg_full_chars'], 1):.0%})")
    print(f"validator false rejections on gold SQL: {len(r['rejects'])}/{r['n']}")
    for q, e, sql in r["rejects"][: a.show]:
        print(f"   {q}: {e} | {sql}")
    print("recall by database:")
    for db, (h, t) in sorted(r["per_db"].items()):
        print(f"   {db:<28} {h}/{t} ({h / t:.0%})")
    print("example retrieval misses:")
    for q, question, missing in r["misses"][: a.show]:
        print(f"   {q}: missing {missing} | {question[:90]}")


if __name__ == "__main__":
    main()
