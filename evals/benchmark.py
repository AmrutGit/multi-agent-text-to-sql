"""Run the pipeline on Spider/BIRD questions and write one JSONL per (model, config).

  python -m evals.benchmark --dataset bird --data-dir data/bird_raw -n 50 \
      --models gemini-flash,qlora-llama

Results go to evals/results/. Summarise/compare them with `python -m evals.report`.
"""
import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import yaml

from evals.datasets import LOADERS, Example, sample
from evals.metrics import classify_error, compare, execute_sql
from src import llm
from src.config import settings
from src.orchestrator import run

RESULTS_DIR = Path("evals/results")


def load_models(path: str = "evals/models.yaml") -> dict[str, llm.ModelConfig]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return {name: llm.ModelConfig(name=name, **cfg) for name, cfg in raw.items()}


def cost_usd(cfg: llm.ModelConfig, tin: int, tout: int, cache_write: int = 0,
             cache_read: int = 0) -> float | None:
    """`tin` is the full prompt size; `cache_write`/`cache_read` are the parts of it that
    were written to / read from the prompt cache and are priced differently."""
    if cfg.price_in is None or cfg.price_out is None:
        return None
    p_write = cfg.price_cache_write if cfg.price_cache_write is not None else 1.25 * cfg.price_in
    p_read = cfg.price_cache_read if cfg.price_cache_read is not None else 0.1 * cfg.price_in
    uncached = tin - cache_write - cache_read
    return (uncached * cfg.price_in + cache_write * p_write + cache_read * p_read
            + tout * cfg.price_out) / 1_000_000


def evaluate_one(ex: Example, cfg: llm.ModelConfig, gold, retries: int, schema_mode: str,
                 use_evidence: bool, timeout_s: float) -> dict:
    """Run one question and score it. Never raises: failures are data."""
    rec = {"id": ex.id, "db_id": ex.db_id, "difficulty": ex.difficulty, "question": ex.question,
           "gold_sql": ex.gold_sql, "model": cfg.name}
    gold_rows, gold_err = gold
    if gold_err:
        return {**rec, "outcome": "gold_error", "correct": None, "error": gold_err}

    start = time.perf_counter()
    result, exc = None, None
    with llm.use_model(cfg), llm.track() as calls:
        try:
            result = run(ex.question, db_path=ex.db_path,
                         evidence=ex.evidence if use_evidence else None, hints="",
                         explain_answer=False, max_retries=retries, schema_mode=schema_mode)
        except Exception as e:  # provider down, auth, blocked response...
            exc = e
    latency = time.perf_counter() - start
    tin = sum(c["in_tokens"] for c in calls)
    tout = sum(c["out_tokens"] for c in calls)
    cw = sum(c.get("cache_write_tokens", 0) for c in calls)
    cr = sum(c.get("cache_read_tokens", 0) for c in calls)
    rec.update(latency_s=round(latency, 3), llm_calls=len(calls), in_tokens=tin, out_tokens=tout,
               cache_write_tokens=cw, cache_read_tokens=cr, cost_usd=cost_usd(cfg, tin, tout, cw, cr))

    if exc is not None:
        return {**rec, "outcome": "llm_error", "correct": False, "error": f"{type(exc).__name__}: {exc}"}

    rec.update(pred_sql=result["sql"], retries=result["retries"], attempts=result["attempts"])
    if not result["valid"]:
        last = result["attempts"][-1]["error"] or ""
        return {**rec, "outcome": classify_error(last), "correct": False, "error": last}

    pred_rows, err = execute_sql(ex.db_path, result["sql"], timeout_s)
    if err:  # passed the agent's checks but fails uncapped / under the eval timeout
        return {**rec, "outcome": classify_error(err), "correct": False, "error": err}
    ok, kind = compare(pred_rows, gold_rows)
    return {**rec, "outcome": "pass" if ok else kind, "correct": ok, "error": ""}


def _evaluate_with_gold(ex, *, cfg, gold, retries, schema_mode, use_evidence, timeout_s):
    return evaluate_one(ex, cfg, gold[ex.id], retries, schema_mode, use_evidence, timeout_s)


def run_with_budget(examples: list, fn, workers: int, max_usd: float | None) -> tuple[list[dict], bool]:
    """Evaluate `examples` in batches of `workers`; stop before the next batch once the
    measured spend reaches `max_usd`. Returns (records, stopped_early).

    Batching keeps the overshoot to at most one batch of questions. Unpriced models
    (cost_usd None) are never stopped.
    """
    records: list[dict] = []
    spent = 0.0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i in range(0, len(examples), workers):
            if max_usd is not None and spent >= max_usd:
                return records, True
            batch = list(pool.map(fn, examples[i:i + workers]))
            spent += sum(r.get("cost_usd") or 0.0 for r in batch)
            records += batch
    return records, False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(LOADERS), required=True)
    ap.add_argument("--data-dir", required=True, help="extracted dataset root")
    ap.add_argument("-n", type=int, default=50, help="number of questions (stratified sample)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", required=True, help="comma-separated names from models.yaml")
    ap.add_argument("--models-file", default="evals/models.yaml")
    ap.add_argument("--retries", type=int, default=2, help="self-correction passes (0 = single shot)")
    ap.add_argument("--schema-mode", choices=["auto", "retrieve", "full"], default="auto")
    ap.add_argument("--no-evidence", action="store_true", help="BIRD: withhold the evidence hint")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=30.0, help="per-query seconds")
    ap.add_argument("--tag", default="", help="suffix for result file names (e.g. 'noretry')")
    ap.add_argument("--max-usd", type=float, default=None,
                    help="stop a model's run once its measured API spend reaches this (per model)")
    args = ap.parse_args(argv)

    settings.query_timeout_s = args.timeout
    models = load_models(args.models_file)
    wanted = [m.strip() for m in args.models.split(",")]
    unknown = [m for m in wanted if m not in models]
    if unknown:
        raise SystemExit(f"Unknown model(s) {unknown}. Defined: {sorted(models)}")

    examples = sample(LOADERS[args.dataset](args.data_dir), args.n, args.seed)
    print(f"{len(examples)} questions from {args.dataset} (seed {args.seed}); computing gold results...")
    gold = {ex.id: execute_sql(ex.db_path, ex.gold_sql, args.timeout) for ex in examples}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=5).stdout.strip()
    except Exception:
        sha = ""
    for name in wanted:
        cfg = models[name]
        print(f"\n== {name} ({cfg.provider}:{cfg.model})")
        fn = partial(_evaluate_with_gold, cfg=cfg, gold=gold, retries=args.retries,
                     schema_mode=args.schema_mode, use_evidence=not args.no_evidence, timeout_s=args.timeout)
        records, stopped = run_with_budget(examples, fn, args.workers, args.max_usd)
        scored = [r for r in records if r["correct"] is not None]
        k = sum(r["correct"] for r in scored)
        spent = sum(r.get("cost_usd") or 0.0 for r in records)
        print(f"   execution accuracy: {k}/{len(scored)} ({k / max(len(scored), 1):.0%}), spend ${spent:.4f}")
        if stopped:
            print(f"   STOPPED at --max-usd {args.max_usd}: {len(records)}/{len(examples)} questions run")
        meta = {"_meta": {
            "model": name, "provider": cfg.provider, "model_id": cfg.model, "dataset": args.dataset,
            "n": len(examples), "seed": args.seed, "retries": args.retries,
            "schema_mode": args.schema_mode, "evidence": not args.no_evidence, "git_sha": sha,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "price_in": cfg.price_in, "price_out": cfg.price_out, "effort": cfg.effort,
            "temperature": cfg.temperature, "workers": args.workers,
            "spend_usd": round(spent, 6), "stopped_by_budget": stopped,
        }}
        tag = f"__{args.tag}" if args.tag else ""
        out = RESULTS_DIR / f"{name}__{args.dataset}__n{len(examples)}__r{args.retries}__{args.schema_mode}{tag}.jsonl"
        with open(out, "w") as f:
            f.write(json.dumps(meta) + "\n")
            for r in records:
                f.write(json.dumps(r, default=str) + "\n")
        print(f"   wrote {out}")


if __name__ == "__main__":
    main()
