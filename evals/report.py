"""Summarise and compare benchmark runs.   python -m evals.report [files...]

Reports per run: execution accuracy (with 95% CI), accuracy by difficulty, failure-type
breakdown, latency (p50/p95) and token cost. With 2+ runs on the SAME questions it adds
a paired comparison (which questions one model gets right and the other doesn't) and an
exact sign-test p-value, because at n=50 two accuracies that differ by 8 points are
usually statistically indistinguishable.
"""
import glob
import json
import sys
from collections import Counter
from functools import partial
from math import comb

from evals.metrics import wilson_interval


def load(path: str) -> dict:
    with open(path) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    meta = lines[0]["_meta"]
    return {"path": path, "meta": meta, "records": lines[1:]}


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = (len(s) - 1) * q
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def summarize(run: dict) -> dict:
    recs = run["records"]
    scored = [r for r in recs if r["correct"] is not None]  # excludes gold_error
    k = sum(r["correct"] for r in scored)
    lo, hi = wilson_interval(k, len(scored))
    by_diff = {}
    for d in ("easy", "medium", "hard", "unknown"):
        sub = [r for r in scored if r["difficulty"] == d]
        if sub:
            by_diff[d] = (sum(r["correct"] for r in sub), len(sub))
    fails = Counter(r["outcome"] for r in scored if not r["correct"])
    lat = [r["latency_s"] for r in scored if "latency_s" in r]
    costs = [r["cost_usd"] for r in scored if r.get("cost_usd") is not None]
    return {
        "k": k, "n": len(scored), "acc": k / max(len(scored), 1), "ci": (lo, hi),
        "gold_errors": len(recs) - len(scored), "by_diff": by_diff, "fails": fails,
        "p50": percentile(lat, 0.5), "p95": percentile(lat, 0.95),
        "tokens_in": sum(r.get("in_tokens", 0) for r in scored) / max(len(scored), 1),
        "tokens_out": sum(r.get("out_tokens", 0) for r in scored) / max(len(scored), 1),
        "cost_per_q": (sum(costs) / len(costs)) if len(costs) == len(scored) and costs else None,
        "retry_used": sum(1 for r in scored if r.get("retries", 0) > 0),
        "retry_fixed": sum(1 for r in scored if r.get("retries", 0) > 0 and r["correct"]),
    }


def _cell(by_diff: dict, key: str) -> str:
    return f"{by_diff[key][0]}/{by_diff[key][1]}" if key in by_diff else "–"


def label(run: dict) -> str:
    m = run["meta"]
    return f"{m['model']} (r{m['retries']}, {m['schema_mode']}{'' if m.get('evidence', True) else ', no-evid'})"


def sign_test(only_a: int, only_b: int) -> float:
    """Two-sided exact sign test on discordant pairs."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def render(runs: list[dict]) -> str:
    sums = [summarize(r) for r in runs]
    out = []
    m0 = runs[0]["meta"]
    out.append(f"### {m0['dataset'].upper()} — n={sums[0]['n']} questions (seed {m0['seed']})\n")

    out.append("| Model / config | Exec. accuracy | 95% CI | Easy | Medium | Hard | p50 s | p95 s | Tokens in/out | $/question |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for r, s in zip(runs, sums, strict=True):
        d = s["by_diff"]
        cell = partial(_cell, d)
        cost = f"${s['cost_per_q']:.4f}" if s["cost_per_q"] is not None else "n/a"
        out.append(f"| {label(r)} | **{s['k']}/{s['n']} ({_pct(s['acc'])})** | {_pct(s['ci'][0])}–{_pct(s['ci'][1])} "
                   f"| {cell('easy')} | {cell('medium')} | {cell('hard')} | {s['p50']:.1f} | {s['p95']:.1f} "
                   f"| {s['tokens_in']:.0f}/{s['tokens_out']:.0f} | {cost} |")

    types = sorted({t for s in sums for t in s["fails"]}, key=lambda t: -sum(s["fails"].get(t, 0) for s in sums))
    if types:
        out.append("\n**Failure types** (count of questions)\n")
        out.append("| Failure type | " + " | ".join(label(r) for r in runs) + " |")
        out.append("|---|" + "---|" * len(runs))
        for t in types:
            out.append(f"| {t} | " + " | ".join(str(s["fails"].get(t, 0)) for s in sums) + " |")

    out.append("\n**Self-correction loop**\n")
    for r, s in zip(runs, sums, strict=True):
        out.append(f"- {label(r)}: retry triggered on {s['retry_used']} questions, "
                   f"{s['retry_fixed']} of those ended correct.")

    if len(runs) > 1:
        out.append("\n**Paired comparison** (same questions; only discordant pairs carry information)\n")
        out.append("| A | B | only A correct | only B correct | sign-test p |")
        out.append("|---|---|---|---|---|")
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                a = {r["id"]: r["correct"] for r in runs[i]["records"] if r["correct"] is not None}
                b = {r["id"]: r["correct"] for r in runs[j]["records"] if r["correct"] is not None}
                shared = a.keys() & b.keys()
                oa = sum(a[q] and not b[q] for q in shared)
                ob = sum(b[q] and not a[q] for q in shared)
                out.append(f"| {label(runs[i])} | {label(runs[j])} | {oa} | {ob} | {sign_test(oa, ob):.2f} |")
        out.append("\n_p < 0.05 is the usual bar; above it, the difference may be noise at this sample size._")
    for r in runs:
        if r["meta"].get("stopped_by_budget"):
            out.append(f"\n_{label(r)} was stopped by --max-usd after {len(r['records'])} of "
                       f"{r['meta']['n']} questions; its numbers cover only those._")
    if any(s["gold_errors"] for s in sums):
        out.append(f"\n_{sums[0]['gold_errors']} question(s) excluded: the gold SQL itself failed to execute._")
    return "\n".join(out)


def render_failures(run: dict, outcome: str | None = None) -> str:
    """Every wrong answer with gold vs predicted SQL, for manual review/labelling."""
    out = [f"## Failures: {label(run)}"]
    for r in run["records"]:
        if r["correct"] is False and (outcome is None or r["outcome"] == outcome):
            out += [f"\n### {r['id']} [{r['difficulty']}] -> {r['outcome']}", f"Q:    {r['question']}",
                    f"GOLD: {r['gold_sql']}", f"PRED: {r.get('pred_sql', '(none)')}"]
            if r.get("error"):
                out.append(f"ERR:  {r['error']}")
    return "\n".join(out)


def main(argv=None):
    argv = list(argv if argv is not None else sys.argv[1:])
    if "--failures" in argv:  # --failures [--outcome TYPE] FILE...
        argv.remove("--failures")
        outcome = None
        if "--outcome" in argv:
            i = argv.index("--outcome")
            outcome = argv[i + 1]
            del argv[i:i + 2]
        print("\n\n".join(render_failures(load(p), outcome) for p in argv))
        return
    paths = argv or sorted(glob.glob("evals/results/*.jsonl"))
    if not paths:
        raise SystemExit("No result files. Run `python -m evals.benchmark ...` first.")
    runs = [load(p) for p in paths]
    groups: dict[tuple, list] = {}
    for r in runs:  # only compare runs over the same dataset+sample
        m = r["meta"]
        groups.setdefault((m["dataset"], m["n"], m["seed"]), []).append(r)
    print("\n\n".join(render(g) for g in groups.values()))


if __name__ == "__main__":
    main()
