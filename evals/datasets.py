"""Benchmark loaders: BIRD (dev) and Spider (dev) -> a common `Example`.

Layouts expected (the official downloads):
  BIRD   <root>/**/dev.json  + <root>/**/dev_databases/<db_id>/<db_id>.sqlite
  Spider <root>/dev.json     + <root>/database/<db_id>/<db_id>.sqlite
"""
import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Example:
    id: str
    dataset: str
    db_id: str
    db_path: str
    question: str
    evidence: str
    gold_sql: str
    difficulty: str  # easy | medium | hard | unknown


_BIRD_DIFF = {"simple": "easy", "moderate": "medium", "challenging": "hard"}


def _find(root: Path, pattern: str) -> Path:
    hits = sorted(root.rglob(pattern))
    if not hits:
        raise FileNotFoundError(f"No '{pattern}' under {root}. Did you download/extract the dataset?")
    return hits[0]


def load_bird(root: str) -> list[Example]:
    root_p = Path(root)
    dev = _find(root_p, "dev.json")
    dbs = _find(root_p, "dev_databases")
    out = []
    for i, r in enumerate(json.loads(dev.read_text(encoding="utf-8"))):
        db = r["db_id"]
        out.append(Example(
            id=f"bird-{r.get('question_id', i)}", dataset="bird", db_id=db,
            db_path=str(dbs / db / f"{db}.sqlite"), question=r["question"],
            evidence=(r.get("evidence") or "").strip(), gold_sql=r["SQL"],
            difficulty=_BIRD_DIFF.get(r.get("difficulty", ""), "unknown"),
        ))
    return out


def load_spider(root: str) -> list[Example]:
    root_p = Path(root)
    dev = _find(root_p, "dev.json")
    dbs = dev.parent / "database"
    out = []
    for i, r in enumerate(json.loads(dev.read_text(encoding="utf-8"))):
        db = r["db_id"]
        out.append(Example(
            id=f"spider-{i}", dataset="spider", db_id=db,
            db_path=str(dbs / db / f"{db}.sqlite"), question=r["question"], evidence="",
            gold_sql=r["query"], difficulty="unknown",
        ))
    return out


LOADERS = {"bird": load_bird, "spider": load_spider}


def sample(examples: list[Example], n: int, seed: int = 0) -> list[Example]:
    """Deterministic, difficulty-stratified sample (same seed -> same questions).

    Every model is evaluated on the identical sample, which is what makes paired
    comparisons valid. Allocation is proportional to each difficulty's share, then
    topped up so the total is exactly n.
    """
    if n >= len(examples):
        return list(examples)
    rng = random.Random(seed)
    groups: dict[str, list[Example]] = {}
    for e in examples:
        groups.setdefault(e.difficulty, []).append(e)
    for g in groups.values():
        g.sort(key=lambda e: e.id)  # stable base order before shuffling
        rng.shuffle(g)
    picked: list[Example] = []
    for key in sorted(groups):
        picked += groups[key][: round(n * len(groups[key]) / len(examples))]
    rest = [e for key in sorted(groups) for e in groups[key] if e not in set(picked)]
    rng.shuffle(rest)
    picked += rest[: max(0, n - len(picked))]
    return sorted(picked[:n], key=lambda e: e.id)
