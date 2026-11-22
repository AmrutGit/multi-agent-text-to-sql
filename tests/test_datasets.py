import json
import sqlite3
from collections import Counter
from pathlib import Path

import pytest

from evals.datasets import Example, load_bird, load_spider, sample


def _mk_db(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sqlite3.connect(path).close()


def test_load_bird_layout_and_difficulty_mapping(tmp_path):
    root = tmp_path / "dev_x"
    _mk_db(root / "dev_databases" / "d1" / "d1.sqlite")
    (root / "dev.json").write_text(json.dumps([
        {"question_id": 7, "db_id": "d1", "question": "q?", "evidence": " hint ", "SQL": "SELECT 1",
         "difficulty": "challenging"}]))
    [ex] = load_bird(str(tmp_path))
    assert ex.id == "bird-7" and ex.difficulty == "hard" and ex.evidence == "hint"
    assert Path(ex.db_path).parts[-3:] == ("dev_databases", "d1", "d1.sqlite")  # OS-independent


def test_load_spider_layout(tmp_path):
    _mk_db(tmp_path / "database" / "d1" / "d1.sqlite")
    (tmp_path / "dev.json").write_text(json.dumps([{"db_id": "d1", "question": "q", "query": "SELECT 1"}]))
    [ex] = load_spider(str(tmp_path))
    assert ex.dataset == "spider" and ex.gold_sql == "SELECT 1" and ex.evidence == ""


def test_missing_dataset_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download"):
        load_bird(str(tmp_path))


def _examples():
    out = []
    for i, d in enumerate(["easy"] * 60 + ["medium"] * 30 + ["hard"] * 10):
        out.append(Example(f"e{i:03d}", "bird", "db", "p", "q", "", "SELECT 1", d))
    return out


def test_sample_is_deterministic_and_exact_size():
    a, b = sample(_examples(), 50, seed=1), sample(_examples(), 50, seed=1)
    assert a == b and len(a) == 50
    assert sample(_examples(), 50, seed=2) != a


def test_sample_is_stratified():
    c = Counter(e.difficulty for e in sample(_examples(), 50, seed=0))
    assert c["easy"] == 30 and c["medium"] == 15 and c["hard"] == 5


def test_sample_larger_than_pool_returns_all():
    assert len(sample(_examples(), 1000)) == 100
