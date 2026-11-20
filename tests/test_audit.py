from evals.audit import audit, gold_tables
from evals.datasets import Example


def test_gold_tables_handles_quoting_and_aliases():
    tabs = ["schools", "scores", "audit_log"]
    assert gold_tables("SELECT * FROM schools s JOIN `scores` c ON c.school_id = s.id", tabs) == {"schools", "scores"}
    assert gold_tables('SELECT 1 FROM "audit_log"', tabs) == {"audit_log"}
    assert gold_tables("SELECT school_id FROM scores", tabs) == {"scores"}  # 'schools' != 'school_id'


def test_audit_reports_recall_and_validator_rejections(tiny_db):
    ex = [
        Example("1", "bird", "t", tiny_db, "math score of each school", "", "SELECT * FROM schools JOIN scores ON school_id = id", "easy"),
        Example("2", "bird", "t", tiny_db, "list all notes", "", "SELECT note FROM audit_log", "easy"),
        Example("3", "bird", "t", tiny_db, "schools", "", "DELETE FROM schools", "easy"),  # unsafe gold
    ]
    r = audit(ex)
    assert r["n"] == 3 and 0 <= r["recall"] <= 1
    assert [x[0] for x in r["rejects"]] == ["3"]
