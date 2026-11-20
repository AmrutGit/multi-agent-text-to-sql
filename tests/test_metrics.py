import pytest

from evals.metrics import classify_error, compare, execute_sql, wilson_interval


def test_compare_ignores_row_order_and_duplicates():
    assert compare([(1,), (2,), (2,)], [(2,), (1,)]) == (True, "")


def test_compare_kinds():
    assert compare([], [(1,)]) == (False, "empty_result")
    assert compare([(1, 2)], [(1,)]) == (False, "wrong_columns")
    assert compare([(1,)], [(1,), (2,)]) == (False, "missing_rows")
    assert compare([(1,), (2,), (3,)], [(1,), (2,)]) == (False, "extra_rows")
    assert compare([(9,)], [(1,)]) == (False, "wrong_values")


@pytest.mark.parametrize("err,label", [
    ("SQLite rejected the query: no such column: x", "hallucinated_column"),
    ("no such table: foo", "hallucinated_table"),
    ('near "FROM": syntax error', "syntax_error"),
    ("interrupted", "timeout"),
    ("Only SELECT queries are allowed (statement starts with 'DROP').", "blocked_by_guard"),
    ("ambiguous column name: id", "ambiguous_column"),
    ("something odd", "other_sql_error"),
])
def test_classify_error(err, label):
    assert classify_error(err) == label


def test_execute_sql_rounds_floats_and_reports_errors(tiny_db):
    rows, err = execute_sql(tiny_db, "SELECT 0.1+0.2")
    assert err is None and rows == [(0.3,)]
    rows, err = execute_sql(tiny_db, "SELECT nope FROM schools")
    assert rows is None and "no such column" in err


def test_execute_sql_is_read_only(tiny_db):
    rows, err = execute_sql(tiny_db, "DELETE FROM schools")
    assert rows is None and "readonly" in err.lower()


def test_execute_sql_is_not_row_capped(tiny_db):
    # The agent caps rows at 200; scoring must not, or big gold results would never match.
    rows, _ = execute_sql(tiny_db, "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c WHERE x<1000) SELECT x FROM c")
    assert len(rows) == 1000


def test_wilson_interval_known_values():
    lo, hi = wilson_interval(30, 50)  # 60% of 50
    assert 0.45 < lo < 0.47 and 0.72 < hi < 0.74
    assert wilson_interval(0, 0) == (0.0, 0.0)
    lo, hi = wilson_interval(50, 50)
    assert hi == 1.0 and lo > 0.9
