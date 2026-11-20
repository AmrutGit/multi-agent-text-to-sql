import pytest

from src.db.schema import get_full_schema

pytestmark = pytest.mark.northwind


def test_schema_lists_tables():
    ddl = get_full_schema()
    assert "CREATE TABLE" in ddl
    assert "Customers" in ddl  # requires `make data` first
