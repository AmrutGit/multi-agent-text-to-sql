import pytest
from src.agents.validator import validate


@pytest.mark.parametrize("sql", [
    "DROP TABLE Customers;",
    "DELETE FROM Orders;",
    "UPDATE Products SET UnitPrice = 0;",
])
def test_rejects_writes(sql):
    """YOUR TODO: once validate() is implemented, this should pass."""
    ok, _ = validate(sql)
    assert ok is False


def test_accepts_select():
    ok, _ = validate("SELECT * FROM Customers LIMIT 1;")
    assert ok is True
