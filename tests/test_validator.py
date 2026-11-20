import pytest

from src.agents.validator import validate

pytestmark = pytest.mark.northwind


@pytest.mark.parametrize("sql", [
    "DROP TABLE Customers;",
    "DELETE FROM Orders;",
    "UPDATE Products SET UnitPrice = 0;",
    "INSERT INTO Shippers (CompanyName) VALUES ('x');",
    "REPLACE INTO Shippers (CompanyName) VALUES ('x');",
    "PRAGMA table_info(Orders);",
    "ATTACH DATABASE 'x.db' AS x;",
    "SELECT 1; DROP TABLE Customers",          # stacked statement
    "/* sneaky */ DELETE FROM Orders",         # leading comment
    "",
    "   ;  ",
])
def test_rejects_unsafe_or_empty(sql):
    ok, err = validate(sql)
    assert ok is False and err


@pytest.mark.parametrize("sql", [
    "SELECT * FROM Customers LIMIT 1;",
    "WITH t AS (SELECT * FROM Products) SELECT COUNT(*) FROM t",
    "SELECT * FROM Customers WHERE CompanyName = 'a;b--c'",   # ; and -- inside a string
    "SELECT * FROM Products WHERE ProductName = 'Drop it'",    # keyword inside a string
    "SELECT replace(City, 'a', 'b') FROM Customers",           # REPLACE() is a function
    'SELECT * FROM "Order Details" LIMIT 1',
])
def test_accepts_valid_selects(sql):
    ok, err = validate(sql)
    assert ok is True, err


def test_planner_error_is_returned_for_unknown_column():
    ok, err = validate("SELECT nope FROM Customers")
    assert ok is False and "no such column" in err


def test_planner_error_is_returned_for_unknown_table():
    ok, err = validate("SELECT * FROM Nope")
    assert ok is False and "no such table" in err
