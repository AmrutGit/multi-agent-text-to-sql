import pytest

from src.agents.schema_retriever import retrieve_schema_context, select_tables

pytestmark = pytest.mark.northwind


def test_simple_question_selects_one_table():
    assert select_tables("How many customers are there?") == ["Customers"]


def test_empty_demographics_tables_are_not_selected():
    assert "CustomerDemographics" not in select_tables("How many customers are there?")


def test_join_path_is_bridged():
    # Customers <-> Order Details have no direct FK; Orders must be pulled in.
    tables = select_tables("Top customers by total revenue")
    assert {"Customers", "Orders", "Order Details"} <= set(tables)


def test_category_value_maps_to_categories_table():
    assert "Categories" in select_tables("List suppliers of Beverages products")


def test_unmatched_question_falls_back_to_full_schema():
    ctx = retrieve_schema_context("What is the weather like?", mode="retrieve")
    assert "CREATE TABLE [Customers]" in ctx and "CREATE TABLE [Shippers]" in ctx


def test_context_is_a_slice_not_the_full_schema():
    ctx = retrieve_schema_context("How many customers are there?", mode="retrieve")
    assert "Customers" in ctx and "[Shippers]" not in ctx


def test_auto_mode_uses_full_schema_when_it_fits(monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "schema_budget_chars", 10**6)
    ctx = retrieve_schema_context("How many customers are there?", mode="auto")
    assert "[Shippers]" in ctx  # not sliced: the whole schema fits the budget


def test_auto_mode_retrieves_when_schema_exceeds_budget(monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "schema_budget_chars", 100)
    ctx = retrieve_schema_context("How many customers are there?", mode="auto")
    assert "[Customers]" in ctx and "[Shippers]" not in ctx
