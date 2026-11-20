"""Schema retrieval (the RAG step).

Approach: lightweight lexical retrieval, no embeddings or extra LLM call.
  1. tokenise the question and score every table on name / column / synonym matches
  2. keep the top-scoring tables
  3. add "bridge" tables so the selected tables can actually be JOINed
     (e.g. Customers + Products need Orders and Order Details in between)
  4. fall back to the FULL schema if nothing matches — a false negative here means
     the writer can't answer at all, whereas a bloated prompt only costs tokens.

Tradeoff: Northwind (13 tables) fits in a prompt, so retrieval is mostly about
discipline and scalability: full-schema is simplest but bloats context and hurts
accuracy as schemas grow; retrieval scales but can miss a table. The fallback and the
bridge step are the guards against that miss.
"""
import re
from collections import deque

from src.config import settings
from src.db.schema import get_empty_tables, get_foreign_keys, get_full_schema, get_table_schemas

MAX_TABLES = 7  # if bridging balloons past this, retrieval isn't helping: use full schema
TOP_K = 4  # tables kept before bridging
RELATIVE_CUTOFF = 0.4  # drop tables scoring < 40% of the best one

# Business words that don't appear in any table/column name but imply a table.
_SYNONYMS = {
    "revenue": ["Order Details"], "sales": ["Order Details"], "sold": ["Order Details"],
    "sell": ["Order Details"], "quantity": ["Order Details"], "discount": ["Order Details"],
    "spent": ["Order Details"], "spend": ["Order Details"], "income": ["Order Details"],
    "salesperson": ["Employees"], "rep": ["Employees"], "staff": ["Employees"],
    "manager": ["Employees"], "boss": ["Employees"], "hired": ["Employees"],
    "buyer": ["Customers"], "client": ["Customers"], "purchase": ["Orders"],
    "bought": ["Orders"], "shipped": ["Orders", "Shippers"], "delivery": ["Orders"],
    "freight": ["Orders"], "carrier": ["Shippers"], "vendor": ["Suppliers"],
    "supplier": ["Suppliers"], "stock": ["Products"], "price": ["Products"],
    "item": ["Products"], "expensive": ["Products"], "cheap": ["Products"],
    "cheapest": ["Products"], "discontinued": ["Products"],
    # category *values* live in data, not in the schema, so map them to Categories
    "beverage": ["Categories"], "condiment": ["Categories"], "seafood": ["Categories"],
    "dairy": ["Categories"], "produce": ["Categories"], "confection": ["Categories"],
    "meat": ["Categories"], "grain": ["Categories"], "cereal": ["Categories"],
}

_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "for", "to", "is", "are", "was", "were", "by",
    "and", "or", "with", "what", "which", "who", "how", "many", "much", "list", "show",
    "me", "all", "each", "per", "than", "that", "have", "has", "had", "most", "least",
    "top", "from", "do", "does", "did", "there", "their", "we", "our", "give", "get",
}

# Few-shot examples: teach SQLite dialect + this schema's quirks. Kept disjoint from
# evals/cases.yaml so the eval measures generalisation, not memorisation.
FEW_SHOT = """\
-- Q: How many orders did each employee handle?
SELECT e.FirstName, e.LastName, COUNT(*) AS order_count
FROM Orders o JOIN Employees e ON e.EmployeeID = o.EmployeeID
GROUP BY e.EmployeeID ORDER BY order_count DESC;

-- Q: Total revenue per category
-- (revenue = UnitPrice * Quantity * (1 - Discount), from "Order Details")
SELECT c.CategoryName, SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS revenue
FROM "Order Details" od
JOIN Products p ON p.ProductID = od.ProductID
JOIN Categories c ON c.CategoryID = p.CategoryID
GROUP BY c.CategoryID;"""


def _stem(word: str) -> str:
    """Crude plural stripping: categories->category, employees->employee, boxes->box."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("sses", "xes", "ches", "shes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _tokens(text: str) -> set[str]:
    # split CamelCase (CustomerID -> customer id) and non-alphanumerics
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return {_stem(w) for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOPWORDS}


def _overlap(q_tokens: set[str], target: set[str]) -> int:
    """How many question tokens match a table/column token, allowing sub-word matches
    (the question says 'SAT', the table is 'satscores'; 'ship' vs 'shippers')."""
    n = 0
    for q in q_tokens:
        for t in target:
            if q == t or (len(q) >= 3 and len(t) >= 3 and (q in t or t in q)):
                n += 1
                break
    return n


def _score_tables(question: str, db_path: str | None = None) -> dict[str, float]:
    q_tokens = _tokens(question)
    schemas = get_table_schemas(db_path)
    scores = {t: 0.0 for t in schemas}
    for table, ddl in schemas.items():
        name_tokens = _tokens(table)
        # Name match is strong evidence, normalised by name length so 'customers' prefers
        # Customers (1/1 tokens) over CustomerDemographics (1/2 tokens).
        scores[table] += 3.0 * _overlap(q_tokens, name_tokens) / max(len(name_tokens), 1)
        # Columns are weaker (ID columns appear everywhere); ignore generic 'id'.
        col_tokens = (_tokens(ddl) - name_tokens) - {"id", "text", "integer", "null", "key",
                                                     "foreign", "primary", "references",
                                                     "table", "create", "numeric", "real",
                                                     "not", "default", "check", "on",
                                                     "action", "no", "update", "delete",
                                                     "datetime", "date", "blob", "autoincrement"}
        scores[table] += 1.0 * _overlap(q_tokens, col_tokens)
    for word in re.findall(r"[a-z]+", question.lower()):
        for table in _SYNONYMS.get(word, []) or _SYNONYMS.get(_stem(word), []):
            if table in scores:
                scores[table] += 4.0
    for table in get_empty_tables(db_path):
        scores[table] *= 0.25
    return scores


def _bridge(selected: list[str], db_path: str | None = None) -> list[str]:
    """Add the tables on the shortest FK path between every selected pair."""
    graph = get_foreign_keys(db_path)
    result = list(selected)
    for i, a in enumerate(selected):
        for b in selected[i + 1:]:
            prev, queue, seen = {}, deque([a]), {a}
            while queue:
                node = queue.popleft()
                if node == b:
                    break
                for nxt in sorted(graph.get(node, ())):  # sorted: set order varies per process
                    if nxt not in seen:
                        seen.add(nxt)
                        prev[nxt] = node
                        queue.append(nxt)
            node = b
            while node in prev:  # walk back, collecting intermediates
                node = prev[node]
                if node != a and node not in result:
                    result.append(node)
    return result


def select_tables(question: str, db_path: str | None = None) -> list[str]:
    scores = _score_tables(question, db_path)
    ranked = sorted(((t, s) for t, s in scores.items() if s > 0), key=lambda kv: -kv[1])
    if not ranked:
        return []
    best = ranked[0][1]
    keep = [t for t, s in ranked[:TOP_K] if s >= RELATIVE_CUTOFF * best]
    return _bridge(keep, db_path)


def retrieve_schema_context(question: str, db_path: str | None = None,
                            mode: str | None = None) -> str:
    """Return the schema context for `question`.

    mode="auto":     full schema if it fits `schema_budget_chars`, else retrieve (default).
    mode="retrieve": only the relevant tables (+ Northwind few-shot when applicable).
    mode="full":     every table.
    Why auto: measured on BIRD dev, lexical retrieval kept all needed tables for only
    ~84% of questions (a hard accuracy ceiling) while shrinking prompts by <50%, because
    typical schemas are tiny. Retrieval only earns its risk once the schema stops fitting.
    """
    schemas = get_table_schemas(db_path)
    mode = mode or settings.schema_mode
    if mode == "auto":
        mode = "full" if len(get_full_schema(db_path)) <= settings.schema_budget_chars else "retrieve"
    tables = select_tables(question, db_path) if mode == "retrieve" else []
    if not tables or len(tables) > MAX_TABLES:
        tables = list(schemas)  # full schema, or fallback when retrieval found nothing
    ddl = "\n\n".join(schemas[t] for t in tables)
    # The few-shot examples reference Northwind tables: only show them on that schema,
    # otherwise they would mislead the model on any other database.
    if "Order Details" in schemas and "Categories" in schemas:
        return f"{ddl}\n\nExamples:\n{FEW_SHOT}"
    return ddl
