.PHONY: install data run eval test lint ci bird audit bench report
install: ; pip install -r requirements-dev.txt
data:    ; python -m src.db.load_data
run:     ; uvicorn src.api.main:app --reload
eval:    ; python -m evals.run_eval                 # quick Northwind sanity eval (15 cases)
test:    ; pytest -q
lint:    ; ruff check .
ci: lint test                                       # exactly what GitHub Actions runs

# --- benchmark (BIRD) ---
bird:    ; python -m evals.fetch_bird               # one-off, ~350 MB
audit:   ; python -m evals.audit --dataset bird --data-dir data/bird_raw   # no LLM needed
# usage: make bench MODELS=gemini-flash,qlora-llama N=50
MODELS ?= gemini-flash
N ?= 50
bench:   ; python -m evals.benchmark --dataset bird --data-dir data/bird_raw -n $(N) --models $(MODELS)
report:  ; python -m evals.report
