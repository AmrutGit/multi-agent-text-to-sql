.PHONY: install data run eval test
install: ; pip install -r requirements.txt
data:    ; python -m src.db.load_data
run:     ; uvicorn src.api.main:app --reload
eval:    ; python -m evals.run_eval
test:    ; pytest -q
