.PHONY: setup data models test lint all report
setup: ; pip install -e ".[dev,fetch]"
data: ; bash scripts/fetch_data.sh
models: ; python scripts/fetch_models.py
test: ; pytest -q
lint: ; ruff check . && ruff format --check .
all: ; bash scripts/run_all.sh
report: ; python -m ftvr.run report
