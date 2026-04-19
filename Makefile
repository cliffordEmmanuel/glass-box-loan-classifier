# Common project recipes. All targets assume `uv sync` has been run.

.PHONY: help install data eda train explain drift pipeline dashboard test lint typecheck fmt clean

help:
	@echo "Targets:"
	@echo "  install    - uv sync (install deps)"
	@echo "  data       - download the Kaggle dataset into data/raw/"
	@echo "  eda        - run exploratory data analysis"
	@echo "  train      - train, tune, calibrate, and compare models"
	@echo "  explain    - generate SHAP / LIME / PDP / consistency outputs"
	@echo "  drift      - compute PSI drift report"
	@echo "  pipeline   - run eda + train + explain + drift"
	@echo "  dashboard  - launch the Streamlit dashboard"
	@echo "  test       - run pytest"
	@echo "  lint       - run ruff"
	@echo "  typecheck  - run mypy"
	@echo "  fmt        - apply ruff formatting and import-sort"
	@echo "  clean      - remove generated artifacts"

install:
	uv sync --all-extras

data:
	mkdir -p data/raw
	curl -L -o data/raw/cs-training.csv \
	  "https://raw.githubusercontent.com/DrIanGregory/Kaggle-GiveMeSomeCredit/master/data/GiveMeSomeCredit-training.csv"

eda:
	uv run python -m src.eda

train:
	uv run python -m src.model_trainer

explain:
	uv run python -m src.xai_engine

drift:
	uv run python -m src.drift

pipeline:
	uv run python main.py

dashboard:
	uv run streamlit run app/main_ui.py

test:
	uv run pytest

lint:
	uv run ruff check .

typecheck:
	uv run mypy src

fmt:
	uv run ruff check --fix .
	uv run ruff format .

clean:
	rm -rf outputs/eda outputs/models outputs/explanations outputs/drift
	rm -f models/*.joblib models/threshold.json
	find . -type d -name __pycache__ -exec rm -rf {} +
