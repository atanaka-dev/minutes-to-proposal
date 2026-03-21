.PHONY: setup run lint format test clean

setup:
	poetry install

run:
	poetry run streamlit run app/main.py

lint:
	poetry run ruff check .

format:
	poetry run ruff format .
	poetry run ruff check . --fix

test:
	poetry run pytest -q

clean:
	rm -rf .pytest_cache .ruff_cache