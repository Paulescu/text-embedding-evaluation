install:
	curl -sSL https://install.python-poetry.org | python3 -
	poetry env use $(shell which python3.10) && \
	poetry install

run-evals:
	poetry run python -m src.run_evaluations
	
lint:
	@echo "Fixing linting issues..."
	poetry run ruff check --fix .

format:
	echo "Formatting Python code..."
	poetry run ruff format .