run-evals:
	poetry run python -m src.run_evaluations
	
lint:
	@echo "Fixing linting issues..."
	poetry run ruff check --fix .

format:
	echo "Formatting Python code..."
	poetry run ruff format .