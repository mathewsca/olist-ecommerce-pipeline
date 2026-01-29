# FabricaIA Makefile

.PHONY: help install test lint format clean docker-build docker-run api jupyter airflow format-check precommit mlflow-server mlflow-ui

help: ## Show this help message
	@echo "FabricaIA - Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt
	pip install -e .

install-dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 isort mypy pre-commit
	pip install -e .
	pre-commit install

test: ## Run tests
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	pytest tests/integration/ -v

lint: ## Run linting
	flake8 src/ tests/
	mypy src/

format: ## Format code
	black src/ tests/
	isort src/ tests/

format-check: ## Check formatting (black/isort) without modifying files
	black --check src/ tests/
	isort --check-only src/ tests/

precommit: ## Run all pre-commit hooks on the codebase
	pre-commit run --all-files

clean: ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/

docker-build: ## Build Docker image
	docker build -t fabricaia .

docker-run: ## Run Docker container
	docker run -p 8000:8000 -v $(PWD)/data:/app/data -v $(PWD)/models:/app/models fabricaia

docker-compose-up: ## Start all services with docker-compose
	docker-compose up -d

docker-compose-down: ## Stop all services
	docker-compose down

api: ## Start API server
	python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

jupyter: ## Start Jupyter notebook
	jupyter notebook notebooks/

jupyter-lab: ## Start Jupyter Lab
	jupyter lab notebooks/

airflow-init: ## Initialize Airflow database
	airflow db init
	airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com --password admin

airflow-webserver: ## Start Airflow webserver
	airflow webserver --port 8080

airflow-scheduler: ## Start Airflow scheduler
	airflow scheduler

airflow: airflow-init airflow-webserver ## Start Airflow (init + webserver)

mlflow-server: ## Start MLflow server
	mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db

mlflow-ui: ## Start MLflow UI (alternative to server)
	mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db

notebook: ## Run example notebook
	jupyter nbconvert --execute --to notebook notebooks/fabricaia_example.ipynb

pipeline: ## Run example pipeline
	python pipelines/scripts/example_pipeline.py

create-project: ## Create new project using cookiecutter
	python create_project.py

setup: install-dev ## Setup development environment
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify installation"

all: clean install-dev test lint ## Run all checks (clean, install, test, lint)
