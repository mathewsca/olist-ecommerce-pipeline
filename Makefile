# FabricaIA Makefile

.PHONY: help install test lint format clean docker-build docker-run api jupyter airflow format-check precommit mlflow-server mlflow-ui mlflow-stop data pipeline airflow-stop airflow-standalone ingest dw-init dw-reset streamlit airflow-connections etl-test

help: ## Show this help message
	@echo "FabricaIA - Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

PYTHON ?= $(shell which python3 2>/dev/null || which python 2>/dev/null || echo "python3")
PIP := $(shell which pip 2>/dev/null || if [ -f "venv/bin/pip" ]; then echo "venv/bin/pip"; elif [ -f ".venv/bin/pip" ]; then echo ".venv/bin/pip"; elif [ -f "venv/Scripts/pip.exe" ]; then echo "venv/Scripts/pip.exe"; else echo ""; fi)
AIRFLOW := $(shell which airflow 2>/dev/null || if [ -f "venv/bin/airflow" ]; then echo "venv/bin/airflow"; elif [ -f ".venv/bin/airflow" ]; then echo ".venv/bin/airflow"; elif [ -f "venv/Scripts/airflow.exe" ]; then echo "venv/Scripts/airflow.exe"; else echo "airflow"; fi)

export AIRFLOW__CORE__LOAD_EXAMPLES ?= False
export AIRFLOW__CORE__DAGS_FOLDER ?= $(shell pwd)/pipelines/dags
export PYTHONPATH := $(shell pwd):$(PYTHONPATH)

check-pip:
	@if [ -z "$(PIP)" ]; then \
		echo "❌ Erro: 'pip' não foi encontrado no PATH nem em 'venv/'."; \
		echo "💡 Dica: Crie e ative o ambiente virtual antes:"; \
		echo "   $(PYTHON) -m venv venv"; \
		echo '   source venv/bin/activate  # (ou venv\Scripts\activate no Windows)'; \
		echo "   make install"; \
		exit 1; \
	fi

install: check-pip ## Install core dependencies (lightweight)
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dl: check-pip ## Install optional Deep Learning dependencies (PyTorch, TensorFlow)
	$(PIP) install -r requirements-dl.txt

install-dev: check-pip ## Install development dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov black flake8 isort mypy pre-commit
	$(PIP) install -e .
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

airflow-init: ## Initialize Airflow database (Airflow 2.7+ uses migrate, older uses init)
	@mkdir -p $(HOME)/airflow/dags
	@ln -sfn $(shell pwd)/pipelines/dags $(HOME)/airflow/dags 2>/dev/null || true
	@$(AIRFLOW) db migrate 2>/dev/null || $(AIRFLOW) db init
	@python3 -c 'import os, json; home = os.environ.get("AIRFLOW_HOME", os.path.expanduser("~/airflow")); os.makedirs(home, exist_ok=True); f = os.path.join(home, "simple_auth_manager_passwords.json.generated"); data = json.load(open(f)) if os.path.exists(f) else {}; data["admin"] = "admin"; json.dump(data, open(f, "w"))' 2>/dev/null || true
	@$(AIRFLOW) users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com --password admin 2>/dev/null || echo "Usuário admin pronto (admin/admin)."
	@$(AIRFLOW) dags reserialize 2>/dev/null || true

airflow-webserver: ## Start Airflow server (Airflow 3: api-server, Airflow 2: webserver)
	@mkdir -p $(HOME)/airflow/dags
	@ln -sfn $(shell pwd)/pipelines/dags $(HOME)/airflow/dags 2>/dev/null || true
	@$(AIRFLOW) api-server --port 8080 2>/dev/null || $(AIRFLOW) webserver --port 8080

airflow-scheduler: ## Start Airflow scheduler
	@mkdir -p $(HOME)/airflow/dags
	@ln -sfn $(shell pwd)/pipelines/dags $(HOME)/airflow/dags 2>/dev/null || true
	$(AIRFLOW) scheduler

airflow-standalone: ## Start Airflow all-in-one (api-server/webserver + scheduler)
	@mkdir -p $(HOME)/airflow/dags
	@ln -sfn $(shell pwd)/pipelines/dags $(HOME)/airflow/dags 2>/dev/null || true
	$(AIRFLOW) standalone

airflow-stop: ## Stop running Airflow services
	@pkill -f "airflow (webserver|api-server|scheduler|standalone)" 2>/dev/null && echo "Serviços do Airflow encerrados." || echo "Nenhum serviço do Airflow em execução."

airflow: airflow-init airflow-webserver ## Start Airflow (init + webserver/api-server)

mlflow-server: ## Start MLflow server
	mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri sqlite:///mlflow.db

mlflow-ui: ## Start MLflow UI (alternative to server)
	mlflow ui --host 0.0.0.0 --port 5001 --backend-store-uri sqlite:///mlflow.db

mlflow-stop: ## Stop running MLflow server
	@pkill -f "mlflow (server|ui)" 2>/dev/null && echo "Servidor MLflow encerrado." || echo "Nenhum servidor MLflow em execução."

notebook: ## Run example notebook
	jupyter nbconvert --execute --to notebook notebooks/fabricaia_example.ipynb

data: ## Generate synthetic dataset for end-to-end pipeline
	python scripts/generate_dataset.py

pipeline: ## Run example pipeline (generates sample data if needed)
	@if [ ! -f "data/raw/obras_publicas.csv" ]; then \
		echo "ℹ️  Dataset 'data/raw/obras_publicas.csv' não encontrado. Gerando dados de exemplo..."; \
		$(MAKE) data; \
	fi
	python pipelines/scripts/example_pipeline.py

ingest: ## Download the Olist dataset from Kaggle into data/raw/ (skips if CSVs already present)
	python scripts/kaggle_ingest.py

dw-init: ## Create the raw/staging/dw schemas and DW tables in Postgres
	psql "postgresql://$${DW_USER:-airflow}:$${DW_PASSWORD:-airflow}@$${DW_HOST:-localhost}:$${DW_PORT:-5432}/$${DW_NAME:-olist_dw}" \
		-f sql/raw/create_raw_schema.sql \
		-f sql/staging/create_staging_schema.sql \
		-f sql/dw/create_dw_schema.sql

dw-reset: ## Drop and recreate the olist_dw schemas (destructive - local teaching use only)
	psql "postgresql://$${DW_USER:-airflow}:$${DW_PASSWORD:-airflow}@$${DW_HOST:-localhost}:$${DW_PORT:-5432}/$${DW_NAME:-olist_dw}" \
		-c "DROP SCHEMA IF EXISTS dw CASCADE; DROP SCHEMA IF EXISTS staging CASCADE; DROP SCHEMA IF EXISTS raw CASCADE;"
	$(MAKE) dw-init

streamlit: ## Start the Streamlit dashboard
	streamlit run src/dashboard/app.py

airflow-connections: ## Register the olist_dw_postgres Airflow connection
	@$(AIRFLOW) connections add olist_dw_postgres \
		--conn-type postgres \
		--conn-host $${DW_HOST:-localhost} \
		--conn-schema $${DW_NAME:-olist_dw} \
		--conn-login $${DW_USER:-airflow} \
		--conn-password $${DW_PASSWORD:-airflow} \
		--conn-port $${DW_PORT:-5432} 2>/dev/null || echo "Conexão 'olist_dw_postgres' já existe."

etl-test: ## Run ETL unit tests with coverage
	pytest tests/unit/test_kaggle_ingestion.py tests/unit/test_staging_transform.py tests/unit/test_db_config.py -v --cov=src/etl --cov-report=term-missing

create-project: ## Create new project using cookiecutter
	python create_project.py

setup: install-dev ## Setup development environment
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify installation"

all: clean install-dev test lint ## Run all checks (clean, install, test, lint)
