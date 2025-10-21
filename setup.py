# FabricaIA Development Environment Setup

# Python version
python_version = "3.11"

# Development dependencies
dev_dependencies = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
    "isort>=5.12.0",
    "mypy>=1.5.0",
    "pre-commit>=3.4.0",
    "jupyter>=1.0.0",
    "ipykernel>=6.25.0"
]

# Core ML dependencies
core_dependencies = [
    "scikit-learn>=1.3.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "scipy>=1.10.0"
]

# Optional deep learning dependencies
deep_learning_dependencies = [
    "torch>=2.0.0",
    "tensorflow>=2.13.0",
    "transformers>=4.30.0"
]

# API dependencies
api_dependencies = [
    "fastapi>=0.100.0",
    "uvicorn>=0.23.0",
    "pydantic>=2.0.0",
    "python-multipart>=0.0.6"
]

# Workflow dependencies
workflow_dependencies = [
    "apache-airflow>=2.6.0",
    "mlflow>=2.5.0",
    "dvc>=3.0.0"
]

# Database dependencies
database_dependencies = [
    "sqlalchemy>=2.0.0",
    "psycopg2-binary>=2.9.0",
    "sqlite3"
]

# Configuration and utilities
utility_dependencies = [
    "pyyaml>=6.0",
    "python-dotenv>=1.0.0",
    "loguru>=0.7.0",
    "tqdm>=4.65.0",
    "click>=8.1.0",
    "openpyxl>=3.1.0",
    "xlrd>=2.0.0"
]

# Environment setup instructions
setup_instructions = """
# FabricaIA Development Setup

## 1. Create Virtual Environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\\Scripts\\activate     # Windows

## 2. Install Dependencies
pip install -r requirements.txt

## 3. Install Development Tools
pip install -e .

## 4. Setup Pre-commit Hooks
pre-commit install

## 5. Run Tests
pytest tests/

## 6. Start Jupyter Notebook
jupyter notebook notebooks/

## 7. Start API Server
python -m uvicorn src.api.main:app --reload

## 8. Start Airflow (if using)
airflow db init
airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com
airflow webserver --port 8080
"""
